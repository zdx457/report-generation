"""读取 xlsx 切片 md 文件，向量化后存入 Milvus Lite（支持增量添加）。

用法示例：
  python build_vector_db.py                                    # 增量模式，只添加新切片
  python build_vector_db.py --input ../xlsx_slices             # 指定输入文件夹
  python build_vector_db.py --rebuild                          # 全量重建
  python build_vector_db.py --batch-size 8                     # 指定批次大小
"""
import argparse
import os
import shutil
import sys
import time

import requests
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

EMBED_URL = os.environ.get("EMBED_URL", "http://14.22.83.225:11002/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
EMBED_DIM = 1024
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "milvus_lite.db")
COLLECTION_NAME = "report_slices"


def parse_md_slice(filepath):
    """解析切片 md 文件，返回 (自然语言文本, 元数据字典)。"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = [l.strip() for l in content.strip().split("\n") if l.strip()]

    header = None
    data_row = None
    for line in lines:
        if line.startswith("|") and "---" in line:
            continue
        if line.startswith("##"):
            continue
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c]
        if not cells:
            continue
        if header is None:
            header = cells
        else:
            data_row = cells
            break

    if not header or not data_row:
        return None, None

    text_parts = []
    metadata = {}
    for h, v in zip(header, data_row):
        text_parts.append(f"{h}：{v}")
        if h in ("检查类型", "部位", "检查项目", "诊断结论"):
            metadata[h] = v

    natural_text = "\n".join(text_parts)
    return natural_text, metadata


def get_embeddings(texts, batch_size=16):
    """调用 bge-m3 接口，批量获取向量，带重试。"""
    all_embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        payload = {
            "model": EMBED_MODEL,
            "input": batch,
        }
        for attempt in range(5):
            try:
                r = requests.post(EMBED_URL, json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()["data"]
                data.sort(key=lambda x: x["index"])
                all_embeddings.extend([d["embedding"] for d in data])
                break
            except Exception as e:
                if attempt < 4:
                    wait = (attempt + 1) * 5
                    print(f"  批次 {i//batch_size+1} 失败({e})，{wait}s 后重试...")
                    time.sleep(wait)
                else:
                    print(f"  批次 {i//batch_size+1} 重试5次仍失败，跳过", file=sys.stderr)
                    all_embeddings.extend([None] * len(batch))
        done = min(i + batch_size, total)
        print(f"\r  向量化进度: {done}/{total} ({done*100//total}%)", end="", flush=True)
        time.sleep(0.2)
    print()
    all_embeddings = [e for e in all_embeddings if e is not None]
    return all_embeddings


def get_existing_sources(client):
    """获取数据库中已有的 source 文件名集合。"""
    if not client.has_collection(COLLECTION_NAME):
        return set()
    try:
        client.load_collection(COLLECTION_NAME)
        results = client.query(COLLECTION_NAME, filter="", output_fields=["source"])
        return {r["source"] for r in results}
    except Exception:
        return set()


def create_collection(client):
    """创建集合（如果不存在）。"""
    if client.has_collection(COLLECTION_NAME):
        return
    schema = CollectionSchema(fields=[
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="检查类型", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="部位", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="检查项目", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="诊断结论", dtype=DataType.VARCHAR, max_length=1024),
    ])

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="COSINE", params={"nlist": 128})

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )


def insert_data(client, texts, metadatas, source_files, embeddings):
    """将数据插入集合。"""
    data = []
    for i in range(len(texts)):
        meta = metadatas[i]
        data.append({
            "vector": embeddings[i],
            "text": texts[i],
            "source": source_files[i],
            "检查类型": meta.get("检查类型", ""),
            "部位": meta.get("部位", ""),
            "检查项目": meta.get("检查项目", ""),
            "诊断结论": meta.get("诊断结论", ""),
        })

    insert_batch = 500
    for i in range(0, len(data), insert_batch):
        client.insert(collection_name=COLLECTION_NAME, data=data[i:i + insert_batch])
    return len(data)


def build_db(input_dir, batch_size=16, rebuild=False):
    if not os.path.isdir(input_dir):
        print(f"输入文件夹不存在: {input_dir}", file=sys.stderr)
        return

    md_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".md")])
    if not md_files:
        print(f"在 {input_dir} 中未找到 md 文件", file=sys.stderr)
        return

    if rebuild:
        print("全量重建模式：删除旧数据...")
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)

    client = MilvusClient(uri=DB_PATH)

    if rebuild:
        create_collection(client)
        new_files = md_files
        print(f"将处理全部 {len(new_files)} 个切片文件")
    else:
        create_collection(client)
        existing = get_existing_sources(client)
        new_files = [f for f in md_files if f not in existing]
        print(f"数据库已有: {len(existing)} 条")
        print(f"新增切片: {len(new_files)} 个")
        if not new_files:
            print("没有新切片需要添加")
            client.close()
            return

    print(f"开始解析 {len(new_files)} 个新切片...")

    texts = []
    metadatas = []
    source_files = []
    for fname in new_files:
        fpath = os.path.join(input_dir, fname)
        text, meta = parse_md_slice(fpath)
        if text:
            texts.append(text)
            metadatas.append(meta or {})
            source_files.append(fname)

    print(f"有效新切片: {len(texts)} 条")
    if not texts:
        print("没有有效切片可添加")
        client.close()
        return

    print(f"开始向量化 (batch_size={batch_size})...")
    embeddings = get_embeddings(texts, batch_size=batch_size)

    if not embeddings:
        print("向量化失败，未获取到任何向量", file=sys.stderr)
        client.close()
        return

    print(f"向量化完成，维度: {len(embeddings[0])}")
    print("写入 Milvus Lite...")

    inserted = insert_data(client, texts, metadatas, source_files, embeddings)

    total = client.query(COLLECTION_NAME, filter="", output_fields=["count(*)"])
    total_count = len(total)
    print(f"本次新增: {inserted} 条")
    print(f"数据库总计: {total_count} 条")
    print(f"数据库文件: {DB_PATH}")
    client.close()


def main():
    default_input = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xlsx_slices")
    parser = argparse.ArgumentParser(description="将 xlsx 切片向量化存入 Milvus Lite（支持增量）")
    parser.add_argument("--input", type=str, default=default_input, help="切片 md 文件所在文件夹")
    parser.add_argument("--batch-size", type=int, default=16, help="向量化批次大小")
    parser.add_argument("--rebuild", action="store_true", help="全量重建（删除旧数据重新导入）")
    args = parser.parse_args()

    build_db(os.path.abspath(args.input), batch_size=args.batch_size, rebuild=args.rebuild)


if __name__ == "__main__":
    main()