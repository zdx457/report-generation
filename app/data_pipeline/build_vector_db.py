"""读取 xlsx 切片 md 文件，向量化后存入 Milvus Lite（支持增量添加）。

用法示例：
  python build_vector_db.py                                    # 增量模式，只添加新切片
  python build_vector_db.py --input ../xlsx_slices             # 指定输入文件夹
  python build_vector_db.py --rebuild                          # 全量重建
  python build_vector_db.py --batch-size 8                     # 指定批次大小
"""
import argparse
import os
import sys
import time

import requests
from dotenv import load_dotenv
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

from config import (
    get_embed_base_url, get_embed_model,
    get_embed_dimension, get_db_path, get_collection_name,
)

EMBED_URL = get_embed_base_url()
EMBED_MODEL = get_embed_model()
EMBED_DIM = get_embed_dimension()
DB_PATH = get_db_path()
COLLECTION_NAME = get_collection_name()


def parse_md_slice(filepath):
    """解析切片 md 文件，返回 (自然语言文本, 元数据字典, 影像学表现文本)。
    
    Returns:
        tuple: (natural_text, metadata, imaging_text)
            - natural_text: 完整行内容的自然语言文本
            - metadata: 元数据字典
            - imaging_text: 仅影像学表现字段的文本
    """
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
        return None, None, None

    text_parts = []
    metadata = {}
    imaging_text = ""
    for h, v in zip(header, data_row):
        text_parts.append(f"{h}：{v}")
        if h in ("检查类型", "部位", "检查项目", "诊断结论"):
            metadata[h] = v
        if h == "影像学表现" and v.strip():
            imaging_text = v.strip()

    natural_text = "\n".join(text_parts)
    return natural_text, metadata, imaging_text


def get_embeddings(texts, batch_size=16, progress_callback=None):
    """调用 bge-m3 接口，批量获取向量，带重试。"""
    def _log(msg, level="info"):
        print(msg, flush=True)
        if progress_callback:
            progress_callback({"level": level, "msg": msg})

    all_embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        payload = {
            "model": EMBED_MODEL,
            "input": batch,
        }
        success = False
        for attempt in range(5):
            try:
                r = requests.post(EMBED_URL, json=payload, timeout=120)
                r.raise_for_status()
                data = r.json()["data"]
                data.sort(key=lambda x: x["index"])
                all_embeddings.extend([d["embedding"] for d in data])
                success = True
                break
            except Exception as e:
                if attempt < 4:
                    wait = (attempt + 1) * 5
                    # 静默等待，不发送日志到前端
                    _log(f"向量化进度: {i//batch_size+1}/{(total + batch_size - 1)//batch_size} 批次 (重试中...)", "info")
                    time.sleep(wait)
                else:
                    _log(f"批次 {i//batch_size+1} 重试5次仍失败: {e}", "error")
                    all_embeddings.extend([None] * len(batch))
        if success:
            done = min(i + batch_size, total)
            _log(f"向量化进度: {done}/{total} ({done*100//total}%)")
        time.sleep(0.2)
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
        FieldSchema(name="vector_type", dtype=DataType.VARCHAR, max_length=64),
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


def insert_data(client, texts, metadatas, source_files, embeddings, vector_types=None):
    """将数据插入集合。
    
    Args:
        vector_types: 向量类型列表，与 texts 一一对应（"full_row" 或 "opinion"）
    """
    if vector_types is None:
        vector_types = ["full_row"] * len(texts)
    
    data = []
    for i in range(len(texts)):
        meta = metadatas[i]
        data.append({
            "vector": embeddings[i],
            "text": texts[i],
            "source": source_files[i],
            "vector_type": vector_types[i],
            "检查类型": meta.get("检查类型", ""),
            "部位": meta.get("部位", ""),
            "检查项目": meta.get("检查项目", ""),
            "诊断结论": meta.get("诊断结论", ""),
        })

    insert_batch = 500
    for i in range(0, len(data), insert_batch):
        client.insert(collection_name=COLLECTION_NAME, data=data[i:i + insert_batch])
    return len(data)


def build_db(input_dir, batch_size=16, rebuild=False, progress_callback=None):
    def _log(msg, level="info"):
        print(msg, flush=True)
        if progress_callback:
            progress_callback({"level": level, "msg": msg})

    # 全量重建：先清空旧切片，然后重新切片
    if rebuild:
        _log("全量重建模式：清空旧数据...")
        
        # 清空旧切片文件（避免新旧命名方式冲突）
        if os.path.isdir(input_dir):
            import shutil
            _log(f"清空切片目录: {input_dir}")
            shutil.rmtree(input_dir)
        
        # 自动重新切片
        report_dir = os.path.join(os.path.dirname(input_dir), "report_template")
        if os.path.isdir(report_dir):
            xlsx_files = [f for f in os.listdir(report_dir) if f.endswith(".xlsx") and not f.startswith("~$")]
            if xlsx_files:
                os.makedirs(input_dir, exist_ok=True)
                
                # 动态导入 xlsx_slicer
                slicer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xlsx_slicer.py")
                import importlib.util
                spec = importlib.util.spec_from_file_location("xlsx_slicer", slicer_path)
                xlsx_slicer = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(xlsx_slicer)
                process_file = xlsx_slicer.process_file
                
                for fname in xlsx_files:
                    fpath = os.path.join(report_dir, fname)
                    _log(f"自动切片: {fname}")
                    process_file(fpath, input_dir, progress_callback)
            else:
                _log(f"在 {report_dir} 中未找到 xlsx 文件", "error")
                return
        else:
            _log(f"报告模板目录不存在: {report_dir}", "error")
            return
    elif not os.path.isdir(input_dir):
        # 增量模式：目录不存在则自动切片
        _log(f"输入文件夹不存在: {input_dir}，尝试自动切片...")
        
        report_dir = os.path.join(os.path.dirname(input_dir), "report_template")
        if os.path.isdir(report_dir):
            xlsx_files = [f for f in os.listdir(report_dir) if f.endswith(".xlsx") and not f.startswith("~$")]
            if xlsx_files:
                os.makedirs(input_dir, exist_ok=True)
                
                slicer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xlsx_slicer.py")
                import importlib.util
                spec = importlib.util.spec_from_file_location("xlsx_slicer", slicer_path)
                xlsx_slicer = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(xlsx_slicer)
                process_file = xlsx_slicer.process_file
                
                for fname in xlsx_files:
                    fpath = os.path.join(report_dir, fname)
                    _log(f"自动切片: {fname}")
                    process_file(fpath, input_dir, progress_callback)
            else:
                _log(f"在 {report_dir} 中未找到 xlsx 文件", "error")
                return
        else:
            _log(f"报告模板目录不存在: {report_dir}", "error")
            return

    md_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".md")])
    if not md_files:
        _log(f"在 {input_dir} 中未找到 md 文件", "error")
        return

    client = MilvusClient(uri=DB_PATH)

    if rebuild:
        _log("清空向量数据库集合...")
        # 关闭客户端再删除数据库文件（避免 Windows 文件锁定问题）
        client.close()
        if os.path.exists(DB_PATH):
            _log(f"删除数据库文件: {DB_PATH}")
            os.remove(DB_PATH)
        # 重新创建客户端和集合
        client = MilvusClient(uri=DB_PATH)
        create_collection(client)
        new_files = md_files
        _log(f"将处理全部 {len(new_files)} 个切片文件")
    else:
        create_collection(client)
        existing = get_existing_sources(client)
        new_files = [f for f in md_files if f not in existing]
        _log(f"数据库已有: {len(existing)} 条")
        _log(f"新增切片: {len(new_files)} 个")
        if not new_files:
            _log("没有新切片需要添加")
            client.close()
            return

    _log(f"开始解析 {len(new_files)} 个新切片...")

    # ── 双通道数据收集 ──
    # full_row 通道：完整行内容
    full_texts = []
    full_metadatas = []
    full_sources = []
    
    # imaging 通道：仅影像学表现
    imaging_texts = []
    imaging_metadatas = []
    imaging_sources = []

    for fname in new_files:
        fpath = os.path.join(input_dir, fname)
        text, meta, imaging = parse_md_slice(fpath)
        if text:
            # 通道1：完整行
            full_texts.append(text)
            full_metadatas.append(meta or {})
            full_sources.append(fname)
            
            # 通道2：影像学表现（如果存在）
            if imaging:
                imaging_texts.append(imaging)
                imaging_metadatas.append(meta or {})
                imaging_sources.append(fname)

    _log(f"完整行通道: {len(full_texts)} 条")
    _log(f"影像学表现通道: {len(imaging_texts)} 条")
    _log(f"总计待向量化: {len(full_texts) + len(imaging_texts)} 条")
    
    if not full_texts:
        _log("没有有效切片可添加")
        client.close()
        return

    _log(f"开始向量化 (batch_size={batch_size})...")
    
    # 合并两个通道的文本进行批量向量化
    all_texts = full_texts + imaging_texts
    all_embeddings = get_embeddings(all_texts, batch_size=batch_size, progress_callback=progress_callback)

    if not all_embeddings:
        _log("向量化失败，未获取到任何向量", "error")
        client.close()
        return

    _log(f"向量化完成，维度: {len(all_embeddings[0])}")
    _log("写入 Milvus Lite...")

    # 分离两个通道的嵌入向量
    full_embeddings = all_embeddings[:len(full_texts)]
    imaging_embeddings = all_embeddings[len(full_texts):]
    
    # 插入完整行通道
    inserted_full = 0
    if full_embeddings:
        inserted_full = insert_data(
            client, full_texts, full_metadatas, full_sources, 
            full_embeddings, vector_types=["full_row"] * len(full_texts)
        )
        _log(f"完整行通道插入: {inserted_full} 条")
    
    # 插入影像学表现通道
    inserted_imaging = 0
    if imaging_embeddings:
        inserted_imaging = insert_data(
            client, imaging_texts, imaging_metadatas, imaging_sources,
            imaging_embeddings, vector_types=["imaging"] * len(imaging_texts)
        )
        _log(f"影像学表现通道插入: {inserted_imaging} 条")

    total = client.query(COLLECTION_NAME, filter="", output_fields=["count(*)"])
    total_count = len(total)
    _log(f"本次新增总计: {inserted_full + inserted_imaging} 条")
    _log(f"数据库总计: {total_count} 条")
    _log(f"数据库文件: {DB_PATH}")
    client.close()
    _log("__DONE__", "done")


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