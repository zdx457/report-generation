"""RAG 问答 Web 界面（Gradio）。

用法示例：
  python web.py
  python web.py --share          # 生成公网链接
  python web.py --top-k 5        # 调整检索数量
  python web.py --debug          # 显示检索详情
"""
import json
import os
import shutil
import sys
import time

import gradio as gr
import requests
from openpyxl import load_workbook
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient

EMBED_URL = os.environ.get("EMBED_URL", "http://14.22.83.225:11002/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
CHAT_URL = os.environ.get("CHAT_URL", "http://14.22.86.97:11001/v1/chat/completions")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen36_27b_lora")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "milvus_lite.db")
COLLECTION_NAME = "report_slices"
REPORT_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template")
XLSX_SLICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xlsx_slices")
EMBED_DIM = 1024

PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.md")


def load_system_prompt():
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "你是一个医疗影像报告生成助手。请根据检索到的参考信息回答用户问题。如果参考信息不足以回答问题，请如实说明。"


SYSTEM_PROMPT = load_system_prompt()

milvus_client = MilvusClient(uri=DB_PATH)
milvus_client.load_collection(COLLECTION_NAME)


def get_embedding(text):
    payload = {"model": EMBED_MODEL, "input": [text]}
    r = requests.post(EMBED_URL, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def search(query_vector, top_k=3):
    results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=top_k,
        output_fields=["text", "source", "检查类型", "部位", "检查项目", "诊断结论"],
    )
    return results[0]


def _xlsx_to_slices(filepath):
    wb = load_workbook(filepath, read_only=True, data_only=True)
    all_rows = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            all_rows.append([cell if cell is not None else "" for cell in row])
    wb.close()
    if not all_rows:
        return [], []
    header = all_rows[0]
    data_rows = all_rows[1:]
    return header, data_rows


def _slice_to_md(header, row, sheet_name="Sheet"):
    lines = [f"## {sheet_name}", ""]
    header_line = "| " + " | ".join(str(h) for h in header) + " |"
    separator = "| " + " | ".join("---" for _ in header) + " |"
    data_line = "| " + " | ".join(str(v) for v in row) + " |"
    lines.extend([header_line, separator, data_line, ""])
    return "\n".join(lines)


def _parse_md_slice(filepath):
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
    return "\n".join(text_parts), metadata


def _get_embeddings_batch(texts, batch_size=16):
    all_embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        payload = {"model": EMBED_MODEL, "input": batch}
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
                    time.sleep((attempt + 1) * 5)
                else:
                    raise RuntimeError(f"向量化批次失败: {e}") from e
        time.sleep(0.2)
    return all_embeddings


def _create_collection_if_needed(client):
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
    client.create_collection(collection_name=COLLECTION_NAME, schema=schema, index_params=index_params)


def upload_and_process(files, progress=gr.Progress()):
    if not files:
        return "请上传 xlsx 文件"

    os.makedirs(REPORT_TEMPLATE_DIR, exist_ok=True)
    os.makedirs(XLSX_SLICES_DIR, exist_ok=True)

    saved_files = []
    for f in files:
        fname = os.path.basename(f.name)
        if not fname.endswith(".xlsx") or fname.startswith("~$"):
            continue
        dest = os.path.join(REPORT_TEMPLATE_DIR, fname)
        shutil.copy2(f.name, dest)
        saved_files.append((fname, dest))

    if not saved_files:
        return "未找到有效的 xlsx 文件（跳过临时文件和非 xlsx 文件）"

    progress(0.1, desc="切片中...")
    total_slices = 0
    new_md_files = []

    for idx, (fname, fpath) in enumerate(saved_files):
        progress((0.1 + 0.2 * idx / len(saved_files)), desc=f"切片: {fname}")
        header, data_rows = _xlsx_to_slices(fpath)
        if not header:
            continue
        basename = os.path.splitext(fname)[0]
        for i, row in enumerate(data_rows, start=1):
            md_content = _slice_to_md(header, row, sheet_name=basename)
            out_name = f"{basename}_row{i}.md"
            out_path = os.path.join(XLSX_SLICES_DIR, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            new_md_files.append(out_name)
            total_slices += 1

    if total_slices == 0:
        return "切片结果为空，请检查 xlsx 文件内容"

    progress(0.3, desc="检查增量...")
    client = MilvusClient(uri=DB_PATH)
    _create_collection_if_needed(client)

    existing = set()
    try:
        client.load_collection(COLLECTION_NAME)
        results = client.query(COLLECTION_NAME, filter="", output_fields=["source"])
        existing = {r["source"] for r in results}
    except Exception:
        pass

    truly_new = [f for f in new_md_files if f not in existing]
    if not truly_new:
        client.close()
        return f"切片完成: {total_slices} 个，但全部已存在于数据库中，无需更新"

    progress(0.4, desc=f"解析 {len(truly_new)} 个新切片...")
    texts = []
    metadatas = []
    source_files = []
    for fname in truly_new:
        fpath = os.path.join(XLSX_SLICES_DIR, fname)
        text, meta = _parse_md_slice(fpath)
        if text:
            texts.append(text)
            metadatas.append(meta or {})
            source_files.append(fname)

    if not texts:
        client.close()
        return "解析完成但无有效切片"

    progress(0.5, desc=f"向量化 {len(texts)} 条切片...")
    embeddings = _get_embeddings_batch(texts, batch_size=16)

    progress(0.85, desc="写入数据库...")
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
    client.close()

    global milvus_client
    milvus_client = MilvusClient(uri=DB_PATH)
    milvus_client.load_collection(COLLECTION_NAME)

    progress(1.0, desc="完成")
    return (
        f"处理完成！\n"
        f"- 上传文件: {len(saved_files)} 个\n"
        f"- 生成切片: {total_slices} 个\n"
        f"- 新增入库: {len(truly_new)} 条\n"
        f"- 已有跳过: {len(new_md_files) - len(truly_new)} 条"
    )


def chat_stream(messages, max_tokens=1024, temperature=0.7):
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}

    full_reply = ""
    with requests.post(CHAT_URL, headers=headers, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
            else:
                data = line
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                if isinstance(obj, dict) and "choices" in obj:
                    for c in obj["choices"]:
                        delta = c.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            full_reply += content
                            yield full_reply
            except Exception:
                pass


def rag_respond(message, history, top_k, temperature):
    query_vec = get_embedding(message)
    hits = search(query_vec, top_k=top_k)

    contexts = []
    ref_details = []
    for i, hit in enumerate(hits, 1):
        entity = hit["entity"]
        score = hit.get("distance", 0)
        contexts.append(f"【参考{i}】(来源: {entity['source']})\n{entity['text']}")
        ref_details.append({
            "index": i,
            "source": entity["source"],
            "score": f"{score:.4f}",
            "检查类型": entity.get("检查类型", ""),
            "部位": entity.get("部位", ""),
            "检查项目": entity.get("检查项目", ""),
            "诊断结论": entity.get("诊断结论", ""),
            "text": entity["text"],
        })

    thinking_html = "<details><summary>🧠 思考过程</summary>\n"
    thinking_html += "<div style='padding:8px;background:#f8f9fa;border-radius:6px;font-size:14px;'>\n"

    thinking_html += "<p><b>🔍 第一步：向量检索</b>（top-{}）</p>\n".format(top_k)
    for ref in ref_details:
        thinking_html += (
            "<div style='margin-left:16px;margin-bottom:8px;'>"
            "<b>参考{index}</b> · 相似度: {score} · 来源: <code>{source}</code><br>"
            "检查类型: {检查类型} | 部位: {部位} | 检查项目: {检查项目}<br>"
            "诊断结论: {诊断结论}<br>"
            "<details><summary>查看全文</summary><pre style='white-space:pre-wrap;'>{text}</pre></details>"
            "</div>\n"
        ).format(**ref)

    context_text = "\n\n".join(contexts)
    user_message = f"参考信息：\n{context_text}\n\n用户问题：{message}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    thinking_html += "<p><b>📝 第二步：构造提示词</b></p>\n"
    thinking_html += "<div style='margin-left:16px;margin-bottom:8px;'>\n"
    thinking_html += "<details><summary>System Prompt</summary><pre style='white-space:pre-wrap;'>{}</pre></details>\n".format(
        SYSTEM_PROMPT.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    thinking_html += "<details><summary>User Prompt</summary><pre style='white-space:pre-wrap;'>{}</pre></details>\n".format(
        user_message.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    thinking_html += "</div>\n"

    thinking_html += "<p><b>🤖 第三步：LLM 生成报告</b></p>\n"
    thinking_html += "<div style='margin-left:16px;margin-bottom:8px;'>"
    thinking_html += "模型: <code>{}</code> | 温度: {}</div>\n".format(CHAT_MODEL, temperature)

    thinking_html += "</div>\n</details>\n"

    yield thinking_html

    partial_reply = ""
    for chunk in chat_stream(messages, temperature=temperature):
        partial_reply = chunk
        yield thinking_html + "\n" + partial_reply


def build_ui():
    with gr.Blocks(title="医疗影像报告生成", css="""
        .chatbot .message { font-size: 16px; line-height: 1.6; }
    """) as demo:
        gr.Markdown("# 🏥 医疗影像报告生成系统")
        gr.Markdown(
            f"Embedding: `{EMBED_MODEL}` | 生成模型: `{CHAT_MODEL}`"
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=800, sanitize_html=False)
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入问题，如：CT弥漫性肺气肿",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                    clear_btn = gr.Button("清空", scale=1)

            with gr.Column(scale=1):
                gr.Markdown("### 参数设置")
                top_k = gr.Slider(1, 10, value=1, step=1, label="检索数量 (Top-K)")
                temperature = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="温度 (Temperature)")

                gr.Markdown("---")
                gr.Markdown("### 📤 上传报告模板")
                file_upload = gr.File(
                    label="上传 xlsx 文件",
                    file_count="multiple",
                    file_types=[".xlsx"],
                )
                upload_btn = gr.Button("上传并处理", variant="secondary")
                upload_status = gr.Textbox(label="处理状态", interactive=False, lines=5)

        def respond(message, history, top_k_val, temp_val):
            reply_text = ""
            for partial in rag_respond(message, history, top_k_val, temp_val):
                reply_text = partial
                display = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": reply_text},
                ]
                yield display

        def clear_chat():
            return []

        msg_input.submit(
            respond,
            inputs=[msg_input, chatbot, top_k, temperature],
            outputs=[chatbot],
        ).then(lambda: "", outputs=msg_input)

        send_btn.click(
            respond,
            inputs=[msg_input, chatbot, top_k, temperature],
            outputs=[chatbot],
        ).then(lambda: "", outputs=msg_input)

        clear_btn.click(clear_chat, outputs=[chatbot])

        upload_btn.click(
            upload_and_process,
            inputs=[file_upload],
            outputs=[upload_status],
        )

    return demo


def main():
    share = "--share" in sys.argv
    debug = "--debug" in sys.argv
    top_k_default = 3
    for arg in sys.argv[1:]:
        if arg.startswith("--top-k="):
            top_k_default = int(arg.split("=")[1])

    demo = build_ui()
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(share=share, debug=debug, theme=gr.themes.Soft(), server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    main()