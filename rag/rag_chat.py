"""RAG 问答：用户提问 → 向量检索 → Qwen 生成回答。

用法示例：
  python rag_chat.py
  python rag_chat.py --top-k 5
"""
import json
import os
import sys

import requests
from pymilvus import MilvusClient

EMBED_URL = os.environ.get("EMBED_URL", "http://14.22.83.225:11002/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
CHAT_URL = os.environ.get("CHAT_URL", "http://14.22.86.97:11001/v1/chat/completions")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen36_27b_lora")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "milvus_lite.db")
COLLECTION_NAME = "report_slices"

PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_prompt.md")


def load_system_prompt():
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "你是一个医疗影像报告分析助手。请根据检索到的参考信息回答用户问题。如果参考信息不足以回答问题，请如实说明。"


def get_embedding(text):
    payload = {"model": EMBED_MODEL, "input": [text]}
    r = requests.post(EMBED_URL, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def search(query_vector, top_k=3, filter_expr=""):
    client = MilvusClient(uri=DB_PATH)
    client.load_collection(COLLECTION_NAME)
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        limit=top_k,
        output_fields=["text", "source", "检查类型", "部位", "检查项目", "诊断结论"],
        filter=filter_expr,
    )
    client.close()
    return results[0]


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
                            print(content, end="", flush=True)
                            full_reply += content
            except Exception:
                pass
    print()
    return full_reply


def rag_query(question, top_k=3, filter_expr="", debug=False):
    print("检索中...", end="", flush=True)
    query_vec = get_embedding(question)
    hits = search(query_vec, top_k=top_k, filter_expr=filter_expr)
    print(" 完成")

    if debug:
        print("\n" + "=" * 60)
        print("【检索结果 Top-{}】".format(top_k))
        print("=" * 60)
        for i, hit in enumerate(hits, 1):
            entity = hit["entity"]
            score = hit.get("distance", "?")
            print(f"\n--- 参考{i} (相似度: {score}) ---")
            print(f"来源: {entity['source']}")
            print(f"检查类型: {entity.get('检查类型', '')}")
            print(f"部位: {entity.get('部位', '')}")
            print(f"检查项目: {entity.get('检查项目', '')}")
            print(f"诊断结论: {entity.get('诊断结论', '')}")
            print(f"全文:\n{entity['text']}")

    contexts = []
    for i, hit in enumerate(hits, 1):
        entity = hit["entity"]
        contexts.append(f"【参考{i}】(来源: {entity['source']})\n{entity['text']}")

    context_text = "\n\n".join(contexts)

    system_prompt = load_system_prompt()
    user_message = f"参考信息：\n{context_text}\n\n用户问题：{question}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    if debug:
        print("\n" + "=" * 60)
        print("【发送给 LLM 的 Prompt】")
        print("=" * 60)
        for msg in messages:
            print(f"\n[{msg['role']}]")
            print(msg["content"])
        print("=" * 60 + "\n")

    print("AI: ", end="", flush=True)
    reply = chat_stream(messages)
    return reply


def main():
    top_k = 1
    debug = False
    for arg in sys.argv[1:]:
        if arg.startswith("--top-k="):
            top_k = int(arg.split("=")[1])
        elif arg == "--debug":
            debug = True

    if not os.path.exists(DB_PATH):
        print("向量数据库不存在，请先运行 build_vector_db.py", file=sys.stderr)
        sys.exit(1)

    print("=== RAG 问答已启动 ===")
    print(f"Embedding: {EMBED_MODEL}")
    print(f"生成模型: {CHAT_MODEL}")
    print(f"检索数量: top-{top_k}")
    if debug:
        print("调试模式: 开启（打印检索结果和Prompt）")
    print("输入 exit 或 quit 退出\n")

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("再见！")
            break

        try:
            rag_query(user_input, top_k=top_k, debug=debug)
        except Exception as e:
            print(f"（出错: {e}）")

        print()


if __name__ == "__main__":
    main()