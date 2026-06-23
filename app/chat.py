"""RAG 问答：用户提问 → 向量检索 → Qwen 生成回答。

用法示例：
  python rag_chat.py
  python rag_chat.py --top-k 5
"""
import json
import os
import sys

import requests
from dotenv import load_dotenv
from pymilvus import MilvusClient

from rerank import get_rerank_config, rerank_documents
from retrieval import multi_recall, parse_query_keywords
from query_rewrite import rewrite_query, needs_rewrite, is_too_vague, get_clarification

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

EMBED_URL = os.environ.get("EMBED_URL", "http://14.22.83.225:11002/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
CHAT_URL = os.environ.get("CHAT_URL", "http://14.22.86.97:11001/v1/chat/completions")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen36_27b_lora")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "milvus_lite.db")
COLLECTION_NAME = "report_slices"

PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt.md")


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


def rag_query(question, top_k=5, rerank_top_k=3, debug=False):
    if rerank_top_k > top_k:
        rerank_top_k = top_k

    if is_too_vague(question):
        clarification = get_clarification(question)
        print("\n⚠️ 您的输入过于模糊，请补充检查部位或诊断信息：")
        print(clarification)
        return clarification

    original_query = question
    rewritten_query = question
    query_was_rewritten = False

    if needs_rewrite(question):
        rewritten_query = rewrite_query(question)
        if rewritten_query != question:
            query_was_rewritten = True
            if debug:
                print(f"\n✏️ 查询改写: '{original_query}' → '{rewritten_query}'")

    search_query = rewritten_query if query_was_rewritten else question

    print("检索中...", end="", flush=True)
    query_vec = get_embedding(search_query)

    keywords = parse_query_keywords(search_query)

    client = MilvusClient(uri=DB_PATH)
    client.load_collection(COLLECTION_NAME)
    candidates = multi_recall(search_query, query_vec, top_k=top_k, client=client)
    client.close()
    print(" 完成")

    if debug:
        print("\n" + "=" * 60)
        print("【多路召回结果】（每路 top-{}）".format(top_k))
        print("=" * 60)
        print(f"解析关键词: 检查类型={keywords.get('检查类型', '-') or '-'} | 部位={keywords.get('部位', '-') or '-'} | 诊断关键词={', '.join(keywords.get('诊断关键词', [])) or '-'}")
        path_counts = {}
        for c in candidates:
            path = c.get("_recall_path", "vector")
            path_counts[path] = path_counts.get(path, 0) + 1
        print(f"召回路径: {' | '.join(f'{k}: {v}条' for k, v in sorted(path_counts.items()))}")
        print(f"合并去重后: {len(candidates)} 条候选")
        for i, c in enumerate(candidates[:10], 1):
            paths_str = "+".join(c.get("_recall_paths", []))
            dist_str = f"相似度: {c.get('_distance', -1):.4f}" if c.get("_distance", -1) >= 0 else "精确匹配"
            print(f"  候选{i} [{paths_str}] {dist_str} | 来源: {c['source']} | 诊断: {c.get('诊断结论', '')}")

    documents = [e["text"] for e in candidates]

    reranked_entities = []
    try:
        print("Rerank中...", end="", flush=True)
        rerank_results = rerank_documents(question, documents, top_n=rerank_top_k)
        print(" 完成")
        for rr in rerank_results:
            idx = rr.get("index", 0)
            rerank_score = rr.get("relevance_score", 0)
            entity = candidates[idx]
            entity["_rerank_score"] = rerank_score
            reranked_entities.append(entity)
    except Exception as e:
        print(f" Rerank失败({e})，使用向量检索结果")
        reranked_entities = candidates[:rerank_top_k]

    if debug:
        print("\n" + "=" * 60)
        print("【Rerank结果 Top-{}】".format(len(reranked_entities)))
        print("=" * 60)
        for i, entity in enumerate(reranked_entities, 1):
            rerank_score = entity.get("_rerank_score", "N/A")
            recall_path = entity.get("_recall_path", "vector")
            print(f"\n--- 参考{i} (Rerank: {rerank_score}, 召回路径: {recall_path}) ---")
            print(f"来源: {entity['source']}")
            print(f"诊断结论: {entity.get('诊断结论', '')}")

    contexts = []
    for i, entity in enumerate(reranked_entities, 1):
        rerank_score = entity.get("_rerank_score", 0)
        contexts.append(f"【参考{i}】(Rerank相关性分数: {rerank_score:.4f}，来源: {entity['source']})\n{entity['text']}")

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
    top_k = 5
    rerank_top_k = 3
    debug = False
    for arg in sys.argv[1:]:
        if arg.startswith("--top-k="):
            top_k = int(arg.split("=")[1])
        elif arg.startswith("--rerank-top-k="):
            rerank_top_k = int(arg.split("=")[1])
        elif arg == "--debug":
            debug = True

    if not os.path.exists(DB_PATH):
        print("向量数据库不存在，请先运行 build_vector_db.py", file=sys.stderr)
        sys.exit(1)

    print("=== RAG 问答已启动 ===")
    print(f"Embedding: {EMBED_MODEL}")
    print(f"Rerank: {get_rerank_config()['rerank_model']}")
    print(f"生成模型: {CHAT_MODEL}")
    print(f"多路召回: 向量检索 + 元数据过滤 + 关键词检索 (每路 top-{top_k})")
    print(f"Rerank返回: top-{rerank_top_k}")
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
            rag_query(user_input, top_k=top_k, rerank_top_k=rerank_top_k, debug=debug)
        except Exception as e:
            print(f"（出错: {e}）")

        print()


if __name__ == "__main__":
    main()