"""完整流程测试: 向量检索 → Rerank → Prompt拼接 → LLM输入预览"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

from pymilvus import MilvusClient
import requests

from rerank import rerank_documents, get_rerank_config

EMBED_URL = os.environ.get("EMBED_URL", "http://14.22.83.225:11002/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "milvus_lite.db")
COLLECTION_NAME = "report_slices"

VEC_TOP_K = 5
RERANK_TOP_K = 3

client = MilvusClient(uri=DB_PATH)
client.load_collection(COLLECTION_NAME)

query = "CT脑出血"

print("=" * 60)
print(f"用户问题: {query}")
print(f"向量检索: top-{VEC_TOP_K}")
print(f"Rerank: top-{RERANK_TOP_K}")
print(f"Rerank模型: {get_rerank_config()['rerank_model']}")
print("=" * 60)

payload = {"model": EMBED_MODEL, "input": [query]}
resp = requests.post(EMBED_URL, json=payload, timeout=30)
query_vec = resp.json()["data"][0]["embedding"]

print(f"\n📌 第一步: 向量检索 (top-{VEC_TOP_K})")
print("-" * 60)
hits = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vec],
    limit=VEC_TOP_K,
    output_fields=["source", "text", "诊断结论", "检查类型", "部位"],
)

documents = []
hit_entities = []
for i, hit in enumerate(hits[0], 1):
    e = hit["entity"]
    documents.append(e["text"])
    entity = {
        "source": e["source"],
        "text": e["text"],
        "诊断结论": e.get("诊断结论", ""),
        "检查类型": e.get("检查类型", ""),
        "部位": e.get("部位", ""),
        "_distance": round(hit["distance"], 4),
    }
    hit_entities.append(entity)
    print(f"  Top-{i}: 相似度={hit['distance']:.4f} | 来源={e['source']}")
    print(f"         诊断={e['诊断结论']} | 类型={e['检查类型']}")

print(f"\n📌 第二步: Rerank 重排序 (top-{RERANK_TOP_K})")
print("-" * 60)
try:
    rerank_results = rerank_documents(query, documents, top_n=RERANK_TOP_K)
    reranked_entities = []
    for rr in rerank_results:
        idx = rr.get("index", 0)
        rerank_score = rr.get("relevance_score", 0)
        entity = hit_entities[idx]
        entity["_rerank_score"] = rerank_score
        reranked_entities.append(entity)
        print(f"  原始位置 Top-{idx+1} → Rerank分数: {rerank_score:.4f}")
        print(f"  来源: {entity['source']}")
        print(f"  诊断: {entity['诊断结论']}")
except Exception as e:
    print(f"  Rerank失败({e})，使用向量检索结果")
    reranked_entities = hit_entities[:RERANK_TOP_K]
    for entity in reranked_entities:
        entity["_rerank_score"] = -1

print(f"\n📌 第三步: Prompt 拼接 (带Rerank分数)")
print("=" * 60)
contexts = []
for i, ent in enumerate(reranked_entities, 1):
    rerank_score = ent.get("_rerank_score", 0)
    contexts.append(f"【参考{i}】(Rerank相关性分数: {rerank_score:.4f}，来源: {ent['source']})\n{ent['text']}")

context_text = "\n\n".join(contexts)
print(context_text)

print("\n" + "=" * 60)
print(f"📌 第四步: 完整 Prompt 预览 (发给 LLM 的内容)")
print("=" * 60)

PROMPT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "prompt.md")
system_prompt = "你是一个医疗影像报告生成助手。"
if os.path.exists(PROMPT_FILE):
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

user_prompt = f"参考信息：\n{context_text}\n\n用户问题：{query}"

print(f"\n[System Prompt]: ({len(system_prompt)} 字符)")
print(system_prompt[:300] + "...")

print(f"\n[User Prompt]:")
print(user_prompt)

print("\n✅ 流程结束 — 以上就是 LLM 收到的完整输入")
client.close()