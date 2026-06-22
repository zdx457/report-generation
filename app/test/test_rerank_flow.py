"""完整流程测试: 向量检索 → Rerank → 最终给LLM的内容"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

from pymilvus import MilvusClient
import requests

EMBED_URL = os.environ.get("EMBED_URL", "http://14.22.83.225:11002/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
RERANK_URL = os.environ.get("RERANK_URL", "https://api.siliconflow.cn/v1/rerank")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "Qwen/Qwen3-VL-Reranker-8B")
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
DB_PATH = os.path.join("app", "milvus_lite.db")
COLLECTION_NAME = "report_slices"

client = MilvusClient(uri=DB_PATH)
client.load_collection(COLLECTION_NAME)

query = "CT脑出血"

print("=" * 60)
print(f"用户问题: {query}")
print("=" * 60)

payload = {"model": EMBED_MODEL, "input": [query]}
resp = requests.post(EMBED_URL, json=payload, timeout=30)
query_vec = resp.json()["data"][0]["embedding"]

print("\n📌 第一步: 向量检索 (top-5)")
print("-" * 60)
hits = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vec],
    limit=5,
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

print(f"\n📌 第二步: Rerank 重排序 (模型: {RERANK_MODEL}, top-1)")
print("-" * 60)
headers = {"Content-Type": "application/json"}
if SILICONFLOW_API_KEY:
    headers["Authorization"] = f"Bearer {SILICONFLOW_API_KEY}"

rerank_payload = {
    "model": RERANK_MODEL,
    "query": query,
    "documents": documents,
    "top_n": 1,
    "return_documents": True,
}
r = requests.post(RERANK_URL, headers=headers, json=rerank_payload, timeout=30)
rerank_data = r.json()
rerank_results = rerank_data.get("results", [])

reranked_entities = []
for rr in rerank_results:
    idx = rr.get("index", 0)
    rerank_score = rr.get("relevance_score", 0)
    entity = hit_entities[idx]
    entity["_rerank_score"] = rerank_score
    reranked_entities.append(entity)
    print(f"  原始位置 Top-{idx+1} → Rerank 第1名")
    print(f"  Rerank分数: {rerank_score:.4f}")
    print(f"  来源: {entity['source']}")
    print(f"  诊断: {entity['诊断结论']}")

print(f"\n📌 第三步: 最终给 LLM 的参考信息 (Prompt 拼接结果)")
print("=" * 60)
ref_parts = []
for i, ent in enumerate(reranked_entities, 1):
    ref_text = (
        f"【参考{i}】(来源: {ent['source']})\n"
        f"{ent['text']}"
    )
    ref_parts.append(ref_text)

reference_info = "\n\n".join(ref_parts)
print(reference_info)

print("\n" + "=" * 60)
print(f"📌 第四步: 完整 Prompt 预览 (发给 LLM 的内容)")
print("=" * 60)
system_prompt = """你是一个专业的医学影像报告生成助手。根据提供的参考信息和用户问题，生成结构化的医学影像报告。

请严格按照以下格式输出：

## 一、影像学表现
（详细描述影像学所见）

## 二、诊断意见
（给出明确的诊断结论）"""

user_prompt = f"""参考信息：
{reference_info}

用户问题：{query}"""

print("\n[System Prompt]:")
print(system_prompt[:200] + "...")

print("\n[User Prompt]:")
print(user_prompt)

print("\n✅ 流程结束 — 以上就是 LLM 收到的完整输入")
client.close()