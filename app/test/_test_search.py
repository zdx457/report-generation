"""快速测试：输入'CT脑出血'会检索出什么"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

from pymilvus import MilvusClient
import requests

EMBED_URL = os.environ.get("EMBED_URL", "http://14.22.83.225:11002/v1/embeddings")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "milvus_lite.db")
COLLECTION_NAME = "report_slices"

client = MilvusClient(uri=DB_PATH)
client.load_collection(COLLECTION_NAME)

results = client.query(
    COLLECTION_NAME,
    filter='诊断结论 like "脑出血%"',
    output_fields=["source", "text", "诊断结论", "检查类型", "部位"],
    limit=20,
)
print(f"=== 库中包含'脑出血'的记录: {len(results)} 条 ===")
for r in results:
    print(f"  [{r['source']}] 类型={r['检查类型']} 部位={r['部位']} 诊断={r['诊断结论']}")

print()
print("=== 向量检索: 'CT脑出血' (top-5) ===")
payload = {"model": EMBED_MODEL, "input": ["CT脑出血"]}
resp = requests.post(EMBED_URL, json=payload, timeout=30)
query_vec = resp.json()["data"][0]["embedding"]

hits = client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vec],
    limit=5,
    output_fields=["source", "text", "诊断结论", "检查类型", "部位"],
)

for i, hit in enumerate(hits[0], 1):
    e = hit["entity"]
    print(f"  Top-{i}: 相似度={hit['distance']:.4f}")
    print(f"         来源: {e['source']}")
    print(f"         类型: {e['检查类型']} | 部位: {e['部位']} | 诊断: {e['诊断结论']}")
    text_preview = e["text"][:150].replace("\n", " | ")
    print(f"         文本预览: {text_preview}...")
    print()

client.close()