"""测试 SiliconFlow BAAI/bge-large-zh-v1.5 Embedding 服务是否可用。"""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))

EMBED_URL = "https://api.siliconflow.cn/v1"
MODEL = "BAAI/bge-large-zh-v1.5"
API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
ENCODING_FORMAT = "float"


def _headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def test_models():
    print("1. 测试 /v1/models 接口...")
    try:
        r = requests.get(f"{EMBED_URL}/models", headers=_headers(), timeout=10)
        print(f"   状态码: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            names = [m.get("id", "") for m in data.get("data", [])]
            matched = [n for n in names if "bge-large-zh" in n]
            print(f"   可用模型总数: {len(names)}")
            print(f"   匹配模型: {matched if matched else '未找到'}")
        else:
            print(f"   响应: {r.text[:300]}")
    except Exception as e:
        print(f"   连接失败: {e}")
    print()


def test_single_embedding():
    print("2. 测试单条文本向量化...")
    try:
        start = time.time()
        r = requests.post(
            f"{EMBED_URL}/embeddings",
            headers=_headers(),
            json={"model": MODEL, "input": "你好，这是一条测试文本", "encoding_format": ENCODING_FORMAT},
            timeout=30,
        )
        elapsed = time.time() - start
        print(f"   状态码: {r.status_code}")
        print(f"   耗时: {elapsed:.2f}s")
        if r.status_code == 200:
            data = r.json()
            emb = data["data"][0]["embedding"]
            print(f"   向量维度: {len(emb)}")
            print(f"   前5个值: {emb[:5]}")
            usage = data.get("usage", {})
            print(f"   Token 用量: prompt={usage.get('prompt_tokens', 'N/A')}, total={usage.get('total_tokens', 'N/A')}")
        else:
            print(f"   响应: {r.text[:300]}")
    except Exception as e:
        print(f"   请求失败: {e}")
    print()


def test_batch_embedding():
    print("3. 测试批量文本向量化 (3条)...")
    try:
        start = time.time()
        r = requests.post(
            f"{EMBED_URL}/embeddings",
            headers=_headers(),
            json={
                "model": MODEL,
                "input": ["CT头颅检查", "MRI腰椎检查", "X光胸部检查"],
                "encoding_format": ENCODING_FORMAT,
            },
            timeout=30,
        )
        elapsed = time.time() - start
        print(f"   状态码: {r.status_code}")
        print(f"   耗时: {elapsed:.2f}s")
        if r.status_code == 200:
            data = r.json()
            print(f"   返回条数: {len(data['data'])}")
            for d in data["data"]:
                print(f"   index={d['index']}, 维度={len(d['embedding'])}")
            usage = data.get("usage", {})
            print(f"   Token 用量: prompt={usage.get('prompt_tokens', 'N/A')}, total={usage.get('total_tokens', 'N/A')}")
        else:
            print(f"   响应: {r.text[:300]}")
    except Exception as e:
        print(f"   请求失败: {e}")
    print()


def test_larger_batch():
    print("4. 测试较大批次 (32条)...")
    texts = [f"测试文本编号{i}" for i in range(32)]
    try:
        start = time.time()
        r = requests.post(
            f"{EMBED_URL}/embeddings",
            headers=_headers(),
            json={"model": MODEL, "input": texts, "encoding_format": ENCODING_FORMAT},
            timeout=60,
        )
        elapsed = time.time() - start
        print(f"   状态码: {r.status_code}")
        print(f"   耗时: {elapsed:.2f}s")
        if r.status_code == 200:
            data = r.json()
            print(f"   返回条数: {len(data['data'])}")
            print(f"   每条耗时: {elapsed/32*1000:.0f}ms")
            usage = data.get("usage", {})
            print(f"   Token 用量: prompt={usage.get('prompt_tokens', 'N/A')}, total={usage.get('total_tokens', 'N/A')}")
        else:
            print(f"   响应: {r.text[:300]}")
    except Exception as e:
        print(f"   请求失败: {e}")
    print()


if __name__ == "__main__":
    if not API_KEY:
        print("⚠️  未设置 SILICONFLOW_API_KEY 环境变量，请先设置：")
        print("   Windows: set SILICONFLOW_API_KEY=sk-xxx")
        print("   Linux/Mac: export SILICONFLOW_API_KEY=sk-xxx")
        print()

    print(f"=== 测试 SiliconFlow BAAI/bge-large-zh-v1.5 服务 ===")
    print(f"地址: {EMBED_URL}")
    print(f"模型: {MODEL}")
    print()
    test_models()
    test_single_embedding()
    test_batch_embedding()
    test_larger_batch()
    print("=== 测试完成 ===")