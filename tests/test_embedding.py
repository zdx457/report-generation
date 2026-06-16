"""测试 bge-m3 embedding 服务是否可用。"""
import requests
import time

EMBED_URL = "http://14.22.83.225:11002/v1"
MODEL = "bge-m3"


def test_models():
    print("1. 测试 /v1/models 接口...")
    try:
        r = requests.get(f"{EMBED_URL}/models", timeout=10)
        print(f"   状态码: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"   可用模型: {data}")
        else:
            print(f"   响应: {r.text[:200]}")
    except Exception as e:
        print(f"   连接失败: {e}")
    print()


def test_single_embedding():
    print("2. 测试单条文本向量化...")
    try:
        start = time.time()
        r = requests.post(
            f"{EMBED_URL}/embeddings",
            json={"model": MODEL, "input": "你好，这是一条测试文本"},
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
            json={
                "model": MODEL,
                "input": ["CT头颅检查", "MRI腰椎检查", "X光胸部检查"],
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
            json={"model": MODEL, "input": texts},
            timeout=60,
        )
        elapsed = time.time() - start
        print(f"   状态码: {r.status_code}")
        print(f"   耗时: {elapsed:.2f}s")
        if r.status_code == 200:
            data = r.json()
            print(f"   返回条数: {len(data['data'])}")
            print(f"   每条耗时: {elapsed/32*1000:.0f}ms")
        else:
            print(f"   响应: {r.text[:300]}")
    except Exception as e:
        print(f"   请求失败: {e}")
    print()


if __name__ == "__main__":
    print(f"=== 测试 bge-m3 Embedding 服务 ===")
    print(f"地址: {EMBED_URL}")
    print(f"模型: {MODEL}")
    print()
    test_models()
    test_single_embedding()
    test_batch_embedding()
    test_larger_batch()
    print("=== 测试完成 ===")