#!/usr/bin/env python3
"""
DeepSearcher API 客户端 — 正确的测试流程
流程：先加载内容到向量库 → 再查询已加载的知识
"""

import json
import sys
from typing import List, Optional

import requests


class DeepSearcherClient:
    """DeepSearcher API 客户端"""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")

    def set_provider_config(
        self, feature: str, provider: str, config: dict,
    ) -> dict:
        url = f"{self.base_url}/set-provider-config/"
        payload = {"feature": feature, "provider": provider, "config": config}
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def load_files(
        self,
        paths: List[str],
        collection_name: Optional[str] = None,
        collection_description: Optional[str] = None,
        batch_size: int = 256,
    ) -> dict:
        url = f"{self.base_url}/load-files/"
        payload = {"paths": paths, "batch_size": batch_size}
        if collection_name:
            payload["collection_name"] = collection_name
        if collection_description:
            payload["collection_description"] = collection_description
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def load_website(
        self,
        urls: List[str],
        collection_name: Optional[str] = None,
        collection_description: Optional[str] = None,
        batch_size: int = 256,
    ) -> dict:
        url = f"{self.base_url}/load-website/"
        payload = {"urls": urls, "batch_size": batch_size}
        if collection_name:
            payload["collection_name"] = collection_name
        if collection_description:
            payload["collection_description"] = collection_description
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def query(
        self, question: str, max_iter: int = 3,
    ) -> dict:
        url = f"{self.base_url}/query/"
        params = {"original_query": question, "max_iter": max_iter}
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


# ============================================================
# 正确的测试流程
# ============================================================
if __name__ == "__main__":
    BASE_URL = "http://220.154.3.166:8001"
    client = DeepSearcherClient(BASE_URL)

    print("=" * 60)
    print("DeepSearcher API 测试")
    print("=" * 60)

    # ---- 1. 基础验证：加载 example.com ----
    print("\n🌐 1. 加载 example.com 到知识库...")
    result = client.load_website(
        urls=["https://example.com"],
        collection_name="test_example",
    )
    print(f"   结果: {result}")

    # ---- 2. 查询刚加载的 example.com ----
    print("\n📝 2. 查询: example.com 网站的主要内容是什么？")
    result = client.query(
        "example.com 网站的主要内容是什么？",
        max_iter=2,
    )
    print(f"   答案: {result['result']}")
    print(f"   消耗 Token: {result['consume_token']}")

    # ---- 3. 测试知乎内容（需要手动保存为本地文件）----
    print("\n📁 3. 测试知乎内容（通过本地文件加载）")
    print("   提示：知乎有反爬机制，建议手动保存 HTML 后加载")
    print("   步骤：")
    print("     1. 浏览器打开知乎问题页面")
    print("     2. Ctrl+S 保存为完整 HTML")
    print("     3. docker cp 到容器：")
    print("        docker cp zhihu_agent.html deepsearcher:/app/data/zhihu_agent.html")
    print("     4. 取消下面代码的注释并运行")
    
    # 取消下面的注释来测试知乎内容
    # print("\n   加载知乎本地文件...")
    # result = client.load_files(
    #     paths=["/app/data/zhihu_agent.html"],
    #     collection_name="zhihu_agent_q",
    # )
    # print(f"   结果: {result}")
    #
    # print("\n   查询知乎内容...")
    # result = client.query(
    #     "这个知乎问题主要讨论了什么？关于 Agent 的核心观点是什么？",
    #     max_iter=2,
    # )
    # print(f"   答案: {result['result']}")
    # print(f"   消耗 Token: {result['consume_token']}")

    # ---- 4. 设置 Provider 配置（可选）----
    print("\n⚙️  4. 设置 Provider 配置（切换 LLM 模型）")
    print("   当前使用：qwen36_27b_lora @ 14.22.86.97:11001")
    print("   如需切换模型，取消下面代码的注释")
    
    # 取消下面的注释来切换模型
    # try:
    #     result = client.set_provider_config(
    #         feature="llm",
    #         provider="DashScope",
    #         config={"model": "qwen-plus"},
    #     )
    #     print(f"   结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
    # except Exception as e:
    #     print(f"   跳过（API Key 可能未配置）: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)