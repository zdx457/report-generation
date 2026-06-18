#!/usr/bin/env python3
"""
DeepSearcher API 客户端
调用 deepsearcher 的 4 个接口
"""

import json
import sys
from typing import Optional

import requests


class DeepSearcherClient:
    """DeepSearcher API 客户端"""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")

    # ============================================================
    # 1. POST /set-provider-config/  — 设置 Provider 配置
    # ============================================================
    def set_provider_config(
        self,
        feature: str,
        provider: str,
        config: dict,
    ) -> dict:
        """
        设置模型/Provider 配置（动态切换 LLM、Embedding、向量库等）。

        Args:
            feature: 功能类型，如 "llm", "embedding", "vector_db"
            provider: Provider 名称，如 "OpenAI", "DashScope"
            config: 配置参数，如 {"model": "qwen-plus", "base_url": "..."}

        Returns:
            成功返回 {"message": "Provider config set successfully", ...}

        示例:
            client.set_provider_config("llm", "OpenAI", {
                "model": "gpt-4o",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-xxx"
            })
        """
        url = f"{self.base_url}/set-provider-config/"
        payload = {
            "feature": feature,
            "provider": provider,
            "config": config,
        }
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    # ============================================================
    # 2. POST /load-files/  — 加载本地文件
    # ============================================================
    def load_files(
        self,
        paths: "list[str]",
        collection_name: Optional[str] = None,
        collection_description: Optional[str] = None,
        batch_size: int = 256,
    ) -> dict:
        """
        加载本地文件（PDF、TXT、Markdown 等）到向量数据库。

        Args:
            paths: 文件路径列表，**注意：路径是容器内的路径**
            collection_name: 可选，集合名称
            collection_description: 可选，集合描述
            batch_size: 批次大小，默认 256

        Returns:
            成功返回 {"message": "Local files loaded successfully."}

        示例:
            # 先 docker cp 文件到容器
            # docker cp doc.pdf deepsearcher:/app/data/doc.pdf
            client.load_files(["/app/data/doc.pdf"])
        """
        url = f"{self.base_url}/load-files/"
        payload = {
            "paths": paths,
            "batch_size": batch_size,
        }
        if collection_name:
            payload["collection_name"] = collection_name
        if collection_description:
            payload["collection_description"] = collection_description

        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    # ============================================================
    # 3. POST /load-website/  — 加载网站
    # ============================================================
    def load_website(
        self,
        urls: "list[str]",
        collection_name: Optional[str] = None,
        collection_description: Optional[str] = None,
        batch_size: int = 256,
    ) -> dict:
        """
        爬取网站并加载到向量数据库。

        Args:
            urls: URL 列表
            collection_name: 可选，集合名称
            collection_description: 可选，集合描述
            batch_size: 批次大小，默认 256

        Returns:
            成功返回 {"message": "Website loaded successfully."}

        示例:
            client.load_website([
                "https://example.com",
                "https://docs.example.com/getting-started",
            ])
        """
        url = f"{self.base_url}/load-website/"
        payload = {
            "urls": urls,
            "batch_size": batch_size,
        }
        if collection_name:
            payload["collection_name"] = collection_name
        if collection_description:
            payload["collection_description"] = collection_description

        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    # ============================================================
    # 4. GET /query/  — 查询搜索
    # ============================================================
    def query(
        self,
        question: str,
        max_iter: int = 3,
    ) -> dict:
        """
        向向量数据库发起查询，DeepSearch 会自动分解问题、迭代搜索、生成答案。

        Args:
            question: 你的问题
            max_iter: 最大迭代次数，默认 3

        Returns:
            {"result": "答案文本", "consume_token": 数字}

        示例:
            resp = client.query("什么是 AI Agent？")
            print(resp["result"])
        """
        url = f"{self.base_url}/query/"
        params = {
            "original_query": question,
            "max_iter": max_iter,
        }
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    BASE_URL = "http://220.154.3.166:8001/"  ## 外部调用
    client = DeepSearcherClient(BASE_URL)

    print("=" * 60)
    print("DeepSearcher API 测试")
    print("=" * 60)

    # ---- 示例 1: 查询 ----
    print("\n📝 1. 查询: https://www.zhihu.com/question/8248918506 是什么？")
    result = client.query("agent是什么", max_iter=2)
    print(f"   答案: {result['result']}")
    print(f"   消耗 Token: {result['consume_token']}")

    # ---- 示例 2: 加载本地文件 ----
    print("\n📁 2. 加载本地文件（需要先 docker cp 文件到容器）")
    print("   docker cp /path/to/doc.pdf deepsearcher:/app/data/doc.pdf")
    try:
        result = client.load_files(
            paths=["/app/data/test.txt"],
            collection_name="my_docs",
        )
        print(f"   结果: {result}")
    except Exception as e:
        print(f"   跳过（文件不存在）: {e}")

    # ---- 示例 3: 加载网站 ----
    print("\n🌐 3. 加载网站: example.com")
    result = client.load_website(
        urls=["https://example.com"],
        collection_name="example_site",
    )
    print(f"   结果: {result}")

    # ---- 示例 4: 设置 Provider 配置 ----
    print("\n⚙️  4. 设置 Provider 配置（切换 LLM 模型）")
    # 示例：切换到 DashScope 的模型（需要 DASHSCOPE_API_KEY 已配置）
    try:
        result = client.set_provider_config(
            feature="llm",
            provider="DashScope",
            config={"model": "qwen-plus"},
        )
        print(f"   结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
    except Exception as e:
        print(f"   跳过（API Key 可能未配置）: {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
