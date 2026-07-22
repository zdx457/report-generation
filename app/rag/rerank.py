"""Rerank 精排模块。

调用 SiliconFlow Rerank API（Qwen3-VL-Reranker-8B），
对向量检索召回的候选文档进行 Cross-Encoder 精排。

用法：
    from rerank import rerank_documents, get_rerank_config

    results = rerank_documents("CT脑出血", ["文档1", "文档2"], top_n=1)
    config = get_rerank_config()
"""
import os

import requests
from dotenv import load_dotenv
from config import get_rerank_base_url, get_rerank_model, get_rerank_api_key, _normalize_base_url
from langsmith import traceable

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env"))


def _get_rerank_url() -> str:
    """获取 Rerank URL，自动补全 /rerank 路径"""
    return _normalize_base_url(get_rerank_base_url(), "/rerank")
RERANK_MODEL = get_rerank_model()
SILICONFLOW_API_KEY = get_rerank_api_key()


@traceable(run_type="tool", name="SiliconFlow_Rerank")
def rerank_documents(query, documents, top_n=3):
    """对候选文档进行 Rerank 精排。

    Args:
        query: 用户查询文本
        documents: 候选文档文本列表
        top_n: 返回前 N 个最相关文档

    Returns:
        list[dict]: 按 relevance_score 降序排列的结果列表，每项包含:
            - index: 原始 documents 列表中的索引
            - relevance_score: 相关性分数
            - document: {"text": 原始文档文本}（需 return_documents=True）

    Raises:
        requests.HTTPError: API 请求失败
        requests.Timeout: 请求超时（默认30秒）
    """
    headers = {"Content-Type": "application/json"}
    if SILICONFLOW_API_KEY:
        headers["Authorization"] = f"Bearer {SILICONFLOW_API_KEY}"
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "return_documents": True,
    }
    r = requests.post(_get_rerank_url(), headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("results", [])


def get_rerank_config():
    """获取当前 Rerank 配置信息。

    Returns:
        dict: 包含 rerank_url, rerank_model, api_key_configured 三个字段
    """
    return {
        "rerank_url": _get_rerank_url(),
        "rerank_model": RERANK_MODEL,
        "api_key_configured": bool(SILICONFLOW_API_KEY),
    }


if __name__ == "__main__":
    print("=== Rerank 模块测试 ===")
    config = get_rerank_config()
    print(f"  Rerank URL: {_get_rerank_url()}")
    print(f"  Rerank Model: {config['rerank_model']}")
    print(f"  API Key: {'已配置' if config['api_key_configured'] else '未配置'}")

    if config["api_key_configured"]:
        print("\n--- 测试 Rerank ---")
        query = "CT脑出血"
        docs = [
            "检查类型：CT\n部位：头颅\n诊断结论：脑出血（破入脑室）\n影像学表现：侧基底节区见高密度影...",
            "检查类型：CT\n部位：头颅\n诊断结论：脑出血\n影像学表现：侧基底节区见团块状高密度影...",
            "检查类型：MRI\n部位：腰椎\n诊断结论：椎间盘突出\n影像学表现：L4/5椎间盘向后突出...",
        ]
        results = rerank_documents(query, docs, top_n=2)
        print(f"  查询: {query}")
        print(f"  候选文档数: {len(docs)}")
        print(f"  Rerank Top-{len(results)}:")
        for i, r in enumerate(results):
            idx = r.get("index", -1)
            score = r.get("relevance_score", 0)
            diag = docs[idx].split("\n")[2] if idx < len(docs) else "N/A"
            print(f"    第{i+1}名: 原始位置={idx}, 分数={score:.4f}, {diag}")
    else:
        print("\n⚠️  API Key 未配置，跳过测试。请在 .env 中设置 SILICONFLOW_API_KEY。")