"""
DeepSearcher API 测试脚本
服务器地址：http://14.22.83.225:8001
"""

import requests

BASE_URL = "http://14.22.83.225:8001"


def load_files(paths, collection_name=None, collection_description=None):
    """加载本地文件到向量库"""
    data = {"paths": paths}
    if collection_name:
        data["collection_name"] = collection_name
    if collection_description:
        data["collection_description"] = collection_description

    resp = requests.post(f"{BASE_URL}/load-files/", json=data)
    print(f"[加载文件] 状态码: {resp.status_code}")
    print(f"[加载文件] 结果: {resp.json()}")
    return resp.json()


def load_website(urls, collection_name=None, collection_description=None):
    """抓取网页内容到向量库"""
    data = {"urls": urls}
    if collection_name:
        data["collection_name"] = collection_name
    if collection_description:
        data["collection_description"] = collection_description

    resp = requests.post(f"{BASE_URL}/load-website/", json=data)
    print(f"[加载网页] 状态码: {resp.status_code}")
    print(f"[加载网页] 结果: {resp.json()}")
    return resp.json()


def query(question, max_iter=3):
    """查询 DeepSearcher"""
    params = {"original_query": question, "max_iter": max_iter}
    resp = requests.get(f"{BASE_URL}/query/", params=params)
    print(f"[查询] 状态码: {resp.status_code}")
    if resp.status_code == 200:
        result = resp.json()
        print(f"[消耗Token] {result.get('consume_token', 'N/A')}")
        print(f"\n{'='*60}")
        print(f"问题: {question}")
        print(f"{'='*60}")
        print(result["result"])
        print(f"{'='*60}\n")
        return result
    else:
        print(f"[错误] {resp.text}")
        return None


if __name__ == "__main__":
    load_website("https://www.zhihu.com/question/661759314/answer/2043451854626063339")
    query("什么是agent？")