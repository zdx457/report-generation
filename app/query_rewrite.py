"""查询改写模块。

当用户输入过短或过于模糊时，使用 LLM 将查询扩展为更具体的描述，
提升多路召回的精准度。

对于过于模糊的输入（如仅输入"CT"，没有部位和诊断），返回追问提示，
引导用户补充信息。

用法：
    from query_rewrite import rewrite_query, needs_rewrite, is_too_vague, get_clarification

    if is_too_vague(user_input):
        reply = get_clarification(user_input)
    elif needs_rewrite(user_input):
        rewritten = rewrite_query(user_input)
    else:
        rewritten = user_input

触发条件：
- is_too_vague: 查询仅有检查类型，缺少部位和诊断（如"CT"、"MR"）
- needs_rewrite: 查询有类型+部位但无诊断（如"CT头颅"、"MR膝关节"）
"""
import json
import os

import requests

CHAT_URL = os.environ.get("CHAT_URL", "http://14.22.86.97:11001/v1/chat/completions")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen36_27b_lora")

KNOWN_TYPES = {"CT", "MR", "DR", "MRI", "X光", "X线"}

KNOWN_PARTS = {
    "头颅", "头颈部", "胸部", "腹部", "盆腔", "脊柱",
    "四肢及关节", "血管", "颈椎", "腰椎", "胸椎",
}

PART_ALIASES = {
    "头": "头颅",
    "脑": "头颅",
    "颅脑": "头颅",
    "头部": "头颅",
    "颈": "头颈部",
    "颈部": "头颈部",
    "胸": "胸部",
    "胸廓": "胸部",
    "腹": "腹部",
    "盆腔": "盆腔",
    "脊柱": "脊柱",
    "脊椎": "脊柱",
    "四肢": "四肢及关节",
    "关节": "四肢及关节",
    "血管": "血管",
}

CLARIFICATION_TEMPLATES = {
    "CT": "您想查询哪个部位的CT检查？例如：\n• CT头颅（脑出血、脑梗死等）\n• CT胸部（肺气肿、肺炎等）\n• CT腹部（肝脏、肾脏等）\n• CT血管（CTA相关）",
    "MR": "您想查询哪个部位的MR检查？例如：\n• MR头颅（脑肿瘤、脑梗死等）\n• MR膝关节（半月板、韧带等）\n• MR脊柱（椎间盘突出等）",
    "DR": "您想查询哪个部位的DR检查？例如：\n• DR胸部（肺部感染等）\n• DR四肢及关节（骨折等）\n• DR脊柱（骨质增生等）",
    "MRI": "您想查询哪个部位的MRI检查？例如：\n• MRI头颅（脑肿瘤等）\n• MRI脊柱（椎间盘等）",
    "X光": "您想查询哪个部位的X光检查？例如：\n• X光胸部\n• X光四肢及关节",
    "X线": "您想查询哪个部位的X线检查？例如：\n• X线胸部\n• X线四肢及关节",
}


def _parse_query_components(query):
    """解析查询中的检查类型、部位和诊断关键词。

    Returns:
        dict: {检查类型: str, 部位: str, has_diag: bool}
    """
    check_type = ""
    for t in KNOWN_TYPES:
        if t in query:
            check_type = t
            break
    if "MRI" in query and not check_type:
        check_type = "MR"

    part = ""
    for alias, standard in PART_ALIASES.items():
        if alias in query:
            part = standard
            break
    if not part:
        for p in KNOWN_PARTS:
            if p in query:
                part = p
                break

    cleaned = query
    for t in KNOWN_TYPES:
        cleaned = cleaned.replace(t, "")
    if part:
        for alias, standard in PART_ALIASES.items():
            if standard == part and alias in cleaned:
                cleaned = cleaned.replace(alias, "")
        cleaned = cleaned.replace(part, "")
    cleaned = cleaned.strip()

    has_diag = len(cleaned) >= 2

    return {"检查类型": check_type, "部位": part, "has_diag": has_diag}


def is_too_vague(query):
    """判断查询是否过于模糊，需要追问用户。

    当查询仅有检查类型，缺少部位和诊断时返回 True。
    例如："CT"、"MR" → True
         "CT头颅"、"脑出血" → False

    Args:
        query: 用户输入的查询文本

    Returns:
        bool: True 表示查询过于模糊，应追问用户
    """
    components = _parse_query_components(query)
    has_type = bool(components["检查类型"])
    has_part = bool(components["部位"])
    has_diag = components["has_diag"]

    if has_type and not has_part and not has_diag:
        return True

    if not has_type and not has_part and not has_diag:
        return True

    return False


def get_clarification(query):
    """根据模糊查询生成追问提示。

    Args:
        query: 用户输入的查询文本

    Returns:
        str: 追问提示文本，引导用户补充部位或诊断信息
    """
    components = _parse_query_components(query)
    check_type = components["检查类型"]

    if check_type and check_type in CLARIFICATION_TEMPLATES:
        return CLARIFICATION_TEMPLATES[check_type]

    if not check_type and not components["部位"]:
        return "请提供更具体的检查信息，例如：\n• CT头颅脑出血\n• MR膝关节半月板损伤\n• DR胸部肺部感染"

    return "请补充检查部位或诊断信息，例如：CT头颅、MR膝关节等。"


REWRITE_SYSTEM_PROMPT = """你是一个医疗影像报告检索系统的查询改写助手。

任务：将用户简短、模糊的查询扩展为更具体、更完整的检索描述。

规则：
1. 保留用户原始意图，不要改变检查类型和部位
2. 补充该检查类型/部位常见的诊断结论关键词
3. 补充影像学表现、影像学意见等检索相关术语
4. 输出格式：直接输出扩展后的查询文本，不要解释
5. 扩展后长度控制在 20-40 字
6. 不要编造用户没有提到的具体诊断

示例：
- 输入: "头颅" → 输出: "头颅CT MR检查 影像学表现 诊断结论"
- 输入: "CT脑出血" → 输出: "CT脑出血 影像学表现 诊断结论 破入脑室"
- 输入: "MR膝关节" → 输出: "MR膝关节检查 半月板 韧带 影像学表现 诊断结论"
- 输入: "CT弥漫性肺气肿" → 输出: "CT弥漫性肺气肿 肺部影像学表现 诊断结论\""""


def needs_rewrite(query):
    """判断查询是否需要改写。

    Args:
        query: 用户输入的查询文本

    Returns:
        bool: True 表示需要改写
    """
    if is_too_vague(query):
        return False

    components = _parse_query_components(query)
    has_type = bool(components["检查类型"])
    has_part = bool(components["部位"])
    has_diag = components["has_diag"]

    if (has_type or has_part) and not has_diag:
        return True

    return False


def rewrite_query(query, chat_url=None, chat_model=None, timeout=10):
    """使用 LLM 改写查询。

    Args:
        query: 用户输入的查询文本
        chat_url: LLM API 地址（默认从环境变量读取）
        chat_model: LLM 模型名（默认从环境变量读取）
        timeout: 超时时间（秒）

    Returns:
        str: 改写后的查询文本。如果改写失败，返回原始查询。
    """
    url = chat_url or CHAT_URL
    model = chat_model or CHAT_MODEL

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        "max_tokens": 128,
        "temperature": 0.3,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        rewritten = data["choices"][0]["message"]["content"].strip()
        if rewritten and len(rewritten) >= len(query):
            return rewritten
        return query
    except Exception:
        return query


if __name__ == "__main__":
    test_queries = [
        "CT",
        "头颅",
        "CT脑出血",
        "脑出血",
        "MR膝关节",
        "CT弥漫性肺气肿",
        "动脉瘤（宽颈多发）",
        "腹部CTA正常",
        "CT颅脑平扫脑梗死",
        "MR",
        "DR",
    ]

    print("查询改写测试")
    print("=" * 70)

    for q in test_queries:
        vague = is_too_vague(q)
        need = needs_rewrite(q)
        if vague:
            clarification = get_clarification(q)
            status = f"⚠️ 过于模糊 → 追问: {clarification[:50]}..."
        elif need:
            rewritten = rewrite_query(q)
            status = f"✏️ 改写 → {rewritten}"
        else:
            status = "✅ 无需改写"
        print(f"  {q:20s} | 模糊={str(vague):5s} | 改写={str(need):5s} | {status}")