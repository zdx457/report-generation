"""工具模块公共工具函数

提供所有工具共用的工具函数，减少代码重复。
"""

import json
import re
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def extract_json(text: str) -> Dict[str, Any]:
    """从 LLM 输出中提取 JSON

    先尝试匹配 ```json 代码块，然后尝试匹配 {} 包裹的 JSON。
    解析失败时返回 {"error": "...", "raw": "..."}, 调用方负责处理。

    Args:
        text: LLM 输出文本

    Returns:
        解析后的 JSON dict，或 error dict
    """
    text = text.strip()

    # 策略1：提取 ```json 代码块
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # 策略2：用括号计数提取最外层 JSON 对象
    json_str = _extract_outermost_json(text)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # 策略3：尝试修复常见问题后重试
    if json_str:
        try:
            fixed = _fix_json(json_str)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    # 策略4：尝试直接解析整个文本（可能只有 JSON）
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    logger.warning("无法解析 JSON: %s...", text[:200])
    return {"error": "JSON 解析失败", "raw": text[:500]}


def _extract_outermost_json(text: str) -> Optional[str]:
    """用括号计数提取最外层 {} 包裹的 JSON"""
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _fix_json(text: str) -> str:
    """修复常见的 JSON 格式问题"""
    # 移除尾随逗号（在 } 或 ] 之前）
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    # 将单引号替换为双引号（但保留字符串内的转义）
    # 简单处理：不在字符串内的单引号 → 双引号
    return text