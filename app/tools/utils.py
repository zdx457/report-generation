"""工具模块公共工具函数

提供所有工具共用的工具函数，减少代码重复。
"""

import json
import re
import logging
from typing import Any, Dict

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

    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("无法解析 JSON: %s...", text[:200])
    return {"error": "JSON 解析失败", "raw": text[:500]}