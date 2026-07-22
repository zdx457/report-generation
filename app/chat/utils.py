"""无状态的工具函数"""

import json
import logging
import re
import time
from functools import wraps

import requests

logger = logging.getLogger(__name__)


# =============================================================================
# 重试装饰器
# =============================================================================
def retry(max_attempts=3, delay=2, exceptions=(requests.RequestException,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(f"{e}，{delay}s 后重试 ({attempt+1}/{max_attempts})")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


# =============================================================================
# JSON 提取工具
# =============================================================================
def _extract_json(text):
    """从 LLM 输出中提取 JSON"""
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

    logger.warning(f"无法解析 JSON: {text[:200]}...")
    return {"error": "JSON 解析失败", "raw": text[:500]}


# =============================================================================
# JSON 报告 → Markdown 展示文本
# =============================================================================
def json_to_display(report_json):
    """将 JSON 报告转换为 Markdown 展示文本

    支持新格式 (results 数组) 和旧格式 (影像学表现/诊断意见 字典) 兼容。
    """
    if isinstance(report_json, str):
        return report_json.replace(r'\n', '\n').replace(r'\t', '\t')

    if not isinstance(report_json, dict) or "error" in report_json:
        raw = report_json.get("raw", str(report_json))
        return raw.replace(r'\n', '\n').replace(r'\t', '\t')

    lines = []

    # 新格式: results 数组 [{影像学表现, 诊断意见}, ...]
    results = report_json.get("results", [])
    if results:
        lines.append("## 一、影像学表现")
        lines.append("")
        for item in results:
            imaging = item.get("影像学表现", "")
            if imaging:
                for line in str(imaging).replace(r'\n', '\n').splitlines():
                    lines.append(line)
                lines.append("")

        lines.append("## 二、诊断意见")
        lines.append("")
        for item in results:
            diagnosis = item.get("诊断意见", "")
            if diagnosis:
                for line in str(diagnosis).replace(r'\n', '\n').splitlines():
                    lines.append(line)
                lines.append("")
    else:
        # 兼容旧格式: {影像学表现: {病变名称: 描述}, 诊断意见: {病变名称: 意见}}
        imaging = report_json.get("影像学表现", {})
        if imaging:
            lines.append("## 一、影像学表现")
            lines.append("")
            if isinstance(imaging, dict):
                for name, desc in imaging.items():
                    for line in str(desc).replace(r'\n', '\n').splitlines():
                        lines.append(line)
                    lines.append("")
            else:
                for line in str(imaging).replace(r'\n', '\n').splitlines():
                    lines.append(line)
                lines.append("")

        diagnosis = report_json.get("诊断意见", {})
        if diagnosis:
            lines.append("## 二、诊断意见")
            lines.append("")
            if isinstance(diagnosis, dict):
                for name, opinion in diagnosis.items():
                    for line in str(opinion).replace(r'\n', '\n').splitlines():
                        lines.append(line)
                    lines.append("")
            else:
                for line in str(diagnosis).replace(r'\n', '\n').splitlines():
                    lines.append(line)
                lines.append("")

    return "\n".join(lines).strip()
