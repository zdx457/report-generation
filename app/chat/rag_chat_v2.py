"""
rag_chat_v2.py - Tool Calling 架构版本

使用 OpenAI 兼容的 Tool Calling 机制，让 LLM 自主决定调用哪个工具，
替代硬编码的意图分类和 if/elif 路由。

核心流程：
1. 预处理（实体提取、意图检测、上下文消解）
2. 构建消息，传入 tools schema
3. LLM 自主决定调用工具或直接回复
4. 执行工具，将结果返回 LLM 生成最终回复
"""

import json
import logging
import os
import re
import sys
import time
import uuid
import shutil
import asyncio
import queue
import traceback
from functools import wraps
from typing import Optional, Callable, Any

import yaml
import requests
from openai import APIConnectionError, APITimeoutError, APIStatusError

# 全局歧义缓存，跨请求存活（按 session_id 索引）
_ambiguity_cache = {}
from dotenv import load_dotenv
from pymilvus import MilvusClient

from memory.entity_tracker import EntityTracker
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.retriever import MemoryRetriever
from memory.session_store import SessionStore
from tools.registry import ToolRegistry, ToolResult
from prompt.builder import PromptBuilder
from rag.rerank import rerank_documents, get_rerank_config
from rag.retrieval import multi_recall
from rag.query_rewrite import (
    parse_query_keywords,
    standardize_query,
    needs_rewrite,
    rewrite_query,
)
from config import (
    get_embed_base_url, get_embed_model, get_embed_api_key,
    get_llm_base_url, get_llm_model, get_llm_api_key,
    get_llm_max_tokens, get_llm_temperature,
    get_db_path, get_collection_name,
    get_rag_top_k, get_rerank_top_k,
    get_rerank_api_key,
    get_max_rounds,
    reload_config,
    get_llm_client,
    get_embed_client,
)
from prompt import load_prompt

from data_pipeline.build_vector_db import build_db
from data_pipeline.extract_metadata import extract_metadata
from data_pipeline.xlsx_slicer import process_file

# ── Web 模式依赖（可选） ──
try:
    from fastapi import FastAPI, Request, UploadFile, File, Query
    from fastapi.responses import StreamingResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
    from typing import Optional, List, Dict, Any
    import uvicorn
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False

# =============================================================================
# API 请求/响应模型
# =============================================================================
class ChatRequest(BaseModel):
    """对话请求"""
    query: str = Field(..., description="用户输入内容")
    session_id: str = Field(default="default", description="会话 ID")
    selected_diagnosis: Optional[str] = Field(default=None, description="选择的诊断（歧义场景）")

class ConfigSaveRequest(BaseModel):
    """保存配置请求"""
    config: Dict[str, Any] = Field(..., description="完整的配置对象")

class TestModelRequest(BaseModel):
    """测试模型连接请求"""
    params: Dict[str, Any] = Field(..., alias="model_config", description="模型配置（包含 base_url, model, api_key）")
    model_type: str = Field(default="llms", description="模型类型：llms/embeddings/reranks")

class KBBuildRequest(BaseModel):
    """构建知识库请求"""
    rebuild: bool = Field(default=False, description="是否重建（清空现有数据）")
    batch_size: int = Field(default=16, description="批次大小")

class SessionTitleUpdate(BaseModel):
    """更新会话标题请求"""
    title: str = Field(..., description="新的会话标题")

class KBStatusResponse(BaseModel):
    """知识库状态响应"""
    total: int = Field(description="知识库文档总数")
    md_count: int = Field(description="MD 切片文件数")
    db_path: str = Field(description="数据库路径")
    metadata_exists: bool = Field(description="metadata.json 是否存在")

class KBFileInfo(BaseModel):
    """知识库文件信息"""
    name: str
    slice_count: int
    size: int
    mtime: float

class KBFilesResponse(BaseModel):
    files: List[KBFileInfo]

class ConfigResponse(BaseModel):
    config: Dict[str, Any]
    path: str

class TestModelResponse(BaseModel):
    success: bool
    message: str

class SessionResponse(BaseModel):
    session_id: str

class SessionsListResponse(BaseModel):
    sessions: List[Dict[str, Any]]

class SessionInfoResponse(BaseModel):
    current_turns: int
    entity_slots: Dict[str, Any]
    has_last_report: bool

class MemoryTurn(BaseModel):
    round: int
    user: str
    assistant: str

class MemoryResponse(BaseModel):
    turns: List[MemoryTurn]
    entities: Dict[str, Any]
    summaries: Any
    current_turns: int
    total_turns: int
    max_rounds: int

class ThinkingResponse(BaseModel):
    thinking: List[Any]

class ClearSessionRequest(BaseModel):
    """清空会话请求"""
    session_id: str = Field(default="default", description="会话 ID")

class StatusResponse(BaseModel):
    status: str = "ok"
    message: Optional[str] = None

# =============================================================================
# 配置
# =============================================================================
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)

EMBED_URL = get_embed_base_url()
EMBED_MODEL = get_embed_model()
EMBED_API_KEY = get_embed_api_key()
CHAT_URL = get_llm_base_url()
CHAT_MODEL = get_llm_model()
CHAT_API_KEY = get_llm_api_key()
DB_PATH = get_db_path()
COLLECTION_NAME = get_collection_name()

RAG_TOP_K = get_rag_top_k()
RERANK_TOP_K = get_rerank_top_k()

# =============================================================================
# 工具函数
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


def load_system_prompt():
    return load_prompt("report_generation")


def get_embedding(text):
    """使用 OpenAI SDK 获取文本向量"""
    try:
        client = get_embed_client()
        response = client.embeddings.create(model=EMBED_MODEL, input=text)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"[get_embedding] 请求失败: {e}", exc_info=True)
        raise


def _estimate_tokens(messages):
    total_chars = sum(len(msg.get("content", "") or "") for msg in messages)
    return total_chars, total_chars // 2


async def chat_stream(messages, max_tokens=2048, temperature=0.3, _emit=None, debug=False, caller="chat_stream"):
    """异步流式 LLM 调用，使用 OpenAI SDK。

    Args:
        messages: 消息列表
        max_tokens: 最大 token 数
        temperature: 温度参数
        _emit: SSE 事件发射器
        debug: 是否打印调试信息
        caller: 调用者标识

    Returns:
        str: 完整生成的文本
    """
    total_chars, est_tokens = _estimate_tokens(messages)
    if debug:
        logger.info(f"[{caller}] 发送请求: {len(messages)} messages, {total_chars} chars, 估算 ~{est_tokens} tokens")

    try:
        client = get_llm_client()
        stream = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        full_text = ""
        token_count = 0
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = delta.content
            if content:
                token_count += 1
                if _emit:
                    _emit("token", {"content": content})
                full_text += content

        if debug:
            logger.info(f"[{caller}] 完成: 收到 {token_count} tokens, {len(full_text)} chars")
        return full_text.strip()

    except APIConnectionError as e:
        logger.error(f"[{caller}] LLM 连接失败: {e}", exc_info=True)
        raise
    except APITimeoutError:
        logger.error(f"[{caller}] LLM 请求超时", exc_info=True)
        raise
    except APIStatusError as e:
        logger.error(f"[{caller}] LLM 返回错误: HTTP {e.status_code} - {e.message}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"[{caller}] 未知错误: {e}", exc_info=True)
        raise

@retry()
def rerank_with_retry(query, documents, top_n=3):
    return rerank_documents(query, documents, top_n=top_n)


def search_reports(query, top_k=RAG_TOP_K, rerank_top_k=RERANK_TOP_K, client=None, _emit=None):
    """RAG 检索：多路召回 + Rerank，返回 (格式化文本, reranked_entities)"""
    if rerank_top_k > top_k:
        rerank_top_k = top_k

    query_vec = get_embedding(query)
    keywords = parse_query_keywords(query)

    candidates, recall_details = multi_recall(query_vec, keywords, top_k=top_k, client=client, return_details=True)

    if not candidates:
        return "未检索到相关报告。", []

    if _emit:
        vec_results = recall_details.get("vector", [])
        meta_results = recall_details.get("metadata", [])
        kw_results = recall_details.get("keyword", [])
        total_before = len(vec_results) + len(meta_results) + len(kw_results)
        _emit("recall", {
            "vector_count": len(vec_results),
            "metadata_count": len(meta_results),
            "keyword_count": len(kw_results),
            "total_before": total_before,
            "total_after": len(candidates),
            "dedup": total_before - len(candidates),
        })

    documents = [e["text"] for e in candidates]

    reranked_entities = []
    try:
        rerank_results = rerank_with_retry(query, documents, top_n=rerank_top_k)
        for rr in rerank_results:
            idx = rr.get("index", 0)
            if idx < len(candidates):
                rerank_score = rr.get("relevance_score", 0)
                entity = candidates[idx]
                entity["_rerank_score"] = rerank_score
                reranked_entities.append(entity)
    except Exception:
        logger.warning("重排序失败，使用原始候选列表", exc_info=True)
        reranked_entities = candidates[:rerank_top_k]
    if _emit:
        _emit("rerank", {
            "results": [
                {
                    "index": i,
                    "score": e.get("_rerank_score", 0),
                    "source": e.get("source", ""),
                    "diagnosis": e.get("诊断结论", ""),
                    "text": e.get("text", ""),
                }
                for i, e in enumerate(reranked_entities)
            ]
        })

    parts = []
    for i, entity in enumerate(reranked_entities, 1):
        score = entity.get("_rerank_score", 0)
        parts.append(f"### 参考{i}（Rerank分数: {score:.4f}）\n{entity['text']}\n")

    return "\n".join(parts), reranked_entities


# =============================================================================
# 从 prompt 目录加载提示词
# =============================================================================
STRUCTURE_PROMPT = load_prompt("structure")
EDIT_PROMPT = load_prompt("edit")
REFINE_PROMPT = load_prompt("refine")
CHAT_SYSTEM_PROMPT = load_prompt("chat")

logger = logging.getLogger(__name__)

# 默认日志格式（Web 启动时 web_main 会重新配置，CLI 启动时 main 会重新配置）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# =============================================================================
# Tool Calling: 非流式 LLM 调用（支持 tools 参数）
# =============================================================================
async def chat_with_tools(messages, tools=None, max_tokens=512, temperature=0.3, debug=False):
    """非流式 LLM 调用，支持 Tool Calling，使用 OpenAI SDK。

    当传入 tools 参数时，API 可能返回 tool_calls 而非 content。
    返回 (content, tool_calls) 元组，其中 tool_calls 为列表或 None。

    Args:
        messages: 消息列表
        tools: OpenAI 兼容的 tools schema 列表，为 None 时不启用工具
        max_tokens: 最大 token 数
        temperature: 温度参数
        debug: 是否打印调试信息

    Returns:
        tuple: (content_text, tool_calls_list)
            - content_text: 文本回复（可能为 None）
            - tool_calls_list: tool_calls 列表，每项为 {"id": str, "name": str, "arguments": dict}
    """
    total_chars, est_tokens = _estimate_tokens(messages)
    if debug:
        logger.info(f"[chat_with_tools] 发送请求: {len(messages)} messages, {total_chars} chars, "
                     f"tools={len(tools) if tools else 0}")

    try:
        client = get_llm_client()
        response = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools if tools else None,
        )

        choice = response.choices[0]
        message = choice.message

        content = message.content or ""
        raw_tool_calls = message.tool_calls or []

        tool_calls = []
        if raw_tool_calls:
            for tc in raw_tool_calls:
                func_name = tc.function.name
                func_args_str = tc.function.arguments

                try:
                    func_args = json.loads(func_args_str)
                except json.JSONDecodeError:
                    func_args = {}

                tool_calls.append({
                    "id": tc.id,
                    "name": func_name,
                    "arguments": func_args,
                })

        if debug:
            if tool_calls:
                logger.info(f"[chat_with_tools] 完成: {len(tool_calls)} tool_calls: "
                             f"{[tc['name'] for tc in tool_calls]}")
            else:
                logger.info(f"[chat_with_tools] 完成: {len(content or '')} chars content")

        return content, tool_calls if tool_calls else None

    except APIConnectionError as e:
        logger.error(f"[chat_with_tools] LLM 连接失败: {e}", exc_info=True)
        raise
    except APITimeoutError:
        logger.error(f"[chat_with_tools] LLM 请求超时", exc_info=True)
        raise
    except APIStatusError as e:
        logger.error(f"[chat_with_tools] LLM 返回错误: HTTP {e.status_code} - {e.message}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"[chat_with_tools] 未知错误: {e}", exc_info=True)
        raise


# =============================================================================
# Stage 1: 意图识别 (已移除 — 由 Tool Calling 替代)
# =============================================================================
# classify_intent 函数已移除。LLM 现在通过 Tool Calling 自主决定操作，
# 不再需要硬编码的意图分类器。


# =============================================================================
# Stage 3A: 结构化提取
# =============================================================================
async def structure_report(search_result, history, last_report, ltm, entity_tracker, _emit=None):
    """将检索结果 + 上一轮报告(如有) 结构化输出为 JSON"""
    sys_prompt = STRUCTURE_PROMPT

    # ── 记忆注入：LTM 偏好（必须放在 System Prompt 最顶部，优先级最高） ──
    pref_prompt = ltm.get_preference_prompt()
    if pref_prompt:
        sys_prompt = pref_prompt + "\n\n" + sys_prompt

    # ── 记忆注入：当前实体上下文 ──
    entity_prompt = entity_tracker.to_context_prompt()
    if entity_prompt:
        sys_prompt = sys_prompt + "\n\n" + entity_prompt

    if last_report and last_report[0]:
        try:
            last_obj = json.loads(last_report[0])
            last_obj.pop("reasoning", None)
            last_text = json.dumps(last_obj, ensure_ascii=False)
        except Exception:
            logger.warning("解析 last_report JSON 失败，使用原文", exc_info=True)
            last_text = last_report[0]
        sys_prompt += f"\n\n## 已生成的报告（仅参考，请勿重复其中的病变）\n{last_text}"

    messages = [{"role": "system", "content": sys_prompt}]

    for msg in history[-4:]:
        content = msg.get("content", "").strip()
        if not content:
            continue
        role = msg.get("role", "")
        if role == "assistant" and len(content) > 200:
            content = "已生成报告（内容较长，此处省略完整文本）"
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": f"检索结果：\n{search_result}\n\n请按 JSON 格式输出结构化报告。"})

    if _emit:
        _emit("status", {"message": "正在生成结构化报告..."})

    output = await chat_stream(messages, max_tokens=2048, temperature=0.3, _emit=None, debug=True, caller="structure_report")

    report_json = _extract_json(output)

    if _emit:
        reasoning = report_json.get("reasoning", "") if isinstance(report_json, dict) else ""
        if reasoning:
            _emit("reasoning", {"text": reasoning})

    return report_json

# =============================================================================
# Stage 3B: 精准编辑
# =============================================================================
async def edit_report(query, last_report, history, _emit=None):
    """根据用户指令修改已有报告"""
    if not last_report or not last_report[0]:
        return {"error": "没有可修改的报告，请先生成一份报告。"}

    old_json_str = last_report[0]

    try:
        old_json = json.loads(old_json_str)
    except json.JSONDecodeError:
        old_json = {"raw": old_json_str}

    sys_prompt = EDIT_PROMPT

    messages = [{"role": "system", "content": sys_prompt}]

    for msg in history[-4:]:
        if msg.get("content", "").strip():
            messages.append(msg)

    messages.append({"role": "user", "content": f"当前报告：\n{old_json_str}\n\n修改指令：{query}\n\n请按 JSON 格式输出修改后的完整报告。"})

    if _emit:
        _emit("status", {"message": "正在修改报告..."})

    output = await chat_stream(messages, max_tokens=2048, temperature=0.3, _emit=None, debug=True, caller="edit_report")

    new_json = _extract_json(output)

    if isinstance(new_json, dict) and isinstance(old_json, dict):
        # 新格式: results 数组
        if "results" in new_json and "results" in old_json:
            new_count = len(new_json["results"])
            old_count = len(old_json["results"])
            if new_count != old_count:
                logger.warning(f"病变数量变化：旧 {old_count} → 新 {new_count}，使用旧报告兜底")
                return old_json
        # 兼容旧格式: 影像学表现/诊断意见 字典
        elif "影像学表现" in new_json and "影像学表现" in old_json:
            old_keys = set(old_json["影像学表现"].keys())
            new_keys = set(new_json["影像学表现"].keys())
            if old_keys != new_keys:
                logger.warning(f"Key 集合变化：旧 {old_keys} → 新 {new_keys}，使用旧报告兜底")
                return old_json

    return new_json if isinstance(new_json, dict) and "error" not in new_json else old_json


# =============================================================================
# Stage 3B-2: 重写报告（REFINE 意图）
# =============================================================================
async def refine_report(query, last_report, history, ltm, entity_tracker, _emit=None):
    """根据用户指令重写报告风格/详细程度，不修改医学内容主干"""
    if not last_report or not last_report[0]:
        return {"error": "没有可重写的报告，请先生成一份报告。"}

    old_json_str = last_report[0]

    try:
        old_json = json.loads(old_json_str)
    except json.JSONDecodeError:
        old_json = {"raw": old_json_str}

    sys_prompt = REFINE_PROMPT

    # 注入 LTM 偏好和 Entity 上下文
    preference_prompt = ""
    if ltm:
        preference_prompt = ltm.get_preference_prompt()
    context_prompt = entity_tracker.get_context_prompt() if entity_tracker else ""

    combined_system = sys_prompt
    if preference_prompt:
        combined_system = preference_prompt + "\n\n" + combined_system
    if context_prompt:
        combined_system = combined_system + "\n\n" + context_prompt

    messages = [{"role": "system", "content": combined_system}]

    for msg in history[-4:]:
        if msg.get("content", "").strip():
            messages.append(msg)

    messages.append({"role": "user", "content": f"当前报告：\n{old_json_str}\n\n重写指令：{query}\n\n请按 JSON 格式输出重写后的完整报告，保持医学内容不变，仅调整风格/表达方式。"})

    if _emit:
        _emit("status", {"message": "正在重写报告..."})

    output = await chat_stream(messages, max_tokens=2048, temperature=0.5, _emit=None, debug=True, caller="refine_report")

    new_json = _extract_json(output)

    if isinstance(new_json, dict) and isinstance(old_json, dict):
        # 新格式: results 数组
        if "results" in new_json and "results" in old_json:
            new_count = len(new_json["results"])
            old_count = len(old_json["results"])
            if new_count != old_count:
                logger.warning(f"REFINE 导致病变数量变化：旧 {old_count} → 新 {new_count}，使用旧报告兜底")
                return old_json
        # 兼容旧格式: 影像学表现/诊断意见 字典
        elif "影像学表现" in new_json and "影像学表现" in old_json:
            old_keys = set(old_json["影像学表现"].keys())
            new_keys = set(new_json["影像学表现"].keys())
            if old_keys != new_keys:
                logger.warning(f"REFINE 导致 Key 集合变化：旧 {old_keys} → 新 {new_keys}，使用旧报告兜底")
                return old_json

    return new_json if isinstance(new_json, dict) and "error" not in new_json else old_json


# =============================================================================
# Stage 3C: 闲聊回复
# =============================================================================
async def chat_reply(query, history, _emit=None):
    """直接回复闲聊"""
    messages = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
    ]
    for msg in history[-4:]:
        if msg.get("content", "").strip():
            messages.append(msg)
    messages.append({"role": "user", "content": query})

    if _emit:
        _emit("status", {"message": "正在回复..."})

    return await chat_stream(messages, max_tokens=512, temperature=0.7, _emit=_emit, debug=True, caller="chat_reply")


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


# =============================================================================
# 主流程：run_pipeline（Tool Calling 架构）
# =============================================================================
def _build_tool_registry(
    ltm, entity_tracker, client, last_report, _emit, selected_diagnosis=None, last_ambiguity=None,
):
    """构建并注册工具到 ToolRegistry。

    将 chat_stream 包装为符合 Tool Handler 签名的函数传入各工具。

    Args:
        ltm: LongTermMemory 实例
        entity_tracker: EntityTracker 实例
        client: MilvusClient 实例
        last_report: 上一轮报告的可变引用
        _emit: SSE 事件发射器

    Returns:
        ToolRegistry: 已注册所有工具的注册中心
    """
    from tools.rag_tool import RAG_SEARCH_SCHEMA, create_rag_search_handler
    from tools.edit_tool import EDIT_REPORT_SCHEMA, create_edit_report_handler
    from tools.refine_tool import REFINE_REPORT_SCHEMA, create_refine_report_handler

    registry = ToolRegistry()

    async def _chat_fn(messages, max_tokens=2048, temperature=0.3, _emit=None, debug=False, caller="tool"):
        return await chat_stream(messages, max_tokens=max_tokens, temperature=temperature,
                           _emit=_emit, debug=debug, caller=caller)

    def _search_reports_fn(query, _emit=None):
        return search_reports(query, client=client, _emit=_emit)

    rag_handler = create_rag_search_handler(
        chat_fn=_chat_fn,
        ltm=ltm,
        entity_tracker=entity_tracker,
        get_embedding_fn=get_embedding,
        search_reports_fn=_search_reports_fn,
        _emit_fn=_emit,
        last_report=last_report,
        selected_diagnosis=selected_diagnosis,
        last_ambiguity=last_ambiguity,
    )
    registry.register("rag_search", RAG_SEARCH_SCHEMA, rag_handler)

    edit_handler = create_edit_report_handler(
        chat_fn=_chat_fn,
        _emit_fn=_emit,
        last_report=last_report,
    )
    registry.register("edit_report", EDIT_REPORT_SCHEMA, edit_handler)

    refine_handler = create_refine_report_handler(
        chat_fn=_chat_fn,
        ltm=ltm,
        entity_tracker=entity_tracker,
        _emit_fn=_emit,
        last_report=last_report,
    )
    registry.register("refine_report", REFINE_REPORT_SCHEMA, refine_handler)

    return registry


async def run_pipeline(query, session_id, stm, entity_tracker, ltm, client, last_report, _emit, selected_diagnosis=None, last_ambiguity=None):
    """Tool Calling 架构主流程

    记忆模块集成点：
    1. Phase 1 (Pre-LLM): 实体提取 → 意图检测 → 上下文消解
    2. 切换意图: 清空 STM 和 last_report (彻底清洗)
    3. Phase 2 (Tool Calling): LLM 自主决定调用工具或直接回复
    4. Phase 3 (Post-LLM): 更新 STM，记录用户偏好
    """
    logger.info(f"── run_pipeline 开始: session={session_id}, query={query[:50]}..., selected_diagnosis={selected_diagnosis}")

    # ── 缓存命中检查：新输入是否匹配缓存中的诊断 ──
    if not selected_diagnosis and last_ambiguity and last_ambiguity[0] is not None:
        _, cached_reranked = last_ambiguity[0]
        for entity in cached_reranked:
            diagnosis_name = entity.get("diagnosis_name", "") or entity.get("诊断结论", "")
            # 检查用户输入是否与缓存中的诊断匹配（完全匹配或包含关系）
            if diagnosis_name and (query == diagnosis_name or diagnosis_name in query or query in diagnosis_name):
                logger.info(f"输入命中缓存诊断: query='{query}' ≈ diagnosis='{diagnosis_name}'，自动跳过检索")
                selected_diagnosis = diagnosis_name
                break

    # ── Phase 1: 输入处理 ──
    # 如果用户点击了歧义选项，或输入命中缓存诊断，跳过实体提取/意图检测/上下文消解，
    # 保留上一轮的 modality/body_part 槽位，确保缓存命中
    if selected_diagnosis:
        logger.info("run_pipeline: 用户通过歧义选项/缓存命中选择诊断，跳过 Phase 1，保留槽位")
        # 明确告知 LLM：用户已选择诊断，请调用 rag_search 生成报告
        enhanced = f"用户选择了诊断：{selected_diagnosis}。请使用 rag_search 工具生成该诊断的结构化报告。"
        _emit("intent", {"intent": "TOOL_CALL"})
        _emit("cache_hit", {"query": query, "matched_diagnosis": selected_diagnosis})
    else:
        # 1. 实体提取：从用户输入提取实体更新槽位
        changes = entity_tracker.update_from_query(query)
        if changes:
            logger.info(f"实体更新: {changes}")
            _emit("entity_update", {"changes": changes, "slots": entity_tracker.slots})

        # 2. 意图检测：new_session / append / switch
        detected_intent = entity_tracker.detect_intent(query)
        logger.info(f"实体意图: {detected_intent}, slots: {entity_tracker.slots}")

        # 切换意图：必须彻底清空，严禁旧病灶残留
        if detected_intent == "switch":
            logger.info("检测到切换意图，清空会话上下文")
            stm.clear(session_id)
            if last_report:
                last_report[0] = ""
            entity_tracker.apply_switch(query)
            _emit("intent_switch", {"message": "已清空旧上下文，开始新检查"})

        # 3. 上下文消解：补全省略信息
        enhanced = entity_tracker.resolve_context(query)
        enhanced = standardize_query(enhanced)
        if enhanced != query:
            logger.info(f"上下文消解: '{query}' → '{enhanced}'")
            _emit("context_resolve", {"original": query, "resolved": enhanced})

    # 如果用户通过歧义选项选择诊断，跳过查询改写
    if not selected_diagnosis and needs_rewrite(enhanced):
        original = enhanced
        rewritten = rewrite_query(enhanced)
        if rewritten and rewritten != enhanced:
            enhanced = rewritten
            logger.info(f"查询改写: '{original}' → '{rewritten}'")
            _emit("query_rewrite", {"original": original, "rewritten": rewritten})

    # ── 模糊输入拦截：如果只有模态，没有部位/诊断，且有 last_report，追问用户意图 ──
    if not selected_diagnosis:
        has_report = last_report and last_report[0]
        
        if has_report:
            # 检查本次查询是否只包含 modality 词（如只输入"CT"）
            query_has_new_modality = entity_tracker._extract_modality_rule(query) is not None
            query_has_body_part = len(entity_tracker._extract_body_part_rule(query)) > 0
            # 检查槽位中的诊断（可能来自歧义选择）
            slots_has_diagnosis = len(entity_tracker.slots.get("diagnosis", [])) > 0
            
            # 只有当查询只包含 modality，且没有任何部位/诊断时，才拦截
            if query_has_new_modality and not query_has_body_part and not slots_has_diagnosis:
                # 输入过于模糊（如只有"CT"），追问用户意图
                logger.info(f"模糊输入拦截：modality={entity_tracker.slots['modality']}, 无部位/诊断，已有报告")
                clarification_msg = (
                    f"检测到您只输入了检查类型 '{entity_tracker.slots['modality']}'，但未指定检查部位或诊断。\n\n"
                    f"请选择您想执行的操作：\n"
                    f"1. **修改当前报告**：修改已有的报告内容\n"
                    f"2. **重新检索**：用 '{entity_tracker.slots['modality']}' 重新检索知识库生成新报告\n"
                    f"3. **补充部位**：如 'CT 头颅'、'CT 腹部' 等\n\n"
                    f"请明确告知您的意图，或直接输入完整的查询（如 'CT 脑出血'）。"
                )
                _emit("message", {"content": clarification_msg})
                stm.add_turn(session_id, query, clarification_msg)
                logger.info(f"── run_pipeline 完成：模糊输入追问")
                return clarification_msg

        # ── 缺少模态拦截：如果有部位/诊断但没有模态，追问检查类型 ──
        has_body_part = len(entity_tracker.slots.get("body_part", [])) > 0
        has_diagnosis = len(entity_tracker.slots.get("diagnosis", [])) > 0
        has_modality = entity_tracker.slots.get("modality") is not None
        
        if not has_modality and (has_body_part or has_diagnosis):
            # 构建提示信息
            parts = []
            if has_body_part:
                parts.append(f"检查部位: {', '.join(entity_tracker.slots['body_part'])}")
            if has_diagnosis:
                parts.append(f"诊断: {', '.join(entity_tracker.slots['diagnosis'])}")
            
            info_text = "、".join(parts)
            clarification_msg = (
                f"已识别到{info_text}，但未指定检查类型。\n\n"
                f"请补充检查类型（如 CT、MR、DR、超声等），例如：\n"
                f"- 'CT 脑出血'\n"
                f"- 'MR 脑部'\n\n"
                f"或回复'继续'使用默认检查类型。"
            )
            _emit("message", {"content": clarification_msg})
            stm.add_turn(session_id, query, clarification_msg)
            logger.info(f"── run_pipeline 完成：缺少模态追问")
            return clarification_msg

    history = stm.get_history(session_id)

    # ── Phase 2: Tool Calling 主循环 ──
    # 构建工具注册中心
    registry = _build_tool_registry(ltm, entity_tracker, client, last_report, _emit, selected_diagnosis=selected_diagnosis, last_ambiguity=last_ambiguity)
    tools_schema = registry.get_tools_schema()
    logger.info(f"已注册工具: {list(registry._tools.keys())}")

    # ── 记忆检索注入：按需检索最相关的 LTM 偏好和 STM 历史 ──
    retriever = MemoryRetriever(get_embedding)
    retriever.index_ltm(ltm.get_preferences())
    retriever.index_stm(history)
    relevant = retriever.search_relevant(enhanced, top_k_ltm=3, top_k_stm=3)
    logger.info(f"记忆检索: LTM={len(relevant['ltm'])}条, STM={len(relevant['stm'])}条")
    if _emit:
        _emit("memory_retrieval", {
            "ltm": relevant["ltm"],
            "stm": relevant["stm"],
            "query": enhanced[:50],
        })

    # 构建系统消息：注入检索后的相关 LTM 偏好 + Entity 上下文
    sys_prompt = PromptBuilder.build(
        "tool_orchestrator",
        ltm_prefs=relevant["ltm"],
        entity_context=entity_tracker.to_context_prompt(),
    )

    # 注入检索后的相关对话历史（供 LLM 参考上下文）
    if relevant["stm"]:
        stm_context = "\n".join(f"- {msg}" for msg in relevant["stm"])
        sys_prompt += f"\n\n---\n\n## 相关历史对话\n{stm_context}"

    # 注入上一轮报告信息（供工具决策参考）
    if last_report and last_report[0]:
        sys_prompt += (
            f"\n\n---\n\n"
            f"## 当前已有报告\n"
            f"以下为上一轮生成的报告 JSON，如果用户要求修改或重写，请直接使用 edit_report 或 "
            f"refine_report 工具，无需重新检索。\n"
            f"```json\n{last_report[0][:2000]}\n```"
        )

    # 构建消息列表
    messages = [{"role": "system", "content": sys_prompt}]

    for msg in history[-6:]:
        content = msg.get("content", "").strip()
        if not content:
            continue
        role = msg.get("role", "")
        if role == "assistant" and len(content) > 500:
            content = content[:500] + "...（已省略后续内容）"
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": enhanced})

    # ── 计算上下文使用率 ──
    total_chars, est_tokens = _estimate_tokens(messages)
    # 模型上下文窗口估算：max_tokens(512) 是输出限制，输入+输出总 token 约 4096
    context_window = 4096
    usage_percent = (est_tokens / context_window) * 100
    if _emit:
        _emit("context_usage", {"percent": usage_percent, "tokens": est_tokens, "chars": total_chars})

    # ── 第一次 LLM 调用（带 tools） ──
    if _emit:
        _emit("status", {"message": "正在分析请求..."})

    logger.info(f"第一次 LLM 调用 (带 tools): {len(messages)} 条消息, tools={[t['function']['name'] for t in tools_schema]}")
    content, tool_calls = await chat_with_tools(
        messages,
        tools=tools_schema,
        max_tokens=512,
        temperature=0.3,
        debug=True,
    )

    # ── 如果没有工具调用，直接回复 ──
    if not tool_calls:
        if content:
            if _emit:
                _emit("message", {"content": content})
            stm.add_turn(session_id, query, content)
            logger.info(f"── run_pipeline 完成: 直接回复 (无工具调用), content长度={len(content)}")
            return content
        else:
            _emit("error", {"message": "模型未返回有效回复"})
            logger.warning(f"── run_pipeline 完成: 模型未返回有效回复")
            return "抱歉，模型未返回有效回复。"

    # ── 执行工具调用 ──
    if _emit:
        _emit("intent", {"intent": "TOOL_CALL", "tools": [tc["name"] for tc in tool_calls]})

    logger.info(f"LLM 选择工具: {[tc['name'] for tc in tool_calls]}")
    for tc in tool_calls:
        logger.info(f"  工具参数: {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})")

    # 将 assistant 消息（含 tool_calls）追加到消息列表
    assistant_tool_calls = []
    for tc in tool_calls:
        assistant_tool_calls.append({
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
            },
        })

    messages.append({
        "role": "assistant",
        "content": content or None,
        "tool_calls": assistant_tool_calls,
    })

    # 执行每个工具
    tool_results = []
    any_final = False

    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["arguments"]

        # 对 edit_report 和 refine_report，自动注入 current_report（主循环兜底）
        if tool_name in ("edit_report", "refine_report"):
            if "current_report" not in tool_args or not tool_args["current_report"]:
                if last_report and last_report[0]:
                    tool_args["current_report"] = last_report[0]
                else:
                    err_msg = f"工具 {tool_name} 需要已有报告，但当前没有可用的报告。"
                    _emit("error", {"message": err_msg})
                    tool_results.append((tc["id"], ToolResult(
                        content=json.dumps({"error": err_msg}, ensure_ascii=False),
                        is_final=False,
                    )))
                    continue

        result = await registry.execute(tc["id"], tool_name, tool_args)
        tool_results.append((tc["id"], result))

        if result.is_final:
            any_final = True

        logger.info(f"工具执行完成: {tool_name}, 结果长度={len(result.content)}, is_final={result.is_final}")
        if _emit:
            _emit("tool_executed", {
                "tool": tool_name,
                "params": tool_args,
                "result_length": len(result.content),
                "is_final": result.is_final,
            })

    # ── 将工具结果追加到消息列表 ──
    for tool_id, result in tool_results:
        messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "content": result.content,
        })

    # ── 决策：是否需要二次 LLM 调用 ──
    # 如果所有工具都返回 is_final=True（报告类结果），跳过二次 LLM 调用，
    # 直接将报告内容发送给前端，避免 LLM 二次总结引入幻觉或改变医学术语。
    final_content = None

    # ── 歧义检测：检查工具结果中是否有 ambiguous 状态 ──
    for tool_id, result in tool_results:
        if not result.is_final:
            continue
        try:
            result_json = json.loads(result.content)
            if isinstance(result_json, dict) and result_json.get("status") == "ambiguous":
                if _emit:
                    _emit("ambiguous", {
                        "question": result_json.get("question", ""),
                        "options": result_json.get("options", []),
                        "scores": result_json.get("scores", []),
                    })
                display = f"🔍 {result_json['question']}\n\n" + "\n".join(
                    f"{i+1}. {opt}" for i, opt in enumerate(result_json.get("options", []))
                )
                stm.add_turn(session_id, query, display)
                logger.info("── run_pipeline 完成: 歧义追问, options=%s", result_json.get("options"))
                return display
        except Exception:
            logger.error("解析工具结果 ambiguous 状态失败", exc_info=True)

    if any_final:
        logger.info("工具返回 is_final=True，跳过二次 LLM 调用，直接发送报告")
        if _emit:
            _emit("status", {"message": "报告已生成", "phase": "done"})

        # ── 在返回前必须保存 last_report，否则下一轮无法编辑 ──
        for tool_id, result in tool_results:
            if not result.is_final:
                continue
            try:
                result_json = json.loads(result.content)
                if isinstance(result_json, dict) and ("results" in result_json or "影像学表现" in result_json or "诊断意见" in result_json):
                    last_report[0] = result.content
                    logger.info(f"已保存 last_report (长度: {len(result.content)})")
                    # 不 break，继续循环，后面发送循环也需要遍历
            except Exception as e:
                logger.warning("保存 last_report 失败", exc_info=True)

        # 从工具结果中提取报告内容，直接发送给前端
        for tool_id, result in tool_results:
            if not result.is_final:
                continue
            try:
                result_json = json.loads(result.content)
                logger.info(f"── 检查工具结果: keys={list(result_json.keys()) if isinstance(result_json, dict) else type(result_json)}")
                if isinstance(result_json, dict) and "error" not in result_json:
                    if "results" in result_json or "影像学表现" in result_json or "诊断意见" in result_json:
                        display = json_to_display(result_json)
                        _emit("report", {"content": display})
                        stm.add_turn(session_id, query, display)
                        logger.info(f"── run_pipeline 完成: is_final 报告直接发送, display长度={len(display)}")
                        return display
                    else:
                        logger.warning(f"── 工具结果不包含报告字段: {list(result_json.keys())}")
                else:
                    logger.warning(f"── 工具结果包含 error 或不是 dict: {result_json.get('error', '') if isinstance(result_json, dict) else type(result_json)}")
            except Exception:
                logger.error("解析工具结果失败", exc_info=True)

        # 兜底：发送工具结果原文
        for tool_id, result in tool_results:
            if result.is_final:
                _emit("report", {"content": result.content[:2000]})
                stm.add_turn(session_id, query, result.content[:2000])
                logger.info(f"── run_pipeline 完成: is_final 兜底发送, content长度={len(result.content[:2000])}")
                return result.content[:2000]
    else:
        # ── 非最终结果：二次 LLM 调用（不带 tools），生成自然语言回复 ──
        if _emit:
            _emit("status", {"message": "正在生成回复..."})

        logger.info("非最终结果，执行第二次 LLM 调用")
        final_content, _ = await chat_with_tools(
            messages,
            tools=None,
            max_tokens=1024,
            temperature=0.3,
            debug=True,
        )

        if not final_content:
            final_content = "操作完成，请查看结果。"
            for tool_id, result in tool_results:
                try:
                    result_json = json.loads(result.content)
                    if isinstance(result_json, dict) and "error" not in result_json:
                        display = json_to_display(result_json)
                        if display:
                            final_content = display
                            break
                except Exception:
                    logger.error("二次LLM调用后解析工具结果失败", exc_info=True)

        if final_content:
            _emit("message", {"content": final_content})
            stm.add_turn(session_id, query, final_content)
            logger.info(f"── run_pipeline 完成: 二次LLM回复, content长度={len(final_content)}")
            return final_content

    # ── Phase 3: 后处理 ──
    # 从工具结果中提取并更新 last_report
    for tool_id, result in tool_results:
        try:
            result_json = json.loads(result.content)
            if isinstance(result_json, dict) and "error" not in result_json:
                if "results" in result_json or "影像学表现" in result_json or "诊断意见" in result_json:
                    last_report[0] = json.dumps(result_json, ensure_ascii=False, indent=2)
                    break
        except Exception:
            logger.error("后处理阶段解析工具结果失败", exc_info=True)

    logger.info(f"── run_pipeline 完成: 后处理兜底, 操作完成")
    return "操作完成"


# =============================================================================
# Web 服务
# =============================================================================
def web_main(port=8000):
    if not WEB_AVAILABLE:
        logger.error("Web 依赖缺失: 需要安装 fastapi 和 uvicorn")
        print("错误: 需要安装 fastapi 和 uvicorn")
        print("  pip install fastapi uvicorn")
        sys.exit(1)

    # 确保日志配置正确
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    app = FastAPI(title="影像报告生成Agent v2")
    store = SessionStore(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "sessions.db"))

    front_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "front")
    if os.path.isdir(front_dir):
        app.mount("/static", StaticFiles(directory=front_dir), name="static")

    @app.get("/")
    async def index():
        index_path = os.path.join(front_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "影像报告生成Agent v2", "docs": "/docs"}

    def _get_or_create_session(session_id):
        """从 SQLite 加载或创建会话，返回内存对象字典"""
        if store.session_exists(session_id):
            # ── 恢复已有会话 ──
            session_data = store.load_session(session_id)
            logger.info(f"恢复会话: {session_id}, 标题={session_data['title']}, 轮次={len(session_data['turns'])}")
            stm = ShortTermMemory(max_rounds=get_max_rounds())
            entity_tracker = EntityTracker(llm_chat_fn=chat_stream)
            ltm = LongTermMemory()
            client = MilvusClient(DB_PATH)
            client.load_collection(COLLECTION_NAME)

            # 恢复对话历史到 STM
            for turn in session_data["turns"]:
                stm.add_turn(session_id, turn["user_input"], turn["assistant_output"])

            # 恢复实体槽位（合并默认值，防止缺失键导致 KeyError）
            if session_data["state"]["entity_slots"]:
                entity_tracker.slots.update(session_data["state"]["entity_slots"])

            # 恢复 last_report
            last_report = [session_data["state"]["last_report"]]

            # 恢复歧义缓存（跨请求存活）
            cached = _ambiguity_cache.get(session_id)
            last_ambiguity = [cached] if cached else [None]

            return {
                "stm": stm,
                "entity_tracker": entity_tracker,
                "ltm": ltm,
                "client": client,
                "last_report": last_report,
                "last_ambiguity": last_ambiguity,
            }
        else:
            # ── 创建新会话 ──
            logger.info(f"创建新会话: {session_id}")
            stm = ShortTermMemory(max_rounds=get_max_rounds())
            entity_tracker = EntityTracker(llm_chat_fn=chat_stream)
            ltm = LongTermMemory()
            client = MilvusClient(DB_PATH)
            client.load_collection(COLLECTION_NAME)

            store.create_session(session_id)

            return {
                "stm": stm,
                "entity_tracker": entity_tracker,
                "ltm": ltm,
                "client": client,
                "last_report": [""],
                "last_ambiguity": [None],
            }

    @app.post("/api/chat", summary="对话", description="发送用户输入并获取流式回复")
    async def chat(request: ChatRequest):
        query = request.query.strip()
        session_id = request.session_id
        selected_diagnosis = request.selected_diagnosis

        if not query:
            return {"error": "query 不能为空"}

        logger.info(f"收到查询: session={session_id}, query={query[:50]}..., selected_diagnosis={selected_diagnosis}")
        session = _get_or_create_session(session_id)
        stm = session["stm"]
        entity_tracker = session["entity_tracker"]
        last_report = session["last_report"]
        last_ambiguity = session["last_ambiguity"]

        async def event_stream():
            thinking_events = []  # 持久化保存思考过程
            start_time = time.time()  # 记录请求开始时间
            event_queue = asyncio.Queue()

            def _emit_sync(event_type, data):
                try:
                    payload = json.dumps({"type": event_type, **data}, ensure_ascii=False)
                    logger.info(f"── _emit_sse: type={event_type}, len={len(payload)}")
                    event_queue.put_nowait(payload)
                    # 记录思考过程事件
                    thinking_events.append({"type": event_type, "data": data})
                except Exception:
                    logger.warning("SSE事件入队失败", exc_info=True)

            async def run():
                try:
                    # 记录本轮对话前的轮次索引
                    info_before = stm.session_info(session_id)
                    turn_index = info_before.get("total_turns", 0)

                    # 记录本轮前缓存是否已存在，用于判断是否清除
                    cache_existed_before = session_id in _ambiguity_cache

                    result = await run_pipeline(
                        query, session_id,
                        stm, entity_tracker, session["ltm"], session["client"],
                        last_report,
                        _emit_sync,
                        selected_diagnosis=selected_diagnosis,
                        last_ambiguity=last_ambiguity,
                    )

                    if result:
                        logger.info(f"run_pipeline 完成: result长度={len(result)}")

                    # ── 同步歧义缓存到全局 dict（跨请求存活）──
                    if last_ambiguity[0] is not None:
                        _ambiguity_cache[session_id] = last_ambiguity[0]
                        logger.info(f"歧义缓存已同步到全局: session={session_id}")

                    # ── 持久化：保存对话记录、会话状态和思考过程 ──
                    try:
                        store.save_turn(session_id, turn_index, query, result or "")
                        store.save_state(session_id, entity_tracker.slots, last_report[0])
                        # 保存思考过程
                        if thinking_events:
                            store.save_thinking(session_id, turn_index, thinking_events)
                        # 如果第一轮对话，自动更新标题
                        if turn_index == 0:
                            title = query[:20] if len(query) > 20 else query
                            store.update_title(session_id, title)
                            logger.info(f"会话标题已更新: {session_id} → {title}")
                        
                        # 同步到长期记忆（更新用户偏好）
                        session["ltm"].sync_from_short_term(stm, session_id, entity_tracker)
                        logger.info(f"长期记忆已同步: session={session_id}")
                    except Exception as e:
                        logger.warning("保存会话/长期记忆失败: %s", e)
                        
                except Exception as e:
                    logger.error("run_pipeline 执行异常", exc_info=True)
                    error_payload = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
                    event_queue.put_nowait(error_payload)
                finally:
                    # 计算总耗时并发送
                    elapsed = time.time() - start_time
                    done_payload = json.dumps({"type": "done", "total_time": round(elapsed, 1)}, ensure_ascii=False)
                    event_queue.put_nowait(done_payload)
                    event_queue.put_nowait("[DONE]")

            # 启动后台任务
            asyncio.create_task(run())

            # 流式推送事件
            while True:
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=120)
                    if event == "[DONE]":
                        logger.info("── event_stream(chat): 发送 [DONE]")
                        yield "data: [DONE]\n\n"
                        break
                    try:
                        evt = json.loads(event)
                        evt_type = evt.get("type", "?")
                        logger.info(f"── event_stream(chat): 发送 type={evt_type}, len={len(event)}")
                    except Exception:
                        logger.info(f"── event_stream(chat): 发送 raw, len={len(event)}")
                    yield f"data: {event}\n\n"
                except asyncio.TimeoutError:
                    logger.info("── event_stream(chat): 队列超时，退出")
                    break

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/api/info", summary="会话信息", description="获取当前会话的轮次、实体槽位和报告状态")
    async def info(session_id: str = Query(default="default", description="会话 ID")):
        session = _get_or_create_session(session_id)
        session_info = session["stm"].session_info(session_id)
        entity = session["entity_tracker"]
        return {
            "current_turns": session_info.get("current_turns", 0),
            "entity_slots": entity.slots,
            "has_last_report": bool(session["last_report"][0]),
        }

    @app.get("/api/memory", summary="记忆信息", description="获取会话的对话历史、实体、摘要等记忆信息")
    async def memory(session_id: str = Query(default="default", description="会话 ID")):
        session = _get_or_create_session(session_id)
        stm = session["stm"]
        entity = session["entity_tracker"]
        info = stm.session_info(session_id)
        history = stm.get_history(session_id)
        entities = entity.slots
        summaries = stm.get_summaries(session_id)

        turns = []
        for i in range(0, len(history), 2):
            user_msg = history[i]["content"] if i < len(history) else ""
            assistant_msg = history[i + 1]["content"] if i + 1 < len(history) else ""
            turns.append({
                "round": i // 2 + 1,
                "user": user_msg,
                "assistant": assistant_msg,
            })

        return {
            "turns": turns,
            "entities": entities,
            "summaries": summaries,
            "current_turns": info.get("current_turns", 0),
            "total_turns": info.get("total_turns", 0),
            "max_rounds": info.get("max_rounds", 5),
        }

    @app.get("/api/kb/status", summary="知识库状态", description="获取知识库文档总数、切片文件数等信息")
    async def kb_status():
        total = 0
        try:
            if os.path.exists(DB_PATH):
                client = MilvusClient(DB_PATH)
                total = len(client.query(COLLECTION_NAME, filter="", output_fields=["count(*)"]))
                client.close()
        except Exception:
            logger.warning("查询 Milvus 知识库状态失败", exc_info=True)
        slices_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "app", "data_pipeline", "xlsx_slices")
        md_count = len([f for f in os.listdir(slices_dir) if f.endswith(".md")]) if os.path.isdir(slices_dir) else 0
        metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline", "report_template", "metadata.json")
        meta_exists = os.path.exists(metadata_path)
        return {"total": total, "md_count": md_count, "db_path": DB_PATH, "metadata_exists": meta_exists}

    @app.get("/api/kb/files", summary="知识库文件列表", description="获取已上传的报告模板文件及其切片信息")
    async def kb_files():
        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline", "report_template")
        slices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline", "xlsx_slices")
        files = []
        if os.path.isdir(report_dir):
            for fname in sorted(os.listdir(report_dir)):
                if fname.endswith(".xlsx") and not fname.startswith("~$"):
                    fpath = os.path.join(report_dir, fname)
                    stat = os.stat(fpath)
                    basename = os.path.splitext(fname)[0]
                    slice_count = 0
                    if os.path.isdir(slices_dir):
                        slice_count = len([f for f in os.listdir(slices_dir) if f.startswith(basename) and f.endswith(".md")])
                    files.append({
                        "name": fname,
                        "slice_count": slice_count,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                    })
        return {"files": files}

    @app.post("/api/kb/build", summary="构建知识库", description="从切片文件构建向量数据库")
    async def kb_build(request: KBBuildRequest):
        rebuild = request.rebuild
        batch_size = request.batch_size
        slices_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "app", "data_pipeline", "xlsx_slices")

        async def event_stream():
            loop = asyncio.get_event_loop()
            event_queue = queue.Queue()

            def _emit_log(data):
                event_queue.put_nowait(json.dumps(data, ensure_ascii=False))

            def run():
                build_db(slices_dir, batch_size=batch_size, rebuild=rebuild, progress_callback=_emit_log)
                event_queue.put_nowait("[DONE]")

            loop.run_in_executor(None, run)

            while True:
                try:
                    event = await loop.run_in_executor(None, event_queue.get, True, 600)
                    if event == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    yield f"data: {event}\n\n"
                except queue.Empty:
                    break

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/api/kb/extract-metadata", summary="提取元数据", description="从报告模板中提取元数据到 metadata.json")
    async def kb_extract_metadata():
        metadata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline", "report_template")
        metadata_path = os.path.join(metadata_dir, "metadata.json")

        async def event_stream():
            loop = asyncio.get_event_loop()
            event_queue = queue.Queue()

            def _emit_log(data):
                event_queue.put_nowait(json.dumps(data, ensure_ascii=False))

            def run():
                extract_metadata(metadata_dir, metadata_path, progress_callback=_emit_log)
                event_queue.put_nowait("[DONE]")

            loop.run_in_executor(None, run)

            while True:
                try:
                    event = await loop.run_in_executor(None, event_queue.get, True, 120)
                    if event == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    yield f"data: {event}\n\n"
                except queue.Empty:
                    break

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/api/kb/upload", summary="上传报告模板", description="上传 .xlsx 报告模板文件并自动切片")
    async def kb_upload(file: UploadFile = File(...)):
        if not file.filename.endswith(".xlsx"):
            return {"error": "只支持 .xlsx 文件"}

        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline", "report_template")
        slices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_pipeline", "xlsx_slices")
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(slices_dir, exist_ok=True)

        filepath = os.path.join(report_dir, file.filename)

        async def event_stream():
            loop = asyncio.get_event_loop()
            event_queue = queue.Queue()

            def _emit_log(data):
                event_queue.put_nowait(json.dumps(data, ensure_ascii=False))

            def run():
                _emit_log({"level": "info", "msg": f"上传文件: {file.filename}"})
                with open(filepath, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                _emit_log({"level": "info", "msg": "切片中..."})
                count = process_file(filepath, slices_dir, progress_callback=_emit_log)
                _emit_log({"level": "done", "msg": f"✅ 切片完成，共生成 {count} 个 md 文件"})
                event_queue.put_nowait("[DONE]")

            loop.run_in_executor(None, run)

            while True:
                try:
                    event = await loop.run_in_executor(None, event_queue.get, True, 120)
                    if event == "[DONE]":
                        yield "data: [DONE]\n\n"
                        break
                    yield f"data: {event}\n\n"
                except queue.Empty:
                    break

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.delete("/api/session", summary="删除会话", description="删除指定会话及其所有数据")
    async def delete_session(session_id: str = Query(default="default", description="会话 ID")):
        store.delete_session(session_id)
        return {"status": "ok"}

    @app.post("/api/clear", summary="清空会话", description="清空指定会话的内容并重新创建")
    async def clear_session_post(request: ClearSessionRequest):
        session_id = request.session_id
        store.delete_session(session_id)
        store.create_session(session_id)
        return {"status": "ok"}

    @app.get("/api/sessions", summary="会话列表", description="获取所有历史会话列表")
    async def list_sessions():
        return {"sessions": store.list_sessions()}

    @app.get("/api/session/thinking", summary="思考过程", description="获取指定会话的思考过程事件记录")
    async def get_thinking(session_id: str = Query(default="default", description="会话 ID")):
        return {"thinking": store.get_thinking(session_id)}

    @app.post("/api/session/new", summary="创建会话", description="创建一个新会话并返回 session_id")
    async def new_session():
        session_id = SessionStore.generate_session_id()
        store.create_session(session_id)
        return {"session_id": session_id}

    @app.get("/api/config", summary="获取配置", description="获取当前系统配置（API 密钥会返回掩码）")
    async def get_config():
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yml")
        if not os.path.exists(config_path):
            return {"error": "配置文件不存在"}
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        # 如果 config.yml 中 api_key 为空但 .env 中有对应的 key，
        # 则返回掩码占位符，让前端知道已配置了 key
        for model_list_key, env_key_func in [
            ("llms", get_llm_api_key),
            ("embeddings", get_embed_api_key),
            ("reranks", get_rerank_api_key),
        ]:
            models = config_data.get(model_list_key)
            if isinstance(models, list):
                for m in models:
                    if not m.get("api_key") and env_key_func():
                        m["api_key"] = "••••••••••••••••••••••••"

        return {"config": config_data, "path": config_path}

    @app.post("/api/config", summary="保存配置", description="保存系统配置并重新加载生效")
    async def save_config(request: ConfigSaveRequest):
        config_data = request.config
        if config_data is None:
            return {"error": "缺少 config 参数"}

        # 如果 api_key 是前端掩码占位符，说明实际 key 在 .env 中，清空避免写入明文
        for model_list_key in ["llms", "embeddings", "reranks"]:
            models = config_data.get(model_list_key)
            if isinstance(models, list):
                for m in models:
                    if m.get("api_key") == "••••••••••••••••••••••••":
                        m["api_key"] = ""

        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config.yml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        reload_config()
        return {"status": "ok", "message": "配置已保存并生效"}

    @app.post("/api/test-model", summary="测试模型连接", description="测试模型 API 连接是否正常", response_model=TestModelResponse)
    async def test_model_connection(request: TestModelRequest):
        model_config = request.params
        model_type = request.model_type

        base_url = model_config.get("base_url", "")
        model_name = model_config.get("model", "")
        api_key = model_config.get("api_key", "")

        # 如果前端没传 key（空或掩码占位符），尝试从环境变量 / .env 兜底
        if not api_key or api_key == "••••••••••••••••••••••••":
            if model_type == "embeddings":
                api_key = get_embed_api_key()
            elif model_type == "reranks":
                api_key = get_rerank_api_key()
            else:
                api_key = get_llm_api_key()

        if not base_url:
            return {"success": False, "message": "API 地址不能为空"}
        if not model_name:
            return {"success": False, "message": "模型名不能为空"}

        try:
            if model_type == "embeddings":
                # 使用 OpenAI SDK 测试 Embedding
                from openai import OpenAI
                client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")
                response = client.embeddings.create(model=model_name, input="你好")
                emb_data = response.data
                if emb_data:
                    dim = len(emb_data[0].embedding)
                    return {"success": True, "message": f"连接成功，向量维度: {dim}"}
                return {"success": True, "message": "连接成功"}
                
            elif model_type == "reranks":
                # Rerank 仍使用 requests（OpenAI SDK 不直接支持 rerank）
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                payload = {
                    "model": model_name,
                    "query": "测试查询",
                    "documents": ["测试文档"],
                }
                r = requests.post(base_url, headers=headers, json=payload, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    results = data.get("results", [])
                    if results:
                        score = results[0].get("relevance_score", "N/A")
                        return {"success": True, "message": f"连接成功，相关性分数: {score}"}
                    return {"success": True, "message": "连接成功"}
                else:
                    error_msg = r.text[:200] if r.text else f"HTTP {r.status_code}"
                    return {"success": False, "message": error_msg}
                    
            else:
                # 使用 OpenAI SDK 测试 LLM
                from openai import AsyncOpenAI
                client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed")
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "你好"}],
                    max_tokens=16,
                    temperature=0.0,
                )
                choices = response.choices
                if choices:
                    reply = choices[0].message.content or ""
                    return {"success": True, "message": f"连接成功，返回: {reply[:50]}"}
                return {"success": True, "message": "连接成功"}
                
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                logger.warning("模型测试：连接超时")
                return {"success": False, "message": "连接超时"}
            elif "connection" in error_msg.lower():
                logger.warning("模型测试：无法连接到服务器")
                return {"success": False, "message": "无法连接到服务器"}
            else:
                logger.error("模型测试失败", exc_info=True)
                # 提取 HTTP 状态码（如果有）
                if "status_code" in error_msg or "http" in error_msg.lower():
                    return {"success": False, "message": error_msg[:200]}
                return {"success": False, "message": error_msg[:200]}

    logger.info(f"Web 服务启动: http://localhost:{port}, 模型={CHAT_MODEL}, Embedding={EMBED_MODEL}")
    print(f"\n{'='*60}")
    print(f"  影像报告生成Agent v2 Web 服务")
    print(f"  {'='*60}")
    print(f"  地址: http://localhost:{port}")
    print(f"  API 文档: http://localhost:{port}/docs")
    print(f"  生成模型: {CHAT_MODEL}")
    print(f"  Embedding: {EMBED_MODEL}")
    print(f"  {'='*60}\n")

    uvicorn.run(app, host="0.0.0.0", port=port)


# =============================================================================
# CLI 模式
# =============================================================================
def main():
    if "--web" in sys.argv:
        web_main()
        return

    print(f"\n{'='*60}")
    print(f"  影像报告生成Agent v2 CLI")
    print(f"  {'='*60}")
    print(f"  输入问题开始对话，输入 clear 清空会话，输入 quit 退出")
    print(f"  {'='*60}\n")

    SESSION_ID = f"rag_v2_{uuid.uuid4().hex[:8]}"
    stm = ShortTermMemory(max_rounds=get_max_rounds())
    entity_tracker = EntityTracker(llm_chat_fn=chat_stream)
    ltm = LongTermMemory()
    client = MilvusClient(DB_PATH)
    client.load_collection(COLLECTION_NAME)
    last_report = [""]

    def _emit(event_type, data):
        if event_type == "report":
            print(f"\n📋 报告:\n{data['content']}\n")
        elif event_type == "reasoning":
            print(f"💭 推理: {data['content']}")
        elif event_type == "intent":
            print(f"🎯 意图: {data['intent']}")
        elif event_type == "entity_update":
            print(f"🔄 实体更新: {data['changes']}, 槽位: {data['slots']}")
        elif event_type == "intent_switch":
            print(f"🔄 {data['message']}")
        elif event_type == "error":
            print(f"❌ 错误: {data['message']}")

    try:
        while True:
            user_input = input("\n👤 用户: ").strip()
            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("👋 再见！")
                break

            if user_input.lower() == "clear":
                stm.clear(SESSION_ID)
                entity_tracker.clear()
                last_report[0] = ""
                print("🧹 会话已清空\n")
                continue

            result = asyncio.run(run_pipeline(
                user_input, SESSION_ID,
                stm, entity_tracker, ltm, client,
                last_report,
                _emit,
            ))
    except KeyboardInterrupt:
        print("\n👋 再见！")
    except Exception as e:
        logger.error("CLI 主循环未捕获异常", exc_info=True)
        print(f"\n❌ 未捕获异常: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()