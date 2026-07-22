"""大模型网络请求封装"""

import logging

from openai import APIConnectionError, APITimeoutError, APIStatusError

from config import get_llm_client, get_embed_client, get_llm_model, get_embed_model

from .utils import retry
from rag.rerank import rerank_documents

logger = logging.getLogger(__name__)

# 从配置加载常量
CHAT_MODEL = get_llm_model()
EMBED_MODEL = get_embed_model()


# =============================================================================
# Embedding
# =============================================================================
def get_embedding(text):
    """使用 OpenAI SDK 获取文本向量"""
    try:
        client = get_embed_client()
        response = client.embeddings.create(model=EMBED_MODEL, input=text)
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"[get_embedding] 请求失败: {e}", exc_info=True)
        raise


# =============================================================================
# Token 估算
# =============================================================================
def _estimate_tokens(messages):
    total_chars = sum(len(msg.get("content", "") or "") for msg in messages)
    return total_chars, total_chars // 2


# =============================================================================
# 流式 LLM 调用
# =============================================================================
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


# =============================================================================
# 非流式 LLM 调用（支持 Tool Calling）
# =============================================================================
import json

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
# Rerank 带重试
# =============================================================================
@retry()
def rerank_with_retry(query, documents, top_n=3):
    return rerank_documents(query, documents, top_n=top_n)
