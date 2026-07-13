"""工具注册中心

提供 OpenAI 兼容的 Tool Calling 机制：
- 注册工具（名称、Schema、处理函数）
- 生成 tools schema 列表供 LLM 调用
- 执行工具并返回结果（含 is_final 标记，用于跳过二次 LLM 调用）
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """工具执行结果

    Attributes:
        content: 工具返回的文本内容（发送给 LLM 的 role:tool 消息）
        is_final: 是否为最终结果。为 True 时，主循环跳过二次 LLM 调用，
                  直接将报告内容发送给前端，避免引入幻觉。
    """
    content: str
    is_final: bool = False


class ToolRegistry:
    """工具注册中心 —— 管理所有可调用的工具

    使用方式：
        registry = ToolRegistry()
        registry.register("rag_search", schema, handler)
        tools_schema = registry.get_tools_schema()
        result = registry.execute(tool_call_id, "rag_search", {"query": "CT脑出血"})
        # result.is_final → True 表示这是最终结果，无需二次 LLM 调用
    """

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register(self, name: str, schema: dict, handler: Callable[[dict], str]):
        """注册一个工具。

        Args:
            name: 工具名称（唯一标识）
            schema: OpenAI 兼容的 function schema，需包含 type, function 字段
            handler: 工具处理函数，签名为 (arguments: dict) -> str，
                     返回的 JSON 字符串中可包含 "_is_final": true 标记
        """
        if name in self._tools:
            logger.warning("工具 '%s' 已存在，将被覆盖", name)

        self._tools[name] = {
            "schema": schema,
            "handler": handler,
        }
        logger.info("工具已注册: %s", name)

    def unregister(self, name: str):
        """注销工具"""
        self._tools.pop(name, None)
        logger.info("工具已注销: %s", name)

    def get_tools_schema(self) -> list[dict]:
        """返回 OpenAI 兼容的 tools 定义列表。

        Returns:
            list[dict]: 每个元素为 {"type": "function", "function": {...}} 格式
        """
        return [t["schema"] for t in self._tools.values()]

    def execute(self, tool_call_id: str, name: str, arguments: dict) -> ToolResult:
        """执行指定工具，返回 ToolResult（含 is_final 标记）。

        工具 Handler 返回的 JSON 字符串中如果包含 "_is_final": true，
        则该字段会被剥离并设置到 ToolResult.is_final 中。

        Args:
            tool_call_id: LLM 返回的 tool_call id（用于日志追踪）
            name: 工具名称
            arguments: 工具参数

        Returns:
            ToolResult: content 为发送给 LLM 的文本，is_final 标记是否跳过二次 LLM
        """
        if name not in self._tools:
            error_msg = f"未知工具: {name}，可用工具: {list(self._tools.keys())}"
            logger.error("工具执行失败 [%s]: %s", tool_call_id, error_msg)
            return ToolResult(
                content=json.dumps({"error": error_msg}, ensure_ascii=False),
                is_final=False,
            )

        handler = self._tools[name]["handler"]
        try:
            logger.info("执行工具 [%s] %s: %s", tool_call_id, name, json.dumps(arguments, ensure_ascii=False)[:200])
            raw_result = handler(arguments)
            if raw_result is None:
                logger.warning("工具 [%s] %s 返回了 None", tool_call_id, name)
                raw_result = json.dumps({"error": "工具未返回结果", "_is_final": False}, ensure_ascii=False)
            logger.info("工具执行完成 [%s] %s: 结果长度 %d", tool_call_id, name, len(raw_result))

            # 解析 is_final 标记
            is_final = False
            try:
                result_json = json.loads(raw_result)
                if isinstance(result_json, dict):
                    is_final = result_json.pop("_is_final", False)
                    raw_result = json.dumps(result_json, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                pass

            return ToolResult(content=raw_result, is_final=is_final)

        except Exception as e:
            error_msg = f"工具执行异常 [{name}]: {str(e)}"
            logger.error("工具执行异常 [%s]: %s", tool_call_id, error_msg)
            return ToolResult(
                content=json.dumps({"error": error_msg}, ensure_ascii=False),
                is_final=False,
            )

    @property
    def tool_names(self) -> list[str]:
        """已注册的工具名称列表"""
        return list(self._tools.keys())

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.tool_names})"