"""RAG 检索工具

Schema: rag_search
- 封装多路召回 + Rerank + 结构化报告生成
- 注入 LTM 偏好和 Entity 上下文
- 返回结构化报告 JSON
"""

import json
import logging
from typing import Callable, Optional

from tools.utils import extract_json

logger = logging.getLogger(__name__)

RAG_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "rag_search",
        "description": (
            "检索医学影像知识库，根据用户查询生成结构化报告。"
            "适用于：用户要求生成一份新的影像报告、查询某个检查类型/部位的参考信息。"
            "当用户提供明确的检查类型（CT/MR/DR等）和部位时，应使用此工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "经过上下文消解和标准化的查询语句，包含检查类型、检查部位等关键信息。例如：'CT头颅 脑出血'",
                },
                "modality": {
                    "type": "string",
                    "enum": ["CT", "MR", "MRI", "DR", "X线", "超声", "PET-CT", "DSA", "CTA", "不限"],
                    "description": "检查类型/模态，用于辅助检索过滤",
                },
                "body_part": {
                    "type": "string",
                    "description": "检查部位，如：头颅、胸部、腹部、肝脏、膝关节等",
                },
            },
            "required": ["query"],
        },
    },
}


def create_rag_search_handler(
    chat_fn: Callable,
    ltm,
    entity_tracker,
    get_embedding_fn: Callable,
    search_reports_fn: Callable,
    _emit_fn: Optional[Callable] = None,
    last_report: Optional[list] = None,
) -> Callable[[dict], str]:
    """创建 rag_search 工具的处理函数。

    Args:
        chat_fn: LLM 对话函数，签名 (messages, max_tokens, temperature, _emit, debug, caller) -> str
        ltm: LongTermMemory 实例，用于注入偏好
        entity_tracker: EntityTracker 实例，用于注入上下文和参数补全
        get_embedding_fn: 向量化函数
        search_reports_fn: RAG 检索函数
        _emit_fn: SSE 事件发射器
        last_report: 上一轮报告的可变引用列表

    Returns:
        handler: 签名为 (arguments: dict) -> str 的处理函数
    """

    def handler(arguments: dict) -> str:
        query = arguments.get("query", "")
        modality = arguments.get("modality", "")
        body_part = arguments.get("body_part", "")

        # ── 参数容错：从 EntityTracker 自动补全缺失参数 ──
        if not modality and entity_tracker:
            modality = entity_tracker.slots.get("modality", "")
            if modality:
                logger.info("rag_search: 从 EntityTracker 自动补全 modality=%s", modality)
        if not body_part and entity_tracker:
            body_part = entity_tracker.slots.get("body_part", "")
            if body_part:
                logger.info("rag_search: 从 EntityTracker 自动补全 body_part=%s", body_part)

        if not query:
            error_result = json.dumps(
                {"error": "缺少 query 参数", "_is_final": False},
                ensure_ascii=False,
            )
            return error_result

        logger.info("rag_search 开始执行: query=%s, modality=%s, body_part=%s", query, modality, body_part)

        try:
            if _emit_fn:
                _emit_fn("status", {"message": f"正在检索: {query}", "phase": "searching"})
                _emit_fn("search", {"query": query, "modality": modality, "body_part": body_part})

            search_result = search_reports_fn(query, _emit=_emit_fn)

            if not search_result or search_result == "未检索到相关报告。":
                logger.warning("rag_search: 未检索到相关报告")
                return json.dumps(
                    {
                        "error": "未检索到相关报告",
                        "message": "知识库中未找到与您查询相关的报告，请尝试更换查询词或检查部位。",
                        "_is_final": False,
                    },
                    ensure_ascii=False,
                )

            if _emit_fn:
                _emit_fn("status", {"message": "检索完成，正在生成结构化报告...", "phase": "generating"})

            from prompt.builder import PromptBuilder

            sys_prompt = PromptBuilder.build(
                "structure",
                ltm_prefs=ltm.get_preference_prompt() if ltm else None,
                entity_context=entity_tracker.to_context_prompt() if entity_tracker else None,
                last_report=last_report[0] if last_report and last_report[0] else None,
            )

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"检索结果：\n{search_result}\n\n请按 JSON 格式输出结构化报告。"},
            ]

            output = chat_fn(
                messages,
                max_tokens=2048,
                temperature=0.3,
                _emit=None,
                debug=True,
                caller="rag_search_tool",
            )

            if _emit_fn:
                _emit_fn("status", {"message": "报告生成完成", "phase": "done"})

            report_json = extract_json(output)

            # ── 标记为最终结果，跳过二次 LLM 调用，避免幻觉 ──
            if isinstance(report_json, dict) and "error" not in report_json:
                report_json["_is_final"] = True
            else:
                report_json["_is_final"] = False

            result_str = json.dumps(report_json, ensure_ascii=False, indent=2)

            if _emit_fn:
                reasoning = report_json.get("reasoning", "") if isinstance(report_json, dict) else ""
                if reasoning:
                    _emit_fn("reasoning", {"text": reasoning})

            logger.info("rag_search 完成: 报告长度 %d, is_final=True", len(result_str))
            return result_str

        except Exception as e:
            logger.error("rag_search 执行失败: %s", e, exc_info=True)
            return json.dumps(
                {"error": f"检索失败: {str(e)}", "_is_final": False},
                ensure_ascii=False,
            )

    return handler