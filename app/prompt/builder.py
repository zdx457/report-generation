"""动态 Prompt Builder

统一管理所有 Prompt 的拼装逻辑，消除 rag_tool.py、refine_tool.py、
rag_chat_v2.py 中重复的 LTM 偏好 + Entity 上下文 + Last Report 拼接代码。

拼接顺序（优先级从高到低）：
1. 用户偏好 (LTM) — 支持全量字符串或检索后的 List[str]
2. 当前上下文 (Entity)
3. 基础模板 (template)
4. 已有报告 (Last Report，自动去除 reasoning 字段)
"""

import json
import logging
from typing import List, Optional, Union
from prompt import load_prompt

logger = logging.getLogger(__name__)


class PromptBuilder:
    """统一 Prompt 拼装器

    使用方式 — 全量注入（向后兼容）：
        sys_prompt = PromptBuilder.build(
            "structure",
            ltm_prefs=ltm.get_preference_prompt(),
            entity_context=entity_tracker.to_context_prompt(),
            last_report=last_report[0] if last_report else None,
        )

    使用方式 — 检索后注入（Phase 3）：
        relevant = retriever.search(query, top_k_ltm=3)
        sys_prompt = PromptBuilder.build(
            "tool_orchestrator",
            ltm_prefs=relevant["ltm"],   # List[str]
            entity_context=entity_tracker.to_context_prompt(),
        )
    """

    @staticmethod
    def build(
        template_name: str,
        ltm_prefs: Optional[Union[str, List[str]]] = None,
        entity_context: Optional[str] = None,
        last_report: Optional[str] = None,
        last_report_label: Optional[str] = None,
    ) -> str:
        """构建完整的系统 Prompt。

        Args:
            template_name: 模板名称（不含 .md 扩展名），如 'structure', 'refine', 'tool_orchestrator'
            ltm_prefs: LTM 偏好，支持两种格式：
                - str: 由 ltm.get_preference_prompt() 生成的全量偏好字符串（向后兼容）
                - List[str]: 由 MemoryRetriever.search() 返回的相关偏好列表（Phase 3）
            entity_context: 实体上下文字符串，由 entity_tracker.to_context_prompt() 生成
            last_report: 上一轮报告 JSON 字符串，会自动去除 reasoning 字段以节省 Token
            last_report_label: 报告段落的标题，默认 "## 已生成的报告（仅参考，请勿重复其中的病变）"

        Returns:
            str: 完整的系统 Prompt 字符串
        """
        parts = []

        # 1. 用户偏好 (LTM) — 最高优先级，放在最前面
        if ltm_prefs:
            if isinstance(ltm_prefs, list):
                # 检索后的偏好列表，join 为字符串
                pref_text = "\n".join(f"- {p}" for p in ltm_prefs)
                if pref_text.strip():
                    parts.append("## 相关用户偏好\n" + pref_text)
            elif isinstance(ltm_prefs, str) and ltm_prefs.strip():
                parts.append(ltm_prefs)

        # 2. 当前上下文 (Entity) — 告知 LLM 当前检查类型和部位
        if entity_context:
            parts.append(entity_context)

        # 3. 基础模板
        template = load_prompt(template_name)
        if template:
            parts.append(template)
        else:
            logger.warning("模板加载失败: %s，将使用空字符串兜底", template_name)

        # 4. 已有报告 (Last Report) — 去除 reasoning 字段以节省 Token
        if last_report:
            try:
                last_obj = json.loads(last_report)
                last_obj.pop("reasoning", None)
                last_text = json.dumps(last_obj, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError):
                last_text = last_report

            label = last_report_label or "## 已生成的报告（包含已有病变，请合并到最终输出，不要丢弃）"
            parts.append(f"{label}\n{last_text}")

        return "\n\n---\n\n".join(parts)