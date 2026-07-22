"""报告编辑工具

Schema: edit_report
- 根据用户指令修改已有报告
- 使用 prompt/edit.md 调用 LLM
- 保证报告结构不被破坏
- 返回修改后的报告 JSON
"""

import json
import logging
from typing import Callable, Optional

from tools.utils import extract_json
from langsmith import traceable

logger = logging.getLogger(__name__)

EDIT_REPORT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit_report",
        "description": (
            "修改已有报告中的指定内容。"
            "适用于：用户要求修改报告中的某个字段、删除某个病变、更新某个数值等。"
            "注意：此工具修改已有报告，不会检索新知识。如果用户要求生成新报告，应使用 rag_search。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "current_report": {
                    "type": "string",
                    "description": "当前报告的 JSON 字符串，包含 '影像学表现' 和 '诊断意见' 等字段",
                },
                "instruction": {
                    "type": "string",
                    "description": "用户的修改指令，描述需要修改什么内容。例如：'把脑出血的CT值改为70HU'、'删除右侧基底节区病变'",
                },
            },
            "required": ["current_report", "instruction"],
        },
    },
}


def create_edit_report_handler(
    chat_fn: Callable,
    _emit_fn: Optional[Callable] = None,
    last_report: Optional[list] = None,
) -> Callable[[dict], str]:
    """创建 edit_report 工具的处理函数。

    Args:
        chat_fn: LLM 对话函数，签名 (messages, max_tokens, temperature, _emit, debug, caller) -> str
        _emit_fn: SSE 事件发射器
        last_report: 上一轮报告的可变引用列表（用于参数容错补全）

    Returns:
        handler: 签名为 (arguments: dict) -> str 的处理函数
    """

    @traceable(run_type="tool", name="Tool_Edit_Report")
    async def handler(arguments: dict) -> str:
        current_report = arguments.get("current_report", "")
        instruction = arguments.get("instruction", "")

        # ── 参数容错：从 last_report 自动补全 current_report ──
        if not current_report and last_report and last_report[0]:
            current_report = last_report[0]
            logger.info("edit_report: 从 last_report 自动补全 current_report")

        if not current_report:
            return json.dumps(
                {"error": "缺少 current_report 参数，没有可修改的报告", "_is_final": False},
                ensure_ascii=False,
            )
        if not instruction:
            return json.dumps(
                {"error": "缺少 instruction 参数，请提供修改指令", "_is_final": False},
                ensure_ascii=False,
            )

        logger.info("edit_report 开始执行: instruction=%s", instruction[:100])

        try:
            old_json = json.loads(current_report)
        except json.JSONDecodeError:
            old_json = {"raw": current_report}

        try:
            from prompt import load_prompt
            EDIT_PROMPT = load_prompt("edit")

            sys_prompt = EDIT_PROMPT
            messages = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": f"当前报告：\n{current_report}\n\n修改指令：{instruction}\n\n请按 JSON 格式输出修改后的完整报告。",
                },
            ]

            if _emit_fn:
                _emit_fn("status", {"message": "正在修改报告...", "phase": "editing"})

            output = await chat_fn(
                messages,
                max_tokens=2048,
                temperature=0.3,
                _emit=None,
                debug=True,
                caller="edit_report_tool",
            )

            if _emit_fn:
                _emit_fn("status", {"message": "报告修改完成", "phase": "done"})

            new_json = extract_json(output)

            # 结构保护：Key 集合不应变化
            if isinstance(new_json, dict) and isinstance(old_json, dict):
                if "影像学表现" in new_json and "影像学表现" in old_json:
                    old_keys = set(old_json["影像学表现"].keys())
                    new_keys = set(new_json["影像学表现"].keys())
                    if old_keys != new_keys:
                        logger.warning("Key 集合变化：旧 %s → 新 %s，使用旧报告兜底", old_keys, new_keys)
                        old_json["_is_final"] = True
                        return json.dumps(old_json, ensure_ascii=False, indent=2)

            result = new_json if isinstance(new_json, dict) and "error" not in new_json else old_json

            # ── 标记为最终结果，跳过二次 LLM 调用，避免幻觉 ──
            if isinstance(result, dict):
                result["_is_final"] = True

            result_str = json.dumps(result, ensure_ascii=False, indent=2)
            logger.info("edit_report 完成: 报告长度 %d, is_final=True", len(result_str))
            return result_str

        except Exception as e:
            logger.error("edit_report 执行失败: %s", e, exc_info=True)
            return json.dumps(
                {"error": f"编辑失败: {str(e)}", "_is_final": False},
                ensure_ascii=False,
            )

    return handler