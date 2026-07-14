"""实体追踪器 (Entity Tracker)

维护当前对话的结构化状态，解决"用户在说什么"的问题。

设计决策：
- 独立类：从 ShortTermMemory 中彻底拆分，职责单一
- 槽位模型：modality / body_part / clinical_history / diagnosis / intent
- 继承性：用户说"再看看肝脏"，自动继承上一轮的模态
- 覆盖性：用户说"换成 MR 膝关节"，清空旧状态，建立新状态
- 提取策略：LLM JSON 结构化提取 + 规则兜底（双引擎，保证鲁棒性）
- 线程安全：每个会话持有独立实例，无需加锁
"""

import json
import logging
import re
from typing import Callable, Optional, Tuple

from prompt import load_prompt

logger = logging.getLogger(__name__)

# 加载 LLM 提取提示词
EXTRACTION_PROMPT = load_prompt("entity_extraction")

# 模态关键词（按长度降序，避免短词误匹配）
MODALITY_PATTERNS = [
    "PET-CT", "PET", "SPECT", "DSA", "CTA", "MRA", "DWI", "SWI", "FLAIR",
    "CT", "MRI", "MR", "X线", "X光", "超声", "B超",
]

# 部位关键词（按长度降序，优先匹配长词）
BODY_PART_PATTERNS = [
    "颅脑", "头颅", "头部", "大脑",
    "胸部", "肺部", "胸腔", "肺",
    "腹部", "肝脏", "肝", "胆囊", "胆", "胰腺", "胰", "脾脏", "脾", "肾脏", "肾",
    "盆腔", "子宫", "卵巢", "前列腺", "膀胱",
    "脊柱", "颈椎", "胸椎", "腰椎", "骶椎",
    "膝关节", "膝", "髋关节", "髋", "肩关节", "肩", "肘关节", "肘", "腕关节", "腕",
    "踝关节", "踝", "足", "手",
    "颈部", "甲状腺",
    "心血管", "心脏", "血管", "冠脉", "主动脉",
    "骨骼", "骨",
    "胃肠", "胃", "肠道", "结肠", "直肠",
    "鼻咽", "咽喉", "口腔",
    "乳腺",
]

# 指代/省略触发词
REFERENCE_PATTERNS = [
    "那个", "刚才", "上面", "之前", "上文", "前文",
    "刚刚", "那个病", "那个检查", "那次",
    "这个", "再看看", "看一下", "接着看",
]

# 切换意图触发词
SWITCH_PATTERNS = [
    "换成", "改为", "改成", "换成别的", "改成别的",
    "重新", "换个", "改成别的", "换一个", "不要这个",
    "重新来", "再来", "重新开始",
]


class EntityTracker:
    """实体追踪器 —— 维护当前对话的结构化上下文状态

    支持双引擎实体提取：
    - LLM 引擎：JSON 结构化提取，准确率高，处理自然语言
    - 规则引擎：关键词降序匹配，兜底保证鲁棒性
    """

    def __init__(self, llm_chat_fn: Optional[Callable[[list[dict]], str]] = None):
        """
        Args:
            llm_chat_fn: LLM 对话函数，签名 (messages: list[dict]) -> str
                        如果提供，则使用 LLM JSON 提取；否则使用规则兜底
        """
        self.slots = {
            "modality": None,          # 当前模态 (CT, MR, DR...)
            "body_part": [],           # 当前部位列表 (Brain, Chest, Liver...)
            "clinical_history": "",    # 病史/症状
            "diagnosis": [],           # 已确认的诊断列表
            "intent": "new_session",   # 当前意图: new_session / append / switch
        }
        self._llm_chat = llm_chat_fn

    # ── 实体提取 ──────────────────────────────────────────

    def update_from_query(self, query: str) -> dict:
        """从用户输入提取实体，更新槽位。

        优先使用 LLM JSON 结构化提取，失败回退到规则匹配。
        body_part 支持多部位：新提取的部位追加到列表中，不去重。

        Returns:
            dict: 本次提取到的实体变更 {"modality": "CT", "body_part": ["肝脏"], ...}
        """
        changes = {}

        if self._llm_chat is not None:
            extracted = self._extract_llm(query)
            if extracted:
                modality, body_parts = extracted
                if modality and modality != self.slots["modality"]:
                    self.slots["modality"] = modality
                    changes["modality"] = modality
                if body_parts:
                    new_parts = [p for p in body_parts if p not in self.slots["body_part"]]
                    if new_parts:
                        self.slots["body_part"].extend(new_parts)
                        changes["body_part"] = new_parts
                if changes:
                    logger.debug("LLM实体更新: %s", changes)
                return changes

        # LLM 不可用或提取失败，回退到规则匹配
        extracted_modality = self._extract_modality_rule(query)
        if extracted_modality and extracted_modality != self.slots["modality"]:
            self.slots["modality"] = extracted_modality
            changes["modality"] = extracted_modality

        extracted_body_parts = self._extract_body_part_rule(query)
        if extracted_body_parts:
            new_parts = [p for p in extracted_body_parts if p not in self.slots["body_part"]]
            if new_parts:
                self.slots["body_part"].extend(new_parts)
                changes["body_part"] = new_parts

        if changes:
            logger.debug("规则实体更新: %s", changes)

        return changes

    def _extract_llm(self, query: str) -> Optional[Tuple[Optional[str], Optional[list]]]:
        """使用 LLM 提取实体，返回 JSON 解析后的 (modality, body_parts)"""
        if self._llm_chat is None:
            return None

        prompt = EXTRACTION_PROMPT + "\n\n" + query
        messages = [
            {"role": "system", "content": "你是一个专业的医学影像实体抽取助手，请严格按要求输出 JSON。"},
            {"role": "user", "content": prompt},
        ]

        try:
            output = self._llm_chat(messages)
            return self._parse_json_output(output)
        except Exception as e:
            logger.warning("LLM 实体提取失败，回退到规则: %s", e)
            return None

    @staticmethod
    def _parse_json_output(output: str) -> Optional[Tuple[Optional[str], Optional[list]]]:
        """从 LLM 输出中解析 JSON，处理可能的 markdown 代码块包裹

        Returns:
            (modality, body_parts) — body_parts 为部位列表
        """
        output = output.strip()

        if output.startswith("```json"):
            output = output[7:]
        if output.startswith("```"):
            output = output[3:]
        if output.endswith("```"):
            output = output[:-3]
        output = output.strip()

        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', output, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.debug("无法解析JSON: %s", output[:100])
                    return None
            else:
                logger.debug("未找到有效JSON: %s", output[:100])
                return None

        modality = data.get("modality")
        body_part = data.get("body_part")

        if modality in ("null", "None", "none", ""):
            modality = None
        if isinstance(modality, str):
            modality = modality.strip()

        if body_part in ("null", "None", "none", "", []):
            body_parts = []
        elif isinstance(body_part, list):
            body_parts = [p.strip() for p in body_part if isinstance(p, str) and p.strip() and p.strip() not in ("null", "None", "none", "")]
        elif isinstance(body_part, str):
            body_part = body_part.strip()
            body_parts = [body_part] if body_part and body_part not in ("null", "None", "none", "") else []
        else:
            body_parts = []

        return (modality, body_parts)

    @staticmethod
    def _extract_modality_rule(query: str) -> Optional[str]:
        """规则兜底：从查询中提取检查类型，按长度降序匹配避免误匹配"""
        query_upper = query.upper()
        for pattern in MODALITY_PATTERNS:
            if pattern.upper() in query_upper:
                return pattern
        return None

    @staticmethod
    def _extract_body_part_rule(query: str) -> list:
        """规则兜底：从查询中提取所有匹配的检查部位，按长度降序匹配。
        自动去重：如果短词是已匹配长词的子串，则跳过。"""
        parts = []
        for pattern in BODY_PART_PATTERNS:
            if pattern in query:
                # 检查是否被已匹配的长词包含（如"膝关节"已匹配，跳过"膝"）
                if not any(pattern in p and pattern != p for p in parts):
                    parts.append(pattern)
        return parts

    def _missing_modality(self, query: str) -> bool:
        """检查查询是否缺少检查类型"""
        if self._llm_chat is not None:
            return self.slots["modality"] is None and self._extract_modality_rule(query) is None
        return self._extract_modality_rule(query) is None

    def _missing_body_part(self, query: str) -> bool:
        """检查查询是否缺少检查部位"""
        if self._llm_chat is not None:
            return len(self.slots["body_part"]) == 0 and len(self._extract_body_part_rule(query)) == 0
        return len(self._extract_body_part_rule(query)) == 0

    # ── 意图识别 ──────────────────────────────────────────

    def detect_intent(self, query: str) -> str:
        """检测用户意图：new_session / append / switch

        - new_session: 首次查询或全新的独立请求
        - append: 补充/追加检查（如"再看看肝脏"）
        - switch: 明确切换话题（如"换成 MR 膝关节"）
        """
        if self._is_switch(query):
            return "switch"
        if self._is_reference(query) and self.slots["modality"] is not None:
            return "append"
        return "new_session"

    @staticmethod
    def _is_switch(query: str) -> bool:
        """检测是否切换意图"""
        for pattern in SWITCH_PATTERNS:
            if pattern in query:
                return True
        # 如果同时包含新模态和新部位，视为切换
        has_modality = EntityTracker._extract_modality_rule(query) is not None
        has_body_part = len(EntityTracker._extract_body_part_rule(query)) > 0
        return has_modality and has_body_part

    @staticmethod
    def _is_reference(query: str) -> bool:
        """检测是否包含指代/省略词"""
        return any(p in query for p in REFERENCE_PATTERNS)

    # ── 上下文消解 ──────────────────────────────────────────

    def resolve_context(self, query: str) -> str:
        """上下文消解：补全省略信息。

        当用户说"再看看肝脏"时，自动补全为"CT 肝脏 再看看肝脏"
        当用户说"这个病灶怎么样"时，自动补全上下文信息

        Returns:
            str: 消解后的完整查询
        """
        needs_modality = self._missing_modality(query)
        needs_body_part = self._missing_body_part(query)
        has_reference = self._is_reference(query)

        if not needs_modality and not needs_body_part and not has_reference:
            return query

        fill_parts = []
        if needs_modality and self.slots["modality"]:
            fill_parts.append(self.slots["modality"])
        if needs_body_part and self.slots["body_part"]:
            fill_parts.extend(self.slots["body_part"])

        if fill_parts:
            return " ".join(fill_parts) + " " + query

        return query

    # ── 状态管理 ──────────────────────────────────────────

    def clear(self):
        """重置所有槽位到初始状态"""
        self.slots = {
            "modality": None,
            "body_part": [],
            "clinical_history": "",
            "diagnosis": [],
            "intent": "new_session",
        }
        logger.debug("实体追踪器已重置")

    def apply_switch(self, query: str):
        """切换意图：清空旧状态，从新查询提取实体"""
        self.clear()
        self.update_from_query(query)
        self.slots["intent"] = "switch"
        logger.debug("实体追踪器已切换: %s", self.slots)

    def set_clinical_history(self, history: str):
        """设置病史"""
        self.slots["clinical_history"] = history

    def add_diagnosis(self, diagnosis: str):
        """添加诊断"""
        if diagnosis not in self.slots["diagnosis"]:
            self.slots["diagnosis"].append(diagnosis)

    def to_dict(self) -> dict:
        """序列化为字典"""
        return dict(self.slots)

    def to_context_prompt(self) -> str:
        """生成上下文提示片段，用于注入 System Prompt"""
        parts = []
        if self.slots["modality"]:
            parts.append(f"当前检查类型: {self.slots['modality']}")
        if self.slots["body_part"]:
            parts.append(f"当前检查部位: {'、'.join(self.slots['body_part'])}")
        if self.slots["clinical_history"]:
            parts.append(f"已知病史: {self.slots['clinical_history']}")
        if self.slots["diagnosis"]:
            parts.append(f"已确认诊断: {', '.join(self.slots['diagnosis'])}")

        if parts:
            return "## 当前上下文\n" + "\n".join(parts)
        return ""

    def __repr__(self) -> str:
        return f"EntityTracker(modality={self.slots['modality']}, body_part={self.slots['body_part']}, intent={self.slots['intent']})"