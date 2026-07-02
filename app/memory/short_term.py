"""短期记忆管理器

基于内存的对话历史与实体追踪，增强多轮对话的上下文理解能力。

设计决策：
- 存储：内存 dict（重启丢失）
- 历史长度：最近 5 轮
- 会话标识：session_id 字符串，由调用方传入
- 定位：增强多轮对话上下文理解，不替代任何框架的对话历史
- 解耦：实体由调用方传入，不依赖特定业务领域
- 槽位填充：逐轮追踪实体，缺失槽位自动从历史填充，置信度按 0.9 衰减
- 智能摘要：淘汰旧轮次时自动压缩为摘要句，防止关键信息丢失
"""

import logging
import threading
import time
from collections import OrderedDict
from typing import Callable

logger = logging.getLogger(__name__)

REFERENCE_PATTERNS = [
    "那个", "刚才", "上面", "之前", "上文", "前文",
    "刚刚", "那个病", "那个检查", "那次",
]

FOLLOW_UP_PATTERNS = [
    "有", "有没有", "是否", "是不是", "会不会",
    "还有", "另外", "再", "也",
    "呢", "吗", "么",
]

MODALITY_PATTERNS = [
    "PET-CT", "PET", "SPECT", "DSA", "CTA", "MRA", "DWI", "SWI", "FLAIR",
    "CT", "MRI", "MR", "X线", "X光", "超声", "B超",
]

DECAY_FACTOR = 0.9

SUMMARIZE_PROMPT = (
    "请将以下对话轮次压缩为一句简洁的摘要（不超过50字），"
    "保留关键实体、核心结论、数值信息，忽略寒暄和无关内容。"
    "只输出摘要句，不要加任何前缀或解释。"
)


class ShortTermMemory:
    def __init__(self, max_rounds: int = 5, decay_factor: float = 0.9,
                 summarize_fn: Callable[[list[dict]], str] = None):
        self._sessions: dict[str, OrderedDict] = {}
        self._entities: dict[str, list[dict]] = {}
        self._counters: dict[str, int] = {}
        self._summaries: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self.max_rounds = max_rounds
        self.decay_factor = decay_factor
        self._summarize_fn = summarize_fn

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str, entities: dict = None):
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = OrderedDict()
                self._entities[session_id] = []
                self._counters[session_id] = 0
                self._summaries[session_id] = []

            self._counters[session_id] += 1
            turn_num = self._counters[session_id]
            turn_key = f"turn_{turn_num}"

            turns = self._sessions[session_id]
            turns[turn_key] = {
                "user": user_msg,
                "assistant": assistant_msg,
                "timestamp": time.time(),
            }

            while len(turns) > self.max_rounds:
                key, evicted_turn = turns.popitem(last=False)
                self._evict_and_summarize(session_id, evicted_turn)

            if entities:
                entry = dict(entities)
                entry["_round"] = turn_num
                self._entities[session_id].append(entry)

    def _evict_and_summarize(self, session_id: str, turn: dict):
        summary = self._summarize_turn(turn)
        if summary:
            self._summaries[session_id].append(summary)
            logger.debug("摘要已生成: %s", summary[:60])

    def _summarize_turn(self, turn: dict) -> str:
        if self._summarize_fn is not None:
            try:
                messages = [
                    {"role": "system", "content": SUMMARIZE_PROMPT},
                    {"role": "user", "content": f"用户: {turn['user']}\n助手: {turn['assistant']}"},
                ]
                return self._summarize_fn(messages)
            except Exception:
                logger.debug("LLM 摘要失败，回退到规则摘要")
        return self._rule_based_summary(turn)

    @staticmethod
    def _rule_based_summary(turn: dict) -> str:
        user = turn["user"][:80]
        assistant = turn["assistant"][:80]
        if len(turn["user"]) > 80:
            user += "..."
        if len(turn["assistant"]) > 80:
            assistant += "..."
        return f"用户: {user} | AI: {assistant}"

    def add_entities(self, session_id: str, **entities):
        with self._lock:
            if session_id not in self._entities:
                self._entities[session_id] = []
            entry = dict(entities)
            entry["_round"] = self._counters.get(session_id, 0)
            self._entities[session_id].append(entry)

    def get_history(self, session_id: str) -> list[dict]:
        with self._lock:
            turns = self._sessions.get(session_id)
            if not turns:
                return []
            messages = []
            for turn in turns.values():
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})
            return messages

    def get_last_turn(self, session_id: str) -> dict | None:
        with self._lock:
            turns = self._sessions.get(session_id)
            if not turns:
                return None
            last_key = next(reversed(turns))
            return dict(turns[last_key])

    def get_entities(self, session_id: str) -> dict:
        with self._lock:
            records = self._entities.get(session_id, [])
            merged = {}
            for record in records:
                for key, val in record.items():
                    if key.startswith("_"):
                        continue
                    if val is None or (isinstance(val, list) and not val):
                        continue
                    if key not in merged:
                        merged[key] = val
                    elif isinstance(val, list):
                        existing = merged.get(key, [])
                        if not isinstance(existing, list):
                            existing = [existing]
                        merged[key] = existing + [v for v in val if v not in existing]
                    elif val != merged[key]:
                        merged[key] = val
            return merged

    def _get_slot_values(self, session_id: str) -> dict[str, tuple]:
        records = self._entities.get(session_id, [])
        current_round = self._counters.get(session_id, 0)
        slot_best: dict[str, tuple] = {}
        for record in records:
            r = record.get("_round", 1)
            weight = self.decay_factor ** (current_round - r)
            for key, val in record.items():
                if key.startswith("_"):
                    continue
                if val is None or (isinstance(val, list) and not val):
                    continue
                prev = slot_best.get(key)
                if prev is None or weight > prev[1]:
                    slot_best[key] = (val, weight)
        return slot_best

    @staticmethod
    def _extract_modality(query: str) -> str | None:
        """从查询中提取检查类型（CT/MRI等），按长度降序匹配避免误匹配"""
        query_upper = query.upper()
        for pattern in MODALITY_PATTERNS:
            if pattern.upper() in query_upper:
                return pattern
        return None

    @staticmethod
    def _missing_modality(query: str) -> bool:
        """检查查询是否缺少检查类型"""
        return ShortTermMemory._extract_modality(query) is None

    def _extract_modality_from_history(self, session_id: str) -> str | None:
        """从历史对话中提取最近一轮的检查类型"""
        turns = self._sessions.get(session_id)
        if not turns:
            return None
        for turn in reversed(turns.values()):
            modality = self._extract_modality(turn["user"])
            if modality:
                return modality
        return None

    def _has_reference(self, query: str) -> bool:
        return any(p in query for p in REFERENCE_PATTERNS)

    def _is_follow_up(self, query: str) -> bool:
        if len(query) <= 6:
            return True
        if any(p in query for p in FOLLOW_UP_PATTERNS):
            return True
        return False

    def resolve_context(self, session_id: str, current_query: str) -> str:
        with self._lock:
            # 优先：如果当前查询缺少检查类型，从历史继承
            if self._missing_modality(current_query):
                modality = self._extract_modality_from_history(session_id)
                if modality:
                    return f"{modality} {current_query}"

            slot_values = self._get_slot_values(session_id)
            if not slot_values:
                return current_query

            needs_resolution = self._has_reference(current_query) or self._is_follow_up(current_query)
            if not needs_resolution:
                return current_query

            fill_parts = []
            for slot, (val, weight) in slot_values.items():
                if isinstance(val, list):
                    for v in val:
                        if v and v not in current_query:
                            fill_parts.append(f"{slot}:{v}({weight:.2f})")
                else:
                    if val and val not in current_query:
                        fill_parts.append(f"{slot}:{val}({weight:.2f})")

            if not fill_parts:
                return current_query

            return f"{current_query}（上下文参考: {' | '.join(fill_parts)}）"

    def clear(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)
            self._entities.pop(session_id, None)
            self._counters.pop(session_id, None)
            self._summaries.pop(session_id, None)

    def build_messages(self, session_id: str, system_prompt: str, current_user_msg: str) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt}]
        summaries = self._summaries.get(session_id, [])
        if summaries:
            summary_text = "## 历史对话摘要（已淘汰轮次的压缩）\n" + "\n".join(
                f"- {s}" for s in summaries
            )
            messages.append({"role": "system", "content": summary_text})
        history = self.get_history(session_id)
        messages.extend(history)
        messages.append({"role": "user", "content": current_user_msg})
        return messages

    def get_summaries(self, session_id: str) -> list[str]:
        with self._lock:
            return list(self._summaries.get(session_id, []))

    def active_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

    def session_info(self, session_id: str) -> dict:
        with self._lock:
            turns = self._sessions.get(session_id)
            counter = self._counters.get(session_id, 0)
            summaries = self._summaries.get(session_id, [])
            return {
                "session_id": session_id,
                "turns": len(turns) if turns else 0,
                "total_turns": counter,
                "max_rounds": self.max_rounds,
                "entities": self.get_entities(session_id),
                "summary_count": len(summaries),
            }

    def cleanup_expired(self, max_age_seconds: int = 3600):
        now = time.time()
        with self._lock:
            expired = []
            for sid, turns in self._sessions.items():
                if not turns:
                    expired.append(sid)
                    continue
                last_turn = next(reversed(turns.values()))
                if now - last_turn.get("timestamp", 0) > max_age_seconds:
                    expired.append(sid)
            for sid in expired:
                self._sessions.pop(sid, None)
                self._entities.pop(sid, None)
                self._counters.pop(sid, None)
                self._summaries.pop(sid, None)
            return len(expired)