"""会话持久化存储

基于 SQLite 将会话数据（对话历史、实体槽位、上一轮报告）持久化到磁盘，
实现会话自动保存、断线恢复、历史会话查询与管理。

数据库路径: data/sessions.db（若目录不存在则自动创建）

表结构:
  - sessions:      会话元数据（id, title, created_at）
  - turns:         对话记录（session_id, turn_index, user_input, assistant_output）
  - session_state: 会话状态（entity_slots JSON, last_report, updated_at）
"""

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SessionStore:
    """SQLite 会话持久化存储"""

    def __init__(self, db_path: str = "data/sessions.db"):
        self._db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    # ── 数据库连接管理 ──────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ── 建表 ────────────────────────────────────────────────

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id         TEXT PRIMARY KEY,
                    title      TEXT NOT NULL DEFAULT '',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS turns (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT NOT NULL,
                    turn_index      INTEGER NOT NULL,
                    user_input      TEXT NOT NULL DEFAULT '',
                    assistant_output TEXT NOT NULL DEFAULT '',
                    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS session_state (
                    session_id   TEXT UNIQUE NOT NULL,
                    entity_slots TEXT NOT NULL DEFAULT '{}',
                    last_report  TEXT NOT NULL DEFAULT '',
                    updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session
                    ON turns(session_id, turn_index);
            """)

    # ── 会话元数据 ──────────────────────────────────────────

    def create_session(self, session_id: str, title: str = "") -> str:
        """创建新会话，返回 session_id"""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title) VALUES (?, ?)",
                (session_id, title),
            )
            conn.execute(
                "INSERT INTO session_state (session_id, entity_slots, last_report) VALUES (?, '{}', '')",
                (session_id,),
            )
        logger.info("会话已创建: %s", session_id)
        return session_id

    def update_title(self, session_id: str, title: str):
        """更新会话标题"""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id),
            )

    # ── 对话记录 ────────────────────────────────────────────

    def save_turn(self, session_id: str, turn_index: int, user_input: str, assistant_output: str):
        """保存单轮对话"""
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO turns (session_id, turn_index, user_input, assistant_output) VALUES (?, ?, ?, ?)",
                (session_id, turn_index, user_input, assistant_output),
            )

    def get_turns(self, session_id: str) -> list[dict]:
        """获取会话的所有对话记录，按 turn_index 升序"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT turn_index, user_input, assistant_output FROM turns WHERE session_id = ? ORDER BY turn_index ASC",
                (session_id,),
            ).fetchall()
        return [{"turn_index": r["turn_index"], "user_input": r["user_input"], "assistant_output": r["assistant_output"]} for r in rows]

    # ── 会话状态 ────────────────────────────────────────────

    def save_state(self, session_id: str, entity_slots: dict, last_report: str):
        """保存会话上下文状态（实体槽位 + 上一轮报告）"""
        slots_json = json.dumps(entity_slots, ensure_ascii=False)
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO session_state (session_id, entity_slots, last_report, updated_at) "
                "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(session_id) DO UPDATE SET entity_slots=excluded.entity_slots, last_report=excluded.last_report, updated_at=CURRENT_TIMESTAMP",
                (session_id, slots_json, last_report),
            )

    def load_state(self, session_id: str) -> Optional[dict]:
        """加载会话状态，返回 {entity_slots, last_report} 或 None"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT entity_slots, last_report FROM session_state WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        try:
            slots = json.loads(row["entity_slots"])
        except (json.JSONDecodeError, TypeError):
            slots = {}
        return {
            "entity_slots": slots,
            "last_report": row["last_report"] or "",
        }

    # ── 完整加载/恢复 ───────────────────────────────────────

    def load_session(self, session_id: str) -> Optional[dict]:
        """加载完整会话数据，返回 {title, turns, state} 或 None"""
        with self._get_conn() as conn:
            session_row = conn.execute(
                "SELECT id, title, created_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if session_row is None:
            return None

        turns = self.get_turns(session_id)
        state = self.load_state(session_id) or {"entity_slots": {}, "last_report": ""}

        return {
            "id": session_row["id"],
            "title": session_row["title"],
            "created_at": session_row["created_at"],
            "turns": turns,
            "state": state,
        }

    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return row is not None

    # ── 会话列表 ────────────────────────────────────────────

    def list_sessions(self) -> list[dict]:
        """查询所有会话，按 created_at 倒序返回"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT s.id, s.title, s.created_at,
                          (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id) AS turn_count
                   FROM sessions s
                   ORDER BY s.created_at DESC"""
            ).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "turn_count": r["turn_count"],
            }
            for r in rows
        ]

    # ── 删除 ────────────────────────────────────────────────

    def delete_session(self, session_id: str):
        """级联删除会话、对话记录和状态"""
        with self._get_conn() as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        logger.info("会话已删除: %s", session_id)

    # ── 工具方法 ────────────────────────────────────────────

    @staticmethod
    def generate_session_id() -> str:
        """生成新的会话 ID"""
        return f"rag_v2_{uuid.uuid4().hex[:8]}"