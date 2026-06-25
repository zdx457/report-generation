"""长期记忆管理器

跨会话持久化存储用户偏好、修正历史，重启不丢失。

设计决策：
- 存储：SQLite 数据库（原子事务，无文件锁竞争，Python 内置无额外依赖）
- 缓存：内存副本，读写不落盘，仅在 on_session_end 或定期（5分钟）持久化
- 用户标识：user_id（默认 "default"）
- 偏好：时间衰减加权（指数衰减），而非简单累加频次
- 修正历史：few-shot 示例
- 线程安全
- 解耦：不依赖具体领域字段，任意 key-value 均可统计

SQLite 表结构：
  preferences(user_id, key, value, timestamp)   — 偏好记录
  corrections(user_id, id, question, original, corrected, timestamp) — 修正历史
  stats(user_id, total_sessions, total_turns, last_updated) — 统计
"""

import json
import logging
import math
import os
import sqlite3
import threading
import time

logger = logging.getLogger(__name__)

DB_FILENAME = "ltm.db"


class LongTermMemory:

    def __init__(self, data_dir: str = None, user_id: str = "default",
                 half_life_days: float = 7.0, max_age_days: float = 30.0,
                 flush_interval: float = 300.0):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self._data_dir = data_dir
        self._user_id = user_id
        self._lock = threading.Lock()

        half_life_seconds = half_life_days * 86400.0
        self._decay_lambda = math.log(2) / half_life_seconds
        self._max_age_seconds = max_age_days * 86400.0
        self._flush_interval = flush_interval

        self._cache = {
            "preferences": {},
            "corrections": [],
            "stats": {
                "total_sessions": 0,
                "total_turns": 0,
                "last_updated": time.time(),
            },
        }
        self._dirty = False
        self._flush_timer = None

        os.makedirs(self._data_dir, exist_ok=True)
        self._init_db()
        self._migrate_from_json()
        self._load_from_db()

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def db_path(self) -> str:
        return os.path.join(self._data_dir, DB_FILENAME)

    @property
    def json_path(self) -> str:
        return os.path.join(self._data_dir, f"{self._user_id}.json")

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS preferences (
                    user_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pref_user_key
                ON preferences(user_id, key)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS corrections (
                    user_id TEXT NOT NULL,
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    original TEXT NOT NULL,
                    corrected TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_corr_user
                ON corrections(user_id)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stats (
                    user_id TEXT PRIMARY KEY,
                    total_sessions INTEGER DEFAULT 0,
                    total_turns INTEGER DEFAULT 0,
                    last_updated REAL DEFAULT 0
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _migrate_from_json(self):
        if not os.path.exists(self.json_path):
            return
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                old = json.load(f)

            old_prefs = old.get("preferences", {})
            old_corrs = old.get("corrections", [])
            old_stats = old.get("stats", {})

            migrated_prefs = self._migrate_preferences(old_prefs)

            conn = self._get_conn()
            try:
                for key, records in migrated_prefs.items():
                    for r in records:
                        conn.execute(
                            "INSERT INTO preferences(user_id, key, value, timestamp) VALUES(?,?,?,?)",
                            (self._user_id, key, r.get("v", ""), r.get("ts", time.time())),
                        )
                for c in old_corrs:
                    conn.execute(
                        "INSERT INTO corrections(user_id, question, original, corrected, timestamp) VALUES(?,?,?,?,?)",
                        (self._user_id, c.get("question", ""), c.get("original", ""),
                         c.get("corrected", ""), c.get("timestamp", time.time())),
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO stats(user_id, total_sessions, total_turns, last_updated) VALUES(?,?,?,?)",
                    (self._user_id, old_stats.get("total_sessions", 0),
                     old_stats.get("total_turns", 0), old_stats.get("last_updated", time.time())),
                )
                conn.commit()
            finally:
                conn.close()

            backup_path = self.json_path + ".bak"
            os.rename(self.json_path, backup_path)
            logger.info("JSON 已迁移至 SQLite，备份: %s", backup_path)
        except Exception as e:
            logger.warning("JSON 迁移失败: %s", e)

    @staticmethod
    def _migrate_preferences(prefs: dict) -> dict:
        now = time.time()
        migrated = {}
        for key, val in prefs.items():
            if isinstance(val, dict):
                first_entry = next(iter(val.values()), None)
                if isinstance(first_entry, dict) and "v" in first_entry:
                    migrated[key] = val
                else:
                    migrated[key] = [{"v": str(v), "ts": now} for v in val]
            elif isinstance(val, list):
                migrated[key] = val
            else:
                migrated[key] = [{"v": str(val), "ts": now}]
        return migrated

    def _load_from_db(self):
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT key, value, timestamp FROM preferences WHERE user_id = ? ORDER BY timestamp",
                (self._user_id,),
            ).fetchall()
            prefs = {}
            for key, value, ts in rows:
                if key not in prefs:
                    prefs[key] = []
                prefs[key].append({"v": value, "ts": ts})

            corr_rows = conn.execute(
                "SELECT question, original, corrected, timestamp FROM corrections WHERE user_id = ? ORDER BY id",
                (self._user_id,),
            ).fetchall()
            corrections = [
                {"question": q, "original": o, "corrected": c, "timestamp": ts}
                for q, o, c, ts in corr_rows
            ]

            stat_row = conn.execute(
                "SELECT total_sessions, total_turns, last_updated FROM stats WHERE user_id = ?",
                (self._user_id,),
            ).fetchone()
            stats = {
                "total_sessions": stat_row[0] if stat_row else 0,
                "total_turns": stat_row[1] if stat_row else 0,
                "last_updated": stat_row[2] if stat_row else time.time(),
            }

            self._cache["preferences"] = prefs
            self._cache["corrections"] = corrections
            self._cache["stats"] = stats

            pref_count = sum(len(v) for v in prefs.values())
            logger.info("长期记忆已加载(SQLite): %s (偏好: %d类, 记录: %d条, 修正: %d条)",
                        self._user_id, len(prefs), pref_count, len(corrections))
        except Exception as e:
            logger.warning("加载长期记忆失败: %s", e)
        finally:
            conn.close()

    def _flush_to_db(self):
        if not self._dirty:
            return
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM preferences WHERE user_id = ?", (self._user_id,))
            for key, records in self._cache["preferences"].items():
                for r in records:
                    conn.execute(
                        "INSERT INTO preferences(user_id, key, value, timestamp) VALUES(?,?,?,?)",
                        (self._user_id, key, r.get("v", ""), r.get("ts", 0)),
                    )

            conn.execute("DELETE FROM corrections WHERE user_id = ?", (self._user_id,))
            for c in self._cache["corrections"]:
                conn.execute(
                    "INSERT INTO corrections(user_id, question, original, corrected, timestamp) VALUES(?,?,?,?,?)",
                    (self._user_id, c.get("question", ""), c.get("original", ""),
                     c.get("corrected", ""), c.get("timestamp", 0)),
                )

            stats = self._cache["stats"]
            conn.execute(
                "INSERT OR REPLACE INTO stats(user_id, total_sessions, total_turns, last_updated) VALUES(?,?,?,?)",
                (self._user_id, stats["total_sessions"], stats["total_turns"], stats["last_updated"]),
            )

            conn.commit()
            self._dirty = False
            logger.debug("长期记忆已刷新至 SQLite: %s", self._user_id)
        except Exception as e:
            logger.error("刷新长期记忆失败: %s", e)
        finally:
            conn.close()

    def _schedule_flush(self):
        if self._flush_timer is not None:
            return
        self._flush_timer = threading.Timer(self._flush_interval, self._on_flush_timer)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _on_flush_timer(self):
        with self._lock:
            self._flush_timer = None
            if self._dirty:
                self._flush_to_db()

    def _cancel_flush_timer(self):
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None

    def _mark_dirty(self):
        self._dirty = True
        self._schedule_flush()

    # ── 偏好 ──────────────────────────────────────────────

    def update_preferences(self, entities: dict):
        with self._lock:
            now = time.time()
            changed = False
            prefs = self._cache["preferences"]

            for key, val in entities.items():
                if key not in prefs:
                    prefs[key] = []

                if isinstance(val, list):
                    for item in val:
                        if item:
                            prefs[key].append({"v": str(item), "ts": now})
                            changed = True
                elif val:
                    prefs[key].append({"v": str(val), "ts": now})
                    changed = True

            if changed:
                self._purge_expired()
                self._mark_dirty()

    def _purge_expired(self):
        cutoff = time.time() - self._max_age_seconds
        for key in list(self._cache["preferences"].keys()):
            records = self._cache["preferences"][key]
            self._cache["preferences"][key] = [r for r in records if r.get("ts", 0) > cutoff]
            if not self._cache["preferences"][key]:
                del self._cache["preferences"][key]

    def _compute_weighted_scores(self) -> dict[str, list[tuple[str, float]]]:
        now = time.time()
        result = {}
        for key, records in self._cache["preferences"].items():
            scores = {}
            for r in records:
                v = r.get("v", "")
                ts = r.get("ts", now)
                if not v:
                    continue
                weight = math.exp(-self._decay_lambda * (now - ts))
                scores[v] = scores.get(v, 0.0) + weight
            if scores:
                sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                result[key] = sorted_items
        return result

    def get_preferences(self) -> dict:
        with self._lock:
            weighted = self._compute_weighted_scores()
            result = {}
            for key, items in weighted.items():
                result[key] = {
                    "top": [item[0] for item in items[:5]],
                    "scores": {item[0]: round(item[1], 4) for item in items},
                }
            return result

    def get_preference_prompt(self) -> str:
        prefs = self.get_preferences()
        if not prefs:
            return ""

        lines = []
        for key, info in prefs.items():
            top = info.get("top", [])
            if top:
                lines.append(f"用户常用 {key}: {', '.join(top)}")

        if lines:
            return "## 用户偏好\n" + "\n".join(lines)
        return ""

    # ── 修正历史 ──────────────────────────────────────────

    def add_correction(self, question: str, original_answer: str, corrected_answer: str):
        with self._lock:
            entry = {
                "question": question,
                "original": original_answer,
                "corrected": corrected_answer,
                "timestamp": time.time(),
            }
            self._cache["corrections"].append(entry)
            if len(self._cache["corrections"]) > 20:
                self._cache["corrections"] = self._cache["corrections"][-20:]
            self._mark_dirty()
            logger.info("修正历史已记录: %s...", question[:30])

    def get_corrections(self, limit: int = 5) -> list[dict]:
        with self._lock:
            return self._cache["corrections"][-limit:]

    def get_correction_prompt(self) -> str:
        corrections = self.get_corrections(limit=3)
        if not corrections:
            return ""

        lines = ["## 历史修正参考（请参考以下修正模式，避免重复之前的错误）"]
        for i, c in enumerate(corrections, 1):
            lines.append(f"修正{i}: 问题「{c['question']}」→ 原文「{c['original'][:80]}」→ 修正为「{c['corrected'][:80]}」")
        return "\n".join(lines)

    # ── 生命周期 ──────────────────────────────────────────

    def sync_from_short_term(self, short_term_memory, session_id: str):
        entities = short_term_memory.get_entities(session_id)
        if entities:
            self.update_preferences(entities)

    def on_session_end(self, short_term_memory, session_id: str):
        self.sync_from_short_term(short_term_memory, session_id)
        total_turns = short_term_memory.session_info(session_id).get("total_turns", 0)
        with self._lock:
            self._cache["stats"]["total_sessions"] += 1
            self._cache["stats"]["total_turns"] += total_turns
            self._cache["stats"]["last_updated"] = time.time()
            self._cancel_flush_timer()
            self._flush_to_db()
        logger.info("会话结束: %s (本会话 %d 轮)", self._user_id, total_turns)

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._cache["stats"])

    def clear(self):
        with self._lock:
            self._cache["preferences"] = {}
            self._cache["corrections"] = []
            self._cache["stats"] = {"total_sessions": 0, "total_turns": 0, "last_updated": time.time()}
            self._cancel_flush_timer()
            self._flush_to_db()
            logger.info("长期记忆已清空: %s", self._user_id)

    def clear_file(self):
        self.clear()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            logger.info("长期记忆数据库已删除: %s", self.db_path)

    def close(self):
        self._cancel_flush_timer()
        with self._lock:
            if self._dirty:
                self._flush_to_db()
        logger.info("长期记忆已关闭: %s", self._user_id)

    def __del__(self):
        try:
            self._cancel_flush_timer()
        except Exception:
            pass