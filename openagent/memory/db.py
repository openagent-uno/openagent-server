"""SQLite-based storage for sessions, messages, and memories.

All data lives in SQLite — no files, no scattered storage.
WAL mode + proper indexes for fast concurrent access.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import aiosqlite


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_agent ON sessions(agent_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_agent_user ON sessions(agent_id, user_id);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    tool_calls TEXT,
    tool_call_id TEXT,
    tool_result TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_time ON messages(session_id, created_at);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    topic TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_agent_user ON memories(agent_id, user_id);
CREATE INDEX IF NOT EXISTS idx_memories_agent_user_topic ON memories(agent_id, user_id, topic);
CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at);
"""


class MemoryDB:
    """SQLite storage for all agent state."""

    def __init__(self, db_path: str = "openagent.db"):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self._conn is not None:
            return
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.executescript(SCHEMA_SQL)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def _ensure_connected(self) -> aiosqlite.Connection:
        if self._conn is None:
            await self.connect()
        return self._conn

    # ── Sessions ──

    async def create_session(self, agent_id: str, user_id: str = "", session_id: str | None = None) -> str:
        conn = await self._ensure_connected()
        sid = session_id or str(uuid.uuid4())
        now = time.time()
        await conn.execute(
            "INSERT OR IGNORE INTO sessions (id, agent_id, user_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (sid, agent_id, user_id, now, now),
        )
        await conn.commit()
        return sid

    async def get_session(self, session_id: str) -> dict | None:
        conn = await self._ensure_connected()
        cursor = await conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_or_create_session(self, agent_id: str, user_id: str = "", session_id: str | None = None) -> str:
        """Get existing session or create a new one. If session_id is given, reuse it."""
        if session_id:
            existing = await self.get_session(session_id)
            if existing:
                return session_id
        return await self.create_session(agent_id, user_id, session_id)

    async def touch_session(self, session_id: str) -> None:
        conn = await self._ensure_connected()
        await conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (time.time(), session_id))
        await conn.commit()

    # ── Messages ──

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str = "",
        tool_calls: list[dict] | None = None,
        tool_call_id: str | None = None,
        tool_result: str | None = None,
    ) -> str:
        """Store a single message immediately."""
        conn = await self._ensure_connected()
        msg_id = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO messages (id, session_id, role, content, tool_calls, tool_call_id, tool_result, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                msg_id, session_id, role, content,
                json.dumps(tool_calls) if tool_calls else None,
                tool_call_id, tool_result,
                time.time(),
            ),
        )
        await conn.commit()
        await self.touch_session(session_id)
        return msg_id

    async def add_messages_batch(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Batch insert multiple messages."""
        conn = await self._ensure_connected()
        now = time.time()
        rows = []
        for msg in messages:
            tc = msg.get("tool_calls")
            rows.append((
                str(uuid.uuid4()), session_id, msg["role"], msg.get("content", ""),
                json.dumps(tc) if tc else None,
                msg.get("tool_call_id"),
                msg.get("tool_result"),
                now,
            ))
        await conn.executemany(
            "INSERT INTO messages (id, session_id, role, content, tool_calls, tool_call_id, tool_result, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        await conn.commit()
        await self.touch_session(session_id)

    async def get_messages(self, session_id: str, limit: int = 50, offset: int = 0) -> list[dict]:
        """Get messages for a session, most recent last. Paginated."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC LIMIT ? OFFSET ?",
            (session_id, limit, offset),
        )
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            msg: dict[str, Any] = {"role": row["role"], "content": row["content"]}
            if row["tool_calls"]:
                msg["tool_calls"] = json.loads(row["tool_calls"])
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            if row["tool_result"]:
                msg["tool_result"] = row["tool_result"]
            result.append(msg)
        return result

    async def get_recent_messages(self, session_id: str, limit: int = 20) -> list[dict]:
        """Get the N most recent messages, returned in chronological order."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        )
        rows = await cursor.fetchall()
        rows = list(reversed(rows))
        result = []
        for row in rows:
            msg: dict[str, Any] = {"role": row["role"], "content": row["content"]}
            if row["tool_calls"]:
                msg["tool_calls"] = json.loads(row["tool_calls"])
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            result.append(msg)
        return result

    # ── Memories ──

    async def add_memory(self, agent_id: str, user_id: str, content: str, topic: str = "") -> str:
        conn = await self._ensure_connected()
        mem_id = str(uuid.uuid4())
        now = time.time()
        await conn.execute(
            "INSERT INTO memories (id, agent_id, user_id, content, topic, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (mem_id, agent_id, user_id, content, topic, now, now),
        )
        await conn.commit()
        return mem_id

    async def update_memory(self, memory_id: str, content: str, topic: str | None = None) -> None:
        conn = await self._ensure_connected()
        if topic is not None:
            await conn.execute(
                "UPDATE memories SET content = ?, topic = ?, updated_at = ? WHERE id = ?",
                (content, topic, time.time(), memory_id),
            )
        else:
            await conn.execute(
                "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
                (content, time.time(), memory_id),
            )
        await conn.commit()

    async def delete_memory(self, memory_id: str) -> None:
        conn = await self._ensure_connected()
        await conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        await conn.commit()

    async def get_memories(
        self,
        agent_id: str,
        user_id: str = "",
        topic: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Retrieve memories for an agent/user pair, optionally filtered by topic."""
        conn = await self._ensure_connected()
        if topic:
            cursor = await conn.execute(
                "SELECT * FROM memories WHERE agent_id = ? AND user_id = ? AND topic = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (agent_id, user_id, topic, limit),
            )
        else:
            cursor = await conn.execute(
                "SELECT * FROM memories WHERE agent_id = ? AND user_id = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (agent_id, user_id, limit),
            )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def search_memories(self, agent_id: str, user_id: str, query: str, limit: int = 10) -> list[dict]:
        """Simple text search in memory content."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM memories WHERE agent_id = ? AND user_id = ? AND content LIKE ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (agent_id, user_id, f"%{query}%", limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
