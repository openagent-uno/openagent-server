"""SQLite storage for scheduled tasks and usage logs."""

from __future__ import annotations

import time
import uuid
from typing import Any

import aiosqlite
from openagent.memory.schedule import (
    ONE_SHOT_PREFIX,
    build_one_shot_expression,
    is_one_shot_expression,
    parse_one_shot_expression,
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS scheduled_tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    cron_expression TEXT NOT NULL,
    prompt TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_run REAL,
    next_run REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_enabled ON scheduled_tasks(enabled);
CREATE INDEX IF NOT EXISTS idx_tasks_next_run ON scheduled_tasks(next_run);

CREATE TABLE IF NOT EXISTS usage_log (
    id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost REAL NOT NULL,
    session_id TEXT,
    year_month TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_year_month ON usage_log(year_month);
CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_log(timestamp);

-- Mapping from OpenAgent session_id (e.g. "tg:155490357") to the
-- provider-native session_id (e.g. Claude SDK UUID) so the provider can
-- --resume the correct transcript after a process restart. Without this
-- the in-memory mapping is wiped by any restart (OOM kill, auto-update,
-- manual restart) and the user's next message starts a brand-new
-- conversation — which presents as "agent forgot everything".
CREATE TABLE IF NOT EXISTS sdk_sessions (
    session_id TEXT PRIMARY KEY,
    sdk_session_id TEXT NOT NULL,
    provider TEXT,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sdk_sessions_updated ON sdk_sessions(updated_at);
"""


class MemoryDB:
    """SQLite storage for scheduled tasks."""

    def __init__(self, db_path: str = "openagent.db"):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self._conn is not None:
            return
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
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

    # ── Scheduled Tasks ──

    async def add_task(self, name: str, cron_expression: str, prompt: str, next_run: float | None = None) -> str:
        conn = await self._ensure_connected()
        task_id = str(uuid.uuid4())
        now = time.time()
        await conn.execute(
            "INSERT INTO scheduled_tasks (id, name, cron_expression, prompt, enabled, next_run, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 1, ?, ?, ?)",
            (task_id, name, cron_expression, prompt, next_run or now, now, now),
        )
        await conn.commit()
        return task_id

    async def get_tasks(self, enabled_only: bool = False) -> list[dict]:
        conn = await self._ensure_connected()
        if enabled_only:
            cursor = await conn.execute("SELECT * FROM scheduled_tasks WHERE enabled = 1 ORDER BY next_run ASC")
        else:
            cursor = await conn.execute("SELECT * FROM scheduled_tasks ORDER BY created_at DESC")
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_task(self, task_id: str) -> dict | None:
        conn = await self._ensure_connected()
        cursor = await conn.execute("SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def update_task(self, task_id: str, **kwargs: Any) -> None:
        conn = await self._ensure_connected()
        allowed = {"name", "cron_expression", "prompt", "enabled", "last_run", "next_run"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        updates["updated_at"] = time.time()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [task_id]
        await conn.execute(f"UPDATE scheduled_tasks SET {set_clause} WHERE id = ?", values)
        await conn.commit()

    async def delete_task(self, task_id: str) -> None:
        conn = await self._ensure_connected()
        await conn.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await conn.commit()

    async def get_due_tasks(self, now: float) -> list[dict]:
        """Get all enabled tasks whose next_run is <= now."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM scheduled_tasks WHERE enabled = 1 AND next_run <= ? ORDER BY next_run ASC",
            (now,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # ── Usage Tracking ──

    async def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        session_id: str | None = None,
    ) -> str:
        conn = await self._ensure_connected()
        row_id = str(uuid.uuid4())
        now = time.time()
        from datetime import datetime, timezone
        ym = datetime.now(timezone.utc).strftime("%Y-%m")
        await conn.execute(
            "INSERT INTO usage_log (id, timestamp, model, input_tokens, output_tokens, cost, session_id, year_month) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (row_id, now, model, input_tokens, output_tokens, cost, session_id, ym),
        )
        await conn.commit()
        return row_id

    async def get_monthly_usage(self, year_month: str | None = None) -> float:
        """Total cost for a given month (default: current month)."""
        conn = await self._ensure_connected()
        if year_month is None:
            from datetime import datetime, timezone
            year_month = datetime.now(timezone.utc).strftime("%Y-%m")
        cursor = await conn.execute(
            "SELECT COALESCE(SUM(cost), 0) FROM usage_log WHERE year_month = ?",
            (year_month,),
        )
        row = await cursor.fetchone()
        return float(row[0])

    async def get_usage_summary(self, year_month: str | None = None) -> dict[str, Any]:
        """Per-model breakdown for a given month."""
        conn = await self._ensure_connected()
        if year_month is None:
            from datetime import datetime, timezone
            year_month = datetime.now(timezone.utc).strftime("%Y-%m")
        cursor = await conn.execute(
            "SELECT model, SUM(cost) as total_cost, SUM(input_tokens) as total_in, "
            "SUM(output_tokens) as total_out, COUNT(*) as calls "
            "FROM usage_log WHERE year_month = ? GROUP BY model",
            (year_month,),
        )
        rows = await cursor.fetchall()
        by_model = {}
        total = 0.0
        for row in rows:
            r = dict(row)
            by_model[r["model"]] = round(r["total_cost"], 6)
            total += r["total_cost"]
        return {"total": round(total, 6), "by_model": by_model}

    # ── SDK Session Mapping ──

    async def set_sdk_session(
        self,
        session_id: str,
        sdk_session_id: str,
        provider: str | None = None,
    ) -> None:
        """Persist the ``session_id → sdk_session_id`` mapping for resume
        after restart. Callers typically fire-and-forget so provider latency
        isn't affected.
        """
        conn = await self._ensure_connected()
        await conn.execute(
            "INSERT INTO sdk_sessions (session_id, sdk_session_id, provider, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "sdk_session_id = excluded.sdk_session_id, "
            "provider = excluded.provider, "
            "updated_at = excluded.updated_at",
            (session_id, sdk_session_id, provider, time.time()),
        )
        await conn.commit()

    async def get_sdk_session(self, session_id: str) -> str | None:
        """Look up the provider-native session_id previously stored for ``session_id``."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT sdk_session_id FROM sdk_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def get_all_sdk_sessions(self, provider: str | None = None) -> dict[str, str]:
        """Return ``{session_id: sdk_session_id}`` for all (or one provider's) rows.

        Used on provider startup to hydrate the in-memory cache from disk so
        the first user message after a restart can resume the right transcript.
        """
        conn = await self._ensure_connected()
        if provider is None:
            cursor = await conn.execute(
                "SELECT session_id, sdk_session_id FROM sdk_sessions"
            )
        else:
            cursor = await conn.execute(
                "SELECT session_id, sdk_session_id FROM sdk_sessions WHERE provider = ?",
                (provider,),
            )
        rows = await cursor.fetchall()
        return {row[0]: row[1] for row in rows}

    async def delete_sdk_session(self, session_id: str) -> None:
        """Remove the stored ``session_id → sdk_session_id`` row.

        Called when the user explicitly asks to forget a conversation
        (``/clear``, ``/new``) so that the next message spawns a fresh
        subprocess without ``--resume`` instead of picking the old
        transcript back up.
        """
        conn = await self._ensure_connected()
        await conn.execute(
            "DELETE FROM sdk_sessions WHERE session_id = ?",
            (session_id,),
        )
        await conn.commit()

    async def get_daily_usage(self, days: int = 7) -> list[dict]:
        """Day-by-day usage breakdown grouped by model."""
        conn = await self._ensure_connected()
        cutoff = time.time() - (days * 86400)
        cursor = await conn.execute(
            "SELECT date(timestamp, 'unixepoch') as date, model, "
            "SUM(cost) as cost, SUM(input_tokens) as input_tokens, "
            "SUM(output_tokens) as output_tokens, COUNT(*) as request_count "
            "FROM usage_log WHERE timestamp >= ? "
            "GROUP BY date, model ORDER BY date DESC, cost DESC",
            (cutoff,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
