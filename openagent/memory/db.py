"""SQLite storage for scheduled tasks and usage logs."""

from __future__ import annotations

import json
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


VALID_MCP_KINDS = ("builtin", "custom", "default")

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

-- Configured MCP servers. Replaces the yaml ``mcp:`` list so the agent
-- itself (via the mcp-manager MCP) can add/remove/toggle servers at
-- runtime without a process restart. A one-shot import from yaml seeds
-- the table on first boot; subsequent yaml edits are ignored.
--
-- ``kind`` discriminates three sources:
--   - ``default``: one of DEFAULT_MCPS, resolved via resolve_default_entry
--   - ``builtin``: user opted-in to one of BUILTIN_MCP_SPECS
--   - ``custom``:  raw command/url entry (pre-resolved)
-- JSON columns are stored as TEXT to keep the schema portable; callers
-- wrap with json.dumps/loads at the Python layer.
CREATE TABLE IF NOT EXISTS mcps (
    name TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('builtin','custom','default')),
    builtin_name TEXT,
    command TEXT,
    args_json TEXT NOT NULL DEFAULT '[]',
    url TEXT,
    env_json TEXT NOT NULL DEFAULT '{}',
    headers_json TEXT NOT NULL DEFAULT '{}',
    oauth INTEGER NOT NULL DEFAULT 0,
    enabled INTEGER NOT NULL DEFAULT 1,
    source TEXT NOT NULL DEFAULT 'user',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mcps_enabled ON mcps(enabled);
CREATE INDEX IF NOT EXISTS idx_mcps_updated ON mcps(updated_at);

-- Configured LLM models. Replaces per-provider ``models:`` lists in yaml
-- so the agent (via the model-manager MCP) can add/remove/toggle models
-- at runtime. ``runtime_id`` is the canonical id (provider:model_id, or
-- claude-cli/model_id) used everywhere in code; see
-- openagent.models.catalog.build_runtime_model_id.
CREATE TABLE IF NOT EXISTS models (
    runtime_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    display_name TEXT,
    input_cost_per_million REAL,
    output_cost_per_million REAL,
    tier_hint TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_models_provider ON models(provider);
CREATE INDEX IF NOT EXISTS idx_models_enabled ON models(enabled);
CREATE INDEX IF NOT EXISTS idx_models_updated ON models(updated_at);

-- Generic string-valued state flags. Used for one-shot bootstrap
-- markers (``mcps_imported``, ``models_imported``) so yaml → DB import
-- runs exactly once per DB file.
CREATE TABLE IF NOT EXISTS config_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

-- Per-session runtime binding. SmartRouter dispatches fresh sessions
-- to either the Agno stack ("agno") or the Claude CLI
-- registry ("claude-cli") based on the classifier; once a session has
-- been served by one side its conversation state lives there
-- (Agno's SqliteDb for agno, Claude's own session store for claude-cli)
-- so the router must respect that lock on subsequent turns.
--
-- Claude-cli bindings are also persisted in ``sdk_sessions`` because
-- that table carries the SDK-native UUID needed for ``--resume``. This
-- table only needs to cover the agno case (no resume id to persist),
-- plus it serves as a fast single-table lookup for SmartRouter.
CREATE TABLE IF NOT EXISTS session_bindings (
    session_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    bound_at REAL NOT NULL
);
"""


class MemoryDB:
    """SQLite storage for scheduled tasks."""

    def __init__(self, db_path: str = "openagent.db"):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self._conn is not None:
            return
        # ``timeout`` is the SQLite-level wait when another connection holds a
        # write lock. WAL mode lets readers proceed without blocking writers,
        # but ``executescript(SCHEMA_SQL)`` below needs a write lock to
        # re-run CREATE TABLE IF NOT EXISTS DDL — and when the same process
        # already has a MemoryDB connection open (gateway agent + scheduler
        # MCP subprocess + a fresh per-test MemoryDB all pointing at the same
        # file), two DDL calls can race. Raise the timeout so the second
        # connect waits a few seconds instead of deadlocking the event loop.
        self._conn = await aiosqlite.connect(self.db_path, timeout=10.0)
        self._conn.row_factory = aiosqlite.Row
        # ``busy_timeout`` gives the same guarantee at every subsequent
        # statement on this connection — not just the initial open.
        await self._conn.execute("PRAGMA busy_timeout = 10000")
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

    # ── MCP Registry ──

    @staticmethod
    def _row_to_mcp(row: aiosqlite.Row) -> dict:
        """Deserialise JSON columns so callers see plain Python values.

        ``command``/``args``/``env``/``headers`` are stored as TEXT-wrapped
        JSON. Upstream (MCPPool.from_db, mcp-manager MCP) expects real
        lists/dicts, so we wrap every read instead of forcing each caller
        to remember.
        """
        d = dict(row)
        for col, default in (("args_json", "[]"), ("env_json", "{}"), ("headers_json", "{}")):
            raw = d.pop(col, default) or default
            key = col[:-5]  # strip "_json"
            try:
                d[key] = json.loads(raw)
            except (TypeError, ValueError):
                d[key] = [] if default == "[]" else {}
        # command is also JSON-wrapped (argv list); None when only url is set.
        raw_cmd = d.get("command")
        if raw_cmd:
            try:
                d["command"] = json.loads(raw_cmd)
            except (TypeError, ValueError):
                d["command"] = None
        d["enabled"] = bool(d.get("enabled"))
        d["oauth"] = bool(d.get("oauth"))
        return d

    async def list_mcps(self, enabled_only: bool = False) -> list[dict]:
        conn = await self._ensure_connected()
        if enabled_only:
            cursor = await conn.execute(
                "SELECT * FROM mcps WHERE enabled = 1 ORDER BY name ASC"
            )
        else:
            cursor = await conn.execute("SELECT * FROM mcps ORDER BY name ASC")
        rows = await cursor.fetchall()
        return [self._row_to_mcp(r) for r in rows]

    async def get_mcp(self, name: str) -> dict | None:
        conn = await self._ensure_connected()
        cursor = await conn.execute("SELECT * FROM mcps WHERE name = ?", (name,))
        row = await cursor.fetchone()
        return self._row_to_mcp(row) if row else None

    async def upsert_mcp(
        self,
        name: str,
        *,
        kind: str,
        builtin_name: str | None = None,
        command: list[str] | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        env: dict | None = None,
        headers: dict | None = None,
        oauth: bool = False,
        enabled: bool = True,
        source: str = "user",
    ) -> None:
        if kind not in VALID_MCP_KINDS:
            raise ValueError(f"invalid kind: {kind!r}")
        if not name:
            raise ValueError("name is required")
        conn = await self._ensure_connected()
        now = time.time()
        cmd_text: str | None
        if command:
            # Store the argv as a single shell-safe string. We keep it as TEXT
            # (not JSON) because the runtime treats command[0] specially
            # (absolute-path resolution in MCPPool._normalise_spec); shell-join
            # would re-parse at the wrong boundary. Use a JSON array instead.
            cmd_text = json.dumps(list(command))
        else:
            cmd_text = None
        await conn.execute(
            "INSERT INTO mcps (name, kind, builtin_name, command, args_json, url, "
            "env_json, headers_json, oauth, enabled, source, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET "
            "kind = excluded.kind, builtin_name = excluded.builtin_name, "
            "command = excluded.command, args_json = excluded.args_json, "
            "url = excluded.url, env_json = excluded.env_json, "
            "headers_json = excluded.headers_json, oauth = excluded.oauth, "
            "enabled = excluded.enabled, source = excluded.source, "
            "updated_at = excluded.updated_at",
            (
                name,
                kind,
                builtin_name,
                cmd_text,
                json.dumps(list(args or [])),
                url,
                json.dumps(dict(env or {})),
                json.dumps(dict(headers or {})),
                1 if oauth else 0,
                1 if enabled else 0,
                source,
                now,
                now,
            ),
        )
        await conn.commit()

    async def set_mcp_enabled(self, name: str, enabled: bool) -> None:
        conn = await self._ensure_connected()
        await conn.execute(
            "UPDATE mcps SET enabled = ?, updated_at = ? WHERE name = ?",
            (1 if enabled else 0, time.time(), name),
        )
        await conn.commit()

    async def delete_mcp(self, name: str) -> None:
        conn = await self._ensure_connected()
        await conn.execute("DELETE FROM mcps WHERE name = ?", (name,))
        await conn.commit()

    async def mcps_max_updated(self) -> float:
        """Return the most recent ``updated_at`` across mcps rows.

        Gateway polls this per message and triggers ``MCPPool.reload()`` when
        it increases. 0.0 when the table is empty — first boot will see
        a bump to the bootstrap write and reload once, which is fine.
        """
        conn = await self._ensure_connected()
        cursor = await conn.execute("SELECT MAX(updated_at) FROM mcps")
        row = await cursor.fetchone()
        return float(row[0] or 0.0) if row else 0.0

    # ── Model Registry ──

    @staticmethod
    def _row_to_model(row: aiosqlite.Row) -> dict:
        d = dict(row)
        raw = d.pop("metadata_json", "{}") or "{}"
        try:
            d["metadata"] = json.loads(raw)
        except (TypeError, ValueError):
            d["metadata"] = {}
        d["enabled"] = bool(d.get("enabled"))
        return d

    async def list_models(
        self, provider: str | None = None, enabled_only: bool = False
    ) -> list[dict]:
        conn = await self._ensure_connected()
        clauses = []
        params: list[Any] = []
        if provider:
            clauses.append("provider = ?")
            params.append(provider)
        if enabled_only:
            clauses.append("enabled = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = await conn.execute(
            f"SELECT * FROM models {where} ORDER BY provider ASC, model_id ASC",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_model(r) for r in rows]

    async def get_model(self, runtime_id: str) -> dict | None:
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM models WHERE runtime_id = ?", (runtime_id,)
        )
        row = await cursor.fetchone()
        return self._row_to_model(row) if row else None

    async def upsert_model(
        self,
        runtime_id: str,
        *,
        provider: str,
        model_id: str,
        display_name: str | None = None,
        input_cost: float | None = None,
        output_cost: float | None = None,
        tier_hint: str | None = None,
        enabled: bool = True,
        metadata: dict | None = None,
    ) -> None:
        if not runtime_id or not provider or not model_id:
            raise ValueError("runtime_id, provider and model_id are required")
        conn = await self._ensure_connected()
        now = time.time()
        await conn.execute(
            "INSERT INTO models (runtime_id, provider, model_id, display_name, "
            "input_cost_per_million, output_cost_per_million, tier_hint, enabled, "
            "metadata_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(runtime_id) DO UPDATE SET "
            "provider = excluded.provider, model_id = excluded.model_id, "
            "display_name = excluded.display_name, "
            "input_cost_per_million = excluded.input_cost_per_million, "
            "output_cost_per_million = excluded.output_cost_per_million, "
            "tier_hint = excluded.tier_hint, enabled = excluded.enabled, "
            "metadata_json = excluded.metadata_json, "
            "updated_at = excluded.updated_at",
            (
                runtime_id,
                provider,
                model_id,
                display_name,
                input_cost,
                output_cost,
                tier_hint,
                1 if enabled else 0,
                json.dumps(dict(metadata or {})),
                now,
                now,
            ),
        )
        await conn.commit()

    async def set_model_enabled(self, runtime_id: str, enabled: bool) -> None:
        conn = await self._ensure_connected()
        await conn.execute(
            "UPDATE models SET enabled = ?, updated_at = ? WHERE runtime_id = ?",
            (1 if enabled else 0, time.time(), runtime_id),
        )
        await conn.commit()

    async def delete_model(self, runtime_id: str) -> None:
        conn = await self._ensure_connected()
        await conn.execute("DELETE FROM models WHERE runtime_id = ?", (runtime_id,))
        await conn.commit()

    async def models_max_updated(self) -> float:
        conn = await self._ensure_connected()
        cursor = await conn.execute("SELECT MAX(updated_at) FROM models")
        row = await cursor.fetchone()
        return float(row[0] or 0.0) if row else 0.0

    async def registry_status(self) -> tuple[float, float, int]:
        """One-shot probe used by the gateway's per-message hot-reload loop.

        Returns ``(mcps_max_updated, models_max_updated, enabled_models_count)``
        in a single round-trip so the dispatcher doesn't pay three SELECTs
        per incoming message.
        """
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT "
            "  COALESCE((SELECT MAX(updated_at) FROM mcps), 0), "
            "  COALESCE((SELECT MAX(updated_at) FROM models), 0), "
            "  COALESCE((SELECT COUNT(*) FROM models WHERE enabled = 1), 0)"
        )
        row = await cursor.fetchone()
        if not row:
            return 0.0, 0.0, 0
        return float(row[0] or 0.0), float(row[1] or 0.0), int(row[2] or 0)

    # ── Session Runtime Bindings ──

    async def get_session_binding(self, session_id: str) -> str | None:
        """Return ``"agno"`` / ``"claude-cli"`` or ``None`` if unbound.

        Checks ``sdk_sessions`` first (source of truth for claude-cli
        resume state) and falls back to ``session_bindings`` for agno.
        """
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT provider FROM sdk_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row and row[0]:
            return str(row[0])
        cursor = await conn.execute(
            "SELECT provider FROM session_bindings WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        return str(row[0]) if row and row[0] else None

    async def set_session_binding(self, session_id: str, provider: str) -> None:
        """Record that ``session_id`` is served by ``provider``.

        Used by SmartRouter after a first successful dispatch so
        subsequent turns are forced to the same side. Claude-cli
        bindings land in ``sdk_sessions`` instead (via
        ``set_sdk_session``) — this table only tracks agno.
        """
        conn = await self._ensure_connected()
        await conn.execute(
            "INSERT INTO session_bindings (session_id, provider, bound_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "provider = excluded.provider, bound_at = excluded.bound_at",
            (session_id, provider, time.time()),
        )
        await conn.commit()

    async def delete_session_binding(self, session_id: str) -> None:
        conn = await self._ensure_connected()
        await conn.execute(
            "DELETE FROM session_bindings WHERE session_id = ?",
            (session_id,),
        )
        await conn.commit()

    # ── Generic state flags ──

    async def get_state(self, key: str) -> str | None:
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT value FROM config_state WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def set_state(self, key: str, value: str) -> None:
        conn = await self._ensure_connected()
        await conn.execute(
            "INSERT INTO config_state (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_at = excluded.updated_at",
            (key, value, time.time()),
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
