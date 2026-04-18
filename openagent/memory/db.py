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

-- Configured MCP servers. The agent itself (via the mcp-manager MCP)
-- can add/remove/toggle servers at runtime without a process restart.
-- A one-shot import from yaml seeds the table on first boot;
-- subsequent yaml edits are ignored.
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

-- Configured LLM models. The model catalog lives here so the agent
-- (via the model-manager MCP) can add/remove/toggle models at runtime.
-- ``runtime_id`` is the canonical id (provider:model_id, or
-- claude-cli/model_id) used everywhere in code; see
-- openagent.models.catalog.build_runtime_model_id.
-- OpenAgent vocabulary (since v0.10.0):
--   provider  = vendor/owner (anthropic, openai, google, z.ai, local…)
--   framework = runtime dispatching the model (agno | claude-cli)
--   model_id  = bare vendor id (gpt-4o-mini, claude-sonnet-4-6…)
-- The same (provider, model_id) can run under different frameworks —
-- notably anthropic models, which run under Agno (direct API) OR under
-- the local Claude CLI binary (Pro/Max subscription). ``runtime_id``
-- encodes all three: ``<provider>:<model>`` for framework=agno, and
-- ``claude-cli:<provider>:<model>`` for framework=claude-cli.
CREATE TABLE IF NOT EXISTS models (
    runtime_id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    framework TEXT NOT NULL DEFAULT 'agno',
    model_id TEXT NOT NULL,
    display_name TEXT,
    tier_hint TEXT,
    notes TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_models_provider ON models(provider);
CREATE INDEX IF NOT EXISTS idx_models_enabled ON models(enabled);
CREATE INDEX IF NOT EXISTS idx_models_updated ON models(updated_at);

-- Generic string-valued state flags. Used for one-shot bootstrap
-- markers (``mcps_imported``) so the yaml → DB MCP import runs
-- exactly once per DB file.
CREATE TABLE IF NOT EXISTS config_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

-- LLM provider credentials. One row per vendor (openai, anthropic,
-- google, zai, …). API keys are stored plaintext; the DB file is
-- owned by the user with 0600 perms.
CREATE TABLE IF NOT EXISTS providers (
    name TEXT PRIMARY KEY,
    api_key TEXT,
    base_url TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_providers_enabled ON providers(enabled);
CREATE INDEX IF NOT EXISTS idx_providers_updated ON providers(updated_at);

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
    bound_at REAL NOT NULL,
    -- Optional per-session *specific* model override. When set,
    -- SmartRouter skips the classifier and dispatches this session
    -- straight to ``runtime_id`` regardless of the picked tier.
    -- NULL means "use the routing table normally for this side".
    runtime_id TEXT
);
"""

# Column-add migrations for tables that exist pre-v0.9.2 / pre-v0.10.
# SQLite's ``CREATE TABLE IF NOT EXISTS`` can't evolve a schema; these
# ``ALTER TABLE`` statements run at every connect and swallow the
# "column already exists" OperationalError on subsequent boots. Safe to
# append more entries as the schema grows.
_SCHEMA_MIGRATIONS = (
    "ALTER TABLE session_bindings ADD COLUMN runtime_id TEXT",
    "ALTER TABLE models ADD COLUMN framework TEXT NOT NULL DEFAULT 'agno'",
    "ALTER TABLE models ADD COLUMN notes TEXT",
    # Drop legacy cost columns. Pricing is resolved live via
    # ``catalog.get_model_pricing`` (claude-cli → 0; agno → OpenRouter
    # cache); the static columns went stale every time a vendor changed
    # tariffs. Requires SQLite ≥ 3.35 — older builds raise OperationalError
    # which is swallowed below, leaving the dead columns in place harmlessly.
    "ALTER TABLE models DROP COLUMN input_cost_per_million",
    "ALTER TABLE models DROP COLUMN output_cost_per_million",
)


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
        # Run column-add migrations for tables that can't be fully
        # re-specified via ``CREATE TABLE IF NOT EXISTS``. Each statement
        # is idempotent-by-try/except: SQLite raises OperationalError
        # when the column already exists, and we swallow that.
        for stmt in _SCHEMA_MIGRATIONS:
            try:
                await self._conn.execute(stmt)
            except aiosqlite.OperationalError:
                pass
        await self._conn.commit()
        # v0.10 data fix-up: pre-v0.10 rows used ``provider='claude-cli'``
        # as a pseudo-provider. In the new vocabulary claude-cli is a
        # *framework* and the underlying provider is ``anthropic``.
        # Translate those rows once so the catalog + SmartRouter see
        # consistent values. Idempotent: subsequent boots find nothing
        # to rewrite because ``framework`` is already set.
        await self._migrate_models_provider_to_framework()

    async def _migrate_models_provider_to_framework(self) -> None:
        """One-shot rewrite of legacy claude-cli rows in the ``models`` table.

        Before v0.10, claude-cli was stored as ``provider='claude-cli'``
        with ``runtime_id='claude-cli/<model>'``. The new shape is
        ``provider='anthropic'``, ``framework='claude-cli'``, and
        ``runtime_id='claude-cli:anthropic:<model>'``. Runs quietly —
        no-op on fresh databases and on already-migrated ones.
        """
        assert self._conn is not None
        try:
            cursor = await self._conn.execute(
                "SELECT runtime_id, model_id FROM models WHERE provider = 'claude-cli'"
            )
            rows = await cursor.fetchall()
        except aiosqlite.OperationalError:
            return
        for row in rows:
            old_rid = row[0]
            model_id = row[1]
            new_rid = f"claude-cli:anthropic:{model_id}"
            try:
                await self._conn.execute(
                    "UPDATE models SET provider = 'anthropic', framework = 'claude-cli', "
                    "runtime_id = ? WHERE runtime_id = ?",
                    (new_rid, old_rid),
                )
            except aiosqlite.IntegrityError:
                # New rid already exists (user re-added under new shape).
                # Drop the legacy row to keep the table consistent.
                await self._conn.execute(
                    "DELETE FROM models WHERE runtime_id = ?", (old_rid,)
                )
        if rows:
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
        # Pre-v0.10 rows may not have the ``framework`` column populated
        # until the migration runs (edge case: a row returned by another
        # aiosqlite connection in the middle of the startup race). Fall
        # back so downstream code doesn't need to special-case the key.
        d.setdefault("framework", "agno")
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
        framework: str = "agno",
        display_name: str | None = None,
        tier_hint: str | None = None,
        notes: str | None = None,
        enabled: bool = True,
        metadata: dict | None = None,
    ) -> None:
        if not runtime_id or not provider or not model_id:
            raise ValueError("runtime_id, provider and model_id are required")
        if framework not in ("agno", "claude-cli"):
            raise ValueError(f"invalid framework: {framework!r}")
        conn = await self._ensure_connected()
        now = time.time()
        await conn.execute(
            "INSERT INTO models (runtime_id, provider, framework, model_id, display_name, "
            "tier_hint, notes, enabled, metadata_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(runtime_id) DO UPDATE SET "
            "provider = excluded.provider, framework = excluded.framework, "
            "model_id = excluded.model_id, display_name = excluded.display_name, "
            "tier_hint = excluded.tier_hint, notes = excluded.notes, "
            "enabled = excluded.enabled, metadata_json = excluded.metadata_json, "
            "updated_at = excluded.updated_at",
            (
                runtime_id,
                provider,
                framework,
                model_id,
                display_name,
                tier_hint,
                notes,
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

    async def delete_models_by_provider(self, provider: str) -> int:
        """Purge every model row owned by ``provider``. Returns the row count.

        Called on provider removal so the models table doesn't accumulate
        orphan entries that can no longer be dispatched (missing API key).
        """
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "DELETE FROM models WHERE provider = ?", (provider,),
        )
        await conn.commit()
        return cursor.rowcount or 0

    async def models_max_updated(self) -> float:
        conn = await self._ensure_connected()
        cursor = await conn.execute("SELECT MAX(updated_at) FROM models")
        row = await cursor.fetchone()
        return float(row[0] or 0.0) if row else 0.0

    # ── Providers (API keys) ──

    @staticmethod
    def _row_to_provider(row: aiosqlite.Row) -> dict[str, Any]:
        metadata = row["metadata_json"] or "{}"
        try:
            meta_parsed = json.loads(metadata) if isinstance(metadata, str) else {}
        except ValueError:
            meta_parsed = {}
        return {
            "name": row["name"],
            "api_key": row["api_key"],
            "base_url": row["base_url"],
            "enabled": bool(row["enabled"]),
            "metadata": meta_parsed,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def list_providers(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        conn = await self._ensure_connected()
        sql = "SELECT * FROM providers"
        params: tuple = ()
        if enabled_only:
            sql += " WHERE enabled = 1"
        sql += " ORDER BY name"
        cursor = await conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [self._row_to_provider(r) for r in rows]

    async def get_provider(self, name: str) -> dict[str, Any] | None:
        conn = await self._ensure_connected()
        cursor = await conn.execute("SELECT * FROM providers WHERE name = ?", (name,))
        row = await cursor.fetchone()
        return self._row_to_provider(row) if row else None

    async def upsert_provider(
        self,
        name: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        conn = await self._ensure_connected()
        now = time.time()
        await conn.execute(
            """
            INSERT INTO providers (name, api_key, base_url, enabled, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                api_key = excluded.api_key,
                base_url = excluded.base_url,
                enabled = excluded.enabled,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                name, api_key, base_url, 1 if enabled else 0,
                json.dumps(metadata or {}), now, now,
            ),
        )
        await conn.commit()

    async def set_provider_enabled(self, name: str, enabled: bool) -> None:
        conn = await self._ensure_connected()
        await conn.execute(
            "UPDATE providers SET enabled = ?, updated_at = ? WHERE name = ?",
            (1 if enabled else 0, time.time(), name),
        )
        await conn.commit()

    async def delete_provider(self, name: str) -> None:
        """Delete a provider row. Does NOT cascade to models — callers
        that want cascade should call ``delete_models_by_provider(name)``
        explicitly. Keeping the two steps separate so tests and tools
        can exercise one without the other."""
        conn = await self._ensure_connected()
        await conn.execute("DELETE FROM providers WHERE name = ?", (name,))
        await conn.commit()

    async def providers_max_updated(self) -> float:
        conn = await self._ensure_connected()
        cursor = await conn.execute("SELECT MAX(updated_at) FROM providers")
        row = await cursor.fetchone()
        return float(row[0] or 0.0) if row else 0.0

    async def registry_status(self) -> tuple[float, float, int, float]:
        """One-shot probe used by the gateway's per-message hot-reload loop.

        Returns ``(mcps_max_updated, models_max_updated, enabled_models_count,
        providers_max_updated)`` in a single round-trip so the dispatcher
        doesn't pay four SELECTs per incoming message.
        """
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT "
            "  COALESCE((SELECT MAX(updated_at) FROM mcps), 0), "
            "  COALESCE((SELECT MAX(updated_at) FROM models), 0), "
            "  COALESCE((SELECT COUNT(*) FROM models WHERE enabled = 1), 0), "
            "  COALESCE((SELECT MAX(updated_at) FROM providers), 0)"
        )
        row = await cursor.fetchone()
        if not row:
            return 0.0, 0.0, 0, 0.0
        return (
            float(row[0] or 0.0), float(row[1] or 0.0),
            int(row[2] or 0), float(row[3] or 0.0),
        )

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

    async def get_session_pin(self, session_id: str) -> str | None:
        """Return the pinned ``runtime_id`` for ``session_id``, or ``None``.

        When non-null, SmartRouter dispatches this session straight to
        ``runtime_id`` without consulting the classifier or the routing
        tiers. Pinned sessions ignore budget degradation too — an
        explicit user choice wins.
        """
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT runtime_id FROM session_bindings WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        return str(row[0]) if row and row[0] else None

    async def set_session_binding(
        self,
        session_id: str,
        provider: str,
        runtime_id: str | None = None,
    ) -> None:
        """Record that ``session_id`` is served by ``provider``.

        Optional ``runtime_id`` pins the session to a specific model.
        Used by SmartRouter after a first successful dispatch so
        subsequent turns are forced to the same side. Claude-cli
        *side* bindings land in ``sdk_sessions`` instead (via
        ``set_sdk_session``); this table tracks agno side + per-session
        explicit model pins for both sides.
        """
        conn = await self._ensure_connected()
        await conn.execute(
            "INSERT INTO session_bindings (session_id, provider, bound_at, runtime_id) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "provider = excluded.provider, bound_at = excluded.bound_at, "
            "runtime_id = excluded.runtime_id",
            (session_id, provider, time.time(), runtime_id),
        )
        await conn.commit()

    async def pin_session_model(self, session_id: str, runtime_id: str) -> None:
        """Pin ``session_id`` to a specific model ``runtime_id``.

        Framework lock: if the session has already been served by one
        framework (rows in ``sdk_sessions`` for claude-cli, or in
        ``session_bindings`` for agno), we refuse to pin it to a model
        from the OTHER framework. Conversation state would split
        across two stores and turns would lose context. Callers should
        ``/clear`` or spawn a fresh session_id if they actually want to
        switch frameworks.
        """
        if not session_id or not runtime_id:
            raise ValueError("session_id and runtime_id are required")
        target_framework = (
            "claude-cli" if runtime_id.startswith("claude-cli") else "agno"
        )
        existing = await self.get_session_binding(session_id)
        if existing and existing != target_framework:
            raise ValueError(
                f"session {session_id!r} is bound to framework={existing!r} "
                f"and cannot be pinned to a {target_framework!r} model — "
                "conversation history lives in the current framework's "
                "store. Use /clear (or a fresh session_id) first."
            )
        await self.set_session_binding(
            session_id, target_framework, runtime_id=runtime_id,
        )

    async def unpin_session_model(self, session_id: str) -> None:
        """Clear the per-session model pin, leaving the side-binding intact.

        The ``runtime_id`` column is set to NULL; SmartRouter resumes
        using the classifier/routing tiers for this session on the next
        turn. The ``provider`` side-binding is *not* touched — a
        session pinned to claude-cli stays on claude-cli even after
        unpinning the specific model.
        """
        conn = await self._ensure_connected()
        await conn.execute(
            "UPDATE session_bindings SET runtime_id = NULL, bound_at = ? "
            "WHERE session_id = ?",
            (time.time(), session_id),
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
