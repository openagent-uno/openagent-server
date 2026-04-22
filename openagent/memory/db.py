"""SQLite storage for scheduled tasks, usage logs, providers, and models."""

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
from openagent.models.catalog import SUPPORTED_FRAMEWORKS


VALID_MCP_KINDS = ("builtin", "custom", "default")
# Alias kept for the ``from openagent.memory.db import VALID_FRAMEWORKS``
# import sites already in the tree; both names point at the canonical
# tuple defined in :mod:`openagent.models.catalog`.
VALID_FRAMEWORKS = SUPPORTED_FRAMEWORKS

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

-- LLM providers. One row per (vendor, framework) pair.
--
-- OpenAgent vocabulary (v0.12+):
--   - **provider**  = a concrete credential + dispatch pair. The same vendor
--                     (``anthropic``) can appear as two rows — one with
--                     ``framework='agno'`` (direct API, needs ``api_key``) and
--                     one with ``framework='claude-cli'`` (local ``claude``
--                     subprocess, ``api_key`` MUST be NULL).
--   - **framework** = how OpenAgent dispatches calls for this provider row:
--                     ``agno`` (Agno SDK hits the vendor's REST API) or
--                     ``claude-cli`` (spawns the user's Pro/Max subscription
--                     via the local binary).
--
-- ``UNIQUE(name, framework)`` lets the UI/API/MCP address a row by its
-- (vendor, framework) pair; the surrogate ``id`` is what the ``models``
-- table joins to.
CREATE TABLE IF NOT EXISTS providers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    framework TEXT NOT NULL CHECK (framework IN ('agno','claude-cli')),
    api_key TEXT,
    base_url TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(name, framework)
);
CREATE INDEX IF NOT EXISTS idx_providers_enabled ON providers(enabled);
CREATE INDEX IF NOT EXISTS idx_providers_updated ON providers(updated_at);
CREATE INDEX IF NOT EXISTS idx_providers_name ON providers(name);

-- Configured LLM models. Each row is a bare vendor id plus a FK to the
-- provider row that owns it. Framework is inherited from the provider —
-- deleting a provider cascades to wipe its models (ON DELETE CASCADE).
--
-- ``model`` is the bare vendor id (e.g. ``gpt-4o-mini``, ``claude-opus-4-7``).
-- The canonical ``runtime_id`` used in logs / session pins / classifier
-- responses is DERIVED at read time from the provider row's (name,
-- framework) pair — no longer stored here.
--
-- ``tier_hint`` absorbs the old ``notes`` column: free-form classifier
-- guidance (``"vision, 200k context, best for code"``, ``"cheap and fast"``).
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id INTEGER NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    display_name TEXT,
    tier_hint TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    is_classifier INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(provider_id, model)
);
CREATE INDEX IF NOT EXISTS idx_models_provider ON models(provider_id);
CREATE INDEX IF NOT EXISTS idx_models_enabled ON models(enabled);
CREATE INDEX IF NOT EXISTS idx_models_updated ON models(updated_at);
-- idx_models_is_classifier is created in _apply_legacy_alters, after
-- the column is guaranteed to exist on legacy DBs (SCHEMA_SQL's
-- CREATE TABLE IF NOT EXISTS can't add columns to an existing table).

-- Generic string-valued state flags. Intended for process-wide
-- markers that need to survive restarts (none in active use — the
-- schema is kept for forward compat).
CREATE TABLE IF NOT EXISTS config_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

-- Per-session runtime binding. SmartRouter dispatches fresh sessions
-- to either the Agno stack (``framework='agno'``) or the Claude CLI
-- registry (``framework='claude-cli'``) based on the classifier; once a
-- session has been served by one side its conversation state lives
-- there so the router must respect that lock on subsequent turns.
--
-- ``runtime_id`` is a human-readable label (e.g.
-- ``claude-cli:anthropic:claude-opus-4-7``) derived from the provider
-- + model rows at pin time; it's not a FK so a later model delete
-- leaves a "stale pin" the router gracefully falls back from rather
-- than throwing an integrity error.
--
-- Claude-cli bindings are ALSO persisted in ``sdk_sessions`` because
-- that table carries the SDK-native UUID needed for ``--resume``. This
-- table covers agno sessions + per-session explicit model pins for
-- both sides, plus it serves as a fast single-table lookup for
-- SmartRouter.
CREATE TABLE IF NOT EXISTS session_bindings (
    session_id TEXT PRIMARY KEY,
    framework TEXT NOT NULL CHECK (framework IN ('agno','claude-cli')),
    runtime_id TEXT,
    bound_at REAL NOT NULL
);

-- Workflow graphs (n8n-style multi-block pipelines). The whole node/
-- edge graph lives inside ``graph_json`` so the AI can round-trip it
-- via a single tool call and React Flow can consume the same shape on
-- the UI. A workflow has no opinion on how it's triggered — any
-- workflow can be fired manually, by the AI, or on a schedule at any
-- time. The scheduling state (cron + next_run_at) is keyed per
-- trigger-schedule *node* in ``workflow_schedules`` below, so a single
-- workflow can carry multiple independent schedules.
--
-- Legacy columns ``trigger_kind`` / ``cron_expression`` / ``next_run_at``
-- shipped in v0.12.10; they are kept on the table for existing DBs
-- (SQLite can't cleanly drop NOT NULL columns pre-3.35) but are no
-- longer read or written by any new code. See ``_apply_legacy_alters``
-- for the migration that backfills ``workflow_schedules`` from the
-- first release's row-level cron.
CREATE TABLE IF NOT EXISTS workflow_tasks (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    description     TEXT,
    graph_json      TEXT NOT NULL DEFAULT '{"version":1,"nodes":[],"edges":[],"variables":{}}',
    trigger_kind    TEXT NOT NULL DEFAULT 'manual',  -- DEPRECATED, ignored
    cron_expression TEXT,                             -- DEPRECATED, ignored
    enabled         INTEGER NOT NULL DEFAULT 1,
    last_run_at     REAL,
    next_run_at     REAL,                              -- DEPRECATED, ignored
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wf_enabled  ON workflow_tasks(enabled);

-- One row per ``trigger-schedule`` block in any workflow's graph. The
-- scheduler polls ``WHERE enabled=1 AND next_run_at <= ?`` with the
-- ``next_run_at`` index so the scan stays O(scheduled) regardless of
-- how many workflows exist. Rows are kept in sync with the graph by
-- ``sync_workflow_schedules`` on every workflow write.
CREATE TABLE IF NOT EXISTS workflow_schedules (
    id              TEXT PRIMARY KEY,
    workflow_id     TEXT NOT NULL REFERENCES workflow_tasks(id) ON DELETE CASCADE,
    node_id         TEXT NOT NULL,
    cron_expression TEXT NOT NULL,
    next_run_at     REAL NOT NULL,
    last_run_at     REAL,
    enabled         INTEGER NOT NULL DEFAULT 1,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    UNIQUE(workflow_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_wfsched_next_run ON workflow_schedules(next_run_at);
CREATE INDEX IF NOT EXISTS idx_wfsched_enabled  ON workflow_schedules(enabled);
CREATE INDEX IF NOT EXISTS idx_wfsched_workflow ON workflow_schedules(workflow_id);

-- Per-execution history + append-only trace. ``trace_json`` is a
-- JSON array of per-block entries:
--   [{node_id, type, started_at, finished_at, status, input, output, error}]
-- Reads: RunHistoryDrawer in the UI, ``get_workflow_run`` MCP tool.
CREATE TABLE IF NOT EXISTS workflow_runs (
    id              TEXT PRIMARY KEY,
    workflow_id     TEXT NOT NULL REFERENCES workflow_tasks(id) ON DELETE CASCADE,
    trigger         TEXT NOT NULL,
    status          TEXT NOT NULL,
    started_at      REAL NOT NULL,
    finished_at     REAL,
    inputs_json     TEXT NOT NULL DEFAULT '{}',
    outputs_json    TEXT NOT NULL DEFAULT '{}',
    error           TEXT,
    trace_json      TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_wfruns_workflow ON workflow_runs(workflow_id);
CREATE INDEX IF NOT EXISTS idx_wfruns_started  ON workflow_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_wfruns_status   ON workflow_runs(status);

-- Cross-process execution queue. The ``workflow-manager`` MCP
-- subprocess cannot touch the live Agent, so ``run_workflow`` drops a
-- row here; the main-process ``Scheduler._check_and_run`` claims it
-- atomically (``claimed_at`` flipped from NULL to now) and drives the
-- ``WorkflowExecutor``. Mirrors the mcp-manager / scheduler pattern:
-- DB-backed hand-off, no in-process IPC.
CREATE TABLE IF NOT EXISTS workflow_run_requests (
    id            TEXT PRIMARY KEY,
    workflow_id   TEXT NOT NULL,
    inputs_json   TEXT NOT NULL DEFAULT '{}',
    trigger       TEXT NOT NULL,
    claimed_at    REAL,
    run_id        TEXT,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wfreq_unclaimed ON workflow_run_requests(claimed_at);
"""


class MemoryDB:
    """SQLite storage for OpenAgent's runtime state."""

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
        # Enable FK constraints per-connection. SQLite's default is OFF,
        # so without this the ON DELETE CASCADE on models.provider_id is
        # silently a no-op and deleting a provider orphans its models.
        await self._conn.execute("PRAGMA foreign_keys = ON")
        await self._conn.executescript(SCHEMA_SQL)
        await self._apply_legacy_alters()
        await self._conn.commit()

    async def _apply_legacy_alters(self) -> None:
        """Idempotent ALTERs for columns added after the schema was first shipped.

        ``CREATE TABLE IF NOT EXISTS`` won't add columns to an existing
        table, so each new column needs a PRAGMA-guarded ALTER here.
        Indexes on post-ship columns also live here — creating them in
        ``SCHEMA_SQL`` would fail on a legacy DB where the column
        doesn't exist yet (the CREATE INDEX runs before the ALTER).
        """
        assert self._conn is not None
        cursor = await self._conn.execute("PRAGMA table_info(models)")
        cols = {row[1] for row in await cursor.fetchall()}
        if "is_classifier" not in cols:
            await self._conn.execute(
                "ALTER TABLE models ADD COLUMN is_classifier "
                "INTEGER NOT NULL DEFAULT 0"
            )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_models_is_classifier "
            "ON models(is_classifier)"
        )

        # v0.12.10 → v0.12.11: per-block scheduling.
        # Rows in ``workflow_tasks.cron_expression`` were the single
        # row-level schedule. The new model carries schedules per
        # ``trigger-schedule`` block in ``workflow_schedules``. For each
        # legacy workflow with a row-level cron, ensure its graph has a
        # matching trigger-schedule block (inject one if missing) and
        # seed the ``workflow_schedules`` row — then clear the legacy
        # column so subsequent boots don't re-migrate.
        await self._migrate_workflow_schedules_from_legacy_columns()

    async def _migrate_workflow_schedules_from_legacy_columns(self) -> None:
        """One-time backfill from v0.12.10's row-level ``cron_expression``
        column into the per-block ``workflow_schedules`` table.
        Idempotent — runs every boot but only does work on rows that
        still carry a legacy cron.
        """
        assert self._conn is not None
        # Probe for the legacy column — absent on fresh installs that
        # started on v0.12.11+, present on upgrades.
        cursor = await self._conn.execute("PRAGMA table_info(workflow_tasks)")
        cols = {row[1] for row in await cursor.fetchall()}
        if "cron_expression" not in cols:
            return

        cursor = await self._conn.execute(
            "SELECT id, graph_json, cron_expression FROM workflow_tasks "
            "WHERE cron_expression IS NOT NULL AND cron_expression != ''"
        )
        rows = await cursor.fetchall()
        if not rows:
            return

        now = time.time()
        # Defer the cron parse to avoid pulling croniter into every boot;
        # only import on actual migration work.
        from openagent.memory.schedule import (
            next_run_for_expression,
            validate_schedule_expression,
        )

        for row in rows:
            wf_id = row[0]
            graph_json = row[1] or '{"version":1,"nodes":[],"edges":[],"variables":{}}'
            legacy_cron = row[2]
            try:
                graph = json.loads(graph_json)
            except (TypeError, ValueError):
                continue

            nodes = graph.setdefault("nodes", [])
            edges = graph.setdefault("edges", [])

            # Does the graph already carry a trigger-schedule block?
            sched_node = next(
                (n for n in nodes if n.get("type") == "trigger-schedule"),
                None,
            )
            if sched_node is None:
                # Inject one so the legacy cron survives the migration.
                used_ids = {n.get("id") for n in nodes}
                i = len(nodes) + 1
                new_id = f"n{i}"
                while new_id in used_ids:
                    i += 1
                    new_id = f"n{i}"
                sched_node = {
                    "id": new_id,
                    "type": "trigger-schedule",
                    "label": "Scheduled",
                    "position": {"x": 120.0, "y": 120.0},
                    "config": {"cron_expression": legacy_cron},
                }
                nodes.insert(0, sched_node)
                await self._conn.execute(
                    "UPDATE workflow_tasks SET graph_json = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(graph), now, wf_id),
                )
            else:
                cfg = sched_node.setdefault("config", {})
                if not cfg.get("cron_expression"):
                    cfg["cron_expression"] = legacy_cron
                    await self._conn.execute(
                        "UPDATE workflow_tasks SET graph_json = ?, updated_at = ? WHERE id = ?",
                        (json.dumps(graph), now, wf_id),
                    )

            # Create a workflow_schedules row if one doesn't exist yet.
            try:
                validate_schedule_expression(sched_node["config"]["cron_expression"])
                nxt = next_run_for_expression(sched_node["config"]["cron_expression"])
            except ValueError:
                continue  # drop invalid legacy crons silently

            await self._conn.execute(
                "INSERT OR IGNORE INTO workflow_schedules "
                "(id, workflow_id, node_id, cron_expression, next_run_at, "
                " enabled, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                (
                    str(uuid.uuid4()),
                    wf_id,
                    sched_node["id"],
                    sched_node["config"]["cron_expression"],
                    nxt,
                    now,
                    now,
                ),
            )
            # Clear the legacy column so the next boot is a no-op.
            await self._conn.execute(
                "UPDATE workflow_tasks SET cron_expression = NULL WHERE id = ?",
                (wf_id,),
            )

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

    # ── Workflow Tasks ──

    @staticmethod
    def _row_to_workflow(row: aiosqlite.Row) -> dict:
        """Hydrate a workflow row. ``graph_json`` is parsed into a
        ``{"version", "nodes", "edges", "variables"}`` dict; ``enabled``
        becomes a real bool. Legacy ``trigger_kind`` / ``cron_expression`` /
        ``next_run_at`` columns (v0.12.10) are stripped — callers read
        schedule state from ``workflow_schedules`` via
        ``list_schedules(workflow_id=...)``.
        """
        d = dict(row)
        raw = d.pop("graph_json", None) or '{"version":1,"nodes":[],"edges":[],"variables":{}}'
        try:
            d["graph"] = json.loads(raw)
        except (TypeError, ValueError):
            d["graph"] = {"version": 1, "nodes": [], "edges": [], "variables": {}}
        d["enabled"] = bool(d.get("enabled"))
        # Drop deprecated row-level fields — they're still stored on the
        # table for backwards-compatibility but callers should not read
        # them. ``_migrate_workflow_schedules_from_legacy_columns``
        # clears them on first boot after the upgrade.
        for deprecated in ("trigger_kind", "cron_expression", "next_run_at"):
            d.pop(deprecated, None)
        return d

    async def list_workflows(
        self,
        *,
        enabled_only: bool = False,
    ) -> list[dict]:
        conn = await self._ensure_connected()
        where = "WHERE enabled = 1" if enabled_only else ""
        cursor = await conn.execute(
            f"SELECT * FROM workflow_tasks {where} ORDER BY updated_at DESC"
        )
        rows = await cursor.fetchall()
        return [self._row_to_workflow(r) for r in rows]

    async def get_workflow(self, id_or_name: str) -> dict | None:
        """Look up a workflow by full id, 8-char id prefix, or unique name."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM workflow_tasks WHERE id = ? OR name = ?",
            (id_or_name, id_or_name),
        )
        row = await cursor.fetchone()
        if row is None and len(id_or_name) >= 4:
            cursor = await conn.execute(
                "SELECT * FROM workflow_tasks WHERE id LIKE ? LIMIT 2",
                (f"{id_or_name}%",),
            )
            matches = await cursor.fetchall()
            if len(matches) == 1:
                row = matches[0]
        return self._row_to_workflow(row) if row else None

    async def add_workflow(
        self,
        *,
        name: str,
        description: str | None = None,
        graph: dict | None = None,
        enabled: bool = True,
    ) -> str:
        if not name or not name.strip():
            raise ValueError("name is required")
        graph_payload = graph or {"version": 1, "nodes": [], "edges": [], "variables": {}}
        conn = await self._ensure_connected()
        workflow_id = str(uuid.uuid4())
        now = time.time()
        await conn.execute(
            "INSERT INTO workflow_tasks "
            "(id, name, description, graph_json, enabled, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                workflow_id,
                name.strip(),
                description,
                json.dumps(graph_payload),
                1 if enabled else 0,
                now,
                now,
            ),
        )
        await conn.commit()
        return workflow_id

    async def update_workflow(self, workflow_id: str, **kwargs: Any) -> None:
        """Partial update. ``graph`` (dict) is serialized to ``graph_json``
        on the way in. Schedule state is kept in sync via
        ``workflow_schedules`` — callers should invoke
        ``sync_workflow_schedules`` after any graph write.
        """
        allowed_direct = {"name", "description", "enabled", "last_run_at"}
        updates: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k == "graph" and v is not None:
                updates["graph_json"] = json.dumps(v)
            elif k in allowed_direct:
                updates[k] = (1 if v else 0) if k == "enabled" and isinstance(v, bool) else v
        if not updates:
            return
        updates["updated_at"] = time.time()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn = await self._ensure_connected()
        await conn.execute(
            f"UPDATE workflow_tasks SET {set_clause} WHERE id = ?",
            list(updates.values()) + [workflow_id],
        )
        await conn.commit()

    async def delete_workflow(self, workflow_id: str) -> None:
        conn = await self._ensure_connected()
        await conn.execute("DELETE FROM workflow_tasks WHERE id = ?", (workflow_id,))
        await conn.commit()

    # ── Workflow Schedules (per trigger-schedule block) ──

    @staticmethod
    def _row_to_schedule(row: aiosqlite.Row) -> dict:
        d = dict(row)
        d["enabled"] = bool(d.get("enabled"))
        return d

    async def list_schedules(
        self,
        *,
        workflow_id: str | None = None,
        enabled_only: bool = False,
    ) -> list[dict]:
        conn = await self._ensure_connected()
        clauses: list[str] = []
        params: list[Any] = []
        if workflow_id is not None:
            clauses.append("workflow_id = ?")
            params.append(workflow_id)
        if enabled_only:
            clauses.append("enabled = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = await conn.execute(
            f"SELECT * FROM workflow_schedules {where} "
            "ORDER BY next_run_at ASC",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_schedule(r) for r in rows]

    async def get_due_schedules(self, now: float) -> list[dict]:
        """Schedules whose next_run_at is <= now. The scheduler loop
        consumes this on every tick to drive per-block triggering."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT s.* FROM workflow_schedules s "
            "JOIN workflow_tasks w ON w.id = s.workflow_id "
            "WHERE s.enabled = 1 AND w.enabled = 1 AND s.next_run_at <= ? "
            "ORDER BY s.next_run_at ASC",
            (now,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_schedule(r) for r in rows]

    async def upsert_schedule(
        self,
        *,
        workflow_id: str,
        node_id: str,
        cron_expression: str,
        next_run_at: float,
        enabled: bool = True,
    ) -> str:
        """Insert or update the schedule row for a given
        (workflow_id, node_id). Returns the row id."""
        conn = await self._ensure_connected()
        now = time.time()
        cursor = await conn.execute(
            "SELECT id, cron_expression, next_run_at FROM workflow_schedules "
            "WHERE workflow_id = ? AND node_id = ?",
            (workflow_id, node_id),
        )
        existing = await cursor.fetchone()
        if existing is not None:
            # Preserve next_run_at when only metadata changed and cron
            # is identical — avoids rolling the scheduler forward on
            # every graph save.
            keep_next = existing["cron_expression"] == cron_expression
            await conn.execute(
                "UPDATE workflow_schedules SET cron_expression = ?, "
                "next_run_at = ?, enabled = ?, updated_at = ? "
                "WHERE id = ?",
                (
                    cron_expression,
                    existing["next_run_at"] if keep_next else next_run_at,
                    1 if enabled else 0,
                    now,
                    existing["id"],
                ),
            )
            await conn.commit()
            return existing["id"]
        sid = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO workflow_schedules "
            "(id, workflow_id, node_id, cron_expression, next_run_at, "
            " enabled, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                sid,
                workflow_id,
                node_id,
                cron_expression,
                next_run_at,
                1 if enabled else 0,
                now,
                now,
            ),
        )
        await conn.commit()
        return sid

    async def update_schedule(self, schedule_id: str, **kwargs: Any) -> None:
        allowed = {"cron_expression", "next_run_at", "last_run_at", "enabled"}
        updates: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            updates[k] = (1 if v else 0) if k == "enabled" and isinstance(v, bool) else v
        if not updates:
            return
        updates["updated_at"] = time.time()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn = await self._ensure_connected()
        await conn.execute(
            f"UPDATE workflow_schedules SET {set_clause} WHERE id = ?",
            list(updates.values()) + [schedule_id],
        )
        await conn.commit()

    async def delete_schedule(self, schedule_id: str) -> None:
        conn = await self._ensure_connected()
        await conn.execute(
            "DELETE FROM workflow_schedules WHERE id = ?", (schedule_id,),
        )
        await conn.commit()

    async def delete_schedules_not_in(
        self,
        workflow_id: str,
        keep_node_ids: list[str],
    ) -> int:
        """Prune schedules whose block no longer exists in the graph.
        Returns the number of rows removed. Called by
        ``sync_workflow_schedules`` after processing graph blocks."""
        conn = await self._ensure_connected()
        if not keep_node_ids:
            cursor = await conn.execute(
                "DELETE FROM workflow_schedules WHERE workflow_id = ?",
                (workflow_id,),
            )
        else:
            placeholders = ",".join("?" for _ in keep_node_ids)
            cursor = await conn.execute(
                f"DELETE FROM workflow_schedules WHERE workflow_id = ? "
                f"AND node_id NOT IN ({placeholders})",
                [workflow_id, *keep_node_ids],
            )
        await conn.commit()
        return cursor.rowcount or 0

    # ── Workflow Runs (execution history) ──

    @staticmethod
    def _row_to_workflow_run(row: aiosqlite.Row) -> dict:
        d = dict(row)
        for col in ("inputs_json", "outputs_json", "trace_json"):
            raw = d.pop(col, None) or ("[]" if col == "trace_json" else "{}")
            key = col[:-5]
            try:
                d[key] = json.loads(raw)
            except (TypeError, ValueError):
                d[key] = [] if key == "trace" else {}
        return d

    async def add_workflow_run(
        self,
        *,
        workflow_id: str,
        trigger: str,
        inputs: dict | None = None,
        run_id: str | None = None,
    ) -> str:
        conn = await self._ensure_connected()
        rid = run_id or str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO workflow_runs "
            "(id, workflow_id, trigger, status, started_at, inputs_json) "
            "VALUES (?, ?, ?, 'running', ?, ?)",
            (rid, workflow_id, trigger, time.time(), json.dumps(inputs or {})),
        )
        await conn.commit()
        return rid

    async def update_workflow_run(self, run_id: str, **kwargs: Any) -> None:
        """Partial update. ``outputs`` / ``trace`` (Python objects) are
        serialized to their ``_json`` columns."""
        allowed_direct = {"status", "finished_at", "error"}
        updates: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k == "outputs" and v is not None:
                updates["outputs_json"] = json.dumps(v)
            elif k == "trace" and v is not None:
                updates["trace_json"] = json.dumps(v)
            elif k in allowed_direct:
                updates[k] = v
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn = await self._ensure_connected()
        await conn.execute(
            f"UPDATE workflow_runs SET {set_clause} WHERE id = ?",
            list(updates.values()) + [run_id],
        )
        await conn.commit()

    async def get_workflow_run(self, run_id: str) -> dict | None:
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM workflow_runs WHERE id = ?", (run_id,),
        )
        row = await cursor.fetchone()
        return self._row_to_workflow_run(row) if row else None

    async def list_workflow_runs(
        self,
        workflow_id: str,
        *,
        limit: int = 20,
        status: str | None = None,
    ) -> list[dict]:
        conn = await self._ensure_connected()
        clauses = ["workflow_id = ?"]
        params: list[Any] = [workflow_id]
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        params.append(int(limit))
        cursor = await conn.execute(
            f"SELECT * FROM workflow_runs WHERE {' AND '.join(clauses)} "
            "ORDER BY started_at DESC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_workflow_run(r) for r in rows]

    async def prune_workflow_runs(
        self,
        workflow_id: str,
        *,
        keep_last: int = 50,
    ) -> int:
        """Delete all runs older than the most recent ``keep_last`` for a
        given workflow. Returns the number of rows removed."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "DELETE FROM workflow_runs WHERE id IN ("
            "  SELECT id FROM workflow_runs WHERE workflow_id = ? "
            "  ORDER BY started_at DESC LIMIT -1 OFFSET ?"
            ")",
            (workflow_id, int(keep_last)),
        )
        await conn.commit()
        return cursor.rowcount or 0

    async def workflow_run_stats(
        self,
        workflow_id: str,
        *,
        sparkline_count: int = 10,
    ) -> dict[str, Any]:
        """Aggregate run statistics for a workflow.

        Powers the workflow editor's RunHistoryDrawer header + the
        list-screen row badges. Returns:
          - total_runs, success_count, failed_count, cancelled_count
          - running_count (for the "something is currently in flight" pill)
          - success_rate (float 0–1, 0 when no runs yet)
          - avg_duration_s (mean of finished_at - started_at for
            success+failed runs; None when no completed runs exist)
          - last: [{id, status, started_at, finished_at, duration_s}]
            newest-first, capped at ``sparkline_count``
        """
        conn = await self._ensure_connected()
        agg_cursor = await conn.execute(
            """
            SELECT status, COUNT(*) AS n,
                   AVG(
                     CASE
                       WHEN finished_at IS NOT NULL THEN finished_at - started_at
                       ELSE NULL
                     END
                   ) AS avg_dur
            FROM workflow_runs
            WHERE workflow_id = ?
            GROUP BY status
            """,
            (workflow_id,),
        )
        agg_rows = await agg_cursor.fetchall()
        stats = {
            "total_runs": 0,
            "success_count": 0,
            "failed_count": 0,
            "cancelled_count": 0,
            "running_count": 0,
            "success_rate": 0.0,
            "avg_duration_s": None,
        }
        weighted_sum = 0.0
        weighted_n = 0
        for row in agg_rows:
            r = dict(row)
            n = int(r.get("n") or 0)
            stats["total_runs"] += n
            key = f"{r['status']}_count"
            if key in stats:
                stats[key] = n
            avg = r.get("avg_dur")
            if avg is not None and r["status"] in ("success", "failed"):
                weighted_sum += float(avg) * n
                weighted_n += n
        terminal = stats["success_count"] + stats["failed_count"]
        if terminal:
            stats["success_rate"] = stats["success_count"] / terminal
        if weighted_n:
            stats["avg_duration_s"] = weighted_sum / weighted_n

        last_cursor = await conn.execute(
            """
            SELECT id, status, started_at, finished_at
            FROM workflow_runs
            WHERE workflow_id = ?
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (workflow_id, int(sparkline_count)),
        )
        last_rows = await last_cursor.fetchall()
        last = []
        for row in last_rows:
            r = dict(row)
            duration = (
                r["finished_at"] - r["started_at"]
                if r.get("finished_at") and r.get("started_at") is not None
                else None
            )
            r["duration_s"] = duration
            last.append(r)
        stats["last"] = last
        return stats

    # ── Workflow run request queue (MCP ↔ main-process hand-off) ──

    async def enqueue_workflow_run_request(
        self,
        *,
        workflow_id: str,
        trigger: str,
        inputs: dict | None = None,
    ) -> str:
        conn = await self._ensure_connected()
        req_id = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO workflow_run_requests "
            "(id, workflow_id, inputs_json, trigger, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (req_id, workflow_id, json.dumps(inputs or {}), trigger, time.time()),
        )
        await conn.commit()
        return req_id

    async def claim_pending_workflow_requests(self, *, limit: int = 5) -> list[dict]:
        """Atomically claim up to ``limit`` unclaimed requests. Each
        returned row has ``claimed_at`` set so concurrent scheduler
        ticks (or stray retries) won't run the same request twice."""
        conn = await self._ensure_connected()
        now = time.time()
        # Select unclaimed ids first, then UPDATE in the same transaction
        # (SQLite doesn't support ``UPDATE ... RETURNING`` on all versions
        # we target). Wrap both statements so a concurrent claimer can't
        # race past the SELECT.
        await conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = await conn.execute(
                "SELECT * FROM workflow_run_requests "
                "WHERE claimed_at IS NULL "
                "ORDER BY created_at ASC LIMIT ?",
                (int(limit),),
            )
            rows = await cursor.fetchall()
            if not rows:
                await conn.execute("COMMIT")
                return []
            ids = [row["id"] for row in rows]
            placeholders = ",".join("?" for _ in ids)
            await conn.execute(
                f"UPDATE workflow_run_requests SET claimed_at = ? "
                f"WHERE id IN ({placeholders})",
                [now, *ids],
            )
            await conn.execute("COMMIT")
        except Exception:
            await conn.execute("ROLLBACK")
            raise
        claimed = []
        for row in rows:
            d = dict(row)
            raw = d.pop("inputs_json", "{}") or "{}"
            try:
                d["inputs"] = json.loads(raw)
            except (TypeError, ValueError):
                d["inputs"] = {}
            d["claimed_at"] = now
            claimed.append(d)
        return claimed

    async def set_workflow_request_run_id(self, request_id: str, run_id: str) -> None:
        """Link a claimed request back to the ``workflow_runs`` row it
        spawned so the MCP tool's ``wait=True`` poller can find the run."""
        conn = await self._ensure_connected()
        await conn.execute(
            "UPDATE workflow_run_requests SET run_id = ? WHERE id = ?",
            (run_id, request_id),
        )
        await conn.commit()

    async def get_workflow_run_request(self, request_id: str) -> dict | None:
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM workflow_run_requests WHERE id = ?", (request_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        d = dict(row)
        raw = d.pop("inputs_json", "{}") or "{}"
        try:
            d["inputs"] = json.loads(raw)
        except (TypeError, ValueError):
            d["inputs"] = {}
        return d

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

    # ── Providers (v0.12: one row per (name, framework) pair) ──

    @staticmethod
    def _row_to_provider(row: aiosqlite.Row) -> dict[str, Any]:
        metadata = row["metadata_json"] or "{}"
        try:
            meta_parsed = json.loads(metadata) if isinstance(metadata, str) else {}
        except ValueError:
            meta_parsed = {}
        return {
            "id": row["id"],
            "name": row["name"],
            "framework": row["framework"],
            "api_key": row["api_key"],
            "base_url": row["base_url"],
            "enabled": bool(row["enabled"]),
            "metadata": meta_parsed,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def list_providers(
        self,
        *,
        enabled_only: bool = False,
        framework: str | None = None,
    ) -> list[dict[str, Any]]:
        conn = await self._ensure_connected()
        clauses: list[str] = []
        params: list[Any] = []
        if enabled_only:
            clauses.append("enabled = 1")
        if framework:
            clauses.append("framework = ?")
            params.append(framework)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = await conn.execute(
            f"SELECT * FROM providers {where} ORDER BY name ASC, framework ASC",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_provider(r) for r in rows]

    async def get_provider(self, provider_id: int) -> dict[str, Any] | None:
        """Fetch one provider row by its surrogate id."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM providers WHERE id = ?", (int(provider_id),),
        )
        row = await cursor.fetchone()
        return self._row_to_provider(row) if row else None

    async def get_provider_by_name(
        self, name: str, framework: str,
    ) -> dict[str, Any] | None:
        """Fetch the provider row for a (name, framework) pair."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM providers WHERE name = ? AND framework = ?",
            (name, framework),
        )
        row = await cursor.fetchone()
        return self._row_to_provider(row) if row else None

    async def upsert_provider(
        self,
        *,
        name: str,
        framework: str,
        api_key: str | None = None,
        base_url: str | None = None,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Upsert a provider row. Returns the provider's surrogate ``id``.

        ``framework='claude-cli'`` providers MUST carry ``api_key=None``
        (the subscription path authenticates through ``~/.claude/``; any
        stored value would poison the subprocess). ``framework='agno'``
        providers can be created with ``api_key=None`` (disabled-until-
        configured state) but dispatch will fail until a key is set.
        """
        if not name or not name.strip():
            raise ValueError("name is required")
        if framework not in VALID_FRAMEWORKS:
            raise ValueError(
                f"invalid framework {framework!r}; expected one of {VALID_FRAMEWORKS}"
            )
        if framework == "claude-cli" and api_key:
            raise ValueError(
                "claude-cli providers must not carry an api_key — the "
                "local `claude` binary authenticates via the Pro/Max "
                "subscription stored under ~/.claude/."
            )
        if framework == "claude-cli" and name.strip().lower() != "anthropic":
            raise ValueError(
                "claude-cli framework is only supported for the "
                "'anthropic' provider — the local `claude` binary "
                "dispatches Anthropic models via the Pro/Max subscription."
            )
        now = time.time()
        conn = await self._ensure_connected()
        await conn.execute(
            """
            INSERT INTO providers (name, framework, api_key, base_url, enabled, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name, framework) DO UPDATE SET
                api_key = excluded.api_key,
                base_url = excluded.base_url,
                enabled = excluded.enabled,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                name.strip(),
                framework,
                (api_key or None),
                (base_url or None),
                1 if enabled else 0,
                json.dumps(metadata or {}),
                now,
                now,
            ),
        )
        await conn.commit()
        # Fetch the id (stable across upserts on conflict).
        cursor = await conn.execute(
            "SELECT id FROM providers WHERE name = ? AND framework = ?",
            (name.strip(), framework),
        )
        row = await cursor.fetchone()
        if not row:
            raise RuntimeError("upsert_provider: row not found after insert")
        return int(row[0])

    async def set_provider_enabled(self, provider_id: int, enabled: bool) -> None:
        conn = await self._ensure_connected()
        await conn.execute(
            "UPDATE providers SET enabled = ?, updated_at = ? WHERE id = ?",
            (1 if enabled else 0, time.time(), int(provider_id)),
        )
        await conn.commit()

    async def delete_provider(self, provider_id: int) -> None:
        """Delete a provider row. FK cascade wipes its models."""
        conn = await self._ensure_connected()
        await conn.execute(
            "DELETE FROM providers WHERE id = ?", (int(provider_id),),
        )
        await conn.commit()

    async def providers_max_updated(self) -> float:
        conn = await self._ensure_connected()
        cursor = await conn.execute("SELECT MAX(updated_at) FROM providers")
        row = await cursor.fetchone()
        return float(row[0] or 0.0) if row else 0.0

    # ── Models (v0.12: provider_id FK, no runtime_id column) ──

    @staticmethod
    def _row_to_model(row: aiosqlite.Row) -> dict:
        d = dict(row)
        raw = d.pop("metadata_json", "{}") or "{}"
        try:
            d["metadata"] = json.loads(raw)
        except (TypeError, ValueError):
            d["metadata"] = {}
        d["enabled"] = bool(d.get("enabled"))
        d["is_classifier"] = bool(d.get("is_classifier"))
        return d

    async def list_models(
        self,
        *,
        provider_id: int | None = None,
        enabled_only: bool = False,
    ) -> list[dict]:
        conn = await self._ensure_connected()
        clauses: list[str] = []
        params: list[Any] = []
        if provider_id is not None:
            clauses.append("provider_id = ?")
            params.append(int(provider_id))
        if enabled_only:
            clauses.append("enabled = 1")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = await conn.execute(
            f"SELECT * FROM models {where} ORDER BY provider_id ASC, model ASC",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_model(r) for r in rows]

    # Common projection for the model-joined-with-provider view. Kept
    # as a constant so :meth:`list_models_enriched`,
    # :meth:`get_model_enriched`, and :meth:`get_model_by_runtime_id`
    # return identical dict shapes.
    _ENRICHED_MODEL_SELECT = """
        SELECT m.id AS id, m.provider_id AS provider_id, m.model AS model,
               m.display_name AS display_name, m.tier_hint AS tier_hint,
               m.enabled AS enabled, m.is_classifier AS is_classifier,
               m.metadata_json AS metadata_json,
               m.created_at AS created_at, m.updated_at AS updated_at,
               p.name AS provider_name, p.framework AS framework,
               p.api_key AS api_key, p.base_url AS base_url,
               p.enabled AS provider_enabled
        FROM models m
        JOIN providers p ON p.id = m.provider_id
    """

    @staticmethod
    def _shape_enriched(row: aiosqlite.Row) -> dict:
        from openagent.models.catalog import build_runtime_model_id

        d = dict(row)
        meta_raw = d.pop("metadata_json", "{}") or "{}"
        try:
            d["metadata"] = json.loads(meta_raw)
        except (TypeError, ValueError):
            d["metadata"] = {}
        d["enabled"] = bool(d["enabled"])
        d["is_classifier"] = bool(d.get("is_classifier"))
        d["provider_enabled"] = bool(d["provider_enabled"])
        d["runtime_id"] = build_runtime_model_id(
            d["provider_name"], d["model"], d["framework"],
        )
        return d

    async def list_models_enriched(
        self,
        *,
        enabled_only: bool = False,
        framework: str | None = None,
        provider_name: str | None = None,
        provider_id: int | None = None,
    ) -> list[dict]:
        """Return each model joined with its provider row.

        Each row carries ``{id, provider_id, model, display_name, tier_hint,
        enabled, metadata, created_at, updated_at, provider_name, framework,
        api_key, base_url, provider_enabled, runtime_id}`` — ``runtime_id``
        is derived on the fly via :func:`openagent.models.catalog.build_runtime_model_id`.
        This is the shape consumed by ``iter_configured_models`` and the REST
        ``/api/models`` list endpoint.
        """
        conn = await self._ensure_connected()
        clauses: list[str] = []
        params: list[Any] = []
        if enabled_only:
            clauses.append("m.enabled = 1")
            clauses.append("p.enabled = 1")
        if framework:
            clauses.append("p.framework = ?")
            params.append(framework)
        if provider_name:
            clauses.append("p.name = ?")
            params.append(provider_name)
        if provider_id is not None:
            clauses.append("m.provider_id = ?")
            params.append(int(provider_id))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = await conn.execute(
            f"{self._ENRICHED_MODEL_SELECT} {where} "
            "ORDER BY p.name ASC, p.framework ASC, m.model ASC",
            params,
        )
        rows = await cursor.fetchall()
        return [self._shape_enriched(r) for r in rows]

    async def get_model_enriched(self, model_id: int) -> dict | None:
        """Fetch a single enriched model row by its surrogate id.

        Same shape as :meth:`list_models_enriched` entries. Used by the
        REST read / create / update / toggle handlers to avoid
        scanning the full catalog for one row.
        """
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            f"{self._ENRICHED_MODEL_SELECT} WHERE m.id = ?",
            (int(model_id),),
        )
        row = await cursor.fetchone()
        return self._shape_enriched(row) if row else None

    async def materialise_providers_config(
        self, *, enabled_only: bool = False,
    ) -> list[dict]:
        """Build the AgnoProvider-consumable providers_config from the DB.

        Produces the flat list shape SmartRouter / AgnoProvider consume:
        one entry per (name, framework) pair, each carrying its nested
        ``models`` list. Used by :meth:`Agent._hydrate_providers_from_db`
        (``enabled_only=True``) and by the smoke-test endpoints that
        want every row regardless of enabled state.

        Single LEFT JOIN keeps this to one SQLite round-trip. A provider
        with no models still shows up (important for the UI's "empty
        provider" state).
        """
        conn = await self._ensure_connected()
        clauses: list[str] = []
        if enabled_only:
            # Model-side filter must go in the JOIN predicate, not WHERE,
            # or providers with zero enabled models would disappear.
            join_filter = " AND m.enabled = 1"
            clauses.append("p.enabled = 1")
        else:
            join_filter = ""
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = await conn.execute(
            f"""
            SELECT p.id AS p_id, p.name AS p_name, p.framework AS p_framework,
                   p.api_key AS p_api_key, p.base_url AS p_base_url,
                   p.enabled AS p_enabled, p.metadata_json AS p_metadata_json,
                   p.created_at AS p_created_at, p.updated_at AS p_updated_at,
                   m.id AS m_id, m.model AS m_model, m.display_name AS m_display_name,
                   m.tier_hint AS m_tier_hint, m.enabled AS m_enabled,
                   m.is_classifier AS m_is_classifier
            FROM providers p
            LEFT JOIN models m ON p.id = m.provider_id{join_filter}
            {where}
            ORDER BY p.name ASC, p.framework ASC, m.model ASC
            """
        )
        rows = await cursor.fetchall()
        by_id: dict[int, dict[str, Any]] = {}
        for r in rows:
            pid = int(r["p_id"])
            bucket = by_id.get(pid)
            if bucket is None:
                try:
                    metadata = json.loads(r["p_metadata_json"] or "{}")
                except (TypeError, ValueError):
                    metadata = {}
                bucket = {
                    "id": pid,
                    "name": r["p_name"],
                    "framework": r["p_framework"],
                    "api_key": r["p_api_key"],
                    "base_url": r["p_base_url"],
                    "enabled": bool(r["p_enabled"]),
                    "metadata": metadata,
                    "created_at": r["p_created_at"],
                    "updated_at": r["p_updated_at"],
                    "models": [],
                }
                by_id[pid] = bucket
            if r["m_id"] is not None:
                bucket["models"].append({
                    "id": int(r["m_id"]),
                    "model": r["m_model"],
                    "display_name": r["m_display_name"],
                    "tier_hint": r["m_tier_hint"],
                    "enabled": bool(r["m_enabled"]),
                    "is_classifier": bool(r["m_is_classifier"]),
                })
        return list(by_id.values())

    async def get_model(self, model_id: int) -> dict | None:
        """Fetch one model row by its surrogate id."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM models WHERE id = ?", (int(model_id),),
        )
        row = await cursor.fetchone()
        return self._row_to_model(row) if row else None

    async def get_model_by_ref(
        self, provider_id: int, model: str,
    ) -> dict | None:
        """Fetch a model row by its (provider_id, bare model) pair."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM models WHERE provider_id = ? AND model = ?",
            (int(provider_id), model),
        )
        row = await cursor.fetchone()
        return self._row_to_model(row) if row else None

    async def get_model_by_runtime_id(self, runtime_id: str) -> dict | None:
        """Fetch an enriched model row via a human-readable ``runtime_id``.

        Used by session-pin + REST/MCP paths where the caller still speaks
        the composite string (``openai:gpt-4o-mini``,
        ``claude-cli:anthropic:claude-opus-4-7``). Returns the same shape
        as :meth:`list_models_enriched`, or ``None`` when no matching
        (provider_name, framework, model) row exists.
        """
        from openagent.models.catalog import framework_of, split_runtime_id

        if not runtime_id:
            return None
        framework = framework_of(runtime_id)
        provider_name, model = split_runtime_id(runtime_id)
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            f"{self._ENRICHED_MODEL_SELECT} "
            "WHERE p.name = ? AND p.framework = ? AND m.model = ?",
            (provider_name, framework, model),
        )
        row = await cursor.fetchone()
        return self._shape_enriched(row) if row else None

    async def upsert_model(
        self,
        *,
        provider_id: int,
        model: str,
        display_name: str | None = None,
        tier_hint: str | None = None,
        enabled: bool = True,
        is_classifier: bool = False,
        metadata: dict | None = None,
    ) -> int:
        """Insert or update a model row. Returns the model's surrogate id."""
        if not provider_id:
            raise ValueError("provider_id is required")
        if not model or not str(model).strip():
            raise ValueError("model is required")
        conn = await self._ensure_connected()
        # FK integrity: make sure the parent provider exists before we
        # try the insert so callers get a clear error instead of the
        # generic "FOREIGN KEY constraint failed".
        prov_row = await (
            await conn.execute(
                "SELECT 1 FROM providers WHERE id = ?", (int(provider_id),),
            )
        ).fetchone()
        if prov_row is None:
            raise ValueError(f"Provider id={provider_id!r} does not exist")
        now = time.time()
        await conn.execute(
            """
            INSERT INTO models (provider_id, model, display_name, tier_hint,
                                enabled, is_classifier, metadata_json,
                                created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider_id, model) DO UPDATE SET
                display_name = excluded.display_name,
                tier_hint = excluded.tier_hint,
                enabled = excluded.enabled,
                is_classifier = excluded.is_classifier,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            """,
            (
                int(provider_id),
                str(model).strip(),
                display_name,
                tier_hint,
                1 if enabled else 0,
                1 if is_classifier else 0,
                json.dumps(dict(metadata or {})),
                now,
                now,
            ),
        )
        await conn.commit()
        cursor = await conn.execute(
            "SELECT id FROM models WHERE provider_id = ? AND model = ?",
            (int(provider_id), str(model).strip()),
        )
        row = await cursor.fetchone()
        if not row:
            raise RuntimeError("upsert_model: row not found after insert")
        return int(row[0])

    async def set_model_enabled(self, model_id: int, enabled: bool) -> None:
        conn = await self._ensure_connected()
        await conn.execute(
            "UPDATE models SET enabled = ?, updated_at = ? WHERE id = ?",
            (1 if enabled else 0, time.time(), int(model_id)),
        )
        await conn.commit()

    async def set_model_is_classifier(self, model_id: int, flag: bool) -> None:
        """Toggle the classifier flag on ``model_id``.

        Multiple rows are allowed to carry the flag simultaneously —
        this is a narrow UPDATE that only touches ``model_id``. The
        SmartRouter resolver picks the first flagged row it sees
        (deterministic catalog order), so having several flagged rows
        is effectively a pool of "eligible classifiers" where the
        first one wins each turn.
        """
        conn = await self._ensure_connected()
        await conn.execute(
            "UPDATE models SET is_classifier = ?, updated_at = ? WHERE id = ?",
            (1 if flag else 0, time.time(), int(model_id)),
        )
        await conn.commit()

    async def delete_model(self, model_id: int) -> None:
        conn = await self._ensure_connected()
        await conn.execute("DELETE FROM models WHERE id = ?", (int(model_id),))
        await conn.commit()

    async def models_max_updated(self) -> float:
        conn = await self._ensure_connected()
        cursor = await conn.execute("SELECT MAX(updated_at) FROM models")
        row = await cursor.fetchone()
        return float(row[0] or 0.0) if row else 0.0

    async def registry_status(self) -> tuple[float, float, int, float]:
        """One-shot probe used by the gateway's per-message hot-reload loop.

        Returns ``(mcps_max_updated, models_max_updated, enabled_models_count,
        providers_max_updated)`` in a single round-trip so the dispatcher
        doesn't pay four SELECTs per incoming message.

        ``enabled_models_count`` requires BOTH the model row AND its
        parent provider to be enabled — a model under a disabled
        provider can't dispatch anyway.
        """
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT "
            "  COALESCE((SELECT MAX(updated_at) FROM mcps), 0), "
            "  COALESCE((SELECT MAX(updated_at) FROM models), 0), "
            "  COALESCE(("
            "    SELECT COUNT(*) FROM models m "
            "    JOIN providers p ON p.id = m.provider_id "
            "    WHERE m.enabled = 1 AND p.enabled = 1"
            "  ), 0), "
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
            "SELECT framework FROM session_bindings WHERE session_id = ?",
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
        framework: str,
        runtime_id: str | None = None,
    ) -> None:
        """Record that ``session_id`` is served by ``framework`` (agno / claude-cli).

        Optional ``runtime_id`` pins the session to a specific model.
        Used by SmartRouter after a first successful dispatch so
        subsequent turns are forced to the same side. Claude-cli
        *side* bindings land in ``sdk_sessions`` instead (via
        ``set_sdk_session``); this table tracks agno side + per-session
        explicit model pins for both sides.
        """
        if framework not in VALID_FRAMEWORKS:
            raise ValueError(
                f"invalid framework {framework!r}; expected one of {VALID_FRAMEWORKS}"
            )
        conn = await self._ensure_connected()
        await conn.execute(
            "INSERT INTO session_bindings (session_id, framework, bound_at, runtime_id) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "framework = excluded.framework, bound_at = excluded.bound_at, "
            "runtime_id = excluded.runtime_id",
            (session_id, framework, time.time(), runtime_id),
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
        from openagent.models.catalog import framework_of

        if not session_id or not runtime_id:
            raise ValueError("session_id and runtime_id are required")
        target_framework = framework_of(runtime_id)
        if target_framework not in VALID_FRAMEWORKS:
            raise ValueError(
                f"runtime_id {runtime_id!r} resolved to an unknown framework {target_framework!r}"
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
        turn. The ``framework`` side-binding is *not* touched — a
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
