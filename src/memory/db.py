"""SQLite storage for scheduled tasks, usage logs, providers, and models."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sqlite3
import time
import uuid
from collections import deque
from typing import Any

import aiosqlite
from src.memory.schedule import (
    ONE_SHOT_PREFIX,
    build_one_shot_expression,
    is_one_shot_expression,
    parse_one_shot_expression,
)
from src.models.catalog import (
    FRAMEWORK_API_BASED,
    LLM_FRAMEWORKS,
    SUPPORTED_FRAMEWORKS,
)


logger = logging.getLogger(__name__)

# Per-process worker identity, stamped on an event-delivery claim so the
# heartbeat can prove "still mine" (``WHERE worker_id = ?``) and an operator
# can see which pod/process owns an in-flight row. Generated once per process;
# a claim + its heartbeat + its dispatch all run in the SAME process (the
# gateway for webhook deliveries, the scheduler for the out-of-process drain),
# so this constant is always the owner of the leases it stamps.
WORKER_ID = str(uuid.uuid4())
WORKER_PID = os.getpid()


# ── One busy timeout for everything that writes this file ────────────────────
#
# SQLite in WAL mode has exactly ONE writer, and this agent has many: the
# gateway, the scheduler, the workflow executor, compaction, the runtime's own
# session store, and every in-tree MCP subprocess — several of them in separate
# processes, so Python-level serialisation buys nothing. A writer that waits
# gets its turn; a writer that gives up early raises "database is locked" and
# leaves work half-done (a workflow_run stranded in 'running', a compaction
# thrown away, an event lease unreaped).
#
# The failures we kept seeing were never "the lock was held forever" — they
# were connections that had been given a *different*, shorter patience than
# everyone else: 10s here, 5s there, none at all in the runtime store. So the
# number lives here, once, and every writer reads it. Tunable per deployment
# via ``OPENAGENT_SQLITE_BUSY_TIMEOUT_MS`` for the rare host that needs it.
_DEFAULT_SQLITE_BUSY_TIMEOUT_MS = 60_000


def sqlite_busy_timeout_ms() -> int:
    """Milliseconds every connection to the agent DB should wait for the
    writer before giving up. Read live so a redeploy is not needed to retune."""
    raw = os.environ.get("OPENAGENT_SQLITE_BUSY_TIMEOUT_MS")
    if raw is None:
        return _DEFAULT_SQLITE_BUSY_TIMEOUT_MS
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return _DEFAULT_SQLITE_BUSY_TIMEOUT_MS
    # A zero/negative timeout would mean "fail instantly", which is the bug
    # this constant exists to remove. Treat it as "use the default".
    return value if value > 0 else _DEFAULT_SQLITE_BUSY_TIMEOUT_MS


def sqlite_busy_timeout_s() -> float:
    """The same budget in seconds, for ``sqlite3.connect(timeout=...)`` — which
    covers the window BEFORE the first PRAGMA can run on a new connection."""
    return sqlite_busy_timeout_ms() / 1000.0

# Lease defaults. The lease is SHORT so a frozen turn is reclaimed in ~LEASE_TTL
# rather than the coarse 30-min stale-sweep age; the heartbeat (a tiny single-row
# write that survives writer contention) keeps a legitimately-running turn's
# lease alive, so failure-to-heartbeat is the freeze signal. Both are read live
# from the environment so an operator can retune without a redeploy.
_LEASE_TTL_ENV = "OPENAGENT_EVENT_LEASE_TTL_SECONDS"
_LEASE_TTL_DEFAULT = 120.0


def _env_bool(name: str, default: bool) -> bool:
    """A boolean env override; unset → ``default``. Off values: 0/false/no/off/''."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _env_float(name: str, default: float) -> float:
    """A float env override that falls back to ``default`` on unset/garbage."""
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    """An int env override that falls back to ``default`` on unset/garbage."""
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _lease_ttl_seconds() -> float:
    """The claim-lease TTL in seconds (>= 1), read live from the environment."""
    return max(1.0, _env_float(_LEASE_TTL_ENV, _LEASE_TTL_DEFAULT))


VALID_MCP_KINDS = ("builtin", "custom", "default")
# Alias kept for the ``from src.memory.db import VALID_FRAMEWORKS``
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
    -- Optional per-task model override (a runtime_id such as
    -- ``anthropic:claude-opus-4-8``). NULL → the firing runs on the agent's
    -- default/router model, exactly like a chat turn with no session pin.
    -- Added by ``_migrate_scheduled_tasks_model_column`` on old DBs.
    model TEXT,
    -- IANA timezone the cron expression is read in (e.g. 'Europe/Rome').
    -- NULL → UTC, which is how every cron already behaved (croniter reads a
    -- float epoch as UTC), so upgrading moves nothing. It is never
    -- backfilled: existing rows hold expressions the operator already
    -- hand-converted to UTC ("23 11 * * 1-5 UTC ≈ 13:23 Europe/Rome"), and
    -- re-reading those under an agent-wide default would shift every
    -- production cron at once. Set per task to opt in; see
    -- ``src/memory/schedule.py`` for the DST rules that follow from it.
    -- Added by ``_migrate_scheduled_tasks_timezone_column`` on old DBs.
    timezone TEXT,
    -- Provider-neutral unattended-run envelope. JSON fields currently:
    -- max_tool_calls, timeout_seconds, allowed_tool_families. NULL preserves
    -- the historical global defaults. Migrated additively on old DBs.
    execution_policy_json TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_enabled ON scheduled_tasks(enabled);
CREATE INDEX IF NOT EXISTS idx_tasks_next_run ON scheduled_tasks(next_run);

-- Per-firing execution history for scheduled_tasks. The Scheduler opens
-- a row when it fires a task and flips it to success/failed when the
-- agent turn returns, capturing an output/error preview + timing. This
-- is the scheduled-task analogue of ``workflow_runs`` — it's what lets
-- the dashboard show a task's run history instead of only a single
-- ``last_run`` timestamp. Reads: ``GET /api/scheduled-tasks/{id}/runs``.
CREATE TABLE IF NOT EXISTS task_runs (
    id          TEXT PRIMARY KEY,
    task_id     TEXT NOT NULL REFERENCES scheduled_tasks(id) ON DELETE CASCADE,
    trigger     TEXT NOT NULL DEFAULT 'schedule',
    status      TEXT NOT NULL,
    started_at  REAL NOT NULL,
    finished_at REAL,
    output      TEXT,
    error       TEXT,
    -- The durable child ``sessions`` row this firing ran as (per-run id
    -- ``scheduler:{task_id}:{run_id}``). Lets the app open a past firing
    -- as a full chat session. Added to existing DBs by
    -- ``_migrate_task_runs_session_id``.
    session_id  TEXT
);
CREATE INDEX IF NOT EXISTS idx_taskruns_task    ON task_runs(task_id);
CREATE INDEX IF NOT EXISTS idx_taskruns_started ON task_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_taskruns_status  ON task_runs(status);

-- On-demand "run now" requests for scheduled tasks. The scheduler MCP
-- subprocess (and any other out-of-process caller) drops a row here to
-- ask the main OpenAgent process to fire a task immediately, out of band
-- from its cron schedule. The Scheduler claims these on its fast
-- cross-process loop (~2s) and runs the task, linking the spawned
-- ``task_runs`` row back via ``run_id`` so a waiting caller can poll for
-- completion. The scheduled-task analogue of ``workflow_run_requests``;
-- firing this way leaves the task's schedule (and enabled flag) untouched.
CREATE TABLE IF NOT EXISTS task_run_requests (
    id          TEXT PRIMARY KEY,
    task_id     TEXT NOT NULL REFERENCES scheduled_tasks(id) ON DELETE CASCADE,
    trigger     TEXT NOT NULL DEFAULT 'manual',
    claimed_at  REAL,
    run_id      TEXT,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_taskreq_unclaimed ON task_run_requests(claimed_at);

-- One row per (run, vault note recalled). The join that lets the vault be a
-- policy instead of a diary: which notes the agent READ before a run, and how
-- that run ended. Written by ``TeamRouterProvider`` next to ``record_cost``
-- (see src/core/vault_recall.py); read by ``vault-gate``'s
-- ``vault_recall_stats``.
--
-- WHY A TABLE AND NOT NOTE FRONTMATTER: the vault is Markdown, git-backed and
-- human-editable (§5). These counters are machine-generated and change on
-- every turn — writing them into frontmatter would rewrite notes the user did
-- not touch, churn the vault's git history, and hand dream mode (§12) a moving
-- target while it tries to consolidate. Scores are ABOUT notes, not part of
-- them, exactly as ``usage_log`` is about model calls and lives here.
--
-- WHY RAW ROWS AND NOT AGGREGATED COUNTERS: what counts as a good outcome is a
-- v1 guess (see ``vault_recall.OUTCOME_*``) and WILL be refined. Per-note
-- counters would bake today's definition in permanently and need a migration
-- plus a backfill to change it; raw rows can simply be re-scored. Cheap, too:
-- the production log's 697 streamed turns over two months would be a few
-- thousand rows.
--
-- ``cost`` is stored, not joined from ``usage_log`` at read time: that ledger
-- is keyed by ``session_id``, and a session has MANY turns, so a join would
-- attribute a whole session's spend to every note any turn in it read. This
-- column is the same ``BudgetTracker.compute_cost`` value ``usage_log`` gets,
-- captured per run.
CREATE TABLE IF NOT EXISTS vault_recall_outcomes (
    id           TEXT PRIMARY KEY,
    timestamp    REAL NOT NULL,
    session_id   TEXT,
    note_path    TEXT NOT NULL,
    -- The vault read tool that surfaced the note (vault_read_note, …).
    tool         TEXT NOT NULL,
    -- 'ok' | 'errored' | 'cancelled'. 'cancelled' is a barge-in and is NEVER
    -- scored — kept only so the recall count stays honest. See SCORABLE.
    outcome      TEXT NOT NULL,
    model        TEXT,
    cost         REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_vault_recall_note ON vault_recall_outcomes(note_path);
CREATE INDEX IF NOT EXISTS idx_vault_recall_ts   ON vault_recall_outcomes(timestamp);

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

-- Budget rules — per-scope spend caps that the router enforces by EXCLUDING
-- an over-cap scope from the enabled catalog for the rest of the window (it
-- never hard-stops a turn; the agent keeps working on whatever stays enabled).
-- First-class and editable everywhere the other automation objects are (yaml
-- seed + REST ``/api/budgets`` + the ``budget-manager`` MCP), exactly like
-- ``events`` / ``models`` / ``mcps``.
--
-- WHY A TABLE (not a yaml-only knob): the operator just put $100 of PAYG on
-- DeepSeek, which is disabled precisely because nothing read a budget before a
-- call. ``BudgetTracker`` only RECORDS cost to ``usage_log``; a dispatcher gate
-- that would have stopped a runaway was removed with a yaml knob. A DB table is
-- what makes a rule editable from the app at 3am without a redeploy — the same
-- reason models/events/mcps are DB-backed and not config.
--
-- ``scope_kind`` + ``scope_value`` name what the rule meters, WITHOUT
-- overloading one string (a provider literally named "global", or one whose
-- name contains a colon, must not collide with a model runtime_id):
--   ``global``   → ALL spend, every model/provider (``scope_value`` = '').
--   ``provider`` → every model of a provider (``scope_value`` = provider name,
--                  e.g. 'deepseek'; the spend query does ``model LIKE 'deepseek:%'``
--                  since runtime_ids are ``provider:model``).
--   ``model``    → one runtime_id (``scope_value`` = 'deepseek:deepseek-v4-pro').
--   ``task``     → one scheduled task's whole run tree (``scope_value`` = task
--                  name; ``session_id LIKE 'scheduler:<task>:%'``). A task is a
--                  CALLER, not a routing target — excluding a model can't "stop
--                  a task" — so ``task`` is REPORTED (usage view) but NOT
--                  enforced by the router gate; its enforcement (scheduler skip
--                  + mid-run cancel) is phase 2. Stored/validated now so that
--                  arrives without a migration.
-- The router gate acts only on ``scope_kind IN (global, provider, model)``.
--
-- ``metric``: ``cost_usd`` sums ``usage_log.cost``; ``tokens`` sums
-- ``input_tokens + output_tokens``. Both exist because token costs differ per
-- model (a token cap is model-agnostic where a dollar cap is not) AND because
-- ``compute_cost`` logs 0 when OpenRouter pricing is unavailable — a token
-- budget is the robust fallback when dollar pricing is uncertain.
--
-- ``window``: ``hour`` / ``day`` / ``month`` are calendar windows (boundaries
-- computed in the agent's timezone, see ``src/core/budget_guard.py``).
-- ``per_run`` has no calendar boundary and applies only to task/run enforcement
-- (phase 2); like ``task`` it is accepted + stored now to avoid a later ALTER,
-- but the router gate ignores it.
--
-- ``scope_value`` is NOT NULL DEFAULT '' (not NULL) so the UNIQUE below also
-- pins one global rule per (metric, window) — SQLite treats NULLs as distinct,
-- which would let two identical global caps coexist.
--
-- ``source`` mirrors the marketplace/mcps marker: 'yaml' rows are seeded
-- additively and only-if-absent (see ``seed_budget``), so an operator who edits
-- a rule in the app is never clobbered on the next boot; 'user'/'agent' rows
-- come from REST / the MCP.
CREATE TABLE IF NOT EXISTS budgets (
    id                    TEXT PRIMARY KEY,
    scope_kind            TEXT NOT NULL DEFAULT 'model'
                              CHECK (scope_kind IN ('global','provider','model','task')),
    scope_value           TEXT NOT NULL DEFAULT '',
    metric                TEXT NOT NULL DEFAULT 'cost_usd'
                              CHECK (metric IN ('cost_usd','tokens')),
    window                TEXT NOT NULL DEFAULT 'day'
                              CHECK (window IN ('hour','day','month','per_run')),
    amount                REAL NOT NULL,
    -- JSON array of fractions in (0,1) at which to emit a ``budget.alert``
    -- event (the 1.0 cap event always fires and is implicit). Default [0.5,0.9].
    alert_thresholds_json TEXT NOT NULL DEFAULT '[0.5,0.9]',
    -- Optional outbound webhook fired once per threshold crossing (in addition
    -- to the structured event the logs MCP already sees).
    webhook_url           TEXT,
    enabled               INTEGER NOT NULL DEFAULT 1,
    source                TEXT NOT NULL DEFAULT 'user',
    created_at            REAL NOT NULL,
    updated_at            REAL NOT NULL,
    UNIQUE(scope_kind, scope_value, metric, window)
);
CREATE INDEX IF NOT EXISTS idx_budgets_enabled ON budgets(enabled);

-- Canonical session table — every chat conversation lives here.
-- Sessions are written natively by the inlined ``SqliteDb`` (which
-- creates this table on first use via its own ORM).
--
-- ``runs`` is a JSON array of RunOutput-shaped objects:
--   [{"run_id": "...", "messages": [{"role":"user","content":"..."},...],
--     "status":"completed", "metrics":{...}, ...}]
--
-- ``metadata`` is a JSON object for framework-private state:
--   - The native path stores its own session/agent/team descriptors
--     (managed by SqliteDb).
--
-- Renamed from the legacy ``agno_sessions`` in v0.14; legacy DBs are migrated
-- transparently by ``_migrate_legacy_agno_sessions_to_sessions`` on
-- bootstrap.
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    session_type TEXT,
    agent_id     TEXT,
    team_id      TEXT,
    workflow_id  TEXT,
    user_id      TEXT,
    session_data TEXT,
    agent_data   TEXT,
    team_data    TEXT,
    workflow_data TEXT,
    metadata     TEXT,
    runs         TEXT,
    summary      TEXT,
    created_at   INTEGER NOT NULL,
    updated_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at);

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

-- LLM and media providers. One row per (vendor, framework) pair.
--
-- OpenAgent vocabulary:
--   - **provider**  = a concrete credential + dispatch pair, e.g.
--                     ``anthropic`` with ``framework='api-based'`` (needs
--                     ``api_key``, native the runtime ``Agent``).
--   - **framework** = which runtime class to instantiate for this row:
--                     ``api-based`` (native ``Agent`` for LLM, or
--                     ``litellm.aspeech``/``atranscription`` for TTS/STT).
--                     The only shipped framework; the column stays the
--                     seam for adding more later.
--
-- ``UNIQUE(name, framework)`` lets the UI/API/MCP address a row by its
-- (vendor, framework) pair; the surrogate ``id`` is what the ``models``
-- table joins to.
-- ``kind`` partitions the registry by capability:
--   ``llm`` — text generation (framework='api-based').
--   ``tts`` — speech synthesis (framework='api-based'; routed via litellm).
--   ``stt`` — speech-to-text (framework='api-based'; routed via litellm).
-- Router/classifier code MUST filter to ``kind='llm'`` before iterating
-- providers, so a TTS row never gets handed to the LLM dispatch path.
CREATE TABLE IF NOT EXISTS providers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    framework TEXT NOT NULL,
    api_key TEXT,
    base_url TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    kind TEXT NOT NULL DEFAULT 'llm' CHECK (kind IN ('llm','tts','stt')),
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(name, framework)
);
CREATE INDEX IF NOT EXISTS idx_providers_enabled ON providers(enabled);
CREATE INDEX IF NOT EXISTS idx_providers_updated ON providers(updated_at);
CREATE INDEX IF NOT EXISTS idx_providers_name ON providers(name);
-- idx_providers_kind is created in _apply_legacy_alters, after the
-- column is guaranteed to exist on legacy DBs (same rationale as
-- idx_models_is_classifier below).

-- Configured LLM models. Each row is a bare vendor id plus a FK to the
-- provider row that owns it. Framework is inherited from the provider —
-- deleting a provider cascades to wipe its models (ON DELETE CASCADE).
--
-- ``model`` is the bare vendor id (e.g. ``gpt-4o-mini``, ``claude-opus-4-7``).
-- The canonical ``runtime_id`` used in logs / session pins / entry-model
-- resolution is DERIVED at read time from the provider row's (name,
-- framework) pair — no longer stored here.
--
-- ``tier_hint`` absorbs the old ``notes`` column: the model's free-form
-- scope (``"vision, 200k context, best for code"``, ``"cheap and fast"``).
-- It is live text, not a note to operators — ``_build_role_blurb`` turns
-- it into the ``role`` the team leader reads when picking who to
-- delegate to.
--
-- ``kind`` partitions the row by capability:
--   ``llm`` — text generation (the dispatcher resolves one as the turn's
--     entry model; the rest join its team as members).
--   ``tts`` — speech synthesis. ``metadata.voice_id`` carries the voice.
--   ``stt`` — speech-to-text.
-- LLM rows route via the provider's framework (``api-based`` builds
-- the runtime ``Agent`` from the provider's vendor class); ``tts`` / ``stt`` rows
-- always dispatch through LiteLLM (``litellm.aspeech`` /
-- ``atranscription``) using the provider's ``name`` as the LiteLLM
-- vendor prefix (``openai/tts-1``).
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider_id INTEGER NOT NULL REFERENCES providers(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    display_name TEXT,
    tier_hint TEXT,
    description TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    is_classifier INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    kind TEXT NOT NULL DEFAULT 'llm' CHECK (kind IN ('llm','tts','stt')),
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

-- Per-session model pin. Optional explicit user/agent choice of a
-- specific runtime_id for a session; the dispatcher honours this pin
-- before falling back to first-enabled. ``runtime_id`` is a human-
-- readable label (e.g. ``anthropic:claude-opus-4-7``)
-- derived from the provider + model rows at pin time; it's not a FK
-- so a later model delete leaves a "stale pin" the router gracefully
-- falls back from rather than throwing an integrity error.
--
-- v0.14+: replaces the old ``session_bindings`` table (which carried
-- a framework lock on top of the pin). With history unified in
-- the runtime's ``sessions`` across every framework, the lock is no longer
-- needed — only the pin value itself survives.
-- ── Session journal: what happened, written while it happened ──
--
-- ``sessions.runs`` is the surface the MODEL sees, and the runtime writes it
-- when a turn ENDS. That is a fine contract for the model and a terrible one
-- for everybody else: a turn interrupted before it closes leaves no trace at
-- all, so an app that was mid-conversation has nothing to reconcile against
-- and a support question about "what did it do at 14:32" has no answer.
--
-- This table is the other half — an append-only journal of the facts of a
-- turn, written as they occur: the user's message, the assistant's message,
-- tool status, compaction, errors, and how the turn ended. Borrowed from
-- DeepSeek Harness's session log, minus what does not apply to us (we do not
-- journal every stream delta: they are re-derivable from the final message and
-- would be the bulk of the volume).
--
-- Rules that make it worth having:
--   * append-only — rows are never updated, only inserted;
--   * ``seq`` is monotonic per session, so a client can ask "what happened
--     after 41" and get an answer instead of inferring from silence;
--   * ``data`` is JSON and may grow new keys; a reader that meets an unknown
--     ``type`` skips it rather than failing (nothing here is load-bearing for
--     reconstruction yet — when something becomes so, it gets an explicit
--     non-ignorable marker, as dsh does).
CREATE TABLE IF NOT EXISTS session_events (
    session_id TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    ts_ms      INTEGER NOT NULL,
    type       TEXT NOT NULL,
    data       TEXT,
    PRIMARY KEY (session_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_session_events_session
    ON session_events(session_id, seq);

CREATE TABLE IF NOT EXISTS pinned_sessions (
    session_id TEXT PRIMARY KEY,
    runtime_id TEXT NOT NULL,
    pinned_at REAL NOT NULL
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
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL UNIQUE,
    description         TEXT,
    graph_json          TEXT NOT NULL DEFAULT '{"version":1,"nodes":[],"edges":[],"variables":{}}',
    trigger_kind        TEXT NOT NULL DEFAULT 'manual',  -- DEPRECATED, ignored
    cron_expression     TEXT,                             -- DEPRECATED, ignored
    enabled             INTEGER NOT NULL DEFAULT 1,
    last_run_at         REAL,
    next_run_at         REAL,                              -- DEPRECATED, ignored
    -- Optional cap on overlapping runs of THIS workflow. NULL means
    -- unlimited (default) — overlapping runs all execute concurrently.
    -- 1 → fully serialized (matches the pre-v0.14.2 behaviour). N>1 →
    -- up to N runs in flight, additional callers wait on a per-workflow
    -- ``asyncio.Semaphore`` inside ``WorkflowExecutor``.
    max_concurrent_runs INTEGER,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL
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

-- ── Network / identity / coordinator ──
--
-- This agent's role in the OpenAgent network model. Singleton row.
-- ``role`` is one of:
--   ``standalone`` — no network configured yet (boot default).
--   ``coordinator`` — this agent runs the embedded coordinator service.
--   ``member``     — this agent is a member of an external network.
-- ``coordinator_pubkey`` is the Ed25519 verify key (32 raw bytes) used to
-- verify device-certs presented by inbound clients on this gateway.
CREATE TABLE IF NOT EXISTS network (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
    role TEXT NOT NULL DEFAULT 'standalone',
    network_id TEXT,
    name TEXT,
    coordinator_node_id TEXT,
    coordinator_pubkey BLOB,
    created_at REAL NOT NULL
);

-- Networks this agent belongs to as a CLIENT (used for agent-to-agent
-- federation: an agent stores other networks here so it can dial peer
-- agents through them). Independent of the singleton ``network`` row,
-- which only describes this agent's own home network.
CREATE TABLE IF NOT EXISTS peer_networks (
    network_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    coordinator_node_id TEXT NOT NULL,
    coordinator_pubkey BLOB NOT NULL,
    our_handle TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    added_at REAL NOT NULL,
    last_seen REAL
);
CREATE INDEX IF NOT EXISTS idx_peer_networks_status ON peer_networks(status);

-- Cached device certificates (CBOR-encoded, Ed25519-signed by the
-- coordinator) per (network, handle). Refreshed at 50% TTL by the
-- session dialer; deleted on logout.
CREATE TABLE IF NOT EXISTS device_certs (
    network_id TEXT NOT NULL,
    handle TEXT NOT NULL,
    cert BLOB NOT NULL,
    expires_at REAL NOT NULL,
    PRIMARY KEY(network_id, handle)
);
CREATE INDEX IF NOT EXISTS idx_device_certs_expires ON device_certs(expires_at);

-- ── Coordinator-only tables ──
-- Populated only when ``network.role='coordinator'``. Other agents
-- carry these tables empty (cheaper than two schemas).

-- Registered users for this network. ``pake_record`` is the SRP-6a
-- verifier (or OPAQUE record once we swap the PAKE backend) — opaque
-- to the coordinator: the password itself never reaches us.
CREATE TABLE IF NOT EXISTS network_users (
    handle TEXT PRIMARY KEY,
    pake_record BLOB NOT NULL,
    pake_algo TEXT NOT NULL DEFAULT 'srp6a',
    status TEXT NOT NULL DEFAULT 'active',
    created_at REAL NOT NULL
);

-- One row per device a user has paired with this network. The
-- coordinator signs a fresh device cert for each device that completes
-- a PAKE login; revocation flips ``status`` to 'revoked' so future
-- cert refreshes fail and inbound dials carrying the old cert are
-- rejected (the agent middleware checks this table on every dial).
CREATE TABLE IF NOT EXISTS network_devices (
    device_pubkey BLOB PRIMARY KEY,
    user_handle TEXT NOT NULL,
    label TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    added_at REAL NOT NULL,
    last_seen REAL,
    FOREIGN KEY(user_handle) REFERENCES network_users(handle) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_network_devices_user ON network_devices(user_handle);

-- Agents registered in this network. Discovery is coordinator-mediated:
-- clients call ``list_agents`` and dial ``node_id`` directly afterwards.
CREATE TABLE IF NOT EXISTS network_agents (
    handle TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    label TEXT,
    owner_handle TEXT NOT NULL,
    added_at REAL NOT NULL,
    last_seen REAL
);

-- One-shot or N-shot invites. ``code`` is a base32 string the user
-- pastes / scans. ``role`` decides what they get on redemption: ``user``
-- registers a new account, ``device`` adds a device to an existing
-- account.
CREATE TABLE IF NOT EXISTS network_invitations (
    code TEXT PRIMARY KEY,
    role TEXT NOT NULL CHECK(role IN ('user','device','agent')),
    created_by TEXT,
    bind_to_handle TEXT,
    uses_left INTEGER NOT NULL DEFAULT 1,
    expires_at REAL NOT NULL,
    created_at REAL NOT NULL,
    used_at REAL
);
CREATE INDEX IF NOT EXISTS idx_invitations_expires ON network_invitations(expires_at);

-- ── Learning tables (DORMANT — no writer, empty everywhere) ────────
-- Both tables below are empty on every deployment and always have been.
-- Their only writers were ``src.learning.user_profile`` / ``.skills``,
-- whose flush + detector hooks had zero callers; both modules were
-- deleted in v0.15.11 for being a second, opaque memory system
-- competing with the vault (§5 wants long-term memory as readable
-- Markdown, not SQLite blobs) — ``src/learning/__init__.py`` carries the
-- full argument. Do not re-point these at a new writer; that data is a
-- vault note.
--
-- They are KEPT rather than dropped because dropping is all cost and no
-- gain: it buys a 15th ``_migrate_*`` to delete tables that never held a
-- row, and edits to the readers that still name them
-- (``bridges/telegram.py``'s ``/export``, ``purge_session``'s
-- ``_SESSION_SATELLITE_TABLES``). Both readers are already try/except'd
-- and degrade to empty payloads, and ``CREATE TABLE IF NOT EXISTS``
-- costs one no-op per boot. Retiring them wants one owner for the schema
-- AND those readers, in one pass.
--
-- ``user_profiles`` was *meant* to accumulate a JSON summary of who the
-- user is per session (preferences, ongoing projects, communication
-- style), flushed every N turns and injected back as system context at
-- resume so the bot didn't restart cold. It never was: nothing writes it.
-- Hermes equivalent: ``memory.user_profile_enabled``.
CREATE TABLE IF NOT EXISTS user_profiles (
    session_id      TEXT PRIMARY KEY,
    profile_json    TEXT NOT NULL DEFAULT '{}',
    turn_count      INTEGER NOT NULL DEFAULT 0,
    last_flushed_at REAL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_user_profiles_updated ON user_profiles(updated_at);

-- ``skills`` was *meant* to store reusable how-tos distilled from
-- recurring request patterns: ``signature_hash`` a stable hash of the
-- normalised trigger so the detector could dedup, ``markdown`` the
-- playbook text a matcher would inject at turn start once relevance hit
-- a threshold. Neither exists any more — the detector that would have
-- written this and the matcher that would have read it are both deleted,
-- so the columns below describe an intent, not a behaviour.
-- Hermes equivalent: skills + curator.
CREATE TABLE IF NOT EXISTS skills (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    signature_hash  TEXT NOT NULL UNIQUE,
    markdown        TEXT NOT NULL,
    tags_json       TEXT NOT NULL DEFAULT '[]',
    usage_count     INTEGER NOT NULL DEFAULT 0,
    last_used_at    REAL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_skills_usage ON skills(usage_count);
CREATE INDEX IF NOT EXISTS idx_skills_last_used ON skills(last_used_at);

-- ``conversation_embeddings`` stores one row per persisted turn pair
-- (user + assistant) with the OpenAI embedding serialised as a JSON
-- float array. The ``memory-search`` MCP queries this table via cosine
-- similarity so the agent can answer "remember when we discussed X?"
-- across past sessions. Embedding vector size depends on the model
-- (1536 for ``text-embedding-3-small``). Pruning is operator-managed
-- via the curator (future) or manual ``DELETE WHERE timestamp < …``.
CREATE TABLE IF NOT EXISTS conversation_embeddings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    role            TEXT NOT NULL,
    text            TEXT NOT NULL,
    embedding_json  TEXT NOT NULL,
    model           TEXT NOT NULL,
    timestamp       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_emb_session ON conversation_embeddings(session_id);
CREATE INDEX IF NOT EXISTS idx_conv_emb_timestamp ON conversation_embeddings(timestamp);
-- ``vault_save_reminders`` tracks per-session turn counts so the
-- reminder injector in ``src.learning.vault_reminder`` can fire
-- a memory-checkpoint prompt into the user turn every N turns.
-- Unlike ``user_profiles``, this table owns only the counter -- no
-- profile payload -- so the two features can be enabled independently.
CREATE TABLE IF NOT EXISTS vault_save_reminders (
    session_id      TEXT PRIMARY KEY,
    turn_count      INTEGER NOT NULL DEFAULT 0,
    last_reminded_at REAL,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vault_reminders_updated ON vault_save_reminders(updated_at);

-- ── Events (webhook channel) ──
--
-- An ``event`` is a first-class inbound trigger (vision §9 "one agent,
-- many doorways"): an external service (or an Iroh peer) hits the webhook
-- and the agent runs a bound action — an existing workflow, a scheduled
-- task, or a fresh chat-prompt session. It is the inbound analogue of the
-- outbound bridges (Telegram/Discord/…): those dial *out* to a platform,
-- an event lets a platform reach *in*.
--
-- ``slug`` is the public path segment: ``POST /hooks/{slug}`` on the
-- dedicated webhook listener. ``type`` is the provider preset that decides
-- how the request is authenticated and how a de-dupe id is extracted
-- (generic | generic-hmac | github | stripe | slack). The per-event secret
-- is NEVER stored in clear: only a salted sha256 (``secret_hash`` +
-- ``secret_salt``) plus a 4-char ``secret_hint`` for the UI. ``action_kind``
-- is one of ``workflow`` | ``scheduled_task`` | ``prompt``; ``action_ref``
-- points at the workflow/task id (NULL for ``prompt``, which carries a
-- ``prompt_template`` rendered against the delivery payload instead).
CREATE TABLE IF NOT EXISTS events (
    id                 TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    slug               TEXT NOT NULL UNIQUE,
    description        TEXT,
    type               TEXT NOT NULL DEFAULT 'generic',
    enabled            INTEGER NOT NULL DEFAULT 1,
    -- The per-event secret, ENCRYPTED at rest (see src.core.event_secret).
    -- Encryption not hashing, because provider HMAC verification needs the
    -- key in clear. Only ``secret_hint`` (last 4 chars) is unencrypted.
    secret_enc         TEXT NOT NULL,
    secret_hint        TEXT,
    -- User-friendly input schema: a JSON array of
    -- ``[{name,type,required,description,path}]`` where ``path`` is the
    -- dot-path into the payload. Drives delivery validation and the
    -- ``{{payload.x}}`` / workflow-inputs surface.
    input_schema_json  TEXT NOT NULL DEFAULT '[]',
    action_kind        TEXT NOT NULL,
    action_ref         TEXT,
    prompt_template    TEXT,
    -- Optional per-event model pin (a runtime_id), same semantics as
    -- ``scheduled_tasks.model``: NULL → the agent's default/router pick.
    model              TEXT,
    -- Optional prompt-event session binding. When enabled, the dispatcher
    -- reads ``session_binding_path`` from the payload and maps that external
    -- value to one durable OpenAgent child session id in
    -- ``event_session_bindings``. Disabled preserves the default:
    -- one fresh event run session per delivery.
    session_binding_enabled INTEGER NOT NULL DEFAULT 0,
    session_binding_path    TEXT,
    -- Optional cheap check run BEFORE the delivery (``_migrate_events_
    -- precondition`` on old DBs). A queued delivery runs against state that has
    -- moved on since it was enqueued; this settles "is there still work here?"
    -- with one HTTP call instead of a model turn. NULL → always run.
    precondition_json  TEXT,
    -- Provider-neutral capability/resource envelope for every delivery.
    -- A nested task can only narrow this envelope, never expand it.
    execution_policy_json TEXT,
    -- Guardrails (the webhook is the first cert-less inbound surface, and
    -- every delivery is a paid LLM run): requests over the cap are 413'd,
    -- more than ``rate_limit_per_min`` deliveries in a rolling minute are
    -- 429'd.
    rate_limit_per_min INTEGER NOT NULL DEFAULT 60,
    max_payload_bytes  INTEGER NOT NULL DEFAULT 262144,
    last_triggered_at  REAL,
    -- Per-event circuit breaker (``_migrate_events_breaker`` on old DBs). Off by
    -- default (gated by ``OPENAGENT_EVENT_BREAKER_ENABLED``); all columns work at
    -- their 0/NULL defaults with no backfill. ``consecutive_failures`` counts ONLY
    -- permanent failures in a row (transient provider-429/throttle/timeout are NOT
    -- counted — the storm must not block a healthy event); once it reaches the
    -- effective limit (per-event ``max_retries``, else the global default) the
    -- breaker trips and ``breaker_tripped_at`` is stamped, parking further
    -- deliveries ``blocked`` until a success resets the counter.
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    max_retries        INTEGER,
    breaker_tripped_at REAL,
    last_failure_error TEXT,
    created_at         REAL NOT NULL,
    updated_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_slug    ON events(slug);
CREATE INDEX IF NOT EXISTS idx_events_enabled ON events(enabled);

-- Stable map from an event-specific external object id (extracted from the
-- payload) to OpenAgent's internal event-run child session id. The payload id
-- is never used as a session id; it only looks up the internal one.
CREATE TABLE IF NOT EXISTS event_session_bindings (
    event_id     TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    binding_key  TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    PRIMARY KEY (event_id, binding_key)
);
CREATE INDEX IF NOT EXISTS idx_ev_session_bindings_session
    ON event_session_bindings(session_id);

-- Per-firing history for events AND the cross-process run queue, unified
-- in one table (what ``task_runs`` + ``task_run_requests`` are for tasks).
-- A row created by an out-of-process caller (the ``events-manager`` MCP)
-- is born with ``claimed_at IS NULL`` and is claimed by the Scheduler's
-- fast (~2s) loop, exactly like ``task_run_requests``. ``source`` records
-- the doorway: ``webhook`` (external HTTP), ``peer`` (Iroh device-cert /
-- agent ALPN via ``/api/events/{id}/trigger``), ``manual`` (app/CLI Test),
-- ``agent`` (the MCP tool). ``external_id`` is the provider delivery id
-- (``X-GitHub-Delivery`` / Stripe event id) used to make redelivery
-- idempotent. Exactly one of ``session_id`` / ``workflow_run_id`` /
-- ``task_run_id`` links the produced unit of work.
CREATE TABLE IF NOT EXISTS event_deliveries (
    id               TEXT PRIMARY KEY,
    event_id         TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    source           TEXT NOT NULL DEFAULT 'webhook',
    external_id      TEXT,
    status           TEXT NOT NULL,
    payload_json     TEXT NOT NULL DEFAULT '{}',
    started_at       REAL NOT NULL,
    finished_at      REAL,
    output           TEXT,
    error            TEXT,
    session_id       TEXT,
    workflow_run_id  TEXT,
    task_run_id      TEXT,
    claimed_at       REAL,
    -- How many times this delivery has been re-enqueued by the orphan reaper
    -- (``reap_orphan_event_deliveries``). At-least-once delivery bounds the
    -- replay budget on this counter so a delivery that keeps orphaning (e.g.
    -- one that reliably kills the process mid-turn) is eventually parked
    -- terminal instead of churning forever. Added on old DBs by
    -- ``_migrate_event_deliveries_reenqueue_count``.
    reenqueue_count  INTEGER NOT NULL DEFAULT 0,
    -- Claim-lease + heartbeat (``_migrate_event_deliveries_lease`` on old DBs).
    -- ``claim_expires`` is when the lease this claim holds runs out; the
    -- dispatch runner heartbeats it forward while the turn is live, and
    -- ``reap_expired_event_leases`` re-enqueues a row whose lease has lapsed
    -- (the freeze signal). ``worker_id``/``worker_pid`` record the owning
    -- process; ``last_heartbeat_at`` is the last time it proved liveness. ALL
    -- NULL on a legacy in-flight row → the lease reaper never touches it (that
    -- is what makes the new columns additive at deploy: pre-existing in-flight
    -- rows stay handled by the age-gated stale sweep / startup reap).
    claim_expires     REAL,
    worker_id         TEXT,
    worker_pid        INTEGER,
    last_heartbeat_at REAL
);
CREATE INDEX IF NOT EXISTS idx_evdel_event     ON event_deliveries(event_id);
CREATE INDEX IF NOT EXISTS idx_evdel_started   ON event_deliveries(started_at);
CREATE INDEX IF NOT EXISTS idx_evdel_status    ON event_deliveries(status);
CREATE INDEX IF NOT EXISTS idx_evdel_external  ON event_deliveries(event_id, external_id);
CREATE INDEX IF NOT EXISTS idx_evdel_unclaimed ON event_deliveries(claimed_at);
-- idx_evdel_lease ON event_deliveries(claim_expires) is created in
-- _migrate_event_deliveries_lease, NOT here: claim_expires is added by that
-- migration on a pre-existing DB, so indexing it in this base schema block —
-- which runs BEFORE _apply_legacy_alters — crashes boot with
-- "no such column: claim_expires". Create the index right after the ADD COLUMN.
"""


def _as_epoch(value: Any) -> float:
    """Epoch da una colonna ``updated_at`` che DOVREBBE essere REAL.

    Il 24-ago-2026 un `update ... set updated_at=datetime('now')` fatto a mano
    ha scritto TEXT in tre righe di ``models``. In SQLite il testo ordina SOPRA
    i numeri, quindi ``MAX(updated_at)`` restituiva quella stringa,
    ``float()`` alzava ValueError e l'idratazione del catalogo moriva li':
    il dispatcher restava con providers_config VUOTO e ogni turno di supporto
    rispondeva "No model is currently enabled." per 17 minuti, con la delivery
    chiusa `success`. Una riga scritta male e' un bug di chi la scrive; farne
    morire il catalogo e' un bug di chi la legge.
    """
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            import datetime as _dt
            return _dt.datetime.strptime(str(value).strip()[:19], fmt).timestamp()
        except ValueError:
            continue
    return 0.0


class MemoryDB:
    """SQLite storage for OpenAgent's runtime state."""

    def __init__(self, db_path: str = "openagent.db"):
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None
        # Per-process claim owner (see module-level ``WORKER_ID``). Exposed on
        # the instance so the dispatch runner can heartbeat "still mine".
        self.worker_id = WORKER_ID
        self.worker_pid = WORKER_PID

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
        self._conn = await aiosqlite.connect(
            self.db_path, timeout=sqlite_busy_timeout_s(),
        )
        self._conn.row_factory = aiosqlite.Row
        # ``busy_timeout`` gives the same guarantee at every subsequent
        # statement on this connection — not just the initial open.
        #
        # 23-ago-2026: alzato da 10s a 60s. Con il solo scrittore WAL occupato
        # da una transazione lunga, il ciclo dello scheduler falliva
        # `_reap_expired_event_leases` con `database is locked` centinaia di
        # volte al giorno (836 in un'ora sola) — e proprio quel reaper e' la
        # via di recupero delle consegne bloccate, quindi si arrendeva quando
        # sarebbe servito di piu'. Il reaper non e' il colpevole della contesa:
        # misurato, aveva ZERO righe da toccare. Aspettare e' corretto,
        # arrendersi no. Il valore alto e' gia' quello usato da
        # `session_retention`, che convive con lo stesso scrittore.
        await self._conn.execute(f"PRAGMA busy_timeout = {sqlite_busy_timeout_ms()}")
        await self._conn.execute("PRAGMA journal_mode=WAL")
        # Canonical history must survive an OS/power loss once SQLite reports
        # the transaction committed.  Rebuildable FTS indexes may use NORMAL;
        # the canonical operational database may not.
        await self._conn.execute("PRAGMA synchronous=FULL")
        # Enable FK constraints per-connection. SQLite's default is OFF,
        # so without this the ON DELETE CASCADE on models.provider_id is
        # silently a no-op and deleting a provider orphans its models.
        await self._conn.execute("PRAGMA foreign_keys = ON")
        # Pre-schema migration: rename the legacy session table
        # to ``sessions`` BEFORE ``executescript(SCHEMA_SQL)`` runs (so
        # the latter's ``CREATE TABLE IF NOT EXISTS sessions`` doesn't
        # create a fresh empty table alongside the legacy data).
        await self._migrate_legacy_agno_sessions_to_sessions()
        await self._conn.executescript(SCHEMA_SQL)
        await self._apply_legacy_alters()
        # Post-schema migration: reclaim any ``sessions`` row whose owner
        # was pinned to a value the runtime won't match (legacy
        # ``'openagent'`` sentinel, device handle, or the ``__bridge``
        # cert handle), so the runtime can read its history + persist new
        # runs again. See ``upsert_session`` and ``RUNTIME_SESSION_USER_ID``.
        await self._migrate_reclaim_session_owners()
        await self._conn.commit()
        # Additive only: takes a verified SQLite backup before the first v2
        # DDL, installs the downgrade journal, and enters shadow.  Legacy
        # ``sessions.runs`` remains intact and canonical until parity gates
        # promote individual reads.
        from src import __version__
        from src.memory.operational.schema import ensure_operational_storage

        # Capture downgrade-trigger evidence before the additive migrator
        # reinstalls any missing trigger.  A clean promoted boot can trust the
        # append-only change journal and avoid reparsing the whole transcript;
        # a table replacement/old binary that dropped a trigger forces the
        # expensive exact audit once.
        required_legacy_triggers = {
            "trg_legacy_sessions_insert_journal",
            "trg_legacy_sessions_update_journal",
            "trg_legacy_sessions_delete_journal",
        }
        trigger_rows = await (
            await self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND name IN (?,?,?)",
                tuple(sorted(required_legacy_triggers)),
            )
        ).fetchall()
        legacy_triggers_intact = {
            str(row[0]) for row in trigger_rows
        } == required_legacy_triggers
        await ensure_operational_storage(
            self._conn,
            self.db_path,
            app_version=__version__,
        )
        # Artifact metadata is always owner-private.  Sharing is inherited
        # from live resource links so later ACL revocation/deletion is honored;
        # keep this normalization outside the shipped v2 schema checksum.
        from src.memory.artifact_acl_migration import ensure_artifact_acl_storage

        await ensure_artifact_acl_storage(
            self._conn,
            app_version=__version__,
        )
        # Custom Views is a separate, additive checksummed migration.  Never
        # append its DDL to operational_storage_v2.sql: completed v2 ledgers
        # intentionally reject checksum drift on already-shipped databases.
        from src.custom_views.migration import ensure_custom_views_storage

        await ensure_custom_views_storage(
            self._conn,
            app_version=__version__,
        )
        # Ordered content is independent from the provider's legacy message
        # JSON and from the UI definition schema.  Install it last because its
        # foreign keys target both operational v2 and Custom Views tables.
        from src.memory.message_parts_migration import ensure_message_parts_storage

        await ensure_message_parts_storage(
            self._conn,
            app_version=__version__,
        )
        # A pre-v2 binary may have written the compatibility table while this
        # build was offline.  Its durable triggers leave pending journal rows;
        # promoted reads must fall back to shadow *before* any runtime writer or
        # scheduler becomes visible, then normal reconciliation can catch up.
        from src.memory.operational.phase import guard_storage_phase_atomic_async

        await guard_storage_phase_atomic_async(
            self._conn,
            writer_version=__version__,
            audit_content=not legacy_triggers_intact,
        )
        # Drain a bounded page before the connection becomes visible to the
        # agent/scheduler. Larger legacy datasets continue through the durable
        # trigger journal and API/search reconciliation; startup never parses
        # the entire historical runs corpus in one blocking transaction.
        try:
            from src.memory.operational.repository import (
                backfill_batch_async,
                reconcile_pending_async,
            )

            await reconcile_pending_async(
                self._conn,
                limit=250,
                worker_id=f"startup:{WORKER_ID}",
            )
            await backfill_batch_async(self._conn, limit=250)
            await self._conn.commit()
        except Exception as exc:  # shadow reads remain legacy-safe
            await self._conn.rollback()
            logger.warning("operational storage startup reconciliation failed: %s", exc)

    async def _migrate_reclaim_session_owners(self) -> None:
        """One-shot UPDATE: NULL any ``sessions.user_id`` the runtime won't
        match, so it can reclaim the row and resume the conversation.

        The runtime owns the ``user_id`` column and stamps the single
        stable ``RUNTIME_SESSION_USER_ID`` ("openagent") sentinel on every
        session; its history read AND its runs write are gated by
        ``user_id == 'openagent' OR user_id IS NULL`` (see
        ``src/memory/store/sqlite/sqlite.py``). Earlier builds let the
        gateway pin a *different* value here — first the legacy
        ``'openagent'`` INSERT sentinel, then (after that was "fixed") the
        authenticated device handle or the ``__bridge`` cert handle every
        bridge connection carries. Any such row is invisible to the
        runtime: it can neither load the stored transcript nor append to
        it, so the agent "forgot" the conversation on every turn (the
        2026-05 Telegram session-reset bug).

        Setting those owners back to NULL makes the runtime's ``IS NULL``
        soft-match fire on the next turn — it reads the existing ``runs``
        (history restored) and re-claims the row as ``'openagent'`` on
        write. Rows already at ``'openagent'`` or NULL are left untouched.
        Idempotent; safe to run on every connect. Tenancy is carried by
        ``session_id`` (e.g. ``tg:<uid>``), never by this column, so
        collapsing owners to one is correct for the single-tenant agent
        (vision §17)."""
        assert self._conn is not None
        # Literal ``'openagent'`` mirrors ``RUNTIME_SESSION_USER_ID`` in
        # ``src/models/catalog.py`` — kept as a literal here to avoid the
        # memory layer importing the models layer. Keep the two in sync.
        try:
            await self._conn.execute(
                "UPDATE sessions SET user_id = NULL "
                "WHERE user_id IS NOT NULL AND user_id != 'openagent'"
            )
        except Exception:
            # ``sessions`` not present yet on a brand-new DB — the
            # SCHEMA_SQL CREATE TABLE just above created it but on
            # legacy paths the migration order may surprise us. Quiet
            # no-op is the right behaviour here.
            pass

    async def _migrate_legacy_agno_sessions_to_sessions(self) -> None:
        """One-shot ALTER TABLE: rename ``agno_sessions`` to ``sessions``.

        v0.14 dropped the legacy ``agno_`` prefix on the canonical session table.
        Existing user databases shipped with the old name; this migration
        renames the table in place so no data is lost, then recreates the
        ``updated_at`` index under the new name.

        Idempotent:
          - Fresh installs: neither table exists yet → no-op (SCHEMA_SQL
            will create ``sessions`` next).
          - Already-migrated installs: only ``sessions`` exists → no-op.
          - Legacy installs: ``agno_sessions`` exists, ``sessions`` does
            not → rename + reindex.
          - Defensive (shouldn't happen): both exist → leave both alone
            and log a warning. Dropping either would risk data loss.

        Must run BEFORE ``executescript(SCHEMA_SQL)`` so the CREATE TABLE
        IF NOT EXISTS for ``sessions`` doesn't create an empty table next
        to the legacy ``agno_sessions``.
        """
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
            "('agno_sessions', 'sessions')"
        )
        present = {r[0] for r in await cursor.fetchall()}
        has_legacy = "agno_sessions" in present
        has_new = "sessions" in present
        if not has_legacy:
            return  # fresh install or already migrated
        if has_new:
            # Both present — refuse to merge or drop either side. The
            # operator can pick a winner manually.
            try:
                from src.core.logging import elog
                elog(
                    "memory.sessions_rename_skipped",
                    level="warning",
                    reason="both_tables_present",
                    note="agno_sessions and sessions both exist; leaving as-is",
                )
            except Exception:
                pass
            return
        # Legacy-only: rename in place. The old ``idx_agno_sessions_updated``
        # follows the table automatically on RENAME, but its name still
        # carries the legacy prefix — drop it and let SCHEMA_SQL recreate
        # the index under the new name (``idx_sessions_updated``).
        await self._conn.execute("ALTER TABLE agno_sessions RENAME TO sessions")
        await self._conn.execute("DROP INDEX IF EXISTS idx_agno_sessions_updated")
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

        # Voice chat: providers gains a ``kind`` column and drops the
        # framework CHECK that locked it to ('agno','claude-cli'), so
        # audio providers (LiteLLM-routed TTS/STT) can live in the same
        # registry.
        await self._migrate_providers_kind_column()
        # Idempotent on both fresh installs and post-rebuild — the
        # column is guaranteed to exist by the line above.
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_providers_kind ON providers(kind)"
        )
        # Cleanup: the very first voice-chat preview shipped with
        # ``framework='elevenlabs'`` for TTS rows. The current code
        # routes via litellm and lives under ``framework='api-based'``
        # (the kind column carries the TTS vs LLM split now), so any
        # surviving ``elevenlabs`` row would be invisible to the
        # resolver. Convert them in place — the row's existing
        # ``api_key`` and ``metadata`` (voice_id, model_id) carry over
        # 1:1 because the LiteLLM ``elevenlabs/<model>`` shape uses the
        # same names.
        await self._migrate_legacy_elevenlabs_to_litellm()
        # v0.14: collapse legacy framework values ``agno`` and
        # ``litellm`` into the single ``api-based`` value (since the runtime
        # is now the execution engine for both LLM paths; ``kind``
        # discriminates LLM vs TTS/STT). Idempotent. Runs after the
        # kind-column migration so the rows are already in their final
        # provider+kind shape.
        await self._migrate_legacy_framework_names_to_api_based()
        # The claude-cli and codex-cli subscription-CLI adapters were
        # removed — drop any provider rows still on those frameworks so a
        # pre-existing DB can't surface them in the catalog. Their
        # ``models`` rows cascade via the FK ON DELETE. Idempotent.
        await self._migrate_drop_subscription_cli_providers()
        # Add ``kind`` to the models table + fold any TTS/STT provider
        # rows (where ``model_id`` and ``voice_id`` lived in
        # ``metadata_json``) into proper model rows. The unified design:
        # one model row per (provider, model_id, kind), no audio config
        # smuggled into provider metadata.
        await self._migrate_models_kind_column()
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_models_kind ON models(kind)"
        )
        await self._migrate_models_description_column()
        # Media routing reads capabilities from models.metadata_json. Legacy
        # rows predate that declaration, so give each LLM a conservative,
        # persisted default (text-only unless its provider/model family is a
        # well-known multimodal one). Idempotent and tiny: this table contains
        # configured models, not conversation/event data.
        await self._migrate_model_input_modalities_metadata()
        await self._migrate_legacy_tables_to_sessions()
        await self._migrate_peer_networks_join_type()
        # v0.14+: fold any legacy ``session_bindings`` rows (which used
        # to carry per-session framework lock + pin) into the new
        # ``pinned_sessions`` table that stores only the pin. Idempotent.
        await self._migrate_session_bindings_to_pinned_sessions()

        # v0.14.2: per-workflow concurrency cap. NULL = unlimited (new
        # default — overlapping runs all execute). Old DBs predate the
        # column; ``CREATE TABLE IF NOT EXISTS`` doesn't add it, so we
        # ALTER once here.
        cursor = await self._conn.execute("PRAGMA table_info(workflow_tasks)")
        cols = {row[1] for row in await cursor.fetchall()}
        if "max_concurrent_runs" not in cols:
            await self._conn.execute(
                "ALTER TABLE workflow_tasks ADD COLUMN max_concurrent_runs INTEGER"
            )
            await self._conn.commit()

        await self._migrate_task_runs_session_id()
        await self._migrate_scheduled_tasks_model_column()
        await self._migrate_scheduled_tasks_timezone_column()
        await self._migrate_scheduled_tasks_execution_policy_column()
        await self._migrate_events_session_binding()
        await self._migrate_events_precondition()
        await self._migrate_events_execution_policy_column()
        await self._migrate_event_deliveries_reenqueue_count()
        await self._migrate_event_deliveries_lease()
        await self._migrate_events_breaker()
        await self._migrate_guarded_changes()

    async def _add_columns_if_missing(
        self, table: str, columns: dict[str, str],
    ) -> None:
        """Idempotent ``ALTER TABLE ADD COLUMN`` for each ``{name: coldef}`` not
        already present. ADD COLUMN only — NEVER a whole-table backfill UPDATE
        (that rewrites/locks the 2 GB event table; the exact outage lesson). New
        columns must work at their NULL/0 defaults with no backfill.

        Swallows SQLite's "duplicate column name" so a concurrent DDL race across
        the gateway + scheduler-MCP subprocess + a per-test connection (all
        pointing at the same file) can't crash a second connect."""
        assert self._conn is not None
        cursor = await self._conn.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in await cursor.fetchall()}
        for name, coldef in columns.items():
            if name in existing:
                continue
            try:
                await self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {name} {coldef}"
                )
                await self._conn.commit()
            except sqlite3.OperationalError as e:
                # Another connection won the DDL race between our PRAGMA read
                # and this ALTER — the column now exists, which is what we
                # wanted. Any other operational error is a real problem.
                if "duplicate column name" not in str(e).lower():
                    raise

    async def _migrate_event_deliveries_lease(self) -> None:
        """Claim-lease + heartbeat columns for at-least-once recovery at
        ~LEASE_TTL instead of the coarse 30-min stale-sweep age. All NULL on a
        legacy in-flight row, so ``reap_expired_event_leases`` (which only acts
        on ``claim_expires IS NOT NULL``) never touches pre-existing rows — the
        age-gated stale sweep / startup reap keep handling those. ADD COLUMN
        only; no backfill."""
        await self._add_columns_if_missing(
            "event_deliveries",
            {
                "claim_expires": "REAL",
                "worker_id": "TEXT",
                "worker_pid": "INTEGER",
                "last_heartbeat_at": "REAL",
            },
        )
        # Index the lease column HERE — the columns above now exist on BOTH a
        # fresh DB (from the CREATE TABLE) and a pre-existing one (just ALTERed
        # in). The base schema block deliberately does NOT index claim_expires,
        # because it runs before this migration and would crash boot on an old
        # DB ("no such column: claim_expires"). IF NOT EXISTS keeps it idempotent.
        assert self._conn is not None
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_evdel_lease "
            "ON event_deliveries(claim_expires)"
        )
        await self._conn.commit()

    async def _migrate_events_breaker(self) -> None:
        """Per-event circuit-breaker columns. Off by default (gated by
        ``OPENAGENT_EVENT_BREAKER_ENABLED``); every column works at its 0/NULL
        default with no backfill, so a fresh column changes nothing until the
        flag is turned on. ADD COLUMN only."""
        await self._add_columns_if_missing(
            "events",
            {
                "consecutive_failures": "INTEGER NOT NULL DEFAULT 0",
                "max_retries": "INTEGER",
                "breaker_tripped_at": "REAL",
                "last_failure_error": "TEXT",
            },
        )

    async def _migrate_guarded_changes(self) -> None:
        """Guarded auto-applied changes — snapshot + watch + auto-rollback +
        blocklist. When an autonomous fix (the self-remediation cycle) applies a
        config/template change, it records a row here; the scheduler's
        ``reap_guarded_changes`` watcher then measures the target's REAL delivery
        failure-rate after the change and, if it regressed, RESTORES the old
        value and marks the row ``rolled_back`` — which also blocklists that exact
        (target, field, new_value) so it is NEVER re-applied. This is what makes
        auto-apply safe: a fix that breaks something self-heals and does not
        repeat. Idle by default: the table is empty until a guarded change runs.

        CREATE TABLE + its indexes together (both reference only columns this
        statement creates — the claim_expires boot-crash lesson: never index a
        column in a block that runs before the column exists)."""
        assert self._conn is not None
        await self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS guarded_changes (
                id                 TEXT PRIMARY KEY,
                target_kind        TEXT NOT NULL,    -- 'event_field'
                target_id          TEXT NOT NULL,    -- e.g. the event id
                field              TEXT NOT NULL,    -- e.g. 'prompt_template'
                old_value          TEXT,             -- snapshot for rollback
                new_value          TEXT,
                metric_event_id    TEXT,             -- event whose fail-rate we watch
                applied_at         REAL NOT NULL,
                check_after        REAL NOT NULL,    -- watcher evaluates at/after this
                baseline_fail_rate REAL,             -- fail-rate in the window BEFORE apply
                baseline_n         INTEGER,
                status             TEXT NOT NULL DEFAULT 'pending', -- pending|confirmed|rolled_back
                after_fail_rate    REAL,
                after_n            INTEGER,
                resolved_at        REAL,
                note               TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_guarded_pending
                ON guarded_changes(status, check_after);
            CREATE INDEX IF NOT EXISTS idx_guarded_block
                ON guarded_changes(target_kind, target_id, field, status);
            """
        )
        await self._conn.commit()

    # ── Guarded-change watcher (deterministic auto-rollback) ────────────────
    # Columns of `events` that a guarded change may restore on rollback. A tight
    # whitelist keeps the rollback UPDATE from ever touching an unexpected field.
    _GUARDED_EVENT_FIELDS = frozenset(
        {"prompt_template", "model", "input_schema_json", "action_ref", "enabled"}
    )

    async def event_fail_rate(
        self, event_id: str, since: float, until: float | None = None
    ) -> tuple[float, int]:
        """(failure_rate, sample_count) for one event's deliveries in
        ``[since, until)``. A render error / broken template shows up as
        ``status='failed'`` rows, so a change that breaks delivery drives this up."""
        assert self._conn is not None
        until = time.time() if until is None else until
        cur = await self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM event_deliveries "
            "WHERE event_id = ? AND started_at >= ? AND started_at < ? "
            "GROUP BY status",
            (event_id, since, until),
        )
        rows = await cur.fetchall()
        total = sum(r["n"] for r in rows)
        failed = sum(r["n"] for r in rows if r["status"] == "failed")
        return (failed / total if total else 0.0), total

    async def is_guarded_change_blocked(
        self, *, target_kind: str, target_id: str, field: str, new_value: str
    ) -> bool:
        """True when this exact (target, field, new_value) was auto-rolled-back
        before — the blocklist that stops a broken auto-fix from repeating."""
        assert self._conn is not None
        cur = await self._conn.execute(
            "SELECT 1 FROM guarded_changes WHERE target_kind=? AND target_id=? "
            "AND field=? AND new_value=? AND status='rolled_back' LIMIT 1",
            (target_kind, target_id, field, new_value),
        )
        return (await cur.fetchone()) is not None

    async def _restore_guarded_target(self, ch: Any) -> bool:
        """Restore a guarded change's snapshot (rollback). Only ``event_field``
        targets, only whitelisted columns. Returns True if a row was restored."""
        assert self._conn is not None
        if ch["target_kind"] != "event_field":
            return False
        field = ch["field"]
        if field not in self._GUARDED_EVENT_FIELDS:
            return False
        # Field name is whitelist-checked above, so this f-string is safe.
        await self._conn.execute(
            f"UPDATE events SET {field}=?, consecutive_failures=0, "
            f"breaker_tripped_at=NULL, last_failure_error=NULL WHERE id=?",
            (ch["old_value"], ch["target_id"]),
        )
        await self._conn.commit()
        return True

    async def reap_guarded_changes(
        self,
        *,
        regression_margin: float = 0.15,
        min_after_samples: int = 3,
        give_up_after_seconds: float = 3600.0,
    ) -> int:
        """Deterministic watcher: resolve every pending guarded change whose
        ``check_after`` has passed. Measures the target's REAL delivery
        failure-rate since the change vs the recorded baseline:

        * regressed (after_rate > baseline + margin, with ≥ min_after_samples
          observed) → RESTORE the snapshot and mark ``rolled_back`` (which also
          blocklists that exact change);
        * stable with enough samples → ``confirmed``;
        * too few samples but past ``give_up_after_seconds`` → ``confirmed``
          (nothing ran → nothing broke); otherwise left pending for a later tick.

        Self-guarded and cheap; returns the number of changes resolved."""
        assert self._conn is not None
        now = time.time()
        cur = await self._conn.execute(
            "SELECT * FROM guarded_changes WHERE status='pending' AND check_after<=?",
            (now,),
        )
        pending = await cur.fetchall()
        resolved = 0
        for ch in pending:
            metric_ev = ch["metric_event_id"] or ch["target_id"]
            after_rate, after_n = await self.event_fail_rate(metric_ev, ch["applied_at"], now)
            base = ch["baseline_fail_rate"] or 0.0
            new_status: str | None = None
            note = ch["note"] or ""
            if after_n >= min_after_samples and after_rate > base + regression_margin:
                restored = await self._restore_guarded_target(ch)
                new_status = "rolled_back"
                note = (
                    f"auto-rollback: fail-rate {after_rate:.0%} (n={after_n}) vs "
                    f"baseline {base:.0%}; snapshot restored={restored}"
                )
            elif after_n >= min_after_samples:
                new_status = "confirmed"
                note = f"confirmed: fail-rate {after_rate:.0%} (n={after_n}) ≤ baseline {base:.0%}+margin"
            elif now - ch["applied_at"] >= give_up_after_seconds:
                new_status = "confirmed"
                note = f"confirmed (idle): only {after_n} deliveries observed, nothing regressed"
            if new_status is not None:
                await self._conn.execute(
                    "UPDATE guarded_changes SET status=?, after_fail_rate=?, "
                    "after_n=?, resolved_at=?, note=? WHERE id=?",
                    (new_status, after_rate, after_n, now, note[:500], ch["id"]),
                )
                await self._conn.commit()
                resolved += 1
                try:
                    from src.core.logging import elog
                    elog(
                        "guarded_change." + new_status,
                        target=f"{ch['target_kind']}:{ch['target_id']}.{ch['field']}",
                        after_fail_rate=round(after_rate, 3),
                        after_n=after_n,
                    )
                except Exception:  # noqa: BLE001 — logging must never break the sweep
                    pass
        return resolved

    async def _migrate_event_deliveries_reenqueue_count(self) -> None:
        """At-least-once event delivery re-enqueues an orphaned (claimed but
        never-completed) delivery on startup instead of dropping it as
        ``failed``. ``reenqueue_count`` bounds that replay so a delivery that
        keeps orphaning can't churn forever. Old DBs predate the column; add
        it idempotently (DEFAULT 0 = never re-enqueued)."""
        assert self._conn is not None
        cursor = await self._conn.execute("PRAGMA table_info(event_deliveries)")
        cols = {row[1] for row in await cursor.fetchall()}
        if "reenqueue_count" not in cols:
            await self._conn.execute(
                "ALTER TABLE event_deliveries "
                "ADD COLUMN reenqueue_count INTEGER NOT NULL DEFAULT 0"
            )
            await self._conn.commit()

    async def _migrate_scheduled_tasks_model_column(self) -> None:
        """A scheduled task can now pin an optional per-task model (a
        runtime_id). The firing runs its child session on that model instead
        of the agent's default/router pick. Old DBs predate the column; add
        it idempotently (NULL = use the default model)."""
        assert self._conn is not None
        cursor = await self._conn.execute("PRAGMA table_info(scheduled_tasks)")
        cols = {row[1] for row in await cursor.fetchall()}
        if "model" not in cols:
            await self._conn.execute(
                "ALTER TABLE scheduled_tasks ADD COLUMN model TEXT"
            )
            await self._conn.commit()

    async def _migrate_scheduled_tasks_timezone_column(self) -> None:
        """A scheduled task can now name the IANA timezone its cron is read
        in. Old DBs predate the column; add it idempotently.

        Deliberately no backfill. Every row already in a production DB holds
        an expression whose hour the operator converted to UTC by hand;
        stamping a real timezone on it would re-read that hour and shift the
        task. NULL keeps the UTC behaviour those rows were written against,
        and an operator opts a task in by editing it."""
        assert self._conn is not None
        cursor = await self._conn.execute("PRAGMA table_info(scheduled_tasks)")
        cols = {row[1] for row in await cursor.fetchall()}
        if "timezone" not in cols:
            await self._conn.execute(
                "ALTER TABLE scheduled_tasks ADD COLUMN timezone TEXT"
            )
            await self._conn.commit()

    async def _migrate_scheduled_tasks_execution_policy_column(self) -> None:
        """Add the generic unattended-run envelope without changing old tasks."""
        assert self._conn is not None
        cursor = await self._conn.execute("PRAGMA table_info(scheduled_tasks)")
        cols = {row[1] for row in await cursor.fetchall()}
        if "execution_policy_json" not in cols:
            await self._conn.execute(
                "ALTER TABLE scheduled_tasks ADD COLUMN execution_policy_json TEXT"
            )
            await self._conn.commit()

    async def _migrate_events_precondition(self) -> None:
        """An event can declare a cheap check that runs BEFORE the delivery does.

        Every prompt delivery is a paid model run, and a queued one runs against
        state that has since moved on. On a support webhook that is the common
        case, not the edge: measured 2026-08-07, ~22% of deliveries reached the
        model only to read the thread and conclude someone had already answered.
        The idempotency check itself was correct — it was just being performed
        by the most expensive component available.

        ``precondition_json`` lets the event state that check declaratively, so
        the dispatcher can settle it with one HTTP call. NULL keeps the old
        behaviour (always run).
        """
        assert self._conn is not None
        cursor = await self._conn.execute("PRAGMA table_info(events)")
        cols = {row[1] for row in await cursor.fetchall()}
        if "precondition_json" not in cols:
            await self._conn.execute(
                "ALTER TABLE events ADD COLUMN precondition_json TEXT"
            )
            await self._conn.commit()

    async def _migrate_events_execution_policy_column(self) -> None:
        """Add per-event resource/capability bounds without changing old events."""
        assert self._conn is not None
        cursor = await self._conn.execute("PRAGMA table_info(events)")
        cols = {row[1] for row in await cursor.fetchall()}
        if "execution_policy_json" not in cols:
            await self._conn.execute(
                "ALTER TABLE events ADD COLUMN execution_policy_json TEXT"
            )
            await self._conn.commit()

    async def _migrate_events_session_binding(self) -> None:
        """Prompt events can optionally bind a payload field (usually an
        external object id) to one durable OpenAgent event-run session.

        Old DBs predate both columns and the lookup table; add them
        idempotently. ``session_binding_enabled=0`` preserves the old
        one-delivery/one-session behaviour.
        """
        assert self._conn is not None
        cursor = await self._conn.execute("PRAGMA table_info(events)")
        cols = {row[1] for row in await cursor.fetchall()}
        if "session_binding_enabled" not in cols:
            await self._conn.execute(
                "ALTER TABLE events ADD COLUMN "
                "session_binding_enabled INTEGER NOT NULL DEFAULT 0"
            )
        if "session_binding_path" not in cols:
            await self._conn.execute(
                "ALTER TABLE events ADD COLUMN session_binding_path TEXT"
            )
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS event_session_bindings (
                event_id     TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
                binding_key  TEXT NOT NULL,
                session_id   TEXT NOT NULL,
                created_at   REAL NOT NULL,
                updated_at   REAL NOT NULL,
                PRIMARY KEY (event_id, binding_key)
            )
            """
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ev_session_bindings_session "
            "ON event_session_bindings(session_id)"
        )
        await self._conn.commit()

    async def _migrate_task_runs_session_id(self) -> None:
        """v0.15: a scheduled-task firing now runs as a durable child
        ``sessions`` row (``scheduler:{task_id}:{run_id}``). ``task_runs``
        records which one so the app can open a past firing as a full chat
        session. Old DBs predate the column; add it idempotently."""
        assert self._conn is not None
        cursor = await self._conn.execute("PRAGMA table_info(task_runs)")
        cols = {row[1] for row in await cursor.fetchall()}
        if "session_id" not in cols:
            await self._conn.execute(
                "ALTER TABLE task_runs ADD COLUMN session_id TEXT"
            )
            await self._conn.commit()

    async def _migrate_legacy_tables_to_sessions(self) -> None:
        """One-time: fold sdk_sessions + chat_sessions + chat_session_runs
        into the canonical ``sessions`` table, then drop them.

        Idempotent: probes for the old tables first; a fresh install
        (where ``sessions`` is the only session table from the start)
        skips all work.
        """
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
            "('sdk_sessions','chat_sessions','chat_session_runs')"
        )
        legacy_tables = {r[0] for r in await cursor.fetchall()}
        if not legacy_tables:
            return

        now = time.time()

        # 1. Upsert sdk_sessions rows into sessions, carrying the
        #    SDK session id in the metadata JSON column.
        if "sdk_sessions" in legacy_tables:
            cursor = await self._conn.execute(
                "SELECT session_id, sdk_session_id, provider, updated_at "
                "FROM sdk_sessions"
            )
            for row in await cursor.fetchall():
                sid = row[0]
                sdk_sid = row[1]
                # Legacy ``sdk_sessions`` was the retired claude-cli resume
                # store; preserve the historical provider label verbatim.
                provider = row[2] or "claude-cli"
                ts = row[3] or now
                meta = json.dumps({
                    "sdk_session_id": sdk_sid,
                    "provider": provider,
                })
                await self._conn.execute(
                    "INSERT OR IGNORE INTO sessions "
                    "(session_id, session_type, user_id, metadata, "
                    " created_at, updated_at) "
                    "VALUES (?, 'agent', 'openagent', ?, ?, ?)",
                    (sid, meta, ts, ts),
                )
                # If a row already existed (created by the session store on
                # an earlier provider run), merge the SDK metadata in.
                await self._conn.execute(
                    "UPDATE sessions SET metadata = ? "
                    "WHERE session_id = ? AND (metadata IS NULL OR metadata = '' "
                    " OR json_extract(metadata, '$.sdk_session_id') IS NULL)",
                    (meta, sid),
                )
            await self._conn.commit()

        # 2. Upsert chat_sessions rows (metadata like title, model, framework).
        if "chat_sessions" in legacy_tables:
            col_list = {"session_id", "client_id", "title", "model",
                         "framework", "created_at", "updated_at", "last_active_at"}
            cursor = await self._conn.execute(
                "SELECT session_id, client_id, title, model, framework, "
                "created_at, updated_at, last_active_at FROM chat_sessions"
            )
            for row in await cursor.fetchall():
                r = dict(row)
                sid = r.get("session_id")
                if not sid:
                    continue
                # Merge extra metadata into the JSON column.
                extra = {k: r[k] for k in ("client_id", "title", "model")
                         if r.get(k)}
                if extra:
                    extra["framework"] = r.get("framework", "")
                # Read existing metadata, merge.
                cursor2 = await self._conn.execute(
                    "SELECT metadata FROM sessions WHERE session_id = ?", (sid,)
                )
                existing = await cursor2.fetchone()
                try:
                    meta = json.loads(existing[0] or "{}") if existing else {}
                except (TypeError, ValueError):
                    meta = {}
                meta.update(extra)
                ts = r.get("updated_at") or r.get("created_at") or now
                await self._conn.execute(
                    "INSERT OR IGNORE INTO sessions "
                    "(session_id, session_type, user_id, metadata, "
                    " created_at, updated_at) "
                    "VALUES (?, 'agent', 'openagent', ?, ?, ?)",
                    (sid, json.dumps(meta), r.get("created_at", ts), ts),
                )
                if existing:
                    await self._conn.execute(
                        "UPDATE sessions SET metadata = ?, updated_at = MAX(updated_at, ?) "
                        "WHERE session_id = ?",
                        (json.dumps(meta), ts, sid),
                    )
            await self._conn.commit()

        # 3. Drop the legacy tables.
        for table in ("chat_session_runs", "chat_sessions", "sdk_sessions"):
            if table in legacy_tables:
                try:
                    await self._conn.execute(f"DROP TABLE IF EXISTS {table}")
                except Exception:
                    pass
        # Drop the legacy agno_memories table — agentic memory is disabled (vault MCP only).
        try:
            await self._conn.execute("DROP TABLE IF EXISTS agno_memories")
        except Exception:
            pass
        await self._conn.commit()

    async def _migrate_models_kind_column(self) -> None:
        """Add ``models.kind`` (idempotent) and lift TTS/STT settings out
        of ``providers.metadata_json`` into model rows.
        """
        assert self._conn is not None
        cursor = await self._conn.execute("PRAGMA table_info(models)")
        cols = {r[1] for r in await cursor.fetchall()}
        if "kind" not in cols:
            await self._conn.execute(
                "ALTER TABLE models ADD COLUMN kind TEXT NOT NULL "
                "DEFAULT 'llm' CHECK (kind IN ('llm','tts','stt'))"
            )

        # For each provider row whose ``providers.kind`` was 'tts'/'stt',
        # promote the model_id+voice_id stored in metadata_json into a
        # proper model row, then reset the provider's kind to 'llm' so
        # the discriminator lives only on the model side.
        cursor = await self._conn.execute(
            "SELECT id, name, metadata_json, kind FROM providers "
            "WHERE kind IN ('tts','stt')"
        )
        rows = await cursor.fetchall()
        if not rows:
            await self._conn.commit()
            return

        now = time.time()
        for row in rows:
            audio_provider_id, vendor, meta_raw, audio_kind = row
            try:
                meta = json.loads(meta_raw or "{}") if isinstance(meta_raw, str) else {}
            except ValueError:
                meta = {}
            model_id = (meta.get("model_id") or "").strip()
            if not model_id:
                # Nothing actionable on this row — drop the kind tag and move on.
                await self._conn.execute(
                    "UPDATE providers SET kind='llm', updated_at=? WHERE id=?",
                    (now, audio_provider_id),
                )
                continue

            # Prefer attaching the audio model to a sibling LLM
            # provider for the same vendor (so the user keeps one
            # api_key per vendor). Fall back to the audio provider's
            # own row when no sibling exists.
            cursor2 = await self._conn.execute(
                "SELECT id FROM providers WHERE name=? AND kind='llm' "
                "ORDER BY id LIMIT 1",
                (vendor,),
            )
            sibling = await cursor2.fetchone()
            target_provider_id = sibling[0] if sibling else audio_provider_id
            model_meta = {k: v for k, v in meta.items() if k != "model_id"}
            await self._conn.execute(
                """
                INSERT OR IGNORE INTO models
                    (provider_id, model, kind, enabled, metadata_json,
                     created_at, updated_at)
                VALUES (?, ?, ?, 1, ?, ?, ?)
                """,
                (target_provider_id, model_id, audio_kind,
                 json.dumps(model_meta), now, now),
            )
            # Provider row now exists only to hold the api_key (if it
            # was the audio-only row) — flip its kind to 'llm' so
            # nothing in the codebase keeps reading the old discriminator.
            await self._conn.execute(
                "UPDATE providers SET kind='llm', metadata_json='{}', "
                "updated_at=? WHERE id=?",
                (now, audio_provider_id),
            )
        await self._conn.commit()

    async def _migrate_legacy_elevenlabs_to_litellm(self) -> None:
        """Idempotent UPDATE of pre-LiteLLM TTS rows. Historically targeted
        ``framework='litellm'``; that value has since collapsed into
        ``api-based`` (see :meth:`_migrate_legacy_framework_names_to_api_based`),
        so we go straight to the canonical value here.
        """
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM providers WHERE framework='elevenlabs'"
        )
        row = await cursor.fetchone()
        if not row or not row[0]:
            return
        # Some rows might collide with an existing (name='elevenlabs',
        # framework='api-based') under the UNIQUE(name, framework)
        # constraint. Pick a non-colliding name like 'elevenlabs-legacy'
        # in that case so we never lose data.
        await self._conn.execute(
            """
            UPDATE providers
               SET framework='api-based',
                   name=CASE
                     WHEN EXISTS (
                       SELECT 1 FROM providers p2
                       WHERE p2.framework='api-based' AND p2.name=providers.name
                     ) THEN providers.name || '-legacy'
                     ELSE providers.name
                   END,
                   updated_at=?
             WHERE framework='elevenlabs'
            """,
            (time.time(),),
        )
        await self._conn.commit()

    async def _migrate_legacy_framework_names_to_api_based(self) -> None:
        """Collapse legacy framework values into the canonical ``api-based``.

        Pre-v0.14: providers.framework could be ``agno`` (legacy LLM via
        the native runtime Agent) or ``litellm`` (TTS/STT via litellm).
        v0.14+: ``api-based`` is the single execution engine and ``kind``
        discriminates TTS/STT from LLM. Only ``agno`` and ``litellm``
        collapse here.

        Idempotent: no-op when no legacy rows remain. Handles
        UNIQUE(name, framework) collisions by suffixing colliding row
        names with ``-legacy`` so no data is lost.
        """
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM providers WHERE framework IN ('agno','litellm')"
        )
        row = await cursor.fetchone()
        if not row or not row[0]:
            return
        await self._conn.execute(
            """
            UPDATE providers
               SET framework='api-based',
                   name=CASE
                     WHEN EXISTS (
                       SELECT 1 FROM providers p2
                       WHERE p2.framework='api-based' AND p2.name=providers.name
                     ) THEN providers.name || '-legacy'
                     ELSE providers.name
                   END,
                   updated_at=?
             WHERE framework IN ('agno','litellm')
            """,
            (time.time(),),
        )
        await self._conn.commit()

    async def _migrate_drop_subscription_cli_providers(self) -> None:
        """Delete provider rows on the retired ``claude-cli`` / ``codex-cli``
        frameworks.

        The subscription-CLI adapters were removed. A pre-existing DB may
        still hold ``providers`` rows on those frameworks (added when the
        adapters shipped); leaving them would let the catalog surface a
        framework the runtime can no longer dispatch. Their ``models`` rows
        cascade-delete via the FK ``ON DELETE CASCADE``.

        Idempotent: no-op when no such rows remain.
        """
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM providers "
            "WHERE framework IN ('claude-cli','codex-cli')"
        )
        row = await cursor.fetchone()
        if not row or not row[0]:
            return
        await self._conn.execute(
            "DELETE FROM providers WHERE framework IN ('claude-cli','codex-cli')"
        )
        await self._conn.commit()

    async def _migrate_session_bindings_to_pinned_sessions(self) -> None:
        """Fold legacy ``session_bindings`` rows into ``pinned_sessions``.

        Pre-v0.14: ``session_bindings`` carried a per-session framework
        lock AND an optional ``runtime_id`` pin. With history unified
        in the canonical ``sessions`` table across every framework, the lock
        is no longer needed — only the pin survives.

        Idempotent: probes for the old table; copies any rows with a
        non-null runtime_id into ``pinned_sessions`` (preserving the
        pin), then drops the legacy table. Fresh installs (where
        SCHEMA_SQL already created ``pinned_sessions``) skip everything.
        """
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='session_bindings'"
        )
        row = await cursor.fetchone()
        if not row or not row[0]:
            return
        cursor = await self._conn.execute(
            "SELECT session_id, runtime_id, bound_at FROM session_bindings "
            "WHERE runtime_id IS NOT NULL AND runtime_id != ''"
        )
        rows = await cursor.fetchall()
        for sid, runtime_id, bound_at in rows:
            await self._conn.execute(
                "INSERT OR IGNORE INTO pinned_sessions "
                "(session_id, runtime_id, pinned_at) VALUES (?, ?, ?)",
                (sid, runtime_id, bound_at or time.time()),
            )
        await self._conn.execute("DROP TABLE session_bindings")
        await self._conn.commit()

    async def _migrate_models_description_column(self) -> None:
        """Add ``description`` to ``models`` (idempotent)."""
        assert self._conn is not None
        cursor = await self._conn.execute("PRAGMA table_info(models)")
        cols = {r[1] for r in await cursor.fetchall()}
        if "description" not in cols:
            await self._conn.execute(
                "ALTER TABLE models ADD COLUMN description TEXT"
            )
            await self._conn.commit()

    async def _migrate_model_input_modalities_metadata(self) -> None:
        """Backfill/repair ``metadata_json.input_modalities`` for LLM rows.

        The field lives in the existing JSON column, so this is additive and
        requires no schema rewrite. Explicit valid declarations are preserved;
        missing or malformed legacy declarations receive conservative defaults.
        ``updated_at`` is intentionally left untouched so a normal boot does not
        masquerade as a user catalog edit to the hot-reload watcher.
        """
        assert self._conn is not None
        from src.models.media_capabilities import normalize_model_metadata

        cursor = await self._conn.execute(
            "SELECT m.id, m.model, m.metadata_json, p.name AS provider_name "
            "FROM models m JOIN providers p ON p.id = m.provider_id "
            "WHERE m.kind = 'llm'"
        )
        for row in await cursor.fetchall():
            raw = row["metadata_json"] or "{}"
            try:
                metadata = json.loads(raw) if isinstance(raw, str) else {}
            except (TypeError, ValueError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            normalized = normalize_model_metadata(
                metadata,
                provider=str(row["provider_name"] or ""),
                model=str(row["model"] or ""),
                strict=False,
            )
            encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
            try:
                existing = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            except (TypeError, ValueError):
                existing = ""
            if encoded == existing:
                continue
            await self._conn.execute(
                "UPDATE models SET metadata_json = ? WHERE id = ?",
                (encoded, int(row["id"])),
            )

    async def _migrate_providers_kind_column(self) -> None:
        """Add ``kind`` to ``providers`` and lift the ``framework`` CHECK.

        Idempotent: detects the old CHECK by inspecting ``sqlite_master``
        and the missing column via ``PRAGMA table_info``. Only does work
        on legacy DBs; fresh installs already get the new shape from
        ``SCHEMA_SQL`` and exit early.
        """
        assert self._conn is not None

        cursor = await self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='providers'"
        )
        row = await cursor.fetchone()
        if not row or not row[0]:
            return  # No providers table — fresh install hasn't created it yet
        table_sql = row[0]
        # The legacy CHECK literal as it appears in SCHEMA_SQL prior to this
        # migration. Substring match is fine because SQLite stores the
        # CREATE TABLE text verbatim.
        has_old_check = "framework IN ('agno','claude-cli')" in table_sql

        cursor = await self._conn.execute("PRAGMA table_info(providers)")
        cols = {r[1] for r in await cursor.fetchall()}
        has_kind = "kind" in cols

        if has_kind and not has_old_check:
            return  # already migrated

        # Close any pending implicit tx so PRAGMA foreign_keys takes effect.
        await self._conn.commit()
        await self._conn.execute("PRAGMA foreign_keys = OFF")
        try:
            await self._conn.execute(
                """
                CREATE TABLE providers_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    api_key TEXT,
                    base_url TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    kind TEXT NOT NULL DEFAULT 'llm'
                        CHECK (kind IN ('llm','tts','stt')),
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(name, framework)
                )
                """
            )
            if has_kind:
                await self._conn.execute(
                    "INSERT INTO providers_new "
                    "(id, name, framework, api_key, base_url, enabled, "
                    " metadata_json, kind, created_at, updated_at) "
                    "SELECT id, name, framework, api_key, base_url, enabled, "
                    " metadata_json, kind, created_at, updated_at FROM providers"
                )
            else:
                await self._conn.execute(
                    "INSERT INTO providers_new "
                    "(id, name, framework, api_key, base_url, enabled, "
                    " metadata_json, kind, created_at, updated_at) "
                    "SELECT id, name, framework, api_key, base_url, enabled, "
                    " metadata_json, 'llm', created_at, updated_at FROM providers"
                )
            await self._conn.execute("DROP TABLE providers")
            await self._conn.execute("ALTER TABLE providers_new RENAME TO providers")
            await self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_providers_enabled ON providers(enabled)"
            )
            await self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_providers_updated ON providers(updated_at)"
            )
            await self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_providers_name ON providers(name)"
            )
            await self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_providers_kind ON providers(kind)"
            )
            await self._conn.commit()
        except Exception:
            await self._conn.rollback()
            raise
        finally:
            await self._conn.execute("PRAGMA foreign_keys = ON")

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
        from src.memory.schedule import (
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

    async def _project_operational_session(self, session_id: str) -> None:
        """Best-effort same-transaction v2 projection for reviewed writers.

        The legacy change trigger is outside this savepoint, so a projection
        bug rolls back only v2 mutations while the durable pending journal row
        survives with the accepted legacy write for later reconciliation.
        """

        if not session_id or self._conn is None:
            return
        savepoint = f"operational_projection_{uuid.uuid4().hex}"
        await self._conn.execute(f"SAVEPOINT {savepoint}")
        try:
            from src.memory.operational.repository import (
                project_legacy_session_async,
            )

            await project_legacy_session_async(self._conn, session_id)
            await self._conn.execute(f"RELEASE SAVEPOINT {savepoint}")
        except Exception as exc:
            await self._conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            await self._conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            logger.warning(
                "operational session projection deferred for %s: %s",
                session_id,
                exc,
            )

    async def set_operational_storage_phase(self, phase: str):
        """Explicitly promote or roll back normalized runtime reads.

        Promotion performs full parity verification in the caller's
        transaction.  The default boot path never invokes it automatically;
        beta operators/tests must opt in after backfill is complete.
        """

        conn = await self._ensure_connected()
        from src import __version__
        from src.memory.operational.phase import (
            transition_storage_phase_atomic_async,
        )

        return await transition_storage_phase_atomic_async(
            conn,
            phase,
            writer_version=__version__,
        )

    async def _write_with_retry(self, do_write, *, attempts: int = 3):
        """Run an idempotent single-row write, retrying on "database is locked".

        The runtime session store commits a big ``runs`` blob per step and can
        hog the single SQLite WAL writer during a rate-limit storm; a finalizer /
        heartbeat / lease-reaper UPDATE that loses that race raises
        ``sqlite3.OperationalError: database is locked``. Before this, a locked
        reaper tick was silently swallowed and the row stayed ``running``
        forever. These writes are idempotent last-writer-wins single-row UPDATEs,
        so a bounded retry (3× with 50–200 ms jittered backoff) simply lands the
        write once the writer frees up. A non-lock OperationalError, or exhausting
        the budget, re-raises to the caller (who logs it)."""
        delay = 0.05
        for i in range(attempts):
            try:
                return await do_write()
            except sqlite3.OperationalError as e:
                if "locked" not in str(e).lower() or i == attempts - 1:
                    raise
                await asyncio.sleep(delay + random.uniform(0.0, 0.05))
                delay = min(delay * 2, 0.2)

    # ── Scheduled Tasks ──

    async def add_task(
        self,
        name: str,
        cron_expression: str,
        prompt: str,
        next_run: float | None = None,
        model: str | None = None,
        timezone: str | None = None,
        execution_policy: Any = None,
    ) -> str:
        """``timezone`` is an IANA name the cron is read in; None = UTC (the
        pre-timezone behaviour). Validated here so a typo can't reach the
        scheduler loop, where it would surface only as a task that mysteriously
        stopped advancing."""
        from src.memory.schedule import validate_timezone

        validate_timezone(timezone)
        from src.core.execution_policy import encode_execution_policy

        execution_policy_json = encode_execution_policy(execution_policy)
        conn = await self._ensure_connected()
        task_id = str(uuid.uuid4())
        now = time.time()
        await conn.execute(
            "INSERT INTO scheduled_tasks (id, name, cron_expression, prompt, enabled, next_run, model, timezone, execution_policy_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)",
            (task_id, name, cron_expression, prompt, next_run or now, model or None,
             (timezone or None), execution_policy_json, now, now),
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
        allowed = {
            "name", "cron_expression", "prompt", "enabled", "last_run",
            "next_run", "model", "timezone", "execution_policy_json",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        if "timezone" in updates:
            from src.memory.schedule import validate_timezone

            # Reject at the write. A bad zone that lands in the row makes
            # every next_run recompute raise inside the scheduler tick, which
            # logs but leaves next_run in the past — the task then re-fires
            # every 30 s instead of failing visibly.
            validate_timezone(updates["timezone"])
            updates["timezone"] = updates["timezone"] or None
        if "execution_policy_json" in updates:
            from src.core.execution_policy import encode_execution_policy

            updates["execution_policy_json"] = encode_execution_policy(
                updates["execution_policy_json"]
            )
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

    # ── Task Runs (scheduled-task execution history) ──
    #
    # Mirrors the Workflow Runs API below: a row is opened ``running`` by
    # the Scheduler when a task fires and flipped to ``success`` /
    # ``failed`` (with an output/error preview + ``finished_at``) when the
    # agent turn returns.

    @staticmethod
    def _row_to_task_run(row: aiosqlite.Row) -> dict:
        return dict(row)

    async def add_task_run(
        self,
        *,
        task_id: str,
        trigger: str = "schedule",
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        conn = await self._ensure_connected()
        rid = run_id or str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO task_runs "
            "(id, task_id, trigger, status, started_at, session_id) "
            "VALUES (?, ?, ?, 'running', ?, ?)",
            (rid, task_id, trigger, time.time(), session_id),
        )
        await conn.commit()
        return rid

    async def update_task_run(self, run_id: str, **kwargs: Any) -> None:
        """Partial update. Only ``status`` / ``finished_at`` / ``output`` /
        ``error`` are writable.

        Same cancellation invariant as ``update_workflow_run``: a firing
        flagged ``status='cancelling'`` may only move to ``cancelled``, so a
        natural finalize (``success`` / ``failed``) that races a "completely
        stop" request is suppressed and the orphan sweep / cancel handler
        records ``cancelled``.
        """
        allowed = {"status", "finished_at", "output", "error", "session_id"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        conn = await self._ensure_connected()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        where = "WHERE id = ?"
        if "status" in updates and updates["status"] != "cancelled":
            where += " AND status != 'cancelling'"
        await conn.execute(
            f"UPDATE task_runs SET {set_clause} {where}",
            list(updates.values()) + [run_id],
        )
        await conn.commit()

    async def get_task_run(self, run_id: str) -> dict | None:
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM task_runs WHERE id = ?", (run_id,),
        )
        row = await cursor.fetchone()
        return self._row_to_task_run(row) if row else None

    async def list_task_runs(
        self,
        task_id: str,
        *,
        limit: int = 20,
        status: str | None = None,
    ) -> list[dict]:
        conn = await self._ensure_connected()
        clauses = ["task_id = ?"]
        params: list[Any] = [task_id]
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        params.append(int(limit))
        cursor = await conn.execute(
            f"SELECT * FROM task_runs WHERE {' AND '.join(clauses)} "
            "ORDER BY started_at DESC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_task_run(r) for r in rows]

    async def get_task_runs_by_status(
        self, status: str, *, limit: int = 500,
    ) -> list[dict]:
        """Every ``task_runs`` row in ``status``, across all scheduled
        tasks. Powers the scheduler's cancellation drain, which scans for
        rows the scheduler MCP flagged ``cancelling`` and stops the
        in-flight firing that owns each one. Backed by ``idx_taskruns_status``
        so the scan stays cheap even when only a handful of rows ever match."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM task_runs WHERE status = ? "
            "ORDER BY started_at ASC LIMIT ?",
            (status, int(limit)),
        )
        rows = await cursor.fetchall()
        return [self._row_to_task_run(r) for r in rows]

    async def reap_orphan_task_runs(self) -> int:
        """Mark every ``task_runs`` row still ``running`` as ``failed`` —
        the scheduled-task analogue of ``reap_orphan_workflow_runs``. A
        ``running`` row that survives a process restart is a zombie: the
        firing that owned it is gone and there is no resume path, so the
        cosmetic "running" badge would otherwise never clear. Called from
        ``AgentServer.start()``. Returns the number of rows reaped."""
        conn = await self._ensure_connected()
        now = time.time()
        cursor = await conn.execute(
            "UPDATE task_runs "
            "SET status='failed', finished_at=?, "
            "    error=COALESCE(error, '') || "
            "          CASE WHEN error IS NULL OR error='' THEN '' ELSE ' | ' END || "
            "          'reaped: orphan from prior process' "
            "WHERE status='running'",
            (now,),
        )
        reaped = cursor.rowcount or 0
        # A firing flagged 'cancelling' that the prior process never finalized
        # (crash between the MCP flag and the scheduler's drain) is also an
        # orphan — finalize it as 'cancelled' (the requested outcome), kept
        # distinct from the 'failed' reap above.
        cancel_cursor = await conn.execute(
            "UPDATE task_runs "
            "SET status='cancelled', finished_at=?, "
            "    error=COALESCE(error, '') || "
            "          CASE WHEN error IS NULL OR error='' THEN '' ELSE ' | ' END || "
            "          'reaped: stop left pending by prior process' "
            "WHERE status='cancelling'",
            (now,),
        )
        await conn.commit()
        return reaped + (cancel_cursor.rowcount or 0)

    async def requeue_interrupted_task_runs(self) -> list[str]:
        """Re-enqueue the tasks whose firing a process restart killed.

        ``reap_orphan_task_runs`` settles the zombie row so the badge stops
        spinning, and treats that as the whole job — the docstring calls it
        cosmetic. It is not: a firing that died under a restart never did its
        work. For an hourly task the next tick covers it; for a weekly one
        (``0 9 * * 1``) the week's output simply does not exist, and the only
        trace is a ``failed`` row nobody reads. Seen on 2026-08-31: a WAL-pin
        restart killed ``esound-manager-review`` mid-run and the Monday review
        was lost until someone went looking for it.

        So an interrupted run goes back in the queue, through the same
        ``task_run_requests`` path a manual "run now" uses.

        Two guards, both about not making it worse:

        * **once per interruption.** If the run BEFORE the reaped one was
          itself a reaped orphan, we already retried and got killed again —
          the likeliest reason is that this task is what takes the process
          down. Retrying it forever would turn a crash into a crash loop, so
          it is left alone for a human.
        * **no duplicate requests.** A task that already has an unclaimed
          request is skipped; the pending one will fire it.

        Call AFTER ``reap_orphan_task_runs``, which is what stamps the marker
        this reads. Returns the task ids re-enqueued.
        """
        conn = await self._ensure_connected()
        marker = "reaped: orphan from prior process"
        cursor = await conn.execute(
            "SELECT r.id, r.task_id FROM task_runs AS r "
            "JOIN scheduled_tasks AS t ON t.id = r.task_id "
            "WHERE r.status = 'failed' AND r.error LIKE ? AND t.enabled = 1",
            (f"%{marker}%",),
        )
        candidates = [(row[0], row[1]) for row in await cursor.fetchall()]

        requeued: list[str] = []
        for run_id, task_id in candidates:
            pending = await conn.execute(
                "SELECT 1 FROM task_run_requests "
                "WHERE task_id = ? AND claimed_at IS NULL LIMIT 1",
                (task_id,),
            )
            if await pending.fetchone():
                continue

            # The run immediately before this one, for the same task.
            prev = await conn.execute(
                "SELECT error FROM task_runs "
                "WHERE task_id = ? AND id <> ? "
                "ORDER BY started_at DESC LIMIT 1",
                (task_id, run_id),
            )
            prev_row = await prev.fetchone()
            if prev_row and marker in (prev_row[0] or ""):
                continue

            if task_id not in requeued:
                await self.enqueue_task_run_request(
                    task_id=task_id, trigger="restart-requeue",
                )
                requeued.append(task_id)

        return requeued

    async def prune_task_runs(self, task_id: str, *, keep_last: int = 50) -> int:
        """Delete all but the most recent ``keep_last`` runs for a task.
        Returns the number of rows removed."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "DELETE FROM task_runs WHERE id IN ("
            "  SELECT id FROM task_runs WHERE task_id = ? "
            "  ORDER BY started_at DESC LIMIT -1 OFFSET ?"
            ")",
            (task_id, int(keep_last)),
        )
        await conn.commit()
        return cursor.rowcount or 0

    async def flag_task_runs_cancelling(self, task_id: str) -> list[str]:
        """Flag every currently-``running`` firing of a task as ``cancelling``
        and return the run ids flagged.

        This is the same cross-process "completely stop" hand-off the
        scheduler MCP's ``stop_scheduled_task`` writes: the scheduler's
        cancellation drain turns the flag into a real hard stop within ~2s
        (cancelling the agent turn) and finalizes the row as ``cancelled``.
        Idempotent — the ``status='running'`` guard on the UPDATE avoids
        clobbering a firing that finished between the SELECT and here, so a
        double "stop" can't resurrect a settled run."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT id FROM task_runs WHERE task_id = ? AND status = 'running'",
            (task_id,),
        )
        ids = [r["id"] for r in await cursor.fetchall()]
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        await conn.execute(
            f"UPDATE task_runs SET status = 'cancelling' "
            f"WHERE id IN ({placeholders}) AND status = 'running'",
            ids,
        )
        await conn.commit()
        return ids

    async def running_task_ids(self) -> set[str]:
        """Set of task ids that have a firing in flight (``running`` or
        ``cancelling``). Lets the dashboard light up a "running" badge and
        offer a Stop control without an N+1 per-task run query."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT DISTINCT task_id FROM task_runs "
            "WHERE status IN ('running', 'cancelling')"
        )
        return {r["task_id"] for r in await cursor.fetchall()}

    # ── Task Run Requests (on-demand "run now" hand-off) ──
    #
    # Mirrors ``workflow_run_requests``: an out-of-process caller (the
    # scheduler MCP subprocess) can't reach the in-process Scheduler, so it
    # enqueues a row here and the main process claims + fires it. The atomic
    # claim guards against a request firing twice if two scheduler loops
    # overlap.

    async def enqueue_task_run_request(
        self, *, task_id: str, trigger: str = "manual",
    ) -> str:
        conn = await self._ensure_connected()
        req_id = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO task_run_requests "
            "(id, task_id, trigger, created_at) VALUES (?, ?, ?, ?)",
            (req_id, task_id, trigger, time.time()),
        )
        await conn.commit()
        return req_id

    async def claim_pending_task_requests(self, *, limit: int = 20) -> list[dict]:
        """Atomically claim up to ``limit`` unclaimed run-now requests.

        Same row-level ``WHERE claimed_at IS NULL`` guard as
        ``claim_pending_workflow_requests`` — a concurrent claimer that
        picked the same rows loses the race because its UPDATE filters out
        already-claimed rows, so no request fires twice.
        """
        conn = await self._ensure_connected()
        now = time.time()
        cursor = await conn.execute(
            "SELECT * FROM task_run_requests "
            "WHERE claimed_at IS NULL ORDER BY created_at ASC LIMIT ?",
            (int(limit),),
        )
        rows = await cursor.fetchall()
        if not rows:
            return []
        ids = [row["id"] for row in rows]
        placeholders = ",".join("?" for _ in ids)
        await conn.execute(
            f"UPDATE task_run_requests SET claimed_at = ? "
            f"WHERE id IN ({placeholders}) AND claimed_at IS NULL",
            [now, *ids],
        )
        await conn.commit()
        cursor = await conn.execute(
            f"SELECT * FROM task_run_requests "
            f"WHERE id IN ({placeholders}) AND claimed_at = ? "
            f"ORDER BY created_at ASC",
            [*ids, now],
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def set_task_request_run_id(self, request_id: str, run_id: str) -> None:
        """Link a claimed request to the ``task_runs`` row it spawned so the
        MCP tool's ``wait`` poller can find the firing."""
        conn = await self._ensure_connected()
        await conn.execute(
            "UPDATE task_run_requests SET run_id = ? WHERE id = ?",
            (run_id, request_id),
        )
        await conn.commit()

    async def get_task_run_request(self, request_id: str) -> dict | None:
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM task_run_requests WHERE id = ?", (request_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    # ── Events (webhook channel) ──
    #
    # An event is an inbound trigger bound to an action (workflow / task /
    # prompt). CRUD mirrors ``scheduled_tasks``; the secret is hashed via
    # ``src.core.event_secret`` and is NEVER returned by a read — callers get
    # ``secret_hint`` only. ``event_deliveries`` is both the run history and
    # the cross-process queue (see ``claim_pending_event_deliveries``).

    @staticmethod
    def _row_to_event(row: aiosqlite.Row, *, include_secret: bool = False) -> dict:
        """Hydrate an event row. ``input_schema_json`` is parsed into a list;
        ``enabled`` becomes a real bool. The encrypted secret is dropped unless
        ``include_secret`` (only the webhook-auth path sets it)."""
        d = dict(row)
        raw = d.pop("input_schema_json", None) or "[]"
        try:
            d["input_schema"] = json.loads(raw)
        except (TypeError, ValueError):
            d["input_schema"] = []
        d["enabled"] = bool(d.get("enabled"))
        d["session_binding_enabled"] = bool(d.get("session_binding_enabled"))
        from src.core.execution_policy import normalize_execution_policy

        d["execution_policy"] = normalize_execution_policy(
            d.pop("execution_policy_json", None)
        )
        if not include_secret:
            d.pop("secret_enc", None)
        return d

    async def add_event(
        self,
        *,
        name: str,
        action_kind: str,
        slug: str,
        secret_enc: str,
        secret_hint: str | None = None,
        event_type: str = "generic",
        description: str | None = None,
        input_schema: list | None = None,
        action_ref: str | None = None,
        prompt_template: str | None = None,
        model: str | None = None,
        session_binding_enabled: bool = False,
        session_binding_path: str | None = None,
        execution_policy: Any = None,
        rate_limit_per_min: int = 60,
        max_payload_bytes: int = 262144,
        enabled: bool = True,
    ) -> str:
        from src.core.execution_policy import encode_execution_policy

        conn = await self._ensure_connected()
        event_id = str(uuid.uuid4())
        now = time.time()
        await conn.execute(
            "INSERT INTO events "
            "(id, name, slug, description, type, enabled, secret_enc, "
            " secret_hint, input_schema_json, action_kind, action_ref, prompt_template, "
            " model, session_binding_enabled, session_binding_path, execution_policy_json, "
            " rate_limit_per_min, max_payload_bytes, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id, name, slug, description or None, event_type,
                1 if enabled else 0, secret_enc, secret_hint,
                json.dumps(input_schema or []), action_kind, action_ref or None,
                prompt_template or None, model or None,
                1 if session_binding_enabled else 0,
                (session_binding_path or "").strip() or None,
                encode_execution_policy(execution_policy),
                int(rate_limit_per_min), int(max_payload_bytes), now, now,
            ),
        )
        await conn.commit()
        return event_id

    async def list_events(self, *, enabled_only: bool = False) -> list[dict]:
        conn = await self._ensure_connected()
        if enabled_only:
            cursor = await conn.execute(
                "SELECT * FROM events WHERE enabled = 1 ORDER BY created_at DESC"
            )
        else:
            cursor = await conn.execute("SELECT * FROM events ORDER BY created_at DESC")
        return [self._row_to_event(r) for r in await cursor.fetchall()]

    async def get_event(self, event_id: str, *, include_secret: bool = False) -> dict | None:
        conn = await self._ensure_connected()
        cursor = await conn.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = await cursor.fetchone()
        return self._row_to_event(row, include_secret=include_secret) if row else None

    async def get_event_by_slug(self, slug: str, *, include_secret: bool = False) -> dict | None:
        conn = await self._ensure_connected()
        cursor = await conn.execute("SELECT * FROM events WHERE slug = ?", (slug,))
        row = await cursor.fetchone()
        return self._row_to_event(row, include_secret=include_secret) if row else None

    async def slug_exists(self, slug: str) -> bool:
        conn = await self._ensure_connected()
        cursor = await conn.execute("SELECT 1 FROM events WHERE slug = ?", (slug,))
        return (await cursor.fetchone()) is not None

    async def update_event(self, event_id: str, **kwargs: Any) -> None:
        conn = await self._ensure_connected()
        allowed = {
            "name", "slug", "description", "type", "enabled", "action_kind",
            "action_ref", "prompt_template", "model", "rate_limit_per_min",
            "max_payload_bytes", "last_triggered_at",
            "session_binding_enabled", "session_binding_path",
            "precondition_json", "execution_policy_json",
            # Secret rotation goes through ``rotate_event_secret``; these are
            # accepted here too so that path can reuse the same UPDATE.
            "secret_enc", "secret_hint",
        }
        updates: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            if k in ("enabled", "session_binding_enabled"):
                v = 1 if v else 0
            if k == "session_binding_path":
                v = (str(v).strip() if v is not None else "") or None
            if k == "precondition_json" and v is not None and not isinstance(v, str):
                # Accept the spec as a dict from callers that build it in code;
                # store the canonical JSON either way.
                v = json.dumps(v)
            if k == "execution_policy_json":
                from src.core.execution_policy import encode_execution_policy

                v = encode_execution_policy(v)
            updates[k] = v
        # ``input_schema`` is passed as a list and serialised here.
        if "input_schema" in kwargs:
            updates["input_schema_json"] = json.dumps(kwargs["input_schema"] or [])
        if not updates:
            return
        updates["updated_at"] = time.time()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        await conn.execute(
            f"UPDATE events SET {set_clause} WHERE id = ?",
            list(updates.values()) + [event_id],
        )
        await conn.commit()

    async def rotate_event_secret(
        self, event_id: str, *, secret_enc: str, secret_hint: str,
    ) -> None:
        """Replace an event's secret. The old secret is invalidated the moment
        this commits (no grace window) — the clear value is generated by the
        caller and returned to the user once."""
        await self.update_event(
            event_id, secret_enc=secret_enc, secret_hint=secret_hint,
        )

    async def delete_event(self, event_id: str) -> None:
        conn = await self._ensure_connected()
        # ON DELETE CASCADE clears event_deliveries.
        await conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
        await conn.commit()

    async def get_event_session_binding(
        self, event_id: str, binding_key: str,
    ) -> dict | None:
        """Return the stored payload-key → internal session binding, if any."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM event_session_bindings "
            "WHERE event_id = ? AND binding_key = ?",
            (event_id, binding_key),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_or_create_event_session_binding(
        self,
        event_id: str,
        binding_key: str,
        *,
        candidate_session_id: str,
    ) -> tuple[str, bool]:
        """Resolve a payload binding to an OpenAgent session id.

        If the ``(event_id, binding_key)`` pair is new, ``candidate_session_id``
        is inserted. If another delivery wins the race first, return the
        already-stored session id instead. The external binding key is never
        used as a session id.
        """
        conn = await self._ensure_connected()
        now = time.time()
        cursor = await conn.execute(
            """
            INSERT OR IGNORE INTO event_session_bindings
                (event_id, binding_key, session_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_id, binding_key, candidate_session_id, now, now),
        )
        created = bool(cursor.rowcount)
        await conn.execute(
            "UPDATE event_session_bindings SET updated_at = ? "
            "WHERE event_id = ? AND binding_key = ?",
            (now, event_id, binding_key),
        )
        await conn.commit()
        row = await self.get_event_session_binding(event_id, binding_key)
        if not row:
            # Defensive fallback; INSERT OR IGNORE + SELECT should make this
            # unreachable unless the row was concurrently deleted.
            return candidate_session_id, created
        return str(row["session_id"]), created

    async def count_recent_deliveries(self, event_id: str, *, window_s: float = 60.0) -> int:
        """Deliveries for an event started within the last ``window_s`` seconds.
        Backstop for the in-memory rate limiter (survives a restart / covers
        multiple listener processes sharing one DB)."""
        conn = await self._ensure_connected()
        cutoff = time.time() - window_s
        cursor = await conn.execute(
            "SELECT COUNT(*) AS n FROM event_deliveries "
            "WHERE event_id = ? AND started_at >= ?",
            (event_id, cutoff),
        )
        row = await cursor.fetchone()
        return int(row["n"]) if row else 0

    # ── Event Deliveries (history + cross-process run queue) ──

    async def add_event_delivery(
        self,
        *,
        event_id: str,
        source: str = "webhook",
        external_id: str | None = None,
        payload: dict | None = None,
        status: str = "received",
        delivery_id: str | None = None,
        claimed: bool = True,
    ) -> str:
        """Open a delivery row. ``claimed=False`` leaves ``claimed_at`` NULL so
        the Scheduler's drain picks it up (the out-of-process MCP path);
        in-process callers pass ``claimed=True`` (the default) since they
        dispatch it themselves.

        An in-process (``claimed=True``) insert also stamps the claim-lease
        (``claim_expires``, ``worker_id``, ``worker_pid``) so the same
        dispatch-runner heartbeat + lease-reaper recovery covers a webhook
        delivery from the moment it is born. An out-of-process insert leaves
        the lease NULL; ``claim_pending_event_deliveries`` stamps it when the
        Scheduler claims the row."""
        conn = await self._ensure_connected()
        did = delivery_id or str(uuid.uuid4())
        now = time.time()
        claim_expires = (now + _lease_ttl_seconds()) if claimed else None
        wid = WORKER_ID if claimed else None
        wpid = WORKER_PID if claimed else None
        await conn.execute(
            "INSERT INTO event_deliveries "
            "(id, event_id, source, external_id, status, payload_json, started_at, "
            " claimed_at, claim_expires, worker_id, worker_pid) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                did, event_id, source, external_id, status,
                json.dumps(payload or {}), now, (now if claimed else None),
                claim_expires, wid, wpid,
            ),
        )
        # Stamp last_triggered_at so the events list can show recency.
        await conn.execute(
            "UPDATE events SET last_triggered_at = ? WHERE id = ?", (now, event_id),
        )
        await conn.commit()
        return did

    async def update_event_delivery(self, delivery_id: str, **kwargs: Any) -> None:
        allowed = {
            "status", "finished_at", "output", "error",
            "session_id", "workflow_run_id", "task_run_id",
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        params = list(updates.values()) + [delivery_id]

        # Lock-surviving: this is the finalizer that a rate-limit storm used to
        # lose to the runtime's big ``runs`` commit — leaving the row ``running``
        # forever. A bounded retry lands the (idempotent, single-row) write once
        # the writer frees up.
        async def _do() -> None:
            conn = await self._ensure_connected()
            await conn.execute(
                f"UPDATE event_deliveries SET {set_clause} WHERE id = ?", params,
            )
            await conn.commit()

        await self._write_with_retry(_do)

    async def get_event_delivery(self, delivery_id: str) -> dict | None:
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM event_deliveries WHERE id = ?", (delivery_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_event_deliveries(
        self, event_id: str, *, limit: int = 20,
    ) -> list[dict]:
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM event_deliveries WHERE event_id = ? "
            "ORDER BY started_at DESC LIMIT ?",
            (event_id, int(limit)),
        )
        return [dict(r) for r in await cursor.fetchall()]

    async def find_delivery_by_external_id(
        self, event_id: str, external_id: str,
    ) -> dict | None:
        """Idempotency check: has this provider delivery id already been
        recorded for this event? Used to dedupe webhook redeliveries."""
        if not external_id:
            return None
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM event_deliveries "
            "WHERE event_id = ? AND external_id = ? LIMIT 1",
            (event_id, external_id),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def claim_pending_event_deliveries(self, *, limit: int = 20) -> list[dict]:
        """Atomically claim up to ``limit`` unclaimed deliveries (those the
        events-manager MCP enqueued out-of-process). Same ``WHERE claimed_at
        IS NULL`` race guard as ``claim_pending_task_requests``.

        The claim also stamps the lease (``claim_expires`` = now + LEASE_TTL,
        ``worker_id``, ``worker_pid``) so the Scheduler's dispatch runner
        heartbeats it while the turn runs and ``reap_expired_event_leases``
        recovers it if the turn freezes."""
        conn = await self._ensure_connected()
        now = time.time()
        claim_expires = now + _lease_ttl_seconds()
        cursor = await conn.execute(
            # ``finished_at IS NULL`` is load-bearing, not tidiness. The claim
            # orders by ``started_at ASC`` and does NOT look at status, so a row
            # that finished without ever being claimed — cancelled out of band,
            # imported, or written by a tool that set an outcome directly — sits
            # at the head of the queue forever and is re-claimed on every tick,
            # spending the whole batch. Measured 19-ago-2026 on a cloned agent:
            # 1057 such rows starved every genuinely pending delivery behind
            # them, and from the outside it looked exactly like "no work".
            "SELECT * FROM event_deliveries "
            "WHERE claimed_at IS NULL AND finished_at IS NULL "
            "ORDER BY started_at ASC LIMIT ?",
            (int(limit),),
        )
        rows = await cursor.fetchall()
        if not rows:
            return []
        ids = [row["id"] for row in rows]
        placeholders = ",".join("?" for _ in ids)
        await conn.execute(
            f"UPDATE event_deliveries SET claimed_at = ?, claim_expires = ?, "
            f"worker_id = ?, worker_pid = ? "
            f"WHERE id IN ({placeholders}) AND claimed_at IS NULL",
            [now, claim_expires, WORKER_ID, WORKER_PID, *ids],
        )
        await conn.commit()
        cursor = await conn.execute(
            f"SELECT * FROM event_deliveries "
            f"WHERE id IN ({placeholders}) AND claimed_at = ? "
            f"ORDER BY started_at ASC",
            [*ids, now],
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def heartbeat_event_delivery(
        self, delivery_id: str, worker_id: str,
    ) -> int:
        """Extend the claim lease on an in-flight delivery: ``claim_expires`` =
        now + LEASE_TTL, ``last_heartbeat_at`` = now, but ONLY while the row is
        still owned by ``worker_id`` (``AND worker_id = ?``) — a row that has
        already been reclaimed and re-dispatched by another owner is left alone.

        This is a tiny single-row write: it survives writer contention far better
        than the runtime's big per-step ``runs`` commit, so *failing* to land it
        (heartbeat stops) is the freeze signal the lease reaper acts on. Wrapped
        in the same bounded lock-surviving retry as ``update_event_delivery``.
        Returns the number of rows updated (0 = the lease was already lost)."""
        now = time.time()
        claim_expires = now + _lease_ttl_seconds()

        async def _do() -> int:
            conn = await self._ensure_connected()
            cur = await conn.execute(
                "UPDATE event_deliveries "
                "SET claim_expires = ?, last_heartbeat_at = ? "
                "WHERE id = ? AND worker_id = ?",
                (claim_expires, now, delivery_id, worker_id),
            )
            await conn.commit()
            return cur.rowcount or 0

        return await self._write_with_retry(_do)

    async def count_open_event_deliveries(self) -> int:
        """Quante delivery non sono ancora finite (``received`` + ``running``).

        Serve al battito dello scheduler: un numero che non scende mentre i beat
        continuano dice che il loop e' vivo ma la coda non avanza — che e' un
        guasto diverso da "il loop e' fermo", e i due vanno distinti."""
        conn = await self._ensure_connected()
        cur = await conn.execute(
            "SELECT COUNT(*) FROM event_deliveries "
            "WHERE status IN ('received','running')"
        )
        row = await cur.fetchone()
        return int(row[0]) if row else 0

    async def reap_expired_event_leases(
        self,
        *,
        enabled: bool | None = None,
        max_attempts: int | None = None,
    ) -> int:
        """Reclaim event deliveries whose claim lease has EXPIRED — the fast
        (~2 s loop) recovery path, so a frozen turn is recovered in ~LEASE_TTL
        instead of the coarse 30-min stale-sweep age.

        A live turn's dispatch runner heartbeats ``claim_expires`` forward, so a
        row with ``claim_expires < now`` provably has not heartbeated for a full
        lease window — its process/turn is frozen (the WAL-writer wedge) or dead.
        Reset it to ``received`` with ``claimed_at``/lease cleared so the drain
        re-dispatches it, and bump ``reenqueue_count``; a row past the replay
        budget is parked terminal ``failed`` instead.

        SAFE AT DEPLOY — the load-bearing property. This ONLY touches rows with
        ``claim_expires IS NOT NULL``. Every in-flight row that predates this
        deploy has ``claim_expires = NULL`` (the column was just added, and only
        a NEW claim stamps it), so the lease reaper never reclaims a pre-existing
        row — those stay handled by the age-gated stale sweep / startup reap.
        Going forward, only leases this build stamped (and thus heartbeats) are
        eligible, so a still-running turn that keeps heartbeating is never
        reclaimed. Respects the shared ``OPENAGENT_EVENT_REENQUEUE_ENABLED``
        kill-switch (off → no-op, returns 0), the redeploy-free escape hatch.

        Returns the number of rows acted on (re-enqueued + parked)."""
        conn = await self._ensure_connected()
        now = time.time()

        if enabled is None:
            enabled = _env_bool("OPENAGENT_EVENT_REENQUEUE_ENABLED", True)
        if not enabled:
            return 0
        if max_attempts is None:
            max_attempts = _env_int("OPENAGENT_EVENT_REENQUEUE_MAX_ATTEMPTS", 5)
        max_attempts = max(1, int(max_attempts))

        async def _do() -> tuple[int, int]:
            # 1. Park expired-lease rows over the replay budget FIRST (so the
            #    re-enqueue below can't bump them past the cap). Clear the lease.
            parked_cur = await conn.execute(
                "UPDATE event_deliveries "
                "SET status='failed', finished_at=?, "
                "    claim_expires=NULL, worker_id=NULL, worker_pid=NULL, "
                "    error=COALESCE(error,'') || "
                "          CASE WHEN error IS NULL OR error='' THEN '' ELSE ' | ' END || "
                "          'lease-reap: retry budget exhausted (' || reenqueue_count || ' attempts)' "
                "WHERE claim_expires IS NOT NULL AND claim_expires < ? "
                "  AND reenqueue_count >= ? "
                "  AND status IN ('received','running')",
                (now, now, max_attempts),
            )
            parked = parked_cur.rowcount or 0

            # 2. Re-enqueue every expired-lease orphan under the budget: reset to
            #    ``received``, drop the claim + lease, clear finished_at, bump the
            #    replay counter (distinct ``lease-reap`` marker).
            requeue_cur = await conn.execute(
                "UPDATE event_deliveries "
                "SET status='received', claimed_at=NULL, finished_at=NULL, "
                "    claim_expires=NULL, worker_id=NULL, worker_pid=NULL, "
                "    reenqueue_count = reenqueue_count + 1, "
                "    error='re-enqueued: lease-reap recovered orphan (attempt ' "
                "          || (reenqueue_count + 1) || ')' "
                "WHERE claim_expires IS NOT NULL AND claim_expires < ? "
                "  AND reenqueue_count < ? "
                "  AND status IN ('received','running')",
                (now, max_attempts),
            )
            requeued = requeue_cur.rowcount or 0
            await conn.commit()
            return requeued, parked

        requeued, parked = await self._write_with_retry(_do)
        if requeued or parked:
            from src.core.logging import elog
            elog(
                "event.orphan_reaped", mode="lease-reap",
                requeued=requeued, parked=parked, max_attempts=max_attempts,
            )
        return requeued + parked

    # ── Per-event circuit breaker (gated by OPENAGENT_EVENT_BREAKER_ENABLED) ──

    def _breaker_enabled(self) -> bool:
        """Master gate. OFF by default → every breaker method is a no-op and
        behaviour is byte-identical to today (mirrors the
        ``OPENAGENT_EVENT_REENQUEUE_ENABLED`` kill-switch shape)."""
        return _env_bool("OPENAGENT_EVENT_BREAKER_ENABLED", False)

    def _breaker_threshold(self) -> int:
        """Global consecutive-permanent-failure limit (>= 1) when an event has
        no per-event ``max_retries`` override."""
        return max(1, _env_int("OPENAGENT_EVENT_BREAKER_THRESHOLD", 5))

    async def record_event_failure(
        self, event_id: str, error: str | None = None,
    ) -> None:
        """Count ONE permanent failure against the event's breaker and trip it
        when the streak reaches the effective limit (per-event ``max_retries``,
        else the global default). No-op when the breaker is disabled — so the
        counter never leaves 0 and nothing is ever blocked (identical to today).

        The caller is responsible for classifying: a transient
        provider-429/throttle/timeout or a cancellation must NOT reach here, or a
        rate-limit storm would trip the breaker on the very event it should keep
        serving."""
        if not self._breaker_enabled():
            return
        now = time.time()
        conn = await self._ensure_connected()
        await conn.execute(
            "UPDATE events "
            "SET consecutive_failures = consecutive_failures + 1, "
            "    last_failure_error = ? "
            "WHERE id = ?",
            ((str(error)[:2000] if error else None), event_id),
        )
        cursor = await conn.execute(
            "SELECT consecutive_failures, max_retries FROM events WHERE id = ?",
            (event_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            await conn.commit()
            return
        limit = row["max_retries"] if row["max_retries"] is not None else self._breaker_threshold()
        limit = max(1, int(limit))
        if int(row["consecutive_failures"]) >= limit:
            # Idempotent: only stamp the trip time once (first time it crosses).
            await conn.execute(
                "UPDATE events SET breaker_tripped_at = ? "
                "WHERE id = ? AND breaker_tripped_at IS NULL",
                (now, event_id),
            )
        await conn.commit()

    async def reset_event_breaker(self, event_id: str) -> None:
        """Clear the breaker on a terminal success: streak → 0, un-trip, drop the
        last-error. No-op when the breaker is disabled."""
        if not self._breaker_enabled():
            return
        conn = await self._ensure_connected()
        await conn.execute(
            "UPDATE events "
            "SET consecutive_failures = 0, breaker_tripped_at = NULL, "
            "    last_failure_error = NULL "
            "WHERE id = ?",
            (event_id,),
        )
        await conn.commit()

    async def is_event_breaker_tripped(self, event_id: str) -> bool:
        """True when the event's breaker is open (``breaker_tripped_at`` set).
        Always False when the breaker is disabled — so the enforcement checks in
        the webhook path and the scheduler drain are inert by default."""
        if not self._breaker_enabled():
            return False
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT breaker_tripped_at FROM events WHERE id = ?", (event_id,),
        )
        row = await cursor.fetchone()
        return bool(row and row["breaker_tripped_at"] is not None)

    # Marker the OLD reaper stamped on the deliveries it (wrongly) failed. It
    # is the one-time backfill signal: a ``failed`` row carrying this exact
    # phrase was never a genuine failure — it was a support ticket dropped by
    # ``reap → failed`` on a prior restart. Matched verbatim so a real
    # application failure (a bad template, a rejected action) is never
    # resurrected. The re-enqueue path never re-writes this phrase, so it
    # only ever marks the pre-fix historical rows.
    _REAP_ORPHAN_MARK = "reaped: orphan from prior process"
    # Un turno che muore senza capacita' di modello (nessun account,
    # 429 ovunque, timeout del provider) chiude la delivery `failed` ma
    # NON e' un evento difettoso: e' lo stesso messaggio del cliente che
    # va riprovato quando la capacita' torna. Marcato a parte dagli
    # orfani del reaper perche' i due casi hanno interruttori diversi:
    # BACKFILL riguarda la storia vecchia, questo il go-forward.
    _RETRYABLE_TURN_MARK = "retryable: turn died without model capacity"

    async def reap_orphan_event_deliveries(
        self,
        *,
        enabled: bool | None = None,
        max_attempts: int | None = None,
        recover_failed: bool | None = None,
    ) -> int:
        """Recover deliveries a prior process left mid-flight so no inbound
        (support ticket) is ever silently dropped.

        The old behaviour marked every ``received`` / ``running`` orphan
        ``failed`` and forgot it — *at-most-once*. A delivery that was claimed
        and dispatched but whose process died before the turn finished was
        never re-fired, so on the live esound-openagent pod **1181 of 1267
        ``failed`` rows carried this exact ``reaped: orphan …`` marker**: not
        failures, dropped tickets.

        This re-enqueues an orphan instead — resets it to ``received`` with
        ``claimed_at = NULL`` — so the Scheduler's ``_drain_event_deliveries``
        loop claims and re-dispatches it (*at-least-once*). Two recoverable
        classes:

        * **in-flight orphans** — ``status IN ('received','running')`` with a
          claim, i.e. interrupted by *this* deploy's restart; and
        * **historical orphans** — ``status='failed'`` carrying the old
          reaper's marker, i.e. dropped by a *prior* restart (a one-time
          backfill; gate off with ``recover_failed=False`` /
          ``OPENAGENT_EVENT_REENQUEUE_BACKFILL=0`` to only fix go-forward).

        SAFETY — why a replay never double-messages a customer. The only
        externally-fired event (``Replio inbound thread``) is a ``prompt``
        action with ``session_binding_enabled=1`` bound on the thread id, and
        an event child session id is the deterministic
        ``event:{event_id}:{delivery_id}``. So a replay *always* resumes the
        SAME session as the original attempt — bound events via
        ``event_session_bindings`` (keyed on the thread), unbound events via
        the deterministic id keyed on the replayed delivery id. The agent
        therefore re-runs with its own prior transcript in context, and the
        customer reply itself goes through Replio's server-side ``reply_guard``
        (LIVE/armed), which suppresses a second outbound when the thread
        already has one newer than the triggering inbound. Re-enqueue adds no
        new reply path; it only re-drives the existing, thread-scoped,
        reply-idempotent one. A ``received`` orphan is safe unconditionally —
        ``dispatch_event`` flips a delivery to ``running`` as its very first
        act, so a still-``received`` row proves the turn never ran and no reply
        could have been sent.

        ``reenqueue_count`` bounds the replay: an orphan re-enqueued
        ``max_attempts`` times is parked terminal ``failed`` so a delivery that
        keeps killing the process can't crash-loop the pod and starve every
        other ticket. Set ``OPENAGENT_EVENT_REENQUEUE_ENABLED=0`` to fall back
        to the legacy mark-``failed`` behaviour without a redeploy.

        Returns the number of rows acted on (re-enqueued + parked), preserving
        the ``int`` contract the ``AgentServer.start()`` caller logs on.
        """
        conn = await self._ensure_connected()
        now = time.time()

        def _flag(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return raw.strip().lower() not in ("0", "false", "no", "off", "")

        if enabled is None:
            enabled = _flag("OPENAGENT_EVENT_REENQUEUE_ENABLED", True)
        if recover_failed is None:
            recover_failed = _flag("OPENAGENT_EVENT_REENQUEUE_BACKFILL", True)
        # Separato da BACKFILL apposta: sui tre agent in produzione BACKFILL e'
        # a 0 (non si resuscita la storia vecchia), ma un turno morto adesso per
        # mancanza di capacita' va ritentato lo stesso.
        recover_transient = _flag("OPENAGENT_EVENT_REENQUEUE_TRANSIENT", True)
        if max_attempts is None:
            try:
                max_attempts = int(
                    os.environ.get("OPENAGENT_EVENT_REENQUEUE_MAX_ATTEMPTS", "5")
                )
            except (TypeError, ValueError):
                max_attempts = 5
        max_attempts = max(1, int(max_attempts))

        if not enabled:
            # Kill-switch: the legacy at-most-once behaviour (mark orphans
            # failed, never re-fire) — a redeploy-free escape hatch.
            cursor = await conn.execute(
                "UPDATE event_deliveries "
                "SET status='failed', finished_at=?, "
                "    error=COALESCE(error,'') || "
                "          CASE WHEN error IS NULL OR error='' THEN '' ELSE ' | ' END || "
                "          ? "
                "WHERE status IN ('received','running') AND claimed_at IS NOT NULL",
                (now, self._REAP_ORPHAN_MARK),
            )
            await conn.commit()
            n = cursor.rowcount or 0
            if n:
                from src.core.logging import elog
                elog("event.orphan_reaped", mode="legacy-failed", count=n)
            return n

        # 1. Park orphans that have exhausted the replay budget FIRST, so a row
        #    at the cap is retired here and not re-enqueued below (order
        #    matters: re-enqueue increments the counter). Only in-flight rows
        #    are parked — a historical ``failed`` row at/over the cap is already
        #    terminal and left alone.
        parked_cur = await conn.execute(
            "UPDATE event_deliveries "
            "SET status='failed', finished_at=?, "
            "    error=COALESCE(error,'') || "
            "          CASE WHEN error IS NULL OR error='' THEN '' ELSE ' | ' END || "
            "          'reaped: retry budget exhausted (' || reenqueue_count || ' attempts)' "
            "WHERE claimed_at IS NOT NULL AND reenqueue_count >= ? "
            "  AND status IN ('received','running')",
            (now, max_attempts),
        )
        parked = parked_cur.rowcount or 0

        # 2. Re-enqueue every recoverable orphan under the budget: reset to
        #    ``received`` + drop the claim so the scheduler drain re-dispatches
        #    it; clear ``finished_at``; bump the replay counter.
        where = (
            "claimed_at IS NOT NULL AND reenqueue_count < ? "
            "AND (status IN ('received','running')"
        )
        params: list[Any] = [max_attempts]
        if recover_failed:
            where += " OR (status='failed' AND error LIKE ?)"
            params.append(f"%{self._REAP_ORPHAN_MARK}%")
        if recover_transient:
            where += " OR (status='failed' AND error LIKE ?)"
            params.append(f"%{self._RETRYABLE_TURN_MARK}%")
        where += ")"
        requeue_cur = await conn.execute(
            "UPDATE event_deliveries "
            "SET status='received', claimed_at=NULL, finished_at=NULL, "
            "    reenqueue_count = reenqueue_count + 1, "
            "    error='re-enqueued: recovered orphan (attempt ' "
            "          || (reenqueue_count + 1) || ')' "
            "WHERE " + where,
            params,
        )
        requeued = requeue_cur.rowcount or 0

        await conn.commit()
        if requeued or parked:
            from src.core.logging import elog
            elog(
                "event.orphan_reaped", mode="re-enqueue",
                requeued=requeued, parked=parked, max_attempts=max_attempts,
            )
        return requeued + parked

    async def reap_stale_event_deliveries(
        self,
        *,
        min_claim_age_seconds: float,
        max_attempts: int | None = None,
        enabled: bool | None = None,
    ) -> int:
        """Periodic, age-gated sibling of ``reap_orphan_event_deliveries`` for
        deliveries orphaned WITHOUT a process restart.

        The startup reap fires once, on boot: every claimed row is *provably* an
        orphan there because the process just started, so it re-enqueues them
        all. But a delivery whose detached dispatch task dies silently while the
        process keeps running — the turn task cancelled, an unhandled crash in
        the background coroutine, an event-loop stall — never gets a restart, so
        it sits ``running``/claimed forever until the next deploy. This sweep,
        called on the Scheduler's fast loop, recovers those go-forward.

        CRITICAL — the age guard. Because the process is *live*, a claimed row is
        NOT provably an orphan: a claim 30 s old is a legitimately-running turn,
        and re-enqueuing it would spawn a SECOND concurrent turn for the same
        delivery. So this only touches rows whose ``claimed_at`` is OLDER than
        ``min_claim_age_seconds`` — a threshold the caller sets comfortably above
        the single-turn wall-clock cap (``OPENAGENT_CHAT_TURN_TIMEOUT``, default
        900 s); the Scheduler passes 2× that (1800 s) by default. A row claimed
        more recently is assumed still running and left strictly alone. Replio's
        thread-scoped ``reply_guard`` remains the final backstop, but the age
        gate is what prevents the double-dispatch in the first place.

        Differences from the startup reap, all deliberate:

        * **age-gated** — only ``claimed_at <= now - min_claim_age_seconds``;
        * **go-forward only** — NEVER touches ``status='failed'`` history (no
          backfill of the old reaper's dropped-ticket rows; that one-time
          recovery is the startup reap's job). Only in-flight
          ``status IN ('received','running')`` rows are eligible.

        Same re-enqueue / park semantics otherwise: reset a recoverable orphan
        to ``received`` with ``claimed_at=NULL`` (so the drain re-dispatches it)
        and bump ``reenqueue_count``; park a stale row that has exhausted the
        ``max_attempts`` budget as terminal ``failed``. Respects the
        ``OPENAGENT_EVENT_REENQUEUE_ENABLED`` kill-switch (off → no-op, returns
        0 — no marking, since a live process has no boot-time proof of orphaning).

        Returns the number of rows acted on (re-enqueued + parked); 0 when it
        finds nothing (and it logs only when it acts).
        """
        conn = await self._ensure_connected()
        now = time.time()

        def _flag(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return default
            return raw.strip().lower() not in ("0", "false", "no", "off", "")

        if enabled is None:
            enabled = _flag("OPENAGENT_EVENT_REENQUEUE_ENABLED", True)
        if not enabled:
            # Kill-switch: unlike the startup reap this does NOT fall back to
            # mark-failed. A periodic sweep on a live process has no proof a
            # claimed row is orphaned (that's the whole point of the age gate),
            # so with re-enqueue disabled the safe action is to do nothing and
            # let the next restart's reap handle it.
            return 0

        if max_attempts is None:
            try:
                max_attempts = int(
                    os.environ.get("OPENAGENT_EVENT_REENQUEUE_MAX_ATTEMPTS", "5")
                )
            except (TypeError, ValueError):
                max_attempts = 5
        max_attempts = max(1, int(max_attempts))

        # A row is "stale" only if it was claimed at/before this cutoff. Never
        # negative — a caller passing 0 would sweep everything, so clamp at 0
        # and let the caller own the safety of the threshold it chooses.
        min_claim_age_seconds = max(0.0, float(min_claim_age_seconds))
        cutoff = now - min_claim_age_seconds

        # 1. Park stale in-flight rows over the replay budget FIRST (retire them
        #    here so the re-enqueue below can't bump them past the cap). Only
        #    stale, in-flight rows — a recently-claimed row at the cap is still
        #    a live turn and must not be parked; a historical ``failed`` row is
        #    already terminal and never matched.
        parked_cur = await conn.execute(
            "UPDATE event_deliveries "
            "SET status='failed', finished_at=?, "
            "    error=COALESCE(error,'') || "
            "          CASE WHEN error IS NULL OR error='' THEN '' ELSE ' | ' END || "
            "          'stale-sweep: retry budget exhausted (' || reenqueue_count || ' attempts)' "
            "WHERE claimed_at IS NOT NULL AND claimed_at <= ? "
            "  AND reenqueue_count >= ? "
            "  AND status IN ('received','running')",
            (now, cutoff, max_attempts),
        )
        parked = parked_cur.rowcount or 0

        # 2. Re-enqueue every stale in-flight orphan under the budget: same SET
        #    clause as the startup reap (received, drop claim, clear finished_at,
        #    bump counter) with a distinct ``stale-sweep`` error marker.
        requeue_cur = await conn.execute(
            "UPDATE event_deliveries "
            "SET status='received', claimed_at=NULL, finished_at=NULL, "
            "    reenqueue_count = reenqueue_count + 1, "
            "    error='re-enqueued: stale-sweep recovered orphan (attempt ' "
            "          || (reenqueue_count + 1) || ')' "
            "WHERE claimed_at IS NOT NULL AND claimed_at <= ? "
            "  AND reenqueue_count < ? "
            "  AND status IN ('received','running')",
            (cutoff, max_attempts),
        )
        requeued = requeue_cur.rowcount or 0

        # 3. E le delivery morte per MANCANZA DI CAPACITA'. Sono terminali
        #    (``failed``), quindi qui non c'e' nessun turno vivo da duplicare e
        #    il cancello dell'eta' non serve a quello: serve a non rilanciarle
        #    dentro lo stesso blackout che le ha uccise, cioe' a dare al pool il
        #    tempo di tornare. Senza questo passaggio sarebbero ripescate solo
        #    dal reap di avvio, cioe' al prossimo riavvio: ore, per il messaggio
        #    di un cliente. Il marcatore lo mette il dispatcher SOLO quando la
        #    morte e' transitoria (un guasto permanente non finisce mai qui) e il
        #    tetto dei tentativi resta quello di sempre.
        try:
            retry_delay = max(0.0, float(
                os.environ.get("OPENAGENT_EVENT_RETRY_DELAY_SECONDS", "300")))
        except (TypeError, ValueError):
            retry_delay = 300.0
        retried = 0
        if _flag("OPENAGENT_EVENT_REENQUEUE_TRANSIENT", True):
            retry_cur = await conn.execute(
                "UPDATE event_deliveries "
                "SET status='received', claimed_at=NULL, finished_at=NULL, "
                "    reenqueue_count = reenqueue_count + 1, "
                "    error='re-enqueued: no model capacity (attempt ' "
                "          || (reenqueue_count + 1) || ')' "
                "WHERE status='failed' AND error LIKE ? "
                "  AND reenqueue_count < ? "
                "  AND finished_at IS NOT NULL AND finished_at <= ?",
                (f"%{self._RETRYABLE_TURN_MARK}%", max_attempts, now - retry_delay),
            )
            retried = retry_cur.rowcount or 0

        await conn.commit()
        if requeued or parked or retried:
            from src.core.logging import elog
            elog(
                "event.orphan_reaped", mode="stale-sweep",
                requeued=requeued, parked=parked, retried=retried,
                max_attempts=max_attempts,
                min_claim_age_seconds=min_claim_age_seconds,
            )
        return requeued + parked + retried

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
        # ``max_concurrent_runs`` is an INTEGER column with NULL =
        # unlimited. Normalize to an ``int | None`` (the column may be
        # absent in legacy rows when reading mid-migration).
        raw_cap = d.get("max_concurrent_runs")
        d["max_concurrent_runs"] = int(raw_cap) if raw_cap is not None else None
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
        max_concurrent_runs: int | None = None,
    ) -> str:
        if not name or not name.strip():
            raise ValueError("name is required")
        if max_concurrent_runs is not None and max_concurrent_runs < 1:
            raise ValueError("max_concurrent_runs must be >= 1 or NULL (unlimited)")
        graph_payload = graph or {"version": 1, "nodes": [], "edges": [], "variables": {}}
        conn = await self._ensure_connected()
        workflow_id = str(uuid.uuid4())
        now = time.time()
        await conn.execute(
            "INSERT INTO workflow_tasks "
            "(id, name, description, graph_json, enabled, "
            " max_concurrent_runs, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                workflow_id,
                name.strip(),
                description,
                json.dumps(graph_payload),
                1 if enabled else 0,
                max_concurrent_runs,
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
        allowed_direct = {
            "name", "description", "enabled", "last_run_at",
            "max_concurrent_runs",
        }
        updates: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k == "graph" and v is not None:
                updates["graph_json"] = json.dumps(v)
            elif k in allowed_direct:
                if k == "enabled" and isinstance(v, bool):
                    updates[k] = 1 if v else 0
                elif k == "max_concurrent_runs":
                    if v is not None and v < 1:
                        raise ValueError(
                            "max_concurrent_runs must be >= 1 or NULL (unlimited)"
                        )
                    updates[k] = v
                else:
                    updates[k] = v
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
        serialized to their ``_json`` columns.

        Cancellation invariant: once a run is flagged ``status='cancelling'``
        (a "completely stop" request from the workflow-manager MCP), the only
        status it may move to is ``cancelled``. A natural finalize that lands
        in the stop window — the executor writing ``success`` / ``failed``
        just after the flag — is suppressed (the UPDATE matches no row),
        leaving the run ``cancelling`` for the scheduler's cancel handler or
        orphan sweep to finalize as ``cancelled``. This keeps "stop"
        authoritative over a run that completed a hair too late, so a flagged
        run never escapes to a non-cancelled terminal state. Writes that don't
        touch ``status`` (e.g. trace-only updates) are never gated.
        """
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
        where = "WHERE id = ?"
        if "status" in updates and updates["status"] != "cancelled":
            where += " AND status != 'cancelling'"
        conn = await self._ensure_connected()
        await conn.execute(
            f"UPDATE workflow_runs SET {set_clause} {where}",
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

    async def get_workflow_runs_by_status(
        self, status: str, *, limit: int = 500,
    ) -> list[dict]:
        """Every ``workflow_runs`` row in ``status``, across all workflows.
        Powers the scheduler's cancellation drain, which scans for runs the
        workflow-manager MCP flagged ``cancelling`` and stops the in-flight
        executor that owns each one. Backed by ``idx_wfruns_status`` so the
        scan stays cheap even when only a handful of rows ever match."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT * FROM workflow_runs WHERE status = ? "
            "ORDER BY started_at ASC LIMIT ?",
            (status, int(limit)),
        )
        rows = await cursor.fetchall()
        return [self._row_to_workflow_run(r) for r in rows]

    async def reap_orphan_workflow_runs(self) -> int:
        """Mark every workflow_run still in ``running`` as ``failed``.

        Called by ``AgentServer.start()`` after the DB is open. A
        ``running`` row that survives a process restart is by definition
        a zombie — the WorkflowExecutor instance that owned it is gone,
        the in-memory per-workflow lock is fresh, and there's no resume
        path. Worse, a stuck row blocks the next scheduled run of the
        same workflow because the executor's per-workflow ``asyncio.Lock``
        would funnel a new run behind the (never-completing) old one if
        the old in-process executor were still around — and the cosmetic
        "running" badge in the UI never clears.

        Closes that loop by finalizing every orphan with a clear error
        marker so the schedule's next tick starts from a clean slate.
        Returns the number of rows reaped (for telemetry).
        """
        conn = await self._ensure_connected()
        now = time.time()
        cursor = await conn.execute(
            "UPDATE workflow_runs "
            "SET status='failed', finished_at=?, "
            "    error=COALESCE(error, '') || "
            "          CASE WHEN error IS NULL OR error='' THEN '' ELSE ' | ' END || "
            "          'reaped: orphan from prior process' "
            "WHERE status='running'",
            (now,),
        )
        reaped = cursor.rowcount or 0
        # A run flagged 'cancelling' that the prior process never finalized
        # (crash between the MCP flag and the scheduler's drain) is also an
        # orphan — finalize it as 'cancelled' (the requested outcome), kept
        # distinct from the 'failed' reap above.
        cancel_cursor = await conn.execute(
            "UPDATE workflow_runs "
            "SET status='cancelled', finished_at=?, "
            "    error=COALESCE(error, '') || "
            "          CASE WHEN error IS NULL OR error='' THEN '' ELSE ' | ' END || "
            "          'reaped: stop left pending by prior process' "
            "WHERE status='cancelling'",
            (now,),
        )
        await conn.commit()
        return reaped + (cancel_cursor.rowcount or 0)

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
        ticks (or stray retries) won't run the same request twice.

        The atomicity comes from the row-level ``WHERE claimed_at IS NULL``
        guard inside the UPDATE — a concurrent claimer that picked the
        same rows from the SELECT loses the race because their UPDATE
        filters out already-claimed rows. The previous implementation
        wrapped the SELECT+UPDATE in an explicit ``BEGIN IMMEDIATE`` /
        ``COMMIT`` pair, which fought with the ``sqlite3`` driver's
        auto-managed transaction state on the shared aiosqlite
        connection: when a sibling coroutine's DML had already
        auto-begun a transaction, the explicit ``BEGIN IMMEDIATE``
        produced ``cannot start a transaction within a transaction``
        and the except branch's ``ROLLBACK`` then chained
        ``cannot rollback - no transaction is active``.
        """
        conn = await self._ensure_connected()
        now = time.time()
        cursor = await conn.execute(
            "SELECT * FROM workflow_run_requests "
            "WHERE claimed_at IS NULL "
            "ORDER BY created_at ASC LIMIT ?",
            (int(limit),),
        )
        rows = await cursor.fetchall()
        if not rows:
            return []
        ids = [row["id"] for row in rows]
        placeholders = ",".join("?" for _ in ids)
        await conn.execute(
            f"UPDATE workflow_run_requests SET claimed_at = ? "
            f"WHERE id IN ({placeholders}) AND claimed_at IS NULL",
            [now, *ids],
        )
        await conn.commit()
        # Only the rows we won carry our ``now`` marker. Any rows a
        # concurrent claimer grabbed in between SELECT and UPDATE
        # filter out via the ``claimed_at IS NULL`` guard above and
        # don't reappear here.
        cursor = await conn.execute(
            f"SELECT * FROM workflow_run_requests "
            f"WHERE id IN ({placeholders}) AND claimed_at = ? "
            f"ORDER BY created_at ASC",
            [*ids, now],
        )
        rows = await cursor.fetchall()
        claimed: list[dict] = []
        for row in rows:
            d = dict(row)
            raw = d.pop("inputs_json", "{}") or "{}"
            try:
                d["inputs"] = json.loads(raw)
            except (TypeError, ValueError):
                d["inputs"] = {}
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

    # ── Vault recall attribution ──

    async def record_vault_recall(
        self,
        *,
        session_id: str | None,
        note_path: str,
        tool: str,
        outcome: str,
        model: str | None = None,
        cost: float = 0.0,
    ) -> str:
        """Record that a run recalled ``note_path`` and ended in ``outcome``.

        One row per (run, note). See ``src/core/vault_recall.py`` for what the
        outcomes mean and why ``cancelled`` is recorded but never scored.
        """
        conn = await self._ensure_connected()
        row_id = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO vault_recall_outcomes "
            "(id, timestamp, session_id, note_path, tool, outcome, model, cost) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (row_id, time.time(), session_id, note_path, tool, outcome, model, cost),
        )
        await conn.commit()
        return row_id

    async def get_vault_recall_stats(
        self,
        *,
        since: float | None = None,
        limit: int = 50,
        note_path: str | None = None,
    ) -> list[dict]:
        """Per-note recall counts grouped by outcome, most-recalled first.

        Returns raw counts — ``ok``/``errored``/``cancelled`` — and leaves the
        judgement to the caller. Deliberately does NOT return a single quality
        score: a run that read a note and finished proves the run finished, not
        that the note helped (nothing here observes whether the answer was any
        good). Aggregation happens at read time so the outcome definition can
        change without a migration or a backfill.
        """
        conn = await self._ensure_connected()
        where: list[str] = []
        params: list[Any] = []
        if since is not None:
            where.append("timestamp >= ?")
            params.append(since)
        if note_path:
            where.append("note_path = ?")
            params.append(note_path)
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        params.append(int(limit))
        sql = f"""
            SELECT note_path,
                   COUNT(*)                                   AS recalls,
                   SUM(outcome = 'ok')                        AS ok,
                   SUM(outcome = 'errored')                   AS errored,
                   SUM(outcome = 'cancelled')                 AS cancelled,
                   SUM(cost)                                  AS cost,
                   MAX(timestamp)                             AS last_recalled
            FROM vault_recall_outcomes
            {clause}
            GROUP BY note_path
            ORDER BY recalls DESC, last_recalled DESC
            LIMIT ?
        """
        rows = await (await conn.execute(sql, tuple(params))).fetchall()
        out: list[dict] = []
        for r in rows:
            d = dict(r)
            ok = int(d.get("ok") or 0)
            errored = int(d.get("errored") or 0)
            # The denominator EXCLUDES barge-ins by construction, not by a
            # filter a caller has to remember. On the production log all 294
            # errored runs were also cancelled; counting those as failures
            # teaches that users interrupting is a defect (§2 says it is not).
            scorable = ok + errored
            d["scorable"] = scorable
            d["ok_rate"] = round(ok / scorable, 3) if scorable else None
            out.append(d)
        return out

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

    # ── Budgets ──
    #
    # CRUD mirrors the events accessors above. The enforcement gate lives in
    # ``src/core/budget_guard.py``; this layer is pure storage + the windowed
    # spend aggregation over ``usage_log`` that both the gate and the REST/MCP
    # usage view read. Keep cost provenance single: spend is summed from the
    # same ``usage_log.cost`` that ``BudgetTracker.compute_cost`` writes — there
    # is no second cost path here.

    # Values the router gate acts on. ``task`` / ``per_run`` are stored and
    # reported but never enforced here (phase 2 — scheduler skip + mid-run
    # cancel). Kept as literals, not imported, so the memory layer stays free
    # of a dependency on the guard module.
    _ENFORCED_SCOPE_KINDS = ("global", "provider", "model")
    _ENFORCED_WINDOWS = ("hour", "day", "month")

    @staticmethod
    def _row_to_budget(row: aiosqlite.Row) -> dict:
        """Hydrate a budget row: parse ``alert_thresholds_json`` into a list of
        floats and coerce ``enabled`` to a real bool."""
        d = dict(row)
        raw = d.pop("alert_thresholds_json", None) or "[]"
        try:
            parsed = json.loads(raw)
            d["alert_thresholds"] = [float(x) for x in parsed] if isinstance(parsed, list) else []
        except (TypeError, ValueError):
            d["alert_thresholds"] = []
        d["enabled"] = bool(d.get("enabled"))
        return d

    async def add_budget(
        self,
        *,
        scope_kind: str,
        scope_value: str = "",
        metric: str = "cost_usd",
        window: str = "day",
        amount: float,
        alert_thresholds: list[float] | None = None,
        webhook_url: str | None = None,
        enabled: bool = True,
        source: str = "user",
    ) -> str:
        """Insert a budget rule and return its id.

        Raises ``sqlite3.IntegrityError`` on a duplicate
        (scope_kind, scope_value, metric, window) — the REST layer maps that to
        409. ``global`` normalises ``scope_value`` to '' so the UNIQUE holds.
        """
        conn = await self._ensure_connected()
        budget_id = str(uuid.uuid4())
        now = time.time()
        sv = "" if scope_kind == "global" else (scope_value or "").strip()
        thresholds = alert_thresholds if alert_thresholds is not None else [0.5, 0.9]
        await conn.execute(
            "INSERT INTO budgets "
            "(id, scope_kind, scope_value, metric, window, amount, "
            " alert_thresholds_json, webhook_url, enabled, source, "
            " created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                budget_id, scope_kind, sv, metric, window, float(amount),
                json.dumps([float(t) for t in thresholds]),
                (webhook_url or "").strip() or None,
                1 if enabled else 0, source, now, now,
            ),
        )
        await conn.commit()
        return budget_id

    async def seed_budget(
        self,
        *,
        scope_kind: str,
        scope_value: str = "",
        metric: str = "cost_usd",
        window: str = "day",
        amount: float,
        alert_thresholds: list[float] | None = None,
        webhook_url: str | None = None,
        enabled: bool = True,
    ) -> bool:
        """Seed a yaml rule additively — INSERT only if no row already owns the
        same (scope_kind, scope_value, metric, window). Returns True if a row was
        inserted, False if one already existed (operator's edit preserved).

        This is the reconcile contract, identical to ``ensure_builtin_mcps``:
        yaml is a floor, never a clobber. An operator who tweaks the amount in
        the app keeps that tweak across reboots; one who deletes a yaml-seeded
        rule gets it back on the next boot (the rule is declared in config).
        """
        conn = await self._ensure_connected()
        sv = "" if scope_kind == "global" else (scope_value or "").strip()
        cursor = await conn.execute(
            "SELECT 1 FROM budgets WHERE scope_kind = ? AND scope_value = ? "
            "AND metric = ? AND window = ?",
            (scope_kind, sv, metric, window),
        )
        if await cursor.fetchone() is not None:
            return False
        await self.add_budget(
            scope_kind=scope_kind, scope_value=sv, metric=metric, window=window,
            amount=amount, alert_thresholds=alert_thresholds,
            webhook_url=webhook_url, enabled=enabled, source="yaml",
        )
        return True

    async def list_budgets(self, *, enabled_only: bool = False) -> list[dict]:
        conn = await self._ensure_connected()
        if enabled_only:
            cursor = await conn.execute(
                "SELECT * FROM budgets WHERE enabled = 1 ORDER BY created_at ASC"
            )
        else:
            cursor = await conn.execute("SELECT * FROM budgets ORDER BY created_at ASC")
        return [self._row_to_budget(r) for r in await cursor.fetchall()]

    async def get_budget(self, budget_id: str) -> dict | None:
        conn = await self._ensure_connected()
        cursor = await conn.execute("SELECT * FROM budgets WHERE id = ?", (budget_id,))
        row = await cursor.fetchone()
        return self._row_to_budget(row) if row else None

    async def update_budget(self, budget_id: str, **kwargs: Any) -> None:
        conn = await self._ensure_connected()
        allowed = {
            "scope_kind", "scope_value", "metric", "window", "amount",
            "webhook_url", "enabled",
        }
        updates: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k not in allowed:
                continue
            if k == "enabled":
                v = 1 if v else 0
            if k == "amount":
                v = float(v)
            if k == "webhook_url":
                v = (str(v).strip() if v is not None else "") or None
            updates[k] = v
        if "alert_thresholds" in kwargs and kwargs["alert_thresholds"] is not None:
            updates["alert_thresholds_json"] = json.dumps(
                [float(t) for t in kwargs["alert_thresholds"]]
            )
        # Keep the global invariant: '' for a global scope so the UNIQUE holds.
        if updates.get("scope_kind") == "global":
            updates["scope_value"] = ""
        if not updates:
            return
        updates["updated_at"] = time.time()
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        await conn.execute(
            f"UPDATE budgets SET {set_clause} WHERE id = ?",
            list(updates.values()) + [budget_id],
        )
        await conn.commit()

    async def delete_budget(self, budget_id: str) -> None:
        conn = await self._ensure_connected()
        await conn.execute("DELETE FROM budgets WHERE id = ?", (budget_id,))
        await conn.commit()

    async def get_scope_spend(
        self,
        *,
        scope_kind: str,
        scope_value: str,
        metric: str,
        since_epoch: float,
    ) -> float:
        """Sum spend for one budget scope since ``since_epoch`` (inclusive).

        The window boundary is computed by the caller in the agent's timezone
        (``budget_guard.window_start_epoch``) and passed as an epoch, so this
        stays a pure aggregation and the timezone logic has one home.

        - ``metric='cost_usd'`` sums ``cost``; ``metric='tokens'`` sums
          ``input_tokens + output_tokens``.
        - Scope filter:
            global   → no model/session filter (the whole window).
            provider → ``model LIKE '<value>:%'`` (runtime_ids are provider:model).
            model    → ``model = '<value>'``.
            task     → ``session_id LIKE 'scheduler:<value>:%'`` (a task run tree;
                       sub-agent child sessions namespace UNDER the parent, so a
                       prefix match captures the whole tree). Reporting only.
        """
        conn = await self._ensure_connected()
        metric_expr = (
            "COALESCE(SUM(input_tokens + output_tokens), 0)"
            if metric == "tokens"
            else "COALESCE(SUM(cost), 0)"
        )
        params: list[Any] = [since_epoch]
        if scope_kind == "global":
            scope_clause = ""
        elif scope_kind == "provider":
            scope_clause = " AND model LIKE ?"
            params.append(f"{scope_value}:%")
        elif scope_kind == "task":
            scope_clause = " AND session_id LIKE ?"
            params.append(f"scheduler:{scope_value}:%")
        else:  # model
            scope_clause = " AND model = ?"
            params.append(scope_value)
        cursor = await conn.execute(
            f"SELECT {metric_expr} FROM usage_log "
            f"WHERE timestamp >= ?{scope_clause}",
            params,
        )
        row = await cursor.fetchone()
        return float(row[0]) if row and row[0] is not None else 0.0

    # ── Session store (sessions) ──
    #
    # ``sessions`` is the single canonical table for all chat sessions:
    # the api-based path (native the runtime ``Agent``) persists through
    # the runtime's ``SqliteDb`` writing to this table.
    #
    # ``delete_sdk_session`` below stays as a thin shim that scrubs the
    # legacy ``metadata.sdk_session_id`` / ``provider`` keys (written by
    # the retired subscription-CLI path) so the gateway's session-delete
    # handler keeps working on older rows.

    async def delete_sdk_session(self, session_id: str) -> None:
        """Clear legacy subscription-CLI resume metadata from a session.

        The retired claude-cli path stored an SDK session id in
        ``sessions.metadata.sdk_session_id``. This helper survives so the
        gateway's ``DELETE /api/sessions/{id}`` handler keeps a place to
        scrub that legacy data. The session row itself is deleted
        separately via :meth:`delete_session`.
        """
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT metadata FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return
        try:
            meta = json.loads(row[0] or "{}")
        except (TypeError, ValueError):
            meta = {}
        if "sdk_session_id" not in meta and "provider" not in meta:
            return  # already clean — no need to bump updated_at
        meta.pop("sdk_session_id", None)
        meta.pop("provider", None)
        await conn.execute(
            "UPDATE sessions SET metadata = ?, updated_at = ? WHERE session_id = ?",
            (json.dumps(meta), int(time.time()), session_id),
        )
        await self._project_operational_session(session_id)
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
        cursor = await conn.execute("SELECT MAX(CAST(updated_at AS REAL)) FROM mcps")
        row = await cursor.fetchone()
        return _as_epoch(row[0]) if row else 0.0

    # ── Providers (v0.12: one row per (name, framework) pair) ──

    @staticmethod
    def _row_to_provider(row: aiosqlite.Row) -> dict[str, Any]:
        metadata = row["metadata_json"] or "{}"
        try:
            meta_parsed = json.loads(metadata) if isinstance(metadata, str) else {}
        except ValueError:
            meta_parsed = {}
        # ``kind`` is post-ship: legacy DBs may surface aiosqlite.Row
        # objects without it during the brief window between schema
        # script and migration. Default to 'llm' to keep LLM-dispatch
        # callers safe.
        try:
            kind = row["kind"] or "llm"
        except (KeyError, IndexError):
            kind = "llm"
        return {
            "id": row["id"],
            "name": row["name"],
            "framework": row["framework"],
            "kind": kind,
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
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        conn = await self._ensure_connected()
        clauses: list[str] = []
        params: list[Any] = []
        if enabled_only:
            clauses.append("enabled = 1")
        if framework:
            clauses.append("framework = ?")
            params.append(framework)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
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
        kind: str = "llm",
    ) -> int:
        """Upsert a provider row. Returns the provider's surrogate ``id``.

        ``framework='api-based'`` providers may be created with
        ``api_key=None`` (a disabled-until-configured state) but dispatch
        will fail until a key is set.

        ``kind`` defaults to ``'llm'`` (text generation). Use ``'tts'``
        for speech synthesis providers (e.g. ElevenLabs) or ``'stt'`` for
        speech-to-text providers — the LLM dispatcher filters these out
        via ``list_providers(kind='llm')``.
        """
        if not name or not name.strip():
            raise ValueError("name is required")
        if kind not in ("llm", "tts", "stt"):
            raise ValueError(f"invalid kind {kind!r}; expected llm/tts/stt")
        # Legacy framework names ``agno`` / ``litellm`` collapsed into
        # ``api-based`` in v0.14. Rewrite at the boundary so callers
        # (older scripts, tests, third-party automations) keep working.
        # Kept as raw strings — they match pre-rename DB values, not the
        # current FRAMEWORK_* constants.
        if framework in ("agno", "litellm"):  # legacy values; map to api-based
            framework = FRAMEWORK_API_BASED
        if kind == "llm" and framework not in LLM_FRAMEWORKS:
            raise ValueError(
                f"invalid framework {framework!r} for kind='llm'; "
                f"expected one of {LLM_FRAMEWORKS}"
            )
        now = time.time()
        conn = await self._ensure_connected()
        await conn.execute(
            """
            INSERT INTO providers (name, framework, api_key, base_url, enabled, metadata_json, kind, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name, framework) DO UPDATE SET
                api_key = excluded.api_key,
                base_url = excluded.base_url,
                enabled = excluded.enabled,
                metadata_json = excluded.metadata_json,
                kind = excluded.kind,
                updated_at = excluded.updated_at
            """,
            (
                name.strip(),
                framework,
                (api_key or None),
                (base_url or None),
                1 if enabled else 0,
                json.dumps(metadata or {}),
                kind,
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
        cursor = await conn.execute("SELECT MAX(CAST(updated_at AS REAL)) FROM providers")
        row = await cursor.fetchone()
        return _as_epoch(row[0]) if row else 0.0

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
        # ``kind`` is post-ship: legacy aiosqlite.Row objects between
        # the schema script and the migration may surface without it.
        try:
            d["kind"] = d.get("kind") or "llm"
        except (KeyError, IndexError):
            d["kind"] = "llm"
        return d

    async def list_models(
        self,
        *,
        provider_id: int | None = None,
        enabled_only: bool = False,
        kind: str | None = None,
    ) -> list[dict]:
        conn = await self._ensure_connected()
        clauses: list[str] = []
        params: list[Any] = []
        if provider_id is not None:
            clauses.append("provider_id = ?")
            params.append(int(provider_id))
        if enabled_only:
            clauses.append("enabled = 1")
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor = await conn.execute(
            f"SELECT * FROM models {where} ORDER BY provider_id ASC, model ASC",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_model(r) for r in rows]

    async def latest_audio_model(self, kind: str) -> dict | None:
        """Return the most-recently-updated enabled model row of ``kind``,
        joined with its provider for credentials.

        Used by the TTS / STT resolvers — the unified pick rule is
        "latest-edited enabled wins" (matches the classifier-row
        convention used elsewhere). Returns ``None`` when no row matches.
        """
        if kind not in ("tts", "stt"):
            raise ValueError(f"latest_audio_model: invalid kind {kind!r}")
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            self._ENRICHED_MODEL_SELECT
            + " WHERE m.kind = ? AND m.enabled = 1 AND p.enabled = 1"
            "  ORDER BY m.updated_at DESC LIMIT 1",
            (kind,),
        )
        row = await cursor.fetchone()
        return self._shape_enriched(row) if row else None

    # Common projection for the model-joined-with-provider view. Kept
    # as a constant so :meth:`list_models_enriched`,
    # :meth:`get_model_enriched`, and :meth:`get_model_by_runtime_id`
    # return identical dict shapes.
    _ENRICHED_MODEL_SELECT = """
        SELECT m.id AS id, m.provider_id AS provider_id, m.model AS model,
               m.display_name AS display_name, m.tier_hint AS tier_hint,
               m.description AS description,
               m.enabled AS enabled, m.is_classifier AS is_classifier,
               m.metadata_json AS metadata_json, m.kind AS kind,
               m.created_at AS created_at, m.updated_at AS updated_at,
               p.name AS provider_name, p.framework AS framework,
               p.api_key AS api_key, p.base_url AS base_url,
               p.enabled AS provider_enabled
        FROM models m
        JOIN providers p ON p.id = m.provider_id
    """

    @staticmethod
    def _shape_enriched(row: aiosqlite.Row) -> dict:
        from src.models.catalog import build_runtime_model_id

        d = dict(row)
        meta_raw = d.pop("metadata_json", "{}") or "{}"
        try:
            d["metadata"] = json.loads(meta_raw)
        except (TypeError, ValueError):
            d["metadata"] = {}
        d["enabled"] = bool(d["enabled"])
        d["is_classifier"] = bool(d.get("is_classifier"))
        d["provider_enabled"] = bool(d["provider_enabled"])
        d["kind"] = d.get("kind") or "llm"
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
        kind: str | None = None,
    ) -> list[dict]:
        """Return each model joined with its provider row.

        Each row carries ``{id, provider_id, model, display_name, tier_hint,
        enabled, kind, metadata, created_at, updated_at, provider_name,
        framework, api_key, base_url, provider_enabled, runtime_id}`` —
        ``runtime_id`` is derived via :func:`openagent.models.catalog.build_runtime_model_id`.
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
        if kind:
            clauses.append("m.kind = ?")
            params.append(kind)
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
        """Build the NativeProvider-consumable providers_config from the DB.

        Produces the flat list shape ModelDispatcher / NativeProvider consume:
        one entry per (name, framework) pair, each carrying its nested
        ``models`` list. Used by :meth:`Agent._hydrate_providers_from_db`
        (``enabled_only=True``) and by the smoke-test endpoints that
        want every row regardless of enabled state.

        Single LEFT JOIN keeps this to one SQLite round-trip. A provider
        with no models still shows up (important for the UI's "empty
        provider" state).
        """
        conn = await self._ensure_connected()
        # LLM-dispatch hydration must skip TTS/STT model rows (they share
        # the table but route through LiteLLM, not the LLM frameworks).
        # The kind discriminator now lives on ``models.kind``; the
        # ``providers.kind='llm'`` check is kept as a no-op safety net
        # for any pre-migration legacy row that still carried it.
        clauses: list[str] = ["p.kind = 'llm'"]
        join_filter = " AND m.kind = 'llm'"
        if enabled_only:
            # Model-side filter must go in the JOIN predicate, not WHERE,
            # or providers with zero enabled models would disappear.
            join_filter += " AND m.enabled = 1"
            clauses.append("p.enabled = 1")
        where = f"WHERE {' AND '.join(clauses)}"
        cursor = await conn.execute(
            f"""
            SELECT p.id AS p_id, p.name AS p_name, p.framework AS p_framework,
                   p.api_key AS p_api_key, p.base_url AS p_base_url,
                   p.enabled AS p_enabled, p.metadata_json AS p_metadata_json,
                   p.created_at AS p_created_at, p.updated_at AS p_updated_at,
                   m.id AS m_id, m.model AS m_model, m.display_name AS m_display_name,
                   m.tier_hint AS m_tier_hint, m.description AS m_description,
                   m.enabled AS m_enabled,
                   m.is_classifier AS m_is_classifier,
                   m.metadata_json AS m_metadata_json
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
                try:
                    model_metadata = json.loads(r["m_metadata_json"] or "{}")
                except (TypeError, ValueError):
                    model_metadata = {}
                if not isinstance(model_metadata, dict):
                    model_metadata = {}
                bucket["models"].append({
                    "id": int(r["m_id"]),
                    "model": r["m_model"],
                    "display_name": r["m_display_name"],
                    "tier_hint": r["m_tier_hint"],
                    "enabled": bool(r["m_enabled"]),
                    "is_classifier": bool(r["m_is_classifier"]),
                    "metadata": model_metadata,
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
        ``anthropic:claude-opus-4-7``). Returns the same shape
        as :meth:`list_models_enriched`, or ``None`` when no matching
        (provider_name, framework, model) row exists.
        """
        from src.models.catalog import framework_of, split_runtime_id

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
        kind: str = "llm",
    ) -> int:
        """Insert or update a model row. Returns the model's surrogate id."""
        if not provider_id:
            raise ValueError("provider_id is required")
        if not model or not str(model).strip():
            raise ValueError("model is required")
        if kind not in ("llm", "tts", "stt"):
            raise ValueError(f"invalid kind {kind!r}; expected llm/tts/stt")
        conn = await self._ensure_connected()
        # FK integrity: make sure the parent provider exists before we
        # try the insert so callers get a clear error instead of the
        # generic "FOREIGN KEY constraint failed".
        prov_row = await (
            await conn.execute(
                "SELECT name FROM providers WHERE id = ?", (int(provider_id),),
            )
        ).fetchone()
        if prov_row is None:
            raise ValueError(f"Provider id={provider_id!r} does not exist")
        stored_metadata = dict(metadata or {})
        if kind == "llm":
            from src.models.media_capabilities import normalize_model_metadata

            stored_metadata = normalize_model_metadata(
                stored_metadata,
                provider=str(prov_row["name"] or ""),
                model=str(model).strip(),
            )
        now = time.time()
        await conn.execute(
            """
            INSERT INTO models (provider_id, model, display_name, tier_hint,
                                enabled, is_classifier, metadata_json,
                                kind, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(provider_id, model) DO UPDATE SET
                display_name = excluded.display_name,
                tier_hint = excluded.tier_hint,
                enabled = excluded.enabled,
                is_classifier = excluded.is_classifier,
                metadata_json = excluded.metadata_json,
                kind = excluded.kind,
                updated_at = excluded.updated_at
            """,
            (
                int(provider_id),
                str(model).strip(),
                display_name,
                tier_hint,
                1 if enabled else 0,
                1 if is_classifier else 0,
                json.dumps(stored_metadata, sort_keys=True, separators=(",", ":")),
                kind,
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
        """Toggle the ``is_classifier`` flag on ``model_id``.

        Despite the column name this arms no classifier: it is the
        user's persistent "default team leader" hint, read as step 2 of
        ``ModelDispatcher._resolve_entry_model`` (session pin → flagged
        row → first enabled). See ``models.catalog.CatalogModel``.

        Multiple rows are allowed to carry the flag simultaneously —
        this is a narrow UPDATE that only touches ``model_id``. The
        resolver takes the first flagged row it sees in deterministic
        catalog order, so with several flagged rows the first simply
        wins and the others are inert.
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
        cursor = await conn.execute("SELECT MAX(CAST(updated_at AS REAL)) FROM models")
        row = await cursor.fetchone()
        return _as_epoch(row[0]) if row else 0.0

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
            "  COALESCE((SELECT MAX(CAST(updated_at AS REAL)) FROM mcps), 0), "
            "  COALESCE((SELECT MAX(CAST(updated_at AS REAL)) FROM models), 0), "
            "  COALESCE(("
            "    SELECT COUNT(*) FROM models m "
            "    JOIN providers p ON p.id = m.provider_id "
            "    WHERE m.enabled = 1 AND p.enabled = 1"
            "  ), 0), "
            "  COALESCE((SELECT MAX(CAST(updated_at AS REAL)) FROM providers), 0)"
        )
        row = await cursor.fetchone()
        if not row:
            return 0.0, 0.0, 0, 0.0
        return (
            _as_epoch(row[0]), _as_epoch(row[1]),
            int(row[2] or 0), _as_epoch(row[3]),
        )

    # ── Per-Session Pin ──

    async def get_session_pin(self, session_id: str) -> str | None:
        """Return the pinned ``runtime_id`` for ``session_id``, or ``None``.

        When non-null, the dispatcher makes ``runtime_id`` the session's
        entry model directly, skipping the ``is_classifier`` default-leader
        flag and the first-enabled fallback. Pinned sessions ignore budget
        degradation too — an explicit user choice wins. The one exception
        is a pin to a model that is no longer enabled:
        ``_resolve_entry_model`` auto-heals it (drops the pin, logs
        ``router.pin_auto_heal``) rather than failing the turn.
        """
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT runtime_id FROM pinned_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        return str(row[0]) if row and row[0] else None

    async def pin_session_model(self, session_id: str, runtime_id: str) -> None:
        """Pin ``session_id`` to a specific model ``runtime_id``.

        With history unified in ``sessions`` across every
        framework, there is no framework lock anymore — any enabled
        runtime_id can be pinned to any session at any time.
        """
        from src.models.catalog import framework_of

        if not session_id or not runtime_id:
            raise ValueError("session_id and runtime_id are required")
        target_framework = framework_of(runtime_id)
        if target_framework not in LLM_FRAMEWORKS:
            raise ValueError(
                f"runtime_id {runtime_id!r} resolved to an unknown framework {target_framework!r}"
            )
        conn = await self._ensure_connected()
        await conn.execute(
            "INSERT INTO pinned_sessions (session_id, runtime_id, pinned_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "runtime_id = excluded.runtime_id, pinned_at = excluded.pinned_at",
            (session_id, runtime_id, time.time()),
        )
        await conn.commit()

    async def unpin_session_model(self, session_id: str) -> None:
        """Drop the per-session model pin.

        On the next turn the session resumes normal entry-model
        resolution: the ``is_classifier``-flagged default leader if one
        is set, else the catalog's first-enabled model.
        """
        conn = await self._ensure_connected()
        await conn.execute(
            "DELETE FROM pinned_sessions WHERE session_id = ?",
            (session_id,),
        )
        await conn.commit()

    # ── Session list / CRUD (sessions) ──

    @staticmethod
    def _parse_metadata(raw: str | None) -> dict:
        try:
            parsed = json.loads(raw or "{}")
        except (TypeError, ValueError):
            return {}
        # Some rows land here with metadata stored as a JSON-encoded string
        # of a dict (the runtime's ``serialize_session_json_fields`` re-encodes
        # ``json.dumps`` when handed a stringified metadata field). Without
        # this unwrap, every such row reports ``client_id == ""`` and gets
        # dropped by the per-handle filter in ``list_all_sessions`` —
        # ``/api/sessions`` returns empty, the desktop sidebar is blank.
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except (TypeError, ValueError):
                return {}
        return parsed if isinstance(parsed, dict) else {}

    async def list_all_sessions(
        self,
        client_id: str | None = None,
        *,
        limit: int = 50,
        exclude_child_origins: tuple[str, ...] = (),
    ) -> list[dict]:
        """Return every session in ``sessions``, ordered by
        ``updated_at`` descending. Metadata columns (client_id, title,
        model, framework) are extracted from the ``metadata`` JSON
        column when present.

        When ``client_id`` is provided, results are filtered to rows
        whose ``metadata.client_id`` either matches it directly OR is
        a device pubkey bound to the same user handle in
        ``network_devices`` (soft-fallback for legacy rows persisted
        before sessions were stamped with the handle).

        The filter runs in SQL so ``LIMIT`` is applied AFTER it — i.e.
        callers get up to ``limit`` matching rows rather than ``limit``
        unfiltered rows of which a handful happen to match. Without
        this, real chat sessions get crowded out of the top page by
        ``:classifier`` and ``workflow:`` rows whose ``updated_at`` is
        more recent. The two-level ``json_extract(json_extract(metadata,
        '$'), '$.client_id')`` works for both proper JSON-object rows
        and the double-encoded form the runtime's
        ``serialize_session_json_fields`` path produces.
        """
        conn = await self._ensure_connected()
        # Optionally drop child sessions of a given origin (the flat history
        # list hides ``delegation`` sub-agents — they're navigable only from
        # their parent's transcript card, never the sidebar). NULL-safe so
        # legacy chat rows (no ``origin``) are always kept. The parent's own
        # cards use ``list_child_sessions`` instead, which is unaffected.
        origin_expr = "json_extract(json_extract(metadata, '$'), '$.origin')"
        excl_clause = ""
        excl_params: list = []
        if exclude_child_origins:
            ph = ",".join("?" * len(exclude_child_origins))
            excl_clause = f" AND ({origin_expr} IS NULL OR {origin_expr} NOT IN ({ph}))"
            excl_params = list(exclude_child_origins)
        if client_id:
            legacy_pubkeys = await self._pubkeys_for_handle(client_id)
            candidates = [client_id, *sorted(legacy_pubkeys)]
            placeholders = ",".join("?" * len(candidates))
            # A row matches if it's owned by the handle directly, OR if its
            # ``parent_session_id`` points at a row the handle owns. The
            # second clause is the owner-inheritance fallback for child
            # sessions whose owner couldn't be stamped at spawn time — most
            # importantly coordinate-team members, spawned from a sync build
            # site that can't resolve the handle — so they still surface in
            # the right user's flat list, nested logically under their parent.
            cursor = await conn.execute(
                f"SELECT session_id, metadata, created_at, updated_at "
                f"FROM sessions "
                f"WHERE (json_extract(json_extract(metadata, '$'), '$.client_id') IN ({placeholders}) "
                f"   OR json_extract(json_extract(metadata, '$'), '$.parent_session_id') IN ("
                f"        SELECT session_id FROM sessions "
                f"        WHERE json_extract(json_extract(metadata, '$'), '$.client_id') IN ({placeholders})"
                f"   )){excl_clause} "
                f"ORDER BY updated_at DESC LIMIT ?",
                (*candidates, *candidates, *excl_params, int(limit)),
            )
        else:
            cursor = await conn.execute(
                f"SELECT session_id, metadata, created_at, updated_at "
                f"FROM sessions "
                f"WHERE 1=1{excl_clause} "
                f"ORDER BY updated_at DESC LIMIT ?",
                (*excl_params, int(limit)),
            )
        results: list[dict] = []
        for row in await cursor.fetchall():
            sid = row[0]
            meta = self._parse_metadata(row[1])
            results.append(self._session_row_to_summary(sid, meta, row[2], row[3]))
        return results

    @staticmethod
    def _session_row_to_summary(
        sid: str, meta: dict, created_at, updated_at,
    ) -> dict:
        """Shape one ``sessions`` row's metadata into the summary dict the
        gateway returns to the app. Shared by ``list_all_sessions`` and
        ``list_child_sessions`` so both surface the same fields, including
        the child-linkage (``parent_session_id`` / ``origin`` / ``kind``)
        the app uses for origin chips and the parent breadcrumb."""
        return {
            "session_id": sid,
            "client_id": meta.get("client_id", ""),
            "title": meta.get("title"),
            "model": meta.get("model"),
            "framework": meta.get("framework") or FRAMEWORK_API_BASED,
            "parent_session_id": meta.get("parent_session_id"),
            "origin": meta.get("origin") or "chat",
            "kind": meta.get("kind"),
            # The delegate-tool run_id this child corresponds to (team-member
            # child sessions only). Lets the parent transcript link a
            # delegate_task_to_member chip to its child session deterministically.
            "child_run_id": meta.get("child_run_id"),
            "created_at": created_at,
            "last_active_at": updated_at,
        }

    async def list_child_sessions(
        self,
        parent_session_id: str,
        *,
        limit: int = 200,
    ) -> list[dict]:
        """Return the sessions spawned by ``parent_session_id`` (delegated
        sub-agents, scheduled-task firings under a task root, or workflow
        AI-prompt nodes under a run root), ordered by ``updated_at`` desc.

        Powers the parent transcript's delegation cards and the ``?parent=``
        gateway query. Filters on ``metadata.parent_session_id`` with the
        same two-level ``json_extract`` form ``list_all_sessions`` uses so
        it matches both proper JSON-object rows and the double-encoded form
        the runtime's ``serialize_session_json_fields`` path produces."""
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT session_id, metadata, created_at, updated_at "
            "FROM sessions "
            "WHERE json_extract(json_extract(metadata, '$'), '$.parent_session_id') = ? "
            "ORDER BY updated_at DESC LIMIT ?",
            (parent_session_id, int(limit)),
        )
        return [
            self._session_row_to_summary(row[0], self._parse_metadata(row[1]), row[2], row[3])
            for row in await cursor.fetchall()
        ]

    async def prune_child_sessions(self, parent_session_id: str, *, keep: int) -> int:
        """Delete the oldest child sessions of ``parent_session_id`` beyond the
        most recent ``keep``. Opt-in retention for automation parents (a
        scheduled-task root, a workflow root) that would otherwise accumulate
        a firing/run session forever. ``keep <= 0`` is a no-op — child
        sessions are durable and navigable by default (vision §4/§16); this
        only trims when an operator sets a cap. Returns the number deleted."""
        if keep is None or keep <= 0:
            return 0
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT session_id FROM sessions "
            "WHERE json_extract(json_extract(metadata, '$'), '$.parent_session_id') = ? "
            "ORDER BY updated_at DESC",
            (parent_session_id,),
        )
        sids = [r[0] for r in await cursor.fetchall()]
        stale = sids[keep:]
        for sid in stale:
            await conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
            await self._project_operational_session(sid)
        if stale:
            await conn.commit()
        return len(stale)

    async def primary_owner_handle(self) -> str | None:
        """The agent's primary owner handle — the earliest active network
        user. Used as the owner for automation child sessions (scheduled-task
        firings, workflow nodes) that have no human parent, so those rows land
        in the owner's flat session list. Returns None on a handle-less /
        coordinator-less deployment (the rows then stay sidebar-hidden, which
        is the correct fallback rather than leaking to a wrong user)."""
        conn = await self._ensure_connected()
        try:
            cursor = await conn.execute(
                "SELECT handle FROM network_users WHERE status = 'active' "
                "ORDER BY created_at ASC LIMIT 1"
            )
            row = await cursor.fetchone()
        except Exception:
            return None
        return row[0] if row else None

    async def _pubkeys_for_handle(self, handle: str) -> set[str]:
        """Return every device pubkey (lowercase hex) bound to ``handle``.

        ``network_devices.device_pubkey`` is a BLOB written by the
        coordinator store; we hex-encode here so the comparison matches
        the ``device_pubkey_hex`` stamped by the auth middleware on
        each request. Returns an empty set when the table is missing
        (e.g. agent-only nodes that never paired through a coordinator)
        or when no devices are registered for that handle."""
        if not handle:
            return set()
        conn = await self._ensure_connected()
        try:
            cursor = await conn.execute(
                "SELECT device_pubkey FROM network_devices WHERE user_handle = ?",
                (handle,),
            )
            rows = await cursor.fetchall()
        except Exception:
            # Table absent on this DB shape — treat as no legacy
            # devices rather than blowing up the entire listing.
            return set()
        out: set[str] = set()
        for r in rows:
            raw = r[0]
            if isinstance(raw, (bytes, bytearray, memoryview)):
                out.add(bytes(raw).hex())
            elif isinstance(raw, str):
                out.add(raw.lower())
        return out

    async def upsert_session(
        self,
        session_id: str,
        *,
        client_id: str | None = None,
        title: str | None = None,
        model: str | None = None,
        framework: str | None = None,
        device_id: str | None = None,
        parent_session_id: str | None = None,
        origin: str | None = None,
        kind: str | None = None,
    ) -> None:
        """Create or update the ``sessions`` row, merging the display
        metadata into the ``metadata`` JSON column.

        ``client_id`` is the row's *owner* — preferably the user handle
        so the session list is cross-device — and is stored in the
        ``metadata`` JSON (which is what ``list_all_sessions`` keys off).
        ``device_id`` (when set) records which device first opened the
        session, so per-device routing (sticky-device retries, WS
        reconnect) still works.

        ``parent_session_id`` / ``origin`` / ``kind`` link a *child*
        session (a delegated sub-agent, a scheduled-task firing, or a
        workflow AI-prompt node) back to its parent and tag what spawned
        it. ``origin`` is one of ``chat | delegation | scheduler |
        workflow``; ``kind`` is the fine label (the delegated model id, a
        task id, ``workflow:node`` …). These are pure metadata, surfaced
        by ``list_all_sessions`` so the app can render an origin chip and
        a navigable parent breadcrumb — they never touch ``user_id``.

        **The gateway is metadata-only on this table — it must NOT write
        ``user_id``.** That column is owned exclusively by the runtime,
        which stamps the single stable ``RUNTIME_SESSION_USER_ID``
        ("openagent") sentinel on every session and gates BOTH its
        history read and its runs write on ``user_id == <that> OR
        user_id IS NULL`` (see ``src/memory/store/sqlite/sqlite.py``).
        Whenever the gateway stamped a *different* value here — the old
        ``'openagent'`` sentinel, then later the device handle /
        ``__bridge`` — that mismatch silently blocked the runtime from
        reading prior runs AND from persisting new ones, so the agent
        "forgot" the conversation every turn (the 2026-05 Telegram
        session-reset bug). Leaving ``user_id`` NULL lets the runtime
        claim and own the row; tenancy is carried by ``session_id``
        (e.g. ``tg:<uid>``), never by this column."""
        conn = await self._ensure_connected()
        now = int(time.time())
        existing = await (
            await conn.execute(
                "SELECT metadata FROM sessions WHERE session_id = ?",
                (session_id,),
            )
        ).fetchone()
        meta = self._parse_metadata(existing[0] if existing else None)
        if client_id:
            meta["client_id"] = client_id
        if device_id:
            meta["device_id"] = device_id
        if title:
            meta["title"] = title[:200]
        if model:
            meta["model"] = model
        if framework:
            meta["framework"] = framework
        if parent_session_id:
            meta["parent_session_id"] = parent_session_id
        if origin:
            meta["origin"] = origin
        if kind:
            meta["kind"] = kind

        if existing:
            # Metadata-only update — never touch ``user_id`` (runtime-owned).
            await conn.execute(
                "UPDATE sessions SET metadata = ?, updated_at = ? "
                "WHERE session_id = ?",
                (json.dumps(meta), now, session_id),
            )
        else:
            # INSERT with a NULL owner so the runtime can claim the row on
            # its first turn (its read/write WHERE accepts ``user_id IS NULL``).
            await conn.execute(
                "INSERT OR IGNORE INTO sessions "
                "(session_id, session_type, user_id, metadata, created_at, updated_at) "
                "VALUES (?, 'agent', NULL, ?, ?, ?)",
                (session_id, json.dumps(meta), now, now),
            )
        await self._project_operational_session(session_id)
        await conn.commit()

    async def delete_session(self, session_id: str) -> None:
        conn = await self._ensure_connected()
        await conn.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        await self._project_operational_session(session_id)
        await conn.commit()

    async def get_session(self, session_id: str) -> dict | None:
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT session_id, metadata, created_at, updated_at "
            "FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        meta = self._parse_metadata(row[1])
        return {
            "session_id": row[0],
            "client_id": meta.get("client_id", ""),
            "title": meta.get("title"),
            "model": meta.get("model"),
            "framework": meta.get("framework") or FRAMEWORK_API_BASED,
            # The child-linkage metadata, so a caller can tell a manual chat
            # (``origin == "chat"``/absent, no parent) from a spawned child
            # (delegation sub-agent, scheduled firing, workflow node) without a
            # second query. Mirrors ``_session_row_to_summary``.
            "parent_session_id": meta.get("parent_session_id"),
            "origin": meta.get("origin") or "chat",
            "kind": meta.get("kind"),
            "created_at": row[2],
            "last_active_at": row[3],
        }

    async def is_session_owned_by(
        self, session_id: str, handle: str, *, row: dict[str, Any] | None = None,
    ) -> bool:
        """Whether ``handle`` owns ``session_id``.

        Ownership is ``metadata.client_id`` being the handle directly, or a
        device pubkey bound to that handle — the SAME soft-fallback
        ``list_all_sessions`` uses, so a session a user can SEE in their list
        is exactly one they can delete. Used to authorise a destructive delete
        in multi-user / federated deploys. Returns False when the row is gone
        or carries no owner (the caller decides how to treat ambiguity).

        Pass ``row`` to reuse an already-fetched session row and skip the
        redundant ``get_session`` read (the delete path has it in hand)."""
        if not handle:
            return False
        if row is None:
            row = await self.get_session(session_id)
        if not row:
            return False
        owner = row.get("client_id") or ""
        if not owner:
            return False
        if owner == handle:
            return True
        return owner in await self._pubkeys_for_handle(handle)

    async def list_descendant_sessions(
        self,
        session_id: str,
        *,
        max_total: int = 5000,
    ) -> list[str]:
        """Every session spawned (transitively) by ``session_id``.

        Walks ``metadata.parent_session_id`` breadth-first so a manual chat's
        delegated sub-agents, *their* sub-agents (delegation nests to any
        depth, vision §4), and any in-chat ``run dream mode`` firing whose
        parent is the chat itself are all collected. The root is excluded.

        Used to cascade a chat-session delete to its whole sub-agent lineage.
        Genuine scheduled-task / workflow run sessions are never reached: their
        ``parent_session_id`` points at a *synthetic* root (``scheduler:<task>``
        / ``workflow:<wf>:<run>``), never at a real chat session id — so a chat
        delete can only ever sweep what that chat actually spawned. A
        ``seen`` set guards against pathological cycles; ``max_total`` bounds a
        runaway fan-out.
        """
        conn = await self._ensure_connected()
        collected: list[str] = []
        seen: set[str] = {session_id}
        queue: deque[str] = deque([session_id])
        while queue and len(collected) < max_total:
            parent = queue.popleft()
            cursor = await conn.execute(
                "SELECT session_id FROM sessions "
                # Double ``json_extract`` — the inner ``$`` normalises rows whose
                # metadata the runtime double-encoded (a JSON string of the dict)
                # so this matches BOTH shapes, exactly like ``list_child_sessions``
                # / ``list_all_sessions``. A single extract returns NULL for the
                # double-encoded form and would silently leave delegation
                # sub-agents orphaned after a chat delete.
                "WHERE json_extract(json_extract(metadata, '$'), '$.parent_session_id') = ?",
                (parent,),
            )
            for (child_sid,) in await cursor.fetchall():
                if child_sid in seen:
                    continue
                seen.add(child_sid)
                collected.append(child_sid)
                queue.append(child_sid)
        return collected

    # Per-session satellite tables that carry a ``session_id`` but have no
    # ON DELETE CASCADE (they are plain TEXT columns, not FKs). They hold data
    # *derived* from a conversation, so they must be cleared when the session
    # is permanently deleted — most importantly ``conversation_embeddings``, or
    # a deleted chat would keep resurfacing through memory-search.
    _SESSION_SATELLITE_TABLES: tuple[str, ...] = (
        "session_events",
        "pinned_sessions",
        "conversation_embeddings",
        "user_profiles",
        "vault_save_reminders",
    )

    async def purge_session(self, session_id: str) -> None:
        """Permanently delete a session row and all of its derived per-session
        data, in a single transaction.

        The ``sessions`` row holds the whole transcript (in its ``runs`` JSON),
        so deleting it removes the history; the satellite deletes then clear the
        derived records that would otherwise be orphaned (no FK cascade exists).
        Idempotent — safe to call for an id whose row is already gone (e.g. the
        runtime ``forget_session`` ran first) and tolerant of older DB shapes
        where a satellite table is absent. Child sessions are handled by the
        caller via :meth:`list_descendant_sessions`."""
        conn = await self._ensure_connected()
        await conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        await self._project_operational_session(session_id)
        for table in self._SESSION_SATELLITE_TABLES:
            try:
                await conn.execute(
                    f"DELETE FROM {table} WHERE session_id = ?", (session_id,)
                )
            except Exception as e:  # noqa: BLE001 — missing table / shape drift
                logger.debug(
                    "purge_session: %s cleanup failed for %s: %s",
                    table, session_id, e,
                )
        await conn.commit()

    # ── Session journal (append-only) ──
    #
    # Which types this build knows, and which of them are merely informational.
    # Taken from dsh's ``ignorable`` flag and the rule that goes with it: a
    # reader that meets an unknown type which is NOT ignorable must refuse to
    # reconstruct rather than quietly skip it and hand back a plausible,
    # incomplete history. Nothing reconstructs from this journal yet — but the
    # rule has to exist before the first consumer does, not after it has
    # already guessed.
    JOURNAL_KNOWN_TYPES: frozenset[str] = frozenset({
        "user/message", "assistant/message", "tool/status",
        "turn/end", "error", "compaction",
    })
    JOURNAL_IGNORABLE_TYPES: frozenset[str] = frozenset({
        # Progress chatter and accounting: losing one cannot change what the
        # conversation WAS.
        "tool/status", "compaction",
    })


    async def append_session_event(
        self, session_id: str, event_type: str, data: dict | None = None,
    ) -> int:
        """Append one fact to a session's journal and return its ``seq``.

        Best-effort by contract: journalling must never be the reason a turn
        fails, so a write error is logged and swallowed (returning 0). The
        journal is a witness, not a participant.

        ``seq`` is allocated as ``max(seq) + 1`` for the session inside the
        same statement, so two writers cannot mint the same number — the
        PRIMARY KEY would reject the loser anyway, and the retry lands on a
        fresh value.
        """
        if not session_id or not event_type:
            return 0
        payload = "{}"
        if data:
            try:
                payload = json.dumps(data, default=str)
            except (TypeError, ValueError):
                payload = json.dumps({"unserializable": True})
        conn = await self._ensure_connected()
        for _attempt in range(3):
            try:
                cursor = await conn.execute(
                    "INSERT INTO session_events (session_id, seq, ts_ms, type, data) "
                    "VALUES (?, COALESCE((SELECT MAX(seq) FROM session_events "
                    "WHERE session_id = ?), 0) + 1, ?, ?, ?) RETURNING seq",
                    (session_id, session_id, int(time.time() * 1000), event_type, payload),
                )
                row = await cursor.fetchone()
                await conn.commit()
                return int(row[0]) if row else 0
            except sqlite3.IntegrityError:
                continue  # lost the seq race — take the next number
            except Exception as e:  # noqa: BLE001
                logger.debug("session journal write failed (%s): %s", event_type, e)
                return 0
        return 0

    async def list_session_events(
        self, session_id: str, *, after_seq: int = 0, limit: int = 500,
    ) -> list[dict]:
        """Read a session's journal in order, from ``after_seq`` exclusive.

        This is what lets a client ask "what happened while I was away" and
        get facts instead of inferring from silence."""
        conn = await self._ensure_connected()
        try:
            cursor = await conn.execute(
                "SELECT seq, ts_ms, type, data FROM session_events "
                "WHERE session_id = ? AND seq > ? ORDER BY seq ASC LIMIT ?",
                (session_id, int(after_seq), max(1, min(int(limit), 2000))),
            )
            rows = await cursor.fetchall()
        except Exception as e:  # noqa: BLE001 — older DB without the table
            logger.debug("session journal read failed: %s", e)
            return []
        out: list[dict] = []
        for row in rows:
            try:
                data = json.loads(row["data"] or "{}")
            except (TypeError, ValueError):
                data = {}
            out.append({
                "seq": int(row["seq"]),
                "ts_ms": int(row["ts_ms"]),
                "type": str(row["type"]),
                "data": data if isinstance(data, dict) else {},
            })
        return out

    # ── Session runs (the runtime SqliteDb owns writes) ──
    #
    # ``sessions.runs`` is now written exclusively by the runtime's
    # ``SqliteDb`` (the native Agent's own storage path for api-based
    # runs). The manual ``add_session_run`` / ``commit_partial_session_run``
    # mirror writes that the retired adapters used are gone with the
    # inlined-runtime migration.
    #
    # ``list_session_runs`` below keeps the read-side surface stable so
    # the gateway's ``GET /api/sessions/{id}/runs`` endpoint can render
    # history regardless of which framework wrote the row.

    async def list_session_runs(
        self,
        session_id: str,
        *,
        limit: int = 20,
    ) -> list[dict]:
        conn = await self._ensure_connected()
        cursor = await conn.execute(
            "SELECT runs FROM sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if not row or not row[0]:
            return []
        try:
            runs = json.loads(row[0])
        except (TypeError, ValueError):
            return []
        # Same double-encoding shape as ``metadata`` — the runtime's
        # ``serialize_session_json_fields`` will store ``runs`` as a
        # JSON-encoded string of a JSON array if handed a stringified
        # value. Without this unwrap, every click on a session shows
        # an empty message list because ``json.loads`` returns a str,
        # ``isinstance(runs, list)`` is False, and we early-out to [].
        if isinstance(runs, str):
            try:
                runs = json.loads(runs)
            except (TypeError, ValueError):
                return []
        if not isinstance(runs, list):
            return []
        return list(reversed(runs[-limit:]))

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

    async def _migrate_peer_networks_join_type(self) -> None:
        """Add ``join_type`` to ``peer_networks`` (idempotent).

        Distinguishes networks joined via agent_login (no-password Iroh auth)
        from those joined via SRP-6a user login, so ``make_dialer_for_peer``
        can pick the right refresh path without a password.
        """
        assert self._conn is not None
        cursor = await self._conn.execute("PRAGMA table_info(peer_networks)")
        cols = {r[1] for r in await cursor.fetchall()}
        if "join_type" not in cols:
            await self._conn.execute(
                "ALTER TABLE peer_networks ADD COLUMN join_type TEXT NOT NULL DEFAULT 'user'"
            )
            await self._conn.commit()

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
