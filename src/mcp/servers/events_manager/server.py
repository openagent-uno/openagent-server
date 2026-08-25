"""Events-manager MCP server.

Exposes OpenAgent's webhook Events over MCP so the agent can create, inspect,
modify, and delete its own events — and fire one on demand — at runtime,
exactly like the scheduler / workflow-manager MCPs do for their objects
(vision §15: the agent knows and reaches for its own levers).

Transport: stdio (launched as a subprocess by MCPPool).
Storage: the same SQLite DB as the main runtime; path from OPENAGENT_DB_PATH
(injected by the Agent — the ``events-manager`` name MUST be in
``resolve_default_entry``'s inject list, else this points at the wrong file).

Writes go straight to the ``events`` table; the main process serves inbound
webhooks and dispatches. For an on-demand fire, this inserts an
``event_deliveries`` row with ``claimed_at IS NULL`` — the main process's
Scheduler drain claims and dispatches it within ~2s (it cannot reach the live
agent from here).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

import aiosqlite
from mcp.server.fastmcp import FastMCP

from src.memory.db import SCHEMA_SQL, sqlite_busy_timeout_ms, sqlite_busy_timeout_s
from src.core.event_secret import make_secret_material, slugify, random_slug_suffix
from src.core.event_types import EVENT_TYPES, DEFAULT_TYPE, is_valid_type, public_types

logger = logging.getLogger(__name__)

_VALID_ACTION_KINDS = ("workflow", "scheduled_task", "prompt")


def _db_path() -> str:
    # Same rule as ``_common.db_path``: never the CWD (see that docstring —
    # under PyInstaller it means an empty database nobody notices).
    from src.mcp.servers._common import db_path as _shared_db_path

    return _shared_db_path()


_conn_lock = asyncio.Lock()
_conn: aiosqlite.Connection | None = None


async def _get_conn() -> aiosqlite.Connection:
    global _conn
    async with _conn_lock:
        if _conn is None:
            path = _db_path()
            conn = await aiosqlite.connect(path, timeout=sqlite_busy_timeout_s())
            conn.row_factory = aiosqlite.Row
            await conn.execute(f"PRAGMA busy_timeout = {sqlite_busy_timeout_ms()}")
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.executescript(SCHEMA_SQL)
            await _ensure_event_session_binding_schema(conn)
            await conn.commit()
            _conn = conn
            logger.info("events-manager MCP connected to %s", path)
        return _conn


async def _ensure_event_session_binding_schema(conn: aiosqlite.Connection) -> None:
    """Subset of MemoryDB's migration needed by this direct-SQL MCP server."""
    cursor = await conn.execute("PRAGMA table_info(events)")
    cols = {row[1] for row in await cursor.fetchall()}
    if "session_binding_enabled" not in cols:
        await conn.execute(
            "ALTER TABLE events ADD COLUMN "
            "session_binding_enabled INTEGER NOT NULL DEFAULT 0"
        )
    if "session_binding_path" not in cols:
        await conn.execute("ALTER TABLE events ADD COLUMN session_binding_path TEXT")
    await conn.execute(
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
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ev_session_bindings_session "
        "ON event_session_bindings(session_id)"
    )


def _public_event(row: aiosqlite.Row) -> dict[str, Any]:
    """Serialise an event WITHOUT the encrypted secret."""
    d = dict(row)
    d.pop("secret_enc", None)
    raw = d.pop("input_schema_json", None) or "[]"
    try:
        d["input_schema"] = json.loads(raw)
    except (TypeError, ValueError):
        d["input_schema"] = []
    d["enabled"] = bool(d.get("enabled"))
    d["session_binding_enabled"] = bool(d.get("session_binding_enabled"))
    d["webhook_path"] = f"/hooks/{d.get('slug')}"
    return d


async def _resolve_event_id(conn: aiosqlite.Connection, id_or_slug: str) -> str:
    """Accept a full id, an 8-char id prefix, or a slug."""
    if not id_or_slug:
        raise ValueError("event id or slug is required")
    cursor = await conn.execute(
        "SELECT id FROM events WHERE id = ? OR slug = ? OR id LIKE ? LIMIT 2",
        (id_or_slug, id_or_slug, f"{id_or_slug}%"),
    )
    rows = await cursor.fetchall()
    if not rows:
        raise ValueError(f"No event matching {id_or_slug!r}")
    if len(rows) > 1:
        raise ValueError(f"Ambiguous {id_or_slug!r}: matches multiple events")
    return rows[0][0]


async def _unique_slug(conn: aiosqlite.Connection, name: str, desired: str | None) -> str:
    slug = slugify(desired or name)
    for candidate in [slug] + [f"{slug}-{random_slug_suffix()}" for _ in range(6)]:
        cur = await conn.execute("SELECT 1 FROM events WHERE slug = ?", (candidate,))
        if await cur.fetchone() is None:
            return candidate
    return f"{slug}-{uuid.uuid4().hex[:6]}"


async def _current_session_binding(
    conn: aiosqlite.Connection, event_id: str,
) -> tuple[bool, str | None]:
    cur = await conn.execute(
        "SELECT session_binding_enabled, session_binding_path "
        "FROM events WHERE id = ?",
        (event_id,),
    )
    row = await cur.fetchone()
    if not row:
        return False, None
    return bool(row[0]), (row[1] or None)


# ── FastMCP server ──

mcp = FastMCP("events-manager")


@mcp.tool()
async def list_events(enabled_only: bool = False) -> list[dict[str, Any]]:
    """List webhook events. Each event binds an inbound trigger (name, type,
    input schema) to an action (a workflow, a scheduled task, or a chat
    prompt). The per-event secret is never returned — only ``secret_hint``."""
    conn = await _get_conn()
    if enabled_only:
        cur = await conn.execute("SELECT * FROM events WHERE enabled = 1 ORDER BY created_at DESC")
    else:
        cur = await conn.execute("SELECT * FROM events ORDER BY created_at DESC")
    return [_public_event(r) for r in await cur.fetchall()]


@mcp.tool()
async def get_event(id_or_slug: str) -> dict[str, Any]:
    """Fetch one event by id (full or 8-char prefix) or slug."""
    conn = await _get_conn()
    eid = await _resolve_event_id(conn, id_or_slug)
    cur = await conn.execute("SELECT * FROM events WHERE id = ?", (eid,))
    row = await cur.fetchone()
    if not row:
        raise ValueError(f"Event {id_or_slug!r} not found")
    return _public_event(row)


@mcp.tool()
async def list_event_types() -> list[dict[str, Any]]:
    """List the webhook ``type`` presets (generic, github, stripe, slack, …)
    with a short description and whether each verifies a provider signature."""
    return public_types()


@mcp.tool()
async def create_event(
    name: str,
    action_kind: str,
    action_ref: str | None = None,
    prompt_template: str | None = None,
    type: str = DEFAULT_TYPE,
    description: str | None = None,
    input_schema: list | None = None,
    model: str | None = None,
    session_binding_enabled: bool = False,
    session_binding_path: str | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Create a webhook event and return it WITH its one-time clear secret.

    Args:
        name: human-facing event name.
        action_kind: one of 'workflow', 'scheduled_task', 'prompt'.
        action_ref: the workflow id/name or scheduled-task id (required unless
            action_kind='prompt').
        prompt_template: for action_kind='prompt', the prompt to run — may
            reference the payload with {{payload.field}}.
        type: webhook preset ('generic', 'generic-hmac', 'github', 'stripe',
            'slack'). Controls signature verification + delivery de-dupe.
        input_schema: optional list of {name,type,required,description,path}
            field descriptors documenting the expected payload.
        model: optional model runtime_id to pin the run to.
        session_binding_enabled: when true, prompt-event deliveries with the
            same payload field value reuse the same internal OpenAgent event
            run session.
        session_binding_path: dot-path in the payload to use as the binding
            key, e.g. "id" or "ticket.id". Required when binding is enabled.
        enabled: whether the event accepts deliveries immediately.

    Returns the event plus ``secret`` (the clear per-event secret) — shown
    ONCE. The caller must save it; only a hint is retrievable afterwards.
    """
    if action_kind not in _VALID_ACTION_KINDS:
        raise ValueError(f"action_kind must be one of {_VALID_ACTION_KINDS}")
    if not is_valid_type(type):
        raise ValueError(f"unknown type {type!r}; use list_event_types")
    if action_kind in ("workflow", "scheduled_task") and not action_ref:
        raise ValueError(f"action_ref is required for action_kind={action_kind}")
    if action_kind == "prompt" and not (prompt_template and prompt_template.strip()):
        raise ValueError("prompt_template is required for action_kind='prompt'")
    binding_path = (session_binding_path or "").strip() or None
    if session_binding_enabled and not binding_path:
        raise ValueError("session_binding_path is required when session binding is enabled")

    conn = await _get_conn()

    # Validate the action target exists.
    if action_kind == "workflow":
        cur = await conn.execute(
            "SELECT id FROM workflow_tasks WHERE id = ? OR name = ? LIMIT 1",
            (action_ref, action_ref),
        )
        row = await cur.fetchone()
        if not row:
            raise ValueError(f"workflow {action_ref!r} not found")
        action_ref = row[0]
    elif action_kind == "scheduled_task":
        cur = await conn.execute(
            "SELECT id FROM scheduled_tasks WHERE id = ? OR id LIKE ? LIMIT 1",
            (action_ref, f"{action_ref}%"),
        )
        row = await cur.fetchone()
        if not row:
            raise ValueError(f"scheduled task {action_ref!r} not found")
        action_ref = row[0]

    slug = await _unique_slug(conn, name, None)
    clear, secret_enc, hint = make_secret_material(db_path=_db_path())
    event_id = str(uuid.uuid4())
    now = time.time()
    await conn.execute(
        "INSERT INTO events "
        "(id, name, slug, description, type, enabled, secret_enc, secret_hint, "
        " input_schema_json, action_kind, action_ref, prompt_template, model, "
        " session_binding_enabled, session_binding_path, "
        " rate_limit_per_min, max_payload_bytes, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 60, 262144, ?, ?)",
        (
            event_id, name, slug, description or None, type, 1 if enabled else 0,
            secret_enc, hint, json.dumps(input_schema or []), action_kind,
            action_ref or None, prompt_template or None, model or None,
            1 if session_binding_enabled else 0, binding_path, now, now,
        ),
    )
    await conn.commit()
    cur = await conn.execute("SELECT * FROM events WHERE id = ?", (event_id,))
    out = _public_event(await cur.fetchone())
    out["secret"] = clear
    out["note"] = "Save the secret now — it is shown only once."
    return out


@mcp.tool()
async def update_event(
    id_or_slug: str,
    name: str | None = None,
    description: str | None = None,
    type: str | None = None,
    enabled: bool | None = None,
    action_kind: str | None = None,
    action_ref: str | None = None,
    prompt_template: str | None = None,
    model: str | None = None,
    session_binding_enabled: bool | None = None,
    session_binding_path: str | None = None,
) -> dict[str, Any]:
    """Update an event's fields (only the ones you pass). Returns the event."""
    conn = await _get_conn()
    eid = await _resolve_event_id(conn, id_or_slug)
    sets: dict[str, Any] = {}
    if name is not None:
        sets["name"] = name
    if description is not None:
        sets["description"] = description
    if type is not None:
        if not is_valid_type(type):
            raise ValueError(f"unknown type {type!r}")
        sets["type"] = type
    if enabled is not None:
        sets["enabled"] = 1 if enabled else 0
    if action_kind is not None:
        if action_kind not in _VALID_ACTION_KINDS:
            raise ValueError(f"action_kind must be one of {_VALID_ACTION_KINDS}")
        sets["action_kind"] = action_kind
    if action_ref is not None:
        sets["action_ref"] = action_ref or None
    if prompt_template is not None:
        sets["prompt_template"] = prompt_template or None
    if model is not None:
        sets["model"] = model or None
    if session_binding_enabled is not None or session_binding_path is not None:
        current_enabled, current_path = await _current_session_binding(conn, eid)
        effective_enabled = (
            bool(session_binding_enabled)
            if session_binding_enabled is not None
            else current_enabled
        )
        effective_path = (
            session_binding_path.strip()
            if session_binding_path is not None
            else current_path
        ) or None
        if effective_enabled and not effective_path:
            raise ValueError("session_binding_path is required when session binding is enabled")
        if session_binding_enabled is not None:
            sets["session_binding_enabled"] = 1 if session_binding_enabled else 0
        if session_binding_path is not None:
            sets["session_binding_path"] = effective_path
    if not sets:
        raise ValueError("no fields to update")
    sets["updated_at"] = time.time()
    clause = ", ".join(f"{k} = ?" for k in sets)
    await conn.execute(f"UPDATE events SET {clause} WHERE id = ?", list(sets.values()) + [eid])
    await conn.commit()
    cur = await conn.execute("SELECT * FROM events WHERE id = ?", (eid,))
    return _public_event(await cur.fetchone())


@mcp.tool()
async def enable_event(id_or_slug: str) -> dict[str, Any]:
    """Enable an event (start accepting deliveries)."""
    return await update_event(id_or_slug, enabled=True)


@mcp.tool()
async def disable_event(id_or_slug: str) -> dict[str, Any]:
    """Disable an event (deliveries return 404 until re-enabled)."""
    return await update_event(id_or_slug, enabled=False)


@mcp.tool()
async def rotate_event_secret(id_or_slug: str) -> dict[str, Any]:
    """Generate a new secret for an event, invalidating the old one
    immediately. Returns the event plus the new clear ``secret`` (shown once)."""
    conn = await _get_conn()
    eid = await _resolve_event_id(conn, id_or_slug)
    clear, secret_enc, hint = make_secret_material(db_path=_db_path())
    await conn.execute(
        "UPDATE events SET secret_enc = ?, secret_hint = ?, updated_at = ? WHERE id = ?",
        (secret_enc, hint, time.time(), eid),
    )
    await conn.commit()
    cur = await conn.execute("SELECT * FROM events WHERE id = ?", (eid,))
    out = _public_event(await cur.fetchone())
    out["secret"] = clear
    return out


@mcp.tool()
async def delete_event(id_or_slug: str) -> dict[str, Any]:
    """Delete an event and its delivery history."""
    conn = await _get_conn()
    eid = await _resolve_event_id(conn, id_or_slug)
    await conn.execute("DELETE FROM events WHERE id = ?", (eid,))
    await conn.commit()
    return {"ok": True, "id": eid}


@mcp.tool()
async def list_event_deliveries(id_or_slug: str, limit: int = 20) -> list[dict[str, Any]]:
    """Recent delivery history for an event (newest first): status, source,
    linked run/session, timing."""
    conn = await _get_conn()
    eid = await _resolve_event_id(conn, id_or_slug)
    cur = await conn.execute(
        "SELECT * FROM event_deliveries WHERE event_id = ? ORDER BY started_at DESC LIMIT ?",
        (eid, int(limit)),
    )
    return [dict(r) for r in await cur.fetchall()]


@mcp.tool()
async def trigger_event(
    id_or_slug: str,
    payload: dict | None = None,
    wait: bool = True,
    timeout_s: int = 120,
) -> dict[str, Any]:
    """Fire an event now with an optional payload, out of band from any real
    webhook — for testing an event or acting on data the agent already has.

    This subprocess has no in-process runtime, so it enqueues an unclaimed
    ``event_deliveries`` row that the main process's Scheduler claims and
    dispatches within ~2s. With ``wait`` (default) it polls until the delivery
    reaches a terminal state and returns it; otherwise it returns immediately.
    """
    conn = await _get_conn()
    eid = await _resolve_event_id(conn, id_or_slug)
    delivery_id = str(uuid.uuid4())
    now = time.time()
    # claimed_at NULL → the Scheduler drain picks it up.
    await conn.execute(
        "INSERT INTO event_deliveries "
        "(id, event_id, source, status, payload_json, started_at, claimed_at) "
        "VALUES (?, ?, 'agent', 'received', ?, ?, NULL)",
        (delivery_id, eid, json.dumps(payload or {}), now),
    )
    await conn.execute("UPDATE events SET last_triggered_at = ? WHERE id = ?", (now, eid))
    await conn.commit()

    if not wait:
        return {"delivery_id": delivery_id, "status": "received"}

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        cur = await conn.execute(
            "SELECT * FROM event_deliveries WHERE id = ?", (delivery_id,),
        )
        row = await cur.fetchone()
        if row and row["status"] not in ("received", "running"):
            return dict(row)
        await asyncio.sleep(0.5)
    raise TimeoutError(
        f"delivery {delivery_id!r} did not finish within {timeout_s}s — it may "
        "still be running; check list_event_deliveries."
    )


@mcp.tool()
async def trigger_events(
    id_or_slug: str,
    payloads: list[dict],
) -> dict[str, Any]:
    """Fire the SAME event once per payload, in ONE call. Fire-and-forget.

    Perche' esiste: `trigger_event` ne accetta uno solo, e chi deve ri-iniettare N
    thread lo chiama N volte. Ogni chiamata a un tool e' un giro completo dal modello,
    con l'intero contesto rispedito: misurato sul task `support-coverage-delegated`,
    un ciclo con lavoro fa fino a **15 trigger separati** — 15 contesti — e il task
    e' il 45% del consumo di tutti gli scheduled task (31,76M token in 8 giorni,
    mediana 353k a giro). Le 15 righe da inserire sono le stesse: e' l'andata e ritorno
    dal modello a costare, non il lavoro.

    Un solo giro, una sola transazione, gli stessi delivery_id di prima. Niente `wait`:
    aspettare N delivery dentro una chiamata sola rimetterebbe il costo da un'altra
    parte, e chi ri-inietta vuole comunque proseguire — gli esiti si leggono dopo con
    `list_event_deliveries` sugli id restituiti.
    """
    if not payloads:
        return {"delivery_ids": [], "count": 0}
    conn = await _get_conn()
    eid = await _resolve_event_id(conn, id_or_slug)
    now = time.time()
    rows = [(str(uuid.uuid4()), eid, json.dumps(p or {}), now) for p in payloads]
    # Una transazione sola: o entrano tutte o nessuna. Con N insert separate un errore
    # a meta' lascerebbe una coda parziale che nessuno sa piu' ricostruire.
    await conn.executemany(
        "INSERT INTO event_deliveries "
        "(id, event_id, source, status, payload_json, started_at, claimed_at) "
        "VALUES (?, ?, 'agent', 'received', ?, ?, NULL)",
        rows,
    )
    await conn.execute("UPDATE events SET last_triggered_at = ? WHERE id = ?", (now, eid))
    await conn.commit()
    return {"delivery_ids": [r[0] for r in rows], "count": len(rows), "status": "received"}


# Fields a GUARDED change may write/rollback. Must match MemoryDB._GUARDED_EVENT_FIELDS.
_GUARDED_FIELDS = ("prompt_template", "model", "input_schema_json", "action_ref", "enabled")

# Baseline window (seconds before apply) for the failure-rate comparison.
_GUARDED_BASELINE_WINDOW = 1800.0


async def _ensure_guarded_changes_schema(conn: aiosqlite.Connection) -> None:
    """Idempotently ensure the guarded_changes table exists from this MCP path
    too (the main process migrates it at startup; this covers a first-boot race).
    Indexes + columns all reference only what this statement creates."""
    await conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS guarded_changes (
            id TEXT PRIMARY KEY, target_kind TEXT NOT NULL, target_id TEXT NOT NULL,
            field TEXT NOT NULL, old_value TEXT, new_value TEXT, metric_event_id TEXT,
            applied_at REAL NOT NULL, check_after REAL NOT NULL,
            baseline_fail_rate REAL, baseline_n INTEGER,
            status TEXT NOT NULL DEFAULT 'pending', after_fail_rate REAL,
            after_n INTEGER, resolved_at REAL, note TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_guarded_pending ON guarded_changes(status, check_after);
        CREATE INDEX IF NOT EXISTS idx_guarded_block ON guarded_changes(target_kind, target_id, field, status);
        """
    )
    await conn.commit()


@mcp.tool()
async def guarded_update_event(
    id_or_slug: str,
    field: str,
    new_value: str,
    watch_minutes: float = 10.0,
) -> dict[str, Any]:
    """Change one field of a webhook event UNDER GUARD — the ONLY sanctioned way
    for an autonomous fix to edit an event; never write the events table directly.

    It (1) REFUSES a change that was auto-rolled-back before (blocklist — so a
    broken fix is never re-applied), (2) SNAPSHOTS the old value, (3) applies the
    change, and (4) registers it so the scheduler's deterministic watcher measures
    this event's REAL delivery failure-rate over the next ``watch_minutes`` and
    AUTO-ROLLS-BACK (restoring the snapshot) + blocklists it if the rate regressed.
    So an auto-apply can safely fail: it self-heals and does not repeat.

    Args:
        id_or_slug: the event to change.
        field: one of prompt_template, model, input_schema_json, action_ref, enabled.
        new_value: the new value (a string; SQLite coerces for enabled).
        watch_minutes: how long the watcher observes before confirming (default 10).

    Returns the guarded-change record (id, status='pending', old/new, baseline).
    """
    if field not in _GUARDED_FIELDS:
        raise ValueError(f"field must be one of {_GUARDED_FIELDS}")
    conn = await _get_conn()
    await _ensure_guarded_changes_schema(conn)
    eid = await _resolve_event_id(conn, id_or_slug)

    # (1) blocklist — this exact change was auto-rolled-back before → refuse.
    cur = await conn.execute(
        "SELECT 1 FROM guarded_changes WHERE target_kind='event_field' AND target_id=? "
        "AND field=? AND new_value=? AND status='rolled_back' LIMIT 1",
        (eid, field, new_value),
    )
    if await cur.fetchone() is not None:
        raise ValueError(
            f"blocked: this exact change to {field!r} on event {eid} was previously "
            "auto-rolled-back for regressing delivery — it will not be re-applied. "
            "Propose it to a human instead."
        )

    # (2) snapshot old value.
    cur = await conn.execute(f"SELECT {field} AS v FROM events WHERE id=?", (eid,))
    row = await cur.fetchone()
    if row is None:
        raise ValueError(f"event {eid} not found")
    old_value = row["v"]
    if str(old_value) == str(new_value):
        return {"status": "noop", "reason": "new_value equals current value", "event_id": eid}

    now = time.time()
    # baseline: failure-rate in the window BEFORE the change.
    cur = await conn.execute(
        "SELECT status, COUNT(*) AS n FROM event_deliveries "
        "WHERE event_id=? AND started_at>=? AND started_at<? GROUP BY status",
        (eid, now - _GUARDED_BASELINE_WINDOW, now),
    )
    brows = await cur.fetchall()
    btotal = sum(r["n"] for r in brows)
    bfailed = sum(r["n"] for r in brows if r["status"] == "failed")
    baseline_rate = (bfailed / btotal) if btotal else 0.0

    # (3) apply + reset the breaker for a clean measurement window.
    await conn.execute(
        f"UPDATE events SET {field}=?, consecutive_failures=0, "
        f"breaker_tripped_at=NULL, last_failure_error=NULL, updated_at=? WHERE id=?",
        (new_value, now, eid),
    )
    # (4) record the pending guarded change for the watcher.
    cid = uuid.uuid4().hex
    await conn.execute(
        "INSERT INTO guarded_changes (id, target_kind, target_id, field, old_value, "
        "new_value, metric_event_id, applied_at, check_after, baseline_fail_rate, "
        "baseline_n, status, note) VALUES (?,?,?,?,?,?,?,?,?,?,?, 'pending', ?)",
        (cid, "event_field", eid, field, old_value, new_value, eid, now,
         now + max(1.0, watch_minutes) * 60.0, baseline_rate, btotal,
         f"guarded apply of {field}"),
    )
    await conn.commit()
    return {
        "id": cid,
        "status": "pending",
        "event_id": eid,
        "field": field,
        "old_value": old_value,
        "new_value": new_value,
        "baseline_fail_rate": round(baseline_rate, 3),
        "baseline_n": btotal,
        "watch_minutes": watch_minutes,
        "note": "applied under guard; watcher will auto-rollback + blocklist if delivery fail-rate regresses",
    }


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("OPENAGENT_EVENTS_MCP_LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    mcp.run()


if __name__ == "__main__":
    main()
