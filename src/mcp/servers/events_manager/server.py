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

from src.memory.db import SCHEMA_SQL
from src.core.event_secret import make_secret_material, slugify, random_slug_suffix
from src.core.event_types import EVENT_TYPES, DEFAULT_TYPE, is_valid_type, public_types

logger = logging.getLogger(__name__)

_VALID_ACTION_KINDS = ("workflow", "scheduled_task", "prompt")


def _db_path() -> str:
    return os.environ.get("OPENAGENT_DB_PATH") or "openagent.db"


_conn_lock = asyncio.Lock()
_conn: aiosqlite.Connection | None = None


async def _get_conn() -> aiosqlite.Connection:
    global _conn
    async with _conn_lock:
        if _conn is None:
            path = _db_path()
            conn = await aiosqlite.connect(path)
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA busy_timeout = 10000")
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.executescript(SCHEMA_SQL)
            await conn.commit()
            _conn = conn
            logger.info("events-manager MCP connected to %s", path)
        return _conn


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
        " rate_limit_per_min, max_payload_bytes, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 60, 262144, ?, ?)",
        (
            event_id, name, slug, description or None, type, 1 if enabled else 0,
            secret_enc, hint, json.dumps(input_schema or []), action_kind,
            action_ref or None, prompt_template or None, model or None, now, now,
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


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("OPENAGENT_EVENTS_MCP_LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    mcp.run()


if __name__ == "__main__":
    main()
