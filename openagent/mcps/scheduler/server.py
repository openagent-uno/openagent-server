"""Scheduler MCP server.

Exposes OpenAgent's scheduled-task database over MCP so the agent can
inspect, create, update and delete its own cron-scheduled prompts at
runtime, without relying on a separate operator CLI flow.

Transport: stdio (launched as a subprocess by MCPRegistry).
Storage: the same SQLite DB used by openagent.scheduler.Scheduler and
openagent.memory.db.MemoryDB. The DB path is read from the
OPENAGENT_DB_PATH env var — injected by the Agent at startup — falling
back to `./openagent.db` to match the default local runtime database.

Writes go straight to the `scheduled_tasks` table; the long-running
Scheduler loop picks up new/updated rows on its next CHECK_INTERVAL tick
(default 30s) because it re-queries `get_due_tasks()` each cycle. No
cross-process signalling is required.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any

import aiosqlite
from croniter import croniter
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Must match openagent.memory.db.SCHEMA_SQL exactly so a bare run against
# a fresh DB (no OpenAgent running yet) still works.
_SCHEMA_SQL = """
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
"""

_ALLOWED_UPDATE_COLUMNS = {
    "name",
    "cron_expression",
    "prompt",
    "enabled",
    "last_run",
    "next_run",
}


def _db_path() -> str:
    """Resolve the SQLite path for this MCP process.

    Precedence:
      1. OPENAGENT_DB_PATH env var (set by the Agent at launch).
      2. ./openagent.db relative to the current working directory — this
         matches the default local runtime database so a
         standalone `python -m openagent.mcps.scheduler.server` run still
         points at the same file.
    """
    return os.environ.get("OPENAGENT_DB_PATH") or "openagent.db"


# Single shared connection per MCP process. SQLite handles this fine
# thanks to WAL (the main OpenAgent process also opens WAL on the same
# file), and keeping one connection avoids per-call open/close overhead.
_conn_lock = asyncio.Lock()
_conn: aiosqlite.Connection | None = None


async def _get_conn() -> aiosqlite.Connection:
    global _conn
    async with _conn_lock:
        if _conn is None:
            path = _db_path()
            conn = await aiosqlite.connect(path)
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.executescript(_SCHEMA_SQL)
            await conn.commit()
            _conn = conn
            logger.info("scheduler MCP connected to %s", path)
        return _conn


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    d = dict(row)
    # Normalise bool + expose pretty timestamps for the agent's benefit.
    d["enabled"] = bool(d.get("enabled"))
    for ts_col in ("last_run", "next_run", "created_at", "updated_at"):
        val = d.get(ts_col)
        if isinstance(val, (int, float)):
            d[f"{ts_col}_iso"] = _iso(val)
    return d


def _iso(epoch: float) -> str:
    import datetime as _dt

    return _dt.datetime.fromtimestamp(epoch).isoformat(timespec="seconds")


def _validate_cron(expr: str) -> None:
    try:
        croniter(expr)
    except (ValueError, KeyError) as e:
        raise ValueError(f"Invalid cron expression {expr!r}: {e}")


def _next_run(expr: str, base: float | None = None) -> float:
    return croniter(expr, base or time.time()).get_next(float)


async def _resolve_task_id(conn: aiosqlite.Connection, task_id: str) -> str:
    """Accept either a full UUID or an 8-char prefix (matches the CLI UX)."""
    if not task_id:
        raise ValueError("task_id is required")
    cursor = await conn.execute(
        "SELECT id FROM scheduled_tasks WHERE id = ? OR id LIKE ? LIMIT 2",
        (task_id, f"{task_id}%"),
    )
    rows = await cursor.fetchall()
    if not rows:
        raise ValueError(f"No scheduled task matching id {task_id!r}")
    if len(rows) > 1:
        raise ValueError(
            f"Ambiguous task id prefix {task_id!r}: matches multiple tasks — "
            "use a longer prefix or the full UUID."
        )
    return rows[0][0]


# ── FastMCP server ──

mcp = FastMCP("scheduler")


@mcp.tool()
async def list_scheduled_tasks(enabled_only: bool = False) -> list[dict[str, Any]]:
    """List scheduled tasks stored in OpenAgent's DB.

    Each task has: id, name, cron_expression, prompt, enabled, last_run,
    next_run, plus ISO-formatted companions (last_run_iso, next_run_iso,
    created_at_iso, updated_at_iso) for readability.

    Args:
        enabled_only: when true, return only tasks with enabled=1.
    """
    conn = await _get_conn()
    if enabled_only:
        cursor = await conn.execute(
            "SELECT * FROM scheduled_tasks WHERE enabled = 1 ORDER BY next_run ASC"
        )
    else:
        cursor = await conn.execute(
            "SELECT * FROM scheduled_tasks ORDER BY created_at DESC"
        )
    rows = await cursor.fetchall()
    return [_row_to_dict(r) for r in rows]


@mcp.tool()
async def get_scheduled_task(task_id: str) -> dict[str, Any]:
    """Fetch a single scheduled task by id (full UUID or 8-char prefix)."""
    conn = await _get_conn()
    full_id = await _resolve_task_id(conn, task_id)
    cursor = await conn.execute(
        "SELECT * FROM scheduled_tasks WHERE id = ?", (full_id,)
    )
    row = await cursor.fetchone()
    if not row:
        raise ValueError(f"Task {task_id!r} not found")
    return _row_to_dict(row)


@mcp.tool()
async def create_scheduled_task(
    name: str,
    cron_expression: str,
    prompt: str,
) -> dict[str, Any]:
    """Create a new recurring task.

    The prompt will be fed to the agent on every cron tick. Cron is a
    standard 5-field expression (minute hour day month weekday), e.g.
    '0 9 * * *' for every day at 09:00 server time. Use the
    describe_cron tool first if you are unsure the expression is valid.
    """
    if not name or not name.strip():
        raise ValueError("name is required")
    if not prompt or not prompt.strip():
        raise ValueError("prompt is required")
    _validate_cron(cron_expression)

    conn = await _get_conn()
    task_id = str(uuid.uuid4())
    now = time.time()
    nr = _next_run(cron_expression, now)

    await conn.execute(
        "INSERT INTO scheduled_tasks "
        "(id, name, cron_expression, prompt, enabled, next_run, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 1, ?, ?, ?)",
        (task_id, name, cron_expression, prompt, nr, now, now),
    )
    await conn.commit()

    cursor = await conn.execute(
        "SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)
    )
    row = await cursor.fetchone()
    return _row_to_dict(row)  # type: ignore[arg-type]


@mcp.tool()
async def update_scheduled_task(
    task_id: str,
    name: str | None = None,
    cron_expression: str | None = None,
    prompt: str | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Partially update a scheduled task.

    Only the fields you pass are changed. Changing cron_expression also
    recomputes next_run so the Scheduler loop picks up the new cadence
    on its next tick.
    """
    conn = await _get_conn()
    full_id = await _resolve_task_id(conn, task_id)

    updates: dict[str, Any] = {}
    if name is not None:
        if not name.strip():
            raise ValueError("name cannot be empty")
        updates["name"] = name
    if prompt is not None:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty")
        updates["prompt"] = prompt
    if cron_expression is not None:
        _validate_cron(cron_expression)
        updates["cron_expression"] = cron_expression
        updates["next_run"] = _next_run(cron_expression)
    if enabled is not None:
        updates["enabled"] = 1 if enabled else 0
        # Re-arming an enabled task: make sure next_run points at the
        # next cron tick so it doesn't fire immediately on stale data.
        if enabled:
            cursor = await conn.execute(
                "SELECT cron_expression FROM scheduled_tasks WHERE id = ?",
                (full_id,),
            )
            row = await cursor.fetchone()
            if row:
                updates.setdefault("next_run", _next_run(row[0]))

    if not updates:
        raise ValueError(
            "No fields to update. Pass at least one of: name, "
            "cron_expression, prompt, enabled."
        )

    # Drop unknown columns as a safety net.
    updates = {k: v for k, v in updates.items() if k in _ALLOWED_UPDATE_COLUMNS}
    updates["updated_at"] = time.time()

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [full_id]
    await conn.execute(
        f"UPDATE scheduled_tasks SET {set_clause} WHERE id = ?", values
    )
    await conn.commit()

    cursor = await conn.execute(
        "SELECT * FROM scheduled_tasks WHERE id = ?", (full_id,)
    )
    row = await cursor.fetchone()
    return _row_to_dict(row)  # type: ignore[arg-type]


@mcp.tool()
async def delete_scheduled_task(task_id: str) -> dict[str, Any]:
    """Delete a scheduled task permanently.

    This cannot be undone. If you only want to stop it running, prefer
    update_scheduled_task with enabled=false.
    """
    conn = await _get_conn()
    full_id = await _resolve_task_id(conn, task_id)

    cursor = await conn.execute(
        "SELECT name FROM scheduled_tasks WHERE id = ?", (full_id,)
    )
    row = await cursor.fetchone()
    name = row[0] if row else ""

    await conn.execute("DELETE FROM scheduled_tasks WHERE id = ?", (full_id,))
    await conn.commit()
    return {"deleted": True, "id": full_id, "name": name}


@mcp.tool()
async def describe_cron(cron_expression: str, count: int = 3) -> dict[str, Any]:
    """Validate a cron expression and preview its next N fire times.

    Use this before create_scheduled_task when you are unsure the cron
    string is valid or want to double-check the cadence matches the
    user's intent.
    """
    _validate_cron(cron_expression)
    count = max(1, min(count, 20))
    base = time.time()
    it = croniter(cron_expression, base)
    upcoming: list[dict[str, Any]] = []
    for _ in range(count):
        nxt = it.get_next(float)
        upcoming.append({"epoch": nxt, "iso": _iso(nxt)})
    return {
        "cron_expression": cron_expression,
        "valid": True,
        "upcoming": upcoming,
    }


def main() -> None:
    """Entrypoint: run the FastMCP server over stdio."""
    logging.basicConfig(
        level=os.environ.get("OPENAGENT_SCHEDULER_MCP_LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    mcp.run()  # stdio transport by default


if __name__ == "__main__":
    main()
