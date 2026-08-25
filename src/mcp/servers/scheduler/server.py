"""Scheduler MCP server.

Exposes OpenAgent's scheduled-task database over MCP so the agent can
inspect, create, update and delete its own cron-scheduled prompts at
runtime, without relying on a separate operator CLI flow.

Transport: stdio (launched as a subprocess by MCPPool).
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
import uuid
from typing import Any

import aiosqlite
from mcp.server.fastmcp import FastMCP
from src.memory.db import SCHEMA_SQL, sqlite_busy_timeout_ms, sqlite_busy_timeout_s
from src.memory.schedule import (
    build_one_shot_expression,
    decorate_scheduled_task,
    default_timezone_name,
    epoch_to_iso,
    is_one_shot_expression,
    next_run_for_expression,
    resolve_timezone,
    validate_schedule_expression,
    validate_timezone,
)
import time

logger = logging.getLogger(__name__)

_ALLOWED_UPDATE_COLUMNS = {
    "name",
    "cron_expression",
    "prompt",
    "enabled",
    "last_run",
    "next_run",
    "model",
    "timezone",
}


def _db_path() -> str:
    """Resolve the SQLite path for this MCP process.

    Precedence:
      1. OPENAGENT_DB_PATH env var (set by the Agent at launch).
      2. ./openagent.db relative to the current working directory — this
         matches the default local runtime database so a
         standalone `python -m openagent.mcp.servers.scheduler.server` run still
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
            conn = await aiosqlite.connect(path, timeout=sqlite_busy_timeout_s())
            conn.row_factory = aiosqlite.Row
            await conn.execute(f"PRAGMA busy_timeout = {sqlite_busy_timeout_ms()}")
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.executescript(SCHEMA_SQL)
            await conn.commit()
            _conn = conn
            logger.info("scheduler MCP connected to %s", path)
        return _conn


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    return decorate_scheduled_task(row)


def _iso(epoch: float, timezone: str | None = None) -> str:
    return epoch_to_iso(epoch, timezone)


def _validate_cron(expr: str, timezone: str | None = None) -> None:
    validate_schedule_expression(expr, timezone)


def _next_run(expr: str, base: float | None = None, timezone: str | None = None) -> float:
    return next_run_for_expression(expr, base, timezone)


def _resolve_new_task_tz(timezone: str | None) -> str | None:
    """Zone to stamp on a task the agent is creating right now.

    An explicit argument wins; otherwise the agent-wide default. Absent
    both, None — UTC, i.e. what this tool did before timezones
    existed. Validated here so a hallucinated zone ("Europe/Roma") comes
    back to the model as a tool error it can correct, rather than silently
    scheduling against the wrong clock."""
    if timezone is not None and str(timezone).strip():
        validate_timezone(timezone)
        return str(timezone).strip()
    return default_timezone_name()


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
    model: str | None = None,
    timezone: str | None = None,
) -> dict[str, Any]:
    """Create a new recurring task.

    The prompt will be fed to the agent on every cron tick. Cron is a
    standard 5-field expression (minute hour day month weekday), e.g.
    '0 9 * * *' for every day at 09:00. Use the describe_cron tool first
    if you are unsure the expression is valid.

    Use this ONLY for repeating schedules. If the user wants something
    to happen once, use create_one_shot_task instead.

    ``model`` (optional) pins the firing to a specific model — a runtime_id
    such as 'anthropic:claude-opus-4-8'. Omit it to run the task on the
    agent's default/router model, like a normal chat turn.

    ``timezone`` (optional) is the IANA zone the cron is read in, e.g.
    'Europe/Rome'. Pass it whenever the user states a time of day — write
    the hour they said and name their zone ('0 9 * * *' + 'Europe/Rome'),
    do NOT convert the hour to UTC yourself: a converted hour is silently
    wrong for half the year, because it cannot follow daylight-saving.
    Omitted, the task uses the agent's configured default zone, or UTC if
    none is set.
    """
    if not name or not name.strip():
        raise ValueError("name is required")
    if not prompt or not prompt.strip():
        raise ValueError("prompt is required")
    tz = _resolve_new_task_tz(timezone)
    _validate_cron(cron_expression, tz)

    conn = await _get_conn()
    task_id = str(uuid.uuid4())
    now = time.time()
    nr = _next_run(cron_expression, now, tz)

    await conn.execute(
        "INSERT INTO scheduled_tasks "
        "(id, name, cron_expression, prompt, enabled, next_run, model, timezone, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)",
        (task_id, name, cron_expression, prompt, nr, (model or None), tz, now, now),
    )
    await conn.commit()

    cursor = await conn.execute(
        "SELECT * FROM scheduled_tasks WHERE id = ?", (task_id,)
    )
    row = await cursor.fetchone()
    return _row_to_dict(row)  # type: ignore[arg-type]


@mcp.tool()
async def create_one_shot_task(
    name: str,
    prompt: str,
    delay_seconds: int | None = None,
    run_at_iso: str | None = None,
    model: str | None = None,
    timezone: str | None = None,
) -> dict[str, Any]:
    """Create a task that runs exactly once.

    Prefer this when the user says things like "once", "one time",
    "in 10 minutes", or gives a specific future timestamp.

    Pass exactly one of:
      - delay_seconds: seconds from now
      - run_at_iso: absolute timestamp like 2026-04-14T09:30:00

    ``model`` (optional) pins the firing to a specific model (a runtime_id);
    omit to use the agent's default/router model.

    ``timezone`` (optional, IANA e.g. 'Europe/Rome') says which clock a
    *bare* run_at_iso is read on. It is only a way to read the input: the
    firing instant is stored as an absolute epoch and never re-interpreted
    afterwards. An offset already in run_at_iso ('2026-04-14T09:30:00+02:00')
    wins outright, and delay_seconds is relative to now, so neither is
    affected by this argument.
    """
    if not name or not name.strip():
        raise ValueError("name is required")
    if not prompt or not prompt.strip():
        raise ValueError("prompt is required")
    if (delay_seconds is None) == (run_at_iso is None):
        raise ValueError("Pass exactly one of delay_seconds or run_at_iso")

    tz = _resolve_new_task_tz(timezone)
    now = time.time()
    if delay_seconds is not None:
        run_at = now + max(1, int(delay_seconds))
    else:
        import datetime as _dt

        try:
            parsed = _dt.datetime.fromisoformat(str(run_at_iso))
        except ValueError as exc:
            raise ValueError(f"Invalid run_at_iso value: {run_at_iso!r}") from exc
        if parsed.tzinfo is None:
            # "09:30" is a wall-clock reading and needs a clock. Prefer the
            # named/default zone; fall back to the host's, which is what a
            # bare fromisoformat().timestamp() meant before timezones existed.
            zone = resolve_timezone(tz)
            if zone is not None:
                parsed = parsed.replace(tzinfo=zone)
        run_at = parsed.timestamp()
    if run_at <= now:
        raise ValueError("One-shot task must be scheduled in the future")

    conn = await _get_conn()
    task_id = str(uuid.uuid4())
    cron_expression = build_one_shot_expression(run_at)
    await conn.execute(
        "INSERT INTO scheduled_tasks "
        "(id, name, cron_expression, prompt, enabled, next_run, model, timezone, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)",
        (task_id, name, cron_expression, prompt, run_at, (model or None), tz, now, now),
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
    model: str | None = None,
    timezone: str | None = None,
) -> dict[str, Any]:
    """Partially update a scheduled task.

    Only the fields you pass are changed. Changing cron_expression also
    recomputes next_run so the Scheduler loop picks up the new cadence
    on its next tick.

    ``model`` sets the per-task model pin (a runtime_id). Pass an empty
    string to clear it so the task reverts to the default/router model.

    ``timezone`` sets the IANA zone the cron is read in (e.g.
    'Europe/Rome'); pass an empty string to clear it back to the server's
    own clock. This moves when the task fires — the same expression under
    a new zone is a different instant — so only set it when the user is
    telling you which clock they meant.
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
    if model is not None:
        # Empty string clears the pin (→ default model); any other value sets it.
        updates["model"] = model.strip() or None

    # The stored row is the baseline for anything the caller didn't pass —
    # a cron edit must keep the task's existing zone, and a zone edit must
    # re-read the task's existing cron.
    cursor = await conn.execute(
        "SELECT cron_expression, timezone FROM scheduled_tasks WHERE id = ?",
        (full_id,),
    )
    current = await cursor.fetchone()
    current_cron = current[0] if current else None
    effective_tz = current[1] if current else None

    if timezone is not None:
        # Empty string clears the zone (→ the UTC default); any other sets it.
        effective_tz = timezone.strip() or None
        validate_timezone(effective_tz)
        updates["timezone"] = effective_tz

    if cron_expression is not None:
        _validate_cron(cron_expression, effective_tz)
        updates["cron_expression"] = cron_expression

    # Recompute next_run whenever the *meaning* of the schedule changed.
    # A zone edit alone changes it just as much as a cron edit does — the
    # same expression on a different clock is a different instant — so
    # skipping this would leave the task firing on the old zone until its
    # next tick happened to rewrite next_run.
    reschedule_cron = cron_expression if cron_expression is not None else current_cron
    if (cron_expression is not None or timezone is not None) and reschedule_cron:
        updates["next_run"] = _next_run(reschedule_cron, None, effective_tz)
    if enabled is not None:
        updates["enabled"] = 1 if enabled else 0
        # Re-arming an enabled task: make sure next_run points at the
        # next cron tick so it doesn't fire immediately on stale data.
        if enabled and reschedule_cron:
            updates.setdefault(
                "next_run", _next_run(reschedule_cron, None, effective_tz),
            )

    if not updates:
        raise ValueError(
            "No fields to update. Pass at least one of: name, "
            "cron_expression, prompt, enabled, model, timezone."
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
async def stop_scheduled_task(
    task_id: str,
    wait: bool = True,
    timeout_s: int = 30,
) -> dict[str, Any]:
    """Completely stop the currently-running firing(s) of a scheduled task.

    A scheduled task fires a full agent turn on its cron tick. This flags
    any in-flight firing of the task for cancellation; the main OpenAgent
    process hard-stops it within a couple of seconds — aborting the agent
    turn and any in-flight model call — and records the run as ``cancelled``.
    Same DB-backed hand-off as the other scheduler tools: this subprocess
    only signals intent and (optionally) waits for the main process to act.

    - ``task_id``: the scheduled task (full UUID or 8-char prefix).
    - ``wait`` (default True): poll until the firing(s) actually stop, so the
      return reflects the real outcome. ``wait=False`` returns immediately.

    Returns ``{task_id, name, stopped: [run_id...], count, runs: [{id,
    status}], note}``. ``count`` is how many firings were flagged; ``0``
    means nothing was running.

    This stops the *current run* only — it does not affect the schedule. To
    stop the task from firing again, use ``update_scheduled_task(
    enabled=false)`` (reversible) or ``delete_scheduled_task`` (permanent).
    """
    conn = await _get_conn()
    full_id = await _resolve_task_id(conn, task_id)

    cursor = await conn.execute(
        "SELECT name FROM scheduled_tasks WHERE id = ?", (full_id,)
    )
    name_row = await cursor.fetchone()
    name = name_row[0] if name_row else ""

    cursor = await conn.execute(
        "SELECT id FROM task_runs WHERE task_id = ? AND status = 'running'",
        (full_id,),
    )
    target_ids = [r["id"] for r in await cursor.fetchall()]

    if not target_ids:
        return {
            "task_id": full_id,
            "name": name,
            "stopped": [],
            "count": 0,
            "runs": [],
            "note": f"No running firing to stop for task {name or full_id!r}.",
        }

    # The ``status='running'`` guard keeps the transition idempotent and
    # avoids clobbering a firing that finished between the SELECT and here.
    placeholders = ",".join("?" for _ in target_ids)
    await conn.execute(
        f"UPDATE task_runs SET status = 'cancelling' "
        f"WHERE id IN ({placeholders}) AND status = 'running'",
        target_ids,
    )
    await conn.commit()

    if wait:
        runs = await _await_task_runs_terminal(conn, target_ids, timeout_s=timeout_s)
    else:
        runs = [{"id": rid, "status": "cancelling"} for rid in target_ids]

    return {
        "task_id": full_id,
        "name": name,
        "stopped": target_ids,
        "count": len(target_ids),
        "runs": runs,
        "note": (
            f"Requested stop of {len(target_ids)} firing(s); the main "
            "process cancels them within ~2s."
        ),
    }


async def _await_task_runs_terminal(
    conn: aiosqlite.Connection, run_ids: list[str], *, timeout_s: int,
) -> list[dict[str, Any]]:
    """Poll ``task_runs`` until each id leaves ``running`` / ``cancelling``,
    or the deadline passes. The main process is the writer (cross-process,
    WAL); a fresh SELECT each pass sees its latest committed status. Returns
    ``[{id, status}]`` with the latest status for every requested id."""
    deadline = time.monotonic() + max(1, timeout_s)
    placeholders = ",".join("?" for _ in run_ids)
    while True:
        cursor = await conn.execute(
            f"SELECT id, status FROM task_runs WHERE id IN ({placeholders})",
            run_ids,
        )
        statuses = {r["id"]: r["status"] for r in await cursor.fetchall()}
        pending = [
            rid for rid in run_ids
            if statuses.get(rid) in ("running", "cancelling")
        ]
        if not pending or time.monotonic() >= deadline:
            return [
                {"id": rid, "status": statuses.get(rid, "unknown")}
                for rid in run_ids
            ]
        await asyncio.sleep(0.4)


def _serialize_task_run(row: aiosqlite.Row) -> dict[str, Any]:
    """Hydrate a ``task_runs`` row with ISO mirrors for its epoch columns —
    matches the gateway's ``_serialize_run`` so callers see one shape."""
    out = dict(row)
    for key in ("started_at", "finished_at"):
        epoch = out.get(key)
        out[f"{key}_iso"] = _iso(epoch) if epoch else None
    return out


async def _poll_task_request(
    conn: aiosqlite.Connection, request_id: str, *, timeout_s: float,
) -> str:
    """Wait for the main process to claim the request and attach a run_id."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        cursor = await conn.execute(
            "SELECT run_id FROM task_run_requests WHERE id = ?", (request_id,),
        )
        row = await cursor.fetchone()
        if row and row["run_id"]:
            return row["run_id"]
        await asyncio.sleep(0.25)
    raise TimeoutError(
        f"run-now request {request_id!r} was not picked up within "
        f"{timeout_s:.0f}s — is the main OpenAgent process running?"
    )


@mcp.tool()
async def run_scheduled_task_now(
    task_id: str,
    wait: bool = True,
    timeout_s: int = 300,
) -> dict[str, Any]:
    """Run a scheduled task immediately, out of band from its cron schedule.

    Fires the task's prompt right now. This does NOT touch the task's
    schedule — its next cron run is unchanged — and the task does not need
    to be enabled, so this also works to test or one-off a disabled task.

    Use this when the user wants a scheduled task to happen "now" /
    "right away" instead of waiting for its next tick. To change *when* it
    recurs, edit ``cron_expression`` instead; to run something brand new
    once, use ``create_one_shot_task``.

    This subprocess has no in-process Scheduler, so it drops a request row
    that the main OpenAgent process claims and executes within a couple of
    seconds. With ``wait=True`` (default) it polls until the firing
    finishes and returns the ``task_runs`` row (status / output preview /
    timing); ``wait=False`` returns as soon as the run is assigned an id.
    """
    conn = await _get_conn()
    full_id = await _resolve_task_id(conn, task_id)

    cursor = await conn.execute(
        "SELECT name FROM scheduled_tasks WHERE id = ?", (full_id,)
    )
    name_row = await cursor.fetchone()
    name = name_row["name"] if name_row else ""

    req_id = str(uuid.uuid4())
    await conn.execute(
        "INSERT INTO task_run_requests (id, task_id, trigger, created_at) "
        "VALUES (?, ?, 'ai', ?)",
        (req_id, full_id, time.time()),
    )
    await conn.commit()

    # Wait for the main process to claim + attach a run_id (bounded so a
    # dead main process surfaces a clear error instead of hanging).
    run_id = await _poll_task_request(conn, req_id, timeout_s=min(timeout_s, 60))
    if not wait:
        return {
            "task_id": full_id,
            "name": name,
            "run_id": run_id,
            "status": "running",
        }

    # Then wait for the firing itself to finish.
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        cursor = await conn.execute(
            "SELECT * FROM task_runs WHERE id = ?", (run_id,),
        )
        run_row = await cursor.fetchone()
        if run_row and run_row["status"] != "running":
            return _serialize_task_run(run_row)
        await asyncio.sleep(0.5)
    raise TimeoutError(
        f"task run {run_id!r} did not finish within {timeout_s}s — it may "
        "still be running; check list_scheduled_tasks / the run history."
    )


@mcp.tool()
async def describe_cron(
    cron_expression: str,
    count: int = 3,
    timezone: str | None = None,
) -> dict[str, Any]:
    """Validate a cron expression and preview its next N fire times.

    Use this before create_scheduled_task when you are unsure the cron
    string is valid or want to double-check the cadence matches the
    user's intent.

    ``timezone`` (optional, IANA e.g. 'Europe/Rome') previews the
    expression on that clock — pass the same value you intend to give
    create_scheduled_task. Omitted, it previews the agent's default zone,
    or UTC if none is configured. The returned ``iso``
    times are rendered on the previewed clock, and the preview walks the
    real scheduler, so daylight-saving is reflected here exactly as it
    will be when the task fires.
    """
    tz = _resolve_new_task_tz(timezone)
    _validate_cron(cron_expression, tz)
    count = max(1, min(count, 20))
    # Step the same function the Scheduler steps, rather than a private
    # croniter walk: a preview that doesn't share the DST rules is a
    # preview that lies exactly when it matters. (The old private walk
    # never ran at all — croniter was not imported in this module.)
    base = time.time()
    upcoming: list[dict[str, Any]] = []
    for _ in range(count):
        nxt = _next_run(cron_expression, base, tz)
        upcoming.append({"epoch": nxt, "iso": _iso(nxt, tz)})
        if is_one_shot_expression(cron_expression):
            break  # fires once; stepping past it would loop on one instant
        base = nxt
    return {
        "cron_expression": cron_expression,
        "timezone": tz,
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
