"""Scheduled tasks REST API — CRUD against the SQLite scheduler table.

GET    /api/scheduled-tasks              → { "tasks": [...] }
POST   /api/scheduled-tasks              → created task (201)
GET    /api/scheduled-tasks/{id}         → task | 404
PATCH  /api/scheduled-tasks/{id}         → updated task | 404
DELETE /api/scheduled-tasks/{id}         → { "ok": true, "id": "..." } | 404

All handlers operate on the same SQLite table the runtime Scheduler
reads from, so changes take effect within the scheduler's next tick
(~30s) without a process restart. Mirrors the tool signatures exposed
by openagent.mcp.servers.scheduler so the app, the CLI, and the agent's
own scheduler MCP all see identical data.

Read handlers use the agent's durable database directly, so definitions and
run history remain inspectable while background workers are intentionally
parked (for example in hermetic local E2E mode). Mutations and execution still
require the live ``Scheduler``: without it there is nothing to recompute
``next_run`` or reconcile enable-flips against, so the safe thing is to reject
writes rather than silently let rows drift.
"""

from __future__ import annotations

import asyncio
import time

from src.core.builtin_tasks import BUILTIN_TASK_NAMES
from src.core.execution_policy import (
    encode_execution_policy,
    normalize_execution_policy,
)
from src.core.logging import elog
from src.memory.schedule import decorate_scheduled_task, epoch_to_iso


def _resolve_scheduler(request):
    """Return (scheduler, error_response). error_response is None on success."""
    from aiohttp import web

    gw = request.app["gateway"]
    scheduler = getattr(gw, "_scheduler", None)
    if scheduler is None:
        return None, web.json_response(
            {"error": "Scheduler is not running"},
            status=503,
        )
    return scheduler, None


def _resolve_read_db(request):
    """Return the durable task DB without requiring a live scheduler.

    The scheduler and agent normally share one ``MemoryDB``. Safe/local
    inspection modes intentionally omit the worker while retaining the DB, so
    GET endpoints must not infer that durable task state is unavailable merely
    because nothing is currently executing it.
    """
    from aiohttp import web

    gw = request.app["gateway"]
    scheduler = getattr(gw, "_scheduler", None)
    db = getattr(scheduler, "db", None)
    if db is None:
        agent = getattr(gw, "agent", None) or getattr(gw, "_agent", None)
        db = getattr(agent, "memory_db", None)
    if db is None:
        return None, web.json_response(
            {"error": "No database configured"},
            status=503,
        )
    return db, None


def _is_builtin(row: dict | None) -> bool:
    return bool(row and row.get("name") in BUILTIN_TASK_NAMES)


async def _reject_if_builtin(scheduler, task_id: str):
    """Return (row, error_response). row is None on error.

    Built-in tasks (``dream-mode``, ``auto-update``)
    are seeded by ``AgentServer`` and managed via ``/api/config/<section>``;
    the gateway serves them read-only (list opt-in, get-by-id, run history)
    but returns 403 for every mutation. Centralised so any new handler added
    later inherits the policy without drift.
    """
    from aiohttp import web

    row = await scheduler.db.get_task(task_id)
    if row is None:
        return None, web.json_response(
            {"error": f"Task {task_id!r} not found"}, status=404,
        )
    if _is_builtin(row):
        return None, web.json_response(
            {"error": "Built-in tasks are managed via /api/config/<section>"},
            status=403,
        )
    return row, None


def _serialize(row: dict, *, running: bool = False) -> dict:
    out = decorate_scheduled_task(row)
    # Whether a firing of this task is in flight right now (``running`` or
    # ``cancelling``). Drives the tile's Run-now ↔ Stop control. Defaults
    # false so callers that don't pass it (e.g. create/update responses,
    # which can't be mid-firing) keep the same shape.
    out["running"] = running
    return out


def _serialize_run(row: dict) -> dict:
    """Add ISO mirrors for the epoch timestamp columns — matches the
    ``_decorate_run`` helper on the workflow-runs endpoint."""
    out = dict(row)
    for key in ("started_at", "finished_at"):
        epoch = out.get(key)
        out[f"{key}_iso"] = epoch_to_iso(epoch) if epoch else None
    return out


async def handle_list(request):
    from aiohttp import web

    db, err = _resolve_read_db(request)
    if err is not None:
        return err

    enabled_only = request.query.get("enabled_only", "").lower() in ("1", "true", "yes")
    # Built-in tasks (``dream-mode``, ``auto-update``) are hidden from the
    # default list — they're managed via ``/api/config/<section>``, not this
    # CRUD surface, so the Scheduled-tasks management screen stays clean. The
    # sidebar's "Recent" activity feed opts in with ``?include_builtin=1`` so a
    # dream-mode firing surfaces there like any other scheduled run — its runs
    # are recorded in the same run history (vision §7/§12). Opt-in exposes
    # READ only; mutations still 403 via ``_reject_if_builtin``.
    include_builtin = request.query.get("include_builtin", "").lower() in ("1", "true", "yes")
    rows = await db.get_tasks(enabled_only=enabled_only)
    if not include_builtin:
        rows = [r for r in rows if not _is_builtin(r)]
    # One query for the whole list tells each tile whether a firing is in
    # flight (so it can show a Stop control instead of Run now).
    scheduler = getattr(request.app["gateway"], "_scheduler", None)
    running = await db.running_task_ids() if scheduler is not None else set()
    return web.json_response(
        {"tasks": [_serialize(r, running=r["id"] in running) for r in rows]}
    )


async def handle_get(request):
    from aiohttp import web

    db, err = _resolve_read_db(request)
    if err is not None:
        return err

    task_id = request.match_info["id"]
    row = await db.get_task(task_id)
    if row is None:
        return web.json_response({"error": f"Task {task_id!r} not found"}, status=404)
    # Built-ins are READABLE by id — the activity feed's run screen and the
    # run-history title fetch resolve them here — but stay non-editable
    # (mutations 403 via ``_reject_if_builtin``).
    scheduler = getattr(request.app["gateway"], "_scheduler", None)
    running = await db.running_task_ids() if scheduler is not None else set()
    return web.json_response(_serialize(row, running=row["id"] in running))


async def handle_runs_list(request):
    """GET /api/scheduled-tasks/{id}/runs — per-firing execution history
    (newest first), the scheduled-task analogue of
    ``/api/workflows/{id}/runs``."""
    from aiohttp import web

    db, err = _resolve_read_db(request)
    if err is not None:
        return err

    task_id = request.match_info["id"]
    row = await db.get_task(task_id)
    if row is None:
        return web.json_response({"error": f"Task {task_id!r} not found"}, status=404)
    # A built-in's run history is readable — dream-mode firings are recorded in
    # ``task_runs`` like any scheduled run, and the sidebar feed / run screen
    # read them here. Only the schedule itself is non-editable.

    limit = int(request.query.get("limit", 20))
    status = request.query.get("status") or None
    runs = await db.list_task_runs(task_id, limit=limit, status=status)
    return web.json_response({"runs": [_serialize_run(r) for r in runs]})


async def handle_run(request):
    """POST /api/scheduled-tasks/{id}/run — fire a task now, out of band
    from its cron schedule. Body: ``{wait, timeout_s}``.

    Mirrors ``/api/workflows/{id}/run``: the gateway shares the scheduler's
    process, so it fast-paths by spawning the firing directly through the
    scheduler's bookkeeping (avoiding the ~2s request-drain tick the
    out-of-process MCP path incurs). The firing leaves the task's schedule
    and enabled flag untouched. When ``wait`` is true (default) it polls for
    completion and returns the ``task_runs`` row; otherwise it returns
    ``{run_id, status:'running'}`` once the run row exists.
    """
    from aiohttp import web

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    task_id = request.match_info["id"]
    existing, reject = await _reject_if_builtin(scheduler, task_id)
    if reject is not None:
        return reject

    try:
        body = await request.json() if request.can_read_body else {}
    except Exception:
        body = {}
    wait = body.get("wait", True)
    timeout_s = int(body.get("timeout_s", 300))

    # Fast path: spawn through the scheduler's tracked task set so concurrent
    # API calls each get their own handle. ``run_task`` mints its own run_id
    # and records the ``task_runs`` row.
    try:
        run_task = scheduler._spawn_workflow(
            scheduler.run_task(existing, trigger="manual")
        )
    except AttributeError:
        return web.json_response(
            {"error": "Scheduler has no run runtime attached"}, status=503,
        )

    gw = request.app["gateway"]
    elog("scheduled_task.run", id=task_id, name=existing.get("name", ""))
    # Flip the tile into a running state on subscribed clients.
    await gw.broadcast_resource("scheduled_task", "updated", task_id)

    if not wait:
        # Short-poll for the row ``run_task`` just opened so we can report
        # its run_id without blocking on the whole firing.
        deadline = time.monotonic() + 3
        latest = None
        while time.monotonic() < deadline:
            runs = await scheduler.db.list_task_runs(task_id, limit=1)
            if runs:
                latest = runs[0]
                break
            await asyncio.sleep(0.05)
        return web.json_response(
            {"run_id": latest["id"] if latest else None, "status": "running"},
            status=202,
        )

    # wait=True: run_task swallows task errors (records them on the row), so
    # awaiting it resolves once the firing reaches a terminal state.
    # ``asyncio.wait`` — NOT ``wait_for``, which cancels the awaited task when
    # the deadline passes. That cancellation killed a real production firing
    # because an HTTP client got bored: the run recorded "Stopped by user" and
    # 22 minutes of completed work were thrown away. A manual trigger is a
    # convenience for the caller; the firing itself belongs to the scheduler
    # and must outlive the request that started it.
    done, _pending = await asyncio.wait({run_task}, timeout=timeout_s)
    if not done:
        runs = await scheduler.db.list_task_runs(task_id, limit=1)
        return web.json_response(
            {
                "status": "running",
                "run_id": runs[0]["id"] if runs else None,
                "detail": (
                    f"still running after {timeout_s}s — left running, not cancelled. "
                    f"Poll /api/scheduled-tasks/{task_id}/runs for the outcome, "
                    f"or POST .../stop to end it deliberately."
                ),
            },
            status=202,
        )
    runs = await scheduler.db.list_task_runs(task_id, limit=1)
    if not runs:
        return web.json_response({"error": "run did not produce a row"}, status=500)
    # Run finished — re-broadcast so the tile flips the spinner off and the
    # "last run" badge picks up the new status.
    await gw.broadcast_resource("scheduled_task", "updated", task_id)
    return web.json_response(_serialize_run(runs[0]))


async def _await_task_runs_terminal(db, run_ids: list[str], *, timeout_s: int):
    """Poll ``task_runs`` until each id leaves ``running`` / ``cancelling``,
    or the deadline passes. The scheduler's cancellation drain is the writer;
    a fresh read each pass sees its latest committed status. Returns
    ``[{id, status}]`` for every requested id."""
    deadline = time.monotonic() + max(1, timeout_s)
    while True:
        statuses: dict[str, str] = {}
        for rid in run_ids:
            row = await db.get_task_run(rid)
            statuses[rid] = row["status"] if row else "unknown"
        pending = [rid for rid in run_ids if statuses.get(rid) in ("running", "cancelling")]
        if not pending or time.monotonic() >= deadline:
            return [{"id": rid, "status": statuses.get(rid, "unknown")} for rid in run_ids]
        await asyncio.sleep(0.4)


async def handle_stop(request):
    """POST /api/scheduled-tasks/{id}/stop — hard-stop the currently-running
    firing(s) of a task. Body: ``{wait, timeout_s}``.

    Flags each in-flight firing ``cancelling`` (the same DB-backed hand-off
    the agent's ``stop_scheduled_task`` MCP tool uses); the scheduler cancels
    them within ~2s — aborting the agent turn and any in-flight model call —
    and records each run ``cancelled``. This stops the *current run* only; it
    does not change the schedule. With ``wait`` true (default) it polls until
    the firing(s) actually stop so the response reflects the real outcome.
    """
    from aiohttp import web

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    task_id = request.match_info["id"]
    existing, reject = await _reject_if_builtin(scheduler, task_id)
    if reject is not None:
        return reject

    try:
        body = await request.json() if request.can_read_body else {}
    except Exception:
        body = {}
    wait = body.get("wait", True)
    timeout_s = int(body.get("timeout_s", 30))

    flagged = await scheduler.db.flag_task_runs_cancelling(task_id)
    gw = request.app["gateway"]
    # Reflect the cancelling state on subscribed clients right away.
    await gw.broadcast_resource("scheduled_task", "updated", task_id)

    if not flagged:
        return web.json_response({
            "task_id": task_id,
            "name": existing.get("name", ""),
            "stopped": [],
            "count": 0,
            "runs": [],
            "note": "No running firing to stop.",
        })

    elog("scheduled_task.stop", id=task_id, count=len(flagged))
    if wait:
        runs = await _await_task_runs_terminal(scheduler.db, flagged, timeout_s=timeout_s)
        # Re-broadcast so the tile drops its running/cancelling state.
        await gw.broadcast_resource("scheduled_task", "updated", task_id)
    else:
        runs = [{"id": rid, "status": "cancelling"} for rid in flagged]

    return web.json_response({
        "task_id": task_id,
        "name": existing.get("name", ""),
        "stopped": flagged,
        "count": len(flagged),
        "runs": runs,
    })


async def handle_create(request):
    from aiohttp import web
    from src.memory.schedule import (
        default_timezone_name,
        validate_schedule_expression,
        validate_timezone,
    )

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    name = (body.get("name") or "").strip()
    cron_expression = (body.get("cron_expression") or "").strip()
    prompt = (body.get("prompt") or "").strip()
    # Optional per-task model pin (a runtime_id). Empty/absent → default model.
    model = (body.get("model") or "").strip() or None
    try:
        execution_policy = normalize_execution_policy(body.get("execution_policy"))
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    if not name:
        return web.json_response({"error": "name is required"}, status=400)
    if not cron_expression:
        return web.json_response({"error": "cron_expression is required"}, status=400)
    if not prompt:
        return web.json_response({"error": "prompt is required"}, status=400)

    # Optional IANA zone the cron is read in. Absent → the agent-wide
    # default; absent there too → UTC, i.e. the behaviour every task created
    # before this field existed still has.
    try:
        if "timezone" in body:
            timezone = (body.get("timezone") or "").strip() or None
            validate_timezone(timezone)
        else:
            timezone = default_timezone_name()
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    try:
        validate_schedule_expression(cron_expression, timezone)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    if name in BUILTIN_TASK_NAMES:
        return web.json_response(
            {"error": f"name {name!r} is reserved for a built-in task"},
            status=400,
        )

    task_id = await scheduler.add_task(
        name, cron_expression, prompt, model=model, timezone=timezone,
        execution_policy=execution_policy,
    )

    # add_task enables by default; honour an explicit enabled=false.
    if body.get("enabled") is False:
        await scheduler.disable_task(task_id)

    row = await scheduler.db.get_task(task_id)
    from .operational import claim_created_resource

    await claim_created_resource(request, "scheduled_definition", task_id)
    elog("scheduled_task.create", id=task_id, name=name)
    gw = request.app["gateway"]
    await gw.broadcast_resource("scheduled_task", "created", task_id)
    return web.json_response(_serialize(row), status=201)


async def handle_update(request):
    from aiohttp import web
    from src.memory.schedule import validate_schedule_expression, validate_timezone

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    task_id = request.match_info["id"]
    existing, reject = await _reject_if_builtin(scheduler, task_id)
    if reject is not None:
        return reject

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    updates: dict = {}
    cron_changed = False

    if "name" in body:
        name = (body["name"] or "").strip()
        if not name:
            return web.json_response({"error": "name cannot be empty"}, status=400)
        updates["name"] = name

    if "prompt" in body:
        prompt = (body["prompt"] or "").strip()
        if not prompt:
            return web.json_response({"error": "prompt cannot be empty"}, status=400)
        updates["prompt"] = prompt

    if "model" in body:
        # Optional per-task model pin. An explicit empty string / null clears
        # it (firing reverts to the default/router model); a runtime_id sets it.
        updates["model"] = (body["model"] or "").strip() or None

    if "execution_policy" in body:
        try:
            updates["execution_policy_json"] = encode_execution_policy(
                body.get("execution_policy")
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    # Fall back to the task's stored zone so a cron edit keeps the zone and
    # a zone edit re-reads the stored cron.
    effective_tz = existing.get("timezone") or None
    if "timezone" in body:
        # Explicit null / "" clears the zone back to the UTC default; an IANA
        # name sets it.
        effective_tz = (body["timezone"] or "").strip() or None
        try:
            validate_timezone(effective_tz)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        updates["timezone"] = effective_tz
        # A zone change relocates every future fire, so it needs the same
        # next_run recompute a cron change gets.
        cron_changed = True

    if "cron_expression" in body:
        cron_expression = (body["cron_expression"] or "").strip()
        if not cron_expression:
            return web.json_response(
                {"error": "cron_expression cannot be empty"}, status=400
            )
        try:
            validate_schedule_expression(cron_expression, effective_tz)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        updates["cron_expression"] = cron_expression
        cron_changed = True

    enabled_change: bool | None = None
    if "enabled" in body:
        enabled_change = bool(body["enabled"])

    if not updates and enabled_change is None:
        return web.json_response(
            {"error": "No fields to update. Pass name, cron_expression, prompt, model, timezone, execution_policy, or enabled."},
            status=400,
        )

    # Apply field updates first. Use the db directly since scheduler has
    # no partial-update helper; we'll reconcile schedule-side state below.
    if updates:
        await scheduler.db.update_task(task_id, **updates)

    # Reconcile scheduler-side state: enable/disable flips and cron
    # changes both need next_run recomputed.
    if enabled_change is True:
        await scheduler.enable_task(task_id)  # also recomputes next_run
    elif enabled_change is False:
        await scheduler.disable_task(task_id)
    elif cron_changed:
        await scheduler.reschedule_task(task_id)

    row = await scheduler.db.get_task(task_id)
    elog(
        "scheduled_task.update",
        id=task_id,
        fields=list(updates.keys()) + (["enabled"] if enabled_change is not None else []),
    )
    gw = request.app["gateway"]
    await gw.broadcast_resource("scheduled_task", "updated", task_id)
    return web.json_response(_serialize(row))


async def handle_delete(request):
    from aiohttp import web

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    task_id = request.match_info["id"]
    existing, reject = await _reject_if_builtin(scheduler, task_id)
    if reject is not None:
        return reject

    await scheduler.remove_task(task_id)
    elog("scheduled_task.delete", id=task_id, name=existing.get("name", ""))
    gw = request.app["gateway"]
    await gw.broadcast_resource("scheduled_task", "deleted", task_id)
    return web.json_response({"ok": True, "id": task_id})
