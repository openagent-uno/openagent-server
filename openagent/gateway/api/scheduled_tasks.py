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

503 is returned when the scheduler isn't running (scheduler.enabled=false
in config). In that case there is no live Scheduler instance to
recompute next_run / reconcile enable-flips against, so the safe thing
is to reject writes rather than silently let rows drift.
"""

from __future__ import annotations

from openagent.core.logging import elog
from openagent.memory.schedule import decorate_scheduled_task


def _resolve_scheduler(request):
    """Return (scheduler, error_response). error_response is None on success."""
    from aiohttp import web

    gw = request.app["gateway"]
    scheduler = getattr(gw, "_scheduler", None)
    if scheduler is None:
        return None, web.json_response(
            {"error": "Scheduler is not running (scheduler.enabled is false)"},
            status=503,
        )
    return scheduler, None


def _serialize(row: dict) -> dict:
    return decorate_scheduled_task(row)


async def handle_list(request):
    from aiohttp import web

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    enabled_only = request.query.get("enabled_only", "").lower() in ("1", "true", "yes")
    rows = await scheduler.db.get_tasks(enabled_only=enabled_only)
    return web.json_response({"tasks": [_serialize(r) for r in rows]})


async def handle_get(request):
    from aiohttp import web

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    task_id = request.match_info["id"]
    row = await scheduler.db.get_task(task_id)
    if row is None:
        return web.json_response({"error": f"Task {task_id!r} not found"}, status=404)
    return web.json_response(_serialize(row))


async def handle_create(request):
    from aiohttp import web
    from openagent.memory.schedule import validate_schedule_expression

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

    if not name:
        return web.json_response({"error": "name is required"}, status=400)
    if not cron_expression:
        return web.json_response({"error": "cron_expression is required"}, status=400)
    if not prompt:
        return web.json_response({"error": "prompt is required"}, status=400)

    try:
        validate_schedule_expression(cron_expression)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    task_id = await scheduler.add_task(name, cron_expression, prompt)

    # add_task enables by default; honour an explicit enabled=false.
    if body.get("enabled") is False:
        await scheduler.disable_task(task_id)

    row = await scheduler.db.get_task(task_id)
    elog("scheduled_task.create", id=task_id, name=name)
    return web.json_response(_serialize(row), status=201)


async def handle_update(request):
    from aiohttp import web
    from openagent.memory.schedule import validate_schedule_expression

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    task_id = request.match_info["id"]
    existing = await scheduler.db.get_task(task_id)
    if existing is None:
        return web.json_response({"error": f"Task {task_id!r} not found"}, status=404)

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

    if "cron_expression" in body:
        cron_expression = (body["cron_expression"] or "").strip()
        if not cron_expression:
            return web.json_response(
                {"error": "cron_expression cannot be empty"}, status=400
            )
        try:
            validate_schedule_expression(cron_expression)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        updates["cron_expression"] = cron_expression
        cron_changed = True

    enabled_change: bool | None = None
    if "enabled" in body:
        enabled_change = bool(body["enabled"])

    if not updates and enabled_change is None:
        return web.json_response(
            {"error": "No fields to update. Pass name, cron_expression, prompt, or enabled."},
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
    return web.json_response(_serialize(row))


async def handle_delete(request):
    from aiohttp import web

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    task_id = request.match_info["id"]
    existing = await scheduler.db.get_task(task_id)
    if existing is None:
        return web.json_response({"error": f"Task {task_id!r} not found"}, status=404)

    await scheduler.remove_task(task_id)
    elog("scheduled_task.delete", id=task_id, name=existing.get("name", ""))
    return web.json_response({"ok": True, "id": task_id})
