"""Workflow tasks REST API — CRUD + run + run history for the n8n-style
workflow engine.

Endpoints:

    GET    /api/workflows                       list all workflows
    POST   /api/workflows                       create a workflow
    GET    /api/workflows/{id}                  fetch one (full graph)
    PATCH  /api/workflows/{id}                  partial update
    DELETE /api/workflows/{id}                  delete + cascade runs
    POST   /api/workflows/{id}/run              body: {inputs, wait}
    GET    /api/workflows/{id}/runs             run history (newest first)
    GET    /api/workflow-runs/{run_id}          fetch one run + trace
    GET    /api/workflow-block-types            static catalog for the UI
    GET    /api/mcp-tools                       live MCP tool inventory

503 is returned when the live ``Scheduler`` isn't attached (same
invariant as /api/scheduled-tasks). Writes bypass the MCP subprocess —
the gateway talks to the same SQLite as the workflow-manager MCP and
the scheduler, so all three stay in sync.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from openagent.core.logging import elog
from openagent.memory.schedule import (
    epoch_to_iso,
    next_run_for_expression,
    validate_schedule_expression,
)
from openagent.workflow.blocks import iter_block_specs
from openagent.workflow.schedule_sync import (
    NOT_PROVIDED as _MISSING,
    derive_schedule_updates,
)
from openagent.workflow.validate import ValidationError, validate_graph


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


def _decorate_workflow(row: dict) -> dict:
    """Shape a DB row for JSON: parse graph_json, add ISO timestamps."""
    out = dict(row)
    if "graph" not in out:
        raw = out.pop("graph_json", None) or '{"version":1,"nodes":[],"edges":[],"variables":{}}'
        try:
            out["graph"] = json.loads(raw)
        except (TypeError, ValueError):
            out["graph"] = {"version": 1, "nodes": [], "edges": [], "variables": {}}
    for key in ("last_run_at", "next_run_at", "created_at", "updated_at"):
        epoch = out.get(key)
        out[f"{key}_iso"] = epoch_to_iso(epoch) if epoch else None
    out["enabled"] = bool(out.get("enabled"))
    return out


def _decorate_run(row: dict) -> dict:
    out = dict(row)
    for key in ("started_at", "finished_at"):
        epoch = out.get(key)
        out[f"{key}_iso"] = epoch_to_iso(epoch) if epoch else None
    return out


async def _find_workflow(scheduler, id_or_name: str) -> dict | None:
    """Accept full id, 8-char id prefix, or unique name. Mirrors the
    MCP's ``_resolve_workflow`` for consistent UX."""
    return await scheduler.db.get_workflow(id_or_name)


# ── list / get / create / update / delete ───────────────────────────


async def handle_list(request):
    from aiohttp import web

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    enabled_only = request.query.get("enabled_only", "").lower() in ("1", "true", "yes")
    trigger_kind = request.query.get("trigger_kind") or None
    rows = await scheduler.db.list_workflows(
        enabled_only=enabled_only, trigger_kind=trigger_kind,
    )
    return web.json_response({"workflows": [_decorate_workflow(r) for r in rows]})


async def handle_get(request):
    from aiohttp import web

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    row = await _find_workflow(scheduler, request.match_info["id"])
    if row is None:
        return web.json_response(
            {"error": f"Workflow {request.match_info['id']!r} not found"},
            status=404,
        )
    return web.json_response(_decorate_workflow(row))


async def handle_create(request):
    from aiohttp import web

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    name = (body.get("name") or "").strip()
    if not name:
        return web.json_response({"error": "name is required"}, status=400)

    trigger_kind = body.get("trigger_kind") or "manual"
    if trigger_kind not in ("manual", "schedule", "ai", "hybrid"):
        return web.json_response(
            {"error": f"invalid trigger_kind {trigger_kind!r}"},
            status=400,
        )

    graph = {
        "version": 1,
        "nodes": body.get("nodes") or [],
        "edges": body.get("edges") or [],
        "variables": body.get("variables") or {},
    }
    try:
        validate_graph(graph)
    except ValidationError as exc:
        return web.json_response({"error": f"graph validation failed: {exc}"}, status=400)

    # Reconcile graph's trigger-schedule block with the row-level
    # cron. Explicit cron in the request body (e.g. from the list
    # screen's quick-create form) wins; otherwise we derive from
    # any trigger-schedule block inside the graph.
    body_cron = body.get("cron_expression")
    try:
        schedule_patch = derive_schedule_updates(
            graph,
            explicit_cron=body_cron if "cron_expression" in body else _MISSING,
            explicit_trigger_kind=trigger_kind,
        )
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    cron_expression = schedule_patch.get("cron_expression")
    next_run_at = schedule_patch.get("next_run_at")
    trigger_kind = schedule_patch.get("trigger_kind", trigger_kind)

    try:
        workflow_id = await scheduler.db.add_workflow(
            name=name,
            description=body.get("description") or None,
            graph=graph,
            trigger_kind=trigger_kind,
            cron_expression=cron_expression,
            next_run_at=next_run_at,
            enabled=bool(body.get("enabled", True)),
        )
    except Exception as exc:  # integrity error on duplicate name, etc.
        if "UNIQUE" in str(exc):
            return web.json_response(
                {"error": f"workflow name {name!r} is already taken"},
                status=409,
            )
        return web.json_response({"error": str(exc)}, status=400)

    row = await scheduler.db.get_workflow(workflow_id)
    elog("workflow.create", id=workflow_id, name=name, trigger_kind=trigger_kind)
    return web.json_response(_decorate_workflow(row), status=201)


async def handle_update(request):
    from aiohttp import web

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    existing = await _find_workflow(scheduler, request.match_info["id"])
    if existing is None:
        return web.json_response(
            {"error": f"Workflow {request.match_info['id']!r} not found"},
            status=404,
        )

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    updates: dict[str, Any] = {}

    if "name" in body:
        name = (body["name"] or "").strip()
        if not name:
            return web.json_response({"error": "name cannot be empty"}, status=400)
        updates["name"] = name

    if "description" in body:
        updates["description"] = body["description"] or None

    if "trigger_kind" in body:
        tk = body["trigger_kind"]
        if tk not in ("manual", "schedule", "ai", "hybrid"):
            return web.json_response(
                {"error": f"invalid trigger_kind {tk!r}"},
                status=400,
            )
        updates["trigger_kind"] = tk

    if "enabled" in body:
        updates["enabled"] = bool(body["enabled"])

    if any(k in body for k in ("nodes", "edges", "variables")):
        current = existing["graph"]
        new_graph = {
            "version": current.get("version", 1),
            "nodes": body["nodes"] if "nodes" in body else current.get("nodes", []),
            "edges": body["edges"] if "edges" in body else current.get("edges", []),
            "variables": (
                body["variables"]
                if "variables" in body
                else current.get("variables", {})
            ),
        }
        try:
            validate_graph(new_graph)
        except ValidationError as exc:
            return web.json_response(
                {"error": f"graph validation failed: {exc}"}, status=400,
            )
        updates["graph"] = new_graph

    # Reconcile schedule column with graph's trigger-schedule block,
    # honouring any explicit cron_expression in the body.
    graph_for_sync = updates.get("graph") or existing.get("graph") or {}
    try:
        schedule_patch = derive_schedule_updates(
            graph_for_sync,
            explicit_cron=(
                body["cron_expression"] if "cron_expression" in body else _MISSING
            ),
            explicit_trigger_kind=updates.get("trigger_kind"),
            current_trigger_kind=existing.get("trigger_kind"),
        )
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    # schedule_patch may override trigger_kind (auto-promote to schedule).
    updates.update(schedule_patch)

    if not updates:
        return web.json_response(
            {"error": "No fields to update."}, status=400,
        )

    try:
        await scheduler.db.update_workflow(existing["id"], **updates)
    except Exception as exc:
        if "UNIQUE" in str(exc):
            return web.json_response(
                {"error": "workflow name is already taken"},
                status=409,
            )
        return web.json_response({"error": str(exc)}, status=400)

    row = await scheduler.db.get_workflow(existing["id"])
    elog(
        "workflow.update",
        id=existing["id"],
        fields=list(updates.keys()),
    )
    return web.json_response(_decorate_workflow(row))


async def handle_delete(request):
    from aiohttp import web

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    existing = await _find_workflow(scheduler, request.match_info["id"])
    if existing is None:
        return web.json_response(
            {"error": f"Workflow {request.match_info['id']!r} not found"},
            status=404,
        )

    await scheduler.db.delete_workflow(existing["id"])
    elog("workflow.delete", id=existing["id"], name=existing.get("name", ""))
    return web.json_response({"ok": True, "id": existing["id"]})


# ── run + run history ───────────────────────────────────────────────


async def handle_run(request):
    """Kick off a workflow execution. Body: ``{inputs, wait, timeout_s}``.

    Always enqueues through ``workflow_run_requests`` so the execution
    path matches what the AI's ``run_workflow`` MCP tool uses. When
    ``wait`` is true (default), polls for completion and returns the
    final run row. Otherwise returns immediately with ``run_id``.
    """
    from aiohttp import web

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    existing = await _find_workflow(scheduler, request.match_info["id"])
    if existing is None:
        return web.json_response(
            {"error": f"Workflow {request.match_info['id']!r} not found"},
            status=404,
        )

    try:
        body = await request.json() if request.can_read_body else {}
    except Exception:
        body = {}
    inputs = body.get("inputs") or {}
    wait = body.get("wait", True)
    timeout_s = int(body.get("timeout_s", 300))

    # Fast path: execute directly against the scheduler's executor —
    # avoids the ~30s scheduler tick latency for UI-triggered runs.
    # Still routes through the same path the scheduler uses for
    # queue-claimed requests so trace/history come out identical.
    run_id = str(uuid.uuid4())
    try:
        # Call private helper intentionally — gateway + scheduler share
        # this module boundary. See Scheduler._run_workflow docstring.
        scheduler._run_workflow_task = asyncio.create_task(
            scheduler._run_workflow(
                existing, trigger="api", inputs=inputs,
            )
        )
    except AttributeError:
        return web.json_response(
            {"error": "Scheduler has no workflow runtime attached"},
            status=503,
        )

    if not wait:
        # We can't report the run_id synchronously without waiting a
        # moment for the executor to insert the row. Short poll for
        # the latest run on this workflow, which will be the one we
        # just started.
        deadline = time.monotonic() + 3
        latest = None
        while time.monotonic() < deadline:
            runs = await scheduler.db.list_workflow_runs(existing["id"], limit=1)
            if runs:
                latest = runs[0]
                break
            await asyncio.sleep(0.05)
        return web.json_response({
            "run_id": latest["id"] if latest else None,
            "status": "running",
        }, status=202)

    # wait=True: let the task finish, then fetch the run row.
    try:
        await asyncio.wait_for(scheduler._run_workflow_task, timeout=timeout_s)
    except asyncio.TimeoutError:
        return web.json_response(
            {"error": f"workflow did not finish within {timeout_s}s"},
            status=504,
        )
    runs = await scheduler.db.list_workflow_runs(existing["id"], limit=1)
    if not runs:
        return web.json_response({"error": "run did not produce a row"}, status=500)
    return web.json_response(_decorate_run(runs[0]))


async def handle_runs_list(request):
    from aiohttp import web

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    existing = await _find_workflow(scheduler, request.match_info["id"])
    if existing is None:
        return web.json_response(
            {"error": f"Workflow {request.match_info['id']!r} not found"},
            status=404,
        )

    limit = int(request.query.get("limit", 20))
    status = request.query.get("status") or None
    runs = await scheduler.db.list_workflow_runs(
        existing["id"], limit=limit, status=status,
    )
    return web.json_response({"runs": [_decorate_run(r) for r in runs]})


async def handle_run_get(request):
    from aiohttp import web

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    run_id = request.match_info["run_id"]
    row = await scheduler.db.get_workflow_run(run_id)
    if row is None:
        return web.json_response({"error": f"run {run_id!r} not found"}, status=404)
    return web.json_response(_decorate_run(row))


async def handle_stats(request):
    """Aggregate stats for the workflow editor's RunHistoryDrawer.

    Returns success rate, avg duration, and a last-N timeline used to
    render the sparkline + "last run" badge on the workflow list row.
    """
    from aiohttp import web

    scheduler, err = _resolve_scheduler(request)
    if err is not None:
        return err

    existing = await _find_workflow(scheduler, request.match_info["id"])
    if existing is None:
        return web.json_response(
            {"error": f"Workflow {request.match_info['id']!r} not found"},
            status=404,
        )

    try:
        count = max(1, min(int(request.query.get("last", 10)), 50))
    except ValueError:
        return web.json_response(
            {"error": "last must be a positive integer"}, status=400,
        )
    stats = await scheduler.db.workflow_run_stats(
        existing["id"], sparkline_count=count,
    )
    # Decorate last[] entries with ISO timestamps for UI display.
    for entry in stats.get("last", []):
        for key in ("started_at", "finished_at"):
            epoch = entry.get(key)
            entry[f"{key}_iso"] = epoch_to_iso(epoch) if epoch else None
    return web.json_response(stats)


# ── introspection: block catalog + MCP tool inventory ───────────────


async def handle_block_types(request):
    """Static catalog from BLOCK_CATALOG — what every block expects in
    config, what handles it publishes, and a human-readable description.
    The workflow editor's block palette and properties panel read from
    here to render their UI.
    """
    from aiohttp import web

    return web.json_response({"block_types": iter_block_specs()})


async def handle_mcp_tools(request):
    """Live enumeration of every connected MCP + the tools it exposes.
    Powers the mcp-tool block's picker in the editor. Reads the live
    pool (not the mcps DB table) so only actually-loaded tools appear.
    """
    from aiohttp import web

    gw = request.app["gateway"]
    agent = getattr(gw, "agent", None) or getattr(gw, "_agent", None)
    if agent is None:
        return web.json_response({"mcps": []})
    pool = getattr(agent, "_mcp", None)
    if pool is None:
        return web.json_response({"mcps": []})
    return web.json_response({"mcps": pool.list_mcp_tools()})


async def handle_cron_describe(request):
    """Validate a cron expression and return the next N fire times.

    Powers the CronPicker's live preview in the workflow editor and
    the list-screen create form. Mirrors the scheduler MCP's
    ``describe_cron`` tool so the UI, the AI, and the CLI see the
    same output shape.

    Query params:
      - ``expression`` (required): cron (``0 9 * * *``) or one-shot
        (``@once:<epoch>``).
      - ``count`` (optional, default 3, max 20): how many upcoming
        fire times to compute.
    """
    from aiohttp import web
    from croniter import croniter

    from openagent.memory.schedule import (
        epoch_to_iso,
        is_one_shot_expression,
        parse_one_shot_expression,
    )

    expr = (request.query.get("expression") or "").strip()
    if not expr:
        return web.json_response(
            {"error": "expression query param is required"}, status=400,
        )
    try:
        count = max(1, min(int(request.query.get("count", 3)), 20))
    except ValueError:
        return web.json_response(
            {"error": "count must be a positive integer"}, status=400,
        )

    try:
        validate_schedule_expression(expr)
    except ValueError as exc:
        return web.json_response(
            {"expression": expr, "valid": False, "error": str(exc)},
            status=400,
        )

    upcoming: list[dict] = []
    if is_one_shot_expression(expr):
        epoch = parse_one_shot_expression(expr)
        upcoming.append({"epoch": epoch, "iso": epoch_to_iso(epoch)})
    else:
        base = time.time()
        it = croniter(expr, base)
        for _ in range(count):
            nxt = it.get_next(float)
            upcoming.append({"epoch": nxt, "iso": epoch_to_iso(nxt)})

    return web.json_response({
        "expression": expr,
        "valid": True,
        "one_shot": is_one_shot_expression(expr),
        "upcoming": upcoming,
    })
