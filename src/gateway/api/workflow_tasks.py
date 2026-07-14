"""Workflow tasks REST API — CRUD + run + run history for the n8n-style
workflow engine.

Endpoints:

    GET    /api/workflows                       list all workflows
    POST   /api/workflows                       create a workflow
    GET    /api/workflows/{id}                  fetch one (full graph)
    PATCH  /api/workflows/{id}                  partial update
    DELETE /api/workflows/{id}                  delete + cascade runs
    POST   /api/workflows/{id}/run              body: {inputs, wait}
    POST   /api/workflows/{id}/stop             body: {run_id, wait}
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

from src.core.logging import elog
from src.memory.schedule import (
    epoch_to_iso,
    next_run_for_expression,
    validate_schedule_expression,
)
from src.workflow.blocks import iter_block_specs
from src.workflow.schedule_sync import (
    sync_workflow_schedules,
    trigger_types_from_graph,
)
from src.workflow.validate import (
    ValidationError,
    mcp_callability_from_pool,
    mcp_inventory_from_pool,
    validate_graph,
)


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


def _resolve_mcp_inventory(request) -> dict[str, dict[str, Any]] | None:
    """Return ``{mcp_name: {tool_name: parameters_schema}}`` from the
    live pool, or ``None`` when no agent/pool is attached. ``None``
    tells ``validate_graph`` to skip the MCP-existence check —
    appropriate during boot, in tests, or any path where the pool
    isn't reachable. A live pool that happens to have zero MCPs
    returns ``{}`` and any mcp-tool block correctly fails validation.
    """
    gw = request.app["gateway"]
    agent = getattr(gw, "agent", None) or getattr(gw, "_agent", None)
    if agent is None:
        return None
    pool = getattr(agent, "_mcp", None)
    return mcp_inventory_from_pool(pool)


def _resolve_mcp_callability(request) -> dict[str, dict[str, bool]] | None:
    """Companion to ``_resolve_mcp_inventory`` returning
    ``{mcp_name: {tool_name: bool}}`` so ``validate_graph`` can reject
    tools whose toolkit registered a non-callable (would otherwise
    raise ``TypeError`` mid-DAG)."""
    gw = request.app["gateway"]
    agent = getattr(gw, "agent", None) or getattr(gw, "_agent", None)
    if agent is None:
        return None
    pool = getattr(agent, "_mcp", None)
    return mcp_callability_from_pool(pool)


def _decorate_schedule(row: dict) -> dict:
    """Shape a workflow_schedules row for JSON."""
    out = dict(row)
    out["enabled"] = bool(out.get("enabled"))
    for key in ("next_run_at", "last_run_at", "created_at", "updated_at"):
        epoch = out.get(key)
        out[f"{key}_iso"] = epoch_to_iso(epoch) if epoch else None
    return out


async def _decorate_workflow(db, row: dict) -> dict:
    """Shape a DB row for JSON: parse graph_json, add ISO timestamps,
    fold in the per-block ``schedules[]`` array + derived
    ``trigger_types[]``. Drops legacy row-level columns
    (``trigger_kind`` / ``cron_expression`` / ``next_run_at``) from
    the response — schedule state lives in ``workflow_schedules``.
    """
    out = dict(row)
    if "graph" not in out:
        raw = out.pop("graph_json", None) or '{"version":1,"nodes":[],"edges":[],"variables":{}}'
        try:
            out["graph"] = json.loads(raw)
        except (TypeError, ValueError):
            out["graph"] = {"version": 1, "nodes": [], "edges": [], "variables": {}}
    for deprecated in ("trigger_kind", "cron_expression", "next_run_at"):
        out.pop(deprecated, None)
    for key in ("last_run_at", "created_at", "updated_at"):
        epoch = out.get(key)
        out[f"{key}_iso"] = epoch_to_iso(epoch) if epoch else None
    out["enabled"] = bool(out.get("enabled"))
    raw_cap = out.get("max_concurrent_runs")
    out["max_concurrent_runs"] = int(raw_cap) if raw_cap is not None else None
    out["trigger_types"] = trigger_types_from_graph(out.get("graph"))
    schedules = await db.list_schedules(workflow_id=out["id"])
    out["schedules"] = [_decorate_schedule(s) for s in schedules]
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
    has_trigger = request.query.get("has_trigger_type") or None
    rows = await scheduler.db.list_workflows(enabled_only=enabled_only)
    decorated = [await _decorate_workflow(scheduler.db, r) for r in rows]
    if has_trigger:
        decorated = [w for w in decorated if has_trigger in w["trigger_types"]]
    return web.json_response({"workflows": decorated})


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
    return web.json_response(await _decorate_workflow(scheduler.db, row))


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

    graph = {
        "version": 1,
        "nodes": body.get("nodes") or [],
        "edges": body.get("edges") or [],
        "variables": body.get("variables") or {},
    }
    try:
        validate_graph(
            graph,
            mcp_inventory=_resolve_mcp_inventory(request),
            mcp_callability=_resolve_mcp_callability(request),
        )
    except ValidationError as exc:
        return web.json_response({"error": f"graph validation failed: {exc}"}, status=400)

    cap_raw = body.get("max_concurrent_runs", None)
    if cap_raw is not None:
        try:
            cap = int(cap_raw)
        except (TypeError, ValueError):
            return web.json_response(
                {"error": "max_concurrent_runs must be an integer or null"},
                status=400,
            )
        if cap < 1:
            return web.json_response(
                {"error": "max_concurrent_runs must be >= 1 (or null for unlimited)"},
                status=400,
            )
    else:
        cap = None

    try:
        workflow_id = await scheduler.db.add_workflow(
            name=name,
            description=body.get("description") or None,
            graph=graph,
            enabled=bool(body.get("enabled", True)),
            max_concurrent_runs=cap,
        )
    except Exception as exc:  # integrity error on duplicate name, etc.
        if "UNIQUE" in str(exc):
            return web.json_response(
                {"error": f"workflow name {name!r} is already taken"},
                status=409,
            )
        return web.json_response({"error": str(exc)}, status=400)

    # Sync the workflow_schedules rows from any trigger-schedule
    # blocks in the new graph.
    try:
        await sync_workflow_schedules(scheduler.db, workflow_id, graph)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    row = await scheduler.db.get_workflow(workflow_id)
    elog("workflow.create", id=workflow_id, name=name)
    await request.app["gateway"].broadcast_resource(
        "workflow", "created", workflow_id,
    )
    return web.json_response(
        await _decorate_workflow(scheduler.db, row), status=201,
    )


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

    if "enabled" in body:
        updates["enabled"] = bool(body["enabled"])

    if "max_concurrent_runs" in body:
        cap_raw = body["max_concurrent_runs"]
        if cap_raw is None:
            updates["max_concurrent_runs"] = None
        else:
            try:
                cap = int(cap_raw)
            except (TypeError, ValueError):
                return web.json_response(
                    {"error": "max_concurrent_runs must be an integer or null"},
                    status=400,
                )
            if cap < 1:
                return web.json_response(
                    {"error": "max_concurrent_runs must be >= 1 (or null for unlimited)"},
                    status=400,
                )
            updates["max_concurrent_runs"] = cap

    new_graph: dict | None = None
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
            validate_graph(
                new_graph,
                mcp_inventory=_resolve_mcp_inventory(request),
                mcp_callability=_resolve_mcp_callability(request),
            )
        except ValidationError as exc:
            return web.json_response(
                {"error": f"graph validation failed: {exc}"}, status=400,
            )
        updates["graph"] = new_graph

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

    # Sync the schedule rows against the graph if it changed.
    if new_graph is not None:
        try:
            await sync_workflow_schedules(scheduler.db, existing["id"], new_graph)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)

    row = await scheduler.db.get_workflow(existing["id"])
    elog(
        "workflow.update",
        id=existing["id"],
        fields=list(updates.keys()),
    )
    await request.app["gateway"].broadcast_resource(
        "workflow", "updated", existing["id"],
    )
    return web.json_response(await _decorate_workflow(scheduler.db, row))


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
    await request.app["gateway"].broadcast_resource(
        "workflow", "deleted", existing["id"],
    )
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
        # Spawn through the scheduler's bookkeeping set so concurrent
        # API calls each get their own task handle (the previous design
        # stashed the task on ``scheduler._run_workflow_task`` — a single
        # attribute that overlapping calls trampled, leaving the earlier
        # handler awaiting whichever task arrived last).
        run_task = scheduler._spawn_workflow(
            scheduler._run_workflow(
                existing, trigger="api", inputs=inputs,
            )
        )
    except AttributeError:
        return web.json_response(
            {"error": "Scheduler has no workflow runtime attached"},
            status=503,
        )
    # Notify subscribed clients that a run kicked off; the workflows
    # screen turns the "Run" button into a spinner on this signal.
    await request.app["gateway"].broadcast_resource(
        "workflow", "updated", existing["id"],
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
        await asyncio.wait_for(run_task, timeout=timeout_s)
    except asyncio.TimeoutError:
        return web.json_response(
            {"error": f"workflow did not finish within {timeout_s}s"},
            status=504,
        )
    runs = await scheduler.db.list_workflow_runs(existing["id"], limit=1)
    if not runs:
        return web.json_response({"error": "run did not produce a row"}, status=500)
    # Run finished — re-broadcast so the screen flips the spinner off
    # and the "last run" badge picks up the new status.
    await request.app["gateway"].broadcast_resource(
        "workflow", "updated", existing["id"],
    )
    return web.json_response(_decorate_run(runs[0]))


async def handle_stop(request):
    """POST /api/workflows/{id}/stop — hard-stop in-flight run(s) of a
    workflow. Body: ``{run_id, wait, timeout_s}``.

    The app could start a workflow it had no way to stop: ``/run`` has always
    been here, and stopping was reachable only through the agent-facing
    ``stop_workflow`` MCP tool — so a user watching a runaway run had to ask
    the agent to kill it. The machinery was all present (the scheduler's
    cancellation drain handles workflow runs already); this is the missing
    door on the gateway, which §10 says is the *only* public surface.

    Flags each in-flight run ``cancelling`` — the same DB-backed hand-off the
    MCP tool writes, via one shared helper — and the scheduler cancels the
    executor mid-DAG within ~2s, aborting any in-flight AI block, then records
    each run ``cancelled``. Deliberately mirrors ``/api/scheduled-tasks/{id}
    /stop`` in status codes and response shape (the app treats the two as the
    same gesture), with one addition the scheduled-task side has no use for:
    ``run_id``, because a workflow may legitimately have several concurrent
    runs and the run screen stops the one it is showing.

    Stopping a run does NOT stop the workflow from firing again — that is
    ``enabled: false`` or removing its trigger-schedule block.
    """
    from aiohttp import web

    from src.workflow.cancel import await_runs_terminal, flag_workflow_runs_cancelling

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
    run_id = (body.get("run_id") or "").strip() or None
    wait = body.get("wait", True)
    timeout_s = int(body.get("timeout_s", 30))

    # The gateway shares a process with the scheduler, so it writes the flag
    # straight to the DB (as the scheduled-task endpoint does) rather than
    # going through the MCP's request queue. The raw connection is needed for
    # the compare-and-set the public partial-update helper cannot express —
    # see ``flag_workflow_runs_cancelling`` for why that guard is load-bearing.
    conn = await scheduler.db._ensure_connected()
    flagged = await flag_workflow_runs_cancelling(
        conn, workflow_id=existing["id"], run_id=run_id,
    )
    gw = request.app["gateway"]
    # Reflect the cancelling state on subscribed clients right away.
    await gw.broadcast_resource("workflow", "updated", existing["id"])

    if not flagged:
        # Not an error: the run finished on its own, was already stopped, or
        # the run_id names a run of some other workflow. The app polls this
        # endpoint from a Stop button that may well be a frame stale, so a
        # 404/409 here would surface as a scary dialog for a benign race —
        # same call the scheduled-task endpoint makes.
        return web.json_response({
            "workflow_id": existing["id"],
            "name": existing.get("name", ""),
            "stopped": [],
            "count": 0,
            "runs": [],
            "note": (
                "No running run to stop"
                + (f" (run_id {run_id!r})" if run_id else "")
                + "."
            ),
        })

    elog("workflow.stop", id=existing["id"], count=len(flagged))
    if wait:
        runs = await await_runs_terminal(conn, flagged, timeout_s=timeout_s)
        # Re-broadcast so the screen drops its running/cancelling state.
        await gw.broadcast_resource("workflow", "updated", existing["id"])
    else:
        runs = [{"id": rid, "status": "cancelling"} for rid in flagged]

    return web.json_response({
        "workflow_id": existing["id"],
        "name": existing.get("name", ""),
        "stopped": flagged,
        "count": len(flagged),
        "runs": runs,
    })


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

    from src.memory.schedule import (
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
