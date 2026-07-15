"""Budgets REST API — CRUD for spend caps + a live spend-vs-limit meter.

GET    /api/budgets            → { "budgets": [...] }
POST   /api/budgets            → created rule (201) | 400 | 409 (duplicate scope)
GET    /api/budgets/usage      → { "usage": [...] } spend vs limit per rule
GET    /api/budgets/{id}       → rule | 404
PUT    /api/budgets/{id}       → updated rule | 404 | 400
DELETE /api/budgets/{id}       → { "ok": true, "id" } | 404

Same device-cert auth middleware as the rest of ``/api/*`` — this is the surface
the app and CLI use to make budgets monitorable and editable without a redeploy.
The ``usage`` view is what lets the app draw a meter ("today: $5.10 of $10.00").

Writes nudge the live :class:`BudgetGuard` so a new/edited cap takes effect on
the next turn rather than one TTL later.
"""

from __future__ import annotations

from src.core.budget_guard import compute_budget_usage, normalize_rule_input
from src.core.logging import elog

# Editable per-rule fields on PUT. ``scope_kind``/``scope``/``metric``/``window``
# form the UNIQUE identity, so changing them is allowed but may 409.
_UPDATABLE = ("scope_kind", "scope_value", "scope", "metric", "window",
              "amount", "alert_thresholds", "webhook_url", "enabled")


def _db(request):
    return request.app["gateway"].agent.memory_db


def _require_db(request):
    from aiohttp import web
    db = _db(request)
    if db is None:
        return None, web.json_response({"error": "No database configured"}, status=503)
    return db, None


def _nudge_guard(request) -> None:
    """Refresh the live guard after a write so the change is reflected promptly.
    Best-effort — the TTL backstop covers a miss."""
    try:
        guard = getattr(request.app["gateway"].agent.model, "budget_guard", None)
        if guard is not None:
            guard.schedule_refresh()
    except Exception:  # noqa: BLE001 — a UI write must not fail on guard plumbing
        pass


async def handle_list(request):
    from aiohttp import web
    db, err = _require_db(request)
    if err is not None:
        return err
    enabled_only = request.query.get("enabled_only", "").lower() in ("1", "true", "yes")
    rows = await db.list_budgets(enabled_only=enabled_only)
    return web.json_response({"budgets": rows})


async def handle_usage(request):
    """GET /api/budgets/usage — current spend vs limit per rule, for the app's
    meter. Registered BEFORE ``/api/budgets/{id}`` so ``usage`` isn't captured
    as an id."""
    from aiohttp import web
    db, err = _require_db(request)
    if err is not None:
        return err
    enabled_only = request.query.get("enabled_only", "").lower() in ("1", "true", "yes")
    usage = await compute_budget_usage(db, enabled_only=enabled_only)
    return web.json_response({"usage": usage})


async def handle_get(request):
    from aiohttp import web
    db, err = _require_db(request)
    if err is not None:
        return err
    row = await db.get_budget(request.match_info["id"])
    if row is None:
        return web.json_response({"error": "Budget not found"}, status=404)
    return web.json_response(row)


async def handle_create(request):
    from aiohttp import web
    db, err = _require_db(request)
    if err is not None:
        return err
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    try:
        norm = normalize_rule_input(body)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    if norm is None:
        return web.json_response({"error": "body must be an object"}, status=400)
    if norm["amount"] <= 0:
        return web.json_response({"error": "amount must be > 0"}, status=400)

    import sqlite3
    try:
        budget_id = await db.add_budget(source="user", **norm)
    except sqlite3.IntegrityError:
        return web.json_response(
            {"error": "a budget for this scope/metric/window already exists"},
            status=409,
        )
    row = await db.get_budget(budget_id)
    elog("budget.create", id=budget_id, scope_kind=norm["scope_kind"],
         scope=norm["scope_value"] or "*", metric=norm["metric"], window=norm["window"])
    _nudge_guard(request)
    await request.app["gateway"].broadcast_resource("budget", "created", budget_id)
    return web.json_response(row, status=201)


async def handle_update(request):
    from aiohttp import web
    db, err = _require_db(request)
    if err is not None:
        return err
    budget_id = request.match_info["id"]
    existing = await db.get_budget(budget_id)
    if existing is None:
        return web.json_response({"error": "Budget not found"}, status=404)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    updates: dict = {}
    for key in _UPDATABLE:
        if key in body:
            updates[key] = body[key]
    # ``scope`` is an alias for ``scope_value``.
    if "scope" in updates and "scope_value" not in updates:
        updates["scope_value"] = updates.pop("scope")
    else:
        updates.pop("scope", None)
    if not updates:
        return web.json_response({"error": "No fields to update"}, status=400)

    # Validate the merged rule so a PUT can't produce an out-of-vocabulary row.
    merged = {**existing, **updates}
    try:
        norm = normalize_rule_input(merged)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    if norm is None or norm["amount"] <= 0:
        return web.json_response({"error": "amount must be > 0"}, status=400)

    import sqlite3
    try:
        await db.update_budget(budget_id, **norm)
    except sqlite3.IntegrityError:
        return web.json_response(
            {"error": "a budget for this scope/metric/window already exists"},
            status=409,
        )
    row = await db.get_budget(budget_id)
    elog("budget.update", id=budget_id, fields=sorted(updates.keys()))
    _nudge_guard(request)
    await request.app["gateway"].broadcast_resource("budget", "updated", budget_id)
    return web.json_response(row)


async def handle_delete(request):
    from aiohttp import web
    db, err = _require_db(request)
    if err is not None:
        return err
    budget_id = request.match_info["id"]
    if await db.get_budget(budget_id) is None:
        return web.json_response({"error": "Budget not found"}, status=404)
    await db.delete_budget(budget_id)
    elog("budget.delete", id=budget_id)
    _nudge_guard(request)
    await request.app["gateway"].broadcast_resource("budget", "deleted", budget_id)
    return web.json_response({"ok": True, "id": budget_id})
