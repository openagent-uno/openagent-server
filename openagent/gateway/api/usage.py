"""GET /api/usage — budget and usage tracking."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web

from openagent.gateway.api.config import _read_resolved


async def _usage_summary_for_agent(agent) -> dict:
    model = agent.model

    # Check if the model has budget tracking
    from openagent.models.smart_router import SmartRouter

    if isinstance(model, SmartRouter) and model._budget:
        return await model._budget.get_usage_summary()

    # For non-smart models, return basic info
    return {
        "monthly_spend": 0,
        "monthly_budget": 0,
        "remaining": None,
        "by_model": {},
    }


async def handle_get(request: web.Request) -> web.Response:
    from aiohttp import web as _web

    gw = request.app["gateway"]
    return _web.json_response(await _usage_summary_for_agent(gw.agent))


async def handle_daily(request: web.Request) -> web.Response:
    """GET /api/usage/daily?days=7 — day-by-day cost breakdown."""
    from aiohttp import web as _web

    days = int(request.query.get("days", "7"))
    gw = request.app["gateway"]
    db = gw.agent._db

    if not db:
        return _web.json_response({"entries": []})

    entries = await db.get_daily_usage(days)
    return _web.json_response({"entries": entries})


async def handle_pricing(request: web.Request) -> web.Response:
    """GET /api/usage/pricing — pricing info for models in usage history."""
    from aiohttp import web as _web
    from openagent.models.catalog import get_model_pricing

    gw = request.app["gateway"]
    db = gw.agent._db
    if not db:
        return _web.json_response({"pricing": {}})

    providers_config = (
        _read_resolved(request).get("providers") or [] if gw.config_path else []
    )

    summary = await db.get_usage_summary()
    pricing = {}
    for model_id in summary.get("by_model", {}).keys():
        pricing[model_id] = get_model_pricing(model_id, providers_config)

    return _web.json_response({"pricing": pricing})
