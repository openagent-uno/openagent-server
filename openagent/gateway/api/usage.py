"""GET /api/usage — budget and usage tracking."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web


async def handle_get(request: web.Request) -> web.Response:
    from aiohttp import web as _web

    gw = request.app["gateway"]
    agent = gw.agent
    model = agent.model

    # Check if the model has budget tracking
    from openagent.models.smart_router import SmartRouter
    if isinstance(model, SmartRouter) and model._budget:
        summary = await model._budget.get_usage_summary()
        return _web.json_response(summary)

    # For non-smart models, return basic info
    return _web.json_response({
        "monthly_spend": 0,
        "monthly_budget": 0,
        "remaining": None,
        "by_model": {},
    })


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

    try:
        from litellm import model_cost
    except ImportError:
        return _web.json_response({"pricing": {}})

    gw = request.app["gateway"]
    db = gw.agent._db
    if not db:
        return _web.json_response({"pricing": {}})

    summary = await db.get_usage_summary()
    pricing = {}
    for model_id in summary.get("by_model", {}).keys():
        info = model_cost.get(model_id, {})
        pricing[model_id] = {
            "input_cost_per_million": (info.get("input_cost_per_token", 0) or 0) * 1_000_000,
            "output_cost_per_million": (info.get("output_cost_per_token", 0) or 0) * 1_000_000,
        }

    return _web.json_response({"pricing": pricing})
