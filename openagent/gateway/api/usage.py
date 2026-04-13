"""GET /api/usage — budget and usage tracking."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web


async def handle_get(request: web.Request) -> web.Response:
    from aiohttp import web as _web

    gw = request.app["gateway"]
    agent = gw.agent
    model = agent._model

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
