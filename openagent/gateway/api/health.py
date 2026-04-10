"""GET /api/health — agent status."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web


async def handle_health(request: web.Request) -> web.Response:
    from aiohttp import web as _web
    import openagent

    gw = request.app["gateway"]
    return _web.json_response({
        "status": "ok",
        "agent": gw.agent.name,
        "version": getattr(openagent, "__version__", "?"),
        "connected_clients": len(gw.clients),
    })
