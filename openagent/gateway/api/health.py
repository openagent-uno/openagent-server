"""GET /api/health — agent status."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web


async def handle_health(request: web.Request) -> web.Response:
    from aiohttp import web as _web

    gw = request.app["gateway"]
    info = gw.runtime_info()
    return _web.json_response({
        "status": "ok",
        "agent": info["agent"],
        "version": info["version"],
        "connected_clients": len(gw.clients),
    })
