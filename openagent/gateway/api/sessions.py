"""REST for per-session model pinning.

``PUT /api/sessions/{session_id}/model`` body ``{"runtime_id": "..."}`` —
pin the session to a specific model. Subsequent turns on that session
skip SmartRouter's classifier and dispatch straight to that model.

``DELETE /api/sessions/{session_id}/model`` — unpin. Session returns
to normal SmartRouter routing.

``GET /api/sessions/{session_id}/model`` — current pin, side binding,
and resolved runtime_id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web


def _db(request):
    return request.app["gateway"].agent.memory_db


async def handle_get(request):
    from aiohttp import web

    db = _db(request)
    if db is None:
        return web.json_response({"error": "memory DB not available"}, status=500)
    session_id = request.match_info["session_id"]
    side = await db.get_session_binding(session_id)
    pin = await db.get_session_pin(session_id)
    return web.json_response({
        "session_id": session_id,
        "side": side,
        "runtime_id": pin,
    })


async def handle_pin(request):
    from aiohttp import web

    db = _db(request)
    if db is None:
        return web.json_response({"error": "memory DB not available"}, status=500)
    session_id = request.match_info["session_id"]
    body = await request.json() if request.can_read_body else {}
    runtime_id = str(body.get("runtime_id") or "").strip()
    if not runtime_id:
        return web.json_response({"error": "runtime_id is required"}, status=400)
    model = await db.get_model(runtime_id)
    if model is None:
        return web.json_response(
            {"error": f"model {runtime_id!r} is not registered"},
            status=404,
        )
    if not model.get("enabled"):
        return web.json_response(
            {"error": f"model {runtime_id!r} is disabled — enable it before pinning"},
            status=400,
        )
    try:
        await db.pin_session_model(session_id, runtime_id)
    except ValueError as e:
        # Cross-framework pin attempt (surfaces a human-readable message).
        return web.json_response({"error": str(e)}, status=409)
    return web.json_response({
        "session_id": session_id,
        "runtime_id": runtime_id,
        "pinned": True,
    })


async def handle_unpin(request):
    from aiohttp import web

    db = _db(request)
    if db is None:
        return web.json_response({"error": "memory DB not available"}, status=500)
    session_id = request.match_info["session_id"]
    await db.unpin_session_model(session_id)
    return web.json_response({
        "session_id": session_id,
        "pinned": False,
    })
