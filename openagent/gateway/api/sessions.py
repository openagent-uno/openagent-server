"""REST for per-session model pinning and session lifecycle.

``GET /api/sessions`` — list persisted chat sessions (query: ``?client_id=...``).
``GET /api/sessions/{session_id}/model`` — current pin, side binding,
    and resolved runtime_id.
``PUT /api/sessions/{session_id}/model`` body ``{"runtime_id": "..."}`` —
    pin the session to a specific model. Subsequent turns on that session
    skip SmartRouter's classifier and dispatch straight to that model.
``DELETE /api/sessions/{session_id}/model`` — unpin. Session returns
    to normal SmartRouter routing.
``DELETE /api/sessions/{session_id}`` — delete a session and its history.
``GET /api/sessions/{session_id}/runs`` — turn history for a session.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web


from openagent.gateway.api._common import gateway_db as _db  # noqa: E402


async def handle_list(request):
    """GET /api/sessions — list persisted sessions.

    Query params:
      ``client_id`` — filter to one client's sessions. When omitted,
      the handler infers it from the authenticated device cert so the
      caller always sees only its own sessions.
      ``limit`` — max results (default 50).
    """
    from aiohttp import web

    db = _db(request)
    if db is None:
        return web.json_response({"error": "memory DB not available"}, status=500)

    client_id = (request.query.get("client_id") or "").strip() or None
    if not client_id:
        client_id = request.get("client_id")

    try:
        limit = min(int(request.query.get("limit", "50")), 200)
    except (TypeError, ValueError):
        limit = 50

    gateway = request.app.get("gateway")
    # DB-backed sessions (chat_sessions + agno_sessions merged).
    rows = await db.list_all_sessions(client_id, limit=limit)

    # Enrich with RAM queue/busy state from SessionManager.
    if gateway is not None and client_id:
        ram_sids = set(gateway.sessions.list_sessions(client_id))
        for r in rows:
            if r["session_id"] in ram_sids:
                r["_live"] = True
    return web.json_response({"sessions": rows})


async def handle_delete(request):
    """DELETE /api/sessions/{session_id} — forget a session and its history."""
    from aiohttp import web

    db = _db(request)
    if db is None:
        return web.json_response({"error": "memory DB not available"}, status=500)

    session_id = request.match_info["session_id"]
    if not session_id:
        return web.json_response({"error": "session_id is required"}, status=400)

    gateway = request.app.get("gateway")
    if gateway is not None:
        # Clean up RAM (find which client owns this session).
        for client_id in list(gateway.sessions._clients.keys()):
            await gateway.sessions.delete_session(client_id, session_id)
        try:
            await gateway.agent.forget_session(session_id)
        except Exception:
            pass
    # Persisted cleanup — covers DB rows even when RAM is gone.
    await db.delete_session_binding(session_id)
    await db.delete_sdk_session(session_id)
    await db.delete_session(session_id)

    return web.json_response({
        "session_id": session_id,
        "deleted": True,
    })


async def handle_get_runs(request):
    """GET /api/sessions/{session_id}/runs — turn history as flat messages.

    Returns messages extracted from ``agno_sessions.runs`` in the shape
    the frontend's ChatMessage array expects: ``{id, role, text, timestamp,
    toolInfo?, attachments?, model?}``. Query: ``?limit=20``.
    """
    from aiohttp import web
    import json as _json

    db = _db(request)
    if db is None:
        return web.json_response({"error": "memory DB not available"}, status=500)

    session_id = request.match_info["session_id"]
    try:
        limit = min(int(request.query.get("limit", "20")), 100)
    except (TypeError, ValueError):
        limit = 20

    runs = await db.list_session_runs(session_id, limit=limit)
    messages: list[dict] = []
    msg_idx = 0
    for run in runs:
        run_tools = run.get("tools") or []
        run_tools_map: dict[str, dict] = {}
        for t in run_tools:
            tn = t.get("tool_name") or t.get("name") or ""
            if tn:
                run_tools_map[tn] = {
                    "tool": tn,
                    "params": t.get("tool_args") or {},
                    "status": "done",
                    "result": t.get("result"),
                    "error": None if not t.get("tool_call_error") else "tool error",
                }
        for m in run.get("messages", []) or []:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                continue
            msg_idx += 1
            entry: dict = {
                "id": f"run-msg-{msg_idx}",
                "role": role,
                "text": content,
                "timestamp": run.get("created_at", 0),
            }
            if role == "tool":
                tname = m.get("name") or m.get("tool_name") or ""
                tool_info = run_tools_map.get(tname)
                if not tool_info:
                    try:
                        parsed = _json.loads(content)
                        if isinstance(parsed, dict) and parsed.get("tool"):
                            tool_info = parsed
                    except Exception:
                        pass
                if tool_info:
                    entry["toolInfo"] = tool_info
            if role == "assistant":
                entry["model"] = run.get("model")
                imgs = run.get("images") or []
                if imgs:
                    atts = []
                    for img in imgs:
                        fp = img.get("filepath") or img.get("url") or ""
                        fn = (img.get("filename") or
                              (fp.split("/")[-1] if "/" in fp else "") or
                              "image.png")
                        if fp:
                            atts.append({"type": "image", "path": fp, "filename": fn})
                    if atts:
                        entry["attachments"] = atts
            messages.append(entry)
    return web.json_response({
        "session_id": session_id,
        "messages": messages,
    })


async def handle_patch_metadata(request):
    """PATCH /api/sessions/{session_id} — update session title/model.

    Body: ``{"title": "...", "model": "..."}``. Both fields optional.
    """
    from aiohttp import web

    db = _db(request)
    if db is None:
        return web.json_response({"error": "memory DB not available"}, status=500)

    session_id = request.match_info["session_id"]
    if not session_id:
        return web.json_response({"error": "session_id is required"}, status=400)

    body = await request.json() if request.can_read_body else {}
    title = str(body.get("title") or "").strip() or None
    model = str(body.get("model") or "").strip() or None
    if not title and not model:
        return web.json_response({"error": "title or model is required"}, status=400)

    await db.upsert_session(session_id, title=title, model=model)
    return web.json_response({"session_id": session_id, "ok": True})


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
