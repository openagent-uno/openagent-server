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


from src.gateway.api._common import gateway_db as _db  # noqa: E402


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

    # Caller-supplied ``?client_id=`` still wins (power-user / debug
    # listing). When absent, default to the authenticated user handle
    # so the list is cross-device — every device the user has paired
    # with this network sees the same sessions. The legacy device
    # pubkey is kept available via ``request['client_id']`` for the
    # RAM enrichment below (which is naturally per-device, since
    # ``SessionManager`` keys by the live WS's client_id).
    client_id = (request.query.get("client_id") or "").strip() or None
    if not client_id:
        client_id = request.get("user_handle") or request.get("client_id")

    try:
        limit = min(int(request.query.get("limit", "50")), 200)
    except (TypeError, ValueError):
        limit = 50

    gateway = request.app.get("gateway")
    # DB-backed sessions (chat_sessions + agno_sessions merged).
    rows = await db.list_all_sessions(client_id, limit=limit)

    # Enrich with RAM queue/busy state from SessionManager. RAM is
    # keyed by device pubkey (the WebSocket's client_id), not by
    # handle, so we look up using the device pubkey when available.
    device_client_id = request.get("client_id")
    if gateway is not None and device_client_id:
        ram_sids = set(gateway.sessions.list_sessions(device_client_id))
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


def _build_run_tool_index(
    run_tools: list[dict],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Two lookup maps for ``runs[].tools[]`` entries used by the
    rehydration walk: by tool_call_id (precise — survives duplicate
    calls of the same name in one turn) and by tool_name (legacy
    fallback for rows that didn't persist tool_call_id).

    Each value is Agno's native ``ToolExecution.to_dict()`` shape — the
    universal app consumes that directly via ``ToolInfo`` and derives
    phase locally from ``tool_call_error`` + ``result`` presence.
    """
    from src.models._tool_status import stored_tool_to_wire

    by_id: dict[str, dict] = {}
    by_name: dict[str, dict] = {}
    for t in run_tools or []:
        info = stored_tool_to_wire(t)
        if info is None:
            continue
        tid = t.get("tool_call_id") or t.get("tool_use_id") or t.get("id") or ""
        tn = t.get("tool_name") or t.get("name") or info.get("tool_name") or ""
        if tid:
            by_id[str(tid)] = info
        if tn and tn not in by_name:
            by_name[str(tn)] = info
    return by_id, by_name


def _attachments_from_images(imgs: list) -> list[dict]:
    out: list[dict] = []
    for img in imgs or []:
        if not isinstance(img, dict):
            continue
        fp = img.get("filepath") or img.get("url") or ""
        fn = (
            img.get("filename")
            or (fp.split("/")[-1] if "/" in fp else "")
            or "image.png"
        )
        if fp:
            out.append({"type": "image", "path": fp, "filename": fn})
    return out


def _expand_run_messages(
    run: dict,
    *,
    timestamp: int,
    msg_counter: list[int],
    parent_model: str | None = None,
    parent_images: list | None = None,
    is_member_run: bool = False,
) -> list[dict]:
    """Expand ONE Agno run dict into the flat ``ChatMessage`` shape the
    universal app expects, mirroring the live-wire event ordering.

    Recurses into ``member_responses`` whenever the leader's
    ``delegate_task_to_member`` tool call sits in the message stream —
    so a specialist's nested tool calls and its delegated content
    surface as their own tool chips + assistant messages with the
    specialist's model attribution. This is what the live path
    produces during streaming (specialist deltas via
    ``IntermediateRunContentEvent``, tool calls via the unified
    STATUS frame); the rehydration walk now matches that 1-for-1.

    The recursive design follows Agno's stored shape exactly — a
    ``TeamRunOutput`` (with ``member_responses``) and a ``RunOutput``
    (without) reuse the same expansion because a member's tool calls
    live in its own ``runs[]``-equivalent ``tools`` list.
    """
    out: list[dict] = []
    run_status = str(run.get("status", "")).lower()
    if run_status in ("cancelled", "canceled"):
        return out

    run_tools = run.get("tools") or []
    run_tools_by_id, run_tools_by_name = _build_run_tool_index(run_tools)

    # Per-run model attribution. TeamRunOutput.to_dict() omits the
    # top-level ``model`` for the leader when it's a Team route — the
    # downstream UI then needs the entry_runtime_id we computed for the
    # synthetic ModelResponse. ``parent_model`` lets a recursing caller
    # override (used so member_responses inherit the team's leader
    # badge only when their own ``model`` is absent).
    run_model = run.get("model") or parent_model

    # Index member responses by member_id AND by stored index so we can
    # splice each delegation result inline. Agno stores ``agent_id`` on
    # the nested RunOutput (the AgnoAgent's name → url_safe_string),
    # which matches the ``member_id`` argument the leader passed to
    # ``delegate_task_to_member``. ``member_idx_by_run_id`` plus the
    # parallel ``agent_id`` map lets the splicing loop below find the
    # right member by id or by stored child_run_id without re-scanning.
    members_by_index: list[dict] = [
        mr for mr in (run.get("member_responses") or [])
        if isinstance(mr, dict)
    ]
    members_by_id: dict[str, int] = {}
    members_by_run_id: dict[str, int] = {}
    for idx, mr in enumerate(members_by_index):
        aid = str(mr.get("agent_id") or mr.get("team_id") or "")
        if aid:
            members_by_id.setdefault(aid, idx)
        rid = str(mr.get("run_id") or "")
        if rid:
            members_by_run_id.setdefault(rid, idx)

    # Track which member responses we've already spliced so we can
    # tack on any unmatched ones at the end (defensive against odd
    # rows where the leader's messages don't carry the delegate tool
    # results — happens with mid-turn cancellations).
    spliced_member_ids: set[int] = set()

    images_for_assistant = (
        (parent_images if parent_images is not None else run.get("images")) or []
    )

    for m in run.get("messages") or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            continue
        if m.get("from_history"):
            continue
        # Member runs (nested under member_responses) carry the leader-
        # generated task prompt as their first user message. That's an
        # internal Agno artifact, not the human's input — surfacing it
        # would make the synthetic prompt show up in the chat IN PLACE
        # OF the user's actual message (which lives at the top-level
        # team run). Skip it; the specialist's assistant content and
        # tool calls still appear.
        if is_member_run and role == "user":
            continue

        # Tool-result message — render as a tool chip with the same
        # JSON envelope the live wire uses. Inline the nested member
        # run (if any) right after, so the specialist's own tool
        # calls + content appear in the correct slot in the
        # transcript.
        if role == "tool":
            msg_counter[0] += 1
            tcall_id = m.get("tool_call_id") or m.get("tool_use_id") or ""
            tname = m.get("name") or m.get("tool_name") or ""
            tool_info: dict | None = None
            if tcall_id:
                tool_info = run_tools_by_id.get(str(tcall_id))
            if tool_info is None and tname:
                tool_info = run_tools_by_name.get(str(tname))
            entry: dict = {
                "id": f"run-msg-{msg_counter[0]}",
                "role": "tool",
                "text": content,
                "timestamp": timestamp,
            }
            if tool_info:
                entry["toolInfo"] = tool_info
            out.append(entry)

            # If this tool result came from a delegate_task_to_member
            # call, splice the matching member_responses entry inline.
            # Match by args.member_id → tools[].child_run_id → next
            # un-spliced member (in stored order, for legacy rows).
            if (
                tool_info
                and tool_info.get("tool_name") == "delegate_task_to_member"
            ):
                member_idx: int | None = None
                params = tool_info.get("tool_args") or {}
                member_id_arg = (
                    str(params.get("member_id") or "")
                    if isinstance(params, dict) else ""
                )
                if member_id_arg:
                    member_idx = members_by_id.get(member_id_arg)
                if member_idx is None and tcall_id:
                    matching_tool = next(
                        (t for t in run_tools
                         if str(t.get("tool_call_id")
                                or t.get("tool_use_id") or "") == str(tcall_id)),
                        None,
                    )
                    if matching_tool:
                        child_rid = str(matching_tool.get("child_run_id") or "")
                        if child_rid:
                            member_idx = members_by_run_id.get(child_rid)
                if member_idx is None:
                    member_idx = next(
                        (i for i in range(len(members_by_index))
                         if i not in spliced_member_ids),
                        None,
                    )

                if member_idx is not None:
                    spliced_member_ids.add(member_idx)
                    out.extend(_expand_run_messages(
                        members_by_index[member_idx],
                        timestamp=timestamp,
                        msg_counter=msg_counter,
                        parent_model=run_model,
                        is_member_run=True,
                    ))
            continue

        # Empty assistant text without a tool-call carrier payload is
        # noise (an LLM that yielded zero deltas). Tool-call carriers
        # have empty content by design and are dropped here too — the
        # ``tool`` role message that follows carries the structured
        # ``toolInfo`` chip the UI renders.
        if not content and role == "assistant":
            continue

        msg_counter[0] += 1
        entry = {
            "id": f"run-msg-{msg_counter[0]}",
            "role": role,
            "text": content,
            "timestamp": timestamp,
        }
        if role == "assistant":
            # Specialist (member) runs carry their own model id; fall
            # back to the leader's badge only when the member row
            # lacks one (very old rows, or external-agent shims).
            if run_model:
                entry["model"] = run_model
            atts = _attachments_from_images(images_for_assistant)
            if atts:
                entry["attachments"] = atts
                images_for_assistant = []  # only emit once per run
        out.append(entry)

    # Any member_responses that weren't spliced inline (e.g. the
    # leader's stored messages don't carry the delegate result, which
    # happens when the row was committed mid-turn) get appended after
    # the leader's content so the specialist's contribution still
    # appears in the transcript.
    for idx, mr in enumerate(members_by_index):
        if idx in spliced_member_ids:
            continue
        out.extend(_expand_run_messages(
            mr,
            timestamp=timestamp,
            msg_counter=msg_counter,
            parent_model=run_model,
            is_member_run=True,
        ))

    return out


async def handle_get_runs(request):
    """GET /api/sessions/{session_id}/runs — turn history as flat messages.

    Returns messages extracted from ``agno_sessions.runs`` in the shape
    the frontend's ChatMessage array expects: ``{id, role, text, timestamp,
    toolInfo?, attachments?, model?}``. Query: ``?limit=20``.

    The expansion walks ``member_responses`` recursively so a delegated
    specialist's tool calls + content appear with the specialist's own
    model attribution — matching what the live stream emitted at turn
    time. See :func:`_expand_run_messages` for the per-run logic and
    :mod:`src.models._tool_status` for the shared status envelope.
    """
    from aiohttp import web

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
    msg_counter = [0]  # mutable counter shared across recursive expansions
    for run in reversed(runs):
        messages.extend(_expand_run_messages(
            run,
            timestamp=int(run.get("created_at", 0) or 0),
            msg_counter=msg_counter,
        ))
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
