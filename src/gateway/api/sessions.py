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


from src.core.child_session import HIDDEN_CHILD_ORIGINS  # noqa: E402
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
    # ``?parent=<sid>`` lists just the children a session spawned (delegated
    # sub-agents, or the AI-node / firing sessions under a workflow-run /
    # scheduled-task root) — powers the parent transcript's delegation cards
    # and the run screen. Otherwise return the flat list, every row now
    # carrying ``parent_session_id`` / ``origin`` / ``kind`` so the app can
    # show an origin chip and a navigable breadcrumb.
    parent = (request.query.get("parent") or "").strip()
    if parent:
        rows = await db.list_child_sessions(parent, limit=limit)
    else:
        # The flat history hides every spawned child session — delegated
        # sub-agents, scheduled firings, and workflow nodes alike. Each is
        # navigable only in context (a parent's transcript card; a run's
        # execution screen), never as a standalone sidebar row. An explicit
        # ``?parent=`` query still lists them for those in-context surfaces.
        rows = await db.list_all_sessions(
            client_id, limit=limit, exclude_child_origins=HIDDEN_CHILD_ORIGINS,
        )

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


async def _teardown_session(gateway, db, sid: str) -> None:
    """Remove one session id everywhere: live RAM, the runtime, the persisted
    row + its derived satellites, then announce the removal. Best-effort per
    layer so one failure never strands the others (and a cascade never aborts
    half-way). Shared by the single-delete and the child-cascade so both tear a
    session down identically."""
    if gateway is not None:
        # RAM: find whichever live client owns this session and drop it.
        for client_id in list(gateway.sessions._clients.keys()):
            try:
                await gateway.sessions.delete_session(client_id, sid)
            except Exception:
                pass
        try:
            await gateway.agent.forget_session(sid)
        except Exception:
            pass
    # Persisted cleanup — covers the row + per-session satellites even when
    # RAM is already gone. ``purge_session`` is idempotent.
    try:
        await db.purge_session(sid)
    except Exception:
        pass
    # Announce so every connected client drops the row from its sidebar live
    # (a session deleted on one device vanishes on the others).
    if gateway is not None:
        try:
            gateway.broadcast_session("deleted", sid)
        except Exception:
            pass


async def handle_delete(request):
    """DELETE /api/sessions/{session_id} — delete a chat session and its
    history, cascading to the sub-agent child sessions it spawned.

    Only a *top-level manual chat* is deletable here: a row with
    ``origin == "chat"`` (or no origin, for legacy rows) and no
    ``parent_session_id``. Scheduled-task and workflow run sessions — and the
    sub-agent children themselves — are not directly deletable; they are
    managed in the context that owns them (the scheduled-task / workflow
    screens, or the parent chat). Deleting the chat cascades to every session
    it spawned (delegated sub-agents at any depth, plus any in-chat
    ``run dream mode`` firing), so no orphan child is left behind."""
    from aiohttp import web

    db = _db(request)
    if db is None:
        return web.json_response({"error": "memory DB not available"}, status=500)

    session_id = request.match_info["session_id"]
    if not session_id:
        return web.json_response({"error": "session_id is required"}, status=400)

    gateway = request.app.get("gateway")

    row = await db.get_session(session_id)

    # Authz: in a multi-user / federated deploy one user must not be able to
    # delete (and now cascade-purge) another's chat by guessing its id. Enforce
    # only when we can resolve BOTH a caller handle and a stamped owner — a
    # trusted-proxy / single-owner / legacy un-owned row has no handle or no
    # owner to compare and falls through unchanged (matching the prior
    # behaviour and the sibling list/runs endpoints). A non-owner gets 404 so
    # the endpoint never confirms the id exists.
    caller = request.get("user_handle") or request.get("client_id")
    if row is not None and caller and (row.get("client_id") or ""):
        if not await db.is_session_owned_by(session_id, caller, row=row):
            return web.json_response({"error": "session not found"}, status=404)

    # Guard: refuse to delete a run / sub-agent session. A missing row means a
    # never-persisted RAM-only chat (or one already gone) — fall through and
    # tear it down anyway, so a delete from the client always converges.
    if row is not None:
        origin = row.get("origin") or "chat"
        if origin != "chat" or row.get("parent_session_id"):
            return web.json_response(
                {
                    "error": (
                        "Only top-level chat sessions can be deleted. "
                        "Sub-agent, scheduled-run and workflow sessions are "
                        "managed from the context that spawned them."
                    ),
                    "origin": origin,
                },
                status=403,
            )

    # Collect the whole sub-agent lineage spawned by this chat, then tear down
    # children first and the parent last.
    descendants = await db.list_descendant_sessions(session_id)
    for child_sid in descendants:
        await _teardown_session(gateway, db, child_sid)
    await _teardown_session(gateway, db, session_id)

    return web.json_response({
        "session_id": session_id,
        "deleted": True,
        # The full set removed (parent + cascaded sub-agents) so a client can
        # prune every affected row and report an accurate count.
        "deleted_session_ids": [session_id, *descendants],
        "deleted_count": 1 + len(descendants),
    })


def _build_run_tool_index(
    run_tools: list[dict],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Two lookup maps for ``runs[].tools[]`` entries used by the
    rehydration walk: by tool_call_id (precise — survives duplicate
    calls of the same name in one turn) and by tool_name (legacy
    fallback for rows that didn't persist tool_call_id).

    Each value is the runtime's native ``ToolExecution.to_dict()`` shape — the
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
    parent_session_id: str | None = None,
    child_by_run_id: dict[str, str] | None = None,
) -> list[dict]:
    """Expand ONE runtime run dict into the flat ``ChatMessage`` shape the
    universal app expects, mirroring the live-wire event ordering.

    Recurses into ``member_responses`` whenever the leader's
    ``delegate_task_to_member`` tool call sits in the message stream —
    so a specialist's nested tool calls and its delegated content
    surface as their own tool chips + assistant messages with the
    specialist's model attribution. This is what the live path
    produces during streaming (specialist deltas via
    ``IntermediateRunContentEvent``, tool calls via the unified
    STATUS frame); the rehydration walk now matches that 1-for-1.

    The recursive design follows the runtime's stored shape exactly — a
    ``TeamRunOutput`` (with ``member_responses``) and a ``RunOutput``
    (without) reuse the same expansion because a member's tool calls
    live in its own ``runs[]``-equivalent ``tools`` list.
    """
    out: list[dict] = []
    run_status = str(run.get("status", "")).lower()
    if run_status in ("cancelled", "canceled"):
        return out

    # In-session compaction recap (vision §2). ``src.core.compaction``
    # folds the oldest runs into a single recap run tagged
    # ``metadata.compaction``. Surface it as a ``compaction`` message —
    # the same tool-style card the live ``session_compacted`` frame draws
    # — instead of letting the recap paragraph render as a bare assistant
    # bubble. The stats persisted alongside the recap let the reopened
    # transcript rebuild the identical card. This is a terminal shape for
    # the run: it has no user/tool/assistant messages worth expanding.
    meta = run.get("metadata")
    if isinstance(meta, dict) and meta.get("compaction"):
        msg_counter[0] += 1
        out.append({
            "id": f"run-msg-{msg_counter[0]}",
            "role": "compaction",
            "text": "",
            "timestamp": timestamp,
            "compaction": {
                "phase": "done",
                "folded_runs": int(meta.get("folded_runs") or 0),
                "kept_runs_count": int(meta.get("kept_runs_count") or 0),
                "summary_chars": int(
                    meta.get("summary_chars") or len(run.get("content") or "")
                ),
                "tokens_before": int(meta.get("tokens_before") or 0),
                "tokens_after": int(meta.get("tokens_after") or 0),
            },
        })
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
    # splice each delegation result inline. the runtime stores ``agent_id`` on
    # the nested RunOutput (the RuntimeAgent's name → url_safe_string),
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
        # internal runtime artifact, not the human's input — surfacing it
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

            # If this tool result came from a delegate_task_to_member call,
            # resolve the matching member_responses entry. Match by
            # args.member_id → tools[].child_run_id → next un-spliced member.
            is_delegate = bool(
                tool_info
                and tool_info.get("tool_name") == "delegate_task_to_member"
            )
            member_idx: int | None = None
            if is_delegate:
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

            # Did the member run as its OWN child session (a navigable row),
            # or nested in this (team) session (legacy)?
            #   1. Primary, deterministic: the parent delegate tool's
            #      ``child_run_id`` maps to a child session (built above from
            #      the child rows). Parallel-safe; independent of the team
            #      run_response object identity.
            #   2. Fallback (legacy rows): the member_response's session_id
            #      differs from the parent.
            child_member_sid = None
            crid = tool_info.get("child_run_id") if tool_info else None
            if crid and child_by_run_id:
                child_member_sid = child_by_run_id.get(str(crid))
            if child_member_sid is None and member_idx is not None:
                mr_sid = members_by_index[member_idx].get("session_id")
                if mr_sid and mr_sid != parent_session_id:
                    child_member_sid = mr_sid
            runs_as_child = bool(child_member_sid and child_member_sid != parent_session_id)

            # In child-session mode the chip MUST carry child_session_id so the
            # app renders a delegation card that deep-links into the member's
            # own session. Rehydration is the source of truth here — the stored
            # ToolExecution may not have captured the id (the live team-tool
            # object and the persisted one can differ), so backfill it from the
            # member_response. Copy the shared tool_info dict before mutating.
            if runs_as_child and tool_info is not None and not tool_info.get("child_session_id"):
                tool_info = {**tool_info, "child_session_id": child_member_sid}

            entry: dict = {
                "id": f"run-msg-{msg_counter[0]}",
                "role": "tool",
                "text": content,
                "timestamp": timestamp,
            }
            if tool_info:
                entry["toolInfo"] = tool_info
            out.append(entry)

            if is_delegate and member_idx is not None:
                spliced_member_ids.add(member_idx)
                # Legacy nested mode: splice the member transcript inline so
                # the specialist's tool calls + content appear in-slot. In
                # child-session mode we DON'T splice — the member's transcript
                # lives in its own row; the parent shows only the card (the
                # member_responses entry exists purely to carry the link).
                if not runs_as_child:
                    out.extend(_expand_run_messages(
                        members_by_index[member_idx],
                        timestamp=timestamp,
                        msg_counter=msg_counter,
                        parent_model=run_model,
                        is_member_run=True,
                        parent_session_id=parent_session_id,
                        child_by_run_id=child_by_run_id,
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
        # Per-message authorship: a human handle/display (so the app shows the
        # real sender instead of a generic "You", and multi-human sessions
        # attribute correctly) or an agent-self seed (the delegated task /
        # scheduled mission / workflow node prompt — rendered as a Mission
        # block). Absent on legacy rows → app falls back to the role label.
        if isinstance(m.get("author"), dict):
            entry["author"] = m["author"]
        if role == "assistant":
            # Specialist (member) runs carry their own model id; fall
            # back to the leader's badge only when the member row
            # lacks one (very old rows, or external-agent shims).
            if run_model:
                entry["model"] = run_model
            # Agent-attached files ride as ``[IMAGE:/p]`` / ``[VIDEO:/p]`` /
            # ``[VOICE:/p]`` / ``[FILE:/p]`` markers embedded in the STORED
            # assistant text — from the attachments MCP's ``send_file_to_user``
            # or auto-emitted for natively generated images. The live turn
            # strips them via ``parse_response_markers`` before the wire
            # ``response`` frame (stream/session.py); rehydration must do the
            # same, else a reopened — or, because the app reconciles the
            # transcript from this endpoint after every ``turn_complete``, even
            # a live — session shows the raw markers as literal text and loses
            # every non-image attachment (only ``run["images"]`` survived).
            atts: list[dict] = []
            if isinstance(content, str) and content:
                from src.channels.base import parse_response_markers

                clean, marker_atts = parse_response_markers(content)
                entry["text"] = clean
                atts = [
                    {"type": a.type, "path": a.path, "filename": a.filename}
                    for a in marker_atts
                ]
            # Merge the runtime-image attachments, deduped by path so a
            # generated image present BOTH as a ``run["images"]`` entry and as an
            # inline ``[IMAGE:/p]`` marker isn't attached twice.
            seen_paths = {a["path"] for a in atts}
            for a in _attachments_from_images(images_for_assistant):
                if a["path"] not in seen_paths:
                    atts.append(a)
                    seen_paths.add(a["path"])
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
        # A member that ran in its own child session is represented by its
        # card (the delegate chip), not by inlining its transcript here.
        mr_sid = mr.get("session_id")
        if mr_sid and mr_sid != parent_session_id:
            continue
        out.extend(_expand_run_messages(
            mr,
            timestamp=timestamp,
            msg_counter=msg_counter,
            parent_model=run_model,
            is_member_run=True,
            parent_session_id=parent_session_id,
            child_by_run_id=child_by_run_id,
        ))

    return out


async def handle_get_runs(request):
    """GET /api/sessions/{session_id}/runs — turn history as flat messages.

    Returns messages extracted from ``sessions.runs`` in the shape
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

    # Map each delegate-tool run_id → its child session id. A team-member
    # delegation runs in its own child session (a navigable sub-agent row);
    # its child session records the member ``child_run_id`` (= the parent
    # delegate tool's ``child_run_id``). This bridge lets us stamp the parent
    # chip's ``child_session_id`` deterministically — the runtime's team
    # run_response the leader mutates is not the persisted one, so the tool's
    # own field can't be relied on. Best-effort (no-op pre-child-session rows).
    child_by_run_id: dict[str, str] = {}
    try:
        for c in await db.list_child_sessions(session_id):
            crid = c.get("child_run_id")
            if crid:
                child_by_run_id[str(crid)] = c["session_id"]
    except Exception:  # noqa: BLE001
        child_by_run_id = {}

    messages: list[dict] = []
    msg_counter = [0]  # mutable counter shared across recursive expansions
    for run in reversed(runs):
        messages.extend(_expand_run_messages(
            run,
            timestamp=int(run.get("created_at", 0) or 0),
            msg_counter=msg_counter,
            parent_session_id=session_id,
            child_by_run_id=child_by_run_id,
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
    pin = await db.get_session_pin(session_id)
    return web.json_response({
        "session_id": session_id,
        # ``side`` was the legacy per-session framework lock; the v0.14
        # ``session_bindings`` → ``pinned_sessions`` migration dropped it (and
        # the ``get_session_binding`` method with it). Kept as a null in the
        # response so the field's shape stays stable for any old client.
        "side": None,
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
