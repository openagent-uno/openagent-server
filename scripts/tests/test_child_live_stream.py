"""Live streaming of a delegated sub-agent into its OWN child session.

A team member runs inside the leader's turn (not as its own gateway
``StreamSession``), so its deltas / tool calls are SUPPRESSED from the parent
stream. These tests lock down the other half: those same events are mirrored
onto the child session's OWN live stream — tagged with the child session_id —
over the active turn's channel (the context-local emitter installed by
``StreamTurnRunner``). The app already routes every frame by session_id, so a
child session then streams piece-by-piece exactly like any session.

Covered:
  - ``emit_child_frame`` / ``current_child_stream_emitter`` contextvar contract.
  - ``_emit_member_event`` translates a member content event → a child-tagged
    ``delta`` frame and a member tool event → a child-tagged ``status`` frame.
  - ``_emit_member_card_link`` stamps ``child_session_id`` onto the leader's
    in-progress delegate tool AND re-streams it as a parent-tagged status so
    the delegation card flips clickable WHILE the sub-agent runs.
  - No emitter installed (bridge / scheduler / headless) → every helper is a
    silent no-op (never raises, never emits).
"""
from __future__ import annotations

import json

from ._framework import TestContext, test

PARENT = "sess-x"
CHILD = "sess-x::member::opus::abcd1234"


class _Capture:
    """A child-stream emitter that records every frame it's handed."""

    def __init__(self) -> None:
        self.frames: list[dict] = []

    async def __call__(self, frame: dict) -> None:
        self.frames.append(frame)


@test("child_live_stream", "member content + tool events mirror onto the child's stream")
async def t_emit_member_event(ctx: TestContext) -> None:
    from src.core._run_state.agent import RunContentEvent, ToolCallStartedEvent
    from src.core._runner.team._default_tools import _emit_member_event
    from src.models.providers.response import ToolExecution
    from src.stream.child_stream import (
        install_child_stream_emitter, reset_child_stream_emitter,
    )

    cap = _Capture()
    tok = install_child_stream_emitter(cap)
    try:
        await _emit_member_event(
            CHILD, RunContentEvent(content="Drafting ", run_id="rm", session_id=CHILD),
        )
        await _emit_member_event(
            CHILD,
            ToolCallStartedEvent(
                tool=ToolExecution(tool_name="write_file", tool_call_id="t1"),
                run_id="rm", session_id=CHILD,
            ),
        )
    finally:
        reset_child_stream_emitter(tok)

    assert len(cap.frames) == 2, cap.frames
    delta, status = cap.frames
    assert delta == {"kind": "delta", "session_id": CHILD, "text": "Drafting "}, delta
    assert status["kind"] == "status" and status["session_id"] == CHILD, status
    tool_json = json.loads(status["text"])
    assert tool_json["tool_name"] == "write_file", tool_json
    # phase=started → result forced None so the UI renders it as "running".
    assert tool_json.get("result") is None, tool_json


@test("child_live_stream", "delegate card-link: stamps child_session_id + streams a clickable parent chip")
async def t_card_link(ctx: TestContext) -> None:
    from src.core._runner.team._default_tools import _emit_member_card_link
    from src.models.providers.response import ToolExecution
    from src.stream.child_stream import (
        install_child_stream_emitter, reset_child_stream_emitter,
    )

    delegate_tool = ToolExecution(
        tool_name="delegate_task_to_member",
        tool_call_id="d1",
        tool_args={"member_id": "opus", "task": "write a haiku"},
    )

    class _RunResponse:
        tools = [delegate_tool]

    class _TeamSession:
        session_id = PARENT

    cap = _Capture()
    tok = install_child_stream_emitter(cap)
    try:
        await _emit_member_card_link(_RunResponse(), _TeamSession(), "opus", CHILD)
    finally:
        reset_child_stream_emitter(tok)

    # The leader's delegate tool now carries the child session id (so the
    # persisted run + the live chip both deep-link into the sub-agent).
    assert delegate_tool.child_session_id == CHILD
    # And a parent-tagged status frame was streamed so the card flips clickable
    # mid-run (carrying child_session_id in its ToolExecution payload).
    assert len(cap.frames) == 1, cap.frames
    frame = cap.frames[0]
    assert frame["kind"] == "status" and frame["session_id"] == PARENT, frame
    payload = json.loads(frame["text"])
    assert payload["tool_call_id"] == "d1", payload
    assert payload["child_session_id"] == CHILD, payload


@test("child_live_stream", "turn_complete frame finalizes the child stream")
async def t_turn_complete(ctx: TestContext) -> None:
    from src.stream.child_stream import (
        emit_child_frame, install_child_stream_emitter, reset_child_stream_emitter,
    )

    cap = _Capture()
    tok = install_child_stream_emitter(cap)
    try:
        await emit_child_frame(CHILD, "turn_complete")
    finally:
        reset_child_stream_emitter(tok)

    assert cap.frames == [{"kind": "turn_complete", "session_id": CHILD}], cap.frames


@test("child_live_stream", "run_child_session(stream=True) streams a detached run via the broadcast sink")
async def t_run_child_session_stream(ctx: TestContext) -> None:
    """A scheduled firing / workflow node has no turn-scoped emitter, so
    ``run_child_session(stream=True)`` installs the gateway's broadcast sink and
    forwards each delta / status / turn_complete tagged with the child sid — the
    run screen renders it token-by-token like any session."""
    from src.core.child_session import run_child_session
    from src.stream.child_stream import set_child_broadcast_sink, broadcast_child_emitter

    class _StreamAgent:
        name = "spy"
        model = None

        def __init__(self) -> None:
            self.released: list[str] = []

        async def run_stream(self, *, message, user_id, session_id,
                             model_override=None, author=None, on_status=None):
            if on_status is not None:
                await on_status("Thinking…")
            for piece in ("Hel", "lo"):
                yield {"kind": "delta", "text": piece}
            yield {"kind": "done", "text": "Hello"}

        async def release_session(self, session_id, *, model_override=None) -> None:
            self.released.append(session_id)

    cap = _Capture()
    assert broadcast_child_emitter() is None  # nothing wired by default
    set_child_broadcast_sink(cap)
    try:
        assert broadcast_child_emitter() is cap
        agent = _StreamAgent()
        result = await run_child_session(
            agent=agent, db=None, parent_session_id="scheduler:t1",
            origin="scheduler", origin_ref={"task_id": "t1", "run_id": "r1"},
            title="Daily", prompt="do it", stream=True,
        )
    finally:
        set_child_broadcast_sink(None)

    sid = "scheduler:t1:r1"
    assert result.session_id == sid, result
    assert result.text == "Hello", result          # "Hel" + "lo" (done not re-emitted)
    assert agent.released == [sid], agent.released   # released, not forgotten
    # Every frame is tagged with the CHILD sid (so the app routes it there).
    assert all(f["session_id"] == sid for f in cap.frames), cap.frames
    kinds = [(f["kind"], f.get("text")) for f in cap.frames]
    # The mission/seed is echoed FIRST (before any delta) carrying the agent-self
    # author, so the run screen shows the Mission block while the run executes.
    assert kinds[0] == ("seed", "do it"), kinds
    assert cap.frames[0].get("author", {}).get("kind") == "agent", cap.frames[0]
    assert ("status", "Thinking…") in kinds, kinds
    assert ("delta", "Hel") in kinds and ("delta", "lo") in kinds, kinds
    # Ends with a committing ``response`` (the full text, for the app to
    # marker-strip + clear its delta buffer) then ``turn_complete``.
    assert kinds[-2:] == [("response", "Hello"), ("turn_complete", None)], kinds


@test("child_live_stream", "emit_card_link re-streams the in-flight chip with child_session_id (clickable mid-run)")
async def t_emit_card_link(ctx: TestContext) -> None:
    """A spawned child's id is minted at run start, but on the MCP delegation /
    run-now paths it only reaches the client in the tool's FINAL result — so the
    card stays inert until the child finishes. ``emit_card_link`` re-streams the
    IN-FLIGHT tool's status frame carrying the child id (correlated by the active
    tool_call_id), tagged with the chat session, so the card flips clickable
    mid-run. It stays a RUNNING frame (no result) so the completion frame still
    matches + replaces it."""
    from src.mcp.servers.delegation import handlers as dh
    from src.stream.card_link import (
        emit_card_link, reset_active_tool_call, set_active_tool_call,
    )
    from src.stream.child_stream import (
        install_child_stream_emitter, reset_child_stream_emitter,
    )

    cap = _Capture()
    etok = install_child_stream_emitter(cap)
    # The delegation context carries the chat session that owns the in-flight
    # tool (set for the whole turn by the main agent).
    ctx_tokens = dh.install_context(
        session_id="chat-1", pool=None, db=None, dispatcher=None,
    )
    ttok = set_active_tool_call("toolu_9", "tool_search_call_tool")
    try:
        await emit_card_link("scheduler:task-1:run-1")
    finally:
        reset_active_tool_call(ttok)
        dh.reset_context(ctx_tokens)
        reset_child_stream_emitter(etok)

    assert len(cap.frames) == 1, cap.frames
    f = cap.frames[0]
    # Tagged with the CHAT session (not the spawned child) so the app patches
    # the parent transcript's chip.
    assert f["kind"] == "status" and f["session_id"] == "chat-1", f
    payload = json.loads(f["text"])
    assert payload["tool_call_id"] == "toolu_9", payload
    assert payload["tool_name"] == "tool_search_call_tool", payload
    assert payload["child_session_id"] == "scheduler:task-1:run-1", payload
    # No result → still "running" (the app keeps it clickable yet in-progress,
    # and the eventual completion frame still matches by tool_call_id).
    assert "result" not in payload, payload


@test("child_live_stream", "emit_card_link is a silent no-op without an active tool / chat session")
async def t_emit_card_link_noop(ctx: TestContext) -> None:
    from src.stream.card_link import (
        emit_card_link, reset_active_tool_call, set_active_tool_call,
    )
    from src.stream.child_stream import (
        install_child_stream_emitter, reset_child_stream_emitter,
    )

    cap = _Capture()
    etok = install_child_stream_emitter(cap)
    try:
        # No active tool + no chat session → nothing emitted.
        await emit_card_link("scheduler:t:r")
        assert cap.frames == [], cap.frames
        # Active tool but no chat session (an autonomous cron fire) → still
        # nothing: there is no in-chat card to link.
        ttok = set_active_tool_call("tc", "x")
        try:
            await emit_card_link("scheduler:t:r")
        finally:
            reset_active_tool_call(ttok)
        assert cap.frames == [], cap.frames
    finally:
        reset_child_stream_emitter(etok)


@test("child_live_stream", "no emitter installed → every child emit is a silent no-op")
async def t_no_emitter_noop(ctx: TestContext) -> None:
    from src.core._run_state.agent import RunContentEvent
    from src.core._runner.team._default_tools import _emit_member_card_link, _emit_member_event
    from src.models.providers.response import ToolExecution
    from src.stream.child_stream import current_child_stream_emitter, emit_child_frame

    # Nothing installed (the default in bridge / scheduler / headless runs).
    assert current_child_stream_emitter() is None

    # None of these raise, and none emit anywhere — they just return.
    await emit_child_frame(CHILD, "delta", text="x")
    await emit_child_frame(CHILD, "turn_complete")
    await _emit_member_event(
        CHILD, RunContentEvent(content="hi", run_id="rm", session_id=CHILD),
    )

    delegate_tool = ToolExecution(
        tool_name="delegate_task_to_member", tool_call_id="d1",
        tool_args={"member_id": "opus"},
    )

    class _RunResponse:
        tools = [delegate_tool]

    class _TeamSession:
        session_id = PARENT

    await _emit_member_card_link(_RunResponse(), _TeamSession(), "opus", CHILD)
    # The card-link stamp still happens (it's not gated on the emitter — it
    # also fixes the persisted run + rehydration), but nothing was streamed.
    assert delegate_tool.child_session_id == CHILD
