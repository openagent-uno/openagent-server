"""Regression tests for ``/stop``, ``/clear``, ``/new``, ``/reset`` semantics.

Bug history
-----------

**2026-04-16, v0.5.25**: ``/clear`` only called ``SessionManager.clear_queue``
— dropped pending messages but left the provider's resume session id
mapping intact. Next message from the user arrived with the same bridge
session id (``tg:<uid>``), the provider found the stored resume id,
resumed the prior transcript, and the previous conversation came back.

**2026-04-16, v0.5.26**: introduced ``forget_session`` but
``_forget_all_client_sessions`` only iterated ``SessionManager.list_sessions``.
After an openagent restart that list is empty (RAM-only) while the
provider's session map had rehydrated from sqlite, so /clear
forgot nothing. v0.5.27 patched it by also iterating the model's
``known_session_ids()`` filtered by a bridge prefix.

**2026-04-16, v0.5.27**: the prefix-filtered wipe was over-broad —
one telegram user's /clear wiped every telegram user on the same bot.
v0.5.28 scopes /stop /clear /new /reset to the sender's session_id
when the bridge passes one, and keeps the legacy client-wide wipe as
a fallback only when ``session_id`` is absent (direct ws admin clients,
etc.).

These tests pin those three fixes in place.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ._framework import TestContext, test


# ── Fakes for the gateway server's dependencies ────────────────────────


class _FakeModel:
    """Records close_session / forget_session calls so tests can assert.

    ``known_ids`` simulates the provider's hydrated map of session_ids — a
    real provider populates this from sqlite on startup.
    """

    def __init__(self, known_ids: list[str] | None = None) -> None:
        self.closed: list[str] = []
        self.forgotten: list[str] = []
        self.known_ids: list[str] = list(known_ids or [])

    async def close_session(self, session_id: str) -> None:
        self.closed.append(session_id)

    async def forget_session(self, session_id: str) -> None:
        # Forget = close + erase resume state; simulate both effects.
        self.closed.append(session_id)
        self.forgotten.append(session_id)
        if session_id in self.known_ids:
            self.known_ids.remove(session_id)

    def known_session_ids(self) -> list[str]:
        return list(self.known_ids)


class _FakeAgent:
    """Just enough Agent surface for ``_handle_command`` to run."""

    def __init__(self, known_ids: list[str] | None = None) -> None:
        self.model = _FakeModel(known_ids=known_ids)
        self._initialized = True

    def _prepare_model_runtime(self, _m: Any) -> None:
        return None

    def known_model_session_ids(self) -> list[str]:
        return list(self.model.known_session_ids())

    async def forget_session(self, session_id: str | None) -> None:
        if not session_id:
            return
        forget = getattr(self.model, "forget_session", None)
        if callable(forget):
            await forget(session_id)
            return
        close = getattr(self.model, "close_session", None)
        if callable(close):
            await close(session_id)

    async def release_session(self, session_id: str | None) -> None:
        if not session_id:
            return
        await self.model.close_session(session_id)


@dataclass
class _SentMsg:
    payload: dict[str, Any] = field(default_factory=dict)


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


class _Harness:
    """Wire up SessionManager + fake agent + the real ``_handle_command``."""

    def __init__(self, *, known_ids: list[str] | None = None) -> None:
        from src.gateway.sessions import SessionManager
        from src.gateway.server import Gateway

        self.sessions = SessionManager(agent_name="test-agent")
        self.agent = _FakeAgent(known_ids=known_ids)

        # Build a minimal Gateway object without going through __init__.
        server = Gateway.__new__(Gateway)
        server.sessions = self.sessions
        server.agent = self.agent
        server.clients = {}
        # The real registry that holds live chat turns. ``__init__`` sets
        # this; tests building the Gateway via ``__new__`` must too, since
        # ``/stop`` now routes through it (``_interrupt_stream_sessions``).
        server._stream_sessions = {}
        server._safe_ws_send_json = self._capture
        self.server = server
        self.ws = _FakeWS()
        self._last_result_text: str | None = None

    async def _capture(self, _ws, payload: dict[str, Any]) -> None:
        if payload.get("type") == "command_result":
            self._last_result_text = payload.get("text")

    async def run_command(
        self, client_id: str, name: str, session_id: str | None = None
    ) -> str:
        self._last_result_text = None
        await self.server._handle_command(self.ws, client_id, name, session_id)
        return self._last_result_text or ""


# ── /stop ──────────────────────────────────────────────────────────────


@test("gateway_commands", "/stop cancels running, clears queue, KEEPS context (client-wide)")
async def t_stop_preserves_context(ctx: TestContext) -> None:
    h = _Harness()
    client = "bridge:telegram"
    sid = h.sessions.get_or_create_session(client, "tg:155490357")

    async def _dummy():
        await asyncio.sleep(10)

    task = asyncio.create_task(_dummy())
    ss = h.sessions._session_state(client, sid)
    ss.current_task = task
    await ss.pending.put(object())

    text = await h.run_command(client, "stop")

    assert "Stopped" in text, text
    assert "cleared 1" in text, text
    assert h.agent.model.forgotten == [], h.agent.model.forgotten
    assert sid in h.sessions.list_sessions(client), h.sessions.list_sessions(client)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@test(
    "gateway_commands",
    "/stop scoped to session_id only cancels MATCHING running task (not siblings)",
)
async def t_stop_scoped_preserves_others(ctx: TestContext) -> None:
    """Two users on the same telegram bot. User B is mid-turn; user A
    issues /stop. A's stop must NOT interrupt B's running task.
    """
    from src.gateway.sessions import _QueuedItem

    h = _Harness()
    client = "bridge:telegram"
    a = h.sessions.get_or_create_session(client, "tg:aaa")
    b = h.sessions.get_or_create_session(client, "tg:bbb")

    async def _long():
        await asyncio.sleep(10)

    running = asyncio.create_task(_long())
    # B's task is currently running — now lives on B's own _SessionState.
    ss_b = h.sessions._session_state(client, b)
    ss_b.current_task = running
    ss_a = h.sessions._session_state(client, a)
    # A has one message queued on A's queue.
    await ss_a.pending.put(_QueuedItem(handler=lambda: None, session_id=a))
    # B has one message queued on B's queue.
    await ss_b.pending.put(_QueuedItem(handler=lambda: None, session_id=b))

    text = await h.run_command(client, "stop", session_id=a)

    # A's stop must NOT cancel B's running task.
    assert not running.done(), "A's /stop cancelled B's running task"
    # A's queued message got dropped; A's queue is now empty.
    assert ss_a.pending.qsize() == 0, ss_a.pending.qsize()
    # B's queued message still sits on B's queue.
    b_remaining = []
    while True:
        try:
            b_remaining.append(ss_b.pending.get_nowait())
        except asyncio.QueueEmpty:
            break
    assert len(b_remaining) == 1, b_remaining
    assert b_remaining[0].session_id == b, b_remaining[0].session_id
    # The response text should reflect "nothing running" for A since the
    # current task isn't theirs, but A's queued msg WAS cleared.
    assert "cleared 1" in text, text
    running.cancel()
    try:
        await running
    except asyncio.CancelledError:
        pass


# ── /clear, /new, /reset (scoped) ──────────────────────────────────────


@test("gateway_commands", "/clear scoped to session_id only forgets THAT session")
async def t_clear_scoped(ctx: TestContext) -> None:
    """The core multi-user bug: user A's /clear must not touch user B."""
    h = _Harness(
        known_ids=[
            "tg:aaa",  # user A
            "tg:bbb",  # user B — must survive
            "tg:ccc",  # user C — must survive
        ],
    )
    client = "bridge:telegram"
    h.sessions.get_or_create_session(client, "tg:aaa")
    h.sessions.get_or_create_session(client, "tg:bbb")
    h.sessions.get_or_create_session(client, "tg:ccc")

    text = await h.run_command(client, "clear", session_id="tg:aaa")

    assert h.agent.model.forgotten == ["tg:aaa"], h.agent.model.forgotten
    assert "forgot 1 prior" in text.lower(), text
    assert "fresh session" in text.lower(), text
    # User B and C still know their own sessions.
    assert "tg:bbb" in h.agent.model.known_session_ids()
    assert "tg:ccc" in h.agent.model.known_session_ids()


@test("gateway_commands", "/new scoped = /clear scoped")
async def t_new_scoped(ctx: TestContext) -> None:
    h = _Harness(known_ids=["tg:a", "tg:b"])
    client = "bridge:telegram"
    h.sessions.get_or_create_session(client, "tg:a")
    h.sessions.get_or_create_session(client, "tg:b")

    await h.run_command(client, "new", session_id="tg:a")

    assert h.agent.model.forgotten == ["tg:a"], h.agent.model.forgotten


@test("gateway_commands", "/reset scoped = /clear scoped")
async def t_reset_scoped(ctx: TestContext) -> None:
    h = _Harness(known_ids=["tg:a", "tg:b"])
    client = "bridge:telegram"
    h.sessions.get_or_create_session(client, "tg:a")
    h.sessions.get_or_create_session(client, "tg:b")

    await h.run_command(client, "reset", session_id="tg:b")

    assert h.agent.model.forgotten == ["tg:b"], h.agent.model.forgotten


# ── /clear (unscoped legacy fallback for direct ws / admin) ────────────


@test(
    "gateway_commands",
    "/clear without session_id falls back to client-wide wipe (legacy / admin path)",
)
async def t_clear_unscoped_wipes_client(ctx: TestContext) -> None:
    """Direct ws clients and administrative flows that don't pass a
    session_id still get the wide behaviour — convenient for a lone
    user clearing everything in one go.
    """
    h = _Harness(known_ids=["tg:aaa", "tg:bbb", "discord:99", "scheduler:uu"])
    client = "bridge:telegram"
    h.sessions.get_or_create_session(client, "tg:aaa")
    h.sessions.get_or_create_session(client, "tg:bbb")

    await h.run_command(client, "clear")  # no session_id

    # Both telegram users get wiped (prefix filter), discord and scheduler survive.
    assert "tg:aaa" in h.agent.model.forgotten, h.agent.model.forgotten
    assert "tg:bbb" in h.agent.model.forgotten, h.agent.model.forgotten
    assert "discord:99" not in h.agent.model.forgotten
    assert "scheduler:uu" not in h.agent.model.forgotten


@test(
    "gateway_commands",
    "/clear unscoped still reaches sessions the model hydrated from disk post-restart",
)
async def t_clear_unscoped_hydrated(ctx: TestContext) -> None:
    """Regression for v0.5.26 bug — the known_session_ids fallback still
    fires when the unscoped fallback path is taken."""
    h = _Harness(known_ids=["tg:155490357", "tg:7295922443"])
    client = "bridge:telegram"
    assert h.sessions.list_sessions(client) == []

    await h.run_command(client, "clear")  # no session_id → legacy wipe

    assert "tg:155490357" in h.agent.model.forgotten
    assert "tg:7295922443" in h.agent.model.forgotten


@test("gateway_commands", "/clear on an empty brand-new client doesn't crash")
async def t_clear_no_sessions(ctx: TestContext) -> None:
    h = _Harness()
    client = "bridge:telegram"
    text = await h.run_command(client, "clear")
    assert "forgot" not in text.lower(), text
    assert "fresh session" in text.lower(), text
    assert h.agent.model.forgotten == []


# ── /stop against a REAL live StreamSession turn ───────────────────────
#
# The tests above hand-inject ``current_task`` onto the SessionManager —
# the slot the live stream path never populates — so they proved only
# that ``stop_current`` cancels a task you give it, NOT that ``/stop``
# stops a running chat turn. These drive the ACTUAL registry: a real
# ``StreamSession`` with a model that is genuinely mid-stream, attached
# to ``gateway._stream_sessions`` exactly as ``_handle_stream_frame``
# does, then assert the command cancels it end to end.


class _EngagedAgent:
    """``run_stream`` yields one delta (engages the turn) then blocks
    forever — so a turn is genuinely streaming when we try to stop it.

    ``engaged`` fires once the first token is yielded (the StreamSession
    flips ``_current_turn_started`` here too); ``cancelled`` fires when
    the asyncio cancellation actually reaches the model loop. A working
    stop must set ``cancelled``; a no-op stop never will.
    """

    name = "engaged"
    db = None

    def __init__(self) -> None:
        self.engaged = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def run_stream(self, *, message, user_id, session_id,
                         attachments=None, on_status=None, author=None):
        yield {"kind": "delta", "text": "thinking…"}
        self.engaged.set()
        try:
            await asyncio.sleep(30)  # never completes on its own
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        yield {"kind": "done", "text": ""}  # pragma: no cover — unreachable

    def last_response_meta(self, _sid):
        return {"model": "engaged"}

    async def request_cancel(self, _sid):
        # No cooperative cancel in this mock → barge-in falls back to the
        # hard asyncio cancel, which this fake honours via ``self.cancelled``.
        return False


async def _poll(condition, *, timeout: float = 3.0, step: float = 0.01) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if condition():
            return True
        await asyncio.sleep(step)
    return False


async def _attach_live_turn(h: "_Harness", client: str, raw_sid: str):
    """Attach a real StreamSession to the gateway and fire one engaged turn.

    Returns ``(session, agent, sid)``. ``coalesce_window_ms=0`` dispatches
    the turn immediately (no debounce wait) so the turn is in flight by the
    time the agent engages.
    """
    from src.stream.session import StreamSession
    from src.stream.channel import RealtimeChannel
    from src.stream.events import TextFinal, now_ms
    from src.gateway.server import _StreamHolder

    sid = h.sessions.get_or_create_session(client, raw_sid)
    agent = _EngagedAgent()
    sess = StreamSession(
        agent, client_id=client, session_id=sid, coalesce_window_ms=0,
    )

    async def _null(_db):
        return None

    await sess.start(stt_factory=_null, tts_factory=_null)
    # A RealtimeChannel draining the outbound queue onto a capture list —
    # mirrors how a real connection consumes frames, so the queue can't
    # fill and the runner's terminal-frame publish never blocks.
    sent: list = []

    async def _send_wire(payload) -> bool:
        sent.append(payload)
        return True

    channel = RealtimeChannel(sess, _send_wire)
    await channel.start()
    h.server._stream_sessions[(client, sid)] = _StreamHolder(
        session=sess, channel=channel,
    )

    await sess.push_in(TextFinal(
        session_id=sid, seq=1, ts_ms=now_ms(), text="hi", source="user_typed",
    ))
    await asyncio.wait_for(agent.engaged.wait(), timeout=3.0)
    return sess, agent, sid, channel, sent


@test("gateway_commands", "/stop cancels a LIVE StreamSession turn end-to-end (the real path)")
async def t_stop_cancels_live_stream_turn(ctx: TestContext) -> None:
    h = _Harness()
    client = "device:abc"
    sess, agent, sid, channel, sent = await _attach_live_turn(h, client, "app:tab1")
    try:
        assert sess.has_active_turn(), "turn should be in flight after engage"

        # The exact wire path the app/CLI/bridges take: a COMMAND 'stop'.
        text = await h.run_command(client, "stop", session_id=sid)

        # Cancellation must reach the model loop — the whole point.
        reached = await _poll(lambda: agent.cancelled.is_set())
        assert reached, "/stop did not cancel the running model loop"
        assert "Stopped" in text, f"expected 'Stopped', got: {text!r}"

        # The live turn slot must free.
        freed = await _poll(lambda: not sess.has_active_turn())
        assert freed, "turn still active after /stop"

        # A terminal turn_complete must reach the client so the UI un-sticks.
        saw_complete = await _poll(
            lambda: any(p.get("type") == "turn_complete" for p in sent),
        )
        assert saw_complete, (
            f"no turn_complete frame after /stop; frames={[p.get('type') for p in sent]}"
        )
    finally:
        await channel.close()


@test("gateway_commands", "/stop on an idle StreamSession reports 'Nothing running.'")
async def t_stop_idle_stream_session(ctx: TestContext) -> None:
    """A /stop with a live session attached but no turn running must not
    claim it stopped something — guards against a false-positive 'Stopped'.
    """
    h = _Harness()
    client = "device:idle"
    sess, agent, sid, channel, sent = await _attach_live_turn(h, client, "app:tab1")
    try:
        # Let the engaged turn be cancelled first so the session goes idle.
        await h.run_command(client, "stop", session_id=sid)
        await _poll(lambda: not sess.has_active_turn())
        # Second /stop — nothing is running now.
        text = await h.run_command(client, "stop", session_id=sid)
        assert "Nothing running" in text, f"expected idle report, got: {text!r}"
    finally:
        await channel.close()


@test("gateway_commands", "/clear cancels a running StreamSession turn AND forgets the session")
async def t_clear_stops_live_turn(ctx: TestContext) -> None:
    """A wipe issued mid-turn must halt the live turn (else a stale reply
    interleaves into the fresh session) and forget prior context.
    """
    h = _Harness(known_ids=["app:tab1"])
    client = "device:abc"
    sess, agent, sid, channel, sent = await _attach_live_turn(h, client, "app:tab1")
    try:
        text = await h.run_command(client, "clear", session_id=sid)
        reached = await _poll(lambda: agent.cancelled.is_set())
        assert reached, "/clear did not cancel the running turn"
        assert "fresh session" in text.lower(), text
        assert h.agent.model.forgotten == [sid], h.agent.model.forgotten
    finally:
        await channel.close()


@test("gateway_commands", "/status reports Busy while a StreamSession turn runs")
async def t_status_reflects_live_turn(ctx: TestContext) -> None:
    """Before the fix, /status read the always-idle SessionManager and
    said 'Idle' mid-turn. It must reflect the StreamSession registry.
    """
    h = _Harness()
    client = "device:abc"
    sess, agent, sid, channel, sent = await _attach_live_turn(h, client, "app:tab1")
    try:
        text = await h.run_command(client, "status", session_id=sid)
        assert "Busy" in text, f"expected Busy mid-turn, got: {text!r}"
    finally:
        await channel.close()
