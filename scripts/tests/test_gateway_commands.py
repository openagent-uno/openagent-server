"""Regression tests for ``/stop``, ``/clear``, ``/new``, ``/reset`` semantics.

Bug history
-----------

**2026-04-16, v0.5.25**: ``/clear`` only called ``SessionManager.clear_queue``
— dropped pending messages but left the Claude SDK session id mapping
intact. Next message from the user arrived with the same bridge session
id (``tg:<uid>``), ``ClaudeCLI._get_client`` found the stored
``sdk_session_id``, spawned claude with ``--resume <old>``, and the
previous transcript came back.

**2026-04-16, v0.5.26**: introduced ``forget_session`` but
``_forget_all_client_sessions`` only iterated ``SessionManager.list_sessions``.
After an openagent restart that list is empty (RAM-only) while
``ClaudeCLI._sdk_sessions`` had rehydrated from sqlite, so /clear
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

    ``known_ids`` simulates the provider's hydrated map of session_ids — the
    real ClaudeCLI populates this from sqlite on startup.
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
        from openagent.gateway.sessions import SessionManager
        from openagent.gateway.server import Gateway

        self.sessions = SessionManager(agent_name="test-agent")
        self.agent = _FakeAgent(known_ids=known_ids)

        # Build a minimal Gateway object without going through __init__.
        server = Gateway.__new__(Gateway)
        server.sessions = self.sessions
        server.agent = self.agent
        server.clients = {}
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
    h.sessions._state(client).current_task = task
    h.sessions._state(client).current_session_id = sid
    await h.sessions._state(client).pending.put(object())

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
    from openagent.gateway.sessions import _QueuedItem

    h = _Harness()
    client = "bridge:telegram"
    a = h.sessions.get_or_create_session(client, "tg:aaa")
    b = h.sessions.get_or_create_session(client, "tg:bbb")

    async def _long():
        await asyncio.sleep(10)

    running = asyncio.create_task(_long())
    # Simulate B's task being the one currently running.
    h.sessions._state(client).current_task = running
    h.sessions._state(client).current_session_id = b
    # A has one message queued.
    await h.sessions._state(client).pending.put(_QueuedItem(handler=lambda: None, session_id=a))
    # B has one message queued.
    await h.sessions._state(client).pending.put(_QueuedItem(handler=lambda: None, session_id=b))

    text = await h.run_command(client, "stop", session_id=a)

    # A's stop must NOT cancel B's running task.
    assert not running.done(), "A's /stop cancelled B's running task"
    # A's queued message got dropped.
    # B's queued message still in the queue.
    pending = h.sessions._state(client).pending
    remaining = []
    while True:
        try:
            remaining.append(pending.get_nowait())
        except asyncio.QueueEmpty:
            break
    assert len(remaining) == 1, remaining
    assert remaining[0].session_id == b, remaining[0].session_id
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
