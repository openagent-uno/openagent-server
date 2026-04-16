"""Regression tests for ``/stop``, ``/clear``, ``/new`` semantics.

Reported bug (2026-04-16): user sent ``/clear`` in Telegram expecting the
agent to forget everything, then ``ci sei?``, and the agent immediately
resumed its previous maestro/android chain. Root cause: ``/clear`` only
cleared the pending message queue; the model's SDK session id mapping for
this chat (``tg:<uid>`` → ``<claude_sdk_session_id>``) was intact, so the
next message went through with ``--resume <prior>`` and picked up the
same conversation.

These tests pin the behaviour for the three relevant commands so a
regression is caught before the next rollout.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ._framework import TestContext, test


# ── Fakes for the gateway server's dependencies ────────────────────────


class _FakeModel:
    """Records close_session / forget_session calls so tests can assert."""

    def __init__(self) -> None:
        self.closed: list[str] = []
        self.forgotten: list[str] = []

    async def close_session(self, session_id: str) -> None:
        self.closed.append(session_id)

    async def forget_session(self, session_id: str) -> None:
        # Forget = close + erase resume state; simulate both effects.
        self.closed.append(session_id)
        self.forgotten.append(session_id)


class _FakeAgent:
    """Just enough Agent surface for ``_handle_command`` to run."""

    def __init__(self) -> None:
        self.model = _FakeModel()
        self._initialized = True

    def _prepare_model_runtime(self, _m: Any) -> None:
        return None

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

    def __init__(self) -> None:
        from openagent.gateway.sessions import SessionManager
        from openagent.gateway.server import Gateway

        self.sessions = SessionManager(agent_name="test-agent")
        self.agent = _FakeAgent()

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

    async def run_command(self, client_id: str, name: str) -> str:
        self._last_result_text = None
        await self.server._handle_command(self.ws, client_id, name)
        return self._last_result_text or ""


# ── Tests ─────────────────────────────────────────────────────────────


@test("gateway_commands", "/stop cancels running, clears queue, KEEPS context")
async def t_stop_preserves_context(ctx: TestContext) -> None:
    h = _Harness()
    client = "bridge:telegram"
    # Attach a pre-existing session so stop has state to touch.
    sid = h.sessions.get_or_create_session(client, "tg:155490357")
    # Pretend there's a running task and something in the queue so /stop's
    # "stopped + cleared N" text path is exercised.

    async def _dummy():
        await asyncio.sleep(10)

    task = asyncio.create_task(_dummy())
    h.sessions._state(client).current_task = task
    await h.sessions._state(client).pending.put(object())

    text = await h.run_command(client, "stop")

    assert "Stopped" in text, text
    assert "cleared 1" in text, text
    # Context is NOT erased — no close or forget should have been issued.
    assert h.agent.model.forgotten == [], h.agent.model.forgotten
    assert h.agent.model.closed == [], h.agent.model.closed
    # Session still exists.
    assert sid in h.sessions.list_sessions(client), h.sessions.list_sessions(client)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@test("gateway_commands", "/clear cancels running, clears queue, AND forgets context")
async def t_clear_wipes_context(ctx: TestContext) -> None:
    h = _Harness()
    client = "bridge:telegram"
    tg_sid = h.sessions.get_or_create_session(client, "tg:155490357")
    # Also add a second session to confirm "all sessions" semantics.
    other_sid = h.sessions.create_session(client)
    await h.sessions._state(client).pending.put(object())
    await h.sessions._state(client).pending.put(object())

    text = await h.run_command(client, "clear")

    # Both pre-existing sessions must be forgotten.
    assert tg_sid in h.agent.model.forgotten, (
        f"expected {tg_sid} in {h.agent.model.forgotten}"
    )
    assert other_sid in h.agent.model.forgotten, (
        f"expected {other_sid} in {h.agent.model.forgotten}"
    )
    lt = text.lower()
    assert "forgot 2 prior" in lt, text
    assert "cleared 2 queued" in lt, text
    assert "fresh session" in lt, text


@test("gateway_commands", "/new is an alias of /clear — full wipe")
async def t_new_wipes_context(ctx: TestContext) -> None:
    h = _Harness()
    client = "bridge:telegram"
    tg_sid = h.sessions.get_or_create_session(client, "tg:155490357")

    text = await h.run_command(client, "new")

    assert tg_sid in h.agent.model.forgotten, h.agent.model.forgotten
    assert "fresh session" in text.lower(), text


@test("gateway_commands", "/reset also wipes (same code path)")
async def t_reset_wipes_context(ctx: TestContext) -> None:
    h = _Harness()
    client = "bridge:telegram"
    tg_sid = h.sessions.get_or_create_session(client, "tg:155490357")

    await h.run_command(client, "reset")

    assert tg_sid in h.agent.model.forgotten, h.agent.model.forgotten


@test("gateway_commands", "/clear on a brand-new client has nothing to forget but doesn't crash")
async def t_clear_no_sessions(ctx: TestContext) -> None:
    h = _Harness()
    client = "bridge:telegram"
    text = await h.run_command(client, "clear")
    # No pre-existing sessions → nothing to forget, no "forgot N" phrase.
    assert "forgot" not in text.lower(), text
    assert "fresh session" in text.lower(), text
    assert h.agent.model.forgotten == []
