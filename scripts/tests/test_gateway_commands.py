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


@test(
    "gateway_commands",
    "/clear reaches sessions the model hydrated from disk that SessionManager never saw "
    "(the restart bug)",
)
async def t_clear_hydrated_sessions(ctx: TestContext) -> None:
    """Regression for the 2026-04-16 bug.

    Scenario: openagent was restarted (service update or manual restart).
    ``SessionManager.sessions`` is RAM-only so it starts empty. Meanwhile
    ``ClaudeCLI._sdk_sessions`` rehydrates from sqlite (seen in
    ``model.sessions_hydrated`` events). The user then types /clear in
    Telegram. Previously ``_forget_all_client_sessions`` only looked at
    ``SessionManager.list_sessions`` — which was empty — so forgot
    nothing. The next message came in on ``tg:<user_id>``, ClaudeCLI
    found the rehydrated mapping, spawned claude with
    ``--resume <old_sid>``, and the prior transcript was back.

    Fix: the gateway also iterates ``agent.known_model_session_ids()``
    filtered by the bridge prefix (``tg:`` for telegram, etc.) so resume
    state that outlived the restart still gets wiped.
    """
    h = _Harness(
        known_ids=[
            "tg:155490357",        # belongs to the telegram user
            "tg:7295922443",       # another telegram user
            "discord:99999",       # belongs to discord — must NOT be forgotten by a telegram /clear
            "scheduler:f2cd26cd",  # internal scheduler session — must NOT be forgotten
        ],
    )
    client = "bridge:telegram"
    # Crucially, do NOT attach any session to SessionManager — simulates
    # the state immediately after a process restart where the in-memory
    # list is empty but the model has rehydrated resume state.
    assert h.sessions.list_sessions(client) == [], (
        "precondition: session manager must be empty to simulate restart"
    )

    text = await h.run_command(client, "clear")

    assert "tg:155490357" in h.agent.model.forgotten, h.agent.model.forgotten
    assert "tg:7295922443" in h.agent.model.forgotten, h.agent.model.forgotten
    assert "discord:99999" not in h.agent.model.forgotten, (
        "discord session forgotten by a telegram /clear — wrong prefix filter"
    )
    assert "scheduler:f2cd26cd" not in h.agent.model.forgotten, (
        "scheduler session must not be wiped by /clear"
    )
    assert "forgot 2 prior" in text.lower(), text
    assert "fresh session" in text.lower(), text
