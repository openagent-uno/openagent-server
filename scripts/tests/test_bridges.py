"""Bridges — import-only smoke.

Full bridge integration needs real tokens (Telegram bot token, Discord
token, WhatsApp Green API ID/token) which we don't have in CI. This
test at least proves the modules compile and expose their primary class
so a typo or dead import doesn't ship silently.
"""
from __future__ import annotations

import inspect

from ._framework import TestContext, test


@test("bridges", "telegram bridge module imports")
async def t_telegram_import(ctx: TestContext) -> None:
    import openagent.bridges.telegram as mod  # noqa: F401
    # Either a TelegramBridge class or a start() coroutine — accept either shape
    has_class = any(inspect.isclass(obj) for _, obj in inspect.getmembers(mod))
    assert has_class, "telegram bridge exposes no class"


@test("bridges", "discord bridge module imports")
async def t_discord_import(ctx: TestContext) -> None:
    import openagent.bridges.discord as mod  # noqa: F401
    has_class = any(inspect.isclass(obj) for _, obj in inspect.getmembers(mod))
    assert has_class, "discord bridge exposes no class"


@test("bridges", "whatsapp bridge module imports")
async def t_whatsapp_import(ctx: TestContext) -> None:
    import openagent.bridges.whatsapp as mod  # noqa: F401
    has_class = any(inspect.isclass(obj) for _, obj in inspect.getmembers(mod))
    assert has_class, "whatsapp bridge exposes no class"


@test("bridges", "BaseBridge exists and has the expected lifecycle methods")
async def t_bridge_base(ctx: TestContext) -> None:
    from openagent.bridges.base import BaseBridge, format_tool_status
    # Each concrete bridge subclasses BaseBridge; confirm the contract
    # surface we rely on is still there.
    for method in ("start", "stop", "send_message", "send_command"):
        assert hasattr(BaseBridge, method), f"BaseBridge is missing {method!r}"
    # format_tool_status is imported by the concrete bridges
    assert format_tool_status("Thinking...") == "Thinking..."
    assert format_tool_status('{"tool":"bash","status":"running"}') == "Using bash..."


class _FakeBridge:
    """Subclass stand-in that skips the WS connect. Only used for send_message
    tests where we control the pending future directly."""

    def __init__(self) -> None:
        from openagent.bridges.base import BaseBridge

        self._real = BaseBridge.__new__(BaseBridge)
        self._real._pending = {}
        self._real._status_callbacks = {}
        self._real._session_locks = {}
        self._real._ws = object()  # non-None bypasses the "not connected" guard
        self._sent: list[dict] = []

        async def fake_send(payload: dict) -> None:
            self._sent.append(payload)

        self._real._send_gateway_json = fake_send  # type: ignore[assignment]

    def future_for(self, sid: str):
        return self._real._pending[sid]

    async def send(self, text: str, sid: str):
        return await self._real.send_message(text=text, session_id=sid)


@test("bridges", "send_message resolves when the gateway future is set")
async def t_send_message_normal(ctx: TestContext) -> None:
    import asyncio

    fb = _FakeBridge()

    async def resolver():
        # Wait until send_message registered the pending future, then resolve.
        for _ in range(500):
            if "s1" in fb._real._pending:
                fb.future_for("s1").set_result({"type": "response", "text": "pong"})
                return
            await asyncio.sleep(0.001)
        raise AssertionError("pending future never appeared")

    result, _ = await asyncio.gather(fb.send("ping", "s1"), resolver())
    assert result["text"] == "pong", result


@test("bridges", "send_message raises CancelledError when /stop cancels the caller")
async def t_send_message_cancelled(ctx: TestContext) -> None:
    import asyncio

    fb = _FakeBridge()
    task = asyncio.create_task(fb.send("ping", "s-cancel"))
    # Give the bridge a moment to register the pending future + send payload.
    for _ in range(500):
        if "s-cancel" in fb._real._pending:
            break
        await asyncio.sleep(0.001)
    assert "s-cancel" in fb._real._pending, "send_message never registered"
    task.cancel()
    raised: BaseException | None = None
    try:
        await task
    except asyncio.CancelledError as e:
        raised = e
    assert raised is not None, "CancelledError was swallowed"
    # Defensive cleanup should have popped the entry.
    assert "s-cancel" not in fb._real._pending, "pending future leaked"
