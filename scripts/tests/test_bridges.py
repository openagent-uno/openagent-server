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
