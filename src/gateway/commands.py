"""Shared metadata for gateway chat commands."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GatewayCommandSpec:
    name: str
    description: str
    help_text: str
    menu_visible: bool = True
    help_visible: bool = True


COMMAND_SPECS = (
    GatewayCommandSpec("new", "Forget everything and start a fresh conversation", "wipe context, clear queue, start fresh"),
    GatewayCommandSpec("reset", "Forget everything and start a fresh conversation", "wipe context, clear queue, start fresh", menu_visible=False, help_visible=False),
    GatewayCommandSpec("clear", "Forget everything and start a fresh conversation", "wipe context, clear queue, start fresh"),
    GatewayCommandSpec("stop", "Cancel the current operation (keeps context)", "cancel the current operation, keep conversation memory"),
    GatewayCommandSpec("status", "Show agent status and queue", "show agent status and queue depth"),
    GatewayCommandSpec("queue", "Show pending messages", "show pending messages", menu_visible=False),
    GatewayCommandSpec("usage", "Show current usage and budget", "show current usage and budget"),
    GatewayCommandSpec("update", "Check for updates and install", "check for updates and install"),
    GatewayCommandSpec("restart", "Restart OpenAgent", "restart OpenAgent"),
    GatewayCommandSpec("help", "Show available commands", "show this help message"),
)

COMMAND_MAP = {spec.name: spec for spec in COMMAND_SPECS}
COMMANDS = tuple(spec.name for spec in COMMAND_SPECS)
BRIDGE_COMMANDS = COMMANDS
BOT_COMMANDS = [(spec.name, spec.description) for spec in COMMAND_SPECS if spec.menu_visible]


def command_help_text() -> str:
    lines = ["Available commands:"]
    for spec in COMMAND_SPECS:
        if spec.help_visible:
            lines.append(f"• /{spec.name} — {spec.help_text}")
    return "\n".join(lines)


def bridge_welcome_text(name: str | None = None) -> str:
    """Shared welcome text shown by chat bridges."""
    greeting = f"👋 Hi {name}! I'm your OpenAgent assistant." if name else "👋 Hi! I'm your OpenAgent assistant."
    return (
        f"{greeting}\n\n"
        "Send me a message, photo, voice note, or file and I'll help.\n\n"
        "Commands:\n"
        "/stop — cancel the current operation (keeps context)\n"
        "/clear — wipe everything and start a fresh conversation\n"
        "/new — same as /clear\n"
        "/status — agent status\n"
        "/help — all commands"
    )
