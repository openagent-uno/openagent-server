#!/usr/bin/env python3
"""Messaging MCP: proactive send to Telegram, Discord, WhatsApp.

Uses the same sender classes as the channel handlers (shared code in senders.py).
Only tools for platforms with configured tokens are registered.

Environment variables:
    TELEGRAM_BOT_TOKEN — enables Telegram tools
    DISCORD_BOT_TOKEN — enables Discord tools
    GREEN_API_ID + GREEN_API_TOKEN — enables WhatsApp tools
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# Add parent packages to path so we can import from openagent
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mcp.server import Server
from mcp.types import TextContent, Tool

server = Server("openagent-messaging-mcp")

ALL_TOOLS: dict[str, Tool] = {}
TOOL_HANDLERS: dict[str, object] = {}


def _define_tool(name: str, description: str, properties: dict, required: list[str]) -> None:
    ALL_TOOLS[name] = Tool(
        name=name,
        description=description,
        inputSchema={
            "type": "object",
            "properties": properties,
            "required": required,
        },
    )


# ── Telegram ──

_tg_token = os.environ.get("TELEGRAM_BOT_TOKEN")

if _tg_token:
    from openagent.channels.senders import TelegramSender
    _tg = TelegramSender(_tg_token)

    _define_tool(
        "telegram_send_message",
        "Send a text message to a Telegram chat or user.",
        {
            "chat_id": {"type": "string", "description": "Telegram chat ID or @username"},
            "text": {"type": "string", "description": "Message text"},
            "parse_mode": {"type": "string", "enum": ["Markdown", "HTML", ""], "description": "Parse mode (optional)"},
        },
        ["chat_id", "text"],
    )
    _define_tool(
        "telegram_send_file",
        "Send a file, image, or voice message to a Telegram chat.",
        {
            "chat_id": {"type": "string", "description": "Telegram chat ID or @username"},
            "file_path": {"type": "string", "description": "Path to the file to send"},
            "caption": {"type": "string", "description": "Optional caption"},
            "type": {"type": "string", "enum": ["auto", "photo", "document", "voice", "video"], "description": "File type (default: auto)"},
        },
        ["chat_id", "file_path"],
    )

    async def _tg_send_msg(args: dict) -> str:
        r = await _tg.send_message(args["chat_id"], args["text"], args.get("parse_mode"))
        return json.dumps(r)

    async def _tg_send_file(args: dict) -> str:
        r = await _tg.send_file(args["chat_id"], args["file_path"], args.get("caption", ""), args.get("type", "auto"))
        return json.dumps(r)

    TOOL_HANDLERS["telegram_send_message"] = _tg_send_msg
    TOOL_HANDLERS["telegram_send_file"] = _tg_send_file


# ── Discord ──

_dc_token = os.environ.get("DISCORD_BOT_TOKEN")

if _dc_token:
    from openagent.channels.senders import DiscordSender
    _dc = DiscordSender(_dc_token)

    _define_tool(
        "discord_send_message",
        "Send a text message to a Discord channel.",
        {
            "channel_id": {"type": "string", "description": "Discord channel ID"},
            "text": {"type": "string", "description": "Message text"},
        },
        ["channel_id", "text"],
    )
    _define_tool(
        "discord_send_file",
        "Send a file to a Discord channel.",
        {
            "channel_id": {"type": "string", "description": "Discord channel ID"},
            "file_path": {"type": "string", "description": "Path to the file to send"},
            "caption": {"type": "string", "description": "Optional message text"},
        },
        ["channel_id", "file_path"],
    )

    async def _dc_send_msg(args: dict) -> str:
        r = await _dc.send_message(args["channel_id"], args["text"])
        return json.dumps(r)

    async def _dc_send_file(args: dict) -> str:
        r = await _dc.send_file(args["channel_id"], args["file_path"], args.get("caption", ""))
        return json.dumps(r)

    TOOL_HANDLERS["discord_send_message"] = _dc_send_msg
    TOOL_HANDLERS["discord_send_file"] = _dc_send_file


# ── WhatsApp ──

_wa_id = os.environ.get("GREEN_API_ID")
_wa_token = os.environ.get("GREEN_API_TOKEN")

if _wa_id and _wa_token:
    from openagent.channels.senders import WhatsAppSender
    _wa = WhatsAppSender(_wa_id, _wa_token)

    _define_tool(
        "whatsapp_send_message",
        "Send a text message via WhatsApp.",
        {
            "phone": {"type": "string", "description": "Phone number with country code (e.g. 393331234567) or chat ID"},
            "text": {"type": "string", "description": "Message text"},
        },
        ["phone", "text"],
    )
    _define_tool(
        "whatsapp_send_file",
        "Send a file via WhatsApp.",
        {
            "phone": {"type": "string", "description": "Phone number or chat ID"},
            "file_path": {"type": "string", "description": "Path to the file to send"},
            "caption": {"type": "string", "description": "Optional caption"},
        },
        ["phone", "file_path"],
    )

    async def _wa_send_msg(args: dict) -> str:
        r = await _wa.send_message(args["phone"], args["text"])
        return json.dumps(r)

    async def _wa_send_file(args: dict) -> str:
        r = await _wa.send_file(args["phone"], args["file_path"], args.get("caption", ""))
        return json.dumps(r)

    TOOL_HANDLERS["whatsapp_send_message"] = _wa_send_msg
    TOOL_HANDLERS["whatsapp_send_file"] = _wa_send_file


# ── MCP Server ──

@server.list_tools()
async def list_tools() -> list[Tool]:
    return list(ALL_TOOLS.values())


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        raise ValueError(f"Unknown tool: {name}")
    result = await handler(arguments)
    return [TextContent(type="text", text=result)]


async def main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
