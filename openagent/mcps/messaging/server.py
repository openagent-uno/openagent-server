#!/usr/bin/env python3
"""Messaging MCP: proactive send to Telegram, Discord, WhatsApp.

This MCP lets the agent initiate messages (not just respond).
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
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

server = Server("openagent-messaging-mcp")

# ── Tool definitions (registered dynamically based on available tokens) ──

ALL_TOOLS: dict[str, Tool] = {}


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
_tg_bot = None

if _tg_token:
    _define_tool(
        "telegram_send_message",
        "Send a text message to a Telegram chat or user.",
        {
            "chat_id": {"type": "string", "description": "Telegram chat ID or @username"},
            "text": {"type": "string", "description": "Message text (supports Markdown)"},
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
            "type": {"type": "string", "enum": ["auto", "photo", "document", "voice", "video"], "description": "File type (default: auto-detect)"},
        },
        ["chat_id", "file_path"],
    )


async def _get_tg_bot():
    global _tg_bot
    if _tg_bot is None:
        from telegram import Bot
        _tg_bot = Bot(token=_tg_token)
    return _tg_bot


async def _telegram_send_message(args: dict) -> str:
    bot = await _get_tg_bot()
    result = await bot.send_message(chat_id=args["chat_id"], text=args["text"], parse_mode="Markdown")
    return json.dumps({"ok": True, "message_id": result.message_id})


async def _telegram_send_file(args: dict) -> str:
    bot = await _get_tg_bot()
    chat_id = args["chat_id"]
    file_path = args["file_path"]
    caption = args.get("caption", "")
    file_type = args.get("type", "auto")

    path = Path(file_path)
    if file_type == "auto":
        suffix = path.suffix.lower()
        if suffix in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
            file_type = "photo"
        elif suffix in (".ogg", ".mp3", ".wav", ".m4a"):
            file_type = "voice"
        elif suffix in (".mp4", ".mov", ".avi"):
            file_type = "video"
        else:
            file_type = "document"

    with open(path, "rb") as f:
        if file_type == "photo":
            result = await bot.send_photo(chat_id=chat_id, photo=f, caption=caption)
        elif file_type == "voice":
            result = await bot.send_voice(chat_id=chat_id, voice=f, caption=caption)
        elif file_type == "video":
            result = await bot.send_video(chat_id=chat_id, video=f, caption=caption)
        else:
            result = await bot.send_document(chat_id=chat_id, document=f, caption=caption, filename=path.name)

    return json.dumps({"ok": True, "message_id": result.message_id})


# ── Discord ──

_dc_token = os.environ.get("DISCORD_BOT_TOKEN")
_dc_client = None
_dc_ready = asyncio.Event()

if _dc_token:
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


async def _get_dc_client():
    global _dc_client
    if _dc_client is None:
        import discord
        intents = discord.Intents.default()
        _dc_client = discord.Client(intents=intents)

        @_dc_client.event
        async def on_ready():
            _dc_ready.set()

        asyncio.create_task(_dc_client.start(_dc_token))
        await asyncio.wait_for(_dc_ready.wait(), timeout=30)
    return _dc_client


async def _discord_send_message(args: dict) -> str:
    client = await _get_dc_client()
    channel = client.get_channel(int(args["channel_id"]))
    if not channel:
        channel = await client.fetch_channel(int(args["channel_id"]))
    msg = await channel.send(args["text"])
    return json.dumps({"ok": True, "message_id": msg.id})


async def _discord_send_file(args: dict) -> str:
    import discord
    client = await _get_dc_client()
    channel = client.get_channel(int(args["channel_id"]))
    if not channel:
        channel = await client.fetch_channel(int(args["channel_id"]))
    file = discord.File(args["file_path"])
    msg = await channel.send(content=args.get("caption", ""), file=file)
    return json.dumps({"ok": True, "message_id": msg.id})


# ── WhatsApp (Green API) ──

_wa_id = os.environ.get("GREEN_API_ID")
_wa_token = os.environ.get("GREEN_API_TOKEN")
_wa_api = None

if _wa_id and _wa_token:
    _define_tool(
        "whatsapp_send_message",
        "Send a text message via WhatsApp.",
        {
            "phone": {"type": "string", "description": "Phone number with country code (e.g. '393331234567') or chat ID (e.g. '393331234567@c.us')"},
            "text": {"type": "string", "description": "Message text"},
        },
        ["phone", "text"],
    )
    _define_tool(
        "whatsapp_send_file",
        "Send a file, image, or voice message via WhatsApp.",
        {
            "phone": {"type": "string", "description": "Phone number or chat ID"},
            "file_path": {"type": "string", "description": "Path to the file to send"},
            "caption": {"type": "string", "description": "Optional caption"},
        },
        ["phone", "file_path"],
    )


def _get_wa_api():
    global _wa_api
    if _wa_api is None:
        from whatsapp_api_client_python import API as GreenAPI
        _wa_api = GreenAPI.GreenApi(_wa_id, _wa_token)
    return _wa_api


def _normalize_wa_chat_id(phone: str) -> str:
    if "@" in phone:
        return phone
    return f"{phone}@c.us"


async def _whatsapp_send_message(args: dict) -> str:
    api = _get_wa_api()
    chat_id = _normalize_wa_chat_id(args["phone"])
    result = await asyncio.to_thread(api.sending.sendMessage, chat_id, args["text"])
    return json.dumps({"ok": True, "id": str(result.data) if hasattr(result, 'data') else "sent"})


async def _whatsapp_send_file(args: dict) -> str:
    api = _get_wa_api()
    chat_id = _normalize_wa_chat_id(args["phone"])
    file_path = args["file_path"]
    caption = args.get("caption", "")
    fname = Path(file_path).name
    result = await asyncio.to_thread(
        api.sending.sendFileByUpload, chat_id, file_path, fname, caption,
    )
    return json.dumps({"ok": True, "id": str(result.data) if hasattr(result, 'data') else "sent"})


# ── MCP Server handlers ──

TOOL_HANDLERS = {
    "telegram_send_message": _telegram_send_message,
    "telegram_send_file": _telegram_send_file,
    "discord_send_message": _discord_send_message,
    "discord_send_file": _discord_send_file,
    "whatsapp_send_message": _whatsapp_send_message,
    "whatsapp_send_file": _whatsapp_send_file,
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return list(ALL_TOOLS.values())


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        raise ValueError(f"Unknown tool: {name}")
    if name not in ALL_TOOLS:
        raise ValueError(f"Tool {name} is not available (missing credentials)")
    result = await handler(arguments)
    return [TextContent(type="text", text=result)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
