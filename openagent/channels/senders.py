"""Shared send logic for all channels.

Used by both channel handlers (responding) and the messaging MCP (proactive send).
Each sender is stateless — just needs credentials + target.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Telegram Sender ──

class TelegramSender:
    """Send messages/files via Telegram Bot API. Shared by channel + MCP."""

    def __init__(self, token: str):
        self.token = token
        self._bot = None

    async def _get_bot(self):
        if self._bot is None:
            from telegram import Bot
            self._bot = Bot(token=self.token)
        return self._bot

    async def send_message(self, chat_id: str | int, text: str, parse_mode: str | None = None) -> dict:
        bot = await self._get_bot()
        result = await bot.send_message(chat_id=chat_id, text=text, parse_mode=parse_mode)
        return {"ok": True, "message_id": result.message_id}

    async def send_file(self, chat_id: str | int, file_path: str, caption: str = "", file_type: str = "auto") -> dict:
        bot = await self._get_bot()
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

        return {"ok": True, "message_id": result.message_id}


# ── Discord Sender ──

class DiscordSender:
    """Send messages/files via Discord bot. Shared by channel + MCP."""

    def __init__(self, token: str):
        self.token = token
        self._client = None
        self._ready = asyncio.Event()

    async def _ensure_client(self):
        if self._client is None:
            import discord
            intents = discord.Intents.default()
            self._client = discord.Client(intents=intents)

            @self._client.event
            async def on_ready():
                self._ready.set()

            asyncio.create_task(self._client.start(self.token))
            await asyncio.wait_for(self._ready.wait(), timeout=30)
        return self._client

    async def _get_channel(self, channel_id: str | int):
        client = await self._ensure_client()
        ch = client.get_channel(int(channel_id))
        if not ch:
            ch = await client.fetch_channel(int(channel_id))
        return ch

    async def send_message(self, channel_id: str | int, text: str) -> dict:
        ch = await self._get_channel(channel_id)
        msg = await ch.send(text)
        return {"ok": True, "message_id": msg.id}

    async def send_file(self, channel_id: str | int, file_path: str, caption: str = "") -> dict:
        import discord
        ch = await self._get_channel(channel_id)
        file = discord.File(file_path)
        msg = await ch.send(content=caption or None, file=file)
        return {"ok": True, "message_id": msg.id}


# ── WhatsApp Sender ──

class WhatsAppSender:
    """Send messages/files via Green API. Shared by channel + MCP."""

    def __init__(self, instance_id: str, api_token: str):
        self.instance_id = instance_id
        self.api_token = api_token
        self._api = None

    def _get_api(self):
        if self._api is None:
            from whatsapp_api_client_python import API as GreenAPI
            self._api = GreenAPI.GreenApi(self.instance_id, self.api_token)
        return self._api

    @staticmethod
    def normalize_chat_id(phone: str) -> str:
        if "@" in phone:
            return phone
        return f"{phone}@c.us"

    async def send_message(self, phone: str, text: str) -> dict:
        api = self._get_api()
        chat_id = self.normalize_chat_id(phone)
        result = await asyncio.to_thread(api.sending.sendMessage, chat_id, text)
        return {"ok": True}

    async def send_file(self, phone: str, file_path: str, caption: str = "") -> dict:
        api = self._get_api()
        chat_id = self.normalize_chat_id(phone)
        fname = Path(file_path).name
        await asyncio.to_thread(api.sending.sendFileByUpload, chat_id, file_path, fname, caption)
        return {"ok": True}
