"""Discord bridge — translates Discord Bot API ↔ Gateway WS protocol.

Security: allowed_users is mandatory. Unauthorized messages are ignored
in silence. Supports DM, mention, and listen_channels modes.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from openagent.bridges.base import BaseBridge
from openagent.channels.base import split_preserving_code_blocks, is_blocked_attachment, parse_response_markers
from openagent.channels.voice import transcribe as transcribe_voice

logger = logging.getLogger(__name__)

DISCORD_MSG_LIMIT = 2000
VOICE_FALLBACK = "[Voice message could not be transcribed. Ask the user to type it.]"


class DiscordBridge(BaseBridge):
    """Discord ↔ Gateway bridge."""

    name = "discord"

    def __init__(
        self,
        token: str,
        allowed_users: list[str],
        allowed_guilds: list[str] | None = None,
        listen_channels: list[str] | None = None,
        dm_only: bool = False,
        gateway_url: str = "ws://localhost:8765/ws",
        gateway_token: str | None = None,
    ):
        super().__init__(gateway_url, gateway_token)
        self.token = token
        self.allowed_users = set(str(u) for u in allowed_users)
        self.allowed_guilds = set(str(g) for g in (allowed_guilds or []))
        self.listen_channels = set(str(c) for c in (listen_channels or []))
        self.dm_only = dm_only
        self._client = None

    async def _run(self) -> None:
        try:
            import discord
        except ImportError:
            raise ImportError("discord.py required. Install: pip install openagent-framework[discord]")

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)
        self._client = client

        @client.event
        async def on_ready():
            logger.info("Discord bridge connected as %s", client.user)

        @client.event
        async def on_message(message):
            if message.author == client.user:
                return
            uid = str(message.author.id)
            if uid not in self.allowed_users:
                return

            is_dm = isinstance(message.channel, discord.DMChannel)
            if not is_dm and self.dm_only:
                return
            if not is_dm and self.allowed_guilds and str(message.guild.id) not in self.allowed_guilds:
                return
            if not is_dm:
                mentioned = client.user and client.user in message.mentions
                in_listen = str(message.channel.id) in self.listen_channels
                if not mentioned and not in_listen:
                    return

            content = message.content or ""
            if client.user:
                content = content.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "").strip()

            # Process attachments
            blocked = []
            files_info = []
            tmp = tempfile.mkdtemp(prefix="oa_dc_")
            for att in message.attachments:
                if is_blocked_attachment(att.filename):
                    blocked.append(att.filename)
                    continue
                ct = att.content_type or ""
                is_voice = ct.startswith("audio/") or att.filename.lower().endswith((".ogg",".mp3",".wav",".m4a"))
                path = str(Path(tmp) / att.filename)
                await att.save(path)

                if is_voice:
                    t = await transcribe_voice(path)
                    if t:
                        content = f"{content}\n{t}" if content else t
                    else:
                        content = f"{content}\n{VOICE_FALLBACK}" if content else VOICE_FALLBACK
                elif ct.startswith("image/"):
                    files_info.append(f"- image: {att.filename} — local path: {path}")
                else:
                    files_info.append(f"- file: {att.filename} — local path: {path}")

            if blocked:
                await message.channel.send(f"⚠️ Blocked: {', '.join(blocked)}")
            if files_info:
                header = "The user attached files:\n" + "\n".join(files_info) + "\nUse Read to inspect them."
                content = f"{header}\n\n{content}" if content else header

            if not content:
                return

            session_id = f"dc:{uid}"
            status_msg = await message.channel.send("⏳ Thinking...")

            async def on_status(status):
                try:
                    await status_msg.edit(content=f"⏳ {status}")
                except Exception:
                    pass

            response = await self.send_message(content, session_id, on_status=on_status)

            try:
                await status_msg.delete()
            except Exception:
                pass

            resp_text = response.get("text", "")
            clean, attachments = parse_response_markers(resp_text)

            if clean:
                for chunk in split_preserving_code_blocks(clean, DISCORD_MSG_LIMIT):
                    await message.channel.send(chunk)

        await client.start(self.token)

    async def stop(self) -> None:
        self._should_stop = True
        if self._client:
            try:
                await self._client.close()
            finally:
                self._client = None
        await super().stop()
