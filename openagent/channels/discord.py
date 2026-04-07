"""Discord channel using discord.py. Supports text, images, files, attachments."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from openagent.channels.base import BaseChannel, Attachment, parse_response_markers

if TYPE_CHECKING:
    from openagent.agent import Agent

logger = logging.getLogger(__name__)


class DiscordChannel(BaseChannel):
    """Discord bot channel with full media support.

    Usage:
        channel = DiscordChannel(agent=agent, token="BOT_TOKEN")
        await channel.start()
    """

    def __init__(self, agent: Agent, token: str):
        super().__init__(agent)
        self.token = token
        self._client = None

    async def start(self) -> None:
        try:
            import discord
        except ImportError:
            raise ImportError(
                "discord.py is required for Discord channel. "
                "Install it with: pip install openagent[discord]"
            )

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)
        self._client = client

        @client.event
        async def on_ready():
            logger.info(f"Discord bot '{self.agent.name}' connected as {client.user}")

        @client.event
        async def on_message(message: discord.Message):
            if message.author == client.user:
                return

            is_dm = isinstance(message.channel, discord.DMChannel)
            is_mentioned = client.user in message.mentions if client.user else False
            if not is_dm and not is_mentioned:
                return

            content = message.content
            if is_mentioned and client.user:
                content = content.replace(f"<@{client.user.id}>", "").strip()

            user_id = str(message.author.id)
            session_id = self._user_session_id("discord", user_id)
            attachments: list[dict] = []

            # Download attachments
            if message.attachments:
                tmp_dir = tempfile.mkdtemp(prefix="openagent_dc_")
                for att in message.attachments:
                    try:
                        path = str(Path(tmp_dir) / att.filename)
                        await att.save(path)
                        ct = att.content_type or ""
                        if ct.startswith("image/"):
                            att_type = "image"
                        elif ct.startswith("audio/") or att.filename.endswith((".ogg", ".mp3", ".wav")):
                            att_type = "voice"
                        elif ct.startswith("video/"):
                            att_type = "video"
                        else:
                            att_type = "file"
                        attachments.append({"type": att_type, "path": path, "filename": att.filename})
                    except Exception as e:
                        logger.error(f"Failed to download Discord attachment: {e}")

            if not content and not attachments:
                return

            try:
                async with message.channel.typing():
                    response = await self.agent.run(
                        message=content,
                        user_id=user_id,
                        session_id=session_id,
                        attachments=attachments if attachments else None,
                    )

                await self._send_response(message.channel, response)
            except Exception as e:
                logger.error(f"Discord handler error: {e}")
                await message.channel.send("Sorry, something went wrong.")

        logger.info(f"Starting Discord bot for agent '{self.agent.name}'")
        await client.start(self.token)

    async def _send_response(self, channel, response: str) -> None:
        """Send agent response, handling file markers."""
        import discord

        clean_text, attachments = parse_response_markers(response)

        # Send attachments
        files = []
        for att in attachments:
            path = Path(att.path)
            if path.exists():
                files.append(discord.File(str(path), filename=att.filename))

        if files:
            # Discord allows up to 10 files per message
            for i in range(0, len(files), 10):
                batch = files[i:i + 10]
                text_chunk = clean_text[:2000] if i == 0 and clean_text else None
                await channel.send(content=text_chunk, files=batch)
                if i == 0 and clean_text:
                    clean_text = clean_text[2000:]

        # Send remaining text
        if clean_text:
            for i in range(0, len(clean_text), 2000):
                await channel.send(clean_text[i:i + 2000])

    async def stop(self) -> None:
        if self._client:
            await self._client.close()
