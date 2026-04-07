"""Discord channel using discord.py."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openagent.channels.base import BaseChannel

if TYPE_CHECKING:
    from openagent.agent import Agent

logger = logging.getLogger(__name__)


class DiscordChannel(BaseChannel):
    """Discord bot channel.

    Usage:
        channel = DiscordChannel(agent=agent, token="BOT_TOKEN")
        await channel.start()  # Blocks, runs the bot
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
            # Ignore own messages
            if message.author == client.user:
                return

            # Only respond to DMs or when mentioned
            is_dm = isinstance(message.channel, discord.DMChannel)
            is_mentioned = client.user in message.mentions if client.user else False

            if not is_dm and not is_mentioned:
                return

            # Strip mention from message
            content = message.content
            if is_mentioned and client.user:
                content = content.replace(f"<@{client.user.id}>", "").strip()

            if not content:
                return

            user_id = str(message.author.id)
            session_id = self._user_session_id("discord", user_id)

            try:
                async with message.channel.typing():
                    response = await self.agent.run(
                        message=content,
                        user_id=user_id,
                        session_id=session_id,
                    )

                # Discord has a 2000 char limit
                for i in range(0, len(response), 2000):
                    await message.channel.send(response[i:i+2000])
            except Exception as e:
                logger.error(f"Discord handler error: {e}")
                await message.channel.send("Sorry, something went wrong.")

        logger.info(f"Starting Discord bot for agent '{self.agent.name}'")
        await client.start(self.token)

    async def stop(self) -> None:
        if self._client:
            await self._client.close()
