"""Discord bridge — translates Discord Bot API ↔ Gateway WS protocol.

Registers native Discord slash commands and handles messages via
the Gateway WebSocket protocol. Authorized users only.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from openagent.bridges.base import BaseBridge, format_tool_status
from openagent.channels.base import (
    build_attachment_context,
    is_blocked_attachment,
    parse_response_markers,
    prepend_context_block,
    split_preserving_code_blocks,
)
from openagent.channels.voice import is_audio_file, transcribe as transcribe_voice
from openagent.gateway.commands import BOT_COMMANDS, bridge_welcome_text

from openagent.core.logging import elog

logger = logging.getLogger(__name__)

DISCORD_MSG_LIMIT = 2000
VOICE_FALLBACK = "[Voice message could not be transcribed. Ask the user to type it.]"


class DiscordBridge(BaseBridge):
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
            from discord import app_commands
        except ImportError:
            raise ImportError("discord.py required. Install: pip install openagent-framework[discord]")

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)
        tree = app_commands.CommandTree(client)
        self._client = client

        # ── Register slash commands ──

        for command_name, description in BOT_COMMANDS:
            async def _handler(interaction: discord.Interaction, _command_name=command_name):
                await self._handle_slash(interaction, _command_name)

            _handler.__name__ = f"_cmd_{command_name.replace('-', '_')}"
            tree.command(name=command_name, description=description)(_handler)

        # Welcome message — symmetric with Telegram /start.  Not part of
        # BOT_COMMANDS because the gateway has no /start command; this is
        # a bridge-local convenience.
        async def _start_handler(interaction: discord.Interaction):
            uid = str(interaction.user.id)
            if uid not in self.allowed_users:
                await interaction.response.send_message("Unauthorized.", ephemeral=True)
                return
            name = interaction.user.display_name or interaction.user.name
            await interaction.response.send_message(bridge_welcome_text(name), ephemeral=True)

        tree.command(name="start", description="Show welcome and command list")(_start_handler)

        # ── Events ──

        @client.event
        async def on_ready():
            logger.info("Discord bridge connected as %s", client.user)
            try:
                if self.allowed_guilds:
                    for gid in self.allowed_guilds:
                        guild = discord.Object(id=int(gid))
                        tree.copy_global_to(guild=guild)
                        await tree.sync(guild=guild)
                else:
                    await tree.sync()
                logger.info("Discord slash commands synced")
            except Exception as e:
                logger.warning("Slash command sync failed: %s", e)

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

            elog("bridge.message", bridge="discord", user_id=uid)
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
                is_voice = is_audio_file(att.filename, ct)
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
                content = prepend_context_block(content, build_attachment_context(files_info))

            if not content:
                return

            session_id = f"dc:{uid}"
            status_msg = await message.channel.send("⏳ Thinking...")

            async def on_status(status):
                try:
                    await status_msg.edit(content=f"⏳ {format_tool_status(status)}")
                except Exception:
                    pass

            response = await self.send_message(content, session_id, on_status=on_status)

            try:
                await status_msg.delete()
            except Exception:
                pass

            resp_text = response.get("text", "")
            clean, attachments = parse_response_markers(resp_text)
            clean = self.append_model_feedback(clean, response.get("model"))

            # Send file attachments
            import discord as _dc
            for att in attachments:
                p = Path(att.path)
                if p.exists():
                    try:
                        await message.channel.send(file=_dc.File(str(p), filename=att.filename))
                    except Exception as e:
                        logger.error("Discord file send error: %s", e)

            if clean:
                for chunk in split_preserving_code_blocks(clean, DISCORD_MSG_LIMIT):
                    await message.channel.send(chunk)

        await client.start(self.token)

    async def _handle_slash(self, interaction, cmd: str) -> None:
        """Handle a Discord slash command via the Gateway."""
        uid = str(interaction.user.id)
        if uid not in self.allowed_users:
            await interaction.response.send_message("Unauthorized.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        result = await self.send_command(cmd)
        await interaction.followup.send(result, ephemeral=True)

    async def stop(self) -> None:
        self._should_stop = True
        if self._client:
            try:
                await self._client.close()
            finally:
                self._client = None
        await super().stop()
