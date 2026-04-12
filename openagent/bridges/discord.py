"""Discord bridge — translates Discord Bot API ↔ Gateway WS protocol.

Registers native Discord slash commands and handles messages via
the Gateway WebSocket protocol. Authorized users only.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from openagent.bridges.base import BaseBridge
from openagent.channels.base import split_preserving_code_blocks, is_blocked_attachment, parse_response_markers
from openagent.channels.voice import transcribe as transcribe_voice

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

        @tree.command(name="new", description="Start a new conversation (fresh context)")
        async def _cmd_new(interaction: discord.Interaction):
            await self._handle_slash(interaction, "new")

        @tree.command(name="stop", description="Cancel the current operation")
        async def _cmd_stop(interaction: discord.Interaction):
            await self._handle_slash(interaction, "stop")

        @tree.command(name="status", description="Show agent status and queue")
        async def _cmd_status(interaction: discord.Interaction):
            await self._handle_slash(interaction, "status")

        @tree.command(name="clear", description="Clear the message queue")
        async def _cmd_clear(interaction: discord.Interaction):
            await self._handle_slash(interaction, "clear")

        @tree.command(name="update", description="Check for updates and install")
        async def _cmd_update(interaction: discord.Interaction):
            await self._handle_slash(interaction, "update")

        @tree.command(name="restart", description="Restart OpenAgent")
        async def _cmd_restart(interaction: discord.Interaction):
            await self._handle_slash(interaction, "restart")

        @tree.command(name="help", description="Show available commands")
        async def _cmd_help(interaction: discord.Interaction):
            await self._handle_slash(interaction, "help")

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
                is_voice = ct.startswith("audio/") or att.filename.lower().endswith((".ogg", ".mp3", ".wav", ".m4a"))
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
