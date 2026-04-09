"""Discord channel using discord.py.

Features:

- Security filter: only explicitly allowed Discord user IDs can talk to the
  bot, optionally restricted by guild id. Unauthorized messages are ignored
  in silence (no "unauthorized" reply, no ack) so the bot gives no signal
  to probers in public servers.
- Multiple server modes:
    • DM (default): always works for allowed users.
    • Mention in any allowed guild: ``@bot ...`` in any channel.
    • Dedicated channels: every message in a listed channel is processed,
      no mention required (Slack-style).
    • ``dm_only: true``: ignores any guild message, DMs only.
- Per-user FIFO message queue via :class:`UserQueueManager` — concurrent
  messages from the same user never race two agent runs, they serialize.
- Slash-command tree: ``/new /stop /status /queue /help /usage`` registered
  as native Discord commands (synced on ready), plus the same commands
  parsed from plain text for convenience.
- Live status message during processing, with an inline **⏹ Stop** button
  that cancels the in-flight task (only the author of the message can use
  it).
- Code-block-aware message splitting — long responses never leave a
  dangling ```fence mid-chunk.
- Executable attachment blocking (``.exe``, ``.bat``, ``.ps1``, …).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from openagent.channels.base import (
    BaseChannel,
    is_blocked_attachment,
    parse_response_markers,
    split_preserving_code_blocks,
)
from openagent.channels.commands import CommandDispatcher
from openagent.channels.queue import UserQueueManager

if TYPE_CHECKING:
    from openagent.agent import Agent

logger = logging.getLogger(__name__)

DISCORD_MSG_LIMIT = 2000


class DiscordChannel(BaseChannel):
    """Discord bot channel with security whitelist, queue, and slash commands."""

    name = "discord"

    def __init__(
        self,
        agent: Agent,
        token: str,
        allowed_users: list[str] | None = None,
        allowed_guilds: list[str] | None = None,
        listen_channels: list[str] | None = None,
        dm_only: bool = False,
    ):
        super().__init__(agent)
        self.token = token
        self.allowed_users = {str(u) for u in (allowed_users or [])}
        self.allowed_guilds = {str(g) for g in (allowed_guilds or [])}
        self.listen_channels = {str(c) for c in (listen_channels or [])}
        self.dm_only = bool(dm_only)
        self._client = None
        self._queue = UserQueueManager(platform="discord", agent_name=agent.name)
        self._commands = CommandDispatcher(agent, self._queue)

    # ── security ───────────────────────────────────────────────────────

    def _is_authorized(self, message) -> bool:
        """Decide whether this author is allowed to talk to the bot at all."""
        import discord

        if not self.allowed_users:
            # No whitelist configured = deny everyone. Fail closed.
            return False

        user_id = str(message.author.id)
        if user_id not in self.allowed_users:
            return False

        is_dm = isinstance(message.channel, discord.DMChannel)
        if is_dm:
            return True
        if self.dm_only:
            return False
        if self.allowed_guilds:
            guild_id = str(message.guild.id) if message.guild else ""
            if guild_id not in self.allowed_guilds:
                return False
        return True

    def _should_respond(self, message, client) -> bool:
        """After authorization — is this message something we should answer?"""
        import discord

        if isinstance(message.channel, discord.DMChannel):
            return True
        # Guild message: either a mention or a listen_channel
        if client.user and client.user in message.mentions:
            return True
        if self.listen_channels and str(message.channel.id) in self.listen_channels:
            return True
        return False

    # ── main loop ──────────────────────────────────────────────────────

    async def _run(self) -> None:
        try:
            import discord
            from discord import app_commands
        except ImportError:
            raise ImportError(
                "discord.py is required for Discord channel. "
                "Install it with: pip install openagent-framework[discord]"
            )

        if not self.allowed_users:
            logger.error(
                "DiscordChannel refusing to start: no 'allowed_users' configured. "
                "Set at least one Discord user ID in openagent.yaml."
            )
            raise RuntimeError("DiscordChannel requires 'allowed_users' to be set")

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)
        tree = app_commands.CommandTree(client)
        self._client = client

        self._register_slash_commands(tree, discord, app_commands)

        @client.event
        async def on_ready():
            logger.info("Discord bot '%s' connected as %s", self.agent.name, client.user)
            try:
                if self.allowed_guilds:
                    for gid in self.allowed_guilds:
                        guild = discord.Object(id=int(gid))
                        tree.copy_global_to(guild=guild)
                        await tree.sync(guild=guild)
                    logger.info("Discord slash commands synced to %d guild(s)", len(self.allowed_guilds))
                else:
                    await tree.sync()
                    logger.info("Discord slash commands synced globally")
            except Exception as e:  # noqa: BLE001
                logger.warning("Discord slash command sync failed: %s", e)

        @client.event
        async def on_message(message: discord.Message):
            if message.author == client.user:
                return
            if not self._is_authorized(message):
                return
            if not self._should_respond(message, client):
                return

            await self._handle_message(message, client)

        logger.info("Starting Discord bot for agent '%s'", self.agent.name)
        await client.start(self.token)

    # ── message handling ──────────────────────────────────────────────

    async def _handle_message(self, message, client) -> None:
        import discord

        content = message.content or ""
        if client.user:
            content = content.replace(f"<@{client.user.id}>", "").strip()
            content = content.replace(f"<@!{client.user.id}>", "").strip()

        user_id = str(message.author.id)

        # Commands — handled synchronously, outside the queue, so they never
        # block behind a running agent run.
        if CommandDispatcher.is_command(content):
            result = await self._commands.dispatch(content, user_id)
            if result is not None:
                await self._send_plain(message.channel, result.text)
                return
            # Unknown slash-looking text → ignore, don't send to agent
            await self._send_plain(message.channel, "Comando sconosciuto. Usa /help.")
            return

        # Attachments
        attachments: list[dict] = []
        blocked: list[str] = []
        if message.attachments:
            tmp_dir = tempfile.mkdtemp(prefix="openagent_dc_")
            for att in message.attachments:
                if is_blocked_attachment(att.filename):
                    blocked.append(att.filename)
                    continue
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
                except Exception as e:  # noqa: BLE001
                    logger.error("Failed to download Discord attachment: %s", e)

        if blocked:
            await message.channel.send(
                f"⚠️ Attachment bloccati (estensione non permessa): {', '.join(blocked)}"
            )

        if not content and not attachments:
            return

        # Build the handler closure — runs inside the per-user worker.
        async def handler():
            await self._process_message(message, client, content, attachments, user_id)

        position = await self._queue.enqueue(user_id, handler)
        if position > 0:
            try:
                await message.add_reaction("🕒")
            except Exception:
                pass

    async def _process_message(
        self,
        message,
        client,
        content: str,
        attachments: list[dict],
        user_id: str,
    ) -> None:
        import discord

        try:
            try:
                await message.clear_reaction("🕒")
            except Exception:
                pass

            view = _StopView(self._queue, user_id)
            status_msg = await message.channel.send("⏳ Thinking...", view=view)

            async def on_status(status: str) -> None:
                try:
                    await status_msg.edit(content=f"⏳ {status}", view=view)
                except Exception:
                    pass

            try:
                async with message.channel.typing():
                    response = await self.agent.run(
                        message=content,
                        user_id=user_id,
                        session_id=self._queue.get_session_id(user_id),
                        attachments=attachments if attachments else None,
                        on_status=on_status,
                    )
            except Exception as e:  # noqa: BLE001
                # Covers cancellation (asyncio.CancelledError subclasses
                # BaseException in 3.8+, but Agent.run catches BaseException
                # already and returns "Error: ...") and any other fault.
                logger.error("Discord agent run failed: %s", e)
                response = f"Error: {e}"

            try:
                await status_msg.delete()
            except Exception:
                pass

            await self._send_response(message.channel, response)

        except Exception as e:  # noqa: BLE001
            logger.error("Discord handler error: %s", e)
            try:
                await message.channel.send(f"Error: {e}")
            except Exception:
                pass

    # ── sending ────────────────────────────────────────────────────────

    async def _send_response(self, channel, response: str) -> None:
        import discord

        clean_text, attachments = parse_response_markers(response)

        files: list[discord.File] = []
        for att in attachments:
            path = Path(att.path)
            if path.exists():
                files.append(discord.File(str(path), filename=att.filename))

        if files:
            # Discord allows up to 10 files per message.
            for i in range(0, len(files), 10):
                batch = files[i : i + 10]
                text_chunk = clean_text[:DISCORD_MSG_LIMIT] if i == 0 and clean_text else None
                await channel.send(content=text_chunk, files=batch)
                if i == 0 and clean_text:
                    clean_text = clean_text[DISCORD_MSG_LIMIT:]

        if clean_text:
            for chunk in split_preserving_code_blocks(clean_text, DISCORD_MSG_LIMIT):
                await channel.send(chunk)

    async def _send_plain(self, channel, text: str) -> None:
        if not text:
            return
        for chunk in split_preserving_code_blocks(text, DISCORD_MSG_LIMIT):
            await channel.send(chunk)

    # ── slash commands ────────────────────────────────────────────────

    def _register_slash_commands(self, tree, discord, app_commands) -> None:
        """Register /new /stop /status /queue /help /usage as slash commands."""

        dispatcher = self._commands

        async def _guard(interaction) -> bool:
            uid = str(interaction.user.id)
            if uid not in self.allowed_users:
                await interaction.response.send_message(
                    "Non sei autorizzato.", ephemeral=True
                )
                return False
            return True

        async def _run(interaction, cmd: str, arg: str = ""):
            if not await _guard(interaction):
                return
            await interaction.response.defer(ephemeral=True, thinking=False)
            result = await dispatcher.dispatch(
                f"/{cmd} {arg}".strip(), str(interaction.user.id)
            )
            text = result.text if result else "Comando sconosciuto."
            if len(text) > DISCORD_MSG_LIMIT:
                text = text[: DISCORD_MSG_LIMIT - 20] + "\n… (troncato)"
            await interaction.followup.send(text, ephemeral=True)

        @tree.command(name="new", description="Avvia una nuova sessione (contesto fresco)")
        async def _new(interaction):
            await _run(interaction, "new")

        @tree.command(name="stop", description="Ferma l'operazione in corso")
        async def _stop(interaction):
            await _run(interaction, "stop")

        @tree.command(name="status", description="Mostra lo stato dell'agent")
        async def _status(interaction):
            await _run(interaction, "status")

        @tree.command(name="queue", description="Mostra i messaggi in coda")
        async def _queue_cmd(interaction):
            await _run(interaction, "queue")

        @tree.command(name="clear", description="Svuota la coda messaggi")
        async def _clear(interaction):
            await _run(interaction, "queue", "clear")

        @tree.command(name="help", description="Lista dei comandi")
        async def _help(interaction):
            await _run(interaction, "help")

        @tree.command(name="usage", description="Mostra l'uso Claude Code (ccusage)")
        async def _usage(interaction):
            await _run(interaction, "usage")

    # ── shutdown ──────────────────────────────────────────────────────

    async def _shutdown(self) -> None:
        try:
            await self._queue.shutdown()
        except Exception:
            pass
        if self._client:
            try:
                await self._client.close()
            finally:
                self._client = None


class _StopView:
    """Discord UI View exposing a single ⏹ Stop button.

    Declared lazily so the module still imports when discord.py is absent.
    """

    def __new__(cls, queue: UserQueueManager, user_id: str):
        import discord
        from discord import ui

        class _Impl(ui.View):
            def __init__(self) -> None:
                super().__init__(timeout=None)

            @ui.button(label="Stop", style=discord.ButtonStyle.danger, emoji="⏹")
            async def _stop(self, interaction: discord.Interaction, button: ui.Button):
                if str(interaction.user.id) != user_id:
                    await interaction.response.send_message(
                        "Non puoi fermare l'operazione di un altro utente.",
                        ephemeral=True,
                    )
                    return
                stopped = queue.stop_current(user_id)
                msg = "⏹ Operazione cancellata." if stopped else "Nessuna operazione in corso."
                await interaction.response.send_message(msg, ephemeral=True)

        return _Impl()
