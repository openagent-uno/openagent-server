"""Discord bridge — translates Discord Bot API ↔ Gateway WS protocol.

Registers native Discord slash commands and handles messages via
the Gateway WebSocket protocol. Authorized users only.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from src.bridges.base import BaseBridge
from src.channels.base import (
    build_attachment_context,
    is_blocked_attachment,
    prepend_context_block,
)
from src.channels.voice import is_audio_file
from src.gateway.commands import BOT_COMMANDS, bridge_welcome_text

from src.core.logging import elog

logger = logging.getLogger(__name__)

DISCORD_MSG_LIMIT = 2000


def _build_picker_view(bridge, uid: str, picker: dict):
    """Build a Discord Select-menu ``View`` for a structured command picker
    (e.g. the /model chooser). Returns ``None`` when discord.ui is
    unavailable or there are no renderable options — the caller then falls
    back to the plain-text option list.

    Discord dispatches component interactions straight to the View, so the
    Select's own ``callback`` re-issues the command with the chosen value;
    no global interaction handler is needed.
    """
    try:
        import discord
    except Exception:
        return None
    command = str(picker.get("command") or "")
    raw = picker.get("options") or []
    if not command or not raw:
        return None
    options = []
    for opt in raw[:25]:  # Discord Select hard cap of 25 options
        value = str(opt.get("value") or "")
        if not value:
            continue
        label = str(opt.get("label") or value)[:100]
        desc = (str(opt.get("subtitle") or "")[:100]) or None
        options.append(discord.SelectOption(
            label=label, value=value[:100], description=desc,
            default=bool(opt.get("active")),
        ))
    if not options:
        return None

    class _PickerSelect(discord.ui.Select):
        def __init__(self) -> None:
            super().__init__(
                placeholder=str(picker.get("prompt") or "Choose…")[:150],
                min_values=1, max_values=1, options=options,
            )

        async def callback(self, interaction) -> None:  # noqa: ANN001
            value = self.values[0]
            result = await bridge.send_command(
                command, session_id=f"dc:{uid}", arg=value,
            )
            try:
                await interaction.response.edit_message(
                    content=result or "Done.", view=None,
                )
            except Exception:
                pass

    view = discord.ui.View(timeout=300)
    view.add_item(_PickerSelect())
    return view


class _DiscordTypingAnimator:
    """Keeps Discord's native "Bot is typing…" indicator lit for the whole
    turn — the in-chat equivalent of Telegram's ``ChatAction.TYPING``
    animator, so the agent never has to post a ``Thinking…`` placeholder.

    ``channel.typing()`` is an async context manager that re-triggers the
    indicator on Discord's ~10 s cadence while the block is open, so we
    just hold it open on a background task until ``stop()`` is called.
    """

    def __init__(self, channel) -> None:
        self._channel = channel
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        async def _loop() -> None:
            try:
                async with self._channel.typing():
                    await self._stop.wait()
            except Exception as e:  # noqa: BLE001
                # Missing permission / deleted channel / network blip —
                # the typing dot is best-effort; the reply still lands.
                logger.debug("discord typing indicator failed: %s", e)
        self._task = asyncio.create_task(_loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                pass
            self._task = None


class DiscordBridge(BaseBridge):
    name = "discord"
    message_limit = DISCORD_MSG_LIMIT

    def __init__(
        self,
        token: str,
        allowed_users: list[str],
        allowed_guilds: list[str] | None = None,
        listen_channels: list[str] | None = None,
        dm_only: bool = False,
        gateway_url: str = "ws://localhost:8765/ws",
        gateway_token: str | None = None,
        personality: str | None = None,
        live: bool = True,
    ):
        super().__init__(gateway_url, gateway_token, personality=personality, live=live)
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

        def _make_command_handler(command_name: str):
            async def _handler(
                interaction: discord.Interaction,
                arg: str | None = None,
            ) -> None:
                await self._handle_slash(interaction, command_name, arg=arg)

            _handler.__name__ = f"_cmd_{command_name.replace('-', '_')}"
            return _handler

        for command_name, description in BOT_COMMANDS:
            tree.command(
                name=command_name,
                description=description,
            )(_make_command_handler(command_name))

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
            # Operator hint: surface the effective receive-side config so
            # silent message drops are debuggable from events.jsonl alone.
            elog("bridge.discord.config",
                 allowed_users=len(self.allowed_users),
                 allowed_guilds=len(self.allowed_guilds),
                 listen_channels=len(self.listen_channels),
                 dm_only=self.dm_only)
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
            cid = str(getattr(message.channel, "id", "")) or None
            gid = str(message.guild.id) if getattr(message, "guild", None) else None
            if uid not in self.allowed_users:
                elog("bridge.discord.dropped", reason="not_allowed_user",
                     user_id=uid, channel_id=cid)
                return

            is_dm = isinstance(message.channel, discord.DMChannel)
            if not is_dm and self.dm_only:
                elog("bridge.discord.dropped", reason="dm_only_rejecting_guild",
                     user_id=uid, channel_id=cid, guild_id=gid)
                return
            if not is_dm and self.allowed_guilds and gid not in self.allowed_guilds:
                elog("bridge.discord.dropped", reason="guild_not_allowed",
                     user_id=uid, channel_id=cid, guild_id=gid)
                return
            if not is_dm:
                mentioned = client.user and client.user in message.mentions
                in_listen = cid in self.listen_channels
                # Authorized users bypass the @mention / listen_channels gate.
                # Symmetry with Telegram: once a user is in allowed_users they
                # can DM-style chat the bot in any channel without ceremony.
                # Safe because DiscordBridge REQUIRES non-empty allowed_users
                # at construction time (openagent/core/server.py:232-234), so
                # this bypass is always narrowly scoped to vetted accounts.
                is_allowed_user = bool(self.allowed_users) and uid in self.allowed_users
                if not mentioned and not in_listen and not is_allowed_user:
                    elog("bridge.discord.dropped", reason="no_mention_no_listen",
                         user_id=uid, channel_id=cid, guild_id=gid)
                    return

            elog("bridge.message", bridge="discord", user_id=uid)
            content = message.content or ""
            if client.user:
                content = content.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "").strip()

            # Process attachments
            blocked = []
            files_info = []
            voice_detected = False
            tmp = tempfile.mkdtemp(prefix="oa_dc_")
            try:
                for att in message.attachments:
                    if is_blocked_attachment(att.filename):
                        blocked.append(att.filename)
                        continue
                    ct = att.content_type or ""
                    is_voice = is_audio_file(att.filename, ct)
                    path = str(Path(tmp) / att.filename)
                    await att.save(path)

                    if is_voice:
                        voice_detected = True
                        t = await self.transcribe_with_fallback(path)
                        content = f"{content}\n{t}" if content else t
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

                await self.dispatch_turn(
                    message.channel, f"dc:{uid}", content,
                    voice_detected=voice_detected,
                )
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        await client.start(self.token)

    async def _on_gateway_lost(self) -> None:
        client = self._client
        if client is None:
            return
        try:
            await client.close()
        finally:
            self._client = None

    # ── Platform primitives (consumed by BaseBridge.dispatch_turn) ──

    async def post_status(self, channel, text: str):
        # Native "is writing" indicator instead of a "Thinking…"
        # message. Tool calls + answer spans surface as live step messages
        # (BaseBridge.dispatch_turn live mode); the typing dot covers the
        # gap before the first one lands.
        try:
            animator = _DiscordTypingAnimator(channel)
            await animator.start()
            return animator
        except Exception:
            return None

    async def update_status(self, animator, text: str) -> None:
        # No-op: the native typing indicator + live step messages replace
        # the old edited "…" placeholder.
        return None

    async def clear_status(self, animator) -> None:
        if animator is None:
            return
        try:
            await animator.stop()
        except Exception:
            pass

    async def send_text_chunk(self, channel, chunk: str) -> None:
        try:
            await channel.send(chunk)
        except Exception as e:  # noqa: BLE001
            logger.error("Discord text send error: %s", e)

    async def post_compaction_notice(self, channel):
        # In-place compaction (vision §2): Discord supports message edits,
        # so post one bubble and flip it in place rather than posting a
        # "Compacting…"/"Compacted" pair (see resolve_compaction_notice).
        try:
            return await channel.send("🗜 *Compacting conversation…*")
        except Exception:
            return None

    async def resolve_compaction_notice(self, channel, handle, text, *, ok):
        if handle is None:
            return await super().resolve_compaction_notice(channel, handle, text, ok=ok)
        try:
            await handle.edit(content=text)
        except Exception:
            pass

    async def send_attachment(self, channel, att) -> None:
        import discord as _dc
        p = Path(att.path)
        if not p.exists():
            return
        try:
            await channel.send(file=_dc.File(str(p), filename=att.filename))
        except Exception as e:  # noqa: BLE001
            logger.error("Discord file send error: %s", e)

    async def _handle_slash(self, interaction, cmd: str, *, arg: str | None = None) -> None:
        """Handle a Discord slash command via the Gateway."""
        uid = str(interaction.user.id)
        if uid not in self.allowed_users:
            await interaction.response.send_message("Unauthorized.", ephemeral=True)
            return
        await interaction.response.defer(ephemeral=True)
        # Scope /stop, /clear, /new, /reset, /compact, /model to THIS user's
        # session so a command from user A doesn't wipe user B's conversation.
        # Forward any inline argument (e.g. /model gpt-4o) via the arg field.
        result = await self.send_command_full(cmd, session_id=f"dc:{uid}", arg=arg or None)
        picker = result.get("picker") if isinstance(result, dict) else None
        if isinstance(picker, dict) and picker.get("options"):
            view = _build_picker_view(self, uid, picker)
            if view is not None:
                prompt = str(picker.get("prompt") or "Choose:")
                await interaction.followup.send(prompt, view=view, ephemeral=True)
                return
        await interaction.followup.send(result.get("text", ""), ephemeral=True)

    async def stop(self) -> None:
        self._should_stop = True
        if self._client:
            try:
                await self._client.close()
            finally:
                self._client = None
        await super().stop()
