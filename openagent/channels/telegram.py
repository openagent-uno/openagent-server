"""Telegram channel using python-telegram-bot.

Features (kept in sync with the Discord channel):

- User whitelist via ``allowed_users`` (Telegram numeric user IDs).
- Per-user FIFO message queue with cancellation support.
- Slash commands: ``/new /reset /stop /status /queue /help /usage`` —
  registered both as Telegram CommandHandlers and understood as plain
  text by the dispatcher.
- Live status message during processing, with an inline **⏹ Stop** button
  that cancels the in-flight task.
- Code-block-aware message splitting (no dangling ``` fences).
- Executable attachment blocking.
- Voice transcription via OpenAI Whisper (optional, needs ``OPENAI_API_KEY``).
"""

from __future__ import annotations

import asyncio
import logging
import os
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
from openagent.channels.formatting import markdown_to_telegram_html
from openagent.channels.queue import UserQueueManager
from openagent.channels.voice import transcribe as transcribe_voice

if TYPE_CHECKING:
    from openagent.core.agent import Agent

logger = logging.getLogger(__name__)

TELEGRAM_MSG_LIMIT = 4096

# Explanation injected into the prompt when we can't transcribe a voice
# message. Clearer than the previous "[Voice message received]" placeholder
# that made models say "I'm a text-only agent".
VOICE_FALLBACK_MSG = (
    "[The user sent a voice message of {duration}s but transcription is "
    "currently unavailable on this deployment. Politely ask them to send "
    "it as text instead — don't claim you can't understand audio in "
    "general, just explain that voice transcription isn't configured.]"
)


class TelegramChannel(BaseChannel):
    """Telegram bot channel with queue, slash commands, and stop button."""

    name = "telegram"

    def __init__(self, agent: Agent, token: str, allowed_users: list[str] | None = None):
        super().__init__(agent)
        self.token = token
        self.allowed_users = set(allowed_users) if allowed_users else None
        self._app = None
        self._queue = UserQueueManager(platform="telegram", agent_name=agent.name)
        self._commands = CommandDispatcher(agent, self._queue)

    # ── authorization ──────────────────────────────────────────────────

    def _is_authorized(self, user_id: str) -> bool:
        if self.allowed_users is None:
            return True  # no whitelist configured
        return user_id in self.allowed_users

    # ── rich-text rendering ────────────────────────────────────────────

    async def _reply_rich(self, msg, text: str, **kwargs) -> None:
        """Send *text* as Telegram HTML, falling back to plain on parse errors.

        The agent's output is markdown — ``**bold**``, ``*italic*``,
        fenced code blocks, links. We convert each chunk to Telegram HTML
        and send with ``parse_mode="HTML"``. If Telegram rejects the
        markup (broken nesting, length, etc.) we retry once without parse
        mode so the user still gets the content, just unstyled.
        """
        if not text:
            return
        for chunk in split_preserving_code_blocks(text, TELEGRAM_MSG_LIMIT):
            rendered = markdown_to_telegram_html(chunk)
            try:
                await msg.reply_text(rendered, parse_mode="HTML",
                                     disable_web_page_preview=True, **kwargs)
            except Exception as e:
                logger.debug("HTML parse failed, falling back to plain: %s", e)
                try:
                    await msg.reply_text(chunk, **kwargs)
                except Exception as e2:
                    logger.error("Telegram reply failed: %s", e2)

    # ── command handlers ──────────────────────────────────────────────

    async def _handle_command(self, update, context, cmd: str) -> None:
        if not update.message:
            return
        user_id = str(update.message.from_user.id)
        if not self._is_authorized(user_id):
            await update.message.reply_text("Unauthorized. Contact the admin.")
            return
        arg = " ".join(context.args) if hasattr(context, "args") and context.args else ""
        result = await self._commands.dispatch(f"/{cmd} {arg}".strip(), user_id)
        if result is None:
            await update.message.reply_text("Comando sconosciuto. Usa /help.")
            return
        await self._reply_rich(update.message, result.text)

    async def _handle_start(self, update, context) -> None:
        await update.message.reply_text(
            f"Ciao! Sono {self.agent.name}. Mandami un messaggio, una foto, "
            f"un vocale o un file.\n\nUsa /help per la lista dei comandi."
        )

    async def _handle_stop_callback(self, update, context) -> None:
        query = update.callback_query
        if query is None:
            return
        data = query.data or ""
        if not data.startswith("stop:"):
            return
        target_user = data.split(":", 1)[1]
        clicker = str(query.from_user.id)
        if clicker != target_user:
            await query.answer("Non puoi fermare l'operazione di un altro utente.", show_alert=True)
            return
        stopped = self._queue.stop_current(target_user)
        await query.answer(
            "⏹ Operazione cancellata." if stopped else "Nessuna operazione in corso.",
            show_alert=False,
        )

    # ── message handling ──────────────────────────────────────────────

    async def _handle_message(self, update, context) -> None:
        if not update.message:
            return
        msg = update.message
        user_id = str(msg.from_user.id)

        if not self._is_authorized(user_id):
            await msg.reply_text("Unauthorized. Contact the admin.")
            return

        text = msg.caption or msg.text or ""

        # Plain-text slash commands (e.g. user typed /stop instead of using
        # the command menu). CommandHandlers already register /new, /stop,
        # etc. so this branch is mainly a fallback for variants like
        # "/queue clear" that Telegram's CommandHandler doesn't parse as a
        # single command invocation.
        if CommandDispatcher.is_command(text) and not msg.photo and not msg.document \
                and not msg.audio and not msg.voice and not msg.video:
            result = await self._commands.dispatch(text, user_id)
            if result is not None:
                await self._reply_rich(msg, result.text)
                return
            # Unknown slash command: CommandHandler already handles known
            # ones, fall through and ignore here.
            return

        attachments: list[dict] = []
        blocked: list[str] = []
        tmp_dir = tempfile.mkdtemp(prefix="openagent_tg_")

        try:
            if msg.photo:
                photo = msg.photo[-1]
                file = await photo.get_file()
                path = str(Path(tmp_dir) / f"photo_{photo.file_unique_id}.jpg")
                await file.download_to_drive(path)
                attachments.append({"type": "image", "path": path, "filename": Path(path).name})

            if msg.voice:
                file = await msg.voice.get_file()
                path = str(Path(tmp_dir) / f"voice_{msg.voice.file_unique_id}.ogg")
                await file.download_to_drive(path)
                transcription = await transcribe_voice(path)
                if transcription:
                    text = transcription if not text else f"{text}\n{transcription}"
                    logger.info(f"Voice transcribed ({msg.voice.duration}s): {transcription[:80]}...")
                else:
                    # No backend succeeded — tell the model clearly instead
                    # of leaving a confusing "[Voice message received]" tag.
                    fallback = VOICE_FALLBACK_MSG.format(duration=msg.voice.duration)
                    text = fallback if not text else f"{text}\n{fallback}"

            if msg.audio:
                fname = msg.audio.file_name or f"audio_{msg.audio.file_unique_id}"
                if is_blocked_attachment(fname):
                    blocked.append(fname)
                else:
                    file = await msg.audio.get_file()
                    path = str(Path(tmp_dir) / fname)
                    await file.download_to_drive(path)
                    attachments.append({"type": "file", "path": path, "filename": fname})

            if msg.document:
                fname = msg.document.file_name or f"doc_{msg.document.file_unique_id}"
                if is_blocked_attachment(fname):
                    blocked.append(fname)
                else:
                    file = await msg.document.get_file()
                    path = str(Path(tmp_dir) / fname)
                    await file.download_to_drive(path)
                    attachments.append({"type": "file", "path": path, "filename": fname})

            if msg.video:
                fname = msg.video.file_name or f"video_{msg.video.file_unique_id}.mp4"
                if is_blocked_attachment(fname):
                    blocked.append(fname)
                else:
                    file = await msg.video.get_file()
                    path = str(Path(tmp_dir) / fname)
                    await file.download_to_drive(path)
                    attachments.append({"type": "video", "path": path, "filename": fname})

            if blocked:
                await msg.reply_text(
                    "⚠️ Attachment bloccati (estensione non permessa): " + ", ".join(blocked)
                )

            if not text and not attachments:
                return

            async def handler():
                await self._process_message(msg, user_id, text, attachments)

            position = await self._queue.enqueue(user_id, handler)
            if position > 0:
                # Give a little acknowledgment that we've received the msg
                # and it's queued (the process handler will update with a
                # real status message when it actually runs).
                try:
                    await msg.reply_text(f"🕒 In coda (posizione {position}).")
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Telegram handler error: {e}")
            try:
                await msg.reply_text(f"Error: {e}")
            except Exception:
                pass

    async def _process_message(self, msg, user_id: str, text: str, attachments: list[dict]) -> None:
        """Runs inside the per-user queue worker."""
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        except ImportError:
            raise

        try:
            kb = InlineKeyboardMarkup(
                [[InlineKeyboardButton("⏹ Stop", callback_data=f"stop:{user_id}")]]
            )
            status_msg = await msg.reply_text("⏳ Thinking...", reply_markup=kb)

            async def on_status(status: str) -> None:
                try:
                    await status_msg.edit_text(f"⏳ {status}", reply_markup=kb)
                except Exception:
                    pass

            try:
                response = await self.agent.run(
                    message=text,
                    user_id=user_id,
                    session_id=self._queue.get_session_id(user_id),
                    attachments=attachments if attachments else None,
                    on_status=on_status,
                )
            except Exception as e:
                logger.error(f"Telegram agent run failed: {e}")
                response = f"Error: {e}"

            try:
                await status_msg.delete()
            except Exception:
                pass

            await self._send_response(msg, response)

        except Exception as e:
            logger.error(f"Telegram process error: {e}")
            try:
                await msg.reply_text(f"Error: {e}")
            except Exception:
                pass

    async def _send_response(self, msg, response: str) -> None:
        clean_text, attachments = parse_response_markers(response)

        for att in attachments:
            try:
                path = Path(att.path)
                if not path.exists():
                    continue
                if att.type == "image":
                    await msg.reply_photo(photo=open(path, "rb"), caption=att.caption)
                elif att.type == "voice":
                    await msg.reply_voice(voice=open(path, "rb"))
                elif att.type == "video":
                    await msg.reply_video(video=open(path, "rb"), caption=att.caption)
                else:
                    await msg.reply_document(document=open(path, "rb"), filename=att.filename)
            except Exception as e:
                logger.error(f"Failed to send attachment {att.filename}: {e}")

        if clean_text:
            await self._reply_rich(msg, clean_text)

    # ── lifecycle ──────────────────────────────────────────────────────

    async def _run(self) -> None:
        try:
            from telegram.ext import (
                ApplicationBuilder,
                CallbackQueryHandler,
                CommandHandler,
                MessageHandler,
                filters,
            )
        except ImportError:
            raise ImportError(
                "python-telegram-bot is required for Telegram channel. "
                "Install it with: pip install openagent-framework[telegram]"
            )

        self._app = ApplicationBuilder().token(self.token).build()
        self._app.add_handler(CommandHandler("start", self._handle_start))

        # Register slash commands as native CommandHandlers so they appear
        # in Telegram's command menu. Each handler delegates to the shared
        # dispatcher.
        for cmd in ("new", "reset", "stop", "status", "queue", "help", "usage"):
            self._app.add_handler(
                CommandHandler(cmd, lambda u, c, _c=cmd: self._handle_command(u, c, _c))
            )

        # Inline keyboard callback (stop button)
        self._app.add_handler(CallbackQueryHandler(self._handle_stop_callback, pattern=r"^stop:"))

        self._app.add_handler(
            MessageHandler(
                filters.TEXT
                | filters.PHOTO
                | filters.VOICE
                | filters.AUDIO
                | filters.Document.ALL
                | filters.VIDEO,
                self._handle_message,
            )
        )

        logger.info(f"Starting Telegram bot for agent '{self.agent.name}'")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

        # Block until stop() is called
        assert self._stop_event is not None
        await self._stop_event.wait()

    async def _shutdown(self) -> None:
        try:
            await self._queue.shutdown()
        except Exception:
            pass
        if self._app:
            try:
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            finally:
                self._app = None
