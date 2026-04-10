"""Telegram bridge — translates Telegram Bot API ↔ Gateway WS protocol.

Thin adapter: receives Telegram messages, forwards to Gateway, sends
responses back to the user. All heavy lifting (queue, sessions, agent,
voice transcription) is done by the Gateway.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from openagent.bridges.base import BaseBridge
from openagent.channels.formatting import markdown_to_telegram_html
from openagent.channels.base import split_preserving_code_blocks, is_blocked_attachment, parse_response_markers
from openagent.channels.voice import transcribe as transcribe_voice

logger = logging.getLogger(__name__)

TELEGRAM_MSG_LIMIT = 4096

VOICE_FALLBACK_MSG = (
    "[The user sent a voice message but transcription is currently "
    "unavailable. Ask them to send it as text instead.]"
)


class TelegramBridge(BaseBridge):
    """Telegram ↔ Gateway bridge."""

    name = "telegram"

    def __init__(
        self,
        token: str,
        allowed_users: list[str] | None = None,
        gateway_url: str = "ws://localhost:8765/ws",
        gateway_token: str | None = None,
    ):
        super().__init__(gateway_url, gateway_token)
        self.token = token
        self.allowed_users = set(allowed_users) if allowed_users else None
        self._app = None

    def _is_authorized(self, user_id: str) -> bool:
        if self.allowed_users is None:
            return True
        return user_id in self.allowed_users

    async def _run(self) -> None:
        try:
            from telegram.ext import (
                ApplicationBuilder, CommandHandler, MessageHandler,
                CallbackQueryHandler, filters,
            )
        except ImportError:
            raise ImportError(
                "python-telegram-bot is required. "
                "Install with: pip install openagent-framework[telegram]"
            )

        self._app = ApplicationBuilder().token(self.token).build()

        # /start welcome
        self._app.add_handler(CommandHandler("start", self._handle_start))

        # Slash commands via Gateway
        for cmd in ("new", "reset", "stop", "status", "queue", "help", "usage"):
            self._app.add_handler(
                CommandHandler(cmd, lambda u, c, _c=cmd: self._handle_command(u, c, _c))
            )

        # Stop button callback
        self._app.add_handler(CallbackQueryHandler(self._handle_stop_callback, pattern=r"^stop:"))

        # Messages
        self._app.add_handler(MessageHandler(
            filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO |
            filters.Document.ALL | filters.VIDEO,
            self._handle_message,
        ))

        logger.info("Telegram bridge started")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

        # Block until stopped
        while not self._should_stop:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._should_stop = True
        if self._app:
            try:
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            finally:
                self._app = None
        await super().stop()

    # ── Handlers ──

    async def _handle_start(self, update, context):
        await update.message.reply_text(
            "Ciao! Mandami un messaggio, una foto, un vocale o un file.\n"
            "Usa /help per la lista dei comandi."
        )

    async def _handle_command(self, update, context, cmd: str):
        if not update.message:
            return
        user_id = str(update.message.from_user.id)
        if not self._is_authorized(user_id):
            await update.message.reply_text("Unauthorized.")
            return
        result = await self.send_command(cmd)
        await self._reply_rich(update.message, result)

    async def _handle_stop_callback(self, update, context):
        query = update.callback_query
        if not query or not query.data.startswith("stop:"):
            return
        target = query.data.split(":", 1)[1]
        if str(query.from_user.id) != target:
            await query.answer("Non puoi fermare l'operazione di un altro utente.", show_alert=True)
            return
        result = await self.send_command("stop")
        await query.answer(result, show_alert=False)

    async def _handle_message(self, update, context):
        if not update.message:
            return
        msg = update.message
        user_id = str(msg.from_user.id)

        if not self._is_authorized(user_id):
            await msg.reply_text("Unauthorized.")
            return

        text = msg.caption or msg.text or ""
        tmp_dir = tempfile.mkdtemp(prefix="oa_tg_")

        # Voice transcription
        if msg.voice:
            file = await msg.voice.get_file()
            path = str(Path(tmp_dir) / f"voice_{msg.voice.file_unique_id}.ogg")
            await file.download_to_drive(path)
            transcription = await transcribe_voice(path)
            if transcription:
                text = transcription if not text else f"{text}\n{transcription}"
            else:
                fallback = VOICE_FALLBACK_MSG
                text = fallback if not text else f"{text}\n{fallback}"

        if not text:
            return

        # Session ID = per-user
        session_id = f"tg:{user_id}"

        # Status message with stop button
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            kb = InlineKeyboardMarkup(
                [[InlineKeyboardButton("⏹ Stop", callback_data=f"stop:{user_id}")]]
            )
            status_msg = await msg.reply_text("⏳ Thinking...", reply_markup=kb)
        except Exception:
            status_msg = None
            kb = None

        async def on_status(status: str):
            if status_msg and kb:
                try:
                    await status_msg.edit_text(f"⏳ {status}", reply_markup=kb)
                except Exception:
                    pass

        # Send through Gateway
        response = await self.send_message(text, session_id, on_status=on_status)

        # Delete status message
        if status_msg:
            try:
                await status_msg.delete()
            except Exception:
                pass

        # Send response
        response_text = response.get("text", "")
        await self._send_response(msg, response_text)

    # ── Sending ──

    async def _reply_rich(self, msg, text: str) -> None:
        if not text:
            return
        for chunk in split_preserving_code_blocks(text, TELEGRAM_MSG_LIMIT):
            rendered = markdown_to_telegram_html(chunk)
            try:
                await msg.reply_text(rendered, parse_mode="HTML", disable_web_page_preview=True)
            except Exception:
                try:
                    await msg.reply_text(chunk)
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
                logger.error("Failed to send attachment %s: %s", att.filename, e)

        if clean_text:
            await self._reply_rich(msg, clean_text)
