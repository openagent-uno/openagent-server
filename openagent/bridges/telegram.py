"""Telegram bridge — translates Telegram Bot API ↔ Gateway WS protocol."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from openagent.bridges.base import BaseBridge, format_tool_status
from openagent.channels.formatting import markdown_to_telegram_html
from openagent.channels.base import split_preserving_code_blocks, is_blocked_attachment, parse_response_markers
from openagent.channels.voice import transcribe as transcribe_voice

from openagent.core.logging import elog

logger = logging.getLogger(__name__)

TELEGRAM_MSG_LIMIT = 4096
VOICE_FALLBACK = "[Voice message not transcribed. Ask the user to type it.]"

# Bot menu commands — shown in Telegram's command picker
BOT_COMMANDS = [
    ("new", "Start a new conversation (fresh context)"),
    ("stop", "Cancel the current operation"),
    ("status", "Show agent status and queue"),
    ("clear", "Clear the message queue"),
    ("update", "Check for updates and install"),
    ("restart", "Restart OpenAgent"),
    ("help", "Show available commands"),
]


class TelegramBridge(BaseBridge):
    name = "telegram"

    def __init__(self, token: str, allowed_users: list[str] | None = None,
                 gateway_url: str = "ws://localhost:8765/ws", gateway_token: str | None = None):
        super().__init__(gateway_url, gateway_token)
        self.token = token
        self.allowed_users = set(allowed_users) if allowed_users else None
        self._app = None

    def _is_authorized(self, uid: str) -> bool:
        return self.allowed_users is None or uid in self.allowed_users

    async def _run(self) -> None:
        from telegram import BotCommand
        from telegram.ext import (
            ApplicationBuilder, CommandHandler, MessageHandler,
            CallbackQueryHandler, filters,
        )

        self._app = ApplicationBuilder().token(self.token).build()

        # Register commands
        self._app.add_handler(CommandHandler("start", self._on_start))
        for cmd in ("new", "reset", "stop", "status", "queue", "clear", "update", "restart", "help"):
            self._app.add_handler(CommandHandler(cmd, lambda u, c, _c=cmd: self._on_command(u, c, _c)))

        # Stop button callback
        self._app.add_handler(CallbackQueryHandler(self._on_stop_cb, pattern=r"^stop:"))

        # Messages (text, photo, voice, audio, documents, video)
        self._app.add_handler(MessageHandler(
            filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO |
            filters.Document.ALL | filters.VIDEO,
            self._on_message,
        ))

        logger.info("Telegram bridge started")
        await self._app.initialize()
        await self._app.start()

        # Set bot menu commands so they appear in Telegram's "/" picker
        try:
            await self._app.bot.set_my_commands([
                BotCommand(cmd, desc) for cmd, desc in BOT_COMMANDS
            ])
            logger.info("Telegram bot commands menu set (%d commands)", len(BOT_COMMANDS))
        except Exception as e:
            logger.warning("Failed to set bot commands: %s", e)

        await self._app.updater.start_polling()
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

    async def _on_start(self, update, context):
        name = update.message.from_user.first_name or "there"
        await update.message.reply_text(
            f"👋 Hi {name}! I'm your OpenAgent assistant.\n\n"
            "Send me a message, photo, voice note, or file and I'll help.\n\n"
            "Commands:\n"
            "/new — fresh conversation\n"
            "/stop — cancel current operation\n"
            "/status — agent status\n"
            "/clear — clear queue\n"
            "/help — all commands"
        )

    async def _on_command(self, update, context, cmd):
        if not update.message:
            return
        if not self._is_authorized(str(update.message.from_user.id)):
            return await update.message.reply_text("Unauthorized.")
        result = await self.send_command(cmd)
        await self._reply_rich(update.message, result)

    async def _on_stop_cb(self, update, context):
        q = update.callback_query
        if not q or not q.data.startswith("stop:"):
            return
        if str(q.from_user.id) != q.data.split(":", 1)[1]:
            return await q.answer("Not your operation.", show_alert=True)
        result = await self.send_command("stop")
        await q.answer(result, show_alert=False)

    async def _on_message(self, update, context):
        msg = update.message
        if not msg:
            return
        uid = str(msg.from_user.id)
        if not self._is_authorized(uid):
            return await msg.reply_text("Unauthorized.")

        elog("bridge.message", bridge="telegram", user_id=uid)
        text = msg.caption or msg.text or ""
        tmp = tempfile.mkdtemp(prefix="oa_tg_")
        files_info: list[str] = []

        # Photo
        if msg.photo:
            photo = msg.photo[-1]
            f = await photo.get_file()
            path = str(Path(tmp) / f"photo_{photo.file_unique_id}.jpg")
            await f.download_to_drive(path)
            files_info.append(f"- image: photo.jpg — local path: {path}")

        # Voice
        if msg.voice:
            f = await msg.voice.get_file()
            path = str(Path(tmp) / f"voice_{msg.voice.file_unique_id}.ogg")
            await f.download_to_drive(path)
            transcription = await transcribe_voice(path)
            if transcription:
                text = transcription if not text else f"{text}\n{transcription}"
            else:
                text = VOICE_FALLBACK if not text else f"{text}\n{VOICE_FALLBACK}"

        # Audio
        if msg.audio:
            fname = msg.audio.file_name or f"audio_{msg.audio.file_unique_id}"
            if not is_blocked_attachment(fname):
                f = await msg.audio.get_file()
                path = str(Path(tmp) / fname)
                await f.download_to_drive(path)
                files_info.append(f"- file: {fname} — local path: {path}")

        # Document
        if msg.document:
            fname = msg.document.file_name or f"doc_{msg.document.file_unique_id}"
            if is_blocked_attachment(fname):
                await msg.reply_text(f"⚠️ Blocked: {fname}")
            else:
                f = await msg.document.get_file()
                path = str(Path(tmp) / fname)
                await f.download_to_drive(path)
                files_info.append(f"- file: {fname} — local path: {path}")

        # Video
        if msg.video:
            fname = msg.video.file_name or f"video_{msg.video.file_unique_id}.mp4"
            if not is_blocked_attachment(fname):
                f = await msg.video.get_file()
                path = str(Path(tmp) / fname)
                await f.download_to_drive(path)
                files_info.append(f"- video: {fname} — local path: {path}")

        # Prepend file info
        if files_info:
            header = "The user attached files:\n" + "\n".join(files_info)
            header += "\nUse the Read tool with the local path to inspect each file."
            text = f"{header}\n\n{text}" if text else header

        if not text:
            return

        session_id = f"tg:{uid}"

        # Status message with stop button
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            kb = InlineKeyboardMarkup([[InlineKeyboardButton("⏹ Stop", callback_data=f"stop:{uid}")]])
            status_msg = await msg.reply_text("⏳ Thinking...", reply_markup=kb)
        except Exception:
            status_msg, kb = None, None

        async def on_status(s):
            if status_msg and kb:
                try:
                    await status_msg.edit_text(f"⏳ {format_tool_status(s)}", reply_markup=kb)
                except Exception:
                    pass

        response = await self.send_message(text, session_id, on_status=on_status)

        if status_msg:
            try:
                await status_msg.delete()
            except Exception:
                pass

        await self._send_response(msg, response.get("text", ""))

    # ── Sending ──

    async def _reply_rich(self, msg, text):
        for chunk in split_preserving_code_blocks(text, TELEGRAM_MSG_LIMIT):
            rendered = markdown_to_telegram_html(chunk)
            try:
                await msg.reply_text(rendered, parse_mode="HTML", disable_web_page_preview=True)
            except Exception:
                try:
                    await msg.reply_text(chunk)
                except Exception:
                    pass

    async def _send_response(self, msg, response):
        clean, attachments = parse_response_markers(response)
        for att in attachments:
            try:
                p = Path(att.path)
                if not p.exists():
                    continue
                with open(p, "rb") as f:
                    if att.type == "image":
                        await msg.reply_photo(photo=f)
                    elif att.type == "voice":
                        await msg.reply_voice(voice=f)
                    elif att.type == "video":
                        await msg.reply_video(video=f)
                    else:
                        await msg.reply_document(document=f, filename=att.filename)
            except Exception as e:
                logger.error("Attachment send error: %s", e)
        if clean:
            await self._reply_rich(msg, clean)
