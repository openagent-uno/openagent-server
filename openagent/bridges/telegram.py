"""Telegram bridge — translates Telegram Bot API ↔ Gateway WS protocol."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from openagent.bridges.base import BaseBridge, format_tool_status
from openagent.channels.formatting import markdown_to_telegram_html
from openagent.channels.base import (
    build_attachment_context,
    is_blocked_attachment,
    parse_response_markers,
    prepend_context_block,
    split_preserving_code_blocks,
)
from openagent.channels.voice import transcribe as transcribe_voice
from openagent.gateway.commands import BOT_COMMANDS, BRIDGE_COMMANDS, bridge_welcome_text

from openagent.core.logging import elog

logger = logging.getLogger(__name__)

TELEGRAM_MSG_LIMIT = 4096
VOICE_FALLBACK = "[Voice message not transcribed. Ask the user to type it.]"

# Hard shutdown deadlines for the python-telegram-bot library calls. These
# exist because ``updater.stop()`` internally POSTs to Telegram's
# ``getUpdates`` via httpx; on a flaky network or during a launchd-initiated
# shutdown the request can hang. We refuse to let library cleanup block the
# entire process restart loop — if it takes more than a few seconds we log
# and move on so the agent can come back up cleanly.
_TG_UPDATER_STOP_TIMEOUT = 3.0
_TG_APP_STOP_TIMEOUT = 3.0
_TG_APP_SHUTDOWN_TIMEOUT = 3.0
# Very short timeout for the manual offset-advance POST — it's a one-shot
# confirmation; if Telegram is slow we'd rather skip than block shutdown.
_TG_OFFSET_FLUSH_TIMEOUT = 2.0


class TelegramBridge(BaseBridge):
    name = "telegram"

    def __init__(self, token: str, allowed_users: list[str] | None = None,
                 gateway_url: str = "ws://localhost:8765/ws", gateway_token: str | None = None):
        super().__init__(gateway_url, gateway_token)
        self.token = token
        self.allowed_users = set(allowed_users) if allowed_users else None
        self._app = None
        # Highest update_id we've seen from Telegram. Used during shutdown
        # to directly ACK the offset so a queued ``/restart`` cannot
        # replay on next boot (which is what caused the lyra-agent
        # mac-mini crash loop: shutdown hung inside
        # ``updater.stop()`` → ``_get_updates_cleanup``, so the offset
        # never advanced; launchd restarted us, and the same ``/restart``
        # Update came right back from ``getUpdates``).
        self._last_update_id: int = 0

    def _is_authorized(self, uid: str) -> bool:
        return self.allowed_users is None or uid in self.allowed_users

    async def _run(self) -> None:
        from telegram import BotCommand
        from telegram.ext import (
            ApplicationBuilder, CommandHandler, MessageHandler,
            CallbackQueryHandler, filters,
        )

        # concurrent_updates(True) is NOT optional: without it python-telegram-bot
        # dispatches Updates for the same chat strictly sequentially, so a
        # user whose message is blocked inside ``send_message`` (waiting on
        # a long agent turn — gradle, docker build, maestro suite, ...) can
        # NOT get a second command through. /stop, the stop-button callback,
        # or any text message just piles up behind the first handler's
        # ``await future`` and never reaches us. With concurrent_updates the
        # second Update gets its own task, hits _on_command, flows through
        # send_command, and ``sm.stop_current`` cancels the original task.
        self._app = (
            ApplicationBuilder()
            .token(self.token)
            .concurrent_updates(True)
            .build()
        )

        # Register commands
        self._app.add_handler(CommandHandler("start", self._on_start))
        for cmd in BRIDGE_COMMANDS:
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

    def _track_update_id(self, update) -> None:
        """Record the highest update_id we've processed so we can ACK it on stop."""
        try:
            uid = getattr(update, "update_id", None)
            if isinstance(uid, int) and uid > self._last_update_id:
                self._last_update_id = uid
        except Exception:
            pass

    async def flush_updates_offset(self) -> None:
        """ACK pending Telegram updates so they don't replay after restart.

        The python-telegram-bot ``Updater.stop()`` path is supposed to call
        ``_get_updates_cleanup`` to confirm our current offset, but during
        an in-flight shutdown (launchd stop, /restart from a queued Update)
        that POST can block or be cancelled — leaving the server-side offset
        unchanged. On next boot ``getUpdates`` re-delivers the same Update
        and we crash-loop.

        We defend against that by doing the ACK directly via a raw httpx
        POST with a short timeout BEFORE the library's own cleanup runs,
        and ignore all errors — best-effort is correct here. If this
        succeeds the Telegram server advances the offset and the queued
        command cannot fire again; if it fails, we're no worse off than
        before the fix.
        """
        next_offset = (self._last_update_id or 0) + 1
        url = f"https://api.telegram.org/bot{self.token}/getUpdates"
        try:
            import httpx
        except ImportError:
            logger.debug("httpx not available — skipping Telegram offset flush")
            return
        try:
            async with httpx.AsyncClient(timeout=_TG_OFFSET_FLUSH_TIMEOUT) as client:
                await client.post(
                    url,
                    json={"offset": next_offset, "timeout": 0, "limit": 1},
                )
            elog("bridge.telegram.offset_flush", offset=next_offset)
        except asyncio.CancelledError:
            # Never let cancellation from the flush derail the surrounding
            # shutdown — swallow and let the caller keep tearing down.
            elog(
                "bridge.telegram.offset_flush_cancelled",
                level="warning",
                offset=next_offset,
            )
        except Exception as e:  # noqa: BLE001 — best-effort
            elog(
                "bridge.telegram.offset_flush_error",
                level="warning",
                offset=next_offset,
                error=str(e),
            )

    async def stop(self) -> None:
        self._should_stop = True
        app = self._app
        # ACK pending updates FIRST so that a queued /restart or /stop can't
        # survive a library-side hang. Idempotent — if flush_updates_offset
        # was already called by the gateway restart path, this is a no-op
        # (at worst a second short POST).
        try:
            await self.flush_updates_offset()
        except asyncio.CancelledError:
            pass  # handled inside flush_updates_offset
        except Exception:
            pass
        if app is not None:
            # Library-side cleanup. Each of updater.stop / app.stop /
            # app.shutdown can internally POST to Telegram, and each has
            # been observed to hang during an outer cancellation (see the
            # events.jsonl trace from lyra-agent: cancel scope cancelled
            # from Gateway._handle_ws.handler while _get_updates_cleanup
            # was mid-POST). Bound each with a short timeout and swallow
            # CancelledError at this boundary — callers above us must be
            # able to finish the rest of shutdown regardless of what
            # python-telegram-bot decides to do.
            self._app = None
            for label, coro_factory, deadline in (
                ("updater.stop", lambda: app.updater.stop(), _TG_UPDATER_STOP_TIMEOUT),
                ("app.stop",     lambda: app.stop(),         _TG_APP_STOP_TIMEOUT),
                ("app.shutdown", lambda: app.shutdown(),     _TG_APP_SHUTDOWN_TIMEOUT),
            ):
                try:
                    await asyncio.wait_for(coro_factory(), timeout=deadline)
                except asyncio.TimeoutError:
                    elog(
                        "bridge.telegram.stop_timeout",
                        level="warning",
                        phase=label,
                        timeout=deadline,
                    )
                except asyncio.CancelledError:
                    # Do not propagate — a CancelledError from httpx/httpcore
                    # during library cleanup must not bleed out into the
                    # gateway shutdown path and derail MCP teardown, etc.
                    elog(
                        "bridge.telegram.stop_cancelled",
                        level="warning",
                        phase=label,
                    )
                except Exception as e:  # noqa: BLE001 — best-effort
                    elog(
                        "bridge.telegram.stop_error",
                        level="warning",
                        phase=label,
                        error=str(e),
                    )
        try:
            await super().stop()
        except asyncio.CancelledError:
            # Same reason — bridge-level cancellation must not escape.
            elog("bridge.telegram.super_stop_cancelled", level="warning")
        except Exception as e:  # noqa: BLE001
            elog("bridge.telegram.super_stop_error", level="warning", error=str(e))

    # ── Handlers ──

    async def _on_start(self, update, context):
        self._track_update_id(update)
        name = update.message.from_user.first_name or "there"
        await update.message.reply_text(bridge_welcome_text(name))

    async def _on_command(self, update, context, cmd):
        self._track_update_id(update)
        if not update.message:
            return
        user_id = str(update.message.from_user.id)
        if not self._is_authorized(user_id):
            return await update.message.reply_text("Unauthorized.")
        # Scope scope-sensitive commands (/stop, /clear, /new, /reset) to
        # just the user who issued them. Other users on the same bot keep
        # their own conversations.
        result = await self.send_command(cmd, session_id=f"tg:{user_id}")
        await self._reply_rich(update.message, result)

    async def _on_stop_cb(self, update, context):
        self._track_update_id(update)
        q = update.callback_query
        if not q or not q.data.startswith("stop:"):
            return
        uid = q.data.split(":", 1)[1]
        if str(q.from_user.id) != uid:
            return await q.answer("Not your operation.", show_alert=True)
        result = await self.send_command("stop", session_id=f"tg:{uid}")
        await q.answer(result, show_alert=False)

    async def _on_message(self, update, context):
        self._track_update_id(update)
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
            text = prepend_context_block(text, build_attachment_context(files_info))

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

        await self._send_response(msg, response.get("text", ""), response.get("model"))

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

    async def _send_response(self, msg, response, model: str | None = None):
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
        if clean or model:
            await self._reply_rich(msg, self.append_model_feedback(clean, model))
