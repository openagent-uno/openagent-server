"""Telegram channel using python-telegram-bot. Supports text, images, files, voice."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from openagent.channels.base import BaseChannel, Attachment, parse_response_markers

if TYPE_CHECKING:
    from openagent.agent import Agent

logger = logging.getLogger(__name__)


class TelegramChannel(BaseChannel):
    """Telegram bot channel with full media support.

    Usage:
        channel = TelegramChannel(agent=agent, token="BOT_TOKEN")
        await channel.start()
    """

    def __init__(self, agent: Agent, token: str):
        super().__init__(agent)
        self.token = token
        self._app = None

    async def _handle_message(self, update, context) -> None:
        """Handle any incoming message (text, photo, voice, file, video)."""
        if not update.message:
            return

        msg = update.message
        user_id = str(msg.from_user.id)
        session_id = self._user_session_id("telegram", user_id)
        text = msg.caption or msg.text or ""
        attachments: list[dict] = []

        # Download media to temp dir
        tmp_dir = tempfile.mkdtemp(prefix="openagent_tg_")

        try:
            if msg.photo:
                photo = msg.photo[-1]  # highest resolution
                file = await photo.get_file()
                path = str(Path(tmp_dir) / f"photo_{photo.file_unique_id}.jpg")
                await file.download_to_drive(path)
                attachments.append({"type": "image", "path": path, "filename": Path(path).name})

            if msg.voice:
                file = await msg.voice.get_file()
                path = str(Path(tmp_dir) / f"voice_{msg.voice.file_unique_id}.ogg")
                await file.download_to_drive(path)
                attachments.append({"type": "voice", "path": path, "filename": Path(path).name})

            if msg.audio:
                file = await msg.audio.get_file()
                fname = msg.audio.file_name or f"audio_{msg.audio.file_unique_id}"
                path = str(Path(tmp_dir) / fname)
                await file.download_to_drive(path)
                attachments.append({"type": "file", "path": path, "filename": fname})

            if msg.document:
                file = await msg.document.get_file()
                fname = msg.document.file_name or f"doc_{msg.document.file_unique_id}"
                path = str(Path(tmp_dir) / fname)
                await file.download_to_drive(path)
                attachments.append({"type": "file", "path": path, "filename": fname})

            if msg.video:
                file = await msg.video.get_file()
                fname = msg.video.file_name or f"video_{msg.video.file_unique_id}.mp4"
                path = str(Path(tmp_dir) / fname)
                await file.download_to_drive(path)
                attachments.append({"type": "video", "path": path, "filename": fname})

            if not text and not attachments:
                return

            response = await self.agent.run(
                message=text,
                user_id=user_id,
                session_id=session_id,
                attachments=attachments if attachments else None,
            )

            await self._send_response(msg, response)

        except Exception as e:
            logger.error(f"Telegram handler error: {e}")
            await msg.reply_text("Sorry, something went wrong.")

    async def _send_response(self, msg, response: str) -> None:
        """Send agent response, handling file markers."""
        clean_text, attachments = parse_response_markers(response)

        # Send attachments
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

        # Send text (split on 4096 char limit)
        if clean_text:
            for i in range(0, len(clean_text), 4096):
                await msg.reply_text(clean_text[i:i + 4096])

    async def _handle_start(self, update, context) -> None:
        await update.message.reply_text(
            f"Hello! I'm {self.agent.name}. Send me a message, photo, voice, or file."
        )

    async def start(self) -> None:
        try:
            from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters
        except ImportError:
            raise ImportError(
                "python-telegram-bot is required for Telegram channel. "
                "Install it with: pip install openagent[telegram]"
            )

        self._app = ApplicationBuilder().token(self.token).build()
        self._app.add_handler(CommandHandler("start", self._handle_start))
        # Handle all content types
        self._app.add_handler(MessageHandler(
            filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO |
            filters.Document.ALL | filters.VIDEO,
            self._handle_message,
        ))

        logger.info(f"Starting Telegram bot for agent '{self.agent.name}'")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

        # Keep alive until stopped
        self._stop_event = asyncio.Event()
        await self._stop_event.wait()

    async def stop(self) -> None:
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
