"""Telegram channel using python-telegram-bot. Supports text, images, files, voice, live status."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from openagent.channels.base import BaseChannel, Attachment, parse_response_markers

if TYPE_CHECKING:
    from openagent.agent import Agent

logger = logging.getLogger(__name__)


async def _transcribe_voice(file_path: str) -> str | None:
    """Transcribe a voice .ogg file using OpenAI Whisper API.

    Returns the transcribed text, or None if transcription is unavailable.
    Requires OPENAI_API_KEY environment variable to be set.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.debug("OPENAI_API_KEY not set — skipping voice transcription")
        return None

    try:
        import httpx
    except ImportError:
        logger.debug("httpx not available — skipping voice transcription")
        return None

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(file_path, "rb") as f:
                resp = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (Path(file_path).name, f, "audio/ogg")},
                    data={"model": "whisper-1"},
                )
            resp.raise_for_status()
            return resp.json().get("text", "").strip() or None
    except Exception as e:
        logger.warning(f"Voice transcription failed: {e}")
        return None


class TelegramChannel(BaseChannel):
    """Telegram bot channel with full media support and live status updates.

    When the agent is processing, sends a status message that updates in real-time:
      "⏳ Thinking..." → "🔧 Using shell_exec..." → "⏳ Thinking..." → final response
    """

    def __init__(self, agent: Agent, token: str, allowed_users: list[str] | None = None):
        super().__init__(agent)
        self.token = token
        self.allowed_users = set(allowed_users) if allowed_users else None
        self._app = None

    async def _handle_message(self, update, context) -> None:
        if not update.message:
            return

        msg = update.message
        user_id = str(msg.from_user.id)

        # Whitelist check
        if self.allowed_users and user_id not in self.allowed_users:
            await msg.reply_text("Unauthorized. Contact the admin.")
            return
        session_id = self._user_session_id("telegram", user_id)
        text = msg.caption or msg.text or ""
        attachments: list[dict] = []

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
                transcription = await _transcribe_voice(path)
                if transcription:
                    # Use transcribed text as the message content
                    text = transcription if not text else f"{text}\n{transcription}"
                    logger.info(f"Voice transcribed ({msg.voice.duration}s): {transcription[:80]}...")
                else:
                    # No transcription available — fall back to attachment
                    text = "[Voice message received]" if not text else text
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

            # Send initial status message
            status_msg = await msg.reply_text("⏳ Thinking...")

            # Status callback: updates the status message in-place
            async def on_status(status: str) -> None:
                try:
                    await status_msg.edit_text(f"⏳ {status}")
                except Exception:
                    pass  # ignore edit failures (message unchanged, rate limit, etc.)

            response = await self.agent.run(
                message=text,
                user_id=user_id,
                session_id=session_id,
                attachments=attachments if attachments else None,
                on_status=on_status,
            )

            # Delete status message
            try:
                await status_msg.delete()
            except Exception:
                pass

            await self._send_response(msg, response)

        except Exception as e:
            logger.error(f"Telegram handler error: {e}")
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
                "Install it with: pip install openagent-framework[telegram]"
            )

        self._app = ApplicationBuilder().token(self.token).build()
        self._app.add_handler(CommandHandler("start", self._handle_start))
        self._app.add_handler(MessageHandler(
            filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO |
            filters.Document.ALL | filters.VIDEO,
            self._handle_message,
        ))

        logger.info(f"Starting Telegram bot for agent '{self.agent.name}'")
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()

        self._stop_event = asyncio.Event()
        await self._stop_event.wait()

    async def stop(self) -> None:
        if hasattr(self, '_stop_event'):
            self._stop_event.set()
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
