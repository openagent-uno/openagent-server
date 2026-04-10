"""WhatsApp channel via Green API.

Features (cross-channel parity with Telegram/Discord where the API allows):

- Optional user whitelist via ``allowed_users`` (phone numbers without
  the ``@c.us`` suffix).
- Per-user FIFO message queue with cancellation via ``/stop``.
- Slash commands: ``/new /reset /stop /status /queue /help /usage`` parsed
  from plain text (Green API has no native command menu, so commands are
  text-driven only).
- Code-block-aware message splitting.
- Executable attachment blocking.

WhatsApp/Green API limitations we can't work around:

- No message editing → no live status updates (only an initial "Thinking"
  acknowledgement).
- No inline buttons → no stop button, users use ``/stop`` text command.
"""

from __future__ import annotations

import asyncio
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
from openagent.channels.formatting import markdown_to_whatsapp
from openagent.channels.queue import UserQueueManager
from openagent.channels.voice import transcribe as transcribe_voice

if TYPE_CHECKING:
    from openagent.core.agent import Agent

logger = logging.getLogger(__name__)

WHATSAPP_MSG_LIMIT = 4000

VOICE_FALLBACK_MSG = (
    "[The user sent a voice/audio message but transcription is currently "
    "unavailable on this deployment. Politely ask them to send it as "
    "text instead — don't claim you can't understand audio in general, "
    "just explain that voice transcription isn't configured.]"
)


class WhatsAppChannel(BaseChannel):
    """WhatsApp channel via Green API with queue and slash commands."""

    name = "whatsapp"

    def __init__(
        self,
        agent: Agent,
        instance_id: str,
        api_token: str,
        allowed_users: list[str] | None = None,
    ):
        super().__init__(agent)
        self.instance_id = instance_id
        self.api_token = api_token
        self.allowed_users = {str(u) for u in allowed_users} if allowed_users else None
        self._greenapi = None
        self._queue = UserQueueManager(platform="whatsapp", agent_name=agent.name)
        self._commands = CommandDispatcher(agent, self._queue)

    # ── authorization ──────────────────────────────────────────────────

    def _is_authorized(self, user_id: str) -> bool:
        if self.allowed_users is None:
            return True
        return user_id in self.allowed_users

    # ── main loop ──────────────────────────────────────────────────────

    async def _run(self) -> None:
        try:
            from whatsapp_api_client_python import API as GreenAPI
        except ImportError:
            raise ImportError(
                "whatsapp-api-client-python is required for WhatsApp channel. "
                "Install it with: pip install openagent-framework[whatsapp]"
            )

        self._greenapi = GreenAPI.GreenApi(self.instance_id, self.api_token)

        logger.info(f"Starting WhatsApp bot for agent '{self.agent.name}'")

        while not self._should_stop:
            try:
                response = await asyncio.to_thread(
                    self._greenapi.receiving.receiveNotification
                )

                if not response or not response.data:
                    await asyncio.sleep(1)
                    continue

                receipt_id = response.data.get("receiptId")
                body = response.data.get("body", {})
                type_webhook = body.get("typeWebhook")

                if type_webhook == "incomingMessageReceived":
                    await self._handle_incoming(body)

                if receipt_id:
                    await asyncio.to_thread(
                        self._greenapi.receiving.deleteNotification,
                        receipt_id,
                    )

            except Exception as e:
                logger.error(f"WhatsApp polling error: {e}")
                await asyncio.sleep(5)

    async def _handle_incoming(self, body: dict) -> None:
        message_data = body.get("messageData", {})
        sender_data = body.get("senderData", {})
        chat_id = sender_data.get("chatId", "")
        user_id = chat_id.replace("@c.us", "").replace("@g.us", "")

        if not self._is_authorized(user_id):
            # Silence — no "unauthorized" reply to avoid probing.
            return

        text = ""
        attachments: list[dict] = []
        blocked: list[str] = []
        msg_type = message_data.get("typeMessage", "")

        if msg_type == "textMessage":
            text = message_data.get("textMessageData", {}).get("textMessage", "")
        elif msg_type == "extendedTextMessage":
            text = message_data.get("extendedTextMessageData", {}).get("text", "")
        elif msg_type == "imageMessage":
            file_data = message_data.get("fileMessageData", {})
            text = file_data.get("caption", "")
            download_url = file_data.get("downloadUrl", "")
            if download_url:
                path = await self._download_file(download_url, file_data.get("fileName", "image.jpg"))
                if path:
                    attachments.append({"type": "image", "path": path, "filename": Path(path).name})
        elif msg_type == "documentMessage":
            file_data = message_data.get("fileMessageData", {})
            text = file_data.get("caption", "")
            download_url = file_data.get("downloadUrl", "")
            fname = file_data.get("fileName", "document")
            if is_blocked_attachment(fname):
                blocked.append(fname)
            elif download_url:
                path = await self._download_file(download_url, fname)
                if path:
                    attachments.append({"type": "file", "path": path, "filename": fname})
        elif msg_type in ("audioMessage", "voiceMessage"):
            file_data = message_data.get("fileMessageData", {})
            download_url = file_data.get("downloadUrl", "")
            if download_url:
                path = await self._download_file(download_url, "voice.ogg")
                if path:
                    transcription = await transcribe_voice(path)
                    if transcription:
                        text = transcription if not text else f"{text}\n{transcription}"
                        logger.info("WhatsApp voice transcribed: %s...", transcription[:80])
                    else:
                        text = VOICE_FALLBACK_MSG if not text else f"{text}\n{VOICE_FALLBACK_MSG}"
        elif msg_type == "videoMessage":
            file_data = message_data.get("fileMessageData", {})
            text = file_data.get("caption", "")
            download_url = file_data.get("downloadUrl", "")
            fname = file_data.get("fileName", "video.mp4")
            if is_blocked_attachment(fname):
                blocked.append(fname)
            elif download_url:
                path = await self._download_file(download_url, fname)
                if path:
                    attachments.append({"type": "video", "path": path, "filename": fname})

        if blocked:
            await self._send_text(chat_id, "⚠️ Attachment bloccati: " + ", ".join(blocked))

        # Slash commands (only when the message is pure text — attachments
        # with commands in captions would be a rare edge case).
        if not attachments and CommandDispatcher.is_command(text):
            result = await self._commands.dispatch(text, user_id)
            if result is not None:
                await self._send_text(chat_id, result.text)
                return
            await self._send_text(chat_id, "Comando sconosciuto. Usa /help.")
            return

        if not text and not attachments:
            return

        async def handler():
            await self._process(chat_id, user_id, text, attachments)

        position = await self._queue.enqueue(user_id, handler)
        if position > 0:
            await self._send_text(chat_id, f"🕒 In coda (posizione {position}).")

    async def _process(self, chat_id: str, user_id: str, text: str, attachments: list[dict]) -> None:
        try:
            await self._send_text(chat_id, "⏳ Thinking...")

            # WhatsApp doesn't allow message editing via Green API — status
            # updates are logged locally but not pushed to the user.
            async def on_status(status: str) -> None:
                logger.debug("WhatsApp status (%s): %s", user_id, status)

            try:
                reply = await self.agent.run(
                    message=text,
                    user_id=user_id,
                    session_id=self._queue.get_session_id(user_id),
                    attachments=attachments if attachments else None,
                    on_status=on_status,
                )
            except Exception as e:
                logger.error(f"WhatsApp agent run failed: {e}")
                reply = f"Error: {e}"

            await self._send_response(chat_id, reply)
        except Exception as e:
            logger.error(f"WhatsApp handler error: {e}")

    # ── sending helpers ───────────────────────────────────────────────

    async def _send_text(self, chat_id: str, text: str) -> None:
        """Send text to the user, converting markdown to WhatsApp syntax."""
        if not text:
            return
        try:
            # Split first in markdown form so code fences stay balanced, then
            # convert each chunk to WhatsApp's single-marker syntax.
            for chunk in split_preserving_code_blocks(text, WHATSAPP_MSG_LIMIT):
                rendered = markdown_to_whatsapp(chunk)
                await asyncio.to_thread(
                    self._greenapi.sending.sendMessage,
                    chat_id,
                    rendered,
                )
        except Exception as e:
            logger.error(f"Failed to send WhatsApp text: {e}")

    async def _download_file(self, url: str, filename: str) -> str | None:
        try:
            import urllib.request
            tmp_dir = tempfile.mkdtemp(prefix="openagent_wa_")
            path = str(Path(tmp_dir) / filename)
            await asyncio.to_thread(urllib.request.urlretrieve, url, path)
            return path
        except Exception as e:
            logger.error(f"Failed to download WhatsApp file: {e}")
            return None

    async def _send_response(self, chat_id: str, response: str) -> None:
        clean_text, attachments = parse_response_markers(response)

        for att in attachments:
            try:
                path = Path(att.path)
                if not path.exists():
                    continue
                await asyncio.to_thread(
                    self._greenapi.sending.sendFileByUpload,
                    chat_id,
                    str(path),
                    att.filename,
                    att.caption or "",
                )
            except Exception as e:
                logger.error(f"Failed to send WhatsApp attachment {att.filename}: {e}")

        if clean_text:
            await self._send_text(chat_id, clean_text)

    async def _shutdown(self) -> None:
        try:
            await self._queue.shutdown()
        except Exception:
            pass
        self._greenapi = None
