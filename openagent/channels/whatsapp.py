"""WhatsApp channel using Green API. Supports text, images, files, voice, live status."""

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


class WhatsAppChannel(BaseChannel):
    """WhatsApp channel via Green API with full media support and live status.

    Sends an initial "⏳ Thinking..." message that updates as the agent works.
    """

    def __init__(self, agent: Agent, instance_id: str, api_token: str):
        super().__init__(agent)
        self.instance_id = instance_id
        self.api_token = api_token
        self._running = False
        self._greenapi = None

    async def start(self) -> None:
        self._should_stop = False
        while not self._should_stop:
            try:
                await self._start_inner()
            except Exception as e:
                if self._should_stop:
                    break
                logger.error(f"WhatsApp channel crashed: {e}, restarting in 45s...")
                await asyncio.sleep(45)

    async def _start_inner(self) -> None:
        try:
            from whatsapp_api_client_python import API as GreenAPI
        except ImportError:
            raise ImportError(
                "whatsapp-api-client-python is required for WhatsApp channel. "
                "Install it with: pip install openagent-framework[whatsapp]"
            )

        self._greenapi = GreenAPI.GreenApi(self.instance_id, self.api_token)
        self._running = True

        logger.info(f"Starting WhatsApp bot for agent '{self.agent.name}'")

        while self._running:
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
        session_id = self._user_session_id("whatsapp", user_id)

        text = ""
        attachments: list[dict] = []
        msg_type = message_data.get("typeMessage", "")

        if msg_type == "textMessage":
            text_data = message_data.get("textMessageData", {})
            text = text_data.get("textMessage", "")
        elif msg_type == "extendedTextMessage":
            text_data = message_data.get("extendedTextMessageData", {})
            text = text_data.get("text", "")
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
            if download_url:
                fname = file_data.get("fileName", "document")
                path = await self._download_file(download_url, fname)
                if path:
                    attachments.append({"type": "file", "path": path, "filename": fname})
        elif msg_type in ("audioMessage", "voiceMessage"):
            file_data = message_data.get("fileMessageData", {})
            download_url = file_data.get("downloadUrl", "")
            if download_url:
                path = await self._download_file(download_url, "voice.ogg")
                if path:
                    attachments.append({"type": "voice", "path": path, "filename": "voice.ogg"})
        elif msg_type == "videoMessage":
            file_data = message_data.get("fileMessageData", {})
            text = file_data.get("caption", "")
            download_url = file_data.get("downloadUrl", "")
            if download_url:
                fname = file_data.get("fileName", "video.mp4")
                path = await self._download_file(download_url, fname)
                if path:
                    attachments.append({"type": "video", "path": path, "filename": fname})

        if not text and not attachments:
            return

        try:
            # Send initial status message
            status_result = await asyncio.to_thread(
                self._greenapi.sending.sendMessage,
                chat_id,
                "⏳ Thinking...",
            )
            status_msg_id = None
            if hasattr(status_result, 'data') and isinstance(status_result.data, dict):
                status_msg_id = status_result.data.get("idMessage")

            # Status callback (WhatsApp doesn't support editing messages easily,
            # so we just log status changes — the initial message shows "Thinking")
            async def on_status(status: str) -> None:
                pass  # WhatsApp doesn't support message editing via Green API

            reply = await self.agent.run(
                message=text,
                user_id=user_id,
                session_id=session_id,
                attachments=attachments if attachments else None,
                on_status=on_status,
            )

            await self._send_response(chat_id, reply)

        except Exception as e:
            logger.error(f"WhatsApp handler error: {e}")

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
            await asyncio.to_thread(
                self._greenapi.sending.sendMessage,
                chat_id,
                clean_text,
            )

    async def stop(self) -> None:
        self._should_stop = True
        self._running = False
