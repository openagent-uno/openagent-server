"""WhatsApp bridge via Green API — translates WA messages ↔ Gateway WS."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from openagent.bridges.base import BaseBridge
from openagent.channels.base import parse_response_markers, is_blocked_attachment
from openagent.channels.formatting import markdown_to_whatsapp
from openagent.channels.voice import transcribe as transcribe_voice

logger = logging.getLogger(__name__)

VOICE_FALLBACK = "[Voice not transcribed. Ask the user to type it.]"


class WhatsAppBridge(BaseBridge):
    """WhatsApp (Green API) ↔ Gateway bridge."""

    name = "whatsapp"

    def __init__(
        self,
        instance_id: str,
        api_token: str,
        allowed_users: list[str] | None = None,
        gateway_url: str = "ws://localhost:8765/ws",
        gateway_token: str | None = None,
    ):
        super().__init__(gateway_url, gateway_token)
        self.instance_id = instance_id
        self.api_token = api_token
        self.allowed_users = set(str(u) for u in allowed_users) if allowed_users else None
        self._greenapi = None

    async def _run(self) -> None:
        try:
            from whatsapp_api_client_python import API as GreenAPI
        except ImportError:
            raise ImportError("Install: pip install openagent-framework[whatsapp]")

        self._greenapi = GreenAPI.GreenApi(self.instance_id, self.api_token)
        logger.info("WhatsApp bridge started")

        while not self._should_stop:
            try:
                response = await asyncio.to_thread(self._greenapi.receiving.receiveNotification)
                if not response or not response.data:
                    await asyncio.sleep(1)
                    continue

                receipt_id = response.data.get("receiptId")
                body = response.data.get("body", {})

                if body.get("typeWebhook") == "incomingMessageReceived":
                    await self._handle(body)

                if receipt_id:
                    await asyncio.to_thread(self._greenapi.receiving.deleteNotification, receipt_id)
            except Exception as e:
                logger.error("WhatsApp poll error: %s", e)
                await asyncio.sleep(5)

    async def _handle(self, body: dict) -> None:
        sender = body.get("senderData", {})
        chat_id = sender.get("chatId", "")
        user_id = chat_id.replace("@c.us", "").replace("@g.us", "")

        if self.allowed_users and user_id not in self.allowed_users:
            return

        msg_data = body.get("messageData", {})
        msg_type = msg_data.get("typeMessage", "")
        text = ""

        if msg_type == "textMessage":
            text = msg_data.get("textMessageData", {}).get("textMessage", "")
        elif msg_type == "extendedTextMessage":
            text = msg_data.get("extendedTextMessageData", {}).get("text", "")
        elif msg_type in ("audioMessage", "voiceMessage"):
            file_data = msg_data.get("fileMessageData", {})
            url = file_data.get("downloadUrl", "")
            if url:
                path = await self._download(url, "voice.ogg")
                if path:
                    t = await transcribe_voice(path)
                    text = t if t else VOICE_FALLBACK

        if not text:
            return

        session_id = f"wa:{user_id}"

        # Send "thinking" (WA can't edit messages)
        await self._send_text(chat_id, "⏳ Thinking...")
        response = await self.send_message(text, session_id)

        resp_text = response.get("text", "")
        clean, attachments = parse_response_markers(resp_text)

        for att in attachments:
            try:
                p = Path(att.path)
                if p.exists():
                    await asyncio.to_thread(
                        self._greenapi.sending.sendFileByUpload, chat_id, str(p), att.filename, ""
                    )
            except Exception as e:
                logger.error("WA attachment error: %s", e)

        if clean:
            await self._send_text(chat_id, markdown_to_whatsapp(clean))

    async def _send_text(self, chat_id: str, text: str) -> None:
        try:
            await asyncio.to_thread(self._greenapi.sending.sendMessage, chat_id, text)
        except Exception as e:
            logger.error("WA send error: %s", e)

    async def _download(self, url: str, filename: str) -> str | None:
        try:
            import urllib.request
            tmp = tempfile.mkdtemp(prefix="oa_wa_")
            path = str(Path(tmp) / filename)
            await asyncio.to_thread(urllib.request.urlretrieve, url, path)
            return path
        except Exception as e:
            logger.error("WA download error: %s", e)
            return None

    async def stop(self) -> None:
        self._should_stop = True
        self._greenapi = None
        await super().stop()
