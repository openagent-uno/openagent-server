"""WhatsApp bridge via Green API — translates WA messages ↔ Gateway WS."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from openagent.bridges.base import BaseBridge
from openagent.channels.base import (
    build_attachment_context,
    is_blocked_attachment,
    prepend_context_block,
)
from openagent.channels.formatting import markdown_to_whatsapp
from openagent.gateway.commands import BRIDGE_COMMANDS, bridge_welcome_text

from openagent.core.logging import elog

logger = logging.getLogger(__name__)

# WhatsApp message size limit (Green API allows up to 65 536 chars; keep
# generous headroom for our own framing).
WHATSAPP_MSG_LIMIT = 4096
# Throttle: at most one progress message every N seconds, to avoid
# spamming the user (WhatsApp can't edit messages, so each status is a
# brand-new chat bubble).
WA_STATUS_THROTTLE_SECS = 8


class WhatsAppBridge(BaseBridge):
    """WhatsApp (Green API) ↔ Gateway bridge."""

    name = "whatsapp"
    message_limit = WHATSAPP_MSG_LIMIT

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
        # Per-chat throttle for status updates: WhatsApp can't edit, so
        # every progress ping is a brand-new bubble. We dedupe identical
        # lines and wait ``WA_STATUS_THROTTLE_SECS`` between distinct
        # updates so the user doesn't see a wall of "Using bash…" pings.
        self._status_throttle: dict[str, dict] = {}

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

        elog("bridge.message", bridge="whatsapp", user_id=user_id)
        msg_data = body.get("messageData", {})
        msg_type = msg_data.get("typeMessage", "")
        text = ""
        voice_detected = False

        files_info = []

        if msg_type == "textMessage":
            text = msg_data.get("textMessageData", {}).get("textMessage", "")
        elif msg_type == "extendedTextMessage":
            text = msg_data.get("extendedTextMessageData", {}).get("text", "")

        # Handle slash commands (text-only, no buttons on WhatsApp)
        if text.startswith("/"):
            cmd = text.strip()[1:].split()[0].lower()
            if cmd in (*BRIDGE_COMMANDS, "start"):
                if cmd == "start":
                    await self._send_text(chat_id, bridge_welcome_text())
                else:
                    # Scope scope-sensitive commands to this user's
                    # session so one WhatsApp contact's /clear doesn't
                    # wipe another's conversation on the same bot.
                    result = await self.send_command(cmd, session_id=f"wa:{user_id}")
                    await self._send_text(chat_id, result)
                return
        elif msg_type in ("audioMessage", "voiceMessage"):
            file_data = msg_data.get("fileMessageData", {})
            url = file_data.get("downloadUrl", "")
            if url:
                path = await self._download(url, "voice.ogg")
                if path:
                    voice_detected = True
                    text = await self.transcribe_with_fallback(path)
        elif msg_type == "imageMessage":
            file_data = msg_data.get("fileMessageData", {})
            text = file_data.get("caption", "")
            url = file_data.get("downloadUrl", "")
            if url:
                fname = file_data.get("fileName", "image.jpg")
                path = await self._download(url, fname)
                if path:
                    files_info.append(f"- image: {fname} — local path: {path}")
        elif msg_type == "documentMessage":
            file_data = msg_data.get("fileMessageData", {})
            text = file_data.get("caption", "")
            url = file_data.get("downloadUrl", "")
            fname = file_data.get("fileName", "document")
            if url and not is_blocked_attachment(fname):
                path = await self._download(url, fname)
                if path:
                    files_info.append(f"- file: {fname} — local path: {path}")
        elif msg_type == "videoMessage":
            file_data = msg_data.get("fileMessageData", {})
            text = file_data.get("caption", "")
            url = file_data.get("downloadUrl", "")
            fname = file_data.get("fileName", "video.mp4")
            if url:
                path = await self._download(url, fname)
                if path:
                    files_info.append(f"- video: {fname} — local path: {path}")

        if files_info:
            text = prepend_context_block(text, build_attachment_context(files_info))

        if not text:
            return

        await self.dispatch_turn(
            chat_id, f"wa:{user_id}", text, voice_detected=voice_detected,
        )

    # ── Platform primitives (consumed by BaseBridge.dispatch_turn) ──
    #
    # WhatsApp's HTTP API can't edit messages — every "update" would be
    # a new chat bubble. ``post_status`` posts the initial line and
    # seeds the per-chat throttle so ``update_status`` only re-posts
    # when the line CHANGES and at least ``WA_STATUS_THROTTLE_SECS``
    # have passed since the last ping. ``clear_status`` wipes the
    # throttle slot (the visible "Thinking…" stays in chat — there's
    # no API to remove it).

    async def post_status(self, chat_id, text: str):
        await self._send_text(chat_id, f"⏳ {text}")
        self._status_throttle[chat_id] = {
            "text": text, "ts": asyncio.get_event_loop().time(),
        }
        return chat_id

    async def update_status(self, chat_id, text: str) -> None:
        last = self._status_throttle.get(chat_id) or {"text": "", "ts": 0.0}
        now = asyncio.get_event_loop().time()
        if text == last["text"]:
            return
        if now - last["ts"] < WA_STATUS_THROTTLE_SECS:
            return
        self._status_throttle[chat_id] = {"text": text, "ts": now}
        try:
            await self._send_text(chat_id, f"⏳ {text}")
        except Exception:
            pass

    async def clear_status(self, chat_id) -> None:
        self._status_throttle.pop(chat_id, None)

    async def send_text_chunk(self, chat_id, chunk: str) -> None:
        await self._send_text(chat_id, markdown_to_whatsapp(chunk))

    async def send_attachment(self, chat_id, att) -> None:
        p = Path(att.path)
        if not p.exists():
            return
        try:
            await asyncio.to_thread(
                self._greenapi.sending.sendFileByUpload,
                chat_id, str(p), att.filename, "",
            )
        except Exception as e:  # noqa: BLE001
            logger.error("WA attachment error: %s", e)

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
