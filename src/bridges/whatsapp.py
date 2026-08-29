"""WhatsApp bridge via Green API — translates WA messages ↔ Gateway WS."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import shutil
import socket
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlsplit

from src.bridges.base import BaseBridge
from src.channels.base import is_blocked_attachment
from src.channels.formatting import markdown_to_whatsapp
from src.core.logging import elog
from src.gateway.commands import BRIDGE_COMMANDS, bridge_welcome_text
from src.memory.artifacts import attachment_limit_bytes, safe_attachment_filename

logger = logging.getLogger(__name__)

# WhatsApp message size limit (Green API allows up to 65 536 chars; keep
# generous headroom for our own framing).
WHATSAPP_MSG_LIMIT = 4096
_WA_MEDIA_DOWNLOAD_TIMEOUT = 120.0
_WA_MAX_REDIRECTS = 5


class _WhatsAppDownloadError(Exception):
    """Media retrieval failed without exposing the remote URL to the user."""


class _WhatsAppAttachmentTooLarge(_WhatsAppDownloadError):
    def __init__(self, size_bytes: int) -> None:
        self.size_bytes = size_bytes
        super().__init__(f"attachment exceeds input limit ({size_bytes} bytes)")


def _is_public_address(value: str) -> bool:
    try:
        return ipaddress.ip_address(value).is_global
    except ValueError:
        return False


def _validate_whatsapp_media_url(url: str) -> str:
    """Reject credentials, unsafe schemes/ports and private IP literals."""
    parsed = urlsplit(str(url or ""))
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise _WhatsAppDownloadError("unsafe media URL")
    try:
        port = parsed.port
    except ValueError as exc:
        raise _WhatsAppDownloadError("invalid media URL port") from exc
    if port not in {None, 443}:
        raise _WhatsAppDownloadError("unsafe media URL port")
    # Literal addresses bypass DNS resolution in some HTTP stacks, so apply
    # the same global-address policy here as in the connector resolver.
    try:
        literal = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        literal = None
    if literal is not None and not literal.is_global:
        raise _WhatsAppDownloadError("private media address")
    return parsed.geturl()


class _PublicAddressResolver:
    """aiohttp resolver that never hands a private address to the socket.

    Resolving inside the connector closes the DNS-rebinding gap created by a
    separate "check then fetch" lookup: the exact validated addresses below
    are the ones aiohttp is allowed to connect to.
    """

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: int = socket.AF_UNSPEC,
    ) -> list[dict]:
        loop = asyncio.get_running_loop()
        infos = await loop.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
            family=family,
        )
        if not infos:
            raise OSError("media host did not resolve")
        resolved: list[dict] = []
        seen: set[tuple[str, int]] = set()
        for address_family, socktype, proto, _canonname, sockaddr in infos:
            address = str(sockaddr[0])
            resolved_port = int(sockaddr[1])
            if not _is_public_address(address):
                # Reject the hostname as a whole if *any* answer is private;
                # silently choosing a public sibling enables mixed-answer
                # rebinding attacks and makes resolver order security-relevant.
                raise OSError("media host resolved to a non-public address")
            key = (address, resolved_port)
            if key in seen:
                continue
            seen.add(key)
            resolved.append({
                "hostname": host,
                "host": address,
                "port": resolved_port,
                "family": address_family,
                "proto": proto,
                "flags": 0,
            })
        return resolved

    async def close(self) -> None:
        return None


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
        personality: str | None = None,
        live: bool = True,
    ):
        super().__init__(gateway_url, gateway_token, personality=personality, live=live)
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

        while not self._should_stop and not self._gateway_lost.is_set():
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
        temp_dirs: set[Path] = set()

        attachments: list[dict] = []

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
                    # Forward any inline argument (e.g. /model gpt-4o).
                    wa_parts = text.strip()[1:].split(None, 1)
                    wa_arg = wa_parts[1] if len(wa_parts) > 1 else None
                    if cmd == "compact":
                        # "Compacting conversation" → "Compacted conversation"
                        # as two short messages (WhatsApp has no edit API),
                        # matching the automatic path — the command has no turn
                        # collector to carry the compaction notice.
                        await self.run_compact_command(chat_id, f"wa:{user_id}")
                        return
                    result = await self.send_command_full(
                        cmd, session_id=f"wa:{user_id}", arg=wa_arg,
                    )
                    # WhatsApp (Green API) has no reliable interactive
                    # button/list primitive, so ``render_picker`` is not
                    # overridden — ``deliver_command_result`` degrades to
                    # the text option list, which now enumerates every
                    # configured model with its id for easy copy/switch.
                    await self.deliver_command_result(chat_id, result)
                return
        elif msg_type in ("audioMessage", "voiceMessage"):
            file_data = msg_data.get("fileMessageData", {})
            url = file_data.get("downloadUrl", "")
            mime = str(file_data.get("mimeType") or "audio/ogg")
            # Green API may deliver regular audio (MP3/M4A) through the
            # same envelope as push-to-talk notes.  Preserve its original
            # filename so provider routing and the channel's outbound
            # primitive do not incorrectly treat every payload as OGG.
            fallback = "voice.ogg" if msg_type == "voiceMessage" else "audio"
            fname = safe_attachment_filename(
                file_data.get("fileName") or fallback,
            )
            path = await self._download_or_notify(chat_id, url, fname)
            if path:
                temp_dirs.add(Path(path).parent)
                voice_detected = True
                text = await self.transcribe_with_fallback(path)
                attachments.append({
                    "type": "voice",
                    "kind": "voice",
                    "path": path,
                    "filename": fname,
                    "mime_type": mime,
                    "size_bytes": Path(path).stat().st_size,
                })
        elif msg_type == "imageMessage":
            file_data = msg_data.get("fileMessageData", {})
            text = file_data.get("caption", "")
            url = file_data.get("downloadUrl", "")
            fname = safe_attachment_filename(file_data.get("fileName", "image.jpg"))
            path = await self._download_or_notify(chat_id, url, fname)
            if path:
                temp_dirs.add(Path(path).parent)
                attachments.append({
                    "type": "image",
                    "kind": "image",
                    "path": path,
                    "filename": fname,
                    "mime_type": file_data.get("mimeType") or "image/jpeg",
                    "size_bytes": Path(path).stat().st_size,
                })
        elif msg_type == "documentMessage":
            file_data = msg_data.get("fileMessageData", {})
            text = file_data.get("caption", "")
            url = file_data.get("downloadUrl", "")
            fname = safe_attachment_filename(file_data.get("fileName", "document"))
            if not is_blocked_attachment(fname):
                path = await self._download_or_notify(chat_id, url, fname)
                if path:
                    temp_dirs.add(Path(path).parent)
                    mime = file_data.get("mimeType") or None
                    kind = "image" if str(mime or "").startswith("image/") else "file"
                    attachments.append({
                        "type": kind,
                        "kind": kind,
                        "path": path,
                        "filename": fname,
                        "mime_type": mime,
                        "size_bytes": Path(path).stat().st_size,
                    })
            else:
                await self._send_text(chat_id, f"⚠️ Blocked: {fname}")
        elif msg_type == "videoMessage":
            file_data = msg_data.get("fileMessageData", {})
            text = file_data.get("caption", "")
            url = file_data.get("downloadUrl", "")
            fname = safe_attachment_filename(file_data.get("fileName", "video.mp4"))
            path = await self._download_or_notify(chat_id, url, fname)
            if path:
                temp_dirs.add(Path(path).parent)
                attachments.append({
                    "type": "video",
                    "kind": "video",
                    "path": path,
                    "filename": fname,
                    "mime_type": file_data.get("mimeType") or "video/mp4",
                    "size_bytes": Path(path).stat().st_size,
                })

        try:
            if not text and not attachments:
                return

            sender_ref = str(
                sender.get("sender")
                or sender.get("senderId")
                or chat_id
            )
            author_id = sender_ref.replace("@c.us", "").replace("@g.us", "")
            await self.dispatch_turn(
                chat_id, f"wa:{user_id}", text, voice_detected=voice_detected,
                attachments=attachments,
                author={
                    "kind": "human",
                    "handle": f"whatsapp:{author_id}",
                    "display": str(sender.get("senderName") or author_id),
                },
            )
        finally:
            for tmp_dir in temp_dirs:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Platform primitives (consumed by BaseBridge.dispatch_turn) ──
    #
    # The Green API exposes no typing/presence primitive, so there is no
    # native "is writing" flag to set — and we no longer post a
    # ``Thinking…`` placeholder. The live step messages (each tool call
    # + answer span, BaseBridge.dispatch_turn live mode) are the progress
    # affordance instead. ``post_status``/``update_status``/``clear_status``
    # are therefore no-ops (the latter two inherit the BaseBridge default).

    async def post_status(self, chat_id, text: str):
        return None

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

    async def _download_or_notify(
        self,
        chat_id: str,
        url: str,
        filename: str,
    ) -> str | None:
        try:
            path = await self._download(url, filename)
            if path:
                return path
            raise _WhatsAppDownloadError("empty download result")
        except _WhatsAppAttachmentTooLarge as exc:
            limit = attachment_limit_bytes(direction="input")
            elog(
                "bridge.whatsapp.attachment_rejected",
                level="warning",
                filename=filename,
                size_bytes=exc.size_bytes,
                limit_bytes=limit,
            )
            await self._send_text(
                chat_id,
                f"⚠️ File troppo grande: {filename} "
                f"({exc.size_bytes} byte; limite {limit} byte).",
            )
        except Exception as exc:  # noqa: BLE001
            elog(
                "bridge.whatsapp.attachment_download_failed",
                level="warning",
                filename=filename,
                error=str(exc) or type(exc).__name__,
            )
            await self._send_text(
                chat_id,
                f"⚠️ Impossibile scaricare {filename}. Riprova più tardi.",
            )
        return None

    async def _download(self, url: str, filename: str) -> str:
        filename = safe_attachment_filename(filename)
        tmp = tempfile.mkdtemp(prefix="oa_wa_")
        path = Path(tmp) / filename

        try:
            import aiohttp

            current_url = _validate_whatsapp_media_url(url)
            limit = attachment_limit_bytes(direction="input")
            timeout = aiohttp.ClientTimeout(total=_WA_MEDIA_DOWNLOAD_TIMEOUT)
            connector = aiohttp.TCPConnector(
                resolver=_PublicAddressResolver(),
                use_dns_cache=False,
            )
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": "OpenAgent/1"},
            ) as session:
                for redirect_count in range(_WA_MAX_REDIRECTS + 1):
                    async with session.get(
                        current_url,
                        allow_redirects=False,
                    ) as response:
                        if response.status in {301, 302, 303, 307, 308}:
                            if redirect_count >= _WA_MAX_REDIRECTS:
                                raise _WhatsAppDownloadError("too many redirects")
                            location = response.headers.get("Location")
                            if not location:
                                raise _WhatsAppDownloadError("redirect without location")
                            current_url = _validate_whatsapp_media_url(
                                urljoin(current_url, location)
                            )
                            continue
                        if response.status != 200:
                            raise _WhatsAppDownloadError(
                                f"media server returned HTTP {response.status}"
                            )
                        try:
                            declared_size = int(
                                response.headers.get("Content-Length") or 0
                            )
                        except ValueError:
                            declared_size = 0
                        if limit and declared_size > limit:
                            raise _WhatsAppAttachmentTooLarge(declared_size)
                        copied = 0
                        with path.open("wb") as out:
                            async for chunk in response.content.iter_chunked(1024 * 1024):
                                copied += len(chunk)
                                if limit and copied > limit:
                                    raise _WhatsAppAttachmentTooLarge(copied)
                                out.write(chunk)
                        break
                else:  # pragma: no cover - loop always exits by break/raise
                    raise _WhatsAppDownloadError("download did not complete")
            return str(path.resolve())
        except Exception:
            with contextlib.suppress(OSError):
                path.unlink()
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    async def stop(self) -> None:
        self._should_stop = True
        self._greenapi = None
        await super().stop()
