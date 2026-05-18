"""Slack bridge — Bolt SDK socket-mode adapter.

Mirrors ``bridges/telegram.py`` in shape: subclass :class:`BaseBridge`,
implement platform-specific event polling, delegate orchestration to
``BaseBridge.dispatch_turn``. Socket mode is the right choice for a
self-hosted agent because the bot doesn't need a public HTTP endpoint
— it dials out to Slack's gateway and receives events over WebSocket.

Status indicator: Slack doesn't expose a typing primitive to bots
(unlike Telegram's chat.sendChatAction), so ``post_status`` posts a
real placeholder message and ``update_status`` / ``clear_status`` edit
or delete it. That's the same fallback Discord and WhatsApp use.

Auth needs TWO tokens:
  - ``bot_token`` (``xoxb-...``) — HTTP API auth, set as bot OAuth scope
  - ``app_token`` (``xapp-...``) — socket mode auth, generated from the
    app's Basic Information page after enabling socket mode

Wire both via ``channels.slack.{bot_token, app_token}`` in
``openagent.yaml`` (or env vars ``SLACK_BOT_TOKEN`` / ``SLACK_APP_TOKEN``).
``allowed_users`` accepts Slack user_ids (``U…``) — keep small for a
personal bot since DMs are the main use case.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from src.bridges.base import BaseBridge
from src.channels.formatting import markdown_to_telegram_html
from src.core.logging import elog

logger = logging.getLogger(__name__)

# Slack hard caps the text field of chat.postMessage at 40_000 chars
# but the practical formatting limit is much lower — long replies are
# split via BaseBridge.message_limit + send_text_chunk.
SLACK_MSG_LIMIT = 3500


class SlackBridge(BaseBridge):
    name = "slack"
    message_limit = SLACK_MSG_LIMIT

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        allowed_users: list[str] | None = None,
        listen_channels: list[str] | None = None,
        gateway_url: str = "ws://localhost:8765/ws",
        gateway_token: str | None = None,
        personality: str | None = None,
    ):
        super().__init__(gateway_url, gateway_token, personality=personality)
        self.bot_token = bot_token
        self.app_token = app_token
        self.allowed_users = (
            set(str(u) for u in allowed_users) if allowed_users else None
        )
        self.listen_channels = (
            set(str(c) for c in listen_channels) if listen_channels else None
        )
        self._app: Any = None
        self._socket_handler: Any = None
        self._bot_user_id: str | None = None

    def _is_authorized(self, user_id: str) -> bool:
        return self.allowed_users is None or user_id in self.allowed_users

    def _channel_listened(self, channel_id: str, is_dm: bool) -> bool:
        # DMs are always listened to (subject to allowed_users), public
        # channels need an explicit listen_channels entry. This matches
        # the principle-of-least-surprise: a bot you DM should just
        # answer; a bot in a channel should only answer where you put
        # it on purpose.
        if is_dm:
            return True
        if self.listen_channels is None:
            return False
        return channel_id in self.listen_channels

    async def _run(self) -> None:
        try:
            from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
            from slack_bolt.async_app import AsyncApp
        except ImportError as e:
            logger.error("slack bridge needs 'slack-bolt' installed (%s)", e)
            elog("bridge.error", name="slack", error="slack_bolt not installed")
            return

        app = AsyncApp(token=self.bot_token)
        self._app = app

        # Resolve bot user id so we can ignore self-replies.
        try:
            auth = await app.client.auth_test()
            self._bot_user_id = auth["user_id"]
        except Exception as e:  # noqa: BLE001
            logger.warning("slack auth_test failed: %s", e)

        @app.event("message")  # type: ignore[misc]
        async def _on_message(event, say, body, client):  # noqa: ANN001
            # Filter out bot echo, edits, joins, etc.
            if event.get("bot_id") or event.get("subtype"):
                return
            user_id = event.get("user")
            channel_id = event.get("channel")
            channel_type = event.get("channel_type") or ""
            is_dm = channel_type == "im"
            text = (event.get("text") or "").strip()
            if not user_id or not channel_id:
                return
            if user_id == self._bot_user_id:
                return
            if not self._is_authorized(str(user_id)):
                return
            if not self._channel_listened(str(channel_id), is_dm):
                return
            if not text:
                return
            # Strip a leading <@BOTID> mention so the model doesn't see
            # the noise; for DMs the mention isn't usually present.
            if self._bot_user_id:
                mention = f"<@{self._bot_user_id}>"
                if text.startswith(mention):
                    text = text[len(mention):].lstrip(": ").strip()

            # Bridge target carries the SDK client + channel for status
            # message edits below.
            target = _SlackTarget(client=client, channel=channel_id, user=user_id)
            session_id = f"sl:{user_id}"
            try:
                await self.dispatch_turn(target, session_id, text)
            except Exception as e:  # noqa: BLE001
                logger.exception("slack dispatch failed: %s", e)
                elog("bridge.error", name="slack", session_id=session_id, error=str(e)[:200])

        # Socket mode handler — long-lived; cancellation comes from
        # BaseBridge.stop() which sets _should_stop and cancels the
        # listener task.
        self._socket_handler = AsyncSocketModeHandler(app, self.app_token)
        try:
            await self._socket_handler.start_async()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            elog("bridge.error", name="slack", error=str(e)[:200])
            raise

    async def _shutdown_platform(self) -> None:
        # Called by BaseBridge.stop() — close the socket mode connection
        # cleanly so the websocket doesn't leak.
        if self._socket_handler is not None:
            with contextlib.suppress(Exception):
                await self._socket_handler.close_async()
            self._socket_handler = None
        if self._app is not None:
            with contextlib.suppress(Exception):
                await self._app.client.session.close()
            self._app = None

    # ── Platform primitives consumed by BaseBridge.dispatch_turn ──

    async def post_status(self, target, text: str):
        # Slack has no native typing indicator for bots; post a real
        # placeholder we can edit/delete. Returns the message handle so
        # update / clear can address it.
        try:
            resp = await target.client.chat_postMessage(
                channel=target.channel, text=f"⏳ {text}"
            )
            return _SlackStatusHandle(
                client=target.client, channel=target.channel, ts=resp["ts"]
            )
        except Exception:
            return None

    async def update_status(self, status_handle, text: str) -> None:
        if status_handle is None:
            return
        try:
            await status_handle.client.chat_update(
                channel=status_handle.channel,
                ts=status_handle.ts,
                text=f"⏳ {text}",
            )
        except Exception:
            pass

    async def clear_status(self, status_handle) -> None:
        if status_handle is None:
            return
        try:
            await status_handle.client.chat_delete(
                channel=status_handle.channel, ts=status_handle.ts
            )
        except Exception:
            pass

    async def send_text_chunk(self, target, chunk: str) -> None:
        # Slack mrkdwn is close enough to plain markdown that the
        # telegram HTML renderer would over-quote things. Send raw text
        # — Slack auto-linkifies URLs and respects \n for paragraphs.
        try:
            await target.client.chat_postMessage(
                channel=target.channel, text=chunk
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("slack send_text_chunk failed: %s", e)

    async def send_attachment(self, target, att) -> None:
        # Minimal v1: only file paths from the local fs. Slack uploads
        # via files_upload_v2 — keep it simple, no captions or
        # thumbnails; the model can describe the file in the reply text
        # if it wants.
        from pathlib import Path
        p = Path(getattr(att, "path", "") or "")
        if not p.exists() or not p.is_file():
            return
        try:
            await target.client.files_upload_v2(
                channel=target.channel,
                file=str(p),
                filename=p.name,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("slack send_attachment failed: %s", e)


class _SlackTarget:
    """Bag of what ``BaseBridge`` primitives need from a Slack event.

    Kept slim on purpose — the message object passed around between
    ``post_status`` / ``send_text_chunk`` / ``send_attachment`` only
    needs the SDK client + the channel id. No need to drag the full
    Bolt ``event`` payload through dispatch.
    """

    __slots__ = ("client", "channel", "user")

    def __init__(self, client, channel: str, user: str):
        self.client = client
        self.channel = channel
        self.user = user


class _SlackStatusHandle:
    """Opaque handle returned by ``post_status`` so ``update_status`` /
    ``clear_status`` can edit / delete that exact message later."""

    __slots__ = ("client", "channel", "ts")

    def __init__(self, client, channel: str, ts: str):
        self.client = client
        self.channel = channel
        self.ts = ts
