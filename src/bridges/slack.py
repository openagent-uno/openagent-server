"""Slack bridge — Bolt SDK socket-mode adapter.

Mirrors ``bridges/telegram.py`` in shape: subclass :class:`BaseBridge`,
implement platform-specific event polling, delegate orchestration to
``BaseBridge.dispatch_turn``. Socket mode is the right choice for a
self-hosted agent because the bot doesn't need a public HTTP endpoint
— it dials out to Slack's gateway and receives events over WebSocket.

Status indicator: Slack doesn't expose a typing primitive to bots
(unlike Telegram's chat.sendChatAction / Discord's typing()), so there
is no "is writing" flag to set and ``post_status`` is a no-op — the live
step messages (each tool call + answer span) are the progress affordance
instead. We no longer post a ``Thinking…`` placeholder.

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


def _build_picker_blocks(picker: dict | None) -> list | None:
    """Build Block Kit blocks with a ``static_select`` for a structured
    command picker (e.g. the /model chooser). Returns ``None`` when there
    are no renderable options — the caller then falls back to plain text.

    Each option's value encodes ``<command>::<runtime_id>`` so the shared
    ``oa_cmd_pick`` action handler can re-issue the right command. Slack
    caps a select at 100 options and each value/label at 75 chars.
    """
    if not isinstance(picker, dict):
        return None
    command = str(picker.get("command") or "")
    raw = picker.get("options") or []
    if not command or not raw:
        return None
    options = []
    for opt in raw[:100]:
        value = str(opt.get("value") or "")
        if not value:
            continue
        encoded = f"{command}::{value}"
        if len(encoded) > 75:  # Slack option value hard cap
            continue
        label = str(opt.get("label") or value)
        if opt.get("active"):
            label = f"✓ {label}"
        options.append({
            "text": {"type": "plain_text", "text": label[:75]},
            "value": encoded,
        })
    if not options:
        return None
    prompt = str(picker.get("prompt") or "Choose:")
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": prompt}},
        {"type": "actions", "elements": [{
            "type": "static_select",
            "action_id": "oa_cmd_pick",
            "placeholder": {"type": "plain_text", "text": "Select…"},
            "options": options,
        }]},
    ]


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
        live: bool = True,
    ):
        super().__init__(gateway_url, gateway_token, personality=personality, live=live)
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

        # Native Slack slash commands, delivered over socket mode (no
        # public request URL needed). Mirrors Telegram's bot commands and
        # Discord's app commands: /stop cancels the issuing user's running
        # turn, /clear|/new|/reset wipe + restart it, /status reports
        # state — all scoped to ``sl:<user_id>`` so one user's command
        # can't touch another's conversation. Slack only delivers these
        # once the user adds them under the app's Features → Slash Commands
        # (socket mode then routes them here). Without that registration
        # Slack swallows ``/stop`` as an "unknown command" and never hands
        # it to the bot — which is why the previous text-only handler could
        # not implement stop on Slack.
        from src.gateway.commands import BOT_COMMANDS

        def _make_slash_handler(command_name: str):
            async def _handler(ack, command, respond):  # noqa: ANN001
                await ack()
                uid = str(command.get("user_id") or "")
                if not self._is_authorized(uid):
                    await respond("Unauthorized.")
                    return
                try:
                    # Slack puts any text after the command name in
                    # command["text"] (e.g. "/model gpt-4o" → "gpt-4o").
                    slack_arg = (command.get("text") or "").strip() or None
                    result = await self.send_command_full(
                        command_name, session_id=f"sl:{uid}", arg=slack_arg,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("slack /%s failed: %s", command_name, e)
                    await respond(f"/{command_name} failed.")
                    return
                picker = result.get("picker") if isinstance(result, dict) else None
                blocks = _build_picker_blocks(picker) if isinstance(picker, dict) else None
                if blocks:
                    await respond(text=str(picker.get("prompt") or "Choose:"), blocks=blocks)
                else:
                    await respond(result.get("text", "") or f"/{command_name} done.")
            return _handler

        for _cmd_name, _ in BOT_COMMANDS:
            app.command(f"/{_cmd_name}")(_make_slash_handler(_cmd_name))

        # Block Kit picker selection (e.g. /model). The static_select's
        # value encodes ``<command>::<runtime_id>``; re-issue the command
        # and replace the picker message with the result. Registered once
        # for the shared action_id used by ``_build_picker_blocks``.
        @app.action("oa_cmd_pick")  # type: ignore[misc]
        async def _on_pick(ack, body, respond):  # noqa: ANN001
            await ack()
            uid = str((body.get("user") or {}).get("id") or "")
            if not self._is_authorized(uid):
                return
            try:
                selected = body["actions"][0]["selected_option"]["value"]
            except (KeyError, IndexError, TypeError):
                return
            command, _, value = selected.partition("::")
            if not command or not value:
                return
            try:
                result = await self.send_command(
                    command, session_id=f"sl:{uid}", arg=value,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("slack picker /%s failed: %s", command, e)
                result = f"/{command} failed."
            with contextlib.suppress(Exception):
                await respond(text=result or "Done.", replace_original=True)

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

    async def stop(self) -> None:
        # Close the slack-bolt socket-mode connection + aiohttp session
        # BEFORE the BaseBridge teardown cancels the ``_run`` task — the
        # base ``stop()`` never knew about the platform handler, so without
        # this the socket-mode websocket and its session leaked on every
        # shutdown / config reload (``_shutdown_platform`` was dead code).
        self._should_stop = True
        await self._shutdown_platform()
        await super().stop()

    async def _shutdown_platform(self) -> None:
        # Close the socket mode connection cleanly so the websocket and its
        # aiohttp session don't leak. Invoked from ``stop()`` above.
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
        # Slack exposes no native bot typing indicator, so there is no
        # "is writing" flag to set and we deliberately do NOT post a
        # "Thinking…" placeholder message. The live step messages (each
        # tool call + answer span, BaseBridge.dispatch_turn live mode)
        # are the progress affordance instead. Returns None — nothing to
        # update or clear.
        return None

    # update_status / clear_status inherit the BaseBridge no-op defaults:
    # there is no placeholder to edit or delete.

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
