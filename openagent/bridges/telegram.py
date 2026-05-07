"""Telegram bridge — translates Telegram Bot API ↔ Gateway WS protocol."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from openagent.bridges.base import BaseBridge
from openagent.channels.formatting import markdown_to_telegram_html
from openagent.channels.base import (
    build_attachment_context,
    is_blocked_attachment,
    prepend_context_block,
)
from openagent.gateway.commands import BOT_COMMANDS, BRIDGE_COMMANDS, bridge_welcome_text

from openagent.core.logging import elog

logger = logging.getLogger(__name__)

TELEGRAM_MSG_LIMIT = 4096


@dataclass
class _Extracted:
    """Output of :meth:`TelegramBridge._extract_files` for one Telegram message."""
    text_addition: str = ""
    files_info: list[str] = field(default_factory=list)
    voice_detected: bool = False

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
# Extra grace for the force-cancel fallback. Only used when the library's
# own ``updater.stop()`` exceeded its deadline and left the polling task
# running in the background.
_TG_FORCE_POLLING_CANCEL_TIMEOUT = 1.0

# How many recent update_ids to remember for duplicate detection. Telegram
# can re-deliver an Update when our offset ACK was lost (network timeout
# during ``getUpdates``, two bot processes racing on the same token, a
# shutdown that SIGKILLed before ``flush_updates_offset`` landed). Without
# dedup the replay reaches ``_on_message`` and the user sees their prior
# message answered again — often "super fast" because the model's prompt
# cache is still warm. 256 is large enough to span any realistic burst
# and small enough that the deque is a no-op on memory.
_SEEN_UPDATE_IDS_MAX = 256

# Telegram delivers a "media group" (a multi-file message from the
# user's perspective) as N independent Updates that share the same
# ``media_group_id`` string. Only one of them carries the user's
# caption. We buffer per ``(uid, media_group_id)`` and flush after a
# rolling debounce so all N siblings collapse into a single agent turn
# instead of N (each starting its own thinking spinner). 1.0s mirrors
# Telegram's own client cadence and is robust to slow uplinks where
# the last sibling lags by several hundred ms.
_MEDIA_GROUP_FLUSH_DELAY = 1.0


class TelegramBridge(BaseBridge):
    name = "telegram"
    message_limit = TELEGRAM_MSG_LIMIT

    def __init__(self, token: str, allowed_users: list[str] | None = None,
                 gateway_url: str = "ws://localhost:8765/ws", gateway_token: str | None = None):
        super().__init__(gateway_url, gateway_token)
        self.token = token
        self.allowed_users = set(allowed_users) if allowed_users else None
        self._app = None
        # Highest update_id we've seen from Telegram. Used during shutdown
        # to directly ACK the offset so a queued ``/restart`` cannot
        # replay on next boot. The original failure mode: shutdown hung
        # inside ``updater.stop()`` → ``_get_updates_cleanup``, so the
        # offset never advanced; the service manager restarted us, and
        # the same ``/restart`` Update came right back from
        # ``getUpdates`` — crash loop.
        self._last_update_id: int = 0
        # Bounded set of recently-seen update_ids used by ``_is_fresh_update``
        # to reject Telegram redeliveries. The deque enforces the size cap
        # while the set gives O(1) membership; they're kept in lockstep.
        self._seen_update_ids: set[int] = set()
        self._seen_update_ids_order: deque[int] = deque(maxlen=_SEEN_UPDATE_IDS_MAX)
        # Buffer for media-group siblings keyed by (uid, media_group_id).
        # Each entry is ``{"messages": [Message...], "timer": Task}``. The
        # lock guards the dict against races where two updates land on
        # different worker threads inside python-telegram-bot.
        self._media_groups: dict[tuple[str, str], dict] = {}
        self._media_group_lock = asyncio.Lock()

    def _is_authorized(self, uid: str) -> bool:
        return self.allowed_users is None or uid in self.allowed_users

    async def _run(self) -> None:
        from telegram import BotCommand
        from telegram.ext import (
            ApplicationBuilder, CommandHandler, MessageHandler, filters,
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

    def _is_fresh_update(self, update) -> bool:
        """Return True on first sight of ``update.update_id``, False on replay.

        Telegram can re-deliver an Update when our offset ACK was lost
        (see the ``_SEEN_UPDATE_IDS_MAX`` comment). Without this guard the
        bridge happily processes the duplicate: the user sees their prior
        message answered a second time, usually faster than a real turn
        because the model's prompt cache is warm.

        ``_track_update_id`` still runs so ``flush_updates_offset`` knows
        the highest id even if a duplicate is rejected.
        """
        self._track_update_id(update)
        try:
            uid = getattr(update, "update_id", None)
        except Exception:
            return True  # unknown id — can't dedup, let it through
        if not isinstance(uid, int):
            return True
        if uid in self._seen_update_ids:
            elog(
                "bridge.telegram.duplicate_update_skipped",
                level="warning",
                update_id=uid,
            )
            return False
        # Bounded set: when the deque evicts an old id, drop it from the
        # lookup set too so the two stay in sync.
        if len(self._seen_update_ids_order) == self._seen_update_ids_order.maxlen:
            evicted = self._seen_update_ids_order[0]
            self._seen_update_ids.discard(evicted)
        self._seen_update_ids_order.append(uid)
        self._seen_update_ids.add(uid)
        return True

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
            # ``httpx.CancelledError`` and a few other shutdown-time
            # exceptions stringify to ``""`` — fall back to the type
            # name so events.jsonl never logs ``error: ""`` (which is
            # unsearchable and prevents postmortem diagnosis).
            elog(
                "bridge.telegram.offset_flush_error",
                level="warning",
                offset=next_offset,
                error=str(e) or type(e).__name__,
            )

    async def _force_stop_updater_poller(self, app) -> None:
        """Best-effort kill switch for a leaked PTB polling task.

        ``python-telegram-bot`` keeps the long-poll loop on a private
        ``Updater.__polling_task``. If ``updater.stop()`` times out while
        awaiting that task, the old ``getUpdates`` call can survive into
        the next bridge boot and Telegram replies with 409 Conflict on
        every poll. We already flush the offset manually before teardown,
        so it's safe to skip PTB's cleanup callback here and cancel the
        task forcefully.
        """
        updater = getattr(app, "updater", None)
        if updater is None:
            return
        poll_task = getattr(updater, "_Updater__polling_task", None)
        stop_event = getattr(updater, "_Updater__polling_task_stop_event", None)
        cleanup_cb = getattr(updater, "_Updater__polling_cleanup_cb", None)

        if stop_event is not None:
            with contextlib.suppress(Exception):
                stop_event.set()

        if isinstance(poll_task, asyncio.Task) and not poll_task.done():
            poll_task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(poll_task),
                    timeout=_TG_FORCE_POLLING_CANCEL_TIMEOUT,
                )
            except asyncio.TimeoutError:
                elog(
                    "bridge.telegram.force_cancel_timeout",
                    level="warning",
                    timeout=_TG_FORCE_POLLING_CANCEL_TIMEOUT,
                )
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                elog(
                    "bridge.telegram.force_cancel_error",
                    level="warning",
                    error=str(e) or type(e).__name__,
                )

        if poll_task is not None:
            setattr(updater, "_Updater__polling_task", None)
        if cleanup_cb is not None:
            setattr(updater, "_Updater__polling_cleanup_cb", None)
        if stop_event is not None:
            with contextlib.suppress(Exception):
                stop_event.clear()
        with contextlib.suppress(Exception):
            updater._running = False
        elog(
            "bridge.telegram.updater_force_stopped",
            had_polling_task=bool(poll_task),
            dropped_cleanup=cleanup_cb is not None,
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
            # been observed to hang during an outer cancellation: the
            # cancel scope from ``Gateway._handle_ws.handler`` fires
            # while ``_get_updates_cleanup`` is mid-POST. Bound each
            # with a short timeout and swallow CancelledError at this
            # boundary — callers above us must be able to finish the
            # rest of shutdown regardless of what python-telegram-bot
            # decides to do.
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
                    if label == "updater.stop":
                        await self._force_stop_updater_poller(app)
                except asyncio.CancelledError:
                    # Do not propagate — a CancelledError from httpx/httpcore
                    # during library cleanup must not bleed out into the
                    # gateway shutdown path and derail MCP teardown, etc.
                    elog(
                        "bridge.telegram.stop_cancelled",
                        level="warning",
                        phase=label,
                    )
                    if label == "updater.stop":
                        await self._force_stop_updater_poller(app)
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
        if not self._is_fresh_update(update):
            return
        name = update.message.from_user.first_name or "there"
        await update.message.reply_text(bridge_welcome_text(name))

    async def _on_command(self, update, context, cmd):
        if not self._is_fresh_update(update):
            return
        if not update.message:
            return
        user_id = str(update.message.from_user.id)
        if not self._is_authorized(user_id):
            return await update.message.reply_text("Unauthorized.")
        # Scope scope-sensitive commands (/stop, /clear, /new, /reset) to
        # just the user who issued them. Other users on the same bot keep
        # their own conversations.
        result = await self.send_command(cmd, session_id=f"tg:{user_id}")
        await self.send_response_text(update.message, result)

    async def _extract_files(self, msg, tmp: str) -> _Extracted:
        """Download every attachment on a single Telegram ``Message`` to ``tmp``.

        Returns an :class:`_Extracted` carrying the voice-transcription
        text (if any), the per-file ``build_attachment_context`` lines,
        and a flag set when the user sent a voice note (so the caller
        can mirror the modality on reply).
        """
        out = _Extracted()

        if msg.photo:
            photo = msg.photo[-1]
            f = await photo.get_file()
            path = str(Path(tmp) / f"photo_{photo.file_unique_id}.jpg")
            await f.download_to_drive(path)
            out.files_info.append(f"- image: photo.jpg — local path: {path}")

        if msg.voice:
            out.voice_detected = True
            f = await msg.voice.get_file()
            path = str(Path(tmp) / f"voice_{msg.voice.file_unique_id}.ogg")
            await f.download_to_drive(path)
            out.text_addition = await self.transcribe_with_fallback(path)

        if msg.audio:
            fname = msg.audio.file_name or f"audio_{msg.audio.file_unique_id}"
            if not is_blocked_attachment(fname):
                f = await msg.audio.get_file()
                path = str(Path(tmp) / fname)
                await f.download_to_drive(path)
                out.files_info.append(f"- file: {fname} — local path: {path}")

        if msg.document:
            fname = msg.document.file_name or f"doc_{msg.document.file_unique_id}"
            if is_blocked_attachment(fname):
                await msg.reply_text(f"⚠️ Blocked: {fname}")
            else:
                f = await msg.document.get_file()
                path = str(Path(tmp) / fname)
                await f.download_to_drive(path)
                out.files_info.append(f"- file: {fname} — local path: {path}")

        if msg.video:
            fname = msg.video.file_name or f"video_{msg.video.file_unique_id}.mp4"
            if not is_blocked_attachment(fname):
                f = await msg.video.get_file()
                path = str(Path(tmp) / fname)
                await f.download_to_drive(path)
                out.files_info.append(f"- video: {fname} — local path: {path}")

        return out

    async def _on_message(self, update, context):
        if not self._is_fresh_update(update):
            return
        msg = update.message
        if not msg:
            return
        uid = str(msg.from_user.id)
        if not self._is_authorized(uid):
            return await msg.reply_text("Unauthorized.")

        # Coalesce media groups: Telegram delivers each attachment in a
        # multi-file message as its own Update sharing ``media_group_id``,
        # so without buffering each file would start its own agent turn.
        # Stash and let ``_flush_media_group`` dispatch one combined
        # message after the rolling debounce expires.
        group_id = getattr(msg, "media_group_id", None)
        if group_id:
            await self._enqueue_media_group(uid, group_id, msg)
            return

        elog("bridge.message", bridge="telegram", user_id=uid)
        text = msg.caption or msg.text or ""
        tmp = tempfile.mkdtemp(prefix="oa_tg_")
        extracted = await self._extract_files(msg, tmp)
        if extracted.text_addition:
            text = f"{text}\n{extracted.text_addition}" if text else extracted.text_addition

        if extracted.files_info:
            text = prepend_context_block(text, build_attachment_context(extracted.files_info))

        if not text:
            return

        await self.dispatch_turn(
            msg, f"tg:{uid}", text, voice_detected=extracted.voice_detected,
        )

    async def _enqueue_media_group(self, uid: str, group_id: str, msg) -> None:
        """Buffer a media-group sibling and (re)arm the flush timer."""
        key = (uid, group_id)
        async with self._media_group_lock:
            entry = self._media_groups.get(key)
            if entry is None:
                entry = {"messages": [msg], "timer": None}
                self._media_groups[key] = entry
            else:
                entry["messages"].append(msg)
                # Cancel and re-arm: rolling debounce, robust to siblings
                # arriving with hundreds of ms of jitter on slow uplinks.
                prior = entry.get("timer")
                if prior is not None and not prior.done():
                    prior.cancel()
            entry["timer"] = asyncio.create_task(
                self._media_group_timer(key)
            )

    async def _media_group_timer(self, key: tuple[str, str]) -> None:
        try:
            await asyncio.sleep(_MEDIA_GROUP_FLUSH_DELAY)
        except asyncio.CancelledError:
            return
        try:
            await self._flush_media_group(key)
        except Exception as e:  # noqa: BLE001
            elog(
                "bridge.media_group_flush_error",
                level="error",
                bridge="telegram",
                error=str(e),
            )

    async def _flush_media_group(self, key: tuple[str, str]) -> None:
        async with self._media_group_lock:
            entry = self._media_groups.pop(key, None)
        if entry is None:
            return
        messages = entry.get("messages") or []
        if not messages:
            return

        uid, _group_id = key
        elog(
            "bridge.media_group_flush",
            bridge="telegram",
            user_id=uid,
            count=len(messages),
        )

        # The caption can sit on any sibling — Telegram's API does not
        # guarantee the first message carries it. Collect every distinct
        # non-empty caption so we recover from forwards that echo it on
        # several siblings without duplicating the text in the prompt.
        seen_captions: set[str] = set()
        captions: list[str] = []
        for m in messages:
            cap = (getattr(m, "caption", None) or getattr(m, "text", None) or "").strip()
            if cap and cap not in seen_captions:
                seen_captions.add(cap)
                captions.append(cap)
        text = "\n".join(captions)

        tmp = tempfile.mkdtemp(prefix="oa_tg_")
        all_files: list[str] = []
        any_voice = False
        for m in messages:
            try:
                extracted = await self._extract_files(m, tmp)
            except Exception as e:  # noqa: BLE001
                elog(
                    "bridge.media_group_extract_error",
                    level="warning",
                    bridge="telegram",
                    error=str(e),
                )
                continue
            if extracted.text_addition:
                text = f"{text}\n{extracted.text_addition}" if text else extracted.text_addition
            all_files.extend(extracted.files_info)
            if extracted.voice_detected:
                any_voice = True

        if all_files:
            text = prepend_context_block(text, build_attachment_context(all_files))

        if not text:
            return

        # Anchor the status reply on the first sibling so the user sees
        # a single in-flight indicator next to the group.
        await self.dispatch_turn(
            messages[0], f"tg:{uid}", text, voice_detected=any_voice,
        )

    # ── Platform primitives (consumed by BaseBridge.dispatch_turn) ──
    #
    # The shared orchestration is in ``BaseBridge.dispatch_turn``; here
    # we only translate between its calls and the python-telegram-bot
    # ``Message`` surface. Voice-mode mirroring (voice in → voice out)
    # is also handled by BaseBridge via ``maybe_prepend_voice_reply``,
    # so this file no longer carries any of that logic.

    async def post_status(self, msg, text: str):
        try:
            return await msg.reply_text(f"⏳ {text}")
        except Exception:
            return None

    async def update_status(self, status_msg, text: str) -> None:
        try:
            await status_msg.edit_text(f"⏳ {text}")
        except Exception:
            pass

    async def clear_status(self, status_msg) -> None:
        try:
            await status_msg.delete()
        except Exception:
            pass

    async def send_text_chunk(self, msg, chunk: str) -> None:
        # Render once per chunk so a giant reply doesn't pay the cost
        # in one go AND so a chunk-local HTML failure can fall back to
        # plain text without losing the rest of the response.
        rendered = markdown_to_telegram_html(chunk)
        try:
            await msg.reply_text(rendered, parse_mode="HTML", disable_web_page_preview=True)
        except Exception:
            try:
                await msg.reply_text(chunk)
            except Exception:
                pass

    async def send_attachment(self, msg, att) -> None:
        p = Path(att.path)
        if not p.exists():
            return
        with open(p, "rb") as f:
            if att.type == "image":
                await msg.reply_photo(photo=f)
            elif att.type == "voice":
                # Telegram's reply_voice requires OGG/Opus; MP3 (the
                # LiteLLM default) ships via reply_audio instead —
                # renders as a music-player bubble.
                if p.suffix.lower() in (".ogg", ".oga", ".opus"):
                    await msg.reply_voice(voice=f)
                else:
                    await msg.reply_audio(audio=f)
            elif att.type == "video":
                await msg.reply_video(video=f)
            else:
                await msg.reply_document(document=f, filename=att.filename)
