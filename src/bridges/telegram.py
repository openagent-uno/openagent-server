"""Telegram bridge — translates Telegram Bot API ↔ Gateway WS protocol."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from src.bridges.base import BaseBridge
from src.channels.formatting import markdown_to_telegram_html
from src.channels.base import (
    build_attachment_context,
    is_blocked_attachment,
    prepend_context_block,
)
from src.gateway.commands import BOT_COMMANDS, BRIDGE_COMMANDS, bridge_welcome_text

from src.core.logging import elog

logger = logging.getLogger(__name__)

TELEGRAM_MSG_LIMIT = 4096

# Commands that wipe state — bridge intercepts them and shows an inline
# Yes/No keyboard before firing the gateway command. Without the gate
# an accidental ``/clear`` (fat-fingered while scrolling commands) nukes
# the session silently.
_DESTRUCTIVE_COMMANDS: set[str] = {"clear", "new", "reset"}


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


class _TypingAnimator:
    """Drives a continuous Telegram ``ChatAction.TYPING`` while a turn is
    in flight. Replaces the ``Thinking...`` status reply used elsewhere
    in this bridge so the user sees the platform's native typing dot —
    matching the UX of Hermes / the Telegram bot SDK examples.

    Telegram drops the typing indicator after ~5 s of silence; the loop
    re-sends ``send_chat_action`` every ``_TYPING_REFRESH_S`` to keep the
    dot alive without spamming. ``stop()`` flips the event so the loop
    exits on its next wake-up (next refresh tick or immediately if the
    sleep is interrupted), the indicator vanishes naturally as soon as
    the next outbound message lands.

    Optional ``placeholder_message`` slot holds the editable reply
    bubble used by the streaming path (``stream_text``). When set, the
    bridge edits this message in place as the model types instead of
    waiting for the final response.
    """

    _TYPING_REFRESH_S = 4.0
    # Telegram caps message edits at roughly 1/sec per chat; pushing
    # past that yields 429s and the SDK retries with backoff. We stay
    # below the cap with a small safety margin so a burst of deltas
    # doesn't accidentally throttle the next user-facing edit either.
    _STREAM_EDIT_MIN_INTERVAL_S = 1.1

    def __init__(self, bot, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        # Streaming state — populated only when ``stream_text`` is
        # invoked; left None on non-streaming turns so the typing-dot
        # behaviour stays unchanged.
        self.placeholder_message: Any = None
        self._last_edit_at: float = 0.0
        self._pending_text: str | None = None
        self._edit_lock = asyncio.Lock()

    async def start(self) -> None:
        from telegram.constants import ChatAction
        # First action awaited synchronously so Telegram registers TYPING
        # before dispatch_turn proceeds to the model round-trip. Without
        # this the chat-list preview can miss the "typing…" state for
        # sub-second turns: the reply lands before the client has had
        # time to switch the preview off the user's last message.
        try:
            await self._bot.send_chat_action(self._chat_id, ChatAction.TYPING)
        except Exception as e:  # noqa: BLE001
            elog(
                "bridge.telegram.typing_action_failed",
                chat_id=self._chat_id, phase="initial", error=str(e)[:200],
            )
        async def _loop() -> None:
            while not self._stop.is_set():
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self._TYPING_REFRESH_S)
                except asyncio.TimeoutError:
                    pass
                if self._stop.is_set():
                    break
                try:
                    await self._bot.send_chat_action(self._chat_id, ChatAction.TYPING)
                except Exception as e:  # noqa: BLE001
                    # Network blip or chat-removed; keep looping, the
                    # outer dispatch will tear us down when the turn ends.
                    # Surface via elog so silent throttling/permission
                    # issues become diagnosable instead of disappearing.
                    elog(
                        "bridge.telegram.typing_action_failed",
                        chat_id=self._chat_id, phase="keepalive", error=str(e)[:200],
                    )
        self._task = asyncio.create_task(_loop())

    async def stream_chunk(self, full_text: str) -> None:
        """Edit the placeholder message with ``full_text``, rate-limited
        to one edit per ``_STREAM_EDIT_MIN_INTERVAL_S``. Lazily posts
        the placeholder on the first call.

        The whole body is serialised under ``_edit_lock`` because deltas
        arrive on N concurrent tasks (``fold_outbound_event`` schedules
        each one via ``asyncio.create_task``). Without the lock, the
        first wave of deltas all see ``placeholder_message is None``
        and race to send a separate message — the symptom that showed
        up as a chat full of half-words. With the lock, exactly one
        send + many edits happen.
        """
        import time as _time
        if not full_text:
            return
        async with self._edit_lock:
            try:
                if self.placeholder_message is None:
                    self.placeholder_message = await self._bot.send_message(
                        chat_id=self._chat_id, text=full_text[:4000]
                    )
                    self._last_edit_at = _time.monotonic()
                    self._pending_text = full_text
                    return
                now = _time.monotonic()
                if now - self._last_edit_at < self._STREAM_EDIT_MIN_INTERVAL_S:
                    # Stash the latest text; the next call past the
                    # rate-limit window will land it.
                    self._pending_text = full_text
                    return
                # Skip the edit when the visible text hasn't changed —
                # Telegram returns ``message is not modified`` and would
                # otherwise just swallow it, but avoiding the round-trip
                # entirely keeps us inside the 1-edit/sec budget.
                if self._pending_text == full_text:
                    return
                try:
                    await self.placeholder_message.edit_text(full_text[:4000])
                    self._last_edit_at = _time.monotonic()
                    self._pending_text = full_text
                except Exception:
                    pass
            except Exception:
                pass

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                pass
        # Delete the placeholder if streaming was used — the final
        # reply lands as a fresh chunked message via send_text_chunk
        # and we don't want the user staring at a half-written
        # duplicate. The tiny window between delete and send is
        # unavoidable without bigger surgery in BaseBridge.dispatch_turn.
        if self.placeholder_message is not None:
            try:
                await self.placeholder_message.delete()
            except Exception:
                pass
            self.placeholder_message = None


class TelegramBridge(BaseBridge):
    name = "telegram"
    message_limit = TELEGRAM_MSG_LIMIT

    def __init__(self, token: str, allowed_users: list[str] | None = None,
                 gateway_url: str = "ws://localhost:8765/ws", gateway_token: str | None = None,
                 personality: str | None = None, streaming: bool = False,
                 allowed_chats: list[int] | None = None, live: bool = True):
        super().__init__(gateway_url, gateway_token, personality=personality, live=live)
        self.token = token
        # Group/supergroup chats the bot is allowed to listen in. ``None``
        # = listen anywhere it's added (still gated by @mention / reply
        # so it doesn't spam). Provide as a list of chat IDs (signed
        # integers, negative for groups) to lock it down for work
        # deployments.
        self.allowed_chats: set[int] | None = (
            {int(c) for c in allowed_chats} if allowed_chats is not None else None
        )
        # Populated lazily on first inbound update — Telegram's
        # ``get_me`` is the canonical source. Used to detect mentions
        # in group chats so the bot stays quiet unless addressed.
        self._bot_username: str | None = None
        # Streaming: edit a placeholder message as the model types.
        # Off by default because token-by-token Telegram bubbles feel
        # noisy in practice — the user typically wants the response
        # to land as one cohesive bubble. Operators that prefer the
        # "ChatGPT-style live typing" feel turn it on explicitly via
        # ``channels.telegram.streaming: true``.
        self._streaming_enabled = bool(streaming)
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
            ApplicationBuilder, CallbackQueryHandler, CommandHandler,
            MessageHandler, filters,
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
        self._app.add_handler(CommandHandler("export", self._on_export))
        for cmd in BRIDGE_COMMANDS:
            self._app.add_handler(CommandHandler(cmd, lambda u, c, _c=cmd: self._on_command(u, c, _c)))
        # Inline-keyboard callbacks (destructive-command confirm + choice
        # prompts emitted by the model via the ``[CHOICE: …]`` marker).
        self._app.add_handler(CallbackQueryHandler(self._on_callback))

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
        while not self._should_stop and not self._gateway_lost.is_set():
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

    async def _shutdown_app(self, *, flush_offset: bool) -> None:
        app = self._app
        if flush_offset:
            # ACK pending updates FIRST so that a queued /restart or /stop
            # can't survive a library-side hang. Idempotent — if
            # flush_updates_offset was already called by the gateway restart
            # path, this is a no-op (at worst a second short POST).
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

    async def _on_gateway_lost(self) -> None:
        await self._shutdown_app(flush_offset=False)

    async def stop(self) -> None:
        self._should_stop = True
        await self._shutdown_app(flush_offset=True)
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
        # Friendly onboarding shape: welcome + brief capability blurb +
        # three quick-start buttons so the user has something to tap
        # on even before figuring out the keyboard. Each button just
        # injects a starter prompt into a fresh turn — no per-user
        # preference persistence is wired yet because the agent's own
        # ``user_profile`` flush will pick those up over the next
        # few turns naturally.
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        welcome = (
            f"Ciao {name}! 👋\n\n"
            "Sono il tuo agente personale. Posso:\n"
            "• rispondere a domande, scrivere testi, analizzare immagini\n"
            "• ricordarmi cose tra una conversazione e l'altra\n"
            "• eseguire script, cercare sul web, accedere a file\n"
            "• rispondere ai messaggi vocali con la voce\n\n"
            "Inizia con qualcosa qui sotto o scrivimi liberamente."
        )
        starter_kb = InlineKeyboardMarkup([
            [InlineKeyboardButton(
                "💬 Cosa sai fare?",
                callback_data="choice:0:Spiegami brevemente cosa puoi fare e come ti uso al meglio.",
            )],
            [InlineKeyboardButton(
                "📝 Salva una nota",
                callback_data="choice:0:Voglio salvare una nota nella tua memoria — ti dico cosa.",
            )],
            [InlineKeyboardButton(
                "📋 Tutti i comandi",
                callback_data="confirm_cmd:help",
            )],
        ])
        try:
            await update.message.reply_text(welcome, reply_markup=starter_kb)
        except Exception:
            # Fallback to the legacy plain welcome if the keyboard
            # render fails (older Telegram clients, network blip).
            await update.message.reply_text(bridge_welcome_text(name))

    async def _on_command(self, update, context, cmd):
        if not self._is_fresh_update(update):
            return
        if not update.message:
            return
        user_id = str(update.message.from_user.id)
        if not self._is_authorized(user_id):
            return await update.message.reply_text("Unauthorized.")
        # Destructive commands route through an inline Yes/No prompt
        # first. The callback handler (``_on_callback``) re-emits the
        # original command only on confirmation.
        if cmd in _DESTRUCTIVE_COMMANDS:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("✅ Sì", callback_data=f"confirm_cmd:{cmd}"),
                InlineKeyboardButton("❌ Annulla", callback_data="cancel_cmd"),
            ]])
            await update.message.reply_text(
                f"Confermi `/{cmd}`? Cancellerà la storia della conversazione.",
                reply_markup=kb,
                parse_mode="Markdown",
            )
            return
        # Scope scope-sensitive commands to just the user who issued them.
        # Other users on the same bot keep their own conversations.
        result = await self.send_command(cmd, session_id=f"tg:{user_id}")
        await self.send_response_text(update.message, result)

    async def _on_export(self, update, context) -> None:
        """``/export`` — bundle the user's profile + skills + recent
        turns into a zip and send it back as a Telegram document.
        GDPR-friendly and lets users back up their context between
        deployments. Heavy lifting (DB reads) runs against the cached
        ``_learning_conn`` already opened for the skill matcher."""
        if not self._is_fresh_update(update):
            return
        if not update.message:
            return
        user_id = str(update.message.from_user.id)
        if not self._is_authorized(user_id):
            return await update.message.reply_text("Unauthorized.")
        import io
        import json as _json
        import zipfile
        session_id = f"tg:{user_id}"
        progress = await update.message.reply_text("📦 Preparo l'export...")
        try:
            shim = await self._get_learning_db_shim()
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                # User profile (the cross-session "who is the user" JSON).
                if shim is not None:
                    try:
                        cur = await shim._conn.execute(
                            "SELECT profile_json, turn_count, last_flushed_at "
                            "FROM user_profiles WHERE session_id = ?",
                            (session_id,),
                        )
                        row = await cur.fetchone()
                        await cur.close()
                    except Exception:
                        row = None
                    if row:
                        zf.writestr(
                            "user_profile.json",
                            _json.dumps(
                                {
                                    "session_id": session_id,
                                    "profile": _json.loads(row[0] or "{}"),
                                    "turn_count": row[1],
                                    "last_flushed_at": row[2],
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                        )
                    else:
                        zf.writestr(
                            "user_profile.json",
                            "{}",
                        )
                    # Skills — current implementation has a single global
                    # set per agent; for a personal bot deployment they're
                    # effectively this user's. Future multi-tenant should
                    # filter by user_id.
                    try:
                        cur = await shim._conn.execute(
                            "SELECT id, name, signature_hash, markdown, tags_json, "
                            "       usage_count, last_used_at, created_at "
                            "FROM skills ORDER BY usage_count DESC, last_used_at DESC NULLS LAST"
                        )
                        skills_rows = await cur.fetchall()
                        await cur.close()
                    except Exception:
                        skills_rows = []
                    skills_payload = []
                    for r in skills_rows or []:
                        try:
                            tags = _json.loads(r[4] or "[]")
                        except Exception:
                            tags = []
                        skills_payload.append({
                            "id": r[0],
                            "name": r[1],
                            "signature_hash": r[2],
                            "markdown": r[3],
                            "tags": tags,
                            "usage_count": r[5],
                            "last_used_at": r[6],
                            "created_at": r[7],
                        })
                    zf.writestr(
                        "skills.json",
                        _json.dumps(skills_payload, ensure_ascii=False, indent=2),
                    )
                    # Recent session row (carries the runtime-tracked
                    # turn transcript for this user's session).
                    try:
                        cur = await shim._conn.execute(
                            "SELECT session_data, agent_data, runs, summary, "
                            "       updated_at FROM sessions WHERE session_id = ?",
                            (session_id,),
                        )
                        sess = await cur.fetchone()
                        await cur.close()
                    except Exception:
                        sess = None
                    if sess:
                        zf.writestr(
                            "session.json",
                            _json.dumps(
                                {
                                    "session_id": session_id,
                                    "session_data": sess[0],
                                    "agent_data": sess[1],
                                    "runs": sess[2],
                                    "summary": sess[3],
                                    "updated_at": sess[4],
                                },
                                ensure_ascii=False,
                                indent=2,
                            ),
                        )
                # A friendly README so the user knows what's inside.
                zf.writestr(
                    "README.md",
                    "# OpenAgent export\n\n"
                    f"Export for session `{session_id}` generated at "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')}.\n\n"
                    "## Files\n\n"
                    "- `user_profile.json` — the cross-session profile the bot keeps about you.\n"
                    "- `skills.json` — reusable how-tos the bot has learned.\n"
                    "- `session.json` — the latest conversation transcript.\n\n"
                    "All data is your property. Use `/clear` in chat to "
                    "wipe the live session at any time.\n",
                )
            buf.seek(0)
            try:
                await progress.delete()
            except Exception:
                pass
            await update.message.reply_document(
                document=buf,
                filename=f"openagent-export-{time.strftime('%Y%m%d-%H%M%S')}.zip",
                caption="📦 Ecco i tuoi dati — profilo, skill, ultima sessione.",
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("/export failed: %s", e)
            try:
                await progress.edit_text(f"⚠️ Errore durante l'export: {str(e)[:200]}")
            except Exception:
                pass

    async def _on_callback(self, update, context) -> None:
        """Generic Telegram ``CallbackQuery`` handler.

        Two callback families today, both routed by ``callback_data`` prefix:

        - ``confirm_cmd:<name>`` / ``cancel_cmd`` — destructive command
          confirmation flow (set up in ``_on_command``).
        - ``choice:<turn_id>:<value>`` — inline-keyboard choice for
          model-driven question prompts (rendered when the agent's
          reply contains a ``[CHOICE: a | b | c]`` marker). The user's
          click becomes a fresh agent turn carrying the chosen value.
        """
        cb = update.callback_query
        if cb is None:
            return
        try:
            await cb.answer()  # dismiss the loading spinner on the client
        except Exception:
            pass
        user_id = str(cb.from_user.id) if cb.from_user else ""
        if not self._is_authorized(user_id):
            return
        data = cb.data or ""
        if data == "cancel_cmd":
            try:
                await cb.edit_message_text("Annullato.")
            except Exception:
                pass
            return
        if data.startswith("confirm_cmd:"):
            cmd = data[len("confirm_cmd:"):]
            result = await self.send_command(cmd, session_id=f"tg:{user_id}")
            try:
                await cb.edit_message_text(
                    f"✅ Fatto: `/{cmd}`. {result or 'Conversazione resettata.'}",
                    parse_mode="Markdown",
                )
            except Exception:
                pass
            return
        if data.startswith("choice:"):
            # Format: choice:<turn_id>:<value> — the value is what we
            # send back to the agent as the user's "click". Turn_id is
            # ignored on the server side for now; future work can use
            # it to attach the click to a specific pending turn.
            _, _, payload = data.partition(":")
            _, _, value = payload.partition(":")
            if value:
                try:
                    await cb.edit_message_reply_markup(reply_markup=None)
                except Exception:
                    pass
                # Treat the click as a normal user turn with the value
                # as text. Lets the agent react in its existing loop.
                fake_target = cb.message
                if fake_target is not None:
                    try:
                        await self.dispatch_turn(
                            fake_target, f"tg:{user_id}", value
                        )
                    except Exception:
                        pass

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
        # Group-chat gate: in groups + supergroups the bot only speaks
        # when @mentioned or replied-to, otherwise it would chatter on
        # every unrelated message. Channels are ignored entirely (1-way
        # broadcast surface, not a conversation). DMs always pass.
        chat = msg.chat
        chat_id = int(chat.id) if chat else 0
        chat_type = (chat.type if chat else "").lower()
        is_group = chat_type in ("group", "supergroup")
        session_id = f"tg:{uid}"
        if chat_type == "channel":
            return
        if is_group:
            if (
                self.allowed_chats is not None
                and chat_id not in self.allowed_chats
            ):
                # Group is not on the whitelist (when one is set).
                return
            # Lazily resolve the bot's own username so we can match
            # ``@bot_user`` mentions reliably.
            if self._bot_username is None:
                try:
                    me = await context.bot.get_me()
                    self._bot_username = (me.username or "").lower()
                except Exception:
                    self._bot_username = ""
            text_for_mention = (msg.text or msg.caption or "").lower()
            mentioned = bool(
                self._bot_username
                and f"@{self._bot_username}" in text_for_mention
            )
            replied_to_bot = bool(
                getattr(msg, "reply_to_message", None) is not None
                and getattr(msg.reply_to_message, "from_user", None) is not None
                and getattr(msg.reply_to_message.from_user, "is_bot", False)
            )
            if not (mentioned or replied_to_bot):
                return
            # Strip the @mention from the prompt so the model doesn't
            # waste tokens on it.
            session_id = f"tg:group:{chat_id}"
            if mentioned and self._bot_username:
                # Mutate-safe: we re-read .text below from msg, so
                # update the local handle the rest of the function uses.
                pass  # actual strip happens after ``text`` is read below

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
        # Reply threading — when the user replies-to a previous message
        # (typically one of the bot's own answers), surface the quoted
        # text to the model as context so "fix that line" / "translate
        # this" / "elaborate on point 2" type follow-ups land coherently
        # instead of getting interpreted against the whole prior session
        # transcript blindly.
        reply_to = getattr(msg, "reply_to_message", None)
        if reply_to is not None:
            quoted = (
                getattr(reply_to, "text", None)
                or getattr(reply_to, "caption", None)
                or ""
            )[:1500]
            if quoted:
                text = (
                    f"[The user is replying to this earlier message: "
                    f"{quoted!r}]\n\n{text}"
                )
        # Surface a transient STT status when the user sent a voice note —
        # transcription routinely takes 1-3 s on Whisper-local and the
        # user shouldn't be left staring at a silent chat. The message
        # is deleted as soon as ``_extract_files`` returns so the
        # following typing-indicator + reply flow stays clean.
        transcribing_msg = None
        if msg.voice is not None:
            try:
                transcribing_msg = await msg.reply_text("🎤 Trascrivo...")
            except Exception:
                transcribing_msg = None
        tmp = tempfile.mkdtemp(prefix="oa_tg_")
        try:
            extracted = await self._extract_files(msg, tmp)
            if transcribing_msg is not None:
                try:
                    await transcribing_msg.delete()
                except Exception:
                    pass
                transcribing_msg = None
            if extracted.text_addition:
                text = f"{text}\n{extracted.text_addition}" if text else extracted.text_addition

            if extracted.files_info:
                text = prepend_context_block(text, build_attachment_context(extracted.files_info))

            if not text:
                return

            # Strip ``@bot_username`` from the text when this is a
            # group-mention so the model doesn't see the boilerplate.
            if is_group and self._bot_username:
                mention_token = f"@{self._bot_username}"
                # Remove first occurrence case-insensitively. Manual
                # find rather than ``re.sub`` because we want to keep
                # surrounding punctuation as-is.
                lower = text.lower()
                idx = lower.find(mention_token)
                if idx != -1:
                    text = (text[:idx] + text[idx + len(mention_token):]).strip()
            await self.dispatch_turn(
                msg, session_id, text, voice_detected=extracted.voice_detected,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

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
        try:
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
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ── Platform primitives (consumed by BaseBridge.dispatch_turn) ──
    #
    # The shared orchestration is in ``BaseBridge.dispatch_turn``; here
    # we only translate between its calls and the python-telegram-bot
    # ``Message`` surface. Voice-mode mirroring (voice in → voice out)
    # is also handled by BaseBridge via ``maybe_prepend_voice_reply``,
    # so this file no longer carries any of that logic.

    async def post_status(self, msg, text: str):
        # Surface the agent's in-flight state as Telegram's native typing
        # action (matches Hermes' UX) instead of a "Thinking..." reply
        # message. The animator self-refreshes every 4s — Telegram clears
        # the typing indicator after ~5s of silence, so the loop period
        # has to undercut that. ``update_status`` becomes a no-op since
        # the typing dot replaces the spinner text entirely, and the agent
        # actual reply lands a few seconds later.
        try:
            animator = _TypingAnimator(msg.get_bot(), msg.chat_id)
            await animator.start()
            return animator
        except Exception:
            return None

    async def update_status(self, status_msg, text: str) -> None:
        # Intentional no-op: see ``post_status`` — tool-progress lines like
        # "Using bash..." would clutter the chat. The typing indicator alone
        # tells the user the agent is still working.
        return None

    async def clear_status(self, status_msg) -> None:
        if status_msg is None:
            return
        try:
            await status_msg.stop()
        except Exception:
            pass

    async def stream_text(self, status_msg, text: str) -> None:
        # Forward to the animator's streaming path. No-op when
        # streaming is disabled — the user then just sees the typing
        # indicator until the final reply lands as a single bubble,
        # which is the default the bridge ships with.
        if status_msg is None or not self._streaming_enabled:
            return
        try:
            await status_msg.stream_chunk(text)
        except Exception:
            pass

    async def ack_follower(self, msg) -> None:
        # Drop a 👀 reaction on follow-up messages that arrived while
        # the bot was still composing the prior turn. Without this the
        # user wonders why their second message is being ignored —
        # server-side coalescence is folding it in, but invisibly.
        try:
            from telegram import ReactionTypeEmoji
            await msg.set_reaction([ReactionTypeEmoji("👀")])
        except Exception:
            # Older clients / unsupported chat types just don't render
            # the reaction — non-fatal.
            pass

    async def send_choice_buttons(self, msg, options: list[str]) -> None:
        # Render the picker as a reply to ``msg`` so the buttons sit
        # underneath the agent's last chunk. Telegram caps inline
        # buttons at ~8 wide visually; we split into rows of 2 so the
        # labels stay readable on mobile. ``callback_data`` carries a
        # ``choice:<turn_id>:<value>`` payload that ``_on_callback``
        # routes back into ``dispatch_turn``.
        if not options:
            return
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        # turn_id is a coarse correlation key; we don't yet thread it
        # through the agent loop, so use the current message_id for
        # uniqueness and to help operators debug callbacks in the log.
        turn_id = str(getattr(msg, "message_id", "0"))
        rows: list[list] = []
        cur: list = []
        for opt in options[:8]:
            cur.append(InlineKeyboardButton(opt[:64], callback_data=f"choice:{turn_id}:{opt[:48]}"))
            if len(cur) == 2:
                rows.append(cur)
                cur = []
        if cur:
            rows.append(cur)
        try:
            await msg.reply_text("Scegli:", reply_markup=InlineKeyboardMarkup(rows))
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
