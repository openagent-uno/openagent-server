"""Base bridge — connects to the Gateway via WS and translates messages.

Subclasses implement platform-specific polling (Telegram, Discord, etc.)
and call ``self.send_message()`` / ``self.send_command()`` to route
through the gateway. Each bridge user maps to one server-side
:class:`StreamSession` so coalescence + barge-in apply uniformly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Awaitable

from openagent.channels.base import (
    parse_response_markers,
    split_preserving_code_blocks,
)
from openagent.gateway import protocol as P

from openagent.core.logging import elog
from openagent.stream.collector import StreamCollector, fold_outbound_event
from openagent.stream.events import SessionOpen, TextFinal, now_ms
from openagent.stream.wire import event_to_wire, wire_to_event

logger = logging.getLogger(__name__)

# Retry cooldown between bridge crashes.
BRIDGE_RETRY_SECONDS = 30

# Single shared fallback message when STT can't transcribe an inbound
# voice note. Forks per bridge had cosmetically-different copy
# ("Voice message not transcribed.", "Voice not transcribed.", etc.) —
# unifying here so the user gets the same prompt across channels.
VOICE_FALLBACK = "[Voice message could not be transcribed. Ask the user to type it.]"

# No per-turn timeout. A runaway or legitimately-long turn is ended by the
# user sending ``/stop`` (which routes to ``sessions.stop_current`` and cancels
# the in-flight asyncio task — see openagent/gateway/server.py), or by
# ``systemctl restart openagent``. Automatic "give up after N minutes" timeouts
# break long workflows like gradle assembleRelease, Electron builds, and
# Maestro suites that legitimately run an hour-plus.

# Bridge users tend to pause longer between messages than webapp typers
# (mobile keyboards, voice-note recording, network round-trips), so the
# bridge default is more generous than the webapp's 500 ms. Voice notes
# flagged ``source="stt"`` bypass the window for instant barge-in.
BRIDGE_COALESCE_WINDOW_MS = 1500


def format_tool_status(raw: str) -> str:
    """Convert a raw status string (possibly JSON tool event) into a
    human-readable line suitable for Telegram/Discord/WhatsApp.

    Structured events look like: ``{"tool":"bash","status":"running",...}``
    Plain strings like ``"Thinking..."`` are returned unchanged.
    """
    from openagent.channels.base import parse_status_event
    evt = parse_status_event(raw)
    if evt is None:
        return raw
    if evt.status == "running":
        return f"Using {evt.tool}..."
    if evt.status == "error":
        return f"✗ {evt.tool} failed: {evt.error or 'unknown error'}"
    # done / anything else
    return f"✓ {evt.tool} done"


class BaseBridge:
    """Abstract base for platform bridges.

    Concrete bridges plug their platform's polling/event primitives into
    :meth:`_run` and override the small set of platform helpers below
    (``post_status``, ``update_status``, ``clear_status``,
    ``send_text_chunk``, ``send_attachment``); the shared turn
    orchestration lives in :meth:`dispatch_turn` so a fix to the
    spam-coalescence + voice-mirror flow lands in every bridge at once.
    """

    name: str = "bridge"

    # Per-platform message size limit used by ``dispatch_turn`` when it
    # splits the response. Overridden in subclasses.
    message_limit: int = 4000

    def __init__(self, gateway_url: str = "ws://localhost:8765/ws", gateway_token: str | None = None):
        self.gateway_url = gateway_url
        self.gateway_token = gateway_token
        self._ws = None
        self._ws_session = None  # aiohttp.ClientSession — must be closed
        self._http_session = None  # cached aiohttp.ClientSession for TTS/STT
        self._listener_task: asyncio.Task | None = None
        self._should_stop = False
        self._command_future: asyncio.Future | None = None
        self._command_lock = asyncio.Lock()
        # NOTE: there is no ``_delta_callbacks`` field. Bridges run in
        # answer-response mode — the gateway streams deltas server-side
        # (the web app consumes them), but bridges only forward the
        # final ``RESPONSE`` text. If a future bridge wants progressive
        # editing it can subclass and tap the WS directly; we don't
        # carry dead infrastructure for a hypothetical caller.
        # ``_stream_opened`` tracks which session_ids have already
        # been ``session_open``'d on the current WS — wiped on reconnect
        # since the gateway tears down server-side sessions on WS drop.
        # ``_stream_pending`` maps session_id → in-flight collector;
        # the listener writes events into it via ``fold_outbound_event``,
        # ``send_message`` awaits ``collector.done``. The tool-status
        # callback lives ON the collector so it dies with the collector
        # (no stale per-session callback dict to keep in sync).
        self._stream_opened: set[str] = set()
        self._stream_pending: dict[str, StreamCollector] = {}

    async def start(self) -> None:
        """Connect to Gateway and start the platform polling loop with retry."""
        self._should_stop = False
        elog("bridge.start", name=self.name)
        while not self._should_stop:
            try:
                await self._connect_gateway()
                await self._run()
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._should_stop:
                    break
                elog("bridge.error", level="error", name=self.name, error=str(e), retry_in=BRIDGE_RETRY_SECONDS)
                await asyncio.sleep(BRIDGE_RETRY_SECONDS)

    async def stop(self) -> None:
        elog("bridge.stop", name=self.name)
        self._should_stop = True
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except (asyncio.CancelledError, Exception):
                pass
            self._listener_task = None
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._ws_session:
            await self._ws_session.close()
            self._ws_session = None
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

    def _resolve_orphaned_futures(self, reason: str) -> None:
        """Resolve all pending futures with an error so callers don't hang."""
        # Drain any in-flight stream collectors with an error and unblock
        # the awaiters.
        orphaned_streams = list(self._stream_pending.items())
        self._stream_pending.clear()
        for sid, collector in orphaned_streams:
            collector.errored = True
            collector.error_text = reason
            collector.done.set()
            logger.warning("Resolved orphaned stream for %s: %s", sid, reason)
        # The gateway tears down the server-side StreamSessions on the
        # WS drop too, so any cached "we already opened it" bookkeeping
        # is stale — wipe it so the next message re-sends session_open.
        self._stream_opened.clear()
        if self._command_future and not self._command_future.done():
            self._command_future.set_result({"type": "error", "text": reason})
        self._command_future = None

    async def _send_gateway_json(self, payload: dict) -> None:
        """Write to the gateway websocket, tolerating reconnect races."""
        if self._ws is None or getattr(self._ws, "closed", False):
            raise ConnectionError("Gateway websocket is not connected")
        try:
            await self._ws.send_json(payload)
        except Exception as e:
            if "closing transport" in str(e).lower():
                raise ConnectionError("Gateway websocket is closing") from e
            raise

    @staticmethod
    def append_model_feedback(text: str, model: str | None) -> str:
        """Append a compact model footer to a response body."""
        if not model:
            return text
        footer = f"Model: {model}"
        return f"{text}\n\n{footer}" if text else footer

    async def _connect_gateway(self) -> None:
        """Connect to the Gateway WebSocket and authenticate."""
        import aiohttp

        # Clean up stale state from any previous connection — the
        # gateway tears down the server-side StreamSessions on the WS
        # drop, so any cached "we already opened it" bookkeeping in
        # ``_resolve_orphaned_futures`` is also wiped.
        self._resolve_orphaned_futures("Reconnecting to gateway")

        # Close any previous session/ws from a prior connection attempt
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._ws_session:
            await self._ws_session.close()
            self._ws_session = None

        session = aiohttp.ClientSession()
        self._ws_session = session
        self._ws = await session.ws_connect(self.gateway_url)

        # Authenticate
        auth_msg = {"type": P.AUTH, "token": self.gateway_token or "", "client_id": f"bridge:{self.name}"}
        await self._send_gateway_json(auth_msg)

        # Wait for auth response
        resp = await self._ws.receive_json()
        if resp.get("type") == P.AUTH_ERROR:
            raise ConnectionError(f"Gateway auth failed: {resp.get('reason')}")
        logger.info("%s bridge connected to Gateway", self.name)

        # Start response listener — store the task so exceptions are not lost
        self._listener_task = asyncio.create_task(
            self._listen_gateway(), name=f"{self.name}:gw-listener"
        )

    async def _listen_gateway(self) -> None:
        """Listen for Gateway responses and dispatch to pending collectors."""
        import aiohttp
        exit_kind = "normal"
        last_msg_type: str | None = None
        try:
            async for msg in self._ws:
                last_msg_type = msg.type.name if hasattr(msg.type, "name") else str(msg.type)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_gateway_frame(json.loads(msg.data))
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    exit_kind = last_msg_type or "closed_or_error"
                    break
        except Exception as e:  # noqa: BLE001
            # Capturing here (not just relying on the silent ``finally``) so
            # the next reproduction has the exception type + traceback. The
            # bridge's ``start()`` retry loop only sees the orphan-future
            # reason string, which strips the exception type and frame info
            # — that's how the v0.12.50+ deflate-error class went undiagnosed
            # for so long: the WS died, futures were orphaned, no one logged
            # what actually killed the listener.
            exit_kind = f"exception:{type(e).__name__}"
            logger.exception(
                "%s bridge gateway listener died (last_msg_type=%s)",
                self.name, last_msg_type,
            )
            elog(
                "bridge.listener_died", level="error", name=self.name,
                error_type=type(e).__name__, error=str(e),
                last_msg_type=last_msg_type,
            )
        finally:
            elog("bridge.listener_exit", name=self.name, exit_kind=exit_kind)
            self._resolve_orphaned_futures("Gateway connection lost")

    async def _handle_gateway_frame(self, data: dict) -> None:
        """Route a single decoded WS frame.

        Stream events go through ``wire_to_event`` + ``fold_outbound_event``
        so the per-bridge code stays in lockstep with the wire codec
        (anything new added to ``OutTextFinal`` / ``OutAudio*`` lands
        here automatically). Side-channels (``status`` for tool pings,
        ``command_result`` for command futures) bypass the typed event
        path because they're not part of a turn's outbound stream.
        """
        t = data.get("type")
        sid = data.get("session_id")
        collector = self._stream_pending.get(sid) if sid else None

        if t == P.STATUS:
            cb = collector.on_status if collector is not None else None
            if cb is not None:
                try:
                    await cb(data.get("text", ""))
                except Exception:
                    pass
            return

        if t == P.COMMAND_RESULT:
            if self._command_future and not self._command_future.done():
                self._command_future.set_result(data)
                self._command_future = None
            return

        if t == P.ERROR and collector is None:
            # Bare gateway errors (auth, handshake, no session attached)
            # would otherwise vanish — surface them so the operator sees
            # the root cause in logs.
            logger.warning(
                "%s: gateway error (no session): %s",
                self.name, data.get("text"),
            )
            return

        evt = wire_to_event(data)
        if evt is None or collector is None:
            return
        if fold_outbound_event(collector, evt):
            collector.done.set()

    async def send_message(
        self,
        text: str,
        session_id: str,
        *,
        target: Any = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        input_was_voice: bool = False,
        source: str = "user_typed",
    ) -> dict:
        """Push ``text`` into the user's stream session and await the reply.

        Each ``session_id`` maps to one server-side
        :class:`StreamSession`. Concurrency is handled by ownership: the
        first concurrent caller for a session creates the collector and
        awaits the reply (the OWNER); subsequent callers send their
        ``text_final`` so the gateway folds them into the same merged
        turn, then return a ``{"type":"duplicate"}`` sentinel so the
        bridge skips posting a redundant response. Without this, three
        concurrent messages would each create their own collector,
        overwriting ``_stream_pending[session_id]`` — the first two
        ``send_message`` calls would hang forever waiting for a
        ``turn_complete`` that's now routed to the third caller's
        collector. Voice notes (``source="stt"``) still bypass server
        coalescence for instant barge-in.

        ``target`` is the platform-specific reply anchor (Telegram
        ``Message`` etc.). Both owner and followers stash it on the
        owner's collector so the merged reply gets posted against the
        LATEST message — without this, the bot visibly replies to the
        FIRST message in the burst, looking like it's "answering an
        old message".
        """
        if session_id not in self._stream_opened:
            await self._send_gateway_json(event_to_wire(SessionOpen(
                session_id=session_id,
                ts_ms=now_ms(),
                profile="batched",
                client_kind=self.name,
                coalesce_window_ms=BRIDGE_COALESCE_WINDOW_MS,
            )))
            self._stream_opened.add(session_id)

        # Atomic ownership check — synchronous, no awaits between read
        # and write so two concurrent tasks can't both win. A collector
        # whose ``done`` is already set is in its post-turn cleanup
        # window; treat it as no-owner so a fresh message starts a new
        # turn instead of latching onto a corpse and overwriting the
        # owner's ``latest_target`` after it's been read.
        existing = self._stream_pending.get(session_id)
        is_owner = (existing is None) or existing.done.is_set()
        if is_owner:
            collector = StreamCollector()
            collector.on_status = on_status
            self._stream_pending[session_id] = collector
        else:
            collector = existing

        # Both owner and followers contribute their target so the owner
        # posts the merged reply against the most-recent message — see
        # docstring rationale. Skip once the collector has fired its
        # ``done`` event so a late follower can't poison the OWNER's
        # already-finalised target read.
        if target is not None and not collector.done.is_set():
            collector.latest_target = target

        try:
            await self._send_gateway_json(event_to_wire(TextFinal(
                session_id=session_id,
                ts_ms=now_ms(),
                text=text,
                source=source,  # type: ignore[arg-type]
            )))
        except Exception:
            if is_owner and self._stream_pending.get(session_id) is collector:
                self._stream_pending.pop(session_id, None)
            raise

        if not is_owner:
            # Follower: text_final has been pushed into the gateway's
            # coalescence buffer. The owner's collector will receive the
            # merged response. Return immediately so the bridge skips
            # posting a duplicate reply.
            return {
                "type": "duplicate",
                "text": "",
                "model": None,
                "attachments": [],
                "target": None,
            }

        try:
            await collector.done.wait()
            return collector.to_legacy_reply()
        finally:
            # Owner cleanup — but only if the slot still holds OUR
            # collector (a brand-new turn could have replaced it after
            # our ``done`` fired).
            if self._stream_pending.get(session_id) is collector:
                self._stream_pending.pop(session_id, None)

    # ── Platform primitives (override per bridge) ──────────────────
    #
    # ``dispatch_turn`` calls these to translate between the shared
    # orchestration and the platform's send/edit primitives. Defaults are
    # no-ops so a bridge that genuinely can't do something (e.g. WhatsApp
    # has no edit API) doesn't have to think about it — the abstract
    # methods are ``send_text_chunk`` and ``send_attachment``, which
    # every bridge MUST override.

    async def post_status(self, target, text: str):
        """Post the initial "Thinking..." status. Return an opaque
        handle (anything the bridge wants — message object, chat_id, …)
        that ``update_status`` / ``clear_status`` will receive back, or
        ``None`` if the bridge has no status surface."""
        return None

    async def update_status(self, handle, text: str) -> None:
        """Update the in-flight status with a new line (e.g. "Using bash…").
        Default: no-op (bridges with no edit API can override to
        post a throttled new message instead)."""
        return None

    async def clear_status(self, handle) -> None:
        """Remove the status indicator once the turn is done. Default:
        no-op (WhatsApp can't delete messages, so the throttled status
        line just stays in the chat)."""
        return None

    async def send_text_chunk(self, target, chunk: str) -> None:
        """Send one already-split chunk of response text. Subclasses
        own platform formatting (HTML render + fallback for Telegram,
        whatsapp markdown, etc.) inside this method."""
        raise NotImplementedError

    async def send_attachment(self, target, att) -> None:
        """Send one ``Attachment`` (image, voice, video, file) to the
        platform. Subclasses dispatch by ``att.type`` as needed."""
        raise NotImplementedError

    async def dispatch_turn(
        self,
        target,
        session_id: str,
        text: str,
        *,
        voice_detected: bool = False,
    ) -> None:
        """Shared orchestration: post status → send_message → render reply.

        Lives here (not in each bridge) so the spam-coalescence
        sentinel, voice-modality mirror, attachment splitting, and
        message-limit chunking all stay in lockstep across Telegram,
        Discord, and WhatsApp. A subclass usually only needs to fill in
        the platform primitives above; the orchestration "just works".
        """
        try:
            status_handle = await self.post_status(target, "Thinking...")
        except Exception as e:  # noqa: BLE001
            logger.debug("%s: post_status failed: %s", self.name, e)
            status_handle = None

        async def on_status(raw: str) -> None:
            if status_handle is None:
                return
            try:
                await self.update_status(status_handle, format_tool_status(raw))
            except Exception:
                pass

        try:
            response = await self.send_message(
                text, session_id,
                target=target,
                on_status=on_status,
                # Voice notes bypass the typed-burst coalescence window
                # for instant barge-in (StreamSession STT-bypass path).
                source="stt" if voice_detected else "user_typed",
            )
        finally:
            if status_handle is not None:
                try:
                    await self.clear_status(status_handle)
                except Exception:
                    pass

        # Concurrent message in the same burst — the owner posts the
        # merged reply; followers exit so the user sees ONE response.
        if response.get("type") == "duplicate":
            return

        # Anchor the merged reply to the LATEST message the burst saw,
        # not the FIRST. Without this, a 5-message spam visibly replies
        # to bubble #1 — looks like the bot is "answering an old
        # message". send_message stashed each follower's target on the
        # collector; the owner's reply now reads back the latest.
        post_target = response.get("target") or target

        response_text = response.get("text", "") or ""
        response_text = await self.maybe_prepend_voice_reply(response_text, voice_detected)

        clean, attachments = parse_response_markers(response_text)
        for att in attachments:
            try:
                await self.send_attachment(post_target, att)
            except Exception as e:  # noqa: BLE001
                logger.error("%s attachment send error: %s", self.name, e)
            finally:
                self._cleanup_owned_temp_artifact(att.path)

        clean = self.append_model_feedback(clean, response.get("model"))
        await self.send_response_text(post_target, clean)

    @staticmethod
    def _cleanup_owned_temp_artifact(path: str | None) -> None:
        """Best-effort unlink of bridge-owned temp files.

        We only remove files we can confidently identify as OpenAgent's
        own ephemeral artifacts: direct children of the OS temp dir whose
        basename starts with ``oa_``. This covers voice replies, stream
        snapshots, uploads, and similar per-turn scratch files while
        avoiding accidental deletion of arbitrary user files the agent may
        have intentionally attached by local path.
        """
        if not path:
            return
        try:
            temp_root = Path(tempfile.gettempdir()).resolve()
            candidate = Path(path).resolve()
        except Exception:
            return
        if candidate.parent != temp_root:
            return
        if not candidate.name.startswith("oa_"):
            return
        try:
            if candidate.is_file() or candidate.is_symlink():
                os.unlink(candidate)
        except FileNotFoundError:
            return
        except OSError as e:
            logger.debug("bridge temp cleanup skipped for %s: %s", candidate, e)

    async def send_response_text(self, target, text: str) -> None:
        """Split ``text`` at ``message_limit`` and send each chunk.

        Shared by ``dispatch_turn`` and command handlers (``/clear``,
        ``/help``) so a long command result obeys the same chunking +
        per-platform rendering as a normal reply.
        """
        if not text:
            return
        for chunk in split_preserving_code_blocks(text, self.message_limit):
            try:
                await self.send_text_chunk(target, chunk)
            except Exception as e:  # noqa: BLE001
                logger.error("%s text send error: %s", self.name, e)

    # ── Voice helpers (shared by every bridge) ──────────────────────

    def _http_base(self) -> str:
        """Map ``ws://host:port/ws`` → ``http://host:port`` (or wss/https)."""
        gw = self.gateway_url or ""
        scheme_map = {"ws://": "http://", "wss://": "https://"}
        for ws_prefix, http_prefix in scheme_map.items():
            if gw.startswith(ws_prefix):
                base = http_prefix + gw[len(ws_prefix):]
                break
        else:
            base = gw
        return base[:-3] if base.endswith("/ws") else base

    async def _audio_session(self):
        """Long-lived ``aiohttp.ClientSession`` for TTS/STT round-trips.

        Created lazily (so importing this module doesn't require aiohttp)
        and reused across calls. ``stop()`` closes it.
        """
        if self._http_session is None:
            try:
                import aiohttp
            except ImportError:
                return None
            self._http_session = aiohttp.ClientSession()
        return self._http_session

    def _auth_headers(self, content_type: str | None = None) -> dict[str, str]:
        h: dict[str, str] = {}
        if content_type:
            h["Content-Type"] = content_type
        if self.gateway_token:
            h["Authorization"] = f"Bearer {self.gateway_token}"
        return h

    async def transcribe_via_gateway(self, file_path: str) -> str | None:
        """POST audio to ``/api/stt/transcribe``; ``None`` on any failure."""
        import aiohttp
        from pathlib import Path as _Path

        session = await self._audio_session()
        if session is None:
            return None
        url = f"{self._http_base()}/api/stt/transcribe"
        try:
            with open(file_path, "rb") as fh:
                form = aiohttp.FormData()
                form.add_field(
                    "file", fh,
                    filename=_Path(file_path).name,
                    content_type="audio/ogg",
                )
                async with session.post(
                    url, data=form, headers=self._auth_headers(),
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        return None
                    body = await resp.json()
                    return (body.get("text") or "").strip() or None
        except FileNotFoundError:
            return None
        except Exception as e:  # noqa: BLE001
            logger.debug("%s.stt.http_exc: %s", self.name, e)
            return None

    async def transcribe_with_fallback(self, file_path: str) -> str:
        """Transcribe ``file_path``: gateway STT first, local Whisper
        fallback, ``VOICE_FALLBACK`` if both produce nothing.

        Used by every bridge for inbound voice notes — keeps the
        gateway-vs-local routing logic in one place.
        """
        from openagent.channels.voice import transcribe as transcribe_local
        text = await self.transcribe_via_gateway(file_path)
        if not text:
            text = await transcribe_local(file_path)
        return text or VOICE_FALLBACK

    async def synthesise_voice_reply(self, text: str) -> bytes | None:
        """POST text to ``/api/tts/synthesize`` and return the audio bytes."""
        import aiohttp

        session = await self._audio_session()
        if session is None:
            return None
        url = f"{self._http_base()}/api/tts/synthesize"
        try:
            async with session.post(
                url, json={"text": text},
                headers=self._auth_headers("application/json"),
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    return None
                return await resp.read()
        except Exception as e:  # noqa: BLE001
            logger.debug("%s.tts.http_exc: %s", self.name, e)
            return None

    async def synthesise_audio_attachment(self, text: str) -> str | None:
        """Synthesize ``text`` to MP3 on disk and return ``[VOICE:/path]``.

        We send the LiteLLM-default MP3 directly — every bridge (Telegram
        ``sendAudio`` / ``sendVoice``, Discord ``File``, WhatsApp
        ``sendFileByUpload``) accepts it. Skipping the OGG/Opus
        transcode keeps the installer free of ffmpeg (~30 MB) and
        removes a native binary from the dependency list.

        Trade-off: Telegram renders MP3 via ``sendVoice`` as a generic
        audio file when the bytes aren't OGG/Opus, so the message
        appears in the music-player UI rather than the native voice-note
        bubble. Acceptable for v1; switch to OGG-Opus output via
        LiteLLM's ``response_format='opus'`` and a tiny pure-Python OGG
        muxer if the voice-note UI matters later.

        ``NamedTemporaryFile(delete=False)`` is used so the bridge can
        finish uploading the file to the platform before deletion —
        the platform sender unlinks it via ``parse_response_markers``-
        driven cleanup once the message lands. (Earlier code used
        ``mkdtemp`` per call with no cleanup, leaking one directory per
        voice reply.)
        """
        import tempfile
        audio = await self.synthesise_voice_reply(text)
        if not audio:
            return None
        with tempfile.NamedTemporaryFile(
            prefix=f"oa_{self.name}_tts_", suffix=".mp3", delete=False,
        ) as fh:
            fh.write(audio)
            mp3_path = fh.name
        return f"[VOICE:{mp3_path}]"

    async def maybe_prepend_voice_reply(
        self, text: str, voice_detected: bool,
    ) -> str:
        """Mirror modality: voice-in → voice-out attachment.

        When the inbound message was voice and we have a non-empty text
        reply, synthesize the reply to MP3 and prepend the
        ``[VOICE:/path]`` marker. Errors during synthesis are logged
        and swallowed — the user still gets the text reply. Used by
        every bridge so the voice-out logic lives in one place.
        """
        if not (voice_detected and text):
            return text
        try:
            voice_marker = await self.synthesise_audio_attachment(text)
        except Exception as e:  # noqa: BLE001 — never drop the text on synth failure
            elog(
                f"{self.name}.tts.error",
                level="warning",
                error_type=type(e).__name__,
                error=str(e) or repr(e),
            )
            return text
        if not voice_marker:
            return text
        return f"{voice_marker}\n{text}"

    async def send_command(self, name: str, session_id: str | None = None) -> str:
        """Send a command and wait for the result.

        ``session_id`` is forwarded to the gateway so scope-sensitive
        commands (``stop``, ``clear``, ``new``, ``reset``) can be limited to
        the specific bridge user who issued them. Bridges that multiplex
        many users onto a single ``client_id`` (telegram, discord) MUST
        pass the user's session_id here; otherwise a ``/clear`` from one
        user wipes everyone else's conversation.
        """
        async with self._command_lock:
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            self._command_future = future
            payload: dict = {"type": P.COMMAND, "name": name}
            if session_id is not None:
                payload["session_id"] = session_id
            try:
                await self._send_gateway_json(payload)
            except Exception:
                if self._command_future is future:
                    self._command_future = None
                raise
            try:
                result = await future
            finally:
                if self._command_future is future:
                    self._command_future = None
            return result.get("text", "")

    async def _run(self) -> None:
        """Platform-specific polling loop. Override in subclass."""
        raise NotImplementedError
