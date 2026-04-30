"""Base bridge — connects to the Gateway via WS and translates messages.

Subclasses implement platform-specific polling (Telegram, Discord, etc.)
and call `self.send_message()` / `self.send_command()` to route through
the Gateway. Responses arrive via `on_response()` / `on_status()` callbacks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Awaitable

from openagent.gateway import protocol as P

from openagent.core.logging import elog

logger = logging.getLogger(__name__)

# Retry cooldown between bridge crashes.
BRIDGE_RETRY_SECONDS = 30

# No per-turn timeout. A runaway or legitimately-long turn is ended by the
# user sending ``/stop`` (which routes to ``sessions.stop_current`` and cancels
# the in-flight asyncio task — see openagent/gateway/server.py), or by
# ``systemctl restart openagent``. Automatic "give up after N minutes" timeouts
# break long workflows like gradle assembleRelease, Electron builds, and
# Maestro suites that legitimately run an hour-plus.


def format_tool_status(raw: str) -> str:
    """Convert a raw status string (possibly JSON tool event) into a
    human-readable line suitable for Telegram/Discord/WhatsApp.

    Structured events look like: ``{"tool":"bash","status":"running",...}``
    Plain strings like ``"Thinking..."`` are returned unchanged.
    """
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or "tool" not in data:
            return raw
    except (json.JSONDecodeError, TypeError):
        return raw

    tool = data["tool"]
    status = data.get("status", "running")

    if status == "running":
        return f"Using {tool}..."
    if status == "error":
        err = data.get("error", "unknown error")
        return f"✗ {tool} failed: {err}"
    # done
    return f"✓ {tool} done"


class BaseBridge:
    """Abstract base for platform bridges."""

    name: str = "bridge"

    def __init__(self, gateway_url: str = "ws://localhost:8765/ws", gateway_token: str | None = None):
        self.gateway_url = gateway_url
        self.gateway_token = gateway_token
        self._ws = None
        self._ws_session = None  # aiohttp.ClientSession — must be closed
        self._http_session = None  # cached aiohttp.ClientSession for TTS/STT
        self._listener_task: asyncio.Task | None = None
        self._should_stop = False
        self._pending: dict[str, asyncio.Future] = {}  # session_id → response future
        self._command_future: asyncio.Future | None = None
        self._command_lock = asyncio.Lock()
        self._status_callbacks: dict[str, Callable] = {}  # session_id → on_status
        # Per-session DELTA frame callback. Populated by
        # ``send_message_streaming``; cleared on RESPONSE/ERROR alongside
        # ``_status_callbacks`` so a slow/disconnected callback can't
        # leak across sessions.
        self._delta_callbacks: dict[str, Callable] = {}  # session_id → on_delta
        self._session_locks: dict[str, asyncio.Lock] = {}  # per-session serialization

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
        orphaned = list(self._pending.items())
        self._pending.clear()
        self._status_callbacks.clear()
        self._delta_callbacks.clear()
        for sid, future in orphaned:
            if not future.done():
                future.set_result({"type": "error", "text": reason})
                logger.warning("Resolved orphaned future for %s: %s", sid, reason)
        if self._command_future and not self._command_future.done():
            self._command_future.set_result({"type": "error", "text": reason})
        self._command_future = None

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a per-session lock to serialize message sending."""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

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

        # Clean up stale state from any previous connection
        self._resolve_orphaned_futures("Reconnecting to gateway")
        self._session_locks.clear()

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
        """Listen for Gateway responses and dispatch to pending futures."""
        import aiohttp
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    t = data.get("type")
                    sid = data.get("session_id")

                    if t == P.STATUS and sid in self._status_callbacks:
                        try:
                            await self._status_callbacks[sid](data.get("text", ""))
                        except Exception:
                            pass
                    elif t == P.DELTA and sid in self._delta_callbacks:
                        # Streaming token chunk for the in-flight turn.
                        # Bridges that opted into streaming via
                        # ``send_message_streaming`` get progressive
                        # text; the trailing RESPONSE still resolves
                        # the pending future with the canonical text.
                        try:
                            await self._delta_callbacks[sid](data.get("text", ""))
                        except Exception:
                            pass
                    elif t == P.RESPONSE and sid in self._pending:
                        future = self._pending.pop(sid)
                        if not future.done():
                            future.set_result(data)
                        self._status_callbacks.pop(sid, None)
                        self._delta_callbacks.pop(sid, None)
                    elif t == P.ERROR:
                        # Errors may or may not have a session_id.  Try to
                        # route to the matching pending future; if no match,
                        # just log it.
                        if sid and sid in self._pending:
                            future = self._pending.pop(sid)
                            if not future.done():
                                future.set_result(data)
                            self._status_callbacks.pop(sid, None)
                            self._delta_callbacks.pop(sid, None)
                    elif t == P.COMMAND_RESULT and self._command_future and not self._command_future.done():
                        self._command_future.set_result(data)
                        self._command_future = None

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        finally:
            # Resolve any futures still waiting so callers don't hang forever
            self._resolve_orphaned_futures("Gateway connection lost")

    async def send_message(
        self,
        text: str,
        session_id: str,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        *,
        input_was_voice: bool = False,
    ) -> dict:
        """Send a message through the Gateway and wait for the response.

        Thin wrapper around :meth:`send_message_streaming` that doesn't
        request streaming deltas. Existing bridge code keeps working
        unchanged — only callers that opt-in via
        ``send_message_streaming`` get progressive text updates.
        """
        return await self.send_message_streaming(
            text, session_id,
            on_status=on_status,
            on_delta=None,
            input_was_voice=input_was_voice,
        )

    async def send_message_streaming(
        self,
        text: str,
        session_id: str,
        *,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        on_delta: Callable[[str], Awaitable[None]] | None = None,
        input_was_voice: bool = False,
    ) -> dict:
        """Send through the Gateway with optional streaming-delta callback.

        Per-session locking keeps only one message in-flight per user so a
        second concurrent message doesn't clobber ``_pending[session_id]``.
        The awaited future is resolved when the Gateway replies, cancelled
        when the user issues ``/stop`` (see ``sessions.stop_current``), or
        raised on gateway disconnect (``_resolve_orphaned_futures``). No
        wall-clock timeout — long tool calls are the point.

        ``on_delta`` is invoked once per ``delta`` WS frame as tokens
        arrive from the LLM. Bridges typically wrap it in a throttle
        (Telegram caps message edits at ~1/sec) and call ``edit_text``
        with the accumulated string. ``None`` (the default) skips
        streaming entirely and matches today's behaviour exactly.

        ``input_was_voice`` mirrors the modality: when True, the gateway
        invokes the streaming TTS pipeline so the reply is voice as well
        as text. Bridges that can't render audio inline (Telegram) still
        set this — the bridge handles synthesis itself by post-processing
        the response (see ``bridges/telegram.py``).
        """
        async with self._get_session_lock(session_id):
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            self._pending[session_id] = future
            if on_status:
                self._status_callbacks[session_id] = on_status
            if on_delta:
                self._delta_callbacks[session_id] = on_delta

            try:
                payload = {
                    "type": P.MESSAGE,
                    "text": text,
                    "session_id": session_id,
                }
                if input_was_voice:
                    payload["input_was_voice"] = True
                await self._send_gateway_json(payload)
            except Exception:
                self._pending.pop(session_id, None)
                self._status_callbacks.pop(session_id, None)
                self._delta_callbacks.pop(session_id, None)
                raise

            try:
                return await future
            finally:
                # Defensive cleanup — the normal path resolves _pending via
                # on_response(), but cancellation (e.g. /stop) unwinds here.
                self._pending.pop(session_id, None)
                self._status_callbacks.pop(session_id, None)
                self._delta_callbacks.pop(session_id, None)

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
        """
        import tempfile
        audio = await self.synthesise_voice_reply(text)
        if not audio:
            return None
        tmp = tempfile.mkdtemp(prefix=f"oa_{self.name}_tts_")
        mp3_path = f"{tmp}/reply.mp3"
        with open(mp3_path, "wb") as f:
            f.write(audio)
        return f"[VOICE:{mp3_path}]"

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
