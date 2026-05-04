"""Channel profiles — :class:`RealtimeChannel` and :class:`BatchedChannel`.

Channels bridge a client transport (gateway WebSocket, bridge polling
loop, CLI client) to a :class:`StreamSession`. They differ in HOW
events flow, not in WHAT events they speak — every channel uses the
same typed event vocabulary.

* :class:`RealtimeChannel` ferries inbound and outbound events through
  transparently. Inbound wire frames decode via :func:`wire_to_event`
  and push into ``session.inbound``; outbound events from
  ``session.outbound`` encode via :func:`event_to_wire` and ship over
  the transport. Used by the universal app's gateway WS path.
* :class:`BatchedChannel` collects an inbound stream into one
  :class:`TextFinal` (with optional attachments), drains the outbound
  stream until :class:`TurnComplete`, and returns a
  :class:`BatchedReply` shaped exactly like the legacy
  ``BaseBridge.send_message`` return value. Used by the
  telegram/discord/whatsapp bridges, whose transports are intrinsically
  request/response.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from openagent.stream.events import (
    OutAudioChunk,
    OutAudioEnd,
    OutAudioStart,
    OutError,
    OutTextDelta,
    OutTextFinal,
    OutToolStatus,
    TextFinal,
    TurnComplete,
    now_ms,
)
from openagent.stream.session import StreamSession
from openagent.stream.wire import event_to_wire, wire_to_event

logger = logging.getLogger(__name__)


@dataclass
class BatchedReply:
    """Result of a :meth:`BatchedChannel.run_one_shot` call.

    Mirrors the shape of legacy ``TurnRunner.run`` return value so
    bridges can keep their existing render path.
    """

    text: str = ""
    audio_chunks: list[bytes] = field(default_factory=list)
    audio_format: str | None = None
    audio_mime: str | None = None
    voice_id: str | None = None
    attachments: list[dict] = field(default_factory=list)
    model: str | None = None
    errored: bool = False
    error_text: str | None = None

    @property
    def audio_bytes(self) -> bytes | None:
        if not self.audio_chunks:
            return None
        return b"".join(self.audio_chunks)


class RealtimeChannel:
    """Pass-through channel for full-duplex transports.

    The send callable returns ``True`` when the frame reached the
    transport and ``False`` when the transport is closed (the gateway's
    ``_safe_ws_send_json`` follows that contract). On ``False`` the pump
    holds the frame and retries — without that, a WS reconnect mid-turn
    would silently drop ``OutTextFinal`` / ``TurnComplete`` and leave the
    UI stuck on ``Thinking…``. After ``unrecoverable_after_s`` of failed
    sends we surrender the frame and fire ``on_unrecoverable`` so the
    gateway can reap the orphaned session.
    """

    profile = "realtime"

    UNRECOVERABLE_AFTER_S = 60.0
    RETRY_INTERVAL_S = 0.5

    def __init__(
        self,
        session: StreamSession,
        send_wire: Callable[[dict], Awaitable[bool]],
        *,
        on_unrecoverable: Callable[[], Awaitable[None]] | None = None,
    ):
        self._session = session
        self._send = send_wire
        self._pump_task: asyncio.Task | None = None
        self._on_unrecoverable = on_unrecoverable

    def rebind(self, send_wire: Callable[[dict], Awaitable[bool]]) -> None:
        """Swap the send target — used when the WS reconnects.

        Atomic on the asyncio loop: the pump rereads ``self._send`` on
        every retry, so a rebind during a stuck send completes the
        delivery on the new transport without losing the frame.
        """
        self._send = send_wire

    async def start(self) -> None:
        if self._pump_task is None:
            self._pump_task = asyncio.create_task(
                self._pump_outbound(),
                name=f"realtime-out:{self._session.session_id}",
            )

    async def on_wire(self, frame: dict) -> None:
        """Decode an inbound wire frame and push it into the session."""
        evt = wire_to_event(frame)
        if evt is None:
            return
        await self._session.push_in(evt)

    async def _pump_outbound(self) -> None:
        try:
            while True:
                evt = await self._session.outbound.get()
                try:
                    payload = event_to_wire(evt)
                except TypeError as e:
                    logger.debug("realtime: drop unwireable event: %s", e)
                    continue
                deadline = time.monotonic() + self.UNRECOVERABLE_AFTER_S
                while True:
                    try:
                        ok = await self._send(payload)
                    except Exception as e:  # noqa: BLE001
                        logger.debug("realtime: send raised: %s", e)
                        ok = False
                    if ok:
                        break
                    if time.monotonic() >= deadline:
                        logger.warning(
                            "realtime: dropping frame, no live ws for %.0fs"
                            " (session=%s)",
                            self.UNRECOVERABLE_AFTER_S,
                            self._session.session_id,
                        )
                        if self._on_unrecoverable is not None:
                            try:
                                await self._on_unrecoverable()
                            except Exception as e:  # noqa: BLE001
                                logger.debug(
                                    "realtime: on_unrecoverable raised: %s", e
                                )
                        return
                    await asyncio.sleep(self.RETRY_INTERVAL_S)
        except asyncio.CancelledError:
            return

    async def close(self) -> None:
        if self._pump_task is not None and not self._pump_task.done():
            self._pump_task.cancel()
            try:
                await self._pump_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._session.close()


class BatchedChannel:
    """Collect-once / send-once channel for bridge-style transports."""

    profile = "batched"

    def __init__(self, session: StreamSession):
        self._session = session

    async def run_one_shot(
        self,
        text: str,
        *,
        attachments: list[dict] | None = None,
        source: str = "user_typed",
    ) -> BatchedReply:
        """Push one user message and drain the outbound stream once."""
        sid = self._session.session_id
        msg = TextFinal(
            session_id=sid,
            seq=self._session.next_seq(),
            ts_ms=now_ms(),
            text=text,
            source=source,  # type: ignore[arg-type]
            attachments=tuple(attachments or ()),
        )
        await self._session.push_in(msg)

        reply = BatchedReply()
        text_parts: list[str] = []
        audio_started = False

        while True:
            evt = await self._session.outbound.get()
            if isinstance(evt, OutTextDelta):
                text_parts.append(evt.text)
            elif isinstance(evt, OutTextFinal):
                reply.text = evt.text
                reply.attachments = list(evt.attachments)
                reply.model = evt.model
            elif isinstance(evt, OutAudioStart):
                audio_started = True
                reply.audio_format = evt.format
                reply.audio_mime = evt.mime
                reply.voice_id = evt.voice_id
            elif isinstance(evt, OutAudioChunk):
                if audio_started:
                    reply.audio_chunks.append(evt.data)
            elif isinstance(evt, (OutAudioEnd, OutToolStatus)):
                # Audio span end: marker only; chunks already buffered above.
                # Tool status: bridges render via on_status side-channel; the
                # event is also still queued for channels that want it.
                pass
            elif isinstance(evt, OutError):
                reply.errored = True
                reply.error_text = evt.text
            elif isinstance(evt, TurnComplete):
                break

        # Prefer the canonical OutTextFinal text when present; fall back
        # to the accumulated deltas (some agents skip the final marker).
        if not reply.text and text_parts:
            reply.text = "".join(text_parts)
        return reply

    async def close(self) -> None:
        await self._session.close()


__all__ = ["BatchedReply", "RealtimeChannel", "BatchedChannel"]
