"""DELTA frame plumbing — gateway emits, bridges silently drop.

The gateway streams text-mode replies via ``DELTA`` WS frames so the
universal client gets a typewriter UX. Bridges, on the other hand, run
in answer-response mode (Discord/Telegram/WhatsApp post the canonical
text once, no in-message editing), so the bridge-side machinery for
opting into per-token callbacks was retired in favour of a single
``send_message`` API.

Tests:

* ``P.DELTA`` constant is still exposed (clients import it by name).
* The gateway server.py text-mode path still emits DELTA payloads with
  the canonical ``{type, text, session_id}`` shape (regression smoke
  for the universal client).
* ``BaseBridge`` exposes a single ``send_message`` method — the dual
  ``send_message_streaming`` API is gone.
* ``BaseBridge`` has no ``_delta_callbacks`` field (no dead state).
* ``send_message`` resolves on RESPONSE and ignores DELTA frames
  silently.
"""
from __future__ import annotations

import asyncio
import json

from ._framework import TestContext, test


class _FakeBridge:
    """Subclass stand-in that skips the WS connect."""

    def __init__(self) -> None:
        from openagent.bridges.base import BaseBridge

        self._real = BaseBridge.__new__(BaseBridge)
        self._real.name = "fake"
        self._real._stream_opened = set()
        self._real._stream_pending = {}
        self._real._status_callbacks = {}
        self._real._ws = object()
        self.sent: list[dict] = []

        async def fake_send(payload: dict) -> None:
            self.sent.append(payload)

        self._real._send_gateway_json = fake_send  # type: ignore[assignment]

    async def feed_listener(self, frames: list[dict]) -> None:
        """Faithful copy of the post-cleanup ``BaseBridge._listen_gateway``
        dispatch — DELTA is a no-op, RESPONSE latches text onto the
        collector, TURN_COMPLETE releases the awaiter. Used to drive
        the bridge end-to-end without a real WebSocket."""
        from openagent.gateway import protocol as P

        for data in frames:
            t = data.get("type")
            sid = data.get("session_id")
            collector = self._real._stream_pending.get(sid) if sid else None

            if t == P.STATUS:
                cb = self._real._status_callbacks.get(sid)
                if cb is not None:
                    await cb(data.get("text", ""))
            elif t == P.DELTA:
                # Bridges ignore DELTA frames — they wait for the
                # canonical RESPONSE / TURN_COMPLETE. This branch must
                # NOT touch any per-session callback dict (none exists
                # for deltas).
                pass
            elif t == P.RESPONSE and collector is not None:
                collector.text = data.get("text", "") or ""
                collector.model = data.get("model")
                collector.attachments = list(data.get("attachments") or [])
            elif t == P.TURN_COMPLETE and collector is not None:
                collector.done.set()


@test("streaming", "P.DELTA constant is exposed by the protocol module")
async def t_protocol_delta_constant(_ctx: TestContext) -> None:
    """Sanity guard — clients import this constant by name; drift in
    the spelling silently breaks the wire format."""
    from openagent.gateway import protocol as P
    assert hasattr(P, "DELTA"), "protocol.py must expose DELTA"
    assert P.DELTA == "delta", f"DELTA should be 'delta', got {P.DELTA!r}"


@test("streaming", "stream wire codec maps OutTextDelta to {type=delta, text, session_id}")
async def t_stream_delta_payload_shape(_ctx: TestContext) -> None:
    """Verifies the wire shape the universal client + CLI receive for
    streaming token frames. The legacy ``TurnRunner`` was retired —
    every text-mode reply now flows through ``StreamSession`` which
    publishes ``OutTextDelta`` events and the wire codec serializes
    them to ``{type: "delta", text, session_id}``. The shape MUST
    stay stable so older clients keep typing-out tokens correctly."""
    from openagent.stream.events import OutTextDelta
    from openagent.stream.wire import event_to_wire

    payload = event_to_wire(OutTextDelta(
        session_id="s1", seq=3, ts_ms=42, text="hello",
    ))
    assert payload["type"] == "delta", payload
    assert payload["text"] == "hello", payload
    assert payload["session_id"] == "s1", payload
    # JSON sanity-check that nothing unparseable lurks in the payload.
    json.dumps(payload)


@test("streaming", "BaseBridge exposes a single send_message API")
async def t_basebridge_single_send_method(_ctx: TestContext) -> None:
    """The dual ``send_message_streaming`` API was retired when the
    progressive-delta UX was removed from Discord/Telegram. Guard
    against accidental re-introduction."""
    from openagent.bridges.base import BaseBridge
    assert hasattr(BaseBridge, "send_message"), "send_message must exist"
    assert not hasattr(BaseBridge, "send_message_streaming"), (
        "send_message_streaming was retired — bridges run in answer-"
        "response mode and don't consume DELTA frames"
    )


@test("streaming", "BaseBridge carries no _delta_callbacks state")
async def t_basebridge_no_delta_callbacks_field(_ctx: TestContext) -> None:
    """No bridge consumes per-delta callbacks now; the storage map is
    dead state and was removed."""
    from openagent.bridges.base import BaseBridge
    instance = BaseBridge.__new__(BaseBridge)
    BaseBridge.__init__(instance)  # type: ignore[misc]
    assert not hasattr(instance, "_delta_callbacks"), (
        "_delta_callbacks should not exist on BaseBridge instances"
    )


@test("streaming", "send_message resolves on TURN_COMPLETE and ignores DELTA frames")
async def t_send_message_ignores_delta(_ctx: TestContext) -> None:
    """End-to-end: a turn that emits DELTA frames before the trailing
    RESPONSE + TURN_COMPLETE must complete cleanly. The DELTAs are
    silently dropped; only the canonical RESPONSE text reaches the
    caller, and TURN_COMPLETE is what releases the awaiter."""
    fb = _FakeBridge()

    async def driver() -> None:
        for _ in range(500):
            if "s-final" in fb._real._stream_pending:
                break
            await asyncio.sleep(0.001)
        await fb.feed_listener([
            {"type": "delta", "session_id": "s-final", "text": "Hello "},
            {"type": "delta", "session_id": "s-final", "text": "world"},
            {"type": "delta", "session_id": "s-final", "text": "!"},
            {"type": "response", "session_id": "s-final", "text": "Hello world!"},
            {"type": "turn_complete", "session_id": "s-final"},
        ])

    result, _ = await asyncio.gather(
        fb._real.send_message("hi", "s-final"),
        driver(),
    )
    assert result["text"] == "Hello world!", result
    # Cleanup — the collector map and status-callback map must be empty
    # so a later turn against the same id starts fresh.
    assert "s-final" not in fb._real._stream_pending, fb._real._stream_pending
    assert "s-final" not in fb._real._status_callbacks, fb._real._status_callbacks
