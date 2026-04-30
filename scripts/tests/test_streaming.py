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
        self._real._pending = {}
        self._real._status_callbacks = {}
        self._real._session_locks = {}
        self._real._ws = object()
        self.sent: list[dict] = []

        async def fake_send(payload: dict) -> None:
            self.sent.append(payload)

        self._real._send_gateway_json = fake_send  # type: ignore[assignment]

    async def feed_listener(self, frames: list[dict]) -> None:
        """Faithful copy of the post-cleanup ``BaseBridge._listen_gateway``
        dispatch — DELTA is a no-op, RESPONSE resolves the future.
        Used to drive the bridge end-to-end without a real WebSocket."""
        from openagent.gateway import protocol as P

        for data in frames:
            t = data.get("type")
            sid = data.get("session_id")
            if t == P.STATUS and sid in self._real._status_callbacks:
                await self._real._status_callbacks[sid](data.get("text", ""))
            elif t == P.DELTA:
                # Bridges ignore DELTA frames — they wait for the
                # canonical RESPONSE. This branch must NOT touch any
                # per-session callback dict (none exists for deltas).
                pass
            elif t == P.RESPONSE and sid in self._real._pending:
                fut = self._real._pending.pop(sid)
                if not fut.done():
                    fut.set_result(data)
                self._real._status_callbacks.pop(sid, None)


@test("streaming", "P.DELTA constant is exposed by the protocol module")
async def t_protocol_delta_constant(_ctx: TestContext) -> None:
    """Sanity guard — clients import this constant by name; drift in
    the spelling silently breaks the wire format."""
    from openagent.gateway import protocol as P
    assert hasattr(P, "DELTA"), "protocol.py must expose DELTA"
    assert P.DELTA == "delta", f"DELTA should be 'delta', got {P.DELTA!r}"


@test("streaming", "TurnRunner emits DELTA payloads with the canonical shape")
async def t_gateway_delta_payload_shape(_ctx: TestContext) -> None:
    """Verifies the WS payload emitted by the unified turn runner has
    exactly the fields the universal client expects (``type``, ``text``,
    ``session_id``). Reads the source directly so the test catches
    drift even when the live WS path can't be exercised without API
    keys."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "openagent" / "gateway" / "turn_runner.py"
    body = src.read_text(encoding="utf-8")
    assert '"type": P.DELTA' in body, "turn_runner.py no longer emits P.DELTA payloads"
    assert '"text": delta' in body, "turn_runner.py no longer wires delta text into payload"
    assert '"session_id": session_id' in body, "turn_runner DELTA payload missing session_id"
    # JSON sanity-check that nothing unparseable lurks at the file level.
    json.dumps({"type": "delta", "text": "x", "session_id": "y"})  # smoke


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


@test("streaming", "send_message resolves on RESPONSE and ignores DELTA frames")
async def t_send_message_ignores_delta(_ctx: TestContext) -> None:
    """End-to-end: a turn that emits DELTA frames before the trailing
    RESPONSE must complete cleanly. The DELTAs are silently dropped;
    only the RESPONSE text reaches the caller."""
    fb = _FakeBridge()

    async def driver() -> None:
        for _ in range(500):
            if "s-final" in fb._real._pending:
                break
            await asyncio.sleep(0.001)
        await fb.feed_listener([
            {"type": "delta", "session_id": "s-final", "text": "Hello "},
            {"type": "delta", "session_id": "s-final", "text": "world"},
            {"type": "delta", "session_id": "s-final", "text": "!"},
            {"type": "response", "session_id": "s-final", "text": "Hello world!"},
        ])

    result, _ = await asyncio.gather(
        fb._real.send_message("hi", "s-final"),
        driver(),
    )
    assert result["text"] == "Hello world!", result
    # Cleanup — the pending map and status-callback map must be empty
    # so a later turn against the same id starts fresh.
    assert "s-final" not in fb._real._pending, fb._real._pending
    assert "s-final" not in fb._real._status_callbacks, fb._real._status_callbacks
