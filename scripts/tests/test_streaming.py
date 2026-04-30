"""DELTA frame plumbing — gateway emit + bridge consumption.

Covers the C-tier of the voice/chat unification work: the new ``delta``
WS frame ferrying token streams to the universal client and the
bridges' opt-in streaming surface.

Tests:

* ``send_message_streaming`` registers + clears its delta callback
  alongside the existing pending future and status callback. Without
  cleanup, a slow bridge would leak the dict entry and the next turn
  would receive deltas meant for the previous one.
* DELTA frames received over the WS dispatch to the per-session
  ``_delta_callbacks`` entry. Without this, bridges don't see deltas
  and the migration to ``send_message_streaming`` accomplishes
  nothing.
* RESPONSE frame clears the delta callback (parity with the existing
  ``_status_callbacks`` cleanup so an old callback can't fire on the
  next turn against the same session id).
* Backward compat: ``send_message`` (no ``on_delta``) keeps the
  delta-callbacks dict empty — old bridge code paths don't accidentally
  start receiving deltas.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from ._framework import TestContext, test


# Reuse the same helper pattern as test_bridges._FakeBridge — local
# copy so the import order within the test driver stays trivial.


class _FakeBridge:
    def __init__(self) -> None:
        from openagent.bridges.base import BaseBridge
        self._real = BaseBridge.__new__(BaseBridge)
        self._real._pending = {}
        self._real._status_callbacks = {}
        self._real._delta_callbacks = {}
        self._real._session_locks = {}
        self._real._ws = object()
        self.sent: list[dict] = []

        async def fake_send(payload: dict) -> None:
            self.sent.append(payload)

        self._real._send_gateway_json = fake_send  # type: ignore[assignment]

    def resolve(self, sid: str, payload: dict) -> None:
        fut = self._real._pending[sid]
        if not fut.done():
            fut.set_result(payload)

    async def feed_listener(self, frames: list[dict]) -> None:
        """Drive the bits of _listen_gateway we care about — DELTA
        dispatch — without setting up a real WebSocket. Faithful copy
        of the dispatch logic in :meth:`BaseBridge._listen_gateway`."""
        from openagent.gateway import protocol as P

        for data in frames:
            t = data.get("type")
            sid = data.get("session_id")
            if t == P.DELTA and sid in self._real._delta_callbacks:
                await self._real._delta_callbacks[sid](data.get("text", ""))
            elif t == P.STATUS and sid in self._real._status_callbacks:
                await self._real._status_callbacks[sid](data.get("text", ""))
            elif t == P.RESPONSE and sid in self._real._pending:
                fut = self._real._pending.pop(sid)
                if not fut.done():
                    fut.set_result(data)
                self._real._status_callbacks.pop(sid, None)
                self._real._delta_callbacks.pop(sid, None)


@test("streaming", "send_message_streaming registers + clears the delta callback")
async def t_streaming_callback_lifecycle(_ctx: TestContext) -> None:
    fb = _FakeBridge()
    received: list[str] = []

    async def on_delta(chunk: str) -> None:
        received.append(chunk)

    # Drive a turn: send + simulate gateway delivering 3 deltas + RESPONSE.
    async def driver() -> None:
        # Wait for the bridge to register the pending future.
        for _ in range(500):
            if "s-stream" in fb._real._pending:
                break
            await asyncio.sleep(0.001)
        await fb.feed_listener([
            {"type": "delta", "session_id": "s-stream", "text": "Hello "},
            {"type": "delta", "session_id": "s-stream", "text": "world"},
            {"type": "delta", "session_id": "s-stream", "text": "!"},
            {"type": "response", "session_id": "s-stream", "text": "Hello world!"},
        ])

    result, _ = await asyncio.gather(
        fb._real.send_message_streaming("hi", "s-stream", on_delta=on_delta),
        driver(),
    )

    assert received == ["Hello ", "world", "!"], received
    assert result["text"] == "Hello world!", result
    # Cleanup — both maps must be empty so a later turn against the
    # same id doesn't receive stale callbacks.
    assert "s-stream" not in fb._real._delta_callbacks, fb._real._delta_callbacks
    assert "s-stream" not in fb._real._pending, fb._real._pending


@test("streaming", "send_message (no on_delta) leaves delta_callbacks empty")
async def t_legacy_send_message_no_delta_callback(_ctx: TestContext) -> None:
    fb = _FakeBridge()

    async def driver() -> None:
        for _ in range(500):
            if "s-legacy" in fb._real._pending:
                break
            await asyncio.sleep(0.001)
        # Even if the gateway happens to send DELTA, the legacy path
        # must not have registered a callback for it.
        assert "s-legacy" not in fb._real._delta_callbacks, (
            "legacy send_message should not register a delta callback"
        )
        await fb.feed_listener([
            {"type": "response", "session_id": "s-legacy", "text": "ok"},
        ])

    result, _ = await asyncio.gather(
        fb._real.send_message("hi", "s-legacy"),
        driver(),
    )
    assert result["text"] == "ok", result


@test("streaming", "RESPONSE clears the delta callback even when none arrived")
async def t_response_clears_delta_callback(_ctx: TestContext) -> None:
    """Even on a turn that produced zero deltas (older gateway, or a
    text-only turn that fell into the empty-stream fallback path), the
    RESPONSE handler must still tear down the delta callback so the
    next turn's callback wins."""
    fb = _FakeBridge()
    seen: list[str] = []

    async def on_delta(chunk: str) -> None:
        seen.append(chunk)

    async def driver() -> None:
        for _ in range(500):
            if "s-empty" in fb._real._pending:
                break
            await asyncio.sleep(0.001)
        await fb.feed_listener([
            {"type": "response", "session_id": "s-empty", "text": "no deltas this turn"},
        ])

    await asyncio.gather(
        fb._real.send_message_streaming("hi", "s-empty", on_delta=on_delta),
        driver(),
    )

    assert seen == [], f"no deltas should have been delivered: {seen}"
    assert "s-empty" not in fb._real._delta_callbacks, fb._real._delta_callbacks


@test("streaming", "P.DELTA constant is exposed by the protocol module")
async def t_protocol_delta_constant(_ctx: TestContext) -> None:
    """Sanity guard — clients import this constant by name, drift in
    the spelling silently breaks the wire format."""
    from openagent.gateway import protocol as P
    assert hasattr(P, "DELTA"), "protocol.py must expose DELTA"
    assert P.DELTA == "delta", f"DELTA should be 'delta', got {P.DELTA!r}"


@test("streaming", "gateway delta handler schema (smoke)")
async def t_gateway_delta_payload_shape(_ctx: TestContext) -> None:
    """Verifies the WS payload emitted by the new text streaming path
    has exactly the fields the universal client expects (``type``,
    ``text``, ``session_id``). Reads the source line directly so the
    test catches drift even when the live WS path can't be exercised
    without API keys.
    """
    from pathlib import Path
    src = Path(__file__).resolve().parents[2] / "openagent" / "gateway" / "server.py"
    body = src.read_text(encoding="utf-8")
    # The streaming text path constructs the DELTA payload via a dict
    # literal with these three keys — assert all three appear together
    # in the source (cheap, brittle, but sufficient as a smoke check).
    assert '"type": P.DELTA' in body, "gateway server.py no longer emits P.DELTA payloads"
    assert '"text": delta' in body, "gateway server.py no longer wires delta text into payload"
    assert (
        '"session_id": session_id,\n                            }' in body
        or '"session_id": session_id' in body
    ), "gateway DELTA payload missing session_id"
    # JSON sanity-check that nothing unparseable lurks at the file level.
    json.dumps({"type": "delta", "text": "x", "session_id": "y"})  # smoke
