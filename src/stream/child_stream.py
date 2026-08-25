"""Stream a spawned child session's frames over the active turn's channel.

A delegated sub-agent runs *inside* the leader's turn (not as its own
gateway ``StreamSession``), so its token deltas and tool calls never reach
the client on their own. But the app already routes every ``delta`` /
``status`` / ``turn_complete`` frame by ``session_id`` — a child session is
just another session to it. So all that's missing is for the server to emit
the child's frames tagged with the child's ``session_id``.

This module is the seam. The gateway's per-turn runner (``StreamTurnRunner``)
installs an **emitter** here for the duration of a turn; it publishes onto
that turn's single serialized outbound queue (the same pump the parent's own
deltas use — so there's no concurrent-write race on the client socket). The
team runner, deep inside the leader's run, reads the emitter via the
contextvar and forwards each member event as a child-tagged frame. When no
streaming client drives the turn (a bridge, the scheduler) the emitter stays
``None`` and child frames are simply not emitted — the durable row + a
``reconcile`` on completion still backfill the transcript.

Same contextvar trick as ``src/core/identity_context.py`` /
``src/mcp/servers/delegation/handlers.py`` — bound for the turn, read deep in
the stack without threading it through every provider signature.

Frame shape (a plain dict the emitter translates into the right ``Out*``
event): ``{"kind": "delta"|"status"|"turn_complete", "session_id": str,
"text"?: str}``.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# Emits one child-tagged frame over the active turn's channel. Async so the
# publish (onto the serialized outbound queue) is awaited inline, preserving
# delta order.
ChildStreamEmitter = Callable[[dict[str, Any]], Awaitable[None]]

_emitter_var: contextvars.ContextVar[Optional[ChildStreamEmitter]] = contextvars.ContextVar(
    "openagent_child_stream_emitter", default=None,
)


def install_child_stream_emitter(emitter: Optional[ChildStreamEmitter]):
    """Bind ``emitter`` for the current turn. Returns a reset token."""
    return _emitter_var.set(emitter)


def reset_child_stream_emitter(token) -> None:
    _emitter_var.reset(token)


def current_child_stream_emitter() -> Optional[ChildStreamEmitter]:
    """The emitter bound for the current turn, or ``None`` (no streaming
    client — bridge / scheduler / headless)."""
    return _emitter_var.get()


# A turn-scoped emitter (above) publishes onto ONE client's serialized channel
# — the right home for a sub-agent running inside a chat turn. But a scheduled
# firing / workflow node runs DETACHED (the scheduler loop, the workflow
# executor — no active turn, no contextvar emitter), yet its run screen should
# still stream live. For those, the gateway registers a BROADCAST sink here
# that fans each child frame out to every connected client (tagged with the
# child sid); ``run_child_session`` installs it as the turn emitter when none is
# already bound. Stays None in headless/bridge runs → child frames are silent.
_broadcast_sink: Optional[ChildStreamEmitter] = None


def set_child_broadcast_sink(cb: Optional[ChildStreamEmitter]) -> None:
    """Register (or clear with None) the gateway's broadcast-to-all-clients
    emitter for detached child runs."""
    global _broadcast_sink
    _broadcast_sink = cb


def broadcast_child_emitter() -> Optional[ChildStreamEmitter]:
    """The registered broadcast emitter, or None when no gateway is wired."""
    return _broadcast_sink


def child_frame_to_event(frame: dict[str, Any], *, seq: int = 0, ts_ms: int = 0):
    """Translate a child-stream ``frame`` dict into the ``Out*`` event the wire
    layer serialises. The single source of truth for the ``kind`` → event
    mapping, shared by the in-turn channel emitter (``StreamTurnRunner``) and
    the detached broadcast sink (the gateway). Returns ``None`` for an unknown
    kind or a frame with no ``session_id``."""
    from src.stream.events import (
        TURN_END_COMPLETED,
        OutSeedMessage, OutTextDelta, OutTextFinal, OutToolStatus, TurnComplete,
    )
    sid = frame.get("session_id")
    if not sid:
        return None
    kind = frame.get("kind")
    if kind == "delta":
        return OutTextDelta(session_id=sid, seq=seq, ts_ms=ts_ms, text=frame.get("text") or "")
    if kind == "status":
        return OutToolStatus(session_id=sid, seq=seq, ts_ms=ts_ms, text=frame.get("text") or "")
    if kind == "response":
        return OutTextFinal(session_id=sid, seq=seq, ts_ms=ts_ms, text=frame.get("text") or "")
    if kind == "seed":
        return OutSeedMessage(
            session_id=sid, seq=seq, ts_ms=ts_ms,
            text=frame.get("text") or "", author=frame.get("author"),
        )
    if kind == "turn_complete":
        return TurnComplete(
            session_id=sid, seq=seq, ts_ms=ts_ms,
            reason=str(frame.get("reason") or TURN_END_COMPLETED),
            error=str(frame.get("error") or ""),
        )
    return None


async def emit_child_frame(
    session_id: str,
    kind: str,
    *,
    text: Optional[str] = None,
    author: Optional[dict[str, Any]] = None,
) -> None:
    """Best-effort: forward one child-tagged frame to the bound emitter.

    A no-op when no emitter is installed. Never raises — streaming a child
    frame must never break the actual delegation it's mirroring. ``author``
    rides only on the ``seed`` frame (the agent-self mission prompt).
    """
    emitter = _emitter_var.get()
    if emitter is None or not session_id:
        return
    frame: dict[str, Any] = {"kind": kind, "session_id": session_id}
    if text is not None:
        frame["text"] = text
    if author is not None:
        frame["author"] = author
    try:
        await emitter(frame)
    except Exception as e:  # noqa: BLE001 — child mirroring is best-effort
        logger.debug("child_stream: emit %s for %s failed: %s", kind, session_id, e)
