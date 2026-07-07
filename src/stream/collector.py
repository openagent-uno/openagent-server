"""Shared accumulator for clients that drive a :class:`StreamSession`
through the gateway WS instead of directly via Python.

The webapp's ``RealtimeChannel`` pumps every outbound event through the
WS into ``event_to_wire`` JSON; in-process callers (tests, future
embeddings) can use :class:`BatchedChannel.run_one_shot` to drain the
session's ``outbound`` queue directly. Bridges and the CLI are a third
case: they speak the wire protocol over a long-lived WS but still want
the answer-response shape ``{text, model, attachments, ...}``. They
buffer outbound frames as they arrive and resolve when ``turn_complete``
lands.

:class:`StreamCollector` extends :class:`BatchedReply` with a single
``done`` event so an external listener (the bridge's
``_listen_gateway`` / the CLI's ``_listen``) can release the awaiter.
:func:`fold_outbound_event` is the single dispatch table used by every
listener; the per-bridge / per-CLI parsers became `wire_to_event` +
this fold instead of hand-rolled type-string cascades.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from src.stream.channel import BatchedReply
from src.stream.events import (
    Event,
    OutAudioChunk,
    OutAudioEnd,
    OutAudioStart,
    OutError,
    OutTextDelta,
    OutTextFinal,
    TurnComplete,
)


@dataclass
class StreamCollector(BatchedReply):
    """:class:`BatchedReply` driven by an external WS listener.

    The listener calls :func:`fold_outbound_event` for each decoded
    event; when it returns ``True`` the awaiter on ``done`` is released.
    """

    done: asyncio.Event = field(default_factory=asyncio.Event)
    # Most recent platform-specific reply anchor (Telegram ``Message``,
    # Discord ``Channel``, WhatsApp ``chat_id``, …) seen during this
    # turn. Followers update it so the OWNER posts the merged reply
    # against the LATEST message instead of the first one — otherwise
    # spam visibly anchors the bot's reply to a stale bubble. ``Any``
    # because the type is platform-defined; the bridge owns the value.
    latest_target: Any = None
    # Tool-status callback for the in-flight turn. Stored ON the
    # collector (not in a per-session dict) so it lives and dies with
    # the collector — a brand-new collector taking over the slot can't
    # have its callback wiped by the previous owner's cleanup.
    on_status: Callable[[str], Awaitable[None]] | None = None
    # Per-token delta callback for bridges that surface streaming
    # responses (Telegram edits the placeholder message as the model
    # types). Receives the *accumulated* text so far so bridges don't
    # need to maintain their own buffer — the same string that would
    # arrive as ``OutTextFinal`` if the turn ended right now.
    on_delta: Callable[[str], Awaitable[None]] | None = None
    # Compaction-progress callback (vision §2 in-place compaction). The
    # WS listener routes ``session_compacted`` frames here so a bridge can
    # post a "Compacting conversation" → "Compacted conversation" notice
    # as the fold runs. Receives the raw frame dict (``phase`` + stats).
    on_compaction: Callable[[dict], Awaitable[None]] | None = None
    # Rolling text buffer used by the delta path; ``OutTextFinal``
    # overwrites it with the canonical reply, so this is only meaningful
    # while the turn is still in flight.
    _delta_buffer: str = ""

    @property
    def accumulated_text(self) -> str:
        """The raw token stream accumulated so far this turn.

        This is the ordered concatenation of every ``OutTextDelta`` —
        ``fold_outbound_event`` appends to it synchronously as each delta
        frame is decoded, so it is always current/ordered even though the
        per-delta ``on_delta`` callbacks fire as detached tasks. Live-mode
        bridges read it to flush the narration that preceded a tool call
        (see ``BaseBridge.dispatch_turn``). Unlike ``text`` it still
        carries any ``[IMAGE:/path]`` / ``[FILE:/path]`` markers — the
        server strips those from the final ``OutTextFinal.text`` but not
        from the deltas.
        """
        return self._delta_buffer or ""

    def to_legacy_reply(self) -> dict:
        """Render the answer-response dict shape bridge / CLI callers expect."""
        base = {
            "model": self.model,
            "attachments": self.attachments,
            "target": self.latest_target,
            # Raw accumulated delta stream for live-mode bridges that
            # posted text incrementally during the turn and only need the
            # still-unposted tail at the end. ``None`` would be wrong here
            # (callers slice it); empty string is the right "nothing
            # streamed" sentinel.
            "accumulated": self.accumulated_text,
        }
        if self.errored:
            return {
                **base,
                "type": "error",
                "text": self.error_text or self.text or "Error",
            }
        return {
            **base,
            "type": "response",
            "text": self.text,
        }


def fold_outbound_event(reply: BatchedReply, evt: Event) -> bool:
    """Apply one outbound :class:`Event` to a :class:`BatchedReply`.

    Returns ``True`` on :class:`TurnComplete` so the caller can flip
    the ``done`` event (or break out of an inline drain loop) — the
    same termination contract :class:`BatchedChannel.run_one_shot`
    uses internally.
    """
    if isinstance(evt, OutTextDelta):
        # Bridges that opted into streaming (``on_delta`` set) get the
        # running text so they can edit a placeholder message in-place.
        # Non-streaming bridges still see the canonical reply land via
        # ``OutTextFinal`` below.
        if isinstance(reply, StreamCollector):
            reply._delta_buffer = (reply._delta_buffer or "") + (evt.text or "")
            cb = reply.on_delta
            if cb is not None:
                # Fire the callback as a background task so a slow
                # bridge edit (e.g. Telegram rate-limit retry) never
                # blocks the wire-event listener loop.
                try:
                    asyncio.create_task(cb(reply._delta_buffer))
                except RuntimeError:
                    # No running loop (test environment); skip silently.
                    pass
        return False
    if isinstance(evt, OutTextFinal):
        reply.text = evt.text
        reply.attachments = list(evt.attachments)
        reply.model = evt.model
        return False
    if isinstance(evt, OutAudioStart):
        reply.audio_format = evt.format
        reply.audio_mime = evt.mime
        reply.voice_id = evt.voice_id
        return False
    if isinstance(evt, OutAudioChunk):
        reply.audio_chunks.append(evt.data)
        return False
    if isinstance(evt, OutAudioEnd):
        return False
    if isinstance(evt, OutError):
        # Treat as terminal so the awaiter resolves immediately — the
        # gateway typically emits OutError when a turn dies before it
        # can publish TurnComplete.
        reply.errored = True
        reply.error_text = evt.text
        return True
    if isinstance(evt, TurnComplete):
        return True
    return False


__all__ = ["StreamCollector", "fold_outbound_event"]
