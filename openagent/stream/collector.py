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

from openagent.stream.channel import BatchedReply
from openagent.stream.events import (
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

    def to_legacy_reply(self) -> dict:
        """Render the answer-response dict shape bridge / CLI callers expect."""
        base = {
            "model": self.model,
            "attachments": self.attachments,
            "target": self.latest_target,
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
        # Bridges + CLI run in answer-response mode and don't need
        # progressive deltas; ``OutTextFinal`` carries the canonical
        # text. Drop silently.
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
