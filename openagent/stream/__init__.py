"""Unified streaming I/O — typed event bus, sessions, and channel profiles.

This package introduces a bidirectional, multi-modal stream protocol that
sits alongside the legacy single-turn :class:`TurnRunner` path. Inbound
events (text, audio, video, attachments, interrupts) flow into a long-lived
:class:`StreamSession`; outbound events (LLM token deltas, TTS audio chunks,
tool status, video frames) flow back out on the same session bus.

Channels (webapp, telegram bridge, etc.) come in two profiles:

* :class:`RealtimeChannel` ferries events through transparently.
* :class:`BatchedChannel` collects an inbound stream into one user message
  and an outbound stream into one finished reply — used by the bot bridges
  whose transports don't support true bidirectional streaming.

The :mod:`openagent.stream.wire` codec round-trips events to/from the
existing JSON wire format (see :mod:`openagent.gateway.protocol`) so older
clients continue to work without changes.
"""

from openagent.stream.events import (
    Event,
    SessionOpen,
    SessionClose,
    TextDelta,
    TextFinal,
    AudioChunk,
    VideoFrame,
    Attachment,
    Interrupt,
    OutTextDelta,
    OutTextFinal,
    OutAudioStart,
    OutAudioChunk,
    OutAudioEnd,
    OutVideoFrame,
    OutToolStatus,
    OutError,
    TurnComplete,
    now_ms,
)

__all__ = [
    "Event",
    "SessionOpen",
    "SessionClose",
    "TextDelta",
    "TextFinal",
    "AudioChunk",
    "VideoFrame",
    "Attachment",
    "Interrupt",
    "OutTextDelta",
    "OutTextFinal",
    "OutAudioStart",
    "OutAudioChunk",
    "OutAudioEnd",
    "OutVideoFrame",
    "OutToolStatus",
    "OutError",
    "TurnComplete",
    "now_ms",
]
