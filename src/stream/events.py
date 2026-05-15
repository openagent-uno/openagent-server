"""Typed event dataclasses for the unified inbound/outbound stream bus.

All events carry ``session_id`` (per-session bus identifier), ``seq``
(monotonic per-session counter assigned by :class:`StreamSession`), and
``ts_ms`` (monotonic-wall-time millisecond stamp, useful for client AV
sync and barge-in latency measurements). Subclasses add modality-specific
payloads.

The set is intentionally small and stable: every adapter — STT, TTS,
realtime providers, channels, bridges — speaks this same vocabulary.
Wire-format mapping lives in :mod:`openagent.stream.wire`.

Inbound (client → server)
-------------------------
* :class:`SessionOpen` / :class:`SessionClose` — session lifecycle.
* :class:`TextDelta` / :class:`TextFinal` — typed text or STT output.
* :class:`AudioChunk` — raw audio bytes; ``end_of_speech`` flags the
  last chunk of an utterance (client VAD or push-to-talk release).
* :class:`VideoFrame` — image bytes from a named stream (``"webcam"``,
  ``"screen"``, …).
* :class:`Attachment` — file reference (image / file / voice / video).
* :class:`Interrupt` — explicit barge-in; subordinate-priority terminal
  events (``TextFinal(source="stt")``, ``TextFinal(source="user_typed")``)
  also trigger interrupt server-side when a turn is in flight.

Outbound (server → client)
--------------------------
* :class:`OutTextDelta` / :class:`OutTextFinal` — LLM token stream + final.
* :class:`OutAudioStart` / :class:`OutAudioChunk` / :class:`OutAudioEnd` —
  TTS audio span.
* :class:`OutVideoFrame` — outbound video (future / vendor-walled).
* :class:`OutToolStatus` — "Using bash..." style tool progress.
* :class:`TurnComplete` — terminal marker for batched channels.
* :class:`OutError` — soft error surfaced inline (gateway already emits a
  trailing ``RESPONSE`` with the message; this is the typed equivalent).

These dataclasses are pure data — no I/O imports — so STT, TTS, agent,
and channel modules can all import them without circular dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


def now_ms() -> int:
    """Wall-clock millisecond timestamp (monotonic across one process).

    Used as the default ``ts_ms`` when callers don't pass one explicitly —
    ``time.monotonic`` is a float in seconds, ``int(... * 1000)`` is a
    sensible compact integer for logs and AV sync.
    """
    return int(time.monotonic() * 1000)


@dataclass(frozen=True)
class Event:
    """Common base — every event carries session/seq/timestamp."""

    session_id: str
    seq: int = 0
    ts_ms: int = 0


# ── Inbound (client → server) ─────────────────────────────────────────


@dataclass(frozen=True)
class SessionOpen(Event):
    """Open a stream session.

    ``profile`` selects the channel-profile semantics: ``"realtime"`` for
    full-duplex (webapp), ``"batched"`` for collect-and-send-once (bridges).

    The ``*_pin`` fields let a client pin a specific provider row (from the
    SQLite ``models`` table) for the lifetime of the session. ``None``
    falls through to ``resolve_*`` defaults.
    """

    profile: Literal["realtime", "batched"] = "realtime"
    llm_pin: str | None = None
    stt_pin: str | None = None
    tts_pin: str | None = None
    language: str | None = None
    client_kind: str | None = None
    # Debounce window for coalescing burst inputs into a single turn.
    # ``None`` (default) means "use the server-side default" —
    # ``StreamSession.DEFAULT_COALESCE_WINDOW_MS`` (500 ms), the OpenAI-
    # Realtime-style merged-burst UX. ``0`` is the explicit opt-out
    # (preempt-on-each-message, legacy behaviour). Any positive int
    # overrides with that exact window. STT/system messages always
    # bypass the window for instant voice barge-in.
    coalesce_window_ms: int | None = None
    # When False, the session NEVER invokes its TTS sidecar even if a
    # provider is resolved — chat-tab style usage where the user
    # doesn't want every reply to also be spoken aloud. Default True
    # preserves the original streaming behaviour (voice-mode + bridge
    # voice-notes both want speak=on by default). Honoured by
    # ``StreamTurnRunner`` via ``StreamSession.speak_enabled``.
    speak: bool = True


@dataclass(frozen=True)
class SessionClose(Event):
    """Close the session — drop providers, cancel any in-flight turn."""


@dataclass(frozen=True)
class TextDelta(Event):
    """Partial typed text or partial STT transcript.

    ``final=True`` is the terminal marker for typed text (user submitted)
    or STT (end-of-utterance). When ``final=True`` the receiver typically
    promotes this to a :class:`TextFinal` for turn dispatch. Pure UI
    deltas (``final=False``) never trigger interrupt by themselves.
    """

    text: str = ""
    final: bool = False


@dataclass(frozen=True)
class TextFinal(Event):
    """Committed user message ready to dispatch as a turn.

    ``source`` records where the text came from so adapters can apply
    different policies (e.g. on-final interrupt for STT, immediate-send
    for typed). ``attachments`` is the structured attachment list the
    agent will see (mirrors ``Agent.run_stream(attachments=...)``).
    """

    text: str = ""
    source: Literal["user_typed", "stt", "system"] = "user_typed"
    attachments: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class AudioChunk(Event):
    """Raw audio bytes. ``end_of_speech`` flags the last chunk."""

    data: bytes = b""
    end_of_speech: bool = False
    sample_rate: int = 0
    encoding: str = ""  # e.g. "pcm16", "opus", "webm"


@dataclass(frozen=True)
class VideoFrame(Event):
    """One frame from a named video stream.

    ``stream`` is a free-form client tag (``"webcam"`` / ``"screen"`` /
    ``"phone-front"``). Multiple streams interleave on the same bus.
    Frames buffer in a ring per stream on the session; the LLM samples
    the latest frame per stream at turn-trigger time.
    """

    stream: str = ""
    image_bytes: bytes = b""
    width: int = 0
    height: int = 0
    keyframe: bool = False


@dataclass(frozen=True)
class Attachment(Event):
    """A file reference uploaded out-of-band (e.g. via /api/upload)."""

    kind: Literal["image", "file", "voice", "video"] = "file"
    path: str | None = None
    filename: str = ""
    mime_type: str | None = None


@dataclass(frozen=True)
class Interrupt(Event):
    """Explicit barge-in. Cancels the active turn if one is in flight."""

    reason: Literal["user_speech", "user_text", "manual"] = "manual"


# ── Outbound (server → client) ────────────────────────────────────────


@dataclass(frozen=True)
class OutTextDelta(Event):
    """LLM token delta. The wire surface for typewriter UX."""

    text: str = ""


@dataclass(frozen=True)
class OutTextFinal(Event):
    """Committed assistant reply (full text + attachments + model)."""

    text: str = ""
    attachments: tuple[dict[str, Any], ...] = ()
    model: str | None = None


@dataclass(frozen=True)
class OutAudioStart(Event):
    """Header for an outbound TTS span."""

    format: str = "mp3"
    mime: str = "audio/mpeg"
    voice_id: str | None = None


@dataclass(frozen=True)
class OutAudioChunk(Event):
    """One TTS audio chunk."""

    data: bytes = b""


@dataclass(frozen=True)
class OutAudioEnd(Event):
    """Tail marker for an outbound TTS span."""

    total_chunks: int = 0


@dataclass(frozen=True)
class OutVideoFrame(Event):
    """Outbound video frame (future / vendor-walled)."""

    stream: str = ""
    image_bytes: bytes = b""
    width: int = 0
    height: int = 0


@dataclass(frozen=True)
class OutToolStatus(Event):
    """Tool progress hint ("Using bash...", "Read done", ...)."""

    text: str = ""


@dataclass(frozen=True)
class OutError(Event):
    """Inline soft error. Channel-batched callers may merge with text."""

    text: str = ""


@dataclass(frozen=True)
class TurnComplete(Event):
    """End-of-assistant-turn marker.

    Realtime channels typically ignore this (they let ``OutTextFinal`` +
    ``OutAudioEnd`` drive the UI). Batched channels rely on it to know
    when to commit one finished message + one finished voice note.
    """


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
