"""JSON wire codec for stream :mod:`events`.

The gateway WebSocket has historically spoken a small set of frame types
(``MESSAGE`` / ``DELTA`` / ``RESPONSE`` / ``AUDIO_*`` / ``STATUS`` /
``ERROR``). The new stream protocol is a strict superset — older clients
keep speaking the legacy frames; newer clients can opt into the typed
event vocabulary.

Two functions:

* :func:`event_to_wire` serialises an outbound :class:`Event` into a JSON-
  compatible dict. For events that map onto a legacy frame
  (``OutTextDelta`` → ``DELTA``, ``OutAudio*`` → ``AUDIO_*``, etc.) the
  dict uses the legacy ``type`` so older readers continue to render them.
  Newer events (``OutVideoFrame``, ``TurnComplete``) get fresh types.
* :func:`wire_to_event` parses an inbound dict into the matching
  :class:`Event`. Legacy ``MESSAGE`` frames map to :class:`TextFinal` so
  the rest of the pipeline doesn't need to know about wire history.

Audio and video bytes are base64-encoded on the wire (matches the legacy
``AUDIO_CHUNK.data`` field). A future binary-WS path can swap that out by
replacing this module without touching the bus.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from openagent.gateway import protocol as P
from openagent.stream.events import (
    Attachment,
    AudioChunk,
    Event,
    Interrupt,
    OutAudioChunk,
    OutAudioEnd,
    OutAudioStart,
    OutError,
    OutTextDelta,
    OutTextFinal,
    OutToolStatus,
    OutVideoFrame,
    SessionClose,
    SessionOpen,
    TextDelta,
    TextFinal,
    TurnComplete,
    VideoFrame,
)

logger = logging.getLogger(__name__)

# New wire-frame ``type`` strings introduced by the stream protocol.
# Existing constants in :mod:`openagent.gateway.protocol` are reused for
# the legacy mappings; the strings below are pure additions.
TEXT_DELTA_IN = "text_delta"  # client → server typed-text delta
TEXT_FINAL_IN = "text_final"  # client → server text commit
AUDIO_CHUNK_IN = "audio_chunk_in"  # client → server PCM/opus chunk
AUDIO_END_IN = "audio_end_in"  # client → server end-of-speech marker
VIDEO_FRAME_IN = "video_frame"  # client → server image frame
ATTACHMENT_IN = "attachment"  # client → server file ref
INTERRUPT = "interrupt"  # client → server barge-in
SESSION_OPEN = "session_open"  # client → server stream open
SESSION_CLOSE = "session_close"  # client → server stream close

VIDEO_FRAME_OUT = "video_frame_out"  # server → client image frame
TURN_COMPLETE = "turn_complete"  # server → client batched-channel sentinel


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii") if data else ""


def _b64decode(s: str | None) -> bytes:
    if not s:
        return b""
    try:
        return base64.b64decode(s)
    except Exception as e:  # noqa: BLE001 — malformed wire bytes are non-fatal
        logger.debug("wire: base64 decode failed: %s", e)
        return b""


def event_to_wire(evt: Event) -> dict[str, Any]:
    """Serialise an event to a JSON-friendly dict.

    Legacy frames are emitted unchanged so older clients (universal app
    pre-stream protocol, CLI, telegram bridge) keep rendering them.
    """
    base = {"session_id": evt.session_id}
    if evt.seq:
        base["seq"] = evt.seq
    if evt.ts_ms:
        base["ts_ms"] = evt.ts_ms

    if isinstance(evt, OutTextDelta):
        return {**base, "type": P.DELTA, "text": evt.text}
    if isinstance(evt, OutTextFinal):
        return {
            **base,
            "type": P.RESPONSE,
            "text": evt.text,
            "attachments": list(evt.attachments) or None,
            "model": evt.model,
        }
    if isinstance(evt, OutAudioStart):
        return {
            **base,
            "type": P.AUDIO_START,
            "format": evt.format,
            "mime": evt.mime,
            "voice_id": evt.voice_id,
        }
    if isinstance(evt, OutAudioChunk):
        return {**base, "type": P.AUDIO_CHUNK, "data": _b64encode(evt.data)}
    if isinstance(evt, OutAudioEnd):
        return {**base, "type": P.AUDIO_END, "total_chunks": evt.total_chunks}
    if isinstance(evt, OutToolStatus):
        return {**base, "type": P.STATUS, "text": evt.text}
    if isinstance(evt, OutError):
        return {**base, "type": P.ERROR, "text": evt.text}
    if isinstance(evt, OutVideoFrame):
        return {
            **base,
            "type": VIDEO_FRAME_OUT,
            "stream": evt.stream,
            "data": _b64encode(evt.image_bytes),
            "width": evt.width,
            "height": evt.height,
        }
    if isinstance(evt, TurnComplete):
        return {**base, "type": TURN_COMPLETE}

    # Inbound types — included for completeness (e.g. tests that
    # round-trip both directions). Servers don't usually emit these.
    if isinstance(evt, TextDelta):
        return {**base, "type": TEXT_DELTA_IN, "text": evt.text, "final": evt.final}
    if isinstance(evt, TextFinal):
        return {
            **base,
            "type": TEXT_FINAL_IN,
            "text": evt.text,
            "source": evt.source,
            "attachments": list(evt.attachments) or None,
        }
    if isinstance(evt, AudioChunk):
        return {
            **base,
            "type": AUDIO_CHUNK_IN,
            "data": _b64encode(evt.data),
            "end_of_speech": evt.end_of_speech,
            "sample_rate": evt.sample_rate or None,
            "encoding": evt.encoding or None,
        }
    if isinstance(evt, VideoFrame):
        return {
            **base,
            "type": VIDEO_FRAME_IN,
            "stream": evt.stream,
            "data": _b64encode(evt.image_bytes),
            "width": evt.width,
            "height": evt.height,
            "keyframe": evt.keyframe,
        }
    if isinstance(evt, Attachment):
        return {
            **base,
            "type": ATTACHMENT_IN,
            "kind": evt.kind,
            "path": evt.path,
            "filename": evt.filename,
            "mime_type": evt.mime_type,
        }
    if isinstance(evt, Interrupt):
        return {**base, "type": INTERRUPT, "reason": evt.reason}
    if isinstance(evt, SessionOpen):
        return {
            **base,
            "type": SESSION_OPEN,
            "profile": evt.profile,
            "llm_pin": evt.llm_pin,
            "stt_pin": evt.stt_pin,
            "tts_pin": evt.tts_pin,
            "language": evt.language,
            "client_kind": evt.client_kind,
            # Only emit the coalesce field when the caller set it
            # explicitly. ``None`` round-trips as "use the server
            # default"; an explicit ``0`` round-trips as "opt out".
            "coalesce_window_ms": evt.coalesce_window_ms,
            # Only emit the speak field when it differs from the default
            # (True) so older clients keep the same wire shape they always
            # saw. ``None`` here gets dropped by the JSON encoder; ``False``
            # becomes ``"speak": false``.
            "speak": None if evt.speak else False,
        }
    if isinstance(evt, SessionClose):
        return {**base, "type": SESSION_CLOSE}

    raise TypeError(f"event_to_wire: unsupported event type {type(evt).__name__}")


def wire_to_event(frame: dict[str, Any]) -> Event | None:
    """Parse a wire frame into an :class:`Event`.

    Returns ``None`` for frames that aren't part of the stream protocol
    (auth, ping, command, …) so callers can keep their existing routing
    for those. Unknown event types log a debug line and return ``None``.
    """
    t = frame.get("type")
    if not t:
        return None
    sid = frame.get("session_id") or ""
    seq = int(frame.get("seq") or 0)
    ts = int(frame.get("ts_ms") or 0)

    # Legacy MESSAGE — the existing universal-app and CLI submit path.
    if t == P.MESSAGE:
        return TextFinal(
            session_id=sid, seq=seq, ts_ms=ts,
            text=str(frame.get("text") or ""),
            source="user_typed",
            attachments=tuple(frame.get("attachments") or ()),
        )

    if t == TEXT_DELTA_IN:
        return TextDelta(
            session_id=sid, seq=seq, ts_ms=ts,
            text=str(frame.get("text") or ""),
            final=bool(frame.get("final")),
        )
    if t == TEXT_FINAL_IN:
        return TextFinal(
            session_id=sid, seq=seq, ts_ms=ts,
            text=str(frame.get("text") or ""),
            source=frame.get("source") or "user_typed",
            attachments=tuple(frame.get("attachments") or ()),
        )
    if t == AUDIO_CHUNK_IN:
        return AudioChunk(
            session_id=sid, seq=seq, ts_ms=ts,
            data=_b64decode(frame.get("data")),
            end_of_speech=bool(frame.get("end_of_speech")),
            sample_rate=int(frame.get("sample_rate") or 0),
            encoding=str(frame.get("encoding") or ""),
        )
    if t == AUDIO_END_IN:
        # End-of-speech sentinel without payload — emit an empty AudioChunk
        # marked as end_of_speech so the STT pump treats it uniformly.
        return AudioChunk(
            session_id=sid, seq=seq, ts_ms=ts,
            data=b"", end_of_speech=True,
        )
    if t == VIDEO_FRAME_IN:
        return VideoFrame(
            session_id=sid, seq=seq, ts_ms=ts,
            stream=str(frame.get("stream") or ""),
            image_bytes=_b64decode(frame.get("data")),
            width=int(frame.get("width") or 0),
            height=int(frame.get("height") or 0),
            keyframe=bool(frame.get("keyframe")),
        )
    if t == ATTACHMENT_IN:
        return Attachment(
            session_id=sid, seq=seq, ts_ms=ts,
            kind=frame.get("kind") or "file",
            path=frame.get("path"),
            filename=str(frame.get("filename") or ""),
            mime_type=frame.get("mime_type"),
        )
    if t == INTERRUPT:
        return Interrupt(
            session_id=sid, seq=seq, ts_ms=ts,
            reason=frame.get("reason") or "manual",
        )
    if t == SESSION_OPEN:
        # ``speak`` defaults to True; only flip if the frame explicitly
        # carries ``False``. Anything else (None / missing / truthy) maps
        # to True so we don't silently disable TTS for older clients.
        speak_field = frame.get("speak")
        speak = False if speak_field is False else True
        # ``coalesce_window_ms`` distinguishes three states on the wire:
        # missing/null → ``None`` → "use the server default". Explicit
        # ``0`` → opt out (legacy preempt-on-each-message). Positive int
        # → that exact window. Don't conflate ``0`` with absent.
        coalesce_raw = frame.get("coalesce_window_ms")
        coalesce: int | None
        if coalesce_raw is None:
            coalesce = None
        else:
            try:
                coalesce = int(coalesce_raw)
            except (TypeError, ValueError):
                coalesce = None
        return SessionOpen(
            session_id=sid, seq=seq, ts_ms=ts,
            profile=frame.get("profile") or "realtime",
            llm_pin=frame.get("llm_pin"),
            stt_pin=frame.get("stt_pin"),
            tts_pin=frame.get("tts_pin"),
            language=frame.get("language"),
            client_kind=frame.get("client_kind"),
            coalesce_window_ms=coalesce,
            speak=speak,
        )
    if t == SESSION_CLOSE:
        return SessionClose(session_id=sid, seq=seq, ts_ms=ts)

    # Outbound types parsed from the wire (used by tests round-tripping
    # both directions). Real clients receive these and don't usually
    # forward them; servers don't usually parse them inbound.
    if t == P.DELTA:
        return OutTextDelta(
            session_id=sid, seq=seq, ts_ms=ts,
            text=str(frame.get("text") or ""),
        )
    if t == P.RESPONSE:
        return OutTextFinal(
            session_id=sid, seq=seq, ts_ms=ts,
            text=str(frame.get("text") or ""),
            attachments=tuple(frame.get("attachments") or ()),
            model=frame.get("model"),
        )
    if t == P.AUDIO_START:
        return OutAudioStart(
            session_id=sid, seq=seq, ts_ms=ts,
            format=str(frame.get("format") or "mp3"),
            mime=str(frame.get("mime") or "audio/mpeg"),
            voice_id=frame.get("voice_id"),
        )
    if t == P.AUDIO_CHUNK:
        return OutAudioChunk(
            session_id=sid, seq=seq, ts_ms=ts,
            data=_b64decode(frame.get("data")),
        )
    if t == P.AUDIO_END:
        return OutAudioEnd(
            session_id=sid, seq=seq, ts_ms=ts,
            total_chunks=int(frame.get("total_chunks") or 0),
        )
    if t == P.STATUS:
        return OutToolStatus(
            session_id=sid, seq=seq, ts_ms=ts,
            text=str(frame.get("text") or ""),
        )
    if t == P.ERROR:
        return OutError(
            session_id=sid, seq=seq, ts_ms=ts,
            text=str(frame.get("text") or ""),
        )
    if t == VIDEO_FRAME_OUT:
        return OutVideoFrame(
            session_id=sid, seq=seq, ts_ms=ts,
            stream=str(frame.get("stream") or ""),
            image_bytes=_b64decode(frame.get("data")),
            width=int(frame.get("width") or 0),
            height=int(frame.get("height") or 0),
        )
    if t == TURN_COMPLETE:
        return TurnComplete(session_id=sid, seq=seq, ts_ms=ts)

    logger.debug("wire_to_event: unknown type %r", t)
    return None


__all__ = [
    "event_to_wire",
    "wire_to_event",
    "TEXT_DELTA_IN",
    "TEXT_FINAL_IN",
    "AUDIO_CHUNK_IN",
    "AUDIO_END_IN",
    "VIDEO_FRAME_IN",
    "ATTACHMENT_IN",
    "INTERRUPT",
    "SESSION_OPEN",
    "SESSION_CLOSE",
    "VIDEO_FRAME_OUT",
    "TURN_COMPLETE",
]
