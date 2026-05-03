"""STT abstraction — model-agnostic, streaming-capable.

Wraps the existing :func:`openagent.channels.voice.transcribe` function in
a small protocol so the unified stream pipeline can dispatch transcription
through one interface. Provider selection comes from the SQLite ``models``
table (rows where ``kind='stt'``) — same source of truth as the LLM and
TTS catalogs.

Two subclasses ship today:

* :class:`LiteLLMSTT` — wraps the ``provider_name`` + ``model`` row via
  ``litellm.atranscription``. One-shot only (litellm has no streaming
  transcription API). The default :meth:`BaseSTT.stream` buffers audio
  chunks into a tempfile by utterance and runs ``transcribe_file`` per
  utterance.
* :class:`WhisperLocalSTT` — wraps the bundled ``faster-whisper`` model
  loader (the long-time default fallback). Same one-shot semantics.

Streaming-capable vendors (Deepgram, AssemblyAI, OpenAI Realtime) do NOT
fit through litellm — they need vendor-specific WebSocket adapters with
``supports_streaming=True`` overriding :meth:`stream`. Adding one is
purely additive: new module, new ``provider_name`` matched in
:func:`resolve_stt`, no changes to callers.

Public surface:

* :class:`STTEvent` — partial / final transcript record.
* :class:`BaseSTT` — abstract base.
* :func:`resolve_stt` — DB-driven factory; falls back to local Whisper
  when no row is configured. Returns ``None`` only when no backend is
  available at all (caller falls through to text-only).
"""

from __future__ import annotations

import asyncio
import logging
import os
import struct
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from openagent.channels import voice as _voice
from openagent.core.logging import elog

logger = logging.getLogger(__name__)


# Encoding strings the wire / clients use for raw 16-bit signed PCM.
# Deepgram calls it ``linear16``; Web Audio API outputs Float32 we
# downsample to ``pcm16`` (Int16). Treat them interchangeably.
_PCM_ENCODINGS = frozenset({"pcm16", "pcm", "linear16"})


def _build_wav_header(
    data_size: int,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Build a 44-byte RIFF/WAVE header for raw PCM data.

    PCM concatenates trivially (no container framing) so we can buffer
    live chunks and prepend this header at end-of-utterance to produce a
    valid WAV file ffmpeg / faster-whisper / litellm.atranscription can
    parse. Defaults match the AudioWorklet's downsampler output: 16 kHz
    mono signed 16-bit little-endian.
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack(
            "<IHHIIHH",
            16,                # fmt chunk size
            1,                 # PCM format
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        )
        + b"data"
        + struct.pack("<I", data_size)
    )


@dataclass(frozen=True)
class STTEvent:
    """One transcript event from a streaming STT pipeline."""

    kind: Literal["partial", "final"]
    text: str
    confidence: float | None = None


class BaseSTT(ABC):
    """Streaming-capable STT interface.

    Default :meth:`stream` is a one-shot fallback: it buffers every
    inbound chunk into a temp file and runs :meth:`transcribe_file` once
    the iterator ends. That makes any one-shot backend usable as a
    "streaming" STT — the latency is the same as today's voice-message
    path, but the call shape matches the streaming contract so downstream
    code (StreamSession's STT pump) doesn't branch.

    Subclasses with native streaming (Deepgram, AssemblyAI, …) override
    :meth:`stream` and set :attr:`supports_streaming` to ``True``.
    """

    supports_streaming: bool = False

    @abstractmethod
    async def transcribe_file(
        self,
        path: str,
        *,
        language: str | None = None,
    ) -> str | None:
        """One-shot: transcribe a file on disk."""

    async def stream(
        self,
        audio_in: AsyncIterator[bytes],
        *,
        language: str | None = None,
        encoding: str = "webm",
        sample_rate: int | None = None,
    ) -> AsyncIterator[STTEvent]:
        """Streaming: consume audio chunks, emit transcript events.

        Default implementation collects every chunk into a temp file and
        runs ``transcribe_file`` once the input iterator drains. Useful
        for vendors without native streaming — the transducer shape is
        the same, just with worse latency.

        Two paths based on ``encoding``:

        * **PCM** (``pcm16`` / ``pcm`` / ``linear16``) — buffer the raw
          bytes, prepend a WAV header at EOS, write to ``.wav`` tempfile.
          PCM concatenates trivially so live-streamed chunks always form
          a valid file ffmpeg can parse. ``sample_rate`` defaults to
          16000 (matches the universal app's AudioWorklet downsampler).
        * **Container** (``webm``, ``mp4``, ``ogg``, ...) — write the
          chunks as-is. Only valid when the iterator yields ONE complete
          container blob (e.g. MediaRecorder.stop() output). Multiple
          partial container chunks do NOT round-trip cleanly through
          ffmpeg, which is why the universal app prefers the PCM path
          when AudioWorklet is available.
        """
        is_pcm = encoding.lower() in _PCM_ENCODINGS
        suffix = ".wav" if is_pcm else "." + (encoding.lower() or "webm").lstrip(".")
        tmp = tempfile.NamedTemporaryFile(prefix="oa_stt_", suffix=suffix, delete=False)
        try:
            if is_pcm:
                pcm = bytearray()
                async for chunk in audio_in:
                    if chunk:
                        pcm.extend(chunk)
                if not pcm:
                    return
                rate = sample_rate or 16000
                tmp.write(_build_wav_header(len(pcm), sample_rate=rate))
                tmp.write(bytes(pcm))
            else:
                written = 0
                async for chunk in audio_in:
                    if chunk:
                        tmp.write(chunk)
                        written += len(chunk)
                if not written:
                    return
            tmp.close()
            text = await self.transcribe_file(tmp.name, language=language)
            if text:
                yield STTEvent(kind="final", text=text)
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


class LiteLLMSTT(BaseSTT):
    """One-shot STT via litellm. Reads provider config from a DB row."""

    def __init__(self, row: dict[str, Any]):
        self._row = row

    async def transcribe_file(
        self,
        path: str,
        *,
        language: str | None = None,
    ) -> str | None:
        return await _voice._transcribe_via_litellm(
            path, self._row, language=language,
        )


class WhisperLocalSTT(BaseSTT):
    """Local faster-whisper. No DB row needed; bundled in core deps."""

    async def transcribe_file(
        self,
        path: str,
        *,
        language: str | None = None,
    ) -> str | None:
        return await _voice._transcribe_local(path, language=language)


class OpenAIWhisperSTT(BaseSTT):
    """Last-resort cloud fallback when ``OPENAI_API_KEY`` is set and
    nothing else is available."""

    async def transcribe_file(
        self,
        path: str,
        *,
        language: str | None = None,
    ) -> str | None:
        return await _voice._transcribe_openai(path, language=language)


async def resolve_stt(db: Any) -> BaseSTT | None:
    """Pick the active STT backend.

    Resolution order matches :func:`openagent.channels.voice.transcribe`
    so behaviour is identical to the legacy path:

    1. Latest enabled ``models`` row with ``kind='stt'`` → :class:`LiteLLMSTT`.
    2. Local ``faster-whisper`` if importable → :class:`WhisperLocalSTT`.
    3. ``OPENAI_API_KEY`` env var → :class:`OpenAIWhisperSTT`.
    4. Otherwise ``None``.
    """
    if db is not None:
        row = await _voice._resolve_stt_provider(db)
        if row is not None:
            vendor = (row.get("provider_name") or "").strip().lower()
            if vendor == "deepgram":
                # Streaming WS adapter — ~10× lower TTFA than litellm REST
                # in voice mode, and gives the same one-shot REST surface
                # via :meth:`transcribe_file` for bridge callers.
                from openagent.channels.stt_deepgram import DeepgramStreamingSTT

                elog(
                    "stt.resolve",
                    vendor=vendor,
                    model=row.get("model"),
                    kind="deepgram_ws",
                )
                return DeepgramStreamingSTT(
                    api_key=row.get("api_key") or "",
                    model=row.get("model") or "nova-2",
                    base_url=row.get("base_url"),
                    metadata=row.get("metadata") or {},
                )
            elog(
                "stt.resolve",
                vendor=vendor,
                model=row.get("model"),
                kind="litellm",
            )
            return LiteLLMSTT(row)

    # Lazy-import to avoid pulling faster-whisper into every gateway boot
    # if the user opted out of voice mode.
    try:
        import faster_whisper  # noqa: F401

        elog("stt.resolve", kind="local_whisper")
        return WhisperLocalSTT()
    except ImportError:
        pass

    if os.environ.get("OPENAI_API_KEY"):
        elog("stt.resolve", kind="openai_whisper")
        return OpenAIWhisperSTT()

    return None


# ── Convenience for ad-hoc one-shot transcription via the new
#    interface. Kept thin — the legacy ``voice.transcribe`` is still the
#    one-call entry point for HTTP/bridge callers. ────────────────────


async def transcribe_one_shot(
    path: str,
    db: Any | None = None,
    *,
    language: str | None = None,
) -> str | None:
    """Resolve once and run ``transcribe_file`` — drop-in for
    :func:`openagent.channels.voice.transcribe` callers that want to
    exercise the new factory path."""
    stt = await resolve_stt(db)
    if stt is None:
        return None
    try:
        return await stt.transcribe_file(path, language=language)
    except FileNotFoundError:
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("transcribe_one_shot failed: %s", e)
        return None


__all__ = [
    "STTEvent",
    "BaseSTT",
    "LiteLLMSTT",
    "WhisperLocalSTT",
    "OpenAIWhisperSTT",
    "resolve_stt",
    "transcribe_one_shot",
]
