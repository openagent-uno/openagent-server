"""TTS abstraction — model-agnostic, streaming-capable.

Wraps the existing :mod:`openagent.channels.tts` machinery in a small
protocol so the unified stream pipeline can pick a TTS provider once and
call into it as a transducer (text-chunks-in → audio-chunks-out).

Three concrete subclasses ship today:

* :class:`LiteLLMTTS` — wraps :func:`tts.synthesize_stream` /
  :func:`tts.synthesize_full` for any LiteLLM-supported vendor (OpenAI,
  ElevenLabs REST, Azure, Groq, Vertex, …). Streaming text-in is
  implemented by the default :meth:`BaseTTS.synthesize_stream` — it pulls
  text deltas through a :class:`SentenceChunker` and synthesises one
  sentence at a time, preserving today's TTFA optimisation for the REST
  path.
* :class:`ElevenLabsWSTTS` — wraps the WebSocket token-stream path in
  :mod:`openagent.channels.tts_streaming`. ``supports_streaming=True``;
  the override pipes deltas straight into the vendor WS and yields
  audio frames as they arrive (sub-second TTFB).
* :class:`LocalPiperTTS` — wraps :mod:`openagent.channels.tts_local`
  (offline). One-shot only — Piper writes the whole WAV at once.

Public surface:

* :class:`BaseTTS` — abstract base.
* :func:`resolve_tts` — DB-driven factory; falls through to local Piper.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from openagent.channels import tts as _tts
from openagent.channels import tts_local as _piper
from openagent.channels import tts_streaming as _wsstream
from openagent.channels.tts_chunker import SentenceChunker
from openagent.core.logging import elog

logger = logging.getLogger(__name__)


class BaseTTS(ABC):
    """Streaming-capable TTS interface.

    Default :meth:`synthesize_stream` chunks the input text iterator at
    sentence boundaries and calls :meth:`synthesize_full` per sentence.
    That preserves the proven REST-per-sentence TTFA optimisation we
    already ship; subclasses with a native token-in/audio-out WebSocket
    (ElevenLabs) override :meth:`synthesize_stream` and set
    :attr:`supports_streaming` to ``True``.
    """

    supports_streaming: bool = False

    @property
    @abstractmethod
    def audio_format(self) -> tuple[str, str]:
        """Return ``(format, mime)`` — e.g. ``("mp3", "audio/mpeg")``."""

    @property
    def voice_id(self) -> str | None:
        return None

    @abstractmethod
    async def synthesize_full(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> bytes | None:
        """One-shot: synthesise the whole text into a single audio blob."""

    async def synthesize_stream(
        self,
        text_chunks: AsyncIterator[str],
        *,
        language: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Streaming: text deltas → audio bytes.

        Default implementation: feed deltas through :class:`SentenceChunker`,
        call :meth:`synthesize_full` per sentence as they emerge, yield
        the audio bytes per sentence. Time-to-first-audio is dominated by
        the first sentence's synthesis latency — typically well under a
        second on cloud REST and on local Piper.
        """
        chunker = SentenceChunker()
        async for delta in text_chunks:
            if not delta:
                continue
            for sentence in chunker.feed(delta):
                audio = await self.synthesize_full(sentence, language=language)
                if audio:
                    yield audio
        tail = chunker.flush()
        if tail:
            audio = await self.synthesize_full(tail, language=language)
            if audio:
                yield audio


class LiteLLMTTS(BaseTTS):
    """Cloud TTS via litellm (OpenAI / Azure / Groq / ElevenLabs REST / …)."""

    def __init__(self, cfg: _tts.TTSConfig):
        self._cfg = cfg

    @property
    def audio_format(self) -> tuple[str, str]:
        fmt = (self._cfg.response_format or "mp3").lower()
        mime = "audio/wav" if fmt == "wav" else "audio/mpeg"
        return fmt, mime

    @property
    def voice_id(self) -> str | None:
        return self._cfg.voice_id

    async def synthesize_full(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> bytes | None:
        return await _tts.synthesize_full(text, self._cfg, language=language)

    async def synthesize_stream(
        self,
        text_chunks: AsyncIterator[str],
        *,
        language: str | None = None,
    ) -> AsyncIterator[bytes]:
        # Per-sentence dispatch via :func:`tts.synthesize_stream` keeps
        # vendor-specific chunked decoding (ElevenLabs ``/stream``,
        # OpenAI ``stream_format``) — those vendors yield multiple chunks
        # per sentence which we forward as-is.
        chunker = SentenceChunker()
        async for delta in text_chunks:
            if not delta:
                continue
            for sentence in chunker.feed(delta):
                async for audio in _tts.synthesize_stream(
                    sentence, self._cfg, language=language,
                ):
                    if audio:
                        yield audio
        tail = chunker.flush()
        if tail:
            async for audio in _tts.synthesize_stream(
                tail, self._cfg, language=language,
            ):
                if audio:
                    yield audio


class ElevenLabsWSTTS(BaseTTS):
    """ElevenLabs WebSocket: token-in / audio-out streaming."""

    supports_streaming = True

    def __init__(self, cfg: _tts.TTSConfig):
        self._cfg = cfg

    @property
    def audio_format(self) -> tuple[str, str]:
        fmt = (self._cfg.response_format or "mp3").lower()
        mime = "audio/wav" if fmt == "wav" else "audio/mpeg"
        return fmt, mime

    @property
    def voice_id(self) -> str | None:
        return self._cfg.voice_id

    async def synthesize_full(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> bytes | None:
        # Fall back to the REST one-shot path for synchronous full-blob
        # callers (bridges synthesising voice notes). The WS path is
        # streaming-only.
        return await _tts.synthesize_full(text, self._cfg, language=language)

    async def synthesize_stream(
        self,
        text_chunks: AsyncIterator[str],
        *,
        language: str | None = None,
    ) -> AsyncIterator[bytes]:
        async for audio in _wsstream.synthesize_token_stream(
            text_chunks, self._cfg,
        ):
            if audio:
                yield audio


class LocalPiperTTS(BaseTTS):
    """Offline Piper. One-shot only — yields one chunk per sentence."""

    def __init__(self, voice_id: str | None = None):
        self._voice_id = voice_id

    @property
    def audio_format(self) -> tuple[str, str]:
        return "wav", "audio/wav"

    @property
    def voice_id(self) -> str | None:
        return self._voice_id

    async def synthesize_full(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> bytes | None:
        text = _tts.sanitize_for_tts(text)
        if not text:
            return None
        voice_arg = self._voice_id
        if (
            voice_arg == _piper.DEFAULT_VOICE
            and not os.environ.get("OPENAGENT_PIPER_VOICE")
        ):
            voice_arg = None
        return await _piper.synthesize(text, voice=voice_arg, language=language)


async def resolve_tts(db: Any) -> BaseTTS | None:
    """Pick the active TTS backend.

    Mirrors :func:`openagent.channels.tts.resolve_tts_provider`:

    1. Latest enabled ``kind='tts'`` row → :class:`ElevenLabsWSTTS` when
       the row sets ``metadata.stream_input=true`` AND vendor is
       ElevenLabs; :class:`LiteLLMTTS` otherwise.
    2. Local Piper when bundled package is importable.
    3. ``None`` (caller falls back to text-only).
    """
    cfg = await _tts.resolve_tts_provider(db)
    if cfg is None:
        return None
    if cfg.vendor == _tts.LOCAL_PIPER_VENDOR:
        elog("tts.resolve", kind="local_piper", voice=cfg.voice_id)
        return LocalPiperTTS(voice_id=cfg.voice_id)
    if cfg.vendor == "elevenlabs" and cfg.stream_input:
        elog("tts.resolve", kind="elevenlabs_ws", model=cfg.model_id)
        return ElevenLabsWSTTS(cfg)
    elog("tts.resolve", kind="litellm", vendor=cfg.vendor, model=cfg.model_id)
    return LiteLLMTTS(cfg)


__all__ = [
    "BaseTTS",
    "LiteLLMTTS",
    "ElevenLabsWSTTS",
    "LocalPiperTTS",
    "resolve_tts",
]
