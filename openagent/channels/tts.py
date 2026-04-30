"""Text-to-speech via LiteLLM (vendor-agnostic) + local Piper fallback.

Two backends, picked in order:

1. **DB-configured cloud TTS via LiteLLM.** Mirrors the Agno pattern on
   the LLM side: one ``framework='litellm'`` adapter dispatches to many
   vendors (OpenAI TTS, ElevenLabs, Azure, Groq, Vertex AI, …) via
   :func:`litellm.aspeech`. The provider row's ``name`` is the vendor,
   ``metadata.model_id`` and ``metadata.voice_id`` carry the rest.
   Latest-edited enabled row wins.
2. **Local Piper (offline).** When no row is configured but
   :mod:`openagent.channels.tts_local` reports the package is available,
   ``resolve_tts_provider`` returns a synthetic ``TTSConfig`` with
   ``vendor=LOCAL_PIPER_VENDOR``. Bundled in the core deps (mirrors
   faster-whisper for STT) so voice mode plays audio out of the box.

Neither available → ``resolve_tts_provider`` returns ``None`` and the
voice pipeline degrades to text-only replies (the historical behaviour).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, AsyncIterator

from openagent.channels import tts_local
from openagent.core.logging import elog

try:
    import litellm as _litellm
except ImportError:  # pragma: no cover — optional at install time
    _litellm = None

logger = logging.getLogger(__name__)

# Sentinel vendor name used by the local-Piper fallback path. Picked
# with a leading underscore so it can't collide with a real LiteLLM
# vendor row even if someone registers a provider literally named
# "piper" in the future.
LOCAL_PIPER_VENDOR = "_local_piper"


DEFAULT_MODEL_BY_VENDOR = {
    "openai": "tts-1",
    "elevenlabs": "eleven_flash_v2_5",
    "azure": "tts-1",
    "groq": "playai-tts",
    "vertex_ai": "text-to-speech",
}
DEFAULT_VOICE_BY_VENDOR = {
    "openai": "alloy",
    "elevenlabs": "21m00Tcm4TlvDq8ikWAM",  # Rachel
}
# mp3 decodes natively in browsers + React Native; supported by every vendor.
DEFAULT_RESPONSE_FORMAT = "mp3"


@dataclass(frozen=True)
class TTSConfig:
    """Resolved TTS configuration — vendor-agnostic."""
    vendor: str               # ``openai`` / ``elevenlabs`` / ``azure`` / …
    model_id: str             # ``tts-1`` / ``eleven_flash_v2_5`` / …
    voice_id: str | None      # vendor-specific identifier or preset name
    api_key: str | None       # may be None for cloud providers using ADC
    base_url: str | None
    response_format: str = DEFAULT_RESPONSE_FORMAT
    speed: float | None = None
    metadata: dict[str, Any] | None = None  # passthrough for vendor-specific knobs

    @property
    def litellm_model(self) -> str:
        """Build the ``<vendor>/<model>`` string LiteLLM consumes."""
        return f"{self.vendor}/{self.model_id}"


async def resolve_tts_provider(db: Any) -> TTSConfig | None:
    """Resolve the active TTS model from the unified ``models`` table.

    Picks any enabled row with ``kind='tts'``, latest-edited wins,
    joined with the provider for credentials. When no row is configured
    AND Piper is importable, returns a synthetic local-Piper config so
    voice mode plays audio without manual setup. Returns ``None`` only
    when neither path is available — caller falls back to text-only.
    """
    row = None
    if db is not None:
        try:
            row = await db.latest_audio_model("tts")
        except Exception as e:  # noqa: BLE001 — db wiring varies in tests
            logger.debug("resolve_tts_provider: latest_audio_model failed: %s", e)
            row = None

    if row is not None:
        vendor = (row.get("provider_name") or "").strip().lower()
        model_id = (row.get("model") or "").strip()
        if vendor and model_id:
            meta = row.get("metadata") or {}
            voice_id = (meta.get("voice_id") or DEFAULT_VOICE_BY_VENDOR.get(vendor) or None)
            if isinstance(voice_id, str):
                voice_id = voice_id.strip() or None
            return TTSConfig(
                vendor=vendor,
                model_id=model_id,
                voice_id=voice_id,
                api_key=(row.get("api_key") or "").strip() or None,
                base_url=row.get("base_url") or None,
                response_format=meta.get("response_format") or DEFAULT_RESPONSE_FORMAT,
                speed=_as_float(meta.get("speed")),
                metadata=dict(meta),
            )

    # No DB-configured cloud TTS — try the bundled local fallback so
    # users get audio out of the box (mirrors faster-whisper for STT).
    if tts_local.is_available():
        return TTSConfig(
            vendor=LOCAL_PIPER_VENDOR,
            model_id="piper",
            voice_id=tts_local._resolve_voice_name(None),
            api_key=None,
            base_url=None,
            response_format="wav",
            speed=None,
            metadata={"local": True},
        )
    return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_kwargs(text: str, cfg: TTSConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": cfg.litellm_model,
        "input": text,
        "response_format": cfg.response_format,
    }
    if cfg.voice_id:
        kwargs["voice"] = cfg.voice_id
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.base_url:
        kwargs["api_base"] = cfg.base_url
    if cfg.speed is not None:
        kwargs["speed"] = cfg.speed
    return kwargs


async def synthesize_stream(
    text: str,
    cfg: TTSConfig,
    *,
    timeout: float = 30.0,
    language: str | None = None,
) -> AsyncIterator[bytes]:
    """Yield audio bytes for ``text``.

    Routes to one of two backends based on ``cfg.vendor``:

    * ``LOCAL_PIPER_VENDOR``: render WAV via Piper (offline, on-CPU).
      Yielded as a single chunk because Piper writes the whole WAV at
      once — the AudioQueuePlayer handles the rest.
    * Anything else: cloud TTS via :func:`litellm.aspeech`. LiteLLM
      returns an HTTPX-backed response that exposes ``aiter_bytes()``
      so vendors that chunk audio (ElevenLabs ``/stream``, OpenAI
      ``stream_format``) deliver progressively. Vendors that ship the
      whole file in one body yield it as a single chunk.

    ``language`` is an ISO-639-1 hint (``"it"`` / ``"es"`` / …)
    propagated from the transcription request. Currently consumed
    only by the local Piper path to swap in a language-matched voice
    so an Italian reply isn't read in an American accent. Cloud TTS
    voices are vendor-specific identifiers (``alloy`` / ``Rachel`` / …)
    that don't map cleanly to language codes — those callers should
    keep configuring per-row voice ids in Models.

    On any error the iterator terminates silently; the caller treats
    a short stream as graceful degradation rather than a crash.
    """
    if not text or not text.strip():
        return

    if cfg.vendor == LOCAL_PIPER_VENDOR:
        elog(
            "tts.stream.start",
            vendor=cfg.vendor, model=cfg.model_id, chars=len(text),
            language=language or "auto",
        )
        # ``cfg.voice_id`` carries the env / default voice. When the
        # user pinned one via ``OPENAGENT_PIPER_VOICE`` we want it to
        # win across languages, so we hand the cfg voice to
        # ``synthesize`` (which honours explicit > env > language >
        # default). When neither was pinned, ``cfg.voice_id`` equals
        # ``DEFAULT_VOICE`` and the language hint takes over inside
        # ``synthesize``. The ``cfg.voice_id == DEFAULT_VOICE`` check
        # below distinguishes the two so a default-cfg request can
        # still pick up the language voice.
        voice_arg = cfg.voice_id
        if voice_arg == tts_local.DEFAULT_VOICE and not os.environ.get("OPENAGENT_PIPER_VOICE"):
            voice_arg = None
        wav_bytes = await tts_local.synthesize(
            text, voice=voice_arg, language=language,
        )
        if wav_bytes:
            elog("tts.stream.done", vendor=cfg.vendor, bytes=len(wav_bytes))
            yield wav_bytes
        return

    if _litellm is None:
        return
    elog("tts.stream.start", vendor=cfg.vendor, model=cfg.model_id, chars=len(text))
    response = await _safe_aspeech(text, cfg, timeout)
    if response is None:
        return
    try:
        emitted = 0
        async for chunk in response.aiter_bytes(chunk_size=16 * 1024):
            if chunk:
                emitted += len(chunk)
                yield chunk
        elog("tts.stream.done", vendor=cfg.vendor, bytes=emitted)
    except Exception as e:  # noqa: BLE001
        logger.warning("LiteLLM aiter_bytes (%s) failed: %s", cfg.vendor, e)
    finally:
        try:
            await response.aclose()
        except Exception:
            pass


async def synthesize_full(
    text: str,
    cfg: TTSConfig,
    *,
    timeout: float = 60.0,
    language: str | None = None,
) -> bytes | None:
    """Synthesize ``text`` and return the full audio blob (for Telegram).

    Returns ``None`` on any failure — caller should fall through to a
    text reply. Routes to Piper when ``cfg`` is a local-fallback
    config so the bridges' "synthesize and post" path benefits from
    the same out-of-the-box-audio fix as the WS streaming path.

    ``language`` matches :func:`synthesize_stream` — an ISO-639-1
    code that lets Piper pick a language-matched voice. Cloud TTS
    rows ignore it.
    """
    if not text or not text.strip():
        return None
    if cfg.vendor == LOCAL_PIPER_VENDOR:
        voice_arg = cfg.voice_id
        if voice_arg == tts_local.DEFAULT_VOICE and not os.environ.get("OPENAGENT_PIPER_VOICE"):
            voice_arg = None
        return await tts_local.synthesize(
            text, voice=voice_arg, language=language,
        )
    if _litellm is None:
        return None
    response = await _safe_aspeech(text, cfg, timeout)
    if response is None:
        return None
    try:
        return await response.aread()
    except Exception as e:  # noqa: BLE001
        logger.warning("LiteLLM aread (%s) failed: %s", cfg.vendor, e)
        return None
    finally:
        try:
            await response.aclose()
        except Exception:
            pass


async def _safe_aspeech(text: str, cfg: TTSConfig, timeout: float):
    try:
        return await _litellm.aspeech(timeout=timeout, **_build_kwargs(text, cfg))
    except Exception as e:  # noqa: BLE001 — soft fail, caller falls through
        logger.warning("LiteLLM aspeech (%s) failed: %s", cfg.vendor, e)
        return None
