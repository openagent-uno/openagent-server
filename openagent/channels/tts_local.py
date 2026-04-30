"""Local Piper TTS — fallback when no kind='tts' row is configured.

Mirrors the faster-whisper integration in :mod:`openagent.channels.voice`:

* Lazy import — ``piper-tts`` is in the ``[voice]`` extra, not core.
  When it isn't installed, :func:`is_available` returns ``False`` and
  :func:`synthesize` returns ``None`` so the gateway falls through to
  text-only replies (current behaviour, no regression).
* Single in-process voice cache (``_VOICE_CACHE``) — loading the ONNX
  model is the expensive part; once warm, subsequent synth calls are
  fast and run entirely on CPU.
* Voice file is auto-downloaded from the rhasspy/piper-voices HF repo
  on first use into ``~/.cache/openagent/piper/``. ~25 MB for the
  default voice (``en_US-amy-medium``); subsequent calls hit disk.
* Output is WAV/PCM (Piper's native format). The voice pipeline emits
  ``mime: audio/wav`` in ``audio_start`` so the universal client's
  AudioQueuePlayer plays it via ``<audio>`` natively — no MP3
  re-encode dependency added.

Override the default voice with ``OPENAGENT_PIPER_VOICE`` (e.g.
``en_GB-alan-medium``, ``it_IT-paola-medium``).

Voice naming convention from rhasspy/piper-voices:
``<lang>_<COUNTRY>-<voice>-<quality>`` →
``<lang>/<lang>_<COUNTRY>/<voice>/<quality>/<full>.onnx``.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import wave
from pathlib import Path
from typing import Any

from openagent.core.logging import elog

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "en_US-amy-medium"
_HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# ISO-639-1 → preferred Piper voice. Hand-picked from the rhasspy
# catalog (https://huggingface.co/rhasspy/piper-voices/tree/main):
# female "medium" voices where available because they consistently
# scored highest in the project's MOS evals. Without this map, every
# transcribed turn was synthesised with the default English voice — so
# an Italian user got their reply spoken in an obvious American accent
# (the user complaint that triggered this map). Languages absent here
# fall through to ``DEFAULT_VOICE`` so the worst case is still audible
# rather than silent. ``OPENAGENT_PIPER_VOICE`` always overrides — a
# user who explicitly picks a voice keeps it across languages.
LANGUAGE_TO_VOICE: dict[str, str] = {
    "en": "en_US-amy-medium",
    "it": "it_IT-paola-medium",
    "es": "es_ES-mls_9972-low",
    "fr": "fr_FR-siwis-medium",
    "de": "de_DE-thorsten-medium",
    "pt": "pt_BR-faber-medium",
    "nl": "nl_NL-mls_5809-low",
    "ru": "ru_RU-irina-medium",
    "zh": "zh_CN-huayan-medium",
}

# Cache: voice_name → loaded PiperVoice instance. Populated lazily by
# ``_load_voice``. Loaded once per process — synth is the hot path.
_VOICE_CACHE: dict[str, Any] = {}
_LOAD_LOCK = asyncio.Lock()


def _cache_dir() -> Path:
    """Resolve ``~/.cache/openagent/piper`` and ensure it exists."""
    base = Path.home() / ".cache" / "openagent" / "piper"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _resolve_voice_name(voice: str | None, *, language: str | None = None) -> str:
    """Pick the Piper voice id, prioritising explicit > env > language > default.

    ``language`` is an ISO-639-1 hint (typically the same code passed
    to Whisper for transcription). When the user hasn't pinned a
    voice, we look it up in :data:`LANGUAGE_TO_VOICE` so an Italian
    transcription gets ``it_IT-paola-medium`` instead of an Italian
    sentence read with an American accent.
    """
    explicit = (voice or "").strip() if voice else ""
    if explicit:
        return explicit
    env = (os.environ.get("OPENAGENT_PIPER_VOICE") or "").strip()
    if env:
        return env
    lang = (language or "").strip().lower().split("-")[0]
    if lang and lang in LANGUAGE_TO_VOICE:
        return LANGUAGE_TO_VOICE[lang]
    return DEFAULT_VOICE


def _voice_url_path(voice: str) -> str | None:
    """Convert ``en_US-amy-medium`` → ``en/en_US/amy/medium/en_US-amy-medium``.

    Returns ``None`` for malformed names so the caller can degrade
    gracefully instead of raising deep inside an HTTP fetch.
    """
    parts = voice.split("-")
    if len(parts) < 3:
        return None
    lang_country, name = parts[0], parts[1]
    quality = "-".join(parts[2:])
    if "_" not in lang_country:
        return None
    lang = lang_country.split("_", 1)[0]
    return f"{lang}/{lang_country}/{name}/{quality}/{voice}"


def is_available() -> bool:
    """True iff the ``piper`` Python package is importable.

    Cheap (one import attempt) so callers can probe per-request without
    perf concerns. Result is not cached because users sometimes install
    the extra mid-session and we want the next probe to see it.
    """
    try:
        import piper  # noqa: F401
        return True
    except ImportError:
        return False


async def _ensure_voice_files(voice: str) -> Path | None:
    """Make sure ``<voice>.onnx`` and ``<voice>.onnx.json`` exist locally.

    Returns the path to the ONNX model, or ``None`` on any download
    failure (network error, 404, write error). Both files are required
    — the JSON carries phoneme + sample-rate metadata.
    """
    cache = _cache_dir()
    onnx_path = cache / f"{voice}.onnx"
    json_path = cache / f"{voice}.onnx.json"

    if onnx_path.exists() and json_path.exists():
        return onnx_path

    url_path = _voice_url_path(voice)
    if url_path is None:
        elog(
            "tts_local.voice_name_invalid",
            level="warning", voice=voice,
            hint="expected '<lang>_<COUNTRY>-<name>-<quality>'",
        )
        return None

    try:
        import httpx
    except ImportError:
        logger.warning("httpx not available — cannot download Piper voice")
        return None

    async def _fetch(url: str, dest: Path) -> bool:
        try:
            async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
                async with client.stream("GET", url) as resp:
                    if resp.status_code != 200:
                        elog(
                            "tts_local.download_failed",
                            level="warning",
                            url=url, status=resp.status_code,
                        )
                        return False
                    tmp = dest.with_suffix(dest.suffix + ".part")
                    with open(tmp, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=1024 * 64):
                            f.write(chunk)
                    tmp.replace(dest)
                    return True
        except Exception as e:  # noqa: BLE001 — network is flaky, surface as warning
            elog(
                "tts_local.download_exception",
                level="warning",
                url=url, error_type=type(e).__name__, error=str(e),
            )
            return False

    elog("tts_local.download.start", voice=voice)
    onnx_url = f"{_HF_BASE}/{url_path}.onnx"
    json_url = f"{_HF_BASE}/{url_path}.onnx.json"
    ok_onnx = await _fetch(onnx_url, onnx_path)
    if not ok_onnx:
        return None
    ok_json = await _fetch(json_url, json_path)
    if not ok_json:
        # Don't leave a half-downloaded onnx around — next call would
        # think it's cached and skip the json fetch entirely.
        try:
            onnx_path.unlink()
        except OSError:
            pass
        return None
    elog(
        "tts_local.download.done",
        voice=voice, onnx_bytes=onnx_path.stat().st_size,
    )
    return onnx_path


async def _load_voice(voice: str) -> Any | None:
    """Return a cached or freshly-loaded ``PiperVoice``."""
    cached = _VOICE_CACHE.get(voice)
    if cached is not None:
        return cached

    async with _LOAD_LOCK:
        # Re-check after acquiring the lock — concurrent first calls
        # would otherwise both download + load.
        cached = _VOICE_CACHE.get(voice)
        if cached is not None:
            return cached

        try:
            from piper import PiperVoice
        except ImportError:
            return None

        onnx_path = await _ensure_voice_files(voice)
        if onnx_path is None:
            return None

        def _load() -> Any:
            return PiperVoice.load(str(onnx_path))

        try:
            piper_voice = await asyncio.to_thread(_load)
        except Exception as e:  # noqa: BLE001
            elog(
                "tts_local.load_failed",
                level="warning",
                voice=voice,
                error_type=type(e).__name__, error=str(e),
            )
            return None

        _VOICE_CACHE[voice] = piper_voice
        elog("tts_local.load.done", voice=voice)
        return piper_voice


async def synthesize(
    text: str,
    *,
    voice: str | None = None,
    language: str | None = None,
) -> bytes | None:
    """Render ``text`` to WAV bytes via Piper.

    ``language`` is an ISO-639-1 hint forwarded by the gateway from
    the original transcription request — ``it`` / ``es`` / ``de`` /
    etc. When ``voice`` and ``OPENAGENT_PIPER_VOICE`` are both unset,
    the language hint picks a matching voice from
    :data:`LANGUAGE_TO_VOICE`. Without it, an Italian reply got read
    by the default American voice.

    Returns ``None`` when piper isn't installed, the voice file can't
    be obtained, or synthesis throws. The caller (voice_pipeline)
    treats ``None`` the same as today's "no TTS configured" path.
    """
    if not text or not text.strip():
        return None
    voice_name = _resolve_voice_name(voice, language=language)
    piper_voice = await _load_voice(voice_name)
    if piper_voice is None:
        return None

    def _synth() -> bytes:
        # piper-tts >=1.3 API: ``synthesize_wav(text, wav_file)``
        # populates the provided wave-file writer with PCM audio and
        # auto-sets the WAV header from the voice's sample rate.
        # Earlier versions exposed ``synthesize(text, wav_file)`` —
        # not supported anymore (``synthesize`` now returns an
        # AudioChunk iterator). We pin to the new API; pyproject.toml
        # asks for >=1.2 but the supported install path is >=1.3.
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            piper_voice.synthesize_wav(text, wav)
        return buf.getvalue()

    try:
        wav_bytes = await asyncio.to_thread(_synth)
    except Exception as e:  # noqa: BLE001
        elog(
            "tts_local.synth_failed",
            level="warning",
            voice=voice_name, chars=len(text),
            error_type=type(e).__name__, error=str(e),
        )
        return None

    elog(
        "tts_local.synth.done",
        voice=voice_name, chars=len(text), bytes=len(wav_bytes),
    )
    return wav_bytes
