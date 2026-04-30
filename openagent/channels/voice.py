"""Voice-message transcription shared by all channels.

Two backends, tried in order:

1. **Local ``faster-whisper``** (recommended). Runs offline on the VPS CPU,
   no rate limits, no API key, no cost. Install with
   ``pip install openagent-framework[voice]`` — this pulls in
   ``faster-whisper`` which downloads the model on first use (~150 MB for
   ``base``). Subsequent calls hit the cached model.
2. **OpenAI Whisper API** (fallback). Used when ``faster-whisper`` isn't
   installed but ``OPENAI_API_KEY`` is set. Cloud call, requires network,
   ~$0.006/minute.

If neither works, ``transcribe`` returns ``None`` and the channel falls
back to a clearer in-prompt message ("the user sent a voice message we
couldn't transcribe — ask them to type it") so the agent knows what
happened instead of claiming to be "a text-only agent".

Model size is controlled by ``OPENAGENT_WHISPER_MODEL`` env var. Default
is ``base`` (multilingual, fast, decent accuracy). Bigger options:
``small``, ``medium``, ``large-v3``. ``large-v3`` is ~3 GB and accurate
but slow on CPU.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from openagent.core.logging import elog

try:
    import litellm as _litellm
except ImportError:  # pragma: no cover — optional at install time
    _litellm = None

logger = logging.getLogger(__name__)

_WHISPER_MODEL: object | None = None
_WHISPER_LOCK = asyncio.Lock()
AUDIO_EXTENSIONS = frozenset({".webm", ".ogg", ".mp3", ".wav", ".m4a", ".opus", ".flac"})

def is_audio_file(filename: str | None, content_type: str | None = None) -> bool:
    """Return True when a file should be treated as an audio upload."""
    if content_type and content_type.startswith("audio/"):
        return True
    if not filename:
        return False
    return Path(filename).suffix.lower() in AUDIO_EXTENSIONS


async def transcribe(
    file_path: str,
    db: object | None = None,
    *,
    language: str | None = None,
) -> str | None:
    """Transcribe an audio file, returning the text or ``None`` on failure.

    Resolution order:

    1. ``providers`` row with ``kind='stt'`` + ``framework='litellm'``
       (latest-edited enabled wins) → :func:`litellm.atranscription`.
    2. Local ``faster-whisper`` (if installed) — runs offline on CPU.
    3. ``OPENAI_API_KEY`` env var → OpenAI Whisper API (legacy fallback).

    Returns ``None`` only when *every* path fails. Pass ``db=None`` (or
    omit) to skip the DB step — this is the single function for all
    transcription needs across the codebase.

    ``language`` is an ISO-639-1 hint (``"it"``, ``"en"``, …) forwarded
    to all three backends. Defaults: client query param > metadata >
    ``OPENAGENT_VOICE_LANG`` env var > Whisper auto-detect. Auto-detect
    on multilingual ``base`` is unreliable for short utterances and has
    bitten users (Italian → Cyrillic gibberish), so passing the user's
    language explicitly is strongly recommended.
    """
    if not file_path:
        return None
    if not language:
        language = os.environ.get("OPENAGENT_VOICE_LANG") or None
    elog("voice.transcribe", filename=Path(file_path).name, language=language or "auto")

    if db is not None:
        row = await _resolve_stt_provider(db)
        if row is not None:
            text = await _transcribe_via_litellm(file_path, row, language=language)
            if text:
                return text

    text = await _transcribe_local(file_path, language=language)
    if text is not None:
        return text
    return await _transcribe_openai(file_path, language=language)


async def _load_local_model():
    """Load (and cache) the faster-whisper model. Thread-safe."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    async with _WHISPER_LOCK:
        if _WHISPER_MODEL is not None:
            return _WHISPER_MODEL
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.debug("faster-whisper not installed — skipping local STT")
            return None

        # ``small`` is the smallest model whose language-ID head is
        # reliable on short utterances. ``base`` was the old default
        # and routinely misidentified Italian as Cyrillic when the
        # user spoke a few words. Override with OPENAGENT_WHISPER_MODEL
        # = base | small | medium | large-v3.
        model_size = os.environ.get("OPENAGENT_WHISPER_MODEL", "small")
        # compute_type=int8 runs fastest on CPU with a small accuracy hit.
        def _load():
            return WhisperModel(model_size, device="cpu", compute_type="int8")

        try:
            logger.info("Loading faster-whisper model '%s' (first call may download)...", model_size)
            _WHISPER_MODEL = await asyncio.to_thread(_load)
            logger.info("faster-whisper model loaded")
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load faster-whisper model: %s", e)
            _WHISPER_MODEL = None

    return _WHISPER_MODEL


async def _transcribe_local(file_path: str, *, language: str | None = None) -> str | None:
    model = await _load_local_model()
    if model is None:
        return None
    try:
        def _run() -> str:
            # Passing ``language=None`` tells faster-whisper to auto-detect.
            # For short utterances on the ``base`` model this is unreliable,
            # so callers are encouraged to forward an explicit hint.
            segments, _info = model.transcribe(
                file_path, vad_filter=True, language=language,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()

        text = await asyncio.to_thread(_run)
        if text:
            logger.info(
                "faster-whisper transcribed %d chars (lang=%s)",
                len(text), language or "auto",
            )
            return text
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("faster-whisper transcription failed: %s", e)
        return None


async def _transcribe_openai(file_path: str, *, language: str | None = None) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        import httpx
    except ImportError:
        logger.debug("httpx not available — skipping OpenAI STT fallback")
        return None

    try:
        data = {"model": "whisper-1"}
        if language:
            data["language"] = language
        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(file_path, "rb") as f:
                resp = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (Path(file_path).name, f, "audio/ogg")},
                    data=data,
                )
            resp.raise_for_status()
            text = (resp.json().get("text") or "").strip()
            if text:
                logger.info(
                    "OpenAI Whisper transcribed %d chars (lang=%s)",
                    len(text), language or "auto",
                )
                return text
            return None
    except Exception as e:  # noqa: BLE001
        logger.warning("OpenAI Whisper transcription failed: %s", e)
        return None


def backend_available() -> bool:
    """Return True if at least one transcription backend is importable."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        pass
    return bool(os.environ.get("OPENAI_API_KEY"))


async def _resolve_stt_provider(db: object) -> dict[str, Any] | None:
    """Find the latest enabled ``kind='stt'`` model row joined with its provider."""
    fn = getattr(db, "latest_audio_model", None)
    if fn is None:
        return None
    try:
        return await fn("stt")
    except Exception as e:  # noqa: BLE001
        logger.debug("STT lookup failed: %s", e)
        return None


async def _transcribe_via_litellm(
    file_path: str,
    row: dict[str, Any],
    *,
    language: str | None = None,
) -> str | None:
    """Dispatch to :func:`litellm.atranscription` for a resolved model row.

    ``language`` from the caller wins over ``metadata.language`` so a
    user-supplied override (Settings → Voice or env) takes effect even
    when the row carries a default.
    """
    if _litellm is None:
        logger.warning("litellm not installed — DB-configured STT unavailable")
        return None
    vendor = str(row.get("provider_name") or "").strip().lower()
    model_id = (row.get("model") or "").strip()
    if not vendor or not model_id:
        return None
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    effective_lang = language or meta.get("language") or None
    try:
        with open(file_path, "rb") as f:
            resp = await _litellm.atranscription(
                model=f"{vendor}/{model_id}",
                file=f,
                api_key=row.get("api_key") or None,
                api_base=row.get("base_url") or None,
                language=effective_lang,
                prompt=meta.get("prompt") or None,
            )
    except FileNotFoundError:
        return None
    except Exception as e:  # noqa: BLE001 — surface as soft fail
        logger.warning("LiteLLM atranscription (%s) failed: %s", vendor, e)
        return None
    text = (getattr(resp, "text", None) or "").strip()
    if text:
        elog(
            "voice.transcribe.litellm",
            vendor=vendor, model=model_id, chars=len(text),
            language=effective_lang or "auto",
        )
        return text
    return None
