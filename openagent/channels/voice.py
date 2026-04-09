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

logger = logging.getLogger(__name__)

_WHISPER_MODEL: object | None = None
_WHISPER_LOCK = asyncio.Lock()


async def transcribe(file_path: str) -> str | None:
    """Transcribe a voice file, returning the text or ``None`` on failure.

    Tries ``faster-whisper`` locally first, then the OpenAI Whisper API.
    Safe to call without either backend installed — returns ``None``.
    """
    if not file_path or not Path(file_path).exists():
        return None

    # 1. Local faster-whisper
    text = await _transcribe_local(file_path)
    if text is not None:
        return text

    # 2. OpenAI API fallback
    text = await _transcribe_openai(file_path)
    if text is not None:
        return text

    return None


# ── local (faster-whisper) ─────────────────────────────────────────────

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

        model_size = os.environ.get("OPENAGENT_WHISPER_MODEL", "base")
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


async def _transcribe_local(file_path: str) -> str | None:
    model = await _load_local_model()
    if model is None:
        return None
    try:
        def _run() -> str:
            segments, _info = model.transcribe(file_path, vad_filter=True)
            return " ".join(seg.text.strip() for seg in segments).strip()

        text = await asyncio.to_thread(_run)
        if text:
            logger.info("faster-whisper transcribed %d chars", len(text))
            return text
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("faster-whisper transcription failed: %s", e)
        return None


# ── OpenAI API fallback ────────────────────────────────────────────────

async def _transcribe_openai(file_path: str) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        import httpx
    except ImportError:
        logger.debug("httpx not available — skipping OpenAI STT fallback")
        return None

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(file_path, "rb") as f:
                resp = await client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (Path(file_path).name, f, "audio/ogg")},
                    data={"model": "whisper-1"},
                )
            resp.raise_for_status()
            text = (resp.json().get("text") or "").strip()
            if text:
                logger.info("OpenAI Whisper transcribed %d chars", len(text))
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
