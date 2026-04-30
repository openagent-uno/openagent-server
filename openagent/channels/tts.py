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
import re
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


DEFAULT_VOICE_BY_VENDOR = {
    "openai": "alloy",
    "elevenlabs": "21m00Tcm4TlvDq8ikWAM",  # Rachel
}
# mp3 decodes natively in browsers + React Native; supported by every vendor.
DEFAULT_RESPONSE_FORMAT = "mp3"


# ── TTS text sanitizer ─────────────────────────────────────────────────
#
# Local Piper (and most vendor TTS models) literally pronounces every
# character it receives — emojis become "smiling face with smiling eyes",
# markdown asterisks become "asterisk asterisk bold asterisk asterisk",
# code fences become "backtick backtick backtick", and bare URLs become a
# tedious letter-by-letter spelling. That makes the spoken reply feel
# unhinged. ``sanitize_for_tts`` strips anything that wouldn't be spoken
# in a natural reading of the text:
#
#   * markdown formatting (bold, italic, strikethrough, headers, lists,
#     blockquotes, inline code) — keep the inner text, drop the markers
#   * links: ``[text](url)`` → ``text``;  ``![alt](url)`` → ``alt``
#   * bare URLs and HTML tags → dropped entirely
#   * code fences are already replaced by SentenceChunker with the
#     "Code shown on screen." placeholder, but we still strip stray
#     leftover backticks defensively
#   * emojis and other pictographic symbols — dropped
#   * pipe-separated table syntax (``| a | b |``) → spaces
#   * collapsed whitespace
#
# Idempotent and safe to call on already-clean text. Pure-Python regex —
# no external dependency.

_EMOJI_RE = re.compile(
    "["                              # any of these unicode ranges →
    "\U0001F300-\U0001F5FF"          # symbols & pictographs
    "\U0001F600-\U0001F64F"          # emoticons
    "\U0001F680-\U0001F6FF"          # transport & map
    "\U0001F700-\U0001F77F"          # alchemical
    "\U0001F780-\U0001F7FF"          # geometric shapes ext
    "\U0001F800-\U0001F8FF"          # supplemental arrows-c
    "\U0001F900-\U0001F9FF"          # supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"          # chess symbols
    "\U0001FA70-\U0001FAFF"          # symbols & pictographs ext-a
    "\U00002600-\U000026FF"          # misc symbols (☀ ☁ ★ …)
    "\U00002700-\U000027BF"          # dingbats (✓ ✗ ❤ …)
    "\U0001F1E6-\U0001F1FF"          # regional indicators (flags)
    "\U0000FE0F"                     # variation selector-16
    "\U0000200D"                     # zero-width joiner (multi-codepoint emoji)
    "]+",
    flags=re.UNICODE,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+", flags=re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^\s*>+\s?", flags=re.MULTILINE)
_LIST_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", flags=re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)|(?<!_)_([^_\n]+)_(?!_)")
_STRIKE_RE = re.compile(r"~~?([^~\n]+)~~?")
_TABLE_PIPE_RE = re.compile(r"\s*\|\s*")
_WHITESPACE_RE = re.compile(r"\s+")
# Punctuation that TTS engines speak literally when they appear alone or
# as runs (``— • ★`` etc.). We keep the everyday ones (``. , ! ? ; :``)
# because they carry prosody.
_NOISE_PUNCT_RE = re.compile(r"[*_~`#|>•·★☆►▶◆◇■□●○]+")


def sanitize_for_tts(text: str, *, preserve_edges: bool = False) -> str:
    """Strip markdown / emoji / URLs from ``text`` so TTS reads it naturally.

    Idempotent. Empty input returns ``""``. By default the result has
    single-space-separated tokens with leading/trailing whitespace
    trimmed.

    Set ``preserve_edges=True`` for per-chunk sanitization on a
    streaming pipeline (e.g. ElevenLabs WS) where leading/trailing
    spaces carry the gap between adjacent tokens — without this an
    input of ``[\"Hello\", \" world\"]`` would collapse to
    ``\"Helloworld\"`` once the vendor concatenates frames.
    """
    if not text:
        return ""
    s = text
    # Remove code blocks first so a stray ``` inside doesn't confuse the
    # inline-code regex below. The chunker already does this for the
    # streaming path but bridges' synthesize_full path goes around it.
    s = _CODE_BLOCK_RE.sub(" ", s)
    s = _INLINE_CODE_RE.sub(r"\1", s)
    # Images BEFORE links — same regex shape, image rule is more specific.
    s = _IMAGE_RE.sub(r"\1", s)
    s = _LINK_RE.sub(r"\1", s)
    s = _URL_RE.sub(" ", s)
    s = _HTML_TAG_RE.sub(" ", s)
    s = _HEADER_RE.sub("", s)
    s = _BLOCKQUOTE_RE.sub("", s)
    s = _LIST_BULLET_RE.sub("", s)
    s = _BOLD_RE.sub(lambda m: m.group(1) or m.group(2), s)
    s = _ITALIC_RE.sub(lambda m: m.group(1) or m.group(2), s)
    s = _STRIKE_RE.sub(r"\1", s)
    s = _TABLE_PIPE_RE.sub(" ", s)
    s = _EMOJI_RE.sub("", s)
    s = _NOISE_PUNCT_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s)
    if not preserve_edges:
        s = s.strip()
    return s


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
    # When True the turn_runner opens a token-in/audio-out WebSocket
    # to the vendor (currently only ElevenLabs implements one) and
    # streams agent deltas directly into it. Beats the per-sentence
    # REST path on TTFB by ~2–5 s on long replies. Opt-in per row via
    # ``metadata.stream_input = true`` so existing rows behave exactly
    # like before this field landed.
    stream_input: bool = False

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
            # ``stream_input`` opts the row into the WebSocket
            # token-in/audio-out path (currently only ElevenLabs has
            # one; gracefully ignored for other vendors so the row
            # falls back to LiteLLM's REST aspeech).
            stream_input = bool(meta.get("stream_input"))
            return TTSConfig(
                vendor=vendor,
                model_id=model_id,
                voice_id=voice_id,
                api_key=(row.get("api_key") or "").strip() or None,
                base_url=row.get("base_url") or None,
                response_format=meta.get("response_format") or DEFAULT_RESPONSE_FORMAT,
                speed=_as_float(meta.get("speed")),
                metadata=dict(meta),
                stream_input=stream_input,
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

    Markdown markers, emojis, code fences, URLs and HTML are stripped
    via :func:`sanitize_for_tts` before synthesis — Piper (and most
    cloud vendors) literally pronounce these characters otherwise.
    """
    text = sanitize_for_tts(text)
    if not text:
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

    Same :func:`sanitize_for_tts` pre-processing as the streaming
    path — bridges that synthesise the full reply benefit from the
    same markdown/emoji stripping.
    """
    text = sanitize_for_tts(text)
    if not text:
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
