"""Token-in / audio-out TTS for vendors that expose a WebSocket API.

LiteLLM's :func:`aspeech` only wraps each vendor's REST endpoint, which
takes a complete utterance per call. That caps voice-mode time-to-first-
audio at roughly ``LLM-time-to-first-sentence + per-sentence-synth-time``
— ~3–5 s on long replies even with the sub-sentence chunking landed in
the same round.

ElevenLabs (and Cartesia / Azure SSML soon) expose a WebSocket
``/stream-input`` endpoint that accepts a stream of partial text frames
and emits audio chunks as they arrive. Sub-second TTFB. This module is
the vendor-direct surface for those endpoints — opt-in per row via
``metadata.stream_input = true``.

The voice pipeline branches on ``cfg.stream_input``. When True:

* :class:`TurnRunner` drives a token queue from the agent's
  delta loop and feeds it to :func:`synthesize_token_stream`.
* The synth coroutine becomes the "speaker" — no per-sentence chunker,
  no `speak_q`, no per-sentence ``aspeech`` calls.

When False (default for every existing row): the orchestrator uses the
historical per-sentence path through :mod:`openagent.channels.tts`. That
path keeps working unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import TYPE_CHECKING, AsyncIterator

from openagent.core.logging import elog

if TYPE_CHECKING:
    from openagent.channels.tts import TTSConfig

logger = logging.getLogger(__name__)


_ELEVENLABS_WS = "wss://api.elevenlabs.io/v1/text-to-speech/{voice}/stream-input"


def supports_token_stream(cfg: "TTSConfig | None") -> bool:
    """Cheap predicate: does this cfg opt into the WS streaming path?

    Used by the voice pipeline to branch BEFORE the agent loop starts.
    Currently only ElevenLabs has a WS endpoint we wrap; other vendors
    just keep using the per-sentence REST path.
    """
    if cfg is None:
        return False
    if not getattr(cfg, "stream_input", False):
        return False
    vendor = (getattr(cfg, "vendor", "") or "").lower()
    return vendor == "elevenlabs"


async def synthesize_token_stream(
    text_iter: AsyncIterator[str],
    cfg: "TTSConfig",
) -> AsyncIterator[bytes]:
    """Pipe token deltas in, yield audio bytes as the vendor returns them.

    Routes by vendor; raises on connection / protocol failures so the
    caller (turn_runner) can fall back to the per-sentence REST path.
    Returns silently when ``cfg`` doesn't opt into the WS path so the
    same call site can be unconditional.
    """
    if not supports_token_stream(cfg):
        return

    vendor = (cfg.vendor or "").lower()
    if vendor == "elevenlabs":
        async for chunk in _elevenlabs_stream(text_iter, cfg):
            yield chunk
        return

    # Defensive — supports_token_stream gated this above.
    return  # pragma: no cover


async def _elevenlabs_stream(
    text_iter: AsyncIterator[str],
    cfg: "TTSConfig",
) -> AsyncIterator[bytes]:
    """ElevenLabs ``/stream-input`` WebSocket adapter.

    Protocol summary (from
    https://elevenlabs.io/docs/api-reference/text-to-speech-websocket):

    1. Open ``wss://api.elevenlabs.io/v1/text-to-speech/<voice>/stream-input``
       with ``model_id`` and ``output_format`` query args.
    2. Send a BOS frame: ``{"text": " ", "voice_settings": {...},
       "xi_api_key": "..."}``.
    3. Forward each LLM delta as ``{"text": "<delta>",
       "try_trigger_generation": true}``. Optional
       ``flush=true`` between sentences for natural pauses.
    4. Send EOS: ``{"text": ""}``.
    5. Read response frames concurrently. Each is JSON
       ``{"audio": "<b64>", "isFinal": false, "normalizedAlignment":
       {...}}``. Decode ``audio`` and yield raw bytes. ``isFinal: true``
       closes the stream.

    The input-feed and output-read tasks run concurrently — ElevenLabs
    is happy to start emitting audio before the input is complete.

    On any protocol-level failure we log and re-raise so the
    turn_runner falls back to the per-sentence REST path. The
    caller-visible iterator simply terminates early on partial
    success — anything we managed to emit before the error is real
    audio the AudioQueuePlayer can play.
    """
    try:
        import websockets
    except ImportError as e:  # pragma: no cover — declared in pyproject
        raise RuntimeError(
            "websockets package missing — required for ElevenLabs WS streaming"
        ) from e

    voice_id = (cfg.voice_id or "").strip()
    if not voice_id:
        raise ValueError(
            "ElevenLabs WS streaming requires a voice_id — set "
            "metadata.voice_id on the providers row"
        )
    if not cfg.api_key:
        raise ValueError(
            "ElevenLabs WS streaming requires an api_key — set "
            "providers.api_key for the elevenlabs row"
        )

    output_format = (cfg.metadata or {}).get("output_format") or "mp3_44100_128"
    model_id = cfg.model_id or "eleven_flash_v2_5"
    url = _ELEVENLABS_WS.format(voice=voice_id) + (
        f"?model_id={model_id}&output_format={output_format}"
    )

    elog(
        "tts.elevenlabs_ws.connect",
        voice=voice_id, model=model_id, output_format=output_format,
    )

    try:
        ws = await websockets.connect(url, max_size=None)
    except Exception as e:  # noqa: BLE001
        elog(
            "tts.elevenlabs_ws.connect_failed",
            level="warning",
            voice=voice_id, model=model_id,
            error_type=type(e).__name__, error=str(e) or repr(e),
        )
        raise

    # Inner queue that the writer task drains: a single-producer,
    # single-consumer FIFO so we don't have to wedge the iterator
    # consumer into the same coroutine as the WS writer.
    text_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _drain_text_iter() -> None:
        """Forward every delta from the agent into ``text_queue``,
        stripping markdown / emojis / URLs first so ElevenLabs doesn't
        literally pronounce them. Sanitization is per-chunk with
        ``preserve_edges=True`` so the leading/trailing spaces between
        adjacent token chunks survive — without this, the vendor would
        concatenate ``["Hello", " world"]`` into ``"Helloworld"``.
        Per-chunk can't catch markdown that straddles two deltas (e.g.
        ``**`` split across tokens) but the common cases (whole emojis,
        list bullets, inline backticks) work fine."""
        from openagent.channels.tts import sanitize_for_tts

        try:
            async for delta in text_iter:
                if not delta:
                    continue
                clean = sanitize_for_tts(delta, preserve_edges=True)
                if clean:
                    await text_queue.put(clean)
        finally:
            await text_queue.put(None)  # EOS sentinel

    async def _writer() -> None:
        """BOS → text frames → EOS. Closes the WS write side cleanly."""
        bos = {
            "text": " ",
            "voice_settings": _voice_settings(cfg),
            "xi_api_key": cfg.api_key,
        }
        gen_cfg = (cfg.metadata or {}).get("generation_config")
        if gen_cfg is not None:
            bos["generation_config"] = gen_cfg
        await ws.send(json.dumps(bos))

        while True:
            piece = await text_queue.get()
            if piece is None:
                break
            await ws.send(json.dumps(
                {"text": piece, "try_trigger_generation": True}
            ))
        # EOS: empty text closes the stream.
        await ws.send(json.dumps({"text": ""}))

    # Start the writer + drain together. The drain needs the text_iter
    # to actually progress (which only happens once the LLM emits
    # deltas), so both tasks live concurrently for the lifetime of the
    # turn.
    drain_task = asyncio.create_task(_drain_text_iter())
    writer_task = asyncio.create_task(_writer())

    emitted_bytes = 0
    try:
        try:
            async for raw in ws:
                # ElevenLabs sends JSON for normal frames. Binary frames
                # would be raw audio (some endpoints support that mode)
                # but stream-input is JSON-with-base64.
                if isinstance(raw, (bytes, bytearray)):
                    yield bytes(raw)
                    emitted_bytes += len(raw)
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                audio_b64 = msg.get("audio")
                if audio_b64:
                    audio = base64.b64decode(audio_b64)
                    if audio:
                        yield audio
                        emitted_bytes += len(audio)
                if msg.get("isFinal"):
                    break
        finally:
            elog(
                "tts.elevenlabs_ws.done",
                voice=voice_id, bytes=emitted_bytes,
            )
    except Exception as e:  # noqa: BLE001
        elog(
            "tts.elevenlabs_ws.error",
            level="warning",
            voice=voice_id,
            error_type=type(e).__name__, error=str(e) or repr(e),
        )
        raise
    finally:
        # Tear down the writer + drain even on early exit. Don't await
        # to completion — they might be stuck on text_queue.put if the
        # outer loop stopped consuming. cancel + suppress is enough.
        for task in (writer_task, drain_task):
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        try:
            await ws.close()
        except Exception:
            pass


def _voice_settings(cfg: "TTSConfig") -> dict:
    """Build the ElevenLabs voice_settings payload from cfg metadata.

    Defaults match ElevenLabs' "balanced" preset. Override per-row via
    ``metadata.voice_settings`` (a JSON object passed straight through).
    """
    meta = cfg.metadata or {}
    override = meta.get("voice_settings")
    if isinstance(override, dict):
        return override
    return {"stability": 0.5, "similarity_boost": 0.75}
