"""Voice-mode turn orchestrator: agent → chunker → TTS → WS audio frames.

Invoked by the gateway when an inbound WS message carries
``input_was_voice=true``. Drives :meth:`Agent.run_stream` through a
:class:`SentenceChunker` and pipes each sentence to LiteLLM TTS,
emitting ``audio_start`` → ``audio_chunk`` ×N → ``audio_end`` and a
final ``RESPONSE`` so text-only clients keep working unchanged.

The RESPONSE frame is the contract that lets the client clear its
"Thinking..." state — every code path through ``run`` MUST send one,
including mid-stream errors, otherwise the voice session hangs forever.
The finally block at the bottom handles that.

Sentence-by-sentence TTS runs through a single background "speaker"
task fed by :class:`asyncio.Queue` so that:

  * status events ("Using ReadFile…") and response sentences both
    enqueue with FIFO ordering — the user hears them in arrival order;
  * the agent stream consumer never blocks waiting for synth to finish
    (it just enqueues and continues), so a slow ElevenLabs call can't
    stall the next delta;
  * cancellation drains cleanly via a sentinel.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

from openagent.channels.base import parse_response_markers
from openagent.channels.tts import (
    LOCAL_PIPER_VENDOR,
    resolve_tts_provider,
    synthesize_stream,
)
from openagent.channels.tts_chunker import SentenceChunker
from openagent.core.logging import elog
from openagent.gateway import protocol as P

logger = logging.getLogger(__name__)


def _status_speech_for(raw: str, seen: set[str]) -> str | None:
    """Convert a status event into a short spoken phrase, deduped by tool.

    Status events come in two shapes (matching what the chat tab parses):

      * JSON: ``{"tool": "ReadFile", "status": "running", ...}``
      * Plain: ``"Using ReadFile..."``

    Returns the phrase to speak, or ``None`` to skip — non-tool statuses
    (e.g. plain "Thinking..."), non-running phases (done/error), and
    repeats of an already-spoken tool all skip. The dedup is per-turn:
    pass a fresh ``seen`` set into each ``run()``.
    """
    text = (raw or "").strip()
    if not text:
        return None

    tool: str | None = None
    status = "running"

    # JSON shape first — chat store does the same try/except.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and parsed.get("tool"):
            tool = str(parsed["tool"])
            status = str(parsed.get("status") or "running")
    except (json.JSONDecodeError, ValueError):
        # Plain "Using X..." text from the legacy code path.
        if text.startswith("Using "):
            tool = text[len("Using "):].rstrip(".").strip()

    if not tool or status != "running":
        return None
    if tool in seen:
        return None
    seen.add(tool)
    # Short phrase keeps the spoken filler out of the way of the actual
    # response. The TTS model handles "ReadFile" / camelCase tool names
    # well enough that a translation layer isn't worth the complexity.
    return f"Using {tool}"


class VoiceTurnOrchestrator:
    def __init__(self, agent: Any, ws_send_json):
        """Args:
            agent: Agent with a configured model + ``.db`` for provider lookup.
            ws_send_json: Async ``(payload: dict) -> None`` writing one WS frame.
        """
        self._agent = agent
        self._db = getattr(agent, "db", None)
        self._send = ws_send_json

    async def run(
        self,
        text: str,
        *,
        client_id: str,
        session_id: str,
        attachments: list[dict] | None = None,
        on_status=None,
        language: str | None = None,
    ) -> dict[str, Any]:
        cfg = await resolve_tts_provider(self._db)
        elog(
            "voice.turn.start",
            session_id=session_id,
            client_id=client_id,
            tts_vendor=(cfg.vendor if cfg else None),
            tts_model=(cfg.model_id if cfg else None),
            input_chars=len(text),
            language=language or "auto",
        )
        if cfg is None:
            # Neither a DB-configured cloud TTS row nor the bundled
            # local Piper fallback is available. Piper ships in core
            # deps now (mirrors faster-whisper for STT) — landing here
            # usually means the user is running a stale install. Tell
            # them to reinstall so the new core dep gets pulled in.
            elog(
                "voice.tts_not_configured",
                level="warning",
                session_id=session_id,
                hint=(
                    "reinstall to pick up bundled Piper (`pip install -e .` from the "
                    "repo, or reinstall the desktop app), or add a kind='tts' Models row"
                ),
            )
            logger.warning(
                "voice mode: no TTS backend available — replies will be text-only "
                "(session_id=%s). Piper-tts is a core dependency now; reinstall "
                "OpenAgent (`pip install -e .` from the repo) to pull it in, or "
                "add a kind='tts' model row in Models for a cloud vendor.",
                session_id,
            )
        elif cfg.vendor == LOCAL_PIPER_VENDOR:
            # Info-level: this is the happy out-of-the-box path now.
            elog(
                "voice.tts_using_local",
                session_id=session_id,
                voice=cfg.voice_id,
            )
        accumulated: list[str] = []
        seq = 0
        audio_started = False
        stream_error: BaseException | None = None
        spoken_tools: set[str] = set()

        # Queue of sentences to speak. ``None`` is the sentinel that
        # tells the speaker task to drain and exit cleanly.
        speak_q: asyncio.Queue[str | None] = asyncio.Queue()
        spoken_count = 0

        async def _ensure_started() -> None:
            nonlocal audio_started
            if audio_started or cfg is None:
                return
            audio_started = True
            # Local Piper writes WAV; cloud vendors default to MP3 via
            # ``DEFAULT_RESPONSE_FORMAT``. The client's AudioQueuePlayer
            # reads ``mime`` to pick the right Blob type so playback
            # works for both — no MP3 re-encode required. ``getattr``
            # rather than attribute access so older test fixtures and
            # any cfg-shaped object missing the field still work.
            fmt = (getattr(cfg, "response_format", None) or "mp3").lower()
            mime = "audio/wav" if fmt == "wav" else "audio/mpeg"
            await self._send({
                "type": P.AUDIO_START,
                "session_id": session_id,
                "format": fmt,
                "voice_id": getattr(cfg, "voice_id", None),
                "mime": mime,
            })

        async def _do_speak(sentence: str) -> None:
            """Synthesize one sentence and forward chunks. Runs on the
            speaker task — never call directly from the agent loop."""
            nonlocal seq, spoken_count
            if cfg is None or not sentence.strip():
                return
            await _ensure_started()
            try:
                async for chunk in synthesize_stream(sentence, cfg, language=language):
                    if not chunk:
                        continue
                    seq += 1
                    await self._send({
                        "type": P.AUDIO_CHUNK,
                        "session_id": session_id,
                        "seq": seq,
                        "data": base64.b64encode(chunk).decode("ascii"),
                    })
                spoken_count += 1
            except Exception as e:  # noqa: BLE001 — WS disconnect mid-turn
                logger.warning("audio_chunk send failed: %s", e)
                elog("voice.audio_chunk.fail", level="warning",
                     session_id=session_id, error=str(e))

        async def _speaker_task() -> None:
            """Drain the speak queue sequentially. One synth at a time so
            chunks arrive at the client in monotonically-increasing seq."""
            while True:
                sentence = await speak_q.get()
                if sentence is None:
                    return
                try:
                    await _do_speak(sentence)
                except Exception as e:  # noqa: BLE001
                    logger.warning("speaker task swallowed exception: %s", e)

        # Only spin up the speaker if TTS is configured — otherwise every
        # enqueue is a no-op and the task just sits idle.
        speaker: asyncio.Task | None = None
        if cfg is not None:
            speaker = asyncio.create_task(_speaker_task())

        def _enqueue(sentence: str) -> None:
            if cfg is None:
                return
            if not sentence or not sentence.strip():
                return
            speak_q.put_nowait(sentence)

        async def _wrapped_on_status(status_text: str) -> None:
            """Forward to the original (WS status frame) and also enqueue
            a spoken summary so the user hears what the agent is doing
            during long tool calls. Deduped by tool name per turn."""
            if on_status is not None:
                try:
                    await on_status(status_text)
                except Exception as e:  # noqa: BLE001
                    logger.debug("forwarded on_status raised: %s", e)
            # Skip the speech extraction entirely when TTS is unconfigured
            # — otherwise we'd populate ``spoken_tools`` for tools we
            # never actually speak, which trips diagnostics + dedup.
            if cfg is None:
                return
            spoken = _status_speech_for(status_text, spoken_tools)
            if spoken:
                elog(
                    "voice.status.spoken",
                    session_id=session_id, phrase=spoken,
                )
                _enqueue(spoken)

        chunker = SentenceChunker()
        try:
            try:
                async for event in self._agent.run_stream(
                    message=text,
                    user_id=client_id,
                    session_id=session_id,
                    attachments=attachments,
                    on_status=_wrapped_on_status,
                ):
                    kind = event.get("kind")
                    # Per-event diagnostic — when the model returns
                    # nothing useful, this lets the user see exactly
                    # what the agent yielded (or didn't) instead of
                    # debugging an empty bubble blind.
                    elog(
                        "voice.agent.event",
                        session_id=session_id,
                        kind=kind,
                        chars=len(event.get("text") or ""),
                    )
                    if kind == "delta":
                        delta = event.get("text") or ""
                        accumulated.append(delta)
                        for sentence in chunker.feed(delta):
                            _enqueue(sentence)
                    elif kind == "iteration_break":
                        # Force-flush a partial sentence across tool turns so
                        # we don't re-narrate it next iteration.
                        chunker.iteration_break()
                    elif kind == "done":
                        if event.get("text") and not accumulated:
                            accumulated.append(event["text"])
                        break
                if tail := chunker.flush():
                    _enqueue(tail)
            except asyncio.CancelledError:
                # Bubble up so the gateway's cancel handler can send its
                # own ERROR frame; we still close audio_end below.
                raise
            except Exception as e:  # noqa: BLE001
                stream_error = e
                logger.warning("voice stream failed mid-turn: %s", e)
                elog("voice.stream.fail", level="warning",
                     session_id=session_id,
                     error_type=type(e).__name__, error=str(e))
        finally:
            # Drain the speaker — sentinel tells it to exit, then we
            # wait briefly so any in-flight synth completes before we
            # close audio_end. Hard cap so a wedged synth can't hang
            # the request indefinitely.
            if speaker is not None:
                speak_q.put_nowait(None)
                try:
                    await asyncio.wait_for(speaker, timeout=20.0)
                except asyncio.TimeoutError:
                    speaker.cancel()
                    logger.warning("speaker task did not drain in 20s — cancelling")
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    logger.warning("speaker task error: %s", e)

            # Always close the audio stream if it was opened, otherwise
            # the client's AudioQueuePlayer waits forever for the tail.
            if audio_started:
                try:
                    await self._send({
                        "type": P.AUDIO_END,
                        "session_id": session_id,
                        "total_chunks": seq,
                    })
                except Exception as e:  # noqa: BLE001
                    logger.warning("audio_end send failed: %s", e)

            # Always send a RESPONSE so the client can clear its
            # ``isProcessing`` state. On a stream error we surface the
            # exception in the response text rather than as an ERROR
            # frame — that way the message lands in the originating
            # session even on older clients that route ERRORs to the
            # active session.
            full_text = "".join(accumulated)
            if not full_text:
                if stream_error is not None:
                    full_text = f"Voice turn failed: {stream_error}"
                else:
                    # Agent finished cleanly but produced no text —
                    # most commonly a tool-only turn that never
                    # produced a final assistant message, or a model
                    # that returned an empty completion. Tell the user
                    # explicitly so they don't see an empty bubble.
                    full_text = (
                        "(No text response — the agent finished without producing "
                        "any output. Check the gateway log for ``voice.agent.event`` "
                        "lines and your model configuration.)"
                    )
                    elog(
                        "voice.empty_response",
                        level="warning",
                        session_id=session_id,
                    )
            clean, attachments_out = parse_response_markers(full_text)
            att_list = [
                {"type": a.type, "path": a.path, "filename": a.filename}
                for a in attachments_out
            ]
            meta_fn = getattr(self._agent, "last_response_meta", None)
            meta = {}
            try:
                meta = meta_fn(session_id) if meta_fn else {}
            except Exception as e:  # noqa: BLE001
                logger.debug("last_response_meta failed: %s", e)
            try:
                await self._send({
                    "type": P.RESPONSE,
                    "text": clean,
                    "session_id": session_id,
                    "attachments": att_list or None,
                    "model": meta.get("model"),
                })
            except Exception as e:  # noqa: BLE001
                logger.warning("voice RESPONSE send failed: %s", e)

            elog(
                "voice.turn.end",
                session_id=session_id,
                response_chars=len(clean),
                audio_chunks=seq,
                spoken_sentences=spoken_count,
                spoken_tools=len(spoken_tools),
                errored=stream_error is not None,
            )

        return {
            "text": clean,
            "attachments": att_list,
            "audio_chunks": seq,
            "spoken_sentences": spoken_count,
            "spoken_tools": len(spoken_tools),
            "errored": stream_error is not None,
        }
