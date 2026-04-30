"""Unified turn runner: agent → DELTA frames → optional TTS sidecar → RESPONSE.

Single entry point for any chat or voice turn flowing through the
Gateway WebSocket. Used by both the typed-text path and the
voice-input path; the only difference is the ``speak`` flag.

Frame protocol (always):

  ``DELTA`` × N  →  ``RESPONSE``

When ``speak=True`` the same loop ALSO produces audio frames in
parallel:

  ``DELTA`` × N
  ``AUDIO_START``
    ``AUDIO_CHUNK`` × N
  ``AUDIO_END``
                   ``RESPONSE``

The agent's text deltas drive both surfaces — text goes straight to
``DELTA``, and (when speaking) gets fed through
:class:`SentenceChunker` and the TTS engine to produce
``AUDIO_CHUNK`` bytes. A single background "speaker" task synthesises
sentences serially so audio chunks reach the client with monotonic
``seq``.

Bridges (Telegram/Discord/WhatsApp) reach this code via the gateway
WS just like the universal app does — they all share the exact same
server-side path now.

The ``RESPONSE`` frame is the contract that lets clients clear their
"Thinking…" state — every code path through ``run`` MUST send one,
including mid-stream errors. The ``finally`` block at the bottom
enforces that.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from typing import Any

from openagent.channels.base import parse_response_markers, parse_status_event
from openagent.channels.tts import (
    LOCAL_PIPER_VENDOR,
    resolve_tts_provider,
    synthesize_stream,
)
from openagent.channels.tts_chunker import SentenceChunker
from openagent.channels.tts_streaming import (
    supports_token_stream,
    synthesize_token_stream,
)
from openagent.core.logging import elog
from openagent.gateway import protocol as P

logger = logging.getLogger(__name__)


def _status_speech_for(raw: str, seen: set[str]) -> str | None:
    """Convert a status event into a short spoken phrase, deduped by tool.

    Returns the phrase to speak, or ``None`` to skip — non-tool statuses
    (e.g. plain "Thinking..."), non-running phases (done/error), and
    repeats of an already-spoken tool all skip. The dedup is per-turn:
    pass a fresh ``seen`` set into each ``run()``.
    """
    evt = parse_status_event(raw)
    if evt is None or evt.status != "running" or evt.tool in seen:
        return None
    seen.add(evt.tool)
    # Short phrase — the TTS model handles "ReadFile" / camelCase tool
    # names well enough that a translation layer isn't worth the
    # complexity. Bare (no trailing ellipsis) reads naturally over Piper.
    return f"Using {evt.tool}"


class TurnRunner:
    """One agent turn over a WebSocket. Text-only or text + spoken audio.

    Construct with the agent and a WS-send coroutine; call :meth:`run`
    once per inbound message. All error paths still send a final
    ``RESPONSE`` frame so the client can clear its "Thinking…" state.
    """

    def __init__(self, agent, ws_send_json):
        self._agent = agent
        # Surface ``agent.db`` for resolve_tts_provider; agents in test
        # fixtures may not set it yet, so guard.
        self._db = getattr(agent, "db", None)
        self._send = ws_send_json

    async def run(
        self,
        text: str,
        *,
        client_id: str,
        session_id: str,
        speak: bool = False,
        attachments: list[dict] | None = None,
        on_status=None,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Run one turn. Always emits DELTA + RESPONSE; emits AUDIO_*
        frames too when ``speak=True``.

        ``language`` is an ISO-639-1 hint forwarded to the TTS engine
        so Piper picks a language-matched voice. Ignored when
        ``speak=False``.

        Returns a summary dict — useful for the voice path's audio
        chunk count, harmless to ignore for text-only callers.
        """
        cfg = await resolve_tts_provider(self._db) if speak else None
        elog(
            "turn.start",
            session_id=session_id,
            client_id=client_id,
            speak=speak,
            tts_vendor=(cfg.vendor if cfg else None),
            tts_model=(cfg.model_id if cfg else None),
            input_chars=len(text),
            language=language or "auto",
        )
        if speak and cfg is None:
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
        elif speak and cfg is not None and cfg.vendor == LOCAL_PIPER_VENDOR:
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

        async def _send_delta(delta: str) -> None:
            try:
                await self._send({
                    "type": P.DELTA,
                    "text": delta,
                    "session_id": session_id,
                })
            except Exception as e:  # noqa: BLE001 — WS dead mid-turn
                logger.debug("delta send failed: %s", e)

        async def _ensure_audio_started() -> None:
            nonlocal audio_started
            if audio_started or cfg is None:
                return
            audio_started = True
            # Local Piper writes WAV; cloud vendors default to MP3 via
            # ``DEFAULT_RESPONSE_FORMAT``. The client's AudioQueuePlayer
            # reads ``mime`` to pick the right Blob type so playback
            # works for both — no MP3 re-encode required.
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
            await _ensure_audio_started()
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

        # Only spin up the speaker if we'd actually use it (speak=True
        # AND TTS is configured). Otherwise every enqueue is a no-op.
        speaker: asyncio.Task | None = None
        if cfg is not None:
            speaker = asyncio.create_task(_speaker_task())

        def _enqueue_sentence(sentence: str) -> None:
            if cfg is None:
                return
            if not sentence or not sentence.strip():
                return
            speak_q.put_nowait(sentence)

        async def _wrapped_on_status(status_text: str) -> None:
            """Forward to the original on_status (status WS frame) and,
            in voice mode, also enqueue a spoken summary so the user
            hears what the agent is doing during long tool calls.
            Deduped by tool name per turn."""
            if on_status is not None:
                try:
                    await on_status(status_text)
                except Exception as e:  # noqa: BLE001
                    logger.debug("forwarded on_status raised: %s", e)
            if cfg is None:
                return
            spoken = _status_speech_for(status_text, spoken_tools)
            if spoken:
                elog("voice.status.spoken", session_id=session_id, phrase=spoken)
                _enqueue_sentence(spoken)

        # WS-streaming TTS path: only relevant when we're speaking AND
        # the configured vendor supports token-in/audio-out (today only
        # ElevenLabs). Otherwise we fall through to the per-sentence
        # chunker path or the text-only path.
        ws_path_active = supports_token_stream(cfg) if cfg is not None else False
        ws_emitted_audio = False
        ws_failed = False

        chunker = SentenceChunker() if cfg is not None else None
        try:
            try:
                if ws_path_active:
                    # Token-stream branch: feed agent deltas directly
                    # into the vendor WebSocket via a queue, run the
                    # WS consumer concurrently. On any error before
                    # audio starts, fall back to the chunker path with
                    # the accumulated text so the user still gets audio.
                    text_q: asyncio.Queue[str | None] = asyncio.Queue()

                    async def _text_iter():
                        while True:
                            piece = await text_q.get()
                            if piece is None:
                                return
                            yield piece

                    async def _drain_ws_audio() -> None:
                        nonlocal seq, ws_emitted_audio, ws_failed
                        try:
                            async for audio in synthesize_token_stream(_text_iter(), cfg):
                                if not audio:
                                    continue
                                await _ensure_audio_started()
                                seq += 1
                                ws_emitted_audio = True
                                try:
                                    await self._send({
                                        "type": P.AUDIO_CHUNK,
                                        "session_id": session_id,
                                        "seq": seq,
                                        "data": base64.b64encode(audio).decode("ascii"),
                                    })
                                except Exception as send_err:  # noqa: BLE001
                                    logger.warning(
                                        "audio_chunk send failed (ws path): %s", send_err,
                                    )
                                    return
                        except Exception as e:  # noqa: BLE001
                            ws_failed = True
                            elog(
                                "voice.tts_fallback_to_rest",
                                level="warning",
                                session_id=session_id,
                                reason="ws_stream_failed",
                                error_type=type(e).__name__,
                                error=str(e) or repr(e),
                            )

                    audio_task = asyncio.create_task(_drain_ws_audio())
                    try:
                        async for event in self._agent.run_stream(
                            message=text,
                            user_id=client_id,
                            session_id=session_id,
                            attachments=attachments,
                            on_status=_wrapped_on_status,
                        ):
                            kind = event.get("kind")
                            elog(
                                "turn.agent.event",
                                session_id=session_id,
                                kind=kind,
                                chars=len(event.get("text") or ""),
                            )
                            if kind == "delta":
                                delta = event.get("text") or ""
                                if not delta:
                                    continue
                                accumulated.append(delta)
                                await _send_delta(delta)
                                await text_q.put(delta)
                            elif kind == "done":
                                if event.get("text") and not accumulated:
                                    accumulated.append(event["text"])
                                    await _send_delta(event["text"])
                                    await text_q.put(event["text"])
                                break
                    finally:
                        await text_q.put(None)  # EOS

                    # Wait for the WS to finish draining — bounded so a
                    # vendor stall can't hang the whole turn.
                    try:
                        await asyncio.wait_for(audio_task, timeout=30.0)
                    except asyncio.TimeoutError:
                        audio_task.cancel()
                        ws_failed = True
                        elog(
                            "voice.tts_fallback_to_rest",
                            level="warning",
                            session_id=session_id,
                            reason="ws_drain_timeout",
                        )

                    # If the WS handshake failed BEFORE any audio was
                    # emitted, transparently fall back to the per-
                    # sentence chunker path with the accumulated text.
                    if ws_failed and not ws_emitted_audio and accumulated and chunker is not None:
                        elog(
                            "voice.tts_fallback_to_rest.engaging",
                            session_id=session_id,
                            chars=len("".join(accumulated)),
                        )
                        for sentence in chunker.feed("".join(accumulated)):
                            _enqueue_sentence(sentence)
                        if tail := chunker.flush():
                            _enqueue_sentence(tail)
                else:
                    # Unified text-only / per-sentence-TTS path. When
                    # ``cfg is None`` the chunker block is skipped and
                    # the loop reduces to "agent → DELTA frames".
                    async for event in self._agent.run_stream(
                        message=text,
                        user_id=client_id,
                        session_id=session_id,
                        attachments=attachments,
                        on_status=_wrapped_on_status,
                    ):
                        kind = event.get("kind")
                        elog(
                            "turn.agent.event",
                            session_id=session_id,
                            kind=kind,
                            chars=len(event.get("text") or ""),
                        )
                        if kind == "delta":
                            delta = event.get("text") or ""
                            if not delta:
                                continue
                            accumulated.append(delta)
                            await _send_delta(delta)
                            if chunker is not None:
                                for sentence in chunker.feed(delta):
                                    _enqueue_sentence(sentence)
                        elif kind == "iteration_break":
                            # Force-flush a partial sentence across tool
                            # turns so we don't re-narrate it next iteration.
                            # Text-only callers ignore this (chunker is None).
                            if chunker is not None:
                                chunker.iteration_break()
                        elif kind == "done":
                            if event.get("text") and not accumulated:
                                accumulated.append(event["text"])
                                await _send_delta(event["text"])
                            break
                    if chunker is not None:
                        if tail := chunker.flush():
                            _enqueue_sentence(tail)
            except asyncio.CancelledError:
                # Bubble up so the gateway's cancel handler can send
                # its own ERROR frame; we still close audio_end below.
                raise
            except Exception as e:  # noqa: BLE001
                stream_error = e
                logger.warning("turn stream failed mid-turn: %s", e)
                elog("turn.stream.fail", level="warning",
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

            # Always close the audio stream if it was opened.
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
            # exception in the response text rather than as a bare
            # ERROR frame (older clients route ERROR globally and
            # might miss the originating session).
            full_text = "".join(accumulated)
            if not full_text:
                if stream_error is not None:
                    full_text = f"Error: {stream_error}"
                elif speak:
                    # Voice-mode empty turns: keep the readable hint so
                    # the user doesn't see an empty bubble after speaking.
                    full_text = (
                        "(No text response — the agent finished without producing "
                        "any output. Check the gateway log for ``turn.agent.event`` "
                        "lines and your model configuration.)"
                    )
                    elog("voice.empty_response", level="warning", session_id=session_id)
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
                logger.warning("RESPONSE send failed: %s", e)

            elog(
                "turn.end",
                session_id=session_id,
                speak=speak,
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
