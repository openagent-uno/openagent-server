"""Long-lived stream session — the spine of the unified I/O protocol.

A :class:`StreamSession` owns inbound + outbound queues and the
coroutines that wire them to the agent core: a dispatch loop that
routes inbound events, an STT pump that turns audio into
``TextFinal(source="stt")``, and per-turn :class:`StreamTurnRunner`
tasks that emit token deltas (and optional TTS audio) back out. Barge-in
on a fresh text or :class:`Interrupt` cancels the in-flight turn.

Channel-agnostic: realtime adapters ferry events to/from the queues;
batched adapters push one :class:`TextFinal` and drain until
:class:`TurnComplete`. :meth:`run_one_shot` is the legacy single-turn
shim for callers that don't drive the queues themselves.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from collections import defaultdict, deque
from typing import Any, Awaitable, Callable

import json

from openagent.channels.stt_base import BaseSTT, resolve_stt
from openagent.channels.tts_base import BaseTTS, resolve_tts
from openagent.core.logging import elog
from openagent.stream.events import (
    Attachment,
    AudioChunk,
    Event,
    Interrupt,
    OutAudioChunk,
    OutAudioEnd,
    OutAudioStart,
    OutError,
    OutTextDelta,
    OutTextFinal,
    OutToolStatus,
    SessionClose,
    SessionOpen,
    TextDelta,
    TextFinal,
    TurnComplete,
    VideoFrame,
    now_ms,
)

logger = logging.getLogger(__name__)


VIDEO_RING_SIZE = 8
SPEAKER_DRAIN_TIMEOUT = 20.0


# Tool-name substring → resource category. Substring (not equality)
# because tool names embed the MCP server prefix differently across
# providers (``mcp__scheduler__add_task`` vs ``scheduler_add_task``).
_MCP_PREFIX_TO_RESOURCE: tuple[tuple[str, str], ...] = (
    ("scheduler", "scheduled_task"),
    ("workflow_manager", "workflow"),
    ("mcp_manager", "mcp"),
    ("vault", "vault"),
)


class StreamSession:
    """Long-lived (client_id, session_id) bus.

    Construct once per session; call :meth:`start` to spin up the pumps,
    :meth:`push_in` to feed inbound events, ``await session.outbound.get()``
    to consume outbound. :meth:`close` shuts everything down.
    """

    DEFAULT_COALESCE_WINDOW_MS = 500

    def __init__(
        self,
        agent: Any,
        *,
        client_id: str,
        session_id: str,
        profile: str = "realtime",
        language: str | None = None,
        coalesce_window_ms: int | None = None,
        speak_enabled: bool = True,
    ):
        self._agent = agent
        self._db = getattr(agent, "db", None)
        self.client_id = client_id
        self.session_id = session_id
        self.profile = profile
        self.language = language
        # ``0`` disables coalescing (legacy preempt-on-each-message);
        # STT/system messages always bypass it regardless.
        if coalesce_window_ms is None:
            coalesce_window_ms = self.DEFAULT_COALESCE_WINDOW_MS
        self.coalesce_window_ms = max(0, int(coalesce_window_ms))
        # When False, typed replies stay silent even if TTS resolved;
        # voice (STT) still speaks via the mirror-modality rule.
        self.speak_enabled = bool(speak_enabled)

        self.inbound: asyncio.Queue[Event] = asyncio.Queue()
        self.outbound: asyncio.Queue[Event] = asyncio.Queue()

        self._stt: BaseSTT | None = None
        self._tts: BaseTTS | None = None

        self._stt_in: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._stt_encoding: str = "webm"
        # PCM needs sample_rate for the WAV header / Deepgram params;
        # ``None`` falls through to vendor defaults (16000).
        self._stt_sample_rate: int | None = None
        self._stt_pump_task: asyncio.Task | None = None
        self._dispatch_task: asyncio.Task | None = None
        self._current_turn: asyncio.Task | None = None

        self._video_buffers: dict[str, deque[VideoFrame]] = defaultdict(
            lambda: deque(maxlen=VIDEO_RING_SIZE)
        )
        self._pending_attachments: list[dict[str, Any]] = []
        self._pending_burst: list[TextFinal] = []
        self._burst_timer: asyncio.Task | None = None
        # Serialises the dispatch-loop's TextFinal handler with the
        # burst-drain task. Without it, a fresh TextFinal arriving in
        # the gap between ``_pending_burst = []`` and the merged-turn
        # ``create_task`` would observe ``in_flight=False`` and dispatch
        # a duplicate turn onto the same ``_current_turn`` slot.
        self._dispatch_lock: asyncio.Lock = asyncio.Lock()
        # When True, the runner's terminal frames (``OutTextFinal`` +
        # ``TurnComplete``) are suppressed so the client UI keeps its
        # "Thinking…" indicator alive across a barge-in until the
        # follow-up turn lands.
        self._suppress_runner_completion: bool = False
        # Deltas of the in-flight assistant turn. Read by
        # ``_cancel_active_turn`` for ``commit_partial_assistant``.
        self._partial_assistant: list[str] = []
        # ``_current_turn_msg`` is the TextFinal that triggered the
        # in-flight turn; the runner flips ``_current_turn_started`` on
        # the FIRST event from ``agent.run_stream`` (the engagement
        # signal — soonest reliable indicator that the prompt reached
        # the provider, post claude-cli spawn / agno arun). Cancels
        # before that point salvage the input back into the burst;
        # cancels after take the partial-commit path.
        self._current_turn_msg: TextFinal | None = None
        self._current_turn_started: bool = False
        self._seq = 0
        self._closed = False
        # Status side-channel for run_one_shot callers.
        self._extra_status_cb: Callable[[str], Awaitable[None]] | None = None

        # ── Optional gateway hooks (set externally) ─────────────────────
        # Returning a non-None string from ``pre_dispatch_hook`` rejects
        # the turn with ``OutError(text=<that string>)`` + ``TurnComplete``.
        self.pre_dispatch_hook: (
            Callable[[TextFinal], Awaitable[str | None]] | None
        ) = None
        # Receives the resource categories tracked during the turn so
        # the gateway can broadcast ``resource_event`` frames.
        self.post_turn_hook: (
            Callable[[set[str]], Awaitable[None]] | None
        ) = None
        self._turn_resources: set[str] = set()

    # ── lifecycle ────────────────────────────────────────────────────

    async def start(
        self,
        *,
        stt_factory: Callable[[Any], Awaitable[BaseSTT | None]] | None = None,
        tts_factory: Callable[[Any], Awaitable[BaseTTS | None]] | None = None,
    ) -> None:
        """Resolve providers, spin up the pumps. Idempotent."""
        if self._dispatch_task is not None:
            return
        stt_factory = stt_factory or resolve_stt
        tts_factory = tts_factory or resolve_tts
        self._stt = await stt_factory(self._db)
        self._tts = await tts_factory(self._db)
        elog(
            "stream.session.start",
            session_id=self.session_id,
            client_id=self.client_id,
            profile=self.profile,
            stt=type(self._stt).__name__ if self._stt else None,
            tts=type(self._tts).__name__ if self._tts else None,
        )
        if self._stt is not None:
            self._stt_pump_task = asyncio.create_task(
                self._stt_pump(),
                name=f"stream-stt:{self.session_id}",
            )
        self._dispatch_task = asyncio.create_task(
            self._dispatch_loop(),
            name=f"stream-dispatch:{self.session_id}",
        )

    async def run_one_shot(
        self,
        text: str,
        *,
        attachments: list[dict] | None = None,
        speak: bool = False,
        on_status: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict[str, Any]:
        """Drive a single turn without using the inbound queue.

        Used by legacy callers (gateway MESSAGE handler, REST shims,
        batched bridges). Status frames also tee to ``on_status`` for
        callers keeping their existing side-channel.
        """
        # Lazy-resolve TTS for callers that skip :meth:`start`.
        if self._tts is None and not self._dispatch_task:
            self._tts = await resolve_tts(self._db)

        self._extra_status_cb = on_status
        try:
            runner = StreamTurnRunner(
                self._agent,
                self,
                tts=self._tts,
                language=self.language,
            )
            return await runner.run(
                text,
                client_id=self.client_id,
                session_id=self.session_id,
                attachments=attachments,
                speak=speak,
            )
        finally:
            self._extra_status_cb = None

    async def close(self) -> None:
        """Drain in-flight work and tear down the pumps."""
        if self._closed:
            return
        self._closed = True
        # Drop the buffered burst — the WS is going away.
        self._cancel_burst_timer()
        self._pending_burst = []
        await self._cancel_active_turn()
        if self._stt_pump_task is not None:
            await self._stt_in.put(None)
            try:
                await asyncio.wait_for(self._stt_pump_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._stt_pump_task.cancel()
            except Exception as e:  # noqa: BLE001
                logger.debug("stt pump close error: %s", e)
        if self._dispatch_task is not None:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except (asyncio.CancelledError, Exception):
                pass
        elog("stream.session.close", session_id=self.session_id)

    # ── inbound surface ─────────────────────────────────────────────

    async def push_in(self, evt: Event) -> None:
        """Append an inbound event for the dispatch loop to handle."""
        await self.inbound.put(evt)

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    # ── publishing helper for the turn runner ───────────────────────

    async def _publish(self, evt: Event) -> None:
        # Suppress the cancelled turn's terminal frames when a
        # follow-up turn is on the way — keeps the client's
        # "Thinking…" / streaming bubble alive across the gap.
        # Intermediate deltas + tool status still flow.
        if self._suppress_runner_completion and isinstance(
            evt, (OutTextFinal, TurnComplete)
        ):
            return
        await self.outbound.put(evt)
        if isinstance(evt, OutToolStatus):
            if self.post_turn_hook is not None:
                self._track_tool_prefix(evt.text)
            if self._extra_status_cb is not None:
                try:
                    await self._extra_status_cb(evt.text)
                except Exception as e:  # noqa: BLE001
                    logger.debug("extra_status_cb raised: %s", e)
        elif isinstance(evt, TurnComplete) and self.post_turn_hook is not None:
            # Reset BEFORE invoking so a hook crash can't poison
            # the next turn's accumulator.
            seen = self._turn_resources
            self._turn_resources = set()
            try:
                await self.post_turn_hook(seen)
            except Exception as e:  # noqa: BLE001
                logger.warning("post_turn_hook raised: %s", e)

    def _track_tool_prefix(self, status_text: str) -> None:
        try:
            data = json.loads(status_text)
        except (ValueError, TypeError):
            return
        if not isinstance(data, dict):
            return
        tool = data.get("tool")
        if not isinstance(tool, str):
            return
        for needle, resource in _MCP_PREFIX_TO_RESOURCE:
            if needle in tool:
                self._turn_resources.add(resource)

    # ── dispatch loop ───────────────────────────────────────────────

    async def _dispatch_loop(self) -> None:
        try:
            while not self._closed:
                evt = await self.inbound.get()
                await self._dispatch(evt)
        except asyncio.CancelledError:
            return
        except Exception as e:  # noqa: BLE001
            logger.exception("stream dispatch loop crashed: %s", e)

    async def _dispatch(self, evt: Event) -> None:
        if isinstance(evt, AudioChunk):
            # The STT pump promotes utterance finals to
            # ``TextFinal(source="stt")`` and re-feeds them through
            # this dispatch.
            if self._stt is not None and evt.data:
                if evt.encoding:
                    self._stt_encoding = evt.encoding
                if evt.sample_rate:
                    self._stt_sample_rate = evt.sample_rate
                await self._stt_in.put(evt.data)
            if evt.end_of_speech and self._stt is not None:
                await self._stt_in.put(b"")  # close the utterance window
            return

        if isinstance(evt, VideoFrame):
            ring = self._video_buffers[evt.stream]
            ring.append(evt)
            # First frame after an empty ring (initial OR post-snapshot
            # reset) — log once per visible stream activation, not per tick.
            if len(ring) == 1:
                elog(
                    "stream.video.frame_in",
                    session_id=self.session_id,
                    stream=evt.stream,
                    bytes=len(evt.image_bytes),
                    width=evt.width,
                    height=evt.height,
                )
            return

        if isinstance(evt, TextDelta):
            # Promote ``final=True`` deltas to a TextFinal turn trigger;
            # interim deltas are UI-only (typed-text preview).
            if evt.final and evt.text:
                promoted = TextFinal(
                    session_id=self.session_id,
                    seq=self.next_seq(),
                    ts_ms=evt.ts_ms or now_ms(),
                    text=evt.text,
                    source="user_typed",
                )
                await self._on_user_turn_complete(promoted)
            return

        if isinstance(evt, TextFinal):
            await self._on_user_turn_complete(evt)
            return

        if isinstance(evt, Attachment):
            self._pending_attachments.append({
                "type": evt.kind,
                "path": evt.path,
                "filename": evt.filename,
                "mime_type": evt.mime_type,
            })
            return

        if isinstance(evt, Interrupt):
            # Lock matches ``_on_user_turn_complete`` — prevents the
            # burst-drain task from dispatching a stale turn during the
            # cancel. No completion-suppression: nothing follows.
            async with self._dispatch_lock:
                self._cancel_burst_timer()
                self._pending_burst = []
                await self._cancel_active_turn(reason=evt.reason)
            return

        if isinstance(evt, SessionOpen):
            # Channel adapter owns SessionOpen — ignore here so a stray
            # frame can't reset session state.
            return

        if isinstance(evt, SessionClose):
            await self.close()
            return

        logger.debug("stream.session: unhandled inbound %s", type(evt).__name__)

    # ── STT pump ────────────────────────────────────────────────────

    async def _stt_pump(self) -> None:
        """Drive the streaming STT transducer.

        Critical: chunks feed ``stt.stream(...)`` as a LIVE async
        iterator (not pre-buffered). This is what delivers the
        streaming-STT TTFA win — Deepgram sees bytes the instant the
        client produces them and can commit a final inside the user's
        last syllable. A pre-buffered iterator works too but caps
        latency at VAD silence detection.

        Sentinels on ``self._stt_in``: ``None`` = close, ``b""`` =
        end-of-utterance (from ``end_of_speech=True`` or VAD fallback).
        """
        assert self._stt is not None
        while not self._closed:
            first = await self._stt_in.get()
            if first is None:
                return
            if first == b"":
                continue  # stray EOS between utterances

            async def _live_audio(_first: bytes = first):
                yield _first
                while True:
                    piece = await self._stt_in.get()
                    if piece is None or piece == b"":
                        return
                    yield piece

            try:
                async for ev in self._stt.stream(
                    _live_audio(),
                    language=self.language,
                    encoding=self._stt_encoding,
                    sample_rate=self._stt_sample_rate,
                ):
                    if ev.kind == "final" and ev.text.strip():
                        promoted = TextFinal(
                            session_id=self.session_id,
                            seq=self.next_seq(),
                            ts_ms=now_ms(),
                            text=ev.text.strip(),
                            source="stt",
                        )
                        # Tee to outbound so the universal app can show
                        # the recognised user line without a REST round-trip.
                        await self._publish(promoted)
                        await self.inbound.put(promoted)
                    elif ev.kind == "partial" and ev.text:
                        await self._publish(OutTextDelta(
                            session_id=self.session_id,
                            seq=self.next_seq(),
                            ts_ms=now_ms(),
                            text=f"[partial] {ev.text}",
                        ))
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("stream stt pump error: %s", e)

    # ── turn dispatch + barge-in ────────────────────────────────────

    async def _on_user_turn_complete(self, msg: TextFinal) -> None:
        # The dispatch lock serialises this with the burst-drain task —
        # see ``_dispatch_lock`` field comment.
        async with self._dispatch_lock:
            # Voice / system messages preempt immediately (matches the
            # voice-call UX where the model stops the instant the user
            # speaks). A typed burst already buffered when STT lands
            # folds into the same merged turn so "typed X then said Y"
            # doesn't split.
            if msg.source != "user_typed":
                self._cancel_burst_timer()
                if self._pending_burst:
                    merged = self._merge_burst(self._pending_burst + [msg])
                    self._pending_burst = []
                    await self._dispatch_turn(merged)
                else:
                    await self._dispatch_turn(msg)
                return

            if self.coalesce_window_ms <= 0:
                await self._dispatch_turn(msg)
                return

            # Typed text ALWAYS funnels through the debounce buffer.
            # An earlier design dispatched the first message immediately
            # and buffered follow-ups, which split bursts like ``a, b,
            # c`` into two turns and orphaned ``a``. Always-buffering
            # costs one debounce window of latency but guarantees the
            # whole burst reaches the agent as ONE merged turn.
            if self._current_turn and not self._current_turn.done():
                # ``salvage_to_burst`` re-buffers the in-flight turn's
                # input if the agent hasn't engaged yet (drain race).
                await self._cancel_active_turn(
                    reason="user_text",
                    suppress_completion=True,
                    salvage_to_burst=True,
                )
            self._pending_burst.append(msg)
            self._restart_burst_timer()

    async def _dispatch_turn(self, msg: TextFinal) -> None:
        """Cancel any in-flight turn, gather context, start a new one."""
        if self._current_turn and not self._current_turn.done():
            # The caller is about to dispatch fresh — suppress the
            # cancelled runner's terminal frames to bridge the UI gap.
            await self._cancel_active_turn(
                reason="user_text", suppress_completion=True,
            )

        if self.pre_dispatch_hook is not None:
            try:
                err = await self.pre_dispatch_hook(msg)
            except Exception as e:  # noqa: BLE001
                logger.warning("pre_dispatch_hook raised: %s", e)
                err = None
            if err:
                await self._publish(OutError(
                    session_id=self.session_id,
                    seq=self.next_seq(),
                    ts_ms=now_ms(),
                    text=err,
                ))
                await self._publish(TurnComplete(
                    session_id=self.session_id,
                    seq=self.next_seq(),
                    ts_ms=now_ms(),
                ))
                return

        self._turn_resources = set()

        attachments = list(msg.attachments)
        attachments.extend(self._pending_attachments)
        self._pending_attachments = []
        attachments.extend(self._snapshot_video_frames())

        text = msg.text
        if not text and not attachments:
            return

        self._partial_assistant = []

        # Mirror modality: voice in → voice out regardless of the
        # session toggle. Merged bursts inherit the last message's
        # source (see ``_merge_burst``).
        from_voice = (msg.source != "user_typed")
        speak = bool(self._tts) and (from_voice or self.speak_enabled)
        runner = StreamTurnRunner(
            self._agent,
            self,
            tts=self._tts,
            language=self.language,
        )
        # Reset the engagement flag — the runner flips it on the first
        # event from ``run_stream``. See ``_current_turn_msg`` field
        # comment for the salvage rationale.
        self._current_turn_msg = msg
        self._current_turn_started = False
        self._current_turn = asyncio.create_task(
            runner.run(
                text,
                client_id=self.client_id,
                session_id=self.session_id,
                attachments=attachments or None,
                speak=speak,
            ),
            name=f"stream-turn:{self.session_id}",
        )

    def _restart_burst_timer(self) -> None:
        """(Re)arm the debounce timer. Cancels any prior pending fire."""
        if self._burst_timer and not self._burst_timer.done():
            self._burst_timer.cancel()
        self._burst_timer = asyncio.create_task(
            self._burst_drain(),
            name=f"burst-drain:{self.session_id}",
        )

    def _cancel_burst_timer(self) -> None:
        if self._burst_timer and not self._burst_timer.done():
            self._burst_timer.cancel()
        self._burst_timer = None

    async def _burst_drain(self) -> None:
        try:
            await asyncio.sleep(self.coalesce_window_ms / 1000.0)
        except asyncio.CancelledError:
            return
        # Lock prevents a racing TextFinal from observing
        # ``has_pending=False, in_flight=False`` and dispatching a
        # parallel turn — see ``_dispatch_lock`` field comment.
        async with self._dispatch_lock:
            if not self._pending_burst:
                return
            msgs = self._pending_burst
            self._pending_burst = []
            self._burst_timer = None
            await self._dispatch_turn(self._merge_burst(msgs))

    def _merge_burst(self, msgs: list[TextFinal]) -> TextFinal:
        # ``\n\n`` reads as a paragraph break to chunkers + LLMs so the
        # merged messages stay distinguishable. Last source wins so
        # downstream policy sees the most recent modality.
        texts = [m.text.strip() for m in msgs if (m.text or "").strip()]
        merged_atts: list[dict[str, Any]] = []
        for m in msgs:
            merged_atts.extend(m.attachments)
        return TextFinal(
            session_id=self.session_id,
            seq=self.next_seq(),
            ts_ms=now_ms(),
            text="\n\n".join(texts),
            source=msgs[-1].source if msgs else "user_typed",
            attachments=tuple(merged_atts),
        )

    async def _cancel_active_turn(
        self,
        *,
        reason: str = "manual",
        suppress_completion: bool = False,
        salvage_to_burst: bool = False,
    ) -> None:
        task = self._current_turn
        if task is None or task.done():
            return
        elog("stream.barge_in", session_id=self.session_id, reason=reason)
        # Snapshot BEFORE cancelling — otherwise the runner can append
        # one more delta between read and cancel.
        partial = "".join(self._partial_assistant).strip()
        self._partial_assistant = []
        # Salvage: re-buffer the input if the agent hasn't engaged yet
        # (typed-burst drain race — merged turn was just scheduled but
        # the runner hasn't reached ``run_stream``). Interrupt/close
        # explicitly discard, so the flag gates this.
        salvaged_msg: TextFinal | None = None
        if (
            salvage_to_burst
            and self._current_turn_msg is not None
            and not self._current_turn_started
        ):
            salvaged_msg = self._current_turn_msg
        # Set BEFORE cancel so the runner's finally-block publishes
        # honour the suppression while the cancel propagates.
        if suppress_completion:
            self._suppress_runner_completion = True
        # Cancel BEFORE any provider control-request — otherwise the
        # provider's reader might be parked behind a full receive
        # channel and ``client.interrupt()``'s ack can't land,
        # burning a 60 s timeout and freezing the dispatch loop.
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        finally:
            self._current_turn = None
            self._current_turn_msg = None
            self._current_turn_started = False
            if suppress_completion:
                self._suppress_runner_completion = False
        if salvaged_msg is not None:
            self._pending_burst.insert(0, salvaged_msg)
        # Persist the partial so history reads as
        # ``user → assistant(partial) → user`` not two adjacent users.
        if partial:
            commit = getattr(self._agent, "commit_partial_assistant", None)
            if callable(commit):
                try:
                    await commit(self.session_id, partial)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "commit_partial_assistant failed (%s): %s",
                        self.session_id, e,
                    )

    def _snapshot_video_frames(self) -> list[dict[str, Any]]:
        """Persist the latest frame per stream as image attachments.

        The ``<stream>-snapshot.jpg`` filename is load-bearing — the
        agent's attachment-context block uses it to tell the LLM
        whether the frame is a webcam or screen feed. Without that
        hint, models reach for an MCP screenshot tool instead of
        reading the attached image.
        """
        out: list[dict[str, Any]] = []
        for stream, ring in list(self._video_buffers.items()):
            if not ring:
                continue
            frame = ring[-1]
            if not frame.image_bytes:
                continue
            try:
                tmp = tempfile.NamedTemporaryFile(
                    prefix=f"oa_{stream}_snapshot_",
                    suffix=".jpg",
                    delete=False,
                )
                tmp.write(frame.image_bytes)
                tmp.close()
                path = tmp.name
                friendly_name = f"{stream}-snapshot.jpg"
            except OSError as e:
                logger.warning("video snapshot write failed: %s", e)
                continue
            elog(
                "stream.video.snapshot",
                session_id=self.session_id,
                stream=stream,
                bytes=len(frame.image_bytes),
                path=path,
            )
            out.append({
                "type": "image",
                "path": path,
                "filename": friendly_name,
            })
        self._video_buffers.clear()
        return out


class StreamTurnRunner:
    """Single-turn runner for the new stream protocol.

    Drives the agent's :meth:`run_stream` and routes events onto the
    session's outbound queue. Optional TTS is plugged in as a parallel
    transducer — deltas pipe into the TTS's text iterator, audio
    chunks emit on the same queue.
    """

    def __init__(
        self,
        agent: Any,
        session: StreamSession,
        *,
        tts: BaseTTS | None = None,
        language: str | None = None,
    ):
        self._agent = agent
        self._session = session
        self._tts = tts
        self._language = language

    async def run(
        self,
        text: str,
        *,
        client_id: str,
        session_id: str,
        attachments: list[dict] | None = None,
        speak: bool = False,
    ) -> dict[str, Any]:
        sess = self._session
        publish = sess._publish
        accumulated: list[str] = []
        audio_started = False
        audio_chunks = 0
        stream_error: BaseException | None = None
        spoken_tools: set[str] = set()

        text_q: asyncio.Queue[str | None] = asyncio.Queue()

        async def text_iter():
            while True:
                piece = await text_q.get()
                if piece is None:
                    return
                yield piece

        async def speaker_task():
            nonlocal audio_started, audio_chunks
            if self._tts is None:
                return
            try:
                async for chunk in self._tts.synthesize_stream(
                    text_iter(), language=self._language,
                ):
                    if not chunk:
                        continue
                    if not audio_started:
                        fmt, mime = self._tts.audio_format
                        await publish(OutAudioStart(
                            session_id=session_id,
                            seq=sess.next_seq(),
                            ts_ms=now_ms(),
                            format=fmt,
                            mime=mime,
                            voice_id=self._tts.voice_id,
                        ))
                        audio_started = True
                    audio_chunks += 1
                    await publish(OutAudioChunk(
                        session_id=session_id,
                        seq=audio_chunks,
                        ts_ms=now_ms(),
                        data=chunk,
                    ))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning("stream speaker task error: %s", e)

        async def on_status(status_text: str) -> None:
            await publish(OutToolStatus(
                session_id=session_id,
                seq=sess.next_seq(),
                ts_ms=now_ms(),
                text=status_text,
            ))

        speaker = asyncio.create_task(speaker_task()) if (speak and self._tts) else None

        try:
            try:
                async for event in self._agent.run_stream(
                    message=text,
                    user_id=client_id,
                    session_id=session_id,
                    attachments=attachments,
                    on_status=on_status,
                ):
                    # Engagement signal — soonest reliable indicator
                    # the prompt reached the provider. Idempotent bool,
                    # safe to set on every event. See ``_dispatch_turn``.
                    sess._current_turn_started = True
                    kind = event.get("kind")
                    if kind == "delta":
                        delta = event.get("text") or ""
                        if not delta:
                            continue
                        accumulated.append(delta)
                        # Tee for ``commit_partial_assistant`` on barge-in.
                        sess._partial_assistant.append(delta)
                        await publish(OutTextDelta(
                            session_id=session_id,
                            seq=sess.next_seq(),
                            ts_ms=now_ms(),
                            text=delta,
                        ))
                        if speaker is not None:
                            await text_q.put(delta)
                    elif kind == "iteration_break":
                        # Synthetic newline forces SentenceChunker to
                        # flush the partial sentence (hard break).
                        if speaker is not None:
                            await text_q.put("\n")
                    elif kind == "done":
                        if event.get("text") and not accumulated:
                            tail = event["text"]
                            accumulated.append(tail)
                            await publish(OutTextDelta(
                                session_id=session_id,
                                seq=sess.next_seq(),
                                ts_ms=now_ms(),
                                text=tail,
                            ))
                            if speaker is not None:
                                await text_q.put(tail)
                        break
                if speaker is not None:
                    await text_q.put(None)
            except asyncio.CancelledError:
                if speaker is not None:
                    await text_q.put(None)
                raise
            except Exception as e:  # noqa: BLE001
                stream_error = e
                if speaker is not None:
                    await text_q.put(None)
                logger.warning("stream turn failed: %s", e)
        finally:
            if speaker is not None:
                try:
                    await asyncio.wait_for(speaker, timeout=SPEAKER_DRAIN_TIMEOUT)
                except asyncio.TimeoutError:
                    speaker.cancel()
                except (asyncio.CancelledError, Exception) as e:
                    if isinstance(e, asyncio.CancelledError):
                        raise
                    logger.debug("speaker cleanup error: %s", e)

            if audio_started:
                await publish(OutAudioEnd(
                    session_id=session_id,
                    seq=sess.next_seq(),
                    ts_ms=now_ms(),
                    total_chunks=audio_chunks,
                ))

            full_text = "".join(accumulated)
            if not full_text and stream_error is not None:
                full_text = f"Error: {stream_error}"
            elif not full_text and speak:
                full_text = (
                    "(No text response — the agent finished without producing "
                    "any output. Check the gateway log for ``turn.agent.event`` "
                    "lines and your model configuration.)"
                )

            from openagent.channels.base import parse_response_markers
            clean, attachments_out = parse_response_markers(full_text)
            att_list = [
                {"type": a.type, "path": a.path, "filename": a.filename}
                for a in attachments_out
            ]
            meta_fn = getattr(self._agent, "last_response_meta", None)
            meta: dict = {}
            try:
                if meta_fn is not None:
                    meta = meta_fn(session_id) or {}
            except Exception as e:  # noqa: BLE001
                logger.debug("last_response_meta failed: %s", e)

            await publish(OutTextFinal(
                session_id=session_id,
                seq=sess.next_seq(),
                ts_ms=now_ms(),
                text=clean,
                attachments=tuple(att_list),
                model=meta.get("model"),
            ))
            await publish(TurnComplete(
                session_id=session_id,
                seq=sess.next_seq(),
                ts_ms=now_ms(),
            ))

            elog(
                "stream.turn.end",
                session_id=session_id,
                response_chars=len(clean),
                audio_chunks=audio_chunks,
                spoken_tools=len(spoken_tools),
                errored=stream_error is not None,
            )

        return {
            "text": clean,
            "attachments": att_list,
            "audio_chunks": audio_chunks,
            "errored": stream_error is not None,
        }


__all__ = ["StreamSession", "StreamTurnRunner"]
