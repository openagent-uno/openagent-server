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
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import json

from src.channels.base import is_reasoning_status, parse_compaction_status
from src.channels.stt_base import BaseSTT, resolve_stt
from src.channels.tts_base import BaseTTS, resolve_tts
from src.core.identity_context import human_author
from src.core.logging import elog
from src.stream.events import (
    Attachment,
    AudioChunk,
    Event,
    Interrupt,
    OutAudioChunk,
    ContextReport,
    OutAudioEnd,
    OutAudioStart,
    OutError,
    OutReasoning,
    OutTextDelta,
    OutTextFinal,
    OutToolStatus,
    SessionClose,
    SessionCompacted,
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
# Upper bound on how long a barge-in waits for the cancelled turn to
# actually finish before detaching it. With a runner that honours
# cancellation this is never hit; it only guards against a provider that
# swallows the cancel — so one misbehaving model can't freeze the session.
BARGE_IN_DRAIN_TIMEOUT = 5.0
# How long to let a COOPERATIVELY-cancelled runtime turn finish on its own
# before falling back to a hard ``task.cancel()``. Cooperative cancel lets the
# runtime (agno) reach its next checkpoint, persist the interrupted run with
# its partial messages, and exit cleanly — so the next turn keeps context. For
# a streaming turn the next checkpoint is sub-second; this only bounds the wait
# when the model is parked in a long non-streaming call.
COOP_CANCEL_DRAIN_TIMEOUT = 4.0
_EMPTY_TURN_FALLBACK_TEXT = (
    "(No text response — the agent finished without producing any output. "
    "Please retry, and check the model/provider logs if this keeps happening.)"
)
_CANCELLED_TURN_FALLBACK_TEXT = (
    "(The turn was interrupted before a response was ready. Please retry.)"
)
_USE_CURRENT_TURN_INGRESS = object()


@dataclass(frozen=True, slots=True)
class _STTInput:
    """One trusted, per-utterance item on the streaming STT queue."""

    data: bytes
    end_of_utterance: bool
    execution_origin: Any
    ingress_identity: Any
    encoding: str = ""
    sample_rate: int = 0


def _same_execution_origin(left: Any, right: Any) -> bool:
    """Compare exact client hosts, including the Gateway registry identity."""

    if left is right:
        return True
    if left is None or right is None:
        return False
    return (
        getattr(left, "device_id", None) == getattr(right, "device_id", None)
        and getattr(left, "client_instance_id", None)
        == getattr(right, "client_instance_id", None)
        and getattr(left, "generation", None) == getattr(right, "generation", None)
        and getattr(left, "auth_epoch", 0) == getattr(right, "auth_epoch", 0)
        and getattr(left, "registry", None) is getattr(right, "registry", None)
    )


def _ingress_key(identity: Any) -> tuple[Any, ...]:
    """Return a stable, non-wire key for trusted per-connection ownership."""

    if identity is None:
        return ("internal",)
    device_id = getattr(identity, "device_id", None)
    connection_id = getattr(identity, "connection_id", None)
    if isinstance(device_id, str) and isinstance(connection_id, str):
        return (
            "gateway",
            device_id,
            connection_id,
            getattr(identity, "client_instance_id", None),
            getattr(identity, "auth_epoch", 0),
        )
    try:
        hash(identity)
    except (TypeError, ValueError):
        return ("opaque", id(identity))
    return ("value", identity)


def _same_ingress(left: Any, right: Any) -> bool:
    return _ingress_key(left) == _ingress_key(right)


def _ingress_device_id(identity: Any) -> str | None:
    """Return the authenticated device owner without trusting wire fields."""

    device_id = getattr(identity, "device_id", None)
    return device_id if isinstance(device_id, str) and device_id else None


def _ingress_auth_epoch(identity: Any) -> int:
    value = getattr(identity, "auth_epoch", 0)
    return value if type(value) is int and value >= 0 else 0


# Tool-name substring → resource category. Substring (not equality)
# because tool names embed the MCP server prefix differently across
# providers (``mcp__scheduler__add_task`` vs ``scheduler_add_task``).
_MCP_PREFIX_TO_RESOURCE: tuple[tuple[str, str], ...] = (
    ("scheduler", "scheduled_task"),
    ("workflow_manager", "workflow"),
    ("events_manager", "event"),
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
        handle: str | None = None,
    ):
        self._agent = agent
        self._db = getattr(agent, "db", None)
        self.client_id = client_id
        self.session_id = session_id
        # The authenticated user handle (stable across the user's devices;
        # vision §11). Recorded as the per-message human author so the app
        # shows real identity instead of a generic "You" and multi-human /
        # bridge sessions attribute each message. ``None`` on handle-less
        # deployments → author stays unset (today's generic rendering).
        self.handle = handle
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
        # Outbound frames share one session queue, but their transport owner is
        # frozen when each frame is published.  A second authenticated device
        # may open the same durable session while a turn is running; the
        # channel must not infer ownership from whichever websocket touched the
        # holder most recently.
        self._outbound_ingresses: dict[int, Any] = {}

        self._stt: BaseSTT | None = None
        self._tts: BaseTTS | None = None

        self._stt_in: asyncio.Queue[_STTInput | None] = asyncio.Queue()
        # Ingress ownership is per utterance, not session state. A second
        # authenticated device sharing this durable session cannot append to
        # or terminate the first device's audio stream.
        self._stt_utterance_open = False
        self._stt_ingress_origin: Any = None
        self._stt_ingress_identity: Any = None
        self._stt_pump_task: asyncio.Task | None = None
        self._dispatch_task: asyncio.Task | None = None
        self._current_turn: asyncio.Task | None = None
        # Turns whose cancellation a misbehaving provider swallowed: held
        # here so they finish in the background without a "Task was
        # destroyed but it is pending" warning, and don't block barge-in.
        self._detached_turns: set[asyncio.Task] = set()

        self._video_buffers: dict[tuple[tuple[Any, ...], str], deque[VideoFrame]] = defaultdict(
            lambda: deque(maxlen=VIDEO_RING_SIZE)
        )
        self._pending_attachments: list[tuple[Any, dict[str, Any]]] = []
        self._pending_burst: list[TextFinal] = []
        # Trusted gateway-origin metadata is kept out of the public Event wire.
        # The map follows each immutable TextFinal through the debounce buffer;
        # a newly-constructed merged event receives the common origin explicitly.
        self._event_origins: dict[int, Any] = {}
        self._event_ingresses: dict[int, Any] = {}
        # Set by the Gateway's live revocation callback. Queued events are
        # checked again at dispatch time so a frame accepted immediately
        # before the callback cannot start work after revocation.
        self._revoked_ingress_epochs: dict[str, int] = {}
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
        # ``_current_turn_msg`` is the TextFinal that triggered the
        # in-flight turn; the runner flips ``_current_turn_started`` on
        # the FIRST event from ``agent.run_stream`` (the engagement
        # signal — soonest reliable indicator that the prompt reached
        # the provider, post runtime arun). Cancels
        # before that point salvage the input back into the burst;
        # cancels after take the partial-commit path.
        self._current_turn_msg: TextFinal | None = None
        self._current_turn_origin: Any = None
        self._current_turn_ingress: Any = None
        # Process-local identity for the runner that currently owns this
        # session. A provider may swallow cancellation and outlive a barge-in;
        # its late frames must never inherit the replacement turn's ingress.
        self._current_turn_token: object | None = None
        self._current_turn_started: bool = False
        # Reason for the active task cancellation currently being
        # orchestrated by ``_cancel_active_turn``. ``None`` means any
        # CancelledError escaping the runner was unexpected (for
        # example, a provider-side cancel leak) and should be
        # finalised into a terminal frame instead of leaving the
        # bridge collector stuck on "Thinking...".
        self._active_cancel_reason: str | None = None
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
                author=human_author(self.handle, display=self.handle),
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
        for pending in self._pending_burst:
            self._event_origins.pop(id(pending), None)
            self._event_ingresses.pop(id(pending), None)
        self._pending_burst = []
        self._outbound_ingresses.clear()
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

    async def push_in(
        self,
        evt: Event,
        *,
        execution_origin: Any = None,
        ingress_identity: Any = None,
    ) -> None:
        """Append an inbound event, optionally with trusted gateway origin.

        Neither trusted value is decoded from the wire. ``ingress_identity``
        always names the authenticated chat connection; ``execution_origin``
        exists only when that exact client currently advertises local tools.
        """
        ingress_device = _ingress_device_id(ingress_identity)
        if self._ingress_is_revoked(ingress_identity):
            elog(
                "stream.ingress.revoked_drop",
                level="warning",
                session_id=self.session_id,
                device_id=ingress_device,
            )
            return
        if isinstance(evt, TextFinal):
            self._event_origins[id(evt)] = execution_origin
        elif isinstance(evt, TextDelta):
            # A promoted final/STT transcript must retain the connection that
            # supplied its source media.
            self._event_origins[id(evt)] = execution_origin
        elif isinstance(evt, AudioChunk):
            self._event_origins[id(evt)] = execution_origin
        if isinstance(evt, (TextFinal, TextDelta, AudioChunk, VideoFrame, Attachment, Interrupt)):
            self._event_ingresses[id(evt)] = ingress_identity
        await self.inbound.put(evt)

    def _ingress_is_revoked(self, identity: Any) -> bool:
        device_id = _ingress_device_id(identity)
        if device_id is None:
            return False
        blocked_epoch = self._revoked_ingress_epochs.get(device_id)
        return (
            blocked_epoch is not None
            and _ingress_auth_epoch(identity) < blocked_epoch
        )

    def allow_ingress_device(self, device_id: str, *, auth_epoch: int = 0) -> None:
        """Validate that a fresh ingress is newer than the revocation barrier.

        The barrier is intentionally retained so already-queued frames from an
        older socket remain denied even after a post-reactivation socket joins.
        """

        blocked_epoch = self._revoked_ingress_epochs.get(device_id)
        if blocked_epoch is not None and auth_epoch < blocked_epoch:
            raise PermissionError("device ingress authorization is stale")

    async def revoke_ingress_device(
        self,
        device_id: str,
        *,
        revocation_epoch: int | None = None,
    ) -> bool:
        """Drop and cancel only work owned by one authenticated device.

        A durable session can be attached by several devices. Revocation must
        cancel A's active turn and buffered media without closing the shared
        holder or disturbing an unrelated turn/burst owned by B.
        """

        affected = False
        previous_epoch = self._revoked_ingress_epochs.get(device_id, -1)
        if revocation_epoch is None:
            revocation_epoch = max(1, previous_epoch + 1)
            owned_epochs = [
                _ingress_auth_epoch(identity)
                for identity in self._event_ingresses.values()
                if _ingress_device_id(identity) == device_id
            ]
            if _ingress_device_id(self._current_turn_ingress) == device_id:
                owned_epochs.append(_ingress_auth_epoch(self._current_turn_ingress))
            if owned_epochs:
                revocation_epoch = max(revocation_epoch, max(owned_epochs) + 1)
        revocation_epoch = max(previous_epoch, int(revocation_epoch))
        self._revoked_ingress_epochs[device_id] = revocation_epoch

        def is_revoked_owner(identity: Any) -> bool:
            return (
                _ingress_device_id(identity) == device_id
                and _ingress_auth_epoch(identity) < revocation_epoch
            )

        async with self._dispatch_lock:
            kept_burst: list[TextFinal] = []
            for msg in self._pending_burst:
                ingress = self._event_ingresses.get(id(msg))
                if is_revoked_owner(ingress):
                    self._event_origins.pop(id(msg), None)
                    self._event_ingresses.pop(id(msg), None)
                    affected = True
                else:
                    kept_burst.append(msg)
            self._pending_burst = kept_burst
            if not kept_burst:
                self._cancel_burst_timer()

            before_attachments = len(self._pending_attachments)
            self._pending_attachments = [
                (owner, attachment)
                for owner, attachment in self._pending_attachments
                if not is_revoked_owner(owner)
            ]
            affected = affected or len(self._pending_attachments) != before_attachments

            for key in list(self._video_buffers):
                owner_key, _stream = key
                if (
                    len(owner_key) > 1
                    and owner_key[0] == "gateway"
                    and owner_key[1] == device_id
                    and (
                        len(owner_key) < 5
                        or not isinstance(owner_key[4], int)
                        or owner_key[4] < revocation_epoch
                    )
                ):
                    self._video_buffers.pop(key, None)
                    affected = True

            if is_revoked_owner(self._stt_ingress_identity):
                self._stt_utterance_open = False
                self._stt_ingress_origin = None
                self._stt_ingress_identity = None
                affected = True
                # Abort the live transducer so chunks from another device are
                # not consumed as the tail of the revoked utterance.
                if self._stt_pump_task is not None and not self._stt_pump_task.done():
                    self._stt_pump_task.cancel()
                    try:
                        await self._stt_pump_task
                    except (asyncio.CancelledError, Exception):
                        pass
                    queued = getattr(self._stt_in, "_queue", None)
                    if queued is not None:
                        kept_audio = [
                            item for item in queued
                            if item is None
                            or not is_revoked_owner(item.ingress_identity)
                        ]
                        queued.clear()
                        queued.extend(kept_audio)
                    if not self._closed and self._stt is not None:
                        self._stt_pump_task = asyncio.create_task(
                            self._stt_pump(),
                            name=f"stream-stt:{self.session_id}",
                        )

            if (
                self._current_turn is not None
                and not self._current_turn.done()
                and is_revoked_owner(self._current_turn_ingress)
            ):
                affected = True
                await self._cancel_active_turn(reason="device_revoked")
        return affected

    def has_active_turn(self) -> bool:
        """True iff a turn is running or a burst is buffered to dispatch.

        The gateway consults this so a COMMAND ``stop`` / lifecycle wipe
        can report honestly ("Stopped" vs "Nothing running.") and decide
        whether injecting an :class:`Interrupt` into this live session is
        worthwhile. A turn buffered in ``_pending_burst`` but not yet
        dispatched still counts — a ``stop`` issued inside the coalesce
        window must drop it, otherwise the merged turn would still fire.
        """
        if self._current_turn is not None and not self._current_turn.done():
            return True
        return bool(self._pending_burst)

    def next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def take_outbound_ingress(self, evt: Event) -> Any:
        """Return and forget the trusted transport owner of ``evt``."""

        return self._outbound_ingresses.pop(id(evt), None)

    # ── publishing helper for the turn runner ───────────────────────

    async def _publish(
        self,
        evt: Event,
        *,
        ingress_identity: Any = _USE_CURRENT_TURN_INGRESS,
    ) -> None:
        # Suppress the cancelled turn's terminal frames when a
        # follow-up turn is on the way — keeps the client's
        # "Thinking…" / streaming bubble alive across the gap.
        # Intermediate deltas + tool status still flow.
        if self._suppress_runner_completion and isinstance(
            evt, (OutTextFinal, TurnComplete)
        ):
            return
        # ``outbound`` is intentionally unbounded, so queueing the
        # frame never needs to suspend. Enqueue FIRST, synchronously:
        # if the current task is cancellation-poisoned by a provider
        # leak, the bridge/web collector still sees the terminal frame
        # and can clear "Thinking..." before any follow-up callback
        # await (status tee, post_turn_hook) has a chance to trip over
        # the cancellation.
        if ingress_identity is _USE_CURRENT_TURN_INGRESS:
            ingress_identity = self._current_turn_ingress
        if ingress_identity is not None:
            self._outbound_ingresses[id(evt)] = ingress_identity
        self.outbound.put_nowait(evt)
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
        # API-native wire shape uses ``tool_name``.
        tool = data.get("tool_name")
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
        ingress = self._event_ingresses.get(id(evt))
        if self._ingress_is_revoked(ingress):
            self._event_origins.pop(id(evt), None)
            self._event_ingresses.pop(id(evt), None)
            return
        if isinstance(evt, AudioChunk):
            # The STT pump promotes utterance finals to
            # ``TextFinal(source="stt")`` and re-feeds them through
            # this dispatch.
            origin = self._event_origins.pop(id(evt), None)
            ingress = self._event_ingresses.pop(id(evt), None)
            if self._stt is None:
                return
            if evt.data:
                if not self._stt_utterance_open:
                    self._stt_utterance_open = True
                    self._stt_ingress_origin = origin
                    self._stt_ingress_identity = ingress
                elif not _same_ingress(ingress, self._stt_ingress_identity):
                    # Never mix audio or accept EOS from another authenticated
                    # host. The other client may start after this utterance's
                    # legitimate owner closes it.
                    elog(
                        "stream.stt.origin_conflict",
                        level="warning",
                        session_id=self.session_id,
                        action="chunk_rejected",
                    )
                    return
                await self._stt_in.put(_STTInput(
                    data=evt.data,
                    end_of_utterance=False,
                    execution_origin=self._stt_ingress_origin,
                    ingress_identity=self._stt_ingress_identity,
                    encoding=evt.encoding,
                    sample_rate=evt.sample_rate,
                ))
            if evt.end_of_speech:
                if not self._stt_utterance_open:
                    return
                if not _same_ingress(ingress, self._stt_ingress_identity):
                    elog(
                        "stream.stt.origin_conflict",
                        level="warning",
                        session_id=self.session_id,
                        action="end_of_speech_rejected",
                    )
                    return
                await self._stt_in.put(_STTInput(
                    data=b"",
                    end_of_utterance=True,
                    execution_origin=self._stt_ingress_origin,
                    ingress_identity=self._stt_ingress_identity,
                ))
                self._stt_utterance_open = False
                self._stt_ingress_origin = None
                self._stt_ingress_identity = None
            return

        if isinstance(evt, VideoFrame):
            ingress = self._event_ingresses.pop(id(evt), None)
            ring = self._video_buffers[(_ingress_key(ingress), evt.stream)]
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
                origin = self._event_origins.pop(id(evt), None)
                ingress = self._event_ingresses.pop(id(evt), None)
                self._event_origins[id(promoted)] = origin
                self._event_ingresses[id(promoted)] = ingress
                await self._on_user_turn_complete(
                    promoted,
                    execution_origin=origin,
                    ingress_identity=ingress,
                )
            self._event_origins.pop(id(evt), None)
            self._event_ingresses.pop(id(evt), None)
            return

        if isinstance(evt, TextFinal):
            await self._on_user_turn_complete(
                evt, execution_origin=self._event_origins.pop(id(evt), None),
                ingress_identity=self._event_ingresses.pop(id(evt), None),
            )
            return

        if isinstance(evt, Attachment):
            ingress = self._event_ingresses.pop(id(evt), None)
            self._pending_attachments.append((
                ingress,
                {
                    "type": evt.kind,
                    "path": evt.path,
                    "filename": evt.filename,
                    "mime_type": evt.mime_type,
                },
            ))
            return

        if isinstance(evt, Interrupt):
            self._event_ingresses.pop(id(evt), None)
            # Lock matches ``_on_user_turn_complete`` — prevents the
            # burst-drain task from dispatching a stale turn during the
            # cancel. No completion-suppression: nothing follows.
            async with self._dispatch_lock:
                self._cancel_burst_timer()
                for pending in self._pending_burst:
                    self._event_origins.pop(id(pending), None)
                    self._event_ingresses.pop(id(pending), None)
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

        ``None`` closes the pump; each ``_STTInput`` otherwise carries its
        trusted utterance origin.  A terminal input ends only that exact
        origin's utterance.
        """
        assert self._stt is not None
        while not self._closed:
            first = await self._stt_in.get()
            if first is None:
                return
            if first.end_of_utterance:
                continue  # stray EOS between utterances
            if (
                self._ingress_is_revoked(first.ingress_identity)
            ):
                continue

            utterance_origin = first.execution_origin
            utterance_ingress = first.ingress_identity
            utterance_encoding = first.encoding or "webm"
            # PCM needs sample_rate for the WAV header / Deepgram params;
            # ``None`` falls through to vendor defaults (16000).
            utterance_sample_rate = first.sample_rate or None

            async def _live_audio(_first: _STTInput = first):
                yield _first.data
                while True:
                    piece = await self._stt_in.get()
                    if piece is None or piece.end_of_utterance:
                        return
                    if (
                        self._ingress_is_revoked(piece.ingress_identity)
                    ):
                        continue
                    if not _same_ingress(piece.ingress_identity, utterance_ingress):
                        # Ingress rejects this already; retain a defensive
                        # boundary in the consumer so a future producer cannot
                        # reintroduce mixed-device audio.
                        elog(
                            "stream.stt.origin_conflict",
                            level="error",
                            session_id=self.session_id,
                            action="queued_chunk_rejected",
                        )
                        continue
                    yield piece.data

            try:
                async for ev in self._stt.stream(
                    _live_audio(),
                    language=self.language,
                    encoding=utterance_encoding,
                    sample_rate=utterance_sample_rate,
                ):
                    if (
                        self._ingress_is_revoked(utterance_ingress)
                    ):
                        break
                    if ev.kind == "final" and ev.text.strip():
                        promoted = TextFinal(
                            session_id=self.session_id,
                            seq=self.next_seq(),
                            ts_ms=now_ms(),
                            text=ev.text.strip(),
                            source="stt",
                        )
                        self._event_origins[id(promoted)] = utterance_origin
                        self._event_ingresses[id(promoted)] = utterance_ingress
                        # Tee to outbound so the universal app can show
                        # the recognised user line without a REST round-trip.
                        await self._publish(
                            promoted, ingress_identity=utterance_ingress,
                        )
                        await self.inbound.put(promoted)
                    elif ev.kind == "partial" and ev.text:
                        await self._publish(OutTextDelta(
                            session_id=self.session_id,
                            seq=self.next_seq(),
                            ts_ms=now_ms(),
                            text=f"[partial] {ev.text}",
                        ), ingress_identity=utterance_ingress)
            except asyncio.CancelledError:
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("stream stt pump error: %s", e)

    # ── turn dispatch + barge-in ────────────────────────────────────

    async def _on_user_turn_complete(
        self,
        msg: TextFinal,
        *,
        execution_origin: Any = None,
        ingress_identity: Any = None,
    ) -> None:
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
                    pending = self._pending_burst
                    origins = [self._event_origins.pop(id(m), None) for m in pending]
                    ingresses = [self._event_ingresses.pop(id(m), None) for m in pending]
                    # Never merge input from two client machines/instances.
                    # Flush the older burst first, then dispatch this voice turn
                    # under its independently frozen origin.
                    if any(
                        not _same_execution_origin(origin, execution_origin)
                        or not _same_ingress(ingress, ingress_identity)
                        for origin, ingress in zip(origins, ingresses)
                    ):
                        self._pending_burst = []
                        await self._dispatch_turn(
                            self._merge_burst(pending),
                            execution_origin=origins[0] if origins else None,
                            ingress_identity=ingresses[0] if ingresses else None,
                        )
                        await self._dispatch_turn(
                            msg,
                            execution_origin=execution_origin,
                            ingress_identity=ingress_identity,
                        )
                        return
                    merged = self._merge_burst(pending + [msg])
                    self._pending_burst = []
                    await self._dispatch_turn(
                        merged,
                        execution_origin=execution_origin,
                        ingress_identity=ingress_identity,
                    )
                else:
                    await self._dispatch_turn(
                        msg,
                        execution_origin=execution_origin,
                        ingress_identity=ingress_identity,
                    )
                return

            if self.coalesce_window_ms <= 0:
                await self._dispatch_turn(
                    msg,
                    execution_origin=execution_origin,
                    ingress_identity=ingress_identity,
                )
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
            # A burst is scoped to one exact execution host. If another client
            # instance writes into the same durable session during the debounce
            # window, flush the first burst as its own turn rather than ever
            # giving one model call a mixed/ambiguous machine context.
            if self._pending_burst:
                first_origin = self._event_origins.get(id(self._pending_burst[0]))
                first_ingress = self._event_ingresses.get(id(self._pending_burst[0]))
                if (
                    not _same_execution_origin(first_origin, execution_origin)
                    or not _same_ingress(first_ingress, ingress_identity)
                ):
                    pending = self._pending_burst
                    self._pending_burst = []
                    origins = [self._event_origins.pop(id(m), None) for m in pending]
                    ingresses = [self._event_ingresses.pop(id(m), None) for m in pending]
                    await self._dispatch_turn(
                        self._merge_burst(pending),
                        execution_origin=origins[0] if origins else None,
                        ingress_identity=ingresses[0] if ingresses else None,
                    )
            self._event_origins[id(msg)] = execution_origin
            self._event_ingresses[id(msg)] = ingress_identity
            self._pending_burst.append(msg)
            self._restart_burst_timer()

    async def _dispatch_turn(
        self,
        msg: TextFinal,
        *,
        execution_origin: Any = None,
        ingress_identity: Any = None,
    ) -> None:
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
                ), ingress_identity=ingress_identity)
                await self._publish(TurnComplete(
                    session_id=self.session_id,
                    seq=self.next_seq(),
                    ts_ms=now_ms(),
                ), ingress_identity=ingress_identity)
                return

        self._turn_resources = set()

        attachments = list(msg.attachments)
        remaining_attachments: list[tuple[Any, dict[str, Any]]] = []
        for owner, attachment in self._pending_attachments:
            if _same_ingress(owner, ingress_identity):
                attachments.append(attachment)
            else:
                remaining_attachments.append((owner, attachment))
        self._pending_attachments = remaining_attachments
        attachments.extend(self._snapshot_video_frames(ingress_identity))

        text = msg.text
        if not text and not attachments:
            return

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
        self._current_turn_origin = execution_origin
        self._current_turn_ingress = ingress_identity
        turn_token = object()
        self._current_turn_token = turn_token
        self._current_turn_started = False
        # Prefer a per-message author carried on the inbound frame (a
        # bridge multiplexing several humans onto one session sets it so
        # each message attributes to the right person); fall back to the
        # session-owner handle for the common single-user case.
        turn_author = getattr(msg, "author", None) or human_author(
            self.handle, display=self.handle,
        )
        self._current_turn = asyncio.create_task(
            runner.run(
                text,
                client_id=self.client_id,
                session_id=self.session_id,
                attachments=attachments or None,
                speak=speak,
                author=turn_author,
                execution_origin=execution_origin,
                ingress_identity=ingress_identity,
                turn_token=turn_token,
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
            origins = [self._event_origins.pop(id(m), None) for m in msgs]
            ingresses = [self._event_ingresses.pop(id(m), None) for m in msgs]
            await self._dispatch_turn(
                self._merge_burst(msgs),
                execution_origin=origins[0] if origins else None,
                ingress_identity=ingresses[0] if ingresses else None,
            )

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
        self._active_cancel_reason = reason
        # ROOT of the "stop then continue loses all context" bug: a raw asyncio
        # ``task.cancel()`` poisons the runner coroutine before it reaches a
        # cancellation checkpoint, so the run is abandoned at ``status=RUNNING``
        # with no messages and never persisted. The runner only records an
        # interrupted turn through its COOPERATIVE cancel path. So ask the
        # runtime to cancel cooperatively and let the turn finish on its own —
        # the runner then persists the run and ``upsert_run`` backfills the
        # user's question into history, so "continua" keeps context. Only if
        # cooperative cancel is unavailable or stalls do we hard-cancel.
        # ``salvage`` (typed-burst merge) skips this — that path wants instant
        # preemption and re-merges the input forward.
        cancelled_cleanly = False
        if salvaged_msg is None:
            try:
                if await self._agent.request_cancel(self.session_id):
                    await asyncio.wait_for(
                        asyncio.shield(task), timeout=COOP_CANCEL_DRAIN_TIMEOUT
                    )
                    cancelled_cleanly = True
                    elog(
                        "stream.barge_in.coop_drained",
                        session_id=self.session_id, reason=reason,
                    )
            except (asyncio.CancelledError, Exception):
                # request_cancel unavailable, the drain timed out, or it
                # failed — fall through to the hard cancel below.
                pass
        if not cancelled_cleanly and not task.done():
            # Hard cancel fallback. A well-behaved runner honours it in
            # milliseconds. If a provider swallows it and keeps generating, do
            # NOT block the dispatch loop (this runs under ``_dispatch_lock``)
            # — detach the task and move on so the next turn can dispatch.
            # ``_suppress_runner_completion`` keeps late zombie frames off the
            # wire.
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=BARGE_IN_DRAIN_TIMEOUT)
            except asyncio.TimeoutError:
                elog(
                    "stream.barge_in.drain_timeout",
                    level="warning",
                    session_id=self.session_id,
                    reason=reason,
                )
                self._detached_turns.add(task)
                task.add_done_callback(self._detached_turns.discard)
            except (asyncio.CancelledError, Exception):
                pass
        self._current_turn = None
        self._current_turn_msg = None
        salvaged_origin = self._current_turn_origin
        salvaged_ingress = self._current_turn_ingress
        self._current_turn_origin = None
        self._current_turn_ingress = None
        self._current_turn_token = None
        self._current_turn_started = False
        self._active_cancel_reason = None
        if suppress_completion:
            self._suppress_runner_completion = False
        if salvaged_msg is not None:
            self._event_origins[id(salvaged_msg)] = salvaged_origin
            self._event_ingresses[id(salvaged_msg)] = salvaged_ingress
            self._pending_burst.insert(0, salvaged_msg)

    def _snapshot_video_frames(self, ingress_identity: Any = None) -> list[dict[str, Any]]:
        """Persist the latest frame per stream as image attachments.

        The ``<stream>-snapshot.jpg`` filename is load-bearing — the
        agent's attachment-context block uses it to tell the LLM
        whether the frame is a webcam or screen feed. Without that
        hint, models reach for an MCP screenshot tool instead of
        reading the attached image.
        """
        out: list[dict[str, Any]] = []
        ingress_key = _ingress_key(ingress_identity)
        consumed: list[tuple[tuple[Any, ...], str]] = []
        for key, ring in list(self._video_buffers.items()):
            owner_key, stream = key
            if owner_key != ingress_key:
                continue
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
            consumed.append(key)
        for key in consumed:
            self._video_buffers.pop(key, None)
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
        author: dict | None = None,
        execution_origin: Any = None,
        ingress_identity: Any = None,
        turn_token: object | None = None,
    ) -> dict[str, Any]:
        sess = self._session

        async def publish(evt: Event) -> None:
            # A detached provider that ignored cancellation is no longer the
            # session's active runner. Drop every late frame rather than
            # looking at mutable session state, which may now belong to a turn
            # from another authenticated device.
            if turn_token is not None and sess._current_turn_token is not turn_token:
                return
            await sess._publish(evt, ingress_identity=ingress_identity)
        accumulated: list[str] = []
        audio_started = False
        audio_chunks = 0
        stream_error: BaseException | None = None
        cancelled = False
        cancel_reason: str | None = None
        spoken_tools: set[str] = set()
        # Tracks whether we've published OutReasoning(active=True) without a
        # matching active=False yet. The agent's "Thinking..."/"Loading
        # context..." statuses become this boolean flag; anything that ends
        # the thinking phase (a tool starting, the first text token, turn
        # end) flips it back via ``_end_reasoning``.
        reasoning_active = False

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

        async def _end_reasoning() -> None:
            """Publish ``OutReasoning(active=False)`` once, if active."""
            nonlocal reasoning_active
            if reasoning_active:
                reasoning_active = False
                await publish(OutReasoning(
                    session_id=session_id,
                    seq=sess.next_seq(),
                    ts_ms=now_ms(),
                    active=False,
                ))

        async def on_status(status_text: str) -> None:
            # Plain "Thinking..."/"Loading context..." → the boolean
            # reasoning flag (the client renders its own indicator; the
            # server does not ship a "Thinking..." UI string). Tool JSON
            # and structured envelopes stay as OutToolStatus — real data.
            nonlocal reasoning_active
            # Session compaction (vision §2) rides the on_status channel as
            # a structured envelope, not tool JSON. Lift it into its own
            # typed frame so clients draw a compaction affordance instead
            # of leaking the raw JSON into a status line. Compaction is
            # visible activity, so it also ends any pending reasoning span.
            comp = parse_compaction_status(status_text)
            if comp is not None:
                await _end_reasoning()
                await publish(SessionCompacted(
                    session_id=session_id,
                    seq=sess.next_seq(),
                    ts_ms=now_ms(),
                    phase=comp["phase"],
                    folded_runs=comp["folded_runs"],
                    kept_runs_count=comp["kept_runs_count"],
                    summary_chars=comp["summary_chars"],
                    tokens_before=comp["tokens_before"],
                    tokens_after=comp["tokens_after"],
                ))
                # After the compaction "done" frame lands, push a fresh
                # ContextReport so the always-visible context panel updates
                # immediately — before the turn completes — showing the
                # freed space. Best-effort: measurement must never block
                # the status handler.
                if comp["phase"] == "done":
                    try:
                        from src.core.context_report import build_context_report
                        report = build_context_report(self._agent, session_id)
                        if report is not None:
                            await publish(ContextReport(
                                session_id=session_id,
                                seq=sess.next_seq(),
                                ts_ms=now_ms(),
                                report=report,
                            ))
                    except Exception:  # noqa: BLE001
                        pass
                return
            if is_reasoning_status(status_text):
                if not reasoning_active:
                    reasoning_active = True
                    await publish(OutReasoning(
                        session_id=session_id,
                        seq=sess.next_seq(),
                        ts_ms=now_ms(),
                        active=True,
                    ))
                return
            # A tool (or other structured update) is the visible activity
            # now, so the thinking phase is over until the next "Thinking..."
            await _end_reasoning()
            await publish(OutToolStatus(
                session_id=session_id,
                seq=sess.next_seq(),
                ts_ms=now_ms(),
                text=status_text,
            ))

        # A spawned child session (a delegated sub-agent) runs inside this
        # turn but is its OWN session to the app. Publish its deltas / tool
        # frames onto THIS turn's outbound queue, tagged with the child's
        # session_id — the app routes them to the child session exactly like
        # any session's stream. Goes through the same serialized pump as the
        # parent's own frames, so there's no concurrent-write race on the
        # client socket. ``ev_sid`` falls back to the parent for safety.
        async def child_emit(frame: dict[str, Any]) -> None:
            # Map via the SAME builder the detached broadcast path uses
            # (``gateway._broadcast_child_frame``) so the two never drift — this
            # inline chain had already fallen behind (no ``response``/``seed``).
            # ``ev_sid`` falls back to the parent for safety.
            if not frame.get("session_id"):
                frame = {**frame, "session_id": session_id}
            evt = child_frame_to_event(frame, seq=sess.next_seq(), ts_ms=now_ms())
            if evt is not None:
                await publish(evt)

        speaker = asyncio.create_task(speaker_task()) if (speak and self._tts) else None
        from src.stream.child_stream import (
            child_frame_to_event, install_child_stream_emitter, reset_child_stream_emitter,
        )
        _child_emit_tok = install_child_stream_emitter(child_emit)
        from src.core.execution_origin import (
            install_execution_origin, reset_execution_origin,
        )
        _execution_origin_tok = install_execution_origin(execution_origin)

        try:
            try:
                async for event in self._agent.run_stream(
                    message=text,
                    user_id=client_id,
                    session_id=session_id,
                    attachments=attachments,
                    on_status=on_status,
                    author=author,
                ):
                    # Engagement signal — soonest reliable indicator
                    # the prompt reached the provider. Idempotent bool,
                    # safe to set on every event. See ``_dispatch_turn``.
                    if turn_token is None or sess._current_turn_token is turn_token:
                        sess._current_turn_started = True
                    kind = event.get("kind")
                    if kind == "delta":
                        delta = event.get("text") or ""
                        if not delta:
                            continue
                        # Visible output has started — reasoning is over.
                        await _end_reasoning()
                        accumulated.append(delta)
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
                cancelled = True
                cancel_reason = getattr(sess, "_active_cancel_reason", None)
                task = asyncio.current_task()
                if task is not None and hasattr(task, "uncancel"):
                    while task.cancelling():
                        task.uncancel()
                if speaker is not None:
                    await text_q.put(None)
                elog(
                    "stream.turn.cancelled",
                    session_id=session_id,
                    reason=cancel_reason or "unexpected",
                    suppress_completion=sess._suppress_runner_completion,
                )
            except Exception as e:  # noqa: BLE001
                stream_error = e
                if speaker is not None:
                    await text_q.put(None)
                logger.warning("stream turn failed: %s", e)
        finally:
            reset_execution_origin(_execution_origin_tok)
            reset_child_stream_emitter(_child_emit_tok)
            if speaker is not None:
                if cancelled:
                    # Barge-in / stop: do NOT drain the TTS tail. The
                    # speaker is a sibling task that is not cancelled by
                    # the runner's own cancellation, so awaiting its full
                    # drain here (up to SPEAKER_DRAIN_TIMEOUT) blocks
                    # ``_cancel_active_turn``'s ``await task`` — and that
                    # runs inside the single dispatch loop holding
                    # ``_dispatch_lock``, so a voice barge-in mid-TTS would
                    # freeze the whole inbound pipeline (no next utterance,
                    # no follow-up interrupt) for up to 20s. Cancel it now.
                    speaker.cancel()
                    try:
                        await speaker
                    except (asyncio.CancelledError, Exception):
                        pass
                else:
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
            elif not full_text and cancelled:
                # Any cancel — an explicit user stop (reason set) or an
                # unexpected provider-side cancel (reason None) — reads as
                # "interrupted", not "the agent produced nothing". The
                # reason is always set for a user Interrupt (wire.py), so
                # gating on ``cancel_reason is None`` showed the wrong
                # "No text response" copy for a deliberate early stop.
                full_text = _CANCELLED_TURN_FALLBACK_TEXT
            elif not full_text:
                full_text = _EMPTY_TURN_FALLBACK_TEXT
                elog(
                    "stream.turn.empty_output",
                    session_id=session_id,
                    speak=speak,
                )

            from src.channels.base import parse_response_markers
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

            # Safety net: clear reasoning before the terminal frames in case
            # the turn produced no deltas (tool-only / empty / errored turn)
            # so the active flag never got flipped off above. Skipped when
            # terminal frames are being suppressed for a barge-in — there
            # the follow-up turn keeps the "thinking" state alive across the
            # gap, exactly like the suppressed OutTextFinal/TurnComplete.
            if not sess._suppress_runner_completion:
                await _end_reasoning()
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

            # Push the fresh context-window composition so a client's
            # always-visible /context panel updates in realtime as the
            # conversation grows. Best-effort — measurement must never
            # break a turn, and text-only channels simply ignore the frame.
            try:
                from src.core.context_report import build_context_report

                report = build_context_report(self._agent, session_id)
                if report is not None:
                    await publish(ContextReport(
                        session_id=session_id,
                        seq=sess.next_seq(),
                        ts_ms=now_ms(),
                        report=report,
                    ))
            except Exception as e:  # noqa: BLE001
                logger.debug("context_report emit failed: %s", e)

            elog(
                "stream.turn.end",
                session_id=session_id,
                response_chars=len(clean),
                audio_chunks=audio_chunks,
                spoken_tools=len(spoken_tools),
                errored=(stream_error is not None) or cancelled,
                cancelled=cancelled,
                cancel_reason=cancel_reason,
            )

        return {
            "text": clean,
            "attachments": att_list,
            "audio_chunks": audio_chunks,
            "errored": stream_error is not None,
        }


__all__ = ["StreamSession", "StreamTurnRunner"]
