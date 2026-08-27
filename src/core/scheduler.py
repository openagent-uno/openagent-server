"""Cron-based task scheduler. Tasks are stored in SQLite and survive reboots.

Owns two DB-polled responsibilities:

1. Legacy ``scheduled_tasks`` rows — a single prompt on a cron. The
   scheduler fires ``agent.run()`` when ``next_run <= now``.
2. Workflow rows + their request queue (Phase 2):
   - ``workflow_tasks`` rows with ``trigger_kind in ('schedule','hybrid')``
     and ``next_run_at <= now`` fire via ``WorkflowExecutor``.
   - ``workflow_run_requests`` rows enqueued by the workflow-manager
     MCP (or the gateway's ``POST /api/workflows/{id}/run``) are
     atomically claimed and executed against the same executor.

The workflow executor is constructed lazily on the first tick — only
if workflows exist — so the existing scheduled-task path carries zero
overhead for users who never adopt workflows.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Callable, TYPE_CHECKING

import time

from src.memory.schedule import (
    is_one_shot_expression,
    next_run_for_expression,
)

if TYPE_CHECKING:
    from src.core.agent import Agent
    from src.memory.db import MemoryDB
    from src.workflow.executor import WorkflowExecutor

from src.core.logging import elog


CHECK_INTERVAL = 30  # seconds between checking for due tasks

# A "stop this run" request crosses the process boundary as a row flagged
# ``status='cancelling'`` (written by the workflow-manager / scheduler MCP
# subprocess, which can't reach the in-process executor or agent). The
# scheduler drains those flags on a dedicated fast loop — far tighter than
# CHECK_INTERVAL — so "completely stop" actually feels immediate instead of
# waiting up to a full 30 s due-task tick.
CANCEL_CHECK_INTERVAL = 2  # seconds between draining cancellation requests
# Battito di vita del loop. Il loop tick ogni 2s in SILENZIO: quando si ferma
# (6-ago-2026: fermo ~17 minuti dopo un riavvio, con 4 delivery in coda che
# nessuno riclamava) non resta traccia di nulla, e "zitto perche' non c'e'
# lavoro" e' indistinguibile da "morto". Un beat ogni 5 minuti costa ~288 righe
# al giorno e rende la differenza misurabile — da fuori bastano due beat
# mancati per sapere che il reaper e i drain non stanno piu' girando.
BEAT_INTERVAL = 300

# Per-tick cap on how many ``cancelling`` rows a single drain processes. A
# healthy system has ~0 at any moment; a high cap only matters after a mass
# crash left stale flags. Hitting the cap is not data loss — the next tick
# (2 s later) picks up the remainder — but we log it so a flood is visible.
_CANCEL_SCAN_LIMIT = 500


def _env_float(name: str, default: float) -> float:
    """A float env override that falls back to ``default`` on unset/garbage."""
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    """An int env override that falls back to ``default`` on unset/garbage."""
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


# Bounded event-delivery dispatch. ``_drain_event_deliveries`` claims the
# ``received`` rows the events-manager MCP enqueued and dispatches each as a
# DETACHED turn. An event turn can be very heavy (a ~250k-token support-thread
# turn), so an UNBOUNDED drain — the old behaviour, which claimed + dispatched a
# whole burst at once — could saturate the runtime and hang the entire pipeline:
# when ~66 deliveries were re-enqueued at once (a manual backfill), all ~66 heavy
# turns fired concurrently and NO delivery completed for ~19 min, including
# brand-new inbound tickets. The DB's ``received`` rows are already the queue;
# this cap is the missing piece. At most OPENAGENT_EVENT_DISPATCH_CONCURRENCY
# event turns run at once; the rest stay ``received`` (unclaimed) and are picked
# up on later ticks as slots free. Read live each tick (like the stale-sweep
# knobs) so an operator can retune the ceiling without a redeploy.
_EVENT_DISPATCH_CONCURRENCY_ENV = "OPENAGENT_EVENT_DISPATCH_CONCURRENCY"
_EVENT_DISPATCH_CONCURRENCY_DEFAULT = 4


# Periodic stale-orphan sweep for event deliveries. The startup reap
# (``server.py`` → ``reap_orphan_event_deliveries``) only recovers CRASH
# orphans: a crash → restart → reap. A delivery orphaned WITHOUT a restart —
# a detached dispatch task that dies silently while the process keeps running —
# would otherwise sit ``running``/claimed until the next restart. This sweep
# re-enqueues those on a cadence, but AGE-GATED so a legitimately-running turn
# is never double-dispatched: only deliveries claimed longer ago than
# STALE_SWEEP_AGE (default 1800 s = 2× the OPENAGENT_CHAT_TURN_TIMEOUT 900 s
# single-turn wall-clock cap) are eligible. Both knobs are read live each tick
# so an operator can retune without a redeploy.
_STALE_SWEEP_INTERVAL_ENV = "OPENAGENT_EVENT_STALE_SWEEP_INTERVAL_SECONDS"
_STALE_SWEEP_AGE_ENV = "OPENAGENT_EVENT_STALE_SWEEP_AGE_SECONDS"
_STALE_SWEEP_INTERVAL_DEFAULT = 600.0   # 10 min between sweeps
_STALE_SWEEP_AGE_DEFAULT = 1800.0       # 30 min = 2× the single-turn cap


# Resource broadcast hook. ``AgentServer`` plugs the Gateway's
# ``broadcast_resource_sync`` in here so that internal mutations (a
# one-shot task auto-disabling itself, a workflow run starting from
# cron) reach the desktop app without going through the REST handlers.
BroadcastHook = Callable[[str, str, "str | None"], None]


def _no_broadcast(resource: str, action: str, id: str | None = None) -> None:
    """Default hook used when the scheduler runs without a gateway
    attached (unit tests, headless invocations)."""
    return None


# Cap the stored task-run output preview. The full transcript still lives
# in the agent's session history; ``task_runs.output`` is only a preview
# for the dashboard's run list, so a chatty turn can't bloat the DB.
_MAX_TASK_RUN_OUTPUT = 4000

_CHILD_FAILURE_STATUSES = {"ERROR", "FAILED"}
_CHILD_CANCELLED_STATUSES = {"CANCELLED"}
_CHILD_INCOMPLETE_STATUSES = {"PAUSED", "PENDING", "RUNNING"}

# The runtime does not raise when a run exhausts its tool-call budget: it feeds
# the model "Tool call limit reached" and lets it write a final answer. The run
# then ends COMPLETED, so the task recorded `success` — while having done
# nothing. Measured on clickup-task-quality-audit: three runs in a row reported
# success with 0/4 lists audited, 0 tasks checked and 0 mutations, and only
# firing the task by hand revealed it. A budget that truncates the work must
# leave a mark in the run history, exactly as the timeout budget does.
_TOOL_LIMIT_MARKER = "Tool call limit reached"


def _run_was_truncated(run: dict) -> bool:
    """True when the stored child run shows the tool-call budget cut it short."""
    try:
        return _TOOL_LIMIT_MARKER in json.dumps(run, default=str)
    except Exception:  # noqa: BLE001 — detection must never fail a run
        return False


def _status_name(status: object) -> str:
    """Normalize runtime RunStatus enum/string values from session JSON."""
    value = getattr(status, "value", status)
    return str(value or "").upper()


def _string_preview(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    try:
        return json.dumps(value, ensure_ascii=False)[:_MAX_TASK_RUN_OUTPUT]
    except Exception:  # noqa: BLE001
        return str(value)[:_MAX_TASK_RUN_OUTPUT]


def _child_run_error_preview(run: dict, *, fallback: str) -> str:
    """Best-effort human error preview from a stored child-session run."""
    content = _string_preview(run.get("content"))
    if content:
        return content
    events = run.get("events")
    if isinstance(events, list):
        for event in reversed(events):
            if not isinstance(event, dict):
                continue
            for key in ("error", "message", "content", "text"):
                preview = _string_preview(event.get(key))
                if preview:
                    return preview
    return fallback


def _durable_child_sessions() -> bool:
    """Whether a scheduled firing persists as a durable child session.

    Default ON: each firing runs as a unique per-run session
    (``scheduler:{task}:{run}``) that survives for navigation + follow-up,
    via ``core.child_session.run_child_session``. The per-run uniqueness is
    what removes the issue-#5 root cause (a *reused* session whose compacted
    transcript made the next firing exit early), so persisting is safe.

    Set ``OPENAGENT_SCHEDULER_DURABLE_SESSIONS=0`` to revert to the exact
    legacy behavior (a reused ``scheduler:{task}`` session wiped via
    ``forget_session`` after every fire) as a safety hatch."""
    return os.environ.get("OPENAGENT_SCHEDULER_DURABLE_SESSIONS", "1").strip() not in ("0", "false", "no")


class Scheduler:
    """Background scheduler that runs agent prompts on cron schedules.

    Tasks are stored in SQLite — they survive process restarts and reboots.
    On startup, recalculates next_run for all tasks to handle missed runs.
    """

    def __init__(
        self,
        db: MemoryDB,
        agent: Agent,
        broadcast: BroadcastHook | None = None,
    ):
        self.db = db
        self.agent = agent
        self._task: asyncio.Task | None = None
        # Dedicated loop that drains ``status='cancelling'`` rows. Kept
        # separate from ``_task`` so the heavy due-task scan stays on its
        # 30 s cadence while stop requests get acted on within seconds.
        self._cancel_task: asyncio.Task | None = None
        # Lazy — created on first workflow tick.
        self._workflow_executor: WorkflowExecutor | None = None
        self._broadcast: BroadcastHook = broadcast or _no_broadcast
        # In-flight per-tick dispatches. ``_check_and_run`` spawns each
        # due task / schedule / queue request as its own ``asyncio.Task``
        # so different workflows actually run concurrently — the previous
        # design awaited each run inline, serialising the whole tick.
        self._workflow_tasks: set[asyncio.Task] = set()
        # run_id → the ``asyncio.Task`` driving that run, so the
        # cancellation drain can hard-stop a specific in-flight run by
        # cancelling its task. Keyed by ``workflow_runs.id`` /
        # ``task_runs.id`` (both UUIDs, so the two maps never collide).
        # Entries are added the moment the run_id is known and removed in
        # the run's ``finally`` — the map only ever holds live runs.
        self._workflow_run_tasks: dict[str, asyncio.Task] = {}
        self._scheduled_run_tasks: dict[str, asyncio.Task] = {}
        # Number of event-delivery dispatch turns currently in flight. The
        # bound that stops a burst from jamming the pipeline: each drain tick
        # claims + dispatches at most ``(concurrency - in_flight)`` deliveries,
        # so at most ``OPENAGENT_EVENT_DISPATCH_CONCURRENCY`` heavy event turns
        # run at once and the rest wait in the DB queue. Incremented
        # synchronously in ``_spawn_event_dispatch`` (before the task is
        # scheduled, so the next tick's free-slot math already counts it) and
        # decremented in that task's ``finally`` — a crashing/cancelled turn
        # still frees its slot. Single-threaded asyncio: no lock needed.
        self._event_dispatch_in_flight: int = 0
        # Monotonic timestamp of the last periodic stale-orphan sweep. Set at
        # the top of ``_cancellation_loop`` so the first sweep fires one
        # interval AFTER boot (the startup reap already covers t=0), then gated
        # to at most once per STALE_SWEEP_INTERVAL.
        self._last_stale_sweep: float = 0.0

    def _next_run(
        self,
        cron_expression: str,
        base: float | None = None,
        timezone: str | None = None,
    ) -> float:
        return next_run_for_expression(cron_expression, base, timezone)

    @staticmethod
    def _task_tz(task: dict) -> str | None:
        """The zone a task's cron is read in; None = UTC (the default).

        ``.get`` rather than ``[]`` so a row read before the timezone
        migration ran (or a hand-built dict in a test) still schedules."""
        return task.get("timezone") or None

    async def start(self) -> None:
        """Start the scheduler background loop."""
        if self._task and not self._task.done():
            return
        await self.db.connect()
        await self._recalculate_next_runs()
        self._task = asyncio.create_task(self._loop())
        self._cancel_task = asyncio.create_task(self._cancellation_loop())
        elog("scheduler.start")

    async def stop(self) -> None:
        """Stop the scheduler."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._cancel_task:
            self._cancel_task.cancel()
            try:
                await self._cancel_task
            except asyncio.CancelledError:
                pass
            self._cancel_task = None
        # Let in-flight runs finish — matches the existing fire-and-forget
        # semantics of ``run_task``. Cancelling here would strand
        # ``workflow_runs`` rows in ``running`` and lose ai-prompt
        # transcripts mid-stream.
        if self._workflow_tasks:
            await asyncio.gather(*self._workflow_tasks, return_exceptions=True)
        elog("scheduler.stop")

    def _spawn_workflow(self, coro) -> asyncio.Task:
        """Dispatch ``coro`` as a tracked background task.

        The set is the only handle the scheduler keeps; ``stop()``
        drains it on shutdown so no run is silently abandoned. Without
        the strong reference, ``asyncio`` may garbage-collect a still-
        running task (see ``asyncio.create_task`` docs).
        """
        task = asyncio.create_task(coro)
        self._workflow_tasks.add(task)
        task.add_done_callback(self._workflow_tasks.discard)
        return task

    async def _recalculate_next_runs(self) -> None:
        """On startup, recalculate next_run for all enabled tasks AND
        every workflow with a cron schedule."""
        tasks = await self.db.get_tasks(enabled_only=True)
        now = time.time()
        for task in tasks:
            try:
                if is_one_shot_expression(task["cron_expression"]):
                    if task.get("last_run"):
                        await self.db.update_task(task["id"], enabled=0, next_run=None)
                    continue
                await self.db.update_task(
                    task["id"],
                    next_run=self._next_run(
                        task["cron_expression"], now, self._task_tz(task),
                    ),
                )
            except ValueError as e:
                elog("scheduler.invalid_cron", level="error", task=task["name"], error=str(e))

        # Per-block schedules in ``workflow_schedules``. Each row is a
        # trigger-schedule block; recalculate its next_run on boot so
        # schedules that elapsed while we were down fire once on next
        # tick rather than stampede once each missed window.
        try:
            schedules = await self.db.list_schedules(enabled_only=True)
        except Exception as e:  # noqa: BLE001
            elog("scheduler.schedules_recalc_skipped", level="warning", error=str(e))
            schedules = []
        for sched in schedules:
            cron = sched.get("cron_expression")
            if not cron:
                continue
            try:
                if is_one_shot_expression(cron):
                    if sched.get("last_run_at"):
                        await self.db.update_schedule(
                            sched["id"], enabled=False,
                        )
                    continue
                await self.db.update_schedule(
                    sched["id"], next_run_at=self._next_run(cron, now),
                )
            except ValueError as e:
                elog(
                    "scheduler.invalid_cron", level="error",
                    schedule=sched.get("id"), error=str(e),
                )

    async def _loop(self) -> None:
        """Main loop: check for due tasks every CHECK_INTERVAL seconds."""
        while True:
            try:
                await self._check_and_run()
            except Exception as e:
                elog("scheduler.loop_error", level="error", error=str(e))
            await asyncio.sleep(CHECK_INTERVAL)

    # ── Run cancellation (cross-process "completely stop") ──
    #
    # The workflow-manager / scheduler MCP subprocesses can't reach the
    # in-process executor or agent, so a "stop this run" request crosses the
    # boundary as a row flagged ``status='cancelling'``. This loop turns the
    # flag into a real hard stop: cancel the ``asyncio.Task`` driving the run
    # (which unwinds the executor / agent turn, aborting any in-flight model
    # call), then let the run's own cancellation handler finalize the row to
    # ``cancelled``. A ``cancelling`` row with no live task — a stale flag
    # left by a crash, or one written a hair after the run finished — is
    # finalized directly so the badge never sticks.

    async def _cancellation_loop(self) -> None:
        """Fast cross-process signal loop, every CANCEL_CHECK_INTERVAL.

        Drains the two cross-process hand-offs an out-of-process MCP
        subprocess uses to reach the in-process runtime: ``status='cancelling'``
        rows (a "completely stop" request) and ``task_run_requests`` rows (a
        "run now" request). Kept off the heavy 30 s due-task scan so both feel
        near-immediate.

        It also hosts the periodic stale-orphan sweep (throttled to
        STALE_SWEEP_INTERVAL via a monotonic gate rather than its own task) so
        an event delivery orphaned without a restart is still recovered."""
        # Anchor the throttle at loop start so the first sweep is one interval
        # out — the startup reap already handled boot orphans at t=0.
        self._last_stale_sweep = time.monotonic()
        self._last_beat = 0.0
        self._ticks = 0
        while True:
            self._ticks += 1
            try:
                await self._drain_cancellations()
            except Exception as e:  # noqa: BLE001
                elog("scheduler.cancel_loop_error", level="error", error=str(e))
            try:
                await self._drain_task_run_requests()
            except Exception as e:  # noqa: BLE001
                elog("scheduler.run_request_loop_error", level="error", error=str(e))
            # FAST lease reclaim, every tick: a delivery whose claim lease has
            # lapsed (its dispatch runner stopped heartbeating — a frozen turn /
            # dead process) is re-enqueued in ~LEASE_TTL, not the coarse 30-min
            # stale-sweep age. Runs BEFORE the drain so a just-recovered row is
            # re-dispatched in the same tick. Only touches rows with a NON-NULL
            # ``claim_expires``, so pre-existing (legacy) in-flight rows are never
            # reclaimed here — the age-gated sweep below still covers those.
            # Battito: prova che il loop e' vivo, con la profondita' della coda.
            # Va PRIMA del reap/drain cosi' esce anche se uno di quelli si pianta.
            try:
                _now = time.monotonic()
                if _now - self._last_beat >= BEAT_INTERVAL:
                    self._last_beat = _now
                    _pending = None
                    if self.db is not None and hasattr(self.db, "count_open_event_deliveries"):
                        try:
                            _pending = await self.db.count_open_event_deliveries()
                        except Exception:  # noqa: BLE001 — il beat non deve mai fallire
                            _pending = None
                    elog("scheduler.beat", ticks=self._ticks,
                         pending=_pending if _pending is not None else -1)
            except Exception as e:  # noqa: BLE001
                elog("scheduler.beat_error", level="warning", error=str(e))
            try:
                await self._reap_expired_event_leases()
            except Exception as e:  # noqa: BLE001
                elog("scheduler.lease_reap_loop_error", level="error", error=str(e))
            try:
                await self._drain_event_deliveries()
            except Exception as e:  # noqa: BLE001
                elog("scheduler.event_delivery_loop_error", level="error", error=str(e))
            # Periodic, age-gated stale-orphan sweep. Self-guarded: a sweep
            # error must never break this loop, so the interval gate and the DB
            # call are wrapped whole. Cheap (one or two UPDATEs) and fires at
            # most once per STALE_SWEEP_INTERVAL.
            try:
                interval = _env_float(
                    _STALE_SWEEP_INTERVAL_ENV, _STALE_SWEEP_INTERVAL_DEFAULT
                )
                nowmono = time.monotonic()
                if nowmono - self._last_stale_sweep >= interval:
                    self._last_stale_sweep = nowmono
                    await self._sweep_stale_event_deliveries()
            except Exception as e:  # noqa: BLE001
                elog("scheduler.stale_sweep_loop_error", level="error", error=str(e))
            # Guarded-change watcher: resolve any auto-applied config/template
            # change whose watch window has elapsed — measure the target's real
            # failure-rate and auto-rollback + blocklist it if it regressed. Each
            # row is self-gated by its own ``check_after`` timestamp, so this is a
            # no-op (one indexed SELECT) unless a guarded change is actually
            # pending. Self-guarded: a watcher error must never break the loop.
            try:
                if hasattr(self.db, "reap_guarded_changes"):
                    await self.db.reap_guarded_changes()
            except Exception as e:  # noqa: BLE001
                elog("scheduler.guarded_change_loop_error", level="error", error=str(e))
            await asyncio.sleep(CANCEL_CHECK_INTERVAL)

    async def _reap_expired_event_leases(self) -> None:
        """Fast-loop lease reclaim: re-enqueue deliveries whose claim lease has
        lapsed (heartbeat stopped → frozen/dead runner), so recovery ≈ LEASE_TTL
        rather than the 30-min stale-sweep age. Only touches rows with a non-NULL
        ``claim_expires``, so a legacy in-flight row (NULL lease at deploy) is
        untouched. Defensive: an older MemoryDB without the method no-ops."""
        if self.db is None:
            return
        if not hasattr(self.db, "reap_expired_event_leases"):
            return
        await self.db.reap_expired_event_leases()

    async def _sweep_stale_event_deliveries(self) -> None:
        """Recover event deliveries orphaned WITHOUT a process restart.

        A detached ``dispatch_event`` task can die silently — its turn task
        cancelled, an unhandled error in the background coroutine — while the
        server keeps running. The startup reap never sees that row (there is no
        restart), so it sits ``running``/claimed until the next deploy. This
        re-enqueues it via ``MemoryDB.reap_stale_event_deliveries``, which
        AGE-GATES on ``claimed_at`` (default 2× the single-turn wall-clock cap)
        so a legitimately-running turn is never re-enqueued into a second
        concurrent dispatch. The DB method logs (``mode='stale-sweep'``) only
        when it acts and stays silent otherwise."""
        if self.db is None:
            return
        # Defensive: an older MemoryDB without the stale reap simply no-ops
        # (mirrors server.py's ``hasattr`` guard around the startup reap).
        if not hasattr(self.db, "reap_stale_event_deliveries"):
            return
        age = _env_float(_STALE_SWEEP_AGE_ENV, _STALE_SWEEP_AGE_DEFAULT)
        await self.db.reap_stale_event_deliveries(min_claim_age_seconds=age)

    async def _drain_task_run_requests(self) -> None:
        """Claim and fire every pending on-demand "run now" request.

        Each request fires its task immediately via ``run_task`` with
        ``trigger`` carried from the request row, leaving the task's cron
        schedule and enabled flag untouched. The firing is dispatched as its
        own tracked ``asyncio.Task`` so a long run can't stall the drain."""
        if self.db is None:
            return
        try:
            requests = await self.db.claim_pending_task_requests(
                limit=_CANCEL_SCAN_LIMIT,
            )
        except Exception as e:  # noqa: BLE001
            elog("scheduler.task_request_claim_failed", level="warning",
                 error=str(e) or type(e).__name__)
            return
        if len(requests) >= _CANCEL_SCAN_LIMIT:
            elog("scheduler.task_request_scan_capped", level="warning",
                 limit=_CANCEL_SCAN_LIMIT)
        for req in requests:
            task_id = req.get("task_id")
            task = await self.db.get_task(task_id) if task_id else None
            if task is None:
                # FK cascade should prevent this, but guard anyway.
                elog("scheduler.task_request_orphan", level="warning",
                     request_id=req.get("id"), task_id=task_id)
                continue
            elog("scheduler.run_now", name=task.get("name"),
                 request_id=req.get("id"))
            self._spawn_workflow(
                self.run_task(
                    task,
                    trigger=req.get("trigger") or "manual",
                    request_id=req.get("id"),
                )
            )

    def _event_dispatch_concurrency(self) -> int:
        """Max event-delivery turns allowed in flight at once (always >= 1).

        Read live each tick from ``OPENAGENT_EVENT_DISPATCH_CONCURRENCY`` so the
        ceiling can be retuned without a redeploy; garbage / unset falls back to
        the default, and a sub-1 value is clamped to 1 so the drain never wedges
        itself into dispatching nothing forever."""
        return max(1, _env_int(
            _EVENT_DISPATCH_CONCURRENCY_ENV, _EVENT_DISPATCH_CONCURRENCY_DEFAULT,
        ))

    def _spawn_event_dispatch(self, coro) -> asyncio.Task:
        """Dispatch one event-delivery turn as a slot-bounded background task.

        Reserves an in-flight slot SYNCHRONOUSLY (before the task is scheduled,
        so a same-tick or next-tick free-slot calculation already accounts for
        it) and releases it in a ``finally`` so a crashing or cancelled turn
        still frees its slot. Tracked in ``_workflow_tasks`` like every other
        detached run, so ``stop()`` drains it on shutdown and the GC can't
        collect a still-running task."""
        self._event_dispatch_in_flight += 1

        async def _runner() -> None:
            try:
                await coro
            finally:
                self._event_dispatch_in_flight -= 1

        task = asyncio.create_task(_runner())
        self._workflow_tasks.add(task)
        task.add_done_callback(self._workflow_tasks.discard)
        return task

    async def _drain_event_deliveries(self) -> None:
        """Claim and dispatch pending event deliveries, BOUNDED so a burst can
        never jam the pipeline.

        The ``events-manager`` MCP subprocess cannot reach the in-process
        runtime, so ``trigger_event`` inserts an ``event_deliveries`` row with
        ``claimed_at IS NULL``; this drain claims it (atomically) and runs the
        bound action via the shared ``dispatch_event``. The webhook listener
        and the REST trigger dispatch in-process and never hit this path (they
        create their rows already ``claimed``).

        BOUND — the fix for the burst-jam. An event turn can be very heavy, so
        this claims + dispatches at most ``(concurrency - in_flight)`` deliveries
        per tick — never more than can currently run under
        ``OPENAGENT_EVENT_DISPATCH_CONCURRENCY``. The claim ``limit`` is EXACTLY
        the number of free slots, so a delivery is never claimed unless it is
        dispatched in the same breath (a claimed-but-undispatched row would
        become a stuck orphan). Deliveries beyond the cap stay ``received``
        (unclaimed) — the DB rows ARE the queue — and later ticks pick them up
        as slots free. The loop keeps running every tick; when every slot is
        busy it simply claims nothing and returns, so a slow/hanging turn holds
        its slot without ever blocking the drain (the stale-orphan reaper
        recovers a truly stuck one)."""
        if self.db is None:
            return
        # Only claim what we can immediately dispatch. When the runtime is
        # saturated (in_flight == concurrency) free is 0 and we touch nothing
        # this tick — the queued ``received`` rows wait in the DB for a slot.
        free = self._event_dispatch_concurrency() - self._event_dispatch_in_flight
        if free <= 0:
            # Say so. A saturated dispatcher and an empty queue look identical
            # from outside — nothing runs, nothing is logged — and an in-flight
            # counter that never came back down stalls the agent in silence.
            # Throttled so a legitimately busy runtime doesn't spam the log.
            now = time.time()
            if now - getattr(self, "_last_saturation_log", 0.0) >= 60.0:
                self._last_saturation_log = now
                elog("scheduler.event_dispatch_saturated", level="warning",
                     in_flight=self._event_dispatch_in_flight,
                     concurrency=self._event_dispatch_concurrency())
            return
        try:
            deliveries = await self.db.claim_pending_event_deliveries(
                limit=free,
            )
        except Exception as e:  # noqa: BLE001
            elog("scheduler.event_claim_failed", level="warning",
                 error=str(e) or type(e).__name__)
            return
        if not deliveries:
            return
        from src.core.event_dispatcher import dispatch_event
        for dl in deliveries:
            event = await self.db.get_event(dl.get("event_id"))
            if event is None:
                elog("scheduler.event_delivery_orphan", level="warning",
                     delivery_id=dl.get("id"), event_id=dl.get("event_id"))
                continue
            # Per-event circuit breaker: an open breaker parks this (already
            # claimed) delivery ``blocked`` and skips dispatch — no slot spent,
            # the row is terminal so it is not re-claimed. Inert by default
            # (``is_event_breaker_tripped`` → False unless the flag is on).
            try:
                if await self.db.is_event_breaker_tripped(dl.get("event_id")):
                    await self.db.update_event_delivery(
                        dl["id"], status="blocked",
                        error="event circuit breaker open", finished_at=time.time(),
                    )
                    elog("scheduler.event_blocked", level="info",
                         delivery_id=dl.get("id"), event_id=dl.get("event_id"))
                    continue
            except Exception as e:  # noqa: BLE001
                elog("scheduler.event_breaker_check_failed", level="warning",
                     error=str(e) or type(e).__name__)
            try:
                payload = json.loads(dl.get("payload_json") or "{}")
            except (TypeError, ValueError):
                payload = {}
            self._spawn_event_dispatch(
                dispatch_event(
                    agent=self.agent, db=self.db, scheduler=self,
                    event=event, payload=payload,
                    delivery_id=dl["id"], source=dl.get("source") or "agent",
                    broadcast=self._broadcast,
                )
            )

    async def _drain_cancellations(self) -> None:
        """Act on every run flagged ``cancelling`` since the last drain."""
        if self.db is None:
            return
        # Workflow runs.
        try:
            wf_runs = await self.db.get_workflow_runs_by_status(
                "cancelling", limit=_CANCEL_SCAN_LIMIT,
            )
        except Exception as e:  # noqa: BLE001
            elog("scheduler.cancel_scan_failed", level="warning",
                 kind="workflow", error=str(e))
            wf_runs = []
        if len(wf_runs) >= _CANCEL_SCAN_LIMIT:
            elog("scheduler.cancel_scan_capped", level="warning",
                 kind="workflow", limit=_CANCEL_SCAN_LIMIT)
        for run in wf_runs:
            await self._apply_cancellation(
                run, registry=self._workflow_run_tasks, kind="workflow",
            )
        # Scheduled-task runs — same shape, different table.
        try:
            task_runs = await self.db.get_task_runs_by_status(
                "cancelling", limit=_CANCEL_SCAN_LIMIT,
            )
        except Exception as e:  # noqa: BLE001
            elog("scheduler.cancel_scan_failed", level="warning",
                 kind="scheduled_task", error=str(e))
            task_runs = []
        if len(task_runs) >= _CANCEL_SCAN_LIMIT:
            elog("scheduler.cancel_scan_capped", level="warning",
                 kind="scheduled_task", limit=_CANCEL_SCAN_LIMIT)
        for run in task_runs:
            await self._apply_cancellation(
                run, registry=self._scheduled_run_tasks, kind="scheduled_task",
            )

    async def _apply_cancellation(
        self, run: dict, *, registry: dict[str, asyncio.Task], kind: str,
    ) -> None:
        """Hard-stop one flagged run, or finalize it if no task owns it."""
        run_id = run.get("id")
        if not run_id:
            return
        task = registry.get(run_id)
        if task is not None:
            # Live run — request cancellation. Its handler finalizes the row
            # to ``cancelled``. Idempotent: re-cancelling an already-
            # cancelling task is harmless if it hasn't unwound by next drain.
            if not task.done():
                elog(
                    "scheduler.cancel_requested", kind=kind, run_id=run_id,
                    target=run.get("workflow_id") or run.get("task_id"),
                )
                task.cancel()
            return
        # No live task owns this row — a stale flag (crash) or a run that
        # finished between the MCP write and now. Finalize directly so the
        # UI doesn't sit on a phantom ``cancelling`` badge.
        await self._finalize_orphan_cancellation(run, kind=kind)

    async def _finalize_orphan_cancellation(self, run: dict, *, kind: str) -> None:
        """Mark a ``cancelling`` row ``cancelled`` when no live task owns it.

        Guarded so a write hiccup can't wedge the drain loop; the next tick
        retries. The flag only lands on rows that were ``running``, so a
        direct overwrite is safe — nothing else races to finalize it."""
        run_id = run.get("id")
        now = time.time()
        try:
            if kind == "workflow":
                await self.db.update_workflow_run(
                    run_id, status="cancelled", finished_at=now,
                    error="Stopped (no live run to cancel)",
                )
                self._broadcast("workflow", "updated", run.get("workflow_id"))
            else:
                await self.db.update_task_run(
                    run_id, status="cancelled", finished_at=now,
                    error="Stopped (no live run to cancel)",
                )
                self._broadcast("scheduled_task", "updated", run.get("task_id"))
            elog("scheduler.cancel_orphan_finalized", kind=kind, run_id=run_id)
        except Exception as e:  # noqa: BLE001
            elog("scheduler.cancel_finalize_failed", level="warning",
                 kind=kind, run_id=run_id, error=str(e))

    def _register_run(self, registry: dict[str, asyncio.Task], run_id: str) -> None:
        """Bind the current run's ``asyncio.Task`` to ``run_id`` so the drain
        can cancel it. Called from inside the run coroutine, where
        ``current_task()`` is exactly that run's task."""
        task = asyncio.current_task()
        if task is not None:
            registry[run_id] = task

    def _unregister_run(self, registry: dict[str, asyncio.Task], run_id: str) -> None:
        registry.pop(run_id, None)

    async def run_task(
        self, task: dict, *, trigger: str = "schedule", request_id: str | None = None,
        context: dict | None = None,
    ) -> None:
        """Execute a single task. Extension point: override or monkey-patch
        this to intercept specific tasks (e.g. auto-update, which uses a
        direct pip subprocess instead of going through the agent).

        Each firing runs as a durable per-run child session
        (``scheduler:{task}:{run}``) — vision §7's "chat with the user's seat
        empty": it appears in the owner's session list, can be opened to read
        the agent's full reasoning, and accepts follow-up messages. The
        per-run uniqueness (not a wiped, reused session) is what fixes
        issue #5. Set ``OPENAGENT_SCHEDULER_DURABLE_SESSIONS=0`` to revert to
        the legacy reused-session + ``forget_session`` behavior.

        ``context`` (optional) is an untrusted payload appended to the task
        prompt — used by the webhook Events channel to feed a delivery's data
        into a scheduled-task action. It is wrapped in a size-capped,
        clearly-delimited "data, not instructions" block. No existing caller
        passes it, so scheduled/manual firings are unchanged."""
        task_name = task["name"]
        # Build the effective prompt: the task prompt, plus (for event-driven
        # firings) an injection-guarded block carrying the delivery payload.
        effective_prompt = task["prompt"]
        if context:
            from src.core.event_dispatcher import render_payload_block
            effective_prompt = effective_prompt + render_payload_block(context)
        from src.core.execution_profile import (
            lean_local_event_scope,
            lean_local_task_scope,
            lean_local_tool_families,
            should_use_lean_local_scheduled_task,
            strict_local_only_scope,
        )
        from src.core.tool_scope import (
            current_tool_allowlist,
            reset_tool_allowlist,
            set_tool_allowlist,
        )
        from src.core.execution_policy import (
            current_execution_policy,
            narrow_execution_policy,
            reset_execution_policy,
            set_execution_policy,
            task_execution_policy,
        )
        from src.core.dry_run import dry_run_scope

        use_lean_local = await should_use_lean_local_scheduled_task(task, self.db)
        # Always bound: the error handler below reads it, and it is only
        # assigned inside the lean-local branch. Leaving it unbound would
        # raise NameError from the handler and bury the real failure.
        run_timeout_s: float | None = None
        # Existing dry-run evaluation tasks predate a schema-level flag.  Make
        # their long-standing, explicit naming/prompt convention an execution
        # boundary as well as prose.  Ordinary tasks remain byte-identical.
        task_name_low = str(task.get("name") or "").strip().lower()
        prompt_head = str(task.get("prompt") or "").lstrip()[:80].lower()
        task_dry_run = (
            "dryrun" in task_name_low
            or "dry-run" in task_name_low
            or prompt_head.startswith("dry run")
            or prompt_head.startswith("dry-run")
        )
        execution_policy = narrow_execution_policy(
            current_execution_policy(), task_execution_policy(task),
        )
        if execution_policy.get("timeout_seconds") is not None:
            run_timeout_s = float(execution_policy["timeout_seconds"])
        elif use_lean_local:
            run_timeout_s = max(5.0, float(os.environ.get(
                "OPENAGENT_LOCAL_SCHEDULED_TASK_TIMEOUT_SECONDS", "120",
            )))
        durable = _durable_child_sessions()
        # Per-run id (durable) so each firing is its own navigable session;
        # legacy mode reuses one per-task id wiped after every fire.
        run_id: str = str(uuid.uuid4())
        # The durable per-run id MUST equal what ``run_child_session`` mints for
        # the same origin_ref, so the ``task_runs.session_id`` link points at the
        # real child row — mint it the one way instead of re-formatting by hand.
        from src.core.child_session import mint_child_session_id
        session_id = (
            mint_child_session_id("scheduler", {"task_id": task["id"], "run_id": run_id})
            if durable else f"scheduler:{task['id']}"
        )
        allowed_families: list[str] | None = execution_policy.get(
            "allowed_tool_families"
        )
        ambient_families = current_tool_allowlist()
        if allowed_families is not None and ambient_families is not None:
            from src.core.tool_scope import normalize_family

            allowed_families = [
                item for item in allowed_families
                if normalize_family(item) in ambient_families
            ]
        if use_lean_local and allowed_families is None:
            pool = getattr(self.agent, "_mcp", None)
            available = list(getattr(pool, "_toolkit_by_name", {}) or {})
            lean_families = lean_local_tool_families(effective_prompt, available)
            allowed_families = lean_families or None
            if allowed_families is not None:
                elog(
                    "task.lean_local_tool_scope",
                    name=task_name,
                    families=",".join(allowed_families),
                    dropped=max(0, len(available) - len(allowed_families)),
                )
        scope_token = (
            set_tool_allowlist(allowed_families)
            if allowed_families is not None else None
        )
        policy_token = set_execution_policy(execution_policy)
        if execution_policy:
            elog(
                "task.execution_policy",
                name=task_name,
                max_tool_calls=execution_policy.get("max_tool_calls"),
                timeout_seconds=execution_policy.get("timeout_seconds"),
                tool_families=allowed_families,
            )
        # The two self-improvement passes run with nobody watching, so they
        # declare themselves: the skill tool then refuses, in code, any write
        # outside their lane (someone else's skill, or a pinned one). Every
        # other task keeps the foreground default — a scheduled report that
        # happens to write a skill is doing it because a human asked for that
        # task, and is not the autonomous curator.
        from src.core.builtin_tasks import (
            SKILL_CURATOR_TASK_NAME, SKILL_DISTILLER_TASK_NAME,
        )
        from src.mcp.servers.skills.provenance import (
            BACKGROUND, reset_write_origin, set_write_origin,
        )

        origin_token = None
        if task_name in (SKILL_CURATOR_TASK_NAME, SKILL_DISTILLER_TASK_NAME):
            origin_token = set_write_origin(BACKGROUND)
        elog("task.run", name=task_name)
        # Record this firing in ``task_runs`` so the dashboard can show a
        # per-task execution history (status / output preview / timing) —
        # the scheduled-task analogue of ``workflow_runs``. Best-effort:
        # logging must never stop the task from running, so the db touch
        # is guarded and skipped entirely when there's no db (e.g. a
        # Scheduler constructed with ``db=None`` in unit tests).
        recorded = False
        if self.db is not None:
            try:
                await self.db.add_task_run(
                    task_id=task["id"], trigger=trigger, run_id=run_id,
                    session_id=session_id if durable else None,
                )
                recorded = True
            except Exception as e:  # noqa: BLE001
                elog("task.run_record_failed", level="warning",
                     name=task_name, error=str(e))
        # The id to finalize the ``task_runs`` row with — None when the row was
        # never recorded (so ``_record_task_finish`` no-ops). ``run_id`` itself
        # stays the real uuid for register/unregister + request linking.
        finish_run_id = run_id if recorded else None
        # Once the run row exists, bind this firing's task so the
        # cancellation drain can hard-stop it on a "completely stop" request.
        if recorded:
            self._register_run(self._scheduled_run_tasks, run_id)
            # Surface the in-flight firing to subscribed clients so the
            # dashboard can flip the tile to "running" and offer a Stop
            # control — for *every* firing (scheduled tick, run-now from the
            # app, or run-now from the agent's MCP), not just app-initiated
            # ones. The matching "no longer running" signal is the finish
            # broadcast in ``_record_task_finish``.
            self._broadcast("scheduled_task", "updated", task["id"])
            # When this firing was kicked off by an out-of-process "run now"
            # request, link the request to this run_id so the MCP tool's
            # wait-poller can find the firing (mirrors the workflow path).
            if request_id is not None:
                try:
                    await self.db.set_task_request_run_id(request_id, run_id)
                except Exception as e:  # noqa: BLE001
                    elog("task.run_request_link_failed", level="warning",
                         name=task_name, error=str(e))
        try:
            # Pick up any providers/models the REST or MCP layer wrote
            # since the last tick. The gateway fires refresh_registries on
            # every user message; the scheduler path bypasses that, so
            # without this hook a freshly-added model stays invisible to
            # scheduler turns until the next gateway message tickles the
            # router. Probe is a single SQLite round-trip; no-op when
            # nothing changed.
            try:
                await self.agent.refresh_registries()
            except Exception as e:  # noqa: BLE001
                elog("scheduler.hot_reload_error", level="warning", error=str(e))
            # Provenance for vault commits made during this scheduled run.
            try:
                from src.memory.vault.vault_origin import note_activity
                note_activity(kind="scheduled_task", task=task["id"],
                              run=run_id, session=session_id)
            except Exception:  # noqa: BLE001
                pass
            # An operator-approved execution block runs deterministically and
            # skips the model entirely. It is checked here, inside the try, so
            # the task_runs row records the outcome exactly like any firing.
            from src.core import task_directive

            # A quality run is judgement plus bookkeeping, and only the first
            # half is the model's job. Measured against this scheduler, the
            # model-driven version skipped the recording step in two firings
            # out of three and once reported a refused write as "ok". So the
            # code fetches, computes and records; the model only supplies the
            # six sub-scores.
            if "[[quality-digest]]" in effective_prompt:
                from src.core import local_quality_scorer

                pool = getattr(self.agent, "_mcp", None)
                if pool is None:
                    raise RuntimeError("quality digest needs an MCP pool")
                product = "lyra" if "lyra" in task_name.lower() else "esound"
                result = await local_quality_scorer.digest(pool, product=product)
                elog("task.quality_digest", name=task_name,
                     systemic=len(result.get("systemic") or []))
                await self._record_task_finish(
                    finish_run_id, task, status="success",
                    output=json.dumps(result, ensure_ascii=False, default=str),
                    error=None,
                )
                return

            if "[[quality-scorer]]" in effective_prompt:
                from src.core import local_quality_scorer

                pool = getattr(self.agent, "_mcp", None)
                if pool is None:
                    raise RuntimeError("quality scorer needs an MCP pool")
                product = "lyra" if "lyra" in task_name.lower() else "esound"
                with (
                    dry_run_scope(task_dry_run),
                    lean_local_event_scope(use_lean_local),
                    lean_local_task_scope(use_lean_local),
                    strict_local_only_scope(use_lean_local),
                ):
                    result = await local_quality_scorer.run(
                        self.agent, {"model": task.get("model") or ""}, pool,
                        f"scheduler:{task['id']}", product=product,
                    )
                elog("task.quality_scored", name=task_name,
                     scored=result.get("scored"), bad=result.get("bad"))
                await self._record_task_finish(
                    finish_run_id, task, status="success",
                    output=json.dumps(result, ensure_ascii=False, default=str),
                    error=None,
                )
                return

            directives = task_directive.parse(effective_prompt)
            if directives:
                pool = getattr(self.agent, "_mcp", None)
                if pool is None:
                    raise RuntimeError("execute block needs an MCP pool")
                ok, receipts = await task_directive.execute(pool, directives)
                summary = json.dumps(
                    {"executed": len(receipts), "ok": ok, "receipts": receipts},
                    ensure_ascii=False, default=str,
                )
                elog(
                    "task.directive_executed",
                    name=task_name, count=len(receipts), ok=ok,
                )
                await self._record_task_finish(
                    finish_run_id, task,
                    status="success" if ok else "failed",
                    output=summary,
                    error=None if ok else "execute block failed",
                )
                return
            if durable:
                # A scheduled firing is the agent acting on a mission it gave
                # itself: spawn a full child session under a per-task root,
                # authored by the agent (so the app renders the prompt as a
                # Mission block), owned by the agent's primary user so the
                # row lands in their session list.
                from src.core.child_session import run_child_session
                from src.core.identity_context import agent_author
                owner = None
                if self.db is not None:
                    try:
                        owner = await self.db.primary_owner_handle()
                    except Exception:  # noqa: BLE001
                        owner = None
                with (
                    dry_run_scope(task_dry_run),
                    lean_local_event_scope(use_lean_local),
                    lean_local_task_scope(use_lean_local),
                    strict_local_only_scope(use_lean_local),
                ):
                    run = run_child_session(
                        agent=self.agent,
                        db=self.db,
                        parent_session_id=f"scheduler:{task['id']}",
                        origin="scheduler",
                        origin_ref={"task_id": task["id"], "run_id": run_id},
                        title=task_name,
                        prompt=effective_prompt,
                        owner_client_id=owner,
                        # Optional per-task model pin: run the firing on the model
                        # the task was configured with. NULL falls back to the
                        # agent's default/router pick inside run_child_session,
                        # exactly like a chat turn with no session pin.
                        model_id=task.get("model") or None,
                        author=agent_author(task_name, agent_name=getattr(self.agent, "name", None)),
                        # Stream the firing live so its run screen fills in
                        # token-by-token like any interactive session.
                        stream=True,
                    )
                    if run_timeout_s is not None:
                        result = await asyncio.wait_for(run, timeout=run_timeout_s)
                    else:
                        result = await run
                response = result.text
                child_issue = await self._child_session_terminal_issue(
                    result.session_id,
                )
            else:
                with (
                    dry_run_scope(task_dry_run),
                    lean_local_event_scope(use_lean_local),
                    lean_local_task_scope(use_lean_local),
                    strict_local_only_scope(use_lean_local),
                ):
                    run = self.agent.run(
                        message=effective_prompt,
                        user_id="scheduler",
                        session_id=session_id,
                    )
                    if run_timeout_s is not None:
                        response = await asyncio.wait_for(run, timeout=run_timeout_s)
                    else:
                        response = await run
                # The non-streaming branch used to skip the check entirely, so a
                # truncated run here looked like a clean success.
                child_issue = await self._child_session_terminal_issue(session_id)
            elog("task.done", name=task_name, preview=str(response)[:100])
            if child_issue is not None:
                final_status, final_error = child_issue
                elog(
                    "task.child_run_not_successful",
                    level="warning",
                    name=task_name,
                    status=final_status,
                    error=final_error[:200],
                )
                await self._record_task_finish(
                    finish_run_id,
                    task,
                    status=final_status,
                    output=str(response),
                    error=final_error,
                )
            else:
                await self._record_task_finish(
                    finish_run_id, task, status="success", output=str(response),
                )
        except asyncio.CancelledError:
            # A stop request (or shutdown) cancelled this firing mid-turn.
            # Finalize the task_runs row as ``cancelled`` — not ``failed`` —
            # before re-raising so the dashboard shows it was stopped, not
            # that it errored. ``CancelledError`` is a BaseException, so the
            # broad ``except Exception`` below never sees it; this branch is
            # the only place the cancelled firing gets recorded.
            elog("task.cancelled", name=task_name)
            await self._record_task_finish(
                finish_run_id, task, status="cancelled", error="Stopped by user",
            )
            raise
        except Exception as e:
            # ``str(e)`` is EMPTY for an exception raised without a message,
            # and several are. That produced `task.error … error=""` — an
            # error event that does not say what went wrong, which is the one
            # thing it exists to do. Keep the type so an argument-less
            # exception still identifies itself, in the log and on the run row
            # the dashboard reads.
            # Keep the CAUSE too. The informative half is usually the wrapped
            # original — a read timeout on the upstream call surfacing as a
            # provider error, say — and it is what tells a timeout on the
            # socket apart from a budget the scheduler itself imposed. Only
            # the lean-local path imposes one, so it is absent for most tasks
            # and must not be implied when it is.
            detail = str(e).strip()
            detail = f"{type(e).__name__}: {detail}" if detail else type(e).__name__
            cause = e.__cause__ or e.__context__
            if cause is not None and type(cause) is not type(e):
                cause_txt = str(cause).strip() or type(cause).__name__
                detail = f"{detail} (caused by {type(cause).__name__}: {cause_txt})"
            if run_timeout_s and isinstance(e, asyncio.TimeoutError):
                detail = f"{detail} — scheduler budget was {run_timeout_s:.0f}s"
            elog("task.error", level="error", name=task_name,
                 error=detail, error_type=type(e).__name__)
            await self._record_task_finish(
                finish_run_id, task, status="failed", error=detail,
            )
        finally:
            reset_execution_policy(policy_token)
            if origin_token is not None:
                reset_write_origin(origin_token)
            if scope_token is not None:
                reset_tool_allowlist(scope_token)
            if recorded:
                self._unregister_run(self._scheduled_run_tasks, run_id)
            if not durable:
                # Legacy fire-and-forget: each tick reuses one per-task
                # session, so ``forget_session`` wipes the provider-native
                # resume id between firings (issue #5 — a resumed, compacted
                # transcript would summarize to "all done" and the next
                # firing would exit without re-running the prompt). The
                # durable path avoids this structurally with per-run ids, so
                # it keeps the row (``run_child_session`` already released
                # the live runtime).
                try:
                    await self.agent.forget_session(session_id)
                except Exception as e:
                    elog("scheduler.forget_failed", task=task_name, error=str(e))

    async def _child_session_terminal_issue(
        self,
        session_id: str,
    ) -> tuple[str, str] | None:
        """Return a task_runs terminal override from child-session metadata.

        ``Agent.run`` and ``Agent.run_stream`` intentionally convert provider
        exceptions into chat-renderable text. The durable child session still
        persists the runtime run with ``status='ERROR'`` / ``'CANCELLED'``;
        scheduled-task history must honor that lower-level truth instead of
        marking the wrapper call as a success.
        """
        if self.db is None:
            return None
        try:
            runs = await self.db.list_session_runs(session_id, limit=1)
        except Exception as e:  # noqa: BLE001
            elog(
                "task.child_run_status_read_failed",
                level="warning",
                session_id=session_id,
                error=str(e) or type(e).__name__,
            )
            return None
        if not runs:
            return None

        latest = runs[0]
        status = _status_name(latest.get("status"))
        if status in _CHILD_FAILURE_STATUSES:
            return (
                "failed",
                _child_run_error_preview(
                    latest,
                    fallback=f"Child session {session_id} ended with runtime status {status}",
                ),
            )
        if status in _CHILD_CANCELLED_STATUSES:
            return (
                "cancelled",
                _child_run_error_preview(
                    latest,
                    fallback=f"Child session {session_id} ended with runtime status {status}",
                ),
            )
        if status in _CHILD_INCOMPLETE_STATUSES:
            return (
                "failed",
                f"Child session {session_id} ended without a completed runtime run: {status}",
            )
        # Prove, non solo esito. Un'esecuzione che dichiara successo senza
        # aver chiamato un solo tool non ha letto, scritto o mandato niente:
        # il resoconto e' un'affermazione, non un risultato. Non la si declassa
        # — un compito che legittimamente non usa tool esiste — ma la si DICE,
        # perche' nell'archivio e' indistinguibile da un lavoro fatto, ed e'
        # cosi' che un compito rotto resta vivo per settimane.
        try:
            from src.core.run_evidence import unevidenced_reason

            reason = unevidenced_reason(
                status="success",
                run=latest,
                output=_string_preview(latest.get("content")),
            )
            if reason:
                elog(
                    "task_run.unevidenced",
                    level="warning",
                    session_id=session_id,
                    detail=reason,
                )
        except Exception as e:  # noqa: BLE001 — un testimone non rompe cio' che osserva
            elog("task_run.evidence_check_failed", level="warning", error=str(e))

        if _run_was_truncated(latest):
            # Completed, but only because the budget stopped it. Say so: a run
            # that never reached its work is not a success, and reading it as
            # one is how this task stayed broken unnoticed.
            return (
                "failed",
                "Run truncated: the tool-call budget was exhausted before the "
                "task finished its work. Raise OPENAGENT_LEAN_EVENT_MAX_TOOL_CALLS "
                "(lean profile) or OPENAGENT_MAX_TOOL_CALLS_PER_RUN.",
            )
        return None

    async def _record_task_finish(
        self,
        run_id: str | None,
        task: dict,
        *,
        status: str,
        output: str | None = None,
        error: str | None = None,
    ) -> None:
        """Finalize the ``task_runs`` row opened in ``run_task``. No-op when
        recording was skipped (no db, or the open failed). Best-effort: a
        failure here is logged, never raised, so a logging hiccup can't
        turn a healthy task run into a reported failure."""
        if run_id is None or self.db is None:
            return
        try:
            updates: dict = {"status": status, "finished_at": time.time()}
            if output is not None:
                updates["output"] = output[:_MAX_TASK_RUN_OUTPUT]
            if error is not None:
                updates["error"] = error[:_MAX_TASK_RUN_OUTPUT]
            await self.db.update_task_run(run_id, **updates)
            # Keep the per-task history bounded.
            await self.db.prune_task_runs(task["id"])
            # Opt-in retention for the durable per-firing child sessions under
            # this task's root. Default 0 = keep all (sessions stay navigable);
            # operators set a cap if firings accumulate.
            try:
                keep = int(os.environ.get("OPENAGENT_SCHEDULER_KEEP_SESSIONS", "0"))
            except (TypeError, ValueError):
                keep = 0
            if keep > 0:
                await self.db.prune_child_sessions(f"scheduler:{task['id']}", keep=keep)
            self._broadcast("scheduled_task", "updated", task["id"])
        except Exception as e:  # noqa: BLE001
            elog("task.run_finish_failed", level="warning",
                 name=task.get("name"), error=str(e))

    async def _check_and_run(self) -> None:
        """Check for due tasks and execute them.

        Each due item is dispatched as its own ``asyncio.Task`` via
        ``_spawn_workflow`` so different workflows actually run in
        parallel. Bookkeeping (``last_run`` / ``next_run`` advances,
        broadcasts) happens *before* dispatch — otherwise a long run
        would still be in-flight when the next 30 s tick fires
        ``get_due_tasks`` again, and the same row would be re-fired
        repeatedly.
        """
        now = time.time()
        due_tasks = await self.db.get_due_tasks(now)

        for task in due_tasks:
            elog("scheduler.run_due", name=task["name"])
            try:
                if is_one_shot_expression(task["cron_expression"]):
                    await self.db.update_task(
                        task["id"],
                        last_run=now,
                        next_run=None,
                        enabled=0,
                    )
                else:
                    await self.db.update_task(
                        task["id"],
                        last_run=now,
                        next_run=self._next_run(
                            task["cron_expression"], now, self._task_tz(task),
                        ),
                    )
                self._broadcast("scheduled_task", "updated", task["id"])
            except ValueError as e:
                elog("scheduler.next_run_update_failed", level="error",
                     task=task["name"], error=str(e))
            self._spawn_workflow(self.run_task(task))

        # Per-block schedules. Each due row fires its own
        # trigger-schedule node as the entry point; a workflow with
        # multiple schedule blocks gets multiple independent firings.
        try:
            due_schedules = await self.db.get_due_schedules(now)
        except Exception as e:  # noqa: BLE001
            elog("scheduler.schedules_fetch_failed", level="warning", error=str(e))
            due_schedules = []
        for sched in due_schedules:
            wf = await self.db.get_workflow(sched["workflow_id"])
            if wf is None:
                # FK cascade should prevent this, but guard anyway.
                await self.db.delete_schedule(sched["id"])
                continue
            elog(
                "scheduler.schedule_due",
                workflow=wf.get("name"),
                node_id=sched.get("node_id"),
                schedule_id=sched.get("id"),
            )
            try:
                cron = sched.get("cron_expression") or ""
                if is_one_shot_expression(cron):
                    await self.db.update_schedule(
                        sched["id"], last_run_at=now, enabled=False,
                    )
                else:
                    await self.db.update_schedule(
                        sched["id"],
                        last_run_at=now,
                        next_run_at=self._next_run(cron, now),
                    )
            except ValueError as e:
                elog(
                    "scheduler.schedule_next_run_failed", level="error",
                    schedule=sched.get("id"), error=str(e),
                )
            # Workflow-level last_run_at surfaces "some schedule fired"
            # to the list UI even when the workflow has many schedules.
            try:
                await self.db.update_workflow(
                    sched["workflow_id"], last_run_at=now,
                )
            except Exception:  # noqa: BLE001
                pass
            self._broadcast("workflow", "updated", sched["workflow_id"])
            self._spawn_workflow(
                self._run_workflow(
                    wf,
                    trigger="schedule",
                    entry_node_id=sched["node_id"],
                )
            )

        # AI-enqueued + manually-enqueued workflow runs (Phase 2).
        try:
            requests = await self.db.claim_pending_workflow_requests(limit=5)
        except Exception as e:  # noqa: BLE001
            # 37e99bd dropped explicit BEGIN/ROLLBACK from the claim path,
            # but both signatures still surface across mixout / performa
            # boss / friday on v0.12.44 — root cause not yet pinned. Capture
            # the type and full traceback so the next run can see which
            # call site inside aiosqlite is raising the auto-begin error.
            elog(
                "scheduler.workflow_claim_failed",
                level="warning",
                error=str(e) or type(e).__name__,
                error_type=type(e).__name__,
                exc_info=True,
            )
            requests = []
        for req in requests:
            workflow_id = req.get("workflow_id")
            wf = await self.db.get_workflow(workflow_id) if workflow_id else None
            if wf is None:
                elog(
                    "scheduler.workflow_request_orphan", level="warning",
                    request_id=req.get("id"), workflow_id=workflow_id,
                )
                continue
            self._spawn_workflow(
                self._run_workflow(
                    wf,
                    trigger=req.get("trigger") or "api",
                    inputs=req.get("inputs") or {},
                    request_id=req.get("id"),
                )
            )

    # ── Workflow helpers (Phase 2) ──

    def _get_workflow_executor(self) -> WorkflowExecutor:
        # Local import keeps openagent.core.scheduler free of a hard
        # dependency on the workflow package at import time — if the
        # user never adopts workflows, the executor class is never
        # loaded.
        if self._workflow_executor is None:
            from src.workflow.executor import WorkflowExecutor

            self._workflow_executor = WorkflowExecutor(
                self.agent, self.db, broadcast=self._broadcast,
            )
        return self._workflow_executor

    async def _run_workflow(
        self,
        wf: dict,
        *,
        trigger: str,
        inputs: dict | None = None,
        request_id: str | None = None,
        entry_node_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Execute a workflow. Mirrors ``run_task``: catches exceptions,
        refreshes registries, and — when this run came from a request
        row — links the request to the new ``run_id`` so the MCP's
        ``run_workflow`` poller can find it without a race.

        ``entry_node_id`` restricts the walk's entry set to a specific
        node, used by the scheduler when a workflow has multiple
        ``trigger-schedule`` blocks and only one of them fired this tick.

        ``run_id`` may be supplied so a caller (the event dispatcher) can
        link the produced run to its own record deterministically; when
        None a fresh id is minted, matching the previous behaviour.
        """
        import uuid

        wf_name = wf.get("name")
        run_id = run_id or str(uuid.uuid4())
        elog(
            "workflow.run", name=wf_name, run_id=run_id,
            trigger=trigger, request_id=request_id,
            entry_node_id=entry_node_id,
        )
        if request_id is not None:
            # Link the request row first so a polling MCP tool can move
            # off "waiting for run_id" the moment the next DB tick lands.
            try:
                await self.db.set_workflow_request_run_id(request_id, run_id)
            except Exception as e:  # noqa: BLE001
                elog(
                    "scheduler.workflow_link_failed", level="warning",
                    request_id=request_id, error=str(e),
                )

        # Bind this run's task so the cancellation drain can hard-stop it.
        # Safe to register before the executor opens the run row: the MCP
        # only flags a row ``cancelling`` once it exists and is ``running``,
        # so no cancel can target this run_id until the executor has created
        # the row a few lines down.
        self._register_run(self._workflow_run_tasks, run_id)
        try:
            try:
                await self.agent.refresh_registries()
            except Exception as e:  # noqa: BLE001
                elog("scheduler.hot_reload_error", level="warning", error=str(e))

            # Provenance for vault commits made during this workflow run.
            try:
                from src.memory.vault.vault_origin import note_activity
                note_activity(kind="workflow", workflow=wf.get("id"), run=run_id)
            except Exception:  # noqa: BLE001
                pass
            executor = self._get_workflow_executor()
            final = await executor.run(
                wf, trigger=trigger, inputs=inputs, run_id=run_id,
                entry_node_id=entry_node_id,
            )
            elog(
                "workflow.done",
                name=wf_name,
                run_id=final.get("id"),
                status=final.get("status"),
            )
        except asyncio.CancelledError:
            # A stop request (or shutdown) cancelled this run. The executor
            # catches only ``Exception``, so ``CancelledError`` unwinds it
            # with the row left ``running`` — finalize it to ``cancelled``
            # here so the badge clears and the MCP's wait-poller stops. The
            # in-flight model call was already aborted as the await chain
            # unwound. Re-raise per asyncio convention.
            elog("workflow.cancelled", name=wf_name, run_id=run_id)
            try:
                await self.db.update_workflow_run(
                    run_id, status="cancelled", finished_at=time.time(),
                    error="Stopped by user",
                )
                self._broadcast("workflow", "updated", wf.get("id"))
            except Exception as e:  # noqa: BLE001
                elog("workflow.cancel_finalize_failed", level="error",
                     run_id=run_id, error=str(e))
            raise
        except Exception as e:  # noqa: BLE001
            elog("workflow.error", level="error", name=wf_name, error=str(e))
        finally:
            self._unregister_run(self._workflow_run_tasks, run_id)

    # ── Task management helpers ──

    async def add_task(
        self,
        name: str,
        cron_expression: str,
        prompt: str,
        model: str | None = None,
        timezone: str | None = None,
        execution_policy: dict | None = None,
    ) -> str:
        """Add a new scheduled task. ``model`` is an optional runtime_id the
        firing runs on (NULL = the agent's default/router model).

        ``timezone`` is an optional IANA name the cron is read in; NULL keeps
        the UTC behaviour every pre-existing task was written against.
        Callers that want the agent-wide default resolve it before
        calling — it is materialised into the row here, not re-resolved at
        fire time (see ``src/memory/schedule.py``)."""
        now = time.time()
        return await self.db.add_task(
            name, cron_expression, prompt,
            self._next_run(cron_expression, now, timezone), model=model,
            timezone=timezone, execution_policy=execution_policy,
        )

    async def list_tasks(self) -> list[dict]:
        return await self.db.get_tasks()

    async def remove_task(self, task_id: str) -> None:
        await self.db.delete_task(task_id)

    async def enable_task(self, task_id: str) -> None:
        await self.reschedule_task(task_id, enabled=1)

    async def reschedule_task(self, task_id: str, *, enabled: int | None = None) -> None:
        now = time.time()
        task = await self.db.get_task(task_id)
        if task:
            updates = {
                "next_run": self._next_run(
                    task["cron_expression"], now, self._task_tz(task),
                ),
            }
            if enabled is not None:
                updates["enabled"] = enabled
            await self.db.update_task(task_id, **updates)

    async def disable_task(self, task_id: str) -> None:
        await self.db.update_task(task_id, enabled=0)
