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

# Per-tick cap on how many ``cancelling`` rows a single drain processes. A
# healthy system has ~0 at any moment; a high cap only matters after a mass
# crash left stale flags. Hitting the cap is not data loss — the next tick
# (2 s later) picks up the remainder — but we log it so a flood is visible.
_CANCEL_SCAN_LIMIT = 500


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

    def _next_run(self, cron_expression: str, base: float | None = None) -> float:
        return next_run_for_expression(cron_expression, base)

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
                await self.db.update_task(task["id"], next_run=self._next_run(task["cron_expression"], now))
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
        near-immediate."""
        while True:
            try:
                await self._drain_cancellations()
            except Exception as e:  # noqa: BLE001
                elog("scheduler.cancel_loop_error", level="error", error=str(e))
            try:
                await self._drain_task_run_requests()
            except Exception as e:  # noqa: BLE001
                elog("scheduler.run_request_loop_error", level="error", error=str(e))
            await asyncio.sleep(CANCEL_CHECK_INTERVAL)

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
        the legacy reused-session + ``forget_session`` behavior."""
        task_name = task["name"]
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
                result = await run_child_session(
                    agent=self.agent,
                    db=self.db,
                    parent_session_id=f"scheduler:{task['id']}",
                    origin="scheduler",
                    origin_ref={"task_id": task["id"], "run_id": run_id},
                    title=task_name,
                    prompt=task["prompt"],
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
                response = result.text
                child_issue = await self._child_session_terminal_issue(
                    result.session_id,
                )
            else:
                response = await self.agent.run(
                    message=task["prompt"],
                    user_id="scheduler",
                    session_id=session_id,
                )
                child_issue = None
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
            elog("task.error", level="error", name=task_name, error=str(e))
            await self._record_task_finish(
                finish_run_id, task, status="failed", error=str(e),
            )
        finally:
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
                        next_run=self._next_run(task["cron_expression"], now),
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
    ) -> None:
        """Execute a workflow. Mirrors ``run_task``: catches exceptions,
        refreshes registries, and — when this run came from a request
        row — links the request to the new ``run_id`` so the MCP's
        ``run_workflow`` poller can find it without a race.

        ``entry_node_id`` restricts the walk's entry set to a specific
        node, used by the scheduler when a workflow has multiple
        ``trigger-schedule`` blocks and only one of them fired this tick.
        """
        import uuid

        wf_name = wf.get("name")
        run_id = str(uuid.uuid4())
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
        self, name: str, cron_expression: str, prompt: str, model: str | None = None,
    ) -> str:
        """Add a new scheduled task. ``model`` is an optional runtime_id the
        firing runs on (NULL = the agent's default/router model)."""
        now = time.time()
        return await self.db.add_task(
            name, cron_expression, prompt,
            self._next_run(cron_expression, now), model=model,
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
            updates = {"next_run": self._next_run(task["cron_expression"], now)}
            if enabled is not None:
                updates["enabled"] = enabled
            await self.db.update_task(task_id, **updates)

    async def disable_task(self, task_id: str) -> None:
        await self.db.update_task(task_id, enabled=0)
