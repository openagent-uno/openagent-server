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
from typing import TYPE_CHECKING

import time

from openagent.memory.schedule import (
    is_one_shot_expression,
    next_run_for_expression,
)

if TYPE_CHECKING:
    from openagent.core.agent import Agent
    from openagent.memory.db import MemoryDB
    from openagent.workflow.executor import WorkflowExecutor

from openagent.core.logging import elog


CHECK_INTERVAL = 30  # seconds between checking for due tasks


class Scheduler:
    """Background scheduler that runs agent prompts on cron schedules.

    Tasks are stored in SQLite — they survive process restarts and reboots.
    On startup, recalculates next_run for all tasks to handle missed runs.
    """

    def __init__(self, db: MemoryDB, agent: Agent):
        self.db = db
        self.agent = agent
        self._task: asyncio.Task | None = None
        # Lazy — created on first workflow tick.
        self._workflow_executor: WorkflowExecutor | None = None

    def _next_run(self, cron_expression: str, base: float | None = None) -> float:
        return next_run_for_expression(cron_expression, base)

    async def start(self) -> None:
        """Start the scheduler background loop."""
        if self._task and not self._task.done():
            return
        await self.db.connect()
        await self._recalculate_next_runs()
        self._task = asyncio.create_task(self._loop())
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
        elog("scheduler.stop")

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

        # Workflows with a schedule trigger — same missed-run handling.
        try:
            workflows = await self.db.list_scheduled_workflows()
        except Exception as e:  # noqa: BLE001 — missing helper or DB blip
            elog("scheduler.workflow_recalc_skipped", level="warning", error=str(e))
            workflows = []
        for wf in workflows:
            cron = wf.get("cron_expression")
            if not cron:
                continue
            try:
                if is_one_shot_expression(cron):
                    if wf.get("last_run_at"):
                        await self.db.update_workflow(
                            wf["id"], enabled=False, next_run_at=None,
                        )
                    continue
                await self.db.update_workflow(
                    wf["id"], next_run_at=self._next_run(cron, now),
                )
            except ValueError as e:
                elog(
                    "scheduler.invalid_cron", level="error",
                    workflow=wf.get("name"), error=str(e),
                )

    async def _loop(self) -> None:
        """Main loop: check for due tasks every CHECK_INTERVAL seconds."""
        while True:
            try:
                await self._check_and_run()
            except Exception as e:
                elog("scheduler.loop_error", level="error", error=str(e))
            await asyncio.sleep(CHECK_INTERVAL)

    async def run_task(self, task: dict) -> None:
        """Execute a single task. Extension point: override or monkey-patch
        this to intercept specific tasks (e.g. auto-update, which uses a
        direct pip subprocess instead of going through the agent)."""
        task_name = task["name"]
        session_id = f"scheduler:{task['id']}"
        elog("task.run", name=task_name)
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
            response = await self.agent.run(
                message=task["prompt"],
                user_id="scheduler",
                session_id=session_id,
            )
            elog("task.done", name=task_name, preview=str(response)[:100])
        except Exception as e:
            elog("task.error", level="error", name=task_name, error=str(e))
        finally:
            try:
                await self.agent.release_session(session_id)
            except Exception as e:
                elog("scheduler.release_failed", task=task_name, error=str(e))

    async def _check_and_run(self) -> None:
        """Check for due tasks and execute them."""
        now = time.time()
        due_tasks = await self.db.get_due_tasks(now)

        for task in due_tasks:
            elog("scheduler.run_due", name=task["name"])
            await self.run_task(task)

            # Update last_run and compute next_run
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
            except ValueError as e:
                elog("scheduler.next_run_update_failed", level="error",
                     task=task["name"], error=str(e))

        # Scheduled workflows (Phase 2).
        try:
            due_workflows = await self.db.get_due_workflow_tasks(now)
        except Exception as e:  # noqa: BLE001
            elog("scheduler.workflow_fetch_failed", level="warning", error=str(e))
            due_workflows = []
        for wf in due_workflows:
            elog("scheduler.workflow_due", name=wf.get("name"), id=wf.get("id"))
            await self._run_workflow(wf, trigger="schedule")
            try:
                cron = wf.get("cron_expression")
                if cron and is_one_shot_expression(cron):
                    await self.db.update_workflow(
                        wf["id"], last_run_at=now, next_run_at=None,
                        enabled=False,
                    )
                elif cron:
                    await self.db.update_workflow(
                        wf["id"], last_run_at=now,
                        next_run_at=self._next_run(cron, now),
                    )
            except ValueError as e:
                elog(
                    "scheduler.workflow_next_run_failed", level="error",
                    workflow=wf.get("name"), error=str(e),
                )

        # AI-enqueued + manually-enqueued workflow runs (Phase 2).
        try:
            requests = await self.db.claim_pending_workflow_requests(limit=5)
        except Exception as e:  # noqa: BLE001
            elog("scheduler.workflow_claim_failed", level="warning", error=str(e))
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
            await self._run_workflow(
                wf,
                trigger=req.get("trigger") or "api",
                inputs=req.get("inputs") or {},
                request_id=req.get("id"),
            )

    # ── Workflow helpers (Phase 2) ──

    def _get_workflow_executor(self) -> WorkflowExecutor:
        # Local import keeps openagent.core.scheduler free of a hard
        # dependency on the workflow package at import time — if the
        # user never adopts workflows, the executor class is never
        # loaded.
        if self._workflow_executor is None:
            from openagent.workflow.executor import WorkflowExecutor

            self._workflow_executor = WorkflowExecutor(self.agent, self.db)
        return self._workflow_executor

    async def _run_workflow(
        self,
        wf: dict,
        *,
        trigger: str,
        inputs: dict | None = None,
        request_id: str | None = None,
    ) -> None:
        """Execute a workflow. Mirrors ``run_task``: catches exceptions,
        refreshes registries, and — when this run came from a request
        row — links the request to the new ``run_id`` so the MCP's
        ``run_workflow`` poller can find it without a race."""
        import uuid

        wf_name = wf.get("name")
        run_id = str(uuid.uuid4())
        elog(
            "workflow.run", name=wf_name, run_id=run_id,
            trigger=trigger, request_id=request_id,
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

        try:
            try:
                await self.agent.refresh_registries()
            except Exception as e:  # noqa: BLE001
                elog("scheduler.hot_reload_error", level="warning", error=str(e))

            executor = self._get_workflow_executor()
            final = await executor.run(
                wf, trigger=trigger, inputs=inputs, run_id=run_id,
            )
            elog(
                "workflow.done",
                name=wf_name,
                run_id=final.get("id"),
                status=final.get("status"),
            )
        except Exception as e:  # noqa: BLE001
            elog("workflow.error", level="error", name=wf_name, error=str(e))

    # ── Task management helpers ──

    async def add_task(self, name: str, cron_expression: str, prompt: str) -> str:
        """Add a new scheduled task."""
        now = time.time()
        return await self.db.add_task(name, cron_expression, prompt, self._next_run(cron_expression, now))

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
