"""Cron-based task scheduler. Tasks are stored in SQLite and survive reboots."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import time

from openagent.memory.schedule import (
    is_one_shot_expression,
    next_run_for_expression,
)

if TYPE_CHECKING:
    from openagent.core.agent import Agent
    from openagent.memory.db import MemoryDB

from openagent.core.logging import elog

logger = logging.getLogger(__name__)

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

    def _next_run(self, cron_expression: str, base: float | None = None) -> float:
        return next_run_for_expression(cron_expression, base)

    async def start(self) -> None:
        """Start the scheduler background loop."""
        if self._task and not self._task.done():
            return
        await self.db.connect()
        await self._recalculate_next_runs()
        self._task = asyncio.create_task(self._loop())
        logger.info("Scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Scheduler stopped")

    async def _recalculate_next_runs(self) -> None:
        """On startup, recalculate next_run for all enabled tasks."""
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
                logger.error(f"Invalid cron for task '{task['name']}': {e}")

    async def _loop(self) -> None:
        """Main loop: check for due tasks every CHECK_INTERVAL seconds."""
        while True:
            try:
                await self._check_and_run()
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
            await asyncio.sleep(CHECK_INTERVAL)

    async def run_task(self, task: dict) -> None:
        """Execute a single task. Extension point: override or monkey-patch
        this to intercept specific tasks (e.g. auto-update, which uses a
        direct pip subprocess instead of going through the agent)."""
        task_name = task["name"]
        session_id = f"scheduler:{task['id']}"
        elog("task.run", name=task_name)
        try:
            response = await self.agent.run(
                message=task["prompt"],
                user_id="scheduler",
                session_id=session_id,
            )
            logger.info(f"Task '{task_name}' completed: {response[:100]}...")
            elog("task.done", name=task_name)
        except Exception as e:
            logger.error(f"Task '{task_name}' failed: {e}")
            elog("task.error", name=task_name, error=str(e))
        finally:
            try:
                await self.agent.release_session(session_id)
            except Exception as e:
                logger.debug("Task '%s' session release failed: %s", task_name, e)

    async def _check_and_run(self) -> None:
        """Check for due tasks and execute them."""
        now = time.time()
        due_tasks = await self.db.get_due_tasks(now)

        for task in due_tasks:
            logger.info(f"Running scheduled task: {task['name']}")
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
                logger.error(f"Failed to update next_run for '{task['name']}': {e}")

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
