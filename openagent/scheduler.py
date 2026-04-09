"""Cron-based task scheduler. Tasks are stored in SQLite and survive reboots."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from croniter import croniter

if TYPE_CHECKING:
    from openagent.agent import Agent
    from openagent.memory.db import MemoryDB

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

    async def start(self) -> None:
        """Start the scheduler background loop."""
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
                cron = croniter(task["cron_expression"], now)
                next_run = cron.get_next(float)
                await self.db.update_task(task["id"], next_run=next_run)
            except (ValueError, KeyError) as e:
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
        try:
            response = await self.agent.run(
                message=task["prompt"],
                user_id="scheduler",
                session_id=f"scheduler:{task['id']}",
            )
            logger.info(f"Task '{task_name}' completed: {response[:100]}...")
        except Exception as e:
            logger.error(f"Task '{task_name}' failed: {e}")

    async def _check_and_run(self) -> None:
        """Check for due tasks and execute them."""
        now = time.time()
        due_tasks = await self.db.get_due_tasks(now)

        for task in due_tasks:
            logger.info(f"Running scheduled task: {task['name']}")
            await self.run_task(task)

            # Update last_run and compute next_run
            try:
                cron = croniter(task["cron_expression"], now)
                next_run = cron.get_next(float)
                await self.db.update_task(
                    task["id"],
                    last_run=now,
                    next_run=next_run,
                )
            except (ValueError, KeyError) as e:
                logger.error(f"Failed to update next_run for '{task['name']}': {e}")

    # ── Task management helpers ──

    async def add_task(self, name: str, cron_expression: str, prompt: str) -> str:
        """Add a new scheduled task."""
        # Validate cron expression
        try:
            croniter(cron_expression)
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid cron expression: {e}")

        now = time.time()
        cron = croniter(cron_expression, now)
        next_run = cron.get_next(float)
        return await self.db.add_task(name, cron_expression, prompt, next_run)

    async def list_tasks(self) -> list[dict]:
        return await self.db.get_tasks()

    async def remove_task(self, task_id: str) -> None:
        await self.db.delete_task(task_id)

    async def enable_task(self, task_id: str) -> None:
        now = time.time()
        task = await self.db.get_task(task_id)
        if task:
            cron = croniter(task["cron_expression"], now)
            next_run = cron.get_next(float)
            await self.db.update_task(task_id, enabled=1, next_run=next_run)

    async def disable_task(self, task_id: str) -> None:
        await self.db.update_task(task_id, enabled=0)
