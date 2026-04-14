"""Scheduler / cron persistence roundtrip at the MemoryDB layer."""
from __future__ import annotations

import time
import uuid

from ._framework import TestContext, test


@test("cron", "MemoryDB.add_task + get_due_tasks")
async def t_cron_dbroundtrip(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB
    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        tid = await db.add_task(
            name=f"test-task-{uuid.uuid4().hex[:6]}",
            cron_expression="0 9 * * *",
            prompt="say hello",
            next_run=time.time() + 3600,
        )
        tasks = await db.get_tasks()
        assert any(t["id"] == tid for t in tasks), "task not found after add_task"
        await db.delete_task(tid)
        tasks_after = await db.get_tasks()
        assert all(t["id"] != tid for t in tasks_after)
    finally:
        await db.close()
