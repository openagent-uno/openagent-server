"""Scheduled-task execution history — ``task_runs`` persistence + the
Scheduler recording path.

A scheduled task used to leave no trace beyond a single ``last_run``
timestamp. ``task_runs`` is the scheduled-task analogue of
``workflow_runs``: the Scheduler opens a ``running`` row when a task
fires and flips it to ``success`` / ``failed`` (with an output/error
preview + ``finished_at``) when the agent turn returns. These tests pin
the DB layer and the recording path without spawning the real agent.
"""
from __future__ import annotations

import uuid

from ._framework import TestContext, test


# ── DB layer ────────────────────────────────────────────────────────────


@test("task_runs", "add/update/get round-trips a task run")
async def t_add_update_get(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskruns-crud-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()

        task_id = await db.add_task("T", "* * * * *", "do the thing")
        run_id = await db.add_task_run(task_id=task_id)

        opened = await db.get_task_run(run_id)
        assert opened is not None
        assert opened["status"] == "running"
        assert opened["task_id"] == task_id
        assert opened["trigger"] == "schedule"
        assert opened["started_at"] is not None
        assert opened["finished_at"] is None

        await db.update_task_run(
            run_id, status="success", finished_at=123.0, output="all done",
        )
        done = await db.get_task_run(run_id)
        assert done["status"] == "success"
        assert done["finished_at"] == 123.0
        assert done["output"] == "all done"

        # Unknown columns are ignored, not crashed on.
        await db.update_task_run(run_id, bogus="x")
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("task_runs", "list returns newest-first, honours limit + status filter")
async def t_list(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskruns-list-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        task_id = await db.add_task("T", "* * * * *", "p")

        ok_id = await db.add_task_run(task_id=task_id)
        await db.update_task_run(ok_id, status="success", finished_at=2.0)
        fail_id = await db.add_task_run(task_id=task_id)
        await db.update_task_run(fail_id, status="failed", error="boom")

        runs = await db.list_task_runs(task_id)
        assert len(runs) == 2
        # started_at DESC — newest first.
        starts = [r["started_at"] for r in runs]
        assert starts == sorted(starts, reverse=True), starts

        only_failed = await db.list_task_runs(task_id, status="failed")
        assert len(only_failed) == 1 and only_failed[0]["id"] == fail_id

        limited = await db.list_task_runs(task_id, limit=1)
        assert len(limited) == 1

        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("task_runs", "reap flips orphaned running rows to failed")
async def t_reap(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskruns-reap-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        task_id = await db.add_task("T", "* * * * *", "p")

        orphan = await db.add_task_run(task_id=task_id)  # left 'running'
        settled = await db.add_task_run(task_id=task_id)
        await db.update_task_run(settled, status="success", finished_at=1.0)

        reaped = await db.reap_orphan_task_runs()
        assert reaped == 1, reaped

        row = await db.get_task_run(orphan)
        assert row["status"] == "failed"
        assert row["finished_at"] is not None
        assert "reaped" in (row["error"] or "")
        # An already-settled run is untouched.
        assert (await db.get_task_run(settled))["status"] == "success"

        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("task_runs", "prune keeps only the most recent N runs")
async def t_prune(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskruns-prune-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        task_id = await db.add_task("T", "* * * * *", "p")

        for _ in range(5):
            await db.add_task_run(task_id=task_id)

        removed = await db.prune_task_runs(task_id, keep_last=2)
        assert removed == 3, removed
        assert len(await db.list_task_runs(task_id, limit=100)) == 2

        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("task_runs", "deleting a task cascades its runs away")
async def t_cascade(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskruns-cascade-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        task_id = await db.add_task("T", "* * * * *", "p")
        run_id = await db.add_task_run(task_id=task_id)

        await db.delete_task(task_id)
        # FK ON DELETE CASCADE (PRAGMA foreign_keys=ON) drops the run too.
        assert await db.get_task_run(run_id) is None

        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


# ── Scheduler recording path ──────────────────────────────────────────


class _SpyAgent:
    def __init__(self, *, raise_on_run: bool = False) -> None:
        self.raise_on_run = raise_on_run
        self.forget_calls: list[str] = []

    async def refresh_registries(self) -> None:
        return None

    async def run(self, *, message: str, user_id: str, session_id: str) -> str:
        if self.raise_on_run:
            raise RuntimeError("boom")
        return "the result"

    async def forget_session(self, session_id: str) -> None:
        self.forget_calls.append(session_id)


@test("task_runs", "run_task records a success run with the output preview")
async def t_scheduler_records_success(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskruns-sched-ok-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        task_id = await db.add_task("Daily", "* * * * *", "do the work")
        task = await db.get_task(task_id)

        scheduler = Scheduler(db=db, agent=_SpyAgent())  # type: ignore[arg-type]
        await scheduler.run_task(task)

        runs = await db.list_task_runs(task_id)
        assert len(runs) == 1, runs
        run = runs[0]
        assert run["status"] == "success", run
        assert run["output"] == "the result"
        assert run["finished_at"] is not None
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("task_runs", "run_task records a failed run with the error")
async def t_scheduler_records_failure(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskruns-sched-err-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        task_id = await db.add_task("Flaky", "* * * * *", "try me")
        task = await db.get_task(task_id)

        agent = _SpyAgent(raise_on_run=True)
        scheduler = Scheduler(db=db, agent=agent)  # type: ignore[arg-type]
        # run_task swallows the agent error; the run row records it.
        await scheduler.run_task(task)

        runs = await db.list_task_runs(task_id)
        assert len(runs) == 1, runs
        assert runs[0]["status"] == "failed", runs[0]
        assert "boom" in (runs[0]["error"] or "")
        # The session is still forgotten on the failure path.
        assert agent.forget_calls == [f"scheduler:{task_id}"], agent.forget_calls
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("task_runs", "run_task is a no-op recorder when constructed without a db")
async def t_scheduler_no_db(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler

    # db=None is how the headless / unit-test scheduler is built; the
    # recording must simply skip rather than AttributeError.
    scheduler = Scheduler(db=None, agent=_SpyAgent())  # type: ignore[arg-type]
    await scheduler.run_task({"id": "x", "name": "X", "prompt": "p"})
    # No assertion beyond "did not raise" — the recording path is guarded.
