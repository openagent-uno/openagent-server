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

import json
import time
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


@test("task_runs", "an interrupted run goes back in the queue")
async def t_requeue_interrupted(ctx: TestContext) -> None:
    """A restart kills the firing; the reap settles the row. The work still
    owes, so the task is re-enqueued through the same path a manual run uses."""
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskruns-requeue-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        weekly = await db.add_task("W", "0 9 * * 1", "p")

        await db.add_task_run(task_id=weekly)  # left 'running' by the restart
        await db.reap_orphan_task_runs()

        requeued = await db.requeue_interrupted_task_runs()
        assert requeued == [weekly], requeued

        pending = await db.claim_pending_task_requests(limit=10)
        assert [r["task_id"] for r in pending] == [weekly]
        assert pending[0]["trigger"] == "restart-requeue"

        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("task_runs", "a task killed twice running is not retried into a loop")
async def t_requeue_stops_after_second_kill(ctx: TestContext) -> None:
    """If the run before the reaped one was ALSO reaped, we already retried and
    got killed again — the likeliest cause is this task taking the process
    down, so it is left for a human instead of looping."""
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskruns-loop-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        task_id = await db.add_task("T", "0 9 * * 1", "p")

        first = await db.add_task_run(task_id=task_id)
        await db.reap_orphan_task_runs()
        assert await db.requeue_interrupted_task_runs() == [task_id]
        # Drain the request so it is not the reason the second pass skips.
        await db.claim_pending_task_requests(limit=10)

        second = await db.add_task_run(task_id=task_id)
        assert second != first
        await db.reap_orphan_task_runs()

        assert await db.requeue_interrupted_task_runs() == []

        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("task_runs", "a disabled task is not resurrected by the requeue")
async def t_requeue_skips_disabled(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskruns-disabled-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        task_id = await db.add_task("D", "0 9 * * 1", "p")
        await db.update_task(task_id, enabled=0)

        await db.add_task_run(task_id=task_id)
        await db.reap_orphan_task_runs()

        assert await db.requeue_interrupted_task_runs() == []

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
    name = "spy"
    model = None

    def __init__(self, *, raise_on_run: bool = False) -> None:
        self.raise_on_run = raise_on_run
        self.forget_calls: list[str] = []
        self.release_calls: list[str] = []

    async def refresh_registries(self) -> None:
        return None

    async def run(self, *, message: str, user_id: str, session_id: str,
                  model_override=None, author=None, on_status=None) -> str:
        if self.raise_on_run:
            raise RuntimeError("boom")
        return "the result"

    async def forget_session(self, session_id: str) -> None:
        self.forget_calls.append(session_id)

    async def release_session(self, session_id: str, *, model_override=None) -> None:
        self.release_calls.append(session_id)


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
        # Durable: the failing firing's per-run child session is RELEASED (row
        # kept for inspection), not forgotten. The run row links to it.
        assert agent.forget_calls == [], agent.forget_calls
        assert len(agent.release_calls) == 1, agent.release_calls
        assert agent.release_calls[0].startswith(f"scheduler:{task_id}:"), agent.release_calls
        assert runs[0]["session_id"] == agent.release_calls[0], runs[0]
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("task_runs", "run_task fails when the durable child session stores runtime ERROR")
async def t_scheduler_records_child_runtime_error(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskruns-child-error-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        task_id = await db.add_task("Provider Broken", "* * * * *", "try me")
        task = await db.get_task(task_id)

        class _RuntimeErrorAgent(_SpyAgent):
            def __init__(self) -> None:
                super().__init__()
                self.db = db

            async def run(self, *, message: str, user_id: str, session_id: str,
                          model_override=None, author=None, on_status=None) -> str:
                self.run_calls = getattr(self, "run_calls", [])
                self.run_calls.append((session_id, message))
                conn = await db._ensure_connected()
                await conn.execute(
                    "UPDATE sessions SET runs = ?, updated_at = ? WHERE session_id = ?",
                    (
                        json.dumps([
                            {
                                "run_id": "runtime-run",
                                "status": "ERROR",
                                "content": "[Errno 2] No such file or directory",
                            }
                        ]),
                        time.time(),
                        session_id,
                    ),
                )
                await conn.commit()
                return "[Errno 2] No such file or directory"

        scheduler = Scheduler(db=db, agent=_RuntimeErrorAgent())  # type: ignore[arg-type]
        await scheduler.run_task(task)

        runs = await db.list_task_runs(task_id)
        assert len(runs) == 1, runs
        run = runs[0]
        assert run["status"] == "failed", run
        assert "[Errno 2]" in (run["error"] or ""), run
        assert "[Errno 2]" in (run["output"] or ""), run
        assert run["session_id"] and run["session_id"].startswith(f"scheduler:{task_id}:")
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


# ── Run-now (on-demand firing) ────────────────────────────────────────────


@test("task_runs", "task run requests: claim is atomic + one-shot, links a run_id")
async def t_run_request_db(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskreq-db-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        task_id = await db.add_task("T", "* * * * *", "p")

        req = await db.enqueue_task_run_request(task_id=task_id, trigger="ai")
        # First claim wins the row; a second claim sees nothing left.
        claimed = await db.claim_pending_task_requests()
        assert len(claimed) == 1 and claimed[0]["id"] == req, claimed
        assert claimed[0]["trigger"] == "ai", claimed[0]
        assert await db.claim_pending_task_requests() == [], "double-claim"

        # The scheduler links the spawned run back so a waiter can find it.
        await db.set_task_request_run_id(req, "run-123")
        row = await db.get_task_run_request(req)
        assert row is not None and row["run_id"] == "run-123", row

        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("task_runs", "run request claim cascades away with its task")
async def t_run_request_cascade(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskreq-cascade-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        task_id = await db.add_task("T", "* * * * *", "p")
        req = await db.enqueue_task_run_request(task_id=task_id)

        await db.delete_task(task_id)
        # FK ON DELETE CASCADE drops the orphaned request too.
        assert await db.get_task_run_request(req) is None
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("task_runs", "flag_task_runs_cancelling flags running firings, no-ops when idle")
async def t_flag_cancelling(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskstop-flag-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        task_id = await db.add_task("T", "* * * * *", "p")

        # Nothing running → empty, no error.
        assert await db.flag_task_runs_cancelling(task_id) == []

        running = await db.add_task_run(task_id=task_id)  # left 'running'
        settled = await db.add_task_run(task_id=task_id)
        await db.update_task_run(settled, status="success", finished_at=1.0)

        flagged = await db.flag_task_runs_cancelling(task_id)
        assert flagged == [running], flagged
        assert (await db.get_task_run(running))["status"] == "cancelling"
        # A settled run is untouched.
        assert (await db.get_task_run(settled))["status"] == "success"

        # Idempotent: re-flagging finds nothing still 'running'.
        assert await db.flag_task_runs_cancelling(task_id) == []
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("task_runs", "running_task_ids reports tasks with an in-flight firing")
async def t_running_ids(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskstop-running-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        a = await db.add_task("A", "* * * * *", "p")
        b = await db.add_task("B", "* * * * *", "p")
        c = await db.add_task("C", "* * * * *", "p")

        run_a = await db.add_task_run(task_id=a)  # running
        run_b = await db.add_task_run(task_id=b)
        await db.update_task_run(run_b, status="cancelling")  # mid-stop, still in flight
        run_c = await db.add_task_run(task_id=c)
        await db.update_task_run(run_c, status="success", finished_at=1.0)  # done

        ids = await db.running_task_ids()
        assert ids == {a, b}, ids  # C finished → excluded
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("task_runs", "stop drain finalizes an orphan cancelling firing as cancelled")
async def t_stop_orphan_drain(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskstop-orphan-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        task_id = await db.add_task("T", "* * * * *", "p")
        run_id = await db.add_task_run(task_id=task_id)

        # A stop request flags the row 'cancelling'. With no live asyncio task
        # owning it (the firing already returned, or a prior-process crash),
        # the scheduler's drain finalizes it directly to 'cancelled' so the UI
        # never sits on a phantom cancelling badge.
        flagged = await db.flag_task_runs_cancelling(task_id)
        assert flagged == [run_id], flagged

        scheduler = Scheduler(db=db, agent=_SpyAgent())  # type: ignore[arg-type]
        await scheduler._drain_cancellations()

        row = await db.get_task_run(run_id)
        assert row["status"] == "cancelled", row
        assert row["finished_at"] is not None
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("task_runs", "drain fires a disabled task, links the run, leaves the schedule alone")
async def t_drain_run_request(ctx: TestContext) -> None:
    import asyncio

    from src.core.scheduler import Scheduler
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"taskreq-drain-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        # A DISABLED task: run-now must still fire it without re-enabling.
        task_id = await db.add_task("Nightly", "0 3 * * *", "do the work")
        await db.update_task(task_id, enabled=0)
        before = await db.get_task(task_id)

        req = await db.enqueue_task_run_request(task_id=task_id, trigger="manual")

        scheduler = Scheduler(db=db, agent=_SpyAgent())  # type: ignore[arg-type]
        await scheduler._drain_task_run_requests()
        # The firing is dispatched as a background task; let it finish.
        await asyncio.gather(*scheduler._workflow_tasks, return_exceptions=True)

        # The request now points at a recorded, successful firing.
        linked = await db.get_task_run_request(req)
        assert linked is not None and linked["run_id"], linked
        run = await db.get_task_run(linked["run_id"])
        assert run is not None and run["status"] == "success", run
        assert run["trigger"] == "manual", run
        assert run["output"] == "the result", run

        # The cron schedule + enabled flag are untouched by a run-now.
        after = await db.get_task(task_id)
        assert after["enabled"] == 0, "run-now must not enable the task"
        assert after["next_run"] == before["next_run"], after
        assert after["last_run"] == before["last_run"], after

        # Claim is one-shot: a second drain finds nothing to fire.
        scheduler._workflow_tasks.clear()
        await scheduler._drain_task_run_requests()
        assert not scheduler._workflow_tasks, "request fired twice"

        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass
