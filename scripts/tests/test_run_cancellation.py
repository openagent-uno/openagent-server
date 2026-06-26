"""Cross-process "completely stop a running run" — workflows + scheduled tasks.

A stop request can't reach the in-process executor / agent from the
workflow-manager / scheduler MCP subprocess, so it crosses the boundary
as a row flagged ``status='cancelling'`` (same DB-backed hand-off pattern
as ``run_workflow``). The main-process ``Scheduler`` drains those flags on
a fast loop, hard-cancels the ``asyncio.Task`` driving the run, and the
run's own cancellation handler finalizes the row to ``cancelled``.

These pin every link in that chain:
  - the DB ``get_*_runs_by_status`` scans the drain relies on
  - the workflow run hard-stop: flag → drain → executor cancelled →
    row finalized ``cancelled`` (not stranded ``running``, not ``failed``)
  - the scheduled-task firing hard-stop: same, via ``run_task``
  - the orphan sweep: a ``cancelling`` row with no live task is finalized
    directly so the badge never sticks
  - the MCP tools ``stop_workflow`` / ``stop_scheduled_task`` write the
    flag, report the targeted runs, and no-op cleanly when nothing runs
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid

from ._framework import TestContext, test


# ── fakes ────────────────────────────────────────────────────────────


class _FakeAgent:
    """Minimal agent: ``run`` blocks forever (so the drain has something
    live to cancel), the rest are no-ops the scheduler calls around it."""

    name = "fake"
    model = None

    def __init__(self) -> None:
        self.forgotten: list[str] = []
        self.released: list[str] = []

    async def refresh_registries(self) -> None:
        return None

    async def run(self, message=None, user_id=None, session_id=None, **kw):
        await asyncio.sleep(30)
        return "should have been cancelled"

    async def forget_session(self, session_id: str) -> None:
        self.forgotten.append(session_id)

    async def release_session(self, session_id: str, *, model_override=None) -> None:
        self.released.append(session_id)


class _SleepingExecutor:
    """Stand-in for ``WorkflowExecutor``: opens the run row exactly like the
    real one (so the cancel path has a ``running`` row to finalize), then
    blocks. Mirrors the real executor's contract of NOT catching
    ``CancelledError`` — the scheduler's handler owns finalization."""

    def __init__(self, db) -> None:
        self.db = db

    async def run(
        self, workflow, *, trigger="manual", inputs=None, run_id=None,
        entry_node_id=None, on_status=None,
    ):
        rid = await self.db.add_workflow_run(
            workflow_id=workflow["id"], trigger=trigger,
            inputs=inputs or {}, run_id=run_id,
        )
        await asyncio.sleep(30)  # cancelled before this returns
        await self.db.update_workflow_run(
            rid, status="success", finished_at=time.time(),
        )
        return await self.db.get_workflow_run(rid)


# ── helpers ──────────────────────────────────────────────────────────


async def _poll(fn, *, timeout: float = 5.0, interval: float = 0.02):
    """Await ``fn`` repeatedly until it returns truthy or time runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        val = await fn()
        if val:
            return val
        await asyncio.sleep(interval)
    return await fn()


async def _wait_done(task: asyncio.Task, *, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if task.done():
            return True
        await asyncio.sleep(0.02)
    return task.done()


async def _reset_mcp_conn(server) -> None:
    """Close + clear a FastMCP server module's cached ``_conn``, resilient to a
    ``close()`` that raises — ``_conn`` is reset to None FIRST so a later test
    can never inherit a stale/closed connection (test-isolation hardening)."""
    conn = getattr(server, "_conn", None)
    server._conn = None
    if conn is not None:
        try:
            await conn.close()
        except Exception:
            pass


def _restore_env(prev: str | None) -> None:
    if prev is None:
        os.environ.pop("OPENAGENT_DB_PATH", None)
    else:
        os.environ["OPENAGENT_DB_PATH"] = prev


# ── DB-level scans the drain depends on ──────────────────────────────


@test("run_cancellation", "get_*_runs_by_status returns only matching rows")
async def t_status_scans(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"cancel-scan-{uuid.uuid4().hex[:8]}.db")
    db: MemoryDB | None = None
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()

        wf_id = await db.add_workflow(name=f"scan-{uuid.uuid4().hex[:6]}")
        r_run = await db.add_workflow_run(workflow_id=wf_id, trigger="manual")
        r_cancelling = await db.add_workflow_run(workflow_id=wf_id, trigger="manual")
        await db.update_workflow_run(r_cancelling, status="cancelling")
        r_done = await db.add_workflow_run(workflow_id=wf_id, trigger="manual")
        await db.update_workflow_run(r_done, status="success", finished_at=time.time())

        cancelling = await db.get_workflow_runs_by_status("cancelling")
        assert {r["id"] for r in cancelling} == {r_cancelling}, \
            f"expected only the cancelling row, got {[r['id'] for r in cancelling]}"
        running = await db.get_workflow_runs_by_status("running")
        assert {r["id"] for r in running} == {r_run}

        task_id = await db.add_task(f"scan-task-{uuid.uuid4().hex[:6]}", "* * * * *", "do")
        tr_run = await db.add_task_run(task_id=task_id, trigger="schedule")
        tr_cancelling = await db.add_task_run(task_id=task_id, trigger="schedule")
        await db.update_task_run(tr_cancelling, status="cancelling")

        tcancelling = await db.get_task_runs_by_status("cancelling")
        assert {r["id"] for r in tcancelling} == {tr_cancelling}
        trunning = await db.get_task_runs_by_status("running")
        assert {r["id"] for r in trunning} == {tr_run}
    finally:
        if db is not None:
            try:
                await db.close()
            except Exception:
                pass
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


# ── workflow run: flag → drain → cancelled ───────────────────────────


@test("run_cancellation", "flagged workflow run is hard-stopped and finalized cancelled")
async def t_workflow_hard_stop(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.core.scheduler import Scheduler

    tmp_db = ctx.db_path.with_name(f"cancel-wf-{uuid.uuid4().hex[:8]}.db")
    db: MemoryDB | None = None
    run_task: asyncio.Task | None = None
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        sched = Scheduler(db, _FakeAgent())
        sched._workflow_executor = _SleepingExecutor(db)

        wf_id = await db.add_workflow(name=f"stop-{uuid.uuid4().hex[:6]}")
        wf = await db.get_workflow(wf_id)
        run_task = sched._spawn_workflow(sched._run_workflow(wf, trigger="manual"))

        # Wait until the run is registered AND its row is ``running``.
        async def _ready():
            if not sched._workflow_run_tasks:
                return None
            rid = next(iter(sched._workflow_run_tasks))
            row = await db.get_workflow_run(rid)
            return rid if (row and row["status"] == "running") else None

        run_id = await _poll(_ready)
        assert run_id, "workflow run never reached running"

        # Simulate the MCP flag, then drain.
        await db.update_workflow_run(run_id, status="cancelling")
        await sched._drain_cancellations()

        assert await _wait_done(run_task), "cancelled run did not unwind"
        assert run_task.cancelled(), "run task should have ended cancelled"
        row = await db.get_workflow_run(run_id)
        assert row["status"] == "cancelled", f"expected cancelled, got {row['status']!r}"
        assert row.get("finished_at"), "cancelled run missing finished_at"
        # Registry is cleaned up by the run's ``finally``.
        assert run_id not in sched._workflow_run_tasks
    finally:
        if run_task is not None and not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except (asyncio.CancelledError, Exception):
                pass
        if db is not None:
            try:
                await db.close()
            except Exception:
                pass
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


# ── scheduled task: flag → drain → cancelled ─────────────────────────


@test("run_cancellation", "flagged scheduled-task firing is hard-stopped and finalized cancelled")
async def t_scheduled_task_hard_stop(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.core.scheduler import Scheduler

    tmp_db = ctx.db_path.with_name(f"cancel-task-{uuid.uuid4().hex[:8]}.db")
    db: MemoryDB | None = None
    fire: asyncio.Task | None = None
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        agent = _FakeAgent()
        sched = Scheduler(db, agent)

        task_id = await sched.add_task(f"firing-{uuid.uuid4().hex[:6]}", "* * * * *", "do it")
        task = await db.get_task(task_id)
        fire = sched._spawn_workflow(sched.run_task(task))

        async def _ready():
            if not sched._scheduled_run_tasks:
                return None
            rid = next(iter(sched._scheduled_run_tasks))
            row = await db.get_task_run(rid)
            return rid if (row and row["status"] == "running") else None

        run_id = await _poll(_ready)
        assert run_id, "task firing never reached running"

        await db.update_task_run(run_id, status="cancelling")
        await sched._drain_cancellations()

        assert await _wait_done(fire), "cancelled firing did not unwind"
        assert fire.cancelled(), "firing task should have ended cancelled"
        row = await db.get_task_run(run_id)
        assert row["status"] == "cancelled", f"expected cancelled, got {row['status']!r}"
        assert row.get("finished_at"), "cancelled firing missing finished_at"
        # Durable: a cancelled firing's per-run child session is RELEASED on
        # the way out (live runtime freed, row kept for inspection), not
        # wiped — the per-run id is unique so there's nothing to inherit.
        assert agent.released, "cancelled firing should release its child session"
        assert not agent.forgotten, "durable cancelled firing must not forget its session"
        assert run_id not in sched._scheduled_run_tasks
    finally:
        if fire is not None and not fire.done():
            fire.cancel()
            try:
                await fire
            except (asyncio.CancelledError, Exception):
                pass
        if db is not None:
            try:
                await db.close()
            except Exception:
                pass
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


# ── orphan sweep: cancelling row with no live task ───────────────────


@test("run_cancellation", "orphan cancelling rows are finalized directly")
async def t_orphan_sweep(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.core.scheduler import Scheduler

    tmp_db = ctx.db_path.with_name(f"cancel-orphan-{uuid.uuid4().hex[:8]}.db")
    db: MemoryDB | None = None
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        # Record broadcasts so we can assert the UI-sync hook fires on finalize.
        events: list[tuple] = []
        sched = Scheduler(
            db, _FakeAgent(),
            broadcast=lambda resource, action, id=None: events.append((resource, action, id)),
        )

        wf_id = await db.add_workflow(name=f"orphan-{uuid.uuid4().hex[:6]}")
        wf_run = await db.add_workflow_run(workflow_id=wf_id, trigger="schedule")
        await db.update_workflow_run(wf_run, status="cancelling")

        task_id = await db.add_task(f"orphan-task-{uuid.uuid4().hex[:6]}", "* * * * *", "do")
        task_run = await db.add_task_run(task_id=task_id, trigger="schedule")
        await db.update_task_run(task_run, status="cancelling")

        await sched._drain_cancellations()

        wf_row = await db.get_workflow_run(wf_run)
        assert wf_row["status"] == "cancelled", f"orphan wf run: {wf_row['status']!r}"
        assert wf_row.get("finished_at")
        t_row = await db.get_task_run(task_run)
        assert t_row["status"] == "cancelled", f"orphan task run: {t_row['status']!r}"
        assert t_row.get("finished_at")
        # The finalize broadcasts an 'updated' for each resource so the desktop
        # app refreshes without going through the REST handlers.
        assert ("workflow", "updated", wf_id) in events, f"missing wf broadcast: {events}"
        assert ("scheduled_task", "updated", task_id) in events, \
            f"missing task broadcast: {events}"
    finally:
        if db is not None:
            try:
                await db.close()
            except Exception:
                pass
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


# ── MCP tools: stop_workflow / stop_scheduled_task ───────────────────


@test("run_cancellation", "workflow-manager stop_workflow flags running runs, no-ops when idle")
async def t_mcp_stop_workflow(ctx: TestContext) -> None:
    import src.mcp.servers.workflow_manager.server as wf_server

    tmp_db = ctx.db_path.with_name(f"cancel-mcpwf-{uuid.uuid4().hex[:8]}.db")
    prev_env = os.environ.get("OPENAGENT_DB_PATH")
    try:
        os.environ["OPENAGENT_DB_PATH"] = str(tmp_db)
        await _reset_mcp_conn(wf_server)

        created = await wf_server.create_workflow(name=f"mcpstop-{uuid.uuid4().hex[:6]}")
        wf_id = created["id"]
        conn = await wf_server._get_conn()
        run_id = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO workflow_runs (id, workflow_id, trigger, status, started_at) "
            "VALUES (?, ?, 'manual', 'running', ?)",
            (run_id, wf_id, time.time()),
        )
        await conn.commit()

        res = await wf_server.stop_workflow(wf_id, wait=False)
        assert res["count"] == 1, f"expected 1 flagged, got {res['count']}"
        assert res["stopped"] == [run_id], f"wrong run reported: {res['stopped']}"

        got = await wf_server.get_workflow_run(run_id)
        assert got["status"] == "cancelling", f"row not flagged: {got['status']!r}"

        # Nothing is ``running`` anymore → clean no-op.
        idle = await wf_server.stop_workflow(wf_id, wait=False)
        assert idle["count"] == 0 and idle["stopped"] == []
        assert "no running run" in idle["note"].lower()
    finally:
        await _reset_mcp_conn(wf_server)
        _restore_env(prev_env)
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("run_cancellation", "scheduler stop_scheduled_task flags running firings, no-ops when idle")
async def t_mcp_stop_scheduled_task(ctx: TestContext) -> None:
    import src.mcp.servers.scheduler.server as sched_server

    tmp_db = ctx.db_path.with_name(f"cancel-mcptask-{uuid.uuid4().hex[:8]}.db")
    prev_env = os.environ.get("OPENAGENT_DB_PATH")
    try:
        os.environ["OPENAGENT_DB_PATH"] = str(tmp_db)
        await _reset_mcp_conn(sched_server)

        created = await sched_server.create_scheduled_task(
            name=f"mcptask-{uuid.uuid4().hex[:6]}",
            cron_expression="* * * * *",
            prompt="do it",
        )
        task_id = created["id"]
        conn = await sched_server._get_conn()
        run_id = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO task_runs (id, task_id, trigger, status, started_at) "
            "VALUES (?, ?, 'schedule', 'running', ?)",
            (run_id, task_id, time.time()),
        )
        await conn.commit()

        res = await sched_server.stop_scheduled_task(task_id, wait=False)
        assert res["count"] == 1, f"expected 1 flagged, got {res['count']}"
        assert res["stopped"] == [run_id], f"wrong run reported: {res['stopped']}"

        cur = await conn.execute("SELECT status FROM task_runs WHERE id = ?", (run_id,))
        row = await cur.fetchone()
        assert row[0] == "cancelling", f"firing not flagged: {row[0]!r}"

        idle = await sched_server.stop_scheduled_task(task_id, wait=False)
        assert idle["count"] == 0 and idle["stopped"] == []
        assert "no running firing" in idle["note"].lower()
    finally:
        await _reset_mcp_conn(sched_server)
        _restore_env(prev_env)
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


# ── the race the whole design hinges on: 'cancelling' is sticky ──────


@test("run_cancellation", "a cancelling run only moves to cancelled, never success/failed")
async def t_cancelling_is_sticky(ctx: TestContext) -> None:
    """The core correctness guarantee. A run flagged 'cancelling' must not be
    overwritten by a natural finalize that lands in the stop window — otherwise
    a run that completed a hair too late would escape to 'success'/'failed' and
    the user's stop would be silently lost. Enforced in update_workflow_run /
    update_task_run. Tested directly so it can't regress unnoticed."""
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"cancel-sticky-{uuid.uuid4().hex[:8]}.db")
    db: MemoryDB | None = None
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        wf_id = await db.add_workflow(name=f"sticky-{uuid.uuid4().hex[:6]}")

        # success must NOT overwrite a cancelling row...
        rid = await db.add_workflow_run(workflow_id=wf_id, trigger="manual")
        await db.update_workflow_run(rid, status="cancelling")
        await db.update_workflow_run(
            rid, status="success", finished_at=time.time(), outputs={"x": 1},
        )
        assert (await db.get_workflow_run(rid))["status"] == "cancelling", \
            "success leaked past the cancelling flag"
        # ...but the authoritative cancel finalize lands.
        await db.update_workflow_run(rid, status="cancelled", finished_at=time.time())
        assert (await db.get_workflow_run(rid))["status"] == "cancelled"

        # failed must NOT overwrite a cancelling row either.
        rid2 = await db.add_workflow_run(workflow_id=wf_id, trigger="manual")
        await db.update_workflow_run(rid2, status="cancelling")
        await db.update_workflow_run(rid2, status="failed", error="boom", finished_at=time.time())
        assert (await db.get_workflow_run(rid2))["status"] == "cancelling"

        # A normal (un-flagged) run still finalizes as usual — no collateral damage.
        rid3 = await db.add_workflow_run(workflow_id=wf_id, trigger="manual")
        await db.update_workflow_run(rid3, status="success", finished_at=time.time())
        assert (await db.get_workflow_run(rid3))["status"] == "success"

        # task_runs carry the same invariant.
        task_id = await db.add_task(f"sticky-task-{uuid.uuid4().hex[:6]}", "* * * * *", "p")
        trid = await db.add_task_run(task_id=task_id, trigger="schedule")
        await db.update_task_run(trid, status="cancelling")
        await db.update_task_run(trid, status="success", output="done", finished_at=time.time())
        assert (await db.get_task_run(trid))["status"] == "cancelling"
        await db.update_task_run(trid, status="cancelled", finished_at=time.time())
        assert (await db.get_task_run(trid))["status"] == "cancelled"
    finally:
        if db is not None:
            try:
                await db.close()
            except Exception:
                pass
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("run_cancellation", "startup reap finalizes crash-stranded cancelling rows as cancelled")
async def t_reap_cancelling(ctx: TestContext) -> None:
    """A crash between the MCP flag and the drain leaves a 'cancelling' row.
    reap_orphan_* (run at serve-start) must finalize it 'cancelled' — not
    'failed', and not leave it stuck — alongside the usual 'running'->'failed'."""
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"cancel-reap-{uuid.uuid4().hex[:8]}.db")
    db: MemoryDB | None = None
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        wf_id = await db.add_workflow(name=f"reapc-{uuid.uuid4().hex[:6]}")
        running = await db.add_workflow_run(workflow_id=wf_id, trigger="schedule")
        cancelling = await db.add_workflow_run(workflow_id=wf_id, trigger="manual")
        await db.update_workflow_run(cancelling, status="cancelling")

        reaped = await db.reap_orphan_workflow_runs()
        assert reaped == 2, f"expected running->failed + cancelling->cancelled, got {reaped}"
        assert (await db.get_workflow_run(running))["status"] == "failed"
        crow = await db.get_workflow_run(cancelling)
        assert crow["status"] == "cancelled", f"stranded cancel not finalized: {crow['status']!r}"
        assert "stop left pending" in (crow.get("error") or "").lower()

        # Idempotent second pass.
        assert await db.reap_orphan_workflow_runs() == 0

        # Same for task_runs.
        task_id = await db.add_task(f"reapc-task-{uuid.uuid4().hex[:6]}", "* * * * *", "p")
        trun = await db.add_task_run(task_id=task_id, trigger="schedule")
        await db.update_task_run(trun, status="cancelling")
        assert await db.reap_orphan_task_runs() == 1
        assert (await db.get_task_run(trun))["status"] == "cancelled"
    finally:
        if db is not None:
            try:
                await db.close()
            except Exception:
                pass
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("run_cancellation", "stop_workflow targets a specific run_id and resolves by name")
async def t_mcp_stop_workflow_targeting(ctx: TestContext) -> None:
    import src.mcp.servers.workflow_manager.server as wf_server

    tmp_db = ctx.db_path.with_name(f"cancel-target-{uuid.uuid4().hex[:8]}.db")
    prev_env = os.environ.get("OPENAGENT_DB_PATH")
    try:
        os.environ["OPENAGENT_DB_PATH"] = str(tmp_db)
        await _reset_mcp_conn(wf_server)

        name = f"target-{uuid.uuid4().hex[:6]}"
        created = await wf_server.create_workflow(name=name)
        wf_id = created["id"]
        conn = await wf_server._get_conn()
        r1, r2 = str(uuid.uuid4()), str(uuid.uuid4())
        for rid in (r1, r2):
            await conn.execute(
                "INSERT INTO workflow_runs (id, workflow_id, trigger, status, started_at) "
                "VALUES (?, ?, 'manual', 'running', ?)",
                (rid, wf_id, time.time()),
            )
        await conn.commit()

        # Target only r1 by run_id — r2 must be left running.
        res = await wf_server.stop_workflow(wf_id, run_id=r1, wait=False)
        assert res["count"] == 1 and res["stopped"] == [r1], f"unexpected: {res}"
        assert (await wf_server.get_workflow_run(r1))["status"] == "cancelling"
        assert (await wf_server.get_workflow_run(r2))["status"] == "running", \
            "targeting one run wrongly touched another"

        # Resolve the workflow by NAME and stop the remainder (r2).
        res2 = await wf_server.stop_workflow(name, wait=False)
        assert res2["count"] == 1 and res2["stopped"] == [r2], f"name resolution failed: {res2}"
        assert (await wf_server.get_workflow_run(r2))["status"] == "cancelling"
    finally:
        await _reset_mcp_conn(wf_server)
        _restore_env(prev_env)
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("run_cancellation", "stop_workflow wait=True blocks until the run reaches a terminal status")
async def t_mcp_stop_workflow_wait(ctx: TestContext) -> None:
    import src.mcp.servers.workflow_manager.server as wf_server

    tmp_db = ctx.db_path.with_name(f"cancel-wait-{uuid.uuid4().hex[:8]}.db")
    prev_env = os.environ.get("OPENAGENT_DB_PATH")
    finisher: asyncio.Task | None = None
    try:
        os.environ["OPENAGENT_DB_PATH"] = str(tmp_db)
        await _reset_mcp_conn(wf_server)

        created = await wf_server.create_workflow(name=f"wait-{uuid.uuid4().hex[:6]}")
        wf_id = created["id"]
        conn = await wf_server._get_conn()
        rid = str(uuid.uuid4())
        await conn.execute(
            "INSERT INTO workflow_runs (id, workflow_id, trigger, status, started_at) "
            "VALUES (?, ?, 'manual', 'running', ?)",
            (rid, wf_id, time.time()),
        )
        await conn.commit()

        # Simulate the main process finalizing the run shortly after the flag.
        async def _finalize_soon():
            await asyncio.sleep(0.3)
            await conn.execute(
                "UPDATE workflow_runs SET status='cancelled', finished_at=? WHERE id=?",
                (time.time(), rid),
            )
            await conn.commit()

        finisher = asyncio.create_task(_finalize_soon())
        res = await wf_server.stop_workflow(wf_id, wait=True, timeout_s=5)
        await finisher

        assert res["count"] == 1
        assert res["runs"] and res["runs"][0]["status"] == "cancelled", \
            f"wait=True did not observe the terminal status: {res['runs']}"
    finally:
        if finisher is not None and not finisher.done():
            finisher.cancel()
            try:
                await finisher
            except (asyncio.CancelledError, Exception):
                pass
        await _reset_mcp_conn(wf_server)
        _restore_env(prev_env)
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass
