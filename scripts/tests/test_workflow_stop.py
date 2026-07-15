"""``POST /api/workflows/{id}/stop`` — the app can stop what it started.

The asymmetry this closes: ``POST /api/workflows/{id}/run`` shipped, and
``POST /api/scheduled-tasks/{id}/stop`` shipped, but the workflow half of stop
was never registered on the gateway. Stopping an in-flight workflow run was
reachable only through the agent-facing ``workflow_manager`` MCP tool, so a
user watching a runaway run from the desktop app had to ask the agent to kill
it — and §10 says the gateway is the *only* public surface, with no privileged
path the first-party client can't take.

Nothing new had to be built: the scheduler's cancellation drain already handles
workflow runs, and the hand-off is a ``status='cancelling'`` row (never IPC),
like ``workflow_run_requests`` carries a start the other way. These tests pin
the missing door and the honest edges around it — a run that already finished,
an unknown run id, a run belonging to a different workflow, and a double-stop.

Pure-unit: a real ``Scheduler`` + a sleeping executor over a temp DB, handlers
driven directly. No gateway lifecycle, no LLM.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from types import SimpleNamespace
from unittest.mock import Mock

from aiohttp import streams
from aiohttp.test_utils import make_mocked_request

from ._framework import TestContext, test


# ── fakes ────────────────────────────────────────────────────────────


class _FakeAgent:
    name = "fake"
    model = None

    async def refresh_registries(self) -> None:
        return None

    async def run(self, message=None, user_id=None, session_id=None, **kw):
        await asyncio.sleep(30)
        return "should have been cancelled"

    async def forget_session(self, session_id: str) -> None:
        return None

    async def release_session(self, session_id: str, *, model_override=None) -> None:
        return None


class _SleepingExecutor:
    """Stand-in for ``WorkflowExecutor``: opens the run row exactly like the
    real one, then blocks. Mirrors the real executor's contract of NOT catching
    ``CancelledError`` — the scheduler's handler owns finalization."""

    def __init__(self, db) -> None:
        self.db = db

    async def run(self, workflow, *, trigger="manual", inputs=None, run_id=None,
                  entry_node_id=None, on_status=None):
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


def _make_request(scheduler, *, method: str = "POST", path: str = "/x",
                  match_info: dict | None = None, body: dict | None = None,
                  broadcasts: list | None = None):
    """A minimal aiohttp request the workflow handlers read: the scheduler off
    ``app['gateway']``, ``match_info``, an optional JSON body, and
    ``broadcast_resource`` (recorded so the UI-signal path is covered).

    The body is fed as a real payload stream rather than a stubbed ``.json()``,
    so ``request.can_read_body`` / ``request.json()`` behave as they do on the
    wire — the handler's own body handling is part of what's under test.
    """

    async def _broadcast(resource, action, id_=None) -> None:
        if broadcasts is not None:
            broadcasts.append((resource, action, id_))

    gw = SimpleNamespace(_scheduler=scheduler, broadcast_resource=_broadcast)
    kwargs: dict = {"match_info": match_info or {}, "app": {"gateway": gw}}
    if body is not None:
        raw = json.dumps(body).encode()
        # ``Mock``, not ``SimpleNamespace``: StreamReader calls back into its
        # protocol (``resume_reading``/``pause_reading``) from paths whose
        # guards move between aiohttp releases, and the signature moves too —
        # 3.13 calls ``resume_reading()``, later versions
        # ``resume_reading(resume_parser=False)``. A SimpleNamespace carrying
        # only ``_reading_paused`` satisfied 3.13 and blew up with
        # ``AttributeError: no attribute 'resume_reading'`` on CI, which
        # resolves ``aiohttp>=3.9`` to the latest because there is no lock in
        # ``uv pip install -e .``. Mock absorbs whatever the library calls,
        # with whatever signature; ``_reading_paused`` stays an explicit False
        # because it is read as a flag, and a bare Mock attribute is truthy.
        protocol = Mock(_reading_paused=False)
        stream = streams.StreamReader(
            protocol, limit=2 ** 16, loop=asyncio.get_running_loop(),
        )
        stream.feed_data(raw)
        stream.feed_eof()
        kwargs["payload"] = stream
        kwargs["headers"] = {
            "Content-Type": "application/json",
            "Content-Length": str(len(raw)),
        }
    return make_mocked_request(method, path, **kwargs)


async def _new_scheduler(ctx: TestContext, tag: str):
    from src.core.scheduler import Scheduler
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"wf-stop-{tag}-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(tmp_db))
    await db.connect()
    sched = Scheduler(db, _FakeAgent())
    sched._workflow_executor = _SleepingExecutor(db)
    return sched, db, tmp_db


async def _cleanup(db, tmp_db, *tasks) -> None:
    for t in tasks:
        if t is not None and not t.done():
            t.cancel()
            try:
                await t
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


# ── the missing route ────────────────────────────────────────────────


@test("workflow_stop", "the gateway registers POST /api/workflows/{id}/stop")
async def t_route_registered(_ctx: TestContext) -> None:
    """The gap was literally an unregistered route: every layer beneath worked.
    Asserted against the real router table, next to its scheduled-task twin so
    the pair can't drift apart again."""
    from aiohttp import web

    from src.gateway.server import Gateway

    # ``_register_routes`` only reaches for ``self._handle_*`` bound methods,
    # so an uninitialised instance is enough — building a real Gateway needs a
    # live Agent + Iroh endpoint, which this assertion has no use for.
    gw = object.__new__(Gateway)
    app = web.Application()
    gw._register_routes(app)

    registered = {
        (r.method, r.resource.canonical)
        for r in app.router.routes()
        if r.resource is not None
    }
    assert ("POST", "/api/workflows/{id}/stop") in registered, (
        "workflow stop is not reachable over REST — the app can start a run "
        f"it cannot stop. workflow routes seen: "
        f"{sorted(p for m, p in registered if p.startswith('/api/workflows'))}"
    )
    # Its twin, and the start it pairs with, must still be there.
    assert ("POST", "/api/scheduled-tasks/{id}/stop") in registered
    assert ("POST", "/api/workflows/{id}/run") in registered


# ── start via REST → stop via REST ───────────────────────────────────


@test("workflow_stop", "a workflow run started via REST is really stopped via REST")
async def t_rest_start_rest_stop(ctx: TestContext) -> None:
    """The end-to-end claim, with no MCP anywhere in the path: POST /run opens
    a real run, POST /stop flags it, the scheduler's own drain hard-cancels the
    executor, and the row finalizes ``cancelled`` — not stranded ``running``,
    not ``failed``."""
    from src.gateway.api.workflow_tasks import handle_run, handle_stop

    sched, db, tmp_db = await _new_scheduler(ctx, "e2e")
    run_task = None
    try:
        wf_id = await db.add_workflow(name=f"e2e-{uuid.uuid4().hex[:6]}")
        broadcasts: list = []

        # 1. Start it the way the app does: POST /run with wait=False.
        resp = await handle_run(_make_request(
            sched, path=f"/api/workflows/{wf_id}/run",
            match_info={"id": wf_id}, body={"wait": False},
            broadcasts=broadcasts,
        ))
        assert resp.status == 202, (resp.status, resp.body)
        started = json.loads(resp.body)
        run_id = started["run_id"]
        assert run_id, started
        assert started["status"] == "running", started

        # The scheduler owns a live task for it (what the drain will cancel).
        run_task = sched._workflow_run_tasks.get(run_id)
        assert run_task is not None, "the run never registered a cancellable task"

        # 2. Stop it the way the app would. wait=False so the response is the
        #    flag itself; the drain below is what makes it real.
        resp = await handle_stop(_make_request(
            sched, path=f"/api/workflows/{wf_id}/stop",
            match_info={"id": wf_id}, body={"wait": False},
            broadcasts=broadcasts,
        ))
        assert resp.status == 200, (resp.status, resp.body)
        stopped = json.loads(resp.body)
        assert stopped["count"] == 1, stopped
        assert stopped["stopped"] == [run_id], stopped
        assert stopped["runs"] == [{"id": run_id, "status": "cancelling"}], stopped
        assert stopped["workflow_id"] == wf_id, stopped
        assert (await db.get_workflow_run(run_id))["status"] == "cancelling"

        # 3. The scheduler's existing drain turns the flag into a hard stop.
        await sched._drain_cancellations()
        assert await _wait_done(run_task), "cancelled run did not unwind"
        assert run_task.cancelled(), "run task should have ended cancelled"

        row = await db.get_workflow_run(run_id)
        assert row["status"] == "cancelled", \
            f"expected cancelled, got {row['status']!r}"
        assert row.get("finished_at"), "cancelled run missing finished_at"
        assert run_id not in sched._workflow_run_tasks, "registry leaked"

        # The screen was told, both when the stop landed and after it settled.
        assert ("workflow", "updated", wf_id) in broadcasts, broadcasts
    finally:
        await _cleanup(db, tmp_db, run_task)


@test("workflow_stop", "stop wait=True reports the run's real terminal status")
async def t_stop_wait_true(ctx: TestContext) -> None:
    """``wait`` defaults to True, so the default response must reflect what
    actually happened rather than the intent — same promise the scheduled-task
    endpoint makes."""
    from src.gateway.api.workflow_tasks import handle_stop

    sched, db, tmp_db = await _new_scheduler(ctx, "wait")
    run_task = drain = None
    try:
        wf_id = await db.add_workflow(name=f"wait-{uuid.uuid4().hex[:6]}")
        wf = await db.get_workflow(wf_id)
        run_task = sched._spawn_workflow(sched._run_workflow(wf, trigger="api"))

        async def _ready():
            if not sched._workflow_run_tasks:
                return None
            rid = next(iter(sched._workflow_run_tasks))
            row = await db.get_workflow_run(rid)
            return rid if (row and row["status"] == "running") else None

        run_id = await _poll(_ready)
        assert run_id, "workflow run never reached running"

        # Stand in for the scheduler's 2s drain loop, which isn't running here.
        async def _drain_soon():
            await asyncio.sleep(0.2)
            await sched._drain_cancellations()

        drain = asyncio.create_task(_drain_soon())
        resp = await handle_stop(_make_request(
            sched, path=f"/api/workflows/{wf_id}/stop",
            match_info={"id": wf_id}, body={"wait": True, "timeout_s": 5},
        ))
        await drain

        body = json.loads(resp.body)
        assert resp.status == 200, (resp.status, body)
        assert body["count"] == 1, body
        assert body["runs"] == [{"id": run_id, "status": "cancelled"}], (
            f"wait=True must not return before the run settles: {body['runs']}"
        )
    finally:
        await _cleanup(db, tmp_db, run_task, drain)


# ── honest edges ─────────────────────────────────────────────────────


@test("workflow_stop", "stopping an unknown workflow is 404; no scheduler is 503")
async def t_unknown_and_no_scheduler(ctx: TestContext) -> None:
    from src.gateway.api.workflow_tasks import handle_stop

    sched, db, tmp_db = await _new_scheduler(ctx, "404")
    try:
        resp = await handle_stop(_make_request(
            sched, path="/api/workflows/nope/stop",
            match_info={"id": "nope"}, body={"wait": False},
        ))
        assert resp.status == 404, (resp.status, resp.body)
        assert "not found" in json.loads(resp.body)["error"], resp.body

        # No live Scheduler attached — same invariant every other handler here
        # holds: reject rather than pretend.
        gw = SimpleNamespace(_scheduler=None)
        req = make_mocked_request(
            "POST", "/api/workflows/x/stop", match_info={"id": "x"},
            app={"gateway": gw},
        )
        resp = await handle_stop(req)
        assert resp.status == 503, (resp.status, resp.body)
    finally:
        await _cleanup(db, tmp_db)


@test("workflow_stop", "stopping a finished run, an unknown run, or twice: 200 + count 0")
async def t_nothing_to_stop(ctx: TestContext) -> None:
    """Every "nothing happened" case answers the same way, and none of them
    invents a failure. The Stop button the app renders is at best one frame
    stale, so a benign race must not surface as an error dialog — this is the
    call ``/api/scheduled-tasks/{id}/stop`` already makes."""
    from src.gateway.api.workflow_tasks import handle_stop

    sched, db, tmp_db = await _new_scheduler(ctx, "noop")
    try:
        wf_id = await db.add_workflow(name=f"noop-{uuid.uuid4().hex[:6]}")

        # (a) A run that already finished on its own.
        done_id = await db.add_workflow_run(workflow_id=wf_id, trigger="api")
        await db.update_workflow_run(
            done_id, status="success", finished_at=time.time(),
        )
        resp = await handle_stop(_make_request(
            sched, path=f"/api/workflows/{wf_id}/stop",
            match_info={"id": wf_id}, body={"wait": False},
        ))
        assert resp.status == 200, (resp.status, resp.body)
        body = json.loads(resp.body)
        assert body["count"] == 0 and body["stopped"] == [], body
        assert body["runs"] == [], body
        assert "No running run to stop" in body["note"], body
        # …and the settled run was NOT rewritten as cancelled.
        assert (await db.get_workflow_run(done_id))["status"] == "success", \
            "stop resurrected a run that had already succeeded"

        # (b) A run id that doesn't exist at all.
        resp = await handle_stop(_make_request(
            sched, path=f"/api/workflows/{wf_id}/stop",
            match_info={"id": wf_id},
            body={"run_id": str(uuid.uuid4()), "wait": False},
        ))
        assert resp.status == 200, (resp.status, resp.body)
        body = json.loads(resp.body)
        assert body["count"] == 0, body
        assert "run_id" in body["note"], \
            f"the note must say which run it looked for: {body['note']}"

        # (c) A double-stop: the second call has nothing left in `running`.
        live_id = await db.add_workflow_run(workflow_id=wf_id, trigger="api")
        first = await handle_stop(_make_request(
            sched, path=f"/api/workflows/{wf_id}/stop",
            match_info={"id": wf_id}, body={"wait": False},
        ))
        assert json.loads(first.body)["count"] == 1, first.body
        second = await handle_stop(_make_request(
            sched, path=f"/api/workflows/{wf_id}/stop",
            match_info={"id": wf_id}, body={"wait": False},
        ))
        assert second.status == 200, second.status
        assert json.loads(second.body)["count"] == 0, second.body
        # The in-flight stop is untouched — not re-flagged, not undone.
        assert (await db.get_workflow_run(live_id))["status"] == "cancelling"
    finally:
        await _cleanup(db, tmp_db)


@test("workflow_stop", "a run_id from another workflow cannot be stopped through this one")
async def t_cross_workflow_run_id(ctx: TestContext) -> None:
    """``run_id`` is caller-supplied, so the workflow in the path must actually
    constrain it — otherwise /api/workflows/{A}/stop could reach into B's run."""
    from src.gateway.api.workflow_tasks import handle_stop

    sched, db, tmp_db = await _new_scheduler(ctx, "cross")
    try:
        wf_a = await db.add_workflow(name=f"a-{uuid.uuid4().hex[:6]}")
        wf_b = await db.add_workflow(name=f"b-{uuid.uuid4().hex[:6]}")
        b_run = await db.add_workflow_run(workflow_id=wf_b, trigger="api")

        resp = await handle_stop(_make_request(
            sched, path=f"/api/workflows/{wf_a}/stop",
            match_info={"id": wf_a}, body={"run_id": b_run, "wait": False},
        ))
        assert resp.status == 200, (resp.status, resp.body)
        assert json.loads(resp.body)["count"] == 0, resp.body
        assert (await db.get_workflow_run(b_run))["status"] == "running", \
            "a stop on workflow A reached into workflow B's run"
    finally:
        await _cleanup(db, tmp_db)


@test("workflow_stop", "REST and the MCP tool write the same flag through one helper")
async def t_rest_and_mcp_share_the_helper(ctx: TestContext) -> None:
    """Two requesters, one hand-off. If either grows its own copy of the SQL,
    the ``status='running'`` guard can drift on one side only — and that guard
    is what stops a settled run being rewritten as cancelled."""
    import src.mcp.servers.workflow_manager.server as wf_server
    from src.gateway.api import workflow_tasks
    from src.workflow.cancel import flag_workflow_runs_cancelling

    assert wf_server.flag_workflow_runs_cancelling is flag_workflow_runs_cancelling

    # The REST handler imports it lazily inside the function body, so pin the
    # source text instead of an attribute (the import must not be re-derived).
    import inspect
    src_text = inspect.getsource(workflow_tasks.handle_stop)
    assert "flag_workflow_runs_cancelling" in src_text, \
        "the REST stop handler no longer uses the shared hand-off helper"

    # The guard itself, at the helper: a settled run is never flagged.
    sched, db, tmp_db = await _new_scheduler(ctx, "guard")
    try:
        wf_id = await db.add_workflow(name=f"guard-{uuid.uuid4().hex[:6]}")
        ok_id = await db.add_workflow_run(workflow_id=wf_id, trigger="api")
        await db.update_workflow_run(ok_id, status="success", finished_at=time.time())
        run_id = await db.add_workflow_run(workflow_id=wf_id, trigger="api")

        conn = await db._ensure_connected()
        flagged = await flag_workflow_runs_cancelling(conn, workflow_id=wf_id)
        assert flagged == [run_id], flagged
        assert (await db.get_workflow_run(ok_id))["status"] == "success"
        assert (await db.get_workflow_run(run_id))["status"] == "cancelling"
    finally:
        await _cleanup(db, tmp_db)


class _RacingConn:
    """Delegates to a real connection, but lets the targeted run finish in the
    window between the helper's SELECT and its UPDATE — the race the
    ``status='running'`` guard exists for, which no amount of timing luck would
    reproduce on demand."""

    def __init__(self, conn, run_id: str) -> None:
        self._conn = conn
        self._run_id = run_id
        self.fired = False

    async def execute(self, sql, params=None):
        if sql.lstrip().upper().startswith("UPDATE") and not self.fired:
            self.fired = True
            # The executor finishes the run a hair after the stop picked it.
            await self._conn.execute(
                "UPDATE workflow_runs SET status='success', finished_at=? WHERE id=?",
                (time.time(), self._run_id),
            )
            await self._conn.commit()
        if params is None:
            return await self._conn.execute(sql)
        return await self._conn.execute(sql, params)

    async def commit(self):
        return await self._conn.commit()


@test("workflow_stop", "a run that finishes mid-stop is never rewritten as cancelled")
async def t_settled_run_not_resurrected(ctx: TestContext) -> None:
    """Without the guard, a run that reached ``success`` between the SELECT and
    the UPDATE gets flagged anyway; the scheduler's drain then finds no live
    task and finalizes the *successful* run as ``cancelled``. History would
    record a lie, and the run screen would show a green run as stopped."""
    from src.workflow.cancel import flag_workflow_runs_cancelling

    sched, db, tmp_db = await _new_scheduler(ctx, "race")
    try:
        wf_id = await db.add_workflow(name=f"race-{uuid.uuid4().hex[:6]}")
        run_id = await db.add_workflow_run(workflow_id=wf_id, trigger="api")

        conn = await db._ensure_connected()
        racing = _RacingConn(conn, run_id)
        flagged = await flag_workflow_runs_cancelling(racing, workflow_id=wf_id)

        assert racing.fired, "the race never triggered — the test proves nothing"
        # The helper reports what it targeted (it cannot know it lost the race)…
        assert flagged == [run_id], flagged
        # …but the row itself must be untouched: still success, not cancelling.
        row = await db.get_workflow_run(run_id)
        assert row["status"] == "success", (
            f"a run that finished mid-stop was rewritten to {row['status']!r} — "
            f"the scheduler's drain would then record the successful run as "
            f"cancelled"
        )
    finally:
        await _cleanup(db, tmp_db)
