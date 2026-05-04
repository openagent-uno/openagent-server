"""Regression: different workflows dispatched on the same scheduler tick
must run concurrently.

Before the fix, ``Scheduler._check_and_run`` awaited each due item inline:

    for req in requests:
        ...
        await self._run_workflow(wf, ...)   # blocks until done

So if the AI enqueued two ``run_workflow`` requests against two different
workflows, the second waited for the first to finish even though they
shared zero state. The user-visible symptom: "I asked for two workflows
to run and they ran one after the other."

The fix dispatches each due item via ``_spawn_workflow``, an
``asyncio.create_task`` wrapper that tracks the handle in
``self._workflow_tasks`` so ``stop()`` can drain it.

Wall-clock timing is the cleanest signal here — same approach as
``test_sessions_parallel_execution.py``. Two 0.5 s ``wait`` blocks on
two distinct workflows must finish in ~0.5 s total (parallel), not
~1.0 s (serialized).

The companion test pins the *intended* per-workflow serialization: the
executor still holds one ``asyncio.Lock`` per workflow id, so two runs
of the SAME workflow remain ordered. That's a different invariant and
this test exists to keep a future refactor from accidentally dropping it.
"""
from __future__ import annotations

import asyncio
import time

from ._framework import TestContext, test


class _StubAgent:
    """Minimal agent stub. The executor only needs ``_mcp`` (None means
    "skip MCP-existence checks" in ``mcp_inventory_from_pool``) and an
    awaitable ``refresh_registries``; the wait block doesn't touch the
    agent at all. ``forget_session`` is referenced by ``_finalize_run``
    on the shared-policy fast path — keep it harmless."""

    _mcp = None

    async def refresh_registries(self) -> None:
        return None

    async def forget_session(self, session_id: str) -> None:
        return None


def _wait_workflow_graph(seconds: float) -> dict:
    """Single-node graph that pauses for ``seconds``. The wait node is
    an entry (in-degree 0) so the walker fires it without needing a
    trigger block."""
    return {
        "version": 1,
        "nodes": [
            {
                "id": "w1",
                "type": "wait",
                "config": {"mode": "duration", "seconds": seconds},
            },
        ],
        "edges": [],
        "variables": {},
    }


@test("workflow_parallel", "two distinct workflows run concurrently in one tick")
async def t_distinct_workflows_run_in_parallel(ctx: TestContext) -> None:
    from openagent.core.scheduler import Scheduler
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    scheduler = Scheduler(db=db, agent=_StubAgent())  # type: ignore[arg-type]

    wf_a = await db.add_workflow(
        name="wf-parallel-A", graph=_wait_workflow_graph(0.5),
    )
    wf_b = await db.add_workflow(
        name="wf-parallel-B", graph=_wait_workflow_graph(0.5),
    )
    try:
        await db.enqueue_workflow_run_request(workflow_id=wf_a, trigger="api")
        await db.enqueue_workflow_run_request(workflow_id=wf_b, trigger="api")

        start = time.monotonic()
        # ``_check_and_run`` claims both requests and dispatches them as
        # background tasks via ``_spawn_workflow``. Snapshot the set
        # before gathering — the done callbacks remove tasks as they
        # finish, which would race a direct ``gather(*self._workflow_tasks)``.
        await scheduler._check_and_run()
        in_flight = list(scheduler._workflow_tasks)
        assert len(in_flight) == 2, (
            f"expected 2 dispatched tasks, got {len(in_flight)}"
        )
        await asyncio.gather(*in_flight, return_exceptions=True)
        total = time.monotonic() - start

        runs_a = await db.list_workflow_runs(wf_a, limit=1)
        runs_b = await db.list_workflow_runs(wf_b, limit=1)
        assert runs_a and runs_a[0]["status"] == "success", runs_a
        assert runs_b and runs_b[0]["status"] == "success", runs_b

        # Parallel: ≈0.5 s. Serialised (old behaviour): ≥ 1.0 s. 0.85 s
        # crosses the parallel/serial boundary with room for SQLite IO
        # and scheduler bookkeeping noise.
        assert total < 0.85, (
            f"workflows serialised instead of running in parallel; "
            f"total={total:.3f}s"
        )
    finally:
        await db.delete_workflow(wf_a)
        await db.delete_workflow(wf_b)
        await db.close()


@test("workflow_parallel", "two runs of the SAME workflow still serialize")
async def t_same_workflow_runs_serialize(ctx: TestContext) -> None:
    """Pins the executor's per-workflow lock as intentional behaviour.

    Even after the dispatcher fix, ``WorkflowExecutor._locks[workflow_id]``
    must keep two runs of one workflow ordered — concurrent same-id runs
    would race on shared session ids and trace persistence. The lock is
    documented at ``openagent/workflow/executor.py:25`` as the design.
    """
    from openagent.core.scheduler import Scheduler
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    scheduler = Scheduler(db=db, agent=_StubAgent())  # type: ignore[arg-type]

    wf = await db.add_workflow(
        name="wf-serial-same", graph=_wait_workflow_graph(0.5),
    )
    try:
        await db.enqueue_workflow_run_request(workflow_id=wf, trigger="api")
        await db.enqueue_workflow_run_request(workflow_id=wf, trigger="api")

        start = time.monotonic()
        await scheduler._check_and_run()
        in_flight = list(scheduler._workflow_tasks)
        assert len(in_flight) == 2, (
            f"expected 2 dispatched tasks, got {len(in_flight)}"
        )
        await asyncio.gather(*in_flight, return_exceptions=True)
        total = time.monotonic() - start

        runs = await db.list_workflow_runs(wf, limit=2)
        assert len(runs) == 2, runs
        assert all(r["status"] == "success" for r in runs), runs

        # Serial via per-workflow lock: ≈1.0 s. Parallel (lock removed):
        # ≈0.5 s. 0.9 s asserts the lock is in effect with headroom for
        # any executor speedup short of full parallelism.
        assert total >= 0.9, (
            f"per-workflow lock appears to have been removed; "
            f"total={total:.3f}s — runs of one workflow are now interleaving"
        )
    finally:
        await db.delete_workflow(wf)
        await db.close()


@test("workflow_parallel", "API handler stores task locally, not on a shared attr")
async def t_handle_run_no_shared_task_attr(ctx: TestContext) -> None:
    """Regression for the second half of the bug.

    ``handle_run`` used to stash the in-flight task on
    ``scheduler._run_workflow_task`` — a single attribute that
    overlapping API calls trampled, leaving the earlier handler awaiting
    whichever task arrived last. The fix routes through
    ``scheduler._spawn_workflow`` and keeps the handle in a local
    variable. This is a code-shape lint — Test 1 already exercises the
    parallel queue path; this guard makes sure a future refactor of
    ``handle_run`` doesn't quietly reintroduce the shared mutable.
    """
    import re
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent.parent / (
        "openagent/gateway/api/workflow_tasks.py"
    )
    text = src.read_text()
    # Catch assignment specifically — the old broken identifier may
    # legitimately appear in comments explaining the fix.
    bad_assign = re.search(r"scheduler\._run_workflow_task\s*=", text)
    assert bad_assign is None, (
        "handle_run must not stash the task on a shared scheduler attr; "
        "use a local variable + scheduler._spawn_workflow instead"
    )
    assert "scheduler._spawn_workflow(" in text, (
        "handle_run should dispatch through scheduler._spawn_workflow"
    )
