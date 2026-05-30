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

Same-workflow concurrency is governed by the workflow row's
``max_concurrent_runs`` column. ``NULL`` (default) means unlimited —
two runs of one workflow execute concurrently. ``1`` fully serializes;
``N>1`` admits up to N concurrent runs via a per-workflow semaphore in
``WorkflowExecutor``. The cap-aware tests below pin each of these
contracts so a future refactor can't silently regress them.
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
    from src.core.scheduler import Scheduler
    from src.memory.db import MemoryDB

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


@test(
    "workflow_parallel",
    "default (no cap) → two runs of the SAME workflow run concurrently",
)
async def t_same_workflow_unlimited_default(ctx: TestContext) -> None:
    """v0.14.2 default: ``max_concurrent_runs`` is NULL → unlimited.

    The executor must not gate concurrent runs of one workflow when no
    cap is set; this is the new contract that supersedes the previous
    per-workflow ``asyncio.Lock``. Two 0.5 s runs of one cap-less
    workflow should finish in ≈0.5 s, not ≈1.0 s.
    """
    from src.core.scheduler import Scheduler
    from src.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    scheduler = Scheduler(db=db, agent=_StubAgent())  # type: ignore[arg-type]

    wf = await db.add_workflow(
        name="wf-unlimited-default", graph=_wait_workflow_graph(0.5),
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

        # Default-unlimited: ≈0.5 s. A regression that reintroduces the
        # per-workflow lock would push this past 0.85 s.
        assert total < 0.85, (
            f"two cap-less runs of one workflow serialized; "
            f"total={total:.3f}s — default concurrency cap is not unlimited"
        )
    finally:
        await db.delete_workflow(wf)
        await db.close()


@test(
    "workflow_parallel",
    "max_concurrent_runs=1 → two runs of the SAME workflow serialize",
)
async def t_same_workflow_cap_one_serializes(ctx: TestContext) -> None:
    """Setting the cap to 1 must reproduce the pre-v0.14.2 behaviour:
    concurrent runs of one workflow execute one at a time. Two 0.5 s
    runs should finish in ≈1.0 s, not ≈0.5 s.
    """
    from src.core.scheduler import Scheduler
    from src.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    scheduler = Scheduler(db=db, agent=_StubAgent())  # type: ignore[arg-type]

    wf = await db.add_workflow(
        name="wf-cap-1",
        graph=_wait_workflow_graph(0.5),
        max_concurrent_runs=1,
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

        # Cap=1 forces serial execution: ≈1.0 s. Anything below 0.9 s
        # means the semaphore is not in effect.
        assert total >= 0.9, (
            f"max_concurrent_runs=1 did not serialize; "
            f"total={total:.3f}s — semaphore not respected"
        )
    finally:
        await db.delete_workflow(wf)
        await db.close()


@test(
    "workflow_parallel",
    "max_concurrent_runs=2 → 3 runs cap at 2 concurrent, third queues",
)
async def t_same_workflow_cap_n_admits_n(ctx: TestContext) -> None:
    """With cap=2 and three concurrently-enqueued 0.5 s runs, the third
    must wait for the first batch to free a permit. Wall clock ≈1.0 s.
    No cap: ≈0.5 s. cap=1: ≈1.5 s. cap=2 sits in the middle, so a
    band [0.9 s, 1.3 s] discriminates correctly.
    """
    from src.core.scheduler import Scheduler
    from src.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    scheduler = Scheduler(db=db, agent=_StubAgent())  # type: ignore[arg-type]

    wf = await db.add_workflow(
        name="wf-cap-2",
        graph=_wait_workflow_graph(0.5),
        max_concurrent_runs=2,
    )
    try:
        for _ in range(3):
            await db.enqueue_workflow_run_request(workflow_id=wf, trigger="api")

        start = time.monotonic()
        await scheduler._check_and_run()
        in_flight = list(scheduler._workflow_tasks)
        assert len(in_flight) == 3, (
            f"expected 3 dispatched tasks, got {len(in_flight)}"
        )
        await asyncio.gather(*in_flight, return_exceptions=True)
        total = time.monotonic() - start

        runs = await db.list_workflow_runs(wf, limit=3)
        assert len(runs) == 3, runs
        assert all(r["status"] == "success" for r in runs), runs

        assert 0.9 <= total < 1.3, (
            f"max_concurrent_runs=2 with 3 runs took {total:.3f}s; "
            f"expected ≈1.0s (≥0.9, <1.3) — the cap is either too "
            f"permissive (<0.9s acts like ∞) or too restrictive (≥1.3s "
            f"acts like cap=1)"
        )
    finally:
        await db.delete_workflow(wf)
        await db.close()


@test(
    "workflow_parallel",
    "executor live-resizes its semaphore when max_concurrent_runs is edited",
)
async def t_cap_resize_takes_effect(ctx: TestContext) -> None:
    """Editing ``max_concurrent_runs`` mid-life must not require a
    restart: the next ``run()`` call rebuilds the semaphore against the
    fresh value. Start serialized (cap=1, ~1.0 s for two runs), then
    edit to ``None`` (unlimited) and assert two more runs finish in
    parallel (~0.5 s).
    """
    from src.core.scheduler import Scheduler
    from src.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    scheduler = Scheduler(db=db, agent=_StubAgent())  # type: ignore[arg-type]

    wf = await db.add_workflow(
        name="wf-cap-resize",
        graph=_wait_workflow_graph(0.4),
        max_concurrent_runs=1,
    )
    try:
        await db.enqueue_workflow_run_request(workflow_id=wf, trigger="api")
        await db.enqueue_workflow_run_request(workflow_id=wf, trigger="api")
        start = time.monotonic()
        await scheduler._check_and_run()
        await asyncio.gather(*list(scheduler._workflow_tasks), return_exceptions=True)
        serial_total = time.monotonic() - start
        assert serial_total >= 0.7, (
            f"cap=1 phase ran in {serial_total:.3f}s — expected ≥0.7s "
            f"(serial)"
        )

        # Resize → unlimited and replay.
        await db.update_workflow(wf, max_concurrent_runs=None)
        await db.enqueue_workflow_run_request(workflow_id=wf, trigger="api")
        await db.enqueue_workflow_run_request(workflow_id=wf, trigger="api")
        start = time.monotonic()
        await scheduler._check_and_run()
        await asyncio.gather(*list(scheduler._workflow_tasks), return_exceptions=True)
        parallel_total = time.monotonic() - start
        assert parallel_total < 0.7, (
            f"after resize to unlimited, two runs took {parallel_total:.3f}s — "
            f"expected <0.7s (parallel); semaphore was not dropped"
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
        "src/gateway/api/workflow_tasks.py"
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
