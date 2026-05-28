"""WorkflowExecutor._finalize_run survives transient SQLite locks.

The lyra-music 2026-05-19 regression: a workflow's ``ai-prompt`` node
hits HTTP 402 from DeepSeek. Inside the node the runtime's SqliteDb
hammers ``sessions`` from a SQLAlchemy connection while
OpenAgent's aiosqlite connection on the same file tries to write the
final ``workflow_runs`` row. Whichever loses the busy_timeout race
raises ``sqlite3.OperationalError: database is locked`` — and if the
loser is ``_finalize_run``, the row sits in ``running`` forever, the
list UI shows the workflow spinning, and the executor's per-workflow
lock blocks the next scheduled run behind a ghost.

Cosmetic side (``update_workflow``) can fall through with a warning,
but the critical write (``update_workflow_run`` flipping status off
``running``) must succeed — or at minimum surface a clear log on
exhaust so the orphan reaper can recognise it next start.

This test instruments ``MemoryDB.update_workflow_run`` to raise an
``OperationalError`` N times before letting through, and asserts
``_finalize_run`` retries until it lands cleanly.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
import uuid

from ._framework import TestContext, test


class _RealRunCtxStub:
    """Tiny stand-in for ``_RunCtx`` carrying only the fields finalize
    actually reads. The real dataclass is annotated with too much
    framework state to construct cleanly outside an executor."""

    def __init__(self, run_id: str, workflow_id: str) -> None:
        self.run_id = run_id
        self.workflow_id = workflow_id
        self.trace: list[dict] = []
        self.nodes: dict[str, dict] = {}


class _FakeDB:
    """Tracks calls + raises OperationalError on the first ``n`` writes."""

    def __init__(self, *, raise_on_run_updates: int = 0,
                 raise_on_workflow_updates: int = 0) -> None:
        self._raise_on_run = raise_on_run_updates
        self._raise_on_workflow = raise_on_workflow_updates
        self.run_update_calls = 0
        self.workflow_update_calls = 0

    async def update_workflow_run(self, run_id, **kwargs):
        self.run_update_calls += 1
        if self.run_update_calls <= self._raise_on_run:
            raise sqlite3.OperationalError("database is locked")

    async def update_workflow(self, workflow_id, **kwargs):
        self.workflow_update_calls += 1
        if self.workflow_update_calls <= self._raise_on_workflow:
            raise sqlite3.OperationalError("database is locked")


def _patch_sleep_to_noop():
    """Replace asyncio.sleep inside the executor with an awaitable
    no-op so the test isn't gated by real wall-clock backoff."""
    import src.workflow.executor as ex

    orig = ex.asyncio.sleep

    async def fast_sleep(_secs):
        return None

    ex.asyncio.sleep = fast_sleep  # type: ignore[assignment]
    return orig


def _restore_sleep(orig):
    import src.workflow.executor as ex

    ex.asyncio.sleep = orig  # type: ignore[assignment]


@test("workflow_finalize_resilience",
      "_finalize_run retries through 2 transient DB locks and lands clean")
async def t_finalize_retries(ctx: TestContext) -> None:
    from src.workflow.executor import WorkflowExecutor

    db = _FakeDB(raise_on_run_updates=2)
    exe = WorkflowExecutor(agent=None, db=db)  # agent unused by finalize
    rc = _RealRunCtxStub(run_id=str(uuid.uuid4()), workflow_id="wf-test")

    orig_sleep = _patch_sleep_to_noop()
    try:
        await exe._finalize_run(rc, status="failed", error="HTTP 402")
    finally:
        _restore_sleep(orig_sleep)

    assert db.run_update_calls == 3, \
        f"expected 3 attempts (2 fail + 1 succeed), got {db.run_update_calls}"
    assert db.workflow_update_calls == 1, \
        f"workflow-row update should fire exactly once after run-row lands, " \
        f"got {db.workflow_update_calls}"


@test("workflow_finalize_resilience",
      "_finalize_run gives up after 4 attempts and logs (no stuck retry loop)")
async def t_finalize_gives_up(ctx: TestContext) -> None:
    from src.workflow.executor import WorkflowExecutor

    # Always fail — we want to confirm the retry is BOUNDED, not infinite.
    db = _FakeDB(raise_on_run_updates=99)
    exe = WorkflowExecutor(agent=None, db=db)
    rc = _RealRunCtxStub(run_id=str(uuid.uuid4()), workflow_id="wf-stuck")

    orig_sleep = _patch_sleep_to_noop()

    # Capture the ERROR log so we can assert the operator gets a clear
    # signal that this run was left stuck. The next process restart's
    # orphan reaper will pick it up.
    captured: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, rec: logging.LogRecord) -> None:
            if rec.levelno >= logging.ERROR:
                captured.append(rec.getMessage())

    capture = _Capture(level=logging.ERROR)
    logging.getLogger("src.workflow.executor").addHandler(capture)
    try:
        await exe._finalize_run(rc, status="failed", error="HTTP 402")
    finally:
        logging.getLogger("src.workflow.executor").removeHandler(capture)
        _restore_sleep(orig_sleep)

    assert db.run_update_calls == 4, \
        f"expected exactly 4 bounded attempts, got {db.run_update_calls}"
    assert any("finalize failed after 4 attempts" in m for m in captured), \
        f"expected give-up log, got messages: {captured!r}"


@test("workflow_finalize_resilience",
      "lock on update_workflow doesn't fail the finalize (cosmetic-only)")
async def t_finalize_skips_workflow_row_on_lock(ctx: TestContext) -> None:
    """The ``last_run_at`` write is purely UI cosmetic; a lock there
    must not propagate and abort finalize. Run-row already landed."""
    from src.workflow.executor import WorkflowExecutor

    db = _FakeDB(raise_on_run_updates=0, raise_on_workflow_updates=99)
    exe = WorkflowExecutor(agent=None, db=db)
    rc = _RealRunCtxStub(run_id=str(uuid.uuid4()), workflow_id="wf-ui-cosmetic")

    orig_sleep = _patch_sleep_to_noop()
    try:
        # Must not raise.
        await exe._finalize_run(rc, status="success")
    finally:
        _restore_sleep(orig_sleep)

    assert db.run_update_calls == 1
    # One attempt on update_workflow; lock swallowed; no retry needed.
    assert db.workflow_update_calls == 1
