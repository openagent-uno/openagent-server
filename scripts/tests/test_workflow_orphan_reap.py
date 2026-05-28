"""Workflow runs stuck in ``running`` are reaped on agent startup.

Regression pin for lyra-music 2026-05-19: the ``dev-coverage`` workflow
hit DeepSeek's HTTP 402 (Insufficient Balance) on every scheduled tick.
Each retry hammered ``sessions`` while OpenAgent's executor was
trying to finalize the run, the two writers contended on the same
SQLite file, ``update_workflow_run`` lost the race, and the row was
left in ``running`` forever. After a process restart, the next tick
fired a new run for the same workflow — and now the executor's
per-workflow lock is fresh, so the new run starts ON TOP of the
zombie. UI shows the workflow as perpetually "running" and the DB
contention compounds.

The reap is mandatory on serve-start: there's no way to resume an
in-flight run across processes (no checkpoint, no journal), so the
only correct action is to mark it ``failed`` with a clear error
marker and let the schedule's next tick start from a clean slate.

This pins:
  - the reap finalizes every ``running`` row (success/failed left alone)
  - it sets a recognizable error marker so post-hoc forensics know
    the row was machine-finalized, not application-finalized
  - it preserves any pre-existing ``error`` text via concat (so a
    partial finalize that wrote a real error before getting stuck
    doesn't lose its provenance)
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid

from ._framework import TestContext, test


@test("workflow_orphan_reap", "reap_orphan_workflow_runs marks all running rows as failed")
async def t_basic_reap(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"reap-{uuid.uuid4().hex[:8]}.db")
    db: MemoryDB | None = None
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()

        # Seed: 2 running, 1 success, 1 failed.
        # workflow_runs.workflow_id has a FK to workflow_tasks(id), so
        # the parent row must exist before we can write run rows.
        wf_id = await db.add_workflow(name=f"reap-test-{uuid.uuid4().hex[:6]}")
        running_a = await db.add_workflow_run(workflow_id=wf_id, trigger="schedule")
        running_b = await db.add_workflow_run(workflow_id=wf_id, trigger="manual")
        done_ok = await db.add_workflow_run(workflow_id=wf_id, trigger="manual")
        await db.update_workflow_run(done_ok, status="success", finished_at=time.time())
        done_fail = await db.add_workflow_run(workflow_id=wf_id, trigger="schedule")
        await db.update_workflow_run(done_fail, status="failed", error="real error",
                                     finished_at=time.time())

        reaped = await db.reap_orphan_workflow_runs()
        assert reaped == 2, f"expected 2 reaped, got {reaped}"

        # Both running rows now failed and carry the reap marker.
        for rid in (running_a, running_b):
            row = await db.get_workflow_run(rid)
            assert row is not None
            assert row["status"] == "failed", f"{rid} not finalized: {row['status']}"
            assert row.get("finished_at"), f"{rid} missing finished_at"
            assert "reaped" in (row.get("error") or "").lower(), \
                f"{rid} missing reap marker: {row.get('error')!r}"

        # Pre-existing terminal rows are untouched (status, error preserved).
        ok = await db.get_workflow_run(done_ok)
        assert ok["status"] == "success"
        assert "reaped" not in (ok.get("error") or "").lower()
        fail = await db.get_workflow_run(done_fail)
        assert fail["status"] == "failed"
        assert fail.get("error") == "real error", \
            f"existing error text shouldn't be touched: {fail.get('error')!r}"
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


@test("workflow_orphan_reap", "second reap is a no-op (idempotent)")
async def t_reap_idempotent(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"reap-idem-{uuid.uuid4().hex[:8]}.db")
    db: MemoryDB | None = None
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        wf_id = await db.add_workflow(name=f"idem-{uuid.uuid4().hex[:6]}")
        await db.add_workflow_run(workflow_id=wf_id, trigger="schedule")
        first = await db.reap_orphan_workflow_runs()
        second = await db.reap_orphan_workflow_runs()
        assert first == 1, f"first reap should have caught 1, got {first}"
        assert second == 0, f"second reap should be no-op, got {second}"
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


@test("workflow_orphan_reap", "reap preserves any pre-existing error text")
async def t_reap_preserves_error(ctx: TestContext) -> None:
    """When a partial finalize wrote a real error message but never
    flipped status to 'failed' (DB-lock mid-update), the reap should
    append its marker, not overwrite the diagnostic content."""
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"reap-merge-{uuid.uuid4().hex[:8]}.db")
    db: MemoryDB | None = None
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        wf_id = await db.add_workflow(name=f"merge-{uuid.uuid4().hex[:6]}")
        rid = await db.add_workflow_run(workflow_id=wf_id, trigger="manual")
        # Imagine the executor wrote 'error' but failed to flip status.
        await db.update_workflow_run(rid, error="HTTP 402 from DeepSeek")
        await db.reap_orphan_workflow_runs()
        row = await db.get_workflow_run(rid)
        assert row["status"] == "failed"
        err = row.get("error") or ""
        assert "HTTP 402" in err, f"original error lost: {err!r}"
        assert "reaped" in err.lower(), f"reap marker missing: {err!r}"
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
