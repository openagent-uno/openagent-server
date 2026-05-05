"""``MemoryDB.claim_pending_workflow_requests`` — the cross-process
hand-off the scheduler uses to pick up workflow run requests that the
``workflow-manager`` MCP subprocess enqueued.

Includes a regression test for the
``scheduler.workflow_claim_failed error='cannot start a transaction
within a transaction'`` pattern observed in production: the previous
implementation issued an explicit ``BEGIN IMMEDIATE`` that fought with
the sqlite3 driver's auto-managed transaction state on the shared
aiosqlite connection.
"""
from __future__ import annotations

import uuid

from ._framework import TestContext, test


@test("db_workflow_claim", "claim picks up enqueued requests, marks them claimed")
async def t_claim_basic(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"wfclaim-basic-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        ids = []
        for i in range(3):
            rid = await db.enqueue_workflow_run_request(
                workflow_id=f"wf-{i}", trigger="api", inputs={"i": i},
            )
            ids.append(rid)

        first = await db.claim_pending_workflow_requests(limit=10)
        assert len(first) == 3
        assert {r["id"] for r in first} == set(ids)
        for r in first:
            assert r["claimed_at"] is not None
            assert r["inputs"]["i"] in (0, 1, 2)

        # Second claim must return nothing — all rows already claimed.
        second = await db.claim_pending_workflow_requests(limit=10)
        assert second == []

        # New request flows through.
        rid4 = await db.enqueue_workflow_run_request(
            workflow_id="wf-4", trigger="api",
        )
        third = await db.claim_pending_workflow_requests(limit=10)
        assert len(third) == 1 and third[0]["id"] == rid4

        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("db_workflow_claim", "claim survives a sibling open transaction (regression)")
async def t_claim_under_open_transaction(ctx: TestContext) -> None:
    """The scheduler shares its aiosqlite connection with every other
    coroutine writing through ``MemoryDB``. When a sibling write has
    already auto-begun a transaction (DML executed but not yet
    committed), the previous ``claim_pending_workflow_requests``
    issued ``BEGIN IMMEDIATE`` and SQLite rejected it with
    ``cannot start a transaction within a transaction``. Reproduce
    that connection state here and assert the claim path now succeeds.
    """
    from openagent.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"wfclaim-tx-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        rid = await db.enqueue_workflow_run_request(
            workflow_id="wf-x", trigger="api",
        )

        # Drop into the live connection and execute a DML statement
        # without committing — the sqlite3 driver auto-begins a
        # transaction the next call to ``claim_pending_workflow_requests``
        # would inherit.
        conn = await db._ensure_connected()
        await conn.execute(
            "UPDATE workflow_run_requests SET trigger = trigger WHERE id = ?",
            (rid,),
        )
        # NOTE: deliberately no ``await conn.commit()`` here.

        claimed = await db.claim_pending_workflow_requests(limit=5)
        assert len(claimed) == 1
        assert claimed[0]["id"] == rid

        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("db_workflow_claim", "claim limit honors the cap")
async def t_claim_limit(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"wfclaim-lim-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        for i in range(7):
            await db.enqueue_workflow_run_request(
                workflow_id=f"wf-{i}", trigger="api",
            )
        first = await db.claim_pending_workflow_requests(limit=3)
        assert len(first) == 3
        second = await db.claim_pending_workflow_requests(limit=3)
        assert len(second) == 3
        third = await db.claim_pending_workflow_requests(limit=3)
        assert len(third) == 1
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass
