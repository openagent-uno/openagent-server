"""BudgetTracker tests.

The budget-exhausted gate is gone with the yaml knob; the
BudgetTracker itself is still wired inside ``ModelDispatcher`` to record
usage (``record`` path) so these tests cover the class in isolation.
"""
from __future__ import annotations

import uuid

from ._framework import TestContext, test


@test("budget", "BudgetTracker.record + get_usage_summary")
async def t_budget_record(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.models.budget import BudgetTracker

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        bt = BudgetTracker(db, monthly_budget=10.0)
        await bt.record(
            model=f"test:{uuid.uuid4().hex[:6]}",
            input_tokens=1000,
            output_tokens=500,
            cost=0.0042,
            session_id=f"budget-test-{uuid.uuid4().hex[:6]}",
        )
        summary = await bt.get_usage_summary()
        assert summary["monthly_spend"] >= 0.0042
        assert summary["monthly_budget"] == 10.0
        remaining = await bt.get_remaining()
        assert remaining < 10.0, f"remaining={remaining} — usage didn't register"
        ratio = await bt.get_budget_ratio()
        assert 0.0 <= ratio <= 1.0
    finally:
        await db.close()


@test("budget", "BudgetTracker.compute_cost matches catalog")
async def t_budget_compute_cost(ctx: TestContext) -> None:
    """compute_cost reads pricing from the OpenRouter cache — prime it
    with a known shape so the test doesn't depend on the live fetch."""
    import time
    from src.models import discovery
    from src.models.budget import BudgetTracker

    prev = discovery._OPENROUTER_CACHE
    try:
        discovery._OPENROUTER_CACHE = (time.time(), [
            {"id": "openai/gpt-4o-mini", "name": "GPT-4o mini",
             "pricing": {"prompt": "0.00000015", "completion": "0.00000060"}},
        ])
        # $0.15 / $0.60 per million → 1M in, 1M out = $0.75
        cost = BudgetTracker.compute_cost(
            "openai:gpt-4o-mini", 1_000_000, 1_000_000,
        )
        assert abs(cost - 0.75) < 1e-9, f"unexpected cost: {cost}"
    finally:
        discovery._OPENROUTER_CACHE = prev


@test("budget", "a usage row survives losing the write lock instead of vanishing")
async def t_usage_row_survives_contention(ctx: TestContext) -> None:
    """The 3-set-2026 accounting hole, reproduced.

    ``BudgetTracker.record`` swallows whatever ``record_usage`` raises — the
    run must not die because accounting could not write — so a row that lost
    the WAL writer race simply disappeared, leaving nothing but a warning
    nobody counts and an ``usage_log`` that looks calm while it is empty.

    The budget here is deliberately tiny (100 ms against a lock held for
    ~350 ms) so the FIRST attempt is guaranteed to lose: without the retry the
    row is gone, with it the write lands on a later attempt.
    """
    import asyncio
    import os as _os
    import uuid as _uuid

    import aiosqlite

    from src.memory.db import MemoryDB
    from src.models.budget import BudgetTracker

    path = ctx.db_path.with_name(f"usagelock-{_uuid.uuid4().hex[:8]}.db")
    _os.environ["OPENAGENT_SQLITE_BUSY_TIMEOUT_MS"] = "100"
    db = MemoryDB(str(path))
    await db.connect()
    blocker = await aiosqlite.connect(str(path))
    try:
        model = f"test:{_uuid.uuid4().hex[:6]}"
        session = f"usage-lock-{_uuid.uuid4().hex[:6]}"

        await blocker.execute("BEGIN IMMEDIATE")
        await blocker.execute("CREATE TABLE IF NOT EXISTS _lockprobe (x INTEGER)")

        async def release_soon() -> None:
            await asyncio.sleep(0.35)
            await blocker.rollback()

        releaser = asyncio.create_task(release_soon())
        bt = BudgetTracker(db, monthly_budget=10.0)
        await bt.record(
            model=model, input_tokens=7, output_tokens=3, cost=0.001,
            session_id=session,
        )
        await releaser

        conn = await db._ensure_connected()
        row = await (await conn.execute(
            "SELECT COUNT(*) FROM usage_log WHERE session_id = ?", (session,),
        )).fetchone()
        assert row[0] == 1, (
            f"{row[0]} righe invece di 1 — la riga di usage e' stata persa "
            "sulla contesa del lock, che e' esattamente il buco di accounting"
        )
    finally:
        _os.environ.pop("OPENAGENT_SQLITE_BUSY_TIMEOUT_MS", None)
        try:
            await blocker.close()
        except Exception:
            pass
        await db.close()
