"""BudgetTracker tests.

The SmartRouter budget-exhausted gate is gone with the yaml knob; the
BudgetTracker itself is still wired inside SmartRouter to record usage
(``record`` path) so these tests cover the class in isolation.
"""
from __future__ import annotations

import uuid

from ._framework import TestContext, test


@test("budget", "BudgetTracker.record + get_usage_summary")
async def t_budget_record(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB
    from openagent.models.budget import BudgetTracker

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
    from openagent.models import discovery
    from openagent.models.budget import BudgetTracker

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
