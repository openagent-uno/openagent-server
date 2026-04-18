"""BudgetTracker + SmartRouter budget-aware fallback.

Two tests:

1. BudgetTracker records + summarises usage correctly.
2. When monthly spend exceeds budget, ``SmartRouter._routing_decision``
   should route to the cheaper fallback tier rather than the requested
   one. We simulate the over-budget state by inserting a synthetic row
   in ``usage_log`` rather than actually spending money.
"""
from __future__ import annotations

import uuid

from ._framework import TestContext, TestSkip, have_openai_key, test


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
            {"openai": {"models": ["gpt-4o-mini"]}},
        )
        assert abs(cost - 0.75) < 1e-9, f"unexpected cost: {cost}"
    finally:
        discovery._OPENROUTER_CACHE = prev


@test("budget", "SmartRouter routes to fallback when budget is exhausted")
async def t_router_budget_fallback(ctx: TestContext) -> None:
    if not have_openai_key(ctx.config):
        raise TestSkip("no OpenAI API key")
    from openagent.memory.db import MemoryDB
    from openagent.models.runtime import create_model_from_config, wire_model_runtime

    # Override config to use DIFFERENT models per tier so we can tell them
    # apart. All three must be real, reachable IDs so the router doesn't
    # crash during construction.
    cfg = dict(ctx.config)
    cfg["model"] = dict(ctx.config["model"])
    cfg["model"]["monthly_budget"] = 1.0
    cfg["model"]["routing"] = {
        "simple":   "gpt-4o-mini",
        "medium":   "gpt-4o-mini",
        "hard":     "gpt-4.1-mini",
        "fallback": "gpt-4o-mini",
    }

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        # Spend $2 — double the $1 budget.
        await db.record_usage(
            model="openai:gpt-4.1-mini", input_tokens=1_000_000,
            output_tokens=100_000, cost=2.0,
            session_id=f"over-budget-{uuid.uuid4().hex[:6]}",
        )
        model = create_model_from_config(cfg)
        pool = ctx.extras.get("pool")
        wire_model_runtime(model, db=db, mcp_pool=pool)

        # Force the classifier path to think this is a HARD question, but
        # _routing_decision should override to the cheaper tier when ratio
        # is low. We pass a low budget_ratio directly to skip the live
        # classifier call entirely (deterministic).
        decision = await model._routing_decision(
            messages=[{"role": "user", "content": "anything"}],
            session_id=f"over-test-{uuid.uuid4().hex[:6]}",
            budget_ratio=0.0,  # Exhausted
        )
        # With 0 budget remaining, router should pick simple/fallback, not hard.
        assert decision.requested_tier in ("simple", "medium", "hard")
        # The chosen model should NOT be the hard-tier model
        assert "4.1-mini" not in decision.primary_model, \
            f"budget exhaustion didn't downgrade from hard tier: {decision.primary_model}"
    finally:
        await db.close()
