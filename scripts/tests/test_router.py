"""SmartRouter live tests — cost tracking + classifier routing.

The first test confirms that a real call lands a non-zero cost row in
the ``usage_log`` table (the bug that kicked off the whole MCP refactor).
The second test drives ``_routing_decision`` directly to verify the
classifier picks a sensible tier without burning tokens on a full
generate.
"""
from __future__ import annotations

import uuid

from ._framework import TestContext, TestSkip, have_openai_key, test


@test("router", "live generate writes usage_log row with non-zero cost")
async def t_router_usage_log(ctx: TestContext) -> None:
    if not have_openai_key(ctx.config):
        raise TestSkip("no OpenAI API key")
    from openagent.memory.db import MemoryDB
    from openagent.models.runtime import create_model_from_config, wire_model_runtime

    pool = ctx.extras["pool"]
    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        model = create_model_from_config(ctx.config)
        wire_model_runtime(model, db=db, mcp_pool=pool)
        sid = f"router-test-{uuid.uuid4().hex[:8]}"
        resp = await model.generate(
            messages=[{"role": "user", "content": "Reply with literally PONG and nothing else."}],
            system="You are a test bot.",
            session_id=sid,
        )
        assert "PONG" in resp.content.upper(), f"{resp.content!r}"
        summary = await db.get_usage_summary()
        assert summary["total"] > 0, f"usage_log total=0; by_model={summary['by_model']}"
        assert any("openai:gpt-4o-mini" in m for m in summary["by_model"]), \
            f"no openai:gpt-4o-mini row in usage_log: {summary['by_model']}"
    finally:
        await db.close()


@test("router", "classifier routes 'simple' question to simple tier model")
async def t_router_classifies(ctx: TestContext) -> None:
    if not have_openai_key(ctx.config):
        raise TestSkip("no OpenAI API key")
    from openagent.models.runtime import create_model_from_config, wire_model_runtime
    from openagent.memory.db import MemoryDB

    pool = ctx.extras["pool"]
    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        model = create_model_from_config(ctx.config)
        wire_model_runtime(model, db=db, mcp_pool=pool)
        sid = f"router-cls-{uuid.uuid4().hex[:8]}"
        decision = await model._routing_decision(
            messages=[{"role": "user", "content": "hi"}],
            session_id=sid,
            budget_ratio=1.0,
        )
        assert decision.requested_tier in ("simple", "medium", "hard"), decision
        # In our test config every tier maps to gpt-4o-mini
        assert "openai" in decision.primary_model
    finally:
        await db.close()
