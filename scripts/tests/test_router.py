"""SmartRouter live tests — cost tracking + entry-model resolution.

The first test confirms that a real call lands a non-zero cost row in
the ``usage_log`` table (the bug that kicked off the whole MCP refactor).
The second drives ``_resolve_entry_model`` directly to verify the
router picks a sensible runtime_id without burning tokens on a full
generate.
"""
from __future__ import annotations

import uuid

from ._framework import TestContext, TestSkip, have_openai_key, test


@test("router", "live generate writes usage_log row with non-zero cost")
async def t_router_usage_log(ctx: TestContext) -> None:
    if not have_openai_key(ctx.config):
        raise TestSkip("no OpenAI API key")
    from src.memory.db import MemoryDB
    from src.models.runtime import create_model_from_config, wire_model_runtime

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


@test("router", "_resolve_entry_model picks the first enabled model")
async def t_router_resolves_entry(ctx: TestContext) -> None:
    if not have_openai_key(ctx.config):
        raise TestSkip("no OpenAI API key")
    from src.models.runtime import create_model_from_config, wire_model_runtime
    from src.memory.db import MemoryDB

    pool = ctx.extras["pool"]
    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        model = create_model_from_config(ctx.config)
        wire_model_runtime(model, db=db, mcp_pool=pool)
        sid = f"router-cls-{uuid.uuid4().hex[:8]}"
        decision = await model._resolve_entry_model(session_id=sid)
        # Resolution: pin → router_flag → first_enabled.
        assert decision.reason in ("first_enabled", "session_pin", "router_flag"), decision
        assert "openai" in decision.primary_model
    finally:
        await db.close()
