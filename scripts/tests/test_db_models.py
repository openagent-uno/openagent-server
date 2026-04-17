"""MemoryDB — Model catalog CRUD and probe.

Covers the ``models`` table backing the new dynamic model catalog.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("db_models", "upsert + list + get roundtrip")
async def t_models_roundtrip(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        await db.upsert_model(
            "openai:gpt-test",
            provider="openai",
            model_id="gpt-test",
            display_name="GPT Test",
            input_cost=1.0,
            output_cost=2.0,
            tier_hint="simple",
        )
        row = await db.get_model("openai:gpt-test")
        assert row is not None
        assert row["provider"] == "openai"
        assert row["model_id"] == "gpt-test"
        assert row["input_cost_per_million"] == 1.0
        assert row["tier_hint"] == "simple"
        listed = await db.list_models(provider="openai", enabled_only=True)
        assert any(r["runtime_id"] == "openai:gpt-test" for r in listed)
        await db.delete_model("openai:gpt-test")
    finally:
        await db.close()


@test("db_models", "disable flips enabled, list_models(enabled_only) honors it")
async def t_models_enable(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        await db.upsert_model("openai:gpt-flip", provider="openai", model_id="gpt-flip")
        await db.set_model_enabled("openai:gpt-flip", False)
        row = await db.get_model("openai:gpt-flip")
        assert row["enabled"] is False
        enabled = await db.list_models(provider="openai", enabled_only=True)
        assert not any(r["runtime_id"] == "openai:gpt-flip" for r in enabled)
        all_rows = await db.list_models(provider="openai")
        assert any(r["runtime_id"] == "openai:gpt-flip" for r in all_rows)
        await db.delete_model("openai:gpt-flip")
    finally:
        await db.close()


@test("db_models", "config_state get/set roundtrip (bootstrap marker)")
async def t_state_roundtrip(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        assert await db.get_state("nonexistent") is None
        await db.set_state("probe-flag", "1")
        assert await db.get_state("probe-flag") == "1"
        # Upsert must overwrite rather than duplicate.
        await db.set_state("probe-flag", "2")
        assert await db.get_state("probe-flag") == "2"
    finally:
        await db.close()
