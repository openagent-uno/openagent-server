"""MemoryDB — Model catalog CRUD and probe (v0.12 schema).

Covers the ``models`` table. Under v0.12, ``models.provider_id`` is a
FK to ``providers.id`` and ``framework`` is inherited from the provider
row. ``runtime_id`` is derived at read time, not stored.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("db_models", "upsert + list + get roundtrip")
async def t_models_roundtrip(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        pid = await db.upsert_provider(
            name="openai", framework="agno", api_key="sk-test",
        )
        mid = await db.upsert_model(
            provider_id=pid,
            model="gpt-test",
            display_name="GPT Test",
            tier_hint="fast, cheap, vision",
        )
        row = await db.get_model(mid)
        assert row is not None
        assert row["provider_id"] == pid
        assert row["model"] == "gpt-test"
        assert row["tier_hint"] == "fast, cheap, vision"

        listed = await db.list_models(provider_id=pid, enabled_only=True)
        assert any(r["id"] == mid for r in listed)

        # Enriched view derives runtime_id on the fly.
        enriched = await db.list_models_enriched(provider_name="openai")
        rids = [r["runtime_id"] for r in enriched]
        assert "openai:gpt-test" in rids

        await db.delete_model(mid)
        await db.delete_provider(pid)
    finally:
        await db.close()


@test("db_models", "disable flips enabled, list_models(enabled_only) honors it")
async def t_models_enable(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        pid = await db.upsert_provider(
            name="openai", framework="agno", api_key="sk-test",
        )
        mid = await db.upsert_model(provider_id=pid, model="gpt-flip")
        await db.set_model_enabled(mid, False)
        row = await db.get_model(mid)
        assert row["enabled"] is False
        enabled = await db.list_models(provider_id=pid, enabled_only=True)
        assert not any(r["id"] == mid for r in enabled)
        all_rows = await db.list_models(provider_id=pid)
        assert any(r["id"] == mid for r in all_rows)
        await db.delete_provider(pid)  # FK cascade wipes the model
    finally:
        await db.close()


@test("db_models", "FK cascade from provider → models")
async def t_cascade_delete(ctx: TestContext) -> None:
    """When a provider row is removed, every model under it is
    cascade-deleted via ``ON DELETE CASCADE`` — replacing the old
    manual ``delete_models_by_provider`` contract."""
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        zai_id = await db.upsert_provider(
            name="zai", framework="agno", api_key="zk",
        )
        oai_id = await db.upsert_provider(
            name="openai", framework="agno", api_key="sk",
        )
        await db.upsert_model(provider_id=zai_id, model="glm-5")
        await db.upsert_model(provider_id=zai_id, model="glm-4.5")
        await db.upsert_model(provider_id=oai_id, model="gpt-4o-mini")

        await db.delete_provider(zai_id)
        remaining = await db.list_models_enriched()
        providers = {r["provider_name"] for r in remaining}
        assert providers == {"openai"}, providers

        # Remaining openai row still works
        assert len(await db.list_models(provider_id=oai_id)) == 1
        await db.delete_provider(oai_id)
    finally:
        await db.close()


@test("db_models", "registry_status.enabled_count reflects model AND provider enable")
async def t_registry_status_empty(ctx: TestContext) -> None:
    """The gate relies on ``registry_status`` returning zero once the
    effective catalog is empty. A model under a DISABLED provider
    can't dispatch anyway, so the count joins on provider.enabled."""
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        oai_id = await db.upsert_provider(
            name="openai", framework="agno", api_key="sk",
        )
        ant_id = await db.upsert_provider(
            name="anthropic", framework="claude-cli",
        )
        mid_a = await db.upsert_model(provider_id=oai_id, model="gpt-gate-a")
        mid_b = await db.upsert_model(provider_id=ant_id, model="sonnet-gate")

        _, _, count, _ = await db.registry_status()
        assert count == 2, count

        # Disable the model row → count drops.
        await db.set_model_enabled(mid_a, False)
        _, _, count, _ = await db.registry_status()
        assert count == 1, count

        # Disable the claude-cli provider row → its model is effectively
        # disabled too, count drops to 0.
        await db.set_provider_enabled(ant_id, False)
        _, _, count, _ = await db.registry_status()
        assert count == 0, f"registry_status still reports {count}"

        await db.delete_model(mid_a)
        await db.delete_model(mid_b)
        await db.delete_provider(oai_id)
        await db.delete_provider(ant_id)
    finally:
        await db.close()


@test("db_models", "get_model_by_runtime_id resolves to the enriched row")
async def t_runtime_id_lookup(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        cli_id = await db.upsert_provider(name="anthropic", framework="claude-cli")
        await db.upsert_model(provider_id=cli_id, model="claude-opus-4-7")
        row = await db.get_model_by_runtime_id("claude-cli:anthropic:claude-opus-4-7")
        assert row is not None
        assert row["framework"] == "claude-cli"
        assert row["provider_name"] == "anthropic"
        assert row["model"] == "claude-opus-4-7"
        assert row["runtime_id"] == "claude-cli:anthropic:claude-opus-4-7"
        # Unknown runtime_id → None (no exception).
        assert await db.get_model_by_runtime_id("openai:nonexistent") is None
        await db.delete_provider(cli_id)
    finally:
        await db.close()


@test("db_models", "upsert_model rejects orphan provider_id")
async def t_reject_orphan(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        raised = False
        try:
            await db.upsert_model(provider_id=99_999, model="nope")
        except ValueError as e:
            raised = True
            assert "does not exist" in str(e).lower()
        assert raised
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
