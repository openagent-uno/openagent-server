"""MemoryDB — Providers table CRUD (v0.12 schema).

Covers the ``providers`` table — the source of truth for LLM
credentials + dispatch framework. One row per (name, framework) pair;
deleting a row cascades to wipe its models via FK.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("db_providers", "upsert + list + get roundtrip")
async def t_providers_roundtrip(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        pid = await db.upsert_provider(
            name="openai",
            framework="agno",
            api_key="sk-test-key",
            base_url="https://api.openai.com/v1",
            enabled=True,
            metadata={"tier": "paid"},
        )
        assert isinstance(pid, int) and pid > 0
        row = await db.get_provider(pid)
        assert row is not None
        assert row["id"] == pid
        assert row["name"] == "openai"
        assert row["framework"] == "agno"
        assert row["api_key"] == "sk-test-key"
        assert row["base_url"] == "https://api.openai.com/v1"
        assert row["enabled"] is True
        assert row["metadata"] == {"tier": "paid"}

        listed = await db.list_providers()
        assert [(r["name"], r["framework"]) for r in listed] == [("openai", "agno")]

        row_by_name = await db.get_provider_by_name("openai", "agno")
        assert row_by_name is not None and row_by_name["id"] == pid

        await db.delete_provider(pid)
        assert await db.get_provider(pid) is None
    finally:
        await db.close()


@test("db_providers", "upsert is idempotent and preserves id + created_at")
async def t_providers_upsert_idempotent(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        pid1 = await db.upsert_provider(name="zai", framework="agno", api_key="key-1")
        first = await db.get_provider(pid1)
        pid2 = await db.upsert_provider(
            name="zai", framework="agno",
            api_key="key-2", base_url="https://api.z.ai/api/paas/v4",
        )
        assert pid1 == pid2, "upsert on same (name, framework) must keep the same id"
        second = await db.get_provider(pid2)
        assert second["api_key"] == "key-2"
        assert second["base_url"] == "https://api.z.ai/api/paas/v4"
        assert second["created_at"] == first["created_at"], "upsert must preserve created_at"
        assert second["updated_at"] >= first["updated_at"]
        await db.delete_provider(pid1)
    finally:
        await db.close()


@test("db_providers", "claude-cli provider forbids api_key (sentinel class of bug)")
async def t_claude_cli_rejects_api_key(ctx: TestContext) -> None:
    """The v0.11.x ``api_key='claude-cli'`` sentinel poisoned the
    claude subprocess via ``ANTHROPIC_API_KEY``. The schema now rejects
    any api_key for claude-cli providers at the DB boundary — no
    downstream filter needed."""
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        raised = False
        try:
            await db.upsert_provider(
                name="anthropic", framework="claude-cli", api_key="claude-cli",
            )
        except ValueError as e:
            raised = True
            assert "api_key" in str(e).lower()
        assert raised, "upsert_provider must raise when claude-cli carries api_key"

        # But claude-cli WITHOUT api_key is the happy path.
        pid = await db.upsert_provider(name="anthropic", framework="claude-cli")
        row = await db.get_provider(pid)
        assert row["api_key"] is None
        assert row["framework"] == "claude-cli"
        await db.delete_provider(pid)
    finally:
        await db.close()


@test("db_providers", "same vendor under both frameworks coexists via UNIQUE(name, framework)")
async def t_dual_framework_provider_rows(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        agno_id = await db.upsert_provider(
            name="anthropic", framework="agno", api_key="sk-ant-live",
        )
        cli_id = await db.upsert_provider(name="anthropic", framework="claude-cli")
        assert agno_id != cli_id
        listed = await db.list_providers()
        pairs = sorted((r["name"], r["framework"]) for r in listed)
        assert pairs == [("anthropic", "agno"), ("anthropic", "claude-cli")]
        await db.delete_provider(agno_id)
        await db.delete_provider(cli_id)
    finally:
        await db.close()


@test("db_providers", "set_provider_enabled flips without touching other fields")
async def t_providers_enable(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        pid = await db.upsert_provider(
            name="anthropic", framework="agno", api_key="sk-ant-test",
        )
        await db.set_provider_enabled(pid, False)
        row = await db.get_provider(pid)
        assert row["enabled"] is False
        assert row["api_key"] == "sk-ant-test"

        only_enabled = await db.list_providers(enabled_only=True)
        assert pid not in {r["id"] for r in only_enabled}
        await db.delete_provider(pid)
    finally:
        await db.close()


@test("db_providers", "delete_provider cascades via FK to wipe models")
async def t_providers_cascade_via_fk(ctx: TestContext) -> None:
    """Deleting a provider row should cascade-delete every model under
    it via ``ON DELETE CASCADE`` on ``models.provider_id``. This replaces
    the old manual ``delete_models_by_provider`` contract."""
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        pid = await db.upsert_provider(
            name="groq", framework="agno", api_key="gk-test",
        )
        mid1 = await db.upsert_model(provider_id=pid, model="llama-3.1-70b")
        mid2 = await db.upsert_model(provider_id=pid, model="llama-3.1-8b")
        assert mid1 != mid2

        # FK cascade: delete the provider → models vanish.
        await db.delete_provider(pid)
        assert await db.get_provider(pid) is None
        assert await db.get_model(mid1) is None
        assert await db.get_model(mid2) is None
    finally:
        await db.close()


@test("db_providers", "registry_status exposes providers_max_updated")
async def t_providers_registry_status(ctx: TestContext) -> None:
    """The gateway's hot-reload probe returns a 4-tuple; the last field
    must bump whenever providers change."""
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        *_, prov_updated = await db.registry_status()
        assert prov_updated == 0.0, "empty table → 0.0"

        pid = await db.upsert_provider(
            name="cerebras", framework="agno", api_key="cb-test",
        )
        *_, prov_updated_after = await db.registry_status()
        assert prov_updated_after > prov_updated

        await db.delete_provider(pid)
    finally:
        await db.close()
