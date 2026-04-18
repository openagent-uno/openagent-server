"""MemoryDB — Providers table CRUD.

Covers the ``providers`` table — the source of truth for LLM
credentials.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("db_providers", "upsert + list + get roundtrip")
async def t_providers_roundtrip(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        await db.upsert_provider(
            "openai",
            api_key="sk-test-key",
            base_url="https://api.openai.com/v1",
            enabled=True,
            metadata={"tier": "paid"},
        )
        row = await db.get_provider("openai")
        assert row is not None
        assert row["name"] == "openai"
        assert row["api_key"] == "sk-test-key"
        assert row["base_url"] == "https://api.openai.com/v1"
        assert row["enabled"] is True
        assert row["metadata"] == {"tier": "paid"}

        listed = await db.list_providers()
        assert [r["name"] for r in listed] == ["openai"]
        await db.delete_provider("openai")
        assert await db.get_provider("openai") is None
    finally:
        await db.close()


@test("db_providers", "upsert is idempotent and preserves created_at")
async def t_providers_upsert_idempotent(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        await db.upsert_provider("zai", api_key="key-1")
        first = await db.get_provider("zai")
        await db.upsert_provider("zai", api_key="key-2", base_url="https://api.z.ai/api/paas/v4")
        second = await db.get_provider("zai")
        assert second["api_key"] == "key-2"
        assert second["base_url"] == "https://api.z.ai/api/paas/v4"
        assert second["created_at"] == first["created_at"], "upsert must preserve created_at"
        assert second["updated_at"] >= first["updated_at"]
        await db.delete_provider("zai")
    finally:
        await db.close()


@test("db_providers", "set_provider_enabled flips without touching other fields")
async def t_providers_enable(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        await db.upsert_provider("anthropic", api_key="sk-ant-test")
        await db.set_provider_enabled("anthropic", False)
        row = await db.get_provider("anthropic")
        assert row["enabled"] is False
        assert row["api_key"] == "sk-ant-test"

        only_enabled = await db.list_providers(enabled_only=True)
        assert "anthropic" not in {r["name"] for r in only_enabled}
        await db.delete_provider("anthropic")
    finally:
        await db.close()


@test("db_providers", "delete_provider does NOT cascade; caller handles models")
async def t_providers_no_cascade(ctx: TestContext) -> None:
    """``delete_provider`` is the minimal op — leaving cascade decisions
    to the caller (REST handler, MCP tool). The higher layers use
    ``delete_models_by_provider`` explicitly so tests can exercise each
    half independently."""
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        await db.upsert_provider("groq", api_key="gk-test")
        await db.upsert_model("groq:llama", provider="groq", model_id="llama")
        await db.delete_provider("groq")
        # The model row is orphaned — higher-level API cascades.
        assert await db.get_model("groq:llama") is not None
        await db.delete_models_by_provider("groq")
        assert await db.get_model("groq:llama") is None
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

        await db.upsert_provider("cerebras", api_key="cb-test")
        *_, prov_updated_after = await db.registry_status()
        assert prov_updated_after > prov_updated

        await db.delete_provider("cerebras")
    finally:
        await db.close()
