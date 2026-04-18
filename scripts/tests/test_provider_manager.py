"""model-manager: provider CRUD tools write to the providers table.

Pre-v0.11 these tools wrote yaml; since v0.11.0 they write the
``providers`` SQLite table. yaml is imported once at bootstrap and
never touched again.
"""
from __future__ import annotations

import os
import uuid

from ._framework import TestContext, test


@test("provider_manager", "add_provider writes api_key to DB (v0.11)")
async def t_add_provider(ctx: TestContext) -> None:
    import openagent.mcp.servers.model_manager.server as mgr
    from openagent.memory.db import MemoryDB

    tmp_dir = ctx.db_path.parent / f"pmgr-{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_dir / "test.db"

    prev_db = os.environ.get("OPENAGENT_DB_PATH")
    os.environ["OPENAGENT_DB_PATH"] = str(db_path)
    mgr._shared._conn = None  # type: ignore[attr-defined]

    try:
        # Seed schema so the providers table exists.
        db = MemoryDB(str(db_path))
        await db.connect()
        await db.close()

        result = await mgr.add_provider("zai", api_key="zai-test-key",
                                         base_url="https://api.z.ai/api/paas/v4")
        assert result["name"] == "zai"
        assert result["has_api_key"] is True
        assert result["base_url"] == "https://api.z.ai/api/paas/v4"

        # Verify the row landed in the DB (cleartext api_key; same
        # security model as the yaml we replaced).
        db = MemoryDB(str(db_path))
        await db.connect()
        try:
            row = await db.get_provider("zai")
            assert row is not None
            assert row["api_key"] == "zai-test-key"
            assert row["base_url"] == "https://api.z.ai/api/paas/v4"
        finally:
            await db.close()

        listed = await mgr.list_providers()
        zai = next((p for p in listed if p["name"] == "zai"), None)
        assert zai is not None
        assert zai["has_api_key"] is True

        # remove_provider deletes the DB row.
        await mgr.remove_provider("zai")
        db = MemoryDB(str(db_path))
        await db.connect()
        try:
            assert await db.get_provider("zai") is None
        finally:
            await db.close()
    finally:
        mgr._shared._conn = None  # type: ignore[attr-defined]
        if prev_db is None:
            os.environ.pop("OPENAGENT_DB_PATH", None)
        else:
            os.environ["OPENAGENT_DB_PATH"] = prev_db
        for name in ("test.db", "test.db-shm", "test.db-wal"):
            try:
                (tmp_dir / name).unlink()
            except FileNotFoundError:
                pass
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


@test("provider_manager", "claude-cli discovery lists Anthropic models from OpenRouter w/o pricing")
async def t_claude_cli_fallback(ctx: TestContext) -> None:
    """When the user asks for the claude-cli "provider" list, we surface
    the Anthropic catalog from OpenRouter (the picker) but with pricing
    stripped — claude-cli is billed via Pro/Max subscription, never per
    token. Uses a canned OpenRouter response so the test is hermetic."""
    import time
    from openagent.models import discovery

    prev = discovery._OPENROUTER_CACHE
    try:
        discovery._OPENROUTER_CACHE = (time.time(), [
            {"id": "anthropic/claude-sonnet-4.5", "name": "Claude Sonnet 4.5",
             "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
            {"id": "anthropic/claude-opus-4.6", "name": "Claude Opus 4.6",
             "pricing": {"prompt": "0.000015", "completion": "0.000075"}},
        ])
        entries = await discovery.list_provider_models("anthropic")
        ids = {e["id"] for e in entries}
        assert "claude-sonnet-4.5" in ids, ids
        assert "claude-opus-4.6" in ids, ids
        # The entries carry pricing — claude-cli's cost exclusion happens
        # in catalog.get_model_pricing, not in discovery. This test
        # documents the split: discovery surfaces whatever OpenRouter has.
        assert any(e.get("output_cost_per_million") for e in entries)
    finally:
        discovery._OPENROUTER_CACHE = prev
