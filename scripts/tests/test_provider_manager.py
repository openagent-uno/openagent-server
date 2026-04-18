"""model-manager: provider CRUD tools mutate the yaml, model rows stay in DB."""
from __future__ import annotations

import os
import uuid

from ._framework import TestContext, test


@test("provider_manager", "add_provider writes api_key to yaml")
async def t_add_provider(ctx: TestContext) -> None:
    import yaml
    import openagent.mcp.servers.model_manager.server as mgr
    from openagent.memory.db import MemoryDB

    tmp_dir = ctx.db_path.parent / f"pmgr-{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_dir / "openagent.yaml"
    db_path = tmp_dir / "test.db"

    # Seed a minimal yaml so load_config has something to return.
    cfg_path.write_text(yaml.safe_dump({"name": "test", "providers": {}}))

    prev_cfg = os.environ.get("OPENAGENT_CONFIG_PATH")
    prev_db = os.environ.get("OPENAGENT_DB_PATH")
    os.environ["OPENAGENT_CONFIG_PATH"] = str(cfg_path)
    os.environ["OPENAGENT_DB_PATH"] = str(db_path)
    mgr._shared._conn = None  # type: ignore[attr-defined]

    try:
        # Need a DB for the list_providers model-count query.
        db = MemoryDB(str(db_path))
        await db.connect()
        await db.close()

        result = await mgr.add_provider("zai", api_key="zai-test-key",
                                         base_url="https://api.z.ai/api/paas/v4")
        assert result["name"] == "zai"
        assert result["has_api_key"] is True
        assert result["base_url"] == "https://api.z.ai/api/paas/v4"
        # Verify yaml on disk was updated.
        raw = yaml.safe_load(cfg_path.read_text())
        assert raw["providers"]["zai"]["api_key"] == "zai-test-key"

        # list_providers now sees it.
        listed = await mgr.list_providers()
        zai = next((p for p in listed if p["name"] == "zai"), None)
        assert zai is not None
        assert zai["has_api_key"] is True

        # remove_provider pulls it back out.
        await mgr.remove_provider("zai")
        raw = yaml.safe_load(cfg_path.read_text())
        assert "zai" not in (raw.get("providers") or {})
    finally:
        mgr._shared._conn = None  # type: ignore[attr-defined]
        if prev_cfg is None:
            os.environ.pop("OPENAGENT_CONFIG_PATH", None)
        else:
            os.environ["OPENAGENT_CONFIG_PATH"] = prev_cfg
        if prev_db is None:
            os.environ.pop("OPENAGENT_DB_PATH", None)
        else:
            os.environ["OPENAGENT_DB_PATH"] = prev_db
        try:
            cfg_path.unlink()
        except FileNotFoundError:
            pass
        try:
            db_path.unlink()
        except FileNotFoundError:
            pass
        try:
            (tmp_dir / "openagent.db-shm").unlink()
        except FileNotFoundError:
            pass
        try:
            (tmp_dir / "openagent.db-wal").unlink()
        except FileNotFoundError:
            pass
        try:
            tmp_dir.rmdir()
        except OSError:
            pass


@test("provider_manager", "bundled fallback for claude-cli returns anthropic models")
async def t_claude_cli_fallback(ctx: TestContext) -> None:
    from openagent.models.discovery import _bundled_fallback

    entries = _bundled_fallback("claude-cli")
    assert entries, "claude-cli fallback must not be empty"
    ids = {e["id"] for e in entries}
    assert any(mid.startswith("claude-sonnet-") for mid in ids), ids
    # Prices should be surfaced (they come from the anthropic: rows).
    assert any(e.get("output_cost_per_million") for e in entries)
