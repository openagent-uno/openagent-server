"""Yaml → DB bootstrap — idempotency + model reference discovery.

The bootstrap copies yaml ``mcp:`` entries and per-provider ``models:``
into the DB on first boot so existing configs migrate transparently.
Re-running is a no-op guarded by ``config_state`` flags.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("bootstrap", "import_yaml_mcps_once writes rows then short-circuits")
async def t_mcps_bootstrap_idempotent(ctx: TestContext) -> None:
    import uuid
    from openagent.memory.db import MemoryDB
    from openagent.memory.bootstrap import import_yaml_mcps_once

    tmp_db = ctx.db_path.with_name(f"bootstrap-mcps-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()

        mcp_config = [{"name": "custom-echo", "command": ["echo", "hi"]}]
        first = await import_yaml_mcps_once(db, mcp_config, include_defaults=False, disable=[])
        assert first is True, "first import must write"
        rows = await db.list_mcps()
        assert any(r["name"] == "custom-echo" for r in rows)

        second = await import_yaml_mcps_once(db, mcp_config, include_defaults=False, disable=[])
        assert second is False, "second import must short-circuit"

        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("bootstrap", "sibling model.provider resolves bare model_id")
async def t_model_id_uses_sibling_provider(ctx: TestContext) -> None:
    """Regression: a yaml with provider=claude-cli + bare model_id=<id>
    used to fail resolution because the bootstrap ignored the sibling
    provider and tried to guess from the pricing table. Observed on
    mixout-agent: ``cannot resolve bare model ref 'claude-sonnet-4-6'``.
    """
    import uuid
    from openagent.memory.db import MemoryDB
    from openagent.memory.bootstrap import import_yaml_models_once

    tmp_db = ctx.db_path.with_name(f"bootstrap-sibling-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        await import_yaml_models_once(
            db,
            providers_config={},  # no providers.X.models — fallback path
            model_cfg={"provider": "claude-cli", "model_id": "claude-sonnet-4-6"},
        )
        runtime_ids = {r["runtime_id"] for r in await db.list_models()}
        # Canonical v0.10 form: ``claude-cli:<provider>:<model>``.
        assert "claude-cli:anthropic:claude-sonnet-4-6" in runtime_ids, runtime_ids
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("bootstrap", "import_yaml_models_once pulls routing-only refs")
async def t_models_bootstrap_routing(ctx: TestContext) -> None:
    import uuid
    from openagent.memory.db import MemoryDB
    from openagent.memory.bootstrap import import_yaml_models_once

    tmp_db = ctx.db_path.with_name(f"bootstrap-models-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()

        # No providers.X.models — the rejection gate would trip without
        # bootstrap's model_cfg fallback. This test is the regression
        # guard.
        providers = {"openai": {"api_key": "sk-test"}}
        model_cfg = {
            "provider": "smart",
            "routing": {"simple": "gpt-4o-mini", "medium": "gpt-4.1-mini", "hard": "gpt-4.1"},
            "classifier_model": "gpt-4o-mini",
        }
        wrote = await import_yaml_models_once(db, providers, model_cfg=model_cfg)
        assert wrote is True

        rows = await db.list_models(enabled_only=True)
        runtime_ids = {r["runtime_id"] for r in rows}
        assert "openai:gpt-4o-mini" in runtime_ids, runtime_ids
        assert "openai:gpt-4.1" in runtime_ids
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass
