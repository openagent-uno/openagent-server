"""Yaml → DB bootstrap — idempotency for MCP imports.

The bootstrap copies yaml ``mcp:`` entries into the DB on first boot
so existing configs migrate transparently. Re-running is a no-op
guarded by a ``config_state`` flag.
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
