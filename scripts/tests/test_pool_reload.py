"""MCPPool.from_db + reload — hot-reload without a restart.

The pool-reload feature lets the mcp-manager MCP (and the REST endpoint)
add/remove MCP servers without stopping the process. This test drives
the from_db + reload path directly against a throwaway DB.
"""
from __future__ import annotations

from ._framework import TestContext, TestSkip, test


@test("pool_reload", "from_db builds specs from the mcps table")
async def t_from_db(ctx: TestContext) -> None:
    import uuid
    from openagent.memory.db import MemoryDB
    from openagent.mcp.pool import MCPPool

    tmp_db = ctx.db_path.with_name(f"pool-fromdb-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        await db.upsert_mcp("filesystem", kind="custom",
                            command=["npx", "-y", "@modelcontextprotocol/server-filesystem"],
                            args=["/tmp"], enabled=True)
        pool = await MCPPool.from_db(db, db_path=str(tmp_db))
        names = [s.name for s in pool.specs]
        assert "filesystem" in names, f"expected filesystem in specs, got {names}"
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("pool_reload", "reload swaps specs in place without a process restart")
async def t_reload_swaps(ctx: TestContext) -> None:
    import uuid
    from openagent.memory.db import MemoryDB
    from openagent.mcp.pool import MCPPool

    tmp_db = ctx.db_path.with_name(f"pool-reload-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        await db.upsert_mcp("alpha", kind="custom", command=["/bin/true"], enabled=True)
        pool = await MCPPool.from_db(db, db_path=str(tmp_db))
        assert [s.name for s in pool.specs] == ["alpha"]

        await db.upsert_mcp("beta", kind="custom", command=["/bin/true"], enabled=True)
        await pool.reload()
        names = sorted(s.name for s in pool.specs)
        assert names == ["alpha", "beta"], f"reload did not pick up new spec: {names}"

        await db.set_mcp_enabled("alpha", False)
        await pool.reload()
        names2 = [s.name for s in pool.specs]
        assert names2 == ["beta"], f"disabled row still in pool: {names2}"
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("pool_reload", "reload is a no-op when the pool was built from_config (tests path)")
async def t_reload_noop_from_config(ctx: TestContext) -> None:
    from openagent.mcp.pool import MCPPool

    pool = MCPPool.from_config(mcp_config=[], include_defaults=False, disable=[], db_path=None)
    # Should not raise — _db is None on the from_config path.
    await pool.reload()
