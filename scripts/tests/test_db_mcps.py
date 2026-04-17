"""MemoryDB — MCP registry CRUD and hot-reload probes.

Covers the new ``mcps`` table added for the db-backed MCP list feature.
Schema idempotency is exercised implicitly: ``_ensure_connected`` runs
``executescript(SCHEMA_SQL)`` on every connect, so repeatedly connecting
against a freshly-created DB must not raise.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("db_mcps", "upsert + list + get roundtrip")
async def t_upsert_roundtrip(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        await db.upsert_mcp(
            "example-mcp",
            kind="custom",
            command=["node", "dist/index.js"],
            args=["--flag"],
            env={"FOO": "bar"},
            enabled=True,
        )
        row = await db.get_mcp("example-mcp")
        assert row is not None, "get_mcp returned None"
        assert row["name"] == "example-mcp"
        assert row["command"] == ["node", "dist/index.js"]
        assert row["args"] == ["--flag"]
        assert row["env"] == {"FOO": "bar"}
        assert row["enabled"] is True

        rows = await db.list_mcps(enabled_only=True)
        assert any(r["name"] == "example-mcp" for r in rows)
        await db.delete_mcp("example-mcp")
    finally:
        # aiosqlite spawns a dedicated thread per Connection — without close
        # the thread lingers for the rest of the suite. Chain that across N
        # tests sharing ``ctx.db_path`` and later tests hit deadlocks when
        # a fresh connect contends with zombie writers.
        await db.close()


@test("db_mcps", "set_mcp_enabled flips enabled without losing other fields")
async def t_set_enabled(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        await db.upsert_mcp("toggle-me", kind="custom", command=["echo"], enabled=True)
        before = await db.get_mcp("toggle-me")
        assert before["enabled"] is True
        await db.set_mcp_enabled("toggle-me", False)
        after = await db.get_mcp("toggle-me")
        assert after["enabled"] is False
        assert after["command"] == ["echo"], "command must survive enable toggle"
        await db.delete_mcp("toggle-me")
    finally:
        await db.close()


@test("db_mcps", "mcps_max_updated is monotonic across writes")
async def t_max_updated_monotonic(ctx: TestContext) -> None:
    import asyncio
    from openagent.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        await db.upsert_mcp("m-a", kind="custom", command=["a"])
        a = await db.mcps_max_updated()
        assert a > 0
        # A tiny sleep — SQLite timestamps are float seconds; two writes in
        # the same microsecond would tie. We don't want to flake.
        await asyncio.sleep(0.01)
        await db.upsert_mcp("m-b", kind="custom", command=["b"])
        b = await db.mcps_max_updated()
        assert b >= a, f"mcps_max_updated regressed: {a} → {b}"
        await db.delete_mcp("m-a")
        await db.delete_mcp("m-b")
    finally:
        await db.close()


@test("db_mcps", "schema creation is idempotent (connect twice)")
async def t_schema_idempotent(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    # A fresh DB path — create, close, reopen. Both ``executescript`` runs
    # must succeed against the same file.
    path = ctx.db_path.with_name("idempotent-mcp-check.db")
    try:
        db1 = MemoryDB(str(path))
        await db1.connect()
        await db1.close()
        db2 = MemoryDB(str(path))
        await db2.connect()
        await db2.close()
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
