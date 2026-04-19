"""MCPPool lifecycle + shape tests.

Verifies the pool builds the right set of specs from config, emits the
correct Claude SDK schema, connects + detects dormant servers, and keeps
the ``<server>_<tool>`` namespacing that Agno relies on.

The ``connect_all`` test stashes the live pool into ``ctx.extras["pool"]``
so downstream MCP / agno / router / gateway tests can reuse it.
"""
from __future__ import annotations

from ._framework import TestContext, TestSkip, test


@test("pool", "from_config builds expected number of specs")
async def t_pool_specs(ctx: TestContext) -> None:
    from openagent.mcp.pool import MCPPool
    pool = MCPPool.from_config(
        mcp_config=ctx.config.get("mcp"),
        include_defaults=True,
        disable=["chrome-devtools", "web-search", "computer-control", "mcp-manager", "model-manager"],
        db_path=str(ctx.db_path),
    )
    names = [s.name for s in pool.specs]
    assert "vault" in names
    assert "filesystem" in names
    assert "scheduler" in names
    assert "messaging" in names
    assert "chrome-devtools" not in names
    assert "web-search" not in names


@test("pool", "claude_sdk_servers shape (command/args/env)")
async def t_pool_claude_shape(ctx: TestContext) -> None:
    from openagent.mcp.pool import MCPPool
    pool = MCPPool.from_config(
        mcp_config=ctx.config.get("mcp"),
        include_defaults=True,
        disable=["chrome-devtools", "web-search", "computer-control", "mcp-manager", "model-manager"],
        db_path=str(ctx.db_path),
    )
    sdk = pool.claude_sdk_servers()
    assert sdk, "claude_sdk_servers returned empty"
    for name, entry in sdk.items():
        if "command" in entry:
            assert isinstance(entry["command"], str), f"{name}: command must be str"
            assert isinstance(entry["args"], list), f"{name}: args must be list"
        elif "url" in entry:
            assert entry.get("type") in ("http", "sse"), f"{name}: missing type"


@test("pool", "connect_all + dormant detection + summary")
async def t_pool_connect(ctx: TestContext) -> None:
    from openagent.mcp.pool import MCPPool
    pool = MCPPool.from_config(
        mcp_config=ctx.config.get("mcp"),
        include_defaults=True,
        disable=["chrome-devtools", "web-search", "computer-control", "mcp-manager", "model-manager"],
        db_path=str(ctx.db_path),
    )
    await pool.connect_all()
    try:
        summary = pool.server_summary()
        assert pool.server_count >= 4, f"expected >=4 servers, got {pool.server_count}"
        assert summary.get("vault", 0) > 0, f"vault has no tools: {summary}"
        assert summary.get("scheduler", 0) > 0, f"scheduler has no tools: {summary}"
        # messaging now always exposes the status tool
        assert summary.get("messaging", 0) >= 1
        ctx.extras["pool"] = pool
        ctx.extras["initial_summary"] = summary
    except Exception:
        await pool.close_all()
        raise


@test("pool", "tool name namespacing follows <server>_<tool>")
async def t_pool_namespacing(ctx: TestContext) -> None:
    pool = ctx.extras.get("pool")
    if pool is None:
        raise TestSkip("requires pool fixture")
    seen_prefixes = set()
    for tk in pool.agno_toolkits:
        prefix = getattr(tk, "tool_name_prefix", None)
        if not prefix:
            continue
        seen_prefixes.add(prefix)
        for fname in (getattr(tk, "functions", {}) or {}):
            assert fname.startswith(prefix + "_"), \
                f"function {fname!r} doesn't follow {prefix}_<tool> convention"
    assert "vault" in seen_prefixes
