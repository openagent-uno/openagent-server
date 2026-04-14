"""Import + stale-reference tests.

Makes sure the package imports cleanly and that nothing still points at
the deleted ``openagent.mcp.client`` or ``openagent.models.tool_factory``
modules (both removed during the MCP migration).
"""
from __future__ import annotations

from pathlib import Path

from ._framework import TestContext, test

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@test("imports", "all openagent modules import")
async def t_imports(ctx: TestContext) -> None:
    import openagent
    import openagent.cli  # noqa: F401
    import openagent.core.agent  # noqa: F401
    import openagent.core.server  # noqa: F401
    import openagent.gateway.server  # noqa: F401
    import openagent.gateway.sessions  # noqa: F401
    import openagent.mcp  # noqa: F401
    import openagent.mcp.pool  # noqa: F401
    import openagent.mcp.builtins  # noqa: F401
    import openagent.mcp.servers.scheduler.server  # noqa: F401
    import openagent.models.agno_provider  # noqa: F401
    import openagent.models.claude_cli  # noqa: F401
    import openagent.models.smart_router  # noqa: F401
    import openagent.models.runtime  # noqa: F401
    import openagent.models.catalog  # noqa: F401
    import openagent.models.budget  # noqa: F401
    import openagent.memory.db  # noqa: F401
    assert openagent.__version__


@test("imports", "no stale legacy refs (MCPRegistry / MCPTools / tool_factory)")
async def t_no_stale_refs(ctx: TestContext) -> None:
    import re
    for p in (REPO_ROOT / "openagent").rglob("*.py"):
        s = p.read_text()
        # Skip legitimate Agno MCPTools references — only flag our deleted classes.
        for line in s.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.search(r"openagent\.mcp\.client\b", stripped):
                raise AssertionError(f"stale openagent.mcp.client ref in {p}: {stripped}")
            if re.search(r"openagent\.models\.tool_factory\b", stripped):
                raise AssertionError(f"stale tool_factory ref in {p}: {stripped}")
