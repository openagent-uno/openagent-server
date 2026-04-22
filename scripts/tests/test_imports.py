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


@test("imports", "groq SDK in deps + agno collected in spec (bundle completeness)")
async def t_bundle_agno_groq(ctx: TestContext) -> None:
    """Verify that the PyInstaller spec collects agno submodules and that the
    groq Python SDK is a declared project dependency.  Both are required so
    ``agno.models.groq`` is importable from the frozen binary; the original
    bug was a per-session ImportError on lyra-virgil whenever a groq model
    was selected."""
    import re

    spec_path = REPO_ROOT / "openagent.spec"
    spec_text = spec_path.read_text()
    assert re.search(r'collect_submodules\("agno"\)', spec_text), \
        "openagent.spec must have collect_submodules(\"agno\") in hiddenimports"
    assert re.search(r'collect_submodules\("groq"\)', spec_text), \
        "openagent.spec must have collect_submodules(\"groq\") in hiddenimports"

    toml_text = (REPO_ROOT / "pyproject.toml").read_text()
    assert re.search(r'"groq[><=!]', toml_text) or re.search(r'"groq"', toml_text), \
        "pyproject.toml must list groq as a dependency"


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
