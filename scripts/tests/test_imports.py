"""Import + stale-reference tests.

Makes sure the package imports cleanly and that nothing still points at
the deleted ``openagent.mcp.client`` or ``openagent.models.tool_factory``
modules (both removed during the MCP migration).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from ._framework import TestContext, test

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@test("imports", "all openagent modules import")
async def t_imports(ctx: TestContext) -> None:
    import src
    import src.cli  # noqa: F401
    import src.core.agent  # noqa: F401
    import src.core.server  # noqa: F401
    import src.gateway.server  # noqa: F401
    import src.gateway.sessions  # noqa: F401
    import src.mcp  # noqa: F401
    import src.mcp.pool  # noqa: F401
    import src.mcp.builtins  # noqa: F401
    import src.mcp.servers.scheduler.server  # noqa: F401
    import src.models.agno_provider  # noqa: F401
    import src.models.claude_cli  # noqa: F401
    import src.models.smart_router  # noqa: F401
    import src.models.runtime  # noqa: F401
    import src.models.catalog  # noqa: F401
    import src.models.budget  # noqa: F401
    import src.memory.db  # noqa: F401
    assert src.__version__


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


@test("imports", "frozen preload list covers agno modules that agno_provider lazy-imports")
async def t_frozen_preload_covers_lazy_agno(ctx: TestContext) -> None:
    """The PyInstaller archive lazy-loads agno submodules on first use.
    When a sibling service swaps the on-disk binary mid-run, that
    deferred zlib extraction blows up with ``zlib.error: Error -3 …
    incorrect header check`` and propagates as ``agent.run.error``.

    Pin the specific agno submodules ``agno_provider._ensure_team``
    (and the surrounding hot paths) lazy-import so the preloader keeps
    them resident in ``sys.modules`` and the runtime never has to crack
    the PYZ archive after startup."""
    import src.core.agent as agent_mod

    required = {
        "agno.agent",
        "agno.team",
        "agno.db.sqlite",
        "agno.session.agent",
        "agno.run.agent",
        "agno.run.team",
        "agno.models.utils",
        "agno.tools.mcp",
    }
    missing = required - set(agent_mod._FROZEN_RUNTIME_PRELOADS)
    assert not missing, (
        f"agno modules missing from _FROZEN_RUNTIME_PRELOADS: {sorted(missing)}"
    )

    import importlib
    for module_name in agent_mod._FROZEN_RUNTIME_PRELOADS:
        importlib.import_module(module_name)


@test("imports", "frozen runtime preloader warms late imports without aborting startup")
async def t_frozen_runtime_preload(ctx: TestContext) -> None:
    import src.core.agent as agent_mod

    imported: list[str] = []

    def fake_import(name: str, package=None):
        imported.append(name)
        if name == "test.module.boom":
            raise RuntimeError("boom")
        return object()

    with patch("src._frozen.is_frozen", return_value=True), \
         patch.object(agent_mod, "_FROZEN_RUNTIME_PRELOADS", (
             "test.module.ok",
             "test.module.boom",
             "test.module.tail",
         )), \
         patch("importlib.import_module", side_effect=fake_import), \
         patch.object(agent_mod, "elog") as elog_mock:
        agent_mod._preload_frozen_runtime_modules()

    assert imported == [
        "test.module.ok",
        "test.module.boom",
        "test.module.tail",
    ], imported
    elog_mock.assert_called_once()
