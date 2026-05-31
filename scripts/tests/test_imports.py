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
    import src.models.native_provider  # noqa: F401
    import src.models.dispatcher  # noqa: F401
    import src.models.runtime  # noqa: F401
    import src.models.catalog  # noqa: F401
    import src.models.budget  # noqa: F401
    import src.memory.db  # noqa: F401
    assert src.__version__


@test("imports", "groq SDK in deps + src.* collected in spec (bundle completeness)")
async def t_bundle_groq_and_src(ctx: TestContext) -> None:
    """Verify that the PyInstaller spec collects every ``src.*`` submodule
    (including the inlined LLM provider drivers under
    ``src.models.providers``) and that the groq Python SDK is a declared
    project dependency. Both are required so
    ``src.models.providers.groq`` is importable from the frozen binary;
    the original incident was a per-session ImportError on lyra-virgil
    whenever a groq model was selected."""
    import re

    spec_path = REPO_ROOT / "openagent.spec"
    spec_text = spec_path.read_text()
    assert re.search(r'collect_submodules\("src"\)', spec_text), \
        "openagent.spec must have collect_submodules(\"src\") in hiddenimports"
    assert re.search(r'collect_submodules\("groq"\)', spec_text), \
        "openagent.spec must have collect_submodules(\"groq\") in hiddenimports"
    # Defensive check: agno should NOT be collected — the framework has
    # been inlined and the dependency dropped.
    assert not re.search(r'collect_submodules\("agno"\)', spec_text), \
        "openagent.spec must NOT collect agno — the framework was inlined into src/"

    toml_text = (REPO_ROOT / "pyproject.toml").read_text()
    assert re.search(r'"groq[><=!]', toml_text) or re.search(r'"groq"', toml_text), \
        "pyproject.toml must list groq as a dependency"
    assert not re.search(r'^\s*"agno[><=!"]', toml_text, re.MULTILINE), \
        "pyproject.toml must NOT list agno as a dependency"


@test("imports", "frozen BUILTIN_MCPS_DIR matches PyInstaller spec layout")
async def t_frozen_builtin_mcps_dir_matches_spec(ctx: TestContext) -> None:
    """Regression for the openagent→src rename (commit 4b5efb5) where
    ``src/mcp/builtins.py`` kept resolving ``BUILTIN_MCPS_DIR`` as
    ``<bundle>/openagent/mcp/servers`` while ``openagent.spec`` started
    writing the MCP tree to ``<bundle>/src/mcp/servers``. The mismatch
    silently disabled every Python/Node built-in MCP — workflow-manager,
    messaging, scheduler, model-manager, mcp-manager, web-search,
    media-gen, memory-search — because ``resolve_builtin_entry`` raised
    FileNotFoundError on the missing directory. In-process MCPs (shell,
    tool-search) bypass the directory check, masking the regression.

    The two paths must stay in lockstep: this test asserts both that the
    runtime constant points at ``src/mcp/servers`` AND that the spec
    bundles MCPs at that same destination prefix."""
    import importlib
    import re
    import sys
    import tempfile

    spec_text = (REPO_ROOT / "openagent.spec").read_text()
    # The spec's append target must be ``src/mcp/servers/<name>``. If a
    # future refactor changes the bundle layout, this test fails loudly.
    assert re.search(
        r'datas\.append\(\(str\(child\),\s*f?"src/mcp/servers/\{child\.name\}"\)\)',
        spec_text,
    ), (
        "openagent.spec must bundle MCPs at 'src/mcp/servers/<name>'; "
        "if the layout changed, update BUILTIN_MCPS_DIR in src/mcp/builtins.py too"
    )

    with tempfile.TemporaryDirectory() as bundle:
        from pathlib import Path as _Path
        bundle_path = _Path(bundle)
        # Simulate the PyInstaller layout the spec produces.
        (bundle_path / "src" / "mcp" / "servers" / "workflow_manager").mkdir(parents=True)

        sys.modules.pop("src.mcp.builtins", None)
        try:
            with patch("src._frozen.is_frozen", return_value=True), \
                 patch("src._frozen.sys") as sys_mock:
                sys_mock._MEIPASS = str(bundle_path)
                sys_mock.executable = str(bundle_path / "openagent")
                builtins_mod = importlib.import_module("src.mcp.builtins")
                resolved = builtins_mod.BUILTIN_MCPS_DIR
                assert resolved == bundle_path / "src" / "mcp" / "servers", (
                    f"frozen BUILTIN_MCPS_DIR must be <bundle>/src/mcp/servers, "
                    f"got {resolved}"
                )
                assert (resolved / "workflow_manager").exists(), (
                    "the resolved path must actually contain the bundled "
                    "MCP directories — if this fails the runtime check in "
                    "resolve_builtin_entry will raise FileNotFoundError"
                )
        finally:
            sys.modules.pop("src.mcp.builtins", None)
            # Re-import under real-frozen=False so later tests get the
            # dev-layout constants back.
            importlib.import_module("src.mcp.builtins")


@test("imports", "no stale legacy refs (MCPRegistry / MCPTools / tool_factory)")
async def t_no_stale_refs(ctx: TestContext) -> None:
    import re
    for p in (REPO_ROOT / "src").rglob("*.py"):
        s = p.read_text()
        # Skip legitimate runtime MCPTools references — only flag our deleted classes.
        for line in s.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.search(r"openagent\.mcp\.client\b", stripped):
                raise AssertionError(f"stale openagent.mcp.client ref in {p}: {stripped}")
            if re.search(r"openagent\.models\.tool_factory\b", stripped):
                raise AssertionError(f"stale tool_factory ref in {p}: {stripped}")


@test("imports", "frozen preload list covers runtime modules that native_provider lazy-imports")
async def t_frozen_preload_covers_lazy_runtime(ctx: TestContext) -> None:
    """The PyInstaller archive lazy-loads runtime submodules on first use.
    When a sibling service swaps the on-disk binary mid-run, that
    deferred zlib extraction blows up with ``zlib.error: Error -3 …
    incorrect header check`` and propagates as ``agent.run.error``.

    Pin the specific submodules ``native_provider._ensure_team`` (and the
    surrounding hot paths) lazy-import so the preloader keeps them
    resident in ``sys.modules`` and the runtime never has to crack
    the PYZ archive after startup."""
    import src.core.agent as agent_mod

    required = {
        "src.core._runner.agent",
        "src.core._runner.team",
        "src.memory.store.sqlite",
        "src.memory.sessions.agent",
        "src.core._run_state.agent",
        "src.core._run_state.team",
        "src.models.providers.utils",
        "src.mcp._runtime.mcp",
    }
    missing = required - set(agent_mod._FROZEN_RUNTIME_PRELOADS)
    assert not missing, (
        f"runtime modules missing from _FROZEN_RUNTIME_PRELOADS: {sorted(missing)}"
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
