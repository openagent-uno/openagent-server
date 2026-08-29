from __future__ import annotations

import asyncio
import inspect
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from ._framework import TestContext, TestSkip, test as register_test

try:
    import pytest
except ModuleNotFoundError:
    # The release suite intentionally runs against production dependencies.
    # Keep these contracts directly runnable under pytest when the dev extra is
    # installed, while allowing the canonical decorator runner to import and
    # execute the same functions without making pytest a runtime dependency.
    class _Raises:
        def __init__(self, error_type):
            self.error_type = error_type
            self.value = None

        def __enter__(self):
            return self

        def __exit__(self, error_type, value, _traceback):
            if error_type is None or not issubclass(error_type, self.error_type):
                return False
            self.value = value
            return True

    class _Mark:
        def __getattr__(self, _name):
            def mark(*args, **_kwargs):
                if len(args) == 1 and callable(args[0]):
                    return args[0]

                def decorate(function):
                    return function

                return decorate

            return mark

    class _PytestCompatibility:
        mark = _Mark()

        @staticmethod
        def raises(error_type):
            return _Raises(error_type)

        @staticmethod
        def skip(reason):
            raise TestSkip(reason)

    pytest = _PytestCompatibility()

from openagent_host_tools.builtins import EditorServer, FilesystemServer, ShellServer
from openagent_host_tools.context import current_principal
from openagent_host_tools.sidecars import (
    AGENT_IN_CHROME_MANIFEST,
    COMPUTER_CONTROL_MANIFEST,
)
from openagent_host_tools.types import HostError, tool_error_result
from src.mcp.builtins import (
    DEFAULT_MCPS,
    _profile_marker_claims_port,
    resolve_builtin_entry,
)
from src.mcp.pool import _resolve_specs, _specs_from_db
from src.mcp._runtime import Function
from src.mcp._runtime.function import classification_from_mcp_annotations
from src.mcp.servers.host_tools.adapters import (
    build_editor_runtime_toolkit,
    build_filesystem_runtime_toolkit,
)
from src.mcp.servers.shell.adapters import (
    build_runtime_toolkit as build_shell_runtime_toolkit,
    reset_session_context,
    set_session_context,
)
from src.mcp.servers.tool_search.adapters import (
    _describe_tool_impl,
    _list_tools_impl,
)


def _async_functions(toolkit):
    return {
        name: function.entrypoint
        for name, function in toolkit.async_functions.items()
    }


@pytest.mark.parametrize(
    "relative",
    (Path("host/browser.js"), Path("README.md")),
)
def test_server_agent_in_chrome_assets_match_shared_host_tools(relative: Path):
    """The server adapter and standalone client host consume one source.

    Prefer the sibling source checkout during development; a standalone server
    checkout falls back to package data from the pinned host-tools wheel.
    """

    import openagent_host_tools

    workspace = Path(__file__).resolve().parents[3]
    shared = workspace / "openagent-host-tools" / "sidecars" / "agent-in-chrome"
    if not shared.is_dir():
        shared = (
            Path(openagent_host_tools.__file__).resolve().parent
            / "sidecars"
            / "agent-in-chrome"
        )
    if not shared.is_dir():
        pytest.skip("the pinned host-tools package does not expose sidecar assets")

    server = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "mcp"
        / "servers"
        / "agent-in-chrome"
    )
    assert (server / relative).read_bytes() == (shared / relative).read_bytes(), (
        f"Agent in Chrome asset drifted from openagent-host-tools: {relative}"
    )


@pytest.mark.asyncio
async def test_server_filesystem_uses_shared_core_and_preserves_server_locality(
    tmp_path: Path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    toolkit = build_filesystem_runtime_toolkit()
    functions = _async_functions(toolkit)
    assert set(functions) == {tool.name for tool in FilesystemServer.manifest.tools}
    for manifest in FilesystemServer.manifest.tools:
        runtime = toolkit.async_functions[manifest.name]
        assert runtime.description == manifest.description
        assert runtime.parameters == manifest.input_schema
        assert runtime.classification == manifest.classification.value
    target = tmp_path / "shared.txt"
    written = await functions["write_file"](str(target), "same-core")
    assert written["isError"] is False
    assert "openagent/location" not in written.get("_meta", {})
    read = await functions["read_text_file"](str(target))
    assert read["content"][0]["text"] == "same-core"
    missing = tmp_path / "missing.txt"
    direct = FilesystemServer(tmp_path)
    with pytest.raises(HostError) as error:
        await direct.call("read_text_file", {"path": str(missing)})
    expected = tool_error_result(error.value).to_wire()
    assert await functions["read_text_file"](str(missing)) == expected


@pytest.mark.asyncio
async def test_server_editor_uses_shared_core(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "edit.txt"
    target.write_text("before")
    toolkit = build_editor_runtime_toolkit()
    functions = _async_functions(toolkit)
    assert set(functions) == {tool.name for tool in EditorServer.manifest.tools}
    for manifest in EditorServer.manifest.tools:
        runtime = toolkit.async_functions[manifest.name]
        assert runtime.description == manifest.description
        assert runtime.parameters == manifest.input_schema
        assert runtime.classification == manifest.classification.value
    result = await functions["edit"](str(target), "before", "after")
    assert result["isError"] is False
    assert target.read_text() == "after"
    missing = tmp_path / "missing-edit.txt"
    direct = EditorServer(tmp_path)
    with pytest.raises(HostError) as error:
        await direct.call(
            "edit",
            {"file_path": str(missing), "old_string": "a", "new_string": "b"},
        )
    assert await functions["edit"](str(missing), "a", "b") == tool_error_result(
        error.value
    ).to_wire()


@pytest.mark.asyncio
async def test_server_filesystem_preserves_configured_and_db_roots(
    tmp_path: Path, monkeypatch
):
    allowed = tmp_path / "allowed"
    sibling = tmp_path / "sibling"
    allowed.mkdir()
    sibling.mkdir()
    (allowed / "inside.txt").write_text("inside")
    (sibling / "outside.txt").write_text("outside")
    monkeypatch.chdir(tmp_path)

    configured = _resolve_specs(
        [{"builtin": "filesystem", "args": [str(allowed)]}],
        include_defaults=False,
        disable=None,
        db_path=None,
    )
    assert configured[0].args == [str(allowed)]
    toolkit = build_filesystem_runtime_toolkit(
        args=configured[0].args,
        env=configured[0].env,
    )
    functions = _async_functions(toolkit)
    listed = await functions["list_allowed_directories"]()
    assert listed["structuredContent"] == {
        "unrestricted": False,
        "roots": [str(allowed)],
    }
    assert (await functions["read_text_file"](str(allowed / "inside.txt")))[
        "isError"
    ] is False
    denied = await functions["read_text_file"](str(sibling / "outside.txt"))
    assert denied["isError"] is True
    assert denied["_meta"]["openagent/error"]["code"] == "access_denied"

    class Db:
        async def list_mcps(self, *, enabled_only):
            assert enabled_only is True
            return [
                {
                    "kind": "default",
                    "name": "filesystem",
                    "builtin_name": None,
                    "command": ["npx"],
                    "args": [],
                    "env": {"OPENAGENT_FILESYSTEM_ROOTS": str(allowed)},
                }
            ]

    from_db = await _specs_from_db(Db(), None)
    assert from_db[0].env["OPENAGENT_FILESYSTEM_ROOTS"] == str(allowed)
    env_toolkit = build_filesystem_runtime_toolkit(
        args=from_db[0].args,
        env=from_db[0].env,
    )
    env_listed = await _async_functions(env_toolkit)["list_allowed_directories"]()
    assert env_listed["structuredContent"]["roots"] == [str(allowed)]


def test_default_specs_route_through_shared_in_process_adapters():
    defaults = [entry.get("builtin") for entry in DEFAULT_MCPS]
    assert "filesystem" in defaults
    for name, factory in (
        ("filesystem", "build_filesystem_runtime_toolkit"),
        ("editor", "build_editor_runtime_toolkit"),
    ):
        spec = resolve_builtin_entry(name)
        assert spec["in_process"] is True
        assert spec["adapter_module"] == "src.mcp.servers.host_tools.adapters"
        assert spec["runtime_toolkit_factory"] == factory


@pytest.mark.asyncio
async def test_server_agent_in_chrome_is_network_scoped_and_collision_safe(
    tmp_path: Path,
):
    class Cursor:
        def __init__(self, row):
            self.row = row

        async def fetchone(self):
            return self.row

    class Db:
        def __init__(self, network_id):
            self.network_id = network_id
            self._conn = self

        async def execute(self, _query):
            return Cursor({
                "role": "coordinator",
                "network_id": self.network_id,
                "name": self.network_id,
                "coordinator_node_id": None,
                "coordinator_pubkey": None,
                "created_at": 0,
            })

        async def list_mcps(self, *, enabled_only):
            assert enabled_only is True
            return [{
                "kind": "builtin",
                "name": "agent-in-chrome",
                "builtin_name": "agent-in-chrome",
                "env": {
                    "OPENAGENT_CHROME_PROFILE_DIR": "/unsafe/global",
                    "OPENAGENT_CHROME_CDP_PORT": "18800",
                },
                "args": [],
            }]

    first_db = tmp_path / "first" / "openagent.db"
    second_db = tmp_path / "second" / "openagent.db"
    first_db.parent.mkdir()
    second_db.parent.mkdir()
    first = (await _specs_from_db(Db("network-a"), str(first_db)))[0]
    first_again = (await _specs_from_db(Db("network-a"), str(first_db)))[0]
    second = (await _specs_from_db(Db("network-b"), str(second_db)))[0]

    assert first.env is not None and second.env is not None
    assert first.env["OPENAGENT_NETWORK_ID"] == "network-a"
    assert second.env["OPENAGENT_NETWORK_ID"] == "network-b"
    assert first.env["OPENAGENT_CHROME_PROFILE_DIR"] != second.env[
        "OPENAGENT_CHROME_PROFILE_DIR"
    ]
    assert first.env["OPENAGENT_CHROME_EXTENSIONS_DIR"] != second.env[
        "OPENAGENT_CHROME_EXTENSIONS_DIR"
    ]
    assert first.env["OPENAGENT_CHROME_CDP_PORT"] != second.env[
        "OPENAGENT_CHROME_CDP_PORT"
    ]
    assert first.env["OPENAGENT_CHROME_CDP_PORT"] == first_again.env[
        "OPENAGENT_CHROME_CDP_PORT"
    ]
    assert str(first_db.parent) in first.env["OPENAGENT_CHROME_PROFILE_DIR"]


def test_agent_in_chrome_profile_marker_allows_only_exact_browser_port(
    tmp_path: Path,
):
    profile = tmp_path / "profile"
    profile.mkdir()
    marker = profile / "DevToolsActivePort"
    marker.write_text("19001\n/devtools/browser/openagent-owned\n")
    assert _profile_marker_claims_port(profile, 19001)
    assert not _profile_marker_claims_port(profile, 19002)
    marker.write_text("19001\n/devtools/page/not-a-browser\n")
    assert not _profile_marker_claims_port(profile, 19001)


def test_frozen_macos_computer_control_prefers_nested_signed_helper_app(
    tmp_path: Path,
    monkeypatch,
):
    import src.mcp.builtins as builtins

    executable = tmp_path / "openagent.app" / "Contents" / "MacOS" / "openagent"
    helper = (
        executable.parent.parent
        / "Helpers"
        / "openagent-computer-control.app"
        / "Contents"
        / "MacOS"
        / "openagent-computer-control"
    )
    helper.parent.mkdir(parents=True)
    helper.write_bytes(b"signed helper")
    monkeypatch.setattr(builtins.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(builtins.sys, "executable", str(executable))
    assert builtins._resolve_native_binary("computer-control") == str(helper)


def test_server_tool_search_preserves_classification_for_all_five_builtins():
    from mcp.types import ToolAnnotations

    manifests = (
        FilesystemServer.manifest,
        EditorServer.manifest,
        ShellServer.manifest,
        COMPUTER_CONTROL_MANIFEST,
        AGENT_IN_CHROME_MANIFEST,
    )
    toolkits = {}
    for manifest in manifests:
        functions = {}
        for tool in manifest.tools:
            annotations = ToolAnnotations(
                readOnlyHint=tool.classification.value == "read_only",
                idempotentHint=tool.classification.value
                in {"read_only", "idempotent"},
                destructiveHint=tool.classification.value == "mutating",
            )
            assert (
                classification_from_mcp_annotations(annotations)
                == tool.classification.value
            )
            functions[tool.name] = Function(
                name=tool.name,
                description=tool.description,
                parameters=tool.input_schema,
                classification=classification_from_mcp_annotations(annotations),
            )
        toolkits[manifest.name] = SimpleNamespace(functions=functions)

    class Pool:
        _toolkit_by_name = toolkits

        def toolkit_by_name(self, name):
            return self._toolkit_by_name.get(name)

    pool = Pool()
    for manifest in manifests:
        expected = {
            tool.name: tool.classification.value for tool in manifest.tools
        }
        if manifest.name == "shell":
            assert expected["shell_output"] == "mutating"
        listed = {
            item["name"]: item["classification"]
            for item in _list_tools_impl(pool, manifest.name)
        }
        assert listed == expected
        for tool in manifest.tools:
            described = _describe_tool_impl(pool, manifest.name, tool.name)
            assert described["classification"] == tool.classification.value
            assert described["input_schema"] == tool.input_schema


@pytest.mark.asyncio
async def test_server_shell_uses_shared_exact_contract(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    toolkit = build_shell_runtime_toolkit()
    functions = _async_functions(toolkit)
    assert set(functions) == {tool.name for tool in ShellServer.manifest.tools}
    for manifest in ShellServer.manifest.tools:
        runtime = toolkit.async_functions[manifest.name]
        assert runtime.description == manifest.description
        assert runtime.parameters == manifest.input_schema
        assert runtime.classification == manifest.classification.value

    local = ShellServer(tmp_path)
    principal_token = current_principal.set("contract-account")
    session_token = set_session_context("contract-session")
    try:
        server_result = await functions["shell_exec"](
            "printf stdout; printf stderr >&2", timeout=10_000
        )
        client_result = (
            await local.call(
                "shell_exec",
                {"command": "printf stdout; printf stderr >&2", "timeout": 10_000},
            )
        ).to_wire()
        expected_foreground = {
            "exit_code",
            "signal",
            "stdout",
            "stderr",
            "duration_ms",
            "timed_out",
            "truncated_stdout",
            "truncated_stderr",
        }
        assert set(server_result) == set(client_result) == {
            "content", "structuredContent", "isError"
        }
        assert set(server_result["structuredContent"]) == expected_foreground
        assert set(client_result["structuredContent"]) == expected_foreground
        assert server_result["structuredContent"]["stdout"] == client_result["structuredContent"]["stdout"] == "stdout"
        assert server_result["structuredContent"]["stderr"] == client_result["structuredContent"]["stderr"] == "stderr"
        import json
        assert json.loads(server_result["content"][0]["text"]) == server_result["structuredContent"]
        assert json.loads(client_result["content"][0]["text"]) == client_result["structuredContent"]

        server_which = await functions["shell_which"]("sh")
        client_which = (await local.call("shell_which", {"command": "sh"})).to_wire()
        assert server_which == client_which

        server_error = await functions["shell_which"]("with/path")
        client_error = tool_error_result(
            HostError(
                "invalid_arguments",
                "command must be a bare program name (no path separator)",
            )
        ).to_wire()
        assert server_error == client_error

        for tool, arguments in (
            ("shell_output", {"shell_id": "missing"}),
            ("shell_input", {"shell_id": "missing", "text": "hello"}),
            ("shell_kill", {"shell_id": "missing"}),
        ):
            with pytest.raises(HostError) as local_error:
                await local.call(tool, arguments)
            server_error = await functions[tool](**arguments)
            assert server_error == tool_error_result(local_error.value).to_wire()

        stopped = await functions["shell_exec"]("true", run_in_background=True)
        stopped_id = stopped["structuredContent"]["shell_id"]
        await asyncio.sleep(0.1)
        stopped_input = await functions["shell_input"](stopped_id, "ignored")
        assert stopped_input == tool_error_result(
            HostError("shell_not_running", f"shell {stopped_id} is not running")
        ).to_wire()

        background = (
            await local.call(
                "shell_exec",
                {"command": "sleep 30", "run_in_background": True},
            )
        ).structured_content
        assert set(background) == {"shell_id", "started_at", "description"}
        shell_id = background["shell_id"]
        output = (
            await local.call("shell_output", {"shell_id": shell_id})
        ).structured_content
        assert set(output) == {
            "stdout_delta",
            "stderr_delta",
            "still_running",
            "exit_code",
            "signal",
            "stdout_bytes_total",
            "stderr_bytes_total",
            "truncated_stdout",
            "truncated_stderr",
        }
        listing = (
            await local.call("shell_list", {"session_id": "ignored-for-auth"})
        ).structured_content
        assert set(listing["shells"][0]) == {
            "shell_id",
            "command",
            "state",
            "started_at",
            "runtime_ms",
            "stdout_bytes",
            "stderr_bytes",
            "exit_code",
            "session_id",
        }
        killed = (
            await local.call("shell_kill", {"shell_id": shell_id})
        ).structured_content
        assert set(killed) == {"killed", "exit_code", "signal"}

        interactive = (
            await local.call(
                "shell_exec",
                {"command": "read value", "run_in_background": True},
            )
        ).structured_content
        written = (
            await local.call(
                "shell_input",
                {"shell_id": interactive["shell_id"], "text": "done"},
            )
        ).structured_content
        assert set(written) == {"bytes_written"}
    finally:
        reset_session_context(session_token)
        current_principal.reset(principal_token)
        await local.close()


class _CanonicalMonkeyPatch:
    """Small reversible fixture for the repository's canonical test runner."""

    def __init__(self):
        self._undo = []

    def chdir(self, path: Path) -> None:
        previous = Path.cwd()
        os.chdir(path)
        self._undo.append(lambda: os.chdir(previous))

    def setattr(self, target, name: str, value) -> None:
        previous = getattr(target, name)
        setattr(target, name, value)
        self._undo.append(lambda: setattr(target, name, previous))

    def undo(self) -> None:
        for restore in reversed(self._undo):
            restore()
        self._undo.clear()


async def _run_with_temp_path(
    ctx: TestContext,
    function,
    *,
    monkeypatch: bool = False,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix="shared-host-tools-", dir=ctx.test_dir
    ) as temporary:
        patcher = _CanonicalMonkeyPatch()
        try:
            # macOS exposes /tmp through a symlink to /private/tmp; the host
            # cores deliberately canonicalize paths, so fixtures must do the
            # same before asserting exact configured roots/helper locations.
            arguments = [Path(temporary).resolve()]
            if monkeypatch:
                arguments.append(patcher)
            result = function(*arguments)
            if inspect.isawaitable(result):
                await result
        finally:
            patcher.undo()


@register_test("host-tools", "server and client Agent in Chrome assets are identical")
async def _registered_agent_in_chrome_assets(_ctx: TestContext) -> None:
    for relative in (Path("host/browser.js"), Path("README.md")):
        test_server_agent_in_chrome_assets_match_shared_host_tools(relative)


@register_test("host-tools", "filesystem shared core preserves server locality")
async def _registered_filesystem_core(ctx: TestContext) -> None:
    await _run_with_temp_path(
        ctx,
        test_server_filesystem_uses_shared_core_and_preserves_server_locality,
        monkeypatch=True,
    )


@register_test("host-tools", "editor shared core contract matches standalone host")
async def _registered_editor_core(ctx: TestContext) -> None:
    await _run_with_temp_path(
        ctx, test_server_editor_uses_shared_core, monkeypatch=True
    )


@register_test("host-tools", "filesystem preserves configured and database roots")
async def _registered_filesystem_roots(ctx: TestContext) -> None:
    await _run_with_temp_path(
        ctx,
        test_server_filesystem_preserves_configured_and_db_roots,
        monkeypatch=True,
    )


@register_test("host-tools", "default MCP specs use shared in-process adapters")
async def _registered_default_adapters(_ctx: TestContext) -> None:
    test_default_specs_route_through_shared_in_process_adapters()


@register_test("host-tools", "Agent in Chrome profile and port are network scoped")
async def _registered_agent_in_chrome_scope(ctx: TestContext) -> None:
    await _run_with_temp_path(
        ctx, test_server_agent_in_chrome_is_network_scoped_and_collision_safe
    )


@register_test("host-tools", "Agent in Chrome marker owns only its exact port")
async def _registered_agent_in_chrome_marker(ctx: TestContext) -> None:
    await _run_with_temp_path(
        ctx, test_agent_in_chrome_profile_marker_allows_only_exact_browser_port
    )


@register_test("host-tools", "frozen macOS server selects signed computer helper")
async def _registered_signed_computer_helper(ctx: TestContext) -> None:
    await _run_with_temp_path(
        ctx,
        test_frozen_macos_computer_control_prefers_nested_signed_helper_app,
        monkeypatch=True,
    )


@register_test("host-tools", "tool-search preserves all host tool classifications")
async def _registered_tool_classification(_ctx: TestContext) -> None:
    test_server_tool_search_preserves_classification_for_all_five_builtins()


@register_test("host-tools", "shell shared core matches the standalone contract")
async def _registered_shell_contract(ctx: TestContext) -> None:
    await _run_with_temp_path(
        ctx, test_server_shell_uses_shared_exact_contract, monkeypatch=True
    )
