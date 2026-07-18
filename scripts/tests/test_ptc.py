"""Programmatic Tool Calling (PTC) — the ``run_python`` tool.

Like ``test_sandbox`` / ``test_dry_run``, these drive the REAL seams rather than
asserting things in isolation: the RPC bridge is exercised over an actual Unix
socket, and the script round-trip actually spawns a child through
``BackgroundShell`` (the same funnel the shell tool uses). The load-bearing test
is the disabled-by-default one — with ``ptc.enabled`` unset the gating seeds
nothing and the framework prompt is byte-identical to a build without PTC.

No live LLM, no docker daemon: the local exec backend runs the child on the host
(``require_sandbox=False``), and a fake pool stands in for the MCP grant.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile

from ._framework import TestContext, test


# ── Fakes: a minimal pool exposing a single in-process tool ─────────


class _FakeToolkit:
    """Duck-typed runtime toolkit — ``_call_tool_impl`` reads ``functions`` /
    ``async_functions`` and calls ``fn.entrypoint or fn``."""

    def __init__(self, functions: dict) -> None:
        self.functions = functions
        self.async_functions: dict = {}


class _FakePool:
    """Duck-typed ``MCPPool`` — only the two members ``_call_tool_impl`` touches."""

    def __init__(self, toolkits: dict) -> None:
        self._toolkit_by_name = toolkits

    def toolkit_by_name(self, name: str):
        return self._toolkit_by_name.get(name)


def _reset_backend() -> None:
    """Drop the memoized exec backend so PTC re-reads the (default: local) env.

    A prior module (test_sandbox) may have left docker selected; without this
    the local path here would misroute. Mirrors test_sandbox's own discipline.
    """
    from src.mcp.servers.shell import backends
    backends._reset_backend_for_tests()


def _local_settings(**over):
    from src.core.config import PtcSettings
    base = dict(enabled=True, require_sandbox=False, timeout_s=30)
    base.update(over)
    return PtcSettings(**base)


# ── Fake docker backend: exercises the file transport without a daemon ──


class _FakePtcDocker:
    """A daemon-free stand-in for ``DockerBackend`` for the PTC docker path.

    Its "container filesystem" is a host tmpdir rooted at ``container_workdir``;
    the container-fs helpers operate on it directly, and ``build_spawn`` runs the
    shipped script LOCALLY (swapping the container's ``python3`` for the test
    interpreter, so the round-trip doesn't depend on a system python3) with the
    per-command env merged in. That makes the file-transport RPC round-trip
    testable end to end without docker. Constructed by ``select_backend`` with a
    ``DockerConfig`` (same contract as the real backend).
    """

    name = "docker"

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg
        self._root = None
        self.prepare_called = False
        self.cleanup_called = False

    async def prepare(self) -> None:
        self.prepare_called = True
        if self._root is None:
            self._root = tempfile.mkdtemp(prefix="oa-ptc-fakedock-")

    @property
    def container_workdir(self) -> str:
        return self._root

    async def container_mkdir(self, path) -> None:
        os.makedirs(path, exist_ok=True)

    async def container_write(self, path, data) -> None:
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)  # atomic — mirrors the real base64+mv

    async def container_read(self, path):
        try:
            with open(path, "rb") as fh:
                return fh.read()
        except OSError:
            return None

    async def container_listdir(self, path):
        try:
            return os.listdir(path)
        except OSError:
            return []

    async def container_rmtree(self, path) -> None:
        shutil.rmtree(path, ignore_errors=True)

    def build_spawn(self, *, command, cwd, env):
        import sys
        from src.mcp.servers.shell.backends import SpawnSpec
        from src.mcp.servers.shell.shells import _pick_shell

        if command.startswith("python3 "):
            command = sys.executable + command[len("python3"):]
        shell, flag = _pick_shell()
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)
        return SpawnSpec(
            argv=[shell, flag, command], env=proc_env, cwd=cwd, start_new_session=True
        )

    async def cleanup(self) -> None:
        self.cleanup_called = True
        if self._root:
            shutil.rmtree(self._root, ignore_errors=True)


# ── 1. RPC server proxies a tool call, args intact ──────────────────


@test("ptc", "rpc server proxies a call through _call_tool_impl with args intact")
async def t_rpc_proxy(_ctx: TestContext) -> None:
    from src.mcp.servers.ptc import handlers

    seen: dict = {}

    def echo(**kwargs):
        seen["args"] = kwargs
        return {"echoed": kwargs, "marker": "PTC_RPC_OK"}

    pool = _FakePool({"fake": _FakeToolkit({"echo": echo})})
    tmp = tempfile.mkdtemp(prefix="oa-ptc-t1-")
    sock = os.path.join(tmp, "rpc.sock")
    token = "token-abc"
    server, state = await handlers.start_rpc_server(pool=pool, sock_path=sock, token=token)
    try:
        # 0600 on the socket — no other user can talk to the bridge.
        assert (os.stat(sock).st_mode & 0o777) == 0o600, oct(os.stat(sock).st_mode)

        reader, writer = await asyncio.open_unix_connection(sock)
        req = {"token": token, "server": "fake", "tool": "echo", "args": {"x": 1, "y": "z"}}
        writer.write((json.dumps(req) + "\n").encode())
        await writer.drain()
        resp = json.loads(await reader.readline())
        writer.close()

        assert resp["ok"] is True, resp
        assert resp["result"] == {"echoed": {"x": 1, "y": "z"}, "marker": "PTC_RPC_OK"}, resp
        assert seen["args"] == {"x": 1, "y": "z"}, seen  # args arrived intact
        assert state["calls"] == 1
    finally:
        server.close()
        await server.wait_closed()
        shutil.rmtree(tmp, ignore_errors=True)


# ── 2. Script round-trip: bare call_tool works, only stdout returns ─


@test("ptc", "run_python round-trips: bare call_tool resolves, stdout carries the result")
async def t_script_round_trip(_ctx: TestContext) -> None:
    from src.mcp.servers.ptc import handlers

    _reset_backend()

    def echo(**kwargs):
        return {"sentinel": "PTC_ROUNDTRIP_OK", "x": kwargs.get("x")}

    pool = _FakePool({"fake": _FakeToolkit({"echo": echo})})
    out = await handlers.run_python_impl(
        'print(call_tool("fake", "echo", {"x": 42}))',
        pool=pool,
        settings=_local_settings(),
        dry_run=False,
    )
    assert out["status"] == "ok", out
    assert out["tool_calls_made"] == 1, out
    # The fake result reaches the model ONLY via stdout — no intermediate leak.
    assert "PTC_ROUNDTRIP_OK" in out["output"], out
    assert "'x': 42" in out["output"], out


# ── 3. Disabled by default → byte-identical (the load-bearing test) ─


@test("ptc", "disabled by default: no gated entry, note empty, prompt byte-identical")
async def t_disabled_by_default(_ctx: TestContext) -> None:
    from src.mcp.builtins import config_gated_mcp_entries, DEFAULT_MCPS
    from src.core.config import ptc_settings
    from src.core.prompts import build_ptc_note, FRAMEWORK_SYSTEM_PROMPT

    # Unset config → OFF, and never in the unconditional default seed set.
    assert ptc_settings({}).enabled is False
    assert all(e.get("builtin") != "ptc" for e in DEFAULT_MCPS)
    assert config_gated_mcp_entries({}) == []
    assert config_gated_mcp_entries({"ptc": {"enabled": True}}) == [
        {"builtin": "ptc", "_default": True}
    ]

    # The note renders "" when disabled. Because the placeholder sits flush
    # against the next header, an empty render leaves the framework prompt
    # BYTE-IDENTICAL to the template with PTC removed entirely.
    assert build_ptc_note(False) == ""
    baseline = FRAMEWORK_SYSTEM_PROMPT.replace("{{PTC_NOTE}}", "")
    with_ptc_off = FRAMEWORK_SYSTEM_PROMPT.replace("{{PTC_NOTE}}", build_ptc_note(False))
    assert with_ptc_off == baseline, "disabled PTC must render byte-identical to no-PTC"
    assert "Programmatic tool calling" not in with_ptc_off

    # And enabled is genuinely distinct (so the test above isn't vacuous).
    assert "run_python" in build_ptc_note(True)


# ── 4. Output cap: a >50k print comes back truncated ────────────────


@test("ptc", "oversized stdout is truncated with the cap marker")
async def t_output_cap(_ctx: TestContext) -> None:
    from src.mcp.servers.ptc import handlers
    from src.core.tool_output import max_tool_result_chars

    _reset_backend()
    limit = max_tool_result_chars()
    out = await handlers.run_python_impl(
        f'print("A" * {limit * 3})',
        pool=_FakePool({}),
        settings=_local_settings(),
        dry_run=False,
    )
    assert out["status"] == "ok", out
    assert "truncated by OpenAgent" in out["output"], out["output"][:200]
    assert len(out["output"]) <= limit + 500, len(out["output"])


# ── 5. dry_run stamping propagates to a proxied write ───────────────


@test("ptc", "dry_run captured at entry stamps call_meta on proxied tool calls")
async def t_dry_run_stamping(_ctx: TestContext) -> None:
    from src.mcp.servers.ptc import handlers
    from src.core.dry_run import call_meta, dry_run_scope, is_dry_run

    _reset_backend()
    seen: dict = {}

    def write_thing(**kwargs):
        seen["meta"] = call_meta()  # a write tool would key its capture off this
        return "written"

    pool = _FakePool({"fake": _FakeToolkit({"write_thing": write_thing})})

    # Capture at entry, exactly as the adapter does (is_dry_run() inside scope).
    with dry_run_scope(True):
        captured = is_dry_run()
    assert captured is True

    out = await handlers.run_python_impl(
        'print(call_tool("fake", "write_thing", {}))',
        pool=pool,
        settings=_local_settings(),
        dry_run=captured,
    )
    assert out["status"] == "ok", out
    assert seen["meta"] == {"dry_run": True}, seen

    # And a live run stamps nothing (identical to not passing meta).
    seen.clear()
    out2 = await handlers.run_python_impl(
        'print(call_tool("fake", "write_thing", {}))',
        pool=pool,
        settings=_local_settings(),
        dry_run=False,
    )
    assert out2["status"] == "ok", out2
    assert seen["meta"] is None, seen


# ── 6. Auth: wrong / absent token is rejected, uncounted ────────────


@test("ptc", "rpc server rejects a wrong or absent token and does not dispatch")
async def t_auth(_ctx: TestContext) -> None:
    from src.mcp.servers.ptc import handlers

    called = {"n": 0}

    def echo(**kwargs):
        called["n"] += 1
        return "should-not-run"

    pool = _FakePool({"fake": _FakeToolkit({"echo": echo})})
    tmp = tempfile.mkdtemp(prefix="oa-ptc-t6-")
    sock = os.path.join(tmp, "rpc.sock")
    server, state = await handlers.start_rpc_server(pool=pool, sock_path=sock, token="the-real-token")
    try:
        # Wrong token.
        reader, writer = await asyncio.open_unix_connection(sock)
        writer.write((json.dumps({"token": "WRONG", "server": "fake", "tool": "echo", "args": {}}) + "\n").encode())
        await writer.drain()
        resp = json.loads(await reader.readline())
        writer.close()
        assert resp["ok"] is False and resp["error"] == "unauthorized", resp

        # Absent token.
        reader2, writer2 = await asyncio.open_unix_connection(sock)
        writer2.write((json.dumps({"server": "fake", "tool": "echo", "args": {}}) + "\n").encode())
        await writer2.drain()
        resp2 = json.loads(await reader2.readline())
        writer2.close()
        assert resp2["ok"] is False and resp2["error"] == "unauthorized", resp2

        # Neither reached dispatch: the tool never ran and the counter is clean.
        assert called["n"] == 0
        assert state["calls"] == 0
    finally:
        server.close()
        await server.wait_closed()
        shutil.rmtree(tmp, ignore_errors=True)


# ── 7. Bonus: require_sandbox / docker fail closed (never run on host) ─


@test("ptc", "require_sandbox and the docker backend both fail closed (no host run)")
async def t_fail_closed(_ctx: TestContext) -> None:
    from src.mcp.servers.ptc import handlers
    from src.mcp.servers.shell import backends

    ran = {"n": 0}

    def tripwire(**kwargs):
        ran["n"] += 1
        return "ran"

    pool = _FakePool({"fake": _FakeToolkit({"tripwire": tripwire})})

    # require_sandbox default True + local backend → refused.
    _reset_backend()
    out = await handlers.run_python_impl(
        'print(call_tool("fake", "tripwire", {}))',
        pool=pool,
        settings=_local_settings(require_sandbox=True),
        dry_run=False,
    )
    assert out["status"] == "refused", out
    assert out["tool_calls_made"] == 0

    # docker backend whose prepare() raises → refused, never on host. Fake the
    # daemon-bring-up failure so this stays hermetic (no real docker required).
    class _PrepFails(_FakePtcDocker):
        async def prepare(self):
            from src.mcp.servers.shell.backends import SandboxUnavailableError
            raise SandboxUnavailableError("no docker daemon in this test")

    prev = os.environ.get("OPENAGENT_SANDBOX_BACKEND")
    orig = backends._BACKENDS["docker"]
    backends._BACKENDS["docker"] = _PrepFails
    try:
        os.environ["OPENAGENT_SANDBOX_BACKEND"] = "docker"
        os.environ["OPENAGENT_SANDBOX_DOCKER"] = "{}"
        backends._reset_backend_for_tests()
        out2 = await handlers.run_python_impl(
            'print(call_tool("fake", "tripwire", {}))',
            pool=pool,
            settings=_local_settings(require_sandbox=False),
            dry_run=False,
        )
        assert out2["status"] == "refused", out2
        assert "docker" in out2["output"], out2
        assert out2["tool_calls_made"] == 0, out2
    finally:
        backends._BACKENDS["docker"] = orig
        if prev is None:
            os.environ.pop("OPENAGENT_SANDBOX_BACKEND", None)
        else:
            os.environ["OPENAGENT_SANDBOX_BACKEND"] = prev
        os.environ.pop("OPENAGENT_SANDBOX_DOCKER", None)
        backends._reset_backend_for_tests()

    # In neither refusal did the fake tool run.
    assert ran["n"] == 0


# ── 8. Docker path: file-transport RPC round-trip (faked docker) ────


@test("ptc", "docker path: file-transport RPC round-trips through _call_tool_impl (fake docker)")
async def t_docker_round_trip(_ctx: TestContext) -> None:
    from src.mcp.servers.ptc import handlers
    from src.mcp.servers.shell import backends

    seen: dict = {}

    def echo(**kwargs):
        seen["args"] = kwargs
        return {"sentinel": "PTC_DOCKER_OK", "x": kwargs.get("x")}

    pool = _FakePool({"fake": _FakeToolkit({"echo": echo})})

    prev = os.environ.get("OPENAGENT_SANDBOX_BACKEND")
    orig = backends._BACKENDS["docker"]
    backends._BACKENDS["docker"] = _FakePtcDocker
    try:
        os.environ["OPENAGENT_SANDBOX_BACKEND"] = "docker"
        os.environ["OPENAGENT_SANDBOX_DOCKER"] = "{}"
        backends._reset_backend_for_tests()

        # require_sandbox=True is SATISFIED by docker → the docker path runs.
        out = await handlers.run_python_impl(
            'print(call_tool("fake", "echo", {"x": 7}))',
            pool=pool,
            settings=_local_settings(require_sandbox=True),
            dry_run=False,
        )
        assert out["status"] == "ok", out
        assert out["tool_calls_made"] == 1, out
        # The fake result reaches the model ONLY via the script's stdout.
        assert "PTC_DOCKER_OK" in out["output"], out
        assert "'x': 7" in out["output"], out
        assert seen["args"] == {"x": 7}, seen  # args survived the file transport
        # It genuinely routed through the docker backend, not the local one.
        inst = backends.get_exec_backend()
        assert isinstance(inst, _FakePtcDocker) and inst.prepare_called, inst
    finally:
        backends._BACKENDS["docker"] = orig
        if prev is None:
            os.environ.pop("OPENAGENT_SANDBOX_BACKEND", None)
        else:
            os.environ["OPENAGENT_SANDBOX_BACKEND"] = prev
        os.environ.pop("OPENAGENT_SANDBOX_DOCKER", None)
        backends._reset_backend_for_tests()


# ── 9. Docker path: wrong token on the file transport is not dispatched ─


@test("ptc", "docker path: a wrong-token request file is rejected, uncounted (file transport)")
async def t_docker_auth(_ctx: TestContext) -> None:
    from src.mcp.servers.ptc import handlers

    ran = {"n": 0}

    def tool_fn(**kwargs):
        ran["n"] += 1
        return "ran-ok"

    pool = _FakePool({"fake": _FakeToolkit({"tool_fn": tool_fn})})

    backend = _FakePtcDocker()
    await backend.prepare()
    rundir = os.path.join(backend.container_workdir, "run")
    await backend.container_mkdir(rundir)

    state: dict = {"calls": 0}
    dispatch = handlers._make_file_dispatch(
        pool=pool, token="the-real-token", dry_run=False,
        allowed_tools=None, max_tool_calls=50, state=state,
    )
    poller = asyncio.create_task(handlers._poll_rundir(backend, rundir, dispatch))

    async def _wait_resp(seq):
        for _ in range(300):
            raw = await backend.container_read(os.path.join(rundir, f"resp_{seq}.json"))
            if raw is not None:
                return json.loads(raw)
            await asyncio.sleep(0.01)
        raise AssertionError(f"no resp_{seq}.json produced")

    try:
        # A wrong-token request, then a correct one.
        await backend.container_write(
            os.path.join(rundir, "req_1.json"),
            json.dumps({"token": "WRONG", "server": "fake", "tool": "tool_fn", "args": {}, "seq": 1}).encode(),
        )
        await backend.container_write(
            os.path.join(rundir, "req_2.json"),
            json.dumps({"token": "the-real-token", "server": "fake", "tool": "tool_fn", "args": {}, "seq": 2}).encode(),
        )
        r1 = await _wait_resp(1)
        r2 = await _wait_resp(2)
    finally:
        poller.cancel()
        await asyncio.gather(poller, return_exceptions=True)
        await backend.cleanup()

    assert r1 == {"ok": False, "error": "unauthorized"}, r1
    assert r2["ok"] is True and r2["result"] == "ran-ok", r2
    assert ran["n"] == 1, ran            # only the authorized call reached the tool
    assert state["calls"] == 1, state    # the wrong-token call never incremented
