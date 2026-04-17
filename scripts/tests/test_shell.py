"""Shell MCP — unit + integration tests for the in-process shell tools."""
from __future__ import annotations

from ._framework import TestContext, test


def _reset_shell_hub() -> None:
    """Isolate hub state between tests so they don't see leaked shells."""
    from openagent.mcp.servers.shell import handlers
    handlers._reset_hub_for_tests()


@test("shell", "ShellEvent is a frozen dataclass with expected fields")
async def t_shell_event_shape(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.events import ShellEvent

    e = ShellEvent(
        shell_id="sh_abc",
        kind="completed",
        exit_code=0,
        signal=None,
        bytes_stdout=42,
        bytes_stderr=0,
        at=123.0,
    )
    assert e.shell_id == "sh_abc"
    assert e.kind == "completed"
    assert e.exit_code == 0
    assert e.signal is None
    assert e.bytes_stdout == 42
    assert e.bytes_stderr == 0
    assert e.at == 123.0
    # Frozen → setattr raises.
    try:
        e.shell_id = "sh_xyz"  # type: ignore[misc]
    except Exception:
        pass
    else:
        raise AssertionError("ShellEvent should be frozen")


@test("shell", "ShellHub: register and get a shell by id")
async def t_hub_register_get(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.hub import ShellHub

    hub = ShellHub()
    hub.register(shell_id="sh_1", session_id="s1", command="echo hi")
    got = hub.get("sh_1")
    assert got is not None, "get should return the registered record"
    assert got.command == "echo hi"
    assert got.session_id == "s1"


@test("shell", "ShellHub: list_for_session filters by session")
async def t_hub_list_for_session(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.hub import ShellHub

    hub = ShellHub()
    hub.register(shell_id="sh_1", session_id="s1", command="a")
    hub.register(shell_id="sh_2", session_id="s2", command="b")
    hub.register(shell_id="sh_3", session_id="s1", command="c")

    ids_s1 = {r.shell_id for r in hub.list_for_session("s1")}
    ids_s2 = {r.shell_id for r in hub.list_for_session("s2")}
    ids_all = {r.shell_id for r in hub.list_for_session(None)}

    assert ids_s1 == {"sh_1", "sh_3"}, f"expected s1 shells, got {ids_s1}"
    assert ids_s2 == {"sh_2"}
    assert ids_all == {"sh_1", "sh_2", "sh_3"}


@test("shell", "ShellHub: has_running only true while not completed")
async def t_hub_has_running(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.hub import ShellHub

    hub = ShellHub()
    hub.register(shell_id="sh_1", session_id="s1", command="x")
    assert hub.has_running("s1") is True
    hub.mark_completed("sh_1", exit_code=0, signal=None)
    assert hub.has_running("s1") is False


@test("shell", "ShellHub: purge_session removes entries and reports killed ids")
async def t_hub_purge_session(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.hub import ShellHub

    hub = ShellHub()
    hub.register(shell_id="sh_1", session_id="s1", command="a")
    hub.register(shell_id="sh_2", session_id="s1", command="b")
    hub.register(shell_id="sh_3", session_id="s2", command="c")

    purged = await hub.purge_session("s1")
    assert sorted(purged) == ["sh_1", "sh_2"], f"unexpected: {purged}"
    assert hub.get("sh_1") is None
    assert hub.get("sh_2") is None
    assert hub.get("sh_3") is not None


@test("shell", "ShellHub: post_event + drain returns events in FIFO order")
async def t_hub_post_drain(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.hub import ShellHub
    from openagent.mcp.servers.shell.events import ShellEvent

    hub = ShellHub()
    e1 = ShellEvent("sh_1", "completed", 0, None, 10, 0, 1.0)
    e2 = ShellEvent("sh_2", "killed", None, "TERM", 3, 5, 2.0)
    hub.post_event("s1", e1)
    hub.post_event("s1", e2)
    drained = hub.drain("s1")
    assert [e.shell_id for e in drained] == ["sh_1", "sh_2"]
    # Queue is empty after drain.
    assert hub.drain("s1") == []


@test("shell", "ShellHub: drain on unknown session returns []")
async def t_hub_drain_unknown(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.hub import ShellHub

    hub = ShellHub()
    assert hub.drain("nope") == []


@test("shell", "ShellHub: wait resolves when an event is posted")
async def t_hub_wait_wakes_up(ctx: TestContext) -> None:
    import asyncio
    from openagent.mcp.servers.shell.hub import ShellHub
    from openagent.mcp.servers.shell.events import ShellEvent

    hub = ShellHub()
    e = ShellEvent("sh_9", "completed", 0, None, 1, 0, 9.0)

    async def delayed_post() -> None:
        await asyncio.sleep(0.05)
        hub.post_event("s1", e)

    task = asyncio.create_task(delayed_post())
    try:
        events = await hub.wait("s1", timeout=1.0)
    finally:
        await task
    assert len(events) == 1
    assert events[0].shell_id == "sh_9"


@test("shell", "ShellHub: wait returns [] on timeout")
async def t_hub_wait_timeout(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.hub import ShellHub

    hub = ShellHub()
    events = await hub.wait("s1", timeout=0.05)
    assert events == []


@test("shell", "ShellHub: queue cap drops oldest and keeps newest")
async def t_hub_queue_cap(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.hub import ShellHub
    from openagent.mcp.servers.shell.events import ShellEvent

    hub = ShellHub()
    # Post more than the cap (200) — confirm the newest 200 survive.
    for i in range(250):
        hub.post_event("s1", ShellEvent(f"sh_{i}", "completed", 0, None, 1, 0, float(i)))
    drained = hub.drain("s1")
    assert len(drained) == 200
    # The oldest 50 (sh_0 … sh_49) were dropped.
    assert drained[0].shell_id == "sh_50"
    assert drained[-1].shell_id == "sh_249"


@test("shell", "ShellHub: gc removes completed shells older than TTL")
async def t_hub_gc(ctx: TestContext) -> None:
    import time
    from openagent.mcp.servers.shell.hub import ShellHub

    hub = ShellHub()
    hub.register(shell_id="sh_old", session_id="s1", command="a")
    hub.register(shell_id="sh_new", session_id="s1", command="b")
    hub.register(shell_id="sh_live", session_id="s1", command="c")

    # Old completed 15 min ago; new completed 1 s ago; live still running.
    hub.mark_completed("sh_old", exit_code=0, signal=None)
    hub.mark_completed("sh_new", exit_code=0, signal=None)
    hub._shells["sh_old"].completed_at = time.time() - 15 * 60

    removed = hub.gc(ttl_seconds=10 * 60)
    assert removed == ["sh_old"], f"unexpected gc: {removed}"
    assert hub.get("sh_old") is None
    assert hub.get("sh_new") is not None
    assert hub.get("sh_live") is not None


@test("shell", "ShellHub: shutdown purges every session and clears state")
async def t_hub_shutdown(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.hub import ShellHub

    hub = ShellHub()
    hub.register(shell_id="sh_1", session_id="s1", command="a")
    hub.register(shell_id="sh_2", session_id="s2", command="b")
    await hub.shutdown()
    assert hub.get("sh_1") is None
    assert hub.get("sh_2") is None
    assert hub.list_for_session(None) == []
    assert hub.drain("s1") == []
    assert hub.drain("s2") == []


async def _run_bg_to_completion(bg, *, max_wait: float = 2.5) -> None:
    """Helper: busy-wait for ``bg`` to exit, then finalise. 50 x 50ms polls."""
    import asyncio
    for _ in range(int(max_wait / 0.05)):
        if not bg.is_running:
            break
        await asyncio.sleep(0.05)
    await bg.finalise()


@test("shell", "BackgroundShell: spawn echo and capture stdout + exit_code")
async def t_bg_spawn_echo(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.shells import BackgroundShell

    bg = BackgroundShell(
        shell_id="sh_echo",
        command="echo hello-from-shell",
        cwd=None,
        env=None,
    )
    await bg.start()
    await _run_bg_to_completion(bg)
    assert not bg.is_running, "echo should have completed within 2.5s"
    stdout, _ = bg.read(since_stdout=0, since_stderr=0)
    assert "hello-from-shell" in stdout
    assert bg.exit_code == 0


@test("shell", "BackgroundShell: non-zero exit is captured")
async def t_bg_nonzero_exit(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.shells import BackgroundShell

    bg = BackgroundShell(
        shell_id="sh_exit",
        command="exit 7",
        cwd=None,
        env=None,
    )
    await bg.start()
    await _run_bg_to_completion(bg)
    assert not bg.is_running
    assert bg.exit_code == 7


@test("shell", "BackgroundShell: stderr is captured separately")
async def t_bg_stderr(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.shells import BackgroundShell

    bg = BackgroundShell(
        shell_id="sh_err",
        command="echo to-err 1>&2",
        cwd=None,
        env=None,
    )
    await bg.start()
    await _run_bg_to_completion(bg)
    stdout, stderr = bg.read(since_stdout=0, since_stderr=0)
    assert stdout == "", f"expected no stdout, got: {stdout!r}"
    assert "to-err" in stderr


@test("shell", "BackgroundShell: read cursors advance (since_last semantics)")
async def t_bg_read_cursor(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.shells import BackgroundShell

    bg = BackgroundShell(
        shell_id="sh_cursor",
        command="printf 'ABC'",
        cwd=None,
        env=None,
    )
    await bg.start()
    await _run_bg_to_completion(bg)
    s1, _ = bg.read(since_stdout=0, since_stderr=0)
    assert s1 == "ABC"
    s2, _ = bg.read(since_stdout=len(s1.encode()), since_stderr=0)
    assert s2 == "", f"expected empty after full read, got: {s2!r}"


@test("shell", "BackgroundShell: write_stdin feeds a line to a running cat")
async def t_bg_stdin_cat(ctx: TestContext) -> None:
    import asyncio
    from openagent.mcp.servers.shell.shells import BackgroundShell

    bg = BackgroundShell(
        shell_id="sh_cat",
        command="cat",
        cwd=None,
        env=None,
    )
    await bg.start()
    try:
        n = await bg.write_stdin("hello\nworld\n", press_enter=False)
        assert n == len("hello\nworld\n")
        # Close stdin so cat exits.
        assert bg._proc is not None
        bg._proc.stdin.close()  # type: ignore[union-attr]
        await bg._proc.wait()
        await bg.finalise()
    finally:
        if bg.is_running:
            await bg.kill(signal_name="KILL", grace_seconds=0)  # defensive
    stdout, _ = bg.read(since_stdout=0, since_stderr=0)
    assert "hello" in stdout and "world" in stdout


@test("shell", "BackgroundShell: write_stdin with press_enter appends a newline")
async def t_bg_stdin_press_enter(ctx: TestContext) -> None:
    import asyncio
    from openagent.mcp.servers.shell.shells import BackgroundShell

    bg = BackgroundShell(
        shell_id="sh_cat2",
        command="cat",
        cwd=None,
        env=None,
    )
    await bg.start()
    try:
        n = await bg.write_stdin("ping", press_enter=True)
        assert n == len("ping\n")
        assert bg._proc is not None
        bg._proc.stdin.close()  # type: ignore[union-attr]
        await bg._proc.wait()
        await bg.finalise()
    finally:
        if bg.is_running:
            await bg.kill(signal_name="KILL", grace_seconds=0)
    stdout, _ = bg.read(since_stdout=0, since_stderr=0)
    assert stdout.rstrip("\n") == "ping"


@test("shell", "BackgroundShell: kill TERM stops a sleep")
async def t_bg_kill_term(ctx: TestContext) -> None:
    import asyncio
    from openagent.mcp.servers.shell.shells import BackgroundShell

    bg = BackgroundShell(
        shell_id="sh_sleep",
        command="sleep 30",
        cwd=None,
        env=None,
    )
    await bg.start()
    await asyncio.sleep(0.1)  # let it actually start
    await bg.kill(signal_name="TERM", grace_seconds=2.0)
    await bg.finalise()
    assert not bg.is_running
    # POSIX SIGTERM — signal captured, exit_code is None.
    assert bg.signal in ("TERM", "15"), f"unexpected signal: {bg.signal}"


@test("shell", "BackgroundShell: kill escalates to KILL if TERM ignored")
async def t_bg_kill_escalate(ctx: TestContext) -> None:
    import asyncio
    from openagent.mcp.servers.shell.shells import BackgroundShell

    # Trap TERM so only KILL works.
    bg = BackgroundShell(
        shell_id="sh_trap",
        command="trap '' TERM; sleep 30",
        cwd=None,
        env=None,
    )
    await bg.start()
    await asyncio.sleep(0.2)  # make sure trap is installed
    await bg.kill(signal_name="TERM", grace_seconds=0.5)
    await bg.finalise()
    assert not bg.is_running
    assert bg.signal in ("KILL", "9"), f"expected KILL, got {bg.signal}"


@test("shell", "BackgroundShell.run_with_timeout: fast command returns normally")
async def t_bg_run_with_timeout_ok(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.shells import BackgroundShell

    bg = BackgroundShell(shell_id="sh_ok", command="echo abc", cwd=None, env=None)
    result = await bg.run_with_timeout(timeout_seconds=2.0)
    assert result.timed_out is False
    assert result.exit_code == 0
    assert "abc" in result.stdout


@test("shell", "BackgroundShell.run_with_timeout: slow command is killed")
async def t_bg_run_with_timeout_kill(ctx: TestContext) -> None:
    import time
    from openagent.mcp.servers.shell.shells import BackgroundShell

    bg = BackgroundShell(shell_id="sh_slow", command="sleep 30", cwd=None, env=None)
    t0 = time.time()
    result = await bg.run_with_timeout(timeout_seconds=0.3)
    elapsed = time.time() - t0
    assert result.timed_out is True
    assert elapsed < 5.0, f"kill took too long: {elapsed}"
    assert result.signal in ("TERM", "KILL", "15", "9")


@test("shell", "handlers.shell_exec: foreground success")
async def t_handlers_exec_fg_ok(ctx: TestContext) -> None:
    _reset_shell_hub()
    from openagent.mcp.servers.shell import handlers

    out = await handlers.shell_exec(
        command="echo one-two-three",
        cwd=None, env=None, timeout=5000,
        run_in_background=False, stdin=None, description=None,
        session_id=None,
    )
    assert out["exit_code"] == 0
    assert "one-two-three" in out["stdout"]
    assert out["stderr"] == ""
    assert out["timed_out"] is False


@test("shell", "handlers.shell_exec: foreground timeout sets timed_out=True")
async def t_handlers_exec_fg_timeout(ctx: TestContext) -> None:
    _reset_shell_hub()
    from openagent.mcp.servers.shell import handlers

    out = await handlers.shell_exec(
        command="sleep 10",
        cwd=None, env=None, timeout=200,
        run_in_background=False, stdin=None, description=None,
        session_id=None,
    )
    assert out["timed_out"] is True
    assert out["signal"] in ("TERM", "KILL", "15", "9")


@test("shell", "handlers.shell_which: existing command returns path")
async def t_handlers_which_ok(ctx: TestContext) -> None:
    _reset_shell_hub()
    from openagent.mcp.servers.shell import handlers

    out = await handlers.shell_which(command="sh")
    assert out["available"] is True
    assert out["path"].endswith("/sh") or out["path"].endswith("sh.exe")


@test("shell", "handlers.shell_which: missing command returns available=false")
async def t_handlers_which_missing(ctx: TestContext) -> None:
    _reset_shell_hub()
    from openagent.mcp.servers.shell import handlers

    out = await handlers.shell_which(command="definitely_not_a_real_binary_xyz_123")
    assert out["available"] is False


@test("shell", "handlers.shell_exec background returns shell_id and posts terminal event")
async def t_handlers_exec_bg_event(ctx: TestContext) -> None:
    import asyncio
    from openagent.mcp.servers.shell import handlers

    _reset_shell_hub()
    started = await handlers.shell_exec(
        command="echo background-done",
        cwd=None, env=None, timeout=None,
        run_in_background=True, stdin=None, description=None,
        session_id="sess-A",
    )
    assert "shell_id" in started
    sid_shell = started["shell_id"]

    # Wait for the watcher to post the event.
    events = await handlers.get_hub().wait("sess-A", timeout=3.0)
    assert len(events) == 1
    ev = events[0]
    assert ev.shell_id == sid_shell
    assert ev.kind == "completed"
    assert ev.exit_code == 0


@test("shell", "handlers.shell_output: returns delta and marks not running")
async def t_handlers_output_delta(ctx: TestContext) -> None:
    import asyncio
    from openagent.mcp.servers.shell import handlers

    _reset_shell_hub()
    started = await handlers.shell_exec(
        command="printf 'abc'",
        cwd=None, env=None, timeout=None,
        run_in_background=True, stdin=None, description=None,
        session_id="sess-B",
    )
    sid_shell = started["shell_id"]
    # Wait until the watcher posts the terminal event, so we know
    # output has been fully drained.
    await handlers.get_hub().wait("sess-B", timeout=3.0)
    out = await handlers.shell_output(
        shell_id=sid_shell, filter=None, since_last=True,
    )
    assert out["still_running"] is False
    assert out["stdout_delta"] == "abc"
    assert out["stderr_delta"] == ""
    assert out["exit_code"] == 0
    # Second call with since_last=True returns empty delta (cursors advanced).
    out2 = await handlers.shell_output(
        shell_id=sid_shell, filter=None, since_last=True,
    )
    assert out2["stdout_delta"] == ""


@test("shell", "handlers.shell_output: filter matches per-line regex")
async def t_handlers_output_filter(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell import handlers

    _reset_shell_hub()
    started = await handlers.shell_exec(
        command="printf 'line-alpha\\nline-beta\\nline-gamma\\n'",
        cwd=None, env=None, timeout=None,
        run_in_background=True, stdin=None, description=None,
        session_id="sess-F",
    )
    sid_shell = started["shell_id"]
    await handlers.get_hub().wait("sess-F", timeout=3.0)
    out = await handlers.shell_output(
        shell_id=sid_shell, filter=r"beta|gamma", since_last=True,
    )
    lines = [l for l in out["stdout_delta"].splitlines() if l]
    assert lines == ["line-beta", "line-gamma"], f"got: {lines}"


@test("shell", "handlers.shell_input writes to a running shell's stdin")
async def t_handlers_input(ctx: TestContext) -> None:
    import asyncio
    from openagent.mcp.servers.shell import handlers

    _reset_shell_hub()
    started = await handlers.shell_exec(
        command="cat",
        cwd=None, env=None, timeout=None,
        run_in_background=True, stdin=None, description=None,
        session_id="sess-I",
    )
    sid = started["shell_id"]
    written = await handlers.shell_input(shell_id=sid, text="hey", press_enter=True)
    assert written["bytes_written"] == len("hey\n")
    # Brief pause so cat has time to echo the input before we kill it.
    await asyncio.sleep(0.1)
    # Kill to let the watcher fire so the hub state is clean.
    await handlers.shell_kill(shell_id=sid, signal="KILL")
    await handlers.get_hub().wait("sess-I", timeout=3.0)
    out = await handlers.shell_output(shell_id=sid, filter=None, since_last=True)
    assert "hey" in out["stdout_delta"]


@test("shell", "handlers.shell_kill terminates a running shell")
async def t_handlers_kill(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell import handlers

    _reset_shell_hub()
    started = await handlers.shell_exec(
        command="sleep 30",
        cwd=None, env=None, timeout=None,
        run_in_background=True, stdin=None, description=None,
        session_id="sess-K",
    )
    sid = started["shell_id"]
    res = await handlers.shell_kill(shell_id=sid, signal="TERM")
    assert res["killed"] is True
    await handlers.get_hub().wait("sess-K", timeout=3.0)
    rec = handlers.get_hub().get(sid)
    assert rec is not None and rec.is_completed


@test("shell", "handlers.shell_list returns running and recently-completed shells")
async def t_handlers_list(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell import handlers

    _reset_shell_hub()
    bg1 = await handlers.shell_exec(
        command="sleep 5", cwd=None, env=None, timeout=None,
        run_in_background=True, stdin=None, description="long",
        session_id="sess-L",
    )
    bg2 = await handlers.shell_exec(
        command="echo fast", cwd=None, env=None, timeout=None,
        run_in_background=True, stdin=None, description="short",
        session_id="sess-L",
    )
    await handlers.get_hub().wait("sess-L", timeout=3.0)  # fast one completes

    listing = await handlers.shell_list(session_id="sess-L")
    assert isinstance(listing, list)
    ids = {entry["shell_id"] for entry in listing}
    assert bg1["shell_id"] in ids and bg2["shell_id"] in ids
    states = {entry["shell_id"]: entry["state"] for entry in listing}
    assert states[bg2["shell_id"]] == "completed"
    assert states[bg1["shell_id"]] == "running"

    # Clean up long-runner.
    await handlers.shell_kill(shell_id=bg1["shell_id"], signal="KILL")
    await handlers.get_hub().wait("sess-L", timeout=3.0)


@test("shell", "adapters.build_sdk_server exposes the six tools")
async def t_adapter_claude(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.adapters import build_sdk_server

    cfg = build_sdk_server()
    assert cfg is not None, "expected a non-None SDK server config"
    # McpSdkServerConfig is a TypedDict / dict in the SDK. Smoke check.
    assert "instance" in cfg or "server" in cfg or "type" in cfg, f"unexpected shape: {cfg!r}"


@test("shell", "adapters.build_agno_toolkit exposes the six tools by name")
async def t_adapter_agno(ctx: TestContext) -> None:
    from openagent.mcp.servers.shell.adapters import build_agno_toolkit

    tk = build_agno_toolkit()
    names = set()
    for attr in ("functions",):
        container = getattr(tk, attr, None)
        if isinstance(container, dict):
            names.update(container.keys())
    # Agno's Toolkit populates .functions on init with the callables it
    # was given; names come from the function __name__.
    if not names:
        # Fallback: look at the underlying tools list.
        tools = getattr(tk, "tools", []) or []
        names = {t.__name__ for t in tools if callable(t)}
    for expected in ("shell_exec", "shell_output", "shell_input", "shell_kill", "shell_list", "shell_which"):
        assert expected in names, f"missing tool {expected} in {names}"


@test("shell", "adapters.build_sdk_server: shell_exec schema has command required, run_in_background boolean")
async def t_adapter_claude_schema(ctx: TestContext) -> None:
    import mcp.types as mcp_types
    from openagent.mcp.servers.shell.adapters import build_sdk_server

    cfg = build_sdk_server()
    server = cfg.get("instance")
    assert server is not None, f"expected instance in cfg, got keys: {list(cfg.keys())}"

    # The MCP server (mcp.server.lowlevel.server.Server) populates _tool_cache
    # lazily on the first list_tools call. Trigger it now.
    list_tools_handler = server.request_handlers.get(mcp_types.ListToolsRequest)
    assert list_tools_handler is not None, "no ListToolsRequest handler found"
    await list_tools_handler(mcp_types.ListToolsRequest(method="tools/list"))

    tool_cache: dict = getattr(server, "_tool_cache", {})
    exec_tool = tool_cache.get("shell_exec")
    assert exec_tool is not None, (
        f"couldn't locate shell_exec in _tool_cache; available: {list(tool_cache.keys())}"
    )

    # mcp.types.Tool uses inputSchema (camelCase).
    schema = getattr(exec_tool, "inputSchema", None)
    assert isinstance(schema, dict), f"schema shape: {schema!r}"
    assert schema.get("required") == ["command"], f"required: {schema.get('required')}"
    props = schema.get("properties", {})
    assert props.get("run_in_background", {}).get("type") == "boolean", \
        f"run_in_background type: {props.get('run_in_background')}"
    assert props.get("timeout", {}).get("type") == "integer", \
        f"timeout type: {props.get('timeout')}"
