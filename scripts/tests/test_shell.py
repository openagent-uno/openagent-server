"""Shell MCP — unit + integration tests for the in-process shell tools."""
from __future__ import annotations

from ._framework import TestContext, test


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
