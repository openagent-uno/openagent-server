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
