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
