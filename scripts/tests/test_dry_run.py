"""Dry-run meta propagation — a run marked dry-run stamps every MCP tool call
with ``meta={"dry_run": True}`` so the server can capture/reject writes. Guards
the two ends: the ContextVar scope, and the single MCP call site.
"""
from __future__ import annotations

from types import SimpleNamespace

from ._framework import TestContext, test


@test("dry_run", "scope sets/resets the flag; default is live")
async def t_scope(_ctx: TestContext) -> None:
    from src.core.dry_run import is_dry_run, call_meta, dry_run_scope

    assert is_dry_run() is False
    assert call_meta() is None
    with dry_run_scope(True):
        assert is_dry_run() is True
        assert call_meta() == {"dry_run": True}
    assert is_dry_run() is False  # reset on exit
    with dry_run_scope(False):
        assert is_dry_run() is False


class _FakeSession:
    """Records the meta each call_tool receives."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def send_ping(self) -> None:  # entrypoint pings before calling
        return None

    async def call_tool(self, name, arguments=None, *_, meta=None, **__):
        from mcp.types import CallToolResult, TextContent

        self.calls.append({"name": name, "arguments": arguments, "meta": meta})
        return CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)


@test("dry_run", "entrypoint stamps dry-run meta only inside a dry_run_scope")
async def t_entrypoint_meta(_ctx: TestContext) -> None:
    from src.core._runner.utils.mcp import get_entrypoint_for_tool
    from src.core.dry_run import dry_run_scope

    fake = _FakeSession()
    tool = SimpleNamespace(name="threads_respond")
    ep = get_entrypoint_for_tool(tool=tool, session=fake, mcp_tools_instance=None)

    # Live run → no meta (identical to not passing it).
    await ep(body_text="hi")
    assert fake.calls[-1]["meta"] is None, fake.calls[-1]

    # Dry-run → meta carries the flag the MCP server keys off.
    with dry_run_scope(True):
        await ep(body_text="hi")
    assert fake.calls[-1]["meta"] == {"dry_run": True}, fake.calls[-1]
    assert fake.calls[-1]["name"] == "threads_respond"


@test("dry_run", "event turn has a wall-clock cap so it can't zombie")
async def t_run_wallclock_cap(_ctx: TestContext) -> None:
    import asyncio
    import src.core.event_dispatcher as ed

    # A finite, positive cap must exist — otherwise a rate-limited turn blocking
    # on backoff would stay "running" for hours (the jam this guards against).
    assert isinstance(ed._EVENT_RUN_TIMEOUT_SECONDS, int)
    assert 0 < ed._EVENT_RUN_TIMEOUT_SECONDS <= 3600

    # The cap is enforced with asyncio.wait_for: a turn that outlives it is
    # cancelled and surfaces TimeoutError (which dispatch_event records as
    # failed → the reconcile sweep re-fires it later).
    async def hang():
        await asyncio.sleep(9999)

    try:
        await asyncio.wait_for(hang(), timeout=0.05)
        assert False, "expected the cap to fire"
    except asyncio.TimeoutError:
        pass
