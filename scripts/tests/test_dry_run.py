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
