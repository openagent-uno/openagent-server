"""``tool_search_call_tool`` — a miss that repeats has to say so.

Observed on a live agent (2026-08-25): a model answered "reply with the word
OK" by spending **22 subscription calls** alternating between ``file``,
``run_command`` and ``shell_run_command``, none of which exist. Every miss
returned a perfectly good error listing the real names — and said nothing
about the fact that the same call had already failed, so nothing ever told
the model that guessing was the problem.

The guard escalates the wording and never refuses: the counter is
process-wide (there is no turn identity at this layer), so it must not be
able to deny anyone a working tool. Any resolved call clears it.
"""
from __future__ import annotations

from ._framework import TestContext, test


class _EmptyPool:
    """A pool with no MCPs loaded — every lookup misses."""

    _toolkit_by_name: dict = {}

    def toolkit_by_name(self, _name):
        return None


class _Toolkit:
    def __init__(self, functions):
        self.functions = functions
        self.async_functions = {}


class _Pool:
    def __init__(self, toolkits):
        self._toolkit_by_name = toolkits

    def toolkit_by_name(self, name):
        return self._toolkit_by_name.get(name)


async def _call(pool, server, tool):
    from src.mcp.servers.tool_search import adapters

    try:
        await adapters._call_tool_impl(pool, server, tool, {})
    except ValueError as e:
        return str(e)
    return ""


@test("tool_search_repeat_miss", "the first misses stay quiet, the repeat shouts")
async def t_escalates_only_on_repeat(ctx: TestContext) -> None:
    from src.mcp.servers.tool_search import adapters

    adapters._clear_misses()
    pool = _EmptyPool()

    first = await _call(pool, "shell", "run_command")
    second = await _call(pool, "shell", "run_command")
    third = await _call(pool, "shell", "run_command")

    # The useful part — what IS loaded — is present every time.
    assert "Known MCPs" in first
    # Two attempts could be a slip; the wording stays unchanged.
    assert "STOP" not in first and "STOP" not in second, (first, second)
    # By the third identical call, guessing is the problem, and the message
    # has to name that rather than repeat itself.
    assert "STOP" in third and "failed 3 times" in third, third
    assert "shell.run_command" in third, third


@test("tool_search_repeat_miss", "a different invented name is counted on its own")
async def t_per_call_counter(ctx: TestContext) -> None:
    from src.mcp.servers.tool_search import adapters

    adapters._clear_misses()
    pool = _EmptyPool()

    for _ in range(3):
        await _call(pool, "shell", "run_command")
    other = await _call(pool, "file", "read")

    # A fresh wrong name has not failed before: it gets the plain error, so
    # the loud wording can never be the first thing a model sees.
    assert "STOP" not in other, other


@test("tool_search_repeat_miss", "a resolved call clears the history")
async def t_success_resets(ctx: TestContext) -> None:
    from src.mcp.servers.tool_search import adapters

    adapters._clear_misses()
    empty = _EmptyPool()
    for _ in range(3):
        await _call(empty, "shell", "run_command")
    assert adapters._MISS_COUNTS, "misses should have been recorded"

    async def _ok(**_kwargs):
        return "done"

    working = _Pool({"shell": _Toolkit({"shell_exec": _ok})})
    await adapters._call_tool_impl(working, "shell", "shell_exec", {})

    assert adapters._MISS_COUNTS == {}, "a working call must reset the guard"
    # And the next miss starts from scratch, quietly.
    again = await _call(empty, "shell", "run_command")
    assert "STOP" not in again, again


@test("tool_search_repeat_miss", "vault keyword search compatibility alias is read-only and exact")
async def t_vault_search_compatibility_alias(_ctx: TestContext) -> None:
    from src.mcp.servers.tool_search import adapters

    calls: list[dict] = []

    async def _search_notes(query: str, limit: int = 20):
        calls.append({"query": query, "limit": limit})
        return {"results": []}

    pool = _Pool({
        "vault": _Toolkit({"vault_search_notes": _search_notes}),
    })
    result = await adapters._call_tool_impl(
        pool, "vault", "vault_search", {"query": "refund policy", "limit": 7},
    )

    assert result == {"results": []}, result
    assert calls == [{"query": "refund policy", "limit": 7}], calls
    assert adapters._candidate_names("vault", "vault_write") == [
        "vault_write", "vault_vault_write", "write",
    ], "mutations must never gain an equivalent-name alias"


@test("tool_search_repeat_miss", "execution scope filters discovery and invocation")
async def t_execution_scope_enforced(_ctx: TestContext) -> None:
    from src.core import tool_scope
    from src.mcp.servers.tool_search import adapters

    async def _ok(**_kwargs):
        return "done"

    pool = _Pool({
        "replio": _Toolkit({"replio_read": _ok}),
        "shell": _Toolkit({"shell_exec": _ok}),
    })
    token = tool_scope.set_tool_allowlist(["replio"])
    try:
        assert [row["name"] for row in adapters._list_servers_impl(pool)] == ["replio"]
        assert await adapters._call_tool_impl(
            pool, "replio", "replio_read", {},
        ) == "done"
        for operation in (
            lambda: adapters._list_tools_impl(pool, "shell"),
            lambda: adapters._describe_tool_impl(pool, "shell", "shell_exec"),
        ):
            try:
                operation()
            except PermissionError:
                pass
            else:
                raise AssertionError("out-of-scope discovery was allowed")
        try:
            await adapters._call_tool_impl(pool, "shell", "shell_exec", {})
        except PermissionError:
            pass
        else:
            raise AssertionError("out-of-scope call was allowed")
    finally:
        tool_scope.reset_tool_allowlist(token)


@test("tool_search_repeat_miss", "the miss table cannot grow without bound")
async def t_bounded(ctx: TestContext) -> None:
    from src.mcp.servers.tool_search import adapters

    adapters._clear_misses()
    for i in range(adapters._MISS_COUNTS_MAX + 10):
        adapters._note_miss("srv", f"tool-{i}")
    assert len(adapters._MISS_COUNTS) <= adapters._MISS_COUNTS_MAX, len(adapters._MISS_COUNTS)
    adapters._clear_misses()
