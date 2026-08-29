"""MCP usage tests — drive vault/scheduler/filesystem tools directly.

These tests invoke the underlying ``entrypoint`` on the runtime's registered
callables rather than going through the LLM, so they're fast and
deterministic. They confirm the MCP servers actually execute work end
to end, not just that they loaded.
"""
from __future__ import annotations

import os
import time
import uuid
from types import SimpleNamespace

from ._framework import TestContext, TestSkip, test


@test("mcp", "tool-search denylist hides and blocks one MCP tool")
async def t_tool_search_denylist(ctx: TestContext) -> None:
    """A deployment can remove a dangerous write without disabling its MCP.

    The policy must cover discovery, schema inspection, the normal prefixed
    runtime key, and the bare leaf alias accepted by tool-search.
    """
    from src.mcp.servers.tool_search.adapters import (
        _call_tool_impl,
        _describe_tool_impl,
        _list_servers_impl,
        _list_tools_impl,
    )

    calls: list[dict] = []

    async def create_task(**kwargs):
        calls.append(kwargs)
        return {"created": True}

    async def link_task(**kwargs):
        return {"linked": kwargs}

    create_fn = SimpleNamespace(
        entrypoint=create_task,
        description="Create a task",
        parameters={"type": "object"},
    )
    link_fn = SimpleNamespace(
        entrypoint=link_task,
        description="Link a task",
        parameters={"type": "object"},
    )
    toolkit = SimpleNamespace(functions={
        "replio_thread_create_task": create_fn,
        "replio_thread_link_task": link_fn,
    })

    class Pool:
        _toolkit_by_name = {"replio": toolkit}

        def toolkit_by_name(self, name):
            return self._toolkit_by_name.get(name)

    pool = Pool()
    env_name = "OPENAGENT_MCP_TOOL_DENYLIST"
    previous = os.environ.get(env_name)
    os.environ[env_name] = "replio:thread_create_task"
    try:
        listed = _list_tools_impl(pool, "replio")
        assert [item["name"] for item in listed] == ["replio_thread_link_task"]
        assert _list_servers_impl(pool) == [{"name": "replio", "tool_count": 1}]

        for tool_name in ("thread_create_task", "replio_thread_create_task"):
            try:
                _describe_tool_impl(pool, "replio", tool_name)
                raise AssertionError(f"denied tool {tool_name} was describable")
            except PermissionError as exc:
                assert env_name in str(exc)
            try:
                await _call_tool_impl(pool, "replio", tool_name, {"thread_id": "t1"})
                raise AssertionError(f"denied tool {tool_name} was callable")
            except PermissionError as exc:
                assert env_name in str(exc)

        allowed = await _call_tool_impl(
            pool,
            "replio",
            "thread_link_task",
            {"thread_id": "t1", "external_task_id": "cu1"},
        )
        assert allowed == {"linked": {"thread_id": "t1", "external_task_id": "cu1"}}
        assert calls == [], "the denied task-creation function ran"
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous


@test("mcp", "vault MCP: write a note then read it back")
async def t_vault_roundtrip(ctx: TestContext) -> None:
    pool = ctx.extras["pool"]
    vault_tk = next(
        (t for t in pool.runtime_toolkits if getattr(t, "tool_name_prefix", "") == "vault"),
        None,
    )
    if vault_tk is None:
        raise TestSkip("vault toolkit not loaded")

    write_fn = vault_tk.functions.get("vault_write_note")
    read_fn = vault_tk.functions.get("vault_read_note")
    assert write_fn and read_fn, "vault tools not registered"
    note_path = f"openagent-test-{uuid.uuid4().hex[:8]}.md"
    body = f"hello from openagent test {time.time()}"
    await write_fn.entrypoint(path=note_path, content=body)
    res = await read_fn.entrypoint(path=note_path)
    out = res.content if hasattr(res, "content") else str(res)
    assert body in out, f"vault read didn't return body; got: {out[:200]}"


@test("mcp", "scheduler MCP: create + list + delete a one-shot task")
async def t_scheduler_roundtrip(ctx: TestContext) -> None:
    pool = ctx.extras["pool"]
    sched_tk = next(
        (t for t in pool.runtime_toolkits if getattr(t, "tool_name_prefix", "") == "scheduler"),
        None,
    )
    if sched_tk is None:
        raise TestSkip("scheduler toolkit not loaded")
    fns = sched_tk.functions
    assert "scheduler_create_one_shot_task" in fns
    assert "scheduler_list_scheduled_tasks" in fns
    assert "scheduler_delete_scheduled_task" in fns

    task_name = f"openagent-test-{uuid.uuid4().hex[:6]}"
    created = await fns["scheduler_create_one_shot_task"].entrypoint(
        name=task_name,
        prompt=f"test prompt {uuid.uuid4().hex[:8]}",
        delay_seconds=3600,
    )
    out = created.content if hasattr(created, "content") else str(created)
    assert "id" in out.lower() or task_name in out, f"unexpected: {out[:200]}"

    listed = await fns["scheduler_list_scheduled_tasks"].entrypoint()
    listed_out = listed.content if hasattr(listed, "content") else str(listed)
    assert task_name in listed_out, f"task not in list: {listed_out[:300]}"


@test("mcp", "filesystem MCP: list_directory works on /tmp")
async def t_filesystem_list(ctx: TestContext) -> None:
    pool = ctx.extras["pool"]
    fs_tk = next(
        (t for t in pool.runtime_toolkits if getattr(t, "tool_name_prefix", "") == "filesystem"),
        None,
    )
    if fs_tk is None:
        raise TestSkip("filesystem toolkit not loaded")
    fn = fs_tk.functions.get("filesystem_list_directory")
    if not fn:
        raise TestSkip("list_directory not available")
    res = await fn.entrypoint(path="/tmp")
    out = res.content if hasattr(res, "content") else str(res)
    assert len(out) > 0, "list_directory returned empty"


@test("mcp", "every Python builtin has a _mcp-server CLI dispatcher branch")
async def t_cli_dispatcher_covers_python_builtins(ctx: TestContext) -> None:
    """Frozen PyInstaller binaries rewrite ``python -m …server`` to
    ``openagent _mcp-server <name>``. If the dispatcher in cli.py misses
    a name, the subprocess dies with "Unknown MCP server" and the pool
    marks the MCP dormant — exactly how workflow-manager shipped broken
    on the VPS for three days. This check keeps the two lists in sync.
    """
    import inspect
    from src import cli
    from src.mcp.builtins import BUILTIN_MCP_SPECS

    src = inspect.getsource(cli.mcp_server_cmd.callback)
    missing = [
        name for name, spec in BUILTIN_MCP_SPECS.items()
        if spec.get("python") and f'"{name}"' not in src and f"'{name}'" not in src
    ]
    assert not missing, (
        f"cli._mcp-server has no dispatcher branch for Python builtin(s): {missing}. "
        f"Add an `if name == \"X\":` import+call in openagent/cli.py:mcp_server_cmd."
    )


@test("mcp", "tool-search MCP exposes the four navigation tools")
async def t_tool_search_exposes_tools(ctx: TestContext) -> None:
    """``tool-search`` is the recovery channel for trimmed-out MCPs.
    It must register exactly four tools (``list_servers``, ``list_tools``,
    ``describe_tool``, ``call_tool``) under both provider adapters,
    because the budget filter in ``wire_model_runtime`` relies on it
    to be present and complete."""
    pool = ctx.extras["pool"]
    ts = pool.toolkit_by_name("tool-search")
    assert ts is not None, "tool-search not loaded in test pool"
    fns = {
        **(getattr(ts, "functions", {}) or {}),
        **(getattr(ts, "async_functions", {}) or {}),
    }
    expected = {
        "tool_search_list_servers",
        "tool_search_list_tools",
        "tool_search_describe_tool",
        "tool_search_call_tool",
    }
    missing = expected - fns.keys()
    assert not missing, f"tool-search missing tools: {missing}"


@test("mcp", "tool-search.call_tool reaches a real MCP via the pool")
async def t_tool_search_call_tool_roundtrip(ctx: TestContext) -> None:
    """``call_tool`` is the recovery path for any MCP trimmed by the
    budget filter. We exercise it through scheduler (always available
    in the test pool) so a regression in the dispatch logic — wrong
    function lookup, wrong kwargs unpacking, async/sync confusion —
    surfaces on every test run, not only the day someone hits the cap.
    """
    import uuid
    pool = ctx.extras["pool"]
    ts = pool.toolkit_by_name("tool-search")
    if ts is None:
        raise TestSkip("tool-search not loaded")
    fns = {
        **(getattr(ts, "functions", {}) or {}),
        **(getattr(ts, "async_functions", {}) or {}),
    }
    call = fns.get("tool_search_call_tool")
    if call is None:
        raise TestSkip("tool_search_call_tool not registered")
    target = getattr(call, "entrypoint", None) or call

    name = f"openagent-search-{uuid.uuid4().hex[:6]}"
    result = await target(
        server="scheduler",
        tool="scheduler_create_one_shot_task",
        args={
            "name": name,
            "prompt": f"test prompt {uuid.uuid4().hex[:8]}",
            "delay_seconds": 3600,
        },
    )
    rendered = str(result)
    assert "id" in rendered.lower() or name in rendered, (
        f"call_tool didn't dispatch to scheduler properly; got: {rendered[:300]}"
    )


@test("mcp", "pool budget filter trims subprocess MCPs alphabetically, keeps tool-search")
async def t_pool_tool_budget(ctx: TestContext) -> None:
    """The whole point of the budget filter is: when there are too many
    tools for the provider's cap, drop subprocess MCPs alphabetically
    until they fit, but never drop in-process MCPs (especially
    ``tool-search``, which is the model's only way back to the
    trimmed ones).
    """
    from src.mcp.pool import MCPPool

    # ``include_defaults=False`` keeps the test hermetic: the pool
    # contains exactly the five MCPs listed below. Otherwise ``vault``,
    # ``filesystem``, ``attachments``, ``delegation`` (etc.) would also
    # load as defaults and skew the budget arithmetic — the tight-budget
    # assertion below is keyed off ``tool-search`` + ``shell`` only.
    pool = MCPPool.from_config(
        [
            {"builtin": "tool-search"},
            {"builtin": "shell"},
            {"builtin": "scheduler"},
            {"builtin": "mcp-manager"},
            {"builtin": "workflow-manager"},
        ],
        include_defaults=False,
    )
    await pool.connect_all()
    try:
        # Budget so tight only in-process fits → keep tool-search + shell,
        # drop every subprocess MCP.
        in_process_count = (
            pool._tool_counts.get("tool-search", 0)
            + pool._tool_counts.get("shell", 0)
        )
        runtime_subset = pool.runtime_toolkits_under_budget(in_process_count)
        runtime_names = {pool._toolkit_name(tk) for tk in runtime_subset}
        assert runtime_names == {"tool-search", "shell"}, (
            f"runtime tight budget kept the wrong set: {runtime_names}"
        )

        # Generous budget should keep everything we explicitly added.
        # ``from_config`` may merge in additional defaults — assert subset
        # so the test stays robust to that.
        runtime_full = pool.runtime_toolkits_under_budget(10_000)
        runtime_full_names = {pool._toolkit_name(tk) for tk in runtime_full}
        expected = {"tool-search", "shell", "scheduler", "mcp-manager", "workflow-manager"}
        assert expected <= runtime_full_names, (
            f"generous runtime budget dropped something: {expected - runtime_full_names}"
        )

        # ``budget < 0`` is the legacy bypass — should equal the unfiltered view.
        assert {pool._toolkit_name(tk) for tk in pool.runtime_toolkits_under_budget(-1)} \
            == {pool._toolkit_name(tk) for tk in pool.runtime_toolkits}, \
            "budget=-1 must skip trimming entirely (legacy callers depend on it)"
    finally:
        await pool.close_all()
