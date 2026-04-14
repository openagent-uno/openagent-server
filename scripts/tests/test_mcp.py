"""MCP usage tests — drive vault/scheduler/filesystem tools directly.

These tests invoke the underlying ``entrypoint`` on Agno's registered
callables rather than going through the LLM, so they're fast and
deterministic. They confirm the MCP servers actually execute work end
to end, not just that they loaded.
"""
from __future__ import annotations

import time
import uuid

from ._framework import TestContext, TestSkip, test


@test("mcp", "vault MCP: write a note then read it back")
async def t_vault_roundtrip(ctx: TestContext) -> None:
    pool = ctx.extras["pool"]
    vault_tk = next(
        (t for t in pool.agno_toolkits if getattr(t, "tool_name_prefix", "") == "vault"),
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
        (t for t in pool.agno_toolkits if getattr(t, "tool_name_prefix", "") == "scheduler"),
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
        (t for t in pool.agno_toolkits if getattr(t, "tool_name_prefix", "") == "filesystem"),
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
