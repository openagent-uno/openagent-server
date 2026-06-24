"""Scheduler run_task hook registry — idempotent, kwarg-preserving.

Guards the refactor of ``AgentServer._install_task_hook`` (formerly
``_wrap_scheduler_run_task``), which previously composed a fresh closure
on every call. Toggling a built-in task's config section at runtime
re-synced its hook and stacked another wrapper each time, so
``_do_auto_update`` (and dream/manager hooks) fired once per accumulated
layer; the old wrapper also dropped the ``trigger`` kwarg.
"""
from __future__ import annotations

from ._framework import TestContext, test


class _FakeScheduler:
    def __init__(self):
        self.base_calls = []

    async def run_task(self, task, *, trigger: str = "schedule"):
        self.base_calls.append((task["name"], trigger))


@test("task_hooks", "re-installing a hook does not stack (no duplicate firing)")
async def t_no_stacking(ctx: TestContext) -> None:
    from src.core.server import AgentServer

    sched = _FakeScheduler()
    fired = {"n": 0}

    async def hook(task, nxt):
        if task["name"] == "AUTO":
            fired["n"] += 1
        await nxt(task)

    # Simulate boot install + 5 runtime config toggles re-syncing the hook.
    for _ in range(6):
        AgentServer._install_task_hook(sched, "AUTO", hook)

    await sched.run_task({"name": "AUTO"})
    assert fired["n"] == 1, f"hook stacked: fired {fired['n']} times"


@test("task_hooks", "the dispatcher forwards the trigger kwarg to the real run_task")
async def t_trigger_forwarded(ctx: TestContext) -> None:
    from src.core.server import AgentServer

    sched = _FakeScheduler()

    async def hook(task, nxt):
        await nxt(task)

    AgentServer._install_task_hook(sched, "X", hook)
    await sched.run_task({"name": "X"}, trigger="manual")
    await sched.run_task({"name": "Y"})  # default trigger
    assert ("X", "manual") in sched.base_calls, sched.base_calls
    assert ("Y", "schedule") in sched.base_calls, sched.base_calls


@test("task_hooks", "passing hook=None removes a previously-installed hook")
async def t_removable(ctx: TestContext) -> None:
    from src.core.server import AgentServer

    sched = _FakeScheduler()
    fired = {"n": 0}

    async def hook(task, nxt):
        fired["n"] += 1
        await nxt(task)

    AgentServer._install_task_hook(sched, "Z", hook)
    await sched.run_task({"name": "Z"})
    AgentServer._install_task_hook(sched, "Z", None)
    await sched.run_task({"name": "Z"})
    assert fired["n"] == 1, "hook still fired after removal"
    # The base task still runs both times (dispatcher stays installed).
    assert sched.base_calls.count(("Z", "schedule")) == 2


@test("task_hooks", "multiple distinct hooks each fire for their own task")
async def t_multiple_hooks(ctx: TestContext) -> None:
    from src.core.server import AgentServer

    sched = _FakeScheduler()
    fired = {"a": 0, "b": 0}

    async def hook_a(task, nxt):
        if task["name"] == "A":
            fired["a"] += 1
        await nxt(task)

    async def hook_b(task, nxt):
        if task["name"] == "B":
            fired["b"] += 1
        await nxt(task)

    AgentServer._install_task_hook(sched, "A", hook_a)
    AgentServer._install_task_hook(sched, "B", hook_b)
    await sched.run_task({"name": "A"})
    await sched.run_task({"name": "B"})
    await sched.run_task({"name": "C"})
    assert (fired["a"], fired["b"]) == (1, 1), fired
    assert ("C", "schedule") in sched.base_calls
