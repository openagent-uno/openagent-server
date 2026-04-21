"""Regression guard for issue #5: scheduled firings must start in a fresh session.

Before the fix, ``Scheduler.run_task`` called ``agent.release_session`` in
its finally block. ``release_session`` disconnects the live Claude CLI
subprocess but keeps the provider-native ``sdk_session_id`` on disk, so
the next firing of the same task spawned a new subprocess with
``--resume <uuid>`` and inherited the previous firing's transcript. Once
that transcript crossed Claude's compaction threshold it would be summarised
to something like *"all work already done, nothing outstanding"*, and every
subsequent firing would silently exit without re-running the task prompt.

The fix swaps ``release_session`` for ``forget_session``, which also drops
the resume id so the next firing gets a blank conversation. This test
pins that contract without spawning the real ``claude`` binary.
"""
from __future__ import annotations

from ._framework import TestContext, test


class _SpyAgent:
    """Minimal Agent stub that records which release method the scheduler called."""

    def __init__(self) -> None:
        self.run_calls: list[tuple[str, str]] = []
        self.forget_calls: list[str] = []
        self.release_calls: list[str] = []

    async def refresh_registries(self) -> None:
        return None

    async def run(self, *, message: str, user_id: str, session_id: str) -> str:
        self.run_calls.append((session_id, message))
        return "ok"

    async def forget_session(self, session_id: str) -> None:
        self.forget_calls.append(session_id)

    async def release_session(self, session_id: str) -> None:  # pragma: no cover
        # Kept so a regression that restores the old call fails loudly
        # via the assertion below rather than AttributeError-ing first.
        self.release_calls.append(session_id)


@test("scheduler_fresh_session", "run_task forgets session between firings (issue #5)")
async def t_run_task_forgets_session(ctx: TestContext) -> None:
    from openagent.core.scheduler import Scheduler

    agent = _SpyAgent()
    scheduler = Scheduler(db=None, agent=agent)  # type: ignore[arg-type]

    task = {"id": "daily-dev", "name": "Daily Dev", "prompt": "do the work"}
    await scheduler.run_task(task)
    await scheduler.run_task(task)

    expected_sid = "scheduler:daily-dev"
    assert agent.run_calls == [
        (expected_sid, "do the work"),
        (expected_sid, "do the work"),
    ], agent.run_calls
    # The fix: forget_session is called between firings so the next run
    # spawns a fresh Claude CLI subprocess without --resume.
    assert agent.forget_calls == [expected_sid, expected_sid], agent.forget_calls
    # Regression guard: release_session (which preserves resume state)
    # must NOT be what the scheduler reaches for.
    assert agent.release_calls == [], agent.release_calls


class _RaisingAgent(_SpyAgent):
    async def run(self, *, message: str, user_id: str, session_id: str) -> str:
        self.run_calls.append((session_id, message))
        raise RuntimeError("boom")


@test("scheduler_fresh_session", "run_task forgets session even when the run raises")
async def t_run_task_forgets_on_error(ctx: TestContext) -> None:
    from openagent.core.scheduler import Scheduler

    agent = _RaisingAgent()
    scheduler = Scheduler(db=None, agent=agent)  # type: ignore[arg-type]

    task = {"id": "flaky", "name": "Flaky", "prompt": "try me"}
    # run_task swallows exceptions from the agent run — the finally
    # block must still wipe the resume state.
    await scheduler.run_task(task)

    assert agent.forget_calls == ["scheduler:flaky"], agent.forget_calls
    assert agent.release_calls == [], agent.release_calls
