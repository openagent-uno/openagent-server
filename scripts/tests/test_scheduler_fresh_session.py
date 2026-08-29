"""Regression guard for issue #5: a scheduled firing must never inherit a
prior firing's transcript.

Issue #5's root cause was that ``Scheduler.run_task`` *reused* one session id
per task and ``release_session`` kept the provider-native resume id on disk —
so the next firing resumed the previous one, and once that transcript crossed
the compaction threshold it summarised to *"all work already done"* and every
subsequent firing silently exited without re-running the prompt.

The current fix removes the root cause structurally: each firing runs as a
UNIQUE per-run child session (``scheduler:{task}:{run_id}``) via
``core.child_session.run_child_session``, so there is no shared session to
resume — the firing is durable (navigable + continuable, vision §7) and
``release_session`` (not ``forget_session``) frees only the live runtime.

A legacy safety hatch (``OPENAGENT_SCHEDULER_DURABLE_SESSIONS=0``) restores
the old reused-session + ``forget_session`` behavior; the second test pins it.

These tests run without spawning the real ``claude`` binary.
"""
from __future__ import annotations

import os
import uuid

from ._framework import TestContext, test


class _SpyAgent:
    """Minimal Agent stub recording how the scheduler drives + releases a run.

    ``run`` mirrors the real signature ``run_child_session`` calls it with
    (``model_override`` / ``author`` / ``on_status`` are passed through)."""

    name = "spy"
    model = None

    def __init__(self) -> None:
        self.run_calls: list[tuple[str, str]] = []
        self.forget_calls: list[str] = []
        self.release_calls: list[str] = []

    async def refresh_registries(self) -> None:
        return None

    async def run(self, *, message: str, user_id: str, session_id: str,
                  model_override=None, author=None, on_status=None) -> str:
        self.run_calls.append((session_id, message))
        return "ok"

    async def forget_session(self, session_id: str) -> None:
        self.forget_calls.append(session_id)

    async def release_session(self, session_id: str, *, model_override=None) -> None:
        self.release_calls.append(session_id)


@test("scheduler_fresh_session", "durable firings get unique per-run sessions (issue #5)")
async def t_run_task_unique_sessions(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler

    os.environ.pop("OPENAGENT_SCHEDULER_DURABLE_SESSIONS", None)  # default = durable
    agent = _SpyAgent()
    scheduler = Scheduler(db=None, agent=agent)  # type: ignore[arg-type]

    task = {"id": "daily-dev", "name": "Daily Dev", "prompt": "do the work"}
    await scheduler.run_task(task)
    await scheduler.run_task(task)

    # Two firings → two DISTINCT per-run session ids, both under the task root,
    # each seeded with the prompt. Distinctness is what removes the issue-#5
    # resume/compaction bug (there is no shared session to inherit).
    assert len(agent.run_calls) == 2, agent.run_calls
    sids = [sid for sid, _ in agent.run_calls]
    assert all(s.startswith("scheduler:daily-dev:") and s != "scheduler:daily-dev" for s in sids), sids
    assert sids[0] != sids[1], sids
    assert all(msg == "do the work" for _, msg in agent.run_calls), agent.run_calls
    # Durable: the row is kept (release frees the runtime); it is NOT wiped.
    assert agent.release_calls == sids, agent.release_calls
    assert agent.forget_calls == [], agent.forget_calls


@test("scheduler_fresh_session", "durable firing still releases when the run raises")
async def t_run_task_releases_on_error(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler

    os.environ.pop("OPENAGENT_SCHEDULER_DURABLE_SESSIONS", None)

    class _RaisingAgent(_SpyAgent):
        async def run(self, *, message: str, user_id: str, session_id: str,
                      model_override=None, author=None, on_status=None) -> str:
            self.run_calls.append((session_id, message))
            raise RuntimeError("boom")

    agent = _RaisingAgent()
    scheduler = Scheduler(db=None, agent=agent)  # type: ignore[arg-type]

    task = {"id": "flaky", "name": "Flaky", "prompt": "try me"}
    # run_task swallows agent-run exceptions; run_child_session's finally must
    # still release the live runtime for the (durable) per-run session.
    await scheduler.run_task(task)

    assert len(agent.run_calls) == 1, agent.run_calls
    sid = agent.run_calls[0][0]
    assert sid.startswith("scheduler:flaky:"), sid
    assert agent.release_calls == [sid], agent.release_calls
    assert agent.forget_calls == [], agent.forget_calls


@test("scheduler_fresh_session", "legacy hatch reuses one session + forgets it")
async def t_run_task_legacy_forget(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler

    os.environ["OPENAGENT_SCHEDULER_DURABLE_SESSIONS"] = "0"
    try:
        agent = _SpyAgent()
        scheduler = Scheduler(db=None, agent=agent)  # type: ignore[arg-type]
        task = {"id": "daily-dev", "name": "Daily Dev", "prompt": "do the work"}
        await scheduler.run_task(task)
        await scheduler.run_task(task)
        # Legacy: one reused per-task id, forgotten between firings.
        assert agent.run_calls == [
            ("scheduler:daily-dev", "do the work"),
            ("scheduler:daily-dev", "do the work"),
        ], agent.run_calls
        assert agent.forget_calls == ["scheduler:daily-dev", "scheduler:daily-dev"], agent.forget_calls
        assert agent.release_calls == [], agent.release_calls
    finally:
        os.environ.pop("OPENAGENT_SCHEDULER_DURABLE_SESSIONS", None)


@test(
    "scheduler_fresh_session",
    "self-hosted scheduled task is lean, strict-local and execution-level dry-run",
)
async def t_local_task_execution_profile(ctx: TestContext) -> None:
    from src.core.dry_run import is_dry_run
    from src.core.execution_profile import (
        lean_local_event_active,
        strict_local_only_active,
    )
    from src.core.scheduler import Scheduler
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"sched-local-{uuid.uuid4().hex[:8]}.db")

    class _ProfileAgent(_SpyAgent):
        def __init__(self) -> None:
            super().__init__()
            class _Model:
                @staticmethod
                def build_override_model(runtime_id: str) -> object:
                    assert runtime_id == "windows-local:qwen3-moe-local"
                    return object()

            self.model = _Model()
            self.profile: tuple[bool, bool, bool] | None = None

        async def run(self, *, message: str, user_id: str, session_id: str,
                      model_override=None, author=None, on_status=None) -> str:
            self.run_calls.append((session_id, message))
            self.profile = (
                lean_local_event_active(),
                strict_local_only_active(),
                is_dry_run(),
            )
            return "ok"

    db = MemoryDB(str(tmp_db))
    await db.connect()
    try:
        await db.upsert_provider(
            name="windows-local",
            framework="api-based",
            base_url="http://192.168.22.145:8099/v1",
        )
        task_id = await db.add_task(
            "gemello-dryrun-test", "* * * * *", "DRY RUN - read only",
            model="windows-local:qwen3-moe-local",
        )
        task = await db.get_task(task_id)
        agent = _ProfileAgent()
        await Scheduler(db=db, agent=agent).run_task(task)  # type: ignore[arg-type]
        assert agent.profile == (True, True, True), agent.profile
    finally:
        await db.close()
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass
