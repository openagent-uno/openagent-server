"""Model-independent execution envelopes for unattended scheduled tasks."""
from __future__ import annotations

from ._framework import TestContext, test


@test("scheduled_task_execution_policy", "validation is strict and canonical")
async def t_policy_validation(_ctx: TestContext) -> None:
    from src.core.execution_policy import (
        narrow_execution_policy,
        normalize_execution_policy,
    )

    assert normalize_execution_policy({
        "max_tool_calls": "6",
        "timeout_seconds": 90,
        "allowed_tool_families": ["replio", "replio", "billingbear"],
    }) == {
        "max_tool_calls": 6,
        "timeout_seconds": 90.0,
        "allowed_tool_families": ["replio", "billingbear"],
    }
    assert normalize_execution_policy({"allowed_tool_families": []}) == {
        "allowed_tool_families": []
    }
    for bad in (
        {"max_tool_calls": 0},
        {"timeout_seconds": 1},
        {"allowed_tool_families": "replio"},
        {"max_tools": 6},
    ):
        try:
            normalize_execution_policy(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid policy accepted: {bad!r}")

    assert narrow_execution_policy(
        {
            "max_tool_calls": 4,
            "timeout_seconds": 60,
            "allowed_tool_families": ["replio", "events-manager"],
        },
        {
            "max_tool_calls": 10,
            "timeout_seconds": 30,
            "allowed_tool_families": ["replio", "billingbear"],
        },
    ) == {
        "max_tool_calls": 4,
        "timeout_seconds": 30.0,
        "allowed_tool_families": ["replio"],
    }


@test("scheduled_task_execution_policy", "DB round-trip and clearing preserve defaults")
async def t_policy_db_roundtrip(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.memory.schedule import decorate_scheduled_task

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        tid = await db.add_task(
            "bounded", "0 */2 * * *", "score replies",
            execution_policy={
                "max_tool_calls": 6,
                "timeout_seconds": 90,
                "allowed_tool_families": ["replio"],
            },
        )
        raw = await db.get_task(tid)
        assert raw is not None and raw["execution_policy_json"], raw
        surfaced = decorate_scheduled_task(raw)
        assert surfaced["execution_policy"] == {
            "max_tool_calls": 6,
            "timeout_seconds": 90.0,
            "allowed_tool_families": ["replio"],
        }
        assert "execution_policy_json" not in surfaced

        await db.update_task(tid, execution_policy_json={})
        cleared = await db.get_task(tid)
        assert cleared is not None and cleared["execution_policy_json"] is None
        assert decorate_scheduled_task(cleared)["execution_policy"] == {}
    finally:
        await db.close()


@test("scheduled_task_execution_policy", "legacy tasks gain a NULL policy without shifting")
async def t_policy_migration(ctx: TestContext) -> None:
    import uuid
    import aiosqlite

    from src.memory.db import MemoryDB

    path = ctx.db_path.with_name(f"policy-migration-{uuid.uuid4().hex[:8]}.db")
    async with aiosqlite.connect(str(path)) as conn:
        await conn.execute(
            "CREATE TABLE scheduled_tasks ("
            "id TEXT PRIMARY KEY, name TEXT NOT NULL, cron_expression TEXT NOT NULL, "
            "prompt TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, "
            "last_run REAL, next_run REAL, model TEXT, timezone TEXT, "
            "created_at REAL NOT NULL, updated_at REAL NOT NULL)"
        )
        await conn.execute(
            "INSERT INTO scheduled_tasks "
            "(id,name,cron_expression,prompt,enabled,next_run,created_at,updated_at) "
            "VALUES ('legacy','legacy','0 9 * * *','p',1,123,0,0)"
        )
        await conn.commit()

    db = MemoryDB(str(path))
    await db.connect()
    try:
        row = await db.get_task("legacy")
        assert row is not None
        assert row["execution_policy_json"] is None
        assert row["cron_expression"] == "0 9 * * *"
        assert row["next_run"] == 123
    finally:
        await db.close()
        path.unlink(missing_ok=True)


@test("scheduled_task_execution_policy", "tool budget context overrides provider global")
async def t_policy_runtime_context(_ctx: TestContext) -> None:
    from src.core.execution_policy import reset_execution_policy, set_execution_policy
    from src.models.native_provider import _max_tool_calls_per_run

    before = _max_tool_calls_per_run()
    token = set_execution_policy({"max_tool_calls": 6})
    try:
        assert _max_tool_calls_per_run() == 6
    finally:
        reset_execution_policy(token)
    assert _max_tool_calls_per_run() == before


@test("scheduled_task_execution_policy", "runner cache separates capability envelopes")
async def t_policy_cache_key(_ctx: TestContext) -> None:
    from src.core.execution_policy import reset_execution_policy, set_execution_policy
    from src.core.tool_scope import reset_tool_allowlist, set_tool_allowlist
    from src.models.native_provider import _execution_cache_key

    plain, system = _execution_cache_key("system")
    assert plain == system == "system"

    policy_token = set_execution_policy({"max_tool_calls": 6})
    scope_token = set_tool_allowlist(["replio"])
    try:
        bounded, real_system = _execution_cache_key("system")
    finally:
        reset_tool_allowlist(scope_token)
        reset_execution_policy(policy_token)
    assert bounded != plain
    assert real_system == "system"
    assert "replio" in bounded and "max_tool_calls" in bounded


@test("scheduled_task_execution_policy", "scheduler installs and then resets the envelope")
async def t_scheduler_policy_scope(ctx: TestContext) -> None:
    import uuid
    from unittest.mock import AsyncMock, patch

    from src.core.execution_policy import current_execution_policy
    from src.core.scheduler import Scheduler
    from src.core.tool_scope import current_tool_allowlist
    from src.memory.db import MemoryDB

    class SpyAgent:
        name = "spy"
        model = None

        def __init__(self) -> None:
            self.seen_policy = None
            self.seen_tools = None

        async def refresh_registries(self) -> None:
            return None

        async def run(self, **_kwargs) -> str:
            self.seen_policy = current_execution_policy()
            self.seen_tools = current_tool_allowlist()
            return "done"

        async def release_session(self, _session_id, **_kwargs) -> None:
            return None

        async def forget_session(self, _session_id) -> None:
            return None

    path = ctx.db_path.with_name(f"policy-run-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(path))
    await db.connect()
    try:
        task_id = await db.add_task(
            "bounded", "0 */2 * * *", "score",
            execution_policy={
                "max_tool_calls": 6,
                "timeout_seconds": 30,
                "allowed_tool_families": ["replio"],
            },
        )
        task = await db.get_task(task_id)
        spy = SpyAgent()
        # An explicit operator policy is authoritative. The lean-local prompt
        # heuristic may under-detect a server (the scorer names tool methods,
        # not the word "replio") and must not intersect the grant down to zero.
        with (
            patch(
                "src.core.execution_profile.should_use_lean_local_scheduled_task",
                AsyncMock(return_value=True),
            ),
            patch(
                "src.core.execution_profile.lean_local_tool_families",
                return_value=["vault", "tool-search"],
            ),
        ):
            await Scheduler(db=db, agent=spy).run_task(task)  # type: ignore[arg-type]
        assert spy.seen_policy == {
            "max_tool_calls": 6,
            "timeout_seconds": 30.0,
            "allowed_tool_families": ["replio"],
        }
        assert spy.seen_tools == frozenset({"replio"})
        assert current_execution_policy() is None
        assert current_tool_allowlist() is None
        run = (await db.list_task_runs(task_id, limit=1))[0]
        assert run["status"] == "success", run
    finally:
        await db.close()
        path.unlink(missing_ok=True)


@test("scheduled_task_execution_policy",
      "un task schedulato ha un budget di tool call suo, non quello dell'evento")
async def t_lean_task_tool_budget(_ctx: TestContext) -> None:
    """Le due corsie condividono il profilo lean, non la forma del lavoro.

    La distinzione era gia' stata fatta per ``max_tokens`` e lasciata a meta'
    sul budget di tool call: dieci chiamate bastano a una risposta di supporto,
    a un audit no. Il 3-set-2026 `quality-scorer` veniva troncato a ogni giro
    (35 fallimenti su 50) ed `escalation-audit` non era MAI andato a buon fine,
    entrambi con "tool-call budget exhausted".
    """
    import os as _os

    from src.core.execution_profile import (
        lean_local_event_scope,
        lean_local_task_scope,
    )
    from src.models.native_provider import _max_tool_calls_per_run

    for key in ("OPENAGENT_LEAN_EVENT_MAX_TOOL_CALLS",
                "OPENAGENT_LEAN_TASK_MAX_TOOL_CALLS"):
        _os.environ.pop(key, None)
    try:
        # evento: la corsia stretta resta stretta
        with lean_local_event_scope(True), lean_local_task_scope(False):
            assert _max_tool_calls_per_run() == 10

        # task schedulato: stesso profilo lean, budget proprio
        with lean_local_event_scope(True), lean_local_task_scope(True):
            assert _max_tool_calls_per_run() == 40

        # e resta regolabile dall'operatore, senza toccare l'altra corsia
        _os.environ["OPENAGENT_LEAN_TASK_MAX_TOOL_CALLS"] = "25"
        with lean_local_event_scope(True), lean_local_task_scope(True):
            assert _max_tool_calls_per_run() == 25
        with lean_local_event_scope(True), lean_local_task_scope(False):
            assert _max_tool_calls_per_run() == 10
    finally:
        for key in ("OPENAGENT_LEAN_EVENT_MAX_TOOL_CALLS",
                    "OPENAGENT_LEAN_TASK_MAX_TOOL_CALLS"):
            _os.environ.pop(key, None)
