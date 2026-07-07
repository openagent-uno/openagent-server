"""Optional per-run model selection for scheduled tasks and delegation.

Two independent, opt-in "run this on a specific model" surfaces share the
same primitive (``run_child_session(model_id=...)``, which resolves NULL to
the agent's default/router model):

  - **Scheduled tasks** gained a ``scheduled_tasks.model`` column. These tests
    lock down the DB round-trip (add/update/clear) and the idempotent ALTER
    migration that adds the column to a pre-existing (old-shape) DB.
  - **Delegation** (`delegation.delegate_task`) made ``model_id`` optional.
    These tests drive the handler with a fake agent and assert: omitting the
    model builds NO override (runs on the default model), passing one builds
    the override and threads it into ``agent.run``, and a blank string is
    normalised to "default".
"""
from __future__ import annotations

from ._framework import TestContext, test


# ── scheduled_tasks.model column ──────────────────────────────────────


@test("scheduled_task_model", "add_task/update_task round-trip the optional model pin")
async def t_task_model_roundtrip(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        # Created WITHOUT a model → NULL (default/router model at fire time).
        tid_default = await db.add_task("nightly", "0 3 * * *", "do the thing")
        row = await db.get_task(tid_default)
        assert row is not None and row["model"] is None, row

        # Created WITH a model pin → persisted verbatim.
        tid_pinned = await db.add_task(
            "report", "0 9 * * *", "write the report",
            model="anthropic:claude-opus-4-8",
        )
        row = await db.get_task(tid_pinned)
        assert row["model"] == "anthropic:claude-opus-4-8", row

        # Update can change the pin …
        await db.update_task(tid_default, model="openai:gpt-4o")
        assert (await db.get_task(tid_default))["model"] == "openai:gpt-4o"

        # … and clear it back to the default (NULL).
        await db.update_task(tid_default, model=None)
        assert (await db.get_task(tid_default))["model"] is None
    finally:
        await db.close()


@test("scheduled_task_model", "migration adds the model column to an old-shape DB, preserving rows")
async def t_task_model_migration(ctx: TestContext) -> None:
    import os
    import tempfile
    import aiosqlite
    from src.memory.db import MemoryDB

    # A FRESH path (the shared ``ctx.db_path`` already carries the full current
    # schema, so it can't stand in for a legacy DB). mkdtemp gives an isolated
    # dir we seed with the old table shape.
    tmpdir = tempfile.mkdtemp(prefix="oa-migrate-")
    path = os.path.join(tmpdir, "legacy.db")

    # Simulate a pre-``model`` DB: the exact old ``scheduled_tasks`` shape with
    # a row already in it. ``CREATE TABLE`` (no IF NOT EXISTS) so this is
    # unambiguously the legacy table.
    conn = await aiosqlite.connect(path)
    await conn.execute(
        "CREATE TABLE scheduled_tasks ("
        "id TEXT PRIMARY KEY, name TEXT NOT NULL, cron_expression TEXT NOT NULL, "
        "prompt TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1, "
        "last_run REAL, next_run REAL, created_at REAL NOT NULL, updated_at REAL NOT NULL)"
    )
    await conn.execute(
        "INSERT INTO scheduled_tasks "
        "(id, name, cron_expression, prompt, enabled, created_at, updated_at) "
        "VALUES ('legacy1', 'old', '* * * * *', 'p', 1, 0, 0)",
    )
    await conn.commit()
    await conn.close()

    # Opening MemoryDB runs SCHEMA_SQL (IF NOT EXISTS leaves the old table) then
    # the ALTER migration adds ``model`` NULL — the legacy row survives.
    db = MemoryDB(path)
    await db.connect()
    try:
        row = await db.get_task("legacy1")
        assert row is not None, "legacy row lost across migration"
        assert "model" in row and row["model"] is None, row
        # And the migrated table now accepts a model pin.
        await db.update_task("legacy1", model="local:phi")
        assert (await db.get_task("legacy1"))["model"] == "local:phi"
    finally:
        await db.close()


# ── delegate_task: optional model_id ──────────────────────────────────


class _FakeModel:
    """Stand-in for ``agent.model`` (the dispatcher): records every
    ``build_override_model`` call and returns a sentinel override object."""

    def __init__(self) -> None:
        self.built: list[str] = []

    def build_override_model(self, runtime_id: str) -> str:
        self.built.append(runtime_id)
        return f"OVERRIDE::{runtime_id}"


class _FakeAgent:
    name = "spy"

    def __init__(self) -> None:
        self.model = _FakeModel()
        self.runs: list[dict] = []

    async def run(self, *, message, user_id, session_id,
                  model_override=None, author=None, on_status=None) -> str:
        self.runs.append({"session_id": session_id, "model_override": model_override})
        return "ANSWER"

    async def release_session(self, session_id, *, model_override=None) -> None:
        pass


class _FakeDB:
    """Minimal DB surface ``run_child_session`` touches (metadata only)."""

    async def get_session(self, sid):
        return {"client_id": "owner-h"}

    async def primary_owner_handle(self):
        return "owner-h"

    async def upsert_session(self, sid, **kwargs):
        return None


async def _run_delegate(task: str, model_id=None):
    """Drive ``handlers.delegate_task`` with a fresh fake context, returning
    ``(result, agent)`` so callers can assert on both."""
    from src.mcp.servers.delegation import handlers

    agent = _FakeAgent()
    db = _FakeDB()
    tokens = handlers.install_context(
        session_id="sess-1", pool=None, db=db, dispatcher=None,
        agent=agent, owner_handle="owner-h",
    )
    try:
        result = await handlers.delegate_task(task=task, model_id=model_id)
    finally:
        handlers.reset_context(tokens)
    return result, agent


@test("delegation", "delegate_task without model_id runs on the default model (no override built)")
async def t_delegate_default_model(ctx: TestContext) -> None:
    result, agent = await _run_delegate("summarise the doc")

    assert result["status"] == "ok", result
    assert result["answer"] == "ANSWER", result
    # No model requested → no override constructed, agent.run got model_override=None.
    assert result["model_id"] is None, result
    assert agent.model.built == [], agent.model.built
    assert agent.runs and agent.runs[-1]["model_override"] is None, agent.runs
    # Still a real child session (deep-linkable card).
    assert result.get("child_session_id"), result


@test("delegation", "delegate_task with model_id builds the override and threads it into the run")
async def t_delegate_pinned_model(ctx: TestContext) -> None:
    result, agent = await _run_delegate("write code", model_id="anthropic:claude-opus-4-8")

    assert result["status"] == "ok", result
    assert result["model_id"] == "anthropic:claude-opus-4-8", result
    assert agent.model.built == ["anthropic:claude-opus-4-8"], agent.model.built
    assert agent.runs[-1]["model_override"] == "OVERRIDE::anthropic:claude-opus-4-8", agent.runs


@test("delegation", "delegate_task normalises a blank model_id to the default model")
async def t_delegate_blank_model(ctx: TestContext) -> None:
    result, agent = await _run_delegate("do it", model_id="   ")

    assert result["model_id"] is None, result
    assert agent.model.built == [], agent.model.built
    assert agent.runs[-1]["model_override"] is None, agent.runs
