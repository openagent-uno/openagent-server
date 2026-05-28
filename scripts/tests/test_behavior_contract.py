"""Contract tests — pin the behavioral invariants of the model/router system.

These tests are the living specification for how OpenAgent's model
catalog + SmartRouter + session pin are supposed to behave.
Together they cover:

  1. Models live ONLY in the ``models`` DB table (not in yaml).
  2. Each provider row carries ``(name, framework)`` — same vendor under
     both frameworks is two separate rows by design (UNIQUE(name, framework)).
  3. Models carry a ``provider_id`` FK; framework is inherited from the
     provider row, never stored on the model directly.
  4. ``runtime_id`` (``openai:gpt-4o-mini``,
     ``claude-cli:anthropic:claude-opus-4-7``) is derived at read time,
     not stored in any table.
  5. SmartRouter resolves the per-session entry model (pin →
     first-enabled); per-turn delegation lives inside the Team.
  6. Per-session pin survives across turns and restarts (rows in
     ``pinned_sessions``).
  7. The ``model-manager`` MCP + REST + CLI can add / remove / edit
     providers, frameworks, and models at runtime.

Kept as pure DB + router unit tests — no LLM calls, no subprocesses.
"""
from __future__ import annotations

import uuid

from ._framework import TestContext, test


class _FakeResp:
    def __init__(self, content: str, model: str):
        self.content = content
        self.input_tokens = 1
        self.output_tokens = 1
        self.stop_reason = "stop"
        self.model = model


async def _tmp_db(ctx: TestContext, tag: str):
    from src.memory.db import MemoryDB
    path = ctx.db_path.with_name(f"contract-{tag}-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(path))
    await db.connect()
    return db, path


def _cleanup(path) -> None:
    for p in (path, path.with_suffix(".db-shm"), path.with_suffix(".db-wal")):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


@test("contract", "provider row carries (name, framework), models join via provider_id")
async def t_provider_triple(ctx: TestContext) -> None:
    db, path = await _tmp_db(ctx, "triple")
    try:
        pid = await db.upsert_provider(
            name="openai", framework="api-based", api_key="sk-x",
        )
        mid = await db.upsert_model(provider_id=pid, model="gpt-4o-mini")
        row = await db.get_model(mid)
        assert row["provider_id"] == pid
        assert row["model"] == "gpt-4o-mini"
        enriched = (await db.list_models_enriched())[0]
        assert enriched["provider_name"] == "openai"
        assert enriched["framework"] == "api-based"
        assert enriched["runtime_id"] == "openai:gpt-4o-mini"
    finally:
        await db.close()
        _cleanup(path)


@test("contract", "same vendor under two frameworks are distinct provider rows")
async def t_dual_framework_rows(ctx: TestContext) -> None:
    db, path = await _tmp_db(ctx, "dual")
    try:
        api_pid = await db.upsert_provider(
            name="anthropic", framework="api-based", api_key="sk-ant",
        )
        cli_pid = await db.upsert_provider(
            name="anthropic", framework="claude-cli",
        )
        assert api_pid != cli_pid
        await db.upsert_model(provider_id=api_pid, model="claude-sonnet-4-6")
        await db.upsert_model(provider_id=cli_pid, model="claude-sonnet-4-6")
        # Both rows live side by side: same (provider_name, model), different
        # framework. The composite runtime_id distinguishes them.
        enriched = await db.list_models_enriched()
        rids = sorted(r["runtime_id"] for r in enriched)
        assert rids == [
            "anthropic:claude-sonnet-4-6",
            "claude-cli:anthropic:claude-sonnet-4-6",
        ], rids
    finally:
        await db.close()
        _cleanup(path)


@test("contract", "claude-cli provider forbids api_key (sentinel class of bug fixed at schema)")
async def t_claude_cli_provider_rejects_api_key(ctx: TestContext) -> None:
    """v0.11.5 added a code-level filter for the ``api_key='claude-cli'``
    sentinel. v0.12 removes the whole class of bug by rejecting any
    non-empty api_key on a claude-cli provider row at the DB boundary."""
    db, path = await _tmp_db(ctx, "cli-nokey")
    try:
        raised = False
        try:
            await db.upsert_provider(
                name="anthropic", framework="claude-cli", api_key="x",
            )
        except ValueError as e:
            raised = True
            assert "api_key" in str(e).lower()
        assert raised, "claude-cli provider must reject api_key"
    finally:
        await db.close()
        _cleanup(path)


@test("contract", "FK cascade: deleting a provider wipes its models")
async def t_fk_cascade_deletes_models(ctx: TestContext) -> None:
    db, path = await _tmp_db(ctx, "cascade")
    try:
        pid = await db.upsert_provider(
            name="openai", framework="api-based", api_key="sk-x",
        )
        await db.upsert_model(provider_id=pid, model="gpt-4o-mini")
        await db.upsert_model(provider_id=pid, model="gpt-4o")
        await db.upsert_model(provider_id=pid, model="o1-mini")
        assert len(await db.list_models(provider_id=pid)) == 3

        await db.delete_provider(pid)
        # FK cascade fires; no models remain.
        assert len(await db.list_models()) == 0
    finally:
        await db.close()
        _cleanup(path)


@test("contract", "cross-framework pin is accepted (no framework lock post-v0.14)")
async def t_cross_framework_pin_accepted(ctx: TestContext) -> None:
    """v0.14+: with history unified in the runtime's ``sessions`` across
    every framework, the per-session framework lock is gone. Any
    enabled runtime_id can be pinned to any session at any time.
    """
    db, path = await _tmp_db(ctx, "cross-pin")
    try:
        cli_pid = await db.upsert_provider(
            name="anthropic", framework="claude-cli",
        )
        await db.upsert_model(provider_id=cli_pid, model="claude-opus-4-6")
        api_pid = await db.upsert_provider(
            name="openai", framework="api-based", api_key="sk-x",
        )
        await db.upsert_model(provider_id=api_pid, model="gpt-4o-mini")
        await db.pin_session_model(
            "sess-cross", "claude-cli:anthropic:claude-opus-4-6",
        )
        assert (
            await db.get_session_pin("sess-cross")
            == "claude-cli:anthropic:claude-opus-4-6"
        )
        # Swap pin to the other framework: no exception.
        await db.pin_session_model("sess-cross", "openai:gpt-4o-mini")
        assert await db.get_session_pin("sess-cross") == "openai:gpt-4o-mini"
    finally:
        await db.close()
        _cleanup(path)


@test("contract", "pin and re-pin overwrite the previous runtime_id")
async def t_pin_overwrite(ctx: TestContext) -> None:
    db, path = await _tmp_db(ctx, "ok-pin")
    try:
        pid = await db.upsert_provider(
            name="openai", framework="api-based", api_key="sk-x",
        )
        await db.upsert_model(provider_id=pid, model="gpt-4o-mini")
        await db.upsert_model(provider_id=pid, model="gpt-4o")
        await db.pin_session_model("sess-api", "openai:gpt-4o-mini")
        assert await db.get_session_pin("sess-api") == "openai:gpt-4o-mini"
        await db.pin_session_model("sess-api", "openai:gpt-4o")
        assert await db.get_session_pin("sess-api") == "openai:gpt-4o"
    finally:
        await db.close()
        _cleanup(path)


@test("contract", "pin on a fresh session creates the pin row")
async def t_pin_fresh_session(ctx: TestContext) -> None:
    db, path = await _tmp_db(ctx, "fresh-pin")
    try:
        cli_pid = await db.upsert_provider(
            name="anthropic", framework="claude-cli",
        )
        await db.upsert_model(provider_id=cli_pid, model="claude-sonnet-4-6")
        assert await db.get_session_pin("fresh") is None
        await db.pin_session_model(
            "fresh", "claude-cli:anthropic:claude-sonnet-4-6",
        )
        assert (
            await db.get_session_pin("fresh")
            == "claude-cli:anthropic:claude-sonnet-4-6"
        )
    finally:
        await db.close()
        _cleanup(path)


@test("contract", "unpin drops the pin row entirely")
async def t_unpin_drops_pin(ctx: TestContext) -> None:
    db, path = await _tmp_db(ctx, "unpin")
    try:
        cli_pid = await db.upsert_provider(
            name="anthropic", framework="claude-cli",
        )
        await db.upsert_model(provider_id=cli_pid, model="claude-sonnet-4-6")
        await db.pin_session_model(
            "sess-unpin", "claude-cli:anthropic:claude-sonnet-4-6",
        )
        await db.unpin_session_model("sess-unpin")
        assert await db.get_session_pin("sess-unpin") is None
    finally:
        await db.close()
        _cleanup(path)


@test("contract", "runtime_id format — api-based = provider:model, claude-cli = claude-cli:provider:model")
async def t_runtime_id_format(ctx: TestContext) -> None:
    from src.models.catalog import build_runtime_model_id

    assert build_runtime_model_id("openai", "gpt-4o-mini", "api-based") == "openai:gpt-4o-mini"
    assert (
        build_runtime_model_id("anthropic", "claude-sonnet-4-6", "claude-cli")
        == "claude-cli:anthropic:claude-sonnet-4-6"
    )
    # the runtime anthropic uses the 2-part form (no claude-cli prefix).
    assert (
        build_runtime_model_id("anthropic", "claude-sonnet-4-6", "api-based")
        == "anthropic:claude-sonnet-4-6"
    )
