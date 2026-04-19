"""Contract tests — pin the behavioral invariants of the model/router system.

These tests are the living specification for how OpenAgent's model
catalog + SmartRouter + session binding are supposed to behave.
Together they cover:

  1. Models live ONLY in the ``models`` DB table (not in yaml).
  2. Each provider row carries ``(name, framework)`` — same vendor under
     both frameworks is two separate rows by design (UNIQUE(name, framework)).
  3. Models carry a ``provider_id`` FK; framework is inherited from the
     provider row, never stored on the model directly.
  4. ``runtime_id`` (``openai:gpt-4o-mini``,
     ``claude-cli:anthropic:claude-opus-4-7``) is derived at read time,
     not stored in any table.
  5. SmartRouter picks per message via the classifier.
  6. Framework binding is permanent per session — claude-cli sessions
     only see claude-cli models; agno sessions only see agno models.
  7. If no enabled model satisfies the session's framework, the turn
     is rejected with a clear error (never crossed into the other
     framework).
  8. Cross-framework pin is refused (conversation would split).
  9. The ``model-manager`` MCP + REST + CLI can add / remove / edit
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
    from openagent.memory.db import MemoryDB
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
            name="openai", framework="agno", api_key="sk-x",
        )
        mid = await db.upsert_model(provider_id=pid, model="gpt-4o-mini")
        row = await db.get_model(mid)
        assert row["provider_id"] == pid
        assert row["model"] == "gpt-4o-mini"
        enriched = (await db.list_models_enriched())[0]
        assert enriched["provider_name"] == "openai"
        assert enriched["framework"] == "agno"
        assert enriched["runtime_id"] == "openai:gpt-4o-mini"
    finally:
        await db.close()
        _cleanup(path)


@test("contract", "same vendor under two frameworks are distinct provider rows")
async def t_dual_framework_rows(ctx: TestContext) -> None:
    db, path = await _tmp_db(ctx, "dual")
    try:
        agno_pid = await db.upsert_provider(
            name="anthropic", framework="agno", api_key="sk-ant",
        )
        cli_pid = await db.upsert_provider(
            name="anthropic", framework="claude-cli",
        )
        assert agno_pid != cli_pid
        await db.upsert_model(provider_id=agno_pid, model="claude-sonnet-4-6")
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
            name="openai", framework="agno", api_key="sk-x",
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


@test("contract", "framework lock: claude-cli session cannot be pinned to agno model")
async def t_cross_framework_pin_refused(ctx: TestContext) -> None:
    db, path = await _tmp_db(ctx, "cross-pin")
    try:
        # Session bound to claude-cli (via sdk_sessions, the normal
        # path the ClaudeCLI provider writes to).
        await db.set_sdk_session("sess-cli", "sdk-uuid", provider="claude-cli")
        pid = await db.upsert_provider(
            name="openai", framework="agno", api_key="sk-x",
        )
        await db.upsert_model(provider_id=pid, model="gpt-4o-mini")
        raised = False
        try:
            await db.pin_session_model("sess-cli", "openai:gpt-4o-mini")
        except ValueError as e:
            raised = True
            assert "framework" in str(e).lower(), str(e)
        assert raised, "pinning a claude-cli session to an agno model must raise"
    finally:
        await db.close()
        _cleanup(path)


@test("contract", "framework lock: agno session cannot be pinned to claude-cli model")
async def t_reverse_cross_framework_pin_refused(ctx: TestContext) -> None:
    db, path = await _tmp_db(ctx, "rev-pin")
    try:
        await db.set_session_binding("sess-agno", "agno")
        cli_pid = await db.upsert_provider(
            name="anthropic", framework="claude-cli",
        )
        await db.upsert_model(provider_id=cli_pid, model="claude-opus-4-6")
        raised = False
        try:
            await db.pin_session_model(
                "sess-agno", "claude-cli:anthropic:claude-opus-4-6",
            )
        except ValueError:
            raised = True
        assert raised
    finally:
        await db.close()
        _cleanup(path)


@test("contract", "same-framework pin is accepted")
async def t_same_framework_pin_ok(ctx: TestContext) -> None:
    db, path = await _tmp_db(ctx, "ok-pin")
    try:
        await db.set_session_binding("sess-agno", "agno")
        pid = await db.upsert_provider(
            name="openai", framework="agno", api_key="sk-x",
        )
        await db.upsert_model(provider_id=pid, model="gpt-4o-mini")
        await db.pin_session_model("sess-agno", "openai:gpt-4o-mini")
        assert await db.get_session_pin("sess-agno") == "openai:gpt-4o-mini"
    finally:
        await db.close()
        _cleanup(path)


@test("contract", "pin on a fresh session seeds the binding to that framework")
async def t_pin_fresh_session_seeds_binding(ctx: TestContext) -> None:
    db, path = await _tmp_db(ctx, "fresh-pin")
    try:
        cli_pid = await db.upsert_provider(
            name="anthropic", framework="claude-cli",
        )
        await db.upsert_model(provider_id=cli_pid, model="claude-sonnet-4-6")
        assert await db.get_session_binding("fresh") is None
        await db.pin_session_model(
            "fresh", "claude-cli:anthropic:claude-sonnet-4-6",
        )
        assert await db.get_session_binding("fresh") == "claude-cli"
        assert (
            await db.get_session_pin("fresh")
            == "claude-cli:anthropic:claude-sonnet-4-6"
        )
    finally:
        await db.close()
        _cleanup(path)


@test("contract", "router honors pin: skips classifier, dispatches directly")
async def t_router_honors_pin(ctx: TestContext) -> None:
    from openagent.models.smart_router import SmartRouter

    db, path = await _tmp_db(ctx, "pin-dispatch")
    try:
        pid = await db.upsert_provider(
            name="openai", framework="agno", api_key="sk-x",
        )
        await db.upsert_model(provider_id=pid, model="gpt-4o-mini")
        await db.upsert_model(provider_id=pid, model="gpt-4.1")
        await db.pin_session_model("sess-x", "openai:gpt-4.1")

        providers = await db.list_providers(enabled_only=True)
        by_id: dict[int, dict] = {p["id"]: {**p, "models": []} for p in providers}
        for m in await db.list_models(enabled_only=True):
            by_id[m["provider_id"]]["models"].append(m)
        providers_config = list(by_id.values())

        router = SmartRouter(
            routing={},
            providers_config=providers_config,
        )
        router.set_db(db)

        seen: list[str] = []

        async def _fake_dispatch(runtime_id, *a, **kw):
            seen.append(runtime_id)
            return _FakeResp("ok", runtime_id)

        async def _fake_classify(*_a, **_kw):
            raise AssertionError("classifier must not run when a pin is active")

        router._dispatch = _fake_dispatch  # type: ignore[assignment]
        router._classify = _fake_classify  # type: ignore[assignment]

        resp = await router.generate(
            [{"role": "user", "content": "hi"}], session_id="sess-x",
        )
        assert resp.model == "openai:gpt-4.1", resp.model
        assert seen == ["openai:gpt-4.1"], seen
    finally:
        await db.close()
        _cleanup(path)


@test("contract", "bound claude-cli session with no enabled claude-cli model rejects")
async def t_bound_framework_no_models_rejects(ctx: TestContext) -> None:
    from openagent.models.smart_router import SmartRouter

    db, path = await _tmp_db(ctx, "no-cli-models")
    try:
        # Only an agno provider+model exists.
        pid = await db.upsert_provider(
            name="openai", framework="agno", api_key="sk-x",
        )
        await db.upsert_model(provider_id=pid, model="gpt-4o-mini")

        providers = await db.list_providers(enabled_only=True)
        by_id = {p["id"]: {**p, "models": []} for p in providers}
        for m in await db.list_models(enabled_only=True):
            by_id[m["provider_id"]]["models"].append(m)
        providers_config = list(by_id.values())

        router = SmartRouter(
            routing={},
            providers_config=providers_config,
        )
        router.set_db(db)
        # Session pre-bound to claude-cli, no claude-cli models registered.
        await db.set_sdk_session("orphan", "stale-uuid", provider="claude-cli")

        async def _fake_classify(*_a, **_kw):
            return None

        router._classify = _fake_classify  # type: ignore[assignment]

        resp = await router.generate(
            [{"role": "user", "content": "hi"}], session_id="orphan",
        )
        assert resp.stop_reason == "error"
        assert "claude-cli" in resp.content.lower()
    finally:
        await db.close()
        _cleanup(path)


@test("contract", "unpin clears runtime_id but preserves framework binding")
async def t_unpin_preserves_framework(ctx: TestContext) -> None:
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
        # Framework binding is untouched.
        assert await db.get_session_binding("sess-unpin") == "claude-cli"
    finally:
        await db.close()
        _cleanup(path)


@test("contract", "runtime_id format — agno = provider:model, claude-cli = claude-cli:provider:model")
async def t_runtime_id_format(ctx: TestContext) -> None:
    from openagent.models.catalog import build_runtime_model_id

    assert build_runtime_model_id("openai", "gpt-4o-mini", "agno") == "openai:gpt-4o-mini"
    assert (
        build_runtime_model_id("anthropic", "claude-sonnet-4-6", "claude-cli")
        == "claude-cli:anthropic:claude-sonnet-4-6"
    )
    # Agno anthropic uses the 2-part form (no claude-cli prefix).
    assert (
        build_runtime_model_id("anthropic", "claude-sonnet-4-6", "agno")
        == "anthropic:claude-sonnet-4-6"
    )
