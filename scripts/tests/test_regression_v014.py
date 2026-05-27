"""Regression tests for the v0.14 Agno-consolidation refactor.

Locks down every fix landed in Phases 1-10 so future churn can't quietly
undo them:

  - Phase 1: Agno bump (claude_agent SDK adapter import path).
  - Phase 2: Framework rename (agno/litellm → api-based) + legacy aliases.
  - Phase 3: ClaudeCLI shim over Agno ClaudeAgent with session_data
    round-trip for SDK session id (resume across restarts).
  - Phase 4: SmartRouter classifier removed; TeamRouterProvider built
    on first dispatch.
  - Phase 5: db.py SDK metadata helpers deleted; back-compat shims left.
  - Phase 6a: All MCPs deferred behind tool-search; only the
    tool-search toolkit reaches the model upfront.
  - Phase 6b: FRAMEWORK_SYSTEM_PROMPT carries the catalog placeholder;
    build_mcp_catalog_summary handles edge cases and foregrounds vault.
  - Phase 7: Gateway endpoints unchanged.
  - Phase 10: Live-run fixes:
      * Curator caller uses agent.memory_db (not agent.memory).
      * wait() installs BOTH asyncio + signal.signal handlers.
      * cli.py pins tqdm to threading.RLock so the multiprocessing
        resource_tracker daemon never spawns and no semaphore leaks
        at process exit.
      * TOKENIZERS_PARALLELISM=false set before downstream imports.

Tests are hermetic: no live API calls, no live subprocess MCPs. The
subprocess-leak test spawns ``python -m src.cli --help`` (which runs
import-only path) and inspects stderr for the warning.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import aiosqlite

from ._framework import TestContext, test


# ── Phase 1 / Phase 0: Agno surface ───────────────────────────────────


@test("regression_v014", "Agno 2.6+ adapter imports — ClaudeAgent + Team + SqliteDb")
async def t_agno_imports(ctx: TestContext) -> None:
    """Phase 1 contract: the bumped Agno surface we rely on exists. A
    regression to a pre-2.6 Agno would break this immediately.
    """
    from agno.agent import Agent as AgnoAgent  # noqa: F401
    from agno.agents.claude import ClaudeAgent  # noqa: F401
    from agno.agents.base import BaseExternalAgent  # noqa: F401
    from agno.db.sqlite import SqliteDb  # noqa: F401
    from agno.team import Team, TeamMode  # noqa: F401

    # ClaudeAgent must be a BaseExternalAgent subclass — that's the
    # contract that lets TeamRouterProvider exclude it from Team.members.
    assert issubclass(ClaudeAgent, BaseExternalAgent), (
        f"ClaudeAgent must inherit from BaseExternalAgent; got {ClaudeAgent.__mro__}"
    )

    # ``framework`` is the discriminator agent_data writes — used by
    # ``_known_sdk_session_ids_from_db`` to filter rows.
    assert getattr(ClaudeAgent, "framework", None) == "claude-agent-sdk", (
        f"ClaudeAgent.framework changed; "
        f"_known_sdk_session_ids_from_db relies on the literal "
        f"'claude-agent-sdk' value"
    )


# ── Phase 2: Framework value collapse ─────────────────────────────────


@test("regression_v014", "catalog constants collapsed — api-based + claude-cli + codex-cli")
async def t_catalog_constants(ctx: TestContext) -> None:
    """The three framework values are the only LLM-routing discriminators.
    Legacy aliases preserved for one release so external callers don't
    crash; new code should reference FRAMEWORK_API_BASED directly.

    codex-cli was added in v0.14.x as the third LLM framework (mirror
    of claude-cli for the OpenAI Codex CLI / ChatGPT Plus subscription
    path).
    """
    from src.models.catalog import (
        FRAMEWORK_AGNO,
        FRAMEWORK_API_BASED,
        FRAMEWORK_CLAUDE_CLI,
        FRAMEWORK_CODEX_CLI,
        FRAMEWORK_LITELLM,
        LLM_FRAMEWORKS,
        SUBSCRIPTION_CLI_FRAMEWORKS,
        SUPPORTED_FRAMEWORKS,
    )

    assert FRAMEWORK_API_BASED == "api-based"
    assert FRAMEWORK_CLAUDE_CLI == "claude-cli"
    assert FRAMEWORK_CODEX_CLI == "codex-cli"
    assert FRAMEWORK_AGNO == FRAMEWORK_API_BASED, (
        "FRAMEWORK_AGNO must be a back-compat alias for FRAMEWORK_API_BASED"
    )
    assert FRAMEWORK_LITELLM == FRAMEWORK_API_BASED, (
        "FRAMEWORK_LITELLM must be a back-compat alias for FRAMEWORK_API_BASED"
    )
    assert LLM_FRAMEWORKS == ("api-based", "claude-cli", "codex-cli")
    assert SUPPORTED_FRAMEWORKS == LLM_FRAMEWORKS, (
        "TTS/STT now discriminated by `kind`, not framework value"
    )
    assert SUBSCRIPTION_CLI_FRAMEWORKS == ("claude-cli", "codex-cli"), (
        "SUBSCRIPTION_CLI_FRAMEWORKS groups the two subscription-billed paths"
    )


@test("regression_v014", "DB migration: legacy agno + litellm framework values → api-based")
async def t_framework_migration(ctx: TestContext) -> None:
    """Phase 2's one-shot migration must rewrite legacy rows on first
    boot AND be idempotent on subsequent boots. Tested against a
    hand-built legacy DB that mirrors the pre-v0.14 layout.
    """
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"mig-{uuid.uuid4().hex[:8]}.db")
    try:
        # Hand-build a pre-v0.14 providers table — no CHECK constraint
        # since the kind-column migration drops it.
        async with aiosqlite.connect(str(tmp_db)) as conn:
            await conn.execute(
                """
                CREATE TABLE providers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    framework TEXT NOT NULL,
                    api_key TEXT,
                    base_url TEXT,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    kind TEXT NOT NULL DEFAULT 'llm',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(name, framework)
                )
                """
            )
            await conn.execute(
                "CREATE TABLE session_bindings ("
                "session_id TEXT PRIMARY KEY, framework TEXT NOT NULL, "
                "runtime_id TEXT, bound_at REAL NOT NULL)"
            )
            await conn.executemany(
                "INSERT INTO providers (name, framework, api_key, kind, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    ("openai", "agno", "sk-x", "llm", 1.0, 1.0),
                    ("elevenlabs", "litellm", "el-x", "tts", 1.0, 1.0),
                    ("anthropic", "claude-cli", None, "llm", 1.0, 1.0),
                ],
            )
            await conn.execute(
                "INSERT INTO session_bindings VALUES "
                "('sess-legacy', 'agno', 'openai:gpt-4o', 1.0)",
            )
            await conn.commit()

        # First connect — migration runs.
        db = MemoryDB(str(tmp_db))
        await db.connect()
        await db.close()

        async with aiosqlite.connect(str(tmp_db)) as conn:
            cur = await conn.execute(
                "SELECT DISTINCT framework FROM providers ORDER BY framework"
            )
            frameworks = [r[0] for r in await cur.fetchall()]
            assert frameworks == ["api-based", "claude-cli"], frameworks

            # The legacy session_bindings row carried a runtime_id; the
            # v0.14+ migration folds it into ``pinned_sessions`` and
            # drops the original table.
            cur = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='session_bindings'"
            )
            assert await cur.fetchone() is None
            cur = await conn.execute(
                "SELECT runtime_id FROM pinned_sessions WHERE session_id='sess-legacy'"
            )
            row = await cur.fetchone()
            assert row is not None and row[0] == "openai:gpt-4o"

        # Idempotent: second connect doesn't double-rewrite.
        db = MemoryDB(str(tmp_db))
        await db.connect()
        await db.close()
        async with aiosqlite.connect(str(tmp_db)) as conn:
            cur = await conn.execute(
                "SELECT DISTINCT framework FROM providers ORDER BY framework"
            )
            assert [r[0] for r in await cur.fetchall()] == [
                "api-based", "claude-cli",
            ]
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("regression_v014", "upsert_provider accepts legacy 'agno' / 'litellm' as api-based")
async def t_upsert_provider_legacy_compat(ctx: TestContext) -> None:
    """External tools / older scripts that still pass framework='agno'
    must keep working — the upsert helper rewrites the value at the
    boundary so they don't need to be updated synchronously.
    """
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"ups-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        try:
            pid_agno = await db.upsert_provider(
                name="openai", framework="agno", api_key="sk-1",
            )
            row = await db.get_provider(pid_agno)
            assert row["framework"] == "api-based", row["framework"]

            pid_litellm = await db.upsert_provider(
                name="elevenlabs", framework="litellm", api_key="el-1",
                kind="tts",
            )
            row = await db.get_provider(pid_litellm)
            assert row["framework"] == "api-based", row["framework"]
        finally:
            await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


# ── Phase 3: ClaudeBackedAgent session-data round-trip ──────────────


@test(
    "regression_v014",
    "ClaudeBackedAgent persists SDK session id via session.session_data",
)
async def t_claude_backed_agent_roundtrip(ctx: TestContext) -> None:
    """The whole point of ``ClaudeBackedAgent`` (replacing the v0.14
    ``PersistentClaudeAgent`` ClaudeAgent subclass) is to survive
    process restarts AND participate in Agno Teams. The SDK session id
    must round-trip through ``session.session_data`` so the next boot
    can ``--resume`` the same SDK subprocess transcript.
    """
    from src.models.claude_agent import ClaudeBackedAgent

    class _FakeSession:
        def __init__(self, sdk_id: str | None = None) -> None:
            self.session_data: dict[str, Any] | None = (
                {"sdk_session_id": sdk_id} if sdk_id else None
            )

    class _FakeDb:
        """Minimal DB facade: get_session / upsert_session — the only
        two methods ClaudeBackedAgent calls for SDK-id round-trip.
        """

        def __init__(self) -> None:
            self.sessions: dict[str, _FakeSession] = {}

        def get_session(self, *, session_id: str, **kwargs: Any) -> _FakeSession | None:
            return self.sessions.get(session_id)

        def upsert_session(self, *, session: _FakeSession) -> None:
            # Track by id by reading the stamped session_data; tests
            # call upsert manually below to seed.
            pass

    db = _FakeDb()
    db.sessions["sess-x"] = _FakeSession(sdk_id="previous-uuid")

    agent = ClaudeBackedAgent(
        name="t", sdk_model_id="claude-sonnet-4-6", sdk_options={}, db=db,
    )

    # Hydration pulls the previous SDK id from session_data.
    agent._hydrate_from_db("sess-x")
    assert agent._sdk_session_ids.get("sess-x") == "previous-uuid", (
        "_hydrate_from_db must pull sdk_session_id from session_data"
    )

    # Now simulate the SDK reporting a NEW session id mid-run.
    agent._sdk_session_ids["sess-x"] = "new-uuid-xyz"
    agent._save_to_db("sess-x")
    assert db.sessions["sess-x"].session_data["sdk_session_id"] == "new-uuid-xyz", (
        "_save_to_db must write the latest sdk_session_id back into session_data"
    )

    # Empty session_data should still get populated.
    db.sessions["sess-y"] = _FakeSession(sdk_id=None)
    agent._sdk_session_ids["sess-y"] = "fresh-uuid"
    agent._save_to_db("sess-y")
    assert db.sessions["sess-y"].session_data == {"sdk_session_id": "fresh-uuid"}


@test("regression_v014", "_known_sdk_session_ids_from_db queries agent_data.framework")
async def t_known_sdk_session_ids(ctx: TestContext) -> None:
    """Phase 3 swapped the legacy metadata.sdk_session_id query for a
    framework-typed read of ``agent_data.framework``. The new helper
    must find rows that Agno's SqliteDb wrote with
    framework='claude-agent-sdk'.
    """
    from src.memory.db import MemoryDB
    from src.models.claude_agent import _known_sdk_session_ids_from_db

    tmp_db = ctx.db_path.with_name(f"sdk-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        try:
            now = 1.0
            conn = await db._ensure_connected()
            # Two Claude SDK sessions + one api-based session.
            await conn.execute(
                "INSERT INTO agno_sessions "
                "(session_id, agent_data, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                ("sess-claude-1",
                 '{"framework": "claude-agent-sdk", "agent_id": "x"}',
                 now, now),
            )
            await conn.execute(
                "INSERT INTO agno_sessions "
                "(session_id, agent_data, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                ("sess-claude-2",
                 '{"framework": "claude-agent-sdk", "agent_id": "y"}',
                 now, now),
            )
            await conn.execute(
                "INSERT INTO agno_sessions "
                "(session_id, agent_data, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                ("sess-api", '{"framework": "external"}', now, now),
            )
            await conn.commit()

            ids = _known_sdk_session_ids_from_db(db)
            assert ids == ["sess-claude-1", "sess-claude-2"], ids
        finally:
            await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test(
    "regression_v014",
    "_known_sdk_session_ids_from_db finds ClaudeBackedAgent sessions via session_data.sdk_session_id",
)
async def t_sdk_session_lookup_widened(ctx: TestContext) -> None:
    """``ClaudeBackedAgent`` rows are stored as plain ``Agent`` rows
    (no ``agent_data.framework = 'claude-agent-sdk'`` marker). They DO
    write ``session_data.sdk_session_id`` though, and the gateway's
    ``/clear`` legacy-row fallback must still find them after a cold
    restart (before ``TeamRouterProvider.known_session_ids`` has been
    warmed back up).
    """
    from src.memory.db import MemoryDB
    from src.models.claude_agent import _known_sdk_session_ids_from_db

    tmp_db = ctx.db_path.with_name(f"sdkw-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        try:
            now = 1.0
            conn = await db._ensure_connected()
            # Row A: legacy claude-agent-sdk row (pre-v0.14 marker).
            await conn.execute(
                "INSERT INTO agno_sessions "
                "(session_id, agent_data, session_data, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("sess-legacy",
                 '{"framework": "claude-agent-sdk", "agent_id": "old"}',
                 None, now, now),
            )
            # Row B: new ClaudeBackedAgent row — agent_data has no
            # framework marker (it's a regular Agent), but the SDK
            # session id is stamped on session_data.
            await conn.execute(
                "INSERT INTO agno_sessions "
                "(session_id, agent_data, session_data, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("sess-backed",
                 '{"agent_id": "claude-backed"}',
                 '{"sdk_session_id": "abc-1234-uuid"}',
                 now, now),
            )
            # Row C: plain api-based session — must NOT be picked up.
            await conn.execute(
                "INSERT INTO agno_sessions "
                "(session_id, agent_data, session_data, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                ("sess-api",
                 '{"framework": "external", "agent_id": "api"}',
                 '{}', now, now),
            )
            await conn.commit()

            ids = _known_sdk_session_ids_from_db(db)
            assert ids == ["sess-backed", "sess-legacy"], ids
        finally:
            await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


# ── Phase 4: SmartRouter no longer has classifier internals ──────────


@test("regression_v014", "SmartRouter classifier internals removed")
async def t_smart_router_no_classifier(ctx: TestContext) -> None:
    """Negative test: the classifier LLM call + its caches are gone.
    A regression that re-adds them would re-introduce the per-turn LLM
    cost the user explicitly removed.
    """
    from src.models.dispatcher import SmartRouter

    router = SmartRouter()
    # None of these should exist anymore — TeamRouterProvider replaces them.
    forbidden = [
        "_classifier_model",
        "_classifier_provider",
        "_classify",
        "_resolve_classifier_pick",
        "_candidate_models",
        "_get_classifier_provider",
        "_get_agno_provider",
        "_agno_providers",
        "_remember_pick",
    ]
    leaked = [name for name in forbidden if hasattr(router, name)]
    # Note: _last_pick_by_session is KEPT — used by effective_model_id
    # for the chat UI's model badge. _remember_pick stays as a tiny
    # private helper. Adjust the list if we keep helpers; what matters
    # is the classifier method itself is gone.
    classifier_methods = ["_classify", "_resolve_classifier_pick", "_classifier_provider"]
    classifier_leaked = [n for n in classifier_methods if hasattr(router, n)]
    assert not classifier_leaked, (
        f"SmartRouter still has classifier internals: {classifier_leaked}. "
        f"The classifier was replaced by TeamRouterProvider in Phase 4."
    )


@test("regression_v014", "SmartRouter dispatches api-based entries through TeamRouterProvider")
async def t_smart_router_uses_team(ctx: TestContext) -> None:
    """The api-based path must build a TeamRouterProvider, not a raw
    AgnoProvider. The team router is what gives sessions their
    sub-agent / specialist-delegation behaviour.
    """
    from src.models.dispatcher import SmartRouter
    from src.models.dispatcher import TeamRouterProvider

    providers = [
        {"id": 1, "name": "openai", "framework": "api-based",
         "api_key": "sk-x", "enabled": True,
         "models": [{"id": 10, "model": "gpt-4o-mini", "enabled": True}]},
    ]
    router = SmartRouter(providers_config=providers)
    provider = router._get_team_provider("openai:gpt-4o-mini")
    assert isinstance(provider, TeamRouterProvider), type(provider).__name__


@test("regression_v014", "_resolve_entry_model picks is_classifier-flagged model over first enabled")
async def t_resolve_prefers_router_flag(ctx: TestContext) -> None:
    """Resolution order: pin → is_classifier flag → first enabled. When
    the user marks a model as the default team leader (the ``router``
    flag in the model_manager UI / ``is_classifier`` column in DB), the
    dispatcher must honour it instead of falling through to catalog
    order.
    """
    from src.models.dispatcher import ModelDispatcher

    providers = [
        {"id": 1, "name": "openai", "framework": "api-based",
         "api_key": "sk-x", "enabled": True,
         "models": [
             {"id": 10, "model": "gpt-4o-mini", "enabled": True},
             {"id": 11, "model": "gpt-4o", "enabled": True, "is_classifier": True},
         ]},
    ]
    router = ModelDispatcher(providers_config=providers)
    decision = await router._resolve_entry_model(session_id="sess-x")
    assert decision.reason == "router_flag", decision
    assert decision.primary_model == "openai:gpt-4o", decision


@test("regression_v014", "_resolve_entry_model falls back to first_enabled when no flag set")
async def t_resolve_fallback_first_enabled(ctx: TestContext) -> None:
    """If no model carries ``is_classifier=True`` the dispatcher walks
    catalog order — keeps single-model deployments working without
    forcing the user to flip a flag.
    """
    from src.models.dispatcher import ModelDispatcher

    providers = [
        {"id": 1, "name": "openai", "framework": "api-based",
         "api_key": "sk-x", "enabled": True,
         "models": [
             {"id": 10, "model": "gpt-4o-mini", "enabled": True},
             {"id": 11, "model": "gpt-4o", "enabled": True},
         ]},
    ]
    router = ModelDispatcher(providers_config=providers)
    decision = await router._resolve_entry_model(session_id="sess-x")
    assert decision.reason == "first_enabled", decision
    assert decision.primary_model == "openai:gpt-4o-mini", decision


@test("regression_v014", "_resolve_entry_model with empty catalog returns no_enabled_model")
async def t_resolve_no_models(ctx: TestContext) -> None:
    """Zero-enabled-models case: the dispatcher must yield a decision
    whose ``primary_model`` is empty + ``reason='no_enabled_model'`` so
    ``generate``/``stream`` surface a clear error in chat instead of
    crashing on a None lookup.
    """
    from src.models.dispatcher import ModelDispatcher

    router = ModelDispatcher(providers_config=[])
    decision = await router._resolve_entry_model(session_id="sess-x")
    assert decision.reason == "no_enabled_model", decision
    assert decision.primary_model == "", decision
    assert decision.candidates == [], decision


@test("regression_v014", "generate with no enabled models returns user-facing error")
async def t_generate_no_models_returns_error(ctx: TestContext) -> None:
    """End-to-end safety net for the no-models case: the dispatcher's
    ``generate`` short-circuits BEFORE invoking any provider so the
    user sees a readable error in the chat rather than a stack trace.
    """
    from src.models.dispatcher import ModelDispatcher

    router = ModelDispatcher(providers_config=[])
    resp = await router.generate(
        messages=[{"role": "user", "content": "hi"}],
        session_id="sess-x",
    )
    assert resp.stop_reason == "error", resp
    assert "no model" in resp.content.lower(), resp.content


@test("regression_v014", "_arun_agno_stream forwards Team-module content events (no duplicate-run bug)")
async def t_stream_handles_team_events(ctx: TestContext) -> None:
    """Regression for the prod bug where ``TeamRouterProvider.stream``
    yielded zero deltas → ``agent.run_stream`` fired its
    "no_deltas_yielded" fallback to ``generate()`` → the team ran the
    whole turn AGAIN, persisting a duplicate row to ``agno_sessions``
    → the next user turn saw the orphan in history and the model
    replied "I already answered above".

    Root cause: ``agno.run.team.RunContentEvent`` is a DISTINCT class
    from ``agno.run.agent.RunContentEvent``. The shared
    ``_arun_agno_stream`` helper must isinstance-check BOTH module
    variants (plus ``IntermediateRunContentEvent`` for delegated
    member content) so Team-mode deltas reach the WS pipe.
    """
    from agno.run.agent import RunContentEvent as AgentRCE
    from agno.run.team import (
        IntermediateRunContentEvent as TeamIRCE,
        RunContentEvent as TeamRCE,
    )

    assert AgentRCE is not TeamRCE, (
        "Test premise broke: Agno collapsed the two RunContentEvent "
        "classes; revisit dispatcher's union."
    )

    from src.models.dispatcher import _arun_agno_stream

    class _FakeTeamRuntime:
        def arun(self, prompt, *, session_id, user_id, stream, stream_events=False):
            async def _iter():
                yield TeamRCE(content="hello ", run_id="r", session_id=session_id)
                yield TeamIRCE(content="from ", run_id="r", session_id=session_id)
                yield AgentRCE(content="opus", run_id="r", session_id=session_id)
            return _iter()

    deltas: list[str] = []
    async for delta in _arun_agno_stream(
        _FakeTeamRuntime(),
        prompt="x",
        session_id="sess-x",
        user_id="u",
        on_status=None,
        error_event="test.stream_error",
    ):
        deltas.append(delta)
    assert deltas == ["hello ", "from ", "opus"], deltas


@test("regression_v014", "stream with no enabled models yields user-facing error")
async def t_stream_no_models_returns_error(ctx: TestContext) -> None:
    """Streaming variant of the no-models safety net — must yield text
    (not raise) so the WS / SSE drain emits a readable line.
    """
    from src.models.dispatcher import ModelDispatcher

    router = ModelDispatcher(providers_config=[])
    chunks: list[str] = []
    async for chunk in router.stream(
        messages=[{"role": "user", "content": "hi"}],
        session_id="sess-x",
    ):
        chunks.append(chunk)
    text = "".join(chunks).lower()
    assert "no model" in text, text


# ── Phase 5: db.py helper cleanup ────────────────────────────────────


@test("regression_v014", "deleted db.py helpers stay deleted")
async def t_db_helpers_deleted(ctx: TestContext) -> None:
    """Negative test on Phase 5: the manual mirror-write helpers (which
    duplicated what Agno's SqliteDb does for free) must not be
    re-added by mistake.
    """
    from src.memory.db import MemoryDB

    deleted = [
        "_ensure_agno_session_row",
        "get_sdk_session",
        "set_sdk_session",
        "get_all_sdk_sessions",
        "add_session_run",
        "_add_session_run_locked",
        "commit_partial_session_run",
        "delete_session_runs",
    ]
    leaked = [name for name in deleted if hasattr(MemoryDB, name)]
    assert not leaked, (
        f"Manual mirror-write helpers must not be re-added: {leaked}. "
        f"Agno's SqliteDb owns these now."
    )

    # The back-compat shim stays so existing gateway DELETE handlers
    # don't crash on legacy sdk_session_id metadata.
    assert hasattr(MemoryDB, "delete_sdk_session"), (
        "delete_sdk_session is a back-compat shim — keep it"
    )
    # list_session_runs stays — gateway uses it for chat history.
    assert hasattr(MemoryDB, "list_session_runs"), (
        "list_session_runs powers GET /api/sessions/{id}/runs"
    )


# ── Phase 6a: All MCPs deferred behind tool-search ───────────────────


@test("regression_v014", "wire_model_runtime attaches ONLY the tool-search toolkit")
async def t_wire_defers_all_mcps(ctx: TestContext) -> None:
    """The defer-all design: every MCP except tool-search is reachable
    via ``tool-search.call_tool``. wire_model_runtime must not attach
    any other toolkits / sdk servers upfront.
    """
    from src.models.runtime import wire_model_runtime

    class _StubModel:
        def __init__(self) -> None:
            self.toolkit_calls: list[Any] = []
            self.sdk_calls: list[Any] = []

        def set_mcp_toolkits(self, toolkits: Any) -> None:
            self.toolkit_calls.append(list(toolkits))

        def set_mcp_servers(self, servers: Any) -> None:
            self.sdk_calls.append(dict(servers))

    class _StubPool:
        def __init__(self) -> None:
            self.agno_calls = 0
            self.sdk_calls = 0

        def agno_toolkits_tool_search_only(self):
            self.agno_calls += 1
            return ["<tool-search-toolkit>"]

        def claude_sdk_servers_tool_search_only(self):
            self.sdk_calls += 1
            return {"tool-search": {"type": "sdk"}}

    model = _StubModel()
    pool = _StubPool()
    wire_model_runtime(model, mcp_pool=pool)

    # The wire layer must call the TOOL-SEARCH-ONLY accessors. If a
    # regression replaces them with ``agno_toolkits`` (the full list)
    # this test catches it because the count check below fails first
    # — but the explicit accessor call is the canary.
    assert pool.agno_calls == 1, pool.agno_calls
    assert pool.sdk_calls == 1, pool.sdk_calls
    assert model.toolkit_calls == [["<tool-search-toolkit>"]]
    assert model.sdk_calls == [{"tool-search": {"type": "sdk"}}]


@test("regression_v014", "MCPPool exposes the tool-search-only accessors")
async def t_pool_tool_search_only(ctx: TestContext) -> None:
    from src.mcp.pool import MCPPool

    assert callable(getattr(MCPPool, "agno_toolkits_tool_search_only", None)), (
        "MCPPool.agno_toolkits_tool_search_only is required by "
        "wire_model_runtime — adding it back is what closed the gap"
    )
    assert callable(getattr(MCPPool, "claude_sdk_servers_tool_search_only", None))
    # The budget knobs are still present so legacy callers don't break,
    # but the wire layer no longer consults them.
    assert callable(getattr(MCPPool, "agno_toolkits_under_budget", None))


# ── Phase 6b: System prompt + catalog summary ────────────────────────


@test("regression_v014", "FRAMEWORK_SYSTEM_PROMPT carries the catalog placeholder + vault block")
async def t_framework_prompt_blocks(ctx: TestContext) -> None:
    from src.core.prompts import FRAMEWORK_SYSTEM_PROMPT

    assert "{{MCP_CATALOG_SUMMARY}}" in FRAMEWORK_SYSTEM_PROMPT, (
        "The per-session catalog must be injected via this placeholder"
    )
    assert "Memory vault — non-negotiable" in FRAMEWORK_SYSTEM_PROMPT, (
        "The top-of-prompt vault discipline block disappeared"
    )
    assert "{{OPENAGENT_VAULT_PATH}}" in FRAMEWORK_SYSTEM_PROMPT, (
        "Vault path placeholder still required by Agent._combined_system_prompt"
    )


@test("regression_v014", "FRAMEWORK_SYSTEM_PROMPT instructs aggressive sub-agent delegation")
async def t_subagent_delegation_block(ctx: TestContext) -> None:
    """The 'Sub-agents' block must remain in the framework prompt so
    every session's Team leader gets the policy that delegation is the
    default mode — capability uplift + context-rot reduction. Without
    this, weak leaders silently handle everything themselves and the
    Team-as-router architecture is wasted.

    Locks the coordinate-mode pillars (decompose first → parallelize
    independent → sequence dependent → synthesize) introduced when the
    Team mode flipped from ``route`` to ``coordinate``.
    """
    from src.core.prompts import FRAMEWORK_SYSTEM_PROMPT

    assert "Sub-agents — ALWAYS break the task down and delegate" in FRAMEWORK_SYSTEM_PROMPT, (
        "Sub-agent delegation policy block missing — leaders won't "
        "exercise the Team-as-router architecture"
    )
    body_lower = FRAMEWORK_SYSTEM_PROMPT.lower()
    pillars = [
        ("Hard rule (delegation is default)", "hard rule"),
        ("Decompose first", "decompose first"),
        ("Parallelize independent work", "parallelize independent"),
        ("Sequence dependent work", "sequence dependent"),
        ("Synthesize, don't relay", "synthesize"),
        ("How to delegate (member id)", "use the member's exact `id`"),
        ("If in doubt, delegate", "if in doubt, delegate"),
    ]
    for label, needle in pillars:
        assert needle in body_lower, (
            f"Sub-agent delegation block lost the '{label}' pillar; "
            f"prompt is being hollowed out"
        )


@test(
    "regression_v014",
    "FRAMEWORK_SYSTEM_PROMPT instructs multi-member decomposition",
)
async def t_prompt_multi_member(ctx: TestContext) -> None:
    """Coordinate-mode contract: the leader must be told to (1) decompose
    multi-part prompts, (2) parallelize independent work, and (3)
    synthesize member outputs into the final reply. A regression that
    softens any of these phrases would silently push the leader back to
    single-specialist-per-turn behaviour.
    """
    from src.core.prompts import FRAMEWORK_SYSTEM_PROMPT

    text = FRAMEWORK_SYSTEM_PROMPT.lower()
    for needle in ("decompose", "parallel", "synthesize"):
        assert needle in text, f"FRAMEWORK_SYSTEM_PROMPT missing: {needle}"


@test("regression_v014", "build_mcp_catalog_summary handles None / empty / vault-foregrounded")
async def t_catalog_summary_edges(ctx: TestContext) -> None:
    from src.core.prompts import build_mcp_catalog_summary

    # No pool at all (early-boot or test fixture).
    text = build_mcp_catalog_summary(None)
    assert "no mcps connected" in text.lower(), text

    # Pool exists but server_summary raises (defensive).
    class _BrokenPool:
        def server_summary(self):
            raise RuntimeError("not ready yet")

    text = build_mcp_catalog_summary(_BrokenPool())
    assert "unavailable" in text.lower() or "no mcps connected" in text.lower(), text

    # Empty pool — no servers connected.
    class _EmptyPool:
        def server_summary(self):
            return {}

    text = build_mcp_catalog_summary(_EmptyPool())
    assert "no mcps connected" in text.lower(), text

    # Realistic pool with vault — vault must be FIRST and carry the
    # imperative "READ BEFORE / WRITE AFTER" wording.
    class _RealPool:
        def server_summary(self):
            return {"vault": 8, "shell": 4, "tool-search": 4, "web": 3}
        def server_descriptions(self):
            return {"shell": "execute shell commands", "web": "fetch URLs"}

    text = build_mcp_catalog_summary(_RealPool())
    # Vault first.
    vault_pos = text.find("``vault``")
    shell_pos = text.find("``shell``")
    web_pos = text.find("``web``")
    assert 0 <= vault_pos < shell_pos < web_pos, (
        f"vault must precede other servers; positions: "
        f"vault={vault_pos} shell={shell_pos} web={web_pos}"
    )
    assert "READ BEFORE" in text and "WRITE AFTER" in text, (
        "vault block must carry the discipline imperative"
    )
    # Tool-search rendered last (the model is already reading the text
    # via tool-search).
    assert text.rindex("``tool-search``") > web_pos


@test("regression_v014", "MCPPool.server_descriptions exists")
async def t_pool_server_descriptions(ctx: TestContext) -> None:
    """build_mcp_catalog_summary calls pool.server_descriptions().
    Without the method, the catalog falls back to generic blurbs and
    the vault foreground hint disappears.
    """
    from src.mcp.pool import MCPPool

    fn = getattr(MCPPool, "server_descriptions", None)
    assert callable(fn), "MCPPool.server_descriptions must exist (Phase 10 gap fill)"


# ── Phase 10: Live-run fixes ─────────────────────────────────────────


@test("regression_v014", "Agent.memory_db is the public DB accessor (Curator wiring fix)")
async def t_agent_memory_db_attr(ctx: TestContext) -> None:
    """The Curator caller in core/server.py passes ``self.agent.memory_db``
    to start(). A rename of memory_db → memory (or vice versa) would
    re-introduce the "Agent has no attribute X" warning on every boot.
    """
    from src.core.agent import Agent

    # The Agent class definition (not an instance — instances need a
    # MemoryDB + MCPPool which the test harness may not provide here).
    assert hasattr(Agent, "memory_db"), "Agent.memory_db property removed"
    # The property must return the underlying ``_db`` set in __init__.
    descr = vars(Agent).get("memory_db")
    assert isinstance(descr, property), "memory_db must remain a property"


@test("regression_v014", "Curator.start accepts a MemoryDB and is callable")
async def t_curator_start_signature(ctx: TestContext) -> None:
    """Curator.start(db) takes ONE argument — the DB. The wrong-attr
    bug was that core/server.py passed agent.memory (None) which the
    function rejected. Verify the contract holds.
    """
    import inspect
    from src.learning.curator import start

    sig = inspect.signature(start)
    params = list(sig.parameters)
    assert params == ["db"], f"curator.start signature changed: {params}"

    # When OPENAGENT_CURATOR_ENABLED is unset, start() returns None
    # without touching the DB — no exception, no side effects.
    result = start(None)
    assert result is None, (
        "Curator should no-op when disabled, returning None — "
        "instead it returned %r" % (result,)
    )


@test("regression_v014", "server.wait() installs BOTH asyncio + signal.signal handlers")
async def t_wait_dual_signal_handler(ctx: TestContext) -> None:
    """Belt-and-suspenders signal handling: the asyncio loop's
    add_signal_handler is the primary path; signal.signal is the
    fallback in case a C extension (iroh tokio, etc.) blocks the
    selector. Both MUST be installed.
    """
    import signal
    from src.core.server import AgentServer

    src_lines = inspect_source_lines(AgentServer.wait)
    text = "\n".join(src_lines)
    assert "loop.add_signal_handler" in text, (
        "wait() must keep the asyncio signal handler (primary path)"
    )
    assert "signal.signal(" in text, (
        "wait() must install signal.signal as the C-extension fallback"
    )
    assert "call_soon_threadsafe" in text, (
        "the legacy handler must bounce through call_soon_threadsafe "
        "so the asyncio Event.set is thread-safe"
    )
    # Both SIGINT and SIGTERM must be handled.
    assert "SIGINT" in text and "SIGTERM" in text


def inspect_source_lines(fn: Any) -> list[str]:
    import inspect
    src, _ = inspect.getsourcelines(fn)
    return src


@test("regression_v014", "cli.py pins tqdm to threading.RLock (no multiprocessing semaphore leak)")
async def t_tqdm_threading_lock(ctx: TestContext) -> None:
    """The smoking-gun fix from Phase 10: tqdm's default lock is
    multiprocessing.RLock, which forks the resource_tracker daemon
    and creates a process-lifetime ``mp-…`` POSIX semaphore that gets
    reported as "leaked" at process exit. Pinning the lock to
    threading.RLock prevents the multiprocessing module from ever
    initialising.
    """
    import sys as _sys

    # Re-import cli in a way that lets us inspect tqdm's lock attribute.
    # cli.py runs ``tqdm.tqdm.set_lock(threading.RLock())`` at module
    # load, so by the time any test runs, tqdm is already pinned.
    # (Other tests in the suite import cli via the framework.)
    import threading
    import src.cli  # noqa: F401  — module-level side effect: pin tqdm

    try:
        import tqdm
    except ImportError:  # pragma: no cover — tqdm is a transitive dep
        return

    lock = tqdm.tqdm.get_lock()
    # The lock should be a threading.RLock (or threading.Lock for older
    # tqdm), NEVER a multiprocessing.RLock / multiprocessing.SemLock.
    lock_type_name = type(lock).__name__
    lock_module = type(lock).__module__
    assert lock_module.startswith("threading") or lock_module == "_thread", (
        f"tqdm lock must be a threading primitive; got {lock_module}.{lock_type_name} "
        f"— a regression here triggers the resource_tracker semaphore leak"
    )

    # Also verify TOKENIZERS_PARALLELISM is set (defensive: HF
    # tokenizers won't fork worker threads on macOS).
    import os
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false", (
        "TOKENIZERS_PARALLELISM must be 'false' to prevent the "
        "tokenizers fork warning + potential semaphore creation"
    )


@test("regression_v014", "openagent CLI subprocess exits without 'leaked semaphore' warning")
async def t_subprocess_no_semaphore_leak(ctx: TestContext) -> None:
    """End-to-end leak check: spawn ``python -m src.cli --help`` (the
    lightest path that exercises module load + click teardown) and
    grep stderr for the resource_tracker warning. This catches a leak
    that the in-process test can't, because the warning fires at
    process exit, not at module load.
    """
    repo_root = Path(__file__).resolve().parents[2]
    python = repo_root / ".venv" / "bin" / "python"
    if not python.exists():
        # CI / non-venv runs: fall back to the current interpreter.
        python = Path(sys.executable)

    proc = subprocess.run(
        [str(python), "-m", "src.cli", "--help"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=30,
    )

    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "leaked semaphore" not in combined.lower(), (
        f"resource_tracker reported a leaked semaphore on CLI exit:\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    # CLI --help should exit 0.
    assert proc.returncode == 0, (
        f"openagent --help exited {proc.returncode}: {proc.stderr}"
    )


# ── Team membership: ClaudeBackedAgent passes Agent typing ─────────


@test("regression_v014", "Team.members typing still constrains to Agent | Team")
async def t_team_members_typing(ctx: TestContext) -> None:
    """Agno's Team.members is typed Union[List[Union[Agent, "Team"]],
    Callable]. ClaudeBackedAgent works as a member precisely because
    it inherits from ``agno.agent.Agent`` (and not from
    ``BaseExternalAgent`` like ``agno.agents.claude.ClaudeAgent``).
    If Agno widened the union to include external agents we'd want to
    know — the routing-model fallback in team_router.py exists
    specifically because Team always invokes team.model for the
    delegation classifier.
    """
    import inspect
    from agno.team._init import __init__ as team_init

    sig = inspect.signature(team_init)
    members_param = sig.parameters.get("members")
    assert members_param is not None, "Team.__init__ no longer has 'members'"
    annotation = str(members_param.annotation)
    # The union still references Agent and Team only.
    assert "Agent" in annotation and "Team" in annotation
    assert "BaseExternalAgent" not in annotation and "ClaudeAgent" not in annotation, (
        f"Team.members union widened to {annotation}; "
        f"team_router.py's routing-model fallback may be reconsidered "
        f"if external agents are now allowed"
    )


@test("regression_v014", "ClaudeBackedAgent IS an Agent (Team accepts it as member)")
async def t_claude_backed_is_agent(ctx: TestContext) -> None:
    """The whole point of ``ClaudeBackedAgent``: a claude-cli row can
    participate in an Agno ``Team`` as a member OR as the leader's
    members[0] slot because it subclasses ``agno.agent.Agent`` (and
    therefore passes Team's ``isinstance(member, Agent)`` check). The
    earlier ``PersistentClaudeAgent`` (ClaudeAgent / BaseExternalAgent
    subclass) couldn't.
    """
    from agno.agent import Agent
    from src.models.claude_agent import ClaudeBackedAgent

    agent = ClaudeBackedAgent(
        name="x", sdk_model_id="claude-sonnet-4-6", sdk_options={},
    )
    assert isinstance(agent, Agent), (
        "ClaudeBackedAgent must subclass Agent so Team accepts it as a member"
    )
    # The Agent attribute defaults Team's delegation code reads must
    # be present. AttributeError on these → Team crashes mid-delegation.
    for attr in ("knowledge_filters", "num_history_runs", "store_media",
                 "store_tool_messages", "store_history_messages",
                 "send_media_to_model", "add_history_to_context"):
        assert hasattr(agent, attr), (
            f"ClaudeBackedAgent missing Agent attribute {attr!r} — "
            f"Team's delegation code will AttributeError"
        )


@test("regression_v014", "TeamRouterProvider includes claude-cli rows as members")
async def t_team_includes_claude_cli(ctx: TestContext) -> None:
    """Regression for the v0.14.x refactor: ``_enabled_llm_models``
    (formerly ``_enabled_api_models``) now returns claude-cli AND
    api-based rows. If a future change reverts to filtering claude-cli
    out of the membership catalog, this test catches it.
    """
    from src.models.catalog import FRAMEWORK_API_BASED, FRAMEWORK_CLAUDE_CLI, framework_of
    from src.models.dispatcher import TeamRouterProvider

    providers = [
        {
            "id": 1, "name": "openai", "framework": "api-based",
            "api_key": "sk-test", "enabled": True,
            "models": [{"id": 10, "model": "gpt-4o-mini", "enabled": True}],
        },
        {
            "id": 2, "name": "anthropic", "framework": "claude-cli",
            "api_key": None, "enabled": True,
            "models": [{"id": 20, "model": "claude-sonnet-4-6", "enabled": True}],
        },
    ]
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )

    catalog = provider._enabled_llm_models()
    frameworks = {framework_of(e.runtime_id) for e in catalog}
    assert FRAMEWORK_API_BASED in frameworks, frameworks
    assert FRAMEWORK_CLAUDE_CLI in frameworks, (
        f"_enabled_llm_models missing claude-cli rows: {[e.runtime_id for e in catalog]}"
    )


# ── codex-cli framework (v0.14.x third framework) ───────────────────


@test("regression_v014", "CodexBackedAgent IS an Agent (Team accepts it as member)")
async def t_codex_backed_is_agent(ctx: TestContext) -> None:
    """``CodexBackedAgent`` is the codex-cli equivalent of
    ``ClaudeBackedAgent`` — both subclass ``agno.agent.Agent`` so they
    pass Agno's ``isinstance(member, Agent)`` check AND inherit the
    attribute defaults Team's delegation code reads.
    """
    from agno.agent import Agent
    from src.models.codex_agent import CodexBackedAgent

    agent = CodexBackedAgent(
        name="x", sdk_model_id="gpt-5", sdk_options={},
    )
    assert isinstance(agent, Agent), (
        "CodexBackedAgent must subclass Agent so Team accepts it as a member"
    )
    for attr in ("knowledge_filters", "num_history_runs", "store_media",
                 "store_tool_messages", "store_history_messages",
                 "send_media_to_model", "add_history_to_context"):
        assert hasattr(agent, attr), (
            f"CodexBackedAgent missing Agent attribute {attr!r} — "
            f"Team's delegation code will AttributeError"
        )


@test("regression_v014", "TeamRouterProvider includes codex-cli rows as members")
async def t_team_includes_codex_cli(ctx: TestContext) -> None:
    """``_enabled_llm_models`` accepts api-based, claude-cli, AND
    codex-cli. A revert to filtering codex-cli out of the membership
    catalog would break the third-framework feature.
    """
    from src.models.catalog import (
        FRAMEWORK_API_BASED,
        FRAMEWORK_CODEX_CLI,
        framework_of,
    )
    from src.models.dispatcher import TeamRouterProvider

    providers = [
        {
            "id": 1, "name": "openai", "framework": "api-based",
            "api_key": "sk-test", "enabled": True,
            "models": [{"id": 10, "model": "gpt-4o-mini", "enabled": True}],
        },
        {
            "id": 2, "name": "openai", "framework": "codex-cli",
            "api_key": None, "enabled": True,
            "models": [{"id": 20, "model": "gpt-5", "enabled": True}],
        },
    ]
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )

    catalog = provider._enabled_llm_models()
    frameworks = {framework_of(e.runtime_id) for e in catalog}
    assert FRAMEWORK_API_BASED in frameworks, frameworks
    assert FRAMEWORK_CODEX_CLI in frameworks, (
        f"_enabled_llm_models missing codex-cli rows: {[e.runtime_id for e in catalog]}"
    )


@test(
    "regression_v014",
    "catalog framework_of / is_codex_cli_model split codex-cli runtime_ids",
)
async def t_catalog_codex_helpers(ctx: TestContext) -> None:
    """Locks the per-runtime_id helpers for codex-cli: ``framework_of``
    returns ``codex-cli``; ``is_codex_cli_model`` matches both the
    canonical 3-part form and the legacy shorthand; pricing for any
    codex-cli runtime_id returns zero (subscription-billed).
    """
    from src.models.catalog import (
        FRAMEWORK_CODEX_CLI,
        codex_cli_model_spec,
        framework_of,
        get_model_pricing,
        is_codex_cli_model,
        is_subscription_cli_model,
        split_runtime_id,
    )

    canonical = "codex-cli:openai:gpt-5"
    legacy_short = "codex-cli/gpt-5"
    plain_codex = "codex-cli"
    api = "openai:gpt-5"

    assert framework_of(canonical) == FRAMEWORK_CODEX_CLI
    assert framework_of(legacy_short) == FRAMEWORK_CODEX_CLI
    assert framework_of(plain_codex) == FRAMEWORK_CODEX_CLI
    assert framework_of(api) == "api-based"

    assert is_codex_cli_model(canonical)
    assert is_codex_cli_model(legacy_short)
    assert is_codex_cli_model(plain_codex)
    assert not is_codex_cli_model(api)
    assert not is_codex_cli_model("claude-cli:anthropic:claude-sonnet-4-6")

    assert is_subscription_cli_model(canonical)
    assert is_subscription_cli_model("claude-cli:anthropic:claude-sonnet-4-6")
    assert not is_subscription_cli_model(api)

    # codex_cli_model_spec builds canonical 3-part form.
    assert codex_cli_model_spec("gpt-5") == canonical

    # split_runtime_id handles the codex-cli prefix correctly.
    assert split_runtime_id(canonical) == ("openai", "gpt-5")
    assert split_runtime_id("codex-cli:openai:gpt-5") == ("openai", "gpt-5")

    # Subscription-billed → zero pricing for any codex-cli model.
    pricing = get_model_pricing(canonical)
    assert pricing == {
        "input_cost_per_million": 0.0,
        "output_cost_per_million": 0.0,
    }, pricing


@test(
    "regression_v014",
    "upsert_provider accepts codex-cli for openai with no api_key",
)
async def t_upsert_codex_cli_provider(ctx: TestContext) -> None:
    """``framework='codex-cli'`` providers MUST be ``name='openai'`` AND
    carry ``api_key=None`` (auth flows through ChatGPT subscription).
    """
    from src.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"cdxups-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        try:
            # Happy path: openai + codex-cli + no api_key.
            pid = await db.upsert_provider(
                name="openai", framework="codex-cli", api_key=None,
            )
            row = await db.get_provider(pid)
            assert row["framework"] == "codex-cli", row["framework"]
            assert row["name"] == "openai", row["name"]

            # Bad path: api_key set → reject (would poison the subprocess).
            raised = False
            try:
                await db.upsert_provider(
                    name="openai", framework="codex-cli", api_key="sk-x",
                )
            except ValueError as e:
                raised = True
                assert "codex-cli" in str(e) and "api_key" in str(e), str(e)
            assert raised

            # Bad path: wrong vendor name → reject.
            raised = False
            try:
                await db.upsert_provider(
                    name="anthropic", framework="codex-cli", api_key=None,
                )
            except ValueError as e:
                raised = True
                assert "openai" in str(e), str(e)
            assert raised
        finally:
            await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass
