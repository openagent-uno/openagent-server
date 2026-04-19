"""SmartRouter hybrid dispatch — agno + claude-cli under one router (v0.12 schema).

Tests here monkey-patch the classifier + dispatch layer so no real LLM
or claude binary is needed. We focus on the routing decision, the
session-side binding, and the cross-side lock.
"""
from __future__ import annotations

import uuid
from typing import Any

from ._framework import TestContext, test


class _FakeResp:
    def __init__(self, content: str, model: str | None = None):
        self.content = content
        self.input_tokens = 10
        self.output_tokens = 5
        self.stop_reason = "stop"
        self.model = model


def _make_router(providers_config: list, routing: dict[str, str] | None = None):
    """Build a SmartRouter for tests.

    ``routing`` is accepted for call-site compat with the legacy yaml
    ``model.routing`` tiers, but v0.12 ignores it — the router reads the
    enabled catalog from ``providers_config`` on every turn.
    """
    from openagent.models.smart_router import SmartRouter

    del routing
    return SmartRouter(providers_config=providers_config)


async def _stub_classifier(router, picked_runtime_id: str | None) -> None:
    """Stub the classifier to return a fixed ``runtime_id`` (or None).

    With classifier-direct routing the classifier returns a concrete
    runtime_id, not a tier. ``None`` lets the router exercise its
    "no pick" fallback path (first enabled model on the bound side).
    """

    async def _fake_classify(messages, session_id, catalog):
        return picked_runtime_id

    router._classify = _fake_classify  # type: ignore[assignment]


async def _stub_dispatch(router, recorded: list[str]):
    """Replace the actual provider dispatch with a recorder."""

    async def _fake(runtime_id, messages, system, tools, on_status, session_id):
        recorded.append(runtime_id)
        return _FakeResp("ok", model=runtime_id)

    router._dispatch = _fake  # type: ignore[assignment]


def _providers_both_frameworks() -> list[dict[str, Any]]:
    """Build a v0.12 flat-list providers_config with agno + claude-cli rows."""
    return [
        {"id": 1, "name": "openai", "framework": "agno",
         "api_key": "sk-x", "base_url": None, "enabled": True,
         "models": [{"id": 10, "model": "gpt-4o-mini", "enabled": True}]},
        {"id": 2, "name": "anthropic", "framework": "claude-cli",
         "api_key": None, "base_url": None, "enabled": True,
         "models": [{"id": 20, "model": "claude-sonnet-4-6", "enabled": True}]},
    ]


@test("smart_router_hybrid", "fresh session uses classifier pick + records binding")
async def t_fresh_agno(ctx: TestContext) -> None:
    import uuid as _uuid
    from openagent.memory.db import MemoryDB

    tmp = ctx.db_path.with_name(f"sr-hybrid-{_uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp))
        await db.connect()
        providers = [
            {"id": 1, "name": "openai", "framework": "agno",
             "api_key": "sk-x", "enabled": True,
             "models": [{"id": 10, "model": "gpt-4o-mini", "enabled": True}]},
        ]
        router = _make_router(providers, {
            "simple": "openai:gpt-4o-mini",
            "medium": "openai:gpt-4o-mini",
            "hard": "openai:gpt-4o-mini",
            "fallback": "openai:gpt-4o-mini",
        })
        router.set_db(db)
        await _stub_classifier(router, "openai:gpt-4o-mini")
        seen: list[str] = []
        await _stub_dispatch(router, seen)

        sid = "tg:42"
        resp = await router.generate([{"role": "user", "content": "hi"}], session_id=sid)
        assert resp.model == "openai:gpt-4o-mini"
        assert seen == ["openai:gpt-4o-mini"]
        assert await db.get_session_binding(sid) == "agno"
        await db.close()
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


@test("smart_router_hybrid", "bound-to-agno session stays on agno even if classifier picks claude-cli")
async def t_bound_side_locked(ctx: TestContext) -> None:
    import uuid as _uuid
    from openagent.memory.db import MemoryDB

    tmp = ctx.db_path.with_name(f"sr-lock-{_uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp))
        await db.connect()
        providers = _providers_both_frameworks()
        router = _make_router(providers, {
            "simple": "openai:gpt-4o-mini",
            "medium": "openai:gpt-4o-mini",
            "hard": "claude-cli:anthropic:claude-sonnet-4-6",
            "fallback": "openai:gpt-4o-mini",
        })
        router.set_db(db)
        # Pre-bind the session to agno as if a prior turn landed there.
        await db.set_session_binding("sess-bound", "agno")
        # Classifier picks a claude-cli model; the bound side filter
        # should drop it and fall back to the first enabled agno entry.
        await _stub_classifier(router, "claude-cli:anthropic:claude-sonnet-4-6")
        seen: list[str] = []
        await _stub_dispatch(router, seen)

        resp = await router.generate(
            [{"role": "user", "content": "this is hard"}],
            session_id="sess-bound",
        )
        assert resp.model.startswith("openai:"), f"should stay on agno, got {resp.model}"
        assert all(m.startswith("openai:") for m in seen), seen
        await db.close()
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


@test("smart_router_hybrid", "bound-to-claude-cli routes via claude-cli only")
async def t_bound_to_claude_cli(ctx: TestContext) -> None:
    import uuid as _uuid
    from openagent.memory.db import MemoryDB

    tmp = ctx.db_path.with_name(f"sr-cli-{_uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp))
        await db.connect()
        providers = _providers_both_frameworks()
        router = _make_router(providers, {
            "simple": "openai:gpt-4o-mini",
            "medium": "openai:gpt-4o-mini",
            "hard": "claude-cli:anthropic:claude-sonnet-4-6",
            "fallback": "openai:gpt-4o-mini",
        })
        router.set_db(db)
        # Claude-cli bindings live in sdk_sessions.
        await db.set_sdk_session("cli-sess", "sdk-uuid", provider="claude-cli")
        await _stub_classifier(router, "claude-cli:anthropic:claude-sonnet-4-6")
        seen: list[str] = []
        await _stub_dispatch(router, seen)

        resp = await router.generate(
            [{"role": "user", "content": "hi"}],
            session_id="cli-sess",
        )
        assert resp.model.startswith("claude-cli:"), f"should stay on claude-cli, got {resp.model}"
        assert seen and seen[0].startswith("claude-cli:"), seen
        await db.close()
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


@test("smart_router_hybrid", "bound side with no enabled models returns a clear error")
async def t_bound_side_empty(ctx: TestContext) -> None:
    import uuid as _uuid
    from openagent.memory.db import MemoryDB

    tmp = ctx.db_path.with_name(f"sr-empty-{_uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp))
        await db.connect()
        providers = [
            {"id": 1, "name": "openai", "framework": "agno",
             "api_key": "sk-x", "enabled": True,
             "models": [{"id": 10, "model": "gpt-4o-mini", "enabled": True}]},
        ]
        router = _make_router(providers, {
            "simple": "openai:gpt-4o-mini",
            "medium": "openai:gpt-4o-mini",
            "hard": "openai:gpt-4o-mini",
            "fallback": "openai:gpt-4o-mini",
        })
        router.set_db(db)
        # Session was bound to claude-cli but we have no claude-cli
        # models configured.
        await db.set_sdk_session("orphan", "sdk-id", provider="claude-cli")
        # No claude-cli model in the catalog → classifier has nothing
        # to pick; resolve_classifier_pick returns the empty-string
        # primary_model and generate surfaces the error.
        await _stub_classifier(router, None)

        resp = await router.generate(
            [{"role": "user", "content": "hi"}],
            session_id="orphan",
        )
        assert resp.stop_reason == "error"
        assert "claude-cli" in resp.content, resp.content
        await db.close()
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


@test("smart_router_hybrid", "dual-framework provider isolation — agno key never leaks into claude-cli env")
async def t_dual_framework_env_isolation(ctx: TestContext) -> None:
    """Regression guard for the v0.11.5 sentinel bug.

    When the same vendor (anthropic) is registered under both frameworks,
    AgnoProvider's env-injection must export ONLY the agno row's api_key.
    The claude-cli row carries api_key=NULL by v0.12 schema, but even if
    legacy data leaked through, AgnoProvider's per-entry framework filter
    must drop anything that's not agno.
    """
    import os as _os
    from openagent.models.agno_provider import AgnoProvider

    providers = [
        {"id": 1, "name": "anthropic", "framework": "agno",
         "api_key": "sk-ant-real", "enabled": True, "models": []},
        {"id": 2, "name": "anthropic", "framework": "claude-cli",
         "api_key": None, "enabled": True, "models": []},
    ]
    prev = _os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        AgnoProvider(model="anthropic:claude-sonnet-4-6", providers_config=providers)
        assert _os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-real"
    finally:
        if prev is None:
            _os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            _os.environ["ANTHROPIC_API_KEY"] = prev
