"""SmartRouter hybrid dispatch — agno + claude-cli under one router.

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


def _make_router(providers_config: dict, routing: dict[str, str]):
    from openagent.models.smart_router import SmartRouter

    r = SmartRouter(
        routing=routing,
        providers_config=providers_config,
        api_key=None,
        monthly_budget=0.0,
    )
    return r


async def _stub_classifier(router, tier: str) -> None:
    async def _fake_classify(messages, session_id=None):
        return tier

    router._classify = _fake_classify  # type: ignore[assignment]


async def _stub_dispatch(router, recorded: list[str]):
    """Replace the actual provider dispatch with a recorder."""

    async def _fake(runtime_id, messages, system, tools, on_status, session_id):
        recorded.append(runtime_id)
        return _FakeResp("ok", model=runtime_id)

    router._dispatch = _fake  # type: ignore[assignment]


@test("smart_router_hybrid", "fresh session uses classifier pick + records binding")
async def t_fresh_agno(ctx: TestContext) -> None:
    import uuid as _uuid
    from openagent.memory.db import MemoryDB

    tmp = ctx.db_path.with_name(f"sr-hybrid-{_uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp))
        await db.connect()
        providers = {"openai": {"api_key": "sk-x", "models": ["gpt-4o-mini"]}}
        router = _make_router(providers, {
            "simple": "openai:gpt-4o-mini",
            "medium": "openai:gpt-4o-mini",
            "hard": "openai:gpt-4o-mini",
            "fallback": "openai:gpt-4o-mini",
        })
        router.set_db(db)
        await _stub_classifier(router, "simple")
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


@test("smart_router_hybrid", "bound-to-agno session stays on agno even if classifier picks hard/claude-cli")
async def t_bound_side_locked(ctx: TestContext) -> None:
    import uuid as _uuid
    from openagent.memory.db import MemoryDB

    tmp = ctx.db_path.with_name(f"sr-lock-{_uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp))
        await db.connect()
        providers = {
            "openai": {"api_key": "sk-x", "models": ["gpt-4o-mini"]},
            # Legacy yaml shape: provider=claude-cli. Bootstrap/catalog
            # translate this to framework=claude-cli, provider=anthropic
            # internally.
            "claude-cli": {"models": ["claude-sonnet-4-6"]},
        }
        router = _make_router(providers, {
            "simple": "openai:gpt-4o-mini",
            "medium": "openai:gpt-4o-mini",
            "hard": "claude-cli:anthropic:claude-sonnet-4-6",
            "fallback": "openai:gpt-4o-mini",
        })
        router.set_db(db)
        # Pre-bind the session to agno as if a prior turn landed there.
        await db.set_session_binding("sess-bound", "agno")
        await _stub_classifier(router, "hard")
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
        providers = {
            "openai": {"api_key": "sk-x", "models": ["gpt-4o-mini"]},
            "claude-cli": {"models": ["claude-sonnet-4-6"]},
        }
        router = _make_router(providers, {
            "simple": "openai:gpt-4o-mini",
            "medium": "openai:gpt-4o-mini",
            "hard": "claude-cli:anthropic:claude-sonnet-4-6",
            "fallback": "openai:gpt-4o-mini",
        })
        router.set_db(db)
        # Claude-cli bindings live in sdk_sessions.
        await db.set_sdk_session("cli-sess", "sdk-uuid", provider="claude-cli")
        await _stub_classifier(router, "simple")
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
        providers = {"openai": {"api_key": "sk-x", "models": ["gpt-4o-mini"]}}
        router = _make_router(providers, {
            "simple": "openai:gpt-4o-mini",
            "medium": "openai:gpt-4o-mini",
            "hard": "openai:gpt-4o-mini",
            "fallback": "openai:gpt-4o-mini",
        })
        router.set_db(db)
        # Session was bound to claude-cli but we have no claude-cli
        # models in the routing table.
        await db.set_sdk_session("orphan", "sdk-id", provider="claude-cli")
        await _stub_classifier(router, "simple")

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
