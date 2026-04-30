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


@test("smart_router_hybrid", "classifier resolves from is_classifier flag, else first enabled")
async def t_classifier_resolution(ctx: TestContext) -> None:
    """SmartRouter picks its classifier from the live catalog.

    Prior releases hardcoded ``openai:gpt-4o-mini`` as the classifier,
    which broke any deployment that didn't have an OpenAI key configured
    (e.g. claude-cli-only installs). The DB-driven resolver must:

      1. prefer the row flagged ``is_classifier=True``,
      2. fall back to the first enabled row when no flag is set,
      3. return empty string (signalling "no classifier available") when
         the catalog has zero enabled rows — caller skips classify and
         surfaces the standard "No model is currently enabled" error.
    """
    from openagent.models.smart_router import _resolve_classifier_model

    # 1. flagged wins over everything
    cfg = [{
        "id": 1, "name": "openai", "framework": "agno", "enabled": True,
        "models": [
            {"id": 1, "model": "gpt-4o-mini", "enabled": True, "is_classifier": False},
            {"id": 2, "model": "gpt-5", "enabled": True, "is_classifier": True},
        ],
    }]
    assert _resolve_classifier_model(cfg) == "openai:gpt-5"

    # 2. no flag → first enabled (deterministic order from materialise)
    cfg = [{
        "id": 1, "name": "openai", "framework": "agno", "enabled": True,
        "models": [
            {"id": 1, "model": "gpt-4o-mini", "enabled": True, "is_classifier": False},
            {"id": 2, "model": "gpt-5", "enabled": True, "is_classifier": False},
        ],
    }]
    assert _resolve_classifier_model(cfg) == "openai:gpt-4o-mini"

    # 3. claude-cli-only install (the lyra-agent scenario) —
    # classifier resolves to the claude-cli model, NOT the dead
    # openai:gpt-4o-mini hardcode.
    cfg = [{
        "id": 1, "name": "anthropic", "framework": "claude-cli", "enabled": True,
        "models": [
            {"id": 1, "model": "claude-sonnet-4-6", "enabled": True, "is_classifier": False},
        ],
    }]
    assert _resolve_classifier_model(cfg) == "claude-cli:anthropic:claude-sonnet-4-6"

    # 4. every model disabled → empty classifier → router skips
    # classify and lets the caller surface "No model is currently
    # enabled" (see _classify's no_classifier_model skip path).
    cfg = [{
        "id": 1, "name": "openai", "framework": "agno", "enabled": True,
        "models": [
            {"id": 1, "model": "gpt-4o-mini", "enabled": False, "is_classifier": True},
        ],
    }]
    assert _resolve_classifier_model(cfg) == ""

    # 5. empty providers_config → empty classifier
    assert _resolve_classifier_model([]) == ""

    # 6. multiple rows flagged → resolver picks the first in catalog
    # order (p.name, p.framework, m.model). Multiple-classifier
    # semantics: the flag opts a row into the pool, it doesn't claim
    # exclusive ownership.
    cfg = [{
        "id": 1, "name": "openai", "framework": "agno", "enabled": True,
        "models": [
            {"id": 1, "model": "gpt-4o-mini", "enabled": True, "is_classifier": True},
            {"id": 2, "model": "gpt-5", "enabled": True, "is_classifier": True},
        ],
    }]
    assert _resolve_classifier_model(cfg) == "openai:gpt-4o-mini"


@test("smart_router_hybrid", "single-model catalog short-circuits the classifier call")
async def t_single_model_skips_classify(ctx: TestContext) -> None:
    """Classifying-of-one is tautological.

    The lyra-agent scenario: only ``claude-cli:anthropic:claude-sonnet-4-6``
    is enabled, so ``_resolve_classifier_model`` returns that same id —
    meaning we'd spawn a claude subprocess just to ask the model "which
    model?". Skip the classify call when the catalog has exactly one
    entry; the decision is forced.

    Asserts: ``_classify`` is NOT called, and the routing decision still
    dispatches the only enabled runtime_id with reason
    ``single_enabled_model``.
    """
    providers = [{
        "id": 1, "name": "anthropic", "framework": "claude-cli",
        "api_key": None, "enabled": True,
        "models": [{"id": 1, "model": "claude-sonnet-4-6", "enabled": True}],
    }]
    router = _make_router(providers)

    classify_calls: list[Any] = []

    async def _spy_classify(messages, session_id, catalog):
        classify_calls.append((messages, session_id, catalog))
        return "claude-cli:anthropic:claude-sonnet-4-6"

    router._classify = _spy_classify  # type: ignore[assignment]
    seen: list[str] = []
    await _stub_dispatch(router, seen)

    resp = await router.generate(
        [{"role": "user", "content": "hi"}],
        session_id="tg:single",
    )
    assert resp.model == "claude-cli:anthropic:claude-sonnet-4-6"
    assert seen == ["claude-cli:anthropic:claude-sonnet-4-6"]
    assert classify_calls == [], (
        f"_classify must not run when the catalog has one entry, got {len(classify_calls)} calls"
    )


@test("smart_router_hybrid", "classifier provider is created with no MCP toolkits attached")
async def t_classifier_no_mcp_injection(ctx: TestContext) -> None:
    """Regression guard for the 302-tools-vs-128-cap bug.

    Before the patch, the classifier shared the dispatch provider's MCP
    pool, which in production deployments (~20 toolkits / 300+ tools)
    overflowed OpenAI's 128-tool limit and every classify call failed
    with ``Invalid 'tools': array too long``. The fix: build the
    classifier with ``mcp_pool=None`` and exclude it from
    ``SmartRouter.set_mcp_pool``'s fan-out loop.

    This test wires a fake pool with a toolkit, flushes the pool into
    the router, triggers classifier creation, and asserts the
    classifier's ``_mcp_toolkits`` stayed empty while the dispatch
    providers picked up the toolkit.
    """
    from openagent.models.smart_router import SmartRouter

    class _FakePool:
        agno_toolkits = ["fake-toolkit-a", "fake-toolkit-b"]

        def claude_sdk_servers(self):
            return {}

        def agno_toolkits_under_budget(self, budget: int) -> list:
            # Mirror the real MCPPool method's signature so dispatch
            # provider construction (set_mcp_toolkits → this method)
            # doesn't AttributeError. Budget is ignored — these fakes
            # cost zero tools.
            return list(self.agno_toolkits)

    providers = [{
        "id": 1, "name": "openai", "framework": "agno",
        "api_key": "sk-x", "enabled": True,
        "models": [{"id": 1, "model": "gpt-4o-mini", "enabled": True}],
    }]
    router = SmartRouter(providers_config=providers)
    router.set_mcp_pool(_FakePool())

    # Build the classifier lazily — mimics what the first classify call
    # does. If this inadvertently pulled from the pool, the toolkit list
    # below would be non-empty.
    classifier = router._get_classifier_provider()
    assert getattr(classifier, "_mcp_toolkits", None) == [], (
        f"classifier must have NO MCP toolkits, got {classifier._mcp_toolkits!r}"
    )

    # A dispatch provider built the same way MUST pick up the pool —
    # otherwise we've over-corrected and broken real turns.
    dispatch = router._get_agno_provider("openai:gpt-4o-mini")
    assert dispatch._mcp_toolkits == ["fake-toolkit-a", "fake-toolkit-b"], (
        "dispatch provider should still receive the MCP pool"
    )


@test("smart_router_hybrid", "rebuild_routing picks up flag flip without restart")
async def t_rebuild_routing_hot_reload(ctx: TestContext) -> None:
    """Hot-reload of the classifier when the ``is_classifier`` flag
    flips on another row. The gateway's per-message refresh calls
    ``rebuild_routing``; the router must pick up the new classifier
    id AND drop any cached provider instance bound to the old id.
    """
    from openagent.models.smart_router import SmartRouter

    providers = [{
        "id": 1, "name": "openai", "framework": "agno",
        "api_key": "sk-x", "enabled": True,
        "models": [
            {"id": 1, "model": "gpt-4o-mini", "enabled": True, "is_classifier": True},
            {"id": 2, "model": "gpt-5", "enabled": True, "is_classifier": False},
        ],
    }]
    router = SmartRouter(providers_config=providers)
    assert router._classifier_model == "openai:gpt-4o-mini"

    # Simulate a cached classifier provider so rebuild_routing has
    # something to invalidate.
    sentinel = object()
    router._classifier_provider = sentinel  # type: ignore[assignment]

    # Flip the flag — the re-materialised config now points to gpt-5.
    providers[0]["models"][0]["is_classifier"] = False
    providers[0]["models"][1]["is_classifier"] = True
    router.rebuild_routing(providers)
    assert router._classifier_model == "openai:gpt-5"
    assert router._classifier_provider is None, (
        "cached classifier provider must be dropped when the resolved id changes"
    )


class _RecordingUnderlying:
    """Stand-in for an underlying model/registry that records lifecycle calls.

    Mirrors the BaseModel surface SmartRouter fans out to. ``known_ids``
    simulates sdk_sessions rehydrated from sqlite after a restart.
    """

    def __init__(self, known_ids: list[str] | None = None) -> None:
        self.closed: list[str] = []
        self.forgotten: list[str] = []
        self._known: list[str] = list(known_ids or [])

    async def close_session(self, session_id: str) -> None:
        self.closed.append(session_id)

    async def forget_session(self, session_id: str) -> None:
        self.forgotten.append(session_id)
        if session_id in self._known:
            self._known.remove(session_id)

    def known_session_ids(self) -> list[str]:
        return list(self._known)


@test(
    "smart_router_hybrid",
    "forget_session fans out to underlying forget_session, not close_session",
)
async def t_forget_session_fans_out_to_forget(ctx: TestContext) -> None:
    """Regression for the ``/clean`` telegram bug (v0.12.16).

    SmartRouter only implemented ``close_session`` — so ``forget_session``
    fell through to ``BaseModel``'s default (which calls ``close_session``).
    That released the claude-cli subprocess but left the ``sdk_sessions``
    resume id intact, and the next turn ``--resume``'d the prior
    transcript. The user saw ``/clean`` echo "forgot 1 prior
    conversation" yet the LLM still remembered.

    The fix is an explicit ``SmartRouter.forget_session`` that fans out
    to the underlying models' own ``forget_session`` (not
    ``close_session``) so resume state is actually erased.
    """
    from openagent.models.smart_router import SmartRouter

    providers = _providers_both_frameworks()
    router = SmartRouter(providers_config=providers)

    agno_fake = _RecordingUnderlying()
    claude_fake = _RecordingUnderlying()
    router._agno_providers["openai:gpt-4o-mini"] = agno_fake  # type: ignore[assignment]
    router._claude_registry = claude_fake  # type: ignore[assignment]

    await router.forget_session("tg:155490357")

    assert agno_fake.forgotten == ["tg:155490357"], agno_fake.forgotten
    assert claude_fake.forgotten == ["tg:155490357"], claude_fake.forgotten
    # The regression: close_session must NOT be the one called, or
    # claude-cli would keep its sdk_sessions row and --resume the
    # prior transcript on the next turn.
    assert agno_fake.closed == [], agno_fake.closed
    assert claude_fake.closed == [], claude_fake.closed


@test(
    "smart_router_hybrid",
    "known_session_ids aggregates from underlying models (post-restart fallback)",
)
async def t_known_session_ids_aggregates(ctx: TestContext) -> None:
    """Gateway ``/clear`` without a session_id filters the model's
    ``known_session_ids`` by bridge prefix to reach sessions rehydrated
    from sqlite. SmartRouter with no override returned ``[]`` (BaseModel
    default), so post-restart /clear from telegram silently forgot
    nothing.
    """
    from openagent.models.smart_router import SmartRouter

    providers = _providers_both_frameworks()
    router = SmartRouter(providers_config=providers)

    router._agno_providers["openai:gpt-4o-mini"] = _RecordingUnderlying(  # type: ignore[assignment]
        known_ids=["tg:aaa", "discord:42"]
    )
    router._claude_registry = _RecordingUnderlying(  # type: ignore[assignment]
        known_ids=["tg:bbb", "tg:ccc"]
    )

    ids = router.known_session_ids()
    assert sorted(ids) == ["discord:42", "tg:aaa", "tg:bbb", "tg:ccc"], ids


