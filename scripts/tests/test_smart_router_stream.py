"""SmartRouter.stream — token-streaming dispatch for both frameworks.

Regression coverage for the bug where ``SmartRouter.stream`` for a
claude-cli route used ``_dispatch`` (one-shot ``generate``) and yielded
the WHOLE reply as a single chunk. Voice mode's TTS pipeline only saw
the first audio after the LLM was done — TTFB ~= full reply latency.

The fix:

* :class:`ClaudeCLIRegistry` got a real ``stream`` override that
  delegates to the per-session ``ClaudeCLI`` instance (which already
  emits token deltas via the SDK's ``on_delta`` plumbing).
* :class:`SmartRouter` no longer special-cases claude-cli into the
  one-shot path; both registries/providers expose ``stream`` and the
  router just forwards.

These tests stub both the routing decision and the registries to keep
the test hermetic (no Anthropic key, no MCP servers, no Agno install).
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from ._framework import TestContext, test


# ── Helpers ─────────────────────────────────────────────────────────


class _FakeClaudeRegistry:
    """Stub for ClaudeCLIRegistry whose ``stream`` yields fixed deltas."""

    def __init__(self, deltas: list[str]):
        self._deltas = list(deltas)
        self.stream_calls: list[dict[str, Any]] = []
        # SmartRouter probes these on shutdown / wiring; provide no-ops.
        self.cleanup_idle = lambda *a, **k: None
        self.shutdown = lambda *a, **k: None

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Any = None,
        session_id: str | None = None,
        model_override: str | None = None,
    ) -> AsyncIterator[str]:
        self.stream_calls.append({
            "session_id": session_id,
            "model_override": model_override,
            "on_status": on_status is not None,
        })
        for d in self._deltas:
            yield d


class _FakeAgnoProvider:
    """Stub for AgnoProvider whose ``stream`` yields fixed deltas.

    Records the kwargs each call receives so tests can verify
    SmartRouter forwards ``session_id`` (multi-tab history isolation)
    and ``on_status`` (parity with the claude-cli branch).
    """

    def __init__(self, deltas: list[str]):
        self._deltas = list(deltas)
        self.stream_calls: list[dict[str, Any]] = []

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Any = None,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        self.stream_calls.append({
            "session_id": session_id,
            "on_status": on_status is not None,
        })
        for d in self._deltas:
            yield d


def _make_router_for_stream(
    registry: _FakeClaudeRegistry | None = None,
    agno_provider: _FakeAgnoProvider | None = None,
):
    """Build a SmartRouter with the registries pre-stubbed.

    Bypasses ``_routing_decision`` so we don't need a real classifier or
    DB — the caller patches ``_routing_decision`` to whichever runtime_id
    they want this turn to land on.
    """
    from openagent.models.smart_router import SmartRouter

    router = SmartRouter(providers_config=[])
    if registry is not None:
        router._claude_registry = registry  # type: ignore[assignment]
    if agno_provider is not None:
        # ``_get_agno_provider`` looks up by runtime_id; stub it to always
        # return the fake. The runtime_id arg is irrelevant for the test.
        router._agno_providers["openai:gpt-4o-mini"] = agno_provider  # type: ignore[index]
    return router


def _stub_routing(router, runtime_id: str, *, bound_framework: str | None = None):
    """Force ``_routing_decision`` to return a specific runtime_id."""
    from openagent.models.smart_router import RoutingDecision

    async def _fake(messages, session_id):
        return RoutingDecision(
            requested_tier="classifier",
            effective_tier="classifier",
            reason="test_stub",
            primary_model=runtime_id,
            candidates=[runtime_id],
            bound_framework=bound_framework,
        )

    router._routing_decision = _fake  # type: ignore[assignment]


# ── Tests ───────────────────────────────────────────────────────────


@test("smart_router_stream", "claude-cli route yields per-token deltas (not one chunk)")
async def t_claude_cli_streams_real_deltas(_ctx: TestContext) -> None:
    """The whole point of this test: SmartRouter.stream for a claude-cli
    runtime must call registry.stream and forward EACH delta. Before the
    fix it called _dispatch (one-shot generate) and yielded the entire
    reply as a single chunk — TTFB-killing for voice mode."""
    fake_registry = _FakeClaudeRegistry(deltas=["Hello", " from", " Claude"])
    router = _make_router_for_stream(registry=fake_registry)
    runtime_id = "claude-cli:anthropic:claude-sonnet-4-6"
    _stub_routing(router, runtime_id)

    yielded: list[str] = []
    async for chunk in router.stream(
        [{"role": "user", "content": "hi"}], session_id="sess-cc-stream",
    ):
        yielded.append(chunk)

    assert yielded == ["Hello", " from", " Claude"], (
        f"router.stream must forward each delta separately, got {yielded}"
    )
    assert len(fake_registry.stream_calls) == 1
    call = fake_registry.stream_calls[0]
    assert call["session_id"] == "sess-cc-stream"
    assert call["model_override"] == runtime_id


@test("smart_router_stream", "claude-cli stream forwards on_status for tool updates")
async def t_claude_cli_stream_forwards_on_status(_ctx: TestContext) -> None:
    """Voice mode wires an on_status callback so tool-running statuses
    surface during streamed turns. Pre-fix, SmartRouter.stream dropped
    on_status when routing to claude-cli (it went through _dispatch but
    the streaming wrapper never picked it up either). Now it must be
    threaded all the way through."""
    fake_registry = _FakeClaudeRegistry(deltas=["x"])
    router = _make_router_for_stream(registry=fake_registry)
    _stub_routing(router, "claude-cli:anthropic:claude-sonnet-4-6")

    statuses: list[str] = []

    async def on_status(text: str) -> None:
        statuses.append(text)

    async for _ in router.stream(
        [{"role": "user", "content": "hi"}],
        session_id="sess-status",
        on_status=on_status,
    ):
        pass

    assert fake_registry.stream_calls[0]["on_status"] is True, (
        "on_status must be forwarded to ClaudeCLIRegistry.stream"
    )


@test("smart_router_stream", "_remember_pick fires before stream so model badge survives")
async def t_remember_pick_fires_before_stream(_ctx: TestContext) -> None:
    """``effective_model_id`` reads ``_recall_pick`` to populate the chat
    UI's model badge. If we forgot to remember the pick (or did it after
    the stream finished), tool-only / empty-stream turns would render a
    badge of ``None``. Same regression as the previous round, easy to
    reintroduce when refactoring the stream path."""
    fake_registry = _FakeClaudeRegistry(deltas=[])  # zero deltas — tool-only turn
    router = _make_router_for_stream(registry=fake_registry)
    runtime_id = "claude-cli:anthropic:claude-sonnet-4-6"
    _stub_routing(router, runtime_id)

    async for _ in router.stream(
        [{"role": "user", "content": "hi"}], session_id="sess-pick",
    ):
        pass

    assert router.effective_model_id("sess-pick") == runtime_id, (
        f"effective_model_id should return the routed runtime even on "
        f"empty-stream turns: {router.effective_model_id('sess-pick')}"
    )


@test("smart_router_stream", "agno route still streams real deltas (regression)")
async def t_agno_stream_unchanged(_ctx: TestContext) -> None:
    """Agno path was already correct; this is a regression guard so the
    claude-cli unification doesn't accidentally break agno streaming."""
    fake_agno = _FakeAgnoProvider(deltas=["agno", "-stream", "-ok"])
    router = _make_router_for_stream(agno_provider=fake_agno)
    _stub_routing(router, "openai:gpt-4o-mini")

    yielded: list[str] = []
    async for chunk in router.stream(
        [{"role": "user", "content": "hi"}], session_id="sess-agno",
    ):
        yielded.append(chunk)

    assert yielded == ["agno", "-stream", "-ok"], yielded
    assert len(fake_agno.stream_calls) == 1


@test("smart_router_stream", "agno route forwards session_id (multi-tab history isolation)")
async def t_agno_stream_forwards_session_id(_ctx: TestContext) -> None:
    """Two browser tabs hitting the same agno model must each keep their
    own RAM history. SmartRouter.stream historically called
    ``provider.stream(messages, system=system, tools=tools)`` WITHOUT
    forwarding session_id — combined with AgnoProvider.stream's
    hardcoded ``sid = "default"``, two concurrent streams collided on
    one Agno session and the second message stomped the first's history
    mid-turn."""
    fake_agno = _FakeAgnoProvider(deltas=["x"])
    router = _make_router_for_stream(agno_provider=fake_agno)
    _stub_routing(router, "openai:gpt-4o-mini")

    async def on_status(_text: str) -> None:
        pass

    async for _ in router.stream(
        [{"role": "user", "content": "hi"}],
        session_id="sess-agno-distinct",
        on_status=on_status,
    ):
        pass

    assert len(fake_agno.stream_calls) == 1
    call = fake_agno.stream_calls[0]
    assert call["session_id"] == "sess-agno-distinct", (
        f"session_id must reach AgnoProvider.stream: {call}"
    )
    assert call["on_status"] is True, (
        "on_status must reach AgnoProvider.stream for parity with claude-cli"
    )
