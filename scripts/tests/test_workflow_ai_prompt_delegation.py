"""Regression: a workflow ``ai-prompt`` block (or any ``model_override``
caller) must still be able to reach other registered models via the
universal ``delegation`` MCP.

Before the fix, ``Agent._run_inner`` and ``_run_inner_stream`` installed
the delegation context with ``dispatcher=active_model``. When
``model_override`` was set, ``active_model`` was a
``TeamRouterProvider`` pinned to a single entry runtime — which does
not implement ``run_delegated``. The first ``delegate_task`` call from
inside a workflow ai-prompt block raised ``AttributeError`` and the
model couldn't reach any other registered model. Vision §15 says AI
blocks fired inside a workflow run with the same sub-agent delegation
as live chat turns; the override path violated that.

The fix prefers ``self.model`` (the canonical ``ModelDispatcher`` that
owns the full catalog) whenever it implements ``run_delegated``, and
falls back to ``active_model`` only when it does not.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from ._framework import TestContext, test


class _FakeModelResponse:
    def __init__(self, content: str = "ok", model: str = "fake") -> None:
        self.content = content
        self.model = model


class _DispatcherCapableModel:
    """Stub for the canonical ``self.model`` — implements ``run_delegated``."""

    history_mode = "caller"
    model = "fake/canonical"

    def __init__(self) -> None:
        self.generate_calls = 0

    def effective_model_id(self, session_id: str | None = None) -> str | None:
        return self.model

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
        **_: Any,
    ) -> _FakeModelResponse:
        self.generate_calls += 1
        return _FakeModelResponse(content="canonical-ok", model=self.model)

    async def run_delegated(self, **_: Any) -> str:
        return "delegated-ok"

    async def close_session(self, session_id: str) -> None:
        return None


class _PinnedOverrideModel:
    """Stub for a ``TeamRouterProvider``-style override — NO ``run_delegated``."""

    history_mode = "caller"
    model = "fake/override"

    def __init__(self) -> None:
        self.generate_calls = 0

    def effective_model_id(self, session_id: str | None = None) -> str | None:
        return self.model

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
        **_: Any,
    ) -> _FakeModelResponse:
        self.generate_calls += 1
        return _FakeModelResponse(content="override-ok", model=self.model)

    async def close_session(self, session_id: str) -> None:
        return None


def _make_agent(model: Any) -> Any:
    from src.core.agent import Agent
    return Agent(name="test", model=model, system_prompt="test", memory=None)


def _patch_install_capture(monkeypatch_target: str, captured: list[Any]):
    """Patch ``install_context`` (imported in agent.py as
    ``install_delegation_context``) to capture the dispatcher arg and
    return a no-op token tuple compatible with ``reset_context``.
    """
    import src.mcp.servers.delegation.handlers as handlers

    original_install = handlers.install_context
    original_reset = handlers.reset_context

    def fake_install(*, session_id, pool, db, dispatcher):
        captured.append({
            "session_id": session_id,
            "pool": pool,
            "db": db,
            "dispatcher": dispatcher,
        })
        return original_install(
            session_id=session_id, pool=pool, db=db, dispatcher=dispatcher,
        )

    handlers.install_context = fake_install
    return lambda: setattr(handlers, "install_context", original_install)


@test(
    "workflow_ai_prompt_delegation",
    "model_override path installs canonical self.model as delegation dispatcher",
)
async def t_override_uses_canonical_dispatcher(_ctx: TestContext) -> None:
    canonical = _DispatcherCapableModel()
    override = _PinnedOverrideModel()
    agent = _make_agent(canonical)

    captured: list[dict[str, Any]] = []
    undo = _patch_install_capture("src.core.agent", captured)
    try:
        text = await agent.run(
            message="hi",
            user_id="workflow",
            session_id="workflow:wf-1:run-1:n1",
            model_override=override,
        )
    finally:
        undo()

    # The override still drives generation for this turn.
    assert override.generate_calls == 1, override.generate_calls
    assert canonical.generate_calls == 0, canonical.generate_calls
    assert text == "override-ok", text

    # ...but the delegation MCP must see the canonical dispatcher so
    # ``delegate_task`` can route to any other catalog model.
    assert captured, "install_delegation_context was never called"
    last = captured[-1]
    assert last["dispatcher"] is canonical, (
        "delegation dispatcher must be the canonical self.model when the "
        "override lacks run_delegated; got "
        f"{type(last['dispatcher']).__name__}"
    )


@test(
    "workflow_ai_prompt_delegation",
    "no override: dispatcher is self.model (live-chat parity)",
)
async def t_no_override_uses_self_model(_ctx: TestContext) -> None:
    canonical = _DispatcherCapableModel()
    agent = _make_agent(canonical)

    captured: list[dict[str, Any]] = []
    undo = _patch_install_capture("src.core.agent", captured)
    try:
        await agent.run(message="hi", user_id="u", session_id="sess-1")
    finally:
        undo()

    assert captured, "install_delegation_context was never called"
    assert captured[-1]["dispatcher"] is canonical, (
        f"live-chat dispatcher must be self.model; got "
        f"{type(captured[-1]['dispatcher']).__name__}"
    )


@test(
    "workflow_ai_prompt_delegation",
    "fallback: self.model without run_delegated → active_model wins",
)
async def t_fallback_when_self_model_cant_dispatch(_ctx: TestContext) -> None:
    # Defensive path: if someone wires Agent with a single provider
    # that has no ``run_delegated`` (e.g. tests), we'd rather pass the
    # active_model than ``None`` — at least the team-leader path inside
    # active_model can still dispatch.
    bare_self = _PinnedOverrideModel()  # no run_delegated
    override = _DispatcherCapableModel()  # has run_delegated
    agent = _make_agent(bare_self)

    captured: list[dict[str, Any]] = []
    undo = _patch_install_capture("src.core.agent", captured)
    try:
        await agent.run(
            message="hi",
            user_id="u",
            session_id="sess-2",
            model_override=override,
        )
    finally:
        undo()

    assert captured, "install_delegation_context was never called"
    assert captured[-1]["dispatcher"] is override, (
        "when self.model lacks run_delegated, the active_model should be "
        f"the fallback dispatcher; got {type(captured[-1]['dispatcher']).__name__}"
    )
