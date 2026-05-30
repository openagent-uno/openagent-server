"""Regression: team members must run with the same MCP setup and the
same framework+persona system prompt as the team leader.

Vision §15 mandates that every AI execution within OpenAgent runs with
the same framework prompt, the same user persona, and the same
deferred-MCP setup — leader and every team member alike, regardless of
which provider backs them.

The dispatcher implements this by:
- ``_compose_member_system(system, role_blurb)`` — wraps the leader's
  ``framework + persona`` base with a short ``── Role ──`` suffix for
  members; same base text in both.
- ``_build_api_agent_for`` and ``_build_subscription_agent_for`` —
  both pull the tool-search toolkit (the single in-process MCP that
  fronts every other capability per vision §6) from the same pool
  accessor.

This file pins those invariants so a refactor can't quietly diverge
leader vs. member setup.
"""
from __future__ import annotations

from typing import Any

from ._framework import TestContext, test


# ── Unit: _compose_member_system ────────────────────────────────────


@test("team_member_parity", "_compose_member_system: empty base → None")
async def t_compose_empty(_ctx: TestContext) -> None:
    from src.models.dispatcher import _compose_member_system
    assert _compose_member_system(None, "coder") is None
    assert _compose_member_system("", "coder") is None
    assert _compose_member_system("   ", "coder") is None


@test("team_member_parity", "_compose_member_system: role-less call → base unchanged")
async def t_compose_no_role(_ctx: TestContext) -> None:
    from src.models.dispatcher import _compose_member_system
    base = "FRAMEWORK_PROMPT\n\nuser persona text"
    assert _compose_member_system(base, "") == base
    assert _compose_member_system(base, "   ") == base


@test(
    "team_member_parity",
    "_compose_member_system: framework+persona base preserved verbatim ahead of role",
)
async def t_compose_keeps_base(_ctx: TestContext) -> None:
    from src.models.dispatcher import _compose_member_system
    base = "FRAMEWORK_PROMPT_TOKEN\n\nuser persona PERSONA_TOKEN"
    composed = _compose_member_system(base, "long-context reasoning")
    assert composed is not None
    # Base must come first (members are framework+persona THEN role suffix —
    # never the other way, since the role would otherwise override the
    # framework guidance).
    assert composed.startswith(base), composed[:200]
    assert "── Role ──" in composed, composed
    assert "long-context reasoning" in composed, composed


# ── Integration: leader vs. member built side-by-side ───────────────


class _FakeToolSearchToolkit:
    """Sentinel mimicking the in-process tool-search toolkit."""

    name = "tool-search"


class _FakePool:
    """Minimal stand-in for ``MCPPool`` exposing only the accessor the
    dispatcher needs to wire MCPs into a member.
    """

    def __init__(self, ts: _FakeToolSearchToolkit) -> None:
        self._ts = ts
        self.runtime_calls = 0
        self.claude_sdk_calls = 0

    def runtime_toolkits_tool_search_only(self) -> list[Any]:
        self.runtime_calls += 1
        return [self._ts]

    def claude_sdk_servers_tool_search_only(self) -> dict[str, dict[str, Any]]:
        self.claude_sdk_calls += 1
        return {"tool-search": {"command": "noop"}}


def _make_provider(pool: _FakePool):
    """Build a TeamRouterProvider rigged for parity testing.

    Replaces ``NativeProvider.build_runtime_model`` with a sentinel so
    the test doesn't need a real provider config or API key — we only
    care about the ``RuntimeAgent`` constructor args (tools, system).
    """
    from src.models.dispatcher import TeamRouterProvider
    from src.models import native_provider as np_mod

    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=[
            {"name": "openai", "api_key": "x", "framework": "api-based"}
        ],
    )
    provider._mcp_pool = pool  # type: ignore[attr-defined]

    original = np_mod.NativeProvider.build_runtime_model

    def fake_build(self) -> Any:
        # ``RuntimeAgent.__init__`` runs the model through ``get_model``
        # which accepts ``None`` as a no-op. We're only inspecting
        # ``tools`` / ``system_message`` so the actual model isn't needed.
        return None

    np_mod.NativeProvider.build_runtime_model = fake_build  # type: ignore[assignment]
    return provider, lambda: setattr(
        np_mod.NativeProvider, "build_runtime_model", original,
    )


def _entry(runtime_id: str = "openai:gpt-4o-mini", tier_hint: str | None = None):
    from src.models.catalog import CatalogModel
    return CatalogModel(
        provider="openai",
        model_id="gpt-4o-mini",
        runtime_id=runtime_id,
        history_mode="platform",
        framework="api-based",
        tier_hint=tier_hint,
    )


@test(
    "team_member_parity",
    "api-based leader & member: same tool-search toolkit instance",
)
async def t_api_mcp_parity(_ctx: TestContext) -> None:
    ts = _FakeToolSearchToolkit()
    pool = _FakePool(ts)
    provider, undo = _make_provider(pool)
    try:
        leader = provider._build_api_agent_for(
            _entry(), name="leader", role=None, system="BASE",
        )
        member = provider._build_api_agent_for(
            _entry(runtime_id="openai:gpt-4o", tier_hint="coding"),
            name="member-coder",
            role="coding",
            system="BASE\n\n── Role ──\nYou are this team's coding specialist…",
        )
    finally:
        undo()

    # Same accessor used for both → same instance returned.
    leader_tools = list(getattr(leader, "tools", []) or [])
    member_tools = list(getattr(member, "tools", []) or [])
    assert leader_tools, f"leader has no tools wired: {leader_tools}"
    assert member_tools, f"member has no tools wired: {member_tools}"
    assert leader_tools[0] is ts, (
        "leader's tool-search toolkit must be the canonical pool instance"
    )
    assert member_tools[0] is ts, (
        "member's tool-search toolkit must be the SAME canonical pool "
        "instance the leader uses — that's how vision §15 'same deferred-MCP "
        "setup' is satisfied"
    )
    # Pool accessor called once per build — confirms no divergent accessor.
    assert pool.runtime_calls == 2, pool.runtime_calls


@test(
    "team_member_parity",
    "api-based leader & member: framework+persona base reaches RuntimeAgent.system_message",
)
async def t_api_system_prompt_parity(_ctx: TestContext) -> None:
    from src.models.dispatcher import _compose_member_system

    ts = _FakeToolSearchToolkit()
    pool = _FakePool(ts)
    provider, undo = _make_provider(pool)
    base = "FRAMEWORK_TOKEN\n\nuser persona PERSONA_TOKEN"
    try:
        leader = provider._build_api_agent_for(
            _entry(), name="leader", role=None, system=base,
        )
        member_system = _compose_member_system(base, "coding")
        member = provider._build_api_agent_for(
            _entry(runtime_id="openai:gpt-4o", tier_hint="coding"),
            name="member-coder",
            role="coding",
            system=member_system,
        )
    finally:
        undo()

    leader_sys = getattr(leader, "system_message", None)
    member_sys = getattr(member, "system_message", None)
    assert leader_sys == base, leader_sys
    assert member_sys is not None and member_sys.startswith(base), (
        "member's system_message must START with the same framework+persona "
        f"base the leader uses; got {member_sys!r}"
    )
    # Member additionally carries the role suffix; leader doesn't.
    assert "── Role ──" not in (leader_sys or ""), leader_sys
    assert "── Role ──" in member_sys, member_sys


# ── Subscription-CLI builders: smoke check that the same pool accessor
# is reached. Avoids invoking the SDK (no claude/codex binary in CI).
# ──────────────────────────────────────────────────────────────────────


@test(
    "team_member_parity",
    "claude-cli builder uses claude_sdk_servers_tool_search_only for both leader+member",
)
async def t_claude_cli_mcp_accessor(_ctx: TestContext) -> None:
    from src.models.dispatcher import TeamRouterProvider
    from src.models import claude_agent as ca_mod

    ts = _FakeToolSearchToolkit()
    pool = _FakePool(ts)
    provider = TeamRouterProvider(
        entry_runtime_id="claude-cli:anthropic:claude-opus-4-7",
        providers_config=[],
    )
    provider._mcp_pool = pool  # type: ignore[attr-defined]

    captured: list[dict[str, Any]] = []

    def fake_build(**kwargs):
        captured.append(kwargs)
        return object()

    original = ca_mod.build_claude_backed_agent
    ca_mod.build_claude_backed_agent = fake_build  # type: ignore[assignment]
    try:
        entry = _entry(
            runtime_id="claude-cli:anthropic:claude-opus-4-7",
        )
        # The dataclass is frozen; rebuild with the claude-cli framework.
        from src.models.catalog import CatalogModel
        entry = CatalogModel(
            provider="anthropic", model_id="claude-opus-4-7",
            runtime_id="claude-cli:anthropic:claude-opus-4-7",
            history_mode="platform", framework="claude-cli",
        )
        provider._build_subscription_agent_for(
            entry, name="leader", role=None, system="BASE",
        )
        provider._build_subscription_agent_for(
            entry, name="member-coder", role="coding",
            system="BASE\n\n── Role ──\n…",
        )
    finally:
        ca_mod.build_claude_backed_agent = original  # type: ignore[assignment]

    # Same accessor invoked both times → same MCP setup.
    assert pool.claude_sdk_calls == 2, pool.claude_sdk_calls
    # The framework prompt reached the SDK in both calls.
    systems = [c.get("system") for c in captured]
    assert all(s and s.startswith("BASE") for s in systems), systems
    # Both builds got the SAME mcp-servers dict — identical tool-search wiring.
    mcp_dicts = [c.get("mcp_servers") for c in captured]
    assert all(d and "tool-search" in d for d in mcp_dicts), mcp_dicts
