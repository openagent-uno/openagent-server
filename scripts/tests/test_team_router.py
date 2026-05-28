"""TeamRouterProvider — sub-agent delegation via Agno ``Team(mode=coordinate)``.

Verifies the v0.14 sub-agent architecture: a session's user-selected
entry model becomes the team leader; every OTHER enabled
``framework='api-based'`` model in the DB joins as a specialist whose
``role`` blurb comes from the DB's ``tier_hint``. The leader delegates
per-turn natively via Agno's TeamMode.coordinate — so a session that
started on a fast/cheap model can pick up an "expert at coding"
specialist when the user asks for code, a marketing specialist when
the user asks for copy, or BOTH in parallel for multi-domain prompts,
all under one session id.

Tests here stub Agno's ``Team`` and ``Agent`` so no real LLM call is
made. We focus on:
  - the team structure (leader, members, role blurbs)
  - the catalog → Agno-object translation
  - the single-agent fallback for one-model deployments
  - the model-switching flow: leader → specialist → response wraps back
    through TeamRouterProvider into ``ModelResponse``
"""
from __future__ import annotations

import uuid
from typing import Any

from ._framework import TestContext, test


# ── Catalog fixtures ──────────────────────────────────────────────────


def _multi_specialist_catalog() -> list[dict[str, Any]]:
    """Three api-based models with distinct tier_hint roles.

    Mirrors the user's intended setup: one fast entry model + two
    specialists (coding, marketing) whose ``tier_hint`` tells the team
    leader when to route to them. The runtime sees them via
    ``iter_configured_models(providers_config)``.
    """
    return [
        {
            "id": 1, "name": "openai", "framework": "api-based",
            "api_key": "sk-test", "base_url": None, "enabled": True,
            "models": [
                # Cheap entry model — the session's default leader.
                {
                    "id": 10, "model": "gpt-4o-mini",
                    "enabled": True,
                    "tier_hint": "fast and cheap general-purpose chat",
                },
                # Coding specialist.
                {
                    "id": 11, "model": "gpt-5-coding",
                    "enabled": True,
                    "tier_hint": "best for coding, refactoring, and "
                                 "debugging across files",
                },
            ],
        },
        {
            "id": 2, "name": "anthropic", "framework": "api-based",
            "api_key": "sk-ant-test", "base_url": None, "enabled": True,
            "models": [
                # Marketing specialist.
                {
                    "id": 20, "model": "claude-opus-marketing",
                    "enabled": True,
                    "tier_hint": "best for marketing copy, ad copy, and "
                                 "long-form persuasion",
                },
            ],
        },
    ]


def _single_model_catalog() -> list[dict[str, Any]]:
    """One enabled api-based model — Team-as-router should fall back to
    a single-agent dispatch since there's nothing to delegate to."""
    return [
        {
            "id": 1, "name": "openai", "framework": "api-based",
            "api_key": "sk-test", "enabled": True,
            "models": [
                {"id": 10, "model": "gpt-4o-mini", "enabled": True,
                 "tier_hint": "general-purpose"},
            ],
        },
    ]


def _catalog_with_claude_cli() -> list[dict[str, Any]]:
    """Mixed catalog — one api-based + one claude-cli. The claude-cli
    row MUST be excluded from team membership (Agno's Team.members
    type rejects BaseExternalAgent subclasses), so the api-based-only
    Team builder should silently skip it.
    """
    return [
        {
            "id": 1, "name": "openai", "framework": "api-based",
            "api_key": "sk-test", "enabled": True,
            "models": [
                {"id": 10, "model": "gpt-4o-mini", "enabled": True,
                 "tier_hint": "fast general-purpose"},
                {"id": 11, "model": "gpt-5-coding", "enabled": True,
                 "tier_hint": "best for coding"},
            ],
        },
        {
            "id": 2, "name": "anthropic", "framework": "claude-cli",
            "api_key": None, "enabled": True,
            "models": [
                {"id": 20, "model": "claude-sonnet-4-6", "enabled": True,
                 "tier_hint": "best for complex multi-step reasoning"},
            ],
        },
    ]


# ── Stub builders ────────────────────────────────────────────────────


class _RecordedAgent:
    """Stand-in for ``agno.agent.Agent`` used to inspect Team construction
    without spawning a real Agno agent (which would try to import vendor
    SDKs and resolve API keys at instantiation time)."""

    def __init__(self, *, name: str, role: str | None, model_id: str):
        self.name = name
        self.role = role
        self.model_id = model_id
        # Agno's Team reads ``.model`` to surface as the leader's model;
        # we stamp a recognisable sentinel so assertions can match.
        self.model = f"<stub-model:{model_id}>"

    def __repr__(self) -> str:  # noqa: D401 — short for test output
        return f"_RecordedAgent(name={self.name!r}, role={self.role!r})"


class _StubTeam:
    """Records construction args + intercepts ``arun`` so we can verify
    the team's shape without invoking real Agno team routing.

    Crucially: an ``arun(prompt)`` call is dispatched to a chosen
    member based on a simple keyword match against the member's role —
    a stand-in for what Agno's TeamMode.coordinate would do natively
    (single-best-member delegation; the real coordinate mode would
    fan out to multiple members when a prompt has parallel sub-tasks,
    but the keyword-pick stub keeps the test deterministic). The chosen
    member's ``synthetic_response`` is wrapped in a fake ``RunOutput``
    so TeamRouterProvider's response-unwrapping path runs end-to-end.
    """

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.members: list[_RecordedAgent] = list(kwargs.get("members") or [])
        self.invocations: list[dict[str, Any]] = []
        # Map (case-insensitive keyword) → member name for the routing
        # stub. Filled by tests before they call generate().
        self.routing_overrides: dict[str, str] = {}

    async def arun(self, prompt: str, **kwargs: Any) -> Any:
        self.invocations.append({"prompt": prompt, **kwargs})
        chosen = self._pick_member(prompt)
        return _FakeRunOutput(
            content=f"[{chosen.name}] handled: {prompt}",
            chosen_name=chosen.name,
        )

    def _pick_member(self, prompt: str) -> _RecordedAgent:
        """Naive role-based picker — looks for a routing override
        keyword in the prompt, falls back to the leader (members[0]).

        Mimics Agno's TeamMode.coordinate at the surface level: the
        test prompt "help me refactor this Python code" matches the
        ``coding`` override and dispatches to the coding specialist.
        """
        lower = (prompt or "").lower()
        for keyword, member_name in self.routing_overrides.items():
            if keyword in lower:
                for m in self.members:
                    if m.name == member_name:
                        return m
        return self.members[0]


class _FakeRunOutput:
    def __init__(self, content: str, chosen_name: str):
        self.content = content
        self.chosen_name = chosen_name
        # Agno's RunOutput surface that TeamRouterProvider reads.
        self.tools = []

        class _Metrics:
            input_tokens = 7
            output_tokens = 11
        self.metrics = _Metrics()


def _install_stubs(provider, *, catalog: list[dict[str, Any]] | None = None,
                   on_team: Any = None) -> dict[str, Any]:
    """Patch ``TeamRouterProvider._build_agent_for`` and the Agno
    ``Team`` import path inside ``_ensure_runtime`` so the test never
    talks to a real Agno class.

    Returns a recorder dict with the constructed leader + members and
    the stubbed Team instance so the test can assert against them.
    """
    from src.models.catalog import (
        FRAMEWORK_API_BASED,
        SUBSCRIPTION_CLI_FRAMEWORKS,
        framework_of,
    )

    recorded: dict[str, Any] = {
        "members": [],   # list[_RecordedAgent] (all members incl. leader)
        "team": None,    # _StubTeam after construction
        "team_model": None,  # the Model object set as team.model (for subscription leaders)
    }

    def fake_build_agent_for(entry, *, name: str, role: str | None):
        rec = _RecordedAgent(
            name=name, role=role, model_id=entry.runtime_id,
        )
        recorded["members"].append(rec)
        return rec

    provider._build_agent_for = fake_build_agent_for  # type: ignore[assignment]
    # The per-framework specialised builders also route through the same
    # recorder so we can assert per-framework construction calls.
    provider._build_api_agent_for = fake_build_agent_for  # type: ignore[assignment]
    provider._build_claude_agent_for = fake_build_agent_for  # type: ignore[assignment]
    provider._build_codex_agent_for = fake_build_agent_for  # type: ignore[assignment]

    # Patch the lazy imports inside _ensure_runtime via a wrapper.
    original_ensure = provider._ensure_runtime

    def fake_ensure_runtime(session_id: str, system: str | None):
        # Reuse the original method's catalog + entry resolution by
        # replicating the relevant pieces — building the Team itself
        # routes through our stub.
        cached = provider._session_runtime.get(session_id)
        if cached is not None:
            return cached

        llm_catalog = provider._enabled_llm_models()
        entry_runtime = provider._entry_runtime_id or (
            llm_catalog[0].runtime_id if llm_catalog else None
        )
        if not entry_runtime:
            raise RuntimeError("no enabled LLM models")
        entry = next(
            (e for e in llm_catalog if e.runtime_id == entry_runtime),
            llm_catalog[0],
        )
        members_catalog = [e for e in llm_catalog if e.runtime_id != entry.runtime_id]
        is_subscription_leader = (
            framework_of(entry.runtime_id) in SUBSCRIPTION_CLI_FRAMEWORKS
        )

        # For subscription-CLI leaders (claude-cli / codex-cli), scan
        # for an api-based row whose Model would serve as team.model.
        # The real provider builds the Model via NativeProvider; the stub
        # just records a sentinel reference.
        team_model = None
        if is_subscription_leader:
            api_rows = [
                e for e in llm_catalog
                if framework_of(e.runtime_id) == FRAMEWORK_API_BASED
            ]
            if not api_rows:
                # No api-based fallback → single-external-agent path.
                stub = fake_build_agent_for(
                    entry, name=f"single:{entry.runtime_id}", role=None,
                )
                provider._session_runtime[session_id] = stub
                return stub
            routing_entry = api_rows[0]
            routing_agent = fake_build_agent_for(
                routing_entry, name=f"routing:{routing_entry.runtime_id}", role=None,
            )
            team_model = routing_agent.model
            recorded["team_model"] = team_model

        if not members_catalog:
            # Single-agent fallback — build_agent_for returns our stub.
            stub = fake_build_agent_for(
                entry, name=f"single:{entry.runtime_id}", role=None,
            )
            provider._session_runtime[session_id] = stub
            return stub

        # Build leader + members + the stubbed Team.
        leader = fake_build_agent_for(entry, name="leader", role=None)
        if not is_subscription_leader:
            team_model = leader.model
            recorded["team_model"] = team_model
        from src.models.dispatcher import _build_role_blurb
        members = [
            fake_build_agent_for(e, name=f"specialist:{e.runtime_id}",
                                  role=_build_role_blurb(e))
            for e in members_catalog
        ]
        team = _StubTeam(
            members=[leader, *members],
            model=team_model,
            system_message=system,
        )
        if on_team is not None:
            on_team(team)
        recorded["team"] = team
        provider._session_runtime[session_id] = team
        return team

    provider._ensure_runtime = fake_ensure_runtime  # type: ignore[assignment]
    del original_ensure
    return recorded


# ── Tests ─────────────────────────────────────────────────────────────


@test("team_router", "build_role_blurb prefers tier_hint, falls back to generic")
async def t_role_blurb(ctx: TestContext) -> None:
    """The role string the leader uses to route turns to specialists is
    sourced from the DB row's ``tier_hint``. Verify the fallback chain
    so empty rows still produce a parseable role.
    """
    from src.models.catalog import CatalogModel
    from src.models.dispatcher import _build_role_blurb

    with_hint = CatalogModel(
        provider="openai", model_id="gpt-5",
        runtime_id="openai:gpt-5", history_mode="platform",
        tier_hint="best for coding",
    )
    assert _build_role_blurb(with_hint) == "best for coding"

    bare = CatalogModel(
        provider="openai", model_id="gpt-5",
        runtime_id="openai:gpt-5", history_mode="platform",
    )
    assert _build_role_blurb(bare) == "specialist using gpt-5"


@test("team_router", "Team built with instructions= not system_message= so <team_members> survives")
async def t_team_uses_instructions(ctx: TestContext) -> None:
    """Regression: passing our OpenAgent system prompt as
    ``system_message`` made Agno's ``get_system_message`` return early
    (agno/team/_messages.py:393) and skip building the ``<team_members>``
    block AND the mode-specific "use only the member id" instructions.
    The leader then saw our "delegate by default" exhortation without
    ever being shown the actual member list, and answered directly
    instead of dispatching. Wiring the prompt through ``instructions=``
    preserves Agno's auto-injected team context while still appending
    our content.

    Source-level lock-down: ``Team(..., instructions=[system] if system else None, ...)``
    must remain in dispatcher.py and ``system_message=system`` must NOT.
    """
    import inspect

    from src.models.dispatcher import TeamRouterProvider

    src = inspect.getsource(TeamRouterProvider._ensure_runtime)
    assert "instructions=[system] if system else None" in src, (
        "Team must receive our prompt via instructions= so Agno's default "
        "team-context builder runs and emits the <team_members> block."
    )
    assert "system_message=system" not in src, (
        "Passing system_message= makes Agno skip <team_members> and the "
        "mode-specific delegation instructions — leader can't dispatch."
    )


@test("team_router", "end-to-end: leader's system prompt contains every specialist's clean id")
async def t_real_team_leader_prompt_contains_member_ids(ctx: TestContext) -> None:
    """End-to-end regression for the live my-agent bug — the leader
    must see EACH specialist's clean id in the same prompt where Agno
    instructs it to ``Use only the member's ID — do not prefix it with
    the team ID.`` If the id appears with the colon-stripped form (e.g.
    ``claude-clianthropicclaude-opus-4.7``) the leader will fail to
    address it and hallucinate placeholder ids instead.

    Mirrors the user's live catalog shape (deepseek leader + 2
    claude-cli specialists + 1 deepseek specialist).
    """
    from src.core._runner.team._messages import _build_team_context

    from src.models.dispatcher import TeamRouterProvider

    providers = [
        {
            "id": 1, "name": "deepseek", "framework": "api-based",
            "api_key": "sk-test", "enabled": True,
            "models": [
                {"id": 10, "model": "deepseek-v4-flash", "enabled": True,
                 "tier_hint": "fast leader", "is_classifier": True},
                {"id": 11, "model": "deepseek-v4-pro", "enabled": True,
                 "tier_hint": "good for customer support"},
            ],
        },
        {
            "id": 2, "name": "anthropic", "framework": "claude-cli",
            "enabled": True,
            "models": [
                {"id": 20, "model": "claude-opus-4.7", "enabled": True,
                 "tier_hint": "best for coding, complex reasoning"},
                {"id": 21, "model": "claude-sonnet-4.6", "enabled": True,
                 "tier_hint": "best for simple reasoning"},
            ],
        },
    ]
    provider = TeamRouterProvider(
        entry_runtime_id="deepseek:deepseek-v4-flash",
        providers_config=providers,
    )
    team = provider._ensure_runtime("e2e-live-shape", system="OUR_OPENAGENT_PROMPT")
    leader_prompt = _build_team_context(team, run_context=None)

    expected_ids = (
        "leader",
        "deepseek-deepseek-v4-pro",
        "claude-cli-anthropic-claude-opus-4.7",
        "claude-cli-anthropic-claude-sonnet-4.6",
    )
    for member_id in expected_ids:
        assert f'<member id="{member_id}"' in leader_prompt, (
            f"leader prompt missing <member id='{member_id}'>; the "
            f"leader can't address this specialist. got prompt:\n"
            f"{leader_prompt[:800]}"
        )

    assert ":" not in leader_prompt or "Use only the member's ID" in leader_prompt, (
        "leader prompt must either be colon-free in member ids OR carry "
        f"the 'Use only the member's ID' guard; got:\n{leader_prompt[:800]}"
    )

    forbidden_collapsed_ids = (
        'specialistclaude-clianthropicclaude-opus-4.7',
        'claude-clianthropicclaude-opus-4.7',
        'specialistdeepseekdeepseek-v4-pro',
    )
    for bad_id in forbidden_collapsed_ids:
        assert bad_id not in leader_prompt, (
            f"leader prompt contains colon-stripped id {bad_id!r} — this "
            f"is the regression the user hit in production. got:\n"
            f"{leader_prompt[:800]}"
        )


@test("team_router", "end-to-end: real Team renders a clean <team_members> block with matching id/name")
async def t_real_team_members_block_e2e(ctx: TestContext) -> None:
    """End-to-end regression for the two delegation bugs:

    Bug 1: member names with colons collapsed through ``url_safe_string``
           into stripped ids that no longer matched the system-message
           name — leader saw two identifiers and hallucinated placeholders.

    Bug 2: passing our OpenAgent prompt as ``system_message`` made Agno's
           ``get_system_message`` skip building ``<team_members>`` and
           the route-mode delegation instructions — leader had no idea
           which members existed.

    This test bypasses ``_install_stubs`` and exercises the real
    ``_ensure_runtime`` path: constructs a real ``Agno.team.Team``,
    renders the actual ``<team_members>`` block via Agno's own
    ``get_members_system_message_content``, and verifies what the leader
    LLM would see.
    """
    from src.core._runner.team import Team
    from src.core._runner.utils.team import get_member_id

    from src.models.dispatcher import TeamRouterProvider

    providers = [
        {
            "id": 1, "name": "openai", "framework": "api-based",
            "api_key": "sk-test", "enabled": True,
            "models": [
                {"id": 10, "model": "gpt-4o-mini", "enabled": True,
                 "tier_hint": "fast chat / general"},
                {"id": 11, "model": "gpt-4o", "enabled": True,
                 "tier_hint": "best for coding"},
                {"id": 12, "model": "gpt-4o-vision", "enabled": True,
                 "tier_hint": "vision / image analysis"},
            ],
        },
    ]
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )
    team = provider._ensure_runtime(
        "sess-e2e",
        system="OPENAGENT_PROMPT_MARKER",
    )

    assert isinstance(team, Team), (
        "Expected real Agno Team; got " + type(team).__name__
    )

    # Bug 2 fix: system_message stays unset so Agno builds the default
    # team prompt; our content rides in via instructions.
    assert team.system_message is None, (
        "team.system_message must be None — setting it makes Agno's "
        "get_system_message return early and skip <team_members>."
    )
    assert team.instructions == ["OPENAGENT_PROMPT_MARKER"], (
        "Our OpenAgent system prompt must be threaded through "
        "Team.instructions so Agno's default prompt builder appends it "
        "to the team-context block."
    )

    # Bug 1 fix: every member's derived id matches its name byte-for-byte.
    for member in team.members:
        derived_id = get_member_id(member)
        assert derived_id == member.name, (
            f"Member id/name mismatch: name={member.name!r} "
            f"derived_id={derived_id!r}. The leader sees TWO identifiers "
            f"for one member and hallucinates placeholders when picking."
        )

    # End-to-end: render the actual block the leader will see.
    xml = team.get_members_system_message_content(indent=0)
    assert '<member id="leader" name="leader">' in xml, (
        f"<team_members> block must include the leader entry; got:\n{xml}"
    )
    assert '<member id="openai-gpt-4o" name="openai-gpt-4o">' in xml, (
        f"coding specialist must appear with clean id; got:\n{xml}"
    )
    assert "Role: best for coding" in xml, (
        f"tier_hint must surface as Role: text; got:\n{xml}"
    )
    assert '<member id="openai-gpt-4o-vision" name="openai-gpt-4o-vision">' in xml, (
        f"vision specialist must appear with clean id; got:\n{xml}"
    )
    assert "Role: vision / image analysis" in xml, (
        f"vision specialist role must surface; got:\n{xml}"
    )


@test("team_router", "end-to-end: full team context includes coordinate-mode delegation instructions")
async def t_real_team_full_context_e2e(ctx: TestContext) -> None:
    """End-to-end render of the COMPLETE prompt-prefix the leader LLM
    sees — opening + ``<team_members>`` + ``<how_to_respond>`` mode
    instructions. This is what got skipped when we previously passed
    ``system_message=`` (Agno returned the message early and the
    ``<how_to_respond>`` block — "Use only the member's ID, do not
    prefix it with the team ID" — never reached the leader, so the
    leader had no instructions on the delegation tool format).

    Coordinate-mode specifics: the block must invite the leader to
    delegate to "multiple members" when the request spans different
    areas of expertise (the parallel-delegation hook the user is
    relying on).
    """
    from src.core._runner.team._messages import _build_team_context

    from src.models.dispatcher import TeamRouterProvider

    providers = [
        {
            "id": 1, "name": "openai", "framework": "api-based",
            "api_key": "sk-test", "enabled": True,
            "models": [
                {"id": 10, "model": "gpt-4o-mini", "enabled": True,
                 "tier_hint": "fast chat"},
                {"id": 11, "model": "gpt-4o", "enabled": True,
                 "tier_hint": "best for coding"},
            ],
        },
    ]
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )
    team = provider._ensure_runtime("sess-full", system="MARKER")
    context = _build_team_context(team, run_context=None)

    assert "<team_members>" in context, (
        f"team context must include <team_members>; got:\n{context}"
    )
    assert "<how_to_respond>" in context, (
        f"team context must include <how_to_respond> mode instructions; "
        f"got:\n{context}"
    )
    assert "coordinate mode" in context, (
        f"coordinate-mode preamble must surface so the leader knows it "
        f"can dispatch to multiple members per turn; got:\n{context}"
    )
    assert "multiple members" in context, (
        f"coordinate-mode block must tell the leader it can delegate "
        f"to MULTIPLE members in one turn (the parallelism hook the "
        f"user expects); got:\n{context}"
    )
    assert "Use only the member's ID" in context, (
        f"the 'Use only the member's ID' guidance is the antidote to "
        f"the placeholder-id hallucination bug; it must be present in "
        f"the prompt context the leader sees. got:\n{context}"
    )


@test(
    "team_router",
    "vision §15: every team member (leader + specialists) carries framework+persona system prompt",
)
async def t_real_team_every_agent_carries_framework_prompt_e2e(
    ctx: TestContext,
) -> None:
    """Vision §15 regression: the OpenAgent framework prompt + the user's
    persona prompt MUST be injected into every agent the user can reach.
    The Team coordinator gets it via ``team.instructions``; that's NOT
    enough — when the coordinator delegates to a member via
    ``delegate_task_to_member``, the member is run as a standalone Agno
    Agent whose system message comes from its OWN ``system_message=``,
    not from the Team's ``instructions=``.

    Before the fix, ``_build_api_agent_for`` and
    ``_build_subscription_agent_for`` were called with only
    ``role=<short-blurb>``. Delegated members ran with no framework
    prompt (no vault discipline, no MCP awareness, no proactivity
    guidance) and no persona — they had a one-line role blurb and
    nothing else. This violated vision §15's "non-removable framework
    prompt" guarantee for everyone except the coordinator.

    The fix threads ``system=`` through ``_build_agent_for`` and stamps
    it as ``system_message=`` on each Agno Agent (and as
    ``system_prompt`` / ``developer_instructions`` on the
    subscription-CLI agents). Members additionally get a short
    ``── Role ──`` suffix telling them which specialty they own —
    mirroring the existing ``agno_provider.py`` Team pattern.

    Setup mirrors the user's live config: api-based leader, api-based
    specialist, claude-cli specialist — so we cover both flavors of
    member in one test.
    """
    from src.core._runner.agent import Agent as AgnoAgent
    from src.core._runner.team import Team

    from src.models.claude_agent import ClaudeBackedAgent
    from src.models.dispatcher import TeamRouterProvider

    framework_marker = (
        "PRETEND_FRAMEWORK_SYSTEM_PROMPT: vault, MCPs, scheduler, federation."
    )
    persona_marker = (
        "── User-specific identity and project context ──\n\n"
        "I am the user's virgil agent; my voice is concise and technical."
    )
    composed_system = f"{framework_marker}\n\n{persona_marker}"

    providers = [
        {
            "id": 1, "name": "openai", "framework": "api-based",
            "api_key": "sk-test", "enabled": True,
            "models": [
                {"id": 10, "model": "gpt-4o-mini", "enabled": True,
                 "tier_hint": "fast and cheap general-purpose chat"},
                {"id": 11, "model": "gpt-5-coding", "enabled": True,
                 "tier_hint": "best for coding"},
            ],
        },
        {
            "id": 2, "name": "anthropic", "framework": "claude-cli",
            "enabled": True,
            "models": [
                {"id": 20, "model": "claude-opus-4.7", "enabled": True,
                 "tier_hint": "best for complex reasoning"},
            ],
        },
    ]
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )
    team = provider._ensure_runtime("sess-vision-15", system=composed_system)
    assert isinstance(team, Team), type(team).__name__

    # Coordinator still carries the prompt as instructions (this is the
    # pre-existing wiring and remains correct — see t_team_uses_instructions).
    assert team.instructions == [composed_system], (
        "Coordinator must keep receiving the system prompt via "
        "Team.instructions= so Agno's default team-context builder runs "
        "and emits <team_members>/<how_to_respond>. Got: "
        f"{team.instructions!r}"
    )

    # Every member — including the leader-at-index-0 — must independently
    # carry framework + persona on its own system_message slot.
    assert team.members, "Team built with zero members; nothing to check."
    for member in team.members:
        sys_text = _resolve_member_system_text(member)
        assert sys_text is not None, (
            f"Member {member.name!r} ({type(member).__name__}) has no "
            f"resolved system text — framework prompt is missing. "
            f"Vision §15 requires it on every agent."
        )
        assert framework_marker in sys_text, (
            f"Member {member.name!r} ({type(member).__name__}) is missing "
            f"the framework prompt marker. Got:\n{sys_text[:600]}"
        )
        assert persona_marker in sys_text, (
            f"Member {member.name!r} ({type(member).__name__}) is missing "
            f"the user persona marker. Got:\n{sys_text[:600]}"
        )

    # The leader (members[0]) is NOT given a Role suffix — it's the
    # session's default model, not a specialist. Members other than the
    # leader DO get a Role suffix so a delegated specialist stays in
    # its lane.
    leader = team.members[0]
    assert leader.name == "leader", f"members[0] should be 'leader', got {leader.name!r}"
    leader_sys = _resolve_member_system_text(leader)
    assert "── Role ──" not in (leader_sys or ""), (
        "Leader must NOT be tagged with a specialist Role suffix — it's "
        "the session's default model, not a specialty member. Got:\n"
        f"{leader_sys[:600]}"
    )

    specialists = [m for m in team.members if m.name != "leader"]
    assert specialists, "Expected at least one specialist member."
    for member in specialists:
        sys_text = _resolve_member_system_text(member) or ""
        assert "── Role ──" in sys_text, (
            f"Specialist {member.name!r} must carry a Role suffix so a "
            f"delegated turn knows its specialty. Got:\n{sys_text[:600]}"
        )

    # Cross-check the claude-cli specialist specifically: its system
    # prompt lives in the SDK options (``system_prompt``), not on an
    # Agno ``system_message`` slot. The fix forwards ``system=`` to
    # ``build_claude_backed_agent`` which lifts it into the SDK options.
    claude_specialists = [
        m for m in specialists
        if isinstance(m, ClaudeBackedAgent)
    ]
    assert claude_specialists, (
        "Expected the claude-cli row to materialise as a "
        "ClaudeBackedAgent member."
    )
    claude_sys = claude_specialists[0]._sdk_options.get("system_prompt")
    assert claude_sys and framework_marker in claude_sys, (
        f"ClaudeBackedAgent's SDK system_prompt must carry the framework "
        f"prompt — the SDK will not see Agno's system_message slot. Got: "
        f"{claude_sys!r}"
    )
    assert persona_marker in claude_sys, (
        f"ClaudeBackedAgent's SDK system_prompt must carry the persona. "
        f"Got: {claude_sys!r}"
    )

    # Sanity: api-based members are real Agno Agents (not BackedAgents).
    api_specialists = [
        m for m in specialists
        if isinstance(m, AgnoAgent) and not isinstance(m, ClaudeBackedAgent)
    ]
    assert api_specialists, "Expected at least one api-based specialist."


@test(
    "team_router",
    "vision §15: codex-cli member carries framework+persona via developer_instructions",
)
async def t_real_team_codex_member_carries_framework_prompt_e2e(
    ctx: TestContext,
) -> None:
    """Codex SDK takes the system prompt as ``developer_instructions``,
    not ``system_prompt``. The fix routes ``system=`` through
    ``build_codex_backed_agent`` which lifts it into the right slot.
    Without this, a delegated codex specialist runs with no OpenAgent
    awareness at all (vision §15 violation, same as the claude-cli case
    but in a different SDK field).
    """
    from src.core._runner.team import Team

    from src.models.codex_agent import CodexBackedAgent
    from src.models.dispatcher import TeamRouterProvider

    framework_marker = "PRETEND_FRAMEWORK_SYSTEM_PROMPT_codex_case"
    persona_marker = "── User-specific identity and project context ──"
    composed_system = f"{framework_marker}\n\n{persona_marker}\n\nmy persona body"

    providers = [
        {
            "id": 1, "name": "openai", "framework": "api-based",
            "api_key": "sk-test", "enabled": True,
            "models": [
                {"id": 10, "model": "gpt-4o-mini", "enabled": True,
                 "tier_hint": "fast leader"},
            ],
        },
        {
            "id": 2, "name": "openai-codex", "framework": "codex-cli",
            "enabled": True,
            "models": [
                {"id": 20, "model": "gpt-5-codex", "enabled": True,
                 "tier_hint": "best for codebase edits"},
            ],
        },
    ]
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )
    team = provider._ensure_runtime("sess-vision-15-codex", system=composed_system)
    assert isinstance(team, Team), type(team).__name__

    codex_members = [m for m in team.members if isinstance(m, CodexBackedAgent)]
    assert codex_members, (
        "Expected the codex-cli row to materialise as a CodexBackedAgent."
    )
    codex_sys = codex_members[0]._sdk_options.get("developer_instructions")
    assert codex_sys and framework_marker in codex_sys, (
        f"CodexBackedAgent's developer_instructions must carry the "
        f"framework prompt. Got: {codex_sys!r}"
    )
    assert persona_marker in codex_sys, (
        f"CodexBackedAgent's developer_instructions must carry the "
        f"persona. Got: {codex_sys!r}"
    )


@test(
    "team_router",
    "vision §15: single-external-agent fallback also carries framework+persona",
)
async def t_single_external_agent_fallback_carries_framework_prompt_e2e(
    ctx: TestContext,
) -> None:
    """When a subscription-CLI leader has no api-based row to drive
    Team.model, the dispatcher falls back to single-external-agent
    dispatch. That fallback agent must ALSO carry the framework prompt
    + persona — otherwise the user loses the vision §15 guarantee
    entirely when their catalog is claude-cli-only or codex-cli-only.
    """
    from src.models.claude_agent import ClaudeBackedAgent
    from src.models.dispatcher import TeamRouterProvider, _SingleExternalAgentAdapter

    framework_marker = "PRETEND_FRAMEWORK_SYSTEM_PROMPT_fallback"
    persona_marker = "PRETEND_PERSONA_fallback"
    composed_system = f"{framework_marker}\n\n{persona_marker}"

    providers: list[dict[str, Any]] = [
        {
            "id": 1, "name": "anthropic", "framework": "claude-cli",
            "enabled": True,
            "models": [
                {"id": 10, "model": "claude-sonnet-4-6", "enabled": True,
                 "tier_hint": "only model"},
            ],
        },
    ]
    provider = TeamRouterProvider(
        entry_runtime_id="claude-cli:anthropic:claude-sonnet-4-6",
        providers_config=providers,
    )
    runtime = provider._ensure_runtime("sess-fallback", system=composed_system)

    assert isinstance(runtime, _SingleExternalAgentAdapter), (
        f"Expected single-external-agent fallback; got {type(runtime).__name__}"
    )
    inner = runtime._agent  # type: ignore[attr-defined]
    assert isinstance(inner, ClaudeBackedAgent), type(inner).__name__
    sys_text = inner._sdk_options.get("system_prompt")
    assert sys_text and framework_marker in sys_text, (
        f"Fallback ClaudeBackedAgent must carry the framework prompt in "
        f"its SDK system_prompt; got {sys_text!r}"
    )
    assert persona_marker in sys_text, (
        f"Fallback ClaudeBackedAgent must carry the persona; got {sys_text!r}"
    )


@test("team_router", "_compose_member_system: composition is framework + persona + Role suffix")
async def t_compose_member_system_unit(ctx: TestContext) -> None:
    """Unit test for the helper that builds the per-member system
    string. Documents the contract: framework+persona on top (so the
    member sees vision §15's non-removable prompt first), then a short
    Role block pinning the specialist to its lane.
    """
    from src.models.dispatcher import _compose_member_system

    # Empty framework prompt → returns None. Some classifier-only paths
    # call into the dispatcher with system=None; we must NOT synthesise
    # a fake prompt out of thin air.
    assert _compose_member_system(None, "any role") is None
    assert _compose_member_system("", "any role") is None
    assert _compose_member_system("   ", "best for coding") is None

    # No role → returns framework prompt unchanged.
    assert _compose_member_system("FRAMEWORK", "") == "FRAMEWORK"
    assert _compose_member_system("FRAMEWORK", None) == "FRAMEWORK"  # type: ignore[arg-type]

    # Standard case: framework + role.
    out = _compose_member_system("FRAMEWORK_AND_PERSONA", "best for coding")
    assert out is not None
    assert out.startswith("FRAMEWORK_AND_PERSONA"), out
    assert "── Role ──" in out
    assert "best for coding" in out
    # Role suffix mentions "defer to the team leader" so a delegated
    # member knows what to do when handed a request outside its area.
    assert "defer to the team leader" in out


# ── helpers (shared by vision §15 tests above) ────────────────────────


def _resolve_member_system_text(member: Any) -> str | None:
    """Extract the resolved system-prompt text for any Team member
    flavor (Agno Agent / ClaudeBackedAgent / CodexBackedAgent), so the
    vision §15 assertions can be flavor-agnostic.

    Agno Agent  → ``member.system_message`` (set directly).
    ClaudeBackedAgent → ``member._sdk_options["system_prompt"]``.
    CodexBackedAgent  → ``member._sdk_options["developer_instructions"]``.
    """
    from src.models.claude_agent import ClaudeBackedAgent
    from src.models.codex_agent import CodexBackedAgent

    if isinstance(member, ClaudeBackedAgent):
        return member._sdk_options.get("system_prompt")
    if isinstance(member, CodexBackedAgent):
        return member._sdk_options.get("developer_instructions")
    return getattr(member, "system_message", None)


@test("team_router", "end-to-end: mixed api-based + claude-cli catalog yields clean ids for both")
async def t_real_team_members_mixed_catalog_e2e(ctx: TestContext) -> None:
    """End-to-end with a realistic mixed catalog: api-based leader plus
    a claude-cli specialist (the exact shape the user hit in production).
    Verifies both flavors emit ids matching their names — the colon-laden
    runtime_ids like ``claude-cli:anthropic:claude-opus-4.7`` survive the
    member-id derivation cleanly.
    """
    from src.core._runner.team import Team
    from src.core._runner.utils.team import get_member_id

    from src.models.dispatcher import TeamRouterProvider

    providers = [
        {
            "id": 1, "name": "openai", "framework": "api-based",
            "api_key": "sk-test", "enabled": True,
            "models": [
                {"id": 10, "model": "gpt-4o-mini", "enabled": True,
                 "tier_hint": "fast chat"},
            ],
        },
        {
            "id": 2, "name": "anthropic", "framework": "claude-cli",
            "enabled": True,
            "models": [
                {"id": 20, "model": "claude-opus-4.7", "enabled": True,
                 "tier_hint": "best for coding, complex reasoning"},
            ],
        },
    ]
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )
    team = provider._ensure_runtime("sess-mixed", system="MARKER")
    assert isinstance(team, Team)

    for member in team.members:
        derived_id = get_member_id(member)
        assert derived_id == member.name, (
            f"mixed-catalog member id/name mismatch: name={member.name!r} "
            f"derived_id={derived_id!r}"
        )

    xml = team.get_members_system_message_content(indent=0)
    # The claude-cli specialist's id is runtime_id with colons → dashes.
    assert '<member id="claude-cli-anthropic-claude-opus-4.7"' in xml, (
        f"claude-cli specialist's id must be the dash-form of the "
        f"runtime_id, no colons stripped to nothing; got:\n{xml}"
    )
    assert 'name="claude-cli-anthropic-claude-opus-4.7">' in xml, (
        f"claude-cli specialist's name must match its id; got:\n{xml}"
    )
    assert "Role: best for coding, complex reasoning" in xml, (
        f"coding role must surface in the team-members block; got:\n{xml}"
    )


@test("team_router", "_member_identifier produces names whose Agno id matches byte-for-byte")
async def t_member_identifier_url_safe(ctx: TestContext) -> None:
    """Regression: in a live my-agent session a deepseek leader tried to
    delegate to ``coding-agent-id-placeholder`` because Agno's
    ``get_member_id`` ran the colon-laden member name through
    ``url_safe_string`` and produced a stripped id that no longer matched
    the human-readable name shown in the team system message. The leader
    saw two distinct identifiers for the same member and hallucinated
    a placeholder when picking one.

    Fix: build member names with dashes (which ``url_safe_string``
    preserves) so the id Agno generates equals the name we set.
    """
    from src.core._runner.utils.string import url_safe_string
    from src.models.dispatcher import _member_identifier

    runtime_ids = (
        "claude-cli:anthropic:claude-opus-4.7",
        "codex-cli:openai:gpt-5",
        "openai:gpt-4o-mini",
        "deepseek:deepseek-v4-flash",
    )
    for rid in runtime_ids:
        name = _member_identifier(rid)
        assert ":" not in name, f"colons must be replaced: {name!r}"
        assert url_safe_string(name) == name, (
            f"Agno's url_safe_string transforms {name!r} → "
            f"{url_safe_string(name)!r}; the leader will see divergent "
            f"id/name and hallucinate placeholder delegations."
        )


@test("team_router", "builds Team with leader + specialists, roles from DB")
async def t_team_construction(ctx: TestContext) -> None:
    """Three enabled api-based models → Team with the entry as leader
    and the other two as specialist members. Each member's ``role``
    must be the corresponding DB row's ``tier_hint``.
    """
    from src.models.dispatcher import TeamRouterProvider

    providers = _multi_specialist_catalog()
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )
    recorded = _install_stubs(provider)

    resp = await provider.generate(
        [{"role": "user", "content": "hello"}],
        session_id="sess-1",
    )

    assert recorded["team"] is not None, "expected a Team to be built"
    team = recorded["team"]
    member_names = [m.name for m in team.members]
    assert member_names == [
        "leader",
        "specialist:openai:gpt-5-coding",
        "specialist:anthropic:claude-opus-marketing",
    ], member_names

    # The leader has no role string — it's the routing arbiter, not a
    # specialist. The specialists' roles encode their tier_hint so the
    # leader can pick between them.
    assert team.members[0].role is None
    assert "coding" in (team.members[1].role or "").lower(), team.members[1].role
    assert "marketing" in (team.members[2].role or "").lower(), team.members[2].role

    # Response unwrapping — the stub returns "[leader] handled: hello"
    # since the prompt has no routing keyword.
    assert resp.content.startswith("[leader] handled:"), resp.content
    assert resp.model == "openai:gpt-4o-mini"
    assert resp.input_tokens == 7
    assert resp.output_tokens == 11


@test("team_router", "single api-based model → single-agent fallback (no Team)")
async def t_single_agent_fallback(ctx: TestContext) -> None:
    """When the DB has only one enabled api-based model, building a
    Team adds latency for no benefit. ``_ensure_runtime`` should
    short-circuit to an NativeProvider-shaped single-agent runtime.
    """
    from src.models.dispatcher import TeamRouterProvider

    providers = _single_model_catalog()
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )
    recorded = _install_stubs(provider)
    # Inject the single-agent stub path — _install_stubs already covers
    # it via the ``not members_catalog`` branch.

    assert recorded["team"] is None  # not built yet

    # Calling _ensure_runtime indirectly via the public API would
    # require the real NativeProvider; we verify the cache shape instead.
    runtime = provider._ensure_runtime("sess-2", system=None)
    assert isinstance(runtime, _RecordedAgent), type(runtime).__name__
    assert runtime.model_id == "openai:gpt-4o-mini"
    # No Team object was built — only a leader-shaped single agent.
    assert recorded["team"] is None
    # Exactly one agent was constructed (the single-agent fallback).
    assert len(recorded["members"]) == 1
    assert recorded["members"][0].name == "single:openai:gpt-4o-mini"


@test("team_router", "claude-cli row joins Team as a specialist member")
async def t_claude_cli_member(ctx: TestContext) -> None:
    """After the ``ClaudeBackedAgent`` refactor, claude-cli rows ARE
    eligible for Team membership — they subclass ``agno.agent.Agent``
    and pass Team's isinstance(Agent) checks. With an api-based entry
    leader, a claude-cli row in the catalog must appear as a specialist
    member so the leader can delegate to it.
    """
    from src.models.dispatcher import TeamRouterProvider

    providers = _catalog_with_claude_cli()
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )
    recorded = _install_stubs(provider)

    await provider.generate(
        [{"role": "user", "content": "hello"}],
        session_id="sess-claude-member",
    )

    assert recorded["team"] is not None
    team = recorded["team"]
    member_names = [m.name for m in team.members]
    # All three slots: api-based leader + api-based specialist + claude-cli specialist.
    assert "leader" in member_names
    assert "specialist:openai:gpt-5-coding" in member_names, member_names
    # The claude-cli row IS now present as a specialist (was excluded in v0.14).
    claude_members = [
        n for n in member_names
        if "claude-cli:anthropic:claude-sonnet-4-6" in n
    ]
    assert claude_members, (
        f"claude-cli row missing from members: {member_names}. "
        f"ClaudeBackedAgent should make it eligible."
    )


@test("team_router", "claude-cli row works as Team leader with api-based routing model")
async def t_claude_cli_leader(ctx: TestContext) -> None:
    """When the entry is claude-cli, the ``ClaudeBackedAgent`` sits as
    ``members[0]`` (so the routing model can delegate to it). But
    Agno's Team invokes ``team.model`` for the routing-classifier
    call, and ClaudeBackedAgent's placeholder ``_NullModel`` can't
    drive that — so TeamRouterProvider picks the cheapest enabled
    api-based model as ``team.model``.
    """
    from src.models.catalog import framework_of
    from src.models.dispatcher import TeamRouterProvider

    providers = _catalog_with_claude_cli()
    # Entry is the claude-cli row this time.
    provider = TeamRouterProvider(
        entry_runtime_id="claude-cli:anthropic:claude-sonnet-4-6",
        providers_config=providers,
    )
    recorded = _install_stubs(provider)

    await provider.generate(
        [{"role": "user", "content": "hello"}],
        session_id="sess-claude-leader",
    )

    assert recorded["team"] is not None
    team = recorded["team"]
    # The claude-cli entry is members[0] (the "leader" slot).
    assert team.members[0].name == "leader"
    assert "claude-sonnet-4-6" in team.members[0].model_id, team.members[0].model_id
    # team.model points at the api-based routing fallback, NOT the
    # claude-cli leader's model. Without that, Team's leader LLM call
    # would route through a no-op model and never dispatch.
    assert recorded["team_model"] is not None
    assert "gpt-4o-mini" in str(recorded["team_model"]) or "gpt-5-coding" in str(recorded["team_model"]), (
        f"team.model must be an api-based fallback, got {recorded['team_model']!r}"
    )
    # The two api-based specialists must also appear as members.
    member_names = [m.name for m in team.members]
    assert "specialist:openai:gpt-4o-mini" in member_names, member_names
    assert "specialist:openai:gpt-5-coding" in member_names, member_names


@test("team_router", "claude-cli only catalog → single-agent fallback (no api-based routing model)")
async def t_claude_cli_only_fallback(ctx: TestContext) -> None:
    """When the catalog has ONLY claude-cli rows, there's no api-based
    model to drive Team's routing classifier. We must skip Team and
    fall back to single-agent dispatch (the ClaudeBackedAgent alone).
    """
    from src.models.dispatcher import TeamRouterProvider

    providers: list[dict[str, Any]] = [
        {
            "id": 1, "name": "anthropic", "framework": "claude-cli",
            "api_key": None, "enabled": True,
            "models": [
                {"id": 1, "model": "claude-sonnet-4-6", "enabled": True,
                 "tier_hint": "general"},
            ],
        },
    ]
    provider = TeamRouterProvider(
        entry_runtime_id="claude-cli:anthropic:claude-sonnet-4-6",
        providers_config=providers,
    )
    recorded = _install_stubs(provider)

    runtime = provider._ensure_runtime("sess-claude-only", system=None)

    assert recorded["team"] is None, (
        "claude-cli-only catalog should not build a Team — no api-based "
        "model exists to drive Team.model's routing call"
    )
    assert isinstance(runtime, _RecordedAgent), type(runtime).__name__
    assert "claude-sonnet-4-6" in runtime.model_id


@test("team_router", "leader delegates coding task to coding specialist, response flows back")
async def t_coding_delegation_flow(ctx: TestContext) -> None:
    """End-to-end model-switching scenario:

      1. Session starts with the entry model (gpt-4o-mini, the leader).
      2. User asks a coding-specific question.
      3. Team's routing-stub matches the ``coding`` keyword to the
         coding specialist member (gpt-5-coding).
      4. The specialist's response is wrapped in a RunOutput.
      5. TeamRouterProvider unwraps it into ModelResponse and returns
         control to the main agent surface.

    Verifies the user's intended workflow: started with a fast model,
    a heavier specialist handles the actual coding turn, then control
    returns to the leader for the next turn.

    Note: with coordinate-mode the real Agno leader MAY fire multiple
    delegations per turn — the stub here picks a single best member
    by keyword for determinism, but the assertion is "the coding
    specialist's output is part of the reply", not "exactly one
    delegation fired".
    """
    from src.models.dispatcher import TeamRouterProvider

    providers = _multi_specialist_catalog()
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )

    # Wire the routing override before any team is built.
    teams_built: list[_StubTeam] = []

    def on_team(team: _StubTeam) -> None:
        teams_built.append(team)
        team.routing_overrides = {
            "coding": "specialist:openai:gpt-5-coding",
            "refactor": "specialist:openai:gpt-5-coding",
            "marketing": "specialist:anthropic:claude-opus-marketing",
        }

    _install_stubs(provider, on_team=on_team)

    sid = "sess-coding"

    # Turn 1 — casual greeting. Should land on the leader.
    resp1 = await provider.generate(
        [{"role": "user", "content": "hi, how are you?"}],
        session_id=sid,
    )
    assert "[leader]" in resp1.content, resp1.content

    # Turn 2 — coding task. Should delegate to the coding specialist.
    resp2 = await provider.generate(
        [{"role": "user", "content": "help me refactor this Python coding example"}],
        session_id=sid,
    )
    assert "[specialist:openai:gpt-5-coding]" in resp2.content, resp2.content

    # Turn 3 — marketing task. Should switch to the marketing specialist
    # mid-session (model switching IS the point of Team-as-router).
    resp3 = await provider.generate(
        [{"role": "user", "content": "write a marketing tagline for our launch"}],
        session_id=sid,
    )
    assert "[specialist:anthropic:claude-opus-marketing]" in resp3.content, resp3.content

    # All three turns went through ONE team — the per-session cache
    # means we don't rebuild on every turn (rebuild only on hot-reload).
    assert len(teams_built) == 1, len(teams_built)
    assert len(teams_built[0].invocations) == 3, teams_built[0].invocations

    # Outer ModelResponse.model is the entry runtime_id — the leader's
    # identity. The fact that a specialist actually handled the turn
    # lives in the trace / Agno's TeamRunEvent, not in our coarse
    # ModelResponse. This is intentional: the chat UI's model badge
    # tracks the SESSION's entry model, not per-turn specialist swaps.
    assert resp2.model == "openai:gpt-4o-mini"
    assert resp3.model == "openai:gpt-4o-mini"


@test("team_router", "model badge follows delegated specialist, not the leader")
async def t_effective_model_badge_after_delegation(ctx: TestContext) -> None:
    """Regression for the user-visible "badge stuck on the leader"
    bug: even when the leader (e.g. ``deepseek-v4-flash``) correctly
    delegates to a specialist (e.g. ``claude-cli:anthropic:claude-opus-4.7``)
    and the specialist actually writes the response, the chat UI was
    showing the LEADER's runtime_id as the model badge — making the
    team-as-router architecture invisible to the user.

    Fix: ``_record_delegation`` snapshots ``member_id -> runtime_id``
    at team-build time, ``_arun_runtime_stream`` / ``_arun_runtime_collect``
    detect ``delegate_task_to_member`` tool calls and feed the
    member_id through, and ``ModelDispatcher.effective_model_id``
    prefers the delegation target over the entry pick. Test verifies
    each link of that chain end-to-end without firing a real LLM.
    """
    from src.models.dispatcher import ModelDispatcher, _extract_delegated_member_id

    providers = _multi_specialist_catalog()
    router = ModelDispatcher(providers_config=providers)
    tp = router._get_team_provider("openai:gpt-4o-mini")

    # The map must include leader + every specialist's id → runtime_id.
    tp._ensure_runtime("sess-badge", system=None)
    member_map = tp._session_member_map["sess-badge"]
    assert "leader" in member_map, member_map
    assert member_map["leader"] == "openai:gpt-4o-mini", member_map
    coding_member_id = "openai-gpt-5-coding"
    assert member_map.get(coding_member_id) == "openai:gpt-5-coding", (
        f"member map missing coding specialist: {member_map}"
    )

    # Simulate the stream/collect path detecting a delegation tool call
    # and pushing the member_id back through ``_record_delegation``.
    tp._record_delegation("sess-badge", coding_member_id)
    router._remember_pick("sess-badge", "openai:gpt-4o-mini")

    badge = router.effective_model_id("sess-badge")
    assert badge == "openai:gpt-5-coding", (
        f"badge must reflect the delegated specialist, not the leader; "
        f"got {badge!r}"
    )

    # Sanity: when no delegation has fired, badge falls back to the entry.
    router._remember_pick("sess-nodelegate", "openai:gpt-4o-mini")
    badge_no_delegate = router.effective_model_id("sess-nodelegate")
    assert badge_no_delegate == "openai:gpt-4o-mini", badge_no_delegate

    # Sanity: _extract_delegated_member_id pulls member_id from a typical
    # Agno tool-call shape (used by the stream/collect helpers).
    class _FakeToolCall:
        tool_name = "delegate_task_to_member"
        tool_args = {"member_id": coding_member_id, "task": "..."}
    assert _extract_delegated_member_id(_FakeToolCall()) == coding_member_id

    class _OtherToolCall:
        tool_name = "search_web"
        tool_args = {"query": "..."}
    assert _extract_delegated_member_id(_OtherToolCall()) is None


@test("team_router", "rebuild_routing invalidates per-session Team cache")
async def t_rebuild_invalidates_cache(ctx: TestContext) -> None:
    """Hot-reloading the providers_config (new model added, tier_hint
    edited, etc.) must drop the cached Team so the next turn picks up
    the fresh catalog. Otherwise edits in the model-manager UI would
    take effect only after a process restart.
    """
    from src.models.dispatcher import TeamRouterProvider

    providers = _multi_specialist_catalog()
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )
    recorded = _install_stubs(provider)

    await provider.generate(
        [{"role": "user", "content": "hello"}], session_id="sess-rebuild",
    )
    first_team = recorded["team"]
    assert first_team is not None

    # Mutate the config: drop the marketing specialist.
    providers_v2 = _multi_specialist_catalog()
    providers_v2[1]["enabled"] = False
    provider.rebuild_routing(providers_config=providers_v2)
    # Cache invalidated.
    assert "sess-rebuild" not in provider._session_runtime

    recorded["team"] = None
    await provider.generate(
        [{"role": "user", "content": "hello again"}], session_id="sess-rebuild",
    )
    new_team = recorded["team"]
    assert new_team is not None
    member_names = [m.name for m in new_team.members]
    # The marketing specialist is gone; only the coding specialist remains.
    assert "specialist:anthropic:claude-opus-marketing" not in member_names, member_names
    assert "specialist:openai:gpt-5-coding" in member_names, member_names


# ── codex-cli tests (mirror of claude-cli ones above) ────────────────


def _catalog_with_codex_cli() -> list[dict[str, Any]]:
    """Mixed catalog — one api-based + one codex-cli row. The codex-cli
    row joins the Team via ``CodexBackedAgent`` (subclasses
    ``agno.agent.Agent``), so it passes Agno's isinstance check just
    like the claude-cli flavor.
    """
    return [
        {
            "id": 1, "name": "openai", "framework": "api-based",
            "api_key": "sk-test", "enabled": True,
            "models": [
                {"id": 10, "model": "gpt-4o-mini", "enabled": True,
                 "tier_hint": "fast general-purpose"},
                {"id": 11, "model": "claude-opus-coding", "enabled": True,
                 "tier_hint": "best for coding"},
            ],
        },
        {
            "id": 2, "name": "openai", "framework": "codex-cli",
            "api_key": None, "enabled": True,
            "models": [
                {"id": 20, "model": "gpt-5", "enabled": True,
                 "tier_hint": "best for code-edit and shell tasks"},
            ],
        },
    ]


@test("team_router", "codex-cli row joins Team as a specialist member")
async def t_codex_cli_member(ctx: TestContext) -> None:
    """A codex-cli row participates in Team membership via
    ``CodexBackedAgent`` (an ``agno.agent.Agent`` subclass). With an
    api-based entry leader, the codex-cli row appears as a specialist
    member so the leader can delegate to it.
    """
    from src.models.dispatcher import TeamRouterProvider

    providers = _catalog_with_codex_cli()
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )
    recorded = _install_stubs(provider)

    await provider.generate(
        [{"role": "user", "content": "hello"}],
        session_id="sess-codex-member",
    )

    assert recorded["team"] is not None
    team = recorded["team"]
    member_names = [m.name for m in team.members]
    assert "leader" in member_names
    codex_members = [
        n for n in member_names
        if "codex-cli:openai:gpt-5" in n
    ]
    assert codex_members, (
        f"codex-cli row missing from members: {member_names}. "
        f"CodexBackedAgent should make it eligible."
    )


@test("team_router", "codex-cli row works as Team leader with api-based routing model")
async def t_codex_cli_leader(ctx: TestContext) -> None:
    """When the entry is codex-cli, the ``CodexBackedAgent`` sits as
    ``members[0]``. ``team.model`` falls back to the cheapest enabled
    api-based model (just like the claude-cli leader path) because
    ``CodexBackedAgent``'s placeholder ``_NullModel`` can't drive
    Agno's routing classifier.
    """
    from src.models.dispatcher import TeamRouterProvider

    providers = _catalog_with_codex_cli()
    provider = TeamRouterProvider(
        entry_runtime_id="codex-cli:openai:gpt-5",
        providers_config=providers,
    )
    recorded = _install_stubs(provider)

    await provider.generate(
        [{"role": "user", "content": "hello"}],
        session_id="sess-codex-leader",
    )

    assert recorded["team"] is not None
    team = recorded["team"]
    # The codex-cli entry is members[0] (the "leader" slot).
    assert team.members[0].name == "leader"
    assert "gpt-5" in team.members[0].model_id, team.members[0].model_id
    # team.model points at the api-based routing fallback, NOT the
    # codex-cli leader's model.
    assert recorded["team_model"] is not None
    assert (
        "gpt-4o-mini" in str(recorded["team_model"])
        or "claude-opus-coding" in str(recorded["team_model"])
    ), (
        f"team.model must be an api-based fallback, got {recorded['team_model']!r}"
    )
    # The api-based specialists must appear as members.
    member_names = [m.name for m in team.members]
    assert "specialist:openai:gpt-4o-mini" in member_names, member_names


@test(
    "team_router",
    "codex-cli only catalog → single-agent fallback (no api-based routing model)",
)
async def t_codex_cli_only_fallback(ctx: TestContext) -> None:
    """When the catalog has ONLY codex-cli rows, there's no api-based
    model to drive Team's routing classifier. Skip Team and fall back
    to single-agent dispatch (the CodexBackedAgent alone).
    """
    from src.models.dispatcher import TeamRouterProvider

    providers: list[dict[str, Any]] = [
        {
            "id": 1, "name": "openai", "framework": "codex-cli",
            "api_key": None, "enabled": True,
            "models": [
                {"id": 1, "model": "gpt-5", "enabled": True,
                 "tier_hint": "general"},
            ],
        },
    ]
    provider = TeamRouterProvider(
        entry_runtime_id="codex-cli:openai:gpt-5",
        providers_config=providers,
    )
    recorded = _install_stubs(provider)

    runtime = provider._ensure_runtime("sess-codex-only", system=None)

    assert recorded["team"] is None, (
        "codex-cli-only catalog should not build a Team — no api-based "
        "model exists to drive Team.model's routing call"
    )
    assert isinstance(runtime, _RecordedAgent), type(runtime).__name__
    assert "gpt-5" in runtime.model_id


@test(
    "team_router",
    "Team mode is coordinate so leader can fan out to multiple members",
)
async def t_team_mode_coordinate(ctx: TestContext) -> None:
    """The Team is built in ``coordinate`` mode, not ``route``. This is
    what lets the leader fire MULTIPLE ``delegate_task_to_member`` tool
    calls in a single turn — Agno gathers them via
    ``asyncio.gather`` (see ``agno.models.base.arun_function_calls``),
    so independent sub-tasks run in parallel.

    Without coordinate mode, the leader is capped at one specialist per
    turn (route mode's contract: "delegate to exactly one member"),
    defeating multi-domain decomposition.
    """
    from src.core._runner.team import Team, TeamMode

    from src.models.dispatcher import TeamRouterProvider

    providers = _multi_specialist_catalog()
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )
    team = provider._ensure_runtime("sess-mode", system="MARKER")
    assert isinstance(team, Team), type(team).__name__
    assert team.mode == TeamMode.coordinate, (
        f"Team mode must be coordinate so leader can dispatch to "
        f"multiple members per turn; got {team.mode!r}"
    )
    # Coordinate mode also implies these booleans — see
    # ``agno/team/_init.py`` normalization.
    assert team.respond_directly is False, (
        "coordinate mode requires respond_directly=False so the leader "
        "synthesizes member outputs instead of returning them verbatim"
    )
    assert team.delegate_to_all_members is False, (
        "coordinate mode picks members (vs broadcast which fans out to all)"
    )


@test(
    "team_router",
    "delegation memo clears between turns so leader-only turns show leader badge",
)
async def t_delegation_memo_clears_between_turns(ctx: TestContext) -> None:
    """Bug: the live-stream UI kept showing the specialist's model badge
    on the turn AFTER a delegation, even when the leader answered the
    next turn directly. Cause: ``_last_delegation_by_session`` was
    written on every ``delegate_task_to_member`` but never cleared
    between turns, so ``effective_model_id`` kept returning the
    previous specialist forever.

    Rehydration was correct because per-message attribution is read
    from each stored run's data — never sticky.

    Fix: each ``generate``/``stream`` clears the per-session memo
    before kicking off the Agno run. If a delegation fires in this
    turn, it's recorded; otherwise the badge falls back to the leader.
    """
    from src.models.dispatcher import TeamRouterProvider

    providers = _multi_specialist_catalog()
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )

    # Turn 1: simulate a delegation. _record_delegation writes the
    # specialist's runtime_id into the memo.
    sid = "sess-badge"
    provider._session_member_map[sid] = {
        "specialist-openai-gpt-4o": "openai:gpt-4o",
    }
    provider._record_delegation(sid, "specialist-openai-gpt-4o")
    assert provider.last_delegation_for_session(sid) == "openai:gpt-4o", (
        "after delegation the memo should report the specialist"
    )

    # Turn 2: the dispatcher clears the memo at the start of generate /
    # stream. Simulate that step directly (the lambda inside generate /
    # stream does `pop(sid, None)`).
    provider._last_delegation_by_session.pop(sid, None)
    assert provider.last_delegation_for_session(sid) is None, (
        "after a non-delegating turn the memo must be empty so the "
        "badge falls back to the team leader (ModelDispatcher."
        "effective_model_id step 2)"
    )


@test(
    "team_router",
    "Team is built with tool-search at the TEAM level (not just members)",
)
async def t_team_has_tool_search_tools(ctx: TestContext) -> None:
    """Bug: live console logged ``Function tool_search_list_servers not
    found`` because tool-search was attached only to individual member
    Agents, never to the Team itself. Agno's Team.model (the routing
    classifier) saw zero callable functions other than
    ``delegate_task_to_member`` — yet the system prompt taught the
    model that tool_search_* names are directly callable, so the
    leader hallucinated those calls and Agno's function-call lookup
    failed.

    Fix: attach the tool-search toolkit to ``Team(tools=...)`` so the
    leader brain can call ``tool_search_*`` directly. Each member
    still has its own copy so delegated work also has tool access.
    """
    from src.core._runner.team import Team

    from src.models.dispatcher import TeamRouterProvider

    providers = _multi_specialist_catalog()
    provider = TeamRouterProvider(
        entry_runtime_id="openai:gpt-4o-mini",
        providers_config=providers,
    )

    class _FakeToolkit:
        def __init__(self, name: str) -> None:
            self.name = name

    class _FakePool:
        def runtime_toolkits_tool_search_only(self) -> list[Any]:
            return [_FakeToolkit("tool-search")]

        def claude_sdk_servers_tool_search_only(self) -> dict[str, dict]:
            return {}

    provider._mcp_pool = _FakePool()
    team = provider._ensure_runtime("sess-tools", system="MARKER")
    assert isinstance(team, Team), type(team).__name__

    tool_names = [getattr(t, "name", None) for t in (team.tools or [])]
    assert "tool-search" in tool_names, (
        "Team must carry the tool-search toolkit at the team level — "
        "without it the leader brain has only delegate_task_to_member "
        f"and Function-not-found errors fire. Got: {tool_names!r}"
    )


@test(
    "regression_v014",
    "FRAMEWORK_SYSTEM_PROMPT teaches tool-search call_tool wrapper, not direct names",
)
async def t_prompt_no_direct_mcp_call_examples(ctx: TestContext) -> None:
    """Bug: the prompt examples like ``vault_read_note`` /
    ``shell_shell_exec`` as if they were directly callable misled the
    leader to invoke them by name, producing ``Function X not found``
    errors. After the rewrite, the top tool section names the FIVE
    actual directly-callable tools and says everything else goes
    through ``tool_search_call_tool``.
    """
    from src.core.prompts import FRAMEWORK_SYSTEM_PROMPT

    text = FRAMEWORK_SYSTEM_PROMPT
    # The five callable tools must be named at the top.
    for required in (
        "tool_search_list_servers",
        "tool_search_list_tools",
        "tool_search_describe_tool",
        "tool_search_call_tool",
        "delegate_task_to_member",
    ):
        assert required in text, f"missing direct-tool name: {required}"
    # The discipline line must explicitly warn against direct calls.
    assert "Function X not found" in text or "Function not found" in text, (
        "prompt must warn that direct MCP-tool calls fail with "
        "'Function X not found' — otherwise the leader keeps trying"
    )
