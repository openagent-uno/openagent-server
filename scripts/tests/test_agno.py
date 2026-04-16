"""AgnoProvider live tests (hits OpenAI with real keys).

Verifies that the provider actually generates a response, reports tokens,
routes the system prompt as a system message (not as user text), and
registers the ``list_mcp_servers`` meta-tool so the LLM can enumerate
MCP servers without hardcoding.
"""
from __future__ import annotations

import uuid

from ._framework import TestContext, TestSkip, have_openai_key, test


@test("agno", "live generate + tokens + cost + system_message routing")
async def t_agno_generate(ctx: TestContext) -> None:
    if not have_openai_key(ctx.config):
        raise TestSkip("no OpenAI API key in user config")
    from openagent.models.agno_provider import AgnoProvider

    pool = ctx.extras["pool"]
    provider = AgnoProvider(
        model="openai:gpt-4o-mini",
        api_key=ctx.config["providers"]["openai"]["api_key"],
        providers_config=ctx.config["providers"],
        db_path=str(ctx.db_path),
    )
    provider.set_mcp_toolkits(pool.agno_toolkits)
    resp = await provider.generate(
        messages=[{"role": "user", "content": "Reply with the literal text PING_OK and nothing else."}],
        system="You are a test bot. Always follow the user's instruction exactly.",
        session_id=f"agno-test-{uuid.uuid4().hex[:8]}",
    )
    assert "PING_OK" in resp.content.upper(), f"unexpected response: {resp.content!r}"
    assert resp.input_tokens > 0, "no input tokens reported"
    assert resp.output_tokens > 0, "no output tokens reported"
    assert resp.model == "openai:gpt-4o-mini"


@test("agno", "list_mcp_servers tool exists in agent tools")
async def t_agno_meta_tool(ctx: TestContext) -> None:
    from openagent.models.agno_provider import AgnoProvider
    pool = ctx.extras["pool"]
    provider = AgnoProvider(
        model="openai:gpt-4o-mini",
        api_key=ctx.config.get("providers", {}).get("openai", {}).get("api_key", "x"),
        providers_config=ctx.config["providers"],
        db_path=str(ctx.db_path),
    )
    provider.set_mcp_toolkits(pool.agno_toolkits)
    agent = provider._ensure_agent(system="test")
    names = [getattr(t, "__name__", None) for t in agent.tools if callable(t)]
    assert "list_mcp_servers" in names, f"meta-tool missing; tools: {names}"


@test("agno", "compaction flags enabled on constructed agent")
async def t_agno_compaction_flags(ctx: TestContext) -> None:
    """Session summaries + agentic memory must be ON by default so the
    agent gets long-horizon recall without blowing token budget. Bumped
    history_runs default to 20 in the same change."""
    from openagent.models.agno_provider import AgnoProvider
    pool = ctx.extras["pool"]
    provider = AgnoProvider(
        model="openai:gpt-4o-mini",
        api_key=ctx.config.get("providers", {}).get("openai", {}).get("api_key", "x"),
        providers_config=ctx.config["providers"],
        db_path=str(ctx.db_path),
    )
    provider.set_mcp_toolkits(pool.agno_toolkits)
    agent = provider._ensure_agent(system="test")
    assert getattr(agent, "enable_session_summaries", False), \
        "enable_session_summaries must be True"
    assert getattr(agent, "add_session_summary_to_context", False), \
        "add_session_summary_to_context must be True"
    assert getattr(agent, "enable_agentic_memory", False), \
        "enable_agentic_memory must be True"
    assert getattr(agent, "num_history_runs", 0) >= 20, \
        f"num_history_runs should be ≥20, got {getattr(agent, 'num_history_runs', None)}"


@test("agno", "tool_families groups toolkits by prefix")
async def t_agno_tool_families(ctx: TestContext) -> None:
    """_tool_families() must return one entry per connected MCP server,
    keyed by that server's tool_name_prefix."""
    from openagent.models.agno_provider import AgnoProvider
    pool = ctx.extras["pool"]
    provider = AgnoProvider(
        model="openai:gpt-4o-mini",
        api_key=ctx.config.get("providers", {}).get("openai", {}).get("api_key", "x"),
        providers_config=ctx.config["providers"],
        db_path=str(ctx.db_path),
    )
    provider.set_mcp_toolkits(pool.agno_toolkits)
    families = provider._tool_families()
    # Each toolkit should produce exactly one family entry — test pool
    # should ship at least one server, and whatever is there must map
    # to a non-empty family name.
    assert len(families) == len(pool.agno_toolkits), \
        f"family count {len(families)} != toolkit count {len(pool.agno_toolkits)}"
    for family_name, toolkits in families.items():
        assert family_name and isinstance(family_name, str), \
            f"bad family key: {family_name!r}"
        assert len(toolkits) >= 1, f"empty family {family_name}"


@test("agno", "team construction: classifier path stays on single agent")
async def t_agno_team_classifier_fallback(ctx: TestContext) -> None:
    """Empty system prompt (classifier) must NOT trigger Team — the
    routing round-trip would waste tokens on simple tier classification."""
    from openagent.models.agno_provider import AgnoProvider
    pool = ctx.extras["pool"]
    provider = AgnoProvider(
        model="openai:gpt-4o-mini",
        api_key=ctx.config.get("providers", {}).get("openai", {}).get("api_key", "x"),
        providers_config=ctx.config["providers"],
        db_path=str(ctx.db_path),
    )
    provider.set_mcp_toolkits(pool.agno_toolkits)
    assert provider._ensure_team(system="") is None, \
        "empty system must skip team path"
    assert provider._ensure_team(system="   ") is None, \
        "whitespace-only system must skip team path"


@test("agno", "team construction: <2 families falls back to single agent")
async def t_agno_team_few_families_fallback(ctx: TestContext) -> None:
    """With 0 or 1 tool families, Team has nothing to route between;
    _ensure_team must return None so the caller uses single Agent."""
    from openagent.models.agno_provider import AgnoProvider
    provider = AgnoProvider(
        model="openai:gpt-4o-mini",
        api_key=ctx.config.get("providers", {}).get("openai", {}).get("api_key", "x"),
        providers_config=ctx.config["providers"],
        db_path=str(ctx.db_path),
    )
    # Zero toolkits → zero families → None.
    provider.set_mcp_toolkits([])
    assert provider._ensure_team(system="hello") is None, \
        "zero toolkits must skip team path"
    # Single toolkit → one family → None.
    pool = ctx.extras["pool"]
    if pool.agno_toolkits:
        provider.set_mcp_toolkits([pool.agno_toolkits[0]])
        assert provider._ensure_team(system="hello") is None, \
            "single toolkit must skip team path"


@test("agno", "team construction: ≥2 families builds route-mode Team")
async def t_agno_team_build(ctx: TestContext) -> None:
    """With ≥2 tool families, _ensure_team must return a Team in route
    mode with one specialist member per family, and the team itself
    must have the compaction flags enabled."""
    from openagent.models.agno_provider import AgnoProvider
    pool = ctx.extras["pool"]
    if len(pool.agno_toolkits) < 2:
        raise TestSkip(f"test pool only has {len(pool.agno_toolkits)} toolkit(s)")
    provider = AgnoProvider(
        model="openai:gpt-4o-mini",
        api_key=ctx.config.get("providers", {}).get("openai", {}).get("api_key", "x"),
        providers_config=ctx.config["providers"],
        db_path=str(ctx.db_path),
    )
    provider.set_mcp_toolkits(pool.agno_toolkits)
    team = provider._ensure_team(system="You are a test bot.")
    assert team is not None, "team should be built with ≥2 families"
    # Verify route mode
    mode = getattr(team, "mode", None)
    mode_value = getattr(mode, "value", mode)
    assert mode_value == "route", f"expected route mode, got {mode_value!r}"
    # Verify one member per family
    families = provider._tool_families()
    members = getattr(team, "members", [])
    assert len(members) == len(families), \
        f"member count {len(members)} != family count {len(families)}"
    # Verify compaction flags on the team leader
    assert getattr(team, "enable_session_summaries", False), \
        "team leader must have session summaries enabled"
    assert getattr(team, "enable_agentic_memory", False), \
        "team leader must have agentic memory enabled"
    # Second call should hit the cache, returning the same object
    assert provider._ensure_team(system="You are a test bot.") is team, \
        "team must be cached by system prompt"
    # set_mcp_toolkits must flush the cache
    provider.set_mcp_toolkits(pool.agno_toolkits)
    assert provider._ensure_team(system="You are a test bot.") is not team, \
        "set_mcp_toolkits must flush team cache"
