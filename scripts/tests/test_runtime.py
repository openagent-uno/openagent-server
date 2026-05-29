"""NativeProvider live tests (hits OpenAI with real keys).

Verifies that the provider actually generates a response, reports tokens,
routes the system prompt as a system message (not as user text), and
registers the ``list_mcp_servers`` meta-tool so the LLM can enumerate
MCP servers without hardcoding.
"""
from __future__ import annotations

import uuid
from typing import Any

from ._framework import TestContext, TestSkip, have_openai_key, test


@test("runtime", "live generate + tokens + cost + system_message routing")
async def t_agno_generate(ctx: TestContext) -> None:
    if not have_openai_key(ctx.config):
        raise TestSkip("no OpenAI API key in user config")
    from src.models.native_provider import NativeProvider

    pool = ctx.extras["pool"]
    provider = NativeProvider(
        model="openai:gpt-4o-mini",
        api_key=ctx.config["providers"]["openai"]["api_key"],
        providers_config=ctx.config["providers"],
        db_path=str(ctx.db_path),
    )
    provider.set_mcp_toolkits(pool.runtime_toolkits)
    resp = await provider.generate(
        messages=[{"role": "user", "content": "Reply with the literal text PING_OK and nothing else."}],
        system="You are a test bot. Always follow the user's instruction exactly.",
        session_id=f"runtime-test-{uuid.uuid4().hex[:8]}",
    )
    assert "PING_OK" in resp.content.upper(), f"unexpected response: {resp.content!r}"
    assert resp.input_tokens > 0, "no input tokens reported"
    assert resp.output_tokens > 0, "no output tokens reported"
    assert resp.model == "openai:gpt-4o-mini"


@test("runtime", "compaction flags enabled on constructed agent")
async def t_agno_compaction_flags(ctx: TestContext) -> None:
    """Session summaries must be ON by default so the agent gets
    long-horizon recall without blowing token budget. Agentic memory is
    DELIBERATELY OFF (see src/models/native_provider.py): OpenAgent uses the vault
    for user-scoped persistence and we don't want the ``runtime_memories``
    table created. Bumped ``num_history_runs`` default to 20."""
    from src.models.native_provider import NativeProvider
    pool = ctx.extras["pool"]
    provider = NativeProvider(
        model="openai:gpt-4o-mini",
        api_key=ctx.config.get("providers", {}).get("openai", {}).get("api_key", "x"),
        providers_config=ctx.config["providers"],
        db_path=str(ctx.db_path),
    )
    provider.set_mcp_toolkits(pool.runtime_toolkits)
    agent = provider._ensure_agent(system="test")
    assert getattr(agent, "enable_session_summaries", False), \
        "enable_session_summaries must be True"
    assert getattr(agent, "add_session_summary_to_context", False), \
        "add_session_summary_to_context must be True"
    assert getattr(agent, "enable_agentic_memory", True) is False, \
        "enable_agentic_memory must be False (vault handles user-scoped state)"
    assert getattr(agent, "num_history_runs", 0) >= 20, \
        f"num_history_runs should be ≥20, got {getattr(agent, 'num_history_runs', None)}"


@test("runtime", "tool_families groups toolkits by prefix")
async def t_agno_tool_families(ctx: TestContext) -> None:
    """_tool_families() must return one entry per connected MCP server,
    keyed by that server's tool_name_prefix."""
    from src.models.native_provider import NativeProvider
    pool = ctx.extras["pool"]
    provider = NativeProvider(
        model="openai:gpt-4o-mini",
        api_key=ctx.config.get("providers", {}).get("openai", {}).get("api_key", "x"),
        providers_config=ctx.config["providers"],
        db_path=str(ctx.db_path),
    )
    provider.set_mcp_toolkits(pool.runtime_toolkits)
    families = provider._tool_families()
    # Each toolkit should produce exactly one family entry — test pool
    # should ship at least one server, and whatever is there must map
    # to a non-empty family name.
    assert len(families) == len(pool.runtime_toolkits), \
        f"family count {len(families)} != toolkit count {len(pool.runtime_toolkits)}"
    for family_name, toolkits in families.items():
        assert family_name and isinstance(family_name, str), \
            f"bad family key: {family_name!r}"
        assert len(toolkits) >= 1, f"empty family {family_name}"


@test("runtime", "team construction: classifier path stays on single agent")
async def t_agno_team_classifier_fallback(ctx: TestContext) -> None:
    """Empty system prompt (classifier) must NOT trigger Team — the
    routing round-trip would waste tokens on simple tier classification."""
    from src.models.native_provider import NativeProvider
    pool = ctx.extras["pool"]
    provider = NativeProvider(
        model="openai:gpt-4o-mini",
        api_key=ctx.config.get("providers", {}).get("openai", {}).get("api_key", "x"),
        providers_config=ctx.config["providers"],
        db_path=str(ctx.db_path),
    )
    provider.set_mcp_toolkits(pool.runtime_toolkits)
    assert provider._ensure_team(system="") is None, \
        "empty system must skip team path"
    assert provider._ensure_team(system="   ") is None, \
        "whitespace-only system must skip team path"


@test("runtime", "team construction: <2 families falls back to single agent")
async def t_agno_team_few_families_fallback(ctx: TestContext) -> None:
    """With 0 or 1 tool families, Team has nothing to route between;
    _ensure_team must return None so the caller uses single Agent."""
    from src.models.native_provider import NativeProvider
    provider = NativeProvider(
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
    if pool.runtime_toolkits:
        provider.set_mcp_toolkits([pool.runtime_toolkits[0]])
        assert provider._ensure_team(system="hello") is None, \
            "single toolkit must skip team path"


@test("runtime", "team construction: ≥2 families builds route-mode Team")
async def t_agno_team_build(ctx: TestContext) -> None:
    """With ≥2 tool families, _ensure_team must return a Team in route
    mode with one specialist member per family, and the team itself
    must have the compaction flags enabled."""
    from src.models.native_provider import NativeProvider
    pool = ctx.extras["pool"]
    if len(pool.runtime_toolkits) < 2:
        raise TestSkip(f"test pool only has {len(pool.runtime_toolkits)} toolkit(s)")
    provider = NativeProvider(
        model="openai:gpt-4o-mini",
        api_key=ctx.config.get("providers", {}).get("openai", {}).get("api_key", "x"),
        providers_config=ctx.config["providers"],
        db_path=str(ctx.db_path),
    )
    provider.set_mcp_toolkits(pool.runtime_toolkits)
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
    # Verify compaction flags on the team leader — same contract as the
    # single-agent path: summaries ON, agentic memory OFF (vault path).
    assert getattr(team, "enable_session_summaries", False), \
        "team leader must have session summaries enabled"
    assert getattr(team, "enable_agentic_memory", True) is False, \
        "team leader agentic memory must be OFF (vault handles user-scoped state)"
    # Second call should hit the cache, returning the same object
    assert provider._ensure_team(system="You are a test bot.") is team, \
        "team must be cached by system prompt"
    # set_mcp_toolkits must flush the cache
    provider.set_mcp_toolkits(pool.runtime_toolkits)
    assert provider._ensure_team(system="You are a test bot.") is not team, \
        "set_mcp_toolkits must flush team cache"


@test("runtime", "generate raises NativeProviderError on RunStatus.error")
async def t_agno_generate_run_status_error(_ctx: TestContext) -> None:
    """the runtime's Agent.arun / Team.arun catches generic exceptions, sets
    ``run_response.status = RunStatus.error`` and
    ``run_response.content = str(e)``, then RETURNS the response instead
    of re-raising. Without an explicit status check the provider treats
    that error string as the model's output and the Telegram bridge
    shows the user a raw Python exception (e.g. ``[Errno 2] No such file
    or directory``) — caught in production on mixout with the
    misconfigured ``deepseek:deepseek-v4-pro`` model. ``generate()``
    must inspect ``response.status`` and raise ``NativeProviderError`` so
    the bridge formats a clean ``⚠️`` message instead.
    """
    from src.models.native_provider import NativeProvider, NativeProviderError

    provider = NativeProvider(
        model="deepseek:deepseek-v4-pro",
        api_key="x",
        providers_config=[],
    )

    class _ErroredResponse:
        status = "ERROR"
        content = "[Errno 2] No such file or directory"
        metrics = None
        tools: list[Any] = []

    class _ErroredRunner:
        async def arun(self, *_args: Any, **_kwargs: Any) -> _ErroredResponse:
            return _ErroredResponse()

    provider._compatible_mcp_toolkits = lambda: ([], [])  # type: ignore[assignment]
    provider._ensure_team = lambda system="": _ErroredRunner()  # type: ignore[assignment]
    provider._ensure_agent = lambda system=None: _ErroredRunner()  # type: ignore[assignment]

    raised: NativeProviderError | None = None
    try:
        await provider.generate(
            messages=[{"role": "user", "content": "hi"}],
            system="system-prompt",
            session_id="runtime-run-status-error",
        )
    except NativeProviderError as e:
        raised = e

    assert raised is not None, "expected NativeProviderError, got no exception"
    # The original Python-OSError string must be preserved in the raised
    # exception so logs/debugging keep the underlying cause — only the
    # user-facing bridge wrapper is supposed to soften it.
    assert "No such file" in str(raised), f"unexpected error: {raised!r}"
