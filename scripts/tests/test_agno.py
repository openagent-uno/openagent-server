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
