"""Gateway lifecycle — boots the HTTP server with a real agent, then
hands the port/agent to downstream tests via ``ctx.extras``.

Runs EARLY in the gateway/sessions/config/models sequence so every
later test that needs the HTTP server (``ctx.extras["gateway_port"]``)
can assume it's up.
"""
from __future__ import annotations

from ._framework import TestContext, TestSkip, free_port, have_openai_key, test


@test("gateway", "gateway starts + /api/health works")
async def t_gateway_health(ctx: TestContext) -> None:
    from openagent.gateway.server import Gateway
    from openagent.core.agent import Agent
    from openagent.models.runtime import create_model_from_config

    if not have_openai_key(ctx.config):
        raise TestSkip("no OpenAI API key")

    pool = ctx.extras["pool"]
    model = create_model_from_config(ctx.config)
    agent = Agent(name="test", model=model, system_prompt="test", mcp_pool=pool)
    await agent.initialize()
    port = free_port()
    gw = Gateway(agent=agent, port=port, host="127.0.0.1",
                 config_path=str(ctx.config_path))
    await gw.start()
    try:
        import aiohttp
        async with aiohttp.ClientSession() as http:
            async with http.get(f"http://127.0.0.1:{port}/api/health") as r:
                assert r.status == 200, f"health returned {r.status}"
                body = await r.json()
                assert body.get("status") in ("ok", "ready", "healthy") or "agent" in body, \
                    f"unexpected health body: {body}"
            ctx.extras["gateway_port"] = port
            ctx.extras["gateway"] = gw
            ctx.extras["agent"] = agent
    except Exception:
        await gw.stop()
        await agent.shutdown()
        raise


@test("gateway", "/api/agent-info returns name + version")
async def t_gateway_agent_info(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/agent-info") as r:
            assert r.status == 200
            body = await r.json()
            assert "name" in body or "agent" in body or "version" in body, body
