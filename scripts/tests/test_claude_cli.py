"""ClaudeCLI (Claude Agent SDK) tests — optional.

The ``--include-claude`` flag gates the live ones because they spawn the
``claude`` binary (a few hundred megabytes of JS runtime per call).

The first test is a cheap presence check and runs unconditionally when
the binary is on PATH. The other two hit the SDK for real and burn
tokens, so they stay behind the flag.
"""
from __future__ import annotations

import asyncio
import shutil
import uuid

from ._framework import TestContext, TestSkip, test


def _find_claude_binary() -> str | None:
    return shutil.which("claude")


@test("claude_cli", "claude binary present + ClaudeCLI imports")
async def t_claude_present(ctx: TestContext) -> None:
    if not _find_claude_binary():
        raise TestSkip("claude binary not on PATH")
    from openagent.models.claude_cli import ClaudeCLI
    cli = ClaudeCLI(model=None, providers_config=ctx.config["providers"])
    assert cli._model_id_for_billing() == "claude-cli"


@test("claude_cli", "live one-shot via Claude SDK with one MCP")
async def t_claude_minimal(ctx: TestContext) -> None:
    if not _find_claude_binary():
        raise TestSkip("claude binary not on PATH")
    if not ctx.extras.get("include_claude"):
        raise TestSkip("claude tests require --include-claude")
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, ResultMessage
    from openagent.mcp.pool import MCPPool

    pool = ctx.extras.get("pool")
    own_pool = False
    if pool is None:
        pool = MCPPool.from_config(
            mcp_config=ctx.config.get("mcp"), include_defaults=True,
            disable=["chrome-devtools", "web-search", "computer-control", "mcp-manager", "model-manager"],
            db_path=str(ctx.db_path))
        await pool.connect_all()
        own_pool = True

    try:
        sdk_servers = pool.claude_sdk_servers()
        if "scheduler" not in sdk_servers:
            raise TestSkip("scheduler MCP not in pool")
        one = {"scheduler": sdk_servers["scheduler"]}
        opts = ClaudeAgentOptions(
            permission_mode="bypassPermissions",
            mcp_servers=one,
            extra_args={"strict-mcp-config": None},
        )
        client = ClaudeSDKClient(options=opts)
        await client.connect()
        try:
            await client.query("Reply with the literal text PING_CLAUDE.")
            async with asyncio.timeout(120):
                async for msg in client.receive_response():
                    if isinstance(msg, ResultMessage):
                        text = msg.result or ""
                        assert "PING_CLAUDE" in text.upper(), f"got: {text}"
                        break
        finally:
            await client.disconnect()
    finally:
        if own_pool:
            await pool.close_all()

@test("claude_cli", "live MCP tool invocation through ClaudeCLI provider")
async def t_claude_provider_mcp_call(ctx: TestContext) -> None:
    if not _find_claude_binary():
        raise TestSkip("claude binary not on PATH")
    if not ctx.extras.get("include_claude"):
        raise TestSkip("claude tests require --include-claude")
    from openagent.models.claude_cli import ClaudeCLI
    from openagent.mcp.pool import MCPPool

    pool = ctx.extras.get("pool")
    own_pool = False
    if pool is None:
        pool = MCPPool.from_config(
            mcp_config=ctx.config.get("mcp"), include_defaults=True,
            disable=["chrome-devtools", "web-search", "computer-control", "mcp-manager", "model-manager"],
            db_path=str(ctx.db_path))
        await pool.connect_all()
        own_pool = True

    try:
        # Use only one MCP because the claude binary is unreliable with many.
        cli = ClaudeCLI(model=None, providers_config=ctx.config.get("providers", {}))
        cli.set_mcp_servers({"scheduler": pool.claude_sdk_servers()["scheduler"]})
        sid = f"claude-mcp-{uuid.uuid4().hex[:8]}"
        resp = await cli.generate(
            messages=[{"role": "user",
                       "content": "Call mcp__scheduler__list_scheduled_tasks. "
                                  "Just report whether the call succeeded with the literal "
                                  "marker SCHEDULER_OK at the end."}],
            session_id=sid,
        )
        assert "SCHEDULER_OK" in resp.content.upper() or "[]" in resp.content, \
            f"unexpected claude response: {resp.content[:300]}"
    finally:
        try:
            await cli.shutdown()
        except Exception:
            pass
        if own_pool:
            await pool.close_all()


@test("claude-cli", "shell MCP round-trip: start bg shell, receive reminder, read output")
async def t_claude_shell_bg_roundtrip(ctx: TestContext) -> None:
    from openagent.core.agent import Agent
    from openagent.models.claude_cli import ClaudeCLI
    from openagent.mcp.servers.shell.handlers import get_hub

    pool = ctx.extras.get("pool")
    if pool is None:
        from scripts.tests._framework import TestSkip
        raise TestSkip("pool fixture not set up")

    model = ClaudeCLI()
    model.set_mcp_servers(pool.claude_sdk_servers())
    agent = Agent(name="shell-e2e", model=model)
    agent._initialized = True

    async def _noop_status(*_a, **_k): pass

    prompt = (
        "Use shell_exec with run_in_background=true to run "
        "'sleep 1 && echo hello-e2e'. Wait for the system reminder, "
        "call shell_output to read the result, and reply with "
        "exactly the text you read."
    )
    result = await agent._run_inner(
        message=prompt,
        attachments=None,
        _status=_noop_status,
        session_id="claude-shell-e2e",
    )
    assert "hello-e2e" in result, f"unexpected result: {result!r}"
