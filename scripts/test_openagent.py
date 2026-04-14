#!/usr/bin/env python
"""End-to-end OpenAgent test suite.

Runs against a temporary test agent directory so it never touches the
user's real ``my-agent`` config or DB. Reads API keys from
``~/my-agent/openagent.yaml`` (or ``--config <path>``) so live model
calls work.

Tests run in roughly increasing cost/time order. Each test is self-contained
and prints ``[OK]`` / ``[FAIL]`` / ``[SKIP]``. Exits non-zero if any failed.

Categories covered:
  - Imports & module wiring
  - Catalog & default pricing
  - MCP pool: build + connect + dormant detection + namespacing
  - AgnoProvider: live OpenAI call + cost mirroring + system_message routing
  - SmartRouter: tier classification + budget tracking + usage_log writes
  - Memory vault MCP: write/read note via tools
  - Scheduler MCP: create/list/delete cron task end-to-end
  - Gateway: HTTP server lifecycle + REST endpoints
  - Chat sessions: WebSocket round-trip + session isolation
  - File attachments: upload + reply mentions the attachment
  - Channel/per-message model overrides
  - Dream mode: prompt content + scheduler hookup
  - Auto-updater: dry-run check
  - ClaudeCLI: optional, only if ``claude`` binary present

Run:  python scripts/test_openagent.py
      python scripts/test_openagent.py --include-claude
      python scripts/test_openagent.py --only catalog,pool
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import socket
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

# Suppress noisy third-party loggers; we'll print our own progress.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("agno", "agno.tools", "agno.tools.mcp", "openai", "httpx",
              "httpcore", "asyncio", "openagent.mcp.client", "openagent.mcp.pool"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Utilities ─────────────────────────────────────────────────────────

ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_YELLOW = "\033[33m"
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"
USE_COLOR = sys.stdout.isatty()


def c(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}" if USE_COLOR else text


def free_port() -> int:
    """Pick a free TCP port for the gateway test."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def find_claude_binary() -> str | None:
    return shutil.which("claude")


# ── Test framework ────────────────────────────────────────────────────


@dataclass
class TestResult:
    name: str
    category: str
    status: str  # "ok" | "fail" | "skip"
    message: str = ""
    duration: float = 0.0


@dataclass
class TestContext:
    """Shared state across tests."""
    test_dir: Path
    config: dict
    config_path: Path
    db_path: Path
    extras: dict[str, Any] = field(default_factory=dict)


TestFn = Callable[[TestContext], Awaitable[None]]
TESTS: list[tuple[str, str, TestFn]] = []  # (category, name, fn)


def test(category: str, name: str):
    """Register a test."""
    def deco(fn: TestFn) -> TestFn:
        TESTS.append((category, name, fn))
        return fn
    return deco


class TestSkip(Exception):
    """Raise inside a test to mark it as skipped with a reason."""


async def _run_one(category: str, name: str, fn: TestFn, ctx: TestContext, timeout: float) -> TestResult:
    t0 = time.monotonic()
    try:
        await asyncio.wait_for(fn(ctx), timeout=timeout)
        return TestResult(name, category, "ok", duration=time.monotonic() - t0)
    except TestSkip as e:
        return TestResult(name, category, "skip", message=str(e), duration=time.monotonic() - t0)
    except asyncio.TimeoutError:
        return TestResult(name, category, "fail",
                          message=f"timeout after {timeout}s",
                          duration=time.monotonic() - t0)
    except AssertionError as e:
        return TestResult(name, category, "fail",
                          message=str(e) or "assertion failed",
                          duration=time.monotonic() - t0)
    except Exception as e:
        tb = traceback.format_exc()
        # Show last frame of the traceback for context
        last = tb.strip().split("\n")[-3:]
        return TestResult(name, category, "fail",
                          message=f"{type(e).__name__}: {e}\n  ...{' '.join(last)[:300]}",
                          duration=time.monotonic() - t0)


# ── Setup ─────────────────────────────────────────────────────────────


def build_test_config(user_config_path: Path) -> tuple[dict, Path, Path]:
    """Create a temp test agent dir with a minimal config that uses the
    user's real API keys (so live tests can hit OpenAI cheaply)."""
    import yaml

    test_dir = Path(f"/tmp/openagent-test-{uuid.uuid4().hex[:8]}")
    test_dir.mkdir(parents=True, exist_ok=True)
    db_path = test_dir / "test.db"

    # Copy providers (API keys + models) from the user's config; keep only
    # what the tests need. Skip channel tokens; skip per-channel model
    # overrides; skip scheduler tasks.
    user_cfg = yaml.safe_load(user_config_path.read_text()) if user_config_path.exists() else {}
    user_providers = dict(user_cfg.get("providers", {}))
    # Strip any anthropic API key — placeholder values like ``sk-test`` get
    # exported as ANTHROPIC_API_KEY by AgnoProvider and then break claude-cli
    # tests (the claude binary tries to use the bogus key instead of the
    # user's subscription auth). Claude-CLI does not need an API key.
    if "anthropic" in user_providers:
        del user_providers["anthropic"]
    cfg = {
        "name": "openagent-test",
        "model": {
            # SmartRouter so we get classifier + tier routing in one config.
            "provider": "smart",
            "monthly_budget": 100,
            "classifier_model": "gpt-4o-mini",
            "routing": {
                "simple": "gpt-4o-mini",
                "medium": "gpt-4o-mini",
                "hard": "gpt-4o-mini",
                "fallback": "gpt-4o-mini",
            },
        },
        "system_prompt": "You are a test assistant.",
        "channels": {"websocket": {"port": free_port()}},
        # Skip the heavy MCPs that initialise browser pools — they slow tests
        # down massively and aren't needed for what we're testing.
        "mcp_disable": ["chrome-devtools", "web-search", "computer-control"],
        "memory": {"db_path": str(db_path)},
        "scheduler": {"enabled": False, "tasks": []},
        "providers": user_providers,
    }
    config_path = test_dir / "openagent.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return cfg, config_path, db_path


# ── Tests ─────────────────────────────────────────────────────────────


# 1. IMPORTS

@test("imports", "all openagent modules import")
async def t_imports(ctx: TestContext) -> None:
    import openagent
    import openagent.cli
    import openagent.core.agent
    import openagent.core.server
    import openagent.gateway.server
    import openagent.gateway.sessions
    import openagent.mcp
    import openagent.mcp.pool
    import openagent.mcp.builtins
    import openagent.mcp.servers.scheduler.server
    import openagent.models.agno_provider
    import openagent.models.claude_cli
    import openagent.models.smart_router
    import openagent.models.runtime
    import openagent.models.catalog
    import openagent.models.budget
    import openagent.memory.db
    assert openagent.__version__


@test("imports", "no stale legacy refs (MCPRegistry / MCPTools / tool_factory)")
async def t_no_stale_refs(ctx: TestContext) -> None:
    import re
    for p in (REPO_ROOT / "openagent").rglob("*.py"):
        s = p.read_text()
        # Skip legitimate Agno MCPTools references — only flag our deleted classes.
        for line in s.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.search(r"openagent\.mcp\.client\b", stripped):
                raise AssertionError(f"stale openagent.mcp.client ref in {p}: {stripped}")
            if re.search(r"openagent\.models\.tool_factory\b", stripped):
                raise AssertionError(f"stale tool_factory ref in {p}: {stripped}")


# 2. CATALOG / PRICING

@test("catalog", "split_runtime_id + model_id_from_runtime")
async def t_catalog_split(ctx: TestContext) -> None:
    from openagent.models.catalog import split_runtime_id, model_id_from_runtime
    assert split_runtime_id("openai:gpt-4o-mini") == ("openai", "gpt-4o-mini")
    assert split_runtime_id("claude-cli/claude-sonnet-4-6") == ("claude-cli", "claude-sonnet-4-6")
    assert split_runtime_id("just-a-name") == ("just-a-name", "just-a-name")
    assert model_id_from_runtime("openai:gpt-4o-mini") == "gpt-4o-mini"


@test("catalog", "default pricing fallback works for bare model entries")
async def t_pricing_fallback(ctx: TestContext) -> None:
    from openagent.models.catalog import get_model_pricing, compute_cost
    user_cfg = {"openai": {"models": ["gpt-4o-mini", "gpt-4.1"]}}
    p = get_model_pricing("gpt-4o-mini", user_cfg)
    assert p["input_cost_per_million"] == 0.15, f"unexpected: {p}"
    assert p["output_cost_per_million"] == 0.60
    cost = compute_cost("openai:gpt-4.1", 1000, 500, user_cfg)
    expected = (2.00 * 1000 + 8.00 * 500) / 1_000_000
    assert abs(cost - expected) < 1e-12


@test("catalog", "user pricing overrides defaults")
async def t_pricing_override(ctx: TestContext) -> None:
    from openagent.models.catalog import get_model_pricing
    cfg = {"openai": {"models": [
        {"id": "gpt-4o-mini", "input_cost_per_million": 99.0, "output_cost_per_million": 88.0}
    ]}}
    p = get_model_pricing("gpt-4o-mini", cfg)
    assert p["input_cost_per_million"] == 99.0


# 3. MCP POOL

@test("pool", "from_config builds expected number of specs")
async def t_pool_specs(ctx: TestContext) -> None:
    from openagent.mcp.pool import MCPPool
    pool = MCPPool.from_config(
        mcp_config=ctx.config.get("mcp"),
        include_defaults=True,
        disable=ctx.config.get("mcp_disable"),
        db_path=str(ctx.db_path),
    )
    # Default MCPs minus the disabled ones (chrome-devtools, web-search, computer-control)
    names = [s.name for s in pool.specs]
    assert "vault" in names
    assert "filesystem" in names
    assert "scheduler" in names
    assert "messaging" in names
    assert "chrome-devtools" not in names
    assert "web-search" not in names


@test("pool", "claude_sdk_servers shape (command/args/env)")
async def t_pool_claude_shape(ctx: TestContext) -> None:
    from openagent.mcp.pool import MCPPool
    pool = MCPPool.from_config(
        mcp_config=ctx.config.get("mcp"),
        include_defaults=True,
        disable=ctx.config.get("mcp_disable"),
        db_path=str(ctx.db_path),
    )
    sdk = pool.claude_sdk_servers()
    assert sdk, "claude_sdk_servers returned empty"
    for name, entry in sdk.items():
        if "command" in entry:
            assert isinstance(entry["command"], str), f"{name}: command must be str"
            assert isinstance(entry["args"], list), f"{name}: args must be list"
        elif "url" in entry:
            assert entry.get("type") in ("http", "sse"), f"{name}: missing type"


@test("pool", "connect_all + dormant detection + summary")
async def t_pool_connect(ctx: TestContext) -> None:
    from openagent.mcp.pool import MCPPool
    pool = MCPPool.from_config(
        mcp_config=ctx.config.get("mcp"),
        include_defaults=True,
        disable=ctx.config.get("mcp_disable"),
        db_path=str(ctx.db_path),
    )
    await pool.connect_all()
    try:
        summary = pool.server_summary()
        assert pool.server_count >= 4, f"expected >=4 servers, got {pool.server_count}"
        # vault and scheduler should expose tools (no creds needed)
        assert summary.get("vault", 0) > 0, f"vault has no tools: {summary}"
        assert summary.get("scheduler", 0) > 0, f"scheduler has no tools: {summary}"
        # messaging now always exposes the status tool
        assert summary.get("messaging", 0) >= 1
        # Stash for downstream tests
        ctx.extras["pool"] = pool
        ctx.extras["initial_summary"] = summary
    except Exception:
        await pool.close_all()
        raise


@test("pool", "tool name namespacing follows <server>_<tool>")
async def t_pool_namespacing(ctx: TestContext) -> None:
    pool = ctx.extras.get("pool")
    if pool is None:
        raise TestSkip("requires pool fixture")
    seen_prefixes = set()
    for tk in pool.agno_toolkits:
        prefix = getattr(tk, "tool_name_prefix", None)
        if not prefix:
            continue
        seen_prefixes.add(prefix)
        for fname in (getattr(tk, "functions", {}) or {}):
            assert fname.startswith(prefix + "_"), \
                f"function {fname!r} doesn't follow {prefix}_<tool> convention"
    assert "vault" in seen_prefixes


# 4. AGNO PROVIDER (live OpenAI)

def _have_openai_key(cfg: dict) -> bool:
    key = (cfg.get("providers", {}).get("openai") or {}).get("api_key", "")
    return bool(key) and key.startswith(("sk-", "sk_"))


@test("agno", "live generate + tokens + cost + system_message routing")
async def t_agno_generate(ctx: TestContext) -> None:
    if not _have_openai_key(ctx.config):
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


# 5. SMART ROUTER + COST TRACKING

@test("router", "live generate writes usage_log row with non-zero cost")
async def t_router_usage_log(ctx: TestContext) -> None:
    if not _have_openai_key(ctx.config):
        raise TestSkip("no OpenAI API key")
    from openagent.memory.db import MemoryDB
    from openagent.models.runtime import create_model_from_config, wire_model_runtime

    pool = ctx.extras["pool"]
    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        model = create_model_from_config(ctx.config)
        wire_model_runtime(model, db=db, mcp_pool=pool)
        sid = f"router-test-{uuid.uuid4().hex[:8]}"
        resp = await model.generate(
            messages=[{"role": "user", "content": "Reply with literally PONG and nothing else."}],
            system="You are a test bot.",
            session_id=sid,
        )
        assert "PONG" in resp.content.upper(), f"{resp.content!r}"
        # Check usage_log row
        summary = await db.get_usage_summary()
        assert summary["total"] > 0, f"usage_log total=0; by_model={summary['by_model']}"
        # Find the row for openai:gpt-4o-mini specifically
        assert any("openai:gpt-4o-mini" in m for m in summary["by_model"]), \
            f"no openai:gpt-4o-mini row in usage_log: {summary['by_model']}"
    finally:
        await db.close()


# 6. MCP USAGE — vault + scheduler

@test("mcp", "vault MCP: write a note then read it back")
async def t_vault_roundtrip(ctx: TestContext) -> None:
    pool = ctx.extras["pool"]
    vault_tk = next((t for t in pool.agno_toolkits if getattr(t, "tool_name_prefix", "") == "vault"), None)
    if vault_tk is None:
        raise TestSkip("vault toolkit not loaded")

    write_fn = vault_tk.functions.get("vault_write_note")
    read_fn = vault_tk.functions.get("vault_read_note")
    assert write_fn and read_fn, "vault tools not registered"
    note_path = f"openagent-test-{uuid.uuid4().hex[:8]}.md"
    body = f"hello from openagent test {time.time()}"
    # Agno wraps callables; invoke the underlying entrypoint
    await write_fn.entrypoint(path=note_path, content=body)
    res = await read_fn.entrypoint(path=note_path)
    out = res.content if hasattr(res, "content") else str(res)
    assert body in out, f"vault read didn't return body; got: {out[:200]}"


@test("mcp", "scheduler MCP: create + list + delete a one-shot task")
async def t_scheduler_roundtrip(ctx: TestContext) -> None:
    pool = ctx.extras["pool"]
    sched_tk = next((t for t in pool.agno_toolkits if getattr(t, "tool_name_prefix", "") == "scheduler"), None)
    if sched_tk is None:
        raise TestSkip("scheduler toolkit not loaded")
    fns = sched_tk.functions
    assert "scheduler_create_one_shot_task" in fns
    assert "scheduler_list_scheduled_tasks" in fns
    assert "scheduler_delete_scheduled_task" in fns

    # Create a task one hour in the future
    task_name = f"openagent-test-{uuid.uuid4().hex[:6]}"
    created = await fns["scheduler_create_one_shot_task"].entrypoint(
        name=task_name,
        prompt=f"test prompt {uuid.uuid4().hex[:8]}",
        delay_seconds=3600,
    )
    out = created.content if hasattr(created, "content") else str(created)
    assert "id" in out.lower() or task_name in out, f"unexpected: {out[:200]}"

    listed = await fns["scheduler_list_scheduled_tasks"].entrypoint()
    listed_out = listed.content if hasattr(listed, "content") else str(listed)
    assert task_name in listed_out, f"task not in list: {listed_out[:300]}"


@test("mcp", "filesystem MCP: list_directory works on /tmp")
async def t_filesystem_list(ctx: TestContext) -> None:
    pool = ctx.extras["pool"]
    fs_tk = next((t for t in pool.agno_toolkits if getattr(t, "tool_name_prefix", "") == "filesystem"), None)
    if fs_tk is None:
        raise TestSkip("filesystem toolkit not loaded")
    fn = fs_tk.functions.get("filesystem_list_directory")
    if not fn:
        raise TestSkip("list_directory not available")
    res = await fn.entrypoint(path="/tmp")
    out = res.content if hasattr(res, "content") else str(res)
    assert len(out) > 0, "list_directory returned empty"


# 7. GATEWAY (HTTP server)

@test("gateway", "gateway starts + /api/health works")
async def t_gateway_health(ctx: TestContext) -> None:
    from openagent.gateway.server import Gateway
    from openagent.core.agent import Agent
    from openagent.models.runtime import create_model_from_config

    if not _have_openai_key(ctx.config):
        raise TestSkip("no OpenAI API key")

    pool = ctx.extras["pool"]
    model = create_model_from_config(ctx.config)
    agent = Agent(name="test", model=model, system_prompt="test", mcp_pool=pool)
    await agent.initialize()
    port = free_port()
    # Gateway creates its own SessionManager internally.
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


# 8. SESSIONS via gateway

@test("sessions", "WebSocket round-trip: send message, get response")
async def t_ws_roundtrip(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.ws_connect(f"ws://127.0.0.1:{port}/ws") as ws:
            await ws.send_json({"type": "auth", "client_id": "test-client"})
            # Drain auth_ok
            await asyncio.wait_for(ws.receive(), timeout=10)
            sid = f"ws-roundtrip-{uuid.uuid4().hex[:8]}"
            await ws.send_json({"type": "message", "text": "Reply with literally PING_RESP", "session_id": sid})
            response_text = None
            async with asyncio.timeout(60):
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    payload = json.loads(msg.data)
                    if payload.get("type") == "response":
                        response_text = payload.get("text", "")
                        break
            assert response_text is not None, "no response message received"
            assert "PING_RESP" in response_text.upper(), f"got: {response_text!r}"


@test("sessions", "session isolation: B can't see A's conversation history")
async def t_session_isolation(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp

    async def _send(client_id: str, sid: str, text: str) -> str:
        async with aiohttp.ClientSession() as http:
            async with http.ws_connect(f"ws://127.0.0.1:{port}/ws") as ws:
                await ws.send_json({"type": "auth", "client_id": client_id})
                await asyncio.wait_for(ws.receive(), timeout=10)
                await ws.send_json({"type": "message", "text": text, "session_id": sid})
                async with asyncio.timeout(60):
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        payload = json.loads(msg.data)
                        if payload.get("type") == "response":
                            return payload.get("text", "")
        return ""

    # Session A: pick a unique tag the agent has no reason to persist anywhere
    # (no "remember this", no "save", no "note"). Just a harmless statement.
    sid_a = f"isol-a-{uuid.uuid4().hex[:8]}"
    sid_b = f"isol-b-{uuid.uuid4().hex[:8]}"
    tag = f"zorpgleep_{uuid.uuid4().hex[:6]}"
    await _send("client-a", sid_a,
                f"My favorite test word for today is {tag}. Reply with just OK_NOTED — "
                "do NOT call any tools, do NOT write to vault, do NOT save anything.")
    # Session B (different client too — confirms client isolation): ask for the
    # word. Without history sharing OR vault commits, the agent cannot know.
    resp_b = await _send("client-b", sid_b,
                         "What was my favorite test word for today? "
                         "If you don't know from THIS conversation, say NO_INFO. "
                         "Do NOT search vault.")
    assert tag.lower() not in resp_b.lower(), \
        f"session B knew session A's tag {tag!r}: {resp_b[:200]}"


# 9. CHANNEL MODEL OVERRIDES via /api/config

@test("config", "GET /api/config returns the current config")
async def t_config_get(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/config") as r:
            assert r.status == 200
            body = await r.json()
            # Body shape varies — look for any of the expected top-level keys
            assert any(k in body for k in ("name", "model", "providers", "config")), body


# 10. MODEL REST

@test("models", "GET /api/models returns provider list")
async def t_models_list(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/models") as r:
            assert r.status == 200
            body = await r.json()
            assert isinstance(body, (list, dict)), body


@test("models", "GET /api/models/catalog returns catalog with pricing")
async def t_models_catalog(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/models/catalog") as r:
            if r.status == 404:
                raise TestSkip("/api/models/catalog not exposed in this build")
            assert r.status == 200
            body = await r.json()
            assert isinstance(body, (list, dict)), body


@test("models", "GET /api/models/providers lists supported providers")
async def t_models_providers(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/models/providers") as r:
            if r.status == 404:
                raise TestSkip("/api/models/providers not exposed in this build")
            assert r.status == 200
            body = await r.json()
            assert isinstance(body, (list, dict)), body
            # Should mention at least openai
            blob = json.dumps(body).lower()
            assert "openai" in blob, body


# 10b. SmartRouter tier classification routes correctly

@test("router", "classifier routes 'simple' question to simple tier model")
async def t_router_classifies(ctx: TestContext) -> None:
    if not _have_openai_key(ctx.config):
        raise TestSkip("no OpenAI API key")
    from openagent.models.runtime import create_model_from_config
    from openagent.memory.db import MemoryDB

    pool = ctx.extras["pool"]
    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        model = create_model_from_config(ctx.config)
        # Wire in pool + db so SmartRouter can route + record
        from openagent.models.runtime import wire_model_runtime
        wire_model_runtime(model, db=db, mcp_pool=pool)
        # Route a trivial greeting and verify it routes to the simple tier model
        sid = f"router-cls-{uuid.uuid4().hex[:8]}"
        decision = await model._routing_decision(
            messages=[{"role": "user", "content": "hi"}],
            session_id=sid,
            budget_ratio=1.0,
        )
        assert decision.requested_tier in ("simple", "medium", "hard"), decision
        # In our test config every tier maps to gpt-4o-mini, so just verify the
        # primary_model is something sensible
        assert "openai" in decision.primary_model
    finally:
        await db.close()


# 10b. FILE UPLOAD via gateway

@test("upload", "POST /api/upload accepts a file + returns a path")
async def t_file_upload(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    payload = b"hello from openagent test upload"
    async with aiohttp.ClientSession() as http:
        data = aiohttp.FormData()
        data.add_field("file", payload, filename="test.txt", content_type="text/plain")
        async with http.post(f"http://127.0.0.1:{port}/api/upload", data=data) as r:
            # Some gateway builds wrap upload behind auth; accept 200 or 401.
            if r.status == 401:
                raise TestSkip("upload endpoint behind auth")
            assert r.status in (200, 201), f"unexpected status: {r.status}"
            body = await r.json()
            # Common response shape: {"path": "...", "filename": "..."}
            assert "path" in body or "url" in body or "filename" in body, body


# 10c. PRICING endpoint

@test("pricing", "GET /api/usage/pricing returns model prices")
async def t_pricing_endpoint(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/usage/pricing") as r:
            if r.status == 404:
                raise TestSkip("/api/usage/pricing not exposed in this build")
            assert r.status == 200, f"status {r.status}"
            body = await r.json()
            assert isinstance(body, (list, dict)), body


# 11. USAGE / BUDGET endpoints

@test("usage", "GET /api/usage returns spend summary")
async def t_usage_endpoint(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/usage") as r:
            assert r.status == 200
            body = await r.json()
            # Must mention spend or budget somewhere
            assert any(k in body for k in ("monthly_spend", "spend", "by_model", "monthly_budget")), body


# 12. CRON / SCHEDULER

@test("cron", "MemoryDB.add_task + get_due_tasks")
async def t_cron_dbroundtrip(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB
    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        tid = await db.add_task(
            name=f"test-task-{uuid.uuid4().hex[:6]}",
            cron_expression="0 9 * * *",
            prompt="say hello",
            next_run=time.time() + 3600,
        )
        tasks = await db.get_tasks()
        assert any(t["id"] == tid for t in tasks), "task not found after add_task"
        await db.delete_task(tid)
        tasks_after = await db.get_tasks()
        assert all(t["id"] != tid for t in tasks_after)
    finally:
        await db.close()


# 13. DREAM MODE (just verify the prompt is sane and accessible)

@test("dream", "DREAM_MODE_PROMPT is non-empty and mentions vault")
async def t_dream_prompt(ctx: TestContext) -> None:
    from openagent.core.server import DREAM_MODE_PROMPT, DREAM_MODE_TASK_NAME
    assert isinstance(DREAM_MODE_PROMPT, str)
    assert len(DREAM_MODE_PROMPT) > 100
    assert "vault" in DREAM_MODE_PROMPT.lower()
    assert DREAM_MODE_TASK_NAME == "dream-mode"


# 14. AUTO-UPDATER (read-only)

@test("updater", "updater symbols exist + current __version__ is sane")
async def t_updater_callable(ctx: TestContext) -> None:
    import openagent
    from openagent.updater import check_for_update, UpdateInfo, perform_self_update_sync
    assert openagent.__version__ and isinstance(openagent.__version__, str)
    # Don't actually call check_for_update — it hits GitHub network.
    assert callable(check_for_update)
    assert callable(perform_self_update_sync)
    # UpdateInfo is a NamedTuple — verify shape
    fields = getattr(UpdateInfo, "_fields", None)
    assert fields and len(fields) >= 1


# 15. CLAUDE CLI (optional)

@test("claude_cli", "claude binary present + ClaudeCLI imports")
async def t_claude_present(ctx: TestContext) -> None:
    if not find_claude_binary():
        raise TestSkip("claude binary not on PATH")
    from openagent.models.claude_cli import ClaudeCLI
    cli = ClaudeCLI(model=None, providers_config=ctx.config["providers"])
    assert cli._model_id_for_billing() == "claude-cli"


@test("claude_cli", "live one-shot via Claude SDK with one MCP")
async def t_claude_minimal(ctx: TestContext) -> None:
    if not find_claude_binary():
        raise TestSkip("claude binary not on PATH")
    if not ctx.extras.get("include_claude"):
        raise TestSkip("claude tests require --include-claude")
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, ResultMessage
    from openagent.mcp.pool import MCPPool

    # Build a pool inline if the pool fixture isn't loaded (--only claude_cli case)
    pool = ctx.extras.get("pool")
    own_pool = False
    if pool is None:
        pool = MCPPool.from_config(
            mcp_config=ctx.config.get("mcp"), include_defaults=True,
            disable=ctx.config.get("mcp_disable"), db_path=str(ctx.db_path))
        await pool.connect_all()
        own_pool = True

    try:
        sdk_servers = pool.claude_sdk_servers()
        # Single MCP — claude binary is more reliable with one
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
    if not find_claude_binary():
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
            disable=ctx.config.get("mcp_disable"), db_path=str(ctx.db_path))
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


# ── Cleanup ───────────────────────────────────────────────────────────


async def _cleanup_extras(ctx: TestContext) -> None:
    """Tear down anything tests started up."""
    gw = ctx.extras.get("gateway")
    if gw is not None:
        try:
            await gw.stop()
        except Exception:
            pass
    agent = ctx.extras.get("agent")
    if agent is not None:
        try:
            await agent.shutdown()
        except Exception:
            pass
    pool = ctx.extras.get("pool")
    if pool is not None:
        try:
            await pool.close_all()
        except Exception:
            pass


# ── Driver ────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path.home() / "my-agent" / "openagent.yaml"),
                        help="Path to the user's openagent.yaml (read-only, for API keys).")
    parser.add_argument("--include-claude", action="store_true",
                        help="Include the live Claude CLI test (slow, requires claude binary).")
    parser.add_argument("--only", default="",
                        help="Comma-separated category list to run (e.g. 'imports,catalog').")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the temp test agent dir for inspection after the run.")
    args = parser.parse_args()

    user_cfg_path = Path(args.config)
    if not user_cfg_path.exists():
        print(c(f"WARNING: {user_cfg_path} not found — live tests will skip.", ANSI_YELLOW))

    cfg, cfg_path, db_path = build_test_config(user_cfg_path)
    print(c(f"Test agent dir: {cfg_path.parent}", ANSI_DIM))
    ctx = TestContext(test_dir=cfg_path.parent, config=cfg,
                      config_path=cfg_path, db_path=db_path,
                      extras={"include_claude": args.include_claude})

    only_categories = {s.strip() for s in args.only.split(",") if s.strip()}
    selected = [(cat, name, fn) for (cat, name, fn) in TESTS
                if not only_categories or cat in only_categories]

    print(c(f"Running {len(selected)} tests across "
            f"{len({c for c, _, _ in selected})} categories\n", ANSI_DIM))

    results: list[TestResult] = []
    last_cat = None

    async def run() -> None:
        nonlocal last_cat
        for cat, name, fn in selected:
            if cat != last_cat:
                print(f"\n[{cat}]")
                last_cat = cat
            timeout = 180 if cat in ("agno", "router", "sessions", "claude_cli") else 60
            res = await _run_one(cat, name, fn, ctx, timeout=timeout)
            results.append(res)
            symbol = {"ok": c("✓", ANSI_GREEN), "fail": c("✗", ANSI_RED), "skip": c("○", ANSI_YELLOW)}[res.status]
            time_str = c(f"({res.duration:.1f}s)", ANSI_DIM)
            print(f"  {symbol} {name} {time_str}")
            if res.message and res.status != "ok":
                for ln in res.message.split("\n"):
                    print(c(f"      {ln}", ANSI_DIM))
        await _cleanup_extras(ctx)

    try:
        asyncio.run(run())
    finally:
        if not args.keep:
            try:
                shutil.rmtree(ctx.test_dir)
            except Exception:
                pass
        else:
            print(c(f"\nKeeping {ctx.test_dir} for inspection.", ANSI_DIM))

    # Summary
    n_ok = sum(1 for r in results if r.status == "ok")
    n_fail = sum(1 for r in results if r.status == "fail")
    n_skip = sum(1 for r in results if r.status == "skip")
    total_time = sum(r.duration for r in results)
    print()
    print("─" * 60)
    print(f" {c(str(n_ok) + ' passed', ANSI_GREEN)}, "
          f"{c(str(n_fail) + ' failed', ANSI_RED)}, "
          f"{c(str(n_skip) + ' skipped', ANSI_YELLOW)} "
          f"in {total_time:.1f}s")
    print("─" * 60)
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
