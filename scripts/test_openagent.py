#!/usr/bin/env python
"""End-to-end OpenAgent test driver.

Each test lives in its own ``scripts/tests/test_<category>.py`` module
and registers with ``@test(category, name)``. This file just:

  1. imports every module in the ``scripts/tests/`` package so the
     ``@test`` side-effect populates the global ``TESTS`` registry,
  2. builds a throwaway agent dir (``/tmp/openagent-test-<uuid>/``)
     with a minimal config that borrows the user's real API keys,
  3. runs the registered tests in order, printing per-category headers
     and a final summary,
  4. tears down anything tests started (pool / gateway / agent).

Run:  bash scripts/test_openagent.sh
      bash scripts/test_openagent.sh --include-claude
      bash scripts/test_openagent.sh --only files,rest
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import shutil
import sys
from pathlib import Path

# Silence noisy third-party loggers; test output is already explicit.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for noisy in ("agno", "agno.tools", "agno.tools.mcp", "openai", "httpx",
              "httpcore", "asyncio", "openagent.mcp.client", "openagent.mcp.pool"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import the test framework AFTER sys.path is set up.
from scripts.tests._framework import (  # noqa: E402
    ANSI_DIM, ANSI_GREEN, ANSI_RED, ANSI_YELLOW, TESTS, TestContext,
    TestResult, c, run_one,
)
from scripts.tests._setup import build_test_config, cleanup_extras  # noqa: E402


# Module load order is SIGNIFICANT — tests register in import order,
# and several of them rely on fixtures set up by earlier tests (pool →
# gateway → sessions/rest/files/...). Changing this list changes the
# execution order of the whole suite, so add new modules deliberately.
_TEST_MODULES: tuple[str, ...] = (
    # 1. Lightweight / pure-unit (no fixtures needed)
    "test_imports",
    "test_setup",
    "test_serve_singleton",
    "test_catalog",
    "test_channels",
    "test_formatting",
    "test_tts_chunker",
    "test_turn_runner",
    # Local Piper TTS fallback — pure-unit, no fixtures. Paired with
    # test_turn_runner.py which covers the runner-side wiring.
    "test_tts_local",
    # TTS text sanitizer — markdown / emoji / URL stripping shared by
    # both synth entry points + the WS-streaming drain. Pure-unit, no
    # fixtures.
    "test_tts_sanitize",
    # ElevenLabs WebSocket streaming TTS — token-in / audio-out path
    # used by TurnRunner when cfg.stream_input is True.
    # Spins up a real local websockets server on a free port to
    # exercise the full BOS / text-frame / EOS protocol.
    "test_tts_elevenlabs_streaming",
    # DELTA frame plumbing for the unified streaming path (web chat +
    # bridges). Pure-unit; relies on the BaseBridge dispatch logic.
    "test_streaming",
    # Agent.run_stream empty-stream safety net — pure-unit, no fixtures.
    # Guards the contract that voice mode (and the soon-to-be-streaming
    # web chat) always gets text even when the streaming provider yields
    # zero deltas (claude_cli tool-only turns, smart_router → claude_cli
    # with empty content, agno when no RunContentEvent fires).
    "test_agent_run_stream",
    # New DB-backed registry tests: pure CRUD against ctx.db_path, no pool.
    "test_db_mcps",
    "test_db_models",
    "test_db_providers",
    "test_db_session_bindings",
    "test_smart_router_hybrid",
    # SmartRouter.stream + ClaudeCLIRegistry.stream — token-streaming
    # dispatch for both frameworks. Pure-unit (stubs the registries) and
    # guards the bug where claude-cli replies came back as one giant
    # chunk through the router (TTFB-killing for voice mode).
    "test_smart_router_stream",
    "test_claude_cli_stream",
    "test_behavior_contract",
    "test_mcp_manager_guards",
    "test_provider_manager",
    # Dynamic provider catalog: bundled fallback only (no live HTTP).
    "test_models_discovery",
    # MCP marketplace — pure schema-mapping unit tests, plus one REST
    # shape check that skips when no gateway fixture is wired.
    "test_marketplace",
    # ClaudeCLI buffer fix — computer-control screenshot regression guard.
    "test_buffer_size",
    # 2. MCP pool — sets ctx.extras["pool"] for everything below
    "test_pool",
    # MCPPool.from_db + reload — runs right after test_pool so it inherits
    # the "pool machinery imports cleanly" guarantee but uses its own
    # throwaway DB to avoid touching the shared pool fixture.
    "test_pool_reload",
    # 3. Provider-level live tests (need pool)
    "test_agno",
    # AgnoProvider.forget_session must wipe stored history so the
    # scheduler's per-fire forget and the gateway's /clear actually
    # reach Agno's SqliteDb-backed session store. Runs here (not in
    # provider live tests) because it uses a synthetic DB and doesn't
    # need the pool fixture.
    "test_agno_forget_clears_history",
    "test_router",
    "test_mcp",
    "test_budget",
    # 4. Gateway — sets ctx.extras["gateway_port"]/gateway/agent
    "test_gateway",
    # 5. HTTP surface + WS + files/images (need gateway)
    "test_sessions",
    "test_upload",
    "test_usage",
    "test_models",
    "test_rest",
    # DB-backed REST endpoints (/api/mcps, /api/models/db) — needs gateway.
    "test_mcps_rest",
    "test_voice",
    "test_files",
    # 6. Misc standalone
    "test_cron",
    # Issue #5 regression — scheduler must start each firing in a fresh session.
    "test_scheduler_fresh_session",
    # Workflow ai-prompt must forget/release at the right moment (same
    # bug class as scheduler issue #5 but for workflows).
    "test_workflow_forgets_session",
    # mcp-tool dispatch + validator callability check — guards against
    # the ``TypeError: 'Function' object is not callable`` regression
    # that broke LLM-authored workflows touching subprocess MCPs.
    "test_workflow_mcp_dispatch",
    # Canonical workflow examples — every example must round-trip
    # through validate_graph so the "reference manual" we ship to the
    # LLM (via list_workflow_examples / get_workflow_example) stays
    # accurate as block schemas evolve.
    "test_workflow_examples",
    "test_dream",
    "test_updater",
    "test_bridges",
    "test_shell",
    # 7. Optional Claude CLI path (needs --include-claude)
    "test_claude_cli",
    # 8. Unit tests for claude_cli text-recovery regression
    "test_claude_cli_text_recovery",
    # Stale-resume self-heal — same monkey-patching pattern.
    "test_claude_cli_stale_resume",
    # forget_session must drain pending sdk_session writes so a
    # background persist can't resurrect the deleted resume id.
    "test_claude_cli_forget_drains_writes",
    # ClaudeCLIRegistry dispatch — runs right after text-recovery since it
    # shares the claude_cli module's monkey-patching patterns.
    "test_claude_cli_registry",
    # 9. Gateway /stop, /clear, /new command semantics
    "test_gateway_commands",
    # SessionManager must run sessions in parallel on one client
    # (each session has its own worker queue). Prior design funneled
    # every message from a client through one queue, serialising chat
    # tabs unnecessarily.
    "test_sessions_parallel_execution",
    # 10. MCPPool resilience — one bad MCP mustn't sink the whole pool
    "test_mcp_pool_resilience",
    # 11. /api/files endpoint — agent-side attachment delivery to remote clients
    "test_files_endpoint",
)


def _discover_test_modules() -> list[str]:
    """Import each registered ``test_*`` module so the ``@test`` side
    effect populates the global registry. Order matters — see
    ``_TEST_MODULES`` above.
    """
    for name in _TEST_MODULES:
        importlib.import_module(f"scripts.tests.{name}")
    return list(_TEST_MODULES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(Path.home() / "my-agent" / "openagent.yaml"),
        help="Path to the user's openagent.yaml (read-only, for API keys).",
    )
    parser.add_argument(
        "--include-claude", action="store_true",
        help="Include the live Claude CLI test (slow, requires claude binary).",
    )
    parser.add_argument(
        "--only", default="",
        help="Comma-separated category list (e.g. 'files,rest,channels').",
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="Keep the temp test agent dir for inspection after the run.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List the discovered tests (category/name) and exit.",
    )
    args = parser.parse_args()

    modules = _discover_test_modules()

    if args.list:
        print(c(f"Discovered {len(modules)} test modules, "
                f"{len(TESTS)} tests total:", ANSI_DIM))
        last_cat = None
        for cat, name, _ in TESTS:
            if cat != last_cat:
                print(f"\n[{cat}]")
                last_cat = cat
            print(f"  {name}")
        return 0

    user_cfg_path = Path(args.config)
    if not user_cfg_path.exists():
        print(c(f"WARNING: {user_cfg_path} not found — live tests will skip.",
                ANSI_YELLOW))

    cfg, cfg_path, db_path = build_test_config(user_cfg_path)
    print(c(f"Test agent dir: {cfg_path.parent}", ANSI_DIM))
    ctx = TestContext(
        test_dir=cfg_path.parent, config=cfg, config_path=cfg_path,
        db_path=db_path,
        extras={"include_claude": args.include_claude},
    )

    only_categories = {s.strip() for s in args.only.split(",") if s.strip()}
    selected = [(cat, name, fn) for (cat, name, fn) in TESTS
                if not only_categories or cat in only_categories]

    print(c(f"Running {len(selected)} tests across "
            f"{len({c for c, _, _ in selected})} categories "
            f"(discovered from {len(modules)} modules)\n", ANSI_DIM))

    results: list[TestResult] = []
    last_cat = None

    async def run() -> None:
        nonlocal last_cat
        for cat, name, fn in selected:
            if cat != last_cat:
                print(f"\n[{cat}]")
                last_cat = cat
            # Long-running categories get extra timeout headroom
            timeout = 180 if cat in (
                "agno", "router", "sessions", "files", "claude_cli"
            ) else 60
            res = await run_one(cat, name, fn, ctx, timeout=timeout)
            results.append(res)
            symbol = {
                "ok":   c("✓", ANSI_GREEN),
                "fail": c("✗", ANSI_RED),
                "skip": c("○", ANSI_YELLOW),
            }[res.status]
            time_str = c(f"({res.duration:.1f}s)", ANSI_DIM)
            print(f"  {symbol} {name} {time_str}")
            if res.message and res.status != "ok":
                for ln in res.message.split("\n"):
                    print(c(f"      {ln}", ANSI_DIM))
        await cleanup_extras(ctx)

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
