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
    "test_cli_cleanup",
    # importlib.metadata fallback for frozen bundles — defense in depth
    # against the agno Team-run crash when pydantic dist-info goes missing
    # under sys._MEIPASS.
    "test_frozen_metadata_patch",
    "test_catalog",
    "test_channels",
    "test_formatting",
    "test_tts_chunker",
    # Local Piper TTS fallback — pure-unit, no fixtures. The legacy
    # ``TurnRunner`` it used to be paired with is gone (every text/voice
    # turn now flows through ``StreamSession`` / ``StreamTurnRunner``);
    # the runner-side wiring is exercised in test_stream.py.
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
    # Deepgram WebSocket streaming STT — audio-in / transcript-out
    # adapter that powers the universal app's voice tab via
    # StreamSession's STT pump. Local fake-WS server exercises the
    # full audio → CloseStream → final protocol.
    "test_stt_deepgram_streaming",
    # Barge-in commit: StreamSession._cancel_active_turn calls
    # Agent.commit_partial_assistant with the accumulated deltas BEFORE
    # cancelling the in-flight turn. Pure-unit; no provider live calls.
    "test_barge_in_commit",
    # ClaudeCLI.commit_partial_assistant routes to client.interrupt()
    # so the SDK's session log records what was emitted up to the
    # interrupt — keeps --resume coherent post-barge-in.
    "test_claude_cli_interrupt",
    # AgnoProvider.commit_partial_assistant injects a synthetic run
    # into the agno_sessions row so the next turn sees ``user →
    # assistant (interrupted) → user`` instead of two adjacent user
    # turns. Round-trips against a throwaway SqliteDb.
    "test_agno_partial_commit",
    # DELTA frame plumbing for the unified streaming path (web chat +
    # bridges). Pure-unit; relies on the BaseBridge dispatch logic.
    "test_streaming",
    # Unified streaming I/O protocol — typed events, wire codec,
    # StreamSession, channel profiles. Pure-unit; uses a fake agent
    # so no network or DB is required.
    "test_stream",
    # Agent.run_stream empty-stream safety net — pure-unit, no fixtures.
    # Guards the contract that voice mode (and the soon-to-be-streaming
    # web chat) always gets text even when the streaming provider yields
    # zero deltas (claude_cli tool-only turns, smart_router → claude_cli
    # with empty content, agno when no RunContentEvent fires).
    "test_agent_run_stream",
    # New DB-backed registry tests: pure CRUD against ctx.db_path, no pool.
    "test_db_mcps",
    # Bootstrap MCP-row seeding: regression for the missing-vault bug.
    "test_bootstrap",
    "test_db_models",
    # Regression: MemoryDB._parse_metadata must always return a dict (mixout
    # crash 2026-05-12, agno_sessions row with literal 'null' metadata).
    "test_db_metadata_parse",
    "test_db_providers",
    "test_db_session_bindings",
    # Concurrent ``add_session_run`` must not drop runs — per-session
    # asyncio.Lock guards the read-modify-write of agno_sessions.runs.
    "test_db_session_run_lock",
    # Cross-device chat visibility: ``upsert_session`` writes the user
    # handle as the row owner and ``list_all_sessions`` soft-falls back
    # to legacy device-pubkey rows via ``network_devices``.
    "test_sessions_cross_device",
    "test_db_workflow_claim",
    "test_smart_router_hybrid",
    "test_agno_tool_filter",
    # AgnoProvider.stream — hermetic zero-delta fallback coverage
    # (Friday's stuck Telegram turns; provider falls back to its own
    # generate() before control returns to Agent.run_stream).
    "test_agno_stream",
    # SmartRouter.stream + ClaudeCLIRegistry.stream — token-streaming
    # dispatch for both frameworks. Pure-unit (stubs the registries) and
    # guards the bug where claude-cli replies came back as one giant
    # chunk through the router (TTFB-killing for voice mode).
    "test_smart_router_stream",
    "test_claude_cli_stream",
    # Persistence-order regression: text and tool blocks must land in the
    # SDK stream order, not bunched user -> assistant_text -> tools.
    "test_claude_cli_ordering",
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
    # Workflow templating filters — fromjson/from_json (load-bearing
    # for ai-prompt → loop.items_expr chaining), regex/url filters,
    # and the safety net that quietly returns "" on resolve errors.
    "test_workflow_templating",
    # Scheduler must dispatch each due workflow as its own asyncio.Task
    # so different workflows on the same tick run concurrently. Companion
    # asserts the per-workflow lock keeps SAME-workflow runs ordered.
    "test_workflow_parallel_execution",
    "test_dream",
    "test_updater",
    "test_bridges",
    "test_bridge_session",
    # Coordinator login_finish must NOT die on SQLite locks for the
    # non-critical touch_device write (lyra-agent outage 2026-05-18 —
    # heavy Agno session writes held the writer lock and broke every
    # returning device's login).
    "test_coordinator_login_resilience",
    # End-to-end multi-user: spin up an in-process coordinator, mint
    # one invite per user, drive the full SRP-6a register+login wire
    # flow for N distinct (handle, device) pairs. Confirms invitations
    # are honoured, certs verify under the coordinator pubkey, and
    # returning-device logins (the touch_device path) survive both
    # clean and DB-lock-rigged runs.
    "test_coordinator_e2e_multi_user",
    # User store keyed by (name, handle): two handles can join one
    # network from a single machine, so a user invite stays redeemable
    # by anyone even when the network is already in the local store.
    "test_user_store",
    # Per-agent network naming: auto-bootstrap drops the legacy
    # ``-personal`` suffix; ``rename_network`` is cosmetic (preserves
    # network_id, coordinator identity, role) so existing pairings
    # survive a rename.
    "test_network_naming",
    # list_agents: coordinator's own agent comes FIRST so default-picking
    # clients (agents[0]) don't dial a foreign-network agent and
    # surface as WS code 1006 (lyra-agent regression 2026-05-19).
    "test_list_agents_ordering",
    # ``openagent invite [HANDLE]`` auto-picks user-role for new
    # handles + device-role bound for existing ones, dropping the
    # ``--role`` flag from the default CLI surface. Existing scripts
    # passing --role still work (advanced, hidden).
    "test_invite_smart",
    # /api/network/{users,agents,invitations} — gateway HTTP surface
    # that the desktop app + openagent-cli use for the members UI
    # and remote invite minting.
    "test_gateway_network_api",
    # workflow_runs left in ``running`` by a prior process must be
    # reaped on startup — without this, the lyra-music ``dev-coverage``
    # zombies pin the executor's per-workflow lock forever and the
    # next scheduled tick can't make progress.
    "test_workflow_orphan_reap",
    # _finalize_run survives transient SQLite locks (Agno's SqliteDb
    # writes racing OpenAgent's aiosqlite). Retries finalize on
    # OperationalError; cosmetic update_workflow lock is non-fatal;
    # retry is BOUNDED so we don't recreate the original loop.
    "test_workflow_finalize_resilience",
    # HTTP 402 / Insufficient Balance from any provider rewrites to a
    # billing-hint message naming the provider — surfaces the actual
    # config issue instead of looking like a transient provider blip.
    "test_provider_402_rewrite",
    # Discord on_message receive-path tests — pin the
    # allowed-users-bypass-mention gate behaviour and confirm silent
    # drops emit a diagnostic event.
    "test_discord_bridge_receive",
    "test_shell",
    # 7. Optional Claude CLI path (needs --include-claude)
    "test_claude_cli",
    # 8. Unit tests for claude_cli text-recovery regression
    "test_claude_cli_text_recovery",
    # Stale-resume self-heal — same monkey-patching pattern.
    "test_claude_cli_stale_resume",
    # Failed connect() must call disconnect() so the partially-spawned
    # claude subprocess doesn't leak (performa boss outage 2026-05-07).
    "test_claude_cli_connect_cleanup",
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
    # MCPPool spec.headers forwarding — regression guard for the
    # auth-gated remote MCP (mixout) 401 → ClosedResourceError chain.
    "test_mcp_pool_headers",
    # 11. /api/files endpoint — agent-side attachment delivery to remote clients
    "test_files_endpoint",
    # 12. Repo-hygiene structural guards — added 2026-05-26 after PR #1
    #     truncated src/workflow/executor.py from 990 to 60 lines via an
    #     LLM-generated "CONTENT OMITTED FOR BREVITY" placeholder that
    #     was committed verbatim. Three orthogonal guards: scan src/ for
    #     LLM placeholder strings, assert every test_*.py is in this
    #     list (so a new test file never silently goes unrun), and
    #     importlib-walk every src/ module so a truncated file is
    #     caught even if no behavioural test exercises its symbols.
    "test_repo_hygiene",
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
