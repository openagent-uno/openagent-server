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
    "test_catalog",
    "test_channels",
    "test_formatting",
    # 2. MCP pool — sets ctx.extras["pool"] for everything below
    "test_pool",
    # 3. Provider-level live tests (need pool)
    "test_agno",
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
    "test_voice",
    "test_files",
    # 6. Misc standalone
    "test_cron",
    "test_dream",
    "test_updater",
    "test_bridges",
    # 7. Optional Claude CLI path (needs --include-claude)
    "test_claude_cli",
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
