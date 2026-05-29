"""Test framework — decorator, context, runner, ANSI helpers.

Tiny on purpose: each test is an ``async def fn(ctx: TestContext) -> None``
that raises ``AssertionError`` (fail), ``TestSkip`` (skip), or returns
normally (pass). Categories group related tests and drive the per-category
timeout used by the driver.
"""
from __future__ import annotations

import asyncio
import socket
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

# ── ANSI helpers ──────────────────────────────────────────────────────

ANSI_GREEN = "\033[32m"
ANSI_RED = "\033[31m"
ANSI_YELLOW = "\033[33m"
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"
USE_COLOR = sys.stdout.isatty()


def c(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}" if USE_COLOR else text


# ── Port/path helpers ────────────────────────────────────────────────


def free_port() -> int:
    """Pick a free TCP port for the gateway test."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Test data structures ────────────────────────────────────────────


@dataclass
class TestResult:
    name: str
    category: str
    status: str  # "ok" | "fail" | "skip"
    message: str = ""
    duration: float = 0.0


@dataclass
class TestContext:
    """Shared state across tests — config, paths, and ``extras`` for
    fixtures (MCP pool, running gateway, etc.) that earlier tests set up
    and later tests consume. Tests should skip cleanly when a required
    extra is missing instead of crashing."""
    test_dir: Path
    config: dict
    config_path: Path
    db_path: Path
    extras: dict[str, Any] = field(default_factory=dict)


TestFn = Callable[[TestContext], Awaitable[None]]

# Global registry — every ``@test`` decorator appends here. Modules register
# at import time; the driver iterates ``TESTS`` in registration order.
TESTS: list[tuple[str, str, TestFn]] = []


def test(category: str, name: str):
    """Register an async test under ``category``/``name``."""
    def deco(fn: TestFn) -> TestFn:
        TESTS.append((category, name, fn))
        return fn
    return deco


class TestSkip(Exception):
    """Raise inside a test to mark it as skipped with a reason."""


async def run_one(category: str, name: str, fn: TestFn,
                  ctx: TestContext, timeout: float) -> TestResult:
    t0 = time.monotonic()
    try:
        await asyncio.wait_for(fn(ctx), timeout=timeout)
        return TestResult(name, category, "ok", duration=time.monotonic() - t0)
    except TestSkip as e:
        return TestResult(name, category, "skip", message=str(e),
                          duration=time.monotonic() - t0)
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
        last = tb.strip().split("\n")[-3:]
        return TestResult(
            name, category, "fail",
            message=f"{type(e).__name__}: {e}\n  ...{' '.join(last)[:300]}",
            duration=time.monotonic() - t0,
        )


# ── Config presence helpers (live tests skip when keys are missing) ──


def have_openai_key(config: dict) -> bool:
    key = (config.get("providers", {}).get("openai", {}) or {}).get("api_key", "")
    return bool(key) and not key.startswith("sk-test") and not key.startswith("${")


def have_any_live_llm_key(config: dict) -> bool:
    """True if at least one configured provider has a real API key.

    Used by tests that just need *any* working LLM (gateway lifecycle,
    REST endpoints, file uploads, vault round-trips, …). The older
    ``have_openai_key`` gate was overly tight — many of these tests
    don't actually use OpenAI; they were gated on it as a heuristic for
    "live keys are available". This helper widens that to any provider
    so a user configured with DeepSeek / Groq / Gemini still gets the
    gateway-dependent tests run instead of skipped.
    """
    providers = config.get("providers", {}) or {}
    for entry in providers.values():
        if not isinstance(entry, dict):
            continue
        key = entry.get("api_key", "") or ""
        if not key:
            continue
        if key.startswith("sk-test") or key.startswith("${"):
            continue
        return True
    return False
