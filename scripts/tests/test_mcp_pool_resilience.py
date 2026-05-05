"""Regression tests for MCPPool one-bad-MCP-must-not-sink-the-boat.

Bug history
-----------

**2026-04-16**: mixout-agent came back from an OVH RAM upgrade with
telegram completely unresponsive. Root cause chain:

  - A Hostinger disk-resize on the VPS wiped ``~/.local/share/uv/`` (the
    uv tool bin dir), turning ``workspace-mcp`` into a dead symlink.
  - systemd tried to spawn openagent, openagent tried to connect every
    MCP in ``MCPPool.connect_all`` via ``AsyncExitStack.enter_async_context``.
  - The google-workspace entry hit a broken handshake. Agno's internal
    ``initialize()`` swallowed ``BaseException`` (CancelledError /
    BaseExceptionGroup), but the *shared* ``AsyncExitStack`` was now in
    a half-entered state across task boundaries.
  - The outer ``except BaseException: stack.aclose()`` then triggered
    anyio's "Attempted to exit a cancel scope that isn't the current
    task's current cancel scope" invariant violation.
  - The pool's lock was never released, the receive loop in ClaudeCLI
    hung in futex_wait, and systemd saw the process as "running".

Four things have to be true after the fix:

1. An ``Exception`` inside one MCP's ``__aenter__`` must be isolated —
   the rest of the MCPs still connect.
2. A ``BaseException`` (CancelledError, anyio cancel-scope violations)
   inside one MCP must also be isolated, not bubble out of ``connect_all``
   and leak partial state.
3. A hung handshake (``__aenter__`` that never returns) must be bounded
   by a timeout so the pool can move on.
4. ``close_all`` must still tear down every toolkit's ``__aexit__``
   (the path anyio expects) even if one raises. To keep this isolation,
   each toolkit lives in its *own* ``AsyncExitStack``, so a busted MCP's
   rollback never touches sibling MCPs' stacks.

These tests stub out ``MCPTools`` so they run in <1s without launching
real subprocesses.
"""
from __future__ import annotations

import asyncio
from typing import Any

from ._framework import TestContext, test


# ── Fake MCPTools ──────────────────────────────────────────────────────


class _FakeToolkit:
    """Stand-in for ``agno.tools.mcp.MCPTools``.

    Supports injection of:

      - ``enter_exc``: BaseException/Exception raised from ``__aenter__``.
      - ``enter_hang``: when True, ``__aenter__`` awaits forever (until
        cancelled) — used to verify the handshake timeout.
      - ``aexit_exc``: exception raised from ``__aexit__`` (which is what
        ``AsyncExitStack.aclose`` calls on shutdown — Agno + anyio use
        this path for proper cancel-scope teardown).
      - ``tool_count``: value used to populate ``functions`` after enter.

    Records ``entered`` / ``exited`` so tests can assert the pool fully
    entered and exited each toolkit.
    """

    def __init__(
        self,
        name: str = "fake",
        *,
        enter_exc: BaseException | None = None,
        enter_hang: bool = False,
        aexit_exc: BaseException | None = None,
        tool_count: int = 1,
    ) -> None:
        self.name = name
        self._enter_exc = enter_exc
        self._enter_hang = enter_hang
        self._aexit_exc = aexit_exc
        self._tool_count = tool_count
        self.tool_name_prefix = name
        self.functions: dict[str, Any] = {}
        self.entered = False
        self.exited = False

    async def __aenter__(self) -> "_FakeToolkit":
        if self._enter_hang:
            # Block forever until someone cancels us — pool's handshake
            # timeout must shield the caller from this.
            await asyncio.Event().wait()
        if self._enter_exc is not None:
            raise self._enter_exc
        self.entered = True
        self.functions = {f"{self.name}_tool_{i}": object() for i in range(self._tool_count)}
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.exited = True
        if self._aexit_exc is not None:
            raise self._aexit_exc


class _StealthFailToolkit(_FakeToolkit):
    """Models the Agno bug: ``__aenter__`` returns OK with an empty tools
    dict (because the wrapped ``initialize`` swallowed a BaseException),
    but a follow-up ``initialize()`` call succeeds.

    ``recover_after``: number of failed ``initialize()`` calls before
    recovery succeeds (0 = first retry wins, ``None`` = never recovers).
    """

    def __init__(
        self,
        name: str = "stealth",
        *,
        recover_after: int | None = 0,
        tool_count: int = 2,
    ) -> None:
        super().__init__(name=name, tool_count=tool_count)
        self._recover_after = recover_after
        self._initialized = False
        self.initialize_calls = 0

    async def __aenter__(self) -> "_StealthFailToolkit":
        # Stealth-fail: enter succeeds, but functions stays empty until
        # something forces another initialize() (just like real Agno).
        self.entered = True
        self.functions = {}
        return self

    async def initialize(self) -> None:
        self.initialize_calls += 1
        # Agno's idempotency guard: skip if already initialized.
        if self._initialized:
            return
        if (
            self._recover_after is not None
            and self.initialize_calls > self._recover_after
        ):
            self.functions = {
                f"{self.name}_tool_{i}": object() for i in range(self._tool_count)
            }
            self._initialized = True
            return
        # Mimic Agno: log the error and silently leave _initialized=False.
        # No exception propagates to the caller.


def _install_pool_fakes(monkey_specs: list[tuple[str, _FakeToolkit]]) -> Any:
    """Install a fake ``_build_and_enter_toolkit`` on ``MCPPool``.

    Returns the pool. Each spec name maps to a pre-built fake toolkit —
    the pool's build function looks up the toolkit by spec name instead
    of importing Agno. This keeps the test independent from Agno's
    current API and gives each test a deterministic per-spec toolkit.
    """
    from openagent.mcp.pool import MCPPool, _ServerSpec

    specs = [_ServerSpec(name=name, command=["/bin/true"]) for name, _ in monkey_specs]
    pool = MCPPool(specs)
    by_name = dict(monkey_specs)

    async def _fake_build(self: MCPPool, spec: _ServerSpec) -> Any:
        toolkit = by_name[spec.name]
        # Enter the toolkit directly; let the new pool do its own bounded
        # wait and BaseException isolation around this call.
        return await pool._safe_enter(toolkit, spec)  # type: ignore[attr-defined]

    # Test monkey-patches at the method level so the real ``_safe_enter``
    # implementation in pool.py is what actually runs (that's what we're
    # testing). If the pool has no ``_safe_enter`` yet, these tests will
    # fail with AttributeError on the current (v0.5.28) pool — by design,
    # they are the failing-first RED phase before the refactor.
    pool._build_and_enter_toolkit = _fake_build.__get__(pool, MCPPool)  # type: ignore
    return pool


# ── Tests ──────────────────────────────────────────────────────────────


@test("mcp_pool_resilience", "Exception in one MCP doesn't stop the others")
async def t_exception_isolated(ctx: TestContext) -> None:
    good_a = _FakeToolkit("good_a", tool_count=3)
    bad = _FakeToolkit("bad", enter_exc=RuntimeError("handshake failed"))
    good_b = _FakeToolkit("good_b", tool_count=2)

    pool = _install_pool_fakes([("good_a", good_a), ("bad", bad), ("good_b", good_b)])

    await pool.connect_all()

    summary = pool.server_summary()
    assert summary == {"good_a": 3, "bad": 0, "good_b": 2}, summary
    assert good_a.entered and good_b.entered
    assert not bad.entered
    assert "bad" in pool.dormant_servers()
    await pool.close_all()


@test(
    "mcp_pool_resilience",
    "BaseException (CancelledError) in one MCP doesn't stop the others",
)
async def t_baseexception_isolated(ctx: TestContext) -> None:
    """The mixout-post-upgrade regression.

    The dead ``workspace-mcp`` symlink caused an anyio cancel scope to
    raise ``BaseExceptionGroup`` out of MCPTools.__aenter__. A plain
    ``except Exception`` (the old behaviour before v0.5.29) would let
    this escape and leave the pool half-built.
    """
    good = _FakeToolkit("good", tool_count=4)
    bad = _FakeToolkit("bad", enter_exc=asyncio.CancelledError())
    pool = _install_pool_fakes([("good", good), ("bad", bad)])

    # Must not raise.
    await pool.connect_all()

    assert good.entered, "good MCP wasn't entered"
    assert not bad.entered
    assert pool.server_summary()["good"] == 4
    assert pool.server_summary()["bad"] == 0
    await pool.close_all()


@test(
    "mcp_pool_resilience",
    "close_all exits every toolkit even when one __aexit__ raises",
)
async def t_close_isolated(ctx: TestContext) -> None:
    """Per-toolkit stacks mean one busted teardown doesn't skip siblings."""
    a = _FakeToolkit("a")
    b = _FakeToolkit("b", aexit_exc=RuntimeError("aexit blew up"))
    c_ = _FakeToolkit("c")
    pool = _install_pool_fakes([("a", a), ("b", b), ("c", c_)])

    await pool.connect_all()
    await pool.close_all()

    # Each toolkit had its own AsyncExitStack, so a raising __aexit__ on
    # ``b`` doesn't prevent ``a`` and ``c`` from tearing down.
    assert a.exited and b.exited and c_.exited, (a.exited, b.exited, c_.exited)


@test(
    "mcp_pool_resilience",
    "Hung __aenter__ is bounded by the handshake timeout",
)
async def t_handshake_hang_times_out(ctx: TestContext) -> None:
    """A broken MCP whose handshake never completes must not pin the pool.

    The test overrides the timeout on the pool instance to 0.2s so we
    don't pay the production default in the suite.
    """
    from openagent.mcp import pool as pool_mod

    good = _FakeToolkit("good", tool_count=1)
    hung = _FakeToolkit("hung", enter_hang=True)
    pool = _install_pool_fakes([("good", good), ("hung", hung)])

    # Shrink the per-handshake timeout so the test is fast.
    saved = pool_mod._MCP_CONNECT_TIMEOUT
    pool_mod._MCP_CONNECT_TIMEOUT = 0.2
    try:
        t0 = asyncio.get_event_loop().time()
        await pool.connect_all()
        elapsed = asyncio.get_event_loop().time() - t0
    finally:
        pool_mod._MCP_CONNECT_TIMEOUT = saved

    # Must NOT have waited anywhere near production defaults (30s).
    assert elapsed < 5.0, f"connect_all took {elapsed:.1f}s — timeout not applied"
    assert good.entered
    assert pool.server_summary()["good"] == 1
    assert pool.server_summary()["hung"] == 0
    await pool.close_all()


@test(
    "mcp_pool_resilience",
    "Stealth-failed Agno init recovers via post-enter initialize() retry",
)
async def t_stealth_fail_recovers(ctx: TestContext) -> None:
    """The mixout-2026-05 regression. Agno's ``MCPTools.initialize()`` wraps
    its real init in ``except (RuntimeError, BaseException): log_error(...)``.
    When the wrapped init raises a ``BaseExceptionGroup`` from anyio's
    TaskGroup (typical under host load), Agno swallows it, leaves
    ``_initialized=False``, and ``__aenter__`` returns OK with zero tools.
    16/20 MCPs went silently dormant on a busy persona because of this.

    The pool must detect ``count == 0`` post-enter and re-run
    ``initialize()`` directly. The toolkit's idempotency-guard means we
    have to reset ``_initialized=False`` between attempts, which the fake
    here mirrors.
    """
    good = _FakeToolkit("good", tool_count=2)
    # Recovers on the FIRST retry (typical case).
    flaky = _StealthFailToolkit("flaky", recover_after=0, tool_count=4)
    pool = _install_pool_fakes([("good", good), ("flaky", flaky)])

    await pool.connect_all()

    summary = pool.server_summary()
    assert summary == {"good": 2, "flaky": 4}, summary
    assert flaky.initialize_calls >= 1, "recovery never re-ran initialize()"
    assert "flaky" not in pool.dormant_servers(), (
        f"flaky shouldn't be dormant after recovery; dormant={pool.dormant_servers()}"
    )
    await pool.close_all()


@test(
    "mcp_pool_resilience",
    "Permanently-broken stealth-fail eventually dormants out (bounded retries)",
)
async def t_stealth_fail_gives_up(ctx: TestContext) -> None:
    """Pool retries are bounded — a permanently-broken MCP must end up
    dormant within a couple seconds, not loop forever."""
    from openagent.mcp import pool as pool_mod

    good = _FakeToolkit("good", tool_count=1)
    # ``recover_after=None`` means initialize() never restores functions.
    dead = _StealthFailToolkit("dead", recover_after=None, tool_count=3)
    pool = _install_pool_fakes([("good", good), ("dead", dead)])

    saved_attempts = pool_mod._DORMANT_RECOVERY_ATTEMPTS
    saved_backoff = pool_mod._DORMANT_RECOVERY_BACKOFF
    pool_mod._DORMANT_RECOVERY_ATTEMPTS = 2
    pool_mod._DORMANT_RECOVERY_BACKOFF = 0.01
    try:
        t0 = asyncio.get_event_loop().time()
        await pool.connect_all()
        elapsed = asyncio.get_event_loop().time() - t0
    finally:
        pool_mod._DORMANT_RECOVERY_ATTEMPTS = saved_attempts
        pool_mod._DORMANT_RECOVERY_BACKOFF = saved_backoff

    assert elapsed < 3.0, f"recovery loop ran too long: {elapsed:.1f}s"
    summary = pool.server_summary()
    assert summary == {"good": 1, "dead": 0}, summary
    assert "dead" in pool.dormant_servers()
    # We attempted recovery the bounded number of times.
    assert dead.initialize_calls == 2, (
        f"expected 2 recovery attempts, got {dead.initialize_calls}"
    )
    await pool.close_all()


@test(
    "mcp_pool_resilience",
    "connect_all doesn't raise when every MCP fails (empty pool is valid)",
)
async def t_all_fail_no_raise(ctx: TestContext) -> None:
    """An agent with no working MCPs is degraded but must still boot —
    otherwise the whole process hangs on startup and systemd can't tell
    anything is wrong. This was the mixout failure mode."""
    a = _FakeToolkit("a", enter_exc=RuntimeError("nope"))
    b = _FakeToolkit("b", enter_exc=asyncio.CancelledError())

    pool = _install_pool_fakes([("a", a), ("b", b)])

    await pool.connect_all()  # must not raise

    assert pool.server_summary() == {"a": 0, "b": 0}
    assert sorted(pool.dormant_servers()) == ["a", "b"]
    await pool.close_all()
