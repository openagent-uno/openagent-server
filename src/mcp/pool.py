"""Process-level pool of MCP toolkits, owned by the Agent.

This replaces the previous in-house ``MCPRegistry`` + ``MCPTools`` pair. The
native runtime ships its own production-grade MCP integration, so OpenAgent owns
only the *product* layer:

  - which servers are configured (``DEFAULT_MCPS``, ``BUILTIN_MCP_SPECS``)
  - how to resolve a server name into a runnable command + env
  - per-server token wiring (messaging MCP credentials, scheduler DB path)

Concretely:

  - ``NativeProvider`` consumes ``pool.runtime_toolkits`` — a list of runtime
    ``MCPTools`` instances that the pool owns and connects once. Multiple
    ``NativeProvider`` tiers (under ``SmartRouter``) share the same toolkit
    list, so we don't spawn N copies of each MCP server process.

The pool also exposes ``server_summary()`` and ``dormant_servers()`` so the
Agent can inject runtime MCP state into the system prompt (e.g. "messaging
is configured but has no tools — set TELEGRAM_BOT_TOKEN to enable").
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import os
import shutil
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from src.core.logging import elog
from src.mcp.builtins import (
    BUILTIN_MCP_SPECS,
    DEFAULT_MCPS,
    resolve_builtin_entry,
    resolve_default_entry,
)

logger = logging.getLogger(__name__)


# Priority order used by ``*_under_budget`` to decide which subprocess MCPs
# appear in the upfront tool list when the total tool count exceeds the
# budget. MCPs listed here win their slot before alphabetical order kicks
# in; trimmed MCPs remain *connected* and reachable via ``tool-search`` —
# the model just has to discover them lazily. Override at runtime by
# setting ``OPENAGENT_MCP_PRIORITY`` to a comma-separated list of MCP
# names (e.g. ``vault,messaging,shell``); names not in the list fall back
# to alphabetical position.
_DEFAULT_MCP_PRIORITY: tuple[str, ...] = (
    "vault",        # long-term memory — keep hot, the user relies on it
    "messaging",    # platform-native Telegram primitives
    "shell",        # run commands / scripts
    "editor",       # file edits
    "filesystem",   # read/list paths
    "web-search",   # quick lookup
    "scheduler",    # cron / reminders
    "agent-bridge", # consult federated peer agents — keep discoverable
)


def _mcp_priority_order() -> tuple[str, ...]:
    raw = (os.environ.get("OPENAGENT_MCP_PRIORITY") or "").strip()
    if not raw:
        return _DEFAULT_MCP_PRIORITY
    parsed = tuple(seg.strip() for seg in raw.split(",") if seg.strip())
    return parsed or _DEFAULT_MCP_PRIORITY


def _mcp_priority_key(name: str) -> tuple[int, str]:
    """Sort key: (priority-rank, name). Lower rank wins. Unknown MCPs get
    a rank past every priority slot so the priority list always sorts
    first; ties (and unknowns) fall back to alphabetical name order.
    """
    order = _mcp_priority_order()
    try:
        rank = order.index(name)
    except ValueError:
        rank = len(order)
    return (rank, name)


# Tokens that must be forwarded from os.environ into the messaging MCP
# subprocess env. The runtime's MCPTools filters env to a safe subset by
# default, so we have to copy these explicitly.
_MESSAGING_TOKEN_ENV_VARS = (
    "TELEGRAM_BOT_TOKEN",
    "DISCORD_BOT_TOKEN",
    "GREEN_API_ID",
    "GREEN_API_TOKEN",
)

# Per-MCP-call timeout. The runtime's MCPTools defaults to 10s, which was too
# tight for cold-start tool calls against npx-launched servers; we bumped to
# 30s and then hit the opposite problem — 30s is *catastrophically* tight for
# tools that legitimately run for minutes, like ``shell_exec`` driving a
# macOS Electron build. A 30-min ceiling matches the shell MCP's own
# MAX_TIMEOUT so the runtime wrapper doesn't cut the call off before the tool
# itself would. Individual MCP tools still enforce their own shorter bounds
# (web-search: 6-10s per fetch, search-engine: 10s per query), so this cap
# only kicks in when a tool is genuinely stuck past its own limit.
_MCP_TIMEOUT_SECONDS = 1800

# Per-MCP *handshake* timeout, distinct from the per-call timeout above.
# This bounds how long a single MCP's ``__aenter__`` can block before the
# pool moves on. Without it, a dead stdio binary (e.g. a symlink in
# ``~/.local/share/uv/`` that no longer points anywhere after a disk
# resize) can pin the whole agent on startup: the subprocess spawn hangs
# inside anyio, the pool never completes, systemd can't tell anything is
# wrong, and telegram becomes unresponsive. 30s is generous for healthy
# handshakes (npx/uv cold-starts usually finish in <5s) and short enough
# that a single broken server is a recoverable blip, not a total outage.
_MCP_CONNECT_TIMEOUT = 30

# Per-MCP *shutdown* timeout — bounds how long we wait for one toolkit's
# supervisor task to exit after we signal it. Matches the handshake
# timeout philosophy: most stdio MCPs drain in milliseconds, but a stuck
# subprocess that ignores SIGTERM could otherwise pin the whole shutdown.
_MCP_CLOSE_TIMEOUT = 5

# Dormant-MCP recovery, for the "stealth-fail" handshake mode.
#
# The runtime's ``MCPTools.initialize()`` (≤ v2.6.4) wraps the actual init in
# ``except (RuntimeError, BaseException): log_error(...)``. When the wrapped
# ``session.initialize()`` / ``build_tools()`` calls raise — typically
# because anyio's TaskGroup surfaces a ``BaseExceptionGroup`` from a
# concurrently-cancelled subprocess scope on a busy host — the runtime
# swallows the error, leaves ``_initialized = False``, returns from
# ``__aenter__`` normally, and reports zero tools. The pool sees a "successful"
# enter with ``functions == {}`` and marks the MCP dormant for the rest of the
# session even though the subprocess is healthy and a second handshake
# would succeed.
#
# Field evidence (mixout-agent, 2026-05-05): on a freshly-restarted
# persona with 20 MCPs configured, 16 came up with 0 tools — every
# subprocess MCP except the four cheapest cold-starts. A manual restart
# of the affected agent recovered them in <2s each. Pattern is "TaskGroup
# under load", not "MCP fundamentally broken".
#
# Mitigation: when the count comes back zero on a successfully-entered
# toolkit, retry ``toolkit.initialize()`` directly a few times with light
# backoff before giving up. The retry is idempotent on the runtime side
# (``if self._initialized: return``), and bounded so a permanently-broken
# MCP still dormants out within a handful of seconds.
_DORMANT_RECOVERY_ATTEMPTS = 3
# Bumped from 15s → 30s after v0.12.51 production data showed the bypass
# bottoming out on TimeoutError for slow Python-builtin MCPs (workflow-
# manager, scheduler, mcp-manager, model-manager). These spawn the
# frozen openagent binary recursively, which extracts the ~220 MB
# PyInstaller archive to /tmp/_MEIxxxxx/ per process; under cold-start
# clustering the disk + import cost can push session.initialize past
# 15s. 30s leaves headroom for a worst-case host without making a
# permanently-broken MCP block startup unreasonably long (3 attempts
# × 30s = 90s upper bound, still well below systemd's 1-minute
# Restart= window).
_DORMANT_RECOVERY_TIMEOUT = 30
_DORMANT_RECOVERY_BACKOFF = 0.5

# Cold-start stagger between subprocess MCP spawns.
#
# Field evidence (friday-agent post-v0.12.51 deploy, 2026-05-05):
# 12/20 MCPs came up cleanly via the dormant-recovery loop, but 8 — all
# slow Python builtins or first-spawn npm packages — still hit
# ``TimeoutError`` in ``session.initialize`` with
# ``bypassed_agno_swallow=True``. Trace pattern:
#   anyio/streams/memory.py:receive_nowait → WouldBlock
#   → CancelledError on memory_stream.receive
#   → wrapped by asyncio.timeouts.__aexit__ in TimeoutError
# Translation: the subprocess is still bootstrapping (PyInstaller
# extract / npm cache miss / Python import graph) when we start
# polling for its initialize() response. The pipe is empty; we time
# out; the runtime swallows; pool flags dormant.
#
# Sleeping a fixed amount between subprocess spawns spreads the
# disk-I/O and CPU cost over wallclock instead of stacking it inside
# one cold-start window. The sequential ``for`` loop already serialises
# spawns at the runtime-pool level; the stagger adds kernel-level
# breathing room that lets the previous spawn at least finish its
# PyInstaller extract before the next one starts thrashing the same
# /tmp filesystem.
#
# 250 ms is the sweet spot from the production trace: long enough
# that workflow-manager finishes its archive extract before we move
# on (~150 ms observed), short enough that 20 MCPs add only ~5 s to
# total startup time — negligible against systemd's 30 s post-boot
# bridge polling delay.
_STARTUP_STAGGER_SECONDS = 0.25

# Error types that indicate the MCP session's underlying transport is
# closed and no amount of re-calling ``session.initialize()`` will
# recover it. When recovery exhausts its attempts AND every attempt
# raised one of these, the in-place retry loop is structurally pointless
# — we need a fresh subprocess + fresh stdio streams. ``connect_all``
# acts on this by re-running ``_build_and_enter_toolkit`` for the spec
# (see the re-spawn branch around line 685).
#
# Discovered 2026-05-23 on lyra-music/virgil v0.13.29: workflow-manager
# specifically (not its sibling Python MCPs) repeatedly came up
# dormant with the recovery loop emitting three identical
# ``ClosedResourceError`` failures from ``send_nowait`` on the
# stdio_client's write stream. The runtime's blanket-except inside
# ``MCPTools.initialize()`` swallows the original cause (a separate
# Instrument task to surface that), but the *symptom* — dead session
# inherited by recovery — is universal and worth handling without
# waiting on the upstream root cause.
_SESSION_DEAD_ERROR_TYPES = frozenset({
    "ClosedResourceError",
    "BrokenResourceError",
    "EndOfStream",
})


def _install_runtime_log_passthrough() -> None:
    """Re-emit the runtime's ``log_error()`` messages through ``elog``.

    The runtime's ``MCPTools.initialize()`` catches every exception and calls
    ``log_error("Failed to initialize MCP toolkit")`` — *without*
    ``exc_info``, so even at DEBUG you only see the string, not the
    underlying cause. The bypass path in ``_recover_dormant_toolkit``
    surfaces the real exception on retry; this handler closes the
    gap for the *first* attempt by routing the swallowed log line
    through our event stream, where ops can grep ``runtime.log_capture``
    to confirm the stealth-fail is happening (vs. some other zero-
    tools cause).

    Idempotent: only attaches one passthrough handler per process.
    """
    runtime_logger = logging.getLogger("openagent")

    class _RuntimePassthrough(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                msg = record.getMessage()
                # Filter to MCP-related ERROR/WARNING noise; we don't
                # want to mirror every runtime log line into events.jsonl.
                if record.levelno < logging.WARNING:
                    return
                if "mcp" not in msg.lower() and "MCP" not in record.name:
                    return
                elog(
                    "runtime.log_capture",
                    level="warning",
                    logger=record.name,
                    levelname=record.levelname,
                    message=msg[:500],
                )
            except Exception:  # noqa: BLE001 — handler must never raise
                pass

    if not any(isinstance(h, _RuntimePassthrough) for h in runtime_logger.handlers):
        runtime_logger.addHandler(_RuntimePassthrough())
        # Don't block propagation — just observe.
        runtime_logger.setLevel(min(runtime_logger.level or logging.WARNING, logging.WARNING))


# Install the passthrough at module import so it's active before any
# pool is built. Cheap (one handler registration), safe (idempotent),
# and the only way to see the runtime's first-attempt log line without
# modifying the runtime itself.
_install_runtime_log_passthrough()


def _safe_prefix(name: str) -> str:
    """Coerce a server name into a valid Python identifier prefix.

    The runtime emits tool names as ``<prefix>_<tool>``; ``computer-control`` →
    ``computer_control``. The hyphen would break OpenAI's function-name
    regex if not normalised.
    """
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


@dataclass
class _ServerSpec:
    """Resolved spec for a single MCP server.

    ``command`` is a fully resolved argv list; ``url`` is set for HTTP/SSE
    servers; ``env`` and ``cwd`` are forwarded to the subprocess.
    ``in_process`` is True for servers whose tools run in the same Python
    process (no subprocess) — the pool loads the adapter module and calls
    the named factory functions directly.
    """
    name: str
    command: list[str] | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None
    headers: dict[str, str] | None = None
    oauth: bool = False
    in_process: bool = False
    adapter_module: str | None = None
    runtime_toolkit_factory: str = "build_runtime_toolkit"

    @property
    def is_stdio(self) -> bool:
        return bool(self.command)


def _normalise_spec(spec: _ServerSpec) -> None:
    """Resolve relative commands to absolute paths and inject host secrets.

    Mutates ``spec`` in place. Two concerns rolled into one pass since both
    apply to the same spec list at the same point in the pipeline:

    - Absolute command path: a stdio MCP whose first argv arg can't be
      resolved on the subprocess ``$PATH`` is silently dropped (an issue
      under systemd). Resolving up-front avoids the footgun.

    - Messaging tokens: the messaging MCP gates each platform's tools on
      ``TELEGRAM_BOT_TOKEN`` etc. We copy those env vars from our own
      environment into the spec's ``env`` so the subprocess sees them.
    """
    if spec.is_stdio and spec.command:
        head = spec.command[0]
        if not os.path.isabs(head):
            resolved = shutil.which(head)
            if resolved:
                spec.command = [resolved] + spec.command[1:]
            else:
                logger.warning(
                    "MCP '%s': command %r not found on PATH — subprocess may fail to start",
                    spec.name, head,
                )

    if spec.name == "messaging" and spec.is_stdio:
        env = dict(spec.env) if spec.env else {}
        for var in _MESSAGING_TOKEN_ENV_VARS:
            val = os.environ.get(var)
            if val and var not in env:
                env[var] = val
        spec.env = env or None


def _resolve_specs(
    mcp_config: list[dict] | None,
    include_defaults: bool,
    disable: list[str] | None,
    db_path: str | None,
) -> list[_ServerSpec]:
    """Apply the same resolution rules as the old ``MCPRegistry.from_config``.

    Defaults are loaded first (skipping disabled and user-overridden ones),
    then user MCPs are merged on top. Cross-cutting host secrets (channel
    tokens for the messaging MCP) are injected here so providers see fully
    wired specs. Returns a flat list of ``_ServerSpec``.
    """
    specs: list[_ServerSpec] = []
    disabled = set(disable or [])
    user_names = {
        (entry.get("name") or entry.get("builtin", "")).strip()
        for entry in (mcp_config or [])
        if entry.get("name") or entry.get("builtin")
    }

    # 1. Defaults
    if include_defaults:
        for default_entry in DEFAULT_MCPS:
            name = default_entry.get("name") or default_entry.get("builtin", "")
            if name in disabled:
                logger.info("Default MCP '%s' disabled by config", name)
                continue
            if name in user_names:
                logger.info("Default MCP '%s' overridden by user config", name)
                continue
            kwargs = resolve_default_entry(default_entry, db_path=db_path)
            if kwargs:
                specs.append(_spec_from_kwargs(kwargs))

    # 2. User-configured
    for entry in (mcp_config or []):
        if "builtin" in entry:
            try:
                kwargs = resolve_builtin_entry(entry["builtin"], env=entry.get("env"))
                specs.append(_spec_from_kwargs(kwargs))
            except Exception as exc:
                logger.error("Failed to load built-in MCP '%s': %s", entry["builtin"], exc)
        else:
            specs.append(_ServerSpec(
                name=entry.get("name", ""),
                command=entry.get("command"),
                args=entry.get("args") or [],
                url=entry.get("url"),
                env=entry.get("env"),
                headers=entry.get("headers"),
                oauth=bool(entry.get("oauth")),
            ))

    # 3. Resolve relative commands and inject host-process secrets.
    for spec in specs:
        _normalise_spec(spec)

    return specs


async def _specs_from_db(db: Any, db_path: str | None) -> list[_ServerSpec]:
    """Translate ``mcps`` table rows into resolved specs.

    Each row keeps the same kwargs shape that ``DEFAULT_MCPS`` entries or
    raw user ``mcp:`` entries use; we then run it through
    ``resolve_default_entry`` / ``resolve_builtin_entry`` so absolute path
    resolution, Python-entrypoint substitution for frozen builds, and the
    scheduler-DB env injection all reuse the existing code path instead
    of being duplicated.
    """
    rows = await db.list_mcps(enabled_only=True)
    specs: list[_ServerSpec] = []
    for row in rows:
        kind = row.get("kind")
        name = row.get("name") or ""
        try:
            if kind == "default":
                # Reconstruct the minimal dict shape ``resolve_default_entry``
                # expects. ``source='yaml-default'`` rows use the builtin
                # indirection when ``builtin_name`` is set; others are raw
                # command/args (e.g. vault, filesystem).
                if row.get("builtin_name"):
                    entry = {"builtin": row["builtin_name"], "env": row.get("env") or None}
                else:
                    entry = {
                        "name": name,
                        "command": row.get("command"),
                        "args": row.get("args") or [],
                        "url": row.get("url"),
                        "env": row.get("env") or None,
                    }
                kwargs = resolve_default_entry(entry, db_path=db_path)
                if kwargs:
                    specs.append(_spec_from_kwargs(kwargs))
            elif kind == "builtin":
                extra_env = dict(row.get("env") or {})
                if row.get("builtin_name") in ("scheduler", "mcp-manager", "model-manager"):
                    # Mirror the db-path injection that resolve_default_entry
                    # does for the scheduler so runtime-created builtin rows
                    # still see the DB.
                    if db_path and "OPENAGENT_DB_PATH" not in extra_env:
                        extra_env["OPENAGENT_DB_PATH"] = os.path.abspath(db_path)
                kwargs = resolve_builtin_entry(
                    row["builtin_name"],
                    env=extra_env or None,
                )
                specs.append(_spec_from_kwargs(kwargs))
            else:  # custom
                specs.append(_ServerSpec(
                    name=name,
                    command=row.get("command"),
                    args=list(row.get("args") or []),
                    url=row.get("url"),
                    env=row.get("env") or None,
                    headers=row.get("headers") or None,
                    oauth=bool(row.get("oauth")),
                ))
        except Exception as exc:  # noqa: BLE001 — one bad row must not pin the pool
            elog("mcp.db_row_error", level="warning", name=name, kind=kind, error=str(exc))

    for spec in specs:
        _normalise_spec(spec)
    return specs


def _spec_from_kwargs(kwargs: dict[str, Any]) -> _ServerSpec:
    """Convert ``resolve_*_entry``'s loose dict into a typed ``_ServerSpec``."""
    return _ServerSpec(
        name=kwargs.get("name", ""),
        command=kwargs.get("command"),
        args=kwargs.get("args") or [],
        url=kwargs.get("url"),
        env=kwargs.get("env"),
        cwd=kwargs.get("_cwd"),
        headers=kwargs.get("headers"),
        oauth=bool(kwargs.get("oauth")),
        in_process=bool(kwargs.get("in_process", False)),
        adapter_module=kwargs.get("adapter_module"),
        runtime_toolkit_factory=kwargs.get("runtime_toolkit_factory", "build_runtime_toolkit"),
    )


@dataclass
class _ToolkitSupervisor:
    """Handle to the per-toolkit supervisor task.

    Each supervised task holds ``async with stack: enter_async_context(toolkit);
    await stop_event.wait()`` for the toolkit's entire lifetime. That means
    ``__aenter__`` AND ``__aexit__`` both run on ``task`` — never on a
    different task.

    Why this matters: the MCP stdio client (``mcp/client/stdio/__init__.py``)
    uses an anyio cancel scope. anyio enforces that a cancel scope opened in
    task A can only be exited by task A; otherwise it raises ``RuntimeError:
    Attempted to exit cancel scope in a different task than it was entered
    in``. Before this refactor, ``__aenter__`` ran on the ``connect_all``
    caller task but ``stack.aclose()`` in ``close_all`` ran on whatever task
    called stop — usually a different task — which tripped that invariant
    during every shutdown and also during reload.
    """

    name: str
    task: asyncio.Task
    stop_event: asyncio.Event


class MCPPool:
    """Owns the lifecycle of MCP toolkits for the current process.

    Connect once at agent startup; close at shutdown. Both providers read
    from the same pool so we don't spawn duplicate MCP server processes
    across SmartRouter tiers (the old ``MCPRegistry`` did the same; this
    keeps that property).
    """

    def __init__(self, specs: list[_ServerSpec]):
        self.specs: list[_ServerSpec] = specs
        # Lazily populated on connect_all. Each toolkit gets its *own*
        # supervisor task (parallel arrays, ``_runtime_toolkits[i]`` is owned
        # by ``_toolkit_supervisors[i]``). The supervisor holds the
        # toolkit's ``AsyncExitStack`` for the toolkit's entire lifetime,
        # which guarantees ``__aenter__`` and ``__aexit__`` run on the
        # SAME task — the only way to satisfy the anyio cancel-scope
        # invariant that the MCP stdio client relies on. The previous
        # per-toolkit ``AsyncExitStack`` design stored the stack itself
        # and called ``aclose()`` from ``close_all``'s caller task, which
        # was fine most of the time but blew up with "Attempted to exit
        # cancel scope in a different task than it was entered in" every
        # time shutdown happened on a different task than connect (every
        # /restart, essentially).
        self._runtime_toolkits: list[Any] = []
        self._toolkit_supervisors: list[_ToolkitSupervisor] = []
        # Name-indexed view of every connected toolkit (subprocess + in-process).
        # Lets callers resolve ``toolkit_by_name("shell")`` without walking the
        # parallel arrays — used by the workflow executor to dispatch
        # ``mcp-tool`` blocks to the right toolkit.
        self._toolkit_by_name: dict[str, Any] = {}
        self._tool_counts: dict[str, int] = {name: 0 for name in (s.name for s in specs)}
        self._last_recovery_error_type = {}
        # Per-MCP last connect/handshake error, keyed by spec name. Set when
        # an MCP ends up dormant (0 tools) so callers — the workflow
        # validator, the UI — can surface the real cause (e.g. a 401 the
        # runtime's blanket-except swallowed) instead of a bare empty tool
        # list. Cleared the moment the MCP connects with tools.
        self._last_connect_error: dict[str, str] = {}
        self._connected = False
        self._lock = asyncio.Lock()
        # In-process MCP state — populated by connect_all for specs with
        # in_process=True. These are kept separate from subprocess toolkits
        # so close_all can handle them independently (no AsyncExitStack needed).
        self._in_process_runtime_toolkits: list[Any] = []
        # DB reference for ``reload()``. Set by ``from_db``; ``None`` for
        # ``from_config`` callers (tests) — reload is a no-op in that mode.
        self._db: Any = None
        self._db_path: str | None = None
        # Cached rendered catalog summary (~16KB string injected into
        # every chat-turn's system prompt). Computed lazily on first use,
        # invalidated by ``reload()`` and ``connect_all`` because both
        # may change ``self._tool_counts``. The render is a 1-2ms walk
        # that adds up across every prompt construction.
        self._catalog_summary_cache: str | None = None

    @classmethod
    def from_config(
        cls,
        mcp_config: list[dict] | None = None,
        include_defaults: bool = True,
        disable: list[str] | None = None,
        db_path: str | None = None,
    ) -> "MCPPool":
        """Build a pool from the same shape ``MCPRegistry.from_config`` accepted."""
        specs = _resolve_specs(mcp_config, include_defaults, disable, db_path)
        return cls(specs)

    @classmethod
    async def from_db(
        cls,
        db: Any,
        *,
        db_path: str | None = None,
    ) -> "MCPPool":
        """Build a pool from the ``mcps`` table in the DB.

        Only ``enabled = 1`` rows are loaded — a disabled row stays in the
        table so the UI can re-enable it without losing its configuration.
        Each row is translated back into the loose-dict shape that the
        existing ``resolve_*_entry`` helpers emit, then flows through the
        same ``_resolve_specs``-style normalization that ``from_config``
        uses (absolute-path resolution, messaging-token injection).
        """
        specs = await _specs_from_db(db, db_path)
        pool = cls(specs)
        pool._db = db
        pool._db_path = db_path
        return pool

    async def rebuild_specs(self) -> list[_ServerSpec]:
        """Re-query the DB and return a fresh spec list.

        Called by ``reload()``. Returns the current ``specs`` unchanged
        when no DB was wired (``from_config`` path) so tests that manually
        construct a pool don't silently get an empty list on reload.
        """
        if self._db is None:
            return list(self.specs)
        return await _specs_from_db(self._db, self._db_path)

    async def reload(self) -> None:
        """Rebuild the pool in-place without a process restart.

        Order: build new specs → swap lists → connect new → close old.
        We swap the backing lists in place (``self._runtime_toolkits[:] = ...``)
        so providers that already captured ``pool.runtime_toolkits`` by
        reference see the new tools on their next turn. Old supervisors are
        torn down AFTER the new pool is live, so there is no window where
        a concurrent turn sees zero tools.

        No-op when the pool was built via ``from_config`` (no DB).
        """
        if self._db is None:
            logger.debug("MCPPool.reload(): no DB wired, skipping")
            return
        new_specs = await self.rebuild_specs()

        async with self._lock:
            old_supervisors = list(self._toolkit_supervisors)
            old_toolkits = list(self._runtime_toolkits)
            old_in_proc_agno = list(self._in_process_runtime_toolkits)
            # Clear in-place so existing references (pool.runtime_toolkits returned
            # by value is list[Any] built on each call, so that's fine; but
            # provider code may also have stashed ``pool``, and after reload
            # calls pool.runtime_toolkits again it must see the new list.)
            self._toolkit_supervisors.clear()
            self._runtime_toolkits.clear()
            self._in_process_runtime_toolkits.clear()
            self._toolkit_by_name.clear()
            self.specs = new_specs
            self._tool_counts = {s.name: 0 for s in new_specs}
            self._last_recovery_error_type = {}
            self._last_connect_error = {}
            self._connected = False
            # Server set + tool counts changed → drop the cached catalog
            # summary; the next prompt render will rebuild it from the
            # fresh self._tool_counts / self.specs.
            self._catalog_summary_cache = None

        # connect_all has its own lock; new subprocesses come up first.
        await self.connect_all()

        # Tear down the old supervisors. Best-effort — a broken shutdown on
        # the old set must not prevent new tools from being used. Close in
        # reverse order to mirror close_all's contract.
        for sup in reversed(old_supervisors):
            await self._shutdown_supervisor(sup)
        elog(
            "mcp.pool.reload",
            new_servers=len(new_specs),
            old_toolkits=len(old_toolkits) + len(old_in_proc_agno),
        )

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect_all(self) -> None:
        """Connect every toolkit. Safe to call multiple times — re-entry is a no-op.

        Per-MCP failures are isolated: if ``bad-mcp`` can't handshake, it's
        logged as dormant and the rest still come online. This is a
        *deliberate* weakening of the old all-or-nothing semantics — in
        production, one crashed MCP must not take the whole agent down
        with it. See the v0.5.29 tests in
        ``scripts/tests/test_mcp_pool_resilience.py`` for the failure
        modes this guards against.
        """
        async with self._lock:
            if self._connected:
                return
            # 1. Subprocess-based MCPs (stdio / HTTP).
            # Stagger spawns to avoid the cold-start clustering that produces
            # the dormant-recovery TimeoutError — see _STARTUP_STAGGER_SECONDS.
            first_subprocess = True
            for spec in self.specs:
                if spec.in_process:
                    continue
                if not first_subprocess and _STARTUP_STAGGER_SECONDS > 0:
                    await asyncio.sleep(_STARTUP_STAGGER_SECONDS)
                first_subprocess = False
                toolkit = await self._build_and_enter_toolkit(spec)
                if toolkit is not None:
                    self._runtime_toolkits.append(toolkit)
                    self._toolkit_by_name[spec.name] = toolkit
                    # MCPTools.functions is the dict of registered tools
                    # (populated during MCPTools.initialize()).
                    count = len(getattr(toolkit, "functions", {}) or {})
                    if count == 0:
                        # Stealth-fail recovery — see _DORMANT_RECOVERY_ATTEMPTS
                        # docstring above. The subprocess is alive (we entered
                        # the stack); the runtime just lost the init result.
                        # Retrying ``initialize()`` directly is cheap and
                        # usually wins.
                        count = await self._recover_dormant_toolkit(toolkit, spec)
                        # Re-spawn escalation: when recovery's bypass loop
                        # exhausts itself entirely on a "session dead"
                        # signature (ClosedResourceError / BrokenResourceError
                        # / EndOfStream — see _SESSION_DEAD_ERROR_TYPES), the
                        # underlying anyio memory stream is gone and no
                        # amount of re-calling initialize on this toolkit
                        # will recover it. Throw the corpse away, build a
                        # fresh subprocess + session pair, and try again.
                        # Empirically the second spawn lands healthy because
                        # PyInstaller's _MEI extract is already warm by
                        # then, so the cold-start race that killed the
                        # first attempt has resolved.
                        if (
                            count == 0
                            and self._last_recovery_error_type.get(spec.name)
                            in _SESSION_DEAD_ERROR_TYPES
                        ):
                            count = await self._respawn_after_dead_session(spec, toolkit)
                    self._tool_counts[spec.name] = count
                    if count > 0:
                        self._last_connect_error.pop(spec.name, None)
                    elog("mcp.connect", name=spec.name, tools=count)
                    if count == 0:
                        # Ensure a dormant MCP carries *some* cause for the
                        # validator/UI even when recovery saw no explicit
                        # exception (the runtime's blanket-except swallowed
                        # it). A specific cause set in _recover_dormant_toolkit
                        # (e.g. "401 Unauthorized") takes precedence.
                        self._last_connect_error.setdefault(
                            spec.name,
                            "connected but exposed 0 tools — the handshake or "
                            "tools/list likely failed (check the MCP's URL, "
                            "Authorization header, and transport)",
                        )
                        elog("mcp.dormant", level="warning", name=spec.name)

            # 2. In-process MCPs — loaded by importing the adapter module and
            #    calling the named factory functions. No subprocess is spawned.
            for spec in self.specs:
                if not spec.in_process:
                    continue
                try:
                    mod = importlib.import_module(spec.adapter_module)  # type: ignore[arg-type]
                    runtime_factory = getattr(mod, spec.runtime_toolkit_factory, None)
                except Exception as e:  # noqa: BLE001
                    elog("mcp.error", level="warning", name=spec.name, error=str(e), phase="import")
                    continue
                if runtime_factory is None:
                    logger.warning(
                        "in-process MCP '%s' missing factory (%s) — skipping",
                        spec.name, spec.runtime_toolkit_factory,
                    )
                    continue
                # Inject ``pool=self`` only for factories that accept it.
                # Existing in-process adapters (e.g. shell) take no kwargs;
                # tool-search needs the pool to navigate other MCPs. We
                # detect this via signature so neither side has to know
                # about the other.
                def _factory_kwargs(factory: Any) -> dict[str, Any]:
                    try:
                        params = inspect.signature(factory).parameters
                    except (TypeError, ValueError):
                        return {}
                    return {"pool": self} if "pool" in params else {}

                try:
                    runtime_tk = runtime_factory(**_factory_kwargs(runtime_factory))
                except Exception as e:  # noqa: BLE001
                    elog("mcp.error", level="warning", name=spec.name, error=str(e), phase="factory")
                    continue
                self._in_process_runtime_toolkits.append(runtime_tk)
                self._toolkit_by_name[spec.name] = runtime_tk
                # Count both sync and async tool dicts — Toolkits with
                # async-only callables register them in ``async_functions``
                # and leave ``functions`` empty, so the prior hardcoded
                # ``count = 6`` (a leftover from when shell was the only
                # in-process MCP) under-counted in general.
                count = (
                    len(getattr(runtime_tk, "functions", {}) or {})
                    + len(getattr(runtime_tk, "async_functions", {}) or {})
                )
                self._tool_counts[spec.name] = count
                elog("mcp.connect", name=spec.name, tools=count, kind="in_process")

            self._connected = True
            # Tool counts may have shifted during connect_all (dormant
            # recoveries, fresh in-process tool registrations) — drop
            # the cached catalog summary so the next prompt render
            # rebuilds it.
            self._catalog_summary_cache = None

    async def close_all(self) -> None:
        """Close every connected toolkit. Per-toolkit supervisors are closed
        independently so one failing teardown doesn't skip the rest.

        Each toolkit is owned by a supervisor task that ran
        ``async with stack: await stack.enter_async_context(toolkit); await
        stop_event.wait()`` from the start. We signal its ``stop_event`` and
        await the task — the ``AsyncExitStack.__aexit__`` runs inside the
        supervisor, which is the same task that called ``__aenter__``. That
        satisfies the anyio cancel-scope invariant that the MCP stdio
        client relies on (see the v0.6.x fix commit message).
        """
        async with self._lock:
            if not self._connected:
                return
            supervisors = list(self._toolkit_supervisors)
            self._toolkit_supervisors.clear()
            self._runtime_toolkits.clear()
            self._in_process_runtime_toolkits.clear()
            self._toolkit_by_name.clear()
            self._connected = False
        # Close in reverse registration order so toolkits that share
        # resources tear down the way AsyncExitStack would have.
        for sup in reversed(supervisors):
            await self._shutdown_supervisor(sup)

    async def _respawn_after_dead_session(
        self, spec: _ServerSpec, dead_toolkit: Any,
    ) -> int:
        """Drop a dormant toolkit whose session is structurally dead and
        bring up a fresh one for the same spec. Returns the new tool count
        (0 if the re-spawn also failed).

        Called from ``connect_all`` ONLY when ``_recover_dormant_toolkit``
        exhausted its retry budget against a session-dead error
        (``ClosedResourceError`` & friends — see ``_SESSION_DEAD_ERROR_TYPES``).
        Re-spawn handles the cold-start race where the first subprocess
        managed to spawn but lost its stdio streams during the runtime's
        swallowed init — by the second spawn PyInstaller's _MEI tree is
        warm and the handshake lands cleanly.

        The dead toolkit is removed from every pool registry (``_runtime_toolkits``,
        ``_toolkit_by_name``) BEFORE we attempt the re-spawn so that even
        if the re-spawn fails the pool's view of the spec stays consistent
        with "no toolkit loaded" rather than "dormant toolkit pinned".
        The dead supervisor is shut down asynchronously so a stuck
        SIGTERM cleanup cannot pin connect_all.
        """
        # Find and detach the dead supervisor; shut it down in the
        # background so we don't pay its _MCP_CLOSE_TIMEOUT inline.
        dead_sup: _ToolkitSupervisor | None = None
        for sup in self._toolkit_supervisors:
            if sup.name == spec.name:
                dead_sup = sup
                break
        if dead_sup is not None:
            self._toolkit_supervisors.remove(dead_sup)
            asyncio.create_task(
                self._shutdown_supervisor(dead_sup),
                name=f"mcp-dead-shutdown:{spec.name}",
            )
        try:
            self._runtime_toolkits.remove(dead_toolkit)
        except ValueError:
            pass
        # Don't pop from _toolkit_by_name yet — we'll overwrite it below
        # if the re-spawn succeeds. Popping then re-inserting would leave
        # a transient window where toolkit_by_name(spec.name) returns None
        # for any caller racing with us.

        elog(
            "mcp.respawn_attempt",
            level="warning",
            name=spec.name,
            reason="dead_session",
            prior_error=self._last_recovery_error_type.get(spec.name, ""),
        )
        new_toolkit = await self._build_and_enter_toolkit(spec)
        if new_toolkit is None:
            # Re-spawn itself failed (subprocess never came up). Pop the
            # by-name entry so the pool reports the spec as not-loaded
            # consistently with the cleared _runtime_toolkits.
            self._toolkit_by_name.pop(spec.name, None)
            elog("mcp.respawn_failed", level="warning", name=spec.name)
            return 0
        self._runtime_toolkits.append(new_toolkit)
        self._toolkit_by_name[spec.name] = new_toolkit
        count = len(getattr(new_toolkit, "functions", {}) or {})
        # If the FRESH toolkit also stealth-fails, run recovery one more
        # time — but do NOT recurse into another re-spawn. A second-spawn
        # session-dead is a real "this MCP is broken" signal and the
        # caller (connect_all) will mark it dormant via the standard
        # path. Capping at one re-spawn keeps the worst-case wallclock
        # bounded for a permanently-broken MCP.
        if count == 0:
            count = await self._recover_dormant_toolkit(new_toolkit, spec)
        if count > 0:
            elog("mcp.respawn_succeeded", name=spec.name, tools=count)
        return count

    async def _shutdown_supervisor(self, sup: _ToolkitSupervisor) -> None:
        """Tell one supervisor to exit its ``async with`` block and wait.

        Bounded by ``_MCP_CLOSE_TIMEOUT`` — a stuck subprocess that ignores
        SIGTERM cannot pin the overall shutdown. On timeout we cancel the
        task and swallow any BaseException that follows (anyio cleanup on a
        cancelled task can still raise during shutdown).
        """
        sup.stop_event.set()
        try:
            await asyncio.wait_for(sup.task, timeout=_MCP_CLOSE_TIMEOUT)
        except asyncio.TimeoutError:
            elog(
                "mcp.close_timeout",
                level="warning",
                name=sup.name,
                seconds=_MCP_CLOSE_TIMEOUT,
            )
            sup.task.cancel()
            try:
                await sup.task
            except BaseException:  # noqa: BLE001 — best-effort cleanup
                pass
        except BaseException as e:  # noqa: BLE001 — best-effort cleanup
            logger.debug("Best-effort MCP supervisor close (%s): %s", sup.name, e)

    async def _safe_enter(self, toolkit: Any, spec: _ServerSpec) -> Any | None:
        """Spawn a supervisor task that owns this toolkit end-to-end, then
        wait up to ``_MCP_CONNECT_TIMEOUT`` for it to finish the handshake.

        Success path: supervisor enters ``stack`` and ``stack.enter_async_context(
        toolkit)``, sets ``ready_event``, and blocks on ``stop_event`` until
        shutdown. We register the supervisor in ``_toolkit_supervisors`` and
        return the toolkit.

        Failure path: exception / timeout inside the handshake exits the
        supervisor with ``ready_event`` still carrying the error info. We
        log, await the supervisor to let the stack unwind cleanly, and
        return ``None``.

        Three distinct failure modes are handled:

        1. ``asyncio.TimeoutError`` — handshake exceeded
           ``_MCP_CONNECT_TIMEOUT``. Signal the supervisor to stop, await
           it, return ``None``.
        2. ``asyncio.CancelledError`` / ``BaseExceptionGroup`` from
           inside the runtime's init — observed after a host disk-resize on a
           production agent. Caught inside the supervisor and reported
           via ``ready_event``.
        3. Regular ``Exception`` — logged + return ``None``.

        Why the supervisor task at all: the MCP stdio client in
        ``mcp/client/stdio/__init__.py`` opens an anyio cancel scope in
        ``__aenter__`` and closes it in ``__aexit__``. anyio enforces that
        both ends happen on the same task; otherwise it raises "Attempted
        to exit cancel scope in a different task than it was entered in".
        Before this refactor, ``__aenter__`` ran on the ``connect_all``
        caller task but ``__aexit__`` ran on the shutdown caller task,
        which reliably tripped that invariant every restart.
        """
        task_out = asyncio.current_task()
        cancelling = (
            getattr(task_out, "cancelling", None) if task_out is not None else None
        )
        cancel_before = cancelling() if callable(cancelling) else 0

        ready_event = asyncio.Event()
        stop_event = asyncio.Event()
        # Holds the exception raised during handshake, if any.
        handshake_error: dict[str, BaseException | None] = {"err": None}

        async def _supervisor() -> None:
            # Open the stack INSIDE this task so __aenter__ and __aexit__
            # are both on the same task — anyio's cancel-scope invariant.
            try:
                async with AsyncExitStack() as stack:
                    try:
                        async with asyncio.timeout(_MCP_CONNECT_TIMEOUT):
                            await stack.enter_async_context(toolkit)
                    except BaseException as e:
                        handshake_error["err"] = e
                        ready_event.set()
                        # Exiting the AsyncExitStack here runs any partial
                        # __aexit__ on the same task — correct.
                        return
                    # Handshake succeeded. Signal the pool and hold the
                    # stack open until shutdown asks us to close.
                    ready_event.set()
                    try:
                        await stop_event.wait()
                    except asyncio.CancelledError:
                        # close_all upgrades to cancel when the graceful
                        # wait_for times out. Let the stack unwind on this
                        # same task — do NOT re-raise until after __aexit__.
                        pass
            except BaseException:  # noqa: BLE001 — best-effort cleanup
                # Any cleanup exception is logged by the caller; keep the
                # task complete so close_all's await returns.
                pass

        sup_task = asyncio.create_task(
            _supervisor(), name=f"mcp-supervisor:{spec.name}"
        )

        async def _wait_handshake() -> None:
            await ready_event.wait()

        try:
            # +1 grace second so ``asyncio.timeout`` inside the supervisor
            # fires FIRST and records the error, rather than this outer
            # wait_for tripping on the same boundary and synthesising a
            # TimeoutError of its own.
            await asyncio.wait_for(
                _wait_handshake(), timeout=_MCP_CONNECT_TIMEOUT + 1
            )
        except asyncio.TimeoutError:
            # Supervisor didn't report ready within our outer bound — stop
            # it, drain it, and report as a timeout. This path is defensive:
            # the inner ``asyncio.timeout(_MCP_CONNECT_TIMEOUT)`` should
            # already have fired and set ready_event with an error.
            elog("mcp.timeout", level="warning", name=spec.name, seconds=_MCP_CONNECT_TIMEOUT)
            stop_event.set()
            try:
                await asyncio.wait_for(sup_task, timeout=_MCP_CLOSE_TIMEOUT)
            except asyncio.TimeoutError:
                sup_task.cancel()
                try:
                    await sup_task
                except BaseException:  # noqa: BLE001
                    pass
            return None
        except BaseException as e:
            # The outer task itself was cancelled while we were waiting.
            # Tell the supervisor to exit, then re-raise so shutdown
            # propagates like it used to.
            stop_event.set()
            try:
                await asyncio.wait_for(sup_task, timeout=_MCP_CLOSE_TIMEOUT)
            except BaseException:  # noqa: BLE001
                sup_task.cancel()
                try:
                    await sup_task
                except BaseException:
                    pass
            if callable(cancelling) and cancelling() > cancel_before:
                raise
            elog("mcp.error", level="warning", name=spec.name, error=str(e), phase="connect")
            return None

        err = handshake_error["err"]
        if err is not None:
            # Handshake failed inside the supervisor. The supervisor has
            # already returned (its async-with block exited on the same
            # task it entered); we just need to await the task to
            # completion and surface the error.
            try:
                await sup_task
            except BaseException:  # noqa: BLE001
                pass
            if isinstance(err, asyncio.TimeoutError):
                elog("mcp.timeout", level="warning", name=spec.name, seconds=_MCP_CONNECT_TIMEOUT)
            else:
                # If the outer task was externally cancelled (not us —
                # something above us asked for shutdown), propagate.
                if callable(cancelling) and cancelling() > cancel_before:
                    raise err
                elog("mcp.error", level="warning", name=spec.name, error=str(err), phase="connect")
            return None

        # Success — register the supervisor so close_all can unwind it.
        self._toolkit_supervisors.append(
            _ToolkitSupervisor(name=spec.name, task=sup_task, stop_event=stop_event)
        )
        return toolkit

    # Last error from the most recent _recover_dormant_toolkit call,
    # keyed by spec.name. Lets connect_all decide whether a dead session
    # warrants a full re-spawn (see _SESSION_DEAD_ERROR_TYPES below).
    # Populated even on success — empty string means "no error captured".
    _last_recovery_error_type: dict[str, str]

    async def _recover_dormant_toolkit(
        self, toolkit: Any, spec: _ServerSpec,
    ) -> int:
        """Re-run init against a stealth-failed handshake AND surface the
        real exception that the runtime's blanket ``except (RuntimeError,
        BaseException): log_error(...)`` would otherwise eat.

        Two paths, picked by introspection so this still works across
        runtime rewrites and on in-process Toolkits:

        1. **Bypass path (preferred)** — when the toolkit looks like
           the runtime's ``MCPTools`` (has ``session`` + ``build_tools``),
           we skip ``initialize()`` and call its two underlying coroutines
           directly: ``session.initialize()`` then ``build_tools()``.
           Identical work to what the runtime's init does, but WITHOUT the
           blanket-except, so the real exception propagates and we can
           log type, message, and traceback. The ``phase`` field tells
           ops which step blew up — ``session.initialize`` (handshake
           with the subprocess) vs ``build_tools`` (parsing the tool
           catalogue) — which is the diagnostic we don't currently have.
        2. **Fallback path** — for toolkits we don't recognise, just
           call ``initialize()``. Same behaviour as before; we still
           catch and log so a permanently-broken MCP eventually dormants.

        ``initialize()`` and the two underlying coroutines do NOT touch
        the anyio cancel scope opened by ``stdio_client.__aenter__``
        (only ``__aexit__`` does). So calling them from this task —
        which is NOT the supervisor task — is safe; the scope stays
        owned end-to-end by the supervisor.

        Returns the tool count after recovery (zero if all retries
        failed). Recovery telemetry is emitted as ``mcp.recovery_failed``
        / ``mcp.recovery_succeeded`` with enough fields to root-cause
        the next incident — see the PR description for the bug history.
        """
        # Reset the per-attempt error trace BEFORE the loop. We populate
        # ``_last_recovery_error_type[spec.name]`` from each attempt's
        # exception class so ``connect_all`` can decide whether to escalate
        # to a full re-spawn (see ``_SESSION_DEAD_ERROR_TYPES``).
        self._last_recovery_error_type[spec.name] = ""

        # Detect the bypass path. We test for callable methods, not for
        # the MCPTools class, so a future runtime rewrite that keeps the
        # same shape still works.
        session = getattr(toolkit, "session", None)
        build_tools = getattr(toolkit, "build_tools", None)
        bypass_available = (
            session is not None
            and callable(getattr(session, "initialize", None))
            and callable(build_tools)
        )
        initialize = getattr(toolkit, "initialize", None)
        if not bypass_available and not callable(initialize):
            # Nothing we can do — toolkit doesn't expose a re-init path.
            return len(getattr(toolkit, "functions", {}) or {})

        for attempt in range(1, _DORMANT_RECOVERY_ATTEMPTS + 1):
            # Reset the idempotency flag: ``initialize()`` short-circuits
            # on ``self._initialized``, even when the previous attempt
            # left ``functions`` empty. Best-effort — the attribute may
            # not exist on non-runtime toolkits.
            try:
                toolkit._initialized = False  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

            phase: str | None = None
            err: BaseException | None = None
            try:
                async with asyncio.timeout(_DORMANT_RECOVERY_TIMEOUT):
                    if bypass_available:
                        # Bypass the runtime's blanket-except so the real
                        # error surfaces. ``session.initialize`` is the MCP
                        # protocol-level handshake (initialize/initialized
                        # message exchange); ``build_tools`` lists tools
                        # and constructs Function wrappers.
                        phase = "session.initialize"
                        await session.initialize()
                        phase = "build_tools"
                        await build_tools()
                        try:
                            toolkit._initialized = True  # type: ignore[attr-defined]
                        except Exception:  # noqa: BLE001
                            pass
                    else:
                        phase = "initialize"
                        result = initialize()  # type: ignore[misc]
                        if inspect.isawaitable(result):
                            await result
            except BaseException as e:  # noqa: BLE001 — we want EVERYTHING
                err = e

            count = len(getattr(toolkit, "functions", {}) or {})
            if count > 0:
                self._last_connect_error.pop(spec.name, None)
                elog(
                    "mcp.recovery_succeeded",
                    name=spec.name,
                    attempt=attempt,
                    tools=count,
                    phase=phase,
                    bypassed_agno_swallow=bypass_available,
                )
                return count

            # Recovery attempt failed — log with full diagnostic context.
            # ``error_type`` is what we actually need to root-cause:
            # BaseExceptionGroup vs CancelledError vs ConnectionError
            # vs ValueError each point at a different class of bug.
            if err is not None:
                self._last_recovery_error_type[spec.name] = type(err).__name__
                # Remember a human-readable cause so the workflow validator
                # and UI can explain a present-but-empty MCP (0 tools) with
                # the REAL reason — e.g. "401 Unauthorized" from a missing
                # Authorization header — instead of a bare empty tool list.
                self._last_connect_error[spec.name] = (
                    f"{type(err).__name__}: {str(err)[:200]}"
                )
                # Pull a short traceback summary — full traceback is too
                # noisy for elog but the last frame usually identifies
                # the failing call (e.g. anyio TaskGroup vs mcp.session).
                import traceback as _tb
                tb_summary = "".join(
                    _tb.format_exception(type(err), err, err.__traceback__)
                )[-1500:]
                # If it's an ExceptionGroup, also unpack inner errors —
                # those are where the real anyio cancel-scope failures
                # live, hidden under a generic group wrapper.
                inner_types: list[str] = []
                inner_msgs: list[str] = []
                exceptions = getattr(err, "exceptions", None)
                if exceptions:
                    for sub in exceptions[:5]:
                        inner_types.append(type(sub).__name__)
                        inner_msgs.append(str(sub)[:200])
                elog(
                    "mcp.recovery_failed",
                    level="warning",
                    name=spec.name,
                    attempt=attempt,
                    phase=phase,
                    error_type=type(err).__name__,
                    error=str(err)[:500],
                    traceback=tb_summary,
                    inner_types=inner_types or None,
                    inner_msgs=inner_msgs or None,
                    bypassed_agno_swallow=bypass_available,
                )
            else:
                # No exception but functions still empty — the runtime's
                # blanket-except triggered on the fallback path and
                # we genuinely have no signal. Log so ops can grep
                # for this case and bump up the runtime's logger to DEBUG.
                elog(
                    "mcp.recovery_silent_fail",
                    level="warning",
                    name=spec.name,
                    attempt=attempt,
                    phase=phase,
                    note=(
                        "the runtime swallowed the init exception; enable "
                        "logger 'openagent' at DEBUG for the underlying "
                        "log_error message"
                    ),
                )

            if attempt < _DORMANT_RECOVERY_ATTEMPTS:
                await asyncio.sleep(_DORMANT_RECOVERY_BACKOFF * attempt)

        return 0

    async def _build_and_enter_toolkit(self, spec: _ServerSpec) -> Any | None:
        """Construct an ``MCPTools`` for one spec, enter it, return it.

        Returns ``None`` on any failure and logs a warning — one bad MCP
        shouldn't kill the agent.
        """
        try:
            # Import directly from the submodule (not via the package
            # ``__init__``) so test fakes installed in ``sys.modules`` at
            # ``src.mcp._runtime.mcp.mcp`` are picked up — going through
            # the package re-uses the real class cached at first
            # import-time.
            from src.mcp._runtime.mcp.mcp import MCPTools
            from src.mcp._runtime.mcp.params import StreamableHTTPClientParams
            from mcp import StdioServerParameters
        except ImportError as exc:
            logger.error("Cannot build MCP toolkits — runtime or mcp SDK missing: %s", exc)
            return None

        try:
            if spec.is_stdio:
                params = StdioServerParameters(
                    command=spec.command[0],
                    args=spec.command[1:] + spec.args,
                    env=spec.env,
                    cwd=spec.cwd,
                )
                toolkit = MCPTools(
                    server_params=params,
                    transport="stdio",
                    tool_name_prefix=_safe_prefix(spec.name),
                    timeout_seconds=_MCP_TIMEOUT_SECONDS,
                )
            elif spec.url:
                # Streamable HTTP is the runtime's default for URL-based servers.
                # Headers (e.g. Authorization: Bearer …) MUST be passed via
                # server_params — MCPTools' top-level kwargs don't surface
                # them to the HTTP transport, so without this every auth-
                # gated remote MCP gets a 401 and dies with
                # anyio.ClosedResourceError on session.initialize.
                http_params = StreamableHTTPClientParams(
                    url=spec.url,
                    headers=spec.headers or None,
                )
                toolkit = MCPTools(
                    server_params=http_params,
                    transport="streamable-http",
                    tool_name_prefix=_safe_prefix(spec.name),
                    timeout_seconds=_MCP_TIMEOUT_SECONDS,
                )
            else:
                logger.warning("MCP spec '%s' has neither command nor url — skipping", spec.name)
                return None
        except Exception as e:
            elog("mcp.error", level="warning", name=spec.name, error=str(e), phase="construct")
            return None

        return await self._safe_enter(toolkit, spec)

    # ── Provider-facing accessors ───────────────────────────────────────

    @property
    def runtime_toolkits(self) -> list[Any]:
        """Connected ``MCPTools`` instances plus in-process Toolkits.

        Pass directly to ``Agent(tools=...)``; both subprocess MCPTools and
        in-process Toolkit objects satisfy the same runtime interface.
        """
        return list(self._runtime_toolkits) + list(self._in_process_runtime_toolkits)

    def toolkit_by_name(self, mcp_name: str) -> Any | None:
        """Resolve a connected toolkit by MCP name, or ``None`` when
        the MCP isn't loaded. Covers both subprocess and in-process
        toolkits — used by the workflow executor to dispatch
        ``mcp-tool`` blocks.
        """
        return self._toolkit_by_name.get(mcp_name)

    def list_mcp_tools(self) -> list[dict[str, Any]]:
        """Shape-for-UI list of every loaded MCP and the tool names it
        exposes. Returns ``[{mcp_name, tools: [{name, description,
        parameters_schema}]}]`` — consumed by the workflow editor's
        tool picker and the ``GET /api/mcp-tools`` endpoint.
        """
        out: list[dict[str, Any]] = []
        for name, toolkit in self._toolkit_by_name.items():
            tools_meta: list[dict[str, Any]] = []
            # Subprocess MCPs (the runtime's MCPTools) populate ``functions``;
            # in-process Toolkits with async-only callables (tool-search,
            # potentially future ones) populate ``async_functions``
            # instead. Merge both so the UI sees every exposed tool —
            # the agent already does the merge via Toolkit's
            # ``get_async_functions()`` so this brings introspection in
            # line with what the model actually sees.
            merged = {
                **(getattr(toolkit, "functions", {}) or {}),
                **(getattr(toolkit, "async_functions", {}) or {}),
            }
            for tool_name, fn in merged.items():
                # The runtime's MCPTools wraps the remote tool; it exposes
                # ``description`` and ``parameters`` on the function
                # object after ``initialize()``. Fall back to empty
                # strings/dicts rather than the inspect machinery so
                # we don't stringify opaque pydantic models.
                tools_meta.append({
                    "name": tool_name,
                    "description": getattr(fn, "description", "") or "",
                    "parameters_schema": getattr(fn, "parameters", None) or {},
                })
            entry: dict[str, Any] = {"mcp_name": name, "tools": tools_meta}
            err = self._last_connect_error.get(name)
            if err and not tools_meta:
                # Surface the connect cause for a present-but-empty MCP so
                # the UI / API can explain 0 tools (e.g. a 401 from a missing
                # Authorization header) rather than showing a blank list.
                entry["error"] = err
            out.append(entry)
        return out

    # ── Tool-budget filtering ───────────────────────────────────────────
    #
    # LLM providers cap how many tools they accept per request (OpenAI:
    # 128). When the pool exposes more tools than the cap, the provider
    # silently drops the alphabetically-late ones — workflow-manager was
    # the canary that surfaced this bug. The method below returns a
    # trimmed view that fits the budget; the trimmed-out MCPs stay
    # reachable via the ``tool-search`` in-process MCP, which the wire
    # layer keeps in the kept set unconditionally.

    def _toolkit_tool_count(self, toolkit: Any) -> int:
        """Tool count for a toolkit, summing sync + async function dicts."""
        return (
            len(getattr(toolkit, "functions", {}) or {})
            + len(getattr(toolkit, "async_functions", {}) or {})
        )

    def _toolkit_name(self, toolkit: Any) -> str:
        """Reverse-lookup a toolkit's MCP name in ``_toolkit_by_name``."""
        for name, tk in self._toolkit_by_name.items():
            if tk is toolkit:
                return name
        return ""

    def runtime_toolkits_under_budget(self, budget: int) -> list[Any]:
        """Subset of ``runtime_toolkits`` whose combined tool count fits ``budget``.

        In-process toolkits (including ``tool-search``) are always
        included — they're cheap and ``tool-search`` is the recovery
        channel for trimmed MCPs. Subprocess toolkits are added in
        alphabetical name order; the next one is dropped when adding it
        would push the total over ``budget``. Dropped MCPs remain
        invocable via ``tool-search.call_tool``.

        Pass ``budget < 0`` to skip trimming entirely (legacy callers /
        tests).
        """
        if budget < 0:
            return self.runtime_toolkits

        in_process = list(self._in_process_runtime_toolkits)
        used = sum(self._toolkit_tool_count(tk) for tk in in_process)
        remaining = max(0, budget - used)

        subprocess_pairs = sorted(
            ((self._toolkit_name(tk), tk) for tk in self._runtime_toolkits),
            key=lambda p: _mcp_priority_key(p[0]),
        )
        kept: list[Any] = []
        for _, tk in subprocess_pairs:
            n = self._toolkit_tool_count(tk)
            # n == 0 means the subprocess connected but contributed no
            # tools (or failed to connect). Including it would sneak past
            # any budget — keeping the noisy tail of dead subprocess MCPs
            # even on the tightest cap. Only keep when there's something
            # to contribute and the budget can absorb it.
            if n > 0 and n <= remaining:
                kept.append(tk)
                remaining -= n
        return kept + in_process

    # ── Defer-all view (v0.14+) ─────────────────────────────────────────
    #
    # The router now feeds the model ONLY the tool-search MCP up front;
    # every other MCP (in-process or subprocess) is reached via
    # ``tool-search.call_tool``. Uniform discovery surface — the model
    # never gets a mixed "some tools upfront, some deferred" picture
    # that confuses delegation. The view below returns just the
    # tool-search MCP for the model's upfront tool list.

    def runtime_toolkits_tool_search_only(self) -> list[Any]:
        """Return ONLY the ``tool-search`` in-process toolkit.

        Used by ``wire_model_runtime`` to keep just the 4 meta-tools
        (``list_servers`` / ``list_tools`` / ``describe_tool`` /
        ``call_tool``) in the model's upfront tool list. All other
        connected MCPs stay alive in the pool but are reached via
        ``tool-search.call_tool``.
        """
        tk = self._toolkit_by_name.get("tool-search")
        return [tk] if tk is not None else []

    # ── Introspection (used by Agent for system prompt + health endpoints) ─

    def server_summary(self) -> dict[str, int]:
        """``{server_name: tool_count}`` for every configured server."""
        return dict(self._tool_counts)

    def server_descriptions(self) -> dict[str, str]:
        """``{server_name: 1-line description}`` for every configured server.

        Pulled from the canonical builtin catalog when a spec matches a
        known builtin (shell, vault, scheduler, …), then from the spec's
        own ``description`` field if it set one, else empty so the
        catalog-summary renderer falls back to a generic blurb. Used by
        :func:`render_catalog_summary` to inject prior-knowledge hints
        into the system prompt so the model doesn't burn a turn on
        ``list_servers``.
        """
        out: dict[str, str] = {}
        for spec in self.specs:
            builtin = BUILTIN_MCP_SPECS.get(spec.name, {}) if isinstance(
                BUILTIN_MCP_SPECS, dict
            ) else {}
            description = (
                getattr(spec, "description", "")
                or builtin.get("description", "")
                or ""
            )
            if description:
                out[spec.name] = description.strip()
        return out

    def server_tool_names(self) -> dict[str, list[str]]:
        """``{server_name: [registered tool keys]}`` for every connected MCP.

        Feeds the catalog summary (:func:`render_catalog_summary`) so the
        model can copy an exact key like ``vault_write_note`` instead of
        guessing — and mis-prefixing — it. The renderer inlines keys only
        for a small allowlist of high-traffic builtins; this returns them
        all and lets the renderer decide.
        """
        out: dict[str, list[str]] = {}
        for name, toolkit in self._toolkit_by_name.items():
            fns = dict(getattr(toolkit, "functions", {}) or {})
            fns.update(getattr(toolkit, "async_functions", {}) or {})
            if fns:
                out[name] = sorted(fns)
        return out

    def render_catalog_summary(self) -> str:
        """Cached markdown block listing every connected MCP server.

        The rendered string is injected into every chat-turn's system
        prompt at the ``{{MCP_CATALOG_SUMMARY}}`` placeholder. The
        server set + tool counts only change on hot-reload, so we cache
        the result and invalidate from ``reload()`` and ``connect_all``.
        """
        if self._catalog_summary_cache is None:
            self._catalog_summary_cache = self._build_catalog_summary()
        return self._catalog_summary_cache

    def _build_catalog_summary(self) -> str:
        """Render the catalog summary from live ``_tool_counts`` /
        ``server_descriptions``. Internal — callers use the cached
        :meth:`render_catalog_summary`.

        Foregrounds ``vault`` (the model's only durable memory) with
        stronger imperative wording. ``tool-search`` always lists last
        because the model already knows about it (it's how the model
        is reading this block).
        """
        summary = dict(self._tool_counts)
        if not summary:
            return "_(no MCPs connected.)_"
        # Delegate the markdown shape to the shared renderer in
        # ``src.core.prompts`` so the pool's cached output and the
        # duck-typed test fallback can't drift apart.
        from src.core.prompts import _render_catalog_summary_lines

        return _render_catalog_summary_lines(
            summary, self.server_descriptions(), self.server_tool_names()
        )

    def dormant_servers(self) -> list[str]:
        return sorted(name for name, count in self._tool_counts.items() if count == 0)

    @property
    def server_count(self) -> int:
        return len(self.specs)

    @property
    def total_tool_count(self) -> int:
        return sum(self._tool_counts.values())

    def __repr__(self) -> str:
        state = "connected" if self._connected else "idle"
        return f"MCPPool(servers={self.server_count}, tools={self.total_tool_count}, state={state})"
