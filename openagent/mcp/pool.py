"""Process-level pool of MCP toolkits, owned by the Agent.

This replaces the previous in-house ``MCPRegistry`` + ``MCPTools`` pair. Both
LLM backends (Agno for API-managed models, Claude Agent SDK for ClaudeCLI)
ship their own production-grade MCP integrations, so OpenAgent owns only the
*product* layer:

  - which servers are configured (``DEFAULT_MCPS``, ``BUILTIN_MCP_SPECS``)
  - how to resolve a server name into a runnable command + env
  - per-server token wiring (messaging MCP credentials, scheduler DB path)

Concretely:

  - ``AgnoProvider`` consumes ``pool.agno_toolkits`` — a list of ``agno.tools.mcp.MCPTools``
    instances that the pool owns and connects once. Multiple ``AgnoProvider``
    tiers (under ``SmartRouter``) share the same toolkit list, so we don't
    spawn N copies of each MCP server process.

  - ``ClaudeCLI`` consumes ``pool.claude_sdk_servers`` — the raw stdio config
    dict that the Claude Agent SDK accepts as its ``mcp_servers`` parameter.
    The SDK spawns its own subprocesses per ``ClaudeSDKClient`` (one set per
    session), as it always has — that's a Claude SDK constraint, not ours.

The pool also exposes ``server_summary()`` and ``dormant_servers()`` so the
Agent can inject runtime MCP state into the system prompt (e.g. "messaging
is configured but has no tools — set TELEGRAM_BOT_TOKEN to enable").
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import shutil
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from openagent.core.logging import elog
from openagent.mcp.builtins import DEFAULT_MCPS, resolve_builtin_entry, resolve_default_entry

logger = logging.getLogger(__name__)

# Tokens that must be forwarded from os.environ into the messaging MCP
# subprocess env. Agno's MCPTools and the Claude SDK both filter env to a
# safe subset by default, so we have to copy these explicitly.
_MESSAGING_TOKEN_ENV_VARS = (
    "TELEGRAM_BOT_TOKEN",
    "DISCORD_BOT_TOKEN",
    "GREEN_API_ID",
    "GREEN_API_TOKEN",
)

# Per-MCP-call timeout. Agno's MCPTools defaults to 10s, which was too tight
# for cold-start tool calls against npx-launched servers; we bumped to 30s
# and then hit the opposite problem — 30s is *catastrophically* tight for
# tools that legitimately run for minutes, like ``shell_exec`` driving a
# macOS Electron build. A 30-min ceiling matches the shell MCP's own
# MAX_TIMEOUT so the Agno wrapper doesn't cut the call off before the tool
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


def _safe_prefix(name: str) -> str:
    """Coerce a server name into a valid Python identifier prefix.

    Agno emits tool names as ``<prefix>_<tool>``; ``computer-control`` →
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
    sdk_server_factory: str = "build_sdk_server"
    agno_toolkit_factory: str = "build_agno_toolkit"

    @property
    def is_stdio(self) -> bool:
        return bool(self.command)

    def claude_sdk_entry(self) -> dict[str, Any]:
        """Format this spec for the Claude Agent SDK's ``mcp_servers`` param.

        SDK schema requires ``command`` to be a string (the executable) and
        ``args`` to be a list of arg strings, NOT a single concatenated argv.
        URL servers need an explicit ``type: "http"`` (or ``"sse"``).
        """
        if self.is_stdio:
            full_cmd = (self.command or []) + self.args
            entry: dict[str, Any] = {
                "command": full_cmd[0],
                "args": full_cmd[1:],
            }
            if self.env:
                entry["env"] = self.env
            return entry
        entry = {"type": "http", "url": self.url}
        if self.headers:
            entry["headers"] = self.headers
        return entry


def _normalise_spec(spec: _ServerSpec) -> None:
    """Resolve relative commands to absolute paths and inject host secrets.

    Mutates ``spec`` in place. Two concerns rolled into one pass since both
    apply to the same spec list at the same point in the pipeline:

    - Absolute command path: the Claude Agent SDK silently drops stdio MCPs
      whose first argv arg can't be resolved on the subprocess ``$PATH``
      (an issue under systemd). Resolving up-front avoids the footgun.

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
        sdk_server_factory=kwargs.get("sdk_server_factory", "build_sdk_server"),
        agno_toolkit_factory=kwargs.get("agno_toolkit_factory", "build_agno_toolkit"),
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
        # supervisor task (parallel arrays, ``_agno_toolkits[i]`` is owned
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
        self._agno_toolkits: list[Any] = []
        self._toolkit_supervisors: list[_ToolkitSupervisor] = []
        self._tool_counts: dict[str, int] = {name: 0 for name in (s.name for s in specs)}
        self._connected = False
        self._lock = asyncio.Lock()
        # In-process MCP state — populated by connect_all for specs with
        # in_process=True. These are kept separate from subprocess toolkits
        # so close_all can handle them independently (no AsyncExitStack needed).
        self._in_process_sdk_servers: dict[str, Any] = {}
        self._in_process_agno_toolkits: list[Any] = []
        # DB reference for ``reload()``. Set by ``from_db``; ``None`` for
        # ``from_config`` callers (tests) — reload is a no-op in that mode.
        self._db: Any = None
        self._db_path: str | None = None

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
        We swap the backing lists in place (``self._agno_toolkits[:] = ...``)
        so Agno providers that already captured ``pool.agno_toolkits`` by
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
            old_agno = list(self._agno_toolkits)
            old_in_proc_agno = list(self._in_process_agno_toolkits)
            # Clear in-place so existing references (pool.agno_toolkits returned
            # by value is list[Any] built on each call, so that's fine; but
            # provider code may also have stashed ``pool``, and after reload
            # calls pool.agno_toolkits again it must see the new list.)
            self._toolkit_supervisors.clear()
            self._agno_toolkits.clear()
            self._in_process_agno_toolkits.clear()
            self._in_process_sdk_servers.clear()
            self.specs = new_specs
            self._tool_counts = {s.name: 0 for s in new_specs}
            self._connected = False

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
            old_toolkits=len(old_agno) + len(old_in_proc_agno),
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
            for spec in self.specs:
                if spec.in_process:
                    continue
                toolkit = await self._build_and_enter_toolkit(spec)
                if toolkit is not None:
                    self._agno_toolkits.append(toolkit)
                    # Agno MCPTools.functions is the dict of registered tools
                    # (populated during MCPTools.initialize()).
                    count = len(getattr(toolkit, "functions", {}) or {})
                    self._tool_counts[spec.name] = count
                    elog("mcp.connect", name=spec.name, tools=count)
                    if count == 0:
                        elog("mcp.dormant", level="warning", name=spec.name)

            # 2. In-process MCPs — loaded by importing the adapter module and
            #    calling the named factory functions. No subprocess is spawned.
            for spec in self.specs:
                if not spec.in_process:
                    continue
                try:
                    mod = importlib.import_module(spec.adapter_module)  # type: ignore[arg-type]
                    sdk_factory = getattr(mod, spec.sdk_server_factory, None)
                    agno_factory = getattr(mod, spec.agno_toolkit_factory, None)
                except Exception as e:  # noqa: BLE001
                    elog("mcp.error", level="warning", name=spec.name, error=str(e), phase="import")
                    continue
                if sdk_factory is None or agno_factory is None:
                    logger.warning(
                        "in-process MCP '%s' missing factories (%s / %s) — skipping",
                        spec.name, spec.sdk_server_factory, spec.agno_toolkit_factory,
                    )
                    continue
                try:
                    sdk_cfg = sdk_factory()
                    agno_tk = agno_factory()
                except Exception as e:  # noqa: BLE001
                    elog("mcp.error", level="warning", name=spec.name, error=str(e), phase="factory")
                    continue
                self._in_process_sdk_servers[spec.name] = sdk_cfg
                self._in_process_agno_toolkits.append(agno_tk)
                count = 6  # six shell tools
                self._tool_counts[spec.name] = count
                elog("mcp.connect", name=spec.name, tools=count, kind="in_process")

            self._connected = True

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
            self._agno_toolkits.clear()
            self._in_process_sdk_servers.clear()
            self._in_process_agno_toolkits.clear()
            self._connected = False
        # Close in reverse registration order so toolkits that share
        # resources tear down the way AsyncExitStack would have.
        for sup in reversed(supervisors):
            await self._shutdown_supervisor(sup)

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
           inside Agno's init — the mixout post-disk-resize regression.
           Caught inside the supervisor and reported via ``ready_event``.
        3. Regular ``Exception`` — logged + return ``None``.

        Why the supervisor task at all: the MCP stdio client in
        ``mcp/client/stdio/__init__.py`` opens an anyio cancel scope in
        ``__aenter__`` and closes it in ``__aexit__``. anyio enforces that
        both ends happen on the same task; otherwise it raises "Attempted
        to exit cancel scope in a different task than it was entered in".
        Before this refactor, ``__aenter__`` ran on the ``connect_all``
        caller task but ``__aexit__`` ran on the shutdown caller task,
        which reliably tripped that invariant every restart (see
        events.jsonl from lyra-agent 2026-04-20).
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

    async def _build_and_enter_toolkit(self, spec: _ServerSpec) -> Any | None:
        """Construct an Agno ``MCPTools`` for one spec, enter it, return it.

        Returns ``None`` on any failure and logs a warning — one bad MCP
        shouldn't kill the agent.
        """
        try:
            from agno.tools.mcp import MCPTools
            from mcp import StdioServerParameters
        except ImportError as exc:
            logger.error("Cannot build MCP toolkits — Agno or mcp SDK missing: %s", exc)
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
                # Streamable HTTP is Agno's default for URL-based servers.
                toolkit = MCPTools(
                    url=spec.url,
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
    def agno_toolkits(self) -> list[Any]:
        """Connected Agno ``MCPTools`` instances plus in-process Toolkits.

        Pass directly to ``Agent(tools=...)``; both subprocess MCPTools and
        in-process Toolkit objects satisfy the same Agno interface.
        """
        return list(self._agno_toolkits) + list(self._in_process_agno_toolkits)

    def claude_sdk_servers(self) -> dict[str, dict[str, Any]]:
        base = {
            spec.name: spec.claude_sdk_entry()
            for spec in self.specs
            if not spec.in_process
        }
        base.update(self._in_process_sdk_servers)
        return base

    # ── Introspection (used by Agent for system prompt + health endpoints) ─

    def server_summary(self) -> dict[str, int]:
        """``{server_name: tool_count}`` for every configured server."""
        return dict(self._tool_counts)

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
