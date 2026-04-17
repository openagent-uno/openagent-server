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
            logger.warning(
                "MCP row %r failed to resolve: %s — skipping (re-enable "
                "via the manager MCP once fixed)",
                name, exc,
            )
            elog("mcp.db_row_error", name=name, kind=kind, error=str(exc))

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
        # ``AsyncExitStack`` (parallel arrays, ``_agno_toolkits[i]`` is
        # owned by ``_toolkit_stacks[i]``) so one broken MCP's anyio
        # cancel-scope violation during startup rolls back in isolation
        # without corrupting the cleanup state of siblings. This is the
        # v0.5.29 change — the old shared-stack design coupled every
        # MCP's cleanup to every other MCP's startup, which is exactly
        # how one dead symlink (``workspace-mcp`` on mixout-agent)
        # hung the entire agent.
        self._agno_toolkits: list[Any] = []
        self._toolkit_stacks: list[AsyncExitStack] = []
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
        reference see the new tools on their next turn. Old stacks are
        torn down AFTER the new pool is live, so there is no window where
        a concurrent turn sees zero tools.

        No-op when the pool was built via ``from_config`` (no DB).
        """
        if self._db is None:
            logger.debug("MCPPool.reload(): no DB wired, skipping")
            return
        new_specs = await self.rebuild_specs()

        async with self._lock:
            old_stacks = list(self._toolkit_stacks)
            old_agno = list(self._agno_toolkits)
            old_in_proc_agno = list(self._in_process_agno_toolkits)
            # Clear in-place so existing references (pool.agno_toolkits returned
            # by value is list[Any] built on each call, so that's fine; but
            # provider code may also have stashed ``pool``, and after reload
            # calls pool.agno_toolkits again it must see the new list.)
            self._toolkit_stacks.clear()
            self._agno_toolkits.clear()
            self._in_process_agno_toolkits.clear()
            self._in_process_sdk_servers.clear()
            self.specs = new_specs
            self._tool_counts = {s.name: 0 for s in new_specs}
            self._connected = False

        # connect_all has its own lock; new subprocesses come up first.
        await self.connect_all()

        # Tear down the old stacks. Best-effort — a broken shutdown on the
        # old set must not prevent new tools from being used. Close in
        # reverse order to mirror close_all's contract.
        for stack in reversed(old_stacks):
            try:
                await stack.aclose()
            except BaseException as e:  # noqa: BLE001 — best-effort cleanup
                logger.debug("reload: best-effort close of old toolkit: %s", e)
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
                        logger.warning(
                            "MCP '%s' connected but registered 0 tools — likely "
                            "missing credentials or env vars. The server stays in "
                            "the pool and tools will appear once configured.",
                            spec.name,
                        )
                        elog("mcp.dormant", name=spec.name)

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
                    logger.warning("in-process MCP '%s' import error: %s", spec.name, e)
                    elog("mcp.error", name=spec.name, error=str(e))
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
                    logger.warning("in-process MCP '%s' factory error: %s", spec.name, e)
                    elog("mcp.error", name=spec.name, error=str(e))
                    continue
                self._in_process_sdk_servers[spec.name] = sdk_cfg
                self._in_process_agno_toolkits.append(agno_tk)
                count = 6  # six shell tools
                self._tool_counts[spec.name] = count
                elog("mcp.connect", name=spec.name, tools=count, kind="in_process")

            self._connected = True

    async def close_all(self) -> None:
        """Close every connected toolkit. Per-toolkit stacks are closed
        independently so one failing teardown doesn't skip the rest.

        Each toolkit was entered inside its own ``AsyncExitStack`` (see
        ``_safe_enter``), which means ``stack.aclose()`` runs the
        toolkit's own ``__aexit__`` — the path Agno + anyio expect for
        proper cancel-scope teardown. We still wrap each close in an
        outer ``except BaseException`` so a single broken subprocess
        can't pin the shutdown.
        """
        async with self._lock:
            if not self._connected:
                return
            stacks = list(self._toolkit_stacks)
            self._toolkit_stacks.clear()
            self._agno_toolkits.clear()
            self._in_process_sdk_servers.clear()
            self._in_process_agno_toolkits.clear()
            self._connected = False
        # Close in reverse registration order so toolkits that share
        # resources tear down the way AsyncExitStack would have.
        for stack in reversed(stacks):
            try:
                await stack.aclose()
            except BaseException as e:  # noqa: BLE001 — best-effort cleanup
                logger.debug("Best-effort MCP pool close: %s", e)

    async def _safe_enter(self, toolkit: Any, spec: _ServerSpec) -> Any | None:
        """Enter one toolkit's own ``AsyncExitStack`` with a handshake
        timeout and BaseException isolation. Returns the toolkit on
        success, or ``None`` after rolling back the per-toolkit stack on
        timeout/failure.

        Three distinct failure modes are handled:

        1. ``asyncio.TimeoutError`` — handshake exceeded
           ``_MCP_CONNECT_TIMEOUT``. Roll back the per-toolkit stack so
           any half-initialised subprocess is cleaned up, then return
           ``None``.
        2. ``asyncio.CancelledError`` / ``BaseExceptionGroup`` from
           inside Agno's init — the mixout post-disk-resize regression.
           Same rollback, swallowed *unless* the outer task was
           externally cancelled (shutdown must propagate).
        3. Regular ``Exception`` — logged + rolled back.

        ``asyncio.timeout`` (not ``asyncio.wait_for``) is used because
        ``wait_for`` would wrap the coroutine in a sub-task. The anyio
        cancel scope opened inside ``MCPTools.__aenter__`` would then
        belong to that sub-task, and when we later ``aclose()`` the
        stack from the outer task, anyio refuses to exit a scope that
        isn't the current task's current scope — that broken state
        bleeds into unrelated awaits later in the same event loop
        (regression observed: ``aiosqlite.connect`` in the cron test
        immediately cancelling).
        """
        task = asyncio.current_task()
        cancelling = (
            getattr(task, "cancelling", None) if task is not None else None
        )
        cancel_before = cancelling() if callable(cancelling) else 0

        stack = AsyncExitStack()
        await stack.__aenter__()

        async def _rollback() -> None:
            try:
                await stack.aclose()
            except BaseException:  # noqa: BLE001 — best-effort cleanup
                pass

        try:
            async with asyncio.timeout(_MCP_CONNECT_TIMEOUT):
                await stack.enter_async_context(toolkit)
        except asyncio.TimeoutError:
            logger.warning(
                "MCP '%s' handshake timed out after %ss — marking dormant",
                spec.name, _MCP_CONNECT_TIMEOUT,
            )
            elog("mcp.timeout", name=spec.name)
            await _rollback()
            return None
        except BaseException as e:
            await _rollback()
            # If the outer task itself was externally cancelled (its
            # ``cancelling()`` counter rose during our await), re-raise
            # so shutdown propagates. Otherwise treat this as a per-MCP
            # failure (anyio cancel-scope, synthetic CancelledError,
            # etc.) and swallow — that's the mixout regression we're
            # fixing.
            if callable(cancelling) and cancelling() > cancel_before:
                raise
            logger.warning("MCP '%s' failed to connect: %s", spec.name, e)
            elog("mcp.error", name=spec.name, error=str(e))
            return None

        # Success — hand the stack to the pool so close_all can unwind it.
        self._toolkit_stacks.append(stack)
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
            logger.warning("MCP '%s' failed to construct: %s", spec.name, e)
            elog("mcp.error", name=spec.name, error=str(e))
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
