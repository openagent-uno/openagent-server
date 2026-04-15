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
    """
    name: str
    command: list[str] | None = None
    args: list[str] = field(default_factory=list)
    url: str | None = None
    env: dict[str, str] | None = None
    cwd: str | None = None
    headers: dict[str, str] | None = None
    oauth: bool = False

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
        # Lazily populated on connect_all.
        self._agno_toolkits: list[Any] = []
        self._tool_counts: dict[str, int] = {name: 0 for name in (s.name for s in specs)}
        self._stack: AsyncExitStack | None = None
        self._connected = False
        self._lock = asyncio.Lock()

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

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect_all(self) -> None:
        """Connect every toolkit. Safe to call multiple times — re-entry is a no-op."""
        async with self._lock:
            if self._connected:
                return
            self._stack = AsyncExitStack()
            try:
                for spec in self.specs:
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
            except BaseException:
                # If anything blows up mid-connect, unwind everything we entered.
                if self._stack is not None:
                    await self._stack.aclose()
                self._stack = None
                self._agno_toolkits.clear()
                raise
            self._connected = True

    async def close_all(self) -> None:
        async with self._lock:
            if not self._connected:
                return
            stack = self._stack
            self._stack = None
            self._agno_toolkits.clear()
            self._connected = False
        if stack is not None:
            try:
                await stack.aclose()
            except Exception as e:
                logger.debug("Best-effort MCP pool close: %s", e)

    async def _build_and_enter_toolkit(self, spec: _ServerSpec) -> Any | None:
        """Construct an Agno ``MCPTools`` for one spec, enter it, return it.

        Returns ``None`` and logs a warning on failure (matching the old
        registry's behaviour: one bad MCP shouldn't kill the agent).
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

            assert self._stack is not None
            await self._stack.enter_async_context(toolkit)
            return toolkit
        except Exception as e:
            logger.warning("MCP '%s' failed to connect: %s", spec.name, e)
            elog("mcp.error", name=spec.name, error=str(e))
            return None

    # ── Provider-facing accessors ───────────────────────────────────────

    @property
    def agno_toolkits(self) -> list[Any]:
        """Connected Agno ``MCPTools`` instances. Pass directly to ``Agent(tools=...)``."""
        return list(self._agno_toolkits)

    def claude_sdk_servers(self) -> dict[str, dict[str, Any]]:
        return {spec.name: spec.claude_sdk_entry() for spec in self.specs}

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
