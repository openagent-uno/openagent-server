"""MCP client: connect to any MCP server (local or remote), list tools, call them.

Configure MCP servers once in openagent.yaml, they get injected into all models.
Includes default MCPs (filesystem, fetch, shell, computer-control) that are always
loaded unless explicitly disabled. User MCPs are merged on top.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from openagent.core.logging import elog
from openagent.mcp.builtins import (
    BUILTIN_MCP_SPECS,
    DEFAULT_MCPS,
    resolve_builtin_entry,
    resolve_default_entry,
)
from openagent.mcp.transport_stdio import ManagedStdioTransport

logger = logging.getLogger(__name__)


class MCPTools:
    """Single MCP server connection.

    Usage:
        # Local server (stdio transport)
        mcp = MCPTools(name="fs", command=["npx", "-y", "@anthropic/mcp-filesystem", "/data"])

        # Remote server (SSE/HTTP transport)
        mcp = MCPTools(name="search", url="http://localhost:8080/sse")

        # Remote server with OAuth (e.g. Quo, ClickUp official)
        mcp = MCPTools(name="quo", url="https://mcp.quo.com/sse", oauth=True)
    """

    def __init__(
        self,
        name: str = "",
        command: list[str] | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        env: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        oauth: bool = False,
        _cwd: str | None = None,
    ):
        self.name = name or (command[0] if command else url or "mcp")
        self.command = command
        self.args = args or []
        self.url = url
        self.env = env
        self.headers = headers
        self.oauth = oauth
        self._cwd = _cwd

        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._stdio_transport: ManagedStdioTransport | None = None
        self._tools: list[dict[str, Any]] = []

    async def connect(self) -> None:
        """Connect to the MCP server and discover tools."""
        if self._session:
            return

        if self.command:
            full_command = self.command + self.args
            # Merge custom env with system env (MCP SDK replaces entirely if env is set)
            import os
            merged_env: dict[str, str] | None = None
            if self.env:
                merged_env = {**os.environ, **self.env}
            if self._cwd:
                merged_env = {**(merged_env or os.environ), "CWD": self._cwd}
            server_params = StdioServerParameters(
                command=full_command[0],
                args=full_command[1:],
                env=merged_env,
                cwd=self._cwd,
            )
            self._stdio_transport = ManagedStdioTransport(server_params)
            try:
                read_stream, write_stream = await self._stdio_transport.start()
                self._session = ClientSession(read_stream, write_stream)
                await self._session.__aenter__()
            except Exception:
                await self._stdio_transport.aclose()
                self._stdio_transport = None
                raise
        elif self.url:
            self._exit_stack = AsyncExitStack()
            # Build auth provider for OAuth-enabled MCPs
            auth = None
            if self.oauth:
                from openagent.mcp.oauth import create_oauth_provider
                auth = create_oauth_provider(self.name, self.url)

            # Try Streamable HTTP first, fallback to SSE
            try:
                http_transport = await self._exit_stack.enter_async_context(
                    streamablehttp_client(self.url, headers=self.headers)
                )
                read_stream, write_stream = http_transport[0], http_transport[1]
                self._session = await self._exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
            except Exception:
                # Fallback to SSE
                self._exit_stack = AsyncExitStack()
                try:
                    sse_transport = await self._exit_stack.enter_async_context(
                        sse_client(self.url, headers=self.headers, auth=auth)
                    )
                    read_stream, write_stream = sse_transport
                    self._session = await self._exit_stack.enter_async_context(
                        ClientSession(read_stream, write_stream)
                    )
                except Exception as e:
                    raise ConnectionError(f"Failed to connect to {self.url}: {e}")
        else:
            raise ValueError("MCPTools requires either 'command' (stdio) or 'url' (HTTP/SSE)")

        await self._session.initialize()

        # Discover tools (some servers don't advertise tools capability)
        try:
            tools_result = await self._session.list_tools()
            self._tools = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {"type": "object", "properties": {}},
                }
                for tool in tools_result.tools
            ]
        except Exception:
            self._tools = []
        logger.info(f"MCP '{self.name}': discovered {len(self._tools)} tools")

    async def close(self) -> None:
        """Close the connection."""
        stack = self._exit_stack
        stdio_transport = self._stdio_transport
        self._session = None
        self._exit_stack = None
        self._stdio_transport = None
        self._tools = []
        if stdio_transport is not None:
            try:
                await asyncio.wait_for(stdio_transport.aclose(), timeout=3)
            except Exception as e:
                logger.debug("Best-effort MCP stdio close for '%s': %s", self.name, e)
        if stack is not None:
            try:
                await asyncio.wait_for(stack.aclose(), timeout=3)
            except Exception as e:
                logger.debug("Best-effort MCP close for '%s': %s", self.name, e)

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Get tool definitions in provider-neutral format."""
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool and return the result as a string."""
        if not self._session:
            raise RuntimeError(f"MCP '{self.name}' is not connected. Call connect() first.")

        result = await self._session.call_tool(name, arguments)

        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "\n".join(parts)

    def __repr__(self) -> str:
        transport = f"stdio:{self.command}" if self.command else f"sse:{self.url}"
        return f"MCPTools(name={self.name!r}, {transport}, tools={len(self._tools)})"


class MCPRegistry:
    """Registry of all MCP servers. Configure once, inject into all agents.

    Default MCPs (filesystem, fetch, shell, computer-control) are always loaded
    unless disabled. User-configured MCPs are merged on top:

        { ...default_mcps, ...user_mcps }

    Disable defaults:
        - In YAML: set `mcp_defaults: false`
        - In code: `MCPRegistry.from_config(config, include_defaults=False)`
        - Disable specific ones: `mcp_disable: ["computer-control", "fetch"]`
    """

    def __init__(self):
        self._servers: list[MCPTools] = []
        self._tool_map: dict[str, MCPTools] = {}  # tool_name -> server

    def add(self, server: MCPTools) -> None:
        self._servers.append(server)

    async def connect_all(self) -> None:
        """Connect to all registered MCP servers."""
        for server in self._servers:
            try:
                await server.connect()
                for tool in server.tools:
                    self._tool_map[tool["name"]] = server
                elog("mcp.connect", name=server.name, tools=len(server.tools))
            except Exception as e:
                # Downgrade to debug for expected failures (no tokens, etc.)
                logger.debug(f"Skipping MCP '{server.name}': {e}")
                elog("mcp.error", name=server.name, error=str(e))

    async def close_all(self) -> None:
        """Close all connections."""
        async def _close_server(server: MCPTools) -> None:
            try:
                await server.close()
            except Exception as e:
                logger.error(f"Failed to close MCP '{server.name}': {e}")

        await asyncio.gather(*(_close_server(server) for server in self._servers), return_exceptions=True)
        self._tool_map.clear()

    def all_tools(self) -> list[dict[str, Any]]:
        """Get a flat list of all tool definitions from all servers."""
        tools = []
        for server in self._servers:
            tools.extend(server.tools)
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Route a tool call to the correct MCP server."""
        server = self._tool_map.get(name)
        if not server:
            raise ValueError(f"Unknown tool: {name}. Available: {list(self._tool_map.keys())}")
        elog("mcp.tool.start", tool=name, server=server.name, params=arguments)
        try:
            result = await server.call_tool(name, arguments)
        except Exception as exc:
            elog("mcp.tool.error", tool=name, server=server.name, error=str(exc))
            raise
        elog("mcp.tool.done", tool=name, server=server.name, result_len=len(result or ""))
        return result

    @classmethod
    def from_config(
        cls,
        mcp_config: list[dict] | None = None,
        include_defaults: bool = True,
        disable: list[str] | None = None,
        db_path: str | None = None,
    ) -> MCPRegistry:
        """Build registry: defaults first, then user MCPs merged on top.

        Args:
            mcp_config: User-configured MCP entries from openagent.yaml
            include_defaults: Whether to include default MCPs (filesystem, fetch, shell, computer-control)
            disable: List of default MCP names to skip (e.g. ["computer-control", "fetch"])
            db_path: Path to the OpenAgent SQLite DB. Forwarded to the
                scheduler MCP so it reads and writes the same scheduled_tasks
                table as the in-process Scheduler.
        """
        registry = cls()
        disabled = set(disable or [])
        user_names = set()

        # Collect user MCP names so defaults don't duplicate them
        for entry in (mcp_config or []):
            name = entry.get("name") or entry.get("builtin", "")
            if name:
                user_names.add(name)

        # 1. Load defaults (skipping disabled and user-overridden ones)
        if include_defaults:
            for default_entry in DEFAULT_MCPS:
                name = default_entry.get("name") or default_entry.get("builtin", "")
                if name in disabled:
                    logger.info(f"Default MCP '{name}' disabled by config")
                    continue
                if name in user_names:
                    logger.info(f"Default MCP '{name}' overridden by user config")
                    continue

                server_kwargs = resolve_default_entry(default_entry, db_path=db_path)
                if server_kwargs:
                    registry.add(MCPTools(**server_kwargs))

        # 2. Load user MCPs on top
        for entry in (mcp_config or []):
            if "builtin" in entry:
                try:
                    server_kwargs = resolve_builtin_entry(entry["builtin"], env=entry.get("env"))
                    registry.add(MCPTools(**server_kwargs))
                except Exception as e:
                    logger.error(f"Failed to load built-in MCP '{entry['builtin']}': {e}")
            else:
                registry.add(MCPTools(
                    name=entry.get("name", ""),
                    command=entry.get("command"),
                    args=entry.get("args"),
                    url=entry.get("url"),
                    env=entry.get("env"),
                    headers=entry.get("headers"),
                    oauth=entry.get("oauth", False),
                ))

        return registry

    def __repr__(self) -> str:
        return f"MCPRegistry(servers={len(self._servers)}, tools={len(self._tool_map)})"
