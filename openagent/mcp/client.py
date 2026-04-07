"""MCP client: connect to any MCP server (local or remote), list tools, call them.

Configure MCP servers once in openagent.yaml, they get injected into all models.
"""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


class MCPTools:
    """Single MCP server connection.

    Usage:
        # Local server (stdio transport)
        mcp = MCPTools(name="fs", command=["npx", "-y", "@anthropic/mcp-filesystem", "/data"])

        # Remote server (SSE transport)
        mcp = MCPTools(name="search", url="http://localhost:8080/sse")
    """

    def __init__(
        self,
        name: str = "",
        command: list[str] | None = None,
        args: list[str] | None = None,
        url: str | None = None,
        env: dict[str, str] | None = None,
    ):
        self.name = name or (command[0] if command else url or "mcp")
        self.command = command
        self.args = args or []
        self.url = url
        self.env = env

        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools: list[dict[str, Any]] = []

    async def connect(self) -> None:
        """Connect to the MCP server and discover tools."""
        if self._session:
            return

        self._exit_stack = AsyncExitStack()

        if self.command:
            full_command = self.command + self.args
            server_params = StdioServerParameters(
                command=full_command[0],
                args=full_command[1:],
                env=self.env,
            )
            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
        elif self.url:
            sse_transport = await self._exit_stack.enter_async_context(
                sse_client(self.url)
            )
            read_stream, write_stream = sse_transport
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
        else:
            raise ValueError("MCPTools requires either 'command' (stdio) or 'url' (SSE)")

        await self._session.initialize()

        # Discover tools
        tools_result = await self._session.list_tools()
        self._tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {"type": "object", "properties": {}},
            }
            for tool in tools_result.tools
        ]
        logger.info(f"MCP '{self.name}': discovered {len(self._tools)} tools")

    async def close(self) -> None:
        """Close the connection."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            self._tools = []

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Get tool definitions in provider-neutral format."""
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool and return the result as a string."""
        if not self._session:
            raise RuntimeError(f"MCP '{self.name}' is not connected. Call connect() first.")

        result = await self._session.call_tool(name, arguments)

        # Combine all content blocks into a string
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

    Usage:
        registry = MCPRegistry()
        registry.add(MCPTools(name="fs", command=["npx", "-y", "@anthropic/mcp-filesystem", "/data"]))
        registry.add(MCPTools(name="search", url="http://localhost:8080/sse"))

        await registry.connect_all()
        tools = registry.all_tools()       # Flat list of all tool definitions
        result = await registry.call("tool_name", {"arg": "val"})
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
            except Exception as e:
                logger.error(f"Failed to connect MCP '{server.name}': {e}")

    async def close_all(self) -> None:
        """Close all connections."""
        for server in self._servers:
            try:
                await server.close()
            except Exception as e:
                logger.error(f"Failed to close MCP '{server.name}': {e}")
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
        return await server.call_tool(name, arguments)

    @classmethod
    def from_config(cls, mcp_config: list[dict]) -> MCPRegistry:
        """Build registry from config list.

        Example config:
            [
                {"name": "fs", "command": ["npx", "-y", "@anthropic/mcp-filesystem"], "args": ["/data"]},
                {"name": "search", "url": "http://localhost:8080/sse"},
            ]
        """
        registry = cls()
        for entry in mcp_config:
            registry.add(MCPTools(
                name=entry.get("name", ""),
                command=entry.get("command"),
                args=entry.get("args"),
                url=entry.get("url"),
                env=entry.get("env"),
            ))
        return registry

    def __repr__(self) -> str:
        return f"MCPRegistry(servers={len(self._servers)}, tools={len(self._tool_map)})"
