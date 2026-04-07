"""MCP client: connect to any MCP server (local or remote), list tools, call them.

Configure MCP servers once in openagent.yaml, they get injected into all models.
Includes default MCPs (filesystem, fetch, shell, computer-use) that are always
loaded unless explicitly disabled. User MCPs are merged on top.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)

# ── Built-in MCPs (custom, ship under mcps/) ──

BUILTIN_MCPS_DIR = Path(__file__).resolve().parent.parent.parent / "mcps"

BUILTIN_MCP_SPECS: dict[str, dict[str, Any]] = {
    "computer-use": {
        "dir": "computer-use",
        "command": ["node", "dist/main.js"],
        "build": ["npm", "run", "build"],
        "install": ["npm", "install"],
    },
    "shell": {
        "dir": "shell",
        "command": ["node", "dist/index.js"],
        "build": ["npm", "run", "build"],
        "install": ["npm", "install"],
    },
    "web-search": {
        "dir": "web-search",
        "command": ["node", "dist/index.js"],
        "build": ["npm", "run", "build"],
        "install": ["npm", "install"],
    },
    "editor": {
        "dir": "editor",
        "command": ["node", "dist/index.js"],
        "build": ["npm", "run", "build"],
        "install": ["npm", "install"],
    },
}

# ── Default MCPs (always injected unless disabled) ──
# Order: defaults first, then user MCPs (like { ...defaults, ...userConfig })

DEFAULT_MCPS: list[dict[str, Any]] = [
    # Official MCP: filesystem read/write/list/search (Node, cross-platform)
    {
        "name": "filesystem",
        "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem"],
        "args": ["."],  # default to cwd, user can override
        "_default": True,
    },
    # Bundled MCP: surgical file editing, grep, glob
    {
        "builtin": "editor",
        "_default": True,
    },
    # Bundled MCP: web search + page content extraction, no API key needed
    # Uses Bing > Brave > DuckDuckGo with Playwright for content extraction
    {
        "builtin": "web-search",
        "_default": True,
    },
    # Custom MCP: cross-platform shell execution
    {
        "builtin": "shell",
        "_default": True,
    },
    # Custom MCP: cross-platform computer use (screenshot, mouse, keyboard)
    {
        "builtin": "computer-use",
        "_default": True,
    },
]


def _check_command_exists(cmd: str) -> bool:
    """Check if a command is available on PATH."""
    return shutil.which(cmd) is not None


def _resolve_builtin(name: str, env: dict[str, str] | None = None) -> MCPTools:
    """Resolve a built-in MCP by name. Auto-installs and builds if needed."""
    if name not in BUILTIN_MCP_SPECS:
        available = ", ".join(BUILTIN_MCP_SPECS.keys())
        raise ValueError(f"Unknown built-in MCP: {name}. Available: {available}")

    spec = BUILTIN_MCP_SPECS[name]
    mcp_dir = BUILTIN_MCPS_DIR / spec["dir"]

    if not mcp_dir.exists():
        raise FileNotFoundError(f"Built-in MCP '{name}' directory not found at {mcp_dir}")

    # Auto-install if node_modules missing
    node_modules = mcp_dir / "node_modules"
    if not node_modules.exists():
        logger.info(f"Installing built-in MCP '{name}'...")
        subprocess.run(spec["install"], cwd=mcp_dir, check=True, capture_output=True)

    # Auto-build if dist missing
    dist_dir = mcp_dir / "dist"
    if not dist_dir.exists():
        logger.info(f"Building built-in MCP '{name}'...")
        subprocess.run(spec["build"], cwd=mcp_dir, check=True, capture_output=True)

    # Resolve the entry point path
    full_command = [str(mcp_dir / c) if "/" in c else c for c in spec["command"]]

    return MCPTools(
        name=name,
        command=full_command,
        env=env,
        _cwd=str(mcp_dir),
    )


def _resolve_default_entry(entry: dict[str, Any]) -> MCPTools | None:
    """Try to resolve a default MCP entry. Returns None if prerequisites are missing."""
    name = entry.get("name") or entry.get("builtin", "")

    if "builtin" in entry:
        # Custom built-in — needs Node.js
        if not _check_command_exists("node"):
            logger.warning(f"Skipping default MCP '{name}': Node.js not found")
            return None
        try:
            return _resolve_builtin(entry["builtin"], env=entry.get("env"))
        except Exception as e:
            logger.warning(f"Skipping default MCP '{name}': {e}")
            return None

    # External package — check if the command exists
    cmd = entry.get("command", [None])[0]
    if cmd and not _check_command_exists(cmd):
        logger.warning(f"Skipping default MCP '{name}': '{cmd}' not found")
        return None

    return MCPTools(
        name=entry.get("name", ""),
        command=entry.get("command"),
        args=entry.get("args"),
        url=entry.get("url"),
        env=entry.get("env"),
    )


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
        _cwd: str | None = None,
    ):
        self.name = name or (command[0] if command else url or "mcp")
        self.command = command
        self.args = args or []
        self.url = url
        self.env = env
        self._cwd = _cwd

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
            env = self.env or {}
            if self._cwd:
                env = {**env, "CWD": self._cwd}
            server_params = StdioServerParameters(
                command=full_command[0],
                args=full_command[1:],
                env=env if env else None,
                cwd=self._cwd,
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

    Default MCPs (filesystem, fetch, shell, computer-use) are always loaded
    unless disabled. User-configured MCPs are merged on top:

        { ...default_mcps, ...user_mcps }

    Disable defaults:
        - In YAML: set `mcp_defaults: false`
        - In code: `MCPRegistry.from_config(config, include_defaults=False)`
        - Disable specific ones: `mcp_disable: ["computer-use", "fetch"]`
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
    def from_config(
        cls,
        mcp_config: list[dict] | None = None,
        include_defaults: bool = True,
        disable: list[str] | None = None,
    ) -> MCPRegistry:
        """Build registry: defaults first, then user MCPs merged on top.

        Args:
            mcp_config: User-configured MCP entries from openagent.yaml
            include_defaults: Whether to include default MCPs (filesystem, fetch, shell, computer-use)
            disable: List of default MCP names to skip (e.g. ["computer-use", "fetch"])
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

                server = _resolve_default_entry(default_entry)
                if server:
                    registry.add(server)

        # 2. Load user MCPs on top
        for entry in (mcp_config or []):
            if "builtin" in entry:
                try:
                    server = _resolve_builtin(entry["builtin"], env=entry.get("env"))
                    registry.add(server)
                except Exception as e:
                    logger.error(f"Failed to load built-in MCP '{entry['builtin']}': {e}")
            else:
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
