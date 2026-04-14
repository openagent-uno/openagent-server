"""MCP client: connect to any MCP server (local or remote), list tools, call them.

Configure MCP servers once in openagent.yaml, they get injected into all models.
Includes default MCPs (filesystem, fetch, shell, computer-control) that are always
loaded unless explicitly disabled. User MCPs are merged on top.
"""

from __future__ import annotations

import logging
import asyncio
import shutil
import subprocess
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import anyio
from anyio.streams.text import TextReceiveStream
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.sse import sse_client
from mcp.client.stdio import (
    PROCESS_TERMINATION_TIMEOUT,
    _create_platform_compatible_process,
    _get_executable_command,
    _terminate_process_tree,
    get_default_environment,
)
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.message import SessionMessage

from openagent.core.logging import elog

logger = logging.getLogger(__name__)

# ── Built-in MCPs (custom, ship under mcps/) ──

from openagent._frozen import is_frozen, bundle_dir

if is_frozen():
    BUILTIN_MCPS_DIR = bundle_dir() / "openagent" / "mcps"
else:
    BUILTIN_MCPS_DIR = Path(__file__).resolve().parent.parent / "mcps"

BUILTIN_MCP_SPECS: dict[str, dict[str, Any]] = {
    "computer-control": {
        "dir": "computer-control",
        "command": ["node", "dist/main.js"],
        "build": ["npm", "run", "build"],
        "install": ["npm", "install"],
        "env": {"DISPLAY": ":1"},  # needed for X11 screen capture on headless VPS
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
        "env": {"NODE_TLS_REJECT_UNAUTHORIZED": "0"},  # some VPS lack updated CA certs
    },
    "editor": {
        "dir": "editor",
        "command": ["node", "dist/index.js"],
        "build": ["npm", "run", "build"],
        "install": ["npm", "install"],
    },
    "chrome-devtools": {
        "dir": "chrome-devtools",
        "command": ["node", "node_modules/chrome-devtools-mcp/build/src/bin/chrome-devtools-mcp.js"],
        "install": ["npm", "install"],
    },
    "messaging": {
        "dir": "messaging",
        "command": ["node", "dist/index.js"],
        "build": ["npm", "run", "build"],
        "install": ["npm", "install"],
    },
    # Python MCP: inspect/create/update/delete OpenAgent's own scheduled
    # tasks from inside the agent loop. Shares the same SQLite DB as
    # openagent.scheduler.Scheduler via the OPENAGENT_DB_PATH env var.
    "scheduler": {
        "dir": "scheduler",
        "command": ["python", "-m", "openagent.mcps.scheduler.server"],
        "python": True,
    },
}

# ── Default MCPs (always injected unless disabled) ──
# Order: defaults first, then user MCPs (like { ...defaults, ...userConfig })

DEFAULT_MCPS: list[dict[str, Any]] = [
    # MCPVault: Obsidian-compatible knowledge base (search, read, write .md files)
    # The vault path is set at runtime from memory.knowledge_dir config (default: ./memories)
    {
        "name": "vault",
        "command": ["npx", "-y", "@bitbonsai/mcpvault@latest"],
        "args": [],  # populated at runtime with vault path
        "_default": True,
    },
    # Official MCP: filesystem read/write/list/search (Node, cross-platform)
    {
        "name": "filesystem",
        "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem"],
        "args": [],  # populated at runtime with home dir
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
        "builtin": "computer-control",
        "_default": True,
    },
    # Chrome DevTools MCP: browser automation, performance, DOM inspection
    {
        "builtin": "chrome-devtools",
        "_default": True,
    },
    # Bundled MCP: proactive messaging (Telegram, Discord, WhatsApp send)
    # Auto-detects available tokens from channel config env vars
    {
        "builtin": "messaging",
        "_default": True,
    },
    # Bundled MCP: read/create/update/delete the agent's own scheduled tasks.
    # The OPENAGENT_DB_PATH env var is injected at runtime so the MCP points
    # at the exact same SQLite file as openagent.scheduler.Scheduler.
    {
        "builtin": "scheduler",
        "_default": True,
    },
]


def _check_command_exists(cmd: str) -> bool:
    """Check if a command is available on PATH."""
    return shutil.which(cmd) is not None


def _node_version() -> tuple[int, int, int] | None:
    if not _check_command_exists("node"):
        return None
    try:
        result = subprocess.run(
            ["node", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    raw = result.stdout.strip().lstrip("v")
    parts = raw.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
    except ValueError:
        return None
    return major, minor, patch


def _node_meets_minimum(major: int, minor: int, patch: int = 0) -> bool:
    current = _node_version()
    if current is None:
        return False
    return current >= (major, minor, patch)


def _resolve_builtin(name: str, env: dict[str, str] | None = None) -> MCPTools:
    """Resolve a built-in MCP by name. Auto-installs and builds if needed."""
    if name not in BUILTIN_MCP_SPECS:
        available = ", ".join(BUILTIN_MCP_SPECS.keys())
        raise ValueError(f"Unknown built-in MCP: {name}. Available: {available}")

    spec = BUILTIN_MCP_SPECS[name]
    mcp_dir = BUILTIN_MCPS_DIR / spec["dir"]

    if not mcp_dir.exists():
        raise FileNotFoundError(f"Built-in MCP '{name}' directory not found at {mcp_dir}")

    is_python = spec.get("python", False)

    if is_python:
        # Python MCP: install deps if requirements.txt exists
        reqs = mcp_dir / "requirements.txt"
        if reqs.exists() and "install" in spec:
            # Check if deps are already installed by looking for a marker
            marker = mcp_dir / ".installed"
            if not marker.exists():
                logger.info(f"Installing built-in MCP '{name}' dependencies...")
                subprocess.run(spec["install"], cwd=mcp_dir, check=True, capture_output=True)
                marker.touch()
    else:
        # Node MCP: install + build
        node_modules = mcp_dir / "node_modules"
        if not node_modules.exists():
            logger.info(f"Installing built-in MCP '{name}'...")
            subprocess.run(spec["install"], cwd=mcp_dir, check=True, capture_output=True)

        dist_dir = mcp_dir / "dist"
        if not dist_dir.exists() and "build" in spec:
            logger.info(f"Building built-in MCP '{name}'...")
            subprocess.run(spec["build"], cwd=mcp_dir, check=True, capture_output=True)

    # Resolve the entry point path
    import sys
    cmd_list = list(spec["command"])
    if is_python and cmd_list and cmd_list[0] in ("python3", "python"):
        import os as _os2
        exe_basename = _os2.path.basename(sys.executable).lower()
        if is_frozen() or "python" not in exe_basename:
            # sys.executable is the openagent binary, not Python.
            # Use the hidden `_mcp-server <name>` subcommand instead of `python -m ...`
            cmd_list = [sys.executable, "_mcp-server", name]
        else:
            cmd_list[0] = sys.executable  # use venv Python

    # Resolve relative paths (like "dist/index.js") to absolute under mcp_dir
    # but skip already-absolute paths (like sys.executable)
    full_command = []
    for c in cmd_list:
        if "/" in c and not Path(c).is_absolute():
            full_command.append(str(mcp_dir / c))
        else:
            full_command.append(c)

    # Merge env from spec + caller
    merged_env = {**(spec.get("env") or {}), **(env or {})}

    # Python MCPs invoked via `python -m openagent.mcps.<name>.server`
    # need to find the `openagent` package on sys.path. When OpenAgent is
    # pip-installed it already is, but when running from a source checkout
    # the subprocess's cwd (the mcp dir) doesn't contain it. Prepend the
    # package parent dir to PYTHONPATH so both cases work.
    if is_python:
        import os as _os
        package_parent = str(BUILTIN_MCPS_DIR.parent.parent)
        existing_pp = merged_env.get("PYTHONPATH") or _os.environ.get("PYTHONPATH", "")
        merged_env["PYTHONPATH"] = (
            package_parent + (_os.pathsep + existing_pp if existing_pp else "")
        )

    return MCPTools(
        name=name,
        command=full_command,
        env=merged_env if merged_env else None,
        _cwd=str(mcp_dir),
    )


def _resolve_default_entry(
    entry: dict[str, Any],
    db_path: str | None = None,
) -> MCPTools | None:
    """Try to resolve a default MCP entry. Returns None if prerequisites are missing."""
    name = entry.get("name") or entry.get("builtin", "")

    if "builtin" in entry:
        spec = BUILTIN_MCP_SPECS.get(entry["builtin"])
        is_python = spec.get("python", False) if spec else False
        if not is_python and not _check_command_exists("node"):
            logger.warning(f"Skipping default MCP '{name}': Node.js not found")
            return None
        if entry["builtin"] == "chrome-devtools" and not _node_meets_minimum(22, 12, 0):
            version = _node_version()
            rendered = ".".join(map(str, version)) if version else "unknown"
            logger.warning(
                "Skipping default MCP '%s': Node 22.12.0+ required (found %s)",
                name,
                rendered,
            )
            return None

        # Per-builtin runtime env injection: the scheduler MCP needs to
        # point at the same SQLite file as the in-process Scheduler.
        extra_env: dict[str, str] = dict(entry.get("env") or {})
        if entry["builtin"] == "scheduler":
            import os as _os
            if db_path:
                extra_env["OPENAGENT_DB_PATH"] = _os.path.abspath(db_path)
            else:
                from openagent.core.paths import default_db_path
                extra_env["OPENAGENT_DB_PATH"] = str(default_db_path())

        try:
            return _resolve_builtin(entry["builtin"], env=extra_env or None)
        except Exception as e:
            logger.warning(f"Skipping default MCP '{name}': {e}")
            return None

    # External package — check if the command exists
    cmd = entry.get("command", [None])[0]
    if cmd and not _check_command_exists(cmd):
        logger.warning(f"Skipping default MCP '{name}': '{cmd}' not found")
        return None

    import os
    from openagent.core.paths import default_vault_path

    args = entry.get("args") or []
    # Expand home dir for filesystem MCP
    if name == "filesystem" and not args:
        args = [os.path.expanduser("~")]
    # Expand vault path for MCPVault (respects agent dir if set)
    if name == "vault" and not args:
        args = [str(default_vault_path())]

    return MCPTools(
        name=entry.get("name", ""),
        command=entry.get("command"),
        args=args,
        url=entry.get("url"),
        env=entry.get("env"),
    )


class _ManagedStdioTransport:
    """Explicit stdio transport lifecycle for MCP subprocesses.

    We avoid the upstream stdio async context manager here because its
    anyio cancel scopes are sensitive to cross-task shutdown on asyncio.
    Keeping the transport lifecycle explicit makes stop() predictable.
    """

    def __init__(self, server: StdioServerParameters):
        self.server = server
        self.process: Any | None = None
        self._task_group: anyio.abc.TaskGroup | None = None
        self.read_stream = None
        self.write_stream = None
        self._read_stream_writer = None
        self._write_stream_reader = None

    async def start(self):
        self._read_stream_writer, self.read_stream = anyio.create_memory_object_stream(0)
        self.write_stream, self._write_stream_reader = anyio.create_memory_object_stream(0)

        command = _get_executable_command(self.server.command)
        env = (
            {**get_default_environment(), **self.server.env}
            if self.server.env is not None
            else get_default_environment()
        )
        self.process = await _create_platform_compatible_process(
            command=command,
            args=self.server.args,
            env=env,
            cwd=self.server.cwd,
        )
        self._task_group = anyio.create_task_group()
        await self._task_group.__aenter__()
        self._task_group.start_soon(self._stdout_reader)
        self._task_group.start_soon(self._stdin_writer)
        return self.read_stream, self.write_stream

    async def _stdout_reader(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        assert self._read_stream_writer is not None
        try:
            async with self._read_stream_writer:
                buffer = ""
                async for chunk in TextReceiveStream(
                    self.process.stdout,
                    encoding=self.server.encoding,
                    errors=self.server.encoding_error_handler,
                ):
                    lines = (buffer + chunk).split("\n")
                    buffer = lines.pop()
                    for line in lines:
                        try:
                            message = types.JSONRPCMessage.model_validate_json(line)
                        except Exception as exc:
                            logger.exception("Failed to parse JSONRPC message from server")
                            await self._read_stream_writer.send(exc)
                            continue
                        await self._read_stream_writer.send(SessionMessage(message))
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async def _stdin_writer(self) -> None:
        assert self.process is not None and self.process.stdin is not None
        assert self._write_stream_reader is not None
        try:
            async with self._write_stream_reader:
                async for session_message in self._write_stream_reader:
                    payload = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                    await self.process.stdin.send(
                        (payload + "\n").encode(
                            encoding=self.server.encoding,
                            errors=self.server.encoding_error_handler,
                        )
                    )
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async def aclose(self) -> None:
        process = self.process
        task_group = self._task_group
        read_stream = self.read_stream
        write_stream = self.write_stream
        read_writer = self._read_stream_writer
        write_reader = self._write_stream_reader

        self.process = None
        self._task_group = None
        self.read_stream = None
        self.write_stream = None
        self._read_stream_writer = None
        self._write_stream_reader = None

        try:
            if process and process.stdin:
                try:
                    await process.stdin.aclose()
                except Exception:
                    pass
        finally:
            if task_group is not None:
                task_group.cancel_scope.cancel()
                try:
                    await task_group.__aexit__(None, None, None)
                except Exception:
                    pass
            if process is not None and process.returncode is None:
                try:
                    await _terminate_process_tree(process, timeout_seconds=1.0)
                except Exception:
                    try:
                        process.kill()
                    except Exception:
                        pass
            for stream in (read_stream, write_stream, read_writer, write_reader):
                if stream is None:
                    continue
                try:
                    await stream.aclose()
                except Exception:
                    pass
            if process is not None:
                try:
                    await process.aclose()
                except Exception:
                    pass


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
        self._stdio_transport: _ManagedStdioTransport | None = None
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
            self._stdio_transport = _ManagedStdioTransport(server_params)
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

                server = _resolve_default_entry(default_entry, db_path=db_path)
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
                    headers=entry.get("headers"),
                    oauth=entry.get("oauth", False),
                ))

        return registry

    def __repr__(self) -> str:
        return f"MCPRegistry(servers={len(self._servers)}, tools={len(self._tool_map)})"
