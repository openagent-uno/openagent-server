"""Core Agent class: orchestrates model, MCP tools, and memory."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Callable, Awaitable

from openagent.models.base import BaseModel, ModelResponse
from openagent.memory.db import MemoryDB
from openagent.mcp.client import MCPRegistry, MCPTools

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 10

# Status callback type: async def on_status(status: str) -> None
StatusCallback = Callable[[str], Awaitable[None]]


class Agent:
    """Main agent class. Ties together a model, MCP tools, and memory.

    Session history is handled by the Claude Agent SDK (resume=session_id).
    Long-term memory is handled by MCPVault (Obsidian vault).
    SQLite (MemoryDB) is only used for scheduled tasks.

    Usage:
        agent = Agent(
            name="assistant",
            model=ClaudeAPI(model="claude-sonnet-4-6"),
            system_prompt="You are a helpful assistant.",
            mcp_tools=[MCPTools(command=["npx", "..."])],
            memory=MemoryDB("agent.db"),
        )
        response = await agent.run("Hello!", user_id="user-1")
    """

    def __init__(
        self,
        name: str = "agent",
        model: BaseModel | None = None,
        system_prompt: str = "You are a helpful assistant.",
        mcp_tools: list[MCPTools] | None = None,
        mcp_registry: MCPRegistry | None = None,
        memory: MemoryDB | str | None = None,
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt

        # MCP
        if mcp_registry:
            self._mcp = mcp_registry
        else:
            self._mcp = MCPRegistry()
            for tool in (mcp_tools or []):
                self._mcp.add(tool)

        # Memory (SQLite for scheduled tasks only; knowledge base in Obsidian vault via MCPVault)
        if isinstance(memory, str):
            self._db = MemoryDB(memory)
        elif isinstance(memory, MemoryDB):
            self._db = memory
        else:
            self._db = None

        self._initialized = False

    async def initialize(self) -> None:
        """Connect MCP servers and initialize memory DB."""
        if self._initialized:
            return
        await self._mcp.connect_all()
        if self._db:
            await self._db.connect()

        # For Claude CLI: pass MCP server configs
        from openagent.models.claude_cli import ClaudeCLI
        if isinstance(self.model, ClaudeCLI):
            mcp_configs = self._build_cli_mcp_configs()
            if mcp_configs:
                self.model.set_mcp_servers(mcp_configs)

        self._initialized = True

    def _build_cli_mcp_configs(self) -> dict[str, dict]:
        """Build MCP server configs for the Claude Agent SDK.

        Supports stdio (command), SSE (url), and HTTP (url) MCP servers.
        """
        import os
        configs = {}
        for server in self._mcp._servers:
            if server.command:
                full_cmd = server.command + server.args
                entry: dict = {
                    "command": full_cmd[0],
                    "args": full_cmd[1:],
                }

                env = dict(server.env) if server.env else {}

                if server.name == "messaging":
                    for var in ("TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "GREEN_API_ID", "GREEN_API_TOKEN"):
                        val = os.environ.get(var)
                        if val:
                            env[var] = val

                if env:
                    entry["env"] = env

                configs[server.name] = entry

            elif server.url:
                configs[server.name] = {"type": "http", "url": server.url}

        return configs

    async def shutdown(self) -> None:
        """Close all connections."""
        await self._mcp.close_all()
        if self._db:
            await self._db.close()
        self._initialized = False

    async def run(
        self,
        message: str,
        user_id: str = "",
        session_id: str | None = None,
        attachments: list[dict] | None = None,
        on_status: StatusCallback | None = None,
    ) -> str:
        """Run the agent with a user message. Returns the final text response.

        Args:
            session_id: Kept for API compatibility. Session continuity is
                handled by the Claude Agent SDK's resume=session_id.
            on_status: Optional async callback for live status updates.
                Called with status strings like "Thinking...", "Using shell_exec...", etc.
                Channels use this to update a live status message.
        """
        if not self.model:
            raise RuntimeError("No model configured. Set agent.model before calling run().")

        await self.initialize()

        async def _status(msg: str) -> None:
            if on_status:
                try:
                    await on_status(msg)
                except Exception:
                    pass

        try:
            return await self._run_inner(message, attachments, _status)
        except BaseException as e:
            logger.error(f"Agent.run() fatal error: {e}")
            return f"Error: {e}"

    async def _run_inner(
        self,
        message: str,
        attachments: list[dict] | None,
        _status,
    ) -> str:
        """Inner run logic, wrapped by run() for crash protection."""
        await _status("Loading context...")

        system = self.system_prompt

        # Build messages
        if attachments:
            att_desc = " ".join(f"[Attached {a.get('type','file')}: {a.get('filename','')}]" for a in attachments)
            message = f"{att_desc}\n{message}" if message else att_desc

        messages: list[dict[str, Any]] = [{"role": "user", "content": message}]

        tools = self._mcp.all_tools() or None

        await _status("Thinking...")

        # Tool-use loop
        response = None
        for iteration in range(MAX_TOOL_ITERATIONS):
            response = await self.model.generate(messages, system=system, tools=tools, on_status=_status)

            if response.tool_calls:
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                }
                messages.append(assistant_msg)

                for tc in response.tool_calls:
                    await _status(f"Using {tc.name}...")

                    try:
                        result = await self._mcp.call_tool(tc.name, tc.arguments)
                    except Exception as e:
                        result = f"Error calling tool {tc.name}: {e}"
                        logger.error(result)

                    tool_msg = {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tc.id,
                    }
                    messages.append(tool_msg)

                if iteration < MAX_TOOL_ITERATIONS - 1:
                    await _status("Thinking...")
            else:
                return response.content

        return response.content if response else "I wasn't able to complete the request."

    async def stream_run(
        self,
        message: str,
        user_id: str = "",
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream the agent's response. Does not support tool use in streaming mode."""
        if not self.model:
            raise RuntimeError("No model configured.")

        await self.initialize()

        system = self.system_prompt
        messages: list[dict[str, Any]] = [{"role": "user", "content": message}]

        async for chunk in self.model.stream(messages, system=system):
            yield chunk

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.shutdown()
