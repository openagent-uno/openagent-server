"""Core Agent class: orchestrates model, MCP tools, and memory."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Callable, Awaitable

from openagent.models.base import BaseModel, ModelResponse
from openagent.memory.db import MemoryDB
from openagent.memory.manager import MemoryManager
from openagent.mcp.client import MCPRegistry, MCPTools

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 10

# Status callback type: async def on_status(status: str) -> None
StatusCallback = Callable[[str], Awaitable[None]]


class Agent:
    """Main agent class. Ties together a model, MCP tools, and memory.

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
        history_limit: int = 50,
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

        # Memory (SQLite for sessions + tasks only; knowledge base in Obsidian vault via MCPVault)
        if isinstance(memory, str):
            self._db = MemoryDB(memory)
        elif isinstance(memory, MemoryDB):
            self._db = memory
        else:
            self._db = None

        self._memory: MemoryManager | None = None
        if self._db:
            self._memory = MemoryManager(self._db, history_limit=history_limit)

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

        All OpenAgent MCPs are passed so every model gets the same tools.
        Format: {"name": {"command": "...", "args": [...], "env": {...}}}
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

                # For messaging MCP: inject channel tokens from os.environ
                if server.name == "messaging":
                    for var in ("TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "GREEN_API_ID", "GREEN_API_TOKEN"):
                        val = os.environ.get(var)
                        if val:
                            env[var] = val

                if env:
                    entry["env"] = env

                configs[server.name] = entry
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
                    pass  # never let status updates break the flow

        await _status("Loading context...")

        # Session + history
        current_session_id = None
        history: list[dict[str, Any]] = []
        system = self.system_prompt

        if self._memory:
            current_session_id = await self._memory.ensure_session(self.name, user_id, session_id)
            history = await self._memory.get_history(current_session_id)

        # Build messages
        if attachments:
            att_desc = " ".join(f"[Attached {a.get('type','file')}: {a.get('filename','')}]" for a in attachments)
            message = f"{att_desc}\n{message}" if message else att_desc

        messages = list(history)
        messages.append({"role": "user", "content": message})

        if self._memory and current_session_id:
            await self._memory.store_message(current_session_id, "user", message)

        tools = self._mcp.all_tools() or None

        await _status("Thinking...")

        # Tool-use loop
        response = None
        for iteration in range(MAX_TOOL_ITERATIONS):
            response = await self.model.generate(messages, system=system, tools=tools)

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
                if self._memory and current_session_id:
                    await self._memory.store_message(
                        current_session_id, "assistant", response.content,
                        tool_calls=assistant_msg["tool_calls"],
                    )

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
                    if self._memory and current_session_id:
                        await self._memory.store_message(
                            current_session_id, "tool", result, tool_call_id=tc.id,
                        )

                if iteration < MAX_TOOL_ITERATIONS - 1:
                    await _status("Thinking...")
            else:
                if self._memory and current_session_id:
                    await self._memory.store_message(current_session_id, "assistant", response.content)
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

        history: list[dict[str, Any]] = []
        system = self.system_prompt
        current_session_id = None

        if self._memory:
            current_session_id = await self._memory.ensure_session(self.name, user_id, session_id)
            history = await self._memory.get_history(current_session_id)
            mem_context = await self._memory.build_memory_context(self.name, user_id)
            if mem_context:
                system = f"{system}\n\n{mem_context}"

        messages = list(history)
        messages.append({"role": "user", "content": message})

        if self._memory and current_session_id:
            await self._memory.store_message(current_session_id, "user", message)

        full_response = []
        async for chunk in self.model.stream(messages, system=system):
            full_response.append(chunk)
            yield chunk

        content = "".join(full_response)
        if self._memory and current_session_id:
            await self._memory.store_message(current_session_id, "assistant", content)

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.shutdown()
