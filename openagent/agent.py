"""Core Agent class: orchestrates model, MCP tools, and memory."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from openagent.models.base import BaseModel, ModelResponse
from openagent.memory.db import MemoryDB
from openagent.memory.manager import MemoryManager
from openagent.mcp.client import MCPRegistry, MCPTools

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 10


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
        auto_extract_memory: bool = True,
        history_limit: int = 50,
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.auto_extract_memory = auto_extract_memory

        # MCP
        if mcp_registry:
            self._mcp = mcp_registry
        else:
            self._mcp = MCPRegistry()
            for tool in (mcp_tools or []):
                self._mcp.add(tool)

        # Memory
        if isinstance(memory, str):
            self._db = MemoryDB(memory)
        elif isinstance(memory, MemoryDB):
            self._db = memory
        else:
            self._db = None

        self._memory: MemoryManager | None = None
        if self._db:
            self._memory = MemoryManager(self._db, auto_extract=auto_extract_memory, history_limit=history_limit)

        self._initialized = False

    async def initialize(self) -> None:
        """Connect MCP servers and initialize memory DB."""
        if self._initialized:
            return
        await self._mcp.connect_all()
        if self._db:
            await self._db.connect()
        self._initialized = True

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
    ) -> str:
        """Run the agent with a user message. Returns the final text response.

        Handles the full tool-use loop: send -> tool call -> execute -> send result -> repeat.
        """
        if not self.model:
            raise RuntimeError("No model configured. Set agent.model before calling run().")

        await self.initialize()

        # Session + history
        current_session_id = None
        history: list[dict[str, Any]] = []
        system = self.system_prompt

        if self._memory:
            current_session_id = await self._memory.ensure_session(self.name, user_id, session_id)
            history = await self._memory.get_history(current_session_id)

            # Inject long-term memories into system prompt
            mem_context = await self._memory.build_memory_context(self.name, user_id)
            if mem_context:
                system = f"{system}\n\n{mem_context}"

        # Build messages
        messages = list(history)
        messages.append({"role": "user", "content": message})

        # Store user message
        if self._memory and current_session_id:
            await self._memory.store_message(current_session_id, "user", message)

        # Get available tools
        tools = self._mcp.all_tools() or None

        # Tool-use loop
        for _ in range(MAX_TOOL_ITERATIONS):
            response = await self.model.generate(messages, system=system, tools=tools)

            if response.tool_calls:
                # Add assistant message with tool calls
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

                # Execute each tool call
                for tc in response.tool_calls:
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
            else:
                # No tool calls — we have the final response
                if self._memory and current_session_id:
                    await self._memory.store_message(current_session_id, "assistant", response.content)

                # Extract memories in background
                if self._memory and self.auto_extract_memory:
                    try:
                        await self._memory.extract_and_store_memories(
                            self.model, self.name, user_id, messages,
                        )
                    except Exception as e:
                        logger.warning(f"Memory extraction failed: {e}")

                return response.content

        # Exceeded max iterations
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
