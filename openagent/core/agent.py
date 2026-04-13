"""Core Agent class: orchestrates model, MCP tools, and memory."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Callable, Awaitable

from openagent.channels.base import build_attachment_context, prepend_context_block
from openagent.models.base import BaseModel, ModelResponse
from openagent.memory.db import MemoryDB
from openagent.mcp.client import MCPRegistry, MCPTools
from openagent.core.prompts import FRAMEWORK_SYSTEM_PROMPT
from openagent.models.runtime import wire_model_runtime

from openagent.core.logging import elog

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 10

# Status callback type: async def on_status(status: str) -> None
StatusCallback = Callable[[str], Awaitable[None]]


class Agent:
    """Main agent class. Ties together a model, MCP tools, and memory.

    OpenAgent owns the MCP topology and the orchestration loop.
    Chat history can be caller-managed, platform-managed, or provider-managed
    depending on the active model's ``history_mode``.

    Long-term memory lives in the Obsidian-style vault exposed through MCP.
    The SQLite database is used for runtime state such as scheduler tasks,
    platform-managed chat sessions, and usage tracking.

    Usage:
        agent = Agent(
            name="assistant",
            model=AgnoProvider(model="anthropic:claude-sonnet-4-20250514"),
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

        # Runtime DB; the long-term knowledge base still lives in the Obsidian vault via MCP.
        if isinstance(memory, str):
            self._db = MemoryDB(memory)
        elif isinstance(memory, MemoryDB):
            self._db = memory
        else:
            self._db = None

        self._initialized = False
        self._idle_cleanup_task: asyncio.Task | None = None
        self._runtime_models: list[BaseModel] = []

    def _register_runtime_model(self, model: BaseModel | None) -> None:
        """Track every model instance that may need lifecycle management."""
        if model is None:
            return
        if any(existing is model for existing in self._runtime_models):
            return
        self._runtime_models.append(model)

    def _prepare_model_runtime(self, model: BaseModel | None) -> None:
        """Wire shared runtime dependencies into models that support them."""
        if model is None:
            return
        self._register_runtime_model(model)
        wire_model_runtime(
            model,
            mcp_registry=self._mcp,
            mcp_servers=self._build_mcp_server_configs() or None,
        )

    def _ensure_idle_cleanup_task(self) -> None:
        """Start the idle cleanup loop if any runtime model supports it."""
        if self._idle_cleanup_task and not self._idle_cleanup_task.done():
            return
        if any(callable(getattr(model, "cleanup_idle", None)) for model in self._runtime_models):
            self._idle_cleanup_task = asyncio.create_task(self._run_idle_cleanup())

    async def initialize(self) -> None:
        """Connect MCP servers and initialize memory DB."""
        if self._initialized:
            return
        elog("agent.initialize.start", agent=self.name, model_class=type(self.model).__name__)
        await self._mcp.connect_all()
        if self._db:
            await self._db.connect()

        self._prepare_model_runtime(self.model)
        self._ensure_idle_cleanup_task()

        self._initialized = True
        elog(
            "agent.initialize.done",
            agent=self.name,
            model_class=type(self.model).__name__,
            mcp_servers=len(self._mcp._servers),
            tools=len(self._mcp.all_tools()),
            has_db=bool(self._db),
        )

    async def _run_idle_cleanup(self) -> None:
        """Periodically release idle provider resources."""
        while True:
            await asyncio.sleep(60)
            for model in list(self._runtime_models):
                cleanup_idle = getattr(model, "cleanup_idle", None)
                if not callable(cleanup_idle):
                    continue
                try:
                    await cleanup_idle()
                except Exception as e:
                    logger.debug("Idle cleanup error: %s", e)

    def _build_mcp_server_configs(self) -> dict[str, dict]:
        """Build MCP server configs for runtime models that consume them.

        Supports stdio (command), SSE (url), and HTTP (url) MCP servers.

        Claude CLI silently drops stdio MCP servers whose `command` cannot
        be resolved in the subprocess's PATH. When OpenAgent runs under a
        systemd unit, `$PATH` is minimal and relative names like `firebase`
        or `github-mcp-server` stop resolving — even when `shutil.which()`
        finds them at process startup. To avoid this footgun, every stdio
        command is resolved to an absolute path here before being handed
        off to the SDK.
        """
        import os
        import shutil

        configs = {}
        for server in self._mcp._servers:
            if server.command:
                full_cmd = server.command + server.args
                cmd_name = full_cmd[0]

                # Resolve to absolute path so Claude CLI doesn't silently
                # drop the server when its minimal PATH can't find it.
                if not os.path.isabs(cmd_name):
                    resolved = shutil.which(cmd_name)
                    if resolved:
                        cmd_name = resolved
                    else:
                        logger.warning(
                            "MCP '%s': command %r not found on PATH — "
                            "Claude CLI will likely drop this server",
                            server.name, cmd_name,
                        )

                entry: dict = {
                    "command": cmd_name,
                    "args": full_cmd[1:],
                }

                env = dict(server.env) if server.env else {}

                if server.name == "messaging":
                    for var in ("TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "GREEN_API_ID", "GREEN_API_TOKEN"):
                        val = os.environ.get(var)
                        if val:
                            env[var] = val

                # The scheduler MCP needs to point at the same SQLite file
                # as the in-process Scheduler. _resolve_default_entry already
                # sets this when MCPRegistry was built with db_path, but fall
                # back to the agent's own DB path here so alternative wiring
                # paths (e.g. direct MCPTools instantiation) still work.
                if server.name == "scheduler" and "OPENAGENT_DB_PATH" not in env:
                    if self._db and getattr(self._db, "db_path", None):
                        env["OPENAGENT_DB_PATH"] = os.path.abspath(self._db.db_path)

                if env:
                    entry["env"] = env

                configs[server.name] = entry

            elif server.url:
                configs[server.name] = {"type": "http", "url": server.url}

        return configs

    async def shutdown(self) -> None:
        """Close all connections."""
        elog("agent.shutdown.start", agent=self.name)
        if self._idle_cleanup_task:
            self._idle_cleanup_task.cancel()
            self._idle_cleanup_task = None
        # Persistent model runtimes may need an explicit shutdown to
        # release subprocesses or cached sessions cleanly.
        seen: set[int] = set()
        for model in [self.model, *self._runtime_models]:
            if model is None or id(model) in seen:
                continue
            seen.add(id(model))
            shutdown = getattr(model, "shutdown", None)
            if callable(shutdown):
                try:
                    await shutdown()
                except Exception as e:  # noqa: BLE001
                    logger.warning("Model shutdown error: %s", e)
        await self._mcp.close_all()
        if self._db:
            await self._db.close()
        self._initialized = False
        self._runtime_models.clear()
        elog("agent.shutdown.done", agent=self.name)

    async def run(
        self,
        message: str,
        user_id: str = "",
        session_id: str | None = None,
        attachments: list[dict] | None = None,
        on_status: StatusCallback | None = None,
        model_override: BaseModel | None = None,
    ) -> str:
        """Run the agent with a user message. Returns the final text response.

        Args:
            session_id: Session key passed through to whichever history mode
                the active model uses.
            on_status: Optional async callback for live status updates.
                Called with status strings like "Thinking...", "Using shell_exec...", etc.
                Channels use this to update a live status message.
        """
        if not self.model:
            raise RuntimeError("No model configured. Set agent.model before calling run().")

        await self.initialize()
        self._prepare_model_runtime(model_override)
        self._ensure_idle_cleanup_task()

        async def _status(msg: str) -> None:
            if on_status:
                try:
                    await on_status(msg)
                except Exception:
                    pass

        try:
            elog(
                "agent.run.start",
                agent=self.name,
                user_id=user_id,
                session_id=session_id,
                model_class=type(model_override or self.model).__name__,
                attachments=len(attachments or []),
            )
            return await self._run_inner(message, attachments, _status, session_id=session_id, model_override=model_override)
        except BaseException as e:
            logger.error(f"Agent.run() fatal error: {e}")
            elog("agent.run.error", agent=self.name, user_id=user_id, session_id=session_id, error=str(e))
            return f"Error: {e}"

    async def _run_inner(
        self,
        message: str,
        attachments: list[dict] | None,
        _status,
        session_id: str | None = None,
        model_override: BaseModel | None = None,
    ) -> str:
        """Inner run logic, wrapped by run() for crash protection."""
        await _status("Loading context...")

        # Combine OpenAgent's framework-level guidelines with the user's
        # project-specific system prompt from openagent.yaml.
        system = self._combined_system_prompt()

        # Include local paths for attachments so the tool layer can inspect them.
        if attachments:
            files_info: list[str] = []
            for a in attachments:
                a_type = a.get("type", "file")
                a_name = a.get("filename", "")
                a_path = a.get("path", "")
                if a_path:
                    files_info.append(f"- {a_type}: {a_name} — local path: {a_path}")
                else:
                    files_info.append(f"- {a_type}: {a_name}")
            message = prepend_context_block(
                message,
                build_attachment_context(
                    files_info,
                    read_hint=(
                        "Use the Read tool (or an MCP tool) with the local path to inspect each file. "
                        "For images, Read returns the image content for you to see directly."
                    ),
                ),
            )

        messages: list[dict[str, Any]] = [{"role": "user", "content": message}]

        tools = self._mcp.all_tools() or None

        await _status("Thinking...")

        # Tool-use loop
        active_model = model_override or self.model
        response = None
        tool_calls_total = 0
        for iteration in range(MAX_TOOL_ITERATIONS):
            elog(
                "agent.run.iteration",
                agent=self.name,
                session_id=session_id,
                iteration=iteration + 1,
                model_class=type(active_model).__name__,
            )
            response = await active_model.generate(messages, system=system, tools=tools, on_status=_status, session_id=session_id)

            if response.tool_calls:
                tool_calls_total += len(response.tool_calls)
                elog(
                    "agent.run.tool_cycle",
                    agent=self.name,
                    session_id=session_id,
                    iteration=iteration + 1,
                    tool_calls=len(response.tool_calls),
                )
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
                    import json as _json
                    # Send structured tool event: running
                    await _status(_json.dumps({
                        "tool": tc.name,
                        "params": tc.arguments,
                        "status": "running",
                    }))
                    elog("tool.start", tool=tc.name, params=tc.arguments)

                    error = None
                    try:
                        result = await self._mcp.call_tool(tc.name, tc.arguments)
                    except Exception as e:
                        result = f"Error calling tool {tc.name}: {e}"
                        error = str(e)
                        logger.error(result)

                    # Send structured tool event: done/error
                    await _status(_json.dumps({
                        "tool": tc.name,
                        "status": "error" if error else "done",
                        "result": (result or "")[:500],
                        "error": error,
                    }))
                    if error:
                        elog("tool.error", tool=tc.name, error=error)
                    else:
                        elog("tool.done", tool=tc.name, result_len=len(result or ""))

                    tool_msg = {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tc.id,
                    }
                    messages.append(tool_msg)

                if iteration < MAX_TOOL_ITERATIONS - 1:
                    await _status("Thinking...")
            else:
                elog(
                    "agent.run.done",
                    agent=self.name,
                    session_id=session_id,
                    iterations=iteration + 1,
                    tool_calls=tool_calls_total,
                    response_len=len(response.content or ""),
                )
                return response.content

        elog(
            "agent.run.max_iterations",
            agent=self.name,
            session_id=session_id,
            iterations=MAX_TOOL_ITERATIONS,
            tool_calls=tool_calls_total,
            response_len=len(response.content or "") if response else 0,
        )
        return response.content if response else "I wasn't able to complete the request."

    def _combined_system_prompt(self) -> str:
        """Prepend the framework-level prompt to the user's system prompt."""
        user = (self.system_prompt or "").strip()
        if not user:
            return FRAMEWORK_SYSTEM_PROMPT
        return (
            FRAMEWORK_SYSTEM_PROMPT
            + "\n\n── User-specific identity and project context ──\n\n"
            + user
        )

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

        system = self._combined_system_prompt()
        messages: list[dict[str, Any]] = [{"role": "user", "content": message}]

        async for chunk in self.model.stream(messages, system=system):
            yield chunk

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.shutdown()
