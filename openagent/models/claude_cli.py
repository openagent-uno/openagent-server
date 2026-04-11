"""Claude model via the Claude Agent SDK with per-session subprocess isolation.

Each unique ``session_id`` gets its own ``ClaudeSDKClient`` instance (a
separate ``claude`` subprocess). This guarantees conversation isolation:
messages in one session never leak into another.

MCP servers are initialized once per subprocess. Idle sessions are kept
alive (the subprocess is lightweight once MCPs are loaded) and only
disconnected on agent shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from openagent.models.base import BaseModel, ModelResponse, ToolCall

logger = logging.getLogger(__name__)


class ClaudeCLI(BaseModel):
    """Claude backed by per-session ``ClaudeSDKClient`` subprocesses."""

    def __init__(
        self,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        permission_mode: str = "bypass",
        mcp_servers: dict[str, dict] | None = None,
    ):
        self.model = model
        self.allowed_tools = allowed_tools or []
        self.permission_mode = permission_mode
        self.mcp_servers: dict[str, dict] = mcp_servers or {}
        self._sessions: dict[str, Any] = {}  # session_id → ClaudeSDKClient
        self._lock = asyncio.Lock()

    def set_mcp_servers(self, servers: dict[str, dict]) -> None:
        self.mcp_servers = servers

    # ── options ──

    def _build_options(self, system: str | None = None) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions
        opts: dict[str, Any] = {}
        if self.permission_mode == "bypass":
            opts["permission_mode"] = "bypassPermissions"
        elif self.permission_mode == "auto":
            opts["permission_mode"] = "acceptEdits"
        if self.mcp_servers:
            opts["mcp_servers"] = self.mcp_servers
        if self.model:
            opts["model"] = self.model
        if system:
            opts["system_prompt"] = system
        return ClaudeAgentOptions(**opts)

    # ── per-session client pool ──

    async def _get_client(self, session_id: str, system: str | None) -> Any:
        """Return (or create) an isolated ClaudeSDKClient for this session."""
        async with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]

            from claude_agent_sdk import ClaudeSDKClient
            logger.info("Creating new Claude session: %s", session_id[-12:])
            client = ClaudeSDKClient(options=self._build_options(system=system))
            try:
                await client.connect()
            except Exception:
                logger.exception("ClaudeSDKClient.connect() failed for session %s", session_id)
                raise
            self._sessions[session_id] = client
            return client

    async def _drop_session(self, session_id: str) -> None:
        """Disconnect and remove a session's client."""
        async with self._lock:
            client = self._sessions.pop(session_id, None)
        if client:
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("Session %s disconnect: %s", session_id, e)

    async def shutdown(self) -> None:
        """Disconnect all session clients."""
        async with self._lock:
            sessions = dict(self._sessions)
            self._sessions.clear()
        for sid, client in sessions.items():
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("Shutdown session %s: %s", sid, e)

    # ── execution ──

    async def _run_once(
        self,
        client: Any,
        prompt: str,
        session_id: str,
        on_status: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        from claude_agent_sdk import AssistantMessage, ResultMessage

        await client.query(prompt, session_id=session_id)

        result_text = ""
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage) and on_status:
                for block in (message.content or []):
                    if getattr(block, "type", None) == "tool_use":
                        name = getattr(block, "name", None)
                        if name:
                            try:
                                await on_status(f"Using {name}...")
                            except Exception:
                                pass
            if isinstance(message, ResultMessage):
                result_text = message.result or ""
        return result_text

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Optional[Callable[[str], Awaitable[None]]] = None,
        session_id: str | None = None,
    ) -> ModelResponse:
        sid = session_id or "default"

        prompt_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"[Previous assistant response] {content}")
        prompt = "\n\n".join(prompt_parts)

        client = await self._get_client(sid, system)

        for attempt in range(2):
            try:
                result = await self._run_once(client, prompt, sid, on_status)
                return ModelResponse(content=result)
            except BaseException as e:
                logger.error("Session %s error (attempt %d): %s", sid[-12:], attempt + 1, e)
                if attempt == 0:
                    await self._drop_session(sid)
                    client = await self._get_client(sid, system)
                    continue
                return ModelResponse(content=f"Error: {e}")

        return ModelResponse(content="Error: max retries exceeded")

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        from claude_agent_sdk import AssistantMessage, ResultMessage

        sid = session_id or "default"
        prompt_parts = [m.get("content", "") for m in messages if m.get("role") == "user"]
        prompt = "\n\n".join(prompt_parts)

        client = await self._get_client(sid, system)
        try:
            await client.query(prompt, session_id=sid)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in (message.content or []):
                        if hasattr(block, "text"):
                            yield block.text
                elif isinstance(message, ResultMessage):
                    if message.result:
                        yield message.result
        except BaseException as e:
            logger.error("Stream error session %s: %s", sid, e)
            await self._drop_session(sid)
            yield f"Error: {e}"
