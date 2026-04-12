"""Claude model via the Claude Agent SDK with a single persistent subprocess.

Uses ONE shared ``ClaudeSDKClient`` subprocess for all sessions. Session
isolation is achieved by passing unique ``session_id`` values to the SDK's
``query()`` method — the SDK routes each session to a separate conversation
thread within the same subprocess.

This avoids the OOM problem of spawning one subprocess (+ ~18 MCP child
processes) per session. A single subprocess uses ~300-500 MB regardless of
how many sessions are active.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from openagent.models.base import BaseModel, ModelResponse, ToolCall

logger = logging.getLogger(__name__)


class ClaudeCLI(BaseModel):
    """Claude backed by a single persistent ``ClaudeSDKClient`` subprocess."""

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
        self._client: Any | None = None
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

    # ── persistent client ──

    async def _ensure_connected(self, system: str | None) -> None:
        async with self._lock:
            if self._client is not None:
                return
            from claude_agent_sdk import ClaudeSDKClient
            logger.info("Starting persistent Claude client...")
            client = ClaudeSDKClient(options=self._build_options(system=system))
            try:
                await client.connect()
            except Exception:
                logger.exception("ClaudeSDKClient.connect() failed")
                raise
            self._client = client
            logger.info("Persistent Claude client connected.")

    async def _reconnect(self, system: str | None) -> None:
        async with self._lock:
            old = self._client
            self._client = None
        if old:
            try:
                await old.disconnect()
            except Exception as e:
                logger.debug("Old client disconnect: %s", e)
        await self._ensure_connected(system)

    async def shutdown(self) -> None:
        async with self._lock:
            client = self._client
            self._client = None
        if client:
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("Client disconnect: %s", e)

    # ── execution ──

    async def _run_once(
        self,
        prompt: str,
        session_id: str,
        on_status: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        from claude_agent_sdk import AssistantMessage, ResultMessage
        assert self._client is not None

        # Pass session_id to the SDK — this is how the SDK isolates
        # conversations within the same persistent subprocess.
        await self._client.query(prompt, session_id=session_id)

        result_text = ""
        async for message in self._client.receive_response():
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

        await self._ensure_connected(system)

        for attempt in range(2):
            try:
                result = await self._run_once(prompt, sid, on_status)
                return ModelResponse(content=result)
            except BaseException as e:
                logger.error("Session %s error (attempt %d): %s", sid[-12:], attempt + 1, e)
                if attempt == 0:
                    logger.info("Reconnecting Claude client...")
                    try:
                        await self._reconnect(system)
                    except Exception as e2:
                        logger.error("Reconnect failed: %s", e2)
                        return ModelResponse(content=f"Error: {e}")
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

        await self._ensure_connected(system)
        try:
            assert self._client is not None
            await self._client.query(prompt, session_id=sid)
            async for message in self._client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in (message.content or []):
                        if hasattr(block, "text"):
                            yield block.text
                elif isinstance(message, ResultMessage):
                    if message.result:
                        yield message.result
        except BaseException as e:
            logger.error("Stream error session %s: %s", sid, e)
            await self._reconnect(system)
            yield f"Error: {e}"
