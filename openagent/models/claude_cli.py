"""Claude model via the Claude Agent SDK.

Uses a single persistent ``ClaudeSDKClient`` subprocess at a time.
When the session_id changes, the client is disconnected and a fresh
one is connected — this guarantees conversation isolation without
spawning multiple subprocesses simultaneously (which caused OOM).

The subprocess is reused for consecutive messages within the same
session (supporting multi-turn tool use). When a different session
sends a message, the current subprocess is recycled.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from openagent.models.base import BaseModel, ModelResponse, ToolCall

logger = logging.getLogger(__name__)

RECEIVE_TIMEOUT = 300  # seconds — generous to allow long tool runs


class ClaudeCLI(BaseModel):
    """Claude backed by a recycled ``ClaudeSDKClient`` subprocess."""

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
        self._current_session: str | None = None
        self._lock = asyncio.Lock()
        self._system: str | None = None

    def set_mcp_servers(self, servers: dict[str, dict]) -> None:
        self.mcp_servers = servers

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

    async def _get_client(self, session_id: str, system: str | None) -> Any:
        """Return a client for this session. Recycles if session changed."""
        async with self._lock:
            # Same session → reuse existing client (multi-turn)
            if self._client is not None and self._current_session == session_id:
                return self._client

            # Different session → disconnect old, connect new
            if self._client is not None:
                logger.info("Recycling Claude client (session %s → %s)",
                            (self._current_session or "?")[-8:], session_id[-8:])
                try:
                    await self._client.disconnect()
                except Exception as e:
                    logger.debug("Disconnect: %s", e)
                self._client = None

            from claude_agent_sdk import ClaudeSDKClient
            logger.info("Connecting Claude client for session %s", session_id[-8:])
            client = ClaudeSDKClient(options=self._build_options(system=system))
            try:
                await client.connect()
            except Exception:
                logger.exception("ClaudeSDKClient.connect() failed")
                raise
            self._client = client
            self._current_session = session_id
            self._system = system
            return client

    async def shutdown(self) -> None:
        async with self._lock:
            client = self._client
            self._client = None
            self._current_session = None
        if client:
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("Shutdown disconnect: %s", e)

    async def _run_once(self, client, prompt, session_id, on_status=None):
        from claude_agent_sdk import AssistantMessage, ResultMessage
        await client.query(prompt, session_id=session_id)
        result_text = ""
        try:
            async with asyncio.timeout(RECEIVE_TIMEOUT):
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
        except TimeoutError:
            logger.error("receive_response() timed out after %ds — forcing reconnect", RECEIVE_TIMEOUT)
            raise
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
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"[Previous assistant response] {content}")
        prompt = "\n\n".join(prompt_parts)

        for attempt in range(2):
            try:
                client = await self._get_client(sid, system)
                result = await self._run_once(client, prompt, sid, on_status)
                return ModelResponse(content=result)
            except BaseException as e:
                logger.error("Session %s error (attempt %d): %s", sid[-8:], attempt + 1, e)
                if attempt == 0:
                    async with self._lock:
                        self._client = None
                        self._current_session = None
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
        try:
            client = await self._get_client(sid, system)
            await client.query(prompt, session_id=sid)
            async with asyncio.timeout(RECEIVE_TIMEOUT):
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in (message.content or []):
                            if hasattr(block, "text"):
                                yield block.text
                    elif isinstance(message, ResultMessage):
                        if message.result:
                            yield message.result
        except TimeoutError:
            logger.error("receive_response() timed out after %ds — forcing reconnect", RECEIVE_TIMEOUT)
            await self._drop_session(sid)
            yield "Error: receive_response() timed out"
        except BaseException as e:
            logger.error("Stream error: %s", e)
            async with self._lock:
                self._client = None
                self._current_session = None
            yield f"Error: {e}"
