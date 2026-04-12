"""Claude model via the Claude Agent SDK with session resume.

Uses persistent ``ClaudeSDKClient`` instances with lazy lifecycle:
- Clients are created on demand and kept alive for fast MCP access.
- Idle clients are closed after IDLE_TTL seconds to free resources.
- SDK session IDs are captured from ResultMessage and passed as
  ``resume`` when creating new clients, so conversation history
  survives subprocess restarts (the SDK persists sessions to disk).
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import time
from typing import Any, AsyncIterator, Awaitable, Callable

from openagent.models.base import BaseModel, ModelResponse

from openagent.core.logging import elog

logger = logging.getLogger(__name__)

RECEIVE_TIMEOUT = 300  # seconds per query
IDLE_TTL = 600  # seconds — close idle clients after 10 min


class ClaudeCLI(BaseModel):
    """Claude backed by ``ClaudeSDKClient`` with lazy lifecycle and session resume."""

    manages_history = True

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
        self._clients: dict[str, Any] = {}  # our_sid → ClaudeSDKClient
        self._sdk_sessions: dict[str, str] = {}  # our_sid → sdk_session_id
        self._last_active: dict[str, float] = {}  # our_sid → timestamp
        self._lock = asyncio.Lock()

    def set_mcp_servers(self, servers: dict[str, dict]) -> None:
        self.mcp_servers = servers

    def _build_options(self, system: str | None = None, session_id: str | None = None) -> Any:
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
        # Resume previous SDK session from disk if available
        if session_id:
            sdk_sid = self._sdk_sessions.get(session_id)
            if sdk_sid:
                opts["resume"] = sdk_sid
        return ClaudeAgentOptions(**opts)

    async def _get_client(self, session_id: str, system: str | None) -> Any:
        """Get or create a client for this session. No cap — idle cleanup handles limits."""
        async with self._lock:
            if session_id in self._clients:
                self._last_active[session_id] = time.time()
                return self._clients[session_id]

            from claude_agent_sdk import ClaudeSDKClient
            logger.info("Creating session %s (%d active)", session_id[-12:], len(self._clients) + 1)
            elog("model.session_create", session_id=session_id, pool_size=len(self._clients) + 1)
            client = ClaudeSDKClient(options=self._build_options(system=system, session_id=session_id))
            try:
                await client.connect()
            except Exception as e:
                logger.exception("ClaudeSDKClient.connect() failed for %s", session_id)
                elog("model.connect_error", session_id=session_id, error=str(e))
                raise
            self._clients[session_id] = client
            self._last_active[session_id] = time.time()
            return client

    async def _drop_client(self, session_id: str) -> None:
        """Close the subprocess but preserve the SDK session_id for resume."""
        async with self._lock:
            client = self._clients.pop(session_id, None)
            self._last_active.pop(session_id, None)
        # Don't remove from _sdk_sessions — needed for resume
        if client:
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("Drop client %s: %s", session_id, e)

    async def cleanup_idle(self) -> None:
        """Close clients idle for more than IDLE_TTL seconds."""
        now = time.time()
        to_close: list[tuple[str, Any]] = []
        async with self._lock:
            for sid, last in list(self._last_active.items()):
                if now - last > IDLE_TTL:
                    client = self._clients.pop(sid, None)
                    self._last_active.pop(sid, None)
                    if client:
                        to_close.append((sid, client))
        for sid, client in to_close:
            logger.info("Closing idle session %s", sid[-12:])
            elog("model.session_idle_close", session_id=sid)
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("Idle close %s: %s", sid, e)

    async def shutdown(self) -> None:
        async with self._lock:
            clients = dict(self._clients)
            self._clients.clear()
            self._last_active.clear()
        for sid, client in clients.items():
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("Shutdown %s: %s", sid, e)

    async def _run_once(self, client: Any, prompt: str, session_id: str, on_status: Any = None) -> str:
        from claude_agent_sdk import AssistantMessage, ResultMessage
        await client.query(prompt, session_id=session_id)
        result_text = ""
        try:
            async with asyncio.timeout(RECEIVE_TIMEOUT):
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage) and on_status:
                        for block in (message.content or []):
                            tool = getattr(block, "name", None)
                            if tool and hasattr(block, "input"):
                                try:
                                    params = getattr(block, "input", {})
                                    await on_status(_json.dumps({
                                        "tool": tool,
                                        "params": params if isinstance(params, dict) else {},
                                        "status": "running",
                                    }))
                                except Exception:
                                    pass
                    if isinstance(message, ResultMessage):
                        result_text = message.result or ""
                        # Capture SDK session ID for future resume
                        sdk_sid = getattr(message, "session_id", None)
                        if sdk_sid:
                            self._sdk_sessions[session_id] = sdk_sid
                            elog("model.session_stored", session_id=session_id, sdk_session_id=sdk_sid)
        except TimeoutError:
            logger.error("receive_response() timed out after %ds", RECEIVE_TIMEOUT)
            elog("model.timeout", session_id=session_id, timeout=RECEIVE_TIMEOUT)
            raise
        return result_text

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
    ) -> ModelResponse:
        sid = session_id or "default"
        elog("model.generate", session_id=sid)
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
            except Exception as e:
                logger.error("Session %s error (attempt %d): %s", sid[-8:], attempt + 1, e)
                # Drop the broken client — next attempt creates a fresh one
                # with resume, recovering history from disk
                await self._drop_client(sid)
                if attempt == 0:
                    continue
                # Second failure — only clear SDK session for non-timeout
                # errors (connection failures, etc.). Timeouts mean the SDK
                # session is still valid on disk; the request was just too
                # complex. Clearing it would destroy conversation history.
                if not isinstance(e, TimeoutError):
                    self._sdk_sessions.pop(sid, None)
                return ModelResponse(
                    content="I'm sorry, that request took too long to process. "
                    "Please try again with a simpler request."
                    if isinstance(e, TimeoutError)
                    else f"Error: {e}"
                )
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
                        sdk_sid = getattr(message, "session_id", None)
                        if sdk_sid:
                            self._sdk_sessions[sid] = sdk_sid
                        if message.result:
                            yield message.result
        except TimeoutError:
            logger.error("Stream timed out after %ds", RECEIVE_TIMEOUT)
            await self._drop_client(sid)
            yield "Error: request timed out"
        except Exception as e:
            logger.error("Stream error %s: %s", sid, e)
            await self._drop_client(sid)
            yield f"Error: {e}"
