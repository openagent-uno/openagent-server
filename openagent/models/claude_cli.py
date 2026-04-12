"""Claude model via the Claude Agent SDK with bounded session pool.

Maintains up to MAX_SESSIONS persistent ``ClaudeSDKClient`` instances.
Each session_id gets its own subprocess with isolated conversation.
When the pool is full, the least-recently-used session is evicted
(disconnected) to make room.

With MAX_SESSIONS=3 and ~18 MCP processes per client, the worst case
is ~54 child processes — fits comfortably in 4 GB RAM.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import Any, AsyncIterator, Awaitable, Callable

from openagent.models.base import BaseModel, ModelResponse

from openagent.core.logging import elog

logger = logging.getLogger(__name__)

RECEIVE_TIMEOUT = 300  # seconds — generous to allow long tool runs
MAX_SESSIONS = 3  # max concurrent Claude subprocesses


class ClaudeCLI(BaseModel):
    """Claude backed by a bounded pool of ``ClaudeSDKClient`` subprocesses."""

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
        self._pool: OrderedDict[str, Any] = OrderedDict()  # session_id → client (LRU)
        self._lock = asyncio.Lock()

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
        """Get or create a client for this session. LRU eviction if pool full."""
        async with self._lock:
            # Hit: move to end (most recent)
            if session_id in self._pool:
                self._pool.move_to_end(session_id)
                return self._pool[session_id]

            # Evict oldest if at capacity
            while len(self._pool) >= MAX_SESSIONS:
                evict_sid, evict_client = self._pool.popitem(last=False)
                logger.info("Evicting session %s (pool full)", evict_sid[-12:])
                elog("model.session_evict", session_id=evict_sid)
                try:
                    await evict_client.disconnect()
                except Exception as e:
                    logger.debug("Evict disconnect: %s", e)

            # Create new
            from claude_agent_sdk import ClaudeSDKClient
            logger.info("Creating session %s (%d/%d in pool)",
                        session_id[-12:], len(self._pool) + 1, MAX_SESSIONS)
            elog("model.session_create", session_id=session_id, pool_size=len(self._pool) + 1)
            client = ClaudeSDKClient(options=self._build_options(system=system))
            try:
                await client.connect()
            except Exception as e:
                logger.exception("ClaudeSDKClient.connect() failed for %s", session_id)
                elog("model.connect_error", session_id=session_id, error=str(e))
                raise
            self._pool[session_id] = client
            return client

    async def _drop_session(self, session_id: str) -> None:
        async with self._lock:
            client = self._pool.pop(session_id, None)
        if client:
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("Drop session %s: %s", session_id, e)

    async def shutdown(self) -> None:
        async with self._lock:
            pool = dict(self._pool)
            self._pool.clear()
        for sid, client in pool.items():
            try:
                await client.disconnect()
            except Exception as e:
                logger.debug("Shutdown %s: %s", sid, e)

    async def _run_once(self, client, prompt, session_id, on_status=None):
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
                                    import json as _json
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
        except TimeoutError:
            logger.error("receive_response() timed out after %ds — forcing reconnect", RECEIVE_TIMEOUT)
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
                is_timeout = isinstance(e, TimeoutError)
                if attempt == 0:
                    if is_timeout:
                        # Client is stuck — drop it and retry with a fresh
                        # subprocess. The SDK may resume history via session_id.
                        await self._drop_session(sid)
                    # For non-timeout errors, retry on the same client
                    # (transient issue, session history still intact).
                    continue
                # Second failure — drop session and return error
                await self._drop_session(sid)
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
            logger.error("Stream error %s: %s", sid, e)
            await self._drop_session(sid)
            yield f"Error: {e}"
