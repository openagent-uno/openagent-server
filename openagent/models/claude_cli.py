"""Claude model via the Claude Agent SDK with a persistent subprocess.

Rationale
---------

Early versions of this backend used the one-shot ``query()`` helper from
``claude_agent_sdk``. That approach spawns a new ``claude`` subprocess for
every user message, and each subprocess re-initialises all MCP servers
from scratch. Claude CLI has an internal 5 s deadline for MCP startup:

    [MCP] --mcp-config servers not ready after 5000ms — proceeding;
          background connection continues

Servers that exceed this deadline (``firebase``, ``google-analytics``,
and any other MCP with non-trivial startup) make it into the background
connection pool, but their tool definitions never land in the system
prompt of that one-shot request. To the model they simply don't exist.

With a **persistent** :class:`ClaudeSDKClient` the MCP startup cost is
paid exactly once: on the first message we ``connect()`` the client,
wait for the MCPs to come up, and then reuse the same subprocess for
every subsequent message. Latency per message also drops dramatically —
the per-request 3-5 s of MCP churn is gone.

Failure handling
----------------

- If the subprocess dies mid-conversation (crash, OOM kill, hook kills
  it) we disconnect, clear state, and reconnect on the next message.
- If a single ``query()`` raises we retry once after a forced reconnect.
- ``shutdown()`` disconnects cleanly; the :class:`AgentServer` calls it
  on SIGTERM.

Session IDs
-----------

``generate()`` still accepts a ``session_id`` from the caller but, for
now, the persistent client sends every user message under the same
underlying conversation — the session_id is passed to the SDK's
``query()`` for future-proofing but channel-level isolation already
happens via :class:`openagent.channels.queue.UserQueueManager`
(one-message-at-a-time per user). Cross-user context bleed is not a
risk in practice because channels serialize all runs through the agent.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from openagent.models.base import BaseModel, ModelResponse, ToolCall

logger = logging.getLogger(__name__)


class ClaudeCLI(BaseModel):
    """Claude backed by a persistent ``ClaudeSDKClient`` subprocess."""

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
        self._client_lock = asyncio.Lock()

    def set_mcp_servers(self, servers: dict[str, dict]) -> None:
        """Set MCP server configs. Called by Agent during initialization."""
        self.mcp_servers = servers

    # ── client lifecycle ───────────────────────────────────────────────

    def _build_options(self, system: str | None = None) -> Any:
        """Build ``ClaudeAgentOptions`` for the persistent client."""
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

    async def _ensure_connected(self, system: str | None) -> None:
        """Connect the persistent client on first use."""
        async with self._client_lock:
            if self._client is not None:
                return
            from claude_agent_sdk import ClaudeSDKClient

            logger.info("Starting persistent Claude client (MCP warmup)...")
            client = ClaudeSDKClient(options=self._build_options(system=system))
            try:
                await client.connect()
            except Exception:
                logger.exception("ClaudeSDKClient.connect() failed")
                raise
            self._client = client
            logger.info("Persistent Claude client connected.")

    async def _reconnect(self, system: str | None) -> None:
        """Disconnect, drop state, and reconnect. Called after a crash."""
        async with self._client_lock:
            old = self._client
            self._client = None
        if old is not None:
            try:
                await old.disconnect()
            except Exception as e:  # noqa: BLE001
                logger.debug("Old client disconnect raised: %s", e)
        await self._ensure_connected(system)

    async def shutdown(self) -> None:
        """Disconnect the persistent client. Called on agent shutdown."""
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            try:
                await client.disconnect()
            except Exception as e:  # noqa: BLE001
                logger.debug("Client disconnect raised: %s", e)

    # ── single-turn execution ─────────────────────────────────────────

    async def _run_once(
        self,
        prompt: str,
        session_id: str,
        on_status: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        """Send one user message through the persistent client.

        Emits status updates via ``on_status`` for every ``tool_use`` block
        and returns the final ``ResultMessage.result`` text.
        """
        from claude_agent_sdk import AssistantMessage, ResultMessage

        assert self._client is not None

        await self._client.query(prompt, session_id=session_id)

        result_text = ""
        async for message in self._client.receive_response():
            if isinstance(message, AssistantMessage) and on_status:
                for block in (message.content or []):
                    if getattr(block, "type", None) == "tool_use":
                        tool_name = getattr(block, "name", None)
                        if tool_name:
                            try:
                                await on_status(f"Using {tool_name}...")
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
    ) -> ModelResponse:
        """Generate a single response via the persistent client.

        Retries once with a fresh connection if the subprocess has died.
        """
        prompt_parts: list[str] = []
        session_id = "default"
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"[Previous assistant response] {content}")
            if msg.get("session_id"):
                session_id = msg["session_id"]
        prompt = "\n\n".join(prompt_parts)

        await self._ensure_connected(system)

        for attempt in range(2):
            try:
                result_text = await self._run_once(prompt, session_id, on_status)
                return ModelResponse(content=result_text)
            except BaseException as e:  # noqa: BLE001
                error_msg = str(e)
                logger.error(
                    "Claude persistent client error (attempt %d): %s",
                    attempt + 1, error_msg,
                )
                if attempt == 0:
                    logger.info("Reconnecting Claude client and retrying...")
                    try:
                        await self._reconnect(system)
                    except Exception as e2:  # noqa: BLE001
                        logger.error("Reconnect failed: %s", e2)
                        return ModelResponse(content=f"Error: {error_msg}")
                    continue
                return ModelResponse(content=f"Error: {error_msg}")

        return ModelResponse(content="Error: max retries exceeded")

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream assistant text chunks as they arrive.

        Note: the Claude Agent SDK emits full ``AssistantMessage`` objects,
        not token-level deltas, so this is effectively chunked-streaming
        rather than real-time.
        """
        from claude_agent_sdk import AssistantMessage, ResultMessage

        prompt_parts: list[str] = []
        session_id = "default"
        for msg in messages:
            if msg.get("role") == "user":
                prompt_parts.append(msg.get("content", ""))
            if msg.get("session_id"):
                session_id = msg["session_id"]
        prompt = "\n\n".join(prompt_parts)

        await self._ensure_connected(system)

        try:
            assert self._client is not None
            await self._client.query(prompt, session_id=session_id)
            async for message in self._client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in (message.content or []):
                        if hasattr(block, "text"):
                            yield block.text
                elif isinstance(message, ResultMessage):
                    if message.result:
                        yield message.result
        except BaseException as e:  # noqa: BLE001
            logger.error("Claude stream error: %s", e)
            await self._reconnect(system)
            yield f"Error: {e}"
