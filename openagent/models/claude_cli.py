"""Claude model via the Claude Agent SDK (persistent sessions, MCP support)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

from openagent.models.base import BaseModel, ModelResponse, ToolCall

logger = logging.getLogger(__name__)


class ClaudeCLI(BaseModel):
    """Claude via the Claude Agent SDK.

    Uses persistent sessions — MCP servers connect once and stay connected
    across multiple messages. No subprocess spawning per message.

    Requires `claude-agent-sdk` installed and Claude CLI authenticated.
    Works with Claude Pro/Max membership (no API key needed).
    """

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
        self._session_id: str | None = None

    def set_mcp_servers(self, servers: dict[str, dict]) -> None:
        """Set MCP server configs. Called by Agent during initialization."""
        self.mcp_servers = servers

    def _build_options(self) -> dict[str, Any]:
        """Build ClaudeAgentOptions kwargs."""
        from claude_agent_sdk import ClaudeAgentOptions

        opts: dict[str, Any] = {}

        if self.permission_mode == "bypass":
            opts["permission_mode"] = "bypassPermissions"
        elif self.permission_mode == "auto":
            opts["permission_mode"] = "acceptEdits"

        if self.mcp_servers:
            opts["mcp_servers"] = self.mcp_servers

        if self._session_id:
            opts["resume"] = self._session_id

        return opts

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage, SystemMessage

        # Build prompt from messages
        prompt_parts = []
        if system:
            prompt_parts.append(f"<system>\n{system}\n</system>")
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(content)
            elif role == "assistant":
                prompt_parts.append(f"[Previous assistant response] {content}")

        prompt = "\n\n".join(prompt_parts)

        opts = self._build_options()
        options = ClaudeAgentOptions(**opts)

        result_text = ""
        try:
            async for message in query(prompt=prompt, options=options):
                # Capture session ID for resume
                if isinstance(message, SystemMessage) and hasattr(message, 'data'):
                    data = message.data if isinstance(message.data, dict) else {}
                    if 'session_id' in data:
                        self._session_id = data['session_id']

                if isinstance(message, ResultMessage):
                    result_text = message.result or ""

        except Exception as e:
            logger.error(f"Claude Agent SDK error: {e}")
            result_text = f"Error: {e}"

        return ModelResponse(content=result_text)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage, SystemMessage, AssistantMessage

        prompt_parts = []
        if system:
            prompt_parts.append(f"<system>\n{system}\n</system>")
        for msg in messages:
            if msg["role"] == "user":
                prompt_parts.append(msg.get("content", ""))

        prompt = "\n\n".join(prompt_parts)

        opts = self._build_options()
        options = ClaudeAgentOptions(**opts)

        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, SystemMessage) and hasattr(message, 'data'):
                    data = message.data if isinstance(message.data, dict) else {}
                    if 'session_id' in data:
                        self._session_id = data['session_id']

                if isinstance(message, AssistantMessage):
                    for block in (message.content or []):
                        if hasattr(block, 'text'):
                            yield block.text

                if isinstance(message, ResultMessage):
                    if message.result:
                        yield message.result

        except Exception as e:
            logger.error(f"Claude Agent SDK stream error: {e}")
            yield f"Error: {e}"
