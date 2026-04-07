"""Claude model via the Anthropic Python SDK."""

from __future__ import annotations

import os
from typing import Any, AsyncIterator

import anthropic

from openagent.models.base import BaseModel, ModelResponse, ToolCall


class ClaudeAPI(BaseModel):
    """Claude via the Anthropic API (supports tool use)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict] | anthropic.NotGiven:
        """Convert neutral tool format to Anthropic format."""
        if not tools:
            return anthropic.NOT_GIVEN
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("input_schema", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict]:
        """Convert neutral message format to Anthropic format."""
        result = []
        for msg in messages:
            role = msg["role"]
            if role == "tool":
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id", ""),
                            "content": str(msg.get("content", "")),
                        }
                    ],
                })
            elif role == "assistant" and msg.get("tool_calls"):
                content: list[dict] = []
                if msg.get("content"):
                    content.append({"type": "text", "text": msg["content"]})
                for tc in msg["tool_calls"]:
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"],
                    })
                result.append({"role": "assistant", "content": content})
            else:
                result.append({"role": role, "content": msg.get("content", "")})
        return result

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._convert_messages(messages),
            "tools": self._convert_tools(tools),
        }
        if system:
            kwargs["system"] = system

        resp = await self._client.messages.create(**kwargs)

        content_parts = []
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input,
                ))

        return ModelResponse(
            content="\n".join(content_parts),
            tool_calls=tool_calls,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            stop_reason=resp.stop_reason,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._convert_messages(messages),
            "tools": self._convert_tools(tools),
        }
        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
