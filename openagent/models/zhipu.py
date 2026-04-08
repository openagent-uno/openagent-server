"""Z.ai GLM model via OpenAI-compatible API.

Also serves as the base for any OpenAI-compatible provider (Ollama, vLLM, etc.).
"""

from __future__ import annotations

import os
import uuid
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from openagent.models.base import BaseModel, ModelResponse, ToolCall


class ZhipuGLM(BaseModel):
    """Z.ai GLM via their OpenAI-compatible API.

    Can be reused for any OpenAI-compatible endpoint by setting base_url.
    """

    def __init__(
        self,
        model: str = "glm-4",
        api_key: str | None = None,
        base_url: str = "https://open.bigmodel.cn/api/paas/v4",
        max_tokens: int = 4096,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(
            api_key=api_key or os.environ.get("ZHIPU_API_KEY", ""),
            base_url=base_url,
        )

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict] | None:
        """Convert neutral tool format to OpenAI function calling format."""
        if not tools:
            return None
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    def _convert_messages(self, messages: list[dict[str, Any]], system: str | None) -> list[dict]:
        """Convert neutral message format to OpenAI format."""
        result = []
        if system:
            result.append({"role": "system", "content": system})
        for msg in messages:
            role = msg["role"]
            if role == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", ""),
                    "content": str(msg.get("content", "")),
                })
            elif role == "assistant" and msg.get("tool_calls"):
                tool_calls = []
                for tc in msg["tool_calls"]:
                    import json as _json
                    tool_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": _json.dumps(tc["arguments"]),
                        },
                    })
                result.append({
                    "role": "assistant",
                    "content": msg.get("content") or None,
                    "tool_calls": tool_calls,
                })
            else:
                result.append({"role": role, "content": msg.get("content", "")})
        return result

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status=None,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._convert_messages(messages, system),
        }
        converted_tools = self._convert_tools(tools)
        if converted_tools:
            kwargs["tools"] = converted_tools

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            import json as _json
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=_json.loads(tc.function.arguments),
                ))

        return ModelResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            stop_reason=choice.finish_reason,
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
            "messages": self._convert_messages(messages, system),
            "stream": True,
        }
        converted_tools = self._convert_tools(tools)
        if converted_tools:
            kwargs["tools"] = converted_tools

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
