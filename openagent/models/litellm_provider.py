"""Unified LLM provider via LiteLLM.

Supports 140+ models (Anthropic, OpenAI, Google, Ollama, etc.) through a
single interface. Replaces per-provider SDK wrappers with one class.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator

import litellm

from openagent.models.base import BaseModel, ModelResponse, ToolCall

logger = logging.getLogger(__name__)

# Suppress litellm's verbose logging
litellm.suppress_debug_info = True


def get_cheapest_model(provider: str) -> str | None:
    """Return the cheapest model for a provider from litellm's catalog."""
    try:
        from litellm import model_cost
    except ImportError:
        return None

    best_id = None
    best_cost = float("inf")
    prefix = f"{provider}."  # litellm uses dots: "anthropic.claude-haiku-4-5"
    prefix_slash = f"{provider}/"

    for model_id, info in model_cost.items():
        if not (model_id.startswith(prefix) or model_id.startswith(prefix_slash)):
            continue
        cost = info.get("input_cost_per_token") or 0
        if 0 < cost < best_cost:
            best_cost = cost
            best_id = model_id

    return best_id


class LiteLLMProvider(BaseModel):
    """Unified API provider powered by LiteLLM.

    Model string uses litellm format:
      - "anthropic/claude-sonnet-4-6"
      - "openai/gpt-4o"
      - "google/gemini-2.5-pro"
      - "ollama/llama3"
    """

    def __init__(
        self,
        model: str = "anthropic/claude-sonnet-4-6",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 4096,
        providers_config: dict | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key
        self._base_url = base_url
        self._providers_config = providers_config or {}

        # Set provider-specific env vars from providers config
        self._inject_provider_keys()

    def _inject_provider_keys(self) -> None:
        """Set API key env vars from the providers config section."""
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GEMINI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "zai": "ZAI_API_KEY",
            "zhipu": "ZAI_API_KEY",
        }
        for provider_name, cfg in self._providers_config.items():
            env_var = env_map.get(provider_name)
            key = cfg.get("api_key")
            if env_var and key and not os.environ.get(env_var):
                os.environ[env_var] = key

    def _convert_tools(self, tools: list[dict[str, Any]] | None) -> list[dict] | None:
        """Convert neutral MCP tool format to OpenAI function calling format."""
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
        """Convert neutral message format to OpenAI format (litellm's common format)."""
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
                    tool_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
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

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        system: str | None,
        tools: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._convert_messages(messages, system),
        }
        converted_tools = self._convert_tools(tools)
        if converted_tools:
            kwargs["tools"] = converted_tools
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["api_base"] = self._base_url
        return kwargs

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status=None,
        session_id: str | None = None,
    ) -> ModelResponse:
        from openagent.core.logging import elog

        kwargs = self._build_kwargs(messages, system, tools)
        n_tools = len(kwargs.get("tools", []))
        n_msgs = len(kwargs.get("messages", []))
        logger.debug("LiteLLM generate: model=%s msgs=%d tools=%d", self.model, n_msgs, n_tools)
        elog(
            "litellm.request",
            model=self.model,
            session_id=session_id,
            messages=n_msgs,
            tools=n_tools,
            has_base_url=bool(self._base_url),
            has_api_key=bool(self._api_key),
        )

        try:
            resp = await litellm.acompletion(**kwargs)
        except Exception as e:
            logger.error("LiteLLM generate failed: model=%s error=%s", self.model, e)
            elog(
                "litellm.error",
                model=self.model,
                session_id=session_id,
                messages=n_msgs,
                tools=n_tools,
                error=str(e),
            )
            raise

        choice = resp.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    args = json.loads(args)
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        in_tok = resp.usage.prompt_tokens if resp.usage else 0
        out_tok = resp.usage.completion_tokens if resp.usage else 0
        logger.info("LiteLLM response: model=%s in=%d out=%d tools=%d stop=%s",
                     self.model, in_tok, out_tok, len(tool_calls), choice.finish_reason)
        elog(
            "litellm.generate",
            model=self.model,
            session_id=session_id,
            input_tokens=in_tok,
            output_tokens=out_tok,
            tool_calls=len(tool_calls),
            stop_reason=choice.finish_reason,
        )

        return ModelResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            input_tokens=in_tok,
            output_tokens=out_tok,
            stop_reason=choice.finish_reason,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        from openagent.core.logging import elog

        kwargs = self._build_kwargs(messages, system, tools)
        kwargs["stream"] = True
        elog(
            "litellm.stream.start",
            model=self.model,
            messages=len(kwargs.get("messages", [])),
            tools=len(kwargs.get("tools", [])),
        )
        try:
            resp = await litellm.acompletion(**kwargs)
            async for chunk in resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            elog("litellm.stream.done", model=self.model)
        except Exception as e:
            elog("litellm.stream.error", model=self.model, error=str(e))
            raise
