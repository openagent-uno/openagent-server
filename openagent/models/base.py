"""Base model interface. All LLM providers implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional


@dataclass
class ToolCall:
    """A tool call requested by the model."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    """Response from a model generation call."""
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None


class BaseModel(ABC):
    """Abstract base for all LLM providers.

    To add a new provider (e.g. Ollama, vLLM), just subclass and implement
    generate() and optionally stream(). No changes needed in agent or MCP layer.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> ModelResponse:
        """Generate a response from the model.

        Args:
            messages: Conversation in [{"role": "user"|"assistant"|"tool", "content": ...}] format.
            system: Optional system prompt.
            tools: Optional list of tool definitions in a provider-neutral format:
                [{"name": str, "description": str, "input_schema": dict}, ...]
            on_status: Optional async callback for live status updates (e.g. tool use).
        """
        ...

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream response text chunks. Default: falls back to generate()."""
        response = await self.generate(messages, system=system, tools=tools)
        yield response.content
