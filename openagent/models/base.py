"""Base model interface. All LLM providers implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable


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
    model: str | None = None


class BaseModel(ABC):
    """Abstract base for all LLM providers.

    To add a new provider (e.g. Ollama, vLLM), just subclass and implement
    generate() and optionally stream(). No changes needed in agent or MCP layer.
    """

    history_mode: str = "caller"
    """How chat history is owned for a session.

    - ``caller``: the caller must pass the relevant history every run
    - ``platform``: OpenAgent/runtime manages persisted chat history
    - ``provider``: the upstream provider manages history internally
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
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

    async def close_session(self, session_id: str) -> None:
        """Release any live runtime resources for one session.

        Provider-managed models may keep per-session subprocesses or sockets
        alive between turns. The default implementation is a no-op because
        caller-managed and platform-managed models often have nothing live to
        tear down.
        """
        return None

    async def forget_session(self, session_id: str) -> None:
        """Drop the subprocess AND erase any resume state for this session.

        Semantically stronger than ``close_session``: after a ``forget_session``,
        the next message on this ``session_id`` must spawn a fresh subprocess
        with no ``--resume`` and no memory of the prior transcript. Used by the
        gateway's ``/clear`` and ``/new`` commands so the user can truly wipe
        the conversation.

        Default implementation falls back to ``close_session`` — caller- and
        platform-managed models have no hidden resume state to erase.
        """
        await self.close_session(session_id)
