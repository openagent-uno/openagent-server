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
    tool_names_called: list[str] = field(default_factory=list)
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
        files: list[Any] | None = None,
        images: list[Any] | None = None,
        audio: list[Any] | None = None,
        videos: list[Any] | None = None,
    ) -> ModelResponse:
        """Generate a response from the model.

        Args:
            messages: Conversation in [{"role": "user"|"assistant"|"tool", "content": ...}] format.
            system: Optional system prompt.
            tools: Optional list of tool definitions in a provider-neutral format:
                [{"name": str, "description": str, "input_schema": dict}, ...]
            on_status: Optional async callback for live status updates (e.g. tool use).
            files: Documents/attachments — runtime ``File`` objects (PDF, JSON,
                text, markdown, …). Forwarded as ``runtime.arun(..., files=...)``.
            images: Image attachments — runtime ``Image`` objects with ``content=bytes``.
                Routed to ``arun(..., images=...)`` for native multimodal handling.
            audio: Audio attachments — runtime ``Audio`` objects. Routed to ``arun(..., audio=...)``.
            videos: Video attachments — runtime ``Video`` objects. Routed to ``arun(..., videos=...)``.

            Splitting media by type at the call boundary matches AgentOS's
            ``process_image / process_audio / process_video / process_document``
            convention so the runtime's model adapters get the right shape for
            their multimodal API calls.
        """
        ...

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        files: list[Any] | None = None,
        images: list[Any] | None = None,
        audio: list[Any] | None = None,
        videos: list[Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream response text chunks. Default: falls back to generate()."""
        response = await self.generate(
            messages, system=system, tools=tools,
            files=files, images=images, audio=audio, videos=videos,
        )
        yield response.content

    def effective_model_id(self, session_id: str | None = None) -> str | None:
        """Return the model id that *actually* produced the last response.

        Used by ``Agent._run_inner_stream`` to populate
        ``last_response_meta()`` for streaming turns — providers don't
        return a ``ModelResponse`` from ``stream()``, so the agent has
        to synthesize one and needs to know which model to credit.

        Default reads ``self.model`` (set by the API-based runtime).
        SmartRouter overrides because it picks per-session and a single
        instance attribute can't capture which routed model handled the
        latest turn. ``None`` is acceptable — the chat UI just hides the
        model badge instead of crashing.
        """
        return getattr(self, "model", None)

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

    async def request_cancel(self, session_id: str) -> bool:
        """Cooperatively cancel the session's in-flight run, if supported.

        Runtime-backed models (agno) override this to mark the run in the
        cancellation registry so it is persisted with its partial messages.
        Default: not supported — the caller falls back to a hard cancel.
        """
        return False

    # ── cooperative-cancel run tracking ──────────────────────────────
    # Runtime-backed providers (single-model NativeProvider, team
    # TeamRouterProvider) stream through agno and must remember the agno
    # ``run_id`` of the in-flight turn so a barge-in can cooperatively cancel
    # THAT run (letting agno persist its partial messages) instead of a raw
    # ``task.cancel()`` that poisons the coroutine before any checkpoint. The
    # tracking map + helpers live here so both paths share one implementation.

    @property
    def _run_ids(self) -> dict[str, str]:
        """Per-session in-flight agno run_id (lazily created, base-owned)."""
        store = self.__dict__.get("_run_ids_store")
        if store is None:
            store = {}
            self.__dict__["_run_ids_store"] = store
        return store

    def _track_run_id(self, session_id: str | None, run_id: str | None) -> None:
        """Record the in-flight agno run_id for a session — first event wins."""
        if run_id:
            self._run_ids.setdefault(session_id or "default", run_id)

    def _clear_run_id(self, session_id: str | None) -> None:
        """Forget the in-flight run_id once a session's stream has drained."""
        self._run_ids.pop(session_id or "default", None)

    async def _coop_cancel(self, session_id: str | None) -> bool:
        """Cooperatively cancel the session's tracked run via the registry.

        Returns True if a run was registered for cancellation; False if none
        is in flight (the caller should then hard-cancel).
        """
        sid = session_id or "default"
        run_id = self._run_ids.get(sid)
        if not run_id:
            return False
        try:
            from src.core._run_state.cancel import acancel_run
            from src.core.logging import elog

            ok = bool(await acancel_run(run_id))
            if ok:
                elog("runtime.coop_cancel", session_id=sid, run_id=run_id)
            return ok
        except Exception:  # noqa: BLE001
            return False

    def known_session_ids(self) -> list[str]:
        """Return every session_id the provider currently has resume state for.

        Provider-managed models override this to expose their internal
        map so the gateway can wipe conversations across restarts.
        The default implementation returns an empty list — nothing to list.
        """
        return []
