"""Core Agent class: orchestrates model, MCP pool, and memory."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Callable, Awaitable

from openagent.channels.base import build_attachment_context, prepend_context_block
from openagent.models.base import BaseModel, ModelResponse
from openagent.memory.db import MemoryDB
from openagent.mcp.pool import MCPPool
from openagent.core.prompts import FRAMEWORK_SYSTEM_PROMPT
from openagent.models.runtime import wire_model_runtime

from openagent.core.logging import elog

logger = logging.getLogger(__name__)

# Status callback type: async def on_status(status: str) -> None
StatusCallback = Callable[[str], Awaitable[None]]


class Agent:
    """Main agent class. Ties together a model, MCP pool, and memory.

    OpenAgent owns the *product* layer (catalog, pricing, gateway, channels,
    memory vault, dormant-MCP detection). Tool execution and the per-call
    tool loop are delegated to the active provider:

      - ``AgnoProvider`` consumes ``MCPPool.agno_toolkits`` (Agno ``MCPTools``
        instances) and Agno's ``Agent`` runs the loop internally, including
        proper image-artifact handling for binary tool results.
      - ``ClaudeCLI`` consumes ``MCPPool.claude_sdk_servers()`` (raw stdio
        config) and the Claude Agent SDK manages everything itself.

    Either way, ``Agent.run`` is a single ``model.generate`` call — the
    provider returns the final content after running its own tool loop.

    Long-term memory lives in the Obsidian-style vault exposed through MCP.
    The SQLite database is used for runtime state such as scheduler tasks,
    platform-managed chat sessions, and usage tracking.

    Usage:
        pool = MCPPool.from_config(mcp_config=cfg.get("mcp"), ...)
        agent = Agent(
            name="assistant",
            model=AgnoProvider(model="anthropic:claude-sonnet-4-20250514"),
            system_prompt="You are a helpful assistant.",
            mcp_pool=pool,
            memory=MemoryDB("agent.db"),
        )
        async with agent:
            response = await agent.run("Hello!", user_id="user-1")
    """

    def __init__(
        self,
        name: str = "agent",
        model: BaseModel | None = None,
        system_prompt: str = "You are a helpful assistant.",
        mcp_pool: MCPPool | None = None,
        memory: MemoryDB | str | None = None,
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt

        # MCPPool — owns the lifecycle of all MCP servers for the process.
        # Pass an empty pool if not provided so dormant detection / system
        # prompt building still work without crashing.
        self._mcp = mcp_pool if mcp_pool is not None else MCPPool([])

        # Runtime DB; the long-term knowledge base still lives in the Obsidian vault via MCP.
        if isinstance(memory, str):
            self._db = MemoryDB(memory)
        elif isinstance(memory, MemoryDB):
            self._db = memory
        else:
            self._db = None

        self._initialized = False
        self._idle_cleanup_task: asyncio.Task | None = None
        self._runtime_models: list[BaseModel] = []
        self._last_response_meta: dict[str, dict[str, Any]] = {}

        # Per-model in-flight counters + drain events. Keyed by id(model).
        # Used by swap_model() to hold old models alive until their last
        # generate() call returns, then shutdown them asynchronously.
        self._inflight_counts: dict[int, int] = {}
        self._drain_events: dict[int, asyncio.Event] = {}

    @staticmethod
    def _response_meta_key(session_id: str | None) -> str:
        return session_id or "__default__"

    def _store_response_meta(self, session_id: str | None, response: ModelResponse | None) -> None:
        key = self._response_meta_key(session_id)
        if response is None or not response.model:
            self._last_response_meta.pop(key, None)
            return
        self._last_response_meta[key] = {"model": response.model}

    def last_response_meta(self, session_id: str | None) -> dict[str, Any]:
        return dict(self._last_response_meta.get(self._response_meta_key(session_id), {}))

    def _register_runtime_model(self, model: BaseModel | None) -> None:
        """Track every model instance that may need lifecycle management."""
        if model is None:
            return
        if any(existing is model for existing in self._runtime_models):
            return
        self._runtime_models.append(model)

    def _unregister_runtime_model(self, model: BaseModel | None) -> None:
        """Remove *model* from the runtime registry (no-op if absent)."""
        if model is None:
            return
        self._runtime_models = [m for m in self._runtime_models if m is not model]

    def _prepare_model_runtime(self, model: BaseModel | None) -> None:
        """Wire shared runtime dependencies into models that support them."""
        if model is None:
            return
        self._register_runtime_model(model)
        wire_model_runtime(model, db=self._db, mcp_pool=self._mcp)

    def _acquire_model_slot(self, model: BaseModel | None) -> BaseModel | None:
        """Increment the in-flight counter for *model*. Returns *model* unchanged."""
        if model is None:
            return None
        key = id(model)
        self._inflight_counts[key] = self._inflight_counts.get(key, 0) + 1
        return model

    def _release_model_slot(self, model: BaseModel | None) -> None:
        """Decrement the in-flight counter for *model*; fire drain event at zero."""
        if model is None:
            return
        key = id(model)
        remaining = self._inflight_counts.get(key, 0) - 1
        if remaining <= 0:
            self._inflight_counts.pop(key, None)
            ev = self._drain_events.pop(key, None)
            if ev is not None:
                ev.set()
        else:
            self._inflight_counts[key] = remaining

    def swap_model(self, new_model: BaseModel) -> tuple[BaseModel | None, asyncio.Event]:
        """Atomically replace ``self.model`` with *new_model*.

        Returns ``(old_model, drain_event)``. The caller should
        ``await drain_event.wait()`` in a background task and then call
        ``old_model.shutdown()`` to release its resources after its last
        in-flight ``generate()`` call has completed.

        If the old model had no in-flight calls, ``drain_event`` is already
        set so the caller can shut down immediately.
        """
        old = self.model
        self._prepare_model_runtime(new_model)
        self.model = new_model
        self._ensure_idle_cleanup_task()

        if old is None or old is new_model:
            ev = asyncio.Event()
            ev.set()
            return old, ev

        key = id(old)
        if self._inflight_counts.get(key, 0) <= 0:
            ev = asyncio.Event()
            ev.set()
        else:
            ev = self._drain_events.setdefault(key, asyncio.Event())

        # Keep *old* in the runtime registry so Agent.shutdown() will
        # still clean it up if the process exits before drain completes.
        # Caller must call _unregister_runtime_model(old) after shutdown.
        return old, ev

    def _ensure_idle_cleanup_task(self) -> None:
        """Start the idle cleanup loop if any runtime model supports it."""
        if self._idle_cleanup_task and not self._idle_cleanup_task.done():
            return
        if any(callable(getattr(model, "cleanup_idle", None)) for model in self._runtime_models):
            self._idle_cleanup_task = asyncio.create_task(self._run_idle_cleanup())

    async def release_session(
        self,
        session_id: str | None,
        *,
        model_override: BaseModel | None = None,
    ) -> None:
        """Release live runtime resources tied to one session, if supported."""
        if not session_id:
            return
        model = model_override or self.model
        if model is None:
            return
        self._prepare_model_runtime(model)
        close_session = getattr(model, "close_session", None)
        if not callable(close_session):
            return
        await close_session(session_id)

    async def initialize(self) -> None:
        """Connect MCP servers and initialize memory DB."""
        if self._initialized:
            return
        elog("agent.initialize.start", agent=self.name, model_class=type(self.model).__name__)
        if self._db:
            await self._db.connect()
        await self._mcp.connect_all()

        self._prepare_model_runtime(self.model)
        self._ensure_idle_cleanup_task()

        self._initialized = True
        elog(
            "agent.initialize.done",
            agent=self.name,
            model_class=type(self.model).__name__,
            mcp_servers=self._mcp.server_count,
            tools=self._mcp.total_tool_count,
            has_db=bool(self._db),
        )

    async def _run_idle_cleanup(self) -> None:
        """Periodically release idle provider resources."""
        while True:
            await asyncio.sleep(60)
            for model in list(self._runtime_models):
                cleanup_idle = getattr(model, "cleanup_idle", None)
                if not callable(cleanup_idle):
                    continue
                try:
                    await cleanup_idle()
                except Exception as e:
                    logger.debug("Idle cleanup error: %s", e)

    async def shutdown(self) -> None:
        """Close all connections."""
        elog("agent.shutdown.start", agent=self.name)
        if self._idle_cleanup_task:
            self._idle_cleanup_task.cancel()
            self._idle_cleanup_task = None
        # Persistent model runtimes may need an explicit shutdown to
        # release subprocesses or cached sessions cleanly.
        seen: set[int] = set()
        for model in [self.model, *self._runtime_models]:
            if model is None or id(model) in seen:
                continue
            seen.add(id(model))
            shutdown = getattr(model, "shutdown", None)
            if callable(shutdown):
                try:
                    await shutdown()
                except Exception as e:  # noqa: BLE001
                    logger.warning("Model shutdown error: %s", e)
        await self._mcp.close_all()
        if self._db:
            await self._db.close()
        self._initialized = False
        self._runtime_models.clear()
        elog("agent.shutdown.done", agent=self.name)

    async def run(
        self,
        message: str,
        user_id: str = "",
        session_id: str | None = None,
        attachments: list[dict] | None = None,
        on_status: StatusCallback | None = None,
        model_override: BaseModel | None = None,
    ) -> str:
        """Run the agent with a user message. Returns the final text response.

        Args:
            session_id: Session key passed through to whichever history mode
                the active model uses.
            on_status: Optional async callback for live status updates.
                Called with status strings like "Thinking...", "Using shell_exec...", etc.
                Channels use this to update a live status message.
        """
        if not self.model:
            raise RuntimeError("No model configured. Set agent.model before calling run().")

        await self.initialize()
        self._prepare_model_runtime(model_override)
        self._ensure_idle_cleanup_task()

        async def _status(msg: str) -> None:
            if on_status:
                try:
                    await on_status(msg)
                except Exception:
                    pass

        try:
            self._store_response_meta(session_id, None)
            elog(
                "agent.run.start",
                agent=self.name,
                user_id=user_id,
                session_id=session_id,
                model_class=type(model_override or self.model).__name__,
                attachments=len(attachments or []),
            )
            return await self._run_inner(message, attachments, _status, session_id=session_id, model_override=model_override)
        except asyncio.CancelledError:
            # Shutdown or task-level cancellation is NOT a fatal error — it's
            # the runtime telling us to stop cleanly. Log it as such, tell the
            # caller something useful (empty ``str(CancelledError)`` used to
            # surface as "Error:" with nothing after), and re-raise so the
            # caller's cancellation semantics are preserved.
            elog(
                "agent.run.cancelled",
                agent=self.name,
                user_id=user_id,
                session_id=session_id,
            )
            logger.info("Agent.run() cancelled for session %s", session_id)
            raise
        except BaseException as e:
            # Log the exception TYPE and repr so we can tell a KeyError from a
            # ConnectionResetError from a RuntimeError. The old ``f"{e}"``
            # format swallowed the type and printed empty strings for
            # exceptions whose ``__str__`` is "" (CancelledError, SystemExit…),
            # which is why these have been appearing as "fatal error: " in
            # the logs.
            logger.error(
                "Agent.run() fatal error: %s: %r",
                type(e).__name__, e,
                exc_info=True,
            )
            elog(
                "agent.run.error",
                agent=self.name,
                user_id=user_id,
                session_id=session_id,
                error_type=type(e).__name__,
                error=str(e) or repr(e),
            )
            return f"Error: {type(e).__name__}: {e}" if str(e) else f"Error: {type(e).__name__}"

    async def _run_inner(
        self,
        message: str,
        attachments: list[dict] | None,
        _status,
        session_id: str | None = None,
        model_override: BaseModel | None = None,
    ) -> str:
        """Inner run logic, wrapped by run() for crash protection.

        Providers handle the tool-use loop internally (Agno via its Agent,
        Claude SDK via its native MCP support), so this method is now a
        single ``model.generate`` call. The provider returns the final
        post-tool-loop content; we just package the prompt and unpack the
        response.
        """
        await _status("Loading context...")

        # Combine OpenAgent's framework-level guidelines with the user's
        # project-specific system prompt from openagent.yaml.
        system = self._combined_system_prompt()

        # Include local paths for attachments so the tool layer can inspect them.
        if attachments:
            files_info: list[str] = []
            for a in attachments:
                a_type = a.get("type", "file")
                a_name = a.get("filename", "")
                a_path = a.get("path", "")
                if a_path:
                    files_info.append(f"- {a_type}: {a_name} — local path: {a_path}")
                else:
                    files_info.append(f"- {a_type}: {a_name}")
            message = prepend_context_block(
                message,
                build_attachment_context(
                    files_info,
                    read_hint=(
                        "Use the Read tool (or an MCP tool) with the local path to inspect each file. "
                        "For images, Read returns the image content for you to see directly."
                    ),
                ),
            )

        messages: list[dict[str, Any]] = [{"role": "user", "content": message}]

        await _status("Thinking...")

        active_model = self._acquire_model_slot(model_override or self.model)
        try:
            response = await active_model.generate(
                messages,
                system=system,
                on_status=_status,
                session_id=session_id,
            )
        finally:
            self._release_model_slot(active_model)

        self._store_response_meta(session_id, response)
        elog(
            "agent.run.done",
            agent=self.name,
            session_id=session_id,
            model_class=type(active_model).__name__,
            response_len=len(response.content or ""),
        )
        return response.content if response else "I wasn't able to complete the request."

    def _combined_system_prompt(self) -> str:
        """Concatenate the framework prompt with the user's project-specific one."""
        user = (self.system_prompt or "").strip()
        if not user:
            return FRAMEWORK_SYSTEM_PROMPT
        return (
            FRAMEWORK_SYSTEM_PROMPT
            + "\n\n── User-specific identity and project context ──\n\n"
            + user
        )

    async def stream_run(
        self,
        message: str,
        user_id: str = "",
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream the agent's response. Does not support tool use in streaming mode."""
        if not self.model:
            raise RuntimeError("No model configured.")

        await self.initialize()

        system = self._combined_system_prompt()
        messages: list[dict[str, Any]] = [{"role": "user", "content": message}]

        async for chunk in self.model.stream(messages, system=system):
            yield chunk

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.shutdown()
