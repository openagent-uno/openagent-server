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


def _format_shell_reminder(events) -> str:
    """Format terminal shell events into a <system-reminder> block."""
    lines = ["Background shell status update since your last message:"]
    for ev in events:
        if ev.kind == "completed":
            detail = f"completed with exit_code={ev.exit_code}"
        elif ev.kind == "timed_out":
            detail = "timed_out"
        else:
            detail = f"killed ({ev.signal or 'unknown'})"
        lines.append(
            f"- shell_id={ev.shell_id}: {detail}. stdout_bytes={ev.bytes_stdout}, "
            f"stderr_bytes={ev.bytes_stderr}. Call shell_output to read."
        )
    lines.append(
        "The user has not sent a new message; continue the task from where "
        "you left off, or summarise and stop if the work is complete."
    )
    body = "\n".join(lines)
    return f"<system-reminder>\n{body}\n</system-reminder>"

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
        config: dict | None = None,
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.config = config or {}

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

    @property
    def memory_db(self) -> MemoryDB | None:
        """Expose the runtime DB. Public accessor for REST handlers
        and manager MCPs so they don't poke at ``_db`` directly.
        """
        return self._db

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
        try:
            from openagent.mcp.servers.shell.handlers import get_hub
            await get_hub().purge_session(session_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("shell hub purge for %s failed: %s", session_id, e)

    def known_model_session_ids(
        self, *, model_override: BaseModel | None = None
    ) -> list[str]:
        """Return every session_id the primary model has resume state for.

        Used by the gateway's ``/clear`` code path to reach past its own
        in-memory SessionManager (which starts empty after a restart) and
        forget conversations whose bridge session ids were hydrated back
        into the model from disk.
        """
        model = model_override or self.model
        if model is None:
            return []
        known = getattr(model, "known_session_ids", None)
        if not callable(known):
            return []
        try:
            return list(known())
        except Exception:
            return []

    async def forget_session(
        self,
        session_id: str | None,
        *,
        model_override: BaseModel | None = None,
    ) -> None:
        """Erase all resume state for ``session_id`` so the next run starts fresh.

        Stronger than :meth:`release_session`: also drops the provider-native
        session id mapping, so the next message spawns a new subprocess
        without ``--resume``. Gateway ``/clear`` and ``/new`` call this so
        users can actually wipe the conversation.
        """
        if not session_id:
            return
        model = model_override or self.model
        if model is None:
            return
        self._prepare_model_runtime(model)
        forget_session = getattr(model, "forget_session", None)
        if callable(forget_session):
            await forget_session(session_id)
        else:
            # Fallback: release live resources even if provider lacks explicit
            # forget support — best-effort; SDK-side resume state may linger.
            close_session = getattr(model, "close_session", None)
            if callable(close_session):
                await close_session(session_id)
        try:
            from openagent.mcp.servers.shell.handlers import get_hub
            await get_hub().purge_session(session_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("shell hub purge for %s failed: %s", session_id, e)

    async def initialize(self) -> None:
        """Connect MCP servers and initialize memory DB.

        On first boot we also run the yaml → DB bootstrap so existing
        users migrate transparently. After bootstrap the MCP pool is
        rebuilt from the ``mcps`` table via ``MCPPool.from_db`` so the
        runtime can hot-reload entries without a process restart
        (see ``reload_mcps_if_changed``).
        """
        if self._initialized:
            return
        elog("agent.initialize.start", agent=self.name, model_class=type(self.model).__name__)
        if self._db:
            await self._db.connect()

        # One-shot yaml → DB bootstrap + swap to the DB-backed pool.
        # Skipped when there is no DB (pure in-memory tests); in that case
        # we fall back to whatever pool the caller passed in.
        if self._db is not None and self.config is not None:
            try:
                from openagent.memory.bootstrap import (
                    import_yaml_mcps_once,
                    import_yaml_models_once,
                )
                mcp_config = self.config.get("mcp", []) or []
                include_defaults = bool(self.config.get("mcp_defaults", True))
                mcp_disable = list(self.config.get("mcp_disable", []) or [])
                await import_yaml_mcps_once(
                    self._db, mcp_config, include_defaults, mcp_disable,
                )
                await import_yaml_models_once(
                    self._db,
                    self.config.get("providers", {}) or {},
                    model_cfg=self.config.get("model", {}) or {},
                )
            except Exception as exc:  # noqa: BLE001 — bootstrap must not block startup
                elog("bootstrap.error", level="warning", error=str(exc))

            try:
                db_path = getattr(self._db, "db_path", None)
                new_pool = await MCPPool.from_db(self._db, db_path=db_path)
                self._mcp = new_pool
                self._mcps_last_updated = await self._db.mcps_max_updated()
            except Exception as exc:  # noqa: BLE001 — leave the existing pool untouched
                elog("pool.from_db_error", level="warning", error=str(exc))

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

    async def refresh_registries(self) -> tuple[bool, int]:
        """Combined hot-reload probe for the gateway's dispatcher.

        One SQLite round-trip (``registry_status``) returns the max
        timestamps for both the mcps and models tables plus the enabled
        model count. We then reload whatever is stale and return the
        count so the caller can short-circuit when zero models are
        enabled. Returns ``(reloaded_anything, enabled_models)``.
        """
        if self._db is None:
            return False, -1
        try:
            mcps_updated, models_updated, enabled_count = await self._db.registry_status()
        except Exception as exc:  # noqa: BLE001 — never gate a message on this probe
            logger.debug("registry_status probe failed: %s", exc)
            return False, -1

        reloaded = False
        if mcps_updated > getattr(self, "_mcps_last_updated", 0.0):
            self._mcps_last_updated = mcps_updated
            try:
                await self._mcp.reload()
                for model in list(self._runtime_models):
                    wire_model_runtime(model, db=self._db, mcp_pool=self._mcp)
                reloaded = True
            except Exception as exc:  # noqa: BLE001
                elog("mcps.reload_error", level="warning", error=str(exc))

        if models_updated > getattr(self, "_models_last_updated", 0.0):
            self._models_last_updated = models_updated
            providers_config = (self.config or {}).get("providers", {})
            for model in list(self._runtime_models):
                rebuild = getattr(model, "rebuild_routing", None)
                if callable(rebuild):
                    try:
                        rebuild(providers_config)
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("rebuild_routing failed: %s", exc)
            elog("models.reload")
            reloaded = True

        return reloaded, enabled_count

    async def _run_idle_cleanup(self) -> None:
        """Periodically release idle provider resources."""
        while True:
            await asyncio.sleep(60)
            for model in list(self._runtime_models):
                cleanup_idle = getattr(model, "cleanup_idle", None)
                if not callable(cleanup_idle):
                    continue
                try:
                    released_ids = await cleanup_idle()
                    if released_ids:
                        try:
                            from openagent.mcp.servers.shell.handlers import get_hub
                            for sid in released_ids:
                                await get_hub().purge_session(sid)
                        except Exception as e:  # noqa: BLE001
                            logger.debug("shell hub purge on idle cleanup failed: %s", e)
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
        try:
            from openagent.mcp.servers.shell.handlers import get_hub
            await get_hub().shutdown()
        except Exception as e:  # noqa: BLE001
            logger.debug("shell hub shutdown failed: %s", e)
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
            # Include error_type so we can tell a KeyError from a
            # ConnectionResetError from a RuntimeError. The old format
            # swallowed the type for exceptions whose ``__str__`` is "".
            elog(
                "agent.run.error",
                level="error",
                exc_info=True,
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
        """Run a single agent turn, continuing the session automatically when
        background shells complete during or shortly after it.

        Providers handle the internal tool-loop (Agno via its Agent, Claude
        SDK via its native MCP support), so each call to ``model.generate``
        returns post-tool-loop content. This method adds a wrapper loop
        above ``generate`` that:

        1. After each turn, drains the shell hub for terminal events
           (shell_exec+run_in_background=True) for ``session_id``.
        2. If any terminal event landed, formats it as a ``<system-reminder>``
           and re-enters ``generate`` on the same session — same subprocess
           (Claude), same Agno history — so the model sees the completion
           mid-conversation.
        3. If no events landed but shells are still running, awaits
           ``hub.wait`` up to ``shell.wake_wait_window_seconds`` before
           giving up and returning to the caller.
        4. Caps at ``shell.autoloop_cap`` iterations to prevent runaway
           chains, logged via ``agent.run.autoloop_cap_hit``.

        Returns the final ``ModelResponse.content`` after the loop settles.
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

        from openagent.mcp.servers.shell.handlers import get_hub
        from openagent.mcp.servers.shell.adapters import set_session_context, reset_session_context
        from openagent.core.config import shell_settings

        hub = get_hub()
        settings = shell_settings(getattr(self, "config", None) or {})
        wake_window = settings.wake_wait_window_seconds
        cap = settings.autoloop_cap

        active_model = self._acquire_model_slot(model_override or self.model)

        current_input = message
        last_response = None
        iter_count = 0

        pending = hub.drain(session_id)
        if pending:
            pre = _format_shell_reminder(pending)
            current_input = f"{pre}\n\n{current_input}"

        try:
            while True:
                iter_count += 1
                if iter_count > cap:
                    elog(
                        "agent.run.autoloop_cap_hit",
                        session_id=session_id,
                        cap=cap,
                    )
                    break

                messages: list[dict[str, Any]] = [{"role": "user", "content": current_input}]
                await _status("Thinking...")

                token = set_session_context(session_id)
                try:
                    response = await active_model.generate(
                        messages,
                        system=system,
                        on_status=_status,
                        session_id=session_id,
                    )
                finally:
                    reset_session_context(token)

                last_response = response

                events = hub.drain(session_id)
                if not events:
                    if not hub.has_running(session_id):
                        break
                    if wake_window > 0:
                        events = await hub.wait(session_id, timeout=wake_window)
                    if not events:
                        break

                elog(
                    "agent.run.autoloop_iter",
                    session_id=session_id,
                    iter=iter_count,
                    events=len(events),
                )
                current_input = _format_shell_reminder(events)
        finally:
            self._release_model_slot(active_model)

        self._store_response_meta(session_id, last_response)
        elog(
            "agent.run.done",
            agent=self.name,
            session_id=session_id,
            model_class=type(active_model).__name__,
            response_len=len((last_response.content if last_response else "") or ""),
        )
        return (last_response.content if last_response else "") or "(Done — no final message was returned.)"

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
