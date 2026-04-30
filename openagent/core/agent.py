"""Core Agent class: orchestrates model, MCP pool, and memory."""

from __future__ import annotations

import asyncio
import inspect
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


def _format_run_error(e: BaseException) -> str:
    """Produce a chat-renderable error string for any agent-run failure.

    Two shapes:
      - ``AgnoProviderError`` carries an already-clean provider message
        (e.g. "API status error from OpenAI API: 403 - You are not
        allowed to sample from this model"). Prefix with a stable
        marker so bridges and the app can detect it as an error and
        style it accordingly.
      - Anything else falls back to ``Error: <ClassName>: <repr>`` so
        the user sees *something* even on novel exception types.
    """
    from openagent.models.agno_provider import AgnoProviderError

    if isinstance(e, AgnoProviderError):
        return f"⚠️ Model provider error\n\n{e}"
    msg = str(e) or repr(e)
    return f"⚠️ {type(e).__name__}: {msg}" if msg else f"⚠️ {type(e).__name__}"


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


_VAULT_WRITE_TOOLS = frozenset({
    "vault_write_note",
    "vault_patch_note",
    "vault_update_frontmatter",
    "vault_delete_note",
    "vault_move_note",
    "vault_manage_tags",
})

_VAULT_READ_TOOLS = frozenset({
    "vault_read_note",
    "vault_read_multiple_notes",
    "vault_search_notes",
    "vault_list_notes",
    "vault_get_frontmatter",
    "vault_list_all_tags",
    "vault_get_vault_stats",
    "vault_get_backlinks",
})


def _emit_tool_call_summary(
    response: Any, *, session_id: str | None, iter_count: int,
) -> None:
    """Log per-iteration tool call breakdown to events.jsonl.

    Used to measure how often the agent actually writes to the vault, so
    prompt tweaks can be evaluated against a numeric baseline. Best-effort:
    silently no-ops when the provider didn't populate ``tool_names_called``.
    """
    tool_names = list(getattr(response, "tool_names_called", None) or [])
    if not tool_names:
        return
    by_server: dict[str, int] = {}
    vault_writes = 0
    vault_reads = 0
    for name in tool_names:
        server = name.split("_", 1)[0] if "_" in name else name
        by_server[server] = by_server.get(server, 0) + 1
        if name in _VAULT_WRITE_TOOLS:
            vault_writes += 1
        elif name in _VAULT_READ_TOOLS:
            vault_reads += 1
    elog(
        "agent.turn.tool_calls",
        session_id=session_id,
        iter=iter_count,
        by_server=by_server,
        vault_writes=vault_writes,
        vault_reads=vault_reads,
        total=len(tool_names),
    )


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
        agent = Agent(
            name="assistant",
            model=AgnoProvider(model="anthropic:claude-sonnet-4-20250514"),
            system_prompt="You are a helpful assistant.",
            mcp_pool=None,  # ``initialize`` rebuilds from the ``mcps`` DB table
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

        # Materialised provider+model catalog from the SQLite ``providers`` /
        # ``models`` tables. Populated by ``_hydrate_providers_from_db`` at
        # boot and on every hot-reload tick; the yaml config is never
        # consulted for this state.
        self._providers_config: list[dict[str, Any]] = []

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

        The ``mcps`` / ``providers`` / ``models`` SQLite tables are the
        sole sources of truth at runtime. ``ensure_builtin_mcps`` runs
        every boot to backfill any missing builtin rows (forward compat
        + safety net); the MCP pool is then (re)built from the DB via
        ``MCPPool.from_db`` so the runtime can hot-reload entries
        without a process restart (see ``reload_mcps_if_changed``).
        """
        if self._initialized:
            return
        elog("agent.initialize.start", agent=self.name, model_class=type(self.model).__name__)
        if self._db:
            await self._db.connect()

        # Hydrate providers/models from the DB and swap to the DB-backed
        # MCP pool. Skipped when there is no DB (pure in-memory tests);
        # in that case we fall back to whatever pool the caller passed in.
        if self._db is not None:
            try:
                from openagent.memory.bootstrap import ensure_builtin_mcps
                # Every boot: re-seed any BUILTIN_MCP_SPECS entry that
                # doesn't have a row yet (forward-compat for future
                # builtins + safety net against manual DB tampering).
                # Existing rows — including disabled ones — are untouched.
                await ensure_builtin_mcps(self._db)
                # Provider keys and the model catalog are DB-backed. Pull
                # the rows into ``self._providers_config`` so SmartRouter
                # / AgnoProvider see the materialised view.
                await self._hydrate_providers_from_db()
                self._providers_last_updated = await self._db.providers_max_updated()
                self._models_last_updated = await self._db.models_max_updated()
                # Hand the freshly-hydrated list to every live runtime
                # model. SmartRouter was constructed with an empty
                # providers_config; without this push it would keep that
                # empty reference until the first hot-reload tick — which
                # only fires on gateway messages, so scheduler turns that
                # run before any user chat would see an empty catalog and
                # reject with "no_enabled_model".
                providers_config = self._providers_config
                for model in list(self._runtime_models) + [self.model]:
                    if model is None:
                        continue
                    rebuild = getattr(model, "rebuild_routing", None)
                    if callable(rebuild):
                        try:
                            rebuild(providers_config)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("rebuild_routing on boot failed: %s", exc)
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

        # Prime OpenRouter's catalog in the background so ``get_model_pricing``
        # has live rates before the first cost attribution, without blocking
        # startup on a network call. Errors are swallowed — the catalog has a
        # bundled offline backstop.
        async def _prime_openrouter() -> None:
            try:
                from openagent.models.discovery import _fetch_openrouter_catalog
                await _fetch_openrouter_catalog()
            except Exception as exc:  # noqa: BLE001
                elog("openrouter.prefetch_error", level="warning", error=str(exc))

        # Warm the local Whisper model in the background so the first
        # voice-tab utterance doesn't pay the 60s+ download/load tax
        # (small ≈ 464 MB; ~10s cold-load even when cached locally).
        # By the time the user records anything, the model is in RAM.
        # Errors swallowed — transcribe() lazy-loads as a fallback.
        async def _prime_whisper() -> None:
            try:
                from openagent.channels.voice import _load_local_model
                await _load_local_model()
                elog("whisper.prefetch_done")
            except Exception as exc:  # noqa: BLE001
                elog("whisper.prefetch_error", level="warning", error=str(exc))

        # Same idea for Piper: cold-load is ~10s for the ONNX model
        # plus a one-time ~25 MB voice-file download. Prefetch so the
        # first reply doesn't sit silent for 12s before audio plays.
        async def _prime_piper() -> None:
            try:
                from openagent.channels import tts_local
                if not tts_local.is_available():
                    return
                # Resolve to the configured default voice and load it.
                # ``_load_voice`` is the exact path synth uses, so a
                # successful prefetch guarantees the next synth is warm.
                voice = tts_local._resolve_voice_name(None)
                loaded = await tts_local._load_voice(voice)
                if loaded is not None:
                    elog("piper.prefetch_done", voice=voice)
            except Exception as exc:  # noqa: BLE001
                elog("piper.prefetch_error", level="warning", error=str(exc))

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_prime_openrouter())
            loop.create_task(_prime_whisper())
            loop.create_task(_prime_piper())
        except RuntimeError:
            # No running loop (sync entry point) — skip; all three
            # backends lazy-load on first request.
            pass

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
        timestamps for the mcps / models / providers tables plus the
        enabled model count. We then reload whatever is stale and
        return the count so the caller can short-circuit when zero
        models are enabled. Returns ``(reloaded_anything, enabled_models)``.

        Provider edits (``api_key``, ``base_url``) invalidate the cached
        ``providers_config`` dict that SmartRouter hands to AgnoProvider
        — without this hook, adding a key would require a restart.
        """
        if self._db is None:
            return False, -1
        try:
            mcps_updated, models_updated, enabled_count, providers_updated = (
                await self._db.registry_status()
            )
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

        providers_changed = providers_updated > getattr(self, "_providers_last_updated", 0.0)
        if providers_changed:
            self._providers_last_updated = providers_updated
            try:
                await self._hydrate_providers_from_db()
                elog("providers.reload")
                reloaded = True
            except Exception as exc:  # noqa: BLE001
                logger.debug("providers hydrate failed: %s", exc)

        # Models or providers changed → rebuild router. Providers affect
        # routing because AgnoProvider's api_key lookup goes through
        # ``providers_config``; models affect it because the classifier
        # picks from the materialised models list.
        models_changed = models_updated > getattr(self, "_models_last_updated", 0.0)
        if models_changed or providers_changed:
            self._models_last_updated = max(
                models_updated, getattr(self, "_models_last_updated", 0.0) or 0.0
            )
            if models_changed and not providers_changed:
                # Providers hydrate already ran above; re-run only when
                # models alone changed so the materialised catalog is fresh.
                try:
                    await self._hydrate_providers_from_db()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("models hydrate failed: %s", exc)
            providers_config = self._providers_config
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

    async def _hydrate_providers_from_db(self) -> None:
        """Pull provider + model rows from the DB into ``self._providers_config``.

        The DB is the source of truth for provider keys AND the model
        catalog. SmartRouter / AgnoProvider consume the v0.12 flat-list
        shape — each entry already carries its ``framework`` and its
        nested ``models`` list, so the same vendor can appear twice
        (anthropic+agno AND anthropic+claude-cli) without a key
        collision. Delegates the SQL materialisation to MemoryDB so
        smoke-test endpoints can reuse the same shape.
        """
        if self._db is None:
            return
        try:
            self._providers_config = await self._db.materialise_providers_config(
                enabled_only=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("providers hydrate failed: %s", exc)
            self._providers_config = []

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
            return _format_run_error(e)

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
        # project-specific system prompt from openagent.yaml. Passing
        # ``session_id`` appends a ``<session-id>`` tag so the LLM can
        # call tools that operate on its own session (e.g. pin_session).
        system = self._combined_system_prompt(session_id=session_id)

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

                _emit_tool_call_summary(
                    response, session_id=session_id, iter_count=iter_count,
                )

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

    async def run_stream(
        self,
        message: str,
        user_id: str = "",
        session_id: str | None = None,
        attachments: list[dict] | None = None,
        on_status: StatusCallback | None = None,
        model_override: BaseModel | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming sibling of :meth:`run` for voice-mode replies.

        Yields events as plain dicts so the gateway/orchestrator can
        consume them without import gymnastics:

        - ``{"kind": "delta", "text": "..."}`` — incremental text from
          the LLM. Feed this to a sentence chunker → TTS pipeline.
        - ``{"kind": "iteration_break"}`` — emitted between autoloop
          iterations (a tool finished and the agent re-enters
          ``model.stream`` with a shell-reminder). The chunker should
          ``flush()`` here so a sentence split by a tool call isn't
          re-narrated. (Risk #1 in the voice-chat plan.)
        - ``{"kind": "done", "text": "<full text>"}`` — final event with
          the assembled response text (post-marker-strip is up to the
          caller; we just emit raw text).

        Cancellation propagates exactly like :meth:`run`: a
        ``CancelledError`` from the model layer is logged and re-raised.
        """
        if not self.model:
            raise RuntimeError("No model configured. Set agent.model before calling run_stream().")

        await self.initialize()
        self._prepare_model_runtime(model_override)
        self._ensure_idle_cleanup_task()

        async def _status(msg: str) -> None:
            if on_status:
                try:
                    await on_status(msg)
                except Exception:
                    pass

        elog(
            "agent.run_stream.start",
            agent=self.name,
            user_id=user_id,
            session_id=session_id,
            model_class=type(model_override or self.model).__name__,
            attachments=len(attachments or []),
        )

        try:
            async for event in self._run_inner_stream(
                message, attachments, _status,
                session_id=session_id, model_override=model_override,
            ):
                yield event
        except asyncio.CancelledError:
            elog(
                "agent.run_stream.cancelled",
                agent=self.name,
                user_id=user_id,
                session_id=session_id,
            )
            raise
        # ``Exception`` (not ``BaseException``) so we don't catch
        # ``GeneratorExit`` — that's how Python tells us the consumer
        # stopped iterating early (the orchestrator's ``break`` after a
        # ``done`` event triggers ``aclose`` → ``GeneratorExit`` here).
        # Catching it and yielding from the cleanup path is illegal —
        # Python raises ``RuntimeError("async generator ignored
        # GeneratorExit")`` and asyncio leaves the cleanup task
        # un-retrieved. Letting it propagate is the right thing.
        except Exception as e:
            elog(
                "agent.run_stream.error",
                level="error",
                exc_info=True,
                agent=self.name,
                user_id=user_id,
                session_id=session_id,
                error_type=type(e).__name__,
                error=str(e) or repr(e),
            )
            yield {"kind": "done", "text": _format_run_error(e)}

    async def _run_inner_stream(
        self,
        message: str,
        attachments: list[dict] | None,
        _status,
        session_id: str | None = None,
        model_override: BaseModel | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming variant of :meth:`_run_inner`.

        Mirrors the autoloop logic line-for-line but calls
        ``active_model.stream(...)`` instead of ``generate(...)``.
        Providers that don't override ``stream`` fall back to a single
        post-hoc yield via ``BaseModel.stream`` — voice-chat still works,
        just without the time-to-first-audio win.
        """
        await _status("Loading context...")
        system = self._combined_system_prompt(session_id=session_id)

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
        accumulated: list[str] = []
        iter_count = 0

        pending = hub.drain(session_id)
        if pending:
            pre = _format_shell_reminder(pending)
            current_input = f"{pre}\n\n{current_input}"

        # When the streaming autoloop yields zero deltas (claude_cli
        # tool-only turns, smart_router → claude_cli with empty content,
        # agno when no RunContentEvent fires), we fall back to a
        # one-shot generate() so callers always receive text. The real
        # ModelResponse from that call wins for last_response_meta()
        # over the synthetic placeholder.
        fallback_response: ModelResponse | None = None
        try:
            while True:
                iter_count += 1
                if iter_count > cap:
                    elog("agent.run_stream.autoloop_cap_hit",
                         session_id=session_id, cap=cap)
                    break

                messages: list[dict[str, Any]] = [{"role": "user", "content": current_input}]
                await _status("Thinking...")

                token = set_session_context(session_id)
                try:
                    # Pass session_id and on_status so SmartRouter.stream
                    # can run the same classifier + binding logic that
                    # ``generate`` uses. Without these, voice turns would
                    # route to "first enabled agno model" instead of the
                    # session's bound side, which 403'd on users whose
                    # first agno model was an OpenAI model their key
                    # couldn't access.
                    #
                    # Introspect once instead of try/except TypeError around
                    # the iteration body — a catch-all TypeError swallows
                    # errors raised mid-iteration (e.g. an SDK shape change
                    # inside ``claude_cli._run_once``) and the silent retry
                    # without ``session_id`` collides on the ``"default"``
                    # subprocess, which then yields zero deltas → fallback
                    # at line 1089 fires → caller sees ONE giant delta.
                    stream_kwargs: dict[str, Any] = {"system": system}
                    try:
                        sig_params = inspect.signature(
                            active_model.stream
                        ).parameters
                    except (TypeError, ValueError):
                        # Builtins / C-coded callables don't expose a
                        # signature. Skip the introspection — call with
                        # only the always-supported args.
                        sig_params = {}
                    if "session_id" in sig_params:
                        stream_kwargs["session_id"] = session_id
                    if "on_status" in sig_params:
                        stream_kwargs["on_status"] = _status
                    async for delta in active_model.stream(
                        messages, **stream_kwargs,
                    ):
                        if not delta:
                            continue
                        accumulated.append(delta)
                        yield {"kind": "delta", "text": delta}
                finally:
                    reset_session_context(token)

                events = hub.drain(session_id)
                if not events:
                    if not hub.has_running(session_id):
                        break
                    if wake_window > 0:
                        events = await hub.wait(session_id, timeout=wake_window)
                    if not events:
                        break

                # Force-flush any in-progress sentence before re-entering the
                # model — a sentence split by a tool call must not be
                # re-narrated. Risk #1 from the voice-chat plan.
                yield {"kind": "iteration_break"}

                elog(
                    "agent.run_stream.autoloop_iter",
                    session_id=session_id, iter=iter_count, events=len(events),
                )
                current_input = _format_shell_reminder(events)

            # Empty-stream safety net: some providers emit zero deltas
            # for tool-only turns, empty completions, or non-streamable
            # backends (claude-cli through smart_router). Without this
            # fallback voice mode (and the soon-to-be-streaming web
            # chat) would surface a confusing "(no output)" message
            # while ``Agent.run()`` worked fine for the same prompt.
            if not accumulated:
                elog(
                    "agent.run_stream.fallback_to_generate",
                    session_id=session_id,
                    reason="no_deltas_yielded",
                )
                try:
                    fallback_response = await active_model.generate(
                        [{"role": "user", "content": message}],
                        system=system,
                        tools=None,
                    )
                    _emit_tool_call_summary(
                        fallback_response,
                        session_id=session_id,
                        iter_count=iter_count,
                    )
                    fallback_text = (fallback_response.content or "").strip()
                    if fallback_text:
                        accumulated.append(fallback_text)
                        # Yield as a final delta so SentenceChunker /
                        # TTS / streaming clients see it the same way
                        # they would a normal delta.
                        yield {"kind": "delta", "text": fallback_text}
                except Exception as e:  # noqa: BLE001 — surface in log, return empty
                    fallback_response = None
                    elog(
                        "agent.run_stream.generate_fallback_failed",
                        level="warning",
                        session_id=session_id,
                        error_type=type(e).__name__,
                        error=str(e) or repr(e),
                    )
        finally:
            self._release_model_slot(active_model)

        full_text = "".join(accumulated)
        # Prefer the real ModelResponse from the generate() fallback so
        # last_response_meta() has accurate model + usage. Otherwise
        # synthesize a minimal stand-in from the accumulated text and
        # the *effective* model id — for SmartRouter this is the
        # runtime actually picked for the session, not a generic
        # instance attribute. ``getattr(active_model, "model_name",
        # None)`` (the previous code) returned ``None`` for every
        # provider in tree (claude_cli/agno expose ``self.model``;
        # SmartRouter exposes neither), which silently dropped the
        # model badge from the chat UI after the streaming migration.
        # ``effective_model_id`` is the provider-aware accessor.
        if fallback_response is not None:
            self._store_response_meta(session_id, fallback_response)
        else:
            model_id = active_model.effective_model_id(session_id)
            synthetic = ModelResponse(content=full_text, model=model_id)
            self._store_response_meta(session_id, synthetic)
        elog(
            "agent.run_stream.done",
            agent=self.name,
            session_id=session_id,
            model_class=type(active_model).__name__,
            response_len=len(full_text),
            used_fallback=fallback_response is not None,
        )
        yield {"kind": "done", "text": full_text}

    def _resolve_vault_path(self) -> str:
        """Return the on-disk path the vault MCP is actually using.

        Mirrors the gateway's resolution order
        ([openagent/gateway/api/vault.py]): a YAML-level
        ``memory.vault_path`` override wins, otherwise falls back to
        ``default_vault_path()`` (which already honours ``--agent-dir``
        via the ``_agent_dir`` global in :mod:`openagent.core.paths`).
        Returned as a string ready to splice into the framework prompt.
        """
        from pathlib import Path
        from openagent.core.paths import default_vault_path

        cfg_path = (
            (self.config or {}).get("memory", {}).get("vault_path")
        )
        if cfg_path:
            return str(Path(cfg_path).expanduser().resolve())
        return str(default_vault_path())

    def _combined_system_prompt(self, session_id: str | None = None) -> str:
        """Concatenate the framework prompt with the user's project-specific one.

        Substitutes ``{{OPENAGENT_VAULT_PATH}}`` in the framework prompt
        with the resolved on-disk path so the agent sees the exact folder
        it must use as memory (and can compare against any rogue path a
        wrapper SDK might inject). Per-agent because each agent runs in
        its own process with its own ``--agent-dir`` (and optional
        ``memory.vault_path`` YAML override).

        When ``session_id`` is provided we append a ``<session-id>`` tag
        so the LLM can learn its own id and pass it to tools that
        operate on "this session" — e.g.
        ``model-manager.pin_session(session_id=..., runtime_id=...)``.
        The tag is stripped of whitespace and comes last so project
        prompts read cleanly above it.
        """
        framework = FRAMEWORK_SYSTEM_PROMPT.replace(
            "{{OPENAGENT_VAULT_PATH}}", self._resolve_vault_path()
        )

        user = (self.system_prompt or "").strip()
        if not user:
            combined = framework
        else:
            combined = (
                framework
                + "\n\n── User-specific identity and project context ──\n\n"
                + user
            )
        if session_id:
            combined += f"\n\n<session-id>{session_id}</session-id>"
        return combined

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

        system = self._combined_system_prompt(session_id=session_id)
        messages: list[dict[str, Any]] = [{"role": "user", "content": message}]

        async for chunk in self.model.stream(messages, system=system):
            yield chunk

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.shutdown()
