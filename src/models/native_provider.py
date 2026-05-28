"""Native API provider for API models.

OpenAgent owns the *product* layer:
- provider/model catalog
- pricing / budget reporting
- gateway, channels, memory vault

The runtime owns the *execution* layer:
- API call execution
- session history persistence
- MCP tool orchestration (via runtime ``MCPTools`` instances supplied
  by the OpenAgent ``MCPPool`` — see ``openagent.mcp.pool``)

Tool wiring: this provider does NOT wrap MCP tools manually. It receives a
list of pre-connected ``MCPTools`` instances from the pool and passes
them straight to the runtime ``Agent``. The runtime handles the tool loop,
content-type serialisation (image artifacts, embedded resources, etc.), and
per-call scheduling. We only need to compute and mirror cost back into the
metrics so ``sessions.runs[*].metrics.cost`` stays queryable.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import importlib
import inspect
import logging
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from src.core.logging import elog
from src.models.base import BaseModel, ModelResponse
from src.models.catalog import (
    DEFAULT_ZAI_BASE_URL,
    FRAMEWORK_API_BASED,
    _iter_provider_entries,
    compute_cost,
    model_id_from_runtime,
    normalize_runtime_model_id,
    split_runtime_id,
)

logger = logging.getLogger(__name__)


# Per-coroutine sink for runtime ERROR log capture. The previous
# implementation installed a global root-logger handler per generate()
# call — under concurrent turns the handlers cross-pollinated and one
# session's error message could surface as another session's failure.
# A contextvar-backed handler isolates each capture to its own coroutine.
_capture_sink_var: contextvars.ContextVar[list[str] | None] = contextvars.ContextVar(
    "runtime_capture_sink", default=None,
)


class _ErrorCaptureHandler(logging.Handler):
    """Append ERROR-level log records to the contextvar-scoped sink.

    Installed exactly once at module import (see ``_install_capture_handler``)
    on the runtime's named logger — NOT the root logger — so we don't see
    every other library's errors and so add/remove churn under concurrent
    turns is gone.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)

    def emit(self, record: logging.LogRecord) -> None:
        sink = _capture_sink_var.get()
        if sink is None or record.levelno < logging.ERROR:
            return
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        # AGNO_LOG_TRACEBACKS=true attaches exc_info to log_error() calls
        # so we can ship the originating file/line out to the fleet log
        # (otherwise the runtime's formatted error string is all we get).
        if record.exc_info:
            try:
                import traceback as _tb
                tb_text = "".join(_tb.format_exception(*record.exc_info))
                msg = f"{msg}\n{tb_text}" if msg else tb_text
            except Exception:
                pass
        if msg:
            sink.append(msg)


def _install_capture_handler() -> None:
    """Idempotently attach :class:`_ErrorCaptureHandler` to the runtime's logger.

    The handler is scoped to a contextvar so each generate() call sees
    only the records emitted within its own coroutine. Safe to call
    repeatedly (e.g. after a logging.basicConfig rewires the root).
    """
    runtime_logger = logging.getLogger("openagent")
    if any(isinstance(h, _ErrorCaptureHandler) for h in runtime_logger.handlers):
        return
    runtime_logger.addHandler(_ErrorCaptureHandler())


@contextlib.contextmanager
def _capture_log_errors():
    """Scope a runtime ERROR-log sink to the current coroutine.

    The handler is installed once at module import; here we just
    publish a per-call sink via contextvar so concurrent generate()
    calls don't share records.
    """
    sink: list[str] = []
    token = _capture_sink_var.set(sink)
    try:
        yield sink
    finally:
        _capture_sink_var.reset(token)


# Configure runtime tracebacks ONCE at import time — used to run per
# generate() call which was global-state thrash on the hot path.
os.environ.setdefault("AGNO_LOG_TRACEBACKS", "true")
try:
    from src.core._runner.utils.log import set_log_tracebacks as _agno_set_log_tracebacks
    _agno_set_log_tracebacks(True)
except Exception:  # noqa: BLE001
    pass
_install_capture_handler()


@functools.lru_cache(maxsize=64)
def _model_class_accepted_params(cls: type) -> frozenset[str]:
    """Cache the parameter names accepted by a runtime model class.

    ``inspect.signature`` is expensive and was called from
    ``_construct_model`` on every model build — once per Team member,
    per session. The runtime class set is small and finite so an unbounded
    cache is fine; the LRU cap is just defensive against pathological
    test stubs.
    """
    return frozenset(inspect.signature(cls).parameters.keys())


_SESSION_ID_TAG_RE = re.compile(r"\n*<session-id>[^<]*</session-id>\s*$")

# Cap per-NativeProvider Agent and Team caches so a deployment with many
# distinct session ids (each baked into the system prompt) doesn't leak
# Agent objects forever.
_AGENT_CACHE_MAX = 64


@functools.lru_cache(maxsize=1)
def _agno_event_types() -> dict[str, tuple]:
    """Module-level cached tuples of runtime event types for isinstance checks.

    The runtime publishes ``RunContentEvent`` / ``ToolCall*Event`` under TWO
    modules — one for plain Agent runs and one for Team(mode=…) runs.
    Each call site needs the union; this helper builds the tuples ONCE
    so stream loops don't pay the import + tuple-rebuild cost on every
    iteration.

    Returns a dict with keys ``content``, ``tool_started``,
    ``tool_completed``, ``tool_error``. Team types are silently
    omitted on runtime builds that don't ship the team module.
    """
    from src.core._run_state.agent import (
        RunContentEvent as AgentRunContentEvent,
        ToolCallStartedEvent as AgentToolCallStartedEvent,
        ToolCallCompletedEvent as AgentToolCallCompletedEvent,
        ToolCallErrorEvent as AgentToolCallErrorEvent,
    )
    content: tuple = (AgentRunContentEvent,)
    tool_started: tuple = (AgentToolCallStartedEvent,)
    tool_completed: tuple = (AgentToolCallCompletedEvent,)
    tool_error: tuple = (AgentToolCallErrorEvent,)
    try:
        from src.core._run_state.team import (
            RunContentEvent as TeamRunContentEvent,
            ToolCallStartedEvent as TeamToolCallStartedEvent,
            ToolCallCompletedEvent as TeamToolCallCompletedEvent,
            ToolCallErrorEvent as TeamToolCallErrorEvent,
        )
    except ImportError:
        pass
    else:
        content = (AgentRunContentEvent, TeamRunContentEvent)
        tool_started = (AgentToolCallStartedEvent, TeamToolCallStartedEvent)
        tool_completed = (AgentToolCallCompletedEvent, TeamToolCallCompletedEvent)
        tool_error = (AgentToolCallErrorEvent, TeamToolCallErrorEvent)
    return {
        "content": content,
        "tool_started": tool_started,
        "tool_completed": tool_completed,
        "tool_error": tool_error,
    }


def _evict_oldest(cache: OrderedDict[str, Any], max_size: int) -> None:
    """Pop the oldest entries from ``cache`` so it doesn't exceed ``max_size``."""
    while len(cache) > max_size:
        cache.popitem(last=False)


def _system_cache_key(system: str | None) -> str:
    """Stable cache key for a system prompt that ignores the per-session
    ``<session-id>`` tag the orchestrator appends.

    Without this, every session generates a unique system prompt
    string, so ``_agno_agents`` / ``_agno_teams`` would miss on every
    session AND grow unbounded with session count.
    """
    if not system:
        return ""
    return _SESSION_ID_TAG_RE.sub("", system).strip()
_INCOMPATIBLE_TOOL_FAMILIES_BY_PROVIDER: dict[str, frozenset[str]] = {
    # DeepSeek v4 flash/pro chat completions are text-only on the official
    # API, so computer-control screenshot artifacts (image parts) fail
    # with "unknown variant image_url, expected text". Filter that toolkit
    # out up front instead of letting sessions crash mid-turn.
    "deepseek": frozenset({"computer_control"}),
}


class NativeProviderError(RuntimeError):
    """Raised when the runtime's underlying provider failed (e.g. OpenAI 403,
    rate-limit, model-not-allowed) but the runtime swallowed the exception
    and returned an empty response. We capture the ERROR log record(s) and
    re-raise with a user-readable message so the chat UI can show what
    actually went wrong instead of a silent placeholder."""


# Single per-process tempdir for tool-call image artifacts so each
# image doesn't leak its own tempdir into ``$TMPDIR``. Resolved on first
# call; cleaned up by the OS at reboot (we never created sub-dirs to
# clean up between, just files within one shared dir).
_AGNO_IMAGE_TMPDIR: str | None = None


def _agno_image_tmpdir() -> str:
    global _AGNO_IMAGE_TMPDIR
    if _AGNO_IMAGE_TMPDIR is None:
        import tempfile
        _AGNO_IMAGE_TMPDIR = tempfile.mkdtemp(prefix="oa_agno_img_")
    return _AGNO_IMAGE_TMPDIR


def _save_agno_image_to_disk(image: Any) -> tuple[str | None, str | None]:
    """Persist a runtime ``Image`` to a temp file. Returns ``(path, filename)``.

    Tries ``content`` (bytes) first, then ``filepath``, then ``url``.
    Returns ``(None, None)`` when no bytes can be resolved synchronously.
    Files land in a single process-wide tempdir so a vision-heavy
    session doesn't leak one tempdir per image.
    """
    import os
    from uuid import uuid4

    mime_type = getattr(image, "mime_type", None) or "image/png"
    fmt = getattr(image, "format", None)
    if not fmt:
        fmt = mime_type.split("/")[-1] if "/" in mime_type else "png"
    if fmt == "jpeg":
        fmt = "jpg"
    filename = getattr(image, "id", None) or f"image.{fmt}"
    if not filename.endswith(f".{fmt}"):
        filename = f"{filename}.{fmt}"

    content = getattr(image, "content", None)
    if content is not None:
        if isinstance(content, str):
            content = content.encode("utf-8")
    elif getattr(image, "filepath", None):
        fp = str(image.filepath)
        try:
            with open(fp, "rb") as f:
                content = f.read()
        except Exception:
            pass
    elif getattr(image, "url", None):
        try:
            import httpx
            content = httpx.get(str(image.url)).content
        except Exception:
            pass

    if not content:
        return None, None

    # Disambiguate per-image with a uuid prefix so two images named
    # ``image.png`` in the same tempdir don't clobber each other.
    safe_name = f"{uuid4().hex[:8]}-{filename}"
    path = os.path.join(_agno_image_tmpdir(), safe_name)
    with open(path, "wb") as f:
        f.write(content)
    return os.path.realpath(path), filename


def _is_error_status(status_obj: Any) -> bool:
    """True when a runtime RunOutput status indicates an error.

    Compares against ``RunStatus.error`` enum (preferred — survives
    enum-value renames). Falls back to the stringly-typed compare for
    the case where status_obj is already a bare string (legacy runtime
    versions / mock responses in tests).
    """
    if status_obj is None:
        return False
    try:
        from src.core._run_state.base import RunStatus
        if status_obj == RunStatus.error:
            return True
    except Exception:  # noqa: BLE001
        pass
    status_val = getattr(status_obj, "value", status_obj)
    return isinstance(status_val, str) and status_val.upper() == "ERROR"


def _extract_tool_names_from_agno_response(response: Any) -> list[str]:
    """Extract executed tool names from a runtime RunOutput.

    The runtime exposes ``RunOutput.tools: list[ToolExecution]``. We read
    that directly. Older defensive ``tool_executions`` /
    ``run_response.tools`` probes were removed — they hadn't matched
    real runtime output since the 1.x line and only hid shape drift.
    Returns ``[]`` on miss — tool telemetry is non-load-bearing.
    """
    tools = getattr(response, "tools", None)
    if not tools:
        return []
    names: list[str] = []
    for entry in tools:
        name = (
            entry.get("tool_name") if isinstance(entry, dict)
            else getattr(entry, "tool_name", None)
        )
        if name:
            names.append(str(name))
    return names


def _summarize_provider_errors(errs: list[str]) -> str:
    """Pick the most useful line from a list of captured ERROR log
    messages. The runtime emits a small flurry of three lines for one
    provider failure ("API status error", "Non-retryable model provider
    error", "Error in Team run") — the second one is the cleanest and
    shortest, so we prefer that pattern, then fall back to the last record.

    Falls back to joining all of them when nothing matches the known
    runtime phrasing.
    """
    if not errs:
        return "Provider returned no content"
    # Prefer the "Non-retryable model provider error: <reason>" line
    # since that's the cleanest provider-side message.
    for line in errs:
        if "Non-retryable model provider error:" in line:
            return line.split("Non-retryable model provider error:", 1)[1].strip() or line
    # Otherwise prefer the API status error which carries the HTTP code.
    for line in errs:
        if "API status error" in line or "status_code" in line.lower():
            return line.strip()
    # Fall back to the last error message.
    return errs[-1].strip()


# Set of env var names NativeProvider has already populated this process.
# Avoids redundant writes from per-member NativeProvider construction in
# TeamRouterProvider, and skips re-walking the providers list once a
# given providers_config has already been processed.
_INJECTED_ENV_KEYS: set[str] = set()
_PROVIDERS_CONFIG_INJECTED: set[int] = set()


def _set_env_key_once(env_var: str | None, value: str) -> None:
    if not env_var or not value:
        return
    if env_var in _INJECTED_ENV_KEYS:
        return
    if not os.environ.get(env_var):
        os.environ[env_var] = value
    _INJECTED_ENV_KEYS.add(env_var)


PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "zai": "ZAI_API_KEY",
    "zhipu": "ZAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
}
RUNTIME_PROVIDER_CLASSES: dict[str, tuple[str, str, dict[str, Any]]] = {
    "anthropic": ("src.models.providers.anthropic", "Claude", {}),
    "openai": ("src.models.providers.openai", "OpenAIChat", {}),
    "google": ("src.models.providers.google", "Gemini", {}),
    "groq": ("src.models.providers.groq", "Groq", {}),
    # ``_provider_overrides.DeepSeekTextOnly`` rewrites multimodal
    # attachments to text before they hit DeepSeek's chat completions
    # API — the official endpoint rejects ``{type:"file"}`` /
    # ``{type:"image_url"}`` / ``{type:"input_audio"}`` content parts
    # with ``unknown variant '<x>', expected 'text'``. Files become
    # inline ``<attachment>`` blocks (or a placeholder for binaries);
    # images / audio become a single ``<media-omitted>`` block listing
    # what was stripped.
    "deepseek": ("src.models._provider_overrides", "DeepSeekTextOnly", {}),
    "zai": ("src.models.providers.openai.like", "OpenAILike", {"name": "ZAI"}),
}


class NativeProvider(BaseModel):
    """API model provider backed by the runtime's session and tool orchestration."""

    history_mode = "platform"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        providers_config: dict | None = None,
        db_path: str | None = None,
        history_runs: int = 20,
    ):
        self._providers_config = providers_config or {}
        self.model = normalize_runtime_model_id(model, self._providers_config)
        self._api_key = api_key
        self._base_url = base_url
        self._db_path = db_path
        self._history_runs = history_runs
        # Pre-connected MCPTools instances supplied by MCPPool. Shared
        # across all NativeProvider instances under the same ModelDispatcher
        # so we don't spawn duplicate MCP server processes per entry model.
        self._mcp_toolkits: list[Any] = []
        # Memoized result of ``_compatible_mcp_toolkits`` so the per-turn
        # path doesn't re-walk the toolkit list.
        self._compatible_cache: tuple[list[Any], list[str]] | None = None
        # One runtime Agent per unique system prompt (the orchestrator
        # appends a per-session ``<session-id>`` tag, so practically
        # one entry per active session). Bounded LRU with
        # ``_AGENT_CACHE_MAX`` entries to cap memory growth — see
        # ``_evict_oldest``.
        self._agno_agents: OrderedDict[str, Any] = OrderedDict()
        self._agno_teams: OrderedDict[str, Any] = OrderedDict()
        # Memoised provider-config row for this model. Resolved once at
        # construction because the providers_config can't change for a
        # given NativeProvider instance (rebuild_routing rebuilds the
        # instance).
        self._provider_config_cache: dict[str, Any] | None = None
        # Path returned by ``_ensured_runtime_db_path`` after we've
        # mkdir()'d the parent — cached so the ensure-agent hot path
        # doesn't redo the stat+mkdir per cache miss.
        self._ensured_db_path: Path | None = None

        self._inject_provider_keys()

    def set_db(self, db) -> None:
        new_path = getattr(db, "db_path", self._db_path)
        if new_path == self._db_path:
            return
        self._db_path = new_path
        self._ensured_db_path = None
        # Force agent rebuild so the new SqliteDb path takes effect.
        self._agno_agents.clear()
        self._agno_teams.clear()

    def set_mcp_toolkits(self, toolkits: list[Any]) -> None:
        """Receive the pool's pre-connected ``MCPTools`` instances.

        Called by ``wire_model_runtime``. The pool owns lifecycle (entered
        once at agent startup, exited at shutdown); we just hold
        references. Always flushes the agent/team caches — the pool may
        swap toolkits in place across hot-reload so we can't safely
        skip rebuild even when the list looks identical.
        """
        self._mcp_toolkits = list(toolkits)
        self._compatible_cache = None
        # Force agent/team rebuild so the new tool list is picked up.
        self._agno_agents.clear()
        self._agno_teams.clear()

    async def close_session(self, session_id: str) -> None:
        """The runtime persists session history in DB but keeps no per-session subprocess."""
        return None

    async def forget_session(self, session_id: str) -> None:
        """Erase the runtime's stored history for ``session_id`` so the next
        call on that session id starts empty.

        Without this, ``add_history_to_context=True`` (see
        :meth:`_ensure_agent`) reloads the full prior transcript AND
        any rolling session summary on the next generate() — so the
        gateway's ``/clear`` and the scheduler's per-fire forget both
        silently break for runtime-backed models.

        Strategy: call the runtime's native ``SqliteDb.delete_session``
        (which drops the sessions row carrying both the transcript
        and the summary). Fall back to raw SQL if the API ever changes.
        The ``_agno_agents`` / ``_agno_teams`` caches don't need to
        be invalidated — the live ``Agent`` re-reads history from the
        DB on every ``arun()``, so deletion takes effect immediately.
        Agentic memory rows (``runtime_memories``) are user-scoped by
        design; wiping them per-session would contradict the product
        model (user-level preferences should survive /clear).
        """
        if not session_id:
            return
        try:
            from src.memory.store.sqlite import SqliteDb
        except ImportError:
            return

        db_path = self._runtime_db_path()
        import asyncio as _asyncio

        def _delete_via_api() -> bool:
            """Sync helper: SqliteDb.delete_session is sync end-to-end."""
            try:
                db = SqliteDb(db_file=db_path)
            except Exception as e:  # noqa: BLE001
                logger.debug("runtime forget_session: SqliteDb init failed: %s", e)
                return False
            delete_fn = getattr(db, "delete_session", None)
            if not callable(delete_fn):
                return False
            try:
                result = delete_fn(session_id=session_id)
                # Defensive — the runtime may switch to async one day.
                if inspect.isawaitable(result):
                    return False
                return True
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "runtime delete_session failed for %s: %s", session_id, e,
                )
                return False

        deleted_via_api = await _asyncio.to_thread(_delete_via_api)
        if not deleted_via_api:
            await self._agno_fallback_sql_forget(db_path, session_id)
        elog("runtime.session_forget", session_id=session_id, db=db_path)

    async def _agno_fallback_sql_forget(
        self, db_path: str, session_id: str
    ) -> None:
        """Raw-SQL delete for when the runtime's ``SqliteDb.delete_session``
        is unavailable or errors out.

        Current schema keeps session history and summary in ``sessions``;
        legacy installs may still carry it as ``sessions`` if the
        rename migration hasn't run yet on that file. We DELETE from
        each candidate and catch the "no such table" OperationalError
        instead of pre-scanning ``sqlite_master`` — a missing table is
        a no-op. ``timeout=2.0`` avoids hanging the gateway ``/clear``
        request when a writer holds the lock. Runs on a worker thread
        so the sqlite I/O doesn't block the event loop.
        """
        import asyncio as _asyncio
        import sqlite3

        candidate_tables = ("sessions", "agno_sessions")

        def _run_sync() -> None:
            try:
                conn = sqlite3.connect(db_path, timeout=2.0)
            except Exception as e:
                logger.debug("runtime fallback sql: connect %s failed: %s", db_path, e)
                return
            try:
                cur = conn.cursor()
                for table in candidate_tables:
                    try:
                        cur.execute(
                            f"DELETE FROM {table} WHERE session_id = ?",
                            (session_id,),
                        )
                    except sqlite3.OperationalError as e:
                        msg = str(e).lower()
                        if "no such table" not in msg:
                            logger.debug(
                                "runtime fallback delete from %s for %s failed: %s",
                                table, session_id, e,
                            )
                conn.commit()
            finally:
                conn.close()

        await _asyncio.to_thread(_run_sync)

    def _provider_name(self) -> str:
        return split_runtime_id(self.model)[0]

    def _inject_provider_keys(self) -> None:
        """Mirror configured API keys into ``os.environ`` for the runtime's
        provider classes (which read from env, not constructor args).

        Module-level deduplication: every key set is tracked in
        :data:`_INJECTED_ENV_KEYS` so a fresh NativeProvider per Team
        member per session doesn't re-walk the providers list or
        re-write the same env vars. Existing env values are not
        overwritten — once a key wins, it wins. (Hot-rotating an API
        key requires a restart; documented limitation.)
        """
        provider_name = self._provider_name()
        if self._api_key:
            _set_env_key_once(PROVIDER_ENV_VARS.get(provider_name), self._api_key)
            if provider_name == "google":
                _set_env_key_once("GEMINI_API_KEY", self._api_key)

        # Only API-based provider rows carry api_keys worth exporting.
        # claude-cli rows have api_key=NULL by schema. The module-level
        # dedupe means we only do the providers-list walk on the FIRST
        # NativeProvider per (config-hash, provider) pair.
        config_id = id(self._providers_config)
        if config_id not in _PROVIDERS_CONFIG_INJECTED:
            for entry in _iter_provider_entries(self._providers_config):
                if entry.get("framework", FRAMEWORK_API_BASED) != FRAMEWORK_API_BASED:
                    continue
                name = str(entry.get("name") or "").strip()
                key = entry.get("api_key")
                if not name or not key:
                    continue
                _set_env_key_once(PROVIDER_ENV_VARS.get(name), key)
                if name == "google":
                    _set_env_key_once("GEMINI_API_KEY", key)
            _PROVIDERS_CONFIG_INJECTED.add(config_id)

        if self._base_url and provider_name == "openai":
            _set_env_key_once("OPENAI_BASE_URL", self._base_url)

    def _runtime_db_path(self) -> str:
        if self._db_path:
            return str(self._db_path)
        from src.core.paths import default_db_path

        return str(default_db_path())

    def _ensured_runtime_db_path(self) -> Path:
        """Return the runtime DB path with its parent dir ensured.

        Caches the ensured-once result on the instance so the
        ``_ensure_agent`` / ``_ensure_team`` hot paths don't repeat
        stat()+mkdir() on every cache miss. ``set_db`` invalidates
        the cache by zeroing ``_ensured_db_path``.
        """
        if self._ensured_db_path is not None:
            return self._ensured_db_path
        path = Path(self._runtime_db_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        self._ensured_db_path = path
        return path

    async def commit_partial_assistant(self, session_id: str, text: str) -> None:
        """Append a synthetic assistant run to the runtime's session row.

        After barge-in, the in-flight runtime run is cancelled mid-stream and
        nothing has been committed to the sessions table's runs column. We
        append a synthetic ``RunOutput`` with ``status=cancelled`` carrying
        the partial assistant text, so the next turn (which reads the session
        via ``add_history_to_context=True``) sees ``user → assistant
        (interrupted) → user`` rather than two adjacent user turns.

        Best-effort: missing session row, schema mismatch, or import
        errors all degrade silently — barge-in must succeed even when
        history persistence fails.
        """
        if not session_id or not text:
            return
        try:
            from src.memory.store.sqlite import SqliteDb
            from src.memory.store.base import SessionType
            from src.memory.sessions.agent import AgentSession
            from src.core._run_state.agent import RunOutput
            from src.core._run_state.base import RunStatus
            from src.models.providers.message import Message
        except ImportError as e:
            logger.debug("runtime commit_partial_assistant: import failed: %s", e)
            return

        db_path = self._runtime_db_path()

        def _commit_sync() -> None:
            # Sync helper: the runtime's SqliteDb is sync end-to-end. Run on a
            # worker thread so barge-in doesn't stall the event loop.
            try:
                db = SqliteDb(db_file=db_path)
            except Exception as e:  # noqa: BLE001
                logger.debug("runtime commit_partial_assistant: SqliteDb init failed: %s", e)
                return

            try:
                session = db.get_session(
                    session_id=session_id,
                    session_type=SessionType.AGENT,
                    deserialize=True,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("runtime commit_partial_assistant: get_session %s: %s", session_id, e)
                return

            if not isinstance(session, AgentSession):
                return

            import time as _time
            # ``agent_id`` is REQUIRED on the run dict — the runtime's
            # ``AgentSession.from_dict`` filters out any run that lacks both
            # ``agent_id`` and ``team_id`` when deserialising. Without it,
            # our synth run round-trips through SqliteDb and silently
            # disappears on the next read.
            run = RunOutput(
                run_id=f"interrupted-{int(_time.time() * 1000)}",
                agent_id=session.agent_id,
                session_id=session_id,
                user_id=session.user_id,
                content=text,
                messages=[Message(role="assistant", content=text)],
                status=RunStatus.cancelled,
            )
            try:
                session.upsert_run(run)
            except Exception as e:  # noqa: BLE001
                logger.debug("runtime commit_partial_assistant: upsert_run failed: %s", e)
                return
            try:
                db.upsert_session(session, deserialize=False)
                elog("runtime.barge_in_commit", session_id=session_id, chars=len(text))
            except Exception as e:  # noqa: BLE001
                logger.debug("runtime commit_partial_assistant: upsert_session failed: %s", e)

        import asyncio as _asyncio
        await _asyncio.to_thread(_commit_sync)

    def _runtime_parts(self) -> tuple[str, str]:
        return split_runtime_id(self.model)

    @staticmethod
    def _toolkit_family_name(toolkit: Any) -> str:
        return str(
            getattr(toolkit, "tool_name_prefix", None)
            or getattr(toolkit, "name", None)
            or "default"
        )

    def _compatible_mcp_toolkits(self) -> tuple[list[Any], list[str]]:
        if self._compatible_cache is not None:
            allowed, filtered = self._compatible_cache
            return list(allowed), list(filtered)
        provider_name, _model_id = self._runtime_parts()
        blocked = _INCOMPATIBLE_TOOL_FAMILIES_BY_PROVIDER.get(provider_name, frozenset())
        if not blocked:
            self._compatible_cache = (list(self._mcp_toolkits), [])
            return list(self._mcp_toolkits), []
        allowed: list[Any] = []
        filtered: set[str] = set()
        for toolkit in self._mcp_toolkits:
            family = self._toolkit_family_name(toolkit)
            if family in blocked:
                filtered.add(family)
                continue
            allowed.append(toolkit)
        self._compatible_cache = (allowed, sorted(filtered))
        return list(allowed), sorted(filtered)

    @staticmethod
    def _is_session_corruption_error(error_msg: str) -> bool:
        lowered = error_msg.lower()
        return (
            "tool_call_id" in lowered
            or "deserialize the json body" in lowered
            or "missing field" in lowered
        )

    def _rewrite_provider_error_detail(self, detail: str) -> str:
        provider_name, _model_id = self._runtime_parts()
        lowered = (detail or "").lower()
        if provider_name == "deepseek" and "unknown variant image_url" in lowered:
            return (
                "DeepSeek's official chat API only accepts text message content, "
                "so computer-control screenshots are incompatible with this "
                "model. Use a non-DeepSeek model for computer-control / GUI tasks."
            )
        if provider_name == "deepseek" and "unknown variant `file`" in lowered:
            # ``DeepSeekTextOnly`` should already inline file attachments
            # as text — if we still see this, an unpatched DeepSeek class
            # leaked through, or a member is using a different OpenAI-like
            # provider that also rejects file parts.
            return (
                "DeepSeek's chat API doesn't accept file attachments as "
                "content parts. File contents are normally inlined as text "
                "for DeepSeek; if you see this, the file may be too large "
                "or in a binary-only format. Switch to anthropic/openai for "
                "richer file analysis."
            )
        # 402 / Insufficient Balance is a billing problem — it won't
        # clear on retry, and a scheduled workflow hammering it just
        # creates writer-lock contention on the agent's SQLite file.
        # Surface a clear, actionable message so the operator knows
        # exactly which provider to refill (or to swap out of the
        # workflow's model config).
        if "insufficient balance" in lowered or "error code: 402" in lowered:
            return (
                f"Provider '{provider_name}' returned HTTP 402 (Insufficient "
                f"Balance). This is a billing issue — refill the provider "
                f"account or change the workflow's model to a different "
                f"provider. Retrying won't help."
            )
        return detail

    def _provider_config(self) -> dict[str, Any]:
        """Return the API-based provider entry matching this model's vendor.

        v0.12 stores providers as a flat list where the same vendor can
        appear twice (API-based + claude-cli). NativeProvider only cares
        about the API-based row — the claude-cli row lives in its own
        registry. Falls back to the legacy dict-shape for early-boot / tests.

        Memoised because providers_config is immutable per instance —
        ``rebuild_routing`` constructs a fresh NativeProvider rather than
        mutating an existing one. Caching saves an O(N_providers) walk
        per ``_build_runtime_model`` (and there are 2 per turn — see
        ``_resolved_api_key`` and ``_resolved_base_url``).
        """
        if self._provider_config_cache is not None:
            return self._provider_config_cache
        provider_name, _ = self._runtime_parts()
        for entry in _iter_provider_entries(self._providers_config):
            if str(entry.get("name") or "").strip() != provider_name:
                continue
            if entry.get("framework", FRAMEWORK_API_BASED) != FRAMEWORK_API_BASED:
                continue
            self._provider_config_cache = dict(entry)
            return self._provider_config_cache
        self._provider_config_cache = {}
        return self._provider_config_cache

    def _provider_setting(self, key: str) -> str | None:
        value = self._provider_config().get(key)
        return str(value).strip() if value is not None else None

    def _resolved_api_key(self) -> str | None:
        return self._api_key or self._provider_setting("api_key")

    def _resolved_base_url(self) -> str | None:
        provider_name, _ = self._runtime_parts()
        if self._base_url:
            return self._base_url
        if provider_name == "zai":
            return self._provider_setting("base_url") or DEFAULT_ZAI_BASE_URL
        return self._provider_setting("base_url")

    def _construct_model(self, cls: type, **kwargs: Any) -> Any:
        accepted = _model_class_accepted_params(cls)
        filtered = {k: v for k, v in kwargs.items() if v is not None and k in accepted}
        return cls(**filtered)

    def _load_runtime_model_class(self, provider_name: str) -> tuple[type | None, dict[str, Any]]:
        spec = RUNTIME_PROVIDER_CLASSES.get(provider_name)
        if not spec:
            return None, {}
        module_name, class_name, extra_kwargs = spec
        module = importlib.import_module(module_name)
        return getattr(module, class_name), dict(extra_kwargs)

    def build_runtime_model(self) -> Any:
        """Construct the underlying ``Model`` instance for this runtime.

        Used by ``TeamRouterProvider`` to build the routing classifier
        without instantiating a full ``Agent`` wrapper.
        """
        provider_name, model_id = self._runtime_parts()
        api_key = self._resolved_api_key()
        base_url = self._resolved_base_url()
        model_class, extra_kwargs = self._load_runtime_model_class(provider_name)
        if model_class is not None:
            return self._construct_model(
                model_class,
                id=model_id,
                api_key=api_key,
                base_url=base_url,
                **extra_kwargs,
            )

        from src.models.providers.utils import get_model

        return get_model(self.model)

    _build_runtime_model = build_runtime_model  # internal alias for back-compat

    def _missing_dependency_hint(self, exc: ImportError) -> str:
        detail = str(exc) or exc.__class__.__name__
        return (
            "Runtime dependencies are incomplete. "
            "Install OpenAgent's API-model dependencies (for example "
            "`sqlalchemy`, provider SDKs like `openai`/`anthropic`/`google-genai`) "
            f"and retry. Original import error: {detail}"
        )

    def _ensure_agent(self, system: str | None = None):
        """Lazily construct one runtime Agent per unique system prompt.

        ``system_message`` is set at construction time so OpenAgent's
        framework prompt reaches OpenAI as a real ``system`` role
        message. The orchestrator appends a per-session ``<session-id>``
        tag to the system prompt so each session generates a unique
        key — meaning each session builds its own Agent. The cache is
        capped at :data:`_AGENT_CACHE_MAX` with LRU eviction to bound
        memory growth; eviction is enforced here AND in close_session
        so long-running deployments don't leak Agents per dead session.
        """
        sys_key = (system or "").strip()
        cached = self._agno_agents.get(sys_key)
        if cached is not None:
            self._agno_agents.move_to_end(sys_key)
            return cached
        try:
            from src.core._runner.agent import Agent as RuntimeAgent
            from src.memory.store.sqlite import SqliteDb
        except ImportError as exc:
            raise RuntimeError(self._missing_dependency_hint(exc)) from exc

        db_path = self._ensured_runtime_db_path()
        compatible_toolkits, filtered_families = self._compatible_mcp_toolkits()
        if filtered_families:
            elog("runtime.toolkits_filtered",
                model=self.model,
                filtered_families=filtered_families,
                runner="agent",
            )
        agent_tools: list[Any] = list(compatible_toolkits)
        agent = RuntimeAgent(
            model=self._build_runtime_model(),
            db=SqliteDb(db_file=str(db_path)),
            tools=agent_tools,
            system_message=sys_key or None,
            add_history_to_context=True,
            num_history_runs=self._history_runs,
            # The runtime maintains a rolling summary of older turns in the same
            # SqliteDb and injects it into context on each call. Combined
            # with ``num_history_runs=20`` this gives us long-horizon
            # recall without blowing the token budget on raw transcript.
            enable_session_summaries=True,
            add_session_summary_to_context=True,
            # Agentic memory is disabled — OpenAgent uses the vault for
            # user-scoped persistence. Keeping it off avoids the legacy agno_memories
            # table creation and keeps all state in the sessions table.
            enable_agentic_memory=False,
            markdown=False,
        )
        self._agno_agents[sys_key] = agent
        _evict_oldest(self._agno_agents, _AGENT_CACHE_MAX)
        return agent

    def _tool_families(self) -> dict[str, list[Any]]:
        """Group connected MCP toolkits by their server prefix.

        Each ``MCPTools`` instance wraps exactly one MCP server, so the
        prefix doubles as a natural "tool family" identifier for Team-mode
        routing. Uses ``tool_name_prefix`` → ``name`` → ``"default"`` as
        the resolution order. Empty dict when no toolkits are connected.
        """
        compatible_toolkits, _filtered = self._compatible_mcp_toolkits()
        families: dict[str, list[Any]] = {}
        for tk in compatible_toolkits:
            family = self._toolkit_family_name(tk)
            families.setdefault(family, []).append(tk)
        return families

    def _ensure_team(self, system: str):
        """Lazily construct a runtime Team in route mode for the main agent.

        Returns ``None`` when the team path is not applicable:
        - empty system prompt (classifier / no-framework-prompt calls)
        - fewer than 2 connected MCP tool families (nothing to route between)
        - Team import failure (older runtime or missing extra)

        The team leader carries the framework prompt and the full
        toolkit list as a safety net. Members are thin specialists:
        each owns one family's toolkit and a short role blurb the
        router uses to pick between them. Members inherit the
        framework prompt too so when the leader synthesises their
        output the persona is preserved.
        """
        sys_key = (system or "").strip()
        if not sys_key:
            return None
        cached = self._agno_teams.get(sys_key)
        if cached is not None:
            self._agno_teams.move_to_end(sys_key)
            return cached

        compatible_toolkits, filtered_families = self._compatible_mcp_toolkits()
        families = self._tool_families()
        if len(families) < 2:
            return None
        if filtered_families:
            elog("runtime.toolkits_filtered",
                model=self.model,
                filtered_families=filtered_families,
                runner="team",
            )

        try:
            from src.core._runner.agent import Agent as RuntimeAgent
            from src.memory.store.sqlite import SqliteDb
            from src.core._runner.team import Team, TeamMode
        except ImportError as exc:
            # Older runtime builds without the team module — fall back to
            # single-agent transparently instead of crashing the session.
            elog("runtime.team.unavailable", model=self.model, error=str(exc))
            return None

        db_path = self._ensured_runtime_db_path()

        members: list[Any] = []
        for family, toolkits in families.items():
            member_role = (
                f"Specialist for the {family} MCP server. "
                f"Handles requests best solved with {family} tools. "
                f"If a request falls outside that area, say so briefly."
            )
            member_system = (
                f"{sys_key}\n\n--- Role ---\nYou are the {family} "
                f"specialist. Prefer {family} tools; defer to the team "
                f"leader if the request is outside your area."
            )
            member = RuntimeAgent(
                model=self._build_runtime_model(),
                tools=list(toolkits),
                system_message=member_system,
                name=f"{family}_specialist",
                role=member_role,
                markdown=False,
            )
            members.append(member)

        # Leader gets the full toolkit list as a safety net so if gpt-4o-mini
        # (or any routing-weak model) decides to answer directly instead of
        # delegating, it still has the tools it needs. In normal route-mode
        # operation the leader delegates to one specialist and never calls
        # these itself — so the fallback costs nothing on the happy path.
        leader_tools: list[Any] = list(compatible_toolkits)

        try:
            team = Team(
                members=members,
                mode=TeamMode.route,
                model=self._build_runtime_model(),
                db=SqliteDb(db_file=str(db_path)),
                tools=leader_tools,
                system_message=sys_key,
                # Surface each member's tool list in the leader's context so
                # routing decisions see capabilities, not just the member's
                # short role blurb. Without this, weak routing models tend
                # to answer directly ("I cannot access files") when the
                # member name alone doesn't make the match obvious.
                add_member_tools_to_context=True,
                add_history_to_context=True,
                num_history_runs=self._history_runs,
                enable_session_summaries=True,
                add_session_summary_to_context=True,
                enable_agentic_memory=False,
                markdown=False,
            )
        except Exception as exc:
            # If Team construction fails for any reason (signature drift
            # between runtime versions, model incompatibility, …), log and
            # fall back rather than breaking the main generate path.
            elog("runtime.team.build_failed",
                model=self.model,
                error_type=type(exc).__name__,
                error=str(exc),
                families=sorted(families.keys()),
            )
            return None

        elog("runtime.team.built",
            model=self.model,
            families=sorted(families.keys()),
            member_count=len(members),
        )
        self._agno_teams[sys_key] = team
        _evict_oldest(self._agno_teams, _AGENT_CACHE_MAX)
        return team

    def _flatten_messages(self, messages: list[dict[str, Any]]) -> str:
        """Render conversation turns as a single user-side prompt for ``arun``.

        The system prompt is NOT included here — it's set on the runtime Agent via
        ``system_message`` (see ``_ensure_agent``) so OpenAI receives it as a
        real ``system`` role message. Including it here would duplicate it as
        user text, undoing the fix that makes procedural instructions
        authoritative for weak models.

        For text-only providers (DeepSeek), ``image_url`` content parts are
        stripped so the API never receives a message variant it can't parse.
        """
        provider_name, _model_id = self._runtime_parts()
        strip_images = provider_name == "deepseek"
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            if isinstance(content, list):
                texts: list[str] = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text" and part.get("text"):
                            texts.append(str(part["text"]))
                        elif strip_images and part.get("type") == "image_url":
                            texts.append("[image omitted]")
                        else:
                            texts.append(str(part))
                    else:
                        texts.append(str(part))
                content = " ".join(texts)
            else:
                content = str(content or "")
            if role == "user":
                parts.append(content)
            elif role == "assistant":
                parts.append(f"[Assistant] {content}")
            elif role == "tool":
                parts.append(f"[Tool:{msg.get('name', 'tool')}] {content}")
        return "\n\n".join(part for part in parts if part).strip()

    @staticmethod
    def _metrics_to_dict(metrics: Any) -> dict[str, Any]:
        """Coerce the runtime's metrics (dataclass / Pydantic / dict / object) into a dict.

        The runtime returns ``response.metrics`` as a ``RunMetrics`` dataclass;
        older versions returned a dict or a Pydantic model. Normalise so token
        and cost extraction works across versions.
        """
        if metrics is None:
            return {}
        if isinstance(metrics, dict):
            return metrics
        if hasattr(metrics, "model_dump"):
            try:
                return metrics.model_dump()
            except Exception:
                pass
        if hasattr(metrics, "__dataclass_fields__"):
            from dataclasses import asdict
            try:
                return asdict(metrics)
            except Exception:
                pass
        if hasattr(metrics, "__dict__"):
            return {k: v for k, v in vars(metrics).items() if not k.startswith("_")}
        return {}

    @staticmethod
    def _extract_metric(data: dict[str, Any], *names: str) -> int:
        for name in names:
            value = data.get(name)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return 0

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
        model_override: str | None = None,
    ) -> ModelResponse:
        """Run a single turn through the runtime.

        Note: ``tools`` is accepted for ``BaseModel`` compatibility but
        ignored — the runtime's Agent already holds the configured
        ``MCPTools`` instances via ``_mcp_toolkits`` and runs the full
        tool loop internally (including image-artifact extraction so
        screenshots no longer blow up the context).

        Media kwargs (``files``/``images``/``audio``/``videos``) are
        forwarded as ``runtime.arun(..., files=..., images=..., ...)``
        so the runtime's native multimodal handling applies (Team
        propagates them to members during ``delegate_task_to_member``).
        """
        prompt = self._flatten_messages(messages)
        sid = session_id or "default"
        # Prefer a Team when the caller supplied a system prompt AND we
        # have ≥2 MCP tool families to route between. Classifier/no-system
        # calls stay on single Agent so they don't pay the leader-routing
        # round trip. _ensure_team() returns None whenever the team path
        # is not applicable, so this collapses to the single-agent path
        # in minimal deployments.
        compatible_toolkits, filtered_families = self._compatible_mcp_toolkits()
        runner = self._ensure_team(system=system or "")
        runner_kind = "team"
        if runner is None:
            runner = self._ensure_agent(system=system)
            runner_kind = "agent"
        elog("runtime.request",
            model=self.model,
            session_id=sid,
            prompt_len=len(prompt),
            mcp_toolkits=len(compatible_toolkits),
            filtered_families=filtered_families or None,
            runner=runner_kind,
        )

        if on_status:
            try:
                await on_status("Thinking...")
            except Exception as e:  # noqa: BLE001
                logger.debug("on_status('Thinking...') raised: %s", e)

        # Note: AGNO_LOG_TRACEBACKS + set_log_tracebacks
        # are configured once at module import so we don't pay the env
        # dict + import lookup on every turn (used to run per generate()).

        # Capture ERROR-level log records from any logger during the run so
        # that when the runtime swallows a provider failure (e.g. OpenAI 403,
        # rate-limit, model-not-allowed), we can surface the underlying
        # message to the user instead of returning the silent empty-result
        # placeholder. The runtime logs the failure with strings like
        # "API status error from OpenAI API: Error code: 403 - ..." and
        # "Non-retryable model provider error: ...", but does not re-raise.
        with _capture_log_errors() as captured_errors:
            # ``user_id`` is required by ``enable_agentic_memory``; we use a
            # stable constant so memory accumulates for the single-tenant
            # OpenAgent deployment. Multi-tenant deployments can wire a
            # real user_id through BaseModel.generate() in a follow-up.
            arun_kwargs: dict[str, Any] = {"session_id": sid, "user_id": "openagent"}
            if files:
                arun_kwargs["files"] = files
            if images:
                arun_kwargs["images"] = images
            if audio:
                arun_kwargs["audio"] = audio
            if videos:
                arun_kwargs["videos"] = videos
            try:
                response = await runner.arun(prompt, **arun_kwargs)
            except Exception as e:
                raw_error = str(e) or repr(e)
                # When a previous run left a message with a missing
                # ``tool_call_id`` (e.g. an interrupted tool call),
                # the runtime's ``add_history_to_context=True`` injects it
                # into the API request and the provider's
                # deserialization fails.  Wipe the corrupted session
                # and retry once with a clean history.
                if sid != "default" and self._is_session_corruption_error(raw_error):
                    elog("runtime.session_corrupt",
                        session_id=sid,
                        error=raw_error[:300],
                    )
                    try:
                        await self.forget_session(sid)
                    except Exception as fe:
                        logger.debug("runtime session recovery forget %s: %s", sid, fe)
                    response = await runner.arun(prompt, **arun_kwargs)
                else:
                    rewritten_error = self._rewrite_provider_error_detail(raw_error)
                    elog("runtime.error",
                        model=self.model,
                        session_id=sid,
                        error_type=type(e).__name__,
                        error=rewritten_error,
                    )
                    if rewritten_error != raw_error:
                        raise NativeProviderError(rewritten_error) from e
                    raise

        # The runtime's Agent.arun / Team.arun catches generic exceptions, sets
        # ``run_response.status = RunStatus.error`` AND
        # ``run_response.content = str(e)``, then RETURNS the response
        # instead of re-raising (see _runner/agent/_run.py:666-696 and
        # _runner/team/_run.py around RunStatus.error assignments). Without
        # this check, openagent reads ``.content`` as a normal model
        # output and the bridge ships the raw Python exception string
        # (e.g. ``[Errno 2] No such file or directory``) to the user as
        # if the model had written it. Translate the status signal back
        # into an NativeProviderError so the surrounding empty-stream /
        # bridge layers format a clean error message.
        status_obj = getattr(response, "status", None)
        if _is_error_status(status_obj):
            raw_error = (getattr(response, "content", None) or "").strip() \
                or "runtime run finished with status=error and no detail"
            detail = self._rewrite_provider_error_detail(raw_error)
            # Ship the captured ERROR-level log records (with tracebacks
            # if AGNO_LOG_TRACEBACKS hooked them in) so the next fleet
            # tick can root-cause WHY the runtime failed instead of guessing
            # from the rewritten ``[Errno 2]`` string alone. Truncated
            # per-entry to keep the event row JSON-friendly.
            captured_snapshot = [c[:2000] for c in captured_errors[-3:]]
            elog("runtime.run_status_error",
                level="error",
                model=self.model,
                session_id=sid,
                detail=detail[:300],
                captured=captured_snapshot,
            )
            raise NativeProviderError(detail)

        # The runtime occasionally returns a response whose ``.content`` is None or
        # empty string. Two cases to distinguish:
        #   1. A *legitimate* tool-only turn — no error logs captured. Fall
        #      back to the placeholder so bridges don't ship zero bytes.
        #   2. A *swallowed provider failure* — the runtime logged ERRORs but
        #      did not re-raise (this is the OpenAI 403 / rate-limit case the
        #      user reported). Raise a clean exception with the captured
        #      message so ``agent.run()`` formats it for the chat UI.
        raw_content = getattr(response, "content", None)
        if raw_content:
            content = raw_content
        else:
            if captured_errors:
                detail = self._rewrite_provider_error_detail(
                    _summarize_provider_errors(captured_errors)
                )
                elog("runtime.provider_error",
                    level="error",
                    model=self.model,
                    session_id=sid,
                    detail=detail[:300],
                )
                raise NativeProviderError(detail)
            elog("runtime.empty_result",
                level="warning",
                model=self.model,
                session_id=sid,
                response_type=type(response).__name__,
                response=repr(response)[:200],
            )
            content = "(Done — no final message was returned.)"

        if on_status:
            tools = getattr(response, "tools", None) or []
            # Symmetric with the streaming path: emit a (started, completed)
            # pair per tool. ``generate()`` runs the turn to completion so
            # the ToolExecution already carries the final ``result`` — for
            # the started frame we shallow-clone it and null ``result`` so
            # the UI's local phase derivation (result presence → completed)
            # reads it as "running". Bridges that use ``generate()``
            # otherwise saw only the completed state and couldn't show a
            # spinner.
            for tool_exec in tools:
                await self._emit_agno_tool_status(
                    on_status, tool_exec, phase="started",
                )
                await self._emit_agno_tool_status(on_status, tool_exec)

        metrics_obj = getattr(response, "metrics", None)
        metrics_dict = self._metrics_to_dict(metrics_obj)

        # Trace event so we can debug if the runtime changes the metrics shape again.
        elog("runtime.metrics.shape",
            model=self.model,
            session_id=sid,
            type=type(metrics_obj).__name__ if metrics_obj is not None else "None",
            keys=sorted(metrics_dict.keys())[:12] if metrics_dict else [],
        )

        input_tokens = self._extract_metric(metrics_dict, "input_tokens", "prompt_tokens", "input")
        output_tokens = self._extract_metric(metrics_dict, "output_tokens", "completion_tokens", "output")
        stop_reason = metrics_dict.get("stop_reason")

        # Compute cost from OpenAgent's catalog and mirror it back into the runtime's
        # metrics so SessionMetrics.cost aggregation works for free.
        cost = self._compute_and_mirror_cost(
            metrics_obj=metrics_obj,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            session_id=sid,
        )

        elog("runtime.generate",
            model=self.model,
            session_id=sid,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            stop_reason=stop_reason or "stop",
        )
        tool_names_called = _extract_tool_names_from_agno_response(response)
        return ModelResponse(
            content=content,
            tool_names_called=tool_names_called,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason or "stop",
            model=self.model,
        )

    async def _emit_agno_tool_status(
        self,
        on_status: Callable[[str], Awaitable[None]] | None,
        tool_exec: Any | None,
        *,
        error_text: str | None = None,
        phase: str | None = None,
    ) -> None:
        """Forward a runtime ``ToolExecution`` as a JSON status frame.

        Thin wrapper that delegates to the shared
        :func:`src.models._tool_status.emit_tool_status` so live
        streaming, generate-time emissions, and the dispatcher's
        team-path emitter all use the same encoder + error handling.
        """
        from src.models._tool_status import emit_tool_status

        await emit_tool_status(
            on_status, tool_exec, error_text=error_text, phase=phase,
        )

    @staticmethod
    def _save_image(image: Any) -> str | None:
        """Write a runtime ``Image`` (from stream content or tool result) to a
        temp file. Returns a ``[IMAGE:/path]`` marker that
        ``parse_response_markers`` extracts downstream, or ``None`` when
        the image has no bytes.
        """
        path, _filename = _save_agno_image_to_disk(image)
        if path is None:
            return None
        return f"\n[IMAGE:{path}]\n"

    async def stream(
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
        model_override: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream content deltas via the runtime's native ``stream=True`` path.

        Yields raw text strings as they arrive from the LLM. Tool-call
        events are forwarded to ``on_status`` in the same JSON format
        used by claude-cli, and image/file artifacts from tool results
        are written to disk and yielded as ``[IMAGE:/path]`` markers.
        On any failure, falls back to :meth:`generate` and yields the
        full content as one chunk so the caller still gets a reply.

        Media kwargs (``files``/``images``/``audio``/``videos``) are
        forwarded to the runtime's native ``runner.arun(...)`` so multimodal
        attachments propagate to Team members (see
        ``_runner/team/_default_tools.py:713``).
        """
        prompt = self._flatten_messages(messages)
        sid = session_id or "default"
        compatible_toolkits, filtered_families = self._compatible_mcp_toolkits()
        runner = self._ensure_team(system=system or "")
        if runner is None:
            runner = self._ensure_agent(system=system)
        elog("runtime.stream.start",
            model=self.model,
            prompt_len=len(prompt),
            mcp_toolkits=len(compatible_toolkits),
            filtered_families=filtered_families or None,
        )
        # Pull the event-type tuples ONCE per call (cached at module
        # level via _agno_event_types) so the per-iteration loop doesn't
        # rebuild them on every non-content event.
        event_types = _agno_event_types()
        content_types = event_types["content"]
        tool_started_types = event_types["tool_started"]
        tool_completed_types = event_types["tool_completed"]
        tool_error_types = event_types["tool_error"]

        try:
            stream_kwargs: dict[str, Any] = {
                "session_id": sid, "user_id": "openagent",
                "stream": True, "stream_events": True,
            }
            if files:
                stream_kwargs["files"] = files
            if images:
                stream_kwargs["images"] = images
            if audio:
                stream_kwargs["audio"] = audio
            if videos:
                stream_kwargs["videos"] = videos
            stream = runner.arun(prompt, **stream_kwargs)
            try:
                emitted = 0
                async for event in stream:
                    if isinstance(event, content_types):
                        text = getattr(event, "content", None) or ""
                        if text:
                            emitted += len(text)
                            yield text
                        image = getattr(event, "image", None)
                        if image is not None:
                            marker = self._save_image(image)
                            if marker:
                                yield marker
                        continue
                    if isinstance(event, tool_started_types):
                        if on_status is not None:
                            await self._emit_agno_tool_status(
                                on_status, getattr(event, "tool", None),
                                phase="started",
                            )
                    elif isinstance(event, tool_completed_types):
                        tool_exec = getattr(event, "tool", None)
                        if on_status is not None:
                            await self._emit_agno_tool_status(
                                on_status, tool_exec,
                            )
                        for img in getattr(event, "images", None) or []:
                            marker = self._save_image(img)
                            if marker:
                                yield marker
                    elif isinstance(event, tool_error_types):
                        if on_status is not None:
                            await self._emit_agno_tool_status(
                                on_status, getattr(event, "tool", None),
                                error_text=getattr(event, "error", None),
                            )
            finally:
                aclose = getattr(stream, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:  # noqa: BLE001
                        pass
            elog("runtime.stream.done", model=self.model, chars=emitted)
            if emitted == 0:
                elog("runtime.stream.empty_fallback",
                    model=self.model,
                    session_id=sid,
                )
                response = await self.generate(
                    messages,
                    system=system,
                    tools=tools,
                    on_status=on_status,
                    session_id=session_id,
                    files=files, images=images, audio=audio, videos=videos,
                )
                if response.content:
                    yield response.content
        except Exception as e:
            raw_error = str(e) or repr(e)
            if sid != "default" and self._is_session_corruption_error(raw_error):
                elog("runtime.stream.corrupt_session",
                    session_id=sid,
                    error=raw_error[:300],
                )
                try:
                    await self.forget_session(sid)
                except Exception as fe:
                    logger.debug("runtime stream recovery forget %s: %s", sid, fe)
            else:
                elog("runtime.stream.fallback",
                    level="warning",
                    model=self.model,
                    error_type=type(e).__name__,
                    error=raw_error,
                )
            response = await self.generate(
                messages,
                system=system,
                tools=tools,
                on_status=on_status,
                session_id=session_id,
            )
            if response.content:
                yield response.content

    def _compute_and_mirror_cost(
        self,
        *,
        metrics_obj: Any,
        input_tokens: int,
        output_tokens: int,
        session_id: str,
    ) -> float:
        """Compute cost from OpenAgent's catalog and write it onto the runtime's metrics.

        The runtime propagates the ``cost`` field through ``MessageMetrics → RunMetrics
        → SessionMetrics``, but never populates it (provider SDKs don't return
        cost). By mutating ``metrics_obj.cost`` (and the per-(provider, id)
        entries in ``metrics.details``) we make the runtime's session-level cost
        aggregation work — so ``sessions.runs[*].metrics.cost`` becomes
        directly queryable and ``SessionMetrics`` sums correctly across runs.

        The canonical cost record still lives in OpenAgent's ``usage_log``
        (written by ``ModelDispatcher``); this is the queryable mirror.
        """
        cost = compute_cost(
            model_ref=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        if input_tokens == 0 and output_tokens == 0:
            elog("runtime.cost_skipped",
                model=self.model,
                session_id=session_id,
                reason="zero_tokens",
            )
            return cost

        if metrics_obj is None:
            elog("runtime.cost_skipped",
                model=self.model,
                session_id=session_id,
                reason="no_metrics_object",
                cost_usd=cost,
            )
            return cost

        bare_id = model_id_from_runtime(self.model)
        mirrored_targets: list[str] = []

        # 1. Top-level RunMetrics.cost
        try:
            setattr(metrics_obj, "cost", cost)
            mirrored_targets.append("run")
        except (AttributeError, TypeError):
            pass

        # 2. Per-(provider, id) ModelMetrics.cost in details["model"], details["output_model"], …
        details = getattr(metrics_obj, "details", None)
        if isinstance(details, dict):
            for model_type, entries in details.items():
                if not entries:
                    continue
                for entry in entries:
                    entry_id = getattr(entry, "id", None)
                    if entry_id and (entry_id == bare_id or entry_id == self.model):
                        try:
                            setattr(entry, "cost", cost)
                            mirrored_targets.append(f"details.{model_type}[{entry_id}]")
                        except (AttributeError, TypeError):
                            pass

        elog("runtime.cost_mirrored",
            model=self.model,
            session_id=session_id,
            cost_usd=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            targets=mirrored_targets,
        )
        return cost
