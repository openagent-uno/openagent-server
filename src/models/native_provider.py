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
import json
import logging
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

from src.core import tool_trace, vault_recall
from src.core.logging import elog
from src.core.tool_scope import current_tool_allowlist, normalize_family
from src.models.base import BaseModel, ModelResponse
from src.models.catalog import (
    DEFAULT_CEREBRAS_BASE_URL,
    DEFAULT_MISTRAL_BASE_URL,
    DEFAULT_MOONSHOT_BASE_URL,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_XAI_BASE_URL,
    DEFAULT_ZAI_BASE_URL,
    FRAMEWORK_API_BASED,
    FULL_SESSION_HISTORY_RUNS,
    RUNTIME_SESSION_USER_ID,
    _iter_provider_entries,
    compute_cost,
    model_id_from_runtime,
    normalize_runtime_model_id,
    split_runtime_id,
)
from src.models.credential_pool import get_or_build_pool

logger = logging.getLogger(__name__)


# Backstop on how many tools ONE run may call before the runtime stops handing
# it more. Not a budget for normal work — real turns measured 3.3 tool calls on
# average, 13 at the worst — but a ceiling on the pathological one: the loop
# re-sends the ENTIRE context on every step, so an agent that keeps calling
# tools pays for its whole history again each time. Past the limit the runtime
# returns "tool call limit reached" instead of executing, and the model has to
# answer with what it has. Tune with OPENAGENT_MAX_TOOL_CALLS_PER_RUN.
_DEFAULT_MAX_TOOL_CALLS_PER_RUN = 60


# ── la guardia sulle chiamate ripetute e' STACCATA ────────────────────────
#
# `src/core/tool_repeat.py` c'e', e' provato (11 test) e funziona. Non e'
# cablato, e la ragione va scritta qui perche' non venga ricablato d'istinto.
#
# Cablarlo come `tool_hooks` ha rotto i tool in produzione DUE volte lo stesso
# giorno:
#   1. parametro chiamato `next_func` — `_build_hook_args` riempie solo i nomi
#      che riconosce, quindi non lo riempiva mai: "missing 1 required
#      positional argument";
#   2. corretto il nome, l'hook sincrono nella catena ASINCRONA restituiva una
#      coroutine che nessuno attendeva, e ogni tool tornava un oggetto
#      coroutine invece del risultato.
#
# Il contratto vero, letto alla fine in `src/mcp/_runtime/function.py`:
#   * catena sincrona  -> gli hook async vengono SCARTATI con un warning;
#   * catena asincrona -> `next_func` e' SEMPRE una coroutine, e l'hook viene
#     atteso SOLO se e' `async def`.
# Quindi serve un hook `async def` (che la catena sincrona scartera'), non uno
# sincuro. E serve un test che percorra la catena VERA del runtime: i miei
# undici test provavano la guardia e nessuno provava che il runtime sapesse
# invocarla, che e' l'unica cosa che poi si e' rotta.
#
# Si ricabla quando quel test esiste ed e' verde. Non prima.


def _max_tool_calls_per_run() -> Optional[int]:
    from src.core.execution_policy import current_max_tool_calls

    policy_limit = current_max_tool_calls()
    if policy_limit is not None:
        return policy_limit
    from src.core.execution_profile import lean_local_event_active

    if lean_local_event_active():
        raw = os.environ.get("OPENAGENT_LEAN_EVENT_MAX_TOOL_CALLS", "10").strip()
        try:
            value = int(raw)
        except ValueError:
            value = 10
        return value if value > 0 else 10
    raw = os.environ.get("OPENAGENT_MAX_TOOL_CALLS_PER_RUN", "").strip()
    if not raw:
        return _DEFAULT_MAX_TOOL_CALLS_PER_RUN
    try:
        val = int(raw)
    except ValueError:
        return _DEFAULT_MAX_TOOL_CALLS_PER_RUN
    # 0 / negative = explicitly unbounded, for an operator who knows why.
    return val if val > 0 else None


def _execution_cache_key(system: str | None) -> tuple[str, str]:
    """Return (cache key, real system message) for the current envelope.

    Runtime agents capture tools and ``tool_call_limit`` at construction. A
    policy change on a reused session therefore needs a distinct cached runner,
    while the synthetic cache suffix must never leak into the model prompt.
    """
    system_text = (system or "").strip()
    from src.core.execution_policy import current_execution_policy

    policy = current_execution_policy()
    allow = current_tool_allowlist()
    if not policy and allow is None:
        return system_text, system_text
    marker = json.dumps(
        {
            "policy": policy or {},
            "tool_families": None if allow is None else sorted(allow),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{system_text}\0{marker}", system_text


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


_SESSION_ID_TAG_RE = re.compile(
    r"\n*(?:<execution-host>[^<]*</execution-host>\s*)?"
    r"(?:<session-id>[^<]*</session-id>\s*)$"
)

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
        RunCompletedEvent as AgentRunCompletedEvent,
        RunContentEvent as AgentRunContentEvent,
        ToolCallStartedEvent as AgentToolCallStartedEvent,
        ToolCallCompletedEvent as AgentToolCallCompletedEvent,
        ToolCallErrorEvent as AgentToolCallErrorEvent,
    )
    content: tuple = (AgentRunContentEvent,)
    tool_started: tuple = (AgentToolCallStartedEvent,)
    tool_completed: tuple = (AgentToolCallCompletedEvent,)
    tool_error: tuple = (AgentToolCallErrorEvent,)
    run_completed: tuple = (AgentRunCompletedEvent,)
    try:
        from src.core._run_state.team import (
            RunCompletedEvent as TeamRunCompletedEvent,
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
        run_completed = (AgentRunCompletedEvent, TeamRunCompletedEvent)
    return {
        "content": content,
        "tool_started": tool_started,
        "tool_completed": tool_completed,
        "tool_error": tool_error,
        "run_completed": run_completed,
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


def _tool_name_args(entry: Any) -> tuple[Any, Any]:
    """Pull ``(tool_name, tool_args)`` off a runtime ``ToolExecution``.

    Handles both the object and the dict shape for the same reason
    ``_extract_tool_names_from_agno_response`` does: the runtime has changed
    this object before, and a shape we fail to read must cost a counter, not
    a turn.
    """
    if isinstance(entry, dict):
        return entry.get("tool_name"), entry.get("tool_args")
    return getattr(entry, "tool_name", None), getattr(entry, "tool_args", None)


def _record_vault_recalls(response: Any) -> None:
    """Book every vault note this non-streamed run read into the recall sink."""
    for entry in getattr(response, "tools", None) or []:
        name, args = _tool_name_args(entry)
        vault_recall.record_tool(name, args)
        tool_trace.record_execution(entry)


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
    # Kimi (Moonshot) and Alibaba Qwen/DashScope — OpenAI-compatible,
    # wired via OpenAILike in RUNTIME_PROVIDER_CLASSES below.
    "moonshot": "MOONSHOT_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
}
def _thinking_kwarg(provider_name: str, model_id: str) -> dict[str, Any]:
    """The ``thinking=`` kwarg for a model build, or ``{}``.

    Wires ``model.extended_thinking_tokens`` (yaml -> ``OPENAGENT_EXTENDED_THINKING_TOKENS``,
    set in ``core/server.py``) through to the Anthropic provider's ``thinking``
    field, which is the one channel that reaches it. Before this the env var was
    write-only — set from yaml, read by nobody — a config that documented a
    feature it did not deliver, the same dead-knob shape as the retired
    ``safety.*`` vars.

    Two gates, both load-bearing:

    * **Anthropic only.** ``thinking`` is an Anthropic request field; other
      providers' constructors do not accept it. (``_construct_model`` would
      filter it out anyway, but not emitting it keeps the intent legible.)
    * **Only models that support it.** ``Claude.supports_extended_thinking``
      excludes the Haiku family. Measured 2026-07-15: Haiku 4.5 returns HTTP
      400 through the subscription proxy when handed a thinking budget. An
      unsupported model is SKIPPED, not injected-and-crashed — the Haiku tier
      is the cheap routing model that fires most often, so a 400 there is the
      worst place to guess wrong. Opus/Sonnet get the budget; Haiku silently
      runs without it.

    Anthropic requires a budget >= 1024 to engage the feature; below that it is
    a no-op, so anything smaller is treated as off rather than sent.
    """
    if provider_name != "anthropic":
        return {}
    raw = (os.environ.get("OPENAGENT_EXTENDED_THINKING_TOKENS") or "").strip()
    if not raw:
        return {}
    try:
        budget = int(raw)
    except (TypeError, ValueError):
        return {}
    if budget < 1024:
        return {}
    from src.models.providers.anthropic import Claude

    if not Claude.supports_extended_thinking(model_id):
        return {}
    return {"thinking": {"type": "enabled", "budget_tokens": budget}}


RUNTIME_PROVIDER_CLASSES: dict[str, tuple[str, str, dict[str, Any]]] = {
    # Anthropic prompt caching is ON here, hardcoded, on purpose.
    #
    # This dict's 3rd element (``extra_kwargs``) is the ONLY channel that
    # reaches a provider constructor — ``_load_runtime_model_class`` reads it
    # and ``build_runtime_model`` splats it into ``_construct_model``. Model-row
    # metadata never becomes provider kwargs, so there is no per-model config
    # escape hatch to defer to. Leaving this ``{}`` did not mean "operator
    # decides"; it meant nobody could turn caching on at all, and Claude's own
    # defaults (cache_system_prompt=False, cache_tools=False) meant every
    # Anthropic call paid full price for a byte-identical prefix.
    #
    # What that cost: each call re-sends the ~12k-token FRAMEWORK_SYSTEM_PROMPT
    # (src/core/prompts.py) + the user persona + every tool schema. And a turn
    # is not one call — it is one call per tool-use iteration (3.3 average, 13
    # worst case; see ``_MAX_HISTORY_TOKENS`` in src/core/compaction.py), times
    # every surface: chat, sub-agent, workflow AI block, cron firing.
    # ``src/core/metrics.py`` has read ``cache_read_tokens`` in ~9 places since
    # it was written — against a counter that was structurally always zero.
    #
    # Why NOT ``extended_cache_time`` (1h): the flags collide, and the local
    # validator will not save us. Anthropic renders tools → system → messages,
    # and ``Claude._apply_cache_tools`` hardcodes the tool breakpoint to a bare
    # ``{"type": "ephemeral"}`` — always 5m; it never consults
    # extended_cache_time. Turning 1h on would place a 1h *system* block after
    # a 5m *tool* block, which is precisely the ordering Anthropic rejects.
    # ``_validate_cache_ttl_order`` only ever inspects the system array (it is
    # called from ``_build_system``, which never sees ``tools``), so it cannot
    # catch that combination — the first symptom would be a 400 on live
    # traffic. Uniform 5m/5m cannot trip the rule from any direction, and it
    # breaks even sooner anyway: 1.25x write + 0.1x read beats 2x uncached at
    # the 2nd call, where a 1h write (2x) needs a 3rd.
    #
    # Breakpoint budget: Anthropic allows 4 per request. We spend 2 — the last
    # tool (cache_tools) and the agent-built system block (cache_system_prompt).
    # The other 2 stay free for ``system_prompt_blocks``, which is user-supplied
    # and which OpenAgent itself never sets.
    #
    # Interaction with compaction: ``compact()`` rewrites ``sessions.runs`` in
    # place, so the replayed transcript changes and any cache over it dies.
    # That is expected and cheap — Anthropic's invalidation is tiered, and a
    # messages-only change leaves the tools+system entries intact. The
    # expensive, stable prefix survives compaction; only the suffix re-bills.
    # (Note the transcript is not cached by these two flags in the first place:
    # the only ``cache_control`` placements on this path are the system blocks
    # and the last tool.)
    #
    # No other provider needs anything here: OpenAI — and the OpenAILike fleet
    # below — caches prefixes implicitly and server-side, with no request-side
    # opt-in and no breakpoints to place. Anthropic is the only provider on this
    # path where caching is something the caller must explicitly ask for.
    "anthropic": (
        "src.models.providers.anthropic",
        "Claude",
        {"cache_system_prompt": True, "cache_tools": True},
    ),
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
    # OpenAI-compatible providers — same OpenAILike base as ZAI, each with a
    # default base_url resolved in ``_resolved_base_url`` via
    # ``PROVIDER_DEFAULT_BASE_URLS``. ``moonshot`` serves Kimi
    # (``kimi-k2-*``); ``qwen`` is Alibaba DashScope's OpenAI-compatible mode.
    # ``openrouter`` / ``cerebras`` were already advertised in
    # SUPPORTED_PROVIDERS + discovery but had no driver and raised at build
    # time — these entries close that gap.
    "moonshot": ("src.models.providers.openai.like", "OpenAILike", {"name": "Moonshot"}),
    "qwen": ("src.models.providers.openai.like", "OpenAILike", {"name": "Qwen"}),
    "openrouter": ("src.models.providers.openai.like", "OpenAILike", {"name": "OpenRouter"}),
    "cerebras": ("src.models.providers.openai.like", "OpenAILike", {"name": "Cerebras"}),
    # xAI (Grok) and Mistral are OpenAI-compatible too. Mistral ships a
    # native SDK upstream, but its ``/v1`` chat+tools endpoint speaks the
    # OpenAI schema, so OpenAILike covers the common path.
    "xai": ("src.models.providers.openai.like", "OpenAILike", {"name": "xAI"}),
    "mistral": ("src.models.providers.openai.like", "OpenAILike", {"name": "Mistral"}),
    # Self-hosted OpenAI-compatible servers (Ollama / vLLM / LM Studio /
    # llama.cpp). No default base_url — the operator supplies one via the
    # provider's ``base_url`` (enforced in ``_resolved_base_url``). Keyless
    # servers get a placeholder api_key in ``_resolved_api_key`` so the
    # OpenAI SDK client still initialises.
    "local": ("src.models.providers.openai.like", "OpenAILike", {"name": "Local"}),
}


# Default base_url per OpenAI-compatible provider built via ``OpenAILike``.
# These vendors have no dedicated driver class — OpenAILike + the right
# base_url IS the integration. A per-provider ``providers.base_url`` DB
# value always wins over these defaults (see ``_resolved_base_url``).
PROVIDER_DEFAULT_BASE_URLS: dict[str, str] = {
    "zai": DEFAULT_ZAI_BASE_URL,
    "moonshot": DEFAULT_MOONSHOT_BASE_URL,
    "qwen": DEFAULT_QWEN_BASE_URL,
    "openrouter": DEFAULT_OPENROUTER_BASE_URL,
    "cerebras": DEFAULT_CEREBRAS_BASE_URL,
    "xai": DEFAULT_XAI_BASE_URL,
    "mistral": DEFAULT_MISTRAL_BASE_URL,
}


# Providers whose endpoint is operator-specific and has no sensible default:
# self-hosted OpenAI-compatible servers (Ollama / vLLM / LM Studio /
# llama.cpp). Unlike the hosted vendors above, these have NO
# ``PROVIDER_DEFAULT_BASE_URLS`` entry — the operator MUST configure a
# ``base_url``. ``_resolved_base_url`` hard-fails when it's missing rather
# than letting ``OpenAILike`` silently fall back to OpenAI's own endpoint.
PROVIDER_REQUIRES_BASE_URL: frozenset[str] = frozenset({"local"})


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
        history_runs: int = FULL_SESSION_HISTORY_RUNS,
    ):
        self._providers_config = providers_config or {}
        self.model = normalize_runtime_model_id(model, self._providers_config)
        self._api_key = api_key
        self._base_url = base_url
        self._db_path = db_path
        self._history_runs = history_runs
        self._fallback_config: Any = None
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
        # In-flight agno run_id per session is tracked by ``BaseModel`` (see
        # ``_track_run_id`` / ``_coop_cancel``) so the single-model and team
        # paths share one cooperative-cancel implementation.
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

    def set_fallback_config(self, fallback_config: Any) -> None:
        if fallback_config is self._fallback_config:
            return
        self._fallback_config = fallback_config
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

        # API-based provider rows carry the api_keys worth exporting.
        # The module-level dedupe means we only do the providers-list
        # walk on the FIRST NativeProvider per (config-hash, provider) pair.
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
        # ``getattr``, not attribute access: the per-model sampling lookup now
        # runs inside ``build_runtime_model``, which some callers reach on an
        # instance built without ``__init__``. A missing path means "no row
        # override", never a crash on the build path.
        db_path = getattr(self, "_db_path", None)
        if db_path:
            return str(db_path)
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

    async def request_cancel(self, session_id: str) -> bool:
        """Cooperatively cancel the in-flight agno run for ``session_id``.

        Marks the run in the runner's global cancellation registry so it
        raises ``RunCancelledException`` at its next checkpoint, sets
        ``status=cancelled``, and persists the run via ``upsert_run`` — which
        backfills the user's question into ``messages`` (see
        ``synth_interrupted_messages``) so the interrupted turn survives in
        history. It's the difference from a raw asyncio ``task.cancel()``,
        which poisons the coroutine before any checkpoint, leaving a
        ``status=RUNNING`` shell that's never persisted — losing the turn.

        Returns True if a run was registered for cancellation; False if none
        is in flight for the session (the caller should then hard-cancel).
        """
        return await self._coop_cancel(session_id)

    def _runtime_parts(self) -> tuple[str, str]:
        return split_runtime_id(self.model)

    @staticmethod
    def _toolkit_family_name(toolkit: Any) -> str:
        return str(
            getattr(toolkit, "tool_name_prefix", None)
            or getattr(toolkit, "name", None)
            or "default"
        )

    def _base_compatible_mcp_toolkits(self) -> tuple[list[Any], list[str]]:
        """The provider-compatible toolkits (drop families this provider can't
        use). This is the whole of the historical behaviour — no per-run
        scoping — extracted so the scoping wrapper below can reuse it."""
        provider_name, _model_id = self._runtime_parts()
        blocked = _INCOMPATIBLE_TOOL_FAMILIES_BY_PROVIDER.get(provider_name, frozenset())
        if not blocked:
            return list(self._mcp_toolkits), []
        allowed: list[Any] = []
        filtered: set[str] = set()
        for toolkit in self._mcp_toolkits:
            family = self._toolkit_family_name(toolkit)
            if family in blocked:
                filtered.add(family)
                continue
            allowed.append(toolkit)
        return allowed, sorted(filtered)

    def _compatible_mcp_toolkits(self) -> tuple[list[Any], list[str]]:
        # Opt-in per-child tool scoping. ``current_tool_allowlist()`` is ``None``
        # for every ordinary run (chat, automation, unrestricted delegation), so
        # this takes the EXACT historical path: the memoised cache fast-path, and
        # the same list it always returned — byte-identical. Only a delegated
        # child that was spawned with an explicit ``allowed_tools`` subset (see
        # ``core.tool_scope`` / ``core.child_session``) installs an allowlist,
        # and that case is request-scoped: it NEITHER reads NOR writes
        # ``self._compatible_cache``, so a restricted run can never pollute the
        # shared cache an unrestricted run relies on.
        allow = current_tool_allowlist()
        if allow is None:
            if self._compatible_cache is not None:
                allowed, filtered = self._compatible_cache
                return list(allowed), list(filtered)
            base, filtered = self._base_compatible_mcp_toolkits()
            self._compatible_cache = (base, filtered)
            return list(base), list(filtered)
        # Restricted: keep only toolkits whose (normalised) family is allowed.
        base, filtered = self._base_compatible_mcp_toolkits()
        restricted = [
            tk for tk in base
            if (
                normalize_family(self._toolkit_family_name(tk)) in allow
                # The model sees only this broker in normal OpenAgent wiring.
                # Its target-server calls are scope-checked inside the adapter;
                # removing it here would turn a restricted run into a tool-less
                # run instead of a run with the intended restricted tools.
                or normalize_family(self._toolkit_family_name(tk)) == "tool_search"
            )
        ]
        return restricted, list(filtered)

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

        v0.12 stores providers as a flat list. NativeProvider resolves
        the API-based row for this model's vendor. Falls back to the
        legacy dict-shape for early-boot / tests.

        Memoised because providers_config is immutable per instance —
        ``rebuild_routing`` constructs a fresh NativeProvider rather than
        mutating an existing one. Caching saves an O(N_providers) walk
        per ``_build_runtime_model`` (and there are 2 per turn — see
        ``_resolved_api_key`` and ``_resolved_base_url``).
        """
        if getattr(self, "_provider_config_cache", None) is not None:
            return self._provider_config_cache
        provider_name, _ = self._runtime_parts()
        for entry in _iter_provider_entries(getattr(self, "_providers_config", None) or []):
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
        explicit = self._api_key or self._provider_setting("api_key")
        if explicit:
            return explicit
        provider_name, _ = self._runtime_parts()
        if provider_name in PROVIDER_REQUIRES_BASE_URL or self._self_hosted_spec(provider_name):
            # Local servers (Ollama / vLLM / LM Studio / llama.cpp) ignore the
            # key, but the OpenAI SDK client refuses to initialise without one —
            # supply a harmless placeholder so a keyless local setup still
            # works. A real key (e.g. a vLLM ``--api-key``) configured on the
            # provider wins above via ``explicit``. This covers BOTH the
            # reserved ``local`` name and any other operator-registered
            # self-hosted server (see ``_self_hosted_spec``): the second one
            # would otherwise build a client with no credential at all.
            return "local"
        return explicit

    def _resolved_base_url(self) -> str | None:
        provider_name, _ = self._runtime_parts()
        if self._base_url:
            return self._base_url
        configured = self._provider_setting("base_url")
        default = PROVIDER_DEFAULT_BASE_URLS.get(provider_name)
        if default is not None:
            return configured or default
        known = provider_name in RUNTIME_PROVIDER_CLASSES
        if not configured and (provider_name in PROVIDER_REQUIRES_BASE_URL or not known):
            raise ValueError(
                f"The '{provider_name}' provider requires a base_url pointing "
                "at your OpenAI-compatible server's /v1 root — e.g. "
                "http://localhost:11434/v1 (Ollama), http://localhost:8000/v1 "
                "(vLLM), or http://localhost:1234/v1 (LM Studio). Set it on the "
                "provider via model-manager add_provider/update_provider or the "
                "Providers UI."
            )
        return configured

    # Parametri di campionamento che un operatore puo' scrivere sulla riga del
    # modello (``models.metadata_json``). Whitelist stretta: qui passa solo cio'
    # che e' campionamento, mai credenziali o url.
    _SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "max_tokens",
                      "presence_penalty", "frequency_penalty", "repeat_penalty", "seed")

    def _sampling_from_model_row(self) -> dict[str, Any]:
        """Sampling params declared on the model row, or ``{}``.

        Before this, ``RUNTIME_PROVIDER_CLASSES``' ``extra_kwargs`` was the only
        channel into a provider constructor, and it is keyed by PROVIDER — so
        two models on the same provider could not differ, and nothing could be
        changed without a release. For a self-hosted server that gap is not
        cosmetic: llama.cpp defaults to ``temperature 0.8``, and a provider that
        sends no temperature (this one does not — ``OpenAIChat.temperature`` is
        ``Optional[float] = None``) inherits it silently. A support agent was
        therefore running far hotter than anyone had chosen, which is exactly
        the setting that turns "I cannot verify that" into an invented answer.

        Read from the DB row rather than yaml so it can be tuned per model with
        one UPDATE, and re-read per build so a change needs a reload, not a
        release.
        """
        raw = (self._model_row_metadata() or {})
        out: dict[str, Any] = {}
        for key in self._SAMPLING_KEYS:
            if key in raw and raw[key] is not None:
                out[key] = raw[key]
        return out

    def _model_row_metadata(self) -> dict[str, Any]:
        import json as _json
        import sqlite3 as _sqlite3

        db_path = self._runtime_db_path() if hasattr(self, "_runtime_db_path") else self._db_path
        if not db_path:
            return {}
        _, model_id = self._runtime_parts()
        try:
            con = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
            try:
                row = con.execute(
                    "select metadata_json from models where model=? limit 1", (model_id,)
                ).fetchone()
            finally:
                con.close()
        except Exception:  # noqa: BLE001 — una config illeggibile non deve fermare un turno
            return {}
        if not row or not row[0]:
            return {}
        try:
            data = _json.loads(row[0])
        except Exception:  # noqa: BLE001
            return {}
        return data if isinstance(data, dict) else {}

    def _construct_model(self, cls: type, **kwargs: Any) -> Any:
        accepted = _model_class_accepted_params(cls)
        # Anti-wedge: give every model call a per-read socket timeout so a hung
        # provider connection RAISES instead of silently wedging the worker.
        # The event claim lease (120s) is shorter than the 600s turn timeout,
        # and the yaml top-level ``timeout:`` key is not wired to model calls,
        # so without this a frozen socket holds the worker until the lease
        # reaper eventually re-enqueues it. This is a per-READ httpx timeout
        # (not a whole-request deadline): it fires only on a genuinely stalled
        # socket, never on a legitimately slow stream — and MUST stay under the
        # 120s lease TTL. An operator can override the 90s default via
        # ``OPENAGENT_MODEL_TIMEOUT_SECONDS`` (wired from ``model.timeout_seconds``
        # in yaml). This is the common constructor for the main agent, team
        # members, and the routing classifier, so it covers every model call.
        if kwargs.get("timeout") is None and "timeout" in accepted:
            env_timeout = os.environ.get("OPENAGENT_MODEL_TIMEOUT_SECONDS")
            try:
                kwargs["timeout"] = float(env_timeout) if env_timeout else 90.0
            except (TypeError, ValueError):
                kwargs["timeout"] = 90.0
        filtered = {k: v for k, v in kwargs.items() if v is not None and k in accepted}
        return cls(**filtered)

    def _load_runtime_model_class(self, provider_name: str) -> tuple[type | None, dict[str, Any]]:
        spec = RUNTIME_PROVIDER_CLASSES.get(provider_name)
        if not spec:
            spec = self._self_hosted_spec(provider_name)
        if not spec:
            return None, {}
        module_name, class_name, extra_kwargs = spec
        module = importlib.import_module(module_name)
        return getattr(module, class_name), dict(extra_kwargs)

    def _self_hosted_spec(self, provider_name: str) -> tuple[str, str, dict[str, Any]] | None:
        """Driver for an operator-registered OpenAI-compatible server.

        ``RUNTIME_PROVIDER_CLASSES`` keys the driver off the provider NAME, so
        ``local`` — the entry meant for self-hosted servers — is a single slot.
        An operator who already spends it (a subscription proxy, say) and then
        registers a second server gets ``Model provider 'x' is not supported``,
        with a suggestion list naming only hosted vendors. Nothing about the
        second server is unsupported: it speaks the same OpenAI schema the
        first one does. The name was never the driver — it is an identity.

        So: a provider row the operator created WITH a ``base_url`` is treated
        as self-hosted and built through ``OpenAILike``, exactly like ``local``.
        The base_url is the discriminator on purpose — it is what an operator
        must supply for a self-hosted endpoint and what no typo produces. A
        misspelled vendor (``anthropci:...``) has no row and no base_url, so it
        still raises, which is the property the strict map was protecting.
        """
        if not provider_name:
            return None
        try:
            configured = (self._provider_setting("base_url") or "").strip()
        except Exception:  # noqa: BLE001 — a missing/odd config must not break the build
            return None
        if not configured:
            return None
        return ("src.models.providers.openai.like", "OpenAILike", {"name": provider_name})

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
            extra_kwargs = {
                **extra_kwargs,
                **_thinking_kwarg(provider_name, model_id),
            }
            # Qwen's llama.cpp template defaults to extended thinking. On a
            # tool loop that means paying a fresh hidden essay before every
            # vault read; measured on Qwen3.5, even "reply exactly OK" exhausted
            # a 32-token cap without producing visible text. The local-event
            # profile values short, evidence-backed execution, so request the
            # model's native non-thinking mode. ``extra_body`` is accepted by
            # OpenAILike and forwarded as ``chat_template_kwargs`` by
            # llama.cpp. Scope it narrowly to self-hosted Qwen aliases so cloud
            # providers and non-Qwen local templates stay untouched.
            from src.core.execution_profile import lean_local_event_active
            if (
                lean_local_event_active()
                and self._self_hosted_spec(provider_name) is not None
                and "qwen" in model_id.lower()
            ):
                extra_kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": False}
                }
                # A self-hosted support turn must have a finite completion
                # budget. llama.cpp otherwise reports ``n_predict=-1`` and a
                # malformed/tool-loop response can occupy the only useful slot
                # indefinitely. Operators can raise this for broader local
                # workloads without changing cloud behaviour.
                # A scheduled task writes a report and calls tools with long
                # argument objects; a support reply is a few sentences. One
                # budget for both truncated a tool call mid-JSON, and the run
                # died on a parse error rather than on anything it did wrong.
                from src.core.execution_profile import lean_local_task_active

                is_task = lean_local_task_active()
                # 2500, not 1200: a tool call carrying an object argument
                # (a quality row with its six sub-scores) was being cut off
                # mid-JSON and the whole run died on a parse error. The budget
                # is still finite, which is what protects the single slot; a
                # support reply never approaches either number.
                env_key, default = (
                    ("OPENAGENT_LEAN_TASK_MAX_TOKENS", "4000") if is_task
                    else ("OPENAGENT_LEAN_EVENT_MAX_TOKENS", "2500")
                )
                try:
                    lean_max_tokens = int(os.environ.get(env_key, default))
                except (TypeError, ValueError):
                    lean_max_tokens = int(default)
                extra_kwargs["max_tokens"] = max(128, lean_max_tokens)
            # I parametri di campionamento della riga del modello vincono sui
            # default del provider: sono la scelta esplicita di un operatore.
            extra_kwargs = {**extra_kwargs, **self._sampling_from_model_row()}
            model = self._construct_model(
                model_class,
                id=model_id,
                api_key=api_key,
                base_url=base_url,
                **extra_kwargs,
            )
            # A self-hosted server's context window is a CAPABILITY, not a
            # sampling parameter, so it does not belong in _SAMPLING_KEYS.
            # Publish it on the model object instead, where compaction's
            # _resolve_max_context already looks first. Without this the qwen
            # row declared 40960 and compaction still budgeted against the
            # 200k default, so history overflowed before it ever compacted.
            declared_context = (self._model_row_metadata() or {}).get("context")
            if isinstance(declared_context, (int, float)) and declared_context > 0:
                try:
                    model.context_window = int(declared_context)
                except Exception:  # noqa: BLE001 - never fail a build over a hint
                    pass
            # Credential pool — inert unless this provider has >= 2 accounts
            # configured (metadata.accounts). When present, seed the model's
            # initial credential from the pool and stash it so the fallback
            # chokepoint can rotate on 429/529 before spilling to DeepSeek.
            pool = get_or_build_pool(provider_name, self._provider_config())
            if pool is not None:
                selected = pool.select()
                if selected is not None:
                    model.api_key = selected.api_key
                    model.base_url = selected.base_url
                    model.client = model.async_client = None
                model._openagent_cred_pool = pool
            return model

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
        cache_key, sys_key = _execution_cache_key(system)
        cached = self._agno_agents.get(cache_key)
        if cached is not None:
            self._agno_agents.move_to_end(cache_key)
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
        from src.core.execution_profile import lean_local_event_active
        summaries_enabled = not lean_local_event_active()
        from src.core.execution_profile import strict_local_only_active

        from src.core.execution_profile import stateless_completion_active

        stateless = stateless_completion_active()
        agent = RuntimeAgent(
            model=self._build_runtime_model(),
            # Controller-owned support composition is explicitly local-only.
            # A local outage must fail closed, never spill customer data or an
            # operational decision into a configured cloud fallback.
            fallback_config=(
                None if strict_local_only_active() else self._fallback_config
            ),
            # A stateless completion writes nothing: no session row, no
            # history replay, no summary. That removes the only writer in the
            # composer path and with it the lock contention above.
            db=None if stateless else SqliteDb(db_file=str(db_path)),
            tools=agent_tools,
            system_message=sys_key or None,
            add_history_to_context=not stateless,
            # Replay the ENTIRE stored transcript for this session, not a
            # trailing window (vision §16). ``_history_runs`` defaults to
            # ``FULL_SESSION_HISTORY_RUNS`` so the runtime's ``runs[-N:]``
            # slice returns everything; in-place compaction
            # (src/core/compaction.py, vision §2) is what bounds the actual
            # token footprint when the context limit nears.
            num_history_runs=self._history_runs,
            # The runtime also maintains a rolling summary of older turns in
            # the same SqliteDb and injects it into context on each call.
            enable_session_summaries=summaries_enabled and not stateless,
            add_session_summary_to_context=summaries_enabled and not stateless,
            # Agentic memory is disabled — OpenAgent uses the vault for
            # user-scoped persistence. Keeping it off avoids the legacy agno_memories
            # table creation and keeps all state in the sessions table.
            enable_agentic_memory=False,
            # Backstop against a run that never stops calling tools: every step
            # re-sends the whole context, so an unbounded loop pays for the
            # history again on each one.
            tool_call_limit=_max_tool_calls_per_run(),
            markdown=False,
        )
        self._agno_agents[cache_key] = agent
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
        cache_key, sys_key = _execution_cache_key(system)
        if not sys_key:
            return None
        cached = self._agno_teams.get(cache_key)
        if cached is not None:
            self._agno_teams.move_to_end(cache_key)
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
                fallback_config=self._fallback_config,
                tools=list(toolkits),
                system_message=member_system,
                name=f"{family}_specialist",
                role=member_role,
                tool_call_limit=_max_tool_calls_per_run(),
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
                fallback_config=self._fallback_config,
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
                tool_call_limit=_max_tool_calls_per_run(),
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
        self._agno_teams[cache_key] = team
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
            # The ``sessions``-row owner is the single stable
            # ``RUNTIME_SESSION_USER_ID`` sentinel for EVERY surface — see
            # the constant's definition in catalog.py. The runtime store
            # gates its history read + runs write on ``user_id == <this>
            # OR IS NULL``; a stable value is what keeps the conversation
            # readable/writable across turns and restarts. Per-user
            # separation lives in ``session_id`` (e.g. ``tg:<uid>``).
            arun_kwargs: dict[str, Any] = {
                "session_id": sid, "user_id": RUNTIME_SESSION_USER_ID,
            }
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
        # Server-side prefix-cache reads (DeepSeek prompt_cache_hit / OpenAI
        # prompt_tokens_details.cached_tokens). Captured by ``_get_metrics``
        # into ``cache_read_tokens``; included IN ``input_tokens`` above, priced
        # at the cheap cache-read rate by ``compute_cost``. Absent → 0 → old flat
        # cost. This is why a 135k-token tool-loop turn (mostly re-sent, cached
        # prefix) was over-billed ~10x before.
        cache_read_tokens = self._extract_metric(
            metrics_dict, "cache_read_tokens", "cached_tokens", "cache_read")
        stop_reason = metrics_dict.get("stop_reason")

        # Compute cost from OpenAgent's catalog and mirror it back into the runtime's
        # metrics so SessionMetrics.cost aggregation works for free.
        cost = self._compute_and_mirror_cost(
            metrics_obj=metrics_obj,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
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
        _record_vault_recalls(response)
        return ModelResponse(
            content=content,
            tool_names_called=tool_names_called,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
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
        events are forwarded to ``on_status`` in a normalized JSON
        format, and image/file artifacts from tool results
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
        run_completed_types = event_types["run_completed"]

        try:
            stream_kwargs: dict[str, Any] = {
                "session_id": sid, "user_id": RUNTIME_SESSION_USER_ID,
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
                    # Capture the agno run_id (first event wins) so a barge-in
                    # can cooperatively cancel THIS run and let agno persist it.
                    self._track_run_id(sid, getattr(event, "run_id", None))
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
                        # Book a vault recall HERE, on completion, not on
                        # ``tool_started``/``tool_error``: a read that failed
                        # never put the note in front of the model, so it is
                        # not a recall. This is the branch that fires in
                        # production — the non-streaming path above ran 11
                        # times against 697 streamed turns.
                        vault_recall.record_tool(*_tool_name_args(tool_exec))
                        tool_trace.record_execution(tool_exec)
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
                    elif isinstance(event, run_completed_types):
                        # The streamed run's tokens. Without this the whole
                        # streaming path is invisible to usage_log — see
                        # src/models/stream_usage.py.
                        from src.models import stream_usage

                        _ev_metrics = getattr(event, "metrics", None)
                        inp, out = stream_usage.metrics_to_tokens(_ev_metrics)
                        cr = stream_usage.metrics_to_cache_read(_ev_metrics)
                        stream_usage.record(
                            input_tokens=inp, output_tokens=out, model=self.model,
                            cache_read_tokens=cr,
                        )
            finally:
                self._clear_run_id(sid)
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
        cache_read_tokens: int = 0,
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
            cache_read_tokens=cache_read_tokens,
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
