"""OpenAI Codex SDK adapter — agent subclass + provider facade in one file.

Mirror of ``src/models/claude_agent.py`` for the OpenAI Codex CLI side.
This module owns the entire codex-cli framework path:

- ``CodexBackedAgent`` — runtime ``Agent`` subclass that drives
  ``openai_codex`` directly. Used wherever you'd otherwise reach for
  the OpenAI Codex CLI but need the result to participate in a runtime
  ``Team`` — as a member, as the leader's ``members[0]`` slot, or as
  both. A placeholder ``_NullModel`` satisfies ``Agent.__init__``;
  we override ``arun`` before any code path could reach
  ``model.invoke``.
- ``CodexCLI`` — single-model wrapper around ``CodexBackedAgent``.
- ``CodexCLIRegistry`` — per-session registry of ``CodexCLI``
  instances. ``runtime.create_model_from_spec`` instantiates one of
  these per codex-cli runtime spec.
- ``build_codex_backed_agent`` — factory used by both the registry
  and ``TeamRouterProvider`` when building team members.
- ``_known_sdk_session_ids_from_db`` — gateway ``/clear`` fallback.

Key differences from the claude-cli flavor:

- SDK is ``openai_codex`` (PyPI: ``openai-codex``, official OpenAI
  package). Its public surface is ``AsyncCodex`` + ``AsyncThread`` +
  per-turn ``thread.run(...)`` / ``thread.turn(...)`` rather than
  claude-cli's stream-of-messages ``query(prompt, options)`` shape.
- Session continuity uses ``thread.id`` + ``codex.thread_resume(tid)``
  rather than claude-cli's ``--resume`` flag. The thread id is stored
  in ``session.session_data["codex_session_id"]``.
- Built-in tools (file edit, shell, web fetch, etc.) come from the
  Codex CLI binary itself, governed by ``approval_mode`` /
  ``sandbox_policy`` rather than an explicit ``allowed_tools`` list.
  ``set_mcp_servers`` is a no-op.
- Streaming notifications are JSON-RPC method names
  (``item/agentMessage/delta``, ``turn/completed``) — translated to
  the runtime's ``RunContentEvent`` / ``ToolCall*Event`` the same way
  ``ClaudeBackedAgent`` translates ``StreamEvent`` deltas.

SDK session ids (thread ids) are round-tripped through the runtime's
session storage (``session.session_data["codex_session_id"]``) so
restart-resume keeps working. Subscription billing is zero per the
catalog; tokens × pricing aren't charged for the codex-cli framework.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Awaitable, Callable, ClassVar, Dict, Optional
from uuid import uuid4

from src.models.providers.response import ToolExecution
from src.core._run_state.agent import (
    RunContentEvent,
    RunOutput,
    ToolCallCompletedEvent,
    ToolCallStartedEvent,
)
from src.core._run_state.base import RunStatus

from src.core.logging import elog
from src.models._backed_agent import (
    BaseSubscriptionBackedAgent,
    _known_sdk_session_ids_by_key,
    _last_user_prompt,
    _make_sqlite_db,
    _prepend_files_for_subscription_cli,
    _stringify_input,
)
from src.models.base import BaseModel, ModelResponse
from src.models.catalog import FRAMEWORK_CODEX_CLI, model_id_from_runtime

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  SDK lazy import + utility helpers
# ════════════════════════════════════════════════════════════════════


def _sdk() -> Any:
    """Lazy-import ``openai_codex`` so non-codex users can still
    ``from src.models.codex_agent import CodexBackedAgent``.
    """
    try:
        import openai_codex  # type: ignore
    except ImportError as e:  # pragma: no cover — exercised only on slim installs
        raise ImportError(
            "openai_codex is required to run CodexBackedAgent. "
            "Install OpenAgent's codex-cli extra (or `pip install openai-codex`)."
        ) from e
    return openai_codex


def _known_sdk_session_ids_from_db(db: Any) -> list[str]:
    """Return ``sessions.session_id`` rows for codex-cli sessions.

    ``CodexBackedAgent`` rows are stored as plain ``Agent`` rows (so
    ``agent_data.framework`` is absent), but the class stamps
    ``session_data.codex_session_id`` with the SDK's thread id once
    known. That's the durable marker for the gateway's ``/clear``
    legacy-row fallback.
    """
    return _known_sdk_session_ids_by_key(db, json_key="codex_session_id")


# ── Notification → runtime event translation ─────────────────────────


def _notification_method(notification: Any) -> str:
    """Extract the JSON-RPC method name from a Codex SDK notification."""
    method = getattr(notification, "method", None)
    if isinstance(method, str):
        return method
    if isinstance(notification, dict):
        return str(notification.get("method") or "")
    return ""


def _notification_params(notification: Any) -> dict[str, Any]:
    """Extract the params payload from a Codex SDK notification."""
    params = getattr(notification, "params", None)
    if isinstance(params, dict):
        return params
    if isinstance(notification, dict):
        raw = notification.get("params")
        if isinstance(raw, dict):
            return raw
    return {}


# Notification kinds emitted by the Codex SDK's JSON-RPC stream. Classifying
# the method name once at the dispatch site keeps the per-kind handler logic
# below from re-scanning the same string for every matcher. The matchers
# stay defensive (substring + suffix) because the SDK has shipped multiple
# slightly different method-name shapes across versions
# (``item/agentMessage/delta``, ``thread/agentMessage/delta``,
# ``turn/completed`` vs ``thread/turn/completed``, ...).
_NK_AGENT_MESSAGE_DELTA = "agent_message_delta"
_NK_TOOL_CALL_STARTED = "tool_call_started"
_NK_TOOL_CALL_COMPLETED = "tool_call_completed"
_NK_TURN_COMPLETED = "turn_completed"
_NK_UNKNOWN = "unknown"


def _classify_notification_method(method: str) -> str:
    """Bucket a JSON-RPC method name into a handler kind.

    Centralises the fragile substring matching that used to be scattered
    across the streaming loop. The matchers are intentionally permissive
    (suffix + keyword) so an SDK rename like
    ``item/agentMessage/delta`` → ``thread/agentMessage/delta`` keeps
    working without code changes — but every pattern lives here, in one
    place, so the next maintainer can see the full vocabulary at a glance.
    """
    if not method:
        return _NK_UNKNOWN
    if method.endswith("/agentMessage/delta") or method.endswith("/message/delta"):
        return _NK_AGENT_MESSAGE_DELTA
    if "toolCall" in method or method.endswith("/tool/started"):
        return _NK_TOOL_CALL_STARTED
    if "toolResult" in method or method.endswith("/tool/completed"):
        return _NK_TOOL_CALL_COMPLETED
    if method == "turn/completed" or method.endswith("/turn/completed"):
        return _NK_TURN_COMPLETED
    return _NK_UNKNOWN


# ════════════════════════════════════════════════════════════════════
#  CodexBackedAgent — Agent subclass that drives openai_codex
# ════════════════════════════════════════════════════════════════════


class CodexBackedAgent(BaseSubscriptionBackedAgent):
    """``Agent`` subclass whose ``arun`` drives ``openai_codex``.

    Use wherever you'd otherwise reach for the OpenAI Codex CLI but
    need the result to participate in a runtime ``Team`` — as a member,
    as the leader's ``members[0]`` slot, or as both.

    Constructor accepts the same shape as ``ClaudeBackedAgent``: a
    model id plus a dict of SDK options. The dict's recognised keys
    map to Codex SDK params:

    - ``system`` / ``base_instructions`` / ``developer_instructions``
      — system-level guidance injected per turn.
    - ``approval_mode`` — ``"auto"`` / ``"suggest"`` / ``"full-auto"``.
    - ``sandbox_policy`` — e.g. ``"workspace-only"``.
    - ``cwd`` — working directory for the Codex subprocess.
    - ``options_kwargs`` — extra kwargs forwarded to ``thread_start``.
    """

    SESSION_DATA_KEY: ClassVar[str] = "codex_session_id"
    PROVIDER_NAME: ClassVar[str] = FRAMEWORK_CODEX_CLI
    NULL_MODEL_ID: ClassVar[str] = "codex-backed-null-model"
    ARUN_ERROR_EVENT: ClassVar[str] = "codex_backed_agent.arun_error"
    STREAM_ERROR_EVENT: ClassVar[str] = "codex_backed_agent.stream_error"

    # ── SDK options construction ─────────────────────────────────────

    def _build_thread_kwargs(self) -> Dict[str, Any]:
        """Compose the kwargs for ``codex.thread_start(...)`` /
        ``codex.thread_resume(...)`` from ``self._sdk_options``.

        Translates the OpenAgent shorthand (``system``) into the SDK's
        canonical field (``developer_instructions``) so callers can
        keep using the same dict shape across both subscription-CLI
        flavors.
        """
        kwargs: Dict[str, Any] = {}
        if self._sdk_model_id:
            kwargs["model"] = self._sdk_model_id

        # System prompt: prefer the SDK's canonical ``developer_instructions``,
        # fall back to ``base_instructions``, finally accept ``system``
        # / ``system_prompt`` as OpenAgent shorthand.
        instructions = (
            self._sdk_options.get("developer_instructions")
            or self._sdk_options.get("base_instructions")
            or self._sdk_options.get("system_prompt")
            or self._sdk_options.get("system")
        )
        if instructions:
            kwargs["developer_instructions"] = instructions

        for key in ("approval_mode", "sandbox_policy", "cwd", "effort"):
            value = self._sdk_options.get(key)
            if value is not None:
                kwargs[key] = value

        extra = self._sdk_options.get("options_kwargs")
        if isinstance(extra, dict):
            kwargs.update(extra)
        return kwargs

    # ── Result helpers ───────────────────────────────────────────────

    @staticmethod
    def _check_turn_result(result: Any) -> None:
        """Mirror ``ClaudeBackedAgent._check_result_message``: raise
        when the SDK reports a non-success status so the Team's task
        path surfaces a failure instead of silently emitting empty
        content.
        """
        status = getattr(result, "status", None)
        # Codex SDK's TurnStatus is an enum-ish object; compare its
        # ``value`` / ``name`` rather than the object identity so any
        # SDK shape variation (str / enum) is tolerated.
        status_name = (
            getattr(status, "name", None)
            or getattr(status, "value", None)
            or status
        )
        if status_name is None:
            return
        status_str = str(status_name).lower()
        if status_str in ("completed", "success", "ok"):
            return
        error = getattr(result, "error", None)
        detail = (
            getattr(error, "message", None)
            or getattr(error, "detail", None)
            or error
            or status_str
        )
        raise RuntimeError(
            f"Codex SDK error (status={status_str}): {detail}"
        )

    @staticmethod
    def _extract_thread_id(thread: Any) -> Optional[str]:
        """Pull the thread id from a Codex SDK thread handle.

        Tolerates the two shapes the SDK exposes: ``AsyncThread.id``
        (dataclass attribute) and ``ThreadStartResponse.thread_id``
        (response object).
        """
        for attr in ("id", "thread_id"):
            value = getattr(thread, attr, None)
            if isinstance(value, str) and value:
                return value
        return None

    @staticmethod
    def _extract_usage(result: Any) -> Dict[str, int]:
        """Pull token usage off a ``TurnResult`` defensively."""
        usage = getattr(result, "usage", None)
        out: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        if usage is None:
            return out
        if isinstance(usage, dict):
            out["input_tokens"] = int(usage.get("input_tokens", 0) or 0)
            out["output_tokens"] = int(usage.get("output_tokens", 0) or 0)
            return out
        # ThreadTokenUsage-shaped object.
        for attr in ("input_tokens", "output_tokens"):
            value = getattr(usage, attr, None)
            if isinstance(value, (int, float)):
                out[attr] = int(value)
        return out

    @staticmethod
    def _extract_tools_from_items(items: Any) -> list[ToolExecution]:
        """Translate Codex SDK ``ThreadItem``s into runtime ``ToolExecution``s.

        Walks the ``items`` list and pulls each tool invocation into a
        ``ToolExecution`` carrying the call id, tool name, args, and
        result text. Codex items come in many flavors (agentMessage,
        toolCall, toolResult, …); we keep this defensive — every
        attribute access falls back to a default so an SDK shape shift
        doesn't crash the parser.
        """
        tools: list[ToolExecution] = []
        if not items:
            return tools
        for item in items:
            item_type = getattr(item, "type", None) or (
                item.get("type") if isinstance(item, dict) else None
            )
            if not item_type or "tool" not in str(item_type).lower():
                continue
            tool_id = (
                getattr(item, "id", None)
                or (item.get("id") if isinstance(item, dict) else None)
                or str(uuid4())
            )
            tool_name = (
                getattr(item, "name", None)
                or getattr(item, "tool_name", None)
                or (
                    item.get("name") if isinstance(item, dict) else None
                )
                or "unknown"
            )
            tool_input = (
                getattr(item, "input", None)
                or getattr(item, "arguments", None)
                or (
                    item.get("input") or item.get("arguments")
                    if isinstance(item, dict) else None
                )
                or {}
            )
            tool_args = (
                tool_input
                if isinstance(tool_input, dict)
                else {"input": tool_input}
            )
            tool_result = (
                getattr(item, "result", None)
                or getattr(item, "output", None)
                or (
                    item.get("result") or item.get("output")
                    if isinstance(item, dict) else None
                )
            )
            tools.append(
                ToolExecution(
                    tool_call_id=str(tool_id),
                    tool_name=str(tool_name),
                    tool_args=tool_args,
                    result=str(tool_result) if tool_result is not None else None,
                )
            )
        return tools

    # ── arun (collect-to-RunOutput) ─────────────────────────────────

    async def _open_codex(self) -> Any:
        """Return an ``AsyncCodex`` context manager.

        Kept as its own method so tests can swap it. The SDK is
        well-behaved as an ``async with`` context manager — both the
        collect and stream paths use it the same way.
        """
        sdk = _sdk()
        return sdk.AsyncCodex()

    async def _ensure_thread(
        self,
        codex: Any,
        *,
        session_id: Optional[str],
    ) -> Any:
        """Resume or start the Codex thread for ``session_id``.

        Returns the thread handle (with ``.id`` attribute). Captures
        the resolved thread id back into ``self._sdk_session_ids`` for
        persistence in ``_save_to_db``.
        """
        kwargs = self._build_thread_kwargs()
        prior_thread_id = (
            self._sdk_session_ids.get(session_id) if session_id else None
        )
        if prior_thread_id:
            thread = await codex.thread_resume(prior_thread_id, **kwargs)
        else:
            thread = await codex.thread_start(**kwargs)
        thread_id = self._extract_thread_id(thread)
        if session_id and thread_id:
            self._sdk_session_ids[session_id] = thread_id
        return thread

    async def _arun_collect(
        self,
        input: Any,
        *,
        user_id: Optional[str],
        session_id: Optional[str],
        run_id: str,
        files: Optional[list[Any]] = None,
        images: Optional[list[Any]] = None,
        audio: Optional[list[Any]] = None,
        videos: Optional[list[Any]] = None,
    ) -> RunOutput:
        prompt = _stringify_input(input)
        if files:
            prompt = _prepend_files_for_subscription_cli(prompt, files)
        self._hydrate_from_db(session_id)

        try:
            async with await self._open_codex() as codex:
                thread = await self._ensure_thread(codex, session_id=session_id)
                result = await thread.run(prompt)
        except Exception as e:  # noqa: BLE001
            elog(
                self.ARUN_ERROR_EVENT,
                session_id=session_id,
                error=str(e),
            )
            raise

        self._check_turn_result(result)

        if session_id:
            self._save_to_db(session_id)

        final_response = getattr(result, "final_response", None)
        content = str(final_response or "")
        tools = self._extract_tools_from_items(getattr(result, "items", None))
        usage = self._extract_usage(result)
        metrics = self._build_metrics(usage["input_tokens"], usage["output_tokens"])

        return RunOutput(
            run_id=run_id,
            agent_id=self.id,
            agent_name=self.name,
            session_id=session_id,
            user_id=user_id,
            content=content,
            content_type="str",
            model=self._sdk_model_id,
            model_provider=FRAMEWORK_CODEX_CLI,
            tools=tools or None,
            metrics=metrics,
            status=RunStatus.completed,
        )

    # ── arun (streaming) ────────────────────────────────────────────

    async def _arun_stream(
        self,
        input: Any,
        *,
        user_id: Optional[str],
        session_id: Optional[str],
        run_id: str,
        yield_run_output: bool,
        files: Optional[list[Any]] = None,
        images: Optional[list[Any]] = None,
        audio: Optional[list[Any]] = None,
        videos: Optional[list[Any]] = None,
    ) -> AsyncIterator[Any]:
        prompt = _stringify_input(input)
        if files:
            prompt = _prepend_files_for_subscription_cli(prompt, files)
        self._hydrate_from_db(session_id)

        emitted_tool_ids: set[str] = set()
        assistant_text_acc = ""
        tools: list[ToolExecution] = []
        usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        final_response_text = ""

        try:
            async with await self._open_codex() as codex:
                thread = await self._ensure_thread(codex, session_id=session_id)
                handle = await thread.turn(prompt)
                async for notification in handle.stream():
                    method = _notification_method(notification)
                    params = _notification_params(notification)
                    kind = _classify_notification_method(method)

                    if kind == _NK_AGENT_MESSAGE_DELTA:
                        delta = str(params.get("delta") or "")
                        if delta:
                            assistant_text_acc += delta
                            yield RunContentEvent(
                                run_id=run_id,
                                agent_id=self.id or "",
                                agent_name=self.name or "",
                                session_id=session_id,
                                content=delta,
                            )
                    elif kind == _NK_TOOL_CALL_STARTED:
                        item = params.get("item") or params
                        tool_id = str(
                            (item.get("id") if isinstance(item, dict) else None)
                            or uuid4()
                        )
                        if tool_id in emitted_tool_ids:
                            continue
                        emitted_tool_ids.add(tool_id)
                        tool_name = str(
                            (
                                item.get("name") or item.get("tool_name")
                                if isinstance(item, dict) else None
                            )
                            or "unknown"
                        )
                        tool_input = (
                            item.get("input") or item.get("arguments")
                            if isinstance(item, dict) else None
                        ) or {}
                        tool_args = (
                            tool_input
                            if isinstance(tool_input, dict)
                            else {"input": tool_input}
                        )
                        te = ToolExecution(
                            tool_call_id=tool_id,
                            tool_name=tool_name,
                            tool_args=tool_args,
                        )
                        tools.append(te)
                        yield ToolCallStartedEvent(
                            run_id=run_id,
                            agent_id=self.id or "",
                            agent_name=self.name or "",
                            session_id=session_id,
                            tool=te,
                        )
                    elif kind == _NK_TOOL_CALL_COMPLETED:
                        item = params.get("item") or params
                        tool_use_id = str(
                            (
                                item.get("tool_call_id")
                                or item.get("id")
                                if isinstance(item, dict) else None
                            )
                            or ""
                        )
                        result_text = str(
                            (
                                item.get("result") or item.get("output")
                                if isinstance(item, dict) else None
                            )
                            or ""
                        )
                        for te in tools:
                            if te.tool_call_id == tool_use_id:
                                te.result = result_text
                                yield ToolCallCompletedEvent(
                                    run_id=run_id,
                                    agent_id=self.id or "",
                                    agent_name=self.name or "",
                                    session_id=session_id,
                                    tool=te,
                                )
                                break
                    elif kind == _NK_TURN_COMPLETED:
                        turn = params.get("turn") or {}
                        if isinstance(turn, dict):
                            final_response_text = str(turn.get("final_response") or "")
                            turn_usage = turn.get("usage") or {}
                            if isinstance(turn_usage, dict):
                                usage["input_tokens"] = int(
                                    turn_usage.get("input_tokens", 0) or 0
                                )
                                usage["output_tokens"] = int(
                                    turn_usage.get("output_tokens", 0) or 0
                                )
                    else:
                        logger.debug("unhandled codex notification: %s", method)
        except Exception as e:  # noqa: BLE001
            elog(
                self.STREAM_ERROR_EVENT,
                session_id=session_id,
                error=str(e),
            )
            raise

        if session_id:
            self._save_to_db(session_id)

        if yield_run_output:
            content = final_response_text or assistant_text_acc
            metrics = self._build_metrics(
                usage["input_tokens"], usage["output_tokens"]
            )
            yield RunOutput(
                run_id=run_id,
                agent_id=self.id,
                agent_name=self.name,
                session_id=session_id,
                user_id=user_id,
                content=content,
                content_type="str",
                model=self._sdk_model_id,
                model_provider=FRAMEWORK_CODEX_CLI,
                tools=tools or None,
                metrics=metrics,
                status=RunStatus.completed,
            )


# ════════════════════════════════════════════════════════════════════
#  Factory + provider facade (CodexCLI / CodexCLIRegistry)
# ════════════════════════════════════════════════════════════════════


def _build_sdk_options(
    *,
    system: str | None,
) -> dict[str, Any]:
    """Compose the dict of SDK options ``CodexBackedAgent`` accepts.

    Pulled out so ``TeamRouterProvider`` can reuse the exact same
    option-building logic when it constructs a ``CodexBackedAgent``
    for a codex-cli row (leader OR member).

    The Codex SDK doesn't expose ``allowed_tools`` / ``mcp_servers``
    in its documented public API — built-in tools come from the Codex
    CLI binary itself, sandboxed by ``approval_mode`` /
    ``sandbox_policy``. So these are NOT forwarded.
    """
    opts: dict[str, Any] = {
        "approval_mode": os.environ.get("OPENAGENT_CODEX_APPROVAL", "auto"),
    }
    sandbox = os.environ.get("OPENAGENT_CODEX_SANDBOX", "").strip()
    if sandbox:
        opts["sandbox_policy"] = sandbox
    if system:
        opts["developer_instructions"] = system
    return opts


def build_codex_backed_agent(
    *,
    name: str,
    model_id: str | None,
    system: str | None = None,
    db: Any = None,
    role: str | None = None,
) -> Any:
    """Factory: instantiate a ``CodexBackedAgent`` configured for either
    standalone use (via ``CodexCLI`` below) or Team membership (via
    ``TeamRouterProvider``).
    """
    sdk_model_id = (model_id or "").strip() or "default"
    sdk_options = _build_sdk_options(system=system)
    return CodexBackedAgent(
        name=name,
        sdk_model_id=sdk_model_id,
        sdk_options=sdk_options,
        db=_make_sqlite_db(db),
        role=role,
    )


class CodexCLI(BaseModel):
    """Single-model wrapper around ``CodexBackedAgent``.

    ``CodexCLIRegistry`` (below) is the real entry point — one
    ``CodexCLI`` per session, keyed by ``session_id``. This class
    stays exported so existing imports (``src.models.__init__``,
    tests, ``runtime.create_model_from_spec``) keep working.
    """

    history_mode = "provider"

    def __init__(
        self,
        model: str | None = None,
        providers_config: Any = None,
    ):
        if model:
            self.model: str | None = model_id_from_runtime(model) or model
        else:
            self.model = model
        self._providers_config = providers_config if providers_config is not None else []
        self._db: Any = None
        self._agent: Any = None
        self._agent_system: str | None = None
        self._session_handles: dict[str, str] = {}

    # ── wiring ────────────────────────────────────────────────────────

    def invalidate_agent_cache(self) -> None:
        """Drop the cached backed Agent so the next call rebuilds it."""
        self._agent = None
        self._agent_system = None

    def set_mcp_servers(self, servers: dict[str, dict]) -> None:
        # Codex CLI manages its own built-in tools via the binary; MCP
        # wiring is a no-op here so the standard pipeline can call this
        # without special-casing.
        return None

    def set_db(self, db: Any) -> None:
        if db is self._db:
            return
        self._db = db
        self.invalidate_agent_cache()

    def set_session_handle(self, session_id: str, handle: str | None) -> None:
        if not session_id:
            return
        if handle and handle.strip():
            self._session_handles[session_id] = handle.strip()
        else:
            self._session_handles.pop(session_id, None)

    # ── agent construction ────────────────────────────────────────────

    def _ensure_agent(self, system: str | None) -> Any:
        if self._agent is not None and self._agent_system == (system or None):
            return self._agent
        bare = self.model or "default"
        self._agent = build_codex_backed_agent(
            name=f"codex-{bare}",
            model_id=bare,
            system=system or None,
            db=self._db,
        )
        self._agent_system = system or None
        return self._agent

    # ── lifecycle (mostly no-ops post-SDK) ────────────────────────────

    async def close_session(self, session_id: str) -> None:
        # ``CodexBackedAgent`` doesn't keep a persistent subprocess
        # between turns — each ``arun`` opens + drains its own
        # ``AsyncCodex`` context. Nothing to tear down per-session.
        return None

    async def forget_session(self, session_id: str) -> None:
        """Erase the session row from the sessions table so the next
        message starts a fresh Codex thread (no resume).
        """
        if not session_id:
            return
        sqlite_db = _make_sqlite_db(self._db)
        if sqlite_db is None:
            return
        try:
            sqlite_db.delete_session(session_id=session_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("CodexCLI.forget_session %s: %s", session_id, e)
        elog("codex_cli.session_forget", session_id=session_id)

    async def commit_partial_assistant(self, session_id: str, text: str) -> None:
        # Barge-in support not exposed by the Codex SDK either.
        return None

    async def cleanup_idle(self) -> None:
        return None

    async def shutdown(self) -> None:
        self.invalidate_agent_cache()

    def known_session_ids(self) -> list[str]:
        return _known_sdk_session_ids_from_db(self._db)

    # ── turn ──────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
        model_override: str | None = None,
        files: list[Any] | None = None,
    ) -> ModelResponse:
        # ``tools`` is ignored: the Codex SDK doesn't accept
        # provider-neutral tool defs — its tools come from the bundled
        # CLI binary.
        agent = self._ensure_agent(system)
        prompt = _last_user_prompt(messages)
        if files:
            prompt = _prepend_files_for_subscription_cli(prompt, files)
        sid = session_id or "default"
        user_id = self._session_handles.get(sid) or "openagent"
        try:
            run_output = await agent.arun(
                prompt, session_id=sid, user_id=user_id, stream=False,
            )
        except Exception as e:  # noqa: BLE001
            elog("codex_cli.generate_error", session_id=sid, error=str(e))
            raise

        content = str(getattr(run_output, "content", "") or "")
        tools_used: list[str] = []
        for tc in getattr(run_output, "tools", None) or []:
            name = getattr(tc, "tool_name", None)
            if name:
                tools_used.append(str(name))
        metrics = getattr(run_output, "metrics", None)
        input_tokens = int(getattr(metrics, "input_tokens", 0) or 0) if metrics else 0
        output_tokens = int(getattr(metrics, "output_tokens", 0) or 0) if metrics else 0
        return ModelResponse(
            content=content,
            tool_names_called=tools_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
        model_override: str | None = None,
        files: list[Any] | None = None,
    ) -> AsyncIterator[str]:
        from src.models._tool_status import tool_exec_to_wire_json

        agent = self._ensure_agent(system)
        prompt = _last_user_prompt(messages)
        if files:
            prompt = _prepend_files_for_subscription_cli(prompt, files)
        sid = session_id or "default"
        user_id = self._session_handles.get(sid) or "openagent"

        stream_iter = None
        try:
            stream_iter = agent.arun(
                prompt, session_id=sid, user_id=user_id, stream=True,
            )
            async for event in stream_iter:
                if isinstance(event, RunContentEvent):
                    delta = event.content or ""
                    if delta:
                        yield delta
                    continue
                if on_status is None:
                    continue
                tool = getattr(event, "tool", None)
                if tool is None or not getattr(tool, "tool_name", None):
                    continue
                if isinstance(event, ToolCallStartedEvent):
                    encoded = tool_exec_to_wire_json(tool, phase="started")
                elif isinstance(event, ToolCallCompletedEvent):
                    encoded = tool_exec_to_wire_json(tool)
                else:
                    continue
                if encoded is None:
                    continue
                try:
                    await on_status(encoded)
                except Exception as e:  # noqa: BLE001
                    logger.debug("on_status callback raised: %s", e)
        except Exception as e:  # noqa: BLE001
            elog("codex_cli.stream_error", session_id=sid, error=str(e))
            raise
        finally:
            if stream_iter is not None:
                aclose = getattr(stream_iter, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:  # noqa: BLE001
                        pass


class CodexCLIRegistry(BaseModel):
    """Per-session registry of ``CodexCLI`` instances.

    Mirrors ``ClaudeCLIRegistry``: ``runtime.create_model_from_spec``
    instantiates one of these for a codex-cli runtime spec; each
    session gets its own ``CodexCLI`` so per-session config (model
    pin, system prompt) can vary without bleed.
    """

    history_mode = "provider"

    def __init__(
        self,
        default_model: str | None = None,
        providers_config: Any = None,
    ):
        self._default_model = (default_model or "").strip() or None
        self._providers_config = providers_config if providers_config is not None else []
        self._db: Any = None
        self._instances: dict[str, CodexCLI] = {}
        self._session_model: dict[str, str] = {}
        self._session_handles: dict[str, str] = {}

    @property
    def model(self) -> str | None:
        return self._default_model

    # ── wiring fan-out ────────────────────────────────────────────────

    def set_db(self, db: Any) -> None:
        self._db = db
        for inst in self._instances.values():
            inst.set_db(db)

    def set_mcp_servers(self, servers: dict[str, dict]) -> None:
        # No-op: see CodexCLI.set_mcp_servers.
        return None

    def set_session_handle(self, session_id: str, handle: str | None) -> None:
        if not session_id:
            return
        if handle and handle.strip():
            self._session_handles[session_id] = handle.strip()
        else:
            self._session_handles.pop(session_id, None)
        for inst in self._instances.values():
            inst.set_session_handle(session_id, handle)

    def set_default_model(self, model_id: str | None) -> None:
        self._default_model = (
            model_id_from_runtime(model_id.strip())
            if (model_id and model_id.strip())
            else None
        )

    def pin_session(self, session_id: str, model_id: str | None) -> None:
        if model_id and model_id.strip():
            self._session_model[session_id] = model_id_from_runtime(model_id.strip())
        else:
            self._session_model.pop(session_id, None)

    # ── lifecycle fan-out ─────────────────────────────────────────────

    async def cleanup_idle(self) -> None:
        for inst in list(self._instances.values()):
            try:
                await inst.cleanup_idle()
            except Exception as e:  # noqa: BLE001
                logger.debug("codex_registry.cleanup_idle: %s", e)

    async def shutdown(self) -> None:
        for inst in list(self._instances.values()):
            try:
                await inst.shutdown()
            except Exception as e:  # noqa: BLE001
                logger.debug("codex_registry.shutdown: %s", e)
        self._instances.clear()

    async def close_session(self, session_id: str) -> None:
        inst = self._instances.get(session_id)
        if inst is not None:
            try:
                await inst.close_session(session_id)
            except Exception as e:  # noqa: BLE001
                logger.debug("codex_registry.close_session: %s", e)
        self._session_model.pop(session_id, None)

    async def forget_session(self, session_id: str) -> None:
        inst = self._instances.pop(session_id, None)
        if inst is not None:
            try:
                await inst.forget_session(session_id)
            except Exception as e:  # noqa: BLE001
                logger.debug("codex_registry.forget_session: %s", e)
            try:
                await inst.shutdown()
            except Exception as e:  # noqa: BLE001
                logger.debug("codex_registry.forget_session shutdown: %s", e)
        self._session_model.pop(session_id, None)

    async def commit_partial_assistant(self, session_id: str, text: str) -> None:
        return None

    def known_session_ids(self) -> list[str]:
        seen: set[str] = set()
        for inst in self._instances.values():
            seen.update(inst.known_session_ids())
        seen.update(self._session_model.keys())
        seen.update(_known_sdk_session_ids_from_db(self._db))
        return sorted(seen)

    # ── per-session routing ───────────────────────────────────────────

    def _resolve_model(self, session_id: str, model_override: str | None) -> str | None:
        if model_override and model_override.strip():
            return model_id_from_runtime(model_override.strip())
        pinned = self._session_model.get(session_id)
        if pinned:
            return pinned
        return self._default_model

    def _get_or_create(self, session_id: str, model_id: str | None) -> CodexCLI:
        inst = self._instances.get(session_id)
        if inst is not None:
            return inst
        if not model_id:
            raise ValueError(
                "CodexCLIRegistry._get_or_create: first-spawn for session "
                f"{session_id!r} requires a model_id; the router must pin a "
                "concrete runtime_id before dispatching to codex-cli."
            )
        bare = model_id_from_runtime(model_id) or model_id
        inst = CodexCLI(
            model=bare,
            providers_config=self._providers_config,
        )
        if self._db is not None:
            inst.set_db(self._db)
        handle = self._session_handles.get(session_id)
        if handle:
            inst.set_session_handle(session_id, handle)
        self._instances[session_id] = inst
        elog(
            "codex_cli_registry.instance_created",
            session_id=session_id,
            initial_model=bare,
        )
        return inst

    @staticmethod
    def _retune_model(inst: CodexCLI, target_model: str | None) -> None:
        if not target_model:
            return
        if inst.model == target_model:
            return
        inst.model = target_model
        inst.invalidate_agent_cache()

    # ── turn ──────────────────────────────────────────────────────────

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
        model_override: str | None = None,
        files: list[Any] | None = None,
    ) -> ModelResponse:
        sid = session_id or "default"
        target_model = self._resolve_model(sid, model_override)
        inst = self._get_or_create(sid, target_model)
        self._retune_model(inst, target_model)
        return await inst.generate(
            messages=messages,
            system=system,
            tools=tools,
            on_status=on_status,
            session_id=sid,
            model_override=target_model,
            files=files,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
        model_override: str | None = None,
        files: list[Any] | None = None,
    ) -> AsyncIterator[str]:
        sid = session_id or "default"
        target_model = self._resolve_model(sid, model_override)
        inst = self._get_or_create(sid, target_model)
        self._retune_model(inst, target_model)
        async for delta in inst.stream(
            messages,
            system=system,
            on_status=on_status,
            session_id=sid,
            model_override=target_model,
            files=files,
        ):
            yield delta
