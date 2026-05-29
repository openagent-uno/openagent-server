"""Claude SDK adapter — agent subclass + provider facade in one file.

This module owns the entire claude-cli framework path:

- ``ClaudeBackedAgent`` — runtime ``Agent`` subclass that drives
  ``claude_agent_sdk`` directly. Used wherever you'd otherwise reach
  for the runtime's built-in Claude agent but need the result to
  participate in a runtime ``Team`` (as a member, as the leader's
  ``members[0]`` slot, or as both). A placeholder ``_NullModel``
  satisfies ``Agent.__init__``'s assumption that ``self.model`` is a
  real ``Model``; we override ``arun`` before any code path could
  reach ``model.invoke``.
- ``ClaudeCLI`` — single-model wrapper around ``ClaudeBackedAgent``.
- ``ClaudeCLIRegistry`` — per-session registry of ``ClaudeCLI``
  instances. ``runtime.create_model_from_spec`` instantiates one of
  these per claude-cli runtime spec.
- ``build_claude_backed_agent`` — factory used by both the registry
  and ``TeamRouterProvider`` when building team members.
- ``_known_sdk_session_ids_from_db`` — gateway ``/clear`` fallback for
  legacy rows.

SDK session ids are round-tripped through the runtime's session storage
(``session.session_data["sdk_session_id"]``) so restart-resume keeps
working.

What's intentionally NOT here (matches the pared-down v0.14 surface):
``can_use_tool`` safety hooks, barge-in / interrupt, stale-resume
self-heal, compression watchdog, ``ResultMessage.total_cost_usd`` —
token-count × catalog price only.
"""

from __future__ import annotations

import logging
import os
import re
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
    _stringify_input,
)
from src.models.base import BaseModel, ModelResponse
from src.models.catalog import FRAMEWORK_CLAUDE_CLI, model_id_from_runtime

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
#  SDK lazy import + utility helpers
# ════════════════════════════════════════════════════════════════════


def _sdk() -> Any:
    """Lazy-import claude_agent_sdk so non-claude users can still
    ``from src.models.claude_agent import ClaudeBackedAgent``.
    """
    try:
        import claude_agent_sdk  # type: ignore
    except ImportError as e:  # pragma: no cover — exercised only on slim installs
        raise ImportError(
            "claude_agent_sdk is required to run ClaudeBackedAgent. "
            "Install OpenAgent's claude-cli extra (or `pip install claude-agent-sdk`)."
        ) from e
    return claude_agent_sdk


# OpenRouter lists Anthropic models with a dotted version separator
# (``claude-sonnet-4.6``) but the Claude SDK only accepts dashes
# (``claude-sonnet-4-6``).
_MODEL_DOTTED_VERSION_RE = re.compile(r"([A-Za-z])-(\d+)\.(\d+)")


def _sanitize_claude_model_id(model_id: str | None) -> str | None:
    """Normalise an Anthropic model id for the Claude SDK."""
    if not model_id:
        return model_id
    return _MODEL_DOTTED_VERSION_RE.sub(r"\1-\2-\3", model_id)


def _known_sdk_session_ids_from_db(db: Any) -> list[str]:
    """Return ``sessions.session_id`` rows for SDK-backed sessions.

    Two markers identify an SDK-backed session row:

    - Legacy rows (pre-v0.14 ``ClaudeAgent`` adapter): the runtime's
      ``BaseExternalAgent`` writes ``agent_data.framework =
      'claude-agent-sdk'``.
    - New ``ClaudeBackedAgent`` rows: stored as plain ``Agent`` rows
      (so ``agent_data.framework`` is absent or ``'agent'``), but the
      class explicitly stamps ``session_data.sdk_session_id`` with the
      SDK's session id once known.

    The widened filter matches EITHER marker so the gateway ``/clear``
    legacy-row fallback can still find new claude-backed rows after a
    cold restart (before ``TeamRouterProvider.known_session_ids`` has
    been re-populated by user activity).
    """
    return _known_sdk_session_ids_by_key(
        db,
        json_key="sdk_session_id",
        extra_or_clauses=[
            "json_extract(agent_data, '$.framework') = 'claude-agent-sdk'",
        ],
    )


# ════════════════════════════════════════════════════════════════════
#  ClaudeBackedAgent — Agent subclass that drives claude_agent_sdk
# ════════════════════════════════════════════════════════════════════


class ClaudeBackedAgent(BaseSubscriptionBackedAgent):
    """``Agent`` subclass whose ``arun`` drives ``claude_agent_sdk``.

    Use wherever you'd otherwise reach for the runtime's built-in Claude
    agent but need the result to participate in a runtime ``Team`` — as a
    member, as the leader's ``members[0]`` slot, or as both.

    Constructor accepts the same shape as the previous
    ``PersistentClaudeAgent`` factory: a model id plus a dict of SDK
    options. The dict's recognised keys mirror ``ClaudeAgentOptions``:
    ``system`` (system_prompt), ``allowed_tools``, ``disallowed_tools``,
    ``permission_mode``, ``max_turns``, ``mcp_servers``, ``cwd``,
    ``max_budget_usd``, plus any ``options_kwargs`` passed through
    verbatim.
    """

    SESSION_DATA_KEY: ClassVar[str] = "sdk_session_id"
    PROVIDER_NAME: ClassVar[str] = FRAMEWORK_CLAUDE_CLI
    NULL_MODEL_ID: ClassVar[str] = "claude-backed-null-model"
    ARUN_ERROR_EVENT: ClassVar[str] = "claude_backed_agent.arun_error"
    STREAM_ERROR_EVENT: ClassVar[str] = "claude_backed_agent.stream_error"

    # ── SDK options construction ─────────────────────────────────────

    def _build_options(self, *, streaming: bool, session_id: Optional[str]) -> Any:
        """Compose a ``ClaudeAgentOptions`` from ``self._sdk_options``
        plus any per-session resume hint.

        ``self._sdk_options`` accepts both the SDK's canonical key
        (``system_prompt``) and the shorthand (``system``) so callers
        can stay terse. We translate at the boundary.
        """
        sdk = _sdk()
        opts: Dict[str, Any] = {}

        # Translate the shorthand caller shape into ClaudeAgentOptions
        # field names. The shorthand is what TeamRouterProvider hands us
        # because it mirrors the older ClaudeCLI dict; the canonical
        # form is what the SDK accepts.
        system_prompt = (
            self._sdk_options.get("system_prompt")
            or self._sdk_options.get("system")
        )
        if system_prompt:
            opts["system_prompt"] = system_prompt

        if self._sdk_model_id:
            opts["model"] = self._sdk_model_id

        for key in (
            "allowed_tools",
            "disallowed_tools",
            "permission_mode",
            "max_turns",
            "max_budget_usd",
            "cwd",
            "mcp_servers",
        ):
            value = self._sdk_options.get(key)
            if value is not None:
                opts[key] = value

        if streaming:
            # Token-level streaming via raw API events.
            opts["include_partial_messages"] = True

        if session_id:
            sdk_sid = self._sdk_session_ids.get(session_id)
            if sdk_sid:
                opts["resume"] = sdk_sid

        # Any options_kwargs the caller wanted passed through verbatim.
        extra = self._sdk_options.get("options_kwargs")
        if isinstance(extra, dict):
            opts.update(extra)

        return sdk.ClaudeAgentOptions(**opts)

    # ── Result-message helpers ───────────────────────────────────────

    @staticmethod
    def _check_result_message(sdk: Any, message: Any) -> None:
        """Mirror ``ClaudeAgent._check_result_message``: raise when the
        SDK reports an error result so the Team's task path surfaces a
        failure instead of silently emitting empty content.
        """
        if not isinstance(message, sdk.ResultMessage):
            return
        is_error = bool(getattr(message, "is_error", False))
        subtype = getattr(message, "subtype", None)
        if is_error or (subtype and subtype != "success"):
            detail = (
                getattr(message, "errors", None)
                or getattr(message, "result", None)
                or getattr(message, "stop_reason", None)
            )
            raise RuntimeError(
                f"Claude SDK error (is_error={is_error}, subtype={subtype}): {detail}"
            )

    # ── arun (collect-to-RunOutput) ─────────────────────────────────

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
        """Drain ``_arun_stream`` and return its terminal ``RunOutput``.

        Both paths walk the same SDK message stream (SystemMessage /
        AssistantMessage / UserMessage / ResultMessage); keeping a single
        implementation in ``_arun_stream`` and consuming it here avoids
        the two-walker drift that the previous duplicate-body collect
        path was prone to. The intermediate stream events
        (RunContentEvent / ToolCall*Event) are dropped — collect callers
        only care about the terminal RunOutput.
        """
        final_output: Optional[RunOutput] = None
        async for event in self._arun_stream(
            input,
            user_id=user_id,
            session_id=session_id,
            run_id=run_id,
            yield_run_output=True,
            files=files, images=images, audio=audio, videos=videos,
        ):
            if isinstance(event, RunOutput):
                final_output = event
        if final_output is None:
            raise RuntimeError(
                "ClaudeBackedAgent._arun_collect: _arun_stream yielded no "
                "terminal RunOutput — SDK stream ended without a ResultMessage."
            )
        return final_output

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
        sdk = _sdk()
        prompt = _stringify_input(input)
        # AgentOS-aligned translation: text content inlined, binary
        # written to a sandbox-safe path under the CLI's effective cwd.
        # Replaces the older path-based ``_prepend_files_for_subscription_cli``
        # which broke on Claude CLI's sandbox.
        from src.models._backed_agent import _inline_attachments_for_subscription_cli
        cwd = self._sdk_options.get("cwd")
        prompt = _inline_attachments_for_subscription_cli(
            prompt,
            files=files, images=images, audio=audio, videos=videos,
            cwd=cwd, session_id=session_id, run_id=run_id,
        )
        self._hydrate_from_db(session_id)
        options = self._build_options(streaming=True, session_id=session_id)

        emitted_tool_ids: set[str] = set()
        tool_info_map: Dict[str, Dict[str, Any]] = {}
        got_stream_events = False
        assistant_text_acc = ""
        captured_sdk_session_id: Optional[str] = None
        tools: list[ToolExecution] = []
        usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        final_result = ""

        try:
            async for message in sdk.query(prompt=prompt, options=options):
                if isinstance(message, sdk.StreamEvent):
                    got_stream_events = True
                    event = message.event
                    event_type = event.get("type", "")
                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                assistant_text_acc += text
                                yield RunContentEvent(
                                    run_id=run_id,
                                    agent_id=self.id or "",
                                    agent_name=self.name or "",
                                    session_id=session_id,
                                    content=text,
                                )
                elif isinstance(message, sdk.SystemMessage):
                    if getattr(message, "subtype", None) == "init":
                        data = getattr(message, "data", {}) or {}
                        sdk_sid = data.get("session_id")
                        if sdk_sid:
                            captured_sdk_session_id = str(sdk_sid)
                elif isinstance(message, sdk.AssistantMessage):
                    for block in message.content:
                        if isinstance(block, sdk.TextBlock):
                            # When StreamEvents are flowing we already
                            # emitted token-level deltas. Otherwise the
                            # whole block content is the assistant's
                            # only chance to surface text.
                            if not got_stream_events and block.text:
                                assistant_text_acc += block.text
                                yield RunContentEvent(
                                    run_id=run_id,
                                    agent_id=self.id or "",
                                    agent_name=self.name or "",
                                    session_id=session_id,
                                    content=block.text,
                                )
                        elif isinstance(block, sdk.ToolUseBlock):
                            tool_id = getattr(block, "id", str(uuid4()))
                            if tool_id in emitted_tool_ids:
                                continue
                            emitted_tool_ids.add(tool_id)
                            tool_name = getattr(block, "name", "unknown")
                            tool_input = getattr(block, "input", {})
                            tool_args = (
                                tool_input
                                if isinstance(tool_input, dict)
                                else {"input": tool_input}
                            )
                            tool_info_map[tool_id] = {
                                "name": tool_name,
                                "args": tool_args,
                            }
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
                elif isinstance(message, sdk.UserMessage):
                    content = message.content
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, sdk.ToolResultBlock):
                                tool_use_id = getattr(block, "tool_use_id", "")
                                result_content = getattr(block, "content", "")
                                if isinstance(result_content, list):
                                    result_str = " ".join(
                                        getattr(item, "text", str(item))
                                        for item in result_content
                                    )
                                else:
                                    result_str = (
                                        str(result_content) if result_content else ""
                                    )
                                info = tool_info_map.get(tool_use_id, {})
                                # Patch the in-memory ToolExecution.
                                for te in tools:
                                    if te.tool_call_id == tool_use_id:
                                        te.result = result_str
                                        break
                                yield ToolCallCompletedEvent(
                                    run_id=run_id,
                                    agent_id=self.id or "",
                                    agent_name=self.name or "",
                                    session_id=session_id,
                                    tool=ToolExecution(
                                        tool_call_id=tool_use_id,
                                        tool_name=info.get("name", ""),
                                        tool_args=info.get("args"),
                                        result=result_str,
                                    ),
                                )
                elif isinstance(message, sdk.ResultMessage):
                    self._check_result_message(sdk, message)
                    sdk_sid = getattr(message, "session_id", None)
                    if sdk_sid:
                        captured_sdk_session_id = str(sdk_sid)
                    if getattr(message, "result", None):
                        final_result = str(message.result)
                    usage_obj = getattr(message, "usage", None)
                    if isinstance(usage_obj, dict):
                        usage["input_tokens"] = int(usage_obj.get("input_tokens", 0) or 0)
                        usage["output_tokens"] = int(usage_obj.get("output_tokens", 0) or 0)
        except Exception as e:  # noqa: BLE001
            elog(
                self.STREAM_ERROR_EVENT,
                session_id=session_id,
                error=str(e),
            )
            raise

        if session_id and captured_sdk_session_id:
            self._sdk_session_ids[session_id] = captured_sdk_session_id
            self._save_to_db(session_id)

        if yield_run_output:
            content = final_result or assistant_text_acc
            metrics = self._build_metrics(usage["input_tokens"], usage["output_tokens"])
            yield RunOutput(
                run_id=run_id,
                agent_id=self.id,
                agent_name=self.name,
                session_id=session_id,
                user_id=user_id,
                content=content,
                content_type="str",
                model=self._sdk_model_id,
                model_provider=FRAMEWORK_CLAUDE_CLI,
                tools=tools or None,
                metrics=metrics,
                status=RunStatus.completed,
            )


# ════════════════════════════════════════════════════════════════════
#  Factory + provider facade (ClaudeCLI / ClaudeCLIRegistry)
# ════════════════════════════════════════════════════════════════════


def _build_sdk_options(
    *,
    allowed_tools: list[str] | None,
    mcp_servers: dict[str, dict] | None,
    system: str | None,
) -> dict[str, Any]:
    """Compose the dict of SDK options ``ClaudeBackedAgent`` accepts.

    Pulled out so ``TeamRouterProvider`` can reuse the exact same
    option-building logic when it constructs a ``ClaudeBackedAgent``
    for a claude-cli row (leader OR member).
    """
    max_turns_raw = os.environ.get("OPENAGENT_CLAUDE_MAX_TURNS", "").strip()
    try:
        max_turns = int(max_turns_raw) if max_turns_raw else 20
    except ValueError:
        max_turns = 20
    # ``bypassPermissions`` matches OpenAgent's pre-approval model. The SDK
    # default (``"default"``) prompts a human for every Write / Bash / MCP
    # call — fine for the interactive ``claude`` CLI where the user clicks
    # "Allow", broken in an agent server where no one is sitting there. The
    # framework prompt at ``src.core.prompts.FRAMEWORK_SYSTEM_PROMPT``
    # already tells the model "tool calls are pre-approved" — this option
    # is what makes that statement true at the SDK boundary. Override via
    # ``OPENAGENT_CLAUDE_PERMISSION=default`` (or any other SDK-recognised
    # mode) when you want stricter behaviour.
    opts: dict[str, Any] = {
        "permission_mode": os.environ.get(
            "OPENAGENT_CLAUDE_PERMISSION", "bypassPermissions",
        ),
        "max_turns": max_turns,
    }
    if system:
        opts["system_prompt"] = system
    if allowed_tools:
        opts["allowed_tools"] = list(allowed_tools)
    if mcp_servers:
        opts["mcp_servers"] = dict(mcp_servers)
    return opts


def build_claude_backed_agent(
    *,
    name: str,
    model_id: str | None,
    allowed_tools: list[str] | None = None,
    mcp_servers: dict[str, dict] | None = None,
    system: str | None = None,
    db: Any = None,
    role: str | None = None,
) -> Any:
    """Factory: instantiate a ``ClaudeBackedAgent`` configured for either
    standalone use (via ``ClaudeCLI`` below) or Team membership (via
    ``TeamRouterProvider``).
    """
    sdk_model_id = _sanitize_claude_model_id(model_id) or "default"
    sdk_options = _build_sdk_options(
        allowed_tools=allowed_tools,
        mcp_servers=mcp_servers,
        system=system,
    )
    return ClaudeBackedAgent(
        name=name,
        sdk_model_id=sdk_model_id,
        sdk_options=sdk_options,
        db=_make_sqlite_db(db),
        role=role,
    )


class ClaudeCLI(BaseModel):
    """Single-model wrapper around ``ClaudeBackedAgent``.

    ``ClaudeCLIRegistry`` (below) is the real entry point — one
    ``ClaudeCLI`` per session, keyed by ``session_id``. This class
    stays exported so existing imports (``src.models.__init__``,
    tests) keep working.
    """

    history_mode = "provider"

    def __init__(
        self,
        model: str | None = None,
        allowed_tools: list[str] | None = None,
        mcp_servers: dict[str, dict] | None = None,
        providers_config: Any = None,
    ):
        if model:
            self.model: str | None = model_id_from_runtime(model) or model
        else:
            self.model = model
        self.allowed_tools = list(allowed_tools or [])
        self.mcp_servers: dict[str, dict] = dict(mcp_servers or {})
        self._providers_config = providers_config if providers_config is not None else []
        self._db: Any = None
        self._agent: Any = None
        self._agent_system: str | None = None
        self._session_handles: dict[str, str] = {}

    # ── wiring ────────────────────────────────────────────────────────

    def invalidate_agent_cache(self) -> None:
        """Drop the cached backed Agent so the next call rebuilds it.

        Public hook called from the registry's ``_retune_model`` after
        a per-session model swap — keeps the registry out of private
        attribute mutation.
        """
        self._agent = None
        self._agent_system = None

    def set_mcp_servers(self, servers: dict[str, dict]) -> None:
        new_servers = dict(servers or {})
        if new_servers == self.mcp_servers:
            return
        self.mcp_servers = new_servers
        self.invalidate_agent_cache()
        elog(
            "claude_cli.mcp_servers_wired",
            count=len(self.mcp_servers),
            names=sorted(self.mcp_servers.keys()),
        )

    def set_db(self, db: Any) -> None:
        if db is self._db:
            return
        self._db = db
        # Rebuild the agent so its ``SqliteDb`` points at the new file.
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
        sanitized = _sanitize_claude_model_id(self.model)
        self._agent = build_claude_backed_agent(
            name=f"claude-{sanitized or 'default'}",
            model_id=sanitized,
            allowed_tools=self.allowed_tools or None,
            mcp_servers=self.mcp_servers or None,
            system=system or None,
            db=self._db,
        )
        self._agent_system = system or None
        return self._agent

    # ── lifecycle (mostly no-ops) ─────────────────────────────────────

    async def close_session(self, session_id: str) -> None:
        # ``ClaudeBackedAgent`` doesn't keep a persistent subprocess
        # between turns — each ``arun`` call spawns + drains its own
        # SDK process. Nothing to tear down per-session.
        return None

    async def forget_session(self, session_id: str) -> None:
        """Erase the session row from the sessions table so the next
        message starts a fresh SDK transcript with no ``--resume``.
        """
        if not session_id:
            return
        sqlite_db = _make_sqlite_db(self._db)
        if sqlite_db is None:
            return
        try:
            sqlite_db.delete_session(session_id=session_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("ClaudeCLI.forget_session %s: %s", session_id, e)
        elog("claude_cli.session_forget", session_id=session_id)

    async def commit_partial_assistant(self, session_id: str, text: str) -> None:
        # Barge-in support is gone — the SDK doesn't expose an interrupt
        # control-request from inside ClaudeBackedAgent either.
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
        images: list[Any] | None = None,
        audio: list[Any] | None = None,
        videos: list[Any] | None = None,
    ) -> ModelResponse:
        # ``tools`` is ignored: the Claude SDK doesn't accept
        # provider-neutral tool defs — its tools come from
        # ``allowed_tools`` + ``mcp_servers``.
        agent = self._ensure_agent(system)
        prompt = _last_user_prompt(messages)
        # All four media kwargs flow through the runtime's ``arun`` signature
        # into ``ClaudeBackedAgent._arun_stream``, which translates them
        # for the Claude CLI subprocess (text inlined into the prompt,
        # binary written to a sandbox-safe path inside the CLI's cwd).
        # Same shape the runtime's native model adapters consume — easy
        # swap to an API-direct provider with no caller changes.
        sid = session_id or "default"
        user_id = self._session_handles.get(sid) or "openagent"
        try:
            run_output = await agent.arun(
                prompt, session_id=sid, user_id=user_id, stream=False,
                files=files, images=images, audio=audio, videos=videos,
            )
        except Exception as e:  # noqa: BLE001
            elog("claude_cli.generate_error", session_id=sid, error=str(e))
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
            model=_sanitize_claude_model_id(self.model),
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
        images: list[Any] | None = None,
        audio: list[Any] | None = None,
        videos: list[Any] | None = None,
    ) -> AsyncIterator[str]:
        from src.models._tool_status import tool_exec_to_wire_json

        agent = self._ensure_agent(system)
        prompt = _last_user_prompt(messages)
        # See ``generate`` above for the media-kwargs contract.
        sid = session_id or "default"
        user_id = self._session_handles.get(sid) or "openagent"

        stream_iter = None
        try:
            # ``arun(stream=True)`` returns an async iterator directly
            # (the runtime's contract — sync function, returns AsyncIterator).
            stream_iter = agent.arun(
                prompt, session_id=sid, user_id=user_id, stream=True,
                files=files, images=images, audio=audio, videos=videos,
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
            elog("claude_cli.stream_error", session_id=sid, error=str(e))
            raise
        finally:
            if stream_iter is not None:
                aclose = getattr(stream_iter, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:  # noqa: BLE001
                        pass


class ClaudeCLIRegistry(BaseModel):
    """Per-session registry of ``ClaudeCLI`` instances.

    ``SmartRouter`` instantiates one of these for the claude-cli
    framework path; each session gets its own ``ClaudeCLI`` so per-
    session config (model pin, system prompt) can vary without bleed.
    """

    history_mode = "provider"

    def __init__(
        self,
        default_model: str | None = None,
        allowed_tools: list[str] | None = None,
        mcp_servers: dict[str, dict] | None = None,
        providers_config: Any = None,
    ):
        self._default_model = (default_model or "").strip() or None
        self._allowed_tools = list(allowed_tools or [])
        self._mcp_servers: dict[str, dict] = dict(mcp_servers or {})
        self._providers_config = providers_config if providers_config is not None else []
        self._db: Any = None
        self._instances: dict[str, ClaudeCLI] = {}
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
        self._mcp_servers = dict(servers or {})
        elog(
            "claude_cli.registry_mcp_wired",
            count=len(self._mcp_servers),
            names=sorted(self._mcp_servers.keys()),
        )
        for inst in self._instances.values():
            inst.set_mcp_servers(self._mcp_servers)

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
                logger.debug("registry.cleanup_idle: %s", e)

    async def shutdown(self) -> None:
        for inst in list(self._instances.values()):
            try:
                await inst.shutdown()
            except Exception as e:  # noqa: BLE001
                logger.debug("registry.shutdown: %s", e)
        self._instances.clear()

    async def close_session(self, session_id: str) -> None:
        inst = self._instances.get(session_id)
        if inst is not None:
            try:
                await inst.close_session(session_id)
            except Exception as e:  # noqa: BLE001
                logger.debug("registry.close_session: %s", e)
        self._session_model.pop(session_id, None)

    async def forget_session(self, session_id: str) -> None:
        inst = self._instances.pop(session_id, None)
        if inst is not None:
            try:
                await inst.forget_session(session_id)
            except Exception as e:  # noqa: BLE001
                logger.debug("registry.forget_session: %s", e)
            try:
                await inst.shutdown()
            except Exception as e:  # noqa: BLE001
                logger.debug("registry.forget_session shutdown: %s", e)
        self._session_model.pop(session_id, None)

    async def commit_partial_assistant(self, session_id: str, text: str) -> None:
        # Barge-in support dropped with the runtime migration; intentional
        # no-op so the gateway's interrupt path stays callable.
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

    def _get_or_create(self, session_id: str, model_id: str | None) -> ClaudeCLI:
        inst = self._instances.get(session_id)
        if inst is not None:
            return inst
        if not model_id:
            raise ValueError(
                "ClaudeCLIRegistry._get_or_create: first-spawn for session "
                f"{session_id!r} requires a model_id; the router must pin a "
                "concrete runtime_id before dispatching to claude-cli."
            )
        bare = model_id_from_runtime(model_id) or model_id
        inst = ClaudeCLI(
            model=bare,
            allowed_tools=self._allowed_tools,
            mcp_servers=self._mcp_servers,
            providers_config=self._providers_config,
        )
        if self._db is not None:
            inst.set_db(self._db)
        handle = self._session_handles.get(session_id)
        if handle:
            inst.set_session_handle(session_id, handle)
        self._instances[session_id] = inst
        elog(
            "claude_cli_registry.instance_created",
            session_id=session_id,
            initial_model=bare,
        )
        return inst

    @staticmethod
    def _retune_model(inst: ClaudeCLI, target_model: str | None) -> None:
        """Swap the per-session ``ClaudeCLI``'s model when routing
        picks a different runtime_id mid-conversation. Invalidates the
        cached agent so the next call rebuilds with the new model id.
        """
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
        images: list[Any] | None = None,
        audio: list[Any] | None = None,
        videos: list[Any] | None = None,
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
            images=images,
            audio=audio,
            videos=videos,
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
        images: list[Any] | None = None,
        audio: list[Any] | None = None,
        videos: list[Any] | None = None,
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
            images=images,
            audio=audio,
            videos=videos,
        ):
            yield delta
