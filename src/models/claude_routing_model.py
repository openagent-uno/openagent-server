"""Claude SDK routing model — runtime ``Model`` that drives the Claude
SDK as the Team coordinator when no api-based model is available.

The runtime's ``Team`` always needs a ``team.model`` for the coordinator
turn. For api-based leaders the team.model is the leader's own runtime
model; for subscription-CLI leaders the dispatcher historically borrowed
the cheapest enabled api-based model. When the catalog contains NO
api-based models — vision §3 allows this — there's nothing to borrow,
and the dispatcher used to fall back to single-agent dispatch with the
team's members dropped entirely.

This module restores the team build by exposing the Claude SDK as a
runtime ``Model``. The SDK has its own internal tool-call loop driven
by ``mcp_servers`` config; we bridge the Team's coordinator tools
(``delegate_task_to_member``, ``get_member_information`` …) into an
in-process SDK MCP server so the SDK can execute them natively. The
runtime Team's outer loop sees no ``tool_calls`` in our
``ModelResponse`` and exits after one iteration with the SDK's final
text answer.

Tradeoffs:
- Intermediate member ``RunOutputEvent`` / ``TeamRunOutputEvent``
  streams produced by ``adelegate_task_to_member`` are drained inside
  the SDK MCP bridge and not re-emitted to the runtime Team's outer
  loop, so the chat UI's mid-turn member progress events don't fire
  for delegations made through this routing path. The final member
  response is still surfaced as the tool result text.
- ``tool_executions`` for each SDK-side tool call are tracked and
  surfaced on the final ``ModelResponse`` so the dispatcher can still
  extract the delegated ``member_id`` and emit ``team_router.delegate``
  log events.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, List, Optional, Union
from uuid import uuid4

from src.core._run_state.agent import RunOutput, RunOutputEvent
from src.core._run_state.team import TeamRunOutput, TeamRunOutputEvent
from src.mcp._runtime.function import Function
from src.models.providers.base import Model
from src.models.providers.message import Message
from src.models.providers.response import (
    ModelResponse,
    ModelResponseEvent,
    ToolExecution,
)

logger = logging.getLogger(__name__)


def _sdk() -> Any:
    """Lazy-import ``claude_agent_sdk`` so non-claude installs still
    parse this module at import time."""
    try:
        import claude_agent_sdk  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "claude_agent_sdk is required for ClaudeSdkRoutingModel. "
            "Install OpenAgent's claude-cli extra (or "
            "`pip install claude-agent-sdk`)."
        ) from e
    return claude_agent_sdk


def _flatten_messages_to_prompt(
    messages: List[Message],
) -> tuple[str, str]:
    """Flatten the runtime's message list into an SDK ``(prompt, system)``.

    The Claude SDK's ``query()`` takes a single string ``prompt`` and a
    string ``system_prompt`` — not a structured message list with tool
    results. Flatten by joining system/developer messages into the
    system prompt and the rest into a ``User:`` / ``Assistant:`` /
    ``Tool result:`` body. The routing turn is single-shot (the SDK
    handles its own internal tool loop), so we don't need round-trip
    fidelity for prior tool results.
    """
    system_parts: list[str] = []
    body_parts: list[str] = []

    for msg in messages:
        role = msg.role
        content = msg.content or ""
        if isinstance(content, list):
            chunks: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    chunks.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    chunks.append(block)
            content = "\n".join(chunks)
        if not isinstance(content, str):
            content = str(content)

        text = content.strip()
        if not text:
            continue

        if role in ("system", "developer"):
            system_parts.append(content)
        elif role == "user":
            body_parts.append(f"User: {content}")
        elif role == "assistant":
            body_parts.append(f"Assistant: {content}")
        elif role == "tool":
            tool_name = getattr(msg, "tool_name", None) or "tool"
            body_parts.append(f"Tool result ({tool_name}): {content}")

    return (
        "\n\n".join(body_parts),
        "\n\n".join(system_parts),
    )


async def _drain_runtime_tool_result(
    fn: Function, args: dict[str, Any]
) -> str:
    """Call a runtime ``Function``'s entrypoint and collect its result text.

    Runtime tools come in three shapes:

    - async generator (``adelegate_task_to_member`` and friends): yields
      a mix of ``str`` chunks (final result text) and
      ``RunOutputEvent`` / ``TeamRunOutputEvent`` objects (intermediate
      progress). We keep only the str chunks since the SDK MCP tool
      result must be plain text — the intermediate events would have
      been bubbled to the team's outer loop if this were a runtime tool
      call, but the SDK boundary doesn't have that pipe.
    - coroutine: ``await fn(**args)`` returns the result directly.
    - regular callable / sync generator: invoked synchronously; sync
      generators are drained the same as async ones.
    """
    entrypoint = fn.entrypoint
    if entrypoint is None:
        return ""

    if inspect.isasyncgenfunction(entrypoint):
        chunks: list[str] = []
        async for chunk in entrypoint(**args):
            if isinstance(chunk, str):
                chunks.append(chunk)
            # RunOutputEvent / TeamRunOutputEvent: dropped intentionally.
        return "\n".join(chunks)

    if inspect.iscoroutinefunction(entrypoint):
        result = await entrypoint(**args)
        return str(result) if result is not None else ""

    result = entrypoint(**args)
    if inspect.isawaitable(result):
        result = await result
    if inspect.isgenerator(result):
        chunks = [str(c) for c in result if isinstance(c, str)]
        return "\n".join(chunks)
    return str(result) if result is not None else ""


def _build_runtime_tool_bridge(
    tools: Optional[List[Any]],
    executions: list[ToolExecution],
) -> Any | None:
    """Wrap each runtime ``Function`` as an in-process SDK MCP tool.

    The bridge's handler:
      1. Records the tool call as a ``ToolExecution`` so the runtime
         Team can read it off the final ``ModelResponse.tool_executions``
         (which the dispatcher uses to emit ``team_router.delegate``).
      2. Executes the runtime ``Function``'s entrypoint via
         ``_drain_runtime_tool_result``.
      3. Returns the result text to the SDK as a plain text content
         block — the SDK then feeds it back to the model as a
         ``ToolResultBlock`` and the model synthesises the final answer.

    Dict-shaped builtin tools are skipped — those aren't called via the
    Team's coordinator anyway; only the ``Function``-shaped delegation
    tools matter for routing.
    """
    if not tools:
        return None

    sdk = _sdk()
    sdk_tools: list[Any] = []

    for tool in tools:
        if not isinstance(tool, Function):
            continue
        if tool.entrypoint is None:
            continue

        fn = tool  # capture in closure

        async def _handler(args: dict, _fn: Function = fn) -> dict:
            execution = ToolExecution(
                tool_call_id=str(uuid4()),
                tool_name=_fn.name,
                tool_args=dict(args) if isinstance(args, dict) else {"input": args},
            )
            executions.append(execution)
            try:
                result_text = await _drain_runtime_tool_result(_fn, args or {})
            except Exception as exc:  # noqa: BLE001
                execution.tool_call_error = True
                execution.result = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "ClaudeSdkRoutingModel: bridged tool %r raised: %s",
                    _fn.name, exc,
                )
                return {
                    "content": [{"type": "text", "text": execution.result}],
                    "is_error": True,
                }
            execution.result = result_text
            return {
                "content": [{"type": "text", "text": result_text or ""}],
            }

        sdk_tools.append(
            sdk.SdkMcpTool(
                name=fn.name,
                description=fn.description or "",
                input_schema=fn.parameters or {
                    "type": "object",
                    "properties": {},
                },
                handler=_handler,
            )
        )

    if not sdk_tools:
        return None

    return sdk.create_sdk_mcp_server(
        name="oa-team-coordinator",
        version="1.0.0",
        tools=sdk_tools,
    )


@dataclass
class ClaudeSdkRoutingModel(Model):
    """Runtime ``Model`` that drives the Claude SDK as the team coordinator.

    Used as ``team.model`` by ``TeamRouterProvider`` when the catalog
    has no enabled api-based model but at least one subscription-CLI
    model exists.

    Overrides ``aresponse_stream`` / ``aresponse`` directly to bypass
    the base ``Model``'s per-turn tool loop — the Claude SDK has its
    own tool loop driven by ``mcp_servers`` config, so we hand it the
    runtime tools bridged through an in-process SDK MCP server and let
    it run the whole coordinator turn in a single ``sdk.query()`` call.
    """

    id: str = "claude-sdk-routing"
    provider: Optional[str] = "claude-cli"

    # SDK-specific configuration. These are set by the dispatcher when
    # the routing model is built (see ``_choose_routing_model``).
    sdk_model_id: Optional[str] = None
    extra_mcp_servers: Optional[dict[str, Any]] = field(default=None)
    permission_mode: str = "bypassPermissions"
    max_turns: int = 20

    async def aresponse(
        self,
        messages: List[Message],
        response_format: Optional[Any] = None,
        tools: Optional[List[Union[Function, dict]]] = None,
        tool_choice: Optional[Any] = None,
        tool_call_limit: Optional[int] = None,
        run_response: Optional[Union[RunOutput, TeamRunOutput]] = None,
        send_media_to_model: bool = True,
        compression_manager: Optional[Any] = None,
    ) -> ModelResponse:
        """Drain ``aresponse_stream`` and return the terminal response.

        Mirrors the base ``Model.aresponse`` contract: callers that want
        a single value instead of a stream get the assistant-response
        event with content + tool_executions accumulated.
        """
        final: Optional[ModelResponse] = None
        async for event in self.aresponse_stream(
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            tool_call_limit=tool_call_limit,
            stream_model_response=False,
            run_response=run_response,
            send_media_to_model=send_media_to_model,
            compression_manager=compression_manager,
        ):
            if isinstance(event, ModelResponse) and (
                event.event == ModelResponseEvent.assistant_response.value
            ):
                final = event
        return final or ModelResponse(content="")

    async def aresponse_stream(
        self,
        messages: List[Message],
        response_format: Optional[Any] = None,
        tools: Optional[List[Union[Function, dict]]] = None,
        tool_choice: Optional[Any] = None,
        tool_call_limit: Optional[int] = None,
        stream_model_response: bool = True,
        run_response: Optional[Union[RunOutput, TeamRunOutput]] = None,
        send_media_to_model: bool = True,
        compression_manager: Optional[Any] = None,
    ) -> AsyncIterator[Union[ModelResponse, RunOutputEvent, TeamRunOutputEvent]]:
        """Drive the Claude SDK as the team coordinator turn.

        Single-pass: bridge runtime tools as SDK MCP, call ``sdk.query``
        once, yield text deltas + a terminal ``assistant_response`` with
        ``tool_executions``. The base ``Model``'s tool loop is bypassed
        entirely.
        """
        sdk = _sdk()
        prompt, system_from_messages = _flatten_messages_to_prompt(messages)

        executions: list[ToolExecution] = []
        coordinator_server = _build_runtime_tool_bridge(tools, executions)

        mcp_servers: dict[str, Any] = {}
        if self.extra_mcp_servers:
            mcp_servers.update(self.extra_mcp_servers)
        if coordinator_server is not None:
            mcp_servers["oa-team-coordinator"] = coordinator_server

        # The bridged tool names must appear in ``allowed_tools`` for
        # the SDK to surface them to the model. Format is
        # ``mcp__<server>__<tool>`` per the Claude SDK convention.
        allowed_tools: list[str] = []
        if coordinator_server is not None:
            for tool in tools or []:
                if isinstance(tool, Function) and tool.name:
                    allowed_tools.append(f"mcp__oa-team-coordinator__{tool.name}")

        opts_kwargs: dict[str, Any] = {
            "permission_mode": self.permission_mode,
            "max_turns": self.max_turns,
        }
        if self.sdk_model_id:
            opts_kwargs["model"] = self.sdk_model_id
        sys_prompt = (self.system_prompt or system_from_messages or "").strip()
        if sys_prompt:
            opts_kwargs["system_prompt"] = sys_prompt
        if mcp_servers:
            opts_kwargs["mcp_servers"] = mcp_servers
        if allowed_tools:
            opts_kwargs["allowed_tools"] = allowed_tools

        options = sdk.ClaudeAgentOptions(**opts_kwargs)

        yield ModelResponse(event=ModelResponseEvent.model_request_started.value)

        final_text = ""
        input_tokens = 0
        output_tokens = 0
        sdk_error: Exception | None = None

        try:
            async for message in sdk.query(prompt=prompt, options=options):
                if isinstance(message, sdk.AssistantMessage):
                    for block in message.content:
                        if isinstance(block, sdk.TextBlock):
                            delta_text = block.text or ""
                            if delta_text:
                                final_text += delta_text
                                if stream_model_response:
                                    yield ModelResponse(content=delta_text)
                elif isinstance(message, sdk.ResultMessage):
                    is_error = bool(getattr(message, "is_error", False))
                    subtype = getattr(message, "subtype", None)
                    if is_error or (subtype and subtype != "success"):
                        detail = (
                            getattr(message, "errors", None)
                            or getattr(message, "result", None)
                            or getattr(message, "stop_reason", None)
                        )
                        sdk_error = RuntimeError(
                            f"Claude SDK routing error "
                            f"(is_error={is_error}, subtype={subtype}): {detail}"
                        )
                        break
                    if getattr(message, "result", None):
                        # ResultMessage.result is the SDK's final synthesised
                        # text. Use it as the canonical final answer.
                        final_text = str(message.result)
                    usage = getattr(message, "usage", None)
                    if isinstance(usage, dict):
                        input_tokens = int(usage.get("input_tokens", 0) or 0)
                        output_tokens = int(usage.get("output_tokens", 0) or 0)
        except Exception as exc:  # noqa: BLE001
            sdk_error = exc

        if sdk_error is not None:
            logger.warning("ClaudeSdkRoutingModel: %s", sdk_error)
            raise sdk_error

        # If we weren't streaming deltas, emit the full text as a single
        # assistant_response with content. If we were streaming, the
        # deltas already covered it; emit a final assistant_response
        # with tool_executions (and no extra content) so the team's
        # outer loop captures them.
        terminal = ModelResponse(
            content=None if stream_model_response else final_text,
            tool_executions=executions or None,
        )
        yield terminal

        yield ModelResponse(
            event=ModelResponseEvent.model_request_completed.value,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    # ── Abstract method stubs ─────────────────────────────────────────
    #
    # The base ``Model``'s per-turn tool loop is bypassed because
    # ``aresponse_stream`` / ``aresponse`` are overridden directly above.
    # Provide trivial stubs so the abstract contract is satisfied; the
    # ones the runtime actually reaches in normal flow are the two
    # public ``aresponse*`` methods above.

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        raise NotImplementedError(
            "ClaudeSdkRoutingModel only supports the async stream path "
            "(aresponse_stream). Sync invocation is not supported."
        )

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        raise NotImplementedError(
            "ClaudeSdkRoutingModel only supports the async stream path "
            "(aresponse_stream); use aresponse() instead of ainvoke()."
        )

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "ClaudeSdkRoutingModel only supports the async stream path "
            "(aresponse_stream). Sync streaming is not supported."
        )

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "ClaudeSdkRoutingModel overrides aresponse_stream directly; "
            "ainvoke_stream is not reached on the runtime Team's hot path."
        )
        # Make this an async generator so the abstract signature matches
        yield  # pragma: no cover

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        raise NotImplementedError

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        raise NotImplementedError
