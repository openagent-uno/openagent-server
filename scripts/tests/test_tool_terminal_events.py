"""Tool calls expose exactly one terminal stream event.

The runtime model reports a failed function execution through a
``ModelResponseEvent.tool_call_completed`` frame whose ``ToolExecution`` has
``tool_call_error=True``.  The runner used to translate that into BOTH a
``ToolCallCompletedEvent`` and a ``ToolCallErrorEvent`` for the same id.  Live
clients consequently received two mutually exclusive terminal states and
rendered duplicate cards until rehydration reconciled the transcript.

These tests pin the contract at every producer boundary: agent/team model
response processing plus synchronous/asynchronous continued tool execution.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ._framework import TestContext, test


def _tool(*, failed: bool, result: str | None = None):
    from src.models.providers.response import ToolExecution

    return ToolExecution(
        tool_call_id="call-terminal-1",
        tool_name="shell_run",
        tool_args={"cmd": "false" if failed else "true"},
        tool_call_error=failed,
        result=result if result is not None else ("exit 1" if failed else "ok"),
    )


def _agent_stub(model: Any | None = None):
    return SimpleNamespace(
        id="agent-terminal-test",
        name="Terminal test agent",
        model=model,
        events_to_skip=[],
        store_events=True,
    )


def _assert_agent_terminal(events: list[Any], *, failed: bool) -> None:
    from src.core._run_state.agent import ToolCallCompletedEvent, ToolCallErrorEvent

    terminals = [e for e in events if isinstance(e, (ToolCallCompletedEvent, ToolCallErrorEvent))]
    assert len(terminals) == 1, f"expected one agent terminal event, got {terminals!r}"
    expected = ToolCallErrorEvent if failed else ToolCallCompletedEvent
    assert isinstance(terminals[0], expected), (
        f"failed={failed} produced {type(terminals[0]).__name__}, expected {expected.__name__}"
    )
    if failed:
        assert terminals[0].error == "exit 1", terminals[0]
    else:
        assert terminals[0].content == "provider content", terminals[0]


def _assert_team_terminal(events: list[Any], *, failed: bool) -> None:
    from src.core._run_state.team import ToolCallCompletedEvent, ToolCallErrorEvent

    terminals = [e for e in events if isinstance(e, (ToolCallCompletedEvent, ToolCallErrorEvent))]
    assert len(terminals) == 1, f"expected one team terminal event, got {terminals!r}"
    expected = ToolCallErrorEvent if failed else ToolCallCompletedEvent
    assert isinstance(terminals[0], expected), (
        f"failed={failed} produced {type(terminals[0]).__name__}, expected {expected.__name__}"
    )
    if failed:
        assert terminals[0].error == "exit 1", terminals[0]
    else:
        assert terminals[0].content == "provider content", terminals[0]


@test("tool_terminal_events", "agent + team response handlers emit one terminal state")
async def t_response_handlers_emit_one_terminal(_ctx: TestContext) -> None:
    # Error is now a first-class (and potentially the only) terminal frame;
    # keep it on the runner's public import surface alongside Completed.
    from src.core._runner.agent import ToolCallErrorEvent as PublicAgentToolCallErrorEvent
    from src.core._runner.team import ToolCallErrorEvent as PublicTeamToolCallErrorEvent
    from src.core._run_state.agent import RunOutput
    from src.core._run_state.team import TeamRunOutput
    from src.core._runner.agent._response import handle_model_response_chunk
    from src.core._runner.team._response import _handle_model_response_chunk
    from src.memory.sessions.agent import AgentSession
    from src.memory.sessions.team import TeamSession
    from src.models.providers.response import ModelResponse, ModelResponseEvent, ToolExecution

    assert PublicAgentToolCallErrorEvent.__name__ == "ToolCallErrorEvent"
    assert PublicTeamToolCallErrorEvent.__name__ == "ToolCallErrorEvent"

    for failed in (False, True):
        final_tool = _tool(failed=failed)
        initial_tool = ToolExecution(
            tool_call_id=final_tool.tool_call_id,
            tool_name=final_tool.tool_name,
            tool_args=final_tool.tool_args,
        )
        response_frame = ModelResponse(
            event=ModelResponseEvent.tool_call_completed.value,
            content="provider content",
            tool_executions=[final_tool],
        )

        agent_run = RunOutput(
            run_id="agent-run",
            agent_id="agent-terminal-test",
            agent_name="Terminal test agent",
            session_id="agent-session",
            tools=[initial_tool],
        )
        agent_events = list(
            handle_model_response_chunk(
                _agent_stub(),
                AgentSession(
                    session_id="agent-session",
                    session_data={"session_state": {}},
                ),
                agent_run,
                ModelResponse(),
                response_frame,
                stream_events=True,
            )
        )
        _assert_agent_terminal(agent_events, failed=failed)
        assert agent_run.tools == [final_tool], agent_run.tools
        assert agent_run.events == agent_events, agent_run.events

        team_tool = _tool(failed=failed)
        team_initial = ToolExecution(
            tool_call_id=team_tool.tool_call_id,
            tool_name=team_tool.tool_name,
            tool_args=team_tool.tool_args,
        )
        team_frame = ModelResponse(
            event=ModelResponseEvent.tool_call_completed.value,
            content="provider content",
            tool_executions=[team_tool],
        )
        team_run = TeamRunOutput(
            run_id="team-run",
            team_id="team-terminal-test",
            team_name="Terminal test team",
            session_id="team-session",
            tools=[team_initial],
        )
        team = SimpleNamespace(
            id="team-terminal-test",
            name="Terminal test team",
            events_to_skip=[],
            store_events=True,
            stream_member_events=True,
        )
        team_events = list(
            _handle_model_response_chunk(
                team,
                TeamSession(
                    session_id="team-session",
                    session_data={"session_state": {}},
                ),
                team_run,
                ModelResponse(),
                team_frame,
                stream_events=True,
            )
        )
        _assert_team_terminal(team_events, failed=failed)
        assert team_run.tools == [team_tool], team_run.tools
        assert team_run.events == team_events, team_run.events


class _SyncToolModel:
    def __init__(self, *, failed: bool) -> None:
        self.failed = failed

    def get_function_call_to_run_from_tool_execution(self, tool, functions):
        return object()

    def run_function_call(self, *, function_call, function_call_results):
        from src.models.providers.response import ModelResponse, ModelResponseEvent

        yield ModelResponse(event=ModelResponseEvent.tool_call_started.value)
        yield ModelResponse(
            event=ModelResponseEvent.tool_call_completed.value,
            content="provider content",
            tool_executions=[_tool(failed=self.failed)],
        )


class _AsyncToolModel:
    def __init__(self, *, failed: bool) -> None:
        self.failed = failed

    def get_function_call_to_run_from_tool_execution(self, tool, functions):
        return object()

    async def arun_function_calls(
        self,
        *,
        function_calls,
        function_call_results,
        skip_pause_check,
    ):
        from src.models.providers.response import ModelResponse, ModelResponseEvent

        yield ModelResponse(event=ModelResponseEvent.tool_call_started.value)
        yield ModelResponse(
            event=ModelResponseEvent.tool_call_completed.value,
            content="provider content",
            tool_executions=[_tool(failed=self.failed)],
        )


def _execution_run(*, team_mode: bool):
    if team_mode:
        from src.core._run_state.team import TeamRunOutput

        return TeamRunOutput(
            run_id="team-tool-run",
            team_id="team-terminal-test",
            team_name="Terminal test team",
            session_id="team-tool-session",
        )

    from src.core._run_state.agent import RunOutput

    return RunOutput(
        run_id="agent-tool-run",
        agent_id="agent-terminal-test",
        agent_name="Terminal test agent",
        session_id="agent-tool-session",
    )


@test("tool_terminal_events", "sync continued tools emit started + one terminal state")
async def t_sync_tool_emits_one_terminal(_ctx: TestContext) -> None:
    from src.core._run_state.messages import RunMessages
    from src.core._runner.agent._tools import run_tool

    for team_mode in (False, True):
        for failed in (False, True):
            run_response = _execution_run(team_mode=team_mode)
            tool = _tool(failed=False, result=None)
            tool.result = None
            events = list(
                run_tool(
                    _agent_stub(_SyncToolModel(failed=failed)),
                    run_response,
                    RunMessages(),
                    tool,
                    stream_events=True,
                    team_mode=team_mode,
                )
            )
            if team_mode:
                _assert_team_terminal(events, failed=failed)
            else:
                _assert_agent_terminal(events, failed=failed)
            assert tool.tool_call_error is failed, tool
            assert tool.result == ("exit 1" if failed else "ok"), tool
            assert run_response.events == events, run_response.events


@test("tool_terminal_events", "async continued tools emit started + one terminal state")
async def t_async_tool_emits_one_terminal(_ctx: TestContext) -> None:
    from src.core._run_state.messages import RunMessages
    from src.core._runner.agent._tools import arun_tool

    for team_mode in (False, True):
        for failed in (False, True):
            run_response = _execution_run(team_mode=team_mode)
            tool = _tool(failed=False, result=None)
            tool.result = None
            events = [
                event
                async for event in arun_tool(
                    _agent_stub(_AsyncToolModel(failed=failed)),
                    run_response,
                    RunMessages(),
                    tool,
                    stream_events=True,
                    team_mode=team_mode,
                )
            ]
            if team_mode:
                _assert_team_terminal(events, failed=failed)
            else:
                _assert_agent_terminal(events, failed=failed)
            assert tool.tool_call_error is failed, tool
            assert tool.result == ("exit 1" if failed else "ok"), tool
            assert run_response.events == events, run_response.events
