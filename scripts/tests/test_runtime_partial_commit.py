"""Interrupted-turn history backfill — ``upsert_run`` + ``synth_interrupted_messages``.

When the user stops a turn the runner persists the run as ``status=CANCELLED``
with the user's question in ``input`` but no ``messages`` — and history skips
CANCELLED runs, so the stopped turn would vanish and the next turn ("continua")
would lose all context. ``Session.upsert_run`` backfills the user message and
re-marks the run COMPLETED so it stays in history. These tests pin that.
"""
from __future__ import annotations

import time

from ._framework import TestContext, TestSkip, test


def _imports():
    from src.memory.sessions.agent import AgentSession
    from src.memory.sessions.team import TeamSession
    from src.core._run_state.agent import RunOutput, RunInput
    from src.core._run_state.team import TeamRunOutput
    from src.core._run_state.base import RunStatus
    from src.models.providers.message import Message
    return AgentSession, TeamSession, RunOutput, RunInput, TeamRunOutput, RunStatus, Message


@test("runtime_partial_commit", "AgentSession.upsert_run backfills a cancelled turn into history")
async def t_agent_backfill(_ctx: TestContext) -> None:
    try:
        AgentSession, _T, RunOutput, RunInput, _TR, RunStatus, _M = _imports()
    except ImportError as e:
        raise TestSkip(f"runtime not available: {e}")

    now = int(time.time())
    s = AgentSession(session_id="sid", agent_id="a", user_id="u", created_at=now, updated_at=now)
    s.upsert_run(RunOutput(
        run_id="r-1", agent_id="a", session_id="sid", user_id="u",
        input=RunInput(input_content="scrivi un poema sul mare"),
        status=RunStatus.cancelled,  # stopped mid-stream, no messages assembled
    ))
    run = s.runs[0]
    assert run.status == RunStatus.completed, run.status  # promoted so history keeps it
    roles = [(m.role, m.content) for m in (run.messages or [])]
    assert roles == [("user", "scrivi un poema sul mare")], roles


@test("runtime_partial_commit", "TeamSession.upsert_run backfills a cancelled turn into history")
async def t_team_backfill(_ctx: TestContext) -> None:
    try:
        _A, TeamSession, _R, RunInput, TeamRunOutput, RunStatus, _M = _imports()
    except ImportError as e:
        raise TestSkip(f"runtime not available: {e}")

    now = int(time.time())
    s = TeamSession(session_id="sid", team_id="t", user_id="u", created_at=now, updated_at=now)
    s.upsert_run(TeamRunOutput(
        run_id="r-1", team_id="t", session_id="sid", user_id="u",
        input=RunInput(input_content="conta da 1 a 300"),
        status=RunStatus.cancelled,
    ))
    run = s.runs[0]
    assert run.status == RunStatus.completed, run.status
    roles = [(m.role, m.content) for m in (run.messages or [])]
    assert roles == [("user", "conta da 1 a 300")], roles


@test("runtime_partial_commit", "a completed run with messages is left untouched")
async def t_completed_untouched(_ctx: TestContext) -> None:
    try:
        AgentSession, _T, RunOutput, _RI, _TR, RunStatus, Message = _imports()
    except ImportError as e:
        raise TestSkip(f"runtime not available: {e}")

    now = int(time.time())
    s = AgentSession(session_id="sid", agent_id="a", user_id="u", created_at=now, updated_at=now)
    original = [Message(role="user", content="q"), Message(role="assistant", content="a")]
    s.upsert_run(RunOutput(
        run_id="r-1", agent_id="a", session_id="sid", user_id="u",
        content="a", messages=list(original), status=RunStatus.completed,
    ))
    run = s.runs[0]
    assert run.status == RunStatus.completed
    assert [(m.role, m.content) for m in run.messages] == [("user", "q"), ("assistant", "a")]


@test("runtime_partial_commit", "synth is a no-op when there is no user input to recover")
async def t_synth_noop(_ctx: TestContext) -> None:
    try:
        _A, _T, RunOutput, _RI, _TR, RunStatus, _M = _imports()
        from src.memory.sessions._synth import synth_interrupted_messages
    except ImportError as e:
        raise TestSkip(f"runtime not available: {e}")

    # No input → nothing to recover; the run (and a RUNNING in-flight run) is
    # left exactly as-is.
    assert synth_interrupted_messages(RunOutput(run_id="x", status=RunStatus.cancelled)) is None

    from src.memory.sessions.agent import AgentSession
    from src.core._run_state.agent import RunInput
    now = int(time.time())
    s = AgentSession(session_id="sid", agent_id="a", user_id="u", created_at=now, updated_at=now)
    # RUNNING (in-flight) with input but no messages must NOT be backfilled/promoted.
    s.upsert_run(RunOutput(
        run_id="r-1", agent_id="a", session_id="sid", user_id="u",
        input=RunInput(input_content="still going"), status=RunStatus.running,
    ))
    assert s.runs[0].status == RunStatus.running, s.runs[0].status
    assert not s.runs[0].messages


@test("runtime_partial_commit", "a stopped run that DID assemble messages is promoted to COMPLETED")
async def t_cancelled_with_messages_promoted(_ctx: TestContext) -> None:
    try:
        AgentSession, _T, RunOutput, _RI, _TR, RunStatus, Message = _imports()
    except ImportError as e:
        raise TestSkip(f"runtime not available: {e}")

    now = int(time.time())
    s = AgentSession(session_id="sid", agent_id="a", user_id="u", created_at=now, updated_at=now)
    # Stopped late enough that the runner already assembled messages (incl. the
    # streamed partial). It must STILL be promoted so history keeps it.
    s.upsert_run(RunOutput(
        run_id="r-1", agent_id="a", session_id="sid", user_id="u",
        content="1\n2\n3", status=RunStatus.cancelled,
        messages=[Message(role="user", content="conta"), Message(role="assistant", content="1\n2\n3")],
    ))
    run = s.runs[0]
    assert run.status == RunStatus.completed, run.status
    assert [(m.role, m.content) for m in run.messages] == [("user", "conta"), ("assistant", "1\n2\n3")]


@test("runtime_partial_commit", "synth keeps the partial reply but skips the cancel sentinels")
async def t_synth_partial_and_sentinels(_ctx: TestContext) -> None:
    try:
        _A, _T, RunOutput, RunInput, _TR, RunStatus, _M = _imports()
        from src.memory.sessions._synth import synth_interrupted_messages
    except ImportError as e:
        raise TestSkip(f"runtime not available: {e}")

    def shell(content):
        return RunOutput(run_id="rid", status=RunStatus.cancelled,
                         input=RunInput(input_content="scrivi una poesia"), content=content)

    # Real partial → kept as the assistant message.
    got = synth_interrupted_messages(shell("Tramonto sul mare\nQuando il giorno"))
    assert [(m.role, m.content) for m in got] == [
        ("user", "scrivi una poesia"), ("assistant", "Tramonto sul mare\nQuando il giorno")]
    # Sentinels (nothing was generated) → user only, no bogus assistant.
    assert [m.role for m in synth_interrupted_messages(shell("Run rid was cancelled"))] == ["user"]
    assert [m.role for m in synth_interrupted_messages(shell("Operation cancelled by user"))] == ["user"]
