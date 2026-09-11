"""A failed turn emits exactly one ``OutError``, wherever it failed.

Two things can go wrong in a turn and they arrive by different routes:
``Agent.run_stream`` catches most failures and reports them as an errored
``done`` frame, while anything raised out of the generator reaches the
runner as an exception. Both set ``stream_error`` and both must look the
same to a client — one error event, published with the other terminal
frames, immediately before the ``TurnComplete`` that says the turn ended
on ``error``. Emitting it at the point of failure instead put it before
the final text on one path and emitted nothing at all on the other.
"""
from __future__ import annotations

from ._framework import TestContext, test


def _make_agent(frames=None, raises=None):
    class Agent:
        db = None

        async def run_stream(self, **_kwargs):
            if raises is not None:
                raise raises
            for frame in frames or []:
                yield frame

        def last_response_meta(self, _session_id):
            return {}

    return Agent


async def _drain(agent_cls):
    from src.stream.session import StreamSession, StreamTurnRunner

    session = StreamSession(
        agent_cls(), client_id="dev", session_id="s", coalesce_window_ms=0,
    )
    result = await StreamTurnRunner(agent_cls(), session).run(
        "hello", client_id="dev", session_id="s",
    )
    frames = []
    while not session.outbound.empty():
        frames.append(session.outbound.get_nowait())
    return result, frames


def _assert_single_error_before_turn_end(frames, label):
    from src.stream.events import (
        OutError, TurnComplete, TURN_END_ERROR,
    )

    errors = [f for f in frames if isinstance(f, OutError)]
    completions = [f for f in frames if isinstance(f, TurnComplete)]
    assert len(errors) == 1, (
        f"{label}: expected exactly one OutError, got {len(errors)} "
        f"({[type(f).__name__ for f in frames]})"
    )
    assert len(completions) == 1, f"{label}: expected one TurnComplete"
    assert completions[0].reason == TURN_END_ERROR, (
        f"{label}: turn ended {completions[0].reason!r}, not error"
    )
    assert isinstance(frames[-1], TurnComplete), (
        f"{label}: TurnComplete must be last, got {type(frames[-1]).__name__}"
    )
    assert isinstance(frames[-2], OutError), (
        f"{label}: OutError must sit immediately before TurnComplete, got "
        f"{type(frames[-2]).__name__}"
    )


@test("turn_error_frames", "an errored done frame produces one OutError at the end")
async def t_errored_done_frame(_ctx: TestContext) -> None:
    agent = _make_agent(frames=[{
        "kind": "done",
        "text": "the provider refused",
        "errored": True,
        "error_code": "auth",
    }])
    result, frames = await _drain(agent)
    assert result["errored"] is True, result
    _assert_single_error_before_turn_end(frames, "reported failure")


@test("turn_error_frames", "a raised failure produces one OutError too")
async def t_raised_failure(_ctx: TestContext) -> None:
    """The path that used to emit nothing. ``stream_error`` was set, the
    turn correctly ended on ``error``, and no error event was ever
    published — so a client that renders ``OutError`` showed a turn that
    simply stopped."""
    agent = _make_agent(raises=RuntimeError("the runner blew up"))
    result, frames = await _drain(agent)
    assert result["errored"] is True, result
    _assert_single_error_before_turn_end(frames, "raised failure")


@test("turn_error_frames", "a healthy turn emits no error frame")
async def t_healthy_turn(_ctx: TestContext) -> None:
    from src.stream.events import OutError, TURN_END_COMPLETED, TurnComplete

    agent = _make_agent(frames=[{"kind": "done", "text": "all good"}])
    result, frames = await _drain(agent)
    assert result["errored"] is False, result
    assert not [f for f in frames if isinstance(f, OutError)], (
        "a successful turn must publish no OutError"
    )
    completions = [f for f in frames if isinstance(f, TurnComplete)]
    assert completions and completions[0].reason == TURN_END_COMPLETED, (
        f"expected a completed turn, got {completions}"
    )
