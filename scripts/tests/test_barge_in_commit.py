"""StreamSession barge-in semantics — partial-assistant commit on cancel.

Verifies the contract documented in
``StreamSession._cancel_active_turn``:

* Accumulated assistant deltas are committed to the agent via
  ``Agent.commit_partial_assistant`` BEFORE the turn task is cancelled.
* Empty partials skip the commit entirely (no synthetic empty turn).
* Provider failures during commit are logged and never propagate — the
  cancel path must complete regardless of history-write outcomes.
"""
from __future__ import annotations

import asyncio

from ._framework import TestContext, test


class _RecordingAgent:
    """Agent stub that records commit calls and yields scripted deltas."""

    name = "fake"
    db = None

    def __init__(self, deltas: list[str], commit_raises: bool = False):
        self._deltas = deltas
        self.commits: list[tuple[str, str]] = []
        self._commit_raises = commit_raises

    async def run_stream(
        self, *, message, user_id, session_id, attachments=None, on_status=None,
    ):
        for d in self._deltas:
            await asyncio.sleep(0.01)
            yield {"kind": "delta", "text": d}
        # Long final pause so the cancel path lands while we're "speaking".
        await asyncio.sleep(2.0)
        yield {"kind": "done", "text": "".join(self._deltas)}

    async def commit_partial_assistant(self, session_id: str, text: str) -> None:
        self.commits.append((session_id, text))
        if self._commit_raises:
            raise RuntimeError("boom")

    def last_response_meta(self, session_id: str) -> dict:
        return {"model": "fake-model"}


async def _null_factory(_db):
    return None


@test("barge_in", "_cancel_active_turn commits partial before cancelling")
async def t_commit_before_cancel(_ctx: TestContext) -> None:
    from openagent.stream.session import StreamSession
    from openagent.stream.events import Interrupt, TextFinal, now_ms

    agent = _RecordingAgent(["hel", "lo ", "world"])
    sess = StreamSession(agent, client_id="c", session_id="s")
    # Skip STT/TTS resolution — keeps the test pure-unit (Piper local
    # TTS synth would otherwise burn ~1 s per sentence in finalize).
    await sess.start(stt_factory=_null_factory, tts_factory=_null_factory)

    # Push a user TextFinal to start a turn.
    await sess.push_in(TextFinal(
        session_id="s", seq=1, ts_ms=now_ms(),
        text="say hi", source="user_typed",
    ))
    # Wait for ≥2 deltas. Breaking on the first one races the dispatch
    # loop: ``_cancel_active_turn`` snapshots ``_partial_assistant`` the
    # instant the Interrupt is processed, and on a fast loop that's
    # before the agent task gets back from ``await asyncio.sleep(0.01)``
    # to yield the second delta — leaving ``partial == "hel"`` and the
    # ``"lo" in text`` assertion below false.
    for _ in range(60):
        await asyncio.sleep(0.05)
        if len(sess._partial_assistant) >= 2:
            break
    assert len(sess._partial_assistant) >= 2, sess._partial_assistant

    # Now interrupt.
    await sess.push_in(Interrupt(
        session_id="s", seq=2, ts_ms=now_ms(), reason="user_speech",
    ))
    # Give the dispatch loop a tick to process the Interrupt.
    await asyncio.sleep(0.2)

    assert agent.commits, "expected commit_partial_assistant to be called"
    sid, text = agent.commits[-1]
    assert sid == "s", sid
    assert "hel" in text and "lo" in text, f"partial mismatch: {text!r}"

    # Buffer should reset post-commit so the next turn starts clean.
    assert sess._partial_assistant == [], sess._partial_assistant

    await sess.close()


@test("barge_in", "empty partials skip commit")
async def t_empty_partial_skips_commit(_ctx: TestContext) -> None:
    from openagent.stream.session import StreamSession

    agent = _RecordingAgent([])
    sess = StreamSession(agent, client_id="c", session_id="s")
    # Don't even start a turn; just trigger cancel directly.
    await sess._cancel_active_turn(reason="manual")
    assert agent.commits == [], (
        f"no partial → no commit; got {agent.commits}"
    )


@test("barge_in", "commit failure does not break cancel path")
async def t_commit_failure_swallowed(_ctx: TestContext) -> None:
    from openagent.stream.session import StreamSession
    from openagent.stream.events import Interrupt, TextFinal, now_ms

    agent = _RecordingAgent(["hi", " there"], commit_raises=True)
    sess = StreamSession(agent, client_id="c", session_id="s")
    await sess.start(stt_factory=_null_factory, tts_factory=_null_factory)
    await sess.push_in(TextFinal(
        session_id="s", seq=1, ts_ms=now_ms(),
        text="hi", source="user_typed",
    ))
    for _ in range(60):
        await asyncio.sleep(0.05)
        if sess._partial_assistant:
            break

    # Should not raise even though commit raises internally.
    await sess.push_in(Interrupt(
        session_id="s", seq=2, ts_ms=now_ms(), reason="manual",
    ))
    await asyncio.sleep(0.2)
    assert agent.commits, "commit was attempted"
    # Session must still be usable — cancel completed cleanly.
    assert sess._current_turn is None
    await sess.close()


@test("barge_in", "Agent.commit_partial_assistant proxies to model")
async def t_agent_proxy(_ctx: TestContext) -> None:
    """``Agent.commit_partial_assistant`` is a thin proxy that forwards
    to ``self.model``. Verifies the routing without touching real
    providers."""
    from openagent.core.agent import Agent

    class _StubModel:
        history_mode = "platform"
        model = "fake"

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def commit_partial_assistant(self, sid: str, text: str) -> None:
            self.calls.append((sid, text))

        # Required surface for Agent's __init__ — these are no-ops.
        async def generate(self, *args, **kwargs):
            raise NotImplementedError

    model = _StubModel()
    agent = Agent.__new__(Agent)
    agent.model = model

    await agent.commit_partial_assistant("sid-1", "partial text")
    assert model.calls == [("sid-1", "partial text")], model.calls

    # Empty text → no call.
    model.calls = []
    await agent.commit_partial_assistant("sid-1", "")
    assert model.calls == [], model.calls

    # Empty session id → no call.
    await agent.commit_partial_assistant("", "x")
    assert model.calls == [], model.calls
