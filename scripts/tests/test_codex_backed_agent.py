"""CodexBackedAgent — Agent-subclass that drives openai_codex.

Mirrors ``test_claude_backed_agent.py`` for the Codex CLI side. Pins
the six guarantees that make codex-cli a viable third framework:

  1. Construction does not crash on a slim install (the
     ``openai_codex`` import is lazy — inside the agent's methods,
     not at module load), and the constructed instance IS an
     ``agno.agent.Agent`` so Team's ``isinstance(member, Agent)``
     check passes.
  2. ``arun(stream=False)`` drives the SDK and assembles a
     ``RunOutput`` with ``status=completed``, content, tools, and
     metrics.
  3. The Codex thread id round-trips through ``session.session_data``
     under the key ``codex_session_id`` so restart-resume keeps the
     same Codex thread across boots.
  4. ``arun(stream=True)`` is an async iterator that emits
     ``RunContentEvent`` per text delta and (when
     ``yield_run_output=True``) a terminal ``RunOutput``.
  5. SDK errors surfaced via the Codex SDK's TurnResult.status raise
     ``RuntimeError`` rather than silently emitting empty content.
  6. Lazy import — module loads without ``openai_codex`` installed.

We monkey-patch ``sys.modules['openai_codex']`` with a small fake
module so the tests don't need the real SDK installed.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from ._framework import TestContext, test


# ── Fake openai_codex module ──────────────────────────────────────────


@dataclass
class _FakeNotification:
    """Mirror of openai_codex's JSON-RPC notification shape."""
    method: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeTurnStatus:
    """Mirror of TurnStatus enum-ish object — uses .name for matching."""
    name: str


@dataclass
class _FakeTurnError:
    message: str = ""
    detail: str = ""


@dataclass
class _FakeTurnResult:
    """TurnResult shape returned from thread.run()."""
    status: _FakeTurnStatus
    final_response: Optional[str] = None
    items: list[Any] = field(default_factory=list)
    usage: Optional[dict[str, int]] = None
    error: Optional[_FakeTurnError] = None


@dataclass
class _FakeThreadItem:
    """ThreadItem shape — mirrors the SDK's tool-call item shape."""
    type: str
    id: str = ""
    name: str = ""
    input: Optional[dict[str, Any]] = None
    result: Optional[str] = None


@dataclass
class _FakeAsyncTurnHandle:
    """Returned by thread.turn() — its .stream() yields notifications."""
    notifications: list[_FakeNotification] = field(default_factory=list)

    async def stream(self) -> AsyncIterator[_FakeNotification]:
        for n in self.notifications:
            yield n


class _FakeAsyncThread:
    """Returned by codex.thread_start / thread_resume."""

    def __init__(self, thread_id: str, state: "_FakeSdkState") -> None:
        self.id = thread_id
        self._state = state

    async def run(self, prompt: Any) -> _FakeTurnResult:
        self._state.last_prompt = prompt
        self._state.run_call_count += 1
        return self._state.next_run_result()

    async def turn(self, prompt: Any) -> _FakeAsyncTurnHandle:
        self._state.last_prompt = prompt
        self._state.turn_call_count += 1
        return _FakeAsyncTurnHandle(notifications=list(self._state.next_stream_script()))


class _FakeAsyncCodex:
    """async with context manager mirroring openai_codex.AsyncCodex."""

    def __init__(self, state: "_FakeSdkState") -> None:
        self._state = state

    async def __aenter__(self) -> "_FakeAsyncCodex":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def thread_start(self, **kwargs: Any) -> _FakeAsyncThread:
        self._state.last_thread_kwargs = dict(kwargs)
        self._state.start_count += 1
        thread_id = self._state.next_thread_id()
        return _FakeAsyncThread(thread_id, self._state)

    async def thread_resume(self, thread_id: str, **kwargs: Any) -> _FakeAsyncThread:
        self._state.last_thread_kwargs = dict(kwargs)
        self._state.last_resume_id = thread_id
        self._state.resume_count += 1
        return _FakeAsyncThread(thread_id, self._state)


@dataclass
class _FakeSdkState:
    """Per-test-controllable script for ``fake.AsyncCodex``.

    Tests stage one or more ``run_results`` and / or ``stream_scripts``
    BEFORE calling the agent. Each ``arun`` call pops the next entry.
    """

    run_results: list[_FakeTurnResult] = field(default_factory=list)
    stream_scripts: list[list[_FakeNotification]] = field(default_factory=list)
    thread_ids: list[str] = field(default_factory=lambda: ["thread-1"])

    last_prompt: Any = None
    last_thread_kwargs: Optional[dict[str, Any]] = None
    last_resume_id: Optional[str] = None
    run_call_count: int = 0
    turn_call_count: int = 0
    start_count: int = 0
    resume_count: int = 0

    def next_thread_id(self) -> str:
        if len(self.thread_ids) == 1:
            return self.thread_ids[0]
        return self.thread_ids.pop(0) if self.thread_ids else "thread-x"

    def next_run_result(self) -> _FakeTurnResult:
        if not self.run_results:
            return _FakeTurnResult(
                status=_FakeTurnStatus(name="completed"),
                final_response="",
                items=[],
                usage={"input_tokens": 0, "output_tokens": 0},
            )
        if len(self.run_results) == 1:
            return self.run_results[0]
        return self.run_results.pop(0)

    def next_stream_script(self) -> list[_FakeNotification]:
        if not self.stream_scripts:
            return []
        if len(self.stream_scripts) == 1:
            return self.stream_scripts[0]
        return self.stream_scripts.pop(0)


def _install_fake_sdk() -> _FakeSdkState:
    """Install (or reset) the fake ``openai_codex`` module."""
    state = _FakeSdkState()

    def _async_codex_factory(*args: Any, **kwargs: Any) -> _FakeAsyncCodex:
        return _FakeAsyncCodex(state)

    fake_module = types.ModuleType("openai_codex")
    fake_module.AsyncCodex = _async_codex_factory  # type: ignore[attr-defined]
    sys.modules["openai_codex"] = fake_module
    return state


# ── Tests ─────────────────────────────────────────────────────────────


@test("codex_backed_agent", "construction with placeholder model")
async def t_construction(ctx: TestContext) -> None:
    """Construction must not require ``openai_codex`` to be installed
    (lazy import inside arun). The constructed instance must be an
    ``Agent`` so Team accepts it as a member, and must carry the
    attribute defaults Team's delegation reads.
    """
    sys.modules.pop("openai_codex", None)

    from agno.agent import Agent
    from src.models.codex_agent import CodexBackedAgent

    agent = CodexBackedAgent(
        name="test",
        sdk_model_id="gpt-5",
        sdk_options={"system": "You are helpful", "approval_mode": "auto"},
    )
    assert isinstance(agent, Agent), (
        "CodexBackedAgent must subclass Agent so Team accepts it as a member"
    )
    for attr, expected_type in [
        ("knowledge_filters", type(None)),
        ("num_history_runs", int),
        ("store_media", bool),
        ("store_tool_messages", bool),
        ("store_history_messages", bool),
        ("send_media_to_model", bool),
    ]:
        value = getattr(agent, attr, "MISSING")
        assert value != "MISSING", f"agent.{attr} missing — Team will crash"
        if expected_type is not type(None):
            assert isinstance(value, expected_type), (
                f"agent.{attr} should be {expected_type.__name__}; got {type(value).__name__}"
            )

    assert agent.name == "test"
    assert agent._sdk_model_id == "gpt-5"
    assert agent._sdk_options["approval_mode"] == "auto"


@test("codex_backed_agent", "arun builds RunOutput with completed status")
async def t_arun_collect(ctx: TestContext) -> None:
    """``arun(stream=False)`` must drive the SDK to completion and
    return a ``RunOutput`` with ``status=completed``, content from the
    TurnResult.final_response, tool calls captured from items, and
    metrics surfaced from the SDK's usage report.
    """
    state = _install_fake_sdk()
    from agno.run.agent import RunOutput
    from agno.run.base import RunStatus
    from src.models.codex_agent import CodexBackedAgent

    state.thread_ids = ["thread-abc"]
    state.run_results = [
        _FakeTurnResult(
            status=_FakeTurnStatus(name="completed"),
            final_response="Hello, world.",
            items=[
                _FakeThreadItem(
                    type="toolCall",
                    id="t1",
                    name="shell",
                    input={"cmd": "echo hi"},
                    result="hi",
                ),
            ],
            usage={"input_tokens": 42, "output_tokens": 17},
        ),
    ]

    agent = CodexBackedAgent(
        name="test", sdk_model_id="gpt-5",
        sdk_options={"system": "be brief"},
    )

    result = await agent.arun("hi", session_id="sess-1", user_id="alice")

    assert isinstance(result, RunOutput), type(result).__name__
    assert result.status == RunStatus.completed
    assert result.content == "Hello, world."
    assert result.session_id == "sess-1"
    assert result.user_id == "alice"
    assert result.model == "gpt-5"
    assert result.model_provider == "codex-cli"
    assert result.run_id  # auto-generated UUID

    # Tool capture from items.
    assert result.tools is not None and len(result.tools) == 1, result.tools
    te = result.tools[0]
    assert te.tool_call_id == "t1"
    assert te.tool_name == "shell"
    assert te.tool_args == {"cmd": "echo hi"}
    assert te.result == "hi"

    # Metrics surfaced.
    assert result.metrics is not None
    assert result.metrics.input_tokens == 42
    assert result.metrics.output_tokens == 17
    assert result.metrics.total_tokens == 59

    # Thread id captured from thread_start.
    assert agent._sdk_session_ids["sess-1"] == "thread-abc"

    # The fake recorded the kwargs we built: system → developer_instructions,
    # plus model.
    assert state.last_thread_kwargs is not None
    assert state.last_thread_kwargs.get("developer_instructions") == "be brief"
    assert state.last_thread_kwargs.get("model") == "gpt-5"
    # First turn started a new thread — no resume.
    assert state.start_count == 1
    assert state.resume_count == 0


@test("codex_backed_agent", "session id round-trips via session_data on DB")
async def t_session_persistence(ctx: TestContext) -> None:
    """The Codex thread id must round-trip through
    ``session.session_data["codex_session_id"]`` so restart-resume
    keeps the same thread across boots.
    """
    state = _install_fake_sdk()
    from src.models.codex_agent import CodexBackedAgent

    class _FakeSession:
        def __init__(self) -> None:
            self.session_data: dict[str, Any] | None = None

    class _FakeDb:
        def __init__(self) -> None:
            self.sessions: dict[str, _FakeSession] = {}

        def get_session(self, *, session_id: str, **kwargs: Any) -> _FakeSession | None:
            return self.sessions.get(session_id)

        def upsert_session(self, *, session: _FakeSession) -> None:
            # session is mutated in-place by _save_to_db; no-op here.
            pass

    db = _FakeDb()
    db.sessions["sess-resume"] = _FakeSession()
    # Pre-seed: the previous boot stamped this thread id in session_data.
    db.sessions["sess-resume"].session_data = {"codex_session_id": "prev-thread-id"}

    agent = CodexBackedAgent(
        name="resume-test", sdk_model_id="gpt-5",
        sdk_options={}, db=db,
    )

    # Turn 1: hydrate finds the previous thread id; the SDK keeps that
    # id (thread_resume returns the same id) and we persist it.
    state.thread_ids = ["prev-thread-id"]
    state.run_results = [
        _FakeTurnResult(
            status=_FakeTurnStatus(name="completed"),
            final_response="ok",
            items=[],
            usage={"input_tokens": 1, "output_tokens": 1},
        ),
    ]

    await agent.arun("hi", session_id="sess-resume")

    # Resume path taken (not thread_start).
    assert state.resume_count == 1, state.resume_count
    assert state.start_count == 0, state.start_count
    assert state.last_resume_id == "prev-thread-id"
    # session_data still carries the same thread id.
    assert (
        db.sessions["sess-resume"].session_data["codex_session_id"]
        == "prev-thread-id"
    )

    # Turn 2: a FRESH agent (simulates a restart) must pick up the
    # persisted thread id and resume the same SDK thread.
    state2 = _install_fake_sdk()
    state2.thread_ids = ["prev-thread-id"]
    state2.run_results = [
        _FakeTurnResult(
            status=_FakeTurnStatus(name="completed"),
            final_response="yo",
            items=[],
            usage={"input_tokens": 1, "output_tokens": 1},
        ),
    ]

    fresh_agent = CodexBackedAgent(
        name="resume-test", sdk_model_id="gpt-5",
        sdk_options={}, db=db,
    )
    await fresh_agent.arun("hello", session_id="sess-resume")

    assert state2.last_resume_id == "prev-thread-id", (
        "fresh agent must restore the persisted codex_session_id from session_data"
    )


@test(
    "codex_backed_agent",
    "_arun_stream yields RunContentEvent per delta + terminal RunOutput",
)
async def t_stream_events(ctx: TestContext) -> None:
    """``arun(stream=True, yield_run_output=True)`` must yield a
    ``RunContentEvent`` per token-level delta and a single terminal
    ``RunOutput`` at the end — the contract Agno Team's streaming
    dispatcher consumes from a member.
    """
    state = _install_fake_sdk()
    from agno.run.agent import RunContentEvent, RunOutput, ToolCallStartedEvent
    from agno.run.base import RunStatus
    from src.models.codex_agent import CodexBackedAgent

    state.thread_ids = ["stream-thread-1"]
    # Codex SDK notification methods: ``item/agentMessage/delta`` for
    # text deltas, ``item/toolCall/started`` for tool-use, and
    # ``turn/completed`` to close the stream with final response + usage.
    state.stream_scripts = [[
        _FakeNotification(
            method="item/agentMessage/delta",
            params={"delta": "Hel"},
        ),
        _FakeNotification(
            method="item/agentMessage/delta",
            params={"delta": "lo"},
        ),
        _FakeNotification(
            method="item/toolCall/started",
            params={"item": {"id": "t-stream", "name": "echo", "input": {"x": 1}}},
        ),
        _FakeNotification(
            method="item/agentMessage/delta",
            params={"delta": "!"},
        ),
        _FakeNotification(
            method="turn/completed",
            params={
                "turn": {
                    "final_response": "Hello!",
                    "usage": {"input_tokens": 3, "output_tokens": 4},
                }
            },
        ),
    ]]

    agent = CodexBackedAgent(
        name="s", sdk_model_id="gpt-5", sdk_options={},
    )

    events: list[Any] = []
    async for ev in agent.arun(
        "hi", session_id="sess-s", stream=True, yield_run_output=True,
    ):
        events.append(ev)

    deltas = [e for e in events if isinstance(e, RunContentEvent)]
    tools_started = [e for e in events if isinstance(e, ToolCallStartedEvent)]
    final_outputs = [e for e in events if isinstance(e, RunOutput)]

    assert [e.content for e in deltas] == ["Hel", "lo", "!"], [e.content for e in deltas]
    assert len(tools_started) == 1
    assert tools_started[0].tool.tool_name == "echo"
    assert len(final_outputs) == 1, len(final_outputs)
    final = final_outputs[0]
    assert final.status == RunStatus.completed
    assert final.content == "Hello!"
    assert final.metrics.input_tokens == 3
    assert final.metrics.output_tokens == 4


@test("codex_backed_agent", "arun raises on SDK TurnResult error status")
async def t_arun_raises_on_error(ctx: TestContext) -> None:
    """Invalid model id, exceeded budget, etc., surface as a
    ``TurnResult`` with a non-success ``status``. The agent must raise
    rather than silently emit empty content.
    """
    state = _install_fake_sdk()
    from src.models.codex_agent import CodexBackedAgent

    state.thread_ids = ["fail-thread"]
    state.run_results = [
        _FakeTurnResult(
            status=_FakeTurnStatus(name="failed"),
            final_response=None,
            items=[],
            usage={"input_tokens": 0, "output_tokens": 0},
            error=_FakeTurnError(message="exceeded turn budget"),
        ),
    ]

    agent = CodexBackedAgent(
        name="e", sdk_model_id="gpt-5", sdk_options={},
    )

    raised = False
    try:
        await agent.arun("hi", session_id="sess-e")
    except RuntimeError as e:
        raised = True
        assert "exceeded turn budget" in str(e) or "failed" in str(e), str(e)
    assert raised, "expected RuntimeError on SDK error result"


@test("codex_backed_agent", "lazy import — module imports without openai_codex")
async def t_lazy_import(ctx: TestContext) -> None:
    """The ``import openai_codex`` MUST be deferred until ``arun``
    actually runs. Importing the module on a slim install (no
    ``openai_codex``) must not crash. We verify by popping the module
    and forcing a fresh import.

    Order-wise this is the LAST test in the module so subsequent test
    modules see the real ``openai_codex`` (or a missing one, if the
    venv is slim) — never our fake.
    """
    sys.modules.pop("openai_codex", None)
    sys.modules.pop("src.models.codex_agent", None)

    import importlib
    mod = importlib.import_module("src.models.codex_agent")

    agent = mod.CodexBackedAgent(
        name="lazy", sdk_model_id="gpt-5", sdk_options={},
    )
    assert agent.name == "lazy"

    # Restore the real ``openai_codex`` if it's importable so
    # downstream tests see the real module (or none).
    try:
        import openai_codex  # noqa: F401
    except ImportError:
        pass
