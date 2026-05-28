"""ClaudeBackedAgent — Agent-subclass that drives claude_agent_sdk.

These tests pin the four guarantees that make the v0.14.x sub-agent
refactor work:

  1. Construction does not crash even on a slim install (the
     ``claude_agent_sdk`` import is lazy — inside the agent's
     methods, not at module load), and the constructed instance
     IS an ``agno.agent.Agent`` so Team's ``isinstance(member, Agent)``
     check passes.
  2. ``arun(stream=False)`` drives the SDK and assembles a
     ``RunOutput`` with ``status=completed``, content, tools, and
     metrics — the shape Agno's Team consumes from a member run.
  3. The SDK session id round-trips through ``session.session_data``
     so restart-resume keeps the same SDK transcript across boots.
  4. ``arun(stream=True)`` is an async iterator that emits
     ``RunContentEvent`` per text delta and (when
     ``yield_run_output=True``) a terminal ``RunOutput`` — the exact
     contract Team's streaming dispatcher expects.

We monkey-patch ``sys.modules['claude_agent_sdk']`` with a small fake
module so the tests don't need the real SDK binary installed and don't
spawn subprocesses.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from ._framework import TestContext, test


# ── Fake claude_agent_sdk module ──────────────────────────────────────


@dataclass
class _FakeTextBlock:
    text: str


@dataclass
class _FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeToolResultBlock:
    tool_use_id: str
    content: str = ""


@dataclass
class _FakeAssistantMessage:
    content: list[Any] = field(default_factory=list)


@dataclass
class _FakeUserMessage:
    content: Any = None


@dataclass
class _FakeSystemMessage:
    subtype: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeResultMessage:
    is_error: bool = False
    subtype: str = "success"
    result: Optional[str] = None
    session_id: Optional[str] = None
    usage: Optional[dict[str, int]] = None
    errors: Any = None
    stop_reason: Optional[str] = None


@dataclass
class _FakeStreamEvent:
    event: dict[str, Any] = field(default_factory=dict)


class _FakeClaudeAgentOptions:
    """Records what the ClaudeBackedAgent passed to the SDK."""

    # Class-level slot so the per-test state instance can read it back
    # across the async-generator boundary (the agent constructs an
    # options object and feeds it to ``query``; the test inspects the
    # last-built kwargs via the closure-captured state).
    _last_kwargs: Optional[dict[str, Any]] = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        type(self)._last_kwargs = dict(kwargs)


@dataclass
class _FakeSdkState:
    """Per-test-controllable script for ``fake.query``.

    Tests rewrite ``script`` to the sequence of messages the SDK
    should yield, then drive ``ClaudeBackedAgent.arun`` and assert on
    the resulting ``RunOutput`` / event stream.
    """

    script: list[Any] = field(default_factory=list)
    last_prompt: Optional[str] = None
    call_count: int = 0

    @property
    def last_options_kwargs(self) -> Optional[dict[str, Any]]:
        return _FakeClaudeAgentOptions._last_kwargs


def _install_fake_sdk() -> _FakeSdkState:
    """Install (or reset) the fake ``claude_agent_sdk`` module.

    Returns the per-test state object; mutate ``state.script`` BEFORE
    calling the agent. Subsequent ``query()`` calls iterate the
    script and yield each item.
    """
    _FakeClaudeAgentOptions._last_kwargs = None
    state = _FakeSdkState()

    async def _fake_query(*, prompt: str, options: Any) -> AsyncIterator[Any]:
        state.call_count += 1
        state.last_prompt = prompt
        for item in state.script:
            yield item

    fake_module = types.ModuleType("claude_agent_sdk")
    fake_module.query = _fake_query  # type: ignore[attr-defined]
    fake_module.ClaudeAgentOptions = _FakeClaudeAgentOptions  # type: ignore[attr-defined]
    fake_module.TextBlock = _FakeTextBlock  # type: ignore[attr-defined]
    fake_module.ToolUseBlock = _FakeToolUseBlock  # type: ignore[attr-defined]
    fake_module.ToolResultBlock = _FakeToolResultBlock  # type: ignore[attr-defined]
    fake_module.AssistantMessage = _FakeAssistantMessage  # type: ignore[attr-defined]
    fake_module.UserMessage = _FakeUserMessage  # type: ignore[attr-defined]
    fake_module.SystemMessage = _FakeSystemMessage  # type: ignore[attr-defined]
    fake_module.ResultMessage = _FakeResultMessage  # type: ignore[attr-defined]
    fake_module.StreamEvent = _FakeStreamEvent  # type: ignore[attr-defined]
    sys.modules["claude_agent_sdk"] = fake_module
    return state


# ── Tests ─────────────────────────────────────────────────────────────


@test("claude_backed_agent", "construction with placeholder model")
async def t_construction(ctx: TestContext) -> None:
    """Importing the module and constructing the agent must not require
    ``claude_agent_sdk`` to be installed — the SDK import is lazy,
    deferred until ``arun`` actually runs. Constructed instance must
    be an ``Agent`` (so Team's isinstance(Agent) check passes) and
    must have the Agent attribute defaults Team's delegation reads
    (``knowledge_filters``, ``num_history_runs``, ``store_media``).
    """
    # Make sure no leftover fake module from a previous test leaks
    # the import shape. We don't install the fake here — the module
    # MUST import without it.
    sys.modules.pop("claude_agent_sdk", None)

    from src.core._runner.agent import Agent
    from src.models.claude_agent import ClaudeBackedAgent

    agent = ClaudeBackedAgent(
        name="test",
        sdk_model_id="claude-sonnet-4-6",
        sdk_options={"system": "You are helpful", "max_turns": 5},
    )
    assert isinstance(agent, Agent), (
        "ClaudeBackedAgent must subclass Agent so Team accepts it as a member"
    )
    # Agent attribute defaults Team reads during delegation must be
    # present — these would AttributeError on a BaseExternalAgent.
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
    assert agent._sdk_model_id == "claude-sonnet-4-6"
    assert agent._sdk_options["max_turns"] == 5


@test("claude_backed_agent", "arun builds RunOutput with completed status")
async def t_arun_collect(ctx: TestContext) -> None:
    """``arun(stream=False)`` must drive the SDK to completion and
    return a ``RunOutput`` with ``status=completed``, content from the
    ResultMessage (or accumulated assistant text), tool calls captured
    from AssistantMessage / UserMessage(ToolResultBlock), and metrics
    surfaced from the SDK's usage report.
    """
    state = _install_fake_sdk()
    from src.core._run_state.agent import RunOutput
    from src.core._run_state.base import RunStatus
    from src.models.claude_agent import ClaudeBackedAgent

    state.script = [
        _FakeSystemMessage(subtype="init", data={"session_id": "sdk-sess-001"}),
        _FakeAssistantMessage(content=[_FakeTextBlock(text="Hello, ")]),
        _FakeAssistantMessage(content=[
            _FakeTextBlock(text="world."),
            _FakeToolUseBlock(id="t1", name="vault_save", input={"key": "x"}),
        ]),
        _FakeUserMessage(content=[
            _FakeToolResultBlock(tool_use_id="t1", content="saved"),
        ]),
        _FakeResultMessage(
            subtype="success", result="Hello, world.",
            session_id="sdk-sess-001",
            usage={"input_tokens": 42, "output_tokens": 17},
        ),
    ]

    agent = ClaudeBackedAgent(
        name="test", sdk_model_id="claude-sonnet-4-6",
        sdk_options={"system": "be brief"},
    )

    result = await agent.arun("hi", session_id="sess-1", user_id="alice")

    assert isinstance(result, RunOutput), type(result).__name__
    assert result.status == RunStatus.completed
    assert result.content == "Hello, world."
    assert result.session_id == "sess-1"
    assert result.user_id == "alice"
    assert result.model == "claude-sonnet-4-6"
    assert result.model_provider == "claude-cli"
    assert result.run_id  # auto-generated UUID

    # Tool capture: one tool_use block → one ToolExecution with result
    # carried forward from the ToolResultBlock.
    assert result.tools is not None and len(result.tools) == 1, result.tools
    te = result.tools[0]
    assert te.tool_call_id == "t1"
    assert te.tool_name == "vault_save"
    assert te.tool_args == {"key": "x"}
    assert te.result == "saved"

    # Metrics: token usage surfaced from the SDK's ResultMessage.
    assert result.metrics is not None
    assert result.metrics.input_tokens == 42
    assert result.metrics.output_tokens == 17
    assert result.metrics.total_tokens == 59

    # SDK session id captured from ResultMessage.session_id.
    assert agent._sdk_session_ids["sess-1"] == "sdk-sess-001"

    # The fake recorded the options we built: system prompt and model.
    assert state.last_options_kwargs is not None
    assert state.last_options_kwargs.get("system_prompt") == "be brief"
    assert state.last_options_kwargs.get("model") == "claude-sonnet-4-6"
    # No resume hint on first turn.
    assert "resume" not in state.last_options_kwargs


@test("claude_backed_agent", "session id round-trips via session_data on DB")
async def t_session_persistence(ctx: TestContext) -> None:
    """The SDK session id must round-trip through
    ``session.session_data`` so restart-resume keeps the same SDK
    transcript. We use a minimal in-memory DB stub that mirrors
    Agno's ``SqliteDb.get_session`` / ``upsert_session`` surface.
    """
    state = _install_fake_sdk()
    from src.models.claude_agent import ClaudeBackedAgent

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
    # Pre-seed: the previous boot stamped this id in session_data.
    db.sessions["sess-resume"].session_data = {"sdk_session_id": "prev-sdk-id"}

    agent = ClaudeBackedAgent(
        name="resume-test", sdk_model_id="claude-sonnet-4-6",
        sdk_options={}, db=db,
    )

    # Turn 1: hydrate finds the previous SDK id, the SDK returns a
    # NEW session id mid-run (e.g. compaction); we must persist that.
    state.script = [
        _FakeSystemMessage(subtype="init", data={"session_id": "new-sdk-id"}),
        _FakeAssistantMessage(content=[_FakeTextBlock(text="ok")]),
        _FakeResultMessage(
            subtype="success", result="ok", session_id="new-sdk-id",
            usage={"input_tokens": 1, "output_tokens": 1},
        ),
    ]

    await agent.arun("hi", session_id="sess-resume")

    # Resume hint must have been sent on this call (we knew the prev id).
    assert state.last_options_kwargs.get("resume") == "prev-sdk-id"
    # The new id replaced the persisted one on session_data.
    assert db.sessions["sess-resume"].session_data["sdk_session_id"] == "new-sdk-id"

    # Turn 2: a fresh agent instance (simulates a restart) must pick
    # up the persisted SDK id and feed it back to the SDK as ``resume``.
    fresh_agent = ClaudeBackedAgent(
        name="resume-test", sdk_model_id="claude-sonnet-4-6",
        sdk_options={}, db=db,
    )
    state.script = [
        _FakeAssistantMessage(content=[_FakeTextBlock(text="yo")]),
        _FakeResultMessage(
            subtype="success", result="yo", session_id="new-sdk-id",
            usage={"input_tokens": 1, "output_tokens": 1},
        ),
    ]
    # Reset the recorded options so the assert below pins THIS call.
    _FakeClaudeAgentOptions._last_kwargs = None

    await fresh_agent.arun("hello", session_id="sess-resume")

    assert state.last_options_kwargs.get("resume") == "new-sdk-id", (
        "fresh agent must restore the persisted sdk_session_id from session_data"
    )


@test("claude_backed_agent", "_arun_stream yields RunContentEvent per delta + terminal RunOutput")
async def t_stream_events(ctx: TestContext) -> None:
    """``arun(stream=True, yield_run_output=True)`` must yield a
    ``RunContentEvent`` per token-level delta (or AssistantMessage
    TextBlock when StreamEvents aren't flowing) and a single terminal
    ``RunOutput`` at the end — the contract Agno Team's streaming
    dispatcher consumes from a member.
    """
    state = _install_fake_sdk()
    from src.core._run_state.agent import RunContentEvent, RunOutput, ToolCallStartedEvent
    from src.core._run_state.base import RunStatus
    from src.models.claude_agent import ClaudeBackedAgent

    # Mix of token-level StreamEvents (text deltas) and a tool call
    # surfaced via the complete AssistantMessage. Then the SDK closes
    # with a ResultMessage.
    state.script = [
        _FakeSystemMessage(subtype="init", data={"session_id": "stream-sdk-1"}),
        _FakeStreamEvent(event={"type": "content_block_delta",
                                "delta": {"type": "text_delta", "text": "Hel"}}),
        _FakeStreamEvent(event={"type": "content_block_delta",
                                "delta": {"type": "text_delta", "text": "lo"}}),
        _FakeAssistantMessage(content=[
            _FakeToolUseBlock(id="t-stream", name="echo", input={"x": 1}),
        ]),
        _FakeStreamEvent(event={"type": "content_block_delta",
                                "delta": {"type": "text_delta", "text": "!"}}),
        _FakeResultMessage(
            subtype="success", result="Hello!", session_id="stream-sdk-1",
            usage={"input_tokens": 3, "output_tokens": 4},
        ),
    ]

    agent = ClaudeBackedAgent(
        name="s", sdk_model_id="claude-sonnet-4-6", sdk_options={},
    )

    events: list[Any] = []
    async for ev in agent.arun(
        "hi", session_id="sess-s", stream=True, yield_run_output=True,
    ):
        events.append(ev)

    # Filter into the three categories we care about.
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

    # Stream mode opts into token-level streaming on the SDK.
    assert state.last_options_kwargs.get("include_partial_messages") is True


@test("claude_backed_agent", "arun raises on SDK ResultMessage error")
async def t_arun_raises_on_error(ctx: TestContext) -> None:
    """Invalid model id, exceeded budget, etc., surface as a
    ``ResultMessage`` with ``is_error=True`` or a non-success subtype.
    The agent must raise rather than silently emit empty content.
    """
    state = _install_fake_sdk()
    from src.models.claude_agent import ClaudeBackedAgent

    state.script = [
        _FakeSystemMessage(subtype="init", data={"session_id": "fail-1"}),
        _FakeResultMessage(
            is_error=True, subtype="error_max_turns",
            result="exceeded max turns", session_id="fail-1",
            stop_reason="max_turns",
        ),
    ]

    agent = ClaudeBackedAgent(
        name="e", sdk_model_id="claude-sonnet-4-6", sdk_options={},
    )

    raised = False
    try:
        await agent.arun("hi", session_id="sess-e")
    except RuntimeError as e:
        raised = True
        assert "error_max_turns" in str(e) or "exceeded max turns" in str(e), str(e)
    assert raised, "expected RuntimeError on SDK error result"


@test("claude_backed_agent", "lazy import — module imports without claude_agent_sdk")
async def t_lazy_import(ctx: TestContext) -> None:
    """The ``from claude_agent_sdk import …`` MUST be deferred until
    ``arun`` runs. Importing the module on a slim install (no
    claude_agent_sdk) must not crash. We verify by pop-ing the
    module and forcing a fresh import.

    Order-wise this is the LAST test in the module so subsequent test
    modules see the real ``claude_agent_sdk`` (or a missing one, if
    the venv is slim) — never our fake. We don't reinstall the fake
    at the end; tests downstream that depend on the real SDK
    (test_shell adapter, etc.) read the real module from sys.modules.
    """
    # Drop the fake (or real) module + the cached claude_agent
    # module so reimport reruns the top-level code path.
    sys.modules.pop("claude_agent_sdk", None)
    sys.modules.pop("src.models.claude_agent", None)

    # The import must succeed without claude_agent_sdk available.
    import importlib
    mod = importlib.import_module("src.models.claude_agent")

    # Construction must also succeed.
    agent = mod.ClaudeBackedAgent(
        name="lazy", sdk_model_id="claude-sonnet-4-6", sdk_options={},
    )
    assert agent.name == "lazy"

    # Restore the real ``claude_agent_sdk`` if it's importable so
    # downstream tests (test_shell adapter etc.) see the real module
    # instead of trying to import-time-resolve a missing one. If the
    # venv is slim (no real SDK), leave the slot empty so those
    # downstream tests get the same ImportError they'd see on any
    # slim install — that's an honest pre-existing limitation.
    try:
        import claude_agent_sdk  # noqa: F401 — populates sys.modules
    except ImportError:
        pass
