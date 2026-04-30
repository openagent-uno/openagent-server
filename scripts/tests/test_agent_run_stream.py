"""Agent.run_stream — empty-stream safety net.

Covers the contract that ``Agent.run_stream`` always produces text when
the underlying provider has it, even if ``provider.stream()`` yielded
zero deltas:

  * Provider yields zero deltas, ``generate()`` returns text →
    ``run_stream`` falls back, yields one ``delta`` + ``done`` carrying
    that text. Without this voice mode (and the soon-to-be-streaming
    web chat) silently surface "(No text response — the agent finished
    without producing any output…)" because the orchestrator's only
    safety net is the fallback message.
  * Provider yields deltas → no fallback call (regression guard so we
    don't double-spend tokens).
  * Both ``stream()`` and ``generate()`` fail-empty → clean exit, no
    crash, ``done`` carries empty text and a warning is logged.
  * On fallback, ``last_response_meta()`` reflects the real
    ``ModelResponse.model`` from ``generate()`` rather than the
    synthetic placeholder.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Awaitable, Callable

from ._framework import TestContext, test


# ── Helpers ─────────────────────────────────────────────────────────


class _FakeModel:
    """BaseModel-compatible stub with controllable stream + generate."""

    history_mode = "caller"

    def __init__(
        self,
        *,
        deltas: list[str] | None = None,
        generate_text: str = "",
        generate_raises: BaseException | None = None,
        model_name: str = "fake/test-model",
    ):
        # ``self.model`` is the attribute real providers set
        # (claude_cli, agno) and the new ``BaseModel.effective_model_id``
        # default reads. We mirror it on ``self.model_name`` only for
        # backwards-compatibility with older test code that referenced
        # the legacy attribute name.
        self.model = model_name
        self.model_name = model_name
        self._deltas = list(deltas or [])
        self._generate_text = generate_text
        self._generate_raises = generate_raises
        self.generate_calls = 0
        self.stream_calls = 0

    def effective_model_id(self, session_id: str | None = None) -> str | None:
        return self.model

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
    ) -> AsyncIterator[str]:
        self.stream_calls += 1
        for d in self._deltas:
            yield d

    async def generate(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        on_status: Callable[[str], Awaitable[None]] | None = None,
        session_id: str | None = None,
    ):
        from openagent.models.base import ModelResponse
        self.generate_calls += 1
        if self._generate_raises is not None:
            raise self._generate_raises
        return ModelResponse(content=self._generate_text, model=self.model_name)

    async def close_session(self, session_id: str) -> None:
        return None


async def _drive(agent, message: str, session_id: str = "sess-A") -> list[dict]:
    events: list[dict] = []
    async for ev in agent.run_stream(message=message, user_id="u", session_id=session_id):
        events.append(ev)
    return events


def _make_agent(model: _FakeModel):
    """Build a DB-less Agent so initialize() short-circuits."""
    from openagent.core.agent import Agent
    return Agent(name="test-agent", model=model, system_prompt="test", memory=None)


# ── Tests ───────────────────────────────────────────────────────────


@test("agent_run_stream", "stream yields no deltas → falls back to generate()")
async def t_fallback_when_stream_empty(_ctx: TestContext) -> None:
    model = _FakeModel(deltas=[], generate_text="Hello from generate")
    agent = _make_agent(model)

    events = await _drive(agent, "hi", session_id="sess-empty")

    deltas = [e for e in events if e.get("kind") == "delta"]
    done = [e for e in events if e.get("kind") == "done"]

    assert model.generate_calls == 1, (
        f"generate() was not invoked as a fallback (calls={model.generate_calls})"
    )
    assert len(deltas) == 1, f"expected one delta from fallback, got {deltas}"
    assert deltas[0]["text"] == "Hello from generate", deltas[0]
    assert len(done) == 1 and done[0]["text"] == "Hello from generate", done


@test("agent_run_stream", "stream yields deltas → no generate() fallback call")
async def t_no_fallback_when_stream_yields(_ctx: TestContext) -> None:
    model = _FakeModel(
        deltas=["Hello", " ", "world"],
        generate_text="SHOULD NOT BE USED",
    )
    agent = _make_agent(model)

    events = await _drive(agent, "hi", session_id="sess-deltas")

    deltas = [e for e in events if e.get("kind") == "delta"]
    done = [e for e in events if e.get("kind") == "done"]
    full_text = "".join(d["text"] for d in deltas)

    assert model.generate_calls == 0, (
        f"generate() should not be called when deltas exist (calls={model.generate_calls})"
    )
    assert full_text == "Hello world", f"deltas concatenated wrong: {full_text!r}"
    assert done and done[0]["text"] == "Hello world", done


@test("agent_run_stream", "stream empty + generate() raises → clean done with empty text")
async def t_fallback_generate_raises(_ctx: TestContext) -> None:
    model = _FakeModel(
        deltas=[],
        generate_raises=RuntimeError("simulated provider error"),
    )
    agent = _make_agent(model)

    events = await _drive(agent, "hi", session_id="sess-fail")

    deltas = [e for e in events if e.get("kind") == "delta"]
    done = [e for e in events if e.get("kind") == "done"]

    assert model.generate_calls == 1, (
        f"generate() must be attempted before giving up (calls={model.generate_calls})"
    )
    assert deltas == [], f"no deltas should be emitted when generate raised: {deltas}"
    assert done and done[0]["text"] == "", (
        f"done event must carry empty text when fallback failed: {done}"
    )


@test("agent_run_stream", "fallback response populates last_response_meta()")
async def t_fallback_meta_uses_real_response(_ctx: TestContext) -> None:
    model = _FakeModel(
        deltas=[],
        generate_text="Backup content",
        model_name="provider-x/model-y",
    )
    agent = _make_agent(model)

    await _drive(agent, "hi", session_id="sess-meta")
    meta = agent.last_response_meta("sess-meta")

    assert meta.get("model") == "provider-x/model-y", (
        f"meta should carry the real model name from the fallback ModelResponse: {meta}"
    )


@test("agent_run_stream", "streaming path stores model meta from effective_model_id")
async def t_streaming_meta_uses_effective_model_id(_ctx: TestContext) -> None:
    # Regression for the chat UI's missing model badge after the
    # streaming migration: the old code did
    # ``getattr(active_model, "model_name", None)`` which returned
    # ``None`` for every real provider in tree, so the chat bubble
    # never showed which model produced the reply. The fix routes
    # through ``BaseModel.effective_model_id`` (default reads
    # ``self.model``; SmartRouter overrides for per-session picks).
    model = _FakeModel(
        deltas=["streamed", " reply"],
        model_name="provider-y/streaming-model",
    )
    agent = _make_agent(model)

    await _drive(agent, "hi", session_id="sess-stream-meta")
    meta = agent.last_response_meta("sess-stream-meta")

    # No fallback should fire — deltas were yielded.
    assert model.generate_calls == 0, model.generate_calls
    assert meta.get("model") == "provider-y/streaming-model", (
        f"streaming-only path must surface the active model id: {meta}"
    )
