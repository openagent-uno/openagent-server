"""Regression tests for claude_cli.py's AssistantMessage text recovery path.

Observed in production: `ResultMessage.result` arrives empty even when the
Claude CLI streamed an AssistantMessage with substantial TextBlock content
(lyra-agent events.jsonl showed `model.empty_result` with `output_tokens=1311`).
The provider used to return "(Done — no final message was returned.)" and
drop all the generated text on the floor.

The fix: accumulate text from every TextBlock we see in AssistantMessages,
then prefer `ResultMessage.result` when non-empty, otherwise fall back to the
accumulated stream.

These are UNIT tests with a fake SDK — no binary required, no tokens burned.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ._framework import TestContext, test


@dataclass
class _FakeTextBlock:
    text: str


@dataclass
class _FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class _FakeAssistantMessage:
    content: list[Any] = field(default_factory=list)


@dataclass
class _FakeResultMessage:
    result: str | None = None
    session_id: str | None = "sdk-sess-1"
    total_cost_usd: float | None = 0.01
    usage: dict[str, Any] | None = None
    model_usage: dict[str, Any] | None = None
    duration_ms: int | None = 100
    duration_api_ms: int | None = 80
    num_turns: int | None = 1


class _FakeSDKClient:
    """Minimal stand-in for ``ClaudeSDKClient`` exposing only what ``_run_once`` uses.

    Waits 1.1 s before the first yield so the stale-response guard in
    ``_run_once`` (which fires on `<1s elapsed` responses) doesn't trip.
    """

    def __init__(self, messages: list[Any], startup_delay: float = 1.1) -> None:
        self._messages = messages
        self._startup_delay = startup_delay

    async def query(self, *_args, **_kwargs) -> None:
        return None

    def receive_response(self):
        messages = self._messages
        delay = self._startup_delay

        async def _gen():
            if delay:
                await asyncio.sleep(delay)
            for msg in messages:
                yield msg

        return _gen()


def _install_fake_sdk_types() -> None:
    """Make ``isinstance(msg, AssistantMessage)`` and ``ResultMessage`` match our fakes.

    `_run_once` imports ``AssistantMessage`` and ``ResultMessage`` lazily from
    ``claude_agent_sdk``, so we monkey-patch the module to point at our fakes.
    That avoids pulling in the real types (and their dataclass constructors
    with required fields).
    """
    import claude_agent_sdk

    claude_agent_sdk.AssistantMessage = _FakeAssistantMessage
    claude_agent_sdk.ResultMessage = _FakeResultMessage


async def _run_once_with(messages: list[Any], *, session_id: str = "test-sess") -> tuple[str, dict]:
    _install_fake_sdk_types()
    from openagent.models.claude_cli import ClaudeCLI

    cli = ClaudeCLI(model=None, providers_config={"anthropic": {"models": ["claude-cli"]}})
    client = _FakeSDKClient(messages)
    # `_drain_stale` reads client._stale_queue; stub it so the drain is a no-op.
    cli._drain_stale = lambda _c: asyncio.sleep(0)  # type: ignore[assignment]
    return await cli._run_once(client, "hi", session_id, on_status=None)


@test("claude_cli_text_recovery", "ResultMessage.result wins when non-empty")
async def t_result_preferred(ctx: TestContext) -> None:
    messages = [
        _FakeAssistantMessage(content=[_FakeTextBlock(text="streamed preview")]),
        _FakeResultMessage(result="final answer"),
    ]
    text, _ = await _run_once_with(messages)
    assert text == "final answer", f"expected 'final answer', got {text!r}"


@test("claude_cli_text_recovery", "AssistantMessage text recovered when ResultMessage.result is empty")
async def t_recover_from_assistant(ctx: TestContext) -> None:
    # This is the production bug: `result` is empty despite streamed text.
    messages = [
        _FakeAssistantMessage(
            content=[_FakeTextBlock(text="Ciao Yoanna! Ecco il report: ")]
        ),
        _FakeAssistantMessage(
            content=[_FakeToolUseBlock(id="t1", name="search_notes", input={"q": "x"})]
        ),
        _FakeAssistantMessage(
            content=[_FakeTextBlock(text="Ordini di oggi: 3.")]
        ),
        _FakeResultMessage(result=""),  # ← the regression
    ]
    text, _ = await _run_once_with(messages)
    assert "Ecco il report" in text, f"recovery failed: {text!r}"
    assert "Ordini di oggi: 3." in text, f"second chunk lost: {text!r}"
    assert "(Done — no final message" not in text, f"fell through to placeholder: {text!r}"


@test("claude_cli_text_recovery", "placeholder still fires when no text anywhere")
async def t_placeholder_when_truly_empty(ctx: TestContext) -> None:
    messages = [
        _FakeAssistantMessage(
            content=[_FakeToolUseBlock(id="t1", name="search_notes", input={})]
        ),
        _FakeResultMessage(result=None),
    ]
    text, _ = await _run_once_with(messages)
    assert text == "(Done — no final message was returned.)", f"got {text!r}"


@test("claude_cli_text_recovery", "AssistantMessage with only None text blocks is ignored")
async def t_none_text_ignored(ctx: TestContext) -> None:
    # Defensive: a block that happens to expose ``text=None`` shouldn't
    # contribute "None" to the recovered string.
    messages = [
        _FakeAssistantMessage(content=[_FakeTextBlock(text=None)]),  # type: ignore[arg-type]
        _FakeResultMessage(result=""),
    ]
    text, _ = await _run_once_with(messages)
    assert text == "(Done — no final message was returned.)", f"got {text!r}"


# ── generate() retry semantics ────────────────────────────────────────


class _RecordingCLI:
    """Wraps ClaudeCLI.generate with a pluggable ``_run_once`` for retry tests."""

    def __init__(self, outcomes):
        _install_fake_sdk_types()
        from openagent.models.claude_cli import ClaudeCLI

        self.cli = ClaudeCLI(model=None, providers_config={"anthropic": {"models": ["claude-cli"]}})
        self.cli._drain_stale = lambda _c: asyncio.sleep(0)  # type: ignore[assignment]
        self.cli._get_client = self._fake_get_client  # type: ignore[assignment]
        self.cli._drop_client = self._fake_drop_client  # type: ignore[assignment]
        self.cli._record_usage = self._fake_record_usage  # type: ignore[assignment]
        self._outcomes = list(outcomes)
        self._calls = 0
        self.dropped = 0

        async def run_once(*_a, **_kw):
            self._calls += 1
            outcome = self._outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        self.cli._run_once = run_once  # type: ignore[assignment]

    async def _fake_get_client(self, *_a, **_kw):
        return object()

    async def _fake_drop_client(self, *_a, **_kw):
        self.dropped += 1

    async def _fake_record_usage(self, *_a, **_kw):
        return 0, 0, 0.0

    @property
    def attempts(self) -> int:
        return self._calls


@test("claude_cli_text_recovery", "generate does NOT retry on TimeoutError")
async def t_no_retry_on_timeout(ctx: TestContext) -> None:
    harness = _RecordingCLI([TimeoutError("idle timeout")])
    resp = await harness.cli.generate(
        [{"role": "user", "content": "ciao"}], session_id="t1"
    )
    assert harness.attempts == 1, f"expected 1 call, got {harness.attempts}"
    assert harness.dropped == 1, f"expected client dropped once, got {harness.dropped}"
    assert "took too long" in resp.content.lower() or "timeout" in resp.content.lower(), resp.content


@test("claude_cli_text_recovery", "generate retries once on non-timeout Exception")
async def t_retry_on_generic_error(ctx: TestContext) -> None:
    harness = _RecordingCLI([RuntimeError("transient network blip"), ("recovered", {})])
    resp = await harness.cli.generate(
        [{"role": "user", "content": "ciao"}], session_id="t2"
    )
    assert harness.attempts == 2, f"expected 2 calls, got {harness.attempts}"
    assert resp.content == "recovered", resp.content


@test("claude_cli_text_recovery", "non-timeout error exhausts retries and reports error")
async def t_retry_exhausted_on_error(ctx: TestContext) -> None:
    harness = _RecordingCLI(
        [RuntimeError("first fail"), RuntimeError("second fail")]
    )
    resp = await harness.cli.generate(
        [{"role": "user", "content": "ciao"}], session_id="t3"
    )
    assert harness.attempts == 2, f"expected 2 calls, got {harness.attempts}"
    assert resp.content.startswith("Error:"), resp.content
