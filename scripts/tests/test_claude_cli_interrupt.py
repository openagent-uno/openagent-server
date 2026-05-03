"""ClaudeCLI.commit_partial_assistant — barge-in via SDK ``interrupt()``.

Covers the dispatch contract:

* Live session → ``client.interrupt()`` is awaited (the SDK's
  ``ClaudeSDKClient.interrupt()`` sends a control-request to the
  subprocess so it closes the turn cleanly with whatever was emitted).
* Missing session → silent no-op (no exception).
* ``client.interrupt()`` raising → caught and logged, never propagates.
* Registry fan-out → forwards to the per-session ClaudeCLI instance.
"""
from __future__ import annotations

import asyncio

from ._framework import TestContext, test


class _RecordingClient:
    """SDK client stub. Records ``interrupt()`` calls."""

    def __init__(self, raise_on_interrupt: bool = False):
        self.calls = 0
        self._raise = raise_on_interrupt

    async def interrupt(self) -> None:
        self.calls += 1
        if self._raise:
            raise RuntimeError("simulated SDK failure")


@test("claude_interrupt", "live session forwards to client.interrupt()")
async def t_live_session_forwards(_ctx: TestContext) -> None:
    from openagent.models.claude_cli import ClaudeCLI, _Session

    cli = ClaudeCLI(model="claude-haiku-4-5-20251001")
    client = _RecordingClient()
    sess = _Session(session_id="sid", client=client)
    cli._sessions["sid"] = sess

    await cli.commit_partial_assistant("sid", "hello world")
    assert client.calls == 1, f"interrupt should fire once; got {client.calls}"


@test("claude_interrupt", "missing session is silent no-op")
async def t_missing_session_noop(_ctx: TestContext) -> None:
    from openagent.models.claude_cli import ClaudeCLI

    cli = ClaudeCLI(model="claude-haiku-4-5-20251001")
    # No registered _Session for this id.
    await cli.commit_partial_assistant("never-seen", "partial")
    # No raise, no state change.


@test("claude_interrupt", "client.interrupt() failure is swallowed")
async def t_failure_swallowed(_ctx: TestContext) -> None:
    from openagent.models.claude_cli import ClaudeCLI, _Session

    cli = ClaudeCLI(model="claude-haiku-4-5-20251001")
    client = _RecordingClient(raise_on_interrupt=True)
    sess = _Session(session_id="sid", client=client)
    cli._sessions["sid"] = sess

    # Must not propagate.
    await cli.commit_partial_assistant("sid", "partial")
    assert client.calls == 1, "interrupt was attempted before raising"


@test("claude_interrupt", "registry forwards to per-session ClaudeCLI")
async def t_registry_fanout(_ctx: TestContext) -> None:
    from openagent.models.claude_cli import (
        ClaudeCLI, ClaudeCLIRegistry, _Session,
    )

    registry = ClaudeCLIRegistry()
    inst = ClaudeCLI(model="claude-haiku-4-5-20251001")
    client = _RecordingClient()
    inst._sessions["sid"] = _Session(session_id="sid", client=client)
    registry._instances["sid"] = inst

    await registry.commit_partial_assistant("sid", "partial")
    assert client.calls == 1, f"registry must forward; got {client.calls}"

    # Unknown session — silent no-op via registry.
    await registry.commit_partial_assistant("unknown", "x")
    assert client.calls == 1, "no extra interrupts for unknown sid"


@test("claude_interrupt", "session without live client is silent no-op")
async def t_no_live_client(_ctx: TestContext) -> None:
    from openagent.models.claude_cli import ClaudeCLI, _Session

    cli = ClaudeCLI(model="claude-haiku-4-5-20251001")
    # Session exists in cache but client is None (idle-closed state).
    cli._sessions["sid"] = _Session(session_id="sid", client=None)
    # Must not raise.
    await cli.commit_partial_assistant("sid", "partial")
