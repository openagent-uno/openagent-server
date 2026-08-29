"""A manual trigger must not kill the firing it started.

``POST /api/scheduled-tasks/{id}/run`` waited with ``asyncio.wait_for``, which
CANCELS the awaited task when the deadline passes. A real production firing of
clickup-task-quality-audit was killed that way after 22 minutes of completed
work: the run recorded "Stopped by user" and the work was thrown away, purely
because the HTTP client had been waiting longer than the default 300s. The
firing belongs to the scheduler and has to outlive the request.
"""
from __future__ import annotations

import asyncio

from ._framework import TestContext, test


@test("manual_run", "asyncio.wait leaves a slow firing running; wait_for would kill it")
async def t_slow_firing_survives(ctx: TestContext) -> None:
    """Pins the primitive the handler now uses, and the one it must not."""
    async def _firing():
        await asyncio.sleep(0.6)
        return "done"

    # The old primitive: the task is cancelled out from under the scheduler.
    victim = asyncio.ensure_future(_firing())
    try:
        await asyncio.wait_for(victim, timeout=0.05)
    except asyncio.TimeoutError:
        pass
    await asyncio.sleep(0.05)
    assert victim.cancelled(), "wait_for used to leave the firing alive?"

    # The new one: the deadline passes, the firing keeps going and completes.
    survivor = asyncio.ensure_future(_firing())
    done, pending = await asyncio.wait({survivor}, timeout=0.05)
    assert not done and pending, "the firing should still be pending here"
    assert not survivor.cancelled(), "asyncio.wait must not cancel the firing"
    assert await survivor == "done", "the firing must still complete on its own"


@test("manual_run", "the handler no longer reaches for the cancelling primitive")
async def t_handler_uses_wait(ctx: TestContext) -> None:
    """A source check, because the regression is invisible at runtime until it
    eats a long run: the endpoint answered 504 either way."""
    from pathlib import Path
    import src.gateway.api.scheduled_tasks as mod

    src = Path(mod.__file__).read_text()
    handler = src[src.index("async def handle_run(request)"):]
    handler = handler[:handler.index("\nasync def ")]
    assert "wait_for(run_task" not in handler, (
        "handle_run is cancelling the firing again")
    assert "asyncio.wait({run_task}" in handler, handler[:200]
    # And it must hand back something the caller can poll, not a bare error.
    assert "run_id" in handler and "202" in handler
