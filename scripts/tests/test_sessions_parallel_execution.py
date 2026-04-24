"""Regression: two sessions on the same client must run in parallel.

Before the fix, ``SessionManager`` had a single pending queue and a
single worker task per CLIENT. Two chat tabs in the app share one
websocket (hence one client_id), so a slow turn in tab A would block
tab B's turn entirely — even though the two conversations are fully
context-isolated at the model level (per-session locks in ClaudeCLI
and per-session keys in AgnoProvider's SqliteDb). The symptom: reply
times looked serialized in the UI.

The fix introduces per-session ``_SessionState`` with its own queue
and worker task. Different sessions run their workers concurrently
under the asyncio scheduler; ordering is preserved *within* a session
(FIFO per session) but no longer *across* sessions.

Wall-clock timing is the clearest signal here: two 0.5 s handlers on
two sessions should finish in ~0.5 s total (parallel), not ~1.0 s
(serialized). Generous headroom keeps the test robust to CI jitter.
"""
from __future__ import annotations

import asyncio
import time

from ._framework import TestContext, test


@test("sessions_parallel", "two sessions on one client run concurrently")
async def t_parallel_sessions(ctx: TestContext) -> None:
    from openagent.gateway.sessions import SessionManager

    sm = SessionManager(agent_name="test-agent")
    client = "ws:client-x"
    a = sm.get_or_create_session(client, "ws:client-x:sid-a")
    b = sm.get_or_create_session(client, "ws:client-x:sid-b")

    done_at: dict[str, float] = {}

    async def sleep_then_mark(sid: str) -> None:
        await asyncio.sleep(0.5)
        done_at[sid] = time.monotonic()

    start = time.monotonic()
    # Two handlers on two different sessions — must dispatch to
    # separate per-session workers and run concurrently.
    await sm.enqueue(client, lambda: sleep_then_mark(a), session_id=a)
    await sm.enqueue(client, lambda: sleep_then_mark(b), session_id=b)

    # Poll until both complete (or timeout).
    deadline = time.monotonic() + 5.0
    while len(done_at) < 2 and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    total = max(done_at.values()) - start if done_at else float("inf")

    try:
        assert len(done_at) == 2, f"only {len(done_at)} sessions completed"
        # Parallel: ≈0.5s. Serialized (old behaviour): ≥1.0s. Use 0.8s
        # as the threshold — crosses the parallel/serial boundary with
        # room for scheduler noise.
        assert total < 0.8, (
            f"sessions serialized instead of parallel; total={total:.3f}s"
        )
    finally:
        await sm.shutdown()


@test("sessions_parallel", "ordering preserved WITHIN a single session (FIFO)")
async def t_fifo_within_session(ctx: TestContext) -> None:
    """Parallelism across sessions must not break ordering inside
    one session — consecutive messages from the same user still
    need to be handled in order."""
    from openagent.gateway.sessions import SessionManager

    sm = SessionManager(agent_name="test-agent")
    client = "ws:single"
    sid = sm.get_or_create_session(client, "ws:single:only")

    order: list[int] = []

    async def record(i: int) -> None:
        # Small sleep so the scheduler could reorder if serialization
        # weren't enforced; order must stay FIFO regardless.
        await asyncio.sleep(0.02)
        order.append(i)

    for i in range(5):
        await sm.enqueue(client, lambda _i=i: record(_i), session_id=sid)

    # Wait for all to complete.
    deadline = time.monotonic() + 3.0
    while len(order) < 5 and time.monotonic() < deadline:
        await asyncio.sleep(0.01)

    try:
        assert order == [0, 1, 2, 3, 4], f"FIFO broken within session: {order}"
    finally:
        await sm.shutdown()


@test("sessions_parallel", "queue cap applies per-session, not per-client")
async def t_queue_cap_per_session(ctx: TestContext) -> None:
    """MAX_QUEUE_SIZE limits each session independently — a noisy
    session shouldn't cause the next session's first message to be
    rejected."""
    from openagent.gateway.sessions import MAX_QUEUE_SIZE, SessionManager

    sm = SessionManager(agent_name="test-agent")
    client = "ws:cap-test"
    a = sm.get_or_create_session(client, "ws:cap-test:a")
    b = sm.get_or_create_session(client, "ws:cap-test:b")

    # Block A's worker with one long-running handler so subsequent
    # enqueues pile up in A's pending queue.
    block = asyncio.Event()

    async def hold() -> None:
        await block.wait()

    # First enqueue starts running; fill A's queue to cap afterward.
    await sm.enqueue(client, lambda: hold(), session_id=a)
    for _ in range(MAX_QUEUE_SIZE):
        await sm.enqueue(client, lambda: asyncio.sleep(0), session_id=a)
    # The next one on A hits the cap and is rejected.
    rejected = await sm.enqueue(client, lambda: asyncio.sleep(0), session_id=a)
    assert rejected == -1, rejected

    # B is independent — its first enqueue must succeed even while A's
    # queue is full.
    accepted_b = await sm.enqueue(client, lambda: asyncio.sleep(0), session_id=b)
    assert accepted_b >= 0, accepted_b

    block.set()
    await sm.shutdown()
