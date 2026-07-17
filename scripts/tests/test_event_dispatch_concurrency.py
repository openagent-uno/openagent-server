"""Bounded event-delivery dispatch: a burst must never jam the pipeline.

The scheduler's event drain (``Scheduler._drain_event_deliveries``, run every
~2 s off ``_cancellation_loop``) claims ``received`` deliveries and dispatches
each bound agent turn as a DETACHED background task. An event turn can be very
heavy (a ~250k-token support-thread turn), so the OLD, unbounded drain — claim a
whole burst, fire every turn at once — could saturate the runtime and HANG the
entire pipeline. In production, ~66 deliveries re-enqueued at once (a manual
backfill) spawned ~66 concurrent heavy turns and NOTHING completed for ~19 min,
including brand-new inbound tickets.

The fix caps in-flight event turns at ``OPENAGENT_EVENT_DISPATCH_CONCURRENCY``
(default 4). The DB's ``received`` rows already ARE the queue; the drain now
claims + dispatches at most ``(concurrency - in_flight)`` deliveries per tick,
so at most ``concurrency`` heavy turns ever run at once and the rest wait in the
DB for a slot. The claim ``limit`` equals the free-slot count, so a delivery is
never claimed unless it is dispatched in the same breath (no claimed-but-
undispatched orphan).

These tests pin the load-bearing guarantees with an INSTRUMENTED, gated fake
dispatch (no real turns, fully deterministic):

  1. the bound holds under a burst — with concurrency=K and N>K claimable
     deliveries, at most K are ever in flight, the rest stay ``received``
     (unclaimed), and as turns complete the queued ones drain until all N
     process — none dropped, none stuck;
  2. a hanging turn holds its slot but does NOT block the drain loop — repeated
     ticks run promptly and dispatch nothing while full, and freeing one slot
     lets exactly one more dispatch (never exceeding the cap).
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid

from ._framework import TestContext, test


def _fresh_db_path(ctx: TestContext):
    return ctx.db_path.with_name(f"evdispatch-{uuid.uuid4().hex[:8]}.db")


async def _make_db(path):
    from src.memory.db import MemoryDB
    db = MemoryDB(str(path))
    await db.connect()
    return db


async def _add_event(db, *, slug: str) -> str:
    from src.core.event_secret import make_secret_material
    _clear, enc, hint = make_secret_material(db_path=db.db_path)
    return await db.add_event(
        name=f"evt-{slug}", action_kind="prompt", slug=slug,
        secret_enc=enc, secret_hint=hint,
        prompt_template="Handle {{payload.ticket.id}}",
        session_binding_enabled=False,
    )


async def _all_rows(db) -> list[dict]:
    conn = await db._ensure_connected()
    cur = await conn.execute(
        "SELECT id, status, claimed_at FROM event_deliveries",
    )
    return [dict(r) for r in await cur.fetchall()]


async def _received_unclaimed(db) -> list[dict]:
    rows = await _all_rows(db)
    return [r for r in rows if r["status"] == "received" and r["claimed_at"] is None]


async def _wait_until(pred, *, timeout: float = 5.0) -> None:
    """Poll ``pred`` until true or the deadline — lets spawned dispatch tasks
    advance to their gate without a fixed-length sleep."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met within timeout")


class _DummyAgent:
    """The drain passes ``agent=self.agent`` straight through to the (faked)
    dispatcher, which ignores it — so a bare stub is enough."""
    name = "dummy"
    model = None


class _GatedDispatch:
    """Instrumented stand-in for ``event_dispatcher.dispatch_event``.

    Mirrors the real lifecycle just enough to be faithful: it flips the
    delivery to ``running`` as its very first act (exactly like the real
    dispatcher, so a claimed row is observably running), then BLOCKS on a
    per-delivery gate the test controls, and on release marks the row
    ``success``. That lets the test hold turns in flight, observe the ceiling,
    then drain them on command. It records the max concurrency ever seen so an
    over-dispatch is caught even if it were only momentary."""

    def __init__(self):
        self.active: set[str] = set()
        self.max_active: int = 0
        self.started: list[str] = []
        self.completed: list[str] = []
        self._gates: dict[str, asyncio.Event] = {}

    def _gate(self, delivery_id: str) -> asyncio.Event:
        ev = self._gates.get(delivery_id)
        if ev is None:
            ev = asyncio.Event()
            self._gates[delivery_id] = ev
        return ev

    def release(self, *delivery_ids: str) -> None:
        for did in delivery_ids:
            self._gate(did).set()

    def release_all(self) -> None:
        for ev in self._gates.values():
            ev.set()

    async def dispatch(self, *, agent, db, scheduler, event, payload,
                       delivery_id, source="agent", broadcast=None):
        # First act mirrors the real dispatch_event: flip to ``running``.
        await db.update_event_delivery(delivery_id, status="running")
        self.active.add(delivery_id)
        self.started.append(delivery_id)
        self.max_active = max(self.max_active, len(self.active))
        try:
            await self._gate(delivery_id).wait()
            await db.update_event_delivery(
                delivery_id, status="success", finished_at=time.time(),
            )
            self.completed.append(delivery_id)
            return {"status": "success"}
        finally:
            self.active.discard(delivery_id)


# ── 1. The bound holds under a burst, and the whole queue eventually drains ──


@test("event_dispatch_concurrency",
      "concurrency=K + burst of N>K: at most K in flight, rest stay received, all N drain")
async def t_bound_holds_and_queue_drains(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler
    import src.core.event_dispatcher as ed

    K = 2
    N = 7  # a burst well over the cap (the production incident was ~66)

    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    harness = _GatedDispatch()
    orig = ed.dispatch_event
    os.environ["OPENAGENT_EVENT_DISPATCH_CONCURRENCY"] = str(K)
    ed.dispatch_event = harness.dispatch
    try:
        eid = await _add_event(db, slug="burst")
        dids = [
            await db.add_event_delivery(
                event_id=eid, payload={"ticket": {"id": f"T-{i}"}},
                claimed=False,  # out-of-process shape: the drain must claim it
            )
            for i in range(N)
        ]

        # Baseline: every delivery is queued (received, unclaimed).
        assert len(await _received_unclaimed(db)) == N

        scheduler = Scheduler(db=db, agent=_DummyAgent())  # type: ignore[arg-type]

        processed = 0
        guard = 0
        while processed < N:
            guard += 1
            assert guard < 50, "drain made no progress"
            expected = min(K, N - processed)

            # One drain tick claims + dispatches at most K.
            await scheduler._drain_event_deliveries()
            await _wait_until(lambda e=expected: len(harness.active) == e)

            # The bound: never more than K in flight, ever — and the
            # scheduler's own counter agrees.
            assert harness.max_active <= K, harness.max_active
            assert scheduler._event_dispatch_in_flight == expected

            # Everything beyond the cap stays received AND unclaimed — no
            # claimed-but-undispatched orphan.
            leftover = await _received_unclaimed(db)
            assert len(leftover) == N - processed - expected, (
                len(leftover), N, processed, expected)

            # A second tick while full must dispatch nothing new (free == 0),
            # and returns immediately rather than blocking.
            await asyncio.wait_for(scheduler._drain_event_deliveries(), timeout=1.0)
            await asyncio.sleep(0)  # give any (wrongly) spawned task a chance
            assert len(harness.active) == expected
            assert harness.max_active <= K

            # Complete this wave; slots free and the next tick dispatches more.
            harness.release(*list(harness.active))
            await _wait_until(lambda: scheduler._event_dispatch_in_flight == 0)
            processed += expected

        # Nothing dropped, nothing stuck: every delivery reached success exactly
        # once, and the cap was respected the whole way through.
        rows = await _all_rows(db)
        assert len(rows) == N
        assert all(r["status"] == "success" for r in rows), [r["status"] for r in rows]
        assert sorted(harness.completed) == sorted(dids)
        assert harness.max_active <= K, harness.max_active
    finally:
        ed.dispatch_event = orig
        os.environ.pop("OPENAGENT_EVENT_DISPATCH_CONCURRENCY", None)
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 2. A hanging turn holds its slot but never blocks the drain loop ─────────


@test("event_dispatch_concurrency",
      "a hanging turn holds its slot but does not block the drain; freeing one dispatches exactly one more")
async def t_hanging_turn_does_not_block_drain(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler
    import src.core.event_dispatcher as ed

    K = 2
    extra = 3
    total = K + extra

    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    harness = _GatedDispatch()
    orig = ed.dispatch_event
    os.environ["OPENAGENT_EVENT_DISPATCH_CONCURRENCY"] = str(K)
    ed.dispatch_event = harness.dispatch
    try:
        eid = await _add_event(db, slug="hang")
        for i in range(total):
            await db.add_event_delivery(
                event_id=eid, payload={"ticket": {"id": f"H-{i}"}}, claimed=False,
            )

        scheduler = Scheduler(db=db, agent=_DummyAgent())  # type: ignore[arg-type]

        # Tick 1: fill all K slots with turns we never release (they "hang").
        await scheduler._drain_event_deliveries()
        await _wait_until(lambda: len(harness.active) == K)
        assert scheduler._event_dispatch_in_flight == K

        # Repeated ticks while every slot is stuck: each must return promptly
        # (the hang does NOT block the loop) and dispatch nothing new.
        for _ in range(4):
            await asyncio.wait_for(scheduler._drain_event_deliveries(), timeout=1.0)
            assert scheduler._event_dispatch_in_flight == K
            assert len(harness.active) == K
            assert harness.max_active == K

        # The queued deliveries wait in the DB, received and unclaimed.
        assert len(await _received_unclaimed(db)) == total - K

        # Free ONE slot → the next tick dispatches exactly one more, still
        # never exceeding the cap.
        stuck = next(iter(harness.active))
        harness.release(stuck)
        await _wait_until(lambda: scheduler._event_dispatch_in_flight == K - 1)
        await scheduler._drain_event_deliveries()
        # The slot is reserved synchronously at spawn, so in_flight is already
        # back to K here; wait on the fake actually STARTING so ``active``
        # reflects the newly-dispatched turn before we assert on it.
        await _wait_until(lambda: len(harness.active) == K)
        assert scheduler._event_dispatch_in_flight == K
        assert harness.max_active == K, harness.max_active
        assert len(await _received_unclaimed(db)) == total - K - 1
    finally:
        harness.release_all()
        try:
            await _wait_until(
                lambda: scheduler._event_dispatch_in_flight == 0, timeout=2.0,
            )
        except Exception:  # noqa: BLE001
            pass
        ed.dispatch_event = orig
        os.environ.pop("OPENAGENT_EVENT_DISPATCH_CONCURRENCY", None)
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
