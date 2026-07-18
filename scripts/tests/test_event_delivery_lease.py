"""Claim-lease + heartbeat: recover a FROZEN event-delivery turn in ~LEASE_TTL.

The existing recovery paths are coarse: the startup reap only fires on a restart,
and the periodic stale sweep uses a 30-min ``claimed_at`` age proxy. Neither
recovers the confirmed incident quickly — the runtime session store hogs the
single SQLite WAL writer during a rate-limit storm, MemoryDB's finalizer loses
the race, and the delivery sits ``running`` for up to half an hour.

A real lease fixes that. Every claim stamps ``claim_expires`` = now + LEASE_TTL
(short, ~120 s) plus ``worker_id``/``worker_pid``; while the turn runs its
dispatch runner HEARTBEATS the lease forward with a tiny single-row write that
survives writer contention. If the turn/process FREEZES the heartbeat stops, the
lease lapses, and ``reap_expired_event_leases`` (on the fast ~2 s loop)
re-enqueues the delivery — recovery ≈ LEASE_TTL, not 30 min.

The load-bearing SAFETY property: the lease reaper only ever touches rows with
``claim_expires IS NOT NULL``. Every in-flight row that predates the deploy has a
NULL lease (the column was just added, only a NEW claim stamps it), so the reaper
never reclaims a legacy row — those stay handled by the age-gated stale sweep.

These tests are fully LLM-free / network-free: a fresh ``MemoryDB`` per test with
columns poked directly, mirroring ``test_event_delivery_stale_sweep``.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import time
import uuid

from ._framework import TestContext, test


def _fresh_db_path(ctx: TestContext):
    return ctx.db_path.with_name(f"evlease-{uuid.uuid4().hex[:8]}.db")


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
    )


async def _set_claim_expires(db, delivery_id: str, ts) -> None:
    """Force a delivery's ``claim_expires`` to an explicit epoch time (or NULL).

    The lease is stamped by the claim path, never by ``update_event_delivery``, so
    a test simulating "lease lapsed / not yet lapsed / legacy NULL" pokes it."""
    conn = await db._ensure_connected()
    await conn.execute(
        "UPDATE event_deliveries SET claim_expires = ? WHERE id = ?",
        (ts, delivery_id),
    )
    await conn.commit()


# ── 1. A lapsed lease is reclaimed; a live (future) lease is left running ──


@test("event_delivery_lease",
      "an expired claim lease is re-enqueued; a future lease is left running")
async def t_expired_lease_reenqueued(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        eid = await _add_event(db, slug="lease-basic")

        # (a) a frozen turn: claimed, running, lease lapsed 5 s ago.
        frozen = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "T-1"}}, claimed=True,
        )
        await db.update_event_delivery(frozen, status="running")
        # Stamp confirms the claim carried a lease + owner.
        row = await db.get_event_delivery(frozen)
        assert row["claim_expires"] is not None, "a claim must stamp a lease"
        assert row["worker_id"] == db.worker_id, row["worker_id"]
        assert row["worker_pid"] == db.worker_pid, row["worker_pid"]
        await _set_claim_expires(db, frozen, time.time() - 5)

        # (b) a healthy turn: claimed, running, lease still in the future.
        live = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "T-2"}}, claimed=True,
        )
        await db.update_event_delivery(live, status="running")
        await _set_claim_expires(db, live, time.time() + 300)

        n = await db.reap_expired_event_leases()
        assert n == 1, f"exactly the expired-lease row should be reclaimed, got {n}"

        fr = await db.get_event_delivery(frozen)
        assert fr["status"] == "received", fr["status"]
        assert fr["claimed_at"] is None, "re-enqueued row drops its claim"
        assert fr["claim_expires"] is None, "re-enqueued row clears its lease"
        assert fr["worker_id"] is None and fr["worker_pid"] is None, fr
        assert fr["reenqueue_count"] == 1, fr["reenqueue_count"]
        assert "lease-reap" in (fr.get("error") or ""), fr.get("error")

        lv = await db.get_event_delivery(live)
        assert lv["status"] == "running", "a live lease must be left running"
        assert lv["claimed_at"] is not None

        # The drain would now claim exactly the recovered row.
        claimed = await db.claim_pending_event_deliveries()
        assert [c["id"] for c in claimed] == [frozen], claimed
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 2. A heartbeat extends the lease → the reaper leaves it alone ──────────


@test("event_delivery_lease",
      "a heartbeat extends the lease so the reaper does not reclaim a live turn")
async def t_heartbeat_extends_lease(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        eid = await _add_event(db, slug="lease-hb")
        did = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "H-1"}}, claimed=True,
        )
        await db.update_event_delivery(did, status="running")
        # The lease has lapsed — WITHOUT a heartbeat this row would be reclaimed.
        await _set_claim_expires(db, did, time.time() - 1)

        # The dispatch runner's heartbeat pushes the lease back into the future.
        matched = await db.heartbeat_event_delivery(did, db.worker_id)
        assert matched == 1, f"heartbeat should match the owned row, got {matched}"
        row = await db.get_event_delivery(did)
        assert row["claim_expires"] > time.time(), "heartbeat must extend the lease"
        assert row["last_heartbeat_at"] is not None

        n = await db.reap_expired_event_leases()
        assert n == 0, f"a heartbeated (live) lease must be left alone, acted on {n}"
        assert (await db.get_event_delivery(did))["status"] == "running"

        # A heartbeat from a DIFFERENT worker (a row already reclaimed and
        # re-dispatched elsewhere) matches nothing — it must not resurrect a
        # lease it no longer owns.
        matched2 = await db.heartbeat_event_delivery(did, "some-other-worker")
        assert matched2 == 0, f"a foreign heartbeat must match nothing, got {matched2}"
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 3. A legacy (NULL-lease) in-flight row is NEVER reclaimed by the reaper ─


@test("event_delivery_lease",
      "a pre-deploy row with a NULL lease is untouched by the lease reaper (deploy safety)")
async def t_legacy_null_lease_untouched(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        eid = await _add_event(db, slug="lease-legacy")
        # Simulate a row claimed by the PREVIOUS build: claimed + running but no
        # lease was ever stamped (the column is new). The lease reaper must never
        # touch it — the age-gated stale sweep / startup reap own those rows.
        did = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "L-1"}}, claimed=True,
        )
        await db.update_event_delivery(did, status="running")
        await _set_claim_expires(db, did, None)  # NULL lease, like a legacy row

        n = await db.reap_expired_event_leases()
        assert n == 0, f"a NULL-lease row must be untouched, acted on {n}"
        row = await db.get_event_delivery(did)
        assert row["status"] == "running", row["status"]
        assert row["claimed_at"] is not None, "legacy claim must be preserved"
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 4. The kill-switch and the replay budget both still apply ──────────────


@test("event_delivery_lease",
      "OPENAGENT_EVENT_REENQUEUE_ENABLED=0 makes the lease reaper a no-op")
async def t_kill_switch_noop(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        eid = await _add_event(db, slug="lease-kill")
        did = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "K-1"}}, claimed=True,
        )
        await db.update_event_delivery(did, status="running")
        await _set_claim_expires(db, did, time.time() - 10)

        n = await db.reap_expired_event_leases(enabled=False)
        assert n == 0, f"kill-switch must no-op, acted on {n}"
        assert (await db.get_event_delivery(did))["status"] == "running"

        os.environ["OPENAGENT_EVENT_REENQUEUE_ENABLED"] = "0"
        try:
            n2 = await db.reap_expired_event_leases()
            assert n2 == 0, f"env kill-switch must no-op, acted on {n2}"
            assert (await db.get_event_delivery(did))["status"] == "running"
        finally:
            os.environ.pop("OPENAGENT_EVENT_REENQUEUE_ENABLED", None)
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


@test("event_delivery_lease",
      "a lease that keeps expiring is parked terminal at the replay budget")
async def t_budget_caps(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        eid = await _add_event(db, slug="lease-budget")
        did = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "B-1"}}, claimed=True,
        )
        await db.update_event_delivery(did, status="running")

        requeues = 0
        for _ in range(5):
            row = await db.get_event_delivery(did)
            if row["claimed_at"] is None:
                await db.claim_pending_event_deliveries()
            await db.update_event_delivery(did, status="running")
            await _set_claim_expires(db, did, time.time() - 10)
            await db.reap_expired_event_leases(max_attempts=2)
            row = await db.get_event_delivery(did)
            if row["status"] == "received":
                requeues += 1
            elif row["status"] == "failed":
                break

        row = await db.get_event_delivery(did)
        assert requeues == 2, f"expected exactly max_attempts=2 re-enqueues, got {requeues}"
        assert row["status"] == "failed", row["status"]
        assert "lease-reap: retry budget exhausted" in (row.get("error") or ""), row.get("error")
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 5. Lock-surviving writes: the bounded retry lands the write ────────────


@test("event_delivery_lease",
      "the bounded retry retries a locked write, propagates a non-lock error, and gives up cleanly")
async def t_write_retry_semantics(ctx: TestContext) -> None:
    """Unit-level guarantees of ``_write_with_retry``: a transient 'database is
    locked' is retried and lands; a non-lock OperationalError propagates at once;
    a persistent lock exhausts the budget and re-raises (no silent swallow, no
    infinite loop)."""
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        # (a) retried then lands.
        calls = {"n": 0}

        async def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        assert await db._write_with_retry(flaky) == "ok"
        assert calls["n"] == 3, calls["n"]

        # (b) a non-lock OperationalError is NOT retried — it propagates.
        nlock = {"n": 0}

        async def other():
            nlock["n"] += 1
            raise sqlite3.OperationalError("no such column: nope")

        raised = False
        try:
            await db._write_with_retry(other)
        except sqlite3.OperationalError as e:
            raised = "no such column" in str(e)
        assert raised, "a non-lock error must propagate immediately"
        assert nlock["n"] == 1, f"a non-lock error must not retry, got {nlock['n']}"

        # (c) a persistent lock exhausts the budget and re-raises.
        always = {"n": 0}

        async def stuck():
            always["n"] += 1
            raise sqlite3.OperationalError("database is locked")

        exhausted = False
        try:
            await db._write_with_retry(stuck, attempts=3)
        except sqlite3.OperationalError:
            exhausted = True
        assert exhausted, "an unrecoverable lock must re-raise, not hang"
        assert always["n"] == 3, f"must try exactly `attempts` times, got {always['n']}"
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


@test("event_delivery_lease",
      "a real second-connection EXCLUSIVE lock: the finalizer write lands after release, loop not wedged")
async def t_locked_writer_lands_after_release(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        eid = await _add_event(db, slug="lease-locked")
        did = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "W-1"}}, claimed=True,
        )
        # Shrink this connection's busy_timeout so a contended write raises
        # "database is locked" quickly and exercises the RETRY path (rather than
        # blocking inside SQLite for the full 10 s).
        conn = await db._ensure_connected()
        await conn.execute("PRAGMA busy_timeout = 30")
        await conn.commit()

        # A second raw connection grabs the write lock (the session-store's role
        # in the incident).
        raw = sqlite3.connect(str(path), timeout=5.0)
        raw.execute("BEGIN EXCLUSIVE")

        # Kick off the finalizer write; it must retry (not wedge the loop).
        write_task = asyncio.create_task(
            db.update_event_delivery(did, status="running")
        )

        # Prove the loop is NOT wedged while the write retries: an unrelated
        # await makes progress.
        progressed = 0
        for _ in range(3):
            await asyncio.sleep(0.02)
            progressed += 1
        assert progressed == 3, "event loop must stay responsive during retry"
        assert not write_task.done(), "the write should still be retrying under the lock"

        # Release the lock; the retry lands the write.
        raw.rollback()
        raw.close()

        await asyncio.wait_for(write_task, timeout=3.0)
        assert (await db.get_event_delivery(did))["status"] == "running", \
            "the retried write must land after the lock is released"
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
