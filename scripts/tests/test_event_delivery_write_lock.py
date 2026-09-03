"""An inbound webhook must not be refused because something else was reading.

The failure (lyra-agent, 2-set-2026): every inbound conversation from Replio
came back HTTP 500 with ``sqlite3.OperationalError: database is locked``, for
hours, while three task runs sat holding the database. Before Replio had a
durable delivery queue those events were simply gone — the customer wrote and
the agent never woke up.

What made it survive a three-minute ``busy_timeout`` is the part worth pinning.
``add_event_delivery`` ran on the long-lived ``MemoryDB._conn``, and SQLite
only takes the write lock when the INSERT executes. If that connection is
already inside a read transaction, promoting it to a write is refused with
SQLITE_BUSY **immediately** — the busy handler is never invoked, so the
timeout buys nothing at all. Any concurrent reader on the shared connection
was enough.

The fix is the shape ``claim_scheduled_task`` already used for the same
reason: a dedicated short-lived connection opened with ``BEGIN IMMEDIATE``,
which asks for the write lock up front — the one case where the busy handler
does run, so the insert waits its turn instead of being refused.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid

import aiosqlite

from ._framework import TestContext, test


async def _make_db(path):
    from src.memory.db import MemoryDB
    db = MemoryDB(str(path))
    await db.connect()
    return db


async def _an_event(db):
    from src.core.event_secret import make_secret_material
    _clear, enc, hint = make_secret_material(db_path=db.db_path)
    return await db.add_event(
        name="evt-lock", action_kind="prompt", slug=f"lock-{uuid.uuid4().hex[:6]}",
        secret_enc=enc, secret_hint=hint, prompt_template="hi",
    )


@test("event_delivery_write_lock",
      "an inbound survives a read snapshot the shared connection cannot promote")
async def t_stale_read_snapshot_does_not_refuse_an_inbound(
    ctx: TestContext,
) -> None:
    """The production failure, reproduced exactly.

    Three steps, and all three are needed — this is why a plain "hold the
    lock" test proves nothing:

      1. the long-lived connection opens a read transaction and READS, which
         pins its snapshot of the WAL;
      2. somebody else writes and COMMITS, moving the database past that
         snapshot;
      3. the long-lived connection now tries to write.

    SQLite cannot promote a snapshot that is no longer current, so it returns
    SQLITE_BUSY *without calling the busy handler*: the failure is instant and
    a three-minute ``busy_timeout`` is not consulted. That is why the timeout
    that was already configured on the Lyra agent did nothing.
    """
    path = ctx.db_path.with_name(f"evlock-{uuid.uuid4().hex[:8]}.db")
    db = await _make_db(path)
    try:
        eid = await _an_event(db)

        shared = await db._ensure_connected()
        await shared.execute("BEGIN")
        await shared.execute("SELECT count(*) FROM events")   # snapshot pinned

        other = await aiosqlite.connect(str(path))
        try:
            await other.execute(f"PRAGMA busy_timeout = 5000")
            await other.execute(
                "UPDATE events SET last_triggered_at = ? WHERE id = ?",
                (time.time(), eid),
            )
            await other.commit()                              # snapshot now stale
        finally:
            await other.close()

        try:
            did = await db.add_event_delivery(event_id=eid, payload={"a": 1})
            assert did, "the delivery was not recorded"
        finally:
            try:
                await shared.execute("ROLLBACK")
            except Exception:
                pass

        rows = await db.list_event_deliveries(event_id=eid)
        assert len(rows) == 1, f"the inbound was lost: {rows!r}"
    finally:
        await db.close()


@test("event_delivery_write_lock",
      "an inbound waits for a write lock held elsewhere instead of being refused")
async def t_inbound_waits_for_the_write_lock(ctx: TestContext) -> None:
    """A behaviour pin, not a proof of the fix.

    This already held before: when the lock is taken by somebody else and the
    connection is NOT sitting on a stale snapshot, the busy handler runs and
    the write waits. It is here so that a future change cannot quietly turn
    the wait into an immediate refusal — which is the failure the test above
    reproduces.
    """
    path = ctx.db_path.with_name(f"evlock2-{uuid.uuid4().hex[:8]}.db")
    db = await _make_db(path)
    # Keep the wait short: the point is that it waits at all, not how long
    # production is willing to.
    os.environ["OPENAGENT_SQLITE_BUSY_TIMEOUT_MS"] = "10000"
    try:
        eid = await _an_event(db)

        blocker = await aiosqlite.connect(str(path))
        await blocker.execute("BEGIN IMMEDIATE")          # holds the write lock
        await blocker.execute(
            "UPDATE events SET last_triggered_at = ? WHERE id = ?", (time.time(), eid)
        )

        started = time.monotonic()
        task = asyncio.create_task(db.add_event_delivery(event_id=eid, payload={}))
        await asyncio.sleep(0.6)
        assert not task.done(), "it gave up instead of waiting for the lock"

        await blocker.commit()
        await blocker.close()

        did = await asyncio.wait_for(task, timeout=10)
        assert did, "the delivery was lost once the lock cleared"
        assert time.monotonic() - started >= 0.5, "it cannot have waited"
    finally:
        os.environ.pop("OPENAGENT_SQLITE_BUSY_TIMEOUT_MS", None)
        await db.close()


@test("event_delivery_write_lock",
      "the delivery id the caller supplied is the one stored")
async def t_supplied_delivery_id_is_kept(ctx: TestContext) -> None:
    # The id is how a replay is recognised as a replay rather than answered
    # twice; moving the insert onto its own connection must not change it.
    path = ctx.db_path.with_name(f"evlock3-{uuid.uuid4().hex[:8]}.db")
    db = await _make_db(path)
    try:
        eid = await _an_event(db)
        mine = str(uuid.uuid4())
        got = await db.add_event_delivery(event_id=eid, delivery_id=mine, payload={})
        assert got == mine
        rows = await db.list_event_deliveries(event_id=eid)
        assert rows[0]["id"] == mine
    finally:
        await db.close()


@test("event_delivery_write_lock",
      "a delivery is finalised even from a connection whose snapshot went stale")
async def t_finaliser_survives_a_stale_snapshot(ctx: TestContext) -> None:
    """The other half, found by fixing the first half.

    With ingest repaired the Lyra agent recorded nine deliveries and finalised
    none: every one sat at ``received``. ``update_event_delivery`` had a
    bounded retry, but it retried the SAME promotion on the SAME connection —
    and a stale snapshot cannot be promoted however many times you ask, so
    three attempts in 350 ms failed three identical times.
    """
    path = ctx.db_path.with_name(f"evfin-{uuid.uuid4().hex[:8]}.db")
    db = await _make_db(path)
    try:
        eid = await _an_event(db)
        did = await db.add_event_delivery(event_id=eid, payload={})

        shared = await db._ensure_connected()
        await shared.execute("BEGIN")
        await shared.execute("SELECT count(*) FROM event_deliveries")

        other = await aiosqlite.connect(str(path))
        try:
            await other.execute("PRAGMA busy_timeout = 5000")
            await other.execute(
                "UPDATE events SET last_triggered_at = ? WHERE id = ?",
                (time.time(), eid),
            )
            await other.commit()
        finally:
            await other.close()

        try:
            await db.update_event_delivery(did, status="success")
        finally:
            try:
                await shared.execute("ROLLBACK")
            except Exception:
                pass

        row = await db.get_event_delivery(did)
        assert row["status"] == "success", (
            f"the delivery was never finalised: {row['status']!r}"
        )
    finally:
        await db.close()


@test("event_delivery_write_lock",
      "an unclaimed delivery is claimed even from a stale snapshot")
async def t_claim_survives_a_stale_snapshot(ctx: TestContext) -> None:
    """The last link, and the one that decides whether anybody answers.

    A claim is a SELECT followed by an UPDATE — the read-then-promote shape
    SQLite refuses instantly on a snapshot the database has moved past. With
    ingest and finalisation already repaired, the Lyra agent logged
    ``scheduler.event_claim_failed`` 49 times in ten minutes and every
    re-enqueued orphan sat unclaimed: the conversation was captured and never
    answered, which from the outside is indistinguishable from losing it.
    """
    path = ctx.db_path.with_name(f"evclaim-{uuid.uuid4().hex[:8]}.db")
    db = await _make_db(path)
    try:
        eid = await _an_event(db)
        # `claimed=False` is the out-of-process shape the drain must pick up —
        # and the shape the orphan reaper restores a row to.
        did = await db.add_event_delivery(event_id=eid, payload={}, claimed=False)

        shared = await db._ensure_connected()
        await shared.execute("BEGIN")
        await shared.execute("SELECT count(*) FROM event_deliveries")

        other = await aiosqlite.connect(str(path))
        try:
            await other.execute("PRAGMA busy_timeout = 5000")
            await other.execute(
                "UPDATE events SET last_triggered_at = ? WHERE id = ?",
                (time.time(), eid),
            )
            await other.commit()
        finally:
            await other.close()

        try:
            claimed = await db.claim_pending_event_deliveries(limit=10)
        finally:
            try:
                await shared.execute("ROLLBACK")
            except Exception:
                pass

        assert [c["id"] for c in claimed] == [did], (
            f"the delivery was not claimed: {claimed!r}"
        )
        row = await db.get_event_delivery(did)
        assert row["claimed_at"] is not None, "the claim did not stick"
    finally:
        await db.close()


@test("event_delivery_write_lock",
      "a delivery write gives up in seconds rather than holding an HTTP request")
async def t_delivery_write_is_bounded(ctx: TestContext) -> None:
    """The cost of moving these writes onto their own connection.

    A sibling holding an uncommitted transaction on the SHARED connection now
    BLOCKS them instead of being joined by them — and the global
    ``busy_timeout`` on the agents is three minutes. On a write that sits on
    an HTTP request that is not a fix, it is a different outage: Replio drops
    a delivery it has waited ten seconds for, which showed up on the receiver
    as ``ConnectionResetError`` while reading the body.

    So the wait is bounded well under that. Several modules outside
    ``memory/db.py`` take the shared connection, so this state is not rare and
    not something this class can rule out.
    """
    import os as _os

    path = ctx.db_path.with_name(f"evbound-{uuid.uuid4().hex[:8]}.db")
    db = await _make_db(path)
    _os.environ["OPENAGENT_SQLITE_BUSY_TIMEOUT_MS"] = "180000"   # come in produzione
    _os.environ["OPENAGENT_DELIVERY_LOCK_WAIT_MS"] = "1000"
    try:
        eid = await _an_event(db)

        blocker = await aiosqlite.connect(str(path))
        await blocker.execute("BEGIN IMMEDIATE")
        await blocker.execute(
            "UPDATE events SET last_triggered_at = ? WHERE id = ?", (time.time(), eid)
        )
        try:
            started = time.monotonic()
            try:
                await asyncio.wait_for(
                    db.add_event_delivery(event_id=eid, payload={}), timeout=20
                )
            except asyncio.TimeoutError:
                raise AssertionError(
                    "the write is still waiting on the global three-minute timeout"
                )
            except Exception:
                pass  # refused is fine; hanging is not
            waited = time.monotonic() - started
            assert waited < 5, f"waited {waited:.1f}s for a lock it should abandon"

            # A claim in the same state must take nothing rather than hold the
            # scheduler: the next tick is seconds away.
            claimed = await asyncio.wait_for(
                db.claim_pending_event_deliveries(limit=5), timeout=20
            )
            assert claimed == []
        finally:
            await blocker.rollback()
            await blocker.close()
    finally:
        _os.environ.pop("OPENAGENT_DELIVERY_LOCK_WAIT_MS", None)
        _os.environ.pop("OPENAGENT_SQLITE_BUSY_TIMEOUT_MS", None)
        await db.close()


@test("event_delivery_write_lock",
      "the lease reaper recovers an orphan from a stale snapshot")
async def t_lease_reaper_survives_a_stale_snapshot(ctx: TestContext) -> None:
    """The recovery path, which is the last thing standing between a frozen
    turn and a customer who is never answered.

    `scheduler.lease_reap_loop_error` fired 1144 times in an hour on the Lyra
    agent. The cause was invisible from `kubectl logs`, because the console
    renderer drops the structured fields — the reason only appears in
    `logs/events.jsonl`, where every one of them reads
    ``"error": "database is locked"``. Same read-then-promote shape as the
    other four sites, on the shared connection.

    Bursty, and tied to load: a quiet window shows zero. So the proof is here,
    not in a production measurement taken while nothing was happening.
    """
    path = ctx.db_path.with_name(f"evreap-{uuid.uuid4().hex[:8]}.db")
    db = await _make_db(path)
    try:
        eid = await _an_event(db)
        did = await db.add_event_delivery(event_id=eid, payload={}, claimed=True)
        # Expire the lease: this is the row a reaper exists to recover.
        async with db._delivery_write() as w:
            await w.execute(
                "UPDATE event_deliveries SET claim_expires = ?, status='running' "
                " WHERE id = ?",
                (time.time() - 3600, did),
            )

        shared = await db._ensure_connected()
        await shared.execute("BEGIN")
        await shared.execute("SELECT count(*) FROM event_deliveries")

        other = await aiosqlite.connect(str(path))
        try:
            await other.execute("PRAGMA busy_timeout = 5000")
            await other.execute(
                "UPDATE events SET last_triggered_at = ? WHERE id = ?",
                (time.time(), eid),
            )
            await other.commit()
        finally:
            await other.close()

        try:
            acted = await db.reap_expired_event_leases()
        finally:
            try:
                await shared.execute("ROLLBACK")
            except Exception:
                pass

        assert acted == 1, f"the orphan was not recovered (acted={acted})"
        row = await db.get_event_delivery(did)
        assert row["status"] == "received", f"still {row['status']!r}"
        assert row["claimed_at"] is None, "the claim was not cleared"
    finally:
        await db.close()
