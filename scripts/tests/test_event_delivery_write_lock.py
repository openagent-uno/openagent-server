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
