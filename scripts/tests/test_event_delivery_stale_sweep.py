"""Periodic, age-gated stale-orphan sweep for event deliveries.

The startup reap (``reap_orphan_event_deliveries``, called once at boot) only
recovers CRASH orphans: crash → restart → reap re-enqueues every claimed row,
which is safe because at boot everything claimed is provably orphaned. But a
delivery orphaned WITHOUT a restart — a detached ``dispatch_event`` task that
dies silently while the process keeps serving — never sees a restart, so it
sits ``running``/claimed until the next deploy.

``reap_stale_event_deliveries`` closes that gap on a periodic cadence. Because
the process is LIVE, a claimed row is NOT provably an orphan (a claim 30 s old
is a legitimately-running turn), so the sweep is AGE-GATED: it only re-enqueues
rows whose ``claimed_at`` is older than ``min_claim_age_seconds`` — a threshold
the Scheduler sets to 2× the single-turn wall-clock cap (default 1800 s). A row
claimed more recently is left strictly alone, which is what prevents a second
concurrent dispatch of a still-running turn.

These tests pin the four load-bearing guarantees:
  1. a long-claimed in-flight orphan IS re-enqueued;
  2. a recently-claimed row is NOT touched (the age guard);
  3. a ``failed`` historical row is NEVER touched (go-forward only, no backfill);
  4. the replay budget still caps it (parked terminal at max attempts);
  5. a recently-claimed row over the budget is NOT parked (age guard on park too);
  6. the ``OPENAGENT_EVENT_REENQUEUE_ENABLED`` kill-switch → no-op.
"""
from __future__ import annotations

import os
import time
import uuid

from ._framework import TestContext, test


def _fresh_db_path(ctx: TestContext) -> "os.PathLike":
    return ctx.db_path.with_name(f"evstale-{uuid.uuid4().hex[:8]}.db")


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
        session_binding_enabled=True, session_binding_path="ticket.id",
    )


async def _set_claimed_at(db, delivery_id: str, ts: float) -> None:
    """Force a delivery's ``claimed_at`` to an explicit epoch time.

    ``update_event_delivery`` deliberately does not expose ``claimed_at``, so a
    test simulating "claimed N seconds ago" pokes the column directly."""
    conn = await db._ensure_connected()
    await conn.execute(
        "UPDATE event_deliveries SET claimed_at = ? WHERE id = ?",
        (ts, delivery_id),
    )
    await conn.commit()


# ── 1. A long-claimed in-flight orphan IS re-enqueued ─────────────────────


@test("event_delivery_stale_sweep",
      "a delivery claimed long ago (still running) is re-enqueued by the stale sweep")
async def t_stale_orphan_reenqueued(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        eid = await _add_event(db, slug="stale-basic")
        did = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "T-1", "body": "help"}},
            claimed=True,
        )
        await db.update_event_delivery(did, status="running")
        # Claimed an hour ago — far past any turn's wall-clock cap: a real
        # orphan whose dispatch task died with no restart to catch it.
        await _set_claimed_at(db, did, time.time() - 3600)

        # Age gate = 30 min; this claim is 60 min old → eligible.
        n = await db.reap_stale_event_deliveries(min_claim_age_seconds=1800)
        assert n == 1, f"expected 1 stale orphan re-enqueued, got {n}"

        row = await db.get_event_delivery(did)
        assert row["status"] == "received", \
            f"stale orphan should be RE-ENQUEUED (received), got {row['status']!r}"
        assert row["claimed_at"] is None, "re-enqueued row must drop its claim"
        assert row["reenqueue_count"] == 1, row["reenqueue_count"]
        assert "stale-sweep" in (row.get("error") or ""), row.get("error")

        # And the drain would now claim exactly this row.
        claimed = await db.claim_pending_event_deliveries()
        assert [c["id"] for c in claimed] == [did], claimed
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 2. A recently-claimed row is NOT touched (the age guard) ──────────────


@test("event_delivery_stale_sweep",
      "a delivery claimed recently (within the age threshold) is left running, not re-enqueued")
async def t_recent_claim_not_touched(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        eid = await _add_event(db, slug="stale-recent")
        did = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "T-2", "body": "x"}},
            claimed=True,
        )
        await db.update_event_delivery(did, status="running")
        # Claimed 30 s ago — a legitimately-running turn. Re-enqueuing this
        # would spawn a SECOND concurrent dispatch: the exact double-fire the
        # age guard exists to prevent.
        await _set_claimed_at(db, did, time.time() - 30)

        n = await db.reap_stale_event_deliveries(min_claim_age_seconds=1800)
        assert n == 0, f"a recently-claimed row must be untouched, acted on {n}"

        row = await db.get_event_delivery(did)
        assert row["status"] == "running", \
            f"live turn must stay running, got {row['status']!r}"
        assert row["claimed_at"] is not None, "live turn must keep its claim"
        assert row["reenqueue_count"] == 0, row["reenqueue_count"]
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 3. A failed historical row is NEVER touched (go-forward only) ─────────


@test("event_delivery_stale_sweep",
      "the stale sweep never touches a failed historical row (no backfill)")
async def t_failed_row_never_touched(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        eid = await _add_event(db, slug="stale-failed")

        # (a) a genuine application failure, claimed & old.
        genuine = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "G", "body": "x"}}, claimed=True,
        )
        await db.update_event_delivery(
            genuine, status="failed", error="EventDispatchError: bad template",
        )
        await _set_claimed_at(db, genuine, time.time() - 7200)

        # (b) a historical orphan the OLD reaper dropped — carries the legacy
        #     marker the STARTUP reap backfills. The periodic sweep is
        #     go-forward only and must leave even this one alone.
        historical = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "H", "body": "y"}}, claimed=True,
        )
        await db.update_event_delivery(
            historical, status="failed",
            error="reaped: orphan from prior process",
        )
        await _set_claimed_at(db, historical, time.time() - 7200)

        n = await db.reap_stale_event_deliveries(min_claim_age_seconds=1800)
        assert n == 0, f"the stale sweep must never touch failed rows, acted on {n}"
        assert (await db.get_event_delivery(genuine))["status"] == "failed"
        assert (await db.get_event_delivery(historical))["status"] == "failed", \
            "the periodic sweep must NOT backfill historical orphans"
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 4. The replay budget still caps a persistently-orphaning delivery ─────


@test("event_delivery_stale_sweep",
      "a stale delivery over the replay budget is parked terminal, not re-enqueued forever")
async def t_budget_caps(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        eid = await _add_event(db, slug="stale-budget")
        did = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "T-3", "body": "x"}}, claimed=True,
        )
        await db.update_event_delivery(did, status="running")

        requeues = 0
        for _ in range(5):
            # Faithful cycle: the row is stale each round (claim aged past the
            # gate), the turn re-starts, dies again → stale sweep.
            row = await db.get_event_delivery(did)
            if row["claimed_at"] is None:
                await db.claim_pending_event_deliveries()
            await db.update_event_delivery(did, status="running")
            await _set_claimed_at(db, did, time.time() - 3600)
            await db.reap_stale_event_deliveries(min_claim_age_seconds=1800, max_attempts=2)
            row = await db.get_event_delivery(did)
            if row["status"] == "received":
                requeues += 1
            elif row["status"] == "failed":
                break

        row = await db.get_event_delivery(did)
        assert requeues == 2, f"expected exactly max_attempts=2 re-enqueues, got {requeues}"
        assert row["status"] == "failed", \
            f"an exhausted stale orphan must be parked terminal, got {row['status']!r}"
        assert "stale-sweep: retry budget exhausted" in (row.get("error") or ""), \
            row.get("error")
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 5. A recently-claimed row over budget is NOT parked (age guard on park) ─


@test("event_delivery_stale_sweep",
      "a recently-claimed row at the budget cap is left running, not parked")
async def t_recent_over_budget_not_parked(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        eid = await _add_event(db, slug="stale-recent-budget")
        did = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "T-4", "body": "x"}}, claimed=True,
        )
        await db.update_event_delivery(did, status="running")
        # Already at the cap, but claimed only moments ago → a live turn that
        # has legitimately been re-enqueued before. The age guard must protect
        # it from being parked mid-run.
        conn = await db._ensure_connected()
        await conn.execute(
            "UPDATE event_deliveries SET reenqueue_count = 5 WHERE id = ?", (did,),
        )
        await conn.commit()
        await _set_claimed_at(db, did, time.time() - 20)

        n = await db.reap_stale_event_deliveries(min_claim_age_seconds=1800, max_attempts=5)
        assert n == 0, f"a live row must not be parked by the sweep, acted on {n}"
        row = await db.get_event_delivery(did)
        assert row["status"] == "running", \
            f"a live over-budget turn must stay running, got {row['status']!r}"
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 6. Kill-switch → no-op (no marking, unlike the startup reap) ──────────


@test("event_delivery_stale_sweep",
      "the OPENAGENT_EVENT_REENQUEUE_ENABLED kill-switch makes the sweep a no-op")
async def t_kill_switch_noop(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        eid = await _add_event(db, slug="stale-killswitch")
        did = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "K", "body": "x"}}, claimed=True,
        )
        await db.update_event_delivery(did, status="running")
        await _set_claimed_at(db, did, time.time() - 3600)

        # Explicit enabled=False (the scheduler reads the env flag; passing it
        # directly pins the same branch without mutating process env).
        n = await db.reap_stale_event_deliveries(
            min_claim_age_seconds=1800, enabled=False,
        )
        assert n == 0, f"kill-switch must make the sweep a no-op, acted on {n}"
        row = await db.get_event_delivery(did)
        # Unlike the startup reap's kill-switch, the periodic sweep does NOT
        # mark the row failed — it leaves it exactly as it was for the next
        # restart's reap to handle.
        assert row["status"] == "running", \
            f"disabled sweep must not touch the row, got {row['status']!r}"
        assert row["claimed_at"] is not None, "disabled sweep must not drop the claim"

        # Env-var form of the same switch, to prove the flag path.
        os.environ["OPENAGENT_EVENT_REENQUEUE_ENABLED"] = "0"
        try:
            n2 = await db.reap_stale_event_deliveries(min_claim_age_seconds=1800)
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
