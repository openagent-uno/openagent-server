"""Per-event circuit breaker — with the load-bearing transient exclusion.

Today the only guard on a repeatedly-failing event is the per-DELIVERY replay
budget; there is no per-EVENT breaker, so a genuinely broken event (a bad
template, a rejected action) is re-attempted delivery after delivery. This adds
a breaker: N consecutive PERMANENT failures trip it, after which further
deliveries are parked ``blocked`` (durable/visible) instead of run, until a
success resets the streak.

THE load-bearing safety property is the TRANSIENT EXCLUSION. The root incident is
a provider rate-limit STORM. A naive breaker that counted every failed turn would
trip on the healthy support event exactly when the provider is throttled and
block it — catastrophic. So a provider-429 / quota / throttle / turn-timeout, and
a barge-in cancellation, are classified TRANSIENT and released WITHOUT a count
(mirroring Hermes' KANBAN_RATE_LIMIT_EXIT_CODE=75). Only a genuine permanent
failure moves the breaker.

The whole breaker is gated behind ``OPENAGENT_EVENT_BREAKER_ENABLED`` (default
OFF); with it unset every method is a no-op and behaviour is identical to today.

LLM-free: the model turn (``_dispatch_prompt``) is replaced with an instrumented
fake that raises/returns on command, so ``dispatch_event``'s real breaker wiring
(classify → record/reset) is exercised without a real turn; the drain-skips-
blocked case swaps the whole ``dispatch_event`` for a call recorder.
"""
from __future__ import annotations

import asyncio
import os
import uuid

from ._framework import TestContext, test


def _fresh_db_path(ctx: TestContext):
    return ctx.db_path.with_name(f"evbreaker-{uuid.uuid4().hex[:8]}.db")


async def _make_db(path):
    from src.memory.db import MemoryDB
    db = MemoryDB(str(path))
    await db.connect()
    return db


async def _add_event(db, *, slug: str, max_retries=None) -> str:
    from src.core.event_secret import make_secret_material
    _clear, enc, hint = make_secret_material(db_path=db.db_path)
    eid = await db.add_event(
        name=f"evt-{slug}", action_kind="prompt", slug=slug,
        secret_enc=enc, secret_hint=hint, prompt_template="Handle it",
    )
    if max_retries is not None:
        conn = await db._ensure_connected()
        await conn.execute(
            "UPDATE events SET max_retries = ? WHERE id = ?", (max_retries, eid),
        )
        await conn.commit()
    return eid


class _DummyAgent:
    name = "dummy"
    model = None


async def _drive_dispatch(db, event, *, raise_exc=None, result=None):
    """Run the REAL ``dispatch_event`` for one fresh delivery with a fake
    ``_dispatch_prompt`` that raises ``raise_exc`` or returns ``result``. Returns
    (delivery_id, outcome) where outcome is 'ok' | 'raised' | 'cancelled'."""
    import src.core.event_dispatcher as ed

    did = await db.add_event_delivery(
        event_id=event["id"], payload={"ticket": {"id": "x"}}, claimed=True,
    )

    async def fake_prompt(*, agent, db, event, payload, delivery_id, source, on_link=None):
        if raise_exc is not None:
            raise raise_exc
        return result or {"status": "success", "session_id": "s", "output": "ok"}

    orig = ed._dispatch_prompt
    ed._dispatch_prompt = fake_prompt
    try:
        await ed.dispatch_event(
            agent=_DummyAgent(), db=db, scheduler=None, event=event,
            payload={"ticket": {"id": "x"}}, delivery_id=did,
            source="webhook", broadcast=None,
        )
        return did, "ok"
    except asyncio.CancelledError:
        return did, "cancelled"
    except Exception:  # noqa: BLE001
        return did, "raised"
    finally:
        ed._dispatch_prompt = orig


# ── 1. N consecutive permanent failures trip the breaker ───────────────────


@test("event_breaker",
      "N consecutive permanent failures trip the breaker (breaker_tripped_at set)")
async def t_trips_on_permanent(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    os.environ["OPENAGENT_EVENT_BREAKER_ENABLED"] = "1"
    os.environ["OPENAGENT_EVENT_BREAKER_THRESHOLD"] = "3"
    try:
        eid = await _add_event(db, slug="brk-perm")
        event = await db.get_event(eid)

        for i in range(2):
            did, outcome = await _drive_dispatch(
                db, event, raise_exc=Exception("bad template: missing field"),
            )
            assert outcome == "raised", outcome
            assert (await db.get_event_delivery(did))["status"] == "failed"
            ev = await db.get_event(eid)
            assert ev["consecutive_failures"] == i + 1, ev["consecutive_failures"]
            assert not await db.is_event_breaker_tripped(eid), "must not trip early"

        # Third permanent failure crosses the threshold → trips.
        await _drive_dispatch(db, event, raise_exc=Exception("rejected action"))
        ev = await db.get_event(eid)
        assert ev["consecutive_failures"] == 3, ev["consecutive_failures"]
        assert ev["breaker_tripped_at"] is not None, "breaker must trip at the limit"
        assert await db.is_event_breaker_tripped(eid) is True
    finally:
        os.environ.pop("OPENAGENT_EVENT_BREAKER_ENABLED", None)
        os.environ.pop("OPENAGENT_EVENT_BREAKER_THRESHOLD", None)
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 2. A tripped breaker parks new deliveries ``blocked`` (not dispatched) ──


@test("event_breaker",
      "a tripped breaker parks a new delivery blocked and the drain never dispatches it")
async def t_tripped_breaker_blocks_drain(ctx: TestContext) -> None:
    import src.core.event_dispatcher as ed
    from src.core.scheduler import Scheduler

    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    os.environ["OPENAGENT_EVENT_BREAKER_ENABLED"] = "1"
    os.environ["OPENAGENT_EVENT_BREAKER_THRESHOLD"] = "2"

    dispatched: list[str] = []

    async def recording_dispatch(**kw):
        dispatched.append(kw["delivery_id"])
        await db.update_event_delivery(kw["delivery_id"], status="success")
        return {"status": "success"}

    orig = ed.dispatch_event
    ed.dispatch_event = recording_dispatch
    try:
        eid = await _add_event(db, slug="brk-block")
        # Trip the breaker directly (the recording semantics are what we test).
        for _ in range(2):
            await db.record_event_failure(eid, "boom")
        assert await db.is_event_breaker_tripped(eid) is True

        # An out-of-process delivery arrives while the breaker is open.
        did = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "Q"}}, claimed=False,
        )
        scheduler = Scheduler(db=db, agent=_DummyAgent())  # type: ignore[arg-type]
        await scheduler._drain_event_deliveries()
        await asyncio.sleep(0)  # let any (wrongly) spawned dispatch run

        assert dispatched == [], f"a blocked event must never dispatch, got {dispatched}"
        row = await db.get_event_delivery(did)
        assert row["status"] == "blocked", f"delivery must be parked blocked, got {row['status']!r}"
        assert "breaker" in (row.get("error") or "").lower(), row.get("error")
    finally:
        ed.dispatch_event = orig
        os.environ.pop("OPENAGENT_EVENT_BREAKER_ENABLED", None)
        os.environ.pop("OPENAGENT_EVENT_BREAKER_THRESHOLD", None)
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 3. THE core safety property: transient failures NEVER trip the breaker ──


@test("event_breaker",
      "transient failures (429 / rate-limit / cancellation) never trip the breaker")
async def t_transient_never_trips(ctx: TestContext) -> None:
    from src.core.runtime_errors import ModelRateLimitError

    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    os.environ["OPENAGENT_EVENT_BREAKER_ENABLED"] = "1"
    os.environ["OPENAGENT_EVENT_BREAKER_THRESHOLD"] = "2"  # trips FAST if buggy
    try:
        eid = await _add_event(db, slug="brk-transient")
        event = await db.get_event(eid)

        # A storm of throttle signals, well over the (deliberately tiny) limit.
        transient_errors = [
            Exception("HTTP 429 Too Many Requests: rate limit exceeded"),
            ModelRateLimitError("provider rate limited", status_code=429),
            Exception("provider quota exhausted"),
            asyncio.TimeoutError(),
            asyncio.CancelledError(),
            Exception("529 overloaded, temporarily unavailable"),
        ]
        for err in transient_errors:
            did, outcome = await _drive_dispatch(db, event, raise_exc=err)
            assert outcome in ("raised", "cancelled"), outcome
            ev = await db.get_event(eid)
            assert ev["consecutive_failures"] == 0, (
                "a transient failure must NOT count — a rate-limit storm cannot "
                f"be allowed to trip the breaker; got {ev['consecutive_failures']}"
            )
            assert await db.is_event_breaker_tripped(eid) is False, \
                "the breaker must never trip on transient failures"

        # And the event is still healthy: it is NOT blocked, so a real delivery
        # would run.
        assert await db.is_event_breaker_tripped(eid) is False
    finally:
        os.environ.pop("OPENAGENT_EVENT_BREAKER_ENABLED", None)
        os.environ.pop("OPENAGENT_EVENT_BREAKER_THRESHOLD", None)
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 4. A terminal success resets the streak ────────────────────────────────


@test("event_breaker",
      "a terminal success resets consecutive_failures to 0")
async def t_success_resets(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    os.environ["OPENAGENT_EVENT_BREAKER_ENABLED"] = "1"
    os.environ["OPENAGENT_EVENT_BREAKER_THRESHOLD"] = "3"
    try:
        eid = await _add_event(db, slug="brk-reset")
        event = await db.get_event(eid)

        for _ in range(2):
            await _drive_dispatch(db, event, raise_exc=Exception("bad template"))
        assert (await db.get_event(eid))["consecutive_failures"] == 2

        # A clean success wipes the streak (and un-trips if it were tripped).
        did, outcome = await _drive_dispatch(db, event, result={"status": "success", "output": "done"})
        assert outcome == "ok", outcome
        assert (await db.get_event_delivery(did))["status"] == "success"
        ev = await db.get_event(eid)
        assert ev["consecutive_failures"] == 0, ev["consecutive_failures"]
        assert ev["breaker_tripped_at"] is None
        assert await db.is_event_breaker_tripped(eid) is False
    finally:
        os.environ.pop("OPENAGENT_EVENT_BREAKER_ENABLED", None)
        os.environ.pop("OPENAGENT_EVENT_BREAKER_THRESHOLD", None)
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 5. A per-event max_retries override beats the global default ───────────


@test("event_breaker",
      "a per-event max_retries overrides the global breaker threshold")
async def t_per_event_max_retries(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    os.environ["OPENAGENT_EVENT_BREAKER_ENABLED"] = "1"
    os.environ["OPENAGENT_EVENT_BREAKER_THRESHOLD"] = "10"  # global is lenient
    try:
        eid = await _add_event(db, slug="brk-override", max_retries=1)  # strict
        event = await db.get_event(eid)
        # One permanent failure hits the per-event limit of 1 → trips, despite
        # the global default of 10.
        await _drive_dispatch(db, event, raise_exc=Exception("bad template"))
        ev = await db.get_event(eid)
        assert ev["consecutive_failures"] == 1, ev["consecutive_failures"]
        assert await db.is_event_breaker_tripped(eid) is True, \
            "per-event max_retries=1 must trip after a single permanent failure"
    finally:
        os.environ.pop("OPENAGENT_EVENT_BREAKER_ENABLED", None)
        os.environ.pop("OPENAGENT_EVENT_BREAKER_THRESHOLD", None)
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 6. OFF by default: failures never block, behaviour identical to today ──


@test("event_breaker",
      "with OPENAGENT_EVENT_BREAKER_ENABLED unset, failures never trip or block (identical to today)")
async def t_off_by_default(ctx: TestContext) -> None:
    # Explicitly ensure the flag is unset.
    os.environ.pop("OPENAGENT_EVENT_BREAKER_ENABLED", None)
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        eid = await _add_event(db, slug="brk-off")
        event = await db.get_event(eid)

        # A long run of permanent failures with the breaker OFF.
        for _ in range(6):
            did, outcome = await _drive_dispatch(db, event, raise_exc=Exception("bad template"))
            assert outcome == "raised", outcome
            # Delivery still records failed (unchanged legacy behaviour) ...
            assert (await db.get_event_delivery(did))["status"] == "failed"

        ev = await db.get_event(eid)
        # ... but the breaker counter never moves and nothing is ever tripped.
        assert ev["consecutive_failures"] == 0, ev["consecutive_failures"]
        assert ev["breaker_tripped_at"] is None
        assert await db.is_event_breaker_tripped(eid) is False

        # And the drain would dispatch normally — no delivery is ever blocked.
        assert await db.is_event_breaker_tripped(eid) is False
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
