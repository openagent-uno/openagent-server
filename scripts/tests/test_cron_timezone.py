"""Timezone-aware scheduled tasks — default-compat, DST, and surfacing.

A cron expression is a wall-clock statement and used to be evaluated
against whatever clock the host happened to have. Containerised agents
run on UTC while operators think in Europe/Rome, so the crons in the
field were hand-converted ("23 11 * * 1-5 UTC ≈ 13:23 Europe/Rome") and
silently slipped an hour at every DST change.

The pre-existing default is **UTC on every host**, not the host's own
clock: ``next_run_for_expression`` hands croniter a float, and croniter
resolves a float through ``fromtimestamp(ts, tz=utc)``. ``t_default_is_utc``
pins that, because the code reads like it would use local time and a
"fix" in that direction would silently re-aim every deployed cron.

Two properties matter here and they pull against each other:

1. **Nothing already deployed may move.** Those hand-converted crons are
   correct *for UTC*. ``t_default_matches_legacy`` pins the no-timezone
   path against a verbatim copy of the pre-timezone implementation, so a
   future edit that "cleans up" the default path fails here rather than in
   production at 09:00.
2. **A tagged cron must hold its wall-clock hour across a transition**,
   firing exactly once — not twice on the repeated hour, not zero times
   on the skipped one.

Dates are real Europe/Rome transitions: 2026-03-29 (02:00→03:00) and
2026-10-25 (03:00→02:00).
"""
from __future__ import annotations

import datetime as dt
import time
import uuid
from zoneinfo import ZoneInfo

from croniter import croniter

from ._framework import TestContext, test

ROME = ZoneInfo("Europe/Rome")
UTC = dt.timezone.utc

# Europe/Rome DST transitions used throughout.
SPRING_2026 = dt.datetime(2026, 3, 29, tzinfo=ROME)   # 02:00 → 03:00 (02:xx gone)
FALL_2026 = dt.datetime(2026, 10, 25, tzinfo=ROME)    # 03:00 → 02:00 (02:xx twice)


def _legacy_next_run(expr: str, base: float | None = None) -> float:
    """Verbatim copy of ``next_run_for_expression`` as it was *before*
    timezones existed (minus the ``@once:`` branch, which is unchanged).

    Kept as a literal duplicate on purpose: the point is to detect the
    new implementation drifting away from the shipped behaviour, so this
    must not import or share code with the thing it is checking.
    """
    return croniter(expr, time.time() if base is None else base).get_next(float)


def _rome(y: int, m: int, d: int, hh: int = 0, mm: int = 0, fold: int = 0) -> float:
    return dt.datetime(y, m, d, hh, mm, tzinfo=ROME, fold=fold).timestamp()


def _walk(expr: str, start: float, count: int, tz: str | None) -> list[float]:
    """Step a schedule the way the Scheduler does — each next_run computed
    from the previous fire — so the sequence reflects real advancement
    rather than a croniter iterator held open across the transition."""
    from src.memory.schedule import next_run_for_expression

    out: list[float] = []
    base = start
    for _ in range(count):
        nxt = next_run_for_expression(expr, base, tz)
        out.append(nxt)
        base = nxt
    return out


# ── 1. The default must reproduce today's behaviour exactly ──


@test("cron_timezone", "default path is identical to the pre-timezone implementation")
async def t_default_matches_legacy(_ctx: TestContext) -> None:
    from src.memory.schedule import next_run_for_expression

    exprs = [
        "0 9 * * *", "23 11 * * 1-5", "*/30 * * * *", "0 * * * *",
        "0 2 * * *", "30 9 * * 1-5", "0 */2 * * *", "* 2 * * *",
        "0 0 1 * *", "@daily", "@hourly", "15 3 * * 0",
    ]
    # Bases include both Europe/Rome DST transitions and a plain winter /
    # summer day, expressed as absolute epochs so the comparison holds
    # whatever timezone the machine running the suite is in.
    bases = [
        _rome(2026, 3, 28, 12, 0), _rome(2026, 3, 29, 1, 30),
        _rome(2026, 10, 24, 12, 0), _rome(2026, 10, 25, 1, 30),
        _rome(2026, 1, 15, 0, 0), _rome(2026, 7, 15, 23, 59),
        time.time(),
    ]
    for expr in exprs:
        for base in bases:
            got = next_run_for_expression(expr, base)          # tz omitted
            want = _legacy_next_run(expr, base)
            assert got == want, (
                f"default path drifted for {expr!r} at base={base}: "
                f"{got!r} != legacy {want!r}"
            )
    # Explicit None must behave the same as omitting it.
    for expr in exprs:
        assert next_run_for_expression(expr, bases[0], None) == _legacy_next_run(
            expr, bases[0]
        ), f"tz=None diverged for {expr!r}"


@test("cron_timezone", "empty-string timezone means the default, not a crash")
async def t_empty_tz_is_default(_ctx: TestContext) -> None:
    from src.memory.schedule import next_run_for_expression, validate_timezone

    base = _rome(2026, 1, 15, 0, 0)
    # "" is what a REST/JSON caller sends to mean "unset"; it must land on
    # the legacy path rather than raising.
    validate_timezone("")
    assert next_run_for_expression("0 9 * * *", base, "") == _legacy_next_run(
        "0 9 * * *", base
    )


@test("cron_timezone", "the no-timezone default is UTC, on any host")
async def t_default_is_utc(_ctx: TestContext) -> None:
    # Pins the semantic the rest of this file leans on. It is not obvious
    # from the source: next_run_for_expression passes a float to croniter,
    # and croniter resolves floats via fromtimestamp(ts, tz=utc) — so the
    # cron is read in UTC no matter what the machine's timezone is. This
    # assertion is written in absolute UTC terms, so it holds on a UTC CI
    # box and on a CEST laptop alike.
    from src.memory.schedule import next_run_for_expression

    base = dt.datetime(2026, 7, 15, 0, 0, tzinfo=UTC).timestamp()
    fired = next_run_for_expression("0 9 * * *", base)
    assert dt.datetime.fromtimestamp(fired, UTC) == dt.datetime(
        2026, 7, 15, 9, 0, tzinfo=UTC
    ), (
        "'0 9 * * *' with no timezone must fire at 09:00 UTC — if this now "
        "tracks the host's local clock, every deployed cron just moved"
    )


@test("cron_timezone", "timezone='UTC' is a no-op: identical to the default")
async def t_utc_equals_default(_ctx: TestContext) -> None:
    # The migration path. An operator can make the implicit explicit — tag
    # a hand-converted cron 'UTC' — and provably not move it, before later
    # re-aiming it at a real zone. If this ever diverges, that safe first
    # step stops being safe.
    from src.memory.schedule import next_run_for_expression

    exprs = ["0 9 * * *", "23 11 * * 1-5", "*/30 * * * *", "0 2 * * *",
             "0 */2 * * *", "@daily", "* 2 * * *", "0 0 1 * *"]
    bases = [
        _rome(2026, 3, 29, 1, 0), _rome(2026, 10, 25, 1, 30),
        _rome(2026, 1, 15, 0, 0), time.time(),
    ]
    for expr in exprs:
        for base in bases:
            assert next_run_for_expression(expr, base, "UTC") == _legacy_next_run(
                expr, base
            ), f"tz='UTC' diverged from the default for {expr!r}"


# ── 2. The point of the feature: a tagged cron holds its wall clock ──


@test("cron_timezone", "tz-tagged cron holds its wall-clock hour across both transitions")
async def t_wall_clock_stable(_ctx: TestContext) -> None:
    # A 09:00 Europe/Rome briefing must read 09:00 on every side of both
    # transitions. On a host-clock cron (UTC container) the same rows would
    # read 09:00 UTC = 10:00/11:00 Rome — the drift that started all this.
    for start in (_rome(2026, 3, 27, 12, 0), _rome(2026, 10, 23, 12, 0)):
        fires = _walk("0 9 * * *", start, 5, "Europe/Rome")
        for epoch in fires:
            local = dt.datetime.fromtimestamp(epoch, ROME)
            assert (local.hour, local.minute) == (9, 0), (
                f"09:00 Europe/Rome drifted to {local.isoformat()}"
            )
        # The wall clock is fixed, so the *absolute* gap must flex: exactly
        # one 23h or 25h day proves it is following DST, not a fixed offset.
        gaps = {round((b - a) / 3600) for a, b in zip(fires, fires[1:])}
        assert gaps & {23, 25}, f"no DST-shifted day in gaps {sorted(gaps)}"


@test("cron_timezone", "a hand-converted UTC cron drifts across DST (the bug being fixed)")
async def t_utc_cron_drifts_in_rome(_ctx: TestContext) -> None:
    # Characterisation of the status quo this feature exists to end. The
    # operator's note is "23 11 * * 1-5 UTC ≈ 13:23 Europe/Rome" — the "≈"
    # is doing a lot of work, because that same expression is 12:23 in Rome
    # for the other half of the year. Nothing about the cron changes; the
    # hour the human sees does.
    from src.memory.schedule import next_run_for_expression

    seen = []
    base = _rome(2026, 3, 27, 12, 0)
    for _ in range(4):
        base = next_run_for_expression("23 11 * * 1-5", base, "UTC")
        seen.append(dt.datetime.fromtimestamp(base, ROME).strftime("%H:%M"))
    assert set(seen) == {"12:23", "13:23"}, (
        f"expected the UTC-pinned cron to shift its Rome hour over DST, got {seen}"
    )
    # And the fix: the same intent expressed in Rome does not drift.
    stable = []
    base = _rome(2026, 3, 27, 12, 0)
    for _ in range(4):
        base = next_run_for_expression("23 13 * * 1-5", base, "Europe/Rome")
        stable.append(dt.datetime.fromtimestamp(base, ROME).strftime("%H:%M"))
    assert set(stable) == {"13:23"}, f"Rome-tagged cron drifted: {stable}"


# ── 3. Spring forward: the skipped hour ──


@test("cron_timezone", "spring forward: 02:00 fires once at the jump, day not skipped")
async def t_spring_forward(_ctx: TestContext) -> None:
    # 2026-03-29 Europe/Rome: 02:00 never exists. Documented semantics
    # (crontab(5)): the job runs once, at the instant the skipped slot
    # would have elapsed — 01:00 UTC, which the clock displays as 03:00.
    fires = _walk("0 2 * * *", _rome(2026, 3, 27, 12, 0), 4, "Europe/Rome")
    on_transition = [
        f for f in fires
        if dt.datetime.fromtimestamp(f, ROME).date() == SPRING_2026.date()
    ]
    assert len(on_transition) == 1, (
        "the skipped hour must neither drop the day nor duplicate it; got "
        f"{[dt.datetime.fromtimestamp(f, ROME).isoformat() for f in on_transition]}"
    )
    fired = on_transition[0]
    would_have_been = dt.datetime(2026, 3, 29, 1, 0, tzinfo=UTC).timestamp()
    assert fired == would_have_been, (
        f"expected the fire at the 02:00-CET instant ({would_have_been}), got {fired}"
    )
    local = dt.datetime.fromtimestamp(fired, ROME)
    assert (local.hour, local.minute) == (3, 0), f"expected 03:00 wall clock, got {local}"

    # And the surrounding days are untouched 02:00s — no cascade.
    assert dt.datetime.fromtimestamp(fires[0], ROME).hour == 2
    assert dt.datetime.fromtimestamp(fires[-1], ROME).hour == 2


@test("cron_timezone", "spring forward: every day is still covered exactly once")
async def t_spring_no_missing_day(_ctx: TestContext) -> None:
    fires = _walk("30 2 * * *", _rome(2026, 3, 26, 12, 0), 6, "Europe/Rome")
    days = [dt.datetime.fromtimestamp(f, ROME).date() for f in fires]
    assert len(days) == len(set(days)), f"a day fired twice: {days}"
    span = (days[-1] - days[0]).days + 1
    assert span == len(days), f"a day was skipped across the gap: {days}"


# ── 4. Fall back: the repeated hour ──


@test("cron_timezone", "fall back: fixed-time cron fires once, on the first pass")
async def t_fall_back_collapses(_ctx: TestContext) -> None:
    # 2026-10-25 Europe/Rome: 02:00 happens twice, an hour apart, and
    # croniter emits both. A daily 2am job must not run twice in one night.
    fires = _walk("0 2 * * *", _rome(2026, 10, 23, 12, 0), 4, "Europe/Rome")
    on_transition = [
        f for f in fires
        if dt.datetime.fromtimestamp(f, ROME).date() == FALL_2026.date()
    ]
    assert len(on_transition) == 1, (
        "the repeated hour must collapse to a single fire; got "
        f"{[dt.datetime.fromtimestamp(f, ROME).isoformat() for f in on_transition]}"
    )
    # ...and it is the FIRST pass (still CEST, +02:00), not the second.
    first_pass = _rome(2026, 10, 25, 2, 0, fold=0)
    second_pass = _rome(2026, 10, 25, 2, 0, fold=1)
    assert first_pass != second_pass, "test setup: passes must be distinct instants"
    assert on_transition[0] == first_pass, (
        f"expected the first (CEST) pass {first_pass}, got {on_transition[0]}"
    )
    # Prove the naive behaviour we are suppressing really is a double fire.
    raw = croniter("0 2 * * *", dt.datetime.fromtimestamp(_rome(2026, 10, 25, 1, 0), ROME))
    raw_hits = [raw.get_next(dt.datetime).timestamp() for _ in range(2)]
    assert raw_hits == [first_pass, second_pass], (
        f"croniter no longer double-fires; the collapse may be dead code: {raw_hits}"
    )


@test("cron_timezone", "fall back: cadence crons keep both passes")
async def t_fall_back_cadence_preserved(_ctx: TestContext) -> None:
    # The carve-out. Collapsing by wall clock alone would take */30 dark
    # for the whole repeated hour — dropping real fires instead of
    # duplicate ones. crontab(5): jobs running more often than hourly are
    # scheduled normally.
    fires = _walk("*/30 * * * *", _rome(2026, 10, 25, 1, 0), 6, "Europe/Rome")
    gaps = {round((b - a) / 60) for a, b in zip(fires, fires[1:])}
    assert gaps == {30}, f"cadence broken across fall back, gaps(min)={sorted(gaps)}"
    assert len(fires) == len(set(fires)), "cadence fired the same instant twice"
    # An hourly job likewise sees both 02:00s — two real hours elapse.
    hourly = _walk("0 * * * *", _rome(2026, 10, 25, 1, 0), 3, "Europe/Rome")
    labels = [dt.datetime.fromtimestamp(f, ROME).strftime("%H:%M") for f in hourly]
    assert labels.count("02:00") == 2, f"hourly job lost a real hour: {labels}"


@test("cron_timezone", "fall back: only the duplicate is dropped, not a whole day")
async def t_fall_back_no_missing_day(_ctx: TestContext) -> None:
    fires = _walk("0 2 * * *", _rome(2026, 10, 23, 12, 0), 6, "Europe/Rome")
    days = [dt.datetime.fromtimestamp(f, ROME).date() for f in fires]
    assert len(days) == len(set(days)), f"a day fired twice: {days}"
    span = (days[-1] - days[0]).days + 1
    assert span == len(days), f"a day was skipped: {days}"
    for f in fires:
        assert dt.datetime.fromtimestamp(f, ROME).hour == 2


# ── 5. The 30 s tick must not double-fire or skip at a transition ──


@test("cron_timezone", "next_run always advances strictly past the tick's base")
async def t_next_run_strictly_advances(_ctx: TestContext) -> None:
    # The Scheduler writes next_run = _next_run(expr, now) right after
    # firing. If that ever returned <= now the row stays due and re-fires
    # every CHECK_INTERVAL. Walk both transitions minute-by-minute.
    from src.memory.schedule import next_run_for_expression

    for expr in ("0 2 * * *", "*/30 * * * *", "0 9 * * *", "* 2 * * *"):
        for start, label in ((SPRING_2026, "spring"), (FALL_2026, "fall")):
            base = (start - dt.timedelta(hours=2)).timestamp()
            end = (start + dt.timedelta(hours=4)).timestamp()
            while base < end:
                nxt = next_run_for_expression(expr, base, "Europe/Rome")
                assert nxt > base, (
                    f"{expr!r} at {label} returned next_run {nxt} <= base {base} "
                    "— the row would re-fire every tick"
                )
                base += 60


@test("cron_timezone", "simulated 30s tick fires a daily task once per transition day")
async def t_tick_simulation_no_double_fire(_ctx: TestContext) -> None:
    # End-to-end-ish: replay the scheduler's own loop shape (poll every
    # CHECK_INTERVAL, fire when next_run <= now, then recompute) across
    # each transition and count real firings.
    from src.core.scheduler import CHECK_INTERVAL
    from src.memory.schedule import next_run_for_expression

    for start in (SPRING_2026, FALL_2026):
        now = (start - dt.timedelta(hours=6)).timestamp()
        end = (start + dt.timedelta(hours=6)).timestamp()
        next_run = next_run_for_expression("0 2 * * *", now, "Europe/Rome")
        fired: list[float] = []
        while now < end:
            if next_run <= now:
                fired.append(next_run)
                next_run = next_run_for_expression("0 2 * * *", now, "Europe/Rome")
            now += CHECK_INTERVAL
        on_day = [
            f for f in fired
            if dt.datetime.fromtimestamp(f, ROME).date() == start.date()
        ]
        assert len(on_day) == 1, (
            f"daily 02:00 fired {len(on_day)}x on {start.date()}: "
            f"{[dt.datetime.fromtimestamp(f, ROME).isoformat() for f in on_day]}"
        )


# ── 6. @once: must not be re-interpreted ──


@test("cron_timezone", "@once: epoch is identical with and without a timezone")
async def t_one_shot_untouched(_ctx: TestContext) -> None:
    from src.memory.schedule import (
        build_one_shot_expression,
        next_run_for_expression,
    )

    # Including instants inside both transitions, where a re-interpretation
    # would be most tempting and most wrong.
    for epoch in (
        _rome(2026, 3, 29, 1, 30),
        _rome(2026, 10, 25, 2, 0, fold=1),
        1774746000.0,
        time.time() + 3600,
    ):
        expr = build_one_shot_expression(epoch)
        bare = next_run_for_expression(expr)
        for tz in (None, "Europe/Rome", "UTC", "America/New_York", "Pacific/Kiritimati"):
            got = next_run_for_expression(expr, None, tz)
            assert got == epoch == bare, (
                f"{expr!r} re-interpreted under tz={tz!r}: {got} != {epoch}"
            )
    # An absolute instant stays absolute even when the zone is nonsense —
    # it is never parsed, so it cannot fail.
    expr = build_one_shot_expression(1774746000.0)
    assert next_run_for_expression(expr, None, "Europe/Roma") == 1774746000.0


@test("cron_timezone", "one-shot rows keep their epoch when tz-tagged in the DB")
async def t_one_shot_db_roundtrip(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.memory.schedule import build_one_shot_expression, decorate_scheduled_task

    run_at = _rome(2026, 10, 25, 2, 0, fold=1)
    tmp = ctx.db_path.with_name(f"tz-once-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(tmp))
    await db.connect()
    try:
        tid = await db.add_task(
            name=f"once-{uuid.uuid4().hex[:6]}",
            cron_expression=build_one_shot_expression(run_at),
            prompt="p",
            next_run=run_at,
            timezone="Europe/Rome",
        )
        row = decorate_scheduled_task(await db.get_task(tid))
        assert row["run_once"] is True
        assert row["run_at"] == run_at, "one-shot instant moved through the DB"
        # The zone is display-only here: it renders the ISO mirror, it does
        # not re-time the firing.
        assert row["run_at_iso"].startswith("2026-10-25T02:00:00")
        assert row["timezone"] == "Europe/Rome"
    finally:
        await db.close()
        tmp.unlink(missing_ok=True)


# ── 7. A bad zone fails loudly, early, and never falls back ──


@test("cron_timezone", "bad timezone raises everywhere instead of silently degrading")
async def t_bad_tz_is_loud(_ctx: TestContext) -> None:
    from src.memory.schedule import (
        next_run_for_expression,
        validate_schedule_expression,
        validate_timezone,
    )

    bad = ["Europe/Roma", "Mars/Olympus", "UTC+2", "Rome", "europe/rome ok"]
    for name in bad:
        for label, call in (
            ("validate_timezone", lambda n=name: validate_timezone(n)),
            ("validate_schedule_expression",
             lambda n=name: validate_schedule_expression("0 9 * * *", n)),
            ("next_run_for_expression",
             lambda n=name: next_run_for_expression("0 9 * * *", time.time(), n)),
        ):
            try:
                call()
            except ValueError as exc:
                # Actionable: name the input and point at the fix.
                assert name in str(exc), f"{label}: error omits the bad name: {exc}"
                assert "IANA" in str(exc) or "Unknown timezone" in str(exc), (
                    f"{label}: error is not actionable: {exc}"
                )
            else:
                raise AssertionError(
                    f"{label} accepted bogus timezone {name!r} — a silent "
                    "fallback to the host clock is the bug we are fixing"
                )


@test("cron_timezone", "DB layer rejects a bad timezone on write")
async def t_bad_tz_rejected_by_db(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp = ctx.db_path.with_name(f"tz-bad-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(tmp))
    await db.connect()
    try:
        try:
            await db.add_task(
                name="bad", cron_expression="0 9 * * *", prompt="p",
                timezone="Europe/Roma",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("add_task accepted a bogus timezone")
        # A good row must not be updatable into a bad state either — that
        # would wedge the scheduler tick on every recompute.
        tid = await db.add_task(
            name="good", cron_expression="0 9 * * *", prompt="p",
            timezone="Europe/Rome",
        )
        try:
            await db.update_task(tid, timezone="Mars/Olympus")
        except ValueError:
            pass
        else:
            raise AssertionError("update_task accepted a bogus timezone")
        assert (await db.get_task(tid))["timezone"] == "Europe/Rome", (
            "a rejected update must leave the stored zone intact"
        )
    finally:
        await db.close()
        tmp.unlink(missing_ok=True)


# ── 8. Persistence, migration, and the scheduler reading the column ──


@test("cron_timezone", "timezone column round-trips and defaults to NULL")
async def t_db_roundtrip(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    tmp = ctx.db_path.with_name(f"tz-rt-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(tmp))
    await db.connect()
    try:
        tagged = await db.add_task(
            name=f"tz-{uuid.uuid4().hex[:6]}", cron_expression="0 9 * * *",
            prompt="p", timezone="Europe/Rome",
        )
        plain = await db.add_task(
            name=f"plain-{uuid.uuid4().hex[:6]}", cron_expression="0 9 * * *",
            prompt="p",
        )
        assert (await db.get_task(tagged))["timezone"] == "Europe/Rome"
        assert (await db.get_task(plain))["timezone"] is None, (
            "a task created without a zone must stay on the host clock"
        )
        await db.update_task(tagged, timezone="America/New_York")
        assert (await db.get_task(tagged))["timezone"] == "America/New_York"
        await db.update_task(tagged, timezone="")
        assert (await db.get_task(tagged))["timezone"] is None, (
            "empty string must clear the zone back to the host clock"
        )
    finally:
        await db.close()
        tmp.unlink(missing_ok=True)


@test("cron_timezone", "migration adds the column to an old DB without backfilling")
async def t_migration_no_backfill(ctx: TestContext) -> None:
    import aiosqlite

    from src.memory.db import MemoryDB

    tmp = ctx.db_path.with_name(f"tz-mig-{uuid.uuid4().hex[:8]}.db")
    # Build a pre-timezone scheduled_tasks table by hand and put a row in
    # it, exactly like a deployment upgrading into this change.
    async with aiosqlite.connect(str(tmp)) as conn:
        await conn.execute(
            "CREATE TABLE scheduled_tasks (id TEXT PRIMARY KEY, name TEXT NOT NULL, "
            "cron_expression TEXT NOT NULL, prompt TEXT NOT NULL, "
            "enabled INTEGER NOT NULL DEFAULT 1, last_run REAL, next_run REAL, "
            "model TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL)"
        )
        await conn.execute(
            "INSERT INTO scheduled_tasks (id, name, cron_expression, prompt, "
            "enabled, next_run, created_at, updated_at) VALUES "
            "('old-1', 'legacy', '23 11 * * 1-5', 'briefing', 1, 0, 0, 0)"
        )
        await conn.commit()

    db = MemoryDB(str(tmp))
    await db.connect()
    try:
        row = await db.get_task("old-1")
        assert row is not None, "migration lost the pre-existing row"
        assert "timezone" in row, "migration did not add the timezone column"
        assert row["timezone"] is None, (
            "the migration backfilled a zone — every hand-converted cron in "
            "the field would shift the moment this shipped"
        )
        assert row["cron_expression"] == "23 11 * * 1-5", "the expression was rewritten"
        # Idempotent: connecting again must not fail on a duplicate column.
        await db.close()
        db2 = MemoryDB(str(tmp))
        await db2.connect()
        await db2.close()
    finally:
        try:
            await db.close()
        except Exception:  # noqa: BLE001 — already closed on the happy path
            pass
        tmp.unlink(missing_ok=True)


@test("cron_timezone", "Scheduler._next_run reads the task's zone")
async def t_scheduler_uses_row_tz(_ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler
    from src.memory.schedule import next_run_for_expression

    sched = Scheduler.__new__(Scheduler)  # no DB/agent needed for the calc
    base = _rome(2026, 7, 15, 0, 0)
    assert Scheduler._task_tz({"timezone": "Europe/Rome"}) == "Europe/Rome"
    assert Scheduler._task_tz({"timezone": None}) is None
    assert Scheduler._task_tz({}) is None, "a pre-migration row must still schedule"

    tagged = sched._next_run("0 9 * * *", base, "Europe/Rome")
    assert dt.datetime.fromtimestamp(tagged, ROME).hour == 9
    # No zone → the untouched legacy path.
    assert sched._next_run("0 9 * * *", base) == _legacy_next_run("0 9 * * *", base)
    assert sched._next_run("0 9 * * *", base, None) == next_run_for_expression(
        "0 9 * * *", base
    )


# ── 9. Surfacing: a zone nobody can read is a zone nobody can trust ──


@test("cron_timezone", "decorate/ISO render on the task's own clock")
async def t_surfacing(_ctx: TestContext) -> None:
    from src.memory.schedule import decorate_scheduled_task, epoch_to_iso

    epoch = _rome(2026, 7, 15, 9, 0)
    # Default rendering stays host-local and naive (workflow surfaces rely
    # on this shape).
    assert epoch_to_iso(epoch) == dt.datetime.fromtimestamp(epoch).isoformat(
        timespec="seconds"
    )
    # Tagged rendering shows the operator their own clock, with the offset.
    assert epoch_to_iso(epoch, "Europe/Rome") == "2026-07-15T09:00:00+02:00"

    row = decorate_scheduled_task({
        "id": "x", "name": "n", "cron_expression": "0 9 * * *", "prompt": "p",
        "enabled": 1, "next_run": epoch, "timezone": "Europe/Rome",
        "created_at": epoch, "updated_at": epoch,
    })
    assert row["timezone"] == "Europe/Rome"
    assert row["next_run_iso"] == "2026-07-15T09:00:00+02:00", row["next_run_iso"]
    # A row with no zone keeps the legacy naive mirror.
    plain = decorate_scheduled_task({
        "id": "x", "name": "n", "cron_expression": "0 9 * * *", "prompt": "p",
        "enabled": 1, "next_run": epoch, "created_at": epoch, "updated_at": epoch,
    })
    assert plain["timezone"] is None
    assert plain["next_run_iso"] == dt.datetime.fromtimestamp(epoch).isoformat(
        timespec="seconds"
    )


@test("cron_timezone", "scheduler MCP: create honours tz; one-shot ISO reading is opt-in")
async def t_mcp_surface(ctx: TestContext) -> None:
    # The agent's own write surface. Exercised in-process against a
    # throwaway DB, mirroring test_run_cancellation's pattern (env swap +
    # _reset_mcp_conn) so the module-global connection can't leak.
    import os

    import src.mcp.servers.scheduler.server as mcp_server
    from src.memory.schedule import DEFAULT_TZ_ENV

    def _call(tool):
        return getattr(tool, "fn", tool)

    tmp = ctx.db_path.with_name(f"tz-mcp-{uuid.uuid4().hex[:8]}.db")
    prev_db = os.environ.get("OPENAGENT_DB_PATH")
    prev_tz = os.environ.get(DEFAULT_TZ_ENV)
    try:
        os.environ["OPENAGENT_DB_PATH"] = str(tmp)
        os.environ.pop(DEFAULT_TZ_ENV, None)
        conn = getattr(mcp_server, "_conn", None)
        mcp_server._conn = None
        if conn is not None:
            try:
                await conn.close()
            except Exception:  # noqa: BLE001
                pass

        # describe_cron previously raised NameError (croniter was never
        # imported in that module) — the agent's "check before you create"
        # tool had never once worked.
        desc = await _call(mcp_server.describe_cron)("0 9 * * *", 2, "Europe/Rome")
        assert desc["timezone"] == "Europe/Rome"
        assert len(desc["upcoming"]) == 2
        for u in desc["upcoming"]:
            assert u["iso"].startswith("2026-") or "T09:00:00+0" in u["iso"], u["iso"]
            assert "T09:00:00+0" in u["iso"], f"preview off the named clock: {u['iso']}"

        # Explicit zone is stamped and drives next_run.
        made = await _call(mcp_server.create_scheduled_task)(
            f"tz-{uuid.uuid4().hex[:6]}", "0 9 * * *", "brief", None, "Europe/Rome",
        )
        assert made["timezone"] == "Europe/Rome"
        assert dt.datetime.fromtimestamp(made["next_run"], ROME).hour == 9

        # No zone anywhere → NULL → the untouched UTC default.
        plain = await _call(mcp_server.create_scheduled_task)(
            f"plain-{uuid.uuid4().hex[:6]}", "0 9 * * *", "brief",
        )
        assert plain["timezone"] is None
        assert dt.datetime.fromtimestamp(plain["next_run"], UTC).hour == 9, (
            "a task created with no zone must still fire on UTC"
        )

        # A hallucinated zone must come back as a correctable tool error.
        try:
            await _call(mcp_server.create_scheduled_task)(
                "bad", "0 9 * * *", "x", None, "Europe/Roma",
            )
        except ValueError as exc:
            assert "Europe/Roma" in str(exc)
        else:
            raise AssertionError("the MCP accepted a bogus timezone")

        # One-shot: a bare ISO with no zone keeps the old host-local
        # reading; naming a zone reads it there instead. Either way the
        # stored instant is absolute.
        future = dt.datetime.now() + dt.timedelta(days=2)
        bare = future.replace(microsecond=0).isoformat()
        legacy_epoch = dt.datetime.fromisoformat(bare).timestamp()
        one = await _call(mcp_server.create_one_shot_task)(
            f"os-{uuid.uuid4().hex[:6]}", "x", None, bare,
        )
        assert one["run_at"] == legacy_epoch, (
            "a bare run_at_iso must keep its pre-timezone (host-local) reading"
        )
        tagged = await _call(mcp_server.create_one_shot_task)(
            f"os2-{uuid.uuid4().hex[:6]}", "x", None, bare, None, "Asia/Tokyo",
        )
        assert tagged["run_at"] == future.replace(
            microsecond=0, tzinfo=ZoneInfo("Asia/Tokyo")
        ).timestamp(), "a named zone must be used to read a bare run_at_iso"
        # An explicit offset in the string wins over the argument.
        aware = future.replace(microsecond=0, tzinfo=UTC).isoformat()
        off = await _call(mcp_server.create_one_shot_task)(
            f"os3-{uuid.uuid4().hex[:6]}", "x", None, aware, None, "Asia/Tokyo",
        )
        assert off["run_at"] == future.replace(
            microsecond=0, tzinfo=UTC
        ).timestamp(), "an explicit offset must not be overridden by the tz argument"
    finally:
        conn = getattr(mcp_server, "_conn", None)
        mcp_server._conn = None
        if conn is not None:
            try:
                await conn.close()
            except Exception:  # noqa: BLE001
                pass
        if prev_db is None:
            os.environ.pop("OPENAGENT_DB_PATH", None)
        else:
            os.environ["OPENAGENT_DB_PATH"] = prev_db
        if prev_tz is None:
            os.environ.pop(DEFAULT_TZ_ENV, None)
        else:
            os.environ[DEFAULT_TZ_ENV] = prev_tz
        tmp.unlink(missing_ok=True)


@test("cron_timezone", "agent-wide default applies to new tasks only, via env")
async def t_default_tz_env(_ctx: TestContext) -> None:
    import os

    from src.memory.schedule import DEFAULT_TZ_ENV, default_timezone_name

    prior = os.environ.get(DEFAULT_TZ_ENV)
    try:
        os.environ.pop(DEFAULT_TZ_ENV, None)
        assert default_timezone_name() is None, "unset default must mean host clock"
        os.environ[DEFAULT_TZ_ENV] = "Europe/Rome"
        assert default_timezone_name() == "Europe/Rome"
        os.environ[DEFAULT_TZ_ENV] = "   "
        assert default_timezone_name() is None, "blank default must mean host clock"
        # A typo in the agent-wide default must be loud, not a silent
        # host-clock fallback across every task the agent creates.
        os.environ[DEFAULT_TZ_ENV] = "Europe/Roma"
        try:
            default_timezone_name()
        except ValueError as exc:
            assert "Europe/Roma" in str(exc)
        else:
            raise AssertionError("a bogus agent-wide default was accepted")
    finally:
        os.environ.pop(DEFAULT_TZ_ENV, None)
        if prior is not None:
            os.environ[DEFAULT_TZ_ENV] = prior


@test("cron_timezone", "syncing an EXISTING task updates its timezone (prod bug)")
async def t_sync_updates_existing_timezone(ctx: TestContext) -> None:
    """The bug that made ``dream_mode.timezone`` do nothing on a live agent.

    ``_sync_scheduled_task`` passed ``timezone`` only to ``add_task`` (create).
    Every built-in is seeded DISABLED on first boot, so by the time an operator
    sets ``dream_mode.enabled: true`` + ``timezone: Europe/Rome`` the row
    already exists — and the update path only ever synced cron + prompt, never
    the timezone. Result: ``timezone`` stayed NULL, "3:00" fired at 03:00 UTC
    (05:00 Rome in summer), and setting the config key changed nothing. Both
    production agents had to have the column patched by hand.

    Drives the real ``AgentServer._sync_scheduled_task`` against a fake db +
    scheduler, so it pins the method's logic, not a reimplementation.
    """
    from src.core.server import AgentServer

    # Fake db holding one existing, enabled, timezone-less task.
    class _Db:
        def __init__(self):
            self.task = {
                "id": "t1", "name": "dream-mode",
                "cron_expression": "0 3 * * *", "prompt": "P",
                "enabled": 1, "timezone": None,
            }
            self.updates = []

        async def get_tasks(self):
            return [dict(self.task)]

        async def update_task(self, task_id, **kw):
            self.updates.append(kw)
            self.task.update(kw)

    class _Sched:
        def __init__(self):
            self.rescheduled = []

        async def add_task(self, **kw):
            return "new"

        async def enable_task(self, tid):
            pass

        async def disable_task(self, tid):
            pass

        async def reschedule_task(self, tid, **kw):
            self.rescheduled.append(tid)

    server = AgentServer.__new__(AgentServer)
    server.agent = type("A", (), {})()
    server.agent._db = _Db()
    sched = _Sched()

    await server._sync_scheduled_task(
        sched, name="dream-mode", enabled=True,
        cron_expr="0 3 * * *", prompt="P", timezone="Europe/Rome",
    )

    tz_updates = [u for u in server.agent._db.updates if "timezone" in u]
    assert tz_updates and tz_updates[0]["timezone"] == "Europe/Rome", (
        "an existing task's timezone was not synced from config — setting "
        f"dream_mode.timezone did nothing. updates: {server.agent._db.updates}"
    )
    assert sched.rescheduled == ["t1"], (
        "the timezone changed but next_run was not recomputed — it would keep "
        "firing at the old (UTC) instant until the next cron change."
    )


@test("cron_timezone", "syncing does NOT rewrite a timezone that already matches")
async def t_sync_timezone_idempotent(ctx: TestContext) -> None:
    """No spurious update/reschedule when the config already matches the DB —
    a needless reschedule on every boot would move next_run around for nothing.
    """
    from src.core.server import AgentServer

    class _Db:
        def __init__(self):
            self.task = {
                "id": "t1", "name": "dream-mode", "cron_expression": "0 3 * * *",
                "prompt": "P", "enabled": 1, "timezone": "Europe/Rome",
            }
            self.updates = []

        async def get_tasks(self):
            return [dict(self.task)]

        async def update_task(self, task_id, **kw):
            self.updates.append(kw)

    class _Sched:
        def __init__(self):
            self.rescheduled = []

        async def enable_task(self, tid):
            pass

        async def disable_task(self, tid):
            pass

        async def reschedule_task(self, tid, **kw):
            self.rescheduled.append(tid)

    server = AgentServer.__new__(AgentServer)
    server.agent = type("A", (), {})()
    server.agent._db = _Db()
    sched = _Sched()

    await server._sync_scheduled_task(
        sched, name="dream-mode", enabled=True,
        cron_expr="0 3 * * *", prompt="P", timezone="Europe/Rome",
    )
    assert not server.agent._db.updates, f"spurious update: {server.agent._db.updates}"
    assert not sched.rescheduled, "rescheduled with nothing changed"
