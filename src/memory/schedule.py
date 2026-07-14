"""Shared helpers for scheduled task expressions.

Timezones
─────────
A cron expression is a *wall-clock* statement — ``0 9 * * *`` means "when
the clock reads 09:00". It says nothing about *whose* clock.

Until this module grew a timezone, the answer was always **UTC**, on every
host. That is worth stating plainly because the code does not look like it:
:func:`next_run_for_expression` passes ``time.time()`` — a float — to
croniter, and croniter's ``timestamp_to_datetime`` resolves a float via
``datetime.fromtimestamp(ts, tz=utc).replace(tzinfo=None)``, i.e. it
converts to UTC and evaluates there. The machine's own timezone never
enters into it. (``epoch_to_iso`` *does* render host-local, so on a
non-UTC host a task's displayed ``next_run_iso`` hour disagrees with the
hour in its cron. That mismatch predates this module and is left alone.)

UTC is unambiguous but it is not where the operator lives. These agents run
containerised on k8s while the operator thinks in Europe/Rome, and their
notes record crons as ``cron 23 11 * * 1-5 UTC ≈ 13:23 Europe/Rome`` — the
conversion done by hand, in a comment. A hand-converted cron is correct
right up until the next DST transition, when a "9am briefing" silently
becomes 8am or 10am: no code change, no error, no log line. Vision §7 calls
a scheduled task "chat with the user's seat empty"; a human would not accept
their 9am meeting drifting an hour in March.

So a schedule may now carry an IANA timezone (``scheduled_tasks.timezone``).

**The default is unchanged UTC, and that is load-bearing.** ``tz=None``
takes a path that is character-for-character the pre-timezone
implementation. Every deployment out there holds crons the operator already
hand-converted to UTC; re-reading those under some new default zone would
shift *every existing cron on every production agent at once*. So ``NULL``
keeps the old meaning, the column is never backfilled, and the agent-wide
default (:func:`default_timezone_name`) is materialised into a row at
*creation* time rather than applied to NULL rows at read time — changing
that default never moves a task that already exists. Moving one is an
explicit edit.

Passing ``timezone='UTC'`` is exactly equivalent to ``NULL`` (UTC has no DST,
so the collapse below is a no-op there); the test suite pins that equality.
An operator can therefore make the implicit explicit, task by task, with no
change in firing instant, and only then re-aim a task at a real zone.

DST semantics (only relevant when a timezone is set)
────────────────────────────────────────────────────
Twice a year a wall-clock reading is either missing or duplicated, and a
naive implementation skips a day or fires twice. We follow the rule
operators already know from ``crontab(5)`` (Vixie cron):

*Spring forward* — the clock jumps 02:00 → 03:00 and ``0 2 * * *`` never
matches. croniter already does the right thing here: it yields
``03:00+02:00``, which is the *same absolute instant* 02:00 would have
been (01:00 UTC). The job fires once, at the moment the skipped slot
would have elapsed. It does not lose a day.

*Fall back* — the clock repeats 02:00, so ``0 2 * * *`` matches twice, an
hour apart, and croniter yields **both**. A daily 2am job firing twice on
one night is the bug; :func:`_collapses_repeated_hour` marks fixed-time
schedules and we drop the second pass. Cadence schedules (``*/30 * * * *``,
``0 * * * *``, ``* 2 * * *``) keep both passes — an hourly job *should*
fire twice when two real hours elapse — matching Vixie's "jobs that run
more frequently [than hourly] are scheduled normally".

``@once:<epoch>`` is an absolute instant and is never re-interpreted: it
short-circuits before any timezone logic. A one-shot may still *carry* a
timezone, but it only affects how the ISO mirror is rendered.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from croniter import croniter

ONE_SHOT_PREFIX = "@once:"

# Agent-wide default timezone for *newly created* schedules. Read from the
# environment because the scheduler MCP runs as its own subprocess and has
# to resolve the same default as the in-process gateway — the same reason
# ``OPENAGENT_DB_PATH`` is plumbed this way. The yaml key that should feed
# this (``scheduler.timezone``) lives in ``core/server.py``.
DEFAULT_TZ_ENV = "OPENAGENT_SCHEDULER_TZ"

# Upper bound on candidates discarded while collapsing a repeated hour.
# A collapsing schedule fires at most once per hour by construction, so a
# real transition discards exactly one candidate; the margin only exists so
# a pathological zone can't spin the loop forever.
_DST_SCAN_LIMIT = 64


def is_one_shot_expression(expr: str | None) -> bool:
    return bool(expr and str(expr).startswith(ONE_SHOT_PREFIX))


def build_one_shot_expression(run_at: float) -> str:
    return f"{ONE_SHOT_PREFIX}{float(run_at)}"


def parse_one_shot_expression(expr: str) -> float:
    if not is_one_shot_expression(expr):
        raise ValueError(f"Not a one-shot schedule expression: {expr!r}")
    try:
        return float(str(expr)[len(ONE_SHOT_PREFIX):])
    except ValueError as exc:
        raise ValueError(f"Invalid one-shot schedule expression: {expr!r}") from exc


# ── Timezones ──


def validate_timezone(name: str | None) -> None:
    """Raise ``ValueError`` unless ``name`` is a loadable IANA timezone.

    Falsy → valid (it means "the UTC default").

    A wrong timezone must fail here, at the write, rather than degrade to
    the default at fire time: a silent fallback is indistinguishable from
    the hand-converted-UTC bug this whole module exists to kill, and would
    only ever be noticed as a briefing arriving at the wrong hour.
    """
    resolve_timezone(name)


def resolve_timezone(name: str | None) -> ZoneInfo | None:
    """``name`` → ``ZoneInfo``; ``None``/empty → ``None`` (the UTC default)."""
    if name is None:
        return None
    if isinstance(name, ZoneInfo):
        return name
    text = str(name).strip()
    if not text:
        return None
    try:
        return ZoneInfo(text)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f"Unknown timezone {text!r}: {exc}. Use an IANA timezone name "
            "such as 'Europe/Rome', 'America/New_York' or 'UTC' "
            "(see zoneinfo.available_timezones())."
        ) from exc
    except (TypeError, OSError) as exc:  # unhashable / unreadable tzdata
        raise ValueError(f"Unknown timezone {text!r}: {exc}") from exc


def default_timezone_name() -> str | None:
    """Agent-wide default timezone for new schedules, or ``None``.

    Applied by the *write* surfaces when a caller doesn't name a timezone,
    and materialised into the row. Deliberately not consulted at fire time:
    see the module docstring — resolving NULL rows through this would shift
    every hand-converted cron the moment an operator set it.
    """
    raw = os.environ.get(DEFAULT_TZ_ENV, "").strip()
    if not raw:
        return None
    validate_timezone(raw)
    return raw


def _collapses_repeated_hour(expr: str) -> bool:
    """Is ``expr`` a fixed-time schedule (vs. a cadence)?

    Fixed-time means "at most one fire per hour, at named hours" — one
    minute value and a non-wildcard hour (``0 2 * * *``, ``30 9 * * 1-5``,
    ``0 */2 * * *``). Those are appointments: they get the repeated hour
    collapsed to a single fire.

    Everything else is a cadence and is left alone, because collapsing it
    would eat real fires rather than duplicate ones — ``*/30 * * * *``
    would go dark for the whole repeated hour, and ``* 2 * * *`` would
    lose 60 fires. This is the ``crontab(5)`` split between jobs with
    "granularity greater than one hour" and jobs that "run more
    frequently"; ``0 * * * *`` is hourly, so it is a cadence and fires
    twice, which is correct — two real hours elapse.
    """
    try:
        expanded = croniter(expr).expanded
    except Exception:  # noqa: BLE001 — validity is the caller's problem
        return False
    if len(expanded) < 2:
        return False
    minutes, hours = expanded[0], expanded[1]
    minute_fixed = minutes != ["*"] and len(minutes) == 1
    hour_named = hours != ["*"]
    return bool(minute_fixed and hour_named)


def _is_repeated_wall_clock(moment: dt.datetime, zone: ZoneInfo) -> bool:
    """Is ``moment`` the *second* pass of a duplicated wall-clock reading?

    On the night the clock falls back, 02:30 happens twice — once at
    +02:00 and again at +01:00 — and croniter emits both. Re-resolving the
    naive reading with ``fold=0`` yields whichever pass is first; if the
    candidate's offset disagrees, the candidate is the later one and the
    schedule already fired for this reading.
    """
    naive = moment.replace(tzinfo=None)
    first_pass = naive.replace(tzinfo=zone, fold=0)
    return first_pass.utcoffset() != moment.utcoffset()


def validate_schedule_expression(expr: str, timezone: str | None = None) -> None:
    """Validate a schedule expression, and its timezone when one is given."""
    validate_timezone(timezone)
    if is_one_shot_expression(expr):
        parse_one_shot_expression(expr)
        return
    try:
        croniter(expr)
    except (ValueError, KeyError) as exc:
        raise ValueError(f"Invalid cron expression {expr!r}: {exc}") from exc


def next_run_for_expression(
    expr: str,
    base: float | None = None,
    timezone: str | None = None,
) -> float:
    """Absolute epoch of the next fire of ``expr`` after ``base``.

    Returns an epoch in every case, timezone or not, so the scheduler's
    ``next_run <= now`` tick keeps comparing two UTC instants — a wall
    clock may jump, the epoch axis never does.
    """
    if is_one_shot_expression(expr):
        # An absolute instant. A timezone must not "helpfully" re-read it.
        return parse_one_shot_expression(expr)
    zone = resolve_timezone(timezone)
    if zone is None:
        # Byte-for-byte the pre-timezone path. Every existing deployment
        # lands here; it must not drift by so much as a rounding step.
        try:
            return croniter(expr, time.time() if base is None else base).get_next(float)
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Invalid cron expression: {exc}") from exc
    return _next_run_in_zone(expr, base, zone)


def _next_run_in_zone(expr: str, base: float | None, zone: ZoneInfo) -> float:
    base_epoch = time.time() if base is None else base
    try:
        start = dt.datetime.fromtimestamp(base_epoch, zone)
        iterator = croniter(expr, start)
    except (ValueError, KeyError, OverflowError, OSError) as exc:
        raise ValueError(f"Invalid cron expression: {exc}") from exc

    collapse = _collapses_repeated_hour(expr)
    for _ in range(_DST_SCAN_LIMIT):
        try:
            candidate = iterator.get_next(dt.datetime)
        except (ValueError, KeyError, OverflowError) as exc:
            raise ValueError(f"Invalid cron expression: {exc}") from exc
        if collapse and _is_repeated_wall_clock(candidate, zone):
            continue
        # ``.timestamp()`` on an aware datetime is exact — including inside
        # a fold, where the offset croniter attached picks the pass.
        return candidate.timestamp()
    raise ValueError(
        f"Could not resolve a next run for {expr!r} in {zone.key!r} within "
        f"{_DST_SCAN_LIMIT} candidates"
    )


def epoch_to_iso(epoch: float, timezone: str | None = None) -> str:
    """Render ``epoch`` as ISO-8601.

    Default (no timezone) stays host-local and naive, as every existing
    caller — including the workflow surfaces — expects. With a timezone,
    the mirror is rendered *in that zone* and carries its offset, so a
    Rome task reads ``09:00+02:00`` instead of the UTC hour the operator
    would have to convert in their head all over again.
    """
    zone = resolve_timezone(timezone)
    if zone is None:
        return dt.datetime.fromtimestamp(epoch).isoformat(timespec="seconds")
    return dt.datetime.fromtimestamp(epoch, zone).isoformat(timespec="seconds")


def decorate_scheduled_task(row: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    task = dict(row)
    task["enabled"] = bool(task.get("enabled"))
    task["run_once"] = is_one_shot_expression(task.get("cron_expression"))
    # Surface the zone even on rows that predate the column, so a reader
    # can tell "defaulted" from "deliberately pinned" instead of guessing.
    zone_name = task.get("timezone") or None
    task["timezone"] = zone_name
    if task["run_once"]:
        run_at = parse_one_shot_expression(task["cron_expression"])
        task["run_at"] = run_at
        task["run_at_iso"] = epoch_to_iso(run_at, zone_name)
    for ts_col in ("last_run", "next_run", "created_at", "updated_at"):
        value = task.get(ts_col)
        if isinstance(value, (int, float)):
            task[f"{ts_col}_iso"] = epoch_to_iso(value, zone_name)
    return task
