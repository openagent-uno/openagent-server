"""Async handlers behind the ``logs`` tools.

READ-ONLY, DELIBERATELY
-----------------------
``src.core.logging`` also exposes ``clear(older_than_days)``, and the REST
surface wires it to ``DELETE /api/logs``. It is **not** exposed here:

* §14 gives the agent one job with the log — *diagnose*. Nothing in that job
  requires destroying evidence.
* Retention already has an owner. Dream mode prunes to ~6 days on a schedule;
  a second, unscheduled pruner reachable from any turn would fight it.
* This log is the substrate for the follow-up outcome-scoring work. A tool
  that can erase the record of a bad run, held by the same agent whose runs
  are being scored, is a conflict of interest baked into the tool surface.
  That it would rarely be *misused* is not the point — it should not be
  reachable by accident either (a model that reads "clear" while hunting for
  a way to reduce noise is one mis-step from an unrecoverable one).

A human keeps ``DELETE /api/logs``, and dream mode keeps its scheduled prune.
Both are the right holders of a destructive, irreversible operation.

TOKEN BUDGET
------------
There IS a global backstop — ``src.core.tool_output.cap_tool_output``, applied
in ``Model.create_function_call_result``, the one point every provider funnels
through. A dict return does not evade it: the runtime ``str()``s any
non-``ToolResult`` result first (``models/providers/base.py:2254``), so it
arrives as a string and gets truncated at 50k chars.

That backstop is a ceiling, not a budget, and it is the wrong tool for this
job on two counts. First, 50k chars is ~12.5k tokens — more than the entire
framework prompt (~11.8k/run) — and ``tool_output``'s own docstring is
emphatic that a tool result is re-sent on *every* following step, so an
uncapped log dump is paid over and over. Second, it truncates the *repr*:
``str(dict)`` cut at 50k hands the model a guillotined pseudo-dict that is
neither valid JSON nor valid Python. A log query must return a well-formed
object that says what it dropped and how to ask again.

So we cap here, at four independent levels, and let the global guard be the
last resort it was built to be:

1. scan bound   — reader stops at ``_MAX_SCAN_BYTES`` of tail
2. row bound    — ``limit`` clamped to ``_MAX_LIMIT``
3. value bound  — every field clamped by ``reader.truncate_value``
4. payload bound— total serialised size clamped by :func:`_fit_budget`

The framework prompt already costs ~11.8k tokens/run; a diagnosis tool that
adds 40k on top would make the agent unable to reason about the very failure
it just fetched.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from src.mcp.servers.logs import reader
from src.mcp.servers.logs.reader import ScanStats

# Row caps. The default returns a page a model can actually reason about;
# the hard cap is what a determined `limit=100000` collapses to.
_DEFAULT_LIMIT = 50
_MAX_LIMIT = 200

# Whole-payload ceiling, ~6k tokens at 4 chars/token. Sized against the
# framework prompt (~11.8k tokens/run): one tool result should not cost more
# than half of what the agent's own self-description costs.
_MAX_RESULT_CHARS = 24_000

# logs_summary group caps — long tails are counted but not enumerated.
_TOP_N = 15
_MAX_TOP_N = 50

# logs_context window caps.
_DEFAULT_CONTEXT = 8
_MAX_CONTEXT = 25

# The levels `elog` can actually emit (its _LEVELS map). A `level=` query
# outside this set is a typo, and answering "0 results" to a typo is how a
# model concludes nothing went wrong.
_QUERYABLE_LEVELS = frozenset({"debug", "info", "warning", "error"})


def _clamp(value: Any, default: int, low: int, high: int) -> int:
    """Coerce a model-supplied count into range instead of raising.

    A model that asks for ``limit=1000`` wants "lots" and should get the cap
    with ``capped: true`` in the result — failing the call would cost a whole
    turn to re-ask for a number it cannot guess. Garbage (``"all"``, ``None``)
    falls back to the default for the same reason.
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, n))


def _severity_notes(from_level: int, inferred: int) -> list[str]:
    """The severity caveat, matched to what this specific log actually holds.

    Three regimes, because a log written across the formatter fix contains
    both schemas and the honest caveat differs for each:

    * all authoritative — say so, so the model trusts the number;
    * all inferred (a log written entirely before the fix) — the original
      warning, unchanged;
    * mixed (the ~6-day transition) — name the exact split, because "some of
      this is a guess" is useless without "how much".
    """
    if inferred and from_level:
        return [
            f"`error_like` mixes two schemas: {from_level} entries had a "
            f"persisted severity level (authoritative) and {inferred} predate "
            f"the level fix, so their severity was INFERRED from the event "
            f"name / error field and may over-report recovered failures.",
        ]
    if inferred:
        return [
            "`error_like` is INFERRED for every entry here (traceback / "
            "errored / error field / error-ish event name): these lines "
            "predate the severity fix, so they carry no level and the count "
            "may over-report recovered failures.",
        ]
    if from_level:
        return [
            "`error_like` was read from each entry's persisted severity level "
            "(authoritative). It counts `error` and `warning` — `mcp.error` "
            "and `mcp.timeout` are logged at warning.",
        ]
    return []


def _int_field(entry: dict[str, Any], key: str) -> int:
    """An int field's value, or 0 — ``bool`` is not a token count."""
    value = entry.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _fit_budget(rows: list[dict[str, Any]], budget: int = _MAX_RESULT_CHARS) -> tuple[list[dict[str, Any]], bool]:
    """Drop rows from the END until the payload fits ``budget`` chars.

    Rows arrive oldest→newest and the newest are the ones a diagnosis needs,
    so this trims from the far end... except the far end IS the newest here.
    We therefore measure from the newest backwards and rebuild in order, so
    what survives a squeeze is the most recent slice, not an arbitrary prefix.

    Only fires when per-value truncation was not enough (e.g. 200 rows that
    are each individually reasonable). Returns ``(rows, was_truncated)``.
    """
    total = 0
    kept: list[dict[str, Any]] = []
    for row in reversed(rows):
        size = len(json.dumps(row, default=str))
        if total + size > budget and kept:
            break
        total += size
        kept.append(row)
    kept.reverse()
    return kept, len(kept) < len(rows)


def _norm_level(value: str | None) -> str | None:
    """Validate a caller-supplied level against the ones ``elog`` can emit."""
    if value is None:
        return None
    level = str(value).strip().lower()
    if not level:
        return None
    if level not in _QUERYABLE_LEVELS:
        raise ValueError(
            f"unknown level {value!r}. Use one of {sorted(_QUERYABLE_LEVELS)}. "
            f"For a failure filter that also covers older, level-less entries, "
            f"use errors_only=true instead."
        )
    return level


def _matches(
    entry: dict[str, Any], *, event: str | None, contains: str | None,
    session_id: str | None, run_id: str | None, errors_only: bool,
    level: str | None, until: float | None,
) -> bool:
    """Apply every filter to one entry. All conditions AND together."""
    if until is not None:
        ts = entry.get("ts")
        if isinstance(ts, (int, float)) and not isinstance(ts, bool) and ts > until:
            return False

    if event:
        name = entry.get("event") or ""
        if not isinstance(name, str):
            return False
        low, q = name.lower(), event.lower()
        # Prefix OR substring. read_tail / GET /api/logs are prefix-only,
        # which fails the way a model actually types: it asks for "cost" and
        # means `runtime.cost_mirrored`. Prefix is still tried first so the
        # precise, namespaced query ("scheduler.") behaves exactly as the
        # REST filter does.
        if not (low.startswith(q) or q in low):
            return False

    if session_id and entry.get("session_id") != session_id:
        return False

    if run_id and entry.get("run_id") != run_id:
        return False

    if errors_only and not reader.classify(entry)[0]:
        return False

    if level is not None and reader.entry_level(entry) != level:
        return False

    if contains:
        try:
            blob = json.dumps(entry, default=str).lower()
        except (TypeError, ValueError):
            blob = str(entry).lower()
        if contains.lower() not in blob:
            return False

    return True


def _query_sync(
    *, event: str | None, contains: str | None, session_id: str | None,
    run_id: str | None, errors_only: bool, level: str | None, since: Any,
    until: Any, limit: Any, offset: Any,
) -> dict[str, Any]:
    since_ts = reader.parse_time(since)
    until_ts = reader.parse_time(until)
    want_level = _norm_level(level)
    n_limit = _clamp(limit, _DEFAULT_LIMIT, 1, _MAX_LIMIT)
    n_offset = _clamp(offset, 0, 0, 100_000)

    stats = ScanStats()
    matched = 0
    picked: list[dict[str, Any]] = []

    # The plain-tail case ("show me the last 50 events"): with no predicate at
    # all, every entry matches, so `matched_in_scan` past a full page would
    # just be "the size of the log" — a number the caller did not ask for,
    # bought by reading the whole file. Every *filtered* query keeps scanning,
    # because there `matched_in_scan` is the real answer to "is there more?"
    # and drives paging. Guard lists every filter explicitly: an earlier draft
    # omitted `errors_only`, which silently truncated the error count on the
    # single most important query this MCP serves.
    unfiltered = (
        since_ts is None and until_ts is None and not contains and not event
        and not session_id and not run_id and not errors_only
        and want_level is None
    )

    # A `level` query cannot judge a pre-fix entry — it has no level to match.
    # Skipping them silently would answer "no warnings yesterday" for a log
    # written before the formatter fix, which is a lie by omission. Count them
    # and say so.
    unjudgeable = 0

    for entry in reader.iter_entries_reverse(since=since_ts, stats=stats):
        if want_level is not None and reader.entry_level(entry) is None:
            unjudgeable += 1
        if not _matches(
            entry, event=event, contains=contains, session_id=session_id,
            run_id=run_id, errors_only=errors_only, level=want_level,
            until=until_ts,
        ):
            continue
        matched += 1
        if matched <= n_offset:
            continue
        if len(picked) < n_limit:
            picked.append(reader.compact_entry(entry))
        elif unfiltered:
            break

    # Scanned newest-first; hand back oldest→newest so the page reads as a
    # timeline. Mirrors read_tail's contract (it reverses before returning),
    # so a caller moving off GET /api/logs sees the same ordering.
    picked.reverse()
    rows, budget_trimmed = _fit_budget(picked)

    out: dict[str, Any] = {
        "entries": rows,
        "returned": len(rows),
        "matched_in_scan": matched,
        "window": {
            "since": reader.iso(since_ts) if since_ts else None,
            "until": reader.iso(until_ts) if until_ts else None,
        },
        "scan": stats.to_dict(),
    }
    if unjudgeable:
        out["entries_without_level_skipped"] = unjudgeable
        out["level_filter_note"] = (
            f"{unjudgeable} scanned entries predate the severity fix and carry "
            f"no level, so `level={want_level!r}` could not judge them — they "
            f"are NOT in these results. Use errors_only=true to span both "
            f"schemas."
        )
    if budget_trimmed:
        out["result_truncated"] = True
        out["hint"] = (
            f"Payload exceeded {_MAX_RESULT_CHARS} chars; oldest rows of the "
            f"page were dropped. Narrow with `event`, `session_id`, or `since`."
        )
    elif matched > len(rows) + n_offset:
        out["hint"] = (
            f"{matched} entries matched but {len(rows)} returned. Page with "
            f"`offset`, or narrow the filter."
        )
    if stats.hit_scan_cap:
        out["hint"] = (
            (out.get("hint", "") + " ").strip()
            + " Scan stopped at the byte cap — older matches may exist beyond "
              "`scan.oldest_entry_scanned`."
        ).strip()
    return out


def _summary_sync(
    *, since: Any, until: Any, session_id: str | None, event: str | None,
    top: Any,
) -> dict[str, Any]:
    since_ts = reader.parse_time(since)
    until_ts = reader.parse_time(until)
    n_top = _clamp(top, _TOP_N, 1, _MAX_TOP_N)

    stats = ScanStats()
    total = 0
    error_like = 0
    from_level = 0
    inferred = 0
    with_level = 0
    by_level: dict[str, int] = {}
    by_event: dict[str, int] = {}
    errors_by_event: dict[str, int] = {}
    error_types: dict[str, int] = {}
    sessions: dict[str, int] = {}
    cost_usd = 0.0
    cost_calls = 0
    tokens_in = 0
    tokens_out = 0
    newest_ts: float | None = None
    oldest_ts: float | None = None
    recent_errors: list[dict[str, Any]] = []

    for entry in reader.iter_entries_reverse(since=since_ts, stats=stats):
        if not _matches(
            entry, event=event, contains=None, session_id=session_id,
            run_id=None, errors_only=False, level=None, until=until_ts,
        ):
            continue
        total += 1

        level = reader.entry_level(entry)
        if level is not None:
            with_level += 1
            by_level[level] = by_level.get(level, 0) + 1

        ts = entry.get("ts")
        if isinstance(ts, (int, float)) and not isinstance(ts, bool):
            if newest_ts is None:
                newest_ts = float(ts)  # reverse scan: first match is newest
            oldest_ts = float(ts)

        name = entry.get("event")
        name = name if isinstance(name, str) else "<unnamed>"
        by_event[name] = by_event.get(name, 0) + 1

        sid = entry.get("session_id")
        if isinstance(sid, str) and sid:
            sessions[sid] = sessions.get(sid, 0) + 1

        if name == reader.COST_EVENT:
            spend = entry.get("cost_usd")
            if isinstance(spend, (int, float)) and not isinstance(spend, bool):
                cost_usd += float(spend)
                cost_calls += 1
            tokens_in += _int_field(entry, "input_tokens")
            tokens_out += _int_field(entry, "output_tokens")

        is_failure, was_inferred = reader.classify(entry)
        if is_failure:
            error_like += 1
            if was_inferred:
                inferred += 1
            else:
                from_level += 1
            errors_by_event[name] = errors_by_event.get(name, 0) + 1
            et = entry.get("error_type")
            if isinstance(et, str) and et:
                error_types[et] = error_types.get(et, 0) + 1
            # A count alone tells the model something broke but not what; a
            # handful of real samples usually ends the investigation without
            # a second call. Bounded hard — this is the only unbounded-ish
            # field in the summary.
            if len(recent_errors) < 5:
                recent_errors.append(reader.compact_entry(entry))

    def _top(d: dict[str, int]) -> list[dict[str, Any]]:
        items = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:n_top]
        return [{"name": k, "count": v} for k, v in items]

    span_h = None
    if newest_ts is not None and oldest_ts is not None:
        span_h = round((newest_ts - oldest_ts) / 3600.0, 2)

    recent_errors.reverse()
    out: dict[str, Any] = {
        "total_events": total,
        "window": {
            "since": reader.iso(since_ts) if since_ts else None,
            "until": reader.iso(until_ts) if until_ts else None,
            "first_event": reader.iso(oldest_ts),
            "last_event": reader.iso(newest_ts),
            "span_hours": span_h,
        },
        "error_like": {
            "count": error_like,
            "rate": round(error_like / total, 4) if total else 0.0,
            # The honesty split. `from_level` was read from a severity the
            # call site declared; `inferred` was guessed from an event name
            # because the entry predates the formatter fix. Without this the
            # count would blend authoritative and guessed severity into one
            # undifferentiated number.
            "from_level": from_level,
            "inferred": inferred,
            "by_event": _top(errors_by_event),
            "by_error_type": _top(error_types),
            "samples": recent_errors,
        },
        "levels": {
            "by_level": _top(by_level),
            "entries_with_level": with_level,
            "entries_without_level": total - with_level,
        },
        "by_event": _top(by_event),
        "distinct_events": len(by_event),
        "sessions": {
            "distinct": len(sessions),
            "busiest": _top(sessions),
        },
        "cost": {
            "total_usd": round(cost_usd, 6),
            "accounted_calls": cost_calls,
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "note": (
                "Mirrored from `runtime.cost_mirrored` events only. The "
                "canonical spend record is the `usage_log` DB table; treat "
                "this as a lower bound."
            ),
        },
        "scan": stats.to_dict(),
        # Stated in-band, every call, and tracking the ACTUAL mix rather than
        # a fixed disclaimer. A model that reads `error_like` as authoritative
        # over a pre-fix log would draw confident, wrong conclusions; one that
        # discounts it as "just a guess" over a post-fix log would ignore real
        # failures. Only the real ratio can say which log it is holding.
        "notes": _severity_notes(from_level, inferred),
    }
    if stats.hit_scan_cap:
        out["notes"].append(
            "Scan stopped at the byte cap — counts cover only the tail up to "
            "`scan.oldest_entry_scanned`, not the whole window."
        )
    return out


def _context_sync(*, ts: Any, before: Any, after: Any) -> dict[str, Any]:
    try:
        anchor = float(ts)
    except (TypeError, ValueError):
        raise ValueError(
            f"`ts` must be the numeric `ts` of an entry from logs_query "
            f"(e.g. 1783960770.0519218), got {ts!r}."
        ) from None

    n_before = _clamp(before, _DEFAULT_CONTEXT, 0, _MAX_CONTEXT)
    n_after = _clamp(after, _DEFAULT_CONTEXT, 0, _MAX_CONTEXT)

    from collections import deque

    stats = ScanStats()
    # Reverse scan reaches the NEWER entries first, so "after" fills before we
    # ever see the anchor. A bounded deque keeps the `n_after` closest to it
    # and lets everything newer fall off the far end for free.
    newer: deque[dict[str, Any]] = deque(maxlen=max(1, n_after))
    anchor_entry: dict[str, Any] | None = None
    older: list[dict[str, Any]] = []

    for entry in reader.iter_entries_reverse(stats=stats):
        e_ts = entry.get("ts")
        if not isinstance(e_ts, (int, float)) or isinstance(e_ts, bool):
            continue
        if anchor_entry is None:
            # First entry at or before the anchor IS the anchor: exact when
            # the ts came from logs_query (the normal path), nearest-older
            # when the model rounded it.
            if e_ts <= anchor:
                anchor_entry = entry
            else:
                newer.append(entry)
            continue
        # Bound-check BEFORE appending: the reverse is an off-by-one that
        # hands back one more entry than asked for, and at before=0 returns a
        # neighbour the caller explicitly declined.
        if len(older) >= n_before:
            break
        older.append(entry)

    if anchor_entry is None:
        return {
            "found": False,
            "error": (
                f"No entry at or before ts={anchor} within the scanned tail. "
                f"The log may have been pruned, or the ts is in the future."
            ),
            "scan": stats.to_dict(),
        }

    # deque filled newest→older as we scanned back toward the anchor; emit
    # chronologically. Empty when the caller asked for no trailing context.
    after_rows = [reader.compact_entry(e) for e in newer] if n_after else []
    after_rows.reverse()
    older.reverse()

    rows = (
        [reader.compact_entry(e) for e in older]
        + [reader.compact_entry(anchor_entry)]
        + after_rows
    )
    rows, trimmed = _fit_budget(rows)

    out: dict[str, Any] = {
        "found": True,
        # Returned in full and separately from `entries`: _fit_budget may drop
        # rows, so any positional index into `entries` would be a lie the
        # moment the budget bites. The anchor is what the caller asked about —
        # it is never merely implied.
        "anchor": reader.compact_entry(anchor_entry),
        "entries": rows,
        "scan": stats.to_dict(),
    }
    if trimmed:
        out["result_truncated"] = True
        out["hint"] = "Context trimmed to fit the payload budget; lower `before`/`after`."
    return out


# ── Tool entry points ───────────────────────────────────────────────
#
# Every one hops to a thread. The scan is blocking file I/O and this MCP is
# in-process — it shares the event loop with live WebSocket voice/chat
# streams, and §2 makes barge-in a first-class behaviour. A multi-hundred-KB
# read on the loop is a stall the user hears.


async def logs_query(
    event: str | None = None, contains: str | None = None,
    session_id: str | None = None, run_id: str | None = None,
    errors_only: bool = False, level: str | None = None,
    since: str | None = None, until: str | None = None,
    limit: int = _DEFAULT_LIMIT, offset: int = 0,
) -> dict:
    return await asyncio.to_thread(
        _query_sync, event=event, contains=contains, session_id=session_id,
        run_id=run_id, errors_only=errors_only, level=level, since=since,
        until=until, limit=limit, offset=offset,
    )


async def logs_summary(
    since: str | None = None, until: str | None = None,
    session_id: str | None = None, event: str | None = None,
    top: int = _TOP_N,
) -> dict:
    return await asyncio.to_thread(
        _summary_sync, since=since, until=until, session_id=session_id,
        event=event, top=top,
    )


async def logs_context(
    ts: float, before: int = _DEFAULT_CONTEXT, after: int = _DEFAULT_CONTEXT,
) -> dict:
    return await asyncio.to_thread(_context_sync, ts=ts, before=before, after=after)
