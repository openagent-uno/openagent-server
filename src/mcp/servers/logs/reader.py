"""Bounded reverse reader + predicates over ``events.jsonl``.

WHY THIS MODULE EXISTS (and does not live in ``src/core/logging.py``)
--------------------------------------------------------------------
``src.core.logging`` owns the write path and exposes exactly two read
primitives: ``read_tail(lines, event_filter)`` and ``clear(older_than_days)``.
``read_tail`` is structurally unusable for this MCP for two reasons:

1. **It slurps.** ``read_text().splitlines()`` materialises the *entire* file
   plus a list of every line before it filters anything. events.jsonl is
   trimmed to ~6 days by dream mode, not to a size — a chatty agent's log is
   unbounded within that window (measured 728 KB / 5365 lines on a light
   install). This MCP runs **in-process**, inside the same event loop that
   serves live WebSocket streams, so a multi-MB blocking slurp is a stall the
   user hears in a voice turn.
2. **Prefix-only.** It filters on ``event`` prefix and nothing else. Every
   question §14 poses needs a time window, and two of them need severity or a
   correlation id.

So we scan **backwards in fixed blocks from EOF** and stop as soon as the
caller's window is satisfied — the log is append-only and every question is
"recent first", so the newest bytes are the only ones worth touching.

The right long-term home for this is ``logging.py`` itself (a generator
primitive shared with ``src/gateway/api/logs.py``, which today calls the same
slurping ``read_tail``). That file is out of scope for this change; see the
handover notes rather than duplicating the reader there later.

A MIXED-SCHEMA LOG (the important one)
--------------------------------------
``_JsonlFormatter`` used to drop ``record.levelname`` — every ``elog`` call
passed a level, and none of it reached the file (measured: 0 of 5365 entries).
That is fixed at the source now, but **the fix only applies going forward**:
every line already on disk has no ``level``, and dream mode keeps ~6 days of
them. So this reader necessarily sees two schemas in one file, often within a
single query window, and must never blend them into one undifferentiated
number:

* entry **has** ``level`` → severity is KNOWN (read it).
* entry **has no** ``level`` → severity is INFERRED (:func:`looks_like_error`).

:func:`classify` returns both the verdict and which regime produced it, and
every caller reports the split. A count that silently mixed the two would be
the exact failure this MCP exists to prevent: a confident answer with no way
to tell how much of it was a guess.

STILL MISSING: no latency is recorded. Across 424 ``elog`` call sites, exactly
one passes ``duration_ms`` and one passes ``elapsed_s``; no MCP call path
emits either. §14's "which MCP call is slowing me down?" cannot be answered by
ranking durations — we would be inventing numbers. We surface the timing
signal that *is* real (failure/timeout hot spots per event, and wall-clock
span) and leave latency ranking to a log that records it.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Read backwards in blocks rather than lines: one 64 KB pread beats thousands
# of syscalls, and it is comfortably larger than the longest line observed in
# a real log (1105 bytes), so a single block almost always closes a partial
# line rather than carrying it across iterations.
_BLOCK_BYTES = 64 * 1024

# Hard ceiling on how much of the tail any single call may touch. This is the
# *token* guard's twin: capping returned rows alone would still let a query
# with a rare filter walk 50 MB of history on the event loop. 2 MB covers
# ~15k entries at the observed 137-byte average — far more than any capped
# result can return — so the bound is invisible in practice and only bites on
# a pathological log.
_MAX_SCAN_BYTES = int(os.environ.get("OPENAGENT_LOGS_MCP_MAX_SCAN_BYTES", str(2_000_000)))

# Value-level truncation. Precedent: tool_search clamps tool descriptions to
# 200 chars so list_tools stays affordable. Log values skew short (137-byte
# average entry) but a `traceback` or a stringified provider payload is
# unbounded and would dominate the result on its own.
_MAX_VALUE_CHARS = 300

# Levels that count as "something went wrong". `warning` is in deliberately:
# `mcp.error`, `mcp.timeout` and `agent.media.read_skip` are all logged at
# warning, so an error-only rule would miss the log's most common real
# failures. `critical` is unreachable via elog's _LEVELS map but costs
# nothing to honour if a level ever arrives from elsewhere.
_FAILURE_LEVELS = frozenset({"error", "warning", "critical"})

# Event-name tokens that imply something went wrong. Used ONLY for entries
# with no persisted level (see module docstring). Deliberately tight: these
# are substrings of real event names in this repo (`mcp.error`, `task.error`,
# `scheduler.invalid_cron`, `mcp.timeout`, `bridge.discord.dropped`,
# `tts.elevenlabs_ws.connect_failed`). A looser list (e.g. "cancel", "skip",
# "orphan") would sweep in routine lifecycle events and make error_rate lie.
_ERROR_NAME_TOKENS = (
    "error", "failed", "failure", "timeout", "invalid",
    "denied", "rejected", "crash", "dropped", "unhandled", "exception",
)

# Only this event carries an accounted spend. `runtime.cost_skipped` also has
# a `cost_usd` field but by definition was NOT mirrored onto the run's
# metrics, so summing it would double-count against the canonical record.
# NOTE: the canonical cost record is the `usage_log` DB table written by
# ModelDispatcher (see native_provider._compute_and_mirror_cost) — what the
# log holds is a mirror, so `logs_summary` reports it as such.
COST_EVENT = "runtime.cost_mirrored"

_REL_TIME_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$", re.IGNORECASE)
_REL_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def events_path() -> Path:
    """Absolute path of the live ``events.jsonl``.

    Resolved through :func:`src.core.paths.log_dir` — the same call
    ``logging.setup_logging`` uses to place the file — so this MCP always
    reads the log of the agent it is running inside, whatever ``--agent-dir``
    said. This is the single piece of file *location* logic we reuse rather
    than re-derive.
    """
    from src.core.paths import log_dir

    return log_dir() / "events.jsonl"


def parse_time(value: Any, *, now: float | None = None) -> float | None:
    """Coerce a model-supplied time bound to an epoch float.

    Accepts what an LLM actually types when asked for "yesterday":

    * relative age — ``"24h"``, ``"90m"``, ``"7d"``, ``"30s"``, ``"2w"``
      (interpreted as *this long ago*, which is what every §14 question wants)
    * ISO 8601 — ``"2026-07-13"`` or ``"2026-07-13T10:00:00"``
    * raw epoch seconds — ``1783960770.05`` or its string form

    Returns ``None`` for ``None``/blank (meaning "unbounded"). Raises
    ``ValueError`` with the accepted forms spelled out, because a silently
    ignored time bound would return a plausible-looking answer for the wrong
    window — the worst failure mode for a diagnosis tool.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"invalid time bound: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    now = now if now is not None else _now()

    m = _REL_TIME_RE.match(text)
    if m:
        return now - float(m.group(1)) * _REL_UNIT_SECONDS[m.group(2).lower()]

    try:
        return float(text)
    except ValueError:
        pass

    iso = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        raise ValueError(
            f"invalid time bound {value!r}. Use a relative age ('24h', '90m', "
            f"'7d'), an ISO date ('2026-07-13' or '2026-07-13T10:00:00'), or "
            f"epoch seconds."
        ) from None
    # A bare date/datetime is the user's local wall clock; ts in the log is
    # time.time() (UTC epoch). Assume local for naive input so "2026-07-13"
    # means the day the user means.
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.timestamp()


def _now() -> float:
    import time

    return time.time()


def iso(ts: Any) -> str | None:
    """Render an epoch ``ts`` as a UTC ISO string, or ``None`` if unusable.

    Raw epoch floats are unreadable to a model reasoning about "yesterday";
    every row and window we return carries the ISO form alongside the float
    (the float stays because ``logs_context`` anchors on it).
    """
    if not isinstance(ts, (int, float)) or isinstance(ts, bool):
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return None


def iter_lines_reverse(
    path: Path, *, max_bytes: int = _MAX_SCAN_BYTES,
) -> Iterator[tuple[bytes, int]]:
    """Yield raw lines newest-first, reading fixed blocks backwards from EOF.

    Yields ``(line, bytes_scanned_so_far)`` so callers can report how much of
    the log a bound actually covered. Stops at ``max_bytes``.

    The first line of a block is held back as ``carry`` because a block
    boundary almost always lands mid-line; it is only emitted once we reach
    byte 0 and know it is whole. If we stop early on ``max_bytes`` the carry
    is a *truncated* line and is dropped rather than handed out as a corrupt
    entry — the scan bound must not manufacture parse errors.
    """
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            pos = fh.tell()
            scanned = 0
            carry = b""
            while pos > 0 and scanned < max_bytes:
                read_size = min(_BLOCK_BYTES, pos, max_bytes - scanned)
                pos -= read_size
                fh.seek(pos)
                block = fh.read(read_size)
                scanned += read_size
                parts = (block + carry).split(b"\n")
                carry = parts[0]
                for part in reversed(parts[1:]):
                    if part.strip():
                        yield part, scanned
            if pos == 0 and carry.strip():
                yield carry, scanned
    except FileNotFoundError:
        return
    except OSError:
        # A log that cannot be opened (permissions, a directory in its place)
        # must degrade to "no entries", never take down the caller's turn.
        return


class ScanStats:
    """How much ground a scan covered — attached to every tool result.

    Without this the model cannot tell "there were no errors yesterday" from
    "I stopped scanning before yesterday", which are opposite conclusions.
    """

    __slots__ = ("lines", "bytes", "corrupt", "hit_scan_cap", "oldest_ts")

    def __init__(self) -> None:
        self.lines = 0
        self.bytes = 0
        self.corrupt = 0
        self.hit_scan_cap = False
        self.oldest_ts: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lines_scanned": self.lines,
            "bytes_scanned": self.bytes,
            "corrupt_lines_skipped": self.corrupt,
            "hit_scan_cap": self.hit_scan_cap,
            "oldest_entry_scanned": iso(self.oldest_ts),
        }


def iter_entries_reverse(
    *, since: float | None = None, max_bytes: int = _MAX_SCAN_BYTES,
    stats: ScanStats | None = None, path: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield parsed entries newest-first, stopping at ``since``.

    ``since`` is the real optimisation: the log is append-only and therefore
    ts-ordered, so the first entry older than the window means every remaining
    byte is older too and we stop reading immediately. "What went wrong in the
    last hour?" touches ~an hour of bytes, not the whole file.

    Corrupt lines are counted and skipped, never raised: a half-written line
    at the tail (the process was killed mid-``write``) is normal for an
    append-only log and must not break a diagnosis query — the exact moment
    you most want to read the log is right after a crash.
    """
    st = stats if stats is not None else ScanStats()
    p = path if path is not None else events_path()

    for raw, scanned in iter_lines_reverse(p, max_bytes=max_bytes):
        st.bytes = scanned
        st.lines += 1
        try:
            entry = json.loads(raw)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError):
            st.corrupt += 1
            continue
        if not isinstance(entry, dict):
            st.corrupt += 1
            continue

        ts = entry.get("ts")
        if isinstance(ts, (int, float)) and not isinstance(ts, bool):
            st.oldest_ts = float(ts)
            if since is not None and ts < since:
                return
        yield entry

    # Distinguish "read the whole file" from "stopped at the cap": only the
    # latter means older matches may exist beyond what we looked at.
    if st.bytes >= max_bytes:
        st.hit_scan_cap = True


def entry_level(entry: dict[str, Any]) -> str | None:
    """The persisted severity, or ``None`` for a pre-fix (level-less) entry.

    ``None`` is the load-bearing case: it means "this line predates the
    formatter fix", NOT "this line is fine".
    """
    level = entry.get("level")
    if isinstance(level, str) and level.strip():
        return level.strip().lower()
    return None


def _has_failure_evidence(entry: dict[str, Any]) -> bool:
    """Structured proof of a failure, independent of the level field.

    Needed even when a level IS present, because a call site can record a
    failure at a permissive level: ``stream.turn.end`` logs the turn's
    outcome via ``errored`` at the default ``info``.

    A **cancelled** turn is excluded, and that is the whole subtlety. A
    barge-in sets ``errored: true`` alongside ``cancelled: true``, but §2
    makes interrupting the agent mid-sentence a first-class behaviour, not a
    fault. Measured on a real log: **230 of 230** ``errored: true`` entries
    were also ``cancelled: true`` — i.e. every single one was a user
    interrupting, and counting them made barge-ins the single largest
    contributor to "what went wrong yesterday".
    """
    if entry.get("traceback"):
        return True
    if entry.get("errored") is True and entry.get("cancelled") is not True:
        return True
    return False


def looks_like_error(entry: dict[str, Any]) -> bool:
    """Best-effort severity for a **level-less** entry (the old schema).

    Only meaningful when :func:`entry_level` returns ``None``; prefer
    :func:`classify`, which picks the right regime for you.

    Keys off the evidence that survives in a pre-fix line, strongest first:

    * a ``traceback``, or a genuinely-errored (non-cancelled) turn
    * a non-empty ``error`` / ``error_type`` field (134 call sites pass one)
    * an error-ish token in the event name (``mcp.error``, ``task.error``,
      ``scheduler.invalid_cron``, ``tts.…connect_failed``)

    It over-reports by construction — ``runtime.stream.fallback`` records an
    error it then recovered from — which is the right bias for a diagnosis
    tool (a missed failure is worse than a listed non-failure), and the event
    name is always returned so the model can judge for itself.
    """
    if _has_failure_evidence(entry):
        return True
    if entry.get("error") or entry.get("error_type"):
        return True
    name = entry.get("event")
    if isinstance(name, str):
        low = name.lower()
        return any(tok in low for tok in _ERROR_NAME_TOKENS)
    return False


def classify(entry: dict[str, Any]) -> tuple[bool, bool]:
    """``(is_failure, inferred)`` for one entry, across both log schemas.

    ``inferred`` is not decoration — it is the caller's obligation to report.
    It says the verdict came from sniffing an event name rather than reading
    a severity the call site actually declared.

    With a level present, ``warning`` counts as a failure alongside ``error``.
    That is not hedging: ``mcp.error`` — 81 occurrences on a real log, plainly
    a failure — is logged at ``level="warning"``, as are ``mcp.timeout`` and
    ``agent.media.read_skip``. Restricting to ``error`` would silently drop
    the most common real failure in the log.
    """
    level = entry_level(entry)
    if level is None:
        return looks_like_error(entry), True
    return (level in _FAILURE_LEVELS or _has_failure_evidence(entry)), False


def truncate_value(
    value: Any, limit: int = _MAX_VALUE_CHARS, *, keep: str = "head",
) -> Any:
    """Clamp one field so a single fat value cannot dominate a result.

    ``keep="tail"`` is for tracebacks: the exception type and message are on
    the *last* line, so the head is the least useful part to keep. Everything
    else keeps its head.

    Scalars (int/float/bool/None) pass through untouched — a model reasoning
    about ``cost_usd`` or ``input_tokens`` needs numbers, not clipped strings.
    Containers are JSON-rendered before measuring: an ``elog`` kwarg can be a
    list or dict (``targets=[...]``, a provider payload), and measuring
    ``len()`` on a dict would count its *keys*, not its weight — so a fat
    nested payload would sail through a naive check.
    """
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        dropped = len(value) - limit
        if keep == "tail":
            return f"…[{dropped} chars trimmed]…" + value[-limit:]
        return value[:limit] + f"…[{dropped} chars trimmed]…"
    if isinstance(value, (list, tuple, dict)):
        try:
            rendered = json.dumps(value, default=str)
        except (TypeError, ValueError):
            rendered = str(value)
        if len(rendered) <= limit:
            return value
        dropped = len(rendered) - limit
        return rendered[:limit] + f"…[{dropped} chars trimmed]…"
    return value


def compact_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """One log entry, shaped for a model: ISO time first, values clamped."""
    out: dict[str, Any] = {}
    ts = entry.get("ts")
    when = iso(ts)
    if when:
        out["time"] = when
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        # Kept because logs_context anchors on the exact float; a model that
        # only had the second-resolution ISO string could not address a
        # specific entry in a busy millisecond.
        out["ts"] = ts
    out["event"] = entry.get("event")
    for key, value in entry.items():
        if key in ("ts", "event"):
            continue
        out[key] = truncate_value(
            value, keep="tail" if key == "traceback" else "head",
        )
    return out
