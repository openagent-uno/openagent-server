"""Unified logging for OpenAgent.

One system, two outputs:

* stdout   — free-form text from ``logging.getLogger(__name__)`` anywhere.
* events.jsonl — structured events via :func:`elog`, one JSON object per line.

Call :func:`setup_logging` once at process start (the CLI does this).

This module owns the events.jsonl *format*, so it owns the reader too:
:func:`iter_events_reverse` is the one primitive that knows how the file is
laid out (append-only, ts-ordered, one JSON object per line, occasionally a
half-written tail line). ``read_tail`` here, ``GET /api/logs``, and the
``logs`` MCP all sit on top of it. The MCP's reader was written first, as its
own bounded implementation, precisely because this module only offered a
whole-file slurp; keeping two readers of one format in sync "by convention" is
a drift we have been paying for elsewhere in the codebase, so the MCP now
layers its policy (a scan cap, severity inference) over this primitive rather
than re-deriving the format.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core.paths import log_dir

EVENT_LOGGER = "openagent.events"
_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}
_configured = False
_event_file_path: Path | None = None


def elog(event: str, level: str = "info", exc_info: bool = False, **data: Any) -> None:
    """Append a structured event to ``events.jsonl`` (and mirror to stdout).

    *level* controls stdout severity (``info``/``warning``/``error``); at
    default verbosity only ``warning``+ shows on the console.  Pass
    ``exc_info=True`` inside an ``except`` block to also capture a traceback
    (into events.jsonl, and on stdout).
    """
    if not _configured:
        setup_logging()
    logging.getLogger(EVENT_LOGGER).log(
        _LEVELS[level], event, exc_info=exc_info, extra={"event_data": data}
    )


def setup_logging(verbose: bool = False) -> None:
    """Configure stdlib logging: stdout for text, events.jsonl for events.

    Safe to call repeatedly. If the resolved :func:`log_dir` changes between
    calls (e.g. the agent directory is set after initial bootstrap), the file
    handler is reopened at the new location so logs follow the agent.
    """
    global _configured, _event_file_path
    stdout_level = logging.DEBUG if verbose else logging.WARNING
    target = events_path()

    if _configured:
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(stdout_level)
        if _event_file_path != target:
            _reopen_event_file(target)
        return

    # Root accepts everything; the console handler is what gates stdout
    # verbosity.  A logger-level gate wouldn't work because records
    # propagated up from child loggers bypass the parent's level check.
    console = logging.StreamHandler()
    console.setLevel(stdout_level)
    console.setFormatter(_ConsoleFormatter())
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)

    # Chatty third-party loggers emit at DEBUG/INFO and drown real output —
    # aiosqlite logs every SQL statement it executes, which buries the signal
    # in schema dumps. Root runs at DEBUG (so events.jsonl captures every
    # level), so we gate these at their own logger level: a record below
    # WARNING is dropped at the source and never reaches any handler.
    for _noisy in ("aiosqlite", "httpx", "httpcore", "hpack", "urllib3",
                   "asyncio", "websockets", "telegram", "PIL"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    events = logging.getLogger(EVENT_LOGGER)
    events.setLevel(logging.DEBUG)  # events.jsonl captures every level
    _reopen_event_file(target)

    _configured = True


def _reopen_event_file(target: Path) -> None:
    """Swap the FileHandler on the event logger to write to *target*."""
    global _event_file_path
    events = logging.getLogger(EVENT_LOGGER)
    for h in list(events.handlers):
        if isinstance(h, logging.FileHandler):
            h.close()
            events.removeHandler(h)
    handler = logging.FileHandler(target, encoding="utf-8")
    handler.setFormatter(_JsonlFormatter())
    events.addHandler(handler)
    _event_file_path = target


# ── reading events.jsonl ─────────────────────────────────────────────
#
# Read backwards in blocks rather than lines: one 64 KB pread beats thousands
# of syscalls, and it is comfortably larger than the longest line observed in
# a real log (1105 bytes), so a single block almost always closes a partial
# line rather than carrying it across iterations.
_BLOCK_BYTES = 64 * 1024


def events_path() -> Path:
    """Absolute path of the live ``events.jsonl``.

    Resolved through :func:`src.core.paths.log_dir` on every call — the agent
    directory can be set *after* bootstrap, and both the writer
    (:func:`setup_logging`) and every reader must follow it to the same file.
    """
    return log_dir() / "events.jsonl"


def iso(ts: Any) -> str | None:
    """Render an epoch ``ts`` as a UTC ISO string, or ``None`` if unusable.

    The ``ts`` this module writes is a raw ``record.created`` float, which is
    unreadable to a human with ``jq`` and to a model reasoning about
    "yesterday"; anything presenting entries renders it through here.
    """
    if not isinstance(ts, (int, float)) or isinstance(ts, bool):
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return None


def iter_lines_reverse(
    path: Path, *, max_bytes: int | None = None,
) -> Iterator[tuple[bytes, int]]:
    """Yield raw lines newest-first, reading fixed blocks backwards from EOF.

    Yields ``(line, bytes_scanned_so_far)`` so callers can report how much of
    the log a bound actually covered. ``max_bytes=None`` means unbounded (read
    to byte 0); a bound stops the scan there.

    The first line of a block is held back as ``carry`` because a block
    boundary almost always lands mid-line; it is only emitted once we reach
    byte 0 and know it is whole. If we stop early on ``max_bytes`` the carry
    is a *truncated* line and is dropped rather than handed out as a corrupt
    entry — a scan bound must not manufacture parse errors.
    """
    limit = float("inf") if max_bytes is None else max_bytes
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            pos = fh.tell()
            scanned = 0
            carry = b""
            while pos > 0 and scanned < limit:
                read_size = int(min(_BLOCK_BYTES, pos, limit - scanned))
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
    """How much ground a scan covered — reported alongside a scan's results.

    Without this a caller cannot tell "there were no errors yesterday" from
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


def iter_events_reverse(
    *, since: float | None = None, max_bytes: int | None = None,
    stats: ScanStats | None = None, path: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield parsed entries newest-first, stopping at ``since``.

    ``max_bytes`` defaults to **unbounded** on purpose. A cap belongs to the
    caller's policy, not to the format: the ``logs`` MCP caps its scans
    because a model-driven query with a rare filter must not walk 50 MB on the
    event loop, but :func:`read_tail` must not — a prefix that only matches
    entries older than the cap has to keep returning them, exactly as the
    slurping implementation did.

    ``since`` is the real optimisation: the log is append-only and therefore
    ts-ordered, so the first entry older than the window means every remaining
    byte is older too and we stop reading immediately.

    Corrupt lines are counted and skipped, never raised: a half-written line
    at the tail (the process was killed mid-``write``) is normal for an
    append-only log and must not break a query — the exact moment you most
    want to read the log is right after a crash.
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
    if max_bytes is not None and st.bytes >= max_bytes:
        st.hit_scan_cap = True


def read_tail(lines: int = 100, event_filter: str | None = None) -> list[dict[str, Any]]:
    """Return the last *lines* entries, optionally filtered by event prefix.

    Blocking (plain file I/O) — an async caller must offload it, which is why
    ``GET /api/logs`` hands it to a thread. It reads only as far back as it
    needs to fill *lines*; it used to ``read_text().splitlines()`` the whole
    file, which on a real ~1 MB log (measured 728 KB / 5365 entries on a light
    install, and dream mode trims by age, not size) stalled the gateway's event
    loop — the same loop carrying live WebSocket streams and voice audio.

    Deliberately **unbounded**: with ``event_filter`` set, the matching entries
    may all be ancient, and a scan cap here would silently return fewer rows
    than the old slurp did.
    """
    out: list[dict[str, Any]] = []
    for entry in iter_events_reverse():
        if event_filter and not entry.get("event", "").startswith(event_filter):
            continue
        out.append(entry)
        if len(out) >= lines:
            break
    out.reverse()
    return out


def clear(older_than_days: float | None = None) -> None:
    """Truncate ``events.jsonl`` and re-open the file handler.

    With no argument, wipes the whole file. If *older_than_days* is given,
    only entries whose ``ts`` is older than that many days are dropped;
    newer (and malformed / ts-less) entries are preserved.
    """
    path = events_path()
    if older_than_days is None:
        path.write_text("", encoding="utf-8")
        _reopen_event_file(path)
        return

    cutoff = time.time() - older_than_days * 86400
    try:
        raw = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        _reopen_event_file(path)
        return

    kept: list[str] = []
    for line in raw:
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            # Preserve unparseable lines rather than silently dropping them.
            kept.append(line)
            continue
        ts = entry.get("ts")
        if not isinstance(ts, (int, float)) or ts >= cutoff:
            kept.append(line)

    path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    _reopen_event_file(path)


class _JsonlFormatter(logging.Formatter):
    """Render one event as a single JSON line: ``{ts, event, level, **data}``.

    ``level`` is written from ``record.levelname`` because for a long time it
    was not: every :func:`elog` call dutifully passed ``level=`` (144 call
    sites), the value reached the stdlib record, and this formatter then
    dropped it on the floor. The result was a log where severity existed
    everywhere except the place you read it — measured at **0 of 5365 entries
    carrying a level** on a real agent. Anything asking "what went wrong?"
    (the logs MCP, dream mode, a human with ``jq``) had to *infer* severity
    from an ``error`` field or an error-ish event name, which over-reports
    recovered failures and cannot see a level-only warning at all.

    Key order is ``{when, what, how-bad, details}``. ``level`` sits before
    ``**event_data`` purely for readability — a ``level`` key in
    ``event_data`` is unreachable, so no order can shadow it: ``elog``
    declares ``level`` as a named parameter, so ``elog("x", level="error")``
    binds the parameter and ``**data`` gets ``{}``. Pinned by
    ``test_logs_mcp``'s schema tripwire.

    NOTE (latent, not fixed here — no call site does it today): ``ts`` has no
    such protection. It is NOT a named parameter of ``elog``, so
    ``elog("x", ts=123)`` lands in ``**data`` and silently overwrites the real
    ``record.created`` timestamp, because ``**event_data`` is splatted last.
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": record.created,
            "event": record.getMessage(),
            "level": record.levelname.lower(),
            **getattr(record, "event_data", {}),
        }
        if record.exc_info:
            entry["traceback"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class _ConsoleFormatter(logging.Formatter):
    """Human-readable stdout format.

    For ``openagent.events`` records (from :func:`elog`), inlines the
    structured fields so operators tailing stdout see the same detail
    the old ``logger.error("msg: %s", err)`` calls used to print — not
    just the event name.
    """

    def format(self, record: logging.LogRecord) -> str:
        if record.name == EVENT_LOGGER:
            data = getattr(record, "event_data", {}) or {}
            fields = " ".join(f"{k}={v!r}" for k, v in data.items())
            base = f"{record.getMessage()}" + (f" {fields}" if fields else "")
            msg = f"{record.name}: {base}"
        else:
            msg = f"{record.name}: {record.getMessage()}"
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return msg
