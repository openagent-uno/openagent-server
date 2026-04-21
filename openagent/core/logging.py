"""Unified logging for OpenAgent.

One system, two outputs:

* stdout   — free-form text from ``logging.getLogger(__name__)`` anywhere.
* events.jsonl — structured events via :func:`elog`, one JSON object per line.

Call :func:`setup_logging` once at process start (the CLI does this).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from openagent.core.paths import log_dir

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
    target = log_dir() / "events.jsonl"

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


def read_tail(lines: int = 100, event_filter: str | None = None) -> list[dict[str, Any]]:
    """Return the last *lines* entries, optionally filtered by event prefix."""
    try:
        raw = (log_dir() / "events.jsonl").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    out: list[dict[str, Any]] = []
    for line in reversed(raw):
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
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
    path = log_dir() / "events.jsonl"
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
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": record.created,
            "event": record.getMessage(),
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
