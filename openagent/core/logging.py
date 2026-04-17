"""Unified logging for OpenAgent.

One system, two outputs:

* stdout   — free-form text from ``logging.getLogger(__name__)`` anywhere.
* events.jsonl — structured events via :func:`elog`, one JSON object per line.

Call :func:`setup_logging` once at process start (the CLI does this).
"""

from __future__ import annotations

import json
import logging
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
    """Configure stdlib logging: stdout for text, events.jsonl for events."""
    global _configured
    stdout_level = logging.DEBUG if verbose else logging.WARNING
    if _configured:
        for h in logging.getLogger().handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(stdout_level)
        return

    # Root accepts everything; the console handler is what gates stdout
    # verbosity.  A logger-level gate wouldn't work because records
    # propagated up from child loggers bypass the parent's level check.
    console = logging.StreamHandler()
    console.setLevel(stdout_level)
    console.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(console)

    events_file = logging.FileHandler(log_dir() / "events.jsonl", encoding="utf-8")
    events_file.setFormatter(_JsonlFormatter())
    events = logging.getLogger(EVENT_LOGGER)
    events.setLevel(logging.INFO)  # capture events even without --verbose
    events.addHandler(events_file)

    _configured = True


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


def clear() -> None:
    """Truncate ``events.jsonl`` and re-open the file handler."""
    events = logging.getLogger(EVENT_LOGGER)
    for h in list(events.handlers):
        h.close()
        events.removeHandler(h)
    path = log_dir() / "events.jsonl"
    path.write_text("", encoding="utf-8")
    new_handler = logging.FileHandler(path, encoding="utf-8")
    new_handler.setFormatter(_JsonlFormatter())
    events.addHandler(new_handler)


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
