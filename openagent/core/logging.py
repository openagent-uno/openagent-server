"""Unified logging for OpenAgent.

One system, two outputs:

* **Console / stdout** — free-form messages from any module via the stdlib
  ``logger = logging.getLogger(__name__)`` pattern.  Captured by systemd/launchd.
* **events.jsonl** — structured events via :func:`elog`, appended as one JSON
  object per line to ``<log_dir>/events.jsonl``.

Everything is plain stdlib ``logging``.  :func:`elog` is just a convenience
wrapper that attaches structured data to a record on the ``openagent.events``
logger; a JSON formatter on that logger's file handler turns the record into
one line of JSONL.

Call :func:`setup_logging` once at process start (the CLI does this).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from openagent.core.paths import log_dir

EVENT_LOGGER_NAME = "openagent.events"

# Marker attribute used to recognise our own handlers so we never attach
# duplicates when setup_logging / _ensure_events_handler run twice.
_OWNED = "_openagent_owned"


# ── Public API ───────────────────────────────────────────────────────────────


def elog(event: str, **data: Any) -> None:
    """Log a structured event to ``events.jsonl`` (and mirror to stdout)."""
    _ensure_events_handler()
    logging.getLogger(EVENT_LOGGER_NAME).info(event, extra={"event_data": data})


def setup_logging(verbose: bool = False) -> None:
    """Configure stdlib logging for the whole process.

    Attaches exactly one console handler to the root logger, and ensures the
    ``openagent.events`` logger has its JSONL file handler.  Safe to call
    repeatedly — duplicate handlers are skipped.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.WARNING)

    if not _has_owned_handler(root):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        setattr(console, _OWNED, True)
        root.addHandler(console)

    _ensure_events_handler()


def read_tail(
    lines: int = 100,
    event_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Return the last *lines* entries from ``events.jsonl``.

    Optionally keep only entries whose event name starts with *event_filter*.
    """
    try:
        raw_lines = _events_path().read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []

    result: list[dict[str, Any]] = []
    for raw in reversed(raw_lines):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if event_filter and not entry.get("event", "").startswith(event_filter):
            continue
        result.append(entry)
        if len(result) >= lines:
            break
    result.reverse()
    return result


def clear() -> None:
    """Truncate ``events.jsonl``.  The next :func:`elog` call reopens the file."""
    events = logging.getLogger(EVENT_LOGGER_NAME)
    for handler in list(events.handlers):
        if getattr(handler, _OWNED, False):
            handler.close()
            events.removeHandler(handler)
    _events_path().write_text("", encoding="utf-8")


# ── Internals ────────────────────────────────────────────────────────────────


def _events_path() -> Path:
    return log_dir() / "events.jsonl"


def _has_owned_handler(logger: logging.Logger) -> bool:
    return any(getattr(h, _OWNED, False) for h in logger.handlers)


def _ensure_events_handler() -> None:
    events = logging.getLogger(EVENT_LOGGER_NAME)
    events.setLevel(logging.INFO)  # capture events even without --verbose
    if _has_owned_handler(events):
        return
    handler = logging.FileHandler(_events_path(), encoding="utf-8")
    handler.setFormatter(_JsonlFormatter())
    setattr(handler, _OWNED, True)
    events.addHandler(handler)


class _JsonlFormatter(logging.Formatter):
    """Format an event record as ``{"ts", "event", **event_data}`` JSON."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": record.created,
            "event": record.getMessage(),
            **getattr(record, "event_data", {}),
        }
        return json.dumps(entry, default=str)
