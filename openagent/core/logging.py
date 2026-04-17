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
_configured = False


def elog(event: str, **data: Any) -> None:
    """Append a structured event to ``events.jsonl`` (and mirror to stdout)."""
    if not _configured:
        setup_logging()
    logging.getLogger(EVENT_LOGGER).info(event, extra={"event_data": data})


def setup_logging(verbose: bool = False) -> None:
    """Configure stdlib logging: stdout for text, events.jsonl for events."""
    global _configured
    if _configured:
        logging.getLogger().setLevel(logging.DEBUG if verbose else logging.WARNING)
        return

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.WARNING)
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
        return json.dumps(
            {"ts": record.created, "event": record.getMessage(),
             **getattr(record, "event_data", {})},
            default=str,
        )
