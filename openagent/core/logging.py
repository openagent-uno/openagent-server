"""Unified structured event logger for OpenAgent.

Every significant event in OpenAgent (session lifecycle, tool execution,
MCP connections, bridge status, errors, restarts …) is logged as a single
JSON-lines entry to ``<log_dir>/events.jsonl``.

Usage::

    from openagent.core.logging import elog

    elog("tool.start", tool="bash", params={"command": "ls"})
    elog("session.create", session_id="abc-123", client_id="app-1")

The convenience function :func:`elog` is the **only** call site most
modules need.  It writes one JSON object per line and also emits
the event to Python's standard ``logging`` at INFO level so that
systemd/launchd stdout capture still works.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from openagent.core.paths import log_dir

_std_logger = logging.getLogger("openagent.events")

# ---------------------------------------------------------------------------
# EventLogger singleton
# ---------------------------------------------------------------------------


class EventLogger:
    """Append-only JSONL logger stored at ``<log_dir>/events.jsonl``."""

    _instance: EventLogger | None = None

    @classmethod
    def get(cls) -> EventLogger:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._path: Path = log_dir() / "events.jsonl"
        self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115

    # -- write --

    def log(self, event: str, **data: Any) -> None:
        """Append one structured event entry."""
        entry: dict[str, Any] = {"ts": time.time(), "event": event, **data}
        line = json.dumps(entry, default=str)
        self._file.write(line + "\n")
        self._file.flush()
        # Mirror to standard logging (truncated) for console/systemd.
        _std_logger.info("[%s] %s", event, json.dumps(data, default=str)[:200])

    # -- read --

    def read_tail(
        self,
        lines: int = 100,
        event_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the last *lines* entries, optionally filtered by event prefix."""
        try:
            all_lines = self._path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        result: list[dict[str, Any]] = []
        for raw in reversed(all_lines):
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

    # -- maintenance --

    def clear(self) -> None:
        """Truncate the log file (called by dream mode daily)."""
        self._file.close()
        self._path.write_text("", encoding="utf-8")
        self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115

    def close(self) -> None:
        self._file.close()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def elog(event: str, **data: Any) -> None:
    """Log a structured event.  Shorthand for ``EventLogger.get().log(…)``."""
    EventLogger.get().log(event, **data)
