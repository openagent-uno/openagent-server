"""Provenance for vault commits — *what* produced a change.

Every vault commit records who caused it: a chat session, a workflow, a
scheduled task, the dream loop, or a human editing in Obsidian. OpenAgent's
own writes carry the origin inline (the REST handler / tool knows the
context). For changes it cannot hook individually (the external vault MCP, a
human in Obsidian), the autocommit sweep attributes them to the most recent
activity via ``note_activity`` — so an external write made during a chat turn
still lands in history tagged with that session.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

_lock = threading.Lock()
_last: dict[str, Any] = {"origin": None, "ts": 0.0}

# Order + labels for the git trailer lines.
_ORDER = ("kind", "session", "workflow", "task", "run", "tool", "user")
_LABELS = {
    "kind": "Origin", "session": "Session", "workflow": "Workflow",
    "task": "Task", "run": "Run", "tool": "Tool", "user": "User",
}


def _clean(origin: dict | None) -> dict:
    if not origin:
        return {}
    return {k: str(v) for k, v in origin.items() if v not in (None, "")}


def note_activity(**origin: Any) -> None:
    """Record the current execution context so the autocommit sweep can
    attribute out-of-band edits to it. Called at chat-turn / workflow /
    scheduled-task boundaries."""
    clean = _clean(origin)
    if not clean:
        return
    with _lock:
        _last["origin"] = clean
        _last["ts"] = time.monotonic()


def recent_origin(max_age_s: float = 180.0) -> Optional[dict]:
    """The most recent activity origin, if it is fresh enough to plausibly
    own an out-of-band edit; otherwise ``None`` (→ "external")."""
    with _lock:
        origin = _last["origin"]
        ts = _last["ts"]
    if origin and (time.monotonic() - ts) <= max_age_s:
        return dict(origin)
    return None


def origin_from_request(request: Any) -> dict:
    """Derive an origin from an aiohttp request — a client may declare its
    session/source via query params or ``X-OpenAgent-*`` headers."""
    try:
        q = request.query
        h = request.headers
        origin = {"kind": q.get("source") or h.get("X-OpenAgent-Source") or "gateway"}
        session = q.get("session") or h.get("X-OpenAgent-Session")
        if session:
            origin["session"] = session
        return origin
    except Exception:  # noqa: BLE001
        return {"kind": "gateway"}


def trailers(origin: dict | None) -> list[str]:
    """Render an origin dict into git trailer lines (``Key: value``)."""
    clean = _clean(origin)
    if not clean:
        return ["Origin: system"]
    out: list[str] = []
    for k in _ORDER:
        if clean.get(k):
            out.append(f"{_LABELS[k]}: {clean[k]}")
    for k, v in clean.items():
        if k not in _ORDER:
            out.append(f"{k[:1].upper() + k[1:]}: {v}")
    return out or ["Origin: system"]
