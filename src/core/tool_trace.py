"""Per-run tool-trace capture — so the quality judge can VERIFY grounding.

WHY THIS EXISTS
---------------
The quality judge (``core/quality_monitor.py``) grades a sampled turn from the
USER message and the ASSISTANT reply alone. It never saw the run's tool calls.
So when the agent correctly cited an id/user/ticket that came *verbatim out of a
tool RESULT* — a Replio thread brief, a ClickUp task id — the judge, seeing no
trace of where that fact came from, ruled it "fabricated / ungrounded / no tool
calls shown" and emitted a false ``bad`` verdict. (Confirmed on real sessions:
a reply quoting ids that were present verbatim in a ``replio_thread_brief``
result was scored ``bad`` for fabrication; a 13-tool-call run was flagged "zero
tool calls shown".)

This module records a COMPACT trace of each run's tool calls — the tool name and
a truncated excerpt of its result — so the judge can check an id/fact against
what the tools actually returned before calling it invented.

WHY A CONTEXTVAR SINK, AND WHY HERE
-----------------------------------
Mirrors ``src/core/vault_recall.py`` and ``src/models/stream_usage.py`` exactly
— same problem, same shape, same reason. The sites that SEE a tool execution
(``NativeProvider.stream``, the Team router's collect/stream loops) cannot
import the dispatcher without a cycle, so the sink lives in a leaf module that
imports nothing but stdlib. The dispatcher opens the sink per call and holds the
dict reference; the inner generators only MUTATE the dict, never rebind the
ContextVar, so nothing depends on context propagating back out of an async
generator. On completion the dispatcher PUBLISHES the run's trace into a small
bounded, per-session hand-off map that ``quality_monitor.spawn_scoring`` drains
synchronously right after the turn — the judge task then carries the trace it
captured, immune to a later turn overwriting it.

OFF BY DEFAULT / §17
--------------------
Keyed on the SAME switch as the quality monitor (``OPENAGENT_QUALITY_MONITOR_
ENABLED``): when the monitor is off, ``maybe_open`` returns no token, ``record``
is a bare no-op (no sink), and ``publish`` stores nothing — so a deployment that
never enabled the monitor is byte-identical. Everything here is best-effort:
a bookkeeping miss must cost a trace, never a turn.
"""
from __future__ import annotations

import contextvars
import os
from collections import OrderedDict
from typing import Any, Optional

_ENABLED_ENV = "OPENAGENT_QUALITY_MONITOR_ENABLED"

# Bounds — a runaway loop, or a tool that returns tens of KB, must cost a
# bounded trace, not unbounded memory and not a ballooning judge prompt.
_MAX_TOOLS = 40            # tool calls captured per run (the rest are dropped)
_MAX_RESULT_CHARS = 600    # per-tool result excerpt cap
_MAX_SESSIONS = 256        # bounded {session_id: trace} hand-off map

_SINK: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "openagent_tool_trace_sink", default=None
)

# Per-session hand-off: the dispatcher publishes a completed run's trace here,
# keyed by session, and ``quality_monitor.spawn_scoring`` pops it synchronously
# right after the turn. One entry per session (latest run overwrites), FIFO-
# evicted past ``_MAX_SESSIONS`` so it can never grow without bound even when
# sampling means most entries are published but never taken.
_PUBLISHED: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def _enabled() -> bool:
    return _truthy(os.environ.get(_ENABLED_ENV, "0"))


def _excerpt(value: Any) -> str:
    """A bounded single-line excerpt of a tool result."""
    if value is None:
        return ""
    try:
        s = value if isinstance(value, str) else str(value)
    except Exception:  # noqa: BLE001 — a weird result repr must never raise here
        return ""
    s = " ".join(s.split())  # collapse whitespace/newlines to keep it compact
    return s if len(s) <= _MAX_RESULT_CHARS else s[:_MAX_RESULT_CHARS] + " …[truncated]"


def maybe_open() -> tuple[Optional[dict], Optional[contextvars.Token]]:
    """Start capturing this call's tool trace, or ``(None, None)`` when off.

    Returns the sink dict (held by the dispatcher for :func:`publish`) and the
    ContextVar token (for :func:`close`), mirroring ``vault_recall.open_sink``.
    """
    if not _enabled():
        return None, None
    sink: dict[str, Any] = {"tools": []}
    return sink, _SINK.set(sink)


def close(token: Optional[contextvars.Token]) -> None:
    if token is None:
        return
    try:
        _SINK.reset(token)
    except (ValueError, LookupError):
        pass


def record(tool_name: Any, result: Any) -> None:
    """Record one completed tool call (name + a truncated result excerpt).

    A no-op when no sink is open (the monitor is off, or a provider streaming
    outside the dispatcher — a test, a direct call). Never raises."""
    sink = _SINK.get()
    if sink is None:
        return
    try:
        name = str(tool_name).strip() if tool_name else ""
        if not name:
            return
        tools = sink["tools"]
        if len(tools) >= _MAX_TOOLS:
            return
        tools.append((name, _excerpt(result)))
    except Exception:  # noqa: BLE001 — a trace miss must cost a row, never a turn
        return


def record_execution(entry: Any) -> None:
    """Record from a runtime ``ToolExecution`` (object or dict shape).

    Handles both shapes for the same reason ``vault_recall._tool_name_args``
    does: the runtime has changed this object before, and a shape we fail to
    read must cost a counter, not a turn."""
    if _SINK.get() is None:
        return
    if isinstance(entry, dict):
        record(entry.get("tool_name"), entry.get("result"))
    else:
        record(getattr(entry, "tool_name", None), getattr(entry, "result", None))


def publish(session_id: Optional[str], sink: Optional[dict]) -> None:
    """Store a completed run's trace for the session, bounded. No-op for an
    empty trace (a no-tool run needs no trace) or when disabled."""
    if not session_id or not sink:
        return
    tools = sink.get("tools") or []
    if not tools:
        return
    try:
        _PUBLISHED[session_id] = list(tools)
        _PUBLISHED.move_to_end(session_id)
        while len(_PUBLISHED) > _MAX_SESSIONS:
            _PUBLISHED.popitem(last=False)
    except Exception:  # noqa: BLE001
        return


def take(session_id: Optional[str]) -> Optional[list[tuple[str, str]]]:
    """Pop the most recent published trace for ``session_id`` (or ``None``)."""
    if not session_id:
        return None
    return _PUBLISHED.pop(session_id, None)


def peek(session_id: Optional[str]) -> Optional[list[tuple[str, str]]]:
    """Read the most recent published trace for ``session_id`` WITHOUT consuming
    it, so a pre-send reader (the reply guard) can inspect this turn's tool calls
    while leaving the entry for ``take`` (the quality judge) to drain afterwards.
    Returns ``None`` when nothing was published (disabled, or a no-tool run)."""
    if not session_id:
        return None
    return _PUBLISHED.get(session_id)


def render(rows: Optional[list[tuple[str, str]]], *, max_chars: int = 3000) -> str:
    """Render a compact, bounded TOOL TRACE block for the judge prompt.

    Returns ``""`` when there is nothing to show. The total is capped at
    ``max_chars`` so the trace can never balloon the (already second) judge call
    — excess tool lines are elided with a count."""
    if not rows:
        return ""
    lines: list[str] = []
    used = 0
    shown = 0
    for i, (name, excerpt) in enumerate(rows, start=1):
        line = f"[{i}] {name} → {excerpt}" if excerpt else f"[{i}] {name} → (no result)"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
        shown += 1
    if shown < len(rows):
        lines.append(f"… (+{len(rows) - shown} more tool calls elided)")
    return "\n".join(lines)
