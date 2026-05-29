"""Telemetry no-op shim.

OpenAgent does not ship the eval / runtime-telemetry subsystem that was
inlined from Agno (vision §17 — self-hosted, no remote calls by default).
The run loop still emits ``log_agent_telemetry()`` calls at key points;
they all resolve here and do nothing.

If a future contributor wires up first-party telemetry, replace these
functions with real implementations — the call sites already pass
``agent``, ``session_id`` and ``run_id``.
"""
from __future__ import annotations

from typing import Any


def log_agent_telemetry(
    agent: Any,
    *,
    session_id: str | None = None,
    run_id: str | None = None,
    **_kwargs: Any,
) -> None:  # noqa: D401 — no-op
    """No-op telemetry hook.

    Kept as a function (not a pass-through ``lambda``) so the call sites
    can be grepped, and so a future telemetry implementation only has to
    edit this file."""
    return None


async def alog_agent_telemetry(
    agent: Any,
    *,
    session_id: str | None = None,
    run_id: str | None = None,
    **_kwargs: Any,
) -> None:
    """Async sibling of :func:`log_agent_telemetry` — same no-op contract."""
    return None
