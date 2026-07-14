"""Per-run dry-run flag, propagated to MCP tool calls as call ``meta``.

When an event fires with ``dry_run: true`` in its payload, the whole run is a
dry-run: every MCP tool call it makes is stamped with ``meta={"dry_run": True}``
so the MCP server on the other side (Replio, billingbear, …) can CAPTURE or
REJECT a write instead of executing it — without OpenAgent needing to know which
tools are writes. Generic: any MCP can honour the flag.

The flag lives in a ``ContextVar`` set for the duration of the run
(:func:`dry_run_scope`) and read at the single MCP call site
(``src.core._runner.utils.mcp.get_entrypoint_for_tool``). ContextVars propagate
into awaited coroutines and into child tasks (copied at ``create_task`` time),
so the value set around ``run_child_session`` reaches every tool call the run
makes.
"""
from __future__ import annotations

import contextlib
import contextvars
from typing import Iterator

_DRY_RUN: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "openagent_dry_run", default=False
)


def is_dry_run() -> bool:
    """True while executing inside a :func:`dry_run_scope`."""
    return _DRY_RUN.get()


def call_meta() -> dict[str, bool] | None:
    """The ``meta`` dict to stamp on an MCP tool call, or None when live."""
    return {"dry_run": True} if _DRY_RUN.get() else None


@contextlib.contextmanager
def dry_run_scope(enabled: bool) -> Iterator[None]:
    """Mark the current run (and everything it awaits) as dry-run or not.

    Always pushes a value — including ``False`` — so a nested live run inside a
    worker that also serves dry-runs can't inherit a stale True from an earlier
    task on the same context.
    """
    token = _DRY_RUN.set(bool(enabled))
    try:
        yield
    finally:
        _DRY_RUN.reset(token)
