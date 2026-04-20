"""Keep a workflow's row-level ``cron_expression`` in sync with the
``trigger-schedule`` block inside its graph.

The scheduler loop polls the ``workflow_tasks.cron_expression`` +
``next_run_at`` columns — that's what makes a workflow fire on a
schedule. The user (or the AI) edits the schedule visually via a
``trigger-schedule`` block's ``config.cron_expression``. Without a
sync step, the two drift: you'd change the block's cron and nothing
would happen because the scheduler reads the row column.

This module exports one pure helper that computes the row updates a
caller should apply after any graph mutation. Call it from the
gateway's REST handlers and from every workflow-manager MCP tool that
touches the graph.

Direction: graph → row. Row-level cron can also be set directly
(e.g. the list-screen create form when the graph is empty); in that
case the caller passes ``explicit_cron`` and the helper honours it.
When both are present, ``explicit_cron`` wins.
"""

from __future__ import annotations

from typing import Any

from openagent.memory.schedule import (
    next_run_for_expression,
    validate_schedule_expression,
)


# Public sentinel distinguishing "caller didn't pass cron" from
# "caller passed None (clear the schedule)". Callers that want to say
# "leave the row cron alone" pass ``NOT_PROVIDED``; callers that want
# to clear it pass ``None``. Any other value is validated as cron.
NOT_PROVIDED = object()


def pick_trigger_schedule_cron(graph: dict[str, Any]) -> str | None:
    """Return the cron_expression of the first ``trigger-schedule``
    block in the graph, or ``None`` if there isn't one. v1 uses the
    first block by iteration order; workflows needing multiple
    distinct schedules should be split into separate workflows.
    """
    for node in graph.get("nodes", []) or []:
        if node.get("type") != "trigger-schedule":
            continue
        cfg = node.get("config") or {}
        cron = cfg.get("cron_expression")
        if cron:
            return str(cron)
    return None


def derive_schedule_updates(
    graph: dict[str, Any] | None,
    *,
    explicit_cron: Any = NOT_PROVIDED,
    explicit_trigger_kind: str | None = None,
    current_trigger_kind: str | None = None,
) -> dict[str, Any]:
    """Compute row patches ``{cron_expression, next_run_at,
    trigger_kind}`` needed after a graph mutation.

    Precedence:
      1. ``explicit_cron`` if the caller passed one (row-level edit).
         ``None`` clears. Anything else is validated + used.
      2. Otherwise the first ``trigger-schedule`` block in the graph.
      3. Otherwise no change (returns ``{}``).

    Auto-promote: when a cron becomes active and ``trigger_kind`` is
    ``None`` or ``'manual'``, bump it to ``'schedule'`` so the
    scheduler loop starts polling. When the caller explicitly passes
    ``'hybrid'`` / ``'ai'`` we leave that alone — they're signaling
    multiple entry points.
    """
    graph = graph or {}
    graph_cron = pick_trigger_schedule_cron(graph)

    if explicit_cron is not NOT_PROVIDED:
        cron_to_use = explicit_cron
    elif graph_cron is not None:
        cron_to_use = graph_cron
    else:
        return {}

    updates: dict[str, Any] = {}
    tk = (
        explicit_trigger_kind
        if explicit_trigger_kind is not None
        else current_trigger_kind
    )

    if cron_to_use:
        validate_schedule_expression(cron_to_use)
        updates["cron_expression"] = str(cron_to_use)
        updates["next_run_at"] = next_run_for_expression(str(cron_to_use))
        if tk in (None, "manual"):
            updates["trigger_kind"] = "schedule"
    else:
        updates["cron_expression"] = None
        updates["next_run_at"] = None
    return updates
