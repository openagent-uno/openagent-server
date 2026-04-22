"""Keep the ``workflow_schedules`` table in sync with each workflow's
graph. Called on every workflow write — the graph is the single source
of truth, the table is its scheduler-friendly index.

Each ``trigger-schedule`` block in a graph becomes a row keyed by
``(workflow_id, node_id)``. Removing a block removes its row; editing
a block's cron updates the row (and re-computes ``next_run_at`` only
when the cron actually changed, so the scheduler doesn't "miss" a
scheduled tick just because someone renamed the block).

A workflow can carry any number of ``trigger-schedule`` blocks. The
row-level scheduler has no opinion on the workflow itself — workflows
no longer carry ``trigger_kind``, ``cron_expression``, or
``next_run_at`` columns. All schedule state is per-block.
"""

from __future__ import annotations

from typing import Any, Iterable

from openagent.memory.schedule import (
    next_run_for_expression,
    validate_schedule_expression,
)


def iter_trigger_schedule_blocks(
    graph: dict[str, Any] | None,
) -> Iterable[dict[str, Any]]:
    """Yield each ``trigger-schedule`` node from ``graph.nodes``.

    Yields nothing for empty or malformed graphs. Consumers should
    check ``node["config"]["cron_expression"]`` — blocks with empty
    cron are intentionally left unsynced so the user can add a block
    in the editor, save, and fill the cron later.
    """
    if not graph:
        return
    for node in graph.get("nodes") or []:
        if node.get("type") == "trigger-schedule":
            yield node


async def sync_workflow_schedules(
    db: Any,
    workflow_id: str,
    graph: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reconcile ``workflow_schedules`` rows against the graph.

    Creates a row for each ``trigger-schedule`` block with a non-empty
    cron; updates existing rows (preserving ``next_run_at`` when the
    cron didn't change); deletes rows for blocks that no longer exist.

    Returns a small summary dict ``{created, updated, removed,
    invalid}`` — mostly useful for tests; production callers can
    ignore it.
    """
    summary = {"created": 0, "updated": 0, "removed": 0, "invalid": 0}
    keep_node_ids: list[str] = []
    existing = await db.list_schedules(workflow_id=workflow_id)
    by_node = {row["node_id"]: row for row in existing}

    for node in iter_trigger_schedule_blocks(graph):
        node_id = node.get("id")
        cfg = node.get("config") or {}
        cron = cfg.get("cron_expression")
        if not node_id or not cron:
            # Block exists but has no cron yet — don't create a row.
            # Any previous row keyed to this node_id will be pruned
            # by ``delete_schedules_not_in`` below (node_id missing
            # from ``keep_node_ids``).
            continue
        try:
            validate_schedule_expression(cron)
        except ValueError:
            summary["invalid"] += 1
            continue
        keep_node_ids.append(node_id)
        try:
            nxt = next_run_for_expression(cron)
        except ValueError:
            summary["invalid"] += 1
            continue
        if node_id in by_node and by_node[node_id]["cron_expression"] == cron:
            summary["updated"] += 1
        else:
            summary["created"] += 1 if node_id not in by_node else 0
            summary["updated"] += 1 if node_id in by_node else 0
        await db.upsert_schedule(
            workflow_id=workflow_id,
            node_id=node_id,
            cron_expression=cron,
            next_run_at=nxt,
        )

    removed = await db.delete_schedules_not_in(workflow_id, keep_node_ids)
    summary["removed"] = removed
    return summary


def trigger_types_from_graph(graph: dict[str, Any] | None) -> list[str]:
    """Derive the set of trigger types present in a workflow's graph.

    Returns a sorted list so ``list_workflows`` responses are stable.
    Callers (UI / AI) use this to render the "How is this triggered?"
    badge without needing a dedicated ``trigger_kind`` column.
    """
    if not graph:
        return []
    seen: set[str] = set()
    for node in graph.get("nodes") or []:
        ntype = node.get("type")
        if isinstance(ntype, str) and ntype.startswith("trigger-"):
            seen.add(ntype)
    return sorted(seen)
