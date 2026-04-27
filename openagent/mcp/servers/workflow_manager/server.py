"""workflow-manager MCP server.

Exposes OpenAgent's workflow engine over MCP so the agent can create,
edit, list, and run its own n8n-style workflows at runtime — no
operator CLI needed. Mirrors the ``scheduler`` / ``mcp-manager`` /
``model-manager`` pattern: stdio subprocess, FastMCP, direct SQLite
writes, the main OpenAgent process picks up changes on its next
poll tick.

``run_workflow`` doesn't execute locally — this subprocess has no
Agent, no MCP pool, no models. It drops a row into
``workflow_run_requests`` and polls ``workflow_runs.status`` until
the main-process scheduler finishes the run. Same pattern the
mcp-manager uses (DB-backed hand-off).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

import aiosqlite
from mcp.server.fastmcp import FastMCP

from openagent.workflow import (
    BLOCK_CATALOG,
    ValidationError,
    iter_block_specs,
    validate_graph,
)
from openagent.workflow.blocks import get_block_spec
from openagent.workflow.examples import (
    get_workflow_example as _get_workflow_example,
    list_workflow_examples as _list_workflow_examples,
)
from openagent.workflow.schedule_sync import (
    iter_trigger_schedule_blocks,
    trigger_types_from_graph,
)
from openagent.memory.db import SCHEMA_SQL
from openagent.memory.schedule import (
    epoch_to_iso,
    next_run_for_expression,
    validate_schedule_expression,
)

logger = logging.getLogger(__name__)


def _db_path() -> str:
    return os.environ.get("OPENAGENT_DB_PATH") or "openagent.db"


_conn_lock = asyncio.Lock()
_conn: aiosqlite.Connection | None = None


async def _get_conn() -> aiosqlite.Connection:
    global _conn
    async with _conn_lock:
        if _conn is None:
            path = _db_path()
            conn = await aiosqlite.connect(path)
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA busy_timeout = 10000")
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.executescript(SCHEMA_SQL)
            await conn.commit()
            _conn = conn
            logger.info("workflow-manager MCP connected to %s", path)
        return _conn


# ── row hydration ────────────────────────────────────────────────────


def _decorate_workflow_row(row: aiosqlite.Row) -> dict[str, Any]:
    """Shape a workflow row without loading its schedules yet.
    ``_decorate_workflow_full`` wraps this + a schedules lookup for
    callers that want the fully-decorated response shape."""
    d = dict(row)
    raw = d.pop("graph_json", None) or '{"version":1,"nodes":[],"edges":[],"variables":{}}'
    try:
        d["graph"] = json.loads(raw)
    except (TypeError, ValueError):
        d["graph"] = {"version": 1, "nodes": [], "edges": [], "variables": {}}
    d["enabled"] = bool(d.get("enabled"))
    # Drop deprecated v0.12.10 row-level fields; schedules now live
    # in workflow_schedules.
    for deprecated in ("trigger_kind", "cron_expression", "next_run_at"):
        d.pop(deprecated, None)
    for key in ("last_run_at", "created_at", "updated_at"):
        epoch = d.get(key)
        d[f"{key}_iso"] = epoch_to_iso(epoch) if epoch else None
    d["trigger_types"] = trigger_types_from_graph(d.get("graph"))
    return d


async def _decorate_workflow(
    conn: aiosqlite.Connection, row: aiosqlite.Row,
) -> dict[str, Any]:
    """Full workflow decoration including the ``schedules`` array
    pulled from ``workflow_schedules``."""
    d = _decorate_workflow_row(row)
    cursor = await conn.execute(
        "SELECT * FROM workflow_schedules WHERE workflow_id = ? "
        "ORDER BY next_run_at ASC",
        (d["id"],),
    )
    rows = await cursor.fetchall()
    schedules = []
    for s in rows:
        sd = dict(s)
        sd["enabled"] = bool(sd.get("enabled"))
        for key in ("next_run_at", "last_run_at", "created_at", "updated_at"):
            epoch = sd.get(key)
            sd[f"{key}_iso"] = epoch_to_iso(epoch) if epoch else None
        schedules.append(sd)
    d["schedules"] = schedules
    return d


def _decorate_run(row: aiosqlite.Row) -> dict[str, Any]:
    d = dict(row)
    for col in ("inputs_json", "outputs_json", "trace_json"):
        raw = d.pop(col, None) or ("[]" if col == "trace_json" else "{}")
        key = col[:-5]
        try:
            d[key] = json.loads(raw)
        except (TypeError, ValueError):
            d[key] = [] if key == "trace" else {}
    for key in ("started_at", "finished_at"):
        epoch = d.get(key)
        d[f"{key}_iso"] = epoch_to_iso(epoch) if epoch else None
    return d


async def _resolve_workflow(
    conn: aiosqlite.Connection, id_or_name: str,
) -> aiosqlite.Row:
    """Accept full id, 8-char id prefix, or unique name. Raises ``ValueError``."""
    if not id_or_name:
        raise ValueError("workflow id or name is required")
    cursor = await conn.execute(
        "SELECT * FROM workflow_tasks WHERE id = ? OR name = ? LIMIT 2",
        (id_or_name, id_or_name),
    )
    rows = await cursor.fetchall()
    if len(rows) == 1:
        return rows[0]
    if len(rows) > 1:
        raise ValueError(
            f"Ambiguous identifier {id_or_name!r}: matches multiple workflows."
        )
    if len(id_or_name) >= 4:
        cursor = await conn.execute(
            "SELECT * FROM workflow_tasks WHERE id LIKE ? LIMIT 2",
            (f"{id_or_name}%",),
        )
        rows = await cursor.fetchall()
        if len(rows) == 1:
            return rows[0]
        if len(rows) > 1:
            raise ValueError(
                f"Ambiguous id prefix {id_or_name!r}: use a longer prefix or "
                "the full UUID."
            )
    raise ValueError(f"No workflow matching {id_or_name!r}")


def _next_node_id(existing: list[dict]) -> str:
    """Generate a short stable id like ``n1``, ``n2``. Skips ids already taken."""
    used = {n["id"] for n in existing if isinstance(n.get("id"), str)}
    i = len(existing) + 1
    while True:
        candidate = f"n{i}"
        if candidate not in used:
            return candidate
        i += 1


def _next_edge_id(existing: list[dict]) -> str:
    used = {e["id"] for e in existing if isinstance(e.get("id"), str)}
    i = len(existing) + 1
    while True:
        candidate = f"e{i}"
        if candidate not in used:
            return candidate
        i += 1


def _auto_position(nodes: list[dict]) -> dict[str, float]:
    """Place new node to the right of the rightmost existing node."""
    if not nodes:
        return {"x": 120.0, "y": 120.0}
    rightmost = max(
        (n.get("position", {}).get("x", 0) for n in nodes),
        default=0,
    )
    topish = min((n.get("position", {}).get("y", 120) for n in nodes), default=120)
    return {"x": float(rightmost) + 240.0, "y": float(topish)}


async def _load_graph(conn: aiosqlite.Connection, workflow_id: str) -> dict:
    cursor = await conn.execute(
        "SELECT graph_json FROM workflow_tasks WHERE id = ?", (workflow_id,),
    )
    row = await cursor.fetchone()
    if not row:
        raise ValueError(f"workflow {workflow_id} vanished mid-edit")
    return json.loads(row[0])


async def _sync_workflow_schedules(
    conn: aiosqlite.Connection, workflow_id: str, graph: dict,
) -> None:
    """Reconcile ``workflow_schedules`` rows against the graph's
    ``trigger-schedule`` blocks. Same logic as
    ``openagent.workflow.schedule_sync.sync_workflow_schedules`` but
    runs against the MCP subprocess's raw ``aiosqlite.Connection``
    without pulling MemoryDB into this process.
    """
    keep_node_ids: list[str] = []
    now = time.time()
    for node in iter_trigger_schedule_blocks(graph):
        node_id = node.get("id")
        cfg = node.get("config") or {}
        cron = cfg.get("cron_expression")
        if not node_id or not cron:
            continue
        try:
            validate_schedule_expression(cron)
            nxt = next_run_for_expression(cron)
        except ValueError:
            continue
        keep_node_ids.append(node_id)
        cursor = await conn.execute(
            "SELECT id, cron_expression FROM workflow_schedules "
            "WHERE workflow_id = ? AND node_id = ?",
            (workflow_id, node_id),
        )
        existing = await cursor.fetchone()
        if existing is not None:
            keep_next = existing["cron_expression"] == cron
            if keep_next:
                await conn.execute(
                    "UPDATE workflow_schedules SET cron_expression = ?, "
                    "updated_at = ? WHERE id = ?",
                    (cron, now, existing["id"]),
                )
            else:
                await conn.execute(
                    "UPDATE workflow_schedules SET cron_expression = ?, "
                    "next_run_at = ?, updated_at = ? WHERE id = ?",
                    (cron, nxt, now, existing["id"]),
                )
        else:
            await conn.execute(
                "INSERT INTO workflow_schedules "
                "(id, workflow_id, node_id, cron_expression, next_run_at, "
                " enabled, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                (
                    str(uuid.uuid4()),
                    workflow_id,
                    node_id,
                    cron,
                    nxt,
                    now,
                    now,
                ),
            )
    # Prune schedules for removed / renamed blocks.
    if keep_node_ids:
        placeholders = ",".join("?" for _ in keep_node_ids)
        await conn.execute(
            f"DELETE FROM workflow_schedules WHERE workflow_id = ? "
            f"AND node_id NOT IN ({placeholders})",
            [workflow_id, *keep_node_ids],
        )
    else:
        await conn.execute(
            "DELETE FROM workflow_schedules WHERE workflow_id = ?",
            (workflow_id,),
        )


async def _save_graph(
    conn: aiosqlite.Connection, workflow_id: str, graph: dict,
) -> None:
    """Persist graph_json + reconcile ``workflow_schedules`` rows.

    After any graph write (``add_block`` / ``update_block`` /
    ``remove_block`` / ``connect_blocks``) the graph is the source of
    truth: adding a trigger-schedule block creates a matching schedule
    row, editing its cron updates the row, removing it prunes the row.
    """
    validate_graph(graph)
    now = time.time()
    await conn.execute(
        "UPDATE workflow_tasks SET graph_json = ?, updated_at = ? WHERE id = ?",
        (json.dumps(graph), now, workflow_id),
    )
    await _sync_workflow_schedules(conn, workflow_id, graph)
    await conn.commit()


# ── FastMCP server ───────────────────────────────────────────────────

mcp = FastMCP("workflow-manager")


@mcp.tool()
async def list_workflows(
    enabled_only: bool = False,
    has_trigger_type: str | None = None,
) -> list[dict[str, Any]]:
    """List workflows with their graphs and schedule rows.

    - ``enabled_only`` filters to workflows whose ``enabled`` flag is set.
    - ``has_trigger_type`` filters to workflows whose graph contains at
      least one block of that type (e.g. ``'trigger-schedule'``,
      ``'trigger-ai'``, ``'trigger-manual'``). Computed in Python from
      the returned graphs since triggers live inside the graph.

    Workflows no longer carry a row-level ``trigger_kind`` — any
    workflow can be triggered manually, by the AI, or on a schedule at
    any time. The trigger nodes inside the graph are the whole story.
    """
    conn = await _get_conn()
    where = "WHERE enabled = 1" if enabled_only else ""
    cursor = await conn.execute(
        f"SELECT * FROM workflow_tasks {where} ORDER BY updated_at DESC"
    )
    rows = await cursor.fetchall()
    decorated = [await _decorate_workflow(conn, r) for r in rows]
    if has_trigger_type:
        decorated = [w for w in decorated if has_trigger_type in w["trigger_types"]]
    return decorated


@mcp.tool()
async def get_workflow(id_or_name: str) -> dict[str, Any]:
    """Return a single workflow: ``{id, name, description, enabled,
    graph: {nodes, edges, variables}, schedules: [...], trigger_types:
    [...], last_run_at, ...}``. Accepts full id, 8-char id prefix, or
    unique name.

    ``schedules`` is one row per ``trigger-schedule`` block inside the
    graph, carrying its ``cron_expression`` + ``next_run_at`` +
    ``last_run_at`` — read this, not a row-level cron column.
    """
    conn = await _get_conn()
    row = await _resolve_workflow(conn, id_or_name)
    return await _decorate_workflow(conn, row)


@mcp.tool()
async def create_workflow(
    name: str,
    description: str = "",
    nodes: list[dict] | None = None,
    edges: list[dict] | None = None,
    variables: dict | None = None,
) -> dict[str, Any]:
    """Create a new workflow.

    - ``name`` must be unique across workflows (case-sensitive).
    - ``nodes``/``edges``/``variables`` default to an empty graph you
      can populate with ``add_block`` + ``connect_blocks``. Use
      ``trigger-manual`` / ``trigger-schedule`` / ``trigger-ai``
      blocks inside the graph to control *how* it fires — the workflow
      row itself has no ``trigger_kind``.
    - ``trigger-schedule`` blocks with a ``cron_expression`` in their
      ``config`` automatically appear in ``workflow_schedules`` and
      are fired by the main-process scheduler.

    Schema (the validator enforces this):

        node = {
          "id":      "<unique str>",          # referenced by edges
          "type":    "<one of BLOCK_CATALOG>",  # see list_block_types
          "label":   "<human label>",         # optional UI hint
          "position": {"x": int, "y": int},   # optional UI hint
          "config":  {<per-block fields>},    # see describe_block_type
        }
        edge = {
          "id":             "<unique str>",
          "source":         "<source node id>",
          "target":         "<target node id>",
          "sourceHandle":   "out" | "true" | "false" | "branch_<i>" | "body" | "done",
          "targetHandle":   "in"  | "body",
        }

    Workflow-authoring playbook:

      1. Call ``list_workflow_examples`` to see canonical patterns.
         If one matches your intent, ``get_workflow_example(name)``
         returns a complete copy-pasteable graph — adapt and create.
      2. Call ``list_block_types`` (or ``describe_block_type(t)``) to
         confirm config_schema for every block you use.
      3. Call ``list_available_tools`` BEFORE referencing ``mcp-tool``
         blocks; tool_name MUST be the prefixed form the pool exposes
         (e.g. ``messaging_telegram_send_message``, NOT bare
         ``telegram_send_message``).
      4. Wire branching/parallel/loop edges with the right
         ``sourceHandle`` (``true``/``false`` for ``if``,
         ``branch_<i>`` for ``parallel``, ``body``/``done`` for
         ``loop``) — the default ``out`` won't fire on those blocks.
      5. Reference upstream block outputs in templated args via
         ``{{nodes.<id>.output.<field>}}``.
    """
    if not name or not name.strip():
        raise ValueError("name is required")
    graph = {
        "version": 1,
        "nodes": nodes or [],
        "edges": edges or [],
        "variables": variables or {},
    }
    try:
        validate_graph(graph)
    except ValidationError as e:
        raise ValueError(f"graph validation failed: {e}") from e

    conn = await _get_conn()
    workflow_id = str(uuid.uuid4())
    now = time.time()
    try:
        await conn.execute(
            "INSERT INTO workflow_tasks "
            "(id, name, description, graph_json, enabled, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, 1, ?, ?)",
            (
                workflow_id,
                name.strip(),
                description or None,
                json.dumps(graph),
                now,
                now,
            ),
        )
        await _sync_workflow_schedules(conn, workflow_id, graph)
        await conn.commit()
    except aiosqlite.IntegrityError as e:
        if "UNIQUE" in str(e):
            raise ValueError(f"workflow name {name!r} is already taken") from None
        raise
    return await get_workflow(workflow_id)


@mcp.tool()
async def update_workflow(
    id_or_name: str,
    name: str | None = None,
    description: str | None = None,
    nodes: list[dict] | None = None,
    edges: list[dict] | None = None,
    variables: dict | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Patch-style update. Only fields you pass are changed.

    Passing ``nodes``/``edges``/``variables`` REPLACES the whole graph —
    for incremental edits use ``add_block`` / ``update_block`` /
    ``remove_block`` / ``connect_blocks`` / ``disconnect_blocks``.

    Scheduling lives inside the graph: add / edit / remove a
    ``trigger-schedule`` block's ``config.cron_expression`` to control
    when the workflow fires. Per-block schedules are kept in
    ``workflow_schedules`` via ``_sync_workflow_schedules`` on every
    write.
    """
    conn = await _get_conn()
    row = await _resolve_workflow(conn, id_or_name)
    workflow_id = row["id"]
    current_graph = json.loads(row["graph_json"])

    updates: dict[str, Any] = {}
    if name is not None:
        if not name.strip():
            raise ValueError("name cannot be empty")
        updates["name"] = name.strip()
    if description is not None:
        updates["description"] = description or None
    if enabled is not None:
        updates["enabled"] = 1 if enabled else 0

    new_graph: dict | None = None
    if nodes is not None or edges is not None or variables is not None:
        new_graph = {
            "version": current_graph.get("version", 1),
            "nodes": nodes if nodes is not None else current_graph.get("nodes", []),
            "edges": edges if edges is not None else current_graph.get("edges", []),
            "variables": (
                variables if variables is not None else current_graph.get("variables", {})
            ),
        }
        try:
            validate_graph(new_graph)
        except ValidationError as e:
            raise ValueError(f"graph validation failed: {e}") from e
        updates["graph_json"] = json.dumps(new_graph)

    if not updates:
        return await _decorate_workflow(conn, row)

    updates["updated_at"] = time.time()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    try:
        await conn.execute(
            f"UPDATE workflow_tasks SET {set_clause} WHERE id = ?",
            list(updates.values()) + [workflow_id],
        )
        # When the graph changed, sync per-block schedule rows before
        # committing so reads in the same transaction see consistent
        # state.
        if new_graph is not None:
            await _sync_workflow_schedules(conn, workflow_id, new_graph)
        await conn.commit()
    except aiosqlite.IntegrityError as e:
        if "UNIQUE" in str(e):
            raise ValueError(f"workflow name {name!r} is already taken") from None
        raise
    return await get_workflow(workflow_id)


@mcp.tool()
async def delete_workflow(id_or_name: str) -> dict[str, Any]:
    """Delete a workflow and all its run history (FK cascade)."""
    conn = await _get_conn()
    row = await _resolve_workflow(conn, id_or_name)
    workflow_id = row["id"]
    name = row["name"]
    await conn.execute("DELETE FROM workflow_tasks WHERE id = ?", (workflow_id,))
    await conn.commit()
    return {"deleted": True, "id": workflow_id, "name": name}


# ── Incremental graph edits ──────────────────────────────────────────


@mcp.tool()
async def add_block(
    workflow_id_or_name: str,
    type: str,
    label: str = "",
    config: dict | None = None,
    position: dict | None = None,
    after_node_id: str | None = None,
) -> dict[str, Any]:
    """Append a block to a workflow.

    - ``type`` must be one of the types returned by ``describe_block_type``
      / ``GET /api/workflow-block-types``.
    - ``position`` auto-computes to the right of existing nodes when omitted.
    - ``after_node_id`` is a convenience: when set, a default-handle edge
      is added from that node to the new block so you don't need a
      separate ``connect_blocks`` call.

    Returns ``{node_id, graph}`` where ``graph`` is the updated graph.
    """
    spec = get_block_spec(type)  # raises KeyError with known types on miss
    conn = await _get_conn()
    row = await _resolve_workflow(conn, workflow_id_or_name)
    workflow_id = row["id"]
    graph = await _load_graph(conn, workflow_id)

    nodes = graph.setdefault("nodes", [])
    edges = graph.setdefault("edges", [])
    new_id = _next_node_id(nodes)
    node = {
        "id": new_id,
        "type": type,
        "label": label or spec.type,
        "position": position or _auto_position(nodes),
        "config": config or {},
    }
    nodes.append(node)

    if after_node_id:
        # Let validation catch missing nodes — but still produce a friendly
        # error instead of a silent no-op.
        if not any(n["id"] == after_node_id for n in nodes):
            raise ValueError(f"after_node_id {after_node_id!r} does not exist")
        edges.append({
            "id": _next_edge_id(edges),
            "source": after_node_id,
            "target": new_id,
            "sourceHandle": "out",
            "targetHandle": "in",
            "label": None,
        })

    await _save_graph(conn, workflow_id, graph)
    return {"node_id": new_id, "graph": graph}


@mcp.tool()
async def update_block(
    workflow_id_or_name: str,
    node_id: str,
    label: str | None = None,
    config: dict | None = None,
    position: dict | None = None,
) -> dict[str, Any]:
    """Patch a single block's label / config / position. ``config`` is
    shallow-merged into the existing value (pass an explicit empty dict
    to clear)."""
    conn = await _get_conn()
    row = await _resolve_workflow(conn, workflow_id_or_name)
    workflow_id = row["id"]
    graph = await _load_graph(conn, workflow_id)

    target = next((n for n in graph.get("nodes", []) if n["id"] == node_id), None)
    if target is None:
        raise ValueError(f"node {node_id!r} not found")
    if label is not None:
        target["label"] = label
    if position is not None:
        target["position"] = position
    if config is not None:
        if not config:
            target["config"] = {}
        else:
            target.setdefault("config", {}).update(config)

    await _save_graph(conn, workflow_id, graph)
    return {"node": target, "graph": graph}


@mcp.tool()
async def remove_block(
    workflow_id_or_name: str, node_id: str,
) -> dict[str, Any]:
    """Remove a block and every edge touching it."""
    conn = await _get_conn()
    row = await _resolve_workflow(conn, workflow_id_or_name)
    workflow_id = row["id"]
    graph = await _load_graph(conn, workflow_id)
    before = len(graph.get("nodes", []))
    graph["nodes"] = [n for n in graph.get("nodes", []) if n["id"] != node_id]
    if len(graph["nodes"]) == before:
        raise ValueError(f"node {node_id!r} not found")
    graph["edges"] = [
        e for e in graph.get("edges", [])
        if e["source"] != node_id and e["target"] != node_id
    ]
    await _save_graph(conn, workflow_id, graph)
    return {"removed": node_id, "graph": graph}


@mcp.tool()
async def connect_blocks(
    workflow_id_or_name: str,
    from_node_id: str,
    to_node_id: str,
    source_handle: str = "out",
    target_handle: str = "in",
    label: str | None = None,
) -> dict[str, Any]:
    """Add an edge between two blocks. ``source_handle`` defaults to
    ``'out'`` for normal blocks; use ``'true' / 'false'`` for ``if``,
    ``'body' / 'done'`` for ``loop``, ``'branch_N'`` for ``parallel``.
    """
    conn = await _get_conn()
    row = await _resolve_workflow(conn, workflow_id_or_name)
    workflow_id = row["id"]
    graph = await _load_graph(conn, workflow_id)

    nodes_by_id = {n["id"]: n for n in graph.get("nodes", [])}
    if from_node_id not in nodes_by_id:
        raise ValueError(f"from_node_id {from_node_id!r} does not exist")
    if to_node_id not in nodes_by_id:
        raise ValueError(f"to_node_id {to_node_id!r} does not exist")

    edges = graph.setdefault("edges", [])
    edge = {
        "id": _next_edge_id(edges),
        "source": from_node_id,
        "target": to_node_id,
        "sourceHandle": source_handle,
        "targetHandle": target_handle,
        "label": label,
    }
    edges.append(edge)
    await _save_graph(conn, workflow_id, graph)
    return {"edge": edge, "graph": graph}


@mcp.tool()
async def disconnect_blocks(
    workflow_id_or_name: str,
    from_node_id: str,
    to_node_id: str,
    source_handle: str | None = None,
) -> dict[str, Any]:
    """Remove all edges between two blocks. When ``source_handle`` is
    provided, only edges on that handle are removed."""
    conn = await _get_conn()
    row = await _resolve_workflow(conn, workflow_id_or_name)
    workflow_id = row["id"]
    graph = await _load_graph(conn, workflow_id)
    edges = graph.get("edges", [])
    kept: list[dict] = []
    removed = 0
    for e in edges:
        matches = e["source"] == from_node_id and e["target"] == to_node_id
        if source_handle is not None:
            matches = matches and (e.get("sourceHandle") == source_handle)
        if matches:
            removed += 1
        else:
            kept.append(e)
    if removed == 0:
        raise ValueError(
            f"no edge from {from_node_id!r} to {to_node_id!r}"
            + (f" on handle {source_handle!r}" if source_handle else "")
        )
    graph["edges"] = kept
    await _save_graph(conn, workflow_id, graph)
    return {"removed_count": removed, "graph": graph}


# ── Introspection ────────────────────────────────────────────────────


@mcp.tool()
async def describe_block_type(type_name: str) -> dict[str, Any]:
    """Return the catalog entry for a block type: description, config
    schema, handles, output shape. Use this when you're unsure how to
    structure a block's ``config`` before calling ``add_block`` /
    ``update_block``.
    """
    return get_block_spec(type_name).as_dict()


@mcp.tool()
async def list_block_types() -> list[dict[str, Any]]:
    """List every available block type with its description + config
    schema. Sibling of ``describe_block_type`` for bulk discovery.
    """
    return iter_block_specs()


@mcp.tool()
async def list_workflow_examples() -> list[dict[str, Any]]:
    """Lightweight index of canonical workflow examples — name +
    description + patterns demonstrated. Cheap to scan when picking
    which example to load in full via ``get_workflow_example``.

    Use this BEFORE calling ``create_workflow`` for any non-trivial
    graph: pick the example whose ``patterns`` match your intent,
    then pull its graph and adapt.
    """
    return _list_workflow_examples()


@mcp.tool()
async def get_workflow_example(name: str) -> dict[str, Any]:
    """Return one canonical workflow example by ``name`` — full graph
    included, ready to adapt and pass to ``create_workflow``. Names
    come from ``list_workflow_examples``. Each example passes
    structural validation; pool-level tool existence is still your
    responsibility (verify via ``list_available_tools``)."""
    return _get_workflow_example(name)


@mcp.tool()
async def list_available_tools() -> list[dict[str, Any]]:
    """Enumerate MCPs (builtin + user-configured) so the AI can pick one
    for an ``mcp-tool`` block.

    Reads directly from the ``mcps`` table — returns one row per
    enabled MCP with its name, kind, and whether it's builtin. Tool
    names per-MCP are fetched from the live gateway via
    ``GET /api/mcp-tools`` when available; this MCP subprocess can
    only see the configuration, not the runtime tool list.
    """
    conn = await _get_conn()
    cursor = await conn.execute(
        "SELECT name, kind, builtin_name, enabled FROM mcps "
        "WHERE enabled = 1 ORDER BY name ASC"
    )
    rows = await cursor.fetchall()
    return [
        {
            "mcp_name": r["name"],
            "kind": r["kind"],
            "builtin_name": r["builtin_name"],
            "note": (
                "Use GET /api/mcp-tools (or the workflow editor's tool "
                "picker) to list the actual tools this MCP exposes."
            ),
        }
        for r in rows
    ]


# ── Execution (enqueue + poll) ───────────────────────────────────────


@mcp.tool()
async def run_workflow(
    id_or_name: str,
    inputs: dict | None = None,
    wait: bool = True,
    timeout_s: int = 300,
) -> dict[str, Any]:
    """Run a workflow.

    Same pattern the mcp-manager uses: this subprocess has no Agent,
    so we drop a row into ``workflow_run_requests`` and the main
    OpenAgent process claims + executes it. When ``wait=True`` we
    poll until completion and return the final run (with trace).
    When ``wait=False`` we return as soon as the main process
    attaches a ``run_id`` — caller can later query via
    ``get_workflow_run``.
    """
    conn = await _get_conn()
    row = await _resolve_workflow(conn, id_or_name)
    workflow_id = row["id"]
    req_id = str(uuid.uuid4())
    await conn.execute(
        "INSERT INTO workflow_run_requests "
        "(id, workflow_id, inputs_json, trigger, created_at) "
        "VALUES (?, ?, ?, 'ai', ?)",
        (req_id, workflow_id, json.dumps(inputs or {}), time.time()),
    )
    await conn.commit()

    # Wait for the main process to claim + attach a run_id.
    run_id = await _poll_request(conn, req_id, timeout_s=min(timeout_s, 60))
    if not wait:
        return {"run_id": run_id, "status": "running"}

    # Then wait for the run itself to finish.
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        cursor = await conn.execute(
            "SELECT * FROM workflow_runs WHERE id = ?", (run_id,),
        )
        run_row = await cursor.fetchone()
        if run_row and run_row["status"] != "running":
            return _decorate_run(run_row)
        await asyncio.sleep(0.5)
    raise TimeoutError(
        f"workflow_run {run_id!r} did not finish within {timeout_s}s"
    )


async def _poll_request(
    conn: aiosqlite.Connection, request_id: str, *, timeout_s: float,
) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        cursor = await conn.execute(
            "SELECT run_id FROM workflow_run_requests WHERE id = ?",
            (request_id,),
        )
        row = await cursor.fetchone()
        if row and row["run_id"]:
            return row["run_id"]
        await asyncio.sleep(0.25)
    raise TimeoutError(
        f"workflow run request {request_id!r} was not picked up within "
        f"{timeout_s}s — is the main OpenAgent process running?"
    )


@mcp.tool()
async def list_workflow_runs(
    id_or_name: str, limit: int = 20, status: str | None = None,
) -> list[dict[str, Any]]:
    """List recent runs of a workflow, newest first."""
    conn = await _get_conn()
    row = await _resolve_workflow(conn, id_or_name)
    workflow_id = row["id"]
    clauses = ["workflow_id = ?"]
    params: list[Any] = [workflow_id]
    if status:
        clauses.append("status = ?")
        params.append(status)
    params.append(int(limit))
    cursor = await conn.execute(
        f"SELECT * FROM workflow_runs WHERE {' AND '.join(clauses)} "
        "ORDER BY started_at DESC LIMIT ?",
        params,
    )
    rows = await cursor.fetchall()
    return [_decorate_run(r) for r in rows]


@mcp.tool()
async def get_workflow_run(run_id: str) -> dict[str, Any]:
    """Fetch a single run row with its full trace."""
    conn = await _get_conn()
    cursor = await conn.execute(
        "SELECT * FROM workflow_runs WHERE id = ?", (run_id,),
    )
    row = await cursor.fetchone()
    if not row:
        raise ValueError(f"no workflow_run with id {run_id!r}")
    return _decorate_run(row)


# ── entrypoint ───────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("OPENAGENT_WORKFLOW_MCP_LOGLEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    mcp.run()


if __name__ == "__main__":
    main()
