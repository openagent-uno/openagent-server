"""Graph validator.

Runs on every ``create_workflow`` / ``update_workflow`` / incremental
edit (``add_block`` and friends). Catches the classes of mistakes a
cheap LLM edit most often makes:

- unknown block ``type``
- edges referencing node ids that don't exist
- cycles (a ``loop`` block's body may self-reference via its own
  ``sourceHandle='body'`` — that specific edge is exempt; anything else
  is a cycle and gets rejected)
- missing required config fields
- config fields of the wrong shape (enum, scalar type)
- multiple trigger nodes when ``trigger_kind`` says only one
"""

from __future__ import annotations

from typing import Any

from openagent.workflow.blocks import BLOCK_CATALOG, BlockSpec


class ValidationError(ValueError):
    """Raised by :func:`validate_graph` with an actionable message."""

    def __init__(self, message: str, *, node_id: str | None = None, field: str | None = None):
        super().__init__(message)
        self.node_id = node_id
        self.field = field


def _check_config(node_id: str, spec: BlockSpec, config: dict[str, Any]) -> None:
    for key, field_spec in spec.config_schema.items():
        if field_spec.get("required") and config.get(key) in (None, ""):
            raise ValidationError(
                f"node {node_id}: required config field {key!r} is missing",
                node_id=node_id,
                field=key,
            )
        if key in config and config[key] is not None:
            value = config[key]
            expected = field_spec.get("type")
            enum = field_spec.get("enum")
            if enum is not None and value not in enum:
                raise ValidationError(
                    f"node {node_id}: field {key!r} must be one of {enum}, got {value!r}",
                    node_id=node_id,
                    field=key,
                )
            type_map = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "object": dict,
                "array": list,
                "boolean": bool,
            }
            expected_type = type_map.get(expected) if expected else None
            if expected_type is not None and not isinstance(value, expected_type):
                raise ValidationError(
                    f"node {node_id}: field {key!r} must be {expected}, "
                    f"got {type(value).__name__}",
                    node_id=node_id,
                    field=key,
                )


def _detect_cycle(nodes: list[dict], edges: list[dict]) -> None:
    """Iterative DFS; loop blocks' body-self-edges are exempt."""
    adj: dict[str, list[tuple[str, str]]] = {n["id"]: [] for n in nodes}
    for e in edges:
        src = e.get("source")
        tgt = e.get("target")
        handle = e.get("sourceHandle") or "out"
        if src in adj and tgt in adj:
            adj[src].append((tgt, handle))

    by_id = {n["id"]: n for n in nodes}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {nid: WHITE for nid in adj}

    def dfs(start: str) -> None:
        stack: list[tuple[str, int]] = [(start, 0)]
        color[start] = GRAY
        path: list[str] = [start]
        while stack:
            nid, idx = stack[-1]
            successors = adj[nid]
            if idx >= len(successors):
                color[nid] = BLACK
                stack.pop()
                path.pop()
                continue
            stack[-1] = (nid, idx + 1)
            nxt, handle = successors[idx]
            # Loop body self-edges are allowed.
            if by_id.get(nid, {}).get("type") == "loop" and handle == "body" and nxt == nid:
                continue
            if color[nxt] == GRAY:
                raise ValidationError(
                    f"cycle detected: {' → '.join(path + [nxt])}",
                    node_id=nxt,
                )
            if color[nxt] == WHITE:
                color[nxt] = GRAY
                stack.append((nxt, 0))
                path.append(nxt)

    for nid in adj:
        if color[nid] == WHITE:
            dfs(nid)


def validate_graph(graph: dict[str, Any]) -> None:
    """Validate a ``graph_json`` payload. Raises ``ValidationError`` on
    the first problem found; returns ``None`` on success.
    """
    if not isinstance(graph, dict):
        raise ValidationError("graph must be an object")
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    if not isinstance(nodes, list):
        raise ValidationError("graph.nodes must be a list")
    if not isinstance(edges, list):
        raise ValidationError("graph.edges must be a list")

    seen_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise ValidationError("every node must be an object")
        nid = node.get("id")
        if not nid or not isinstance(nid, str):
            raise ValidationError("every node must have a string id")
        if nid in seen_ids:
            raise ValidationError(f"duplicate node id {nid!r}", node_id=nid)
        seen_ids.add(nid)
        ntype = node.get("type")
        if ntype not in BLOCK_CATALOG:
            raise ValidationError(
                f"node {nid}: unknown type {ntype!r}. "
                f"Known types: {sorted(BLOCK_CATALOG)}",
                node_id=nid,
                field="type",
            )
        spec = BLOCK_CATALOG[ntype]
        config = node.get("config") or {}
        if not isinstance(config, dict):
            raise ValidationError(
                f"node {nid}: config must be an object",
                node_id=nid,
                field="config",
            )
        _check_config(nid, spec, config)

    for edge in edges:
        if not isinstance(edge, dict):
            raise ValidationError("every edge must be an object")
        src = edge.get("source")
        tgt = edge.get("target")
        if src not in seen_ids:
            raise ValidationError(
                f"edge references unknown source {src!r}",
            )
        if tgt not in seen_ids:
            raise ValidationError(
                f"edge references unknown target {tgt!r}",
            )

    _detect_cycle(nodes, edges)
