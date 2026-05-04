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
- ``mcp-tool`` blocks pointing at MCPs/tools that aren't loaded (only
  when an ``mcp_inventory`` is supplied — see ``validate_graph``)
"""

from __future__ import annotations

import difflib
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


def _check_mcp_tool(
    node_id: str,
    config: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
    callability: dict[str, dict[str, bool]] | None = None,
) -> None:
    """Validate (and lightly repair) an ``mcp-tool`` block's mcp_name +
    tool_name against the live ``inventory`` snapshot.

    ``inventory`` shape: ``{mcp_name: {tool_name: parameters_schema_or_{}}}``.

    Auto-repair: when ``tool_name`` doesn't exist on the MCP but
    ``f"{mcp_name}_{tool_name}"`` does, the block's config is rewritten
    in place. Agno prefixes remote tool names with the MCP name; LLM-
    authored workflows routinely emit the bare upstream name (e.g.
    ``telegram_send_message`` instead of ``messaging_telegram_send_message``).
    Auto-repair eliminates that whole failure class without forcing the
    LLM to re-author.

    ``callability`` (optional, shape:
    ``{mcp_name: {tool_name: bool}}``) is a parallel snapshot built by
    :func:`mcp_callability_from_pool`. When supplied, the resolved tool
    must be invocable — either a raw callable in the toolkit, or an
    Agno ``Function`` descriptor with a non-None ``entrypoint``. A
    ``False`` here means the executor would fail at run-time with
    ``TypeError: 'Function' object is not callable``; we surface it as
    a clear validation error instead.
    """
    mcp_name = config.get("mcp_name")
    tool_name = config.get("tool_name")
    if not isinstance(mcp_name, str) or not isinstance(tool_name, str):
        return  # required-field check elsewhere catches this

    tools = inventory.get(mcp_name)
    if tools is None:
        suggestions = difflib.get_close_matches(mcp_name, inventory.keys(), n=3)
        hint = f" Did you mean: {suggestions}?" if suggestions else ""
        raise ValidationError(
            f"node {node_id}: MCP {mcp_name!r} is not loaded. "
            f"Known MCPs: {sorted(inventory)}.{hint}",
            node_id=node_id,
            field="mcp_name",
        )

    if tool_name not in tools:
        prefixed = f"{mcp_name}_{tool_name}"
        if prefixed in tools:
            # Forward repair: bare upstream name → Agno-prefixed name.
            config["tool_name"] = prefixed
            tool_name = prefixed
        else:
            # Reverse repair: strip a redundant leading mcp_name_ prefix
            # (e.g. shell_shell_exec → shell_exec). Happens when an LLM
            # emits mcp_name + "_" + already-prefixed tool_name.
            mcp_prefix = f"{mcp_name}_"
            if tool_name.startswith(mcp_prefix):
                stripped = tool_name[len(mcp_prefix):]
                if stripped in tools:
                    config["tool_name"] = stripped
                    tool_name = stripped

        if tool_name not in tools:
            suggestions = difflib.get_close_matches(tool_name, tools.keys(), n=3)
            hint = f" Did you mean: {suggestions}?" if suggestions else ""
            raise ValidationError(
                f"node {node_id}: MCP {mcp_name!r} has no tool {tool_name!r}. "
                f"Available: {sorted(tools)}.{hint}",
                node_id=node_id,
                field="tool_name",
            )

    # Callability: catch the case where the toolkit registered a
    # non-callable (e.g. an Agno ``Function`` descriptor whose
    # ``entrypoint`` never got bound). The executor would otherwise
    # raise ``TypeError: 'Function' object is not callable`` mid-DAG.
    if callability is not None:
        per_mcp = callability.get(mcp_name) or {}
        if per_mcp.get(tool_name) is False:
            raise ValidationError(
                f"node {node_id}: tool {mcp_name}.{tool_name} resolved to "
                f"a non-callable in the live MCP pool — likely a stale "
                f"subprocess MCP toolkit registration. Reload the MCP "
                f"and retry.",
                node_id=node_id,
                field="tool_name",
            )

    # Best-effort args sanity: only flag plain-missing required keys.
    # Templated values like ``"{{ctx.inputs.x}}"`` are valid strings
    # and count as present — we don't try to resolve them here.
    args = config.get("args") or {}
    if not isinstance(args, dict):
        return  # type check elsewhere catches this
    schema = tools.get(tool_name) or {}
    required = schema.get("required") if isinstance(schema, dict) else None
    if isinstance(required, list):
        missing = [k for k in required if k not in args]
        if missing:
            raise ValidationError(
                f"node {node_id}: tool {mcp_name}.{tool_name} requires "
                f"args {missing} that are not set",
                node_id=node_id,
                field="args",
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


def validate_graph(
    graph: dict[str, Any],
    *,
    mcp_inventory: dict[str, dict[str, Any]] | None = None,
    mcp_callability: dict[str, dict[str, bool]] | None = None,
) -> None:
    """Validate a ``graph_json`` payload. Raises ``ValidationError`` on
    the first problem found; returns ``None`` on success.

    When ``mcp_inventory`` is supplied (shape:
    ``{mcp_name: {tool_name: parameters_schema}}``) ``mcp-tool`` blocks
    are additionally cross-checked against the loaded MCPs. Tool-name
    prefix mismatches (LLM emitting ``telegram_send_message`` when the
    pool exposes ``messaging_telegram_send_message``) are auto-repaired
    in place. Pass ``None`` (default) on code paths that don't have a
    pool handy — the rest of the validation still runs.

    When ``mcp_callability`` is supplied (shape:
    ``{mcp_name: {tool_name: bool}}``) every ``mcp-tool`` block whose
    resolved tool maps to ``False`` is rejected with an actionable
    error. Built by :func:`mcp_callability_from_pool` alongside the
    inventory snapshot.

    Build the inventory with :func:`mcp_inventory_from_pool` from any
    caller that has an ``MCPPool`` in scope.
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
        # Auto-repair: ``arguments`` → ``args`` for mcp-tool blocks.
        # The MCP JSON-RPC spec uses ``arguments``; LLM-authored
        # workflows routinely emit that name. The block schema's
        # canonical key is ``args``. Repair in place so trace_json,
        # the missing-required-args check below, and the executor all
        # see the canonical form. Same spirit as the tool_name repair.
        if (
            ntype == "mcp-tool"
            and "args" not in config
            and isinstance(config.get("arguments"), dict)
        ):
            config["args"] = config.pop("arguments")
        _check_config(nid, spec, config)
        if ntype == "mcp-tool" and mcp_inventory is not None:
            _check_mcp_tool(nid, config, mcp_inventory, mcp_callability)

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


def mcp_inventory_from_pool(pool: Any) -> dict[str, dict[str, Any]] | None:
    """Snapshot a ``MCPPool`` into the shape ``validate_graph`` expects:
    ``{mcp_name: {tool_name: parameters_schema}}``.

    Returns ``None`` when the pool can't be introspected (``None``, no
    ``list_mcp_tools`` method, exception during enumeration). Callers
    pass this directly to ``validate_graph(graph, mcp_inventory=...)``
    and ``None`` means "skip MCP-existence checks" — the right
    behavior at boot, in tests, or anywhere a pool isn't attached yet.
    A genuine empty pool returns ``{}``, which causes mcp-tool blocks
    to fail validation (no MCPs are loaded, so referencing one is wrong).
    """
    if pool is None:
        return None
    list_fn = getattr(pool, "list_mcp_tools", None)
    if not callable(list_fn):
        return None
    try:
        listing = list_fn()
    except Exception:
        return None
    out: dict[str, dict[str, Any]] = {}
    for entry in listing or []:
        name = entry.get("mcp_name")
        if not isinstance(name, str):
            continue
        tools_meta = entry.get("tools") or []
        out[name] = {
            t["name"]: t.get("parameters_schema") or {}
            for t in tools_meta
            if isinstance(t, dict) and isinstance(t.get("name"), str)
        }
    return out


def mcp_callability_from_pool(pool: Any) -> dict[str, dict[str, bool]] | None:
    """Snapshot a ``MCPPool`` into ``{mcp_name: {tool_name: bool}}``,
    where ``True`` means "the executor can dispatch this tool" and
    ``False`` means "calling it would raise ``TypeError`` mid-DAG".

    A tool is considered callable when the toolkit registers either:
    a raw Python callable (in-process toolkits like ``tool-search``);
    or an Agno ``Function`` descriptor whose ``entrypoint`` field is
    itself callable (the standard subprocess-MCP shape).

    Returns ``None`` when the pool can't be introspected — same
    convention as :func:`mcp_inventory_from_pool`. Pass alongside the
    inventory to :func:`validate_graph`.
    """
    if pool is None:
        return None
    by_name = getattr(pool, "_toolkit_by_name", None)
    if not isinstance(by_name, dict):
        return None
    out: dict[str, dict[str, bool]] = {}
    for mcp_name, toolkit in by_name.items():
        merged = {
            **(getattr(toolkit, "functions", {}) or {}),
            **(getattr(toolkit, "async_functions", {}) or {}),
        }
        per: dict[str, bool] = {}
        for tool_name, fn in merged.items():
            entrypoint = getattr(fn, "entrypoint", None)
            per[tool_name] = bool(callable(entrypoint) or callable(fn))
        out[mcp_name] = per
    return out
