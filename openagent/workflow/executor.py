"""WorkflowExecutor — drives a workflow graph through one run.

Responsibilities:

- Create a ``workflow_runs`` row at start, flip it to ``success`` /
  ``failed`` at the end, append per-block entries to ``trace_json`` so
  the UI and the AI see progress incrementally.
- Walk the DAG: find entry nodes (trigger nodes + orphans with
  in-degree 0), run them, enqueue successors whose incoming edges are
  all satisfied or skipped.
- Route by ``sourceHandle``: an ``if`` block takes only its ``true``
  or ``false`` handle; a ``loop`` takes ``body`` for each iteration
  and ``done`` after the last one; ``parallel`` takes all ``branch_*``
  handles concurrently.
- Dispatch each node to its handler. Templating: every string in
  ``config`` goes through ``resolve_templates`` with a ``ctx`` that
  includes ``inputs``, ``vars``, ``nodes.<id>.output``, ``now``,
  ``run_id``.
- Error handling: ``config.on_error`` defaults to ``halt`` — any block
  failure marks the run failed and stops. ``continue`` records the
  error but keeps walking. ``branch`` routes to an ``error`` handle if
  one exists.
- Concurrency: batches of currently-ready nodes run in parallel via
  ``asyncio.gather``, so ``parallel`` blocks really do fan out. A
  per-workflow lock prevents overlapping runs of the *same* workflow.

The executor does **not** run in the workflow-manager MCP subprocess —
it lives in the main OpenAgent process so it can call ``agent.run()``
and touch the live ``MCPPool``. The MCP subprocess hands work over
via the ``workflow_run_requests`` queue table.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from openagent.memory.db import MemoryDB
from openagent.workflow.blocks import BLOCK_CATALOG, get_block_spec
from openagent.workflow.templating import (
    TemplateError,
    evaluate_expression,
    resolve_templates,
)
from openagent.workflow.validate import validate_graph

logger = logging.getLogger(__name__)

StatusCallback = Callable[[str], Awaitable[None]]


class WorkflowExecutionError(RuntimeError):
    """Raised when a block fails and ``on_error='halt'`` is in effect.
    The ``run_row`` attribute is the final persisted row so the caller
    can report without a re-fetch."""

    def __init__(self, message: str, run_row: dict | None = None):
        super().__init__(message)
        self.run_row = run_row


# Sentinel wrapping a handler's output when the handler wants to steer
# edge routing. ``taken`` is the set of ``sourceHandle`` names whose
# outgoing edges the walker should treat as satisfied. Edges whose
# handle isn't in the set are marked skipped — dead paths cascade
# downstream so merges don't block forever and unreachable branches
# surface as ``status='skipped'`` in the trace.
@dataclass
class NodeResult:
    output: Any
    taken: frozenset[str] | None = None  # None → every outgoing edge taken


@dataclass
class _RunCtx:
    run_id: str
    workflow_id: str
    inputs: dict[str, Any]
    vars: dict[str, Any]
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    # Populated by the walker at start of each run so handlers (merge
    # in particular) can introspect incoming edges without pulling the
    # graph from the gateway.
    graph: dict[str, Any] | None = None
    # When we're inside a loop iteration, the parent ctx's trace/vars
    # are reached via ``parent``. ``nodes`` is fresh per iteration so
    # iteration N doesn't leak results into iteration N+1.
    parent: "_RunCtx | None" = None

    def to_template_ctx(self) -> dict[str, Any]:
        return {
            "inputs": self.inputs,
            "vars": self.vars,
            "nodes": self.nodes,
            "now": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
        }


class WorkflowExecutor:
    """Runs workflows against the live Agent and MCPPool. One instance
    per main process is enough — per-run state lives in ``_RunCtx``.
    """

    def __init__(self, agent: Any, db: MemoryDB):
        self.agent = agent
        self.db = db
        self._locks: dict[str, asyncio.Lock] = {}
        self._trace_lock = asyncio.Lock()

    # ── public API ──────────────────────────────────────────────────

    async def run(
        self,
        workflow: dict,
        *,
        trigger: str = "manual",
        inputs: dict[str, Any] | None = None,
        on_status: StatusCallback | None = None,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        graph = workflow.get("graph") or {}
        validate_graph(graph)

        workflow_id = workflow["id"]
        lock = self._locks.setdefault(workflow_id, asyncio.Lock())

        async with lock:
            run_id = await self.db.add_workflow_run(
                workflow_id=workflow_id,
                trigger=trigger,
                inputs=inputs or {},
                run_id=run_id,
            )
            ctx = _RunCtx(
                run_id=run_id,
                workflow_id=workflow_id,
                inputs=dict(inputs or {}),
                vars=dict(graph.get("variables") or {}),
                graph=graph,
            )

            if on_status is not None:
                try:
                    await on_status(f"workflow.{workflow['name']} started")
                except Exception:  # noqa: BLE001
                    pass

            try:
                await self._walk(graph, ctx, on_status)
            except WorkflowExecutionError as exc:
                await self._finalize_run(
                    ctx, status="failed", error=str(exc),
                )
            except Exception as exc:  # noqa: BLE001 — unexpected
                logger.exception("workflow %s crashed", workflow_id)
                await self._finalize_run(
                    ctx, status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            else:
                outputs = self._collect_outputs(graph, ctx)
                await self._finalize_run(
                    ctx, status="success", outputs=outputs,
                )

            final = await self.db.get_workflow_run(run_id)
            if final is None:
                return {
                    "id": run_id,
                    "workflow_id": workflow_id,
                    "status": "failed",
                    "error": "run row vanished",
                    "trace": ctx.trace,
                }
            return final

    # ── DAG walk ────────────────────────────────────────────────────

    async def _walk(
        self,
        graph: dict,
        ctx: _RunCtx,
        on_status: StatusCallback | None,
    ) -> None:
        """Batch-parallel walker with per-edge routing.

        Edges start ``pending``. When a node runs, its ``taken``
        sourceHandles turn matching edges into ``satisfied``; the
        rest become ``skipped``. A node becomes runnable when ALL its
        incoming edges are resolved and AT LEAST ONE is satisfied; a
        node whose incoming edges are all skipped is marked ``dead``
        and its outgoing edges cascade-skip.

        This naturally handles ``if`` (one branch satisfied, the other
        skipped+cascaded), ``parallel`` (all branches satisfied, run
        concurrently on the next tick), ``merge`` (waits for upstream,
        handles partial-skip), and plain linear flow.
        """
        nodes_by_id = {n["id"]: n for n in graph.get("nodes", [])}
        edges = list(graph.get("edges", []))

        # Adjacency + predecessor counts.
        outgoing: dict[str, list[dict]] = {nid: [] for nid in nodes_by_id}
        incoming: dict[str, list[dict]] = {nid: [] for nid in nodes_by_id}
        for e in edges:
            if e["source"] in outgoing:
                outgoing[e["source"]].append(e)
            if e["target"] in incoming:
                incoming[e["target"]].append(e)

        in_degree = {nid: len(incoming[nid]) for nid in nodes_by_id}
        waiting_on = dict(in_degree)
        sat_count = {nid: 0 for nid in nodes_by_id}
        completed: set[str] = set()
        dead: set[str] = set()

        ready: list[str] = [nid for nid, deg in in_degree.items() if deg == 0]

        async def _resolve_edge(src: str, edge: dict, satisfied: bool) -> None:
            """Mark ``edge`` as satisfied or skipped; promote target to
            ready or dead when all incoming edges are resolved."""
            tgt = edge["target"]
            if tgt in completed or tgt in dead:
                return
            waiting_on[tgt] -= 1
            if satisfied:
                sat_count[tgt] += 1
            if waiting_on[tgt] != 0:
                return
            if sat_count[tgt] > 0:
                if tgt not in ready:
                    ready.append(tgt)
            else:
                dead.add(tgt)
                ctx.trace.append({
                    "node_id": tgt,
                    "type": nodes_by_id[tgt].get("type"),
                    "status": "skipped",
                    "started_at": None,
                    "finished_at": None,
                    "input": None,
                    "output": None,
                    "error": None,
                })
                ctx.nodes[tgt] = {"output": None, "status": "skipped"}
                # Cascade — every outgoing edge is skipped.
                for e2 in outgoing.get(tgt, []):
                    await _resolve_edge(tgt, e2, satisfied=False)

        while ready:
            # Snapshot the ready set; clear so we can re-populate as
            # downstream nodes become unblocked during processing.
            batch = [nid for nid in ready if nid not in completed and nid not in dead]
            ready.clear()
            if not batch:
                break

            # Run every ready node concurrently. Exceptions inside
            # _run_node are turned into NodeResult-or-skip signals;
            # halt-mode failures re-raise to abort the whole walk.
            results = await asyncio.gather(
                *[self._run_node(nodes_by_id[nid], ctx, on_status) for nid in batch],
                return_exceptions=True,
            )

            # Halt-mode failure surfaces as WorkflowExecutionError;
            # any other exception is a handler bug — re-raise both so
            # the run is marked ``failed`` cleanly.
            for nid, res in zip(batch, results):
                if isinstance(res, WorkflowExecutionError):
                    raise res
                if isinstance(res, Exception):
                    raise WorkflowExecutionError(
                        f"block {nid!r} raised {type(res).__name__}: {res}",
                    ) from res

            # Apply routing + mark completed in a serial pass so the
            # waiting_on / sat_count state stays consistent.
            for nid, res in zip(batch, results):
                completed.add(nid)
                taken: frozenset[str] | None = res if isinstance(res, frozenset) else None
                for e in outgoing.get(nid, []):
                    sh = e.get("sourceHandle") or "out"
                    satisfied = taken is None or sh in taken
                    await _resolve_edge(nid, e, satisfied)

        # Anything that never ran and wasn't marked dead (orphan
        # subgraphs) gets recorded as skipped so the trace matches
        # the graph shape 1:1.
        recorded = {e["node_id"] for e in ctx.trace}
        for nid in nodes_by_id:
            if nid in completed or nid in dead:
                continue
            if nid in recorded:
                continue
            ctx.trace.append({
                "node_id": nid,
                "type": nodes_by_id[nid].get("type"),
                "status": "skipped",
                "started_at": None,
                "finished_at": None,
                "input": None,
                "output": None,
                "error": None,
            })

    async def _run_node(
        self,
        node: dict,
        ctx: _RunCtx,
        on_status: StatusCallback | None,
    ) -> frozenset[str] | None:
        """Execute one block. Returns a frozenset of taken sourceHandle
        names (routing), or ``None`` for "take every outgoing edge".
        Raises ``WorkflowExecutionError`` when ``on_error='halt'`` and
        the block fails."""
        node_id = node["id"]
        ntype = node["type"]
        handler = _HANDLERS.get(ntype)
        if handler is None:
            raise WorkflowExecutionError(
                f"node {node_id}: no handler for block type {ntype!r}"
            )

        if on_status is not None:
            try:
                await on_status(f"{ntype} ({node.get('label') or node_id}) running")
            except Exception:  # noqa: BLE001
                pass

        entry: dict[str, Any] = {
            "node_id": node_id,
            "type": ntype,
            "started_at": time.time(),
            "finished_at": None,
            "status": "running",
            "input": None,
            "output": None,
            "error": None,
        }
        ctx.trace.append(entry)
        await self._persist_trace(ctx)

        raw_config = node.get("config") or {}
        template_ctx = ctx.to_template_ctx()
        try:
            resolved_config = resolve_templates(raw_config, template_ctx)
            entry["input"] = resolved_config
            handler_out = await handler(self, node, resolved_config, ctx)
        except Exception as exc:  # noqa: BLE001
            on_error = raw_config.get("on_error", "halt")
            entry["status"] = "failed"
            entry["finished_at"] = time.time()
            entry["error"] = f"{type(exc).__name__}: {exc}"
            ctx.nodes[node_id] = {
                "output": None,
                "status": "failed",
                "error": entry["error"],
            }
            await self._persist_trace(ctx)
            if on_error == "halt":
                raise WorkflowExecutionError(
                    f"block {node_id!r} failed: {entry['error']}"
                ) from exc
            if on_error == "branch":
                # Route only via the 'error' handle; all other outgoing
                # edges skip so normal downstream doesn't run.
                return frozenset({"error"})
            # 'continue' — keep walking. Downstream sees the failure
            # via ctx.nodes[<id>].status='failed' but normal edges are
            # still satisfied so the workflow doesn't stall.
            return None

        # Normalize output vs routing-decision from handlers.
        if isinstance(handler_out, NodeResult):
            output = handler_out.output
            taken = handler_out.taken
        else:
            output = handler_out
            taken = None

        entry["status"] = "success"
        entry["finished_at"] = time.time()
        entry["output"] = output
        ctx.nodes[node_id] = {"output": output, "status": "success"}
        await self._persist_trace(ctx)
        return taken

    # ── persistence helpers ─────────────────────────────────────────

    async def _persist_trace(self, ctx: _RunCtx) -> None:
        async with self._trace_lock:
            await self.db.update_workflow_run(ctx.run_id, trace=ctx.trace)

    async def _finalize_run(
        self,
        ctx: _RunCtx,
        *,
        status: str,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        now = time.time()
        kwargs: dict[str, Any] = {
            "status": status,
            "finished_at": now,
            "trace": ctx.trace,
        }
        if outputs is not None:
            kwargs["outputs"] = outputs
        if error is not None:
            kwargs["error"] = error
        await self.db.update_workflow_run(ctx.run_id, **kwargs)
        await self.db.update_workflow(ctx.workflow_id, last_run_at=now)

    def _collect_outputs(self, graph: dict, ctx: _RunCtx) -> dict[str, Any]:
        nodes = {n["id"]: n for n in graph.get("nodes", [])}
        edges = graph.get("edges", [])
        with_outgoing = {e["source"] for e in edges}
        terminals = [nid for nid in nodes if nid not in with_outgoing]
        if not terminals:
            return {}
        if len(terminals) == 1:
            nid = terminals[0]
            out = ctx.nodes.get(nid, {}).get("output")
            return {"value": out, "terminal_node": nid}
        return {
            nid: ctx.nodes.get(nid, {}).get("output")
            for nid in terminals
        }

    # ── subgraph runner (used by loop) ──────────────────────────────

    async def _run_subgraph(
        self,
        subgraph: dict,
        parent_ctx: _RunCtx,
        on_status: StatusCallback | None,
    ) -> dict[str, Any]:
        """Run a body subgraph (used by ``loop``). Shares ``vars`` with
        the parent by reference so accumulators work, but has a fresh
        ``nodes`` map so successive iterations don't see each other's
        outputs. Trace entries bubble up into the parent trace."""
        child_ctx = _RunCtx(
            run_id=parent_ctx.run_id,
            workflow_id=parent_ctx.workflow_id,
            inputs=parent_ctx.inputs,
            vars=parent_ctx.vars,  # shared by reference
            nodes={},
            trace=parent_ctx.trace,  # shared trace
            graph=subgraph,
            parent=parent_ctx,
        )
        await self._walk(subgraph, child_ctx, on_status)
        # Return the terminal node's output so the loop handler can
        # accumulate it.
        outputs = self._collect_outputs(subgraph, child_ctx)
        return outputs.get("value") if "value" in outputs else outputs


# ── block handlers ──────────────────────────────────────────────────


async def _h_trigger_manual(
    exe: WorkflowExecutor, node: dict, cfg: dict, ctx: _RunCtx,
) -> dict[str, Any]:
    return dict(ctx.inputs)


async def _h_trigger_ai(
    exe: WorkflowExecutor, node: dict, cfg: dict, ctx: _RunCtx,
) -> dict[str, Any]:
    return dict(ctx.inputs)


async def _h_trigger_schedule(
    exe: WorkflowExecutor, node: dict, cfg: dict, ctx: _RunCtx,
) -> dict[str, Any]:
    return {"triggered_at": datetime.now(timezone.utc).isoformat()}


async def _h_set_variable(
    exe: WorkflowExecutor, node: dict, cfg: dict, ctx: _RunCtx,
) -> dict[str, Any]:
    key = cfg.get("key")
    if not key:
        raise ValueError("set-variable: 'key' is required")
    expr = cfg.get("value_expr", "")
    template_ctx = ctx.to_template_ctx()
    value = evaluate_expression(expr, template_ctx) if expr else None
    ctx.vars[str(key)] = value
    return {"key": key, "value": value}


async def _h_mcp_tool(
    exe: WorkflowExecutor, node: dict, cfg: dict, ctx: _RunCtx,
) -> dict[str, Any]:
    mcp_name = cfg.get("mcp_name")
    tool_name = cfg.get("tool_name")
    args = cfg.get("args") or {}
    if not mcp_name or not tool_name:
        raise ValueError(
            "mcp-tool: both 'mcp_name' and 'tool_name' are required"
        )
    pool = getattr(exe.agent, "_mcp", None)
    if pool is None:
        raise RuntimeError("mcp-tool: agent has no MCP pool attached")
    toolkit = pool.toolkit_by_name(mcp_name)
    if toolkit is None:
        raise RuntimeError(
            f"mcp-tool: MCP {mcp_name!r} is not loaded. Known MCPs: "
            f"{sorted(pool._toolkit_by_name)}"
        )
    functions = getattr(toolkit, "functions", {}) or {}
    fn = functions.get(tool_name)
    if fn is None:
        raise RuntimeError(
            f"mcp-tool: MCP {mcp_name!r} has no tool {tool_name!r}. "
            f"Available: {sorted(functions)}"
        )
    if not isinstance(args, dict):
        raise ValueError(
            f"mcp-tool: 'args' must be an object, got {type(args).__name__}"
        )
    result = fn(**args)
    if inspect.isawaitable(result):
        result = await result
    return {"result": _coerce_to_jsonable(result)}


async def _h_ai_prompt(
    exe: WorkflowExecutor, node: dict, cfg: dict, ctx: _RunCtx,
) -> dict[str, Any]:
    prompt = cfg.get("prompt")
    if not prompt:
        raise ValueError("ai-prompt: 'prompt' is required")
    policy = cfg.get("session_policy", "ephemeral")
    session_id = (
        f"workflow:{ctx.workflow_id}:{ctx.run_id}"
        if policy == "shared"
        else f"workflow:{ctx.workflow_id}:{ctx.run_id}:{node['id']}"
    )
    override_id = cfg.get("model_override")
    override = None
    if override_id:
        smart = getattr(exe.agent, "model", None)
        if smart is not None and hasattr(smart, "build_override_model"):
            override = smart.build_override_model(override_id)
        else:
            logger.warning(
                "ai-prompt: model_override=%r requested but active model "
                "%r does not support overrides; using default",
                override_id, type(smart).__name__,
            )
    try:
        text = await exe.agent.run(
            message=str(prompt),
            user_id="workflow",
            session_id=session_id,
            model_override=override,
        )
    finally:
        release = getattr(exe.agent, "release_session", None)
        if callable(release):
            maybe = release(session_id)
            if inspect.isawaitable(maybe):
                try:
                    await maybe
                except Exception:  # noqa: BLE001
                    logger.debug("release_session failed for %s", session_id)
    return {"text": text}


async def _h_if(
    exe: WorkflowExecutor, node: dict, cfg: dict, ctx: _RunCtx,
) -> NodeResult:
    expr = cfg.get("expression")
    if not expr:
        raise ValueError("if: 'expression' is required")
    template_ctx = ctx.to_template_ctx()
    try:
        value = evaluate_expression(str(expr), template_ctx)
    except TemplateError as e:
        raise ValueError(f"if: expression failed: {e}") from e
    taken = "true" if value else "false"
    return NodeResult(
        output={"result": bool(value), "branch": taken},
        taken=frozenset({taken}),
    )


async def _h_wait(
    exe: WorkflowExecutor, node: dict, cfg: dict, ctx: _RunCtx,
) -> dict[str, Any]:
    mode = cfg.get("mode") or "duration"
    if mode == "duration":
        seconds = float(cfg.get("seconds") or 0)
        if seconds < 0:
            raise ValueError("wait: duration must be non-negative")
        start = time.time()
        if seconds > 0:
            await asyncio.sleep(seconds)
        return {"waited_ms": int((time.time() - start) * 1000)}
    if mode == "until":
        raw = cfg.get("until_iso")
        if not raw:
            raise ValueError("wait: 'until_iso' is required for mode='until'")
        try:
            target = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(f"wait: invalid until_iso {raw!r}: {e}") from e
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = (target - now).total_seconds()
        start = time.time()
        if delta > 0:
            await asyncio.sleep(delta)
        return {"waited_ms": int((time.time() - start) * 1000)}
    raise ValueError(f"wait: unknown mode {mode!r}")


async def _h_parallel(
    exe: WorkflowExecutor, node: dict, cfg: dict, ctx: _RunCtx,
) -> dict[str, Any]:
    """Fan-out node. The walker's default routing (``taken=None``)
    satisfies every outgoing edge — ``branch_0``, ``branch_1``, … all
    become runnable in the next tick and run concurrently via
    ``asyncio.gather`` in ``_walk``.
    """
    return {"branches": int(cfg.get("branches") or 2)}


async def _h_merge(
    exe: WorkflowExecutor, node: dict, cfg: dict, ctx: _RunCtx,
) -> dict[str, Any]:
    strategy = cfg.get("strategy") or "all"
    # Identify upstream nodes via the graph's incoming edges to this
    # merge node. Inputs whose upstream was skipped come through as
    # None so strategy='first' can pick the earliest satisfied value.
    graph = ctx.graph or {}
    edges = graph.get("edges", [])
    my_id = node["id"]
    upstream_outputs: list[Any] = []
    for e in edges:
        if e.get("target") != my_id:
            continue
        src = e.get("source")
        upstream_state = ctx.nodes.get(src, {})
        if upstream_state.get("status") == "success":
            upstream_outputs.append(upstream_state.get("output"))
    if strategy == "first":
        return {"collected": upstream_outputs[0] if upstream_outputs else None}
    if strategy == "last":
        return {"collected": upstream_outputs[-1] if upstream_outputs else None}
    # default 'all'
    key = cfg.get("collect_as") or "collected"
    return {key: upstream_outputs}


async def _h_loop(
    exe: WorkflowExecutor, node: dict, cfg: dict, ctx: _RunCtx,
) -> NodeResult:
    items_expr = cfg.get("items_expr")
    if not items_expr:
        raise ValueError("loop: 'items_expr' is required")
    max_iter = int(cfg.get("max_iterations") or 100)
    iteration_var = cfg.get("iteration_var") or "item"

    template_ctx = ctx.to_template_ctx()
    items = evaluate_expression(str(items_expr), template_ctx)
    if items is None:
        items = []
    if not isinstance(items, (list, tuple)):
        raise ValueError(
            f"loop: items_expr must resolve to a list, got {type(items).__name__}"
        )

    subgraph = _extract_body_subgraph(ctx.graph or {}, node["id"])

    results: list[Any] = []
    for i, item in enumerate(items):
        if i >= max_iter:
            break
        ctx.vars[str(iteration_var)] = item
        ctx.vars["_iteration_index"] = i
        output = await exe._run_subgraph(subgraph, ctx, on_status=None)
        results.append(output)
    # Route via 'done' handle once all iterations finish.
    return NodeResult(
        output={"results": results, "iterations": len(results)},
        taken=frozenset({"done"}),
    )


async def _h_http_request(
    exe: WorkflowExecutor, node: dict, cfg: dict, ctx: _RunCtx,
) -> dict[str, Any]:
    method = (cfg.get("method") or "GET").upper()
    url = cfg.get("url")
    if not url:
        raise ValueError("http-request: 'url' is required")
    headers = cfg.get("headers") or {}
    body = cfg.get("body")
    timeout_s = float(cfg.get("timeout_s") or 30)
    # aiohttp is a gateway dep already, so no new package needed.
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.request(
            method, str(url), headers=headers, data=body,
        ) as resp:
            text = await resp.text()
            result: dict[str, Any] = {
                "status": resp.status,
                "headers": dict(resp.headers),
                "body": text,
            }
            ctype = (resp.headers.get("content-type") or "").lower()
            if "json" in ctype:
                try:
                    result["json"] = json.loads(text)
                except ValueError:
                    pass
            return result


_HANDLERS: dict[str, Callable[..., Awaitable[Any]]] = {
    "trigger-manual": _h_trigger_manual,
    "trigger-schedule": _h_trigger_schedule,
    "trigger-ai": _h_trigger_ai,
    "set-variable": _h_set_variable,
    "mcp-tool": _h_mcp_tool,
    "ai-prompt": _h_ai_prompt,
    "if": _h_if,
    "wait": _h_wait,
    "parallel": _h_parallel,
    "merge": _h_merge,
    "loop": _h_loop,
    "http-request": _h_http_request,
}


# ── helpers ─────────────────────────────────────────────────────────


def _extract_body_subgraph(graph: dict, loop_id: str) -> dict:
    """Return the subgraph reachable from the ``loop`` node's ``body``
    handle, excluding back-edges to the loop itself. Body nodes share
    ids with the outer graph — nested loops get their own subgraph
    recursively at execution time, not at extraction time.
    """
    nodes = graph.get("nodes") or []
    edges = graph.get("edges") or []
    nodes_by_id = {n["id"]: n for n in nodes}

    body_roots = [
        e["target"]
        for e in edges
        if e.get("source") == loop_id
        and (e.get("sourceHandle") or "out") == "body"
    ]
    if not body_roots:
        return {"nodes": [], "edges": [], "version": graph.get("version", 1), "variables": {}}

    reachable: set[str] = set()
    stack = list(body_roots)
    while stack:
        nid = stack.pop()
        if nid == loop_id:
            continue
        if nid in reachable:
            continue
        reachable.add(nid)
        for e in edges:
            if e.get("source") != nid:
                continue
            tgt = e.get("target")
            if tgt and tgt != loop_id and tgt not in reachable:
                stack.append(tgt)

    body_edges = [
        e for e in edges
        if e.get("source") in reachable and e.get("target") in reachable
    ]
    body_nodes = [nodes_by_id[nid] for nid in reachable if nid in nodes_by_id]
    return {
        "version": graph.get("version", 1),
        "nodes": body_nodes,
        "edges": body_edges,
        "variables": {},  # shared with parent via ctx.vars reference
    }


def _coerce_to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _coerce_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_coerce_to_jsonable(v) for v in value]
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:  # noqa: BLE001
        return repr(value)
