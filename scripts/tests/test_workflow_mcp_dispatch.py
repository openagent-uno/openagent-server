"""Regression: ``mcp-tool`` blocks must dispatch to subprocess MCPs
correctly. Without the fix, ``_h_mcp_tool`` would do ``fn(**args)``
on Agno's ``Function`` Pydantic descriptor (which has no ``__call__``)
and the workflow would die with::

    TypeError: 'Function' object is not callable

The actual callable lives on ``Function.entrypoint``. The same
``getattr(fn, "entrypoint", None) or fn`` pattern is already used by
``tool_search/adapters.py:129`` — the executor was the lone hold-out.

This module also covers the validator side: when a callability snapshot
is supplied, ``validate_graph`` rejects mcp-tool blocks whose tool maps
to a non-callable, so the failure surfaces at workflow-save / run-start
instead of mid-DAG.
"""
from __future__ import annotations

from typing import Any

from ._framework import TestContext, test


# ── Shared stubs ────────────────────────────────────────────────────


class _RecordingAgent:
    """Minimal Agent stub. Workflow executor only pokes ``_mcp`` and
    ``forget_session`` (the latter from _finalize_run; we no-op it).
    """

    def __init__(self, pool: Any | None = None) -> None:
        self._mcp = pool
        self.model = None

    async def forget_session(self, session_id: str) -> None:
        return None

    async def release_session(self, session_id: str) -> None:
        return None


class _StubDB:
    """Minimal MemoryDB shape used by WorkflowExecutor."""

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self._counter = 0

    async def add_workflow_run(
        self,
        *,
        workflow_id: str,
        trigger: str,
        inputs: dict,
        run_id: str | None = None,
    ) -> str:
        self._counter += 1
        rid = run_id or f"run-{self._counter}"
        self.runs[rid] = {
            "id": rid,
            "workflow_id": workflow_id,
            "status": "running",
            "trigger": trigger,
            "inputs": inputs,
            "trace": [],
        }
        return rid

    async def update_workflow_run(self, run_id: str, **kwargs: Any) -> None:
        self.runs[run_id].update(kwargs)

    async def update_workflow(self, workflow_id: str, **kwargs: Any) -> None:
        return None

    async def get_workflow_run(self, run_id: str) -> dict | None:
        return self.runs.get(run_id)


def _agno_function_factory():
    """Return Agno's ``Function`` class. Skip the test when Agno isn't
    importable rather than crashing the suite."""
    from agno.tools.function import Function
    return Function


def _make_pool(toolkits: dict[str, Any]) -> Any:
    """Build a shape-compatible pool stub: ``_toolkit_by_name`` (dict)
    + ``toolkit_by_name`` + ``list_mcp_tools`` so the validator's
    ``mcp_inventory_from_pool`` and the executor's ``_h_mcp_tool``
    both see the same toolkits."""
    class _PoolStub:
        def __init__(self, by_name):
            self._toolkit_by_name = by_name

        def toolkit_by_name(self, name):
            return self._toolkit_by_name.get(name)

        def list_mcp_tools(self):
            out = []
            for name, tk in self._toolkit_by_name.items():
                merged = {
                    **(getattr(tk, "functions", {}) or {}),
                    **(getattr(tk, "async_functions", {}) or {}),
                }
                out.append({
                    "mcp_name": name,
                    "tools": [
                        {
                            "name": tn,
                            "description": getattr(fn, "description", "") or "",
                            "parameters_schema": getattr(fn, "parameters", None) or {},
                        }
                        for tn, fn in merged.items()
                    ],
                })
            return out

    return _PoolStub(toolkits)


class _ToolkitStub:
    """Mimics Agno's ``MCPTools`` shape for our purposes: ``functions``
    dict (subprocess MCPs) + ``async_functions`` dict (in-process
    toolkits). The executor's ``_h_mcp_tool`` merges both, exactly
    like the live pool does."""

    def __init__(
        self,
        *,
        functions: dict[str, Any] | None = None,
        async_functions: dict[str, Any] | None = None,
    ) -> None:
        self.functions = functions or {}
        self.async_functions = async_functions or {}


# ── Direct executor tests (per-handler) ─────────────────────────────


@test("workflow_mcp_dispatch", "mcp-tool calls Agno Function.entrypoint")
async def t_function_entrypoint_called(ctx: TestContext) -> None:
    """Regression for ``TypeError: 'Function' object is not callable``.
    Without the executor fix, calling ``fn(**args)`` on a Pydantic
    ``Function`` blows up here. With the fix, the call routes to
    ``fn.entrypoint``."""
    from openagent.workflow.executor import (
        WorkflowExecutor, _RunCtx, _h_mcp_tool,
    )

    Function = _agno_function_factory()

    seen: dict[str, Any] = {}

    async def entrypoint(**kwargs: Any) -> str:
        seen["kwargs"] = kwargs
        return "ok"

    fn = Function(name="messaging_telegram_send_message", entrypoint=entrypoint)
    toolkit = _ToolkitStub(functions={fn.name: fn})
    pool = _make_pool({"messaging": toolkit})

    executor = WorkflowExecutor(agent=_RecordingAgent(pool=pool), db=_StubDB())  # type: ignore[arg-type]
    run_ctx = _RunCtx(run_id="r", workflow_id="w", inputs={}, vars={})

    result = await _h_mcp_tool(
        executor,
        {"id": "n1", "type": "mcp-tool"},
        {
            "mcp_name": "messaging",
            "tool_name": "messaging_telegram_send_message",
            "args": {"chat_id": "123", "text": "hi"},
        },
        run_ctx,
    )

    assert result == {"result": "ok"}, result
    assert seen["kwargs"] == {"chat_id": "123", "text": "hi"}, seen


@test("workflow_mcp_dispatch", "mcp-tool falls back to raw async callable")
async def t_async_function_raw_callable(ctx: TestContext) -> None:
    """In-process toolkits (e.g. ``tool-search``) register raw
    callables in ``async_functions`` — no ``Function`` wrapper. The
    ``or fn`` fallback in the executor must still invoke them."""
    from openagent.workflow.executor import (
        WorkflowExecutor, _RunCtx, _h_mcp_tool,
    )

    seen: dict[str, Any] = {}

    async def call_tool(**kwargs: Any) -> str:
        seen["kwargs"] = kwargs
        return "raw-ok"

    toolkit = _ToolkitStub(async_functions={"tool_search_call_tool": call_tool})
    pool = _make_pool({"tool-search": toolkit})

    executor = WorkflowExecutor(agent=_RecordingAgent(pool=pool), db=_StubDB())  # type: ignore[arg-type]
    run_ctx = _RunCtx(run_id="r", workflow_id="w", inputs={}, vars={})

    result = await _h_mcp_tool(
        executor,
        {"id": "n1", "type": "mcp-tool"},
        {
            "mcp_name": "tool-search",
            "tool_name": "tool_search_call_tool",
            "args": {"server": "x", "tool": "y", "args": {}},
        },
        run_ctx,
    )

    assert result == {"result": "raw-ok"}, result
    assert seen["kwargs"] == {"server": "x", "tool": "y", "args": {}}, seen


@test("workflow_mcp_dispatch", "mcp-tool handles sync callable without await")
async def t_sync_callable_no_await(ctx: TestContext) -> None:
    """A sync entrypoint returns a non-awaitable; the
    ``inspect.isawaitable`` branch must skip cleanly."""
    from openagent.workflow.executor import (
        WorkflowExecutor, _RunCtx, _h_mcp_tool,
    )

    Function = _agno_function_factory()

    def entrypoint(**kwargs: Any) -> dict:
        return {"echo": kwargs}

    fn = Function(name="util_echo", entrypoint=entrypoint)
    toolkit = _ToolkitStub(functions={fn.name: fn})
    pool = _make_pool({"util": toolkit})

    executor = WorkflowExecutor(agent=_RecordingAgent(pool=pool), db=_StubDB())  # type: ignore[arg-type]
    run_ctx = _RunCtx(run_id="r", workflow_id="w", inputs={}, vars={})

    result = await _h_mcp_tool(
        executor,
        {"id": "n1", "type": "mcp-tool"},
        {
            "mcp_name": "util",
            "tool_name": "util_echo",
            "args": {"a": 1},
        },
        run_ctx,
    )

    assert result == {"result": {"echo": {"a": 1}}}, result


# ── Validator tests ─────────────────────────────────────────────────


@test("workflow_mcp_dispatch", "validate_graph rejects non-callable Function")
async def t_validate_rejects_uncallable_function(ctx: TestContext) -> None:
    """When a callability snapshot says a tool is non-callable,
    ``validate_graph`` must reject the graph before it ever runs."""
    from openagent.workflow.validate import ValidationError, validate_graph

    graph = {
        "version": 1,
        "nodes": [
            {
                "id": "n1",
                "type": "mcp-tool",
                "config": {
                    "mcp_name": "messaging",
                    "tool_name": "messaging_telegram_send_message",
                    "args": {"chat_id": "1", "text": "x"},
                },
            },
        ],
        "edges": [],
        "variables": {},
    }
    inventory = {
        "messaging": {"messaging_telegram_send_message": {}},
    }
    callability = {
        "messaging": {"messaging_telegram_send_message": False},
    }

    raised = False
    try:
        validate_graph(
            graph,
            mcp_inventory=inventory,
            mcp_callability=callability,
        )
    except ValidationError as exc:
        raised = True
        assert "non-callable" in str(exc), str(exc)
        assert exc.node_id == "n1", exc.node_id
        assert exc.field == "tool_name", exc.field
    assert raised, "validate_graph should have raised on non-callable tool"


@test("workflow_mcp_dispatch", "mcp_callability_from_pool snapshots the live pool")
async def t_callability_snapshot_shape(ctx: TestContext) -> None:
    """The helper must mark Function-with-entrypoint and raw callables
    as ``True``, and Function-without-entrypoint as ``False``."""
    from openagent.workflow.validate import mcp_callability_from_pool

    Function = _agno_function_factory()

    async def entrypoint(**_: Any) -> str:
        return ""

    good = Function(name="good_tool", entrypoint=entrypoint)
    bad = Function(name="bad_tool")  # entrypoint defaults to None
    toolkit_sub = _ToolkitStub(functions={good.name: good, bad.name: bad})

    async def raw(**_: Any) -> str:
        return ""

    toolkit_inproc = _ToolkitStub(async_functions={"raw_tool": raw})

    pool = _make_pool({"sub": toolkit_sub, "inproc": toolkit_inproc})
    snapshot = mcp_callability_from_pool(pool)

    assert snapshot is not None, snapshot
    assert snapshot["sub"]["good_tool"] is True, snapshot["sub"]
    assert snapshot["sub"]["bad_tool"] is False, snapshot["sub"]
    assert snapshot["inproc"]["raw_tool"] is True, snapshot["inproc"]


# ── End-to-end: executor re-validates at run-start ──────────────────


@test(
    "workflow_mcp_dispatch",
    "executor re-validates at run-start with live pool",
)
async def t_executor_revalidates_at_run_start(ctx: TestContext) -> None:
    """Workflows authored via the workflow-manager subprocess validate
    *without* the live pool. The executor must re-validate at run-start
    so a stale tool reference (or the Function-not-callable bug class)
    fails as a finalized ``failed`` run with a clear error in
    ``trace_json``, not a mid-DAG ``TypeError``."""
    from openagent.workflow.executor import WorkflowExecutor

    Function = _agno_function_factory()

    bad = Function(name="messaging_telegram_send_message")  # no entrypoint
    toolkit = _ToolkitStub(functions={bad.name: bad})
    pool = _make_pool({"messaging": toolkit})

    executor = WorkflowExecutor(agent=_RecordingAgent(pool=pool), db=_StubDB())  # type: ignore[arg-type]

    workflow = {
        "id": "wf-1",
        "name": "broken",
        "graph": {
            "version": 1,
            "nodes": [
                {
                    "id": "n2",
                    "type": "mcp-tool",
                    "config": {
                        "mcp_name": "messaging",
                        "tool_name": "messaging_telegram_send_message",
                        "args": {"chat_id": "1", "text": "x"},
                    },
                },
            ],
            "edges": [],
            "variables": {},
        },
    }

    final = await executor.run(workflow, trigger="manual")

    assert final["status"] == "failed", final
    error = final.get("error") or ""
    assert "non-callable" in error, error
    assert "messaging" in error, error
    # Crucially NOT the opaque mid-DAG TypeError we used to get.
    assert "'Function' object is not callable" not in error, error


@test(
    "workflow_mcp_dispatch",
    "validate_graph auto-repairs double-prefixed tool names (shell_shell_exec → shell_exec)",
)
async def t_validate_repairs_double_prefix(ctx: TestContext) -> None:
    """When an LLM emits mcp_name + '_' + already-prefixed tool_name
    (e.g. tool_name='shell_shell_exec' for MCP 'shell'), the validator
    must strip the redundant prefix and repair in place rather than
    raising a ValidationError. Regression for the mixout Git Sync
    workflow that kept crashing with ValidationError: shell_shell_exec."""
    from openagent.workflow.validate import validate_graph

    config = {
        "mcp_name": "shell",
        "tool_name": "shell_shell_exec",  # double-prefixed — the bug
        "args": {},
    }
    graph = {
        "version": 1,
        "nodes": [{"id": "n1", "type": "mcp-tool", "config": config}],
        "edges": [],
        "variables": {},
    }
    inventory = {
        "shell": {
            "shell_exec": {},
            "shell_input": {},
            "shell_kill": {},
        },
    }

    # Must not raise — double-prefix should be silently repaired.
    validate_graph(graph, mcp_inventory=inventory)

    # Verify the config was repaired in-place.
    assert config["tool_name"] == "shell_exec", config["tool_name"]
