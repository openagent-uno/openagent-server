"""Regression: a workflow ``mcp-tool`` block that calls
``delegation.delegate_task`` must run with a live delegation context.

The delegation MCP handler resolves "the current session / pool / db /
dispatcher" from process-wide ``contextvars`` that the agent runtime
binds around each turn (``agent.run`` → ``install_delegation_context``).
Workflow ``mcp-tool`` blocks reach the in-process tool *directly* via
``_h_mcp_tool`` — they never go through ``agent.run`` — so without the
executor installing the same context, ``delegate_task`` short-circuits
with::

    delegate_task called outside an agent turn. ... the runtime didn't
    install a delegation context.

That is exactly the failure a "support coverage" workflow hit: it
delegates to sub-agents (team members) from ``mcp-tool`` blocks. The fix
installs the delegation context for the whole run in
``WorkflowExecutor.run`` (mirroring ``agent.run``). Vision §8 makes a
workflow that delegates to sub-agents the *preferred* shape, so this is
the supported path — not an escape hatch.

These tests pin:
1. delegate_task succeeds end-to-end through ``executor.run`` (context
   present → dispatcher.run_delegated is reached).
2. The same handler, called *without* the executor wrapper, still errors
   — proving the context (not the block plumbing) is what the fix adds.
3. The context is torn down after the run (no contextvar leak).
"""
from __future__ import annotations

from typing import Any

from ._framework import TestContext, test
from .test_workflow_mcp_dispatch import _StubDB, _ToolkitStub, _make_pool


# ── Stubs ───────────────────────────────────────────────────────────


class _StubDispatcher:
    """Stand-in for ``ModelDispatcher``. Only ``run_delegated`` is
    exercised by the delegation handler; record the call so the test can
    prove the context routed through to a real delegated run."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run_delegated(
        self,
        *,
        model_id: str,
        task: str,
        parent_session_id: str | None,
        pool: Any,
        db: Any,
    ) -> str:
        self.calls.append(
            {
                "model_id": model_id,
                "task": task,
                "parent_session_id": parent_session_id,
                "pool": pool,
                "db": db,
            }
        )
        return f"sub-agent handled: {task}"


class _DelegatingAgent:
    """Agent stub exposing exactly what the executor reads to build the
    delegation context: ``_mcp`` (pool), ``_db`` (catalog handle), and
    ``model`` (the dispatcher with ``run_delegated``)."""

    def __init__(self, *, pool: Any, db: Any, model: Any) -> None:
        self._mcp = pool
        self._db = db
        self.model = model

    async def forget_session(self, session_id: str) -> None:
        return None

    async def release_session(self, session_id: str) -> None:
        return None


def _delegation_toolkit() -> _ToolkitStub:
    """A ``delegation`` toolkit whose ``delegate_task`` is the *real*
    handler wrapper — so it reads the live contextvars exactly like the
    in-process MCP does. Lands in ``async_functions`` (async callable),
    which ``_h_mcp_tool`` merges and dispatches over."""
    from src.mcp.servers.delegation import handlers

    async def delegate_task(model_id: str, task: str) -> dict:
        return await handlers.delegate_task(model_id=model_id, task=task)

    return _ToolkitStub(async_functions={"delegate_task": delegate_task})


def _delegating_workflow() -> dict:
    return {
        "id": "wf-support-coverage",
        "name": "support coverage",
        "graph": {
            "version": 1,
            "nodes": [
                {
                    "id": "n1",
                    "type": "mcp-tool",
                    "config": {
                        "mcp_name": "delegation",
                        "tool_name": "delegate_task",
                        "args": {
                            "model_id": "deepseek:v4-pro",
                            "task": "Cover the support queue.",
                        },
                    },
                },
            ],
            "edges": [],
            "variables": {},
        },
    }


# ── Tests ───────────────────────────────────────────────────────────


@test(
    "workflow_delegation_context",
    "executor.run installs a delegation context so delegate_task succeeds",
)
async def t_delegate_task_succeeds_in_workflow(ctx: TestContext) -> None:
    from src.workflow.executor import WorkflowExecutor

    dispatcher = _StubDispatcher()
    db = _StubDB()
    pool = _make_pool({"delegation": _delegation_toolkit()})
    agent = _DelegatingAgent(pool=pool, db=db, model=dispatcher)

    executor = WorkflowExecutor(agent=agent, db=db)  # type: ignore[arg-type]
    final = await executor.run(_delegating_workflow(), trigger="manual")

    # The run must succeed — before the fix it failed at the mcp-tool
    # node with the "outside an agent turn" error.
    assert final["status"] == "success", final

    node = final["trace"][0]
    assert node["node_id"] == "n1", node
    assert node["status"] == "success", node
    inner = node["output"]["result"]
    assert inner["status"] == "ok", inner
    assert inner["answer"] == "sub-agent handled: Cover the support queue.", inner

    # The context actually routed through to a real delegated run with
    # the parent's pool + db + the workflow-scoped session id.
    assert len(dispatcher.calls) == 1, dispatcher.calls
    call = dispatcher.calls[0]
    assert call["model_id"] == "deepseek:v4-pro", call
    assert call["pool"] is pool, call
    assert call["db"] is db, call
    assert call["parent_session_id"] == "workflow:wf-support-coverage:" + final["id"], call


@test(
    "workflow_delegation_context",
    "delegate_task without the executor context still errors (the fix is the context)",
)
async def t_delegate_task_errors_without_context(ctx: TestContext) -> None:
    """Calling the handler outside ``executor.run`` — i.e. with no
    delegation context installed — must still hit the guard. This pins
    the cause: it is the missing context the executor now installs, not
    any change to how the block resolves the tool."""
    from src.mcp.servers.delegation import handlers

    # No context installed in this scope (default contextvars → None).
    result = await handlers.delegate_task(
        model_id="deepseek:v4-pro", task="x",
    )
    assert result["status"] == "error", result
    assert "outside an agent turn" in result["error"], result


@test(
    "workflow_delegation_context",
    "executor.run resets the delegation context after the run (no leak)",
)
async def t_context_reset_after_run(ctx: TestContext) -> None:
    from src.mcp.servers.delegation import handlers
    from src.workflow.executor import WorkflowExecutor

    dispatcher = _StubDispatcher()
    db = _StubDB()
    pool = _make_pool({"delegation": _delegation_toolkit()})
    agent = _DelegatingAgent(pool=pool, db=db, model=dispatcher)

    executor = WorkflowExecutor(agent=agent, db=db)  # type: ignore[arg-type]
    await executor.run(_delegating_workflow(), trigger="manual")

    # finally-block reset must have restored the module contextvars to
    # their unset default — otherwise a delegate_task in an unrelated
    # context (or a later workflow) could inherit this run's dispatcher.
    assert handlers._dispatcher_var.get() is None
    assert handlers._db_var.get() is None
    assert handlers._session_id_var.get() is None
    assert handlers._pool_var.get() is None
