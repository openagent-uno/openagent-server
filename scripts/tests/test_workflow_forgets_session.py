"""Regression: ``ai-prompt`` nodes and workflow run-finalisation must
forget provider-native resume state at the right moment so a re-run of
the same workflow doesn't inherit the prior run's transcript.

Before the fix, ``_h_ai_prompt`` always called ``release_session`` in
its finally block — the same bug class as scheduler issue #5, just for
workflows. With ephemeral policy each node had its own unique session
id that was never wiped, so ``sdk_sessions`` grew unboundedly and
claude-cli kept resuming the old transcript. With shared policy the
first firing of the workflow populated a session id keyed on
``(workflow_id, run_id)``; the next manual run with a different run_id
was fine, but the rows never got cleaned up.

The fix:
- ``ephemeral`` nodes call ``forget_session`` at node-end (matches the
  scheduler's post-#5 behaviour).
- ``shared`` nodes still call ``release_session`` at node-end so
  successive ai-prompt nodes in the same run can chain context; the
  full forget happens once in ``_finalize_run``.
"""
from __future__ import annotations

from ._framework import TestContext, test


class _SpyAgent:
    """Minimal Agent stub recording forget vs release calls."""

    def __init__(self) -> None:
        self.forget_calls: list[str] = []
        self.release_calls: list[str] = []
        self.run_calls: list[tuple[str, str]] = []
        # SmartRouter attribute poked by _h_ai_prompt's model_override path;
        # stay None so the override branch is a no-op.
        self.model = None

    async def run(
        self,
        *,
        message: str,
        user_id: str,
        session_id: str,
        model_override=None,
    ) -> str:
        self.run_calls.append((session_id, message))
        return "ok"

    async def forget_session(self, session_id: str) -> None:
        self.forget_calls.append(session_id)

    async def release_session(self, session_id: str) -> None:
        self.release_calls.append(session_id)


class _StubDB:
    """update_* no-ops — _finalize_run only cares that awaits resolve."""

    async def update_workflow_run(self, run_id: str, **kwargs) -> None:
        return None

    async def update_workflow(self, workflow_id: str, **kwargs) -> None:
        return None


@test("workflow_forget", "ai-prompt ephemeral node forgets at node-end")
async def t_ephemeral_forget(ctx: TestContext) -> None:
    from openagent.workflow.executor import (
        WorkflowExecutor,
        _RunCtx,
        _h_ai_prompt,
    )

    agent = _SpyAgent()
    executor = WorkflowExecutor(agent=agent, db=_StubDB())  # type: ignore[arg-type]
    run_ctx = _RunCtx(
        run_id="run-1",
        workflow_id="wf-1",
        inputs={},
        vars={},
    )
    node = {"id": "n1", "type": "ai-prompt"}
    cfg = {"prompt": "hi", "session_policy": "ephemeral"}
    await _h_ai_prompt(executor, node, cfg, run_ctx)

    expected_sid = "workflow:wf-1:run-1:n1"
    assert agent.run_calls == [(expected_sid, "hi")], agent.run_calls
    # The fix: ephemeral must forget (not release) so provider-native
    # resume state is actually erased.
    assert agent.forget_calls == [expected_sid], agent.forget_calls
    assert agent.release_calls == [], agent.release_calls


@test("workflow_forget", "ai-prompt shared node releases per-node; forget at run-end")
async def t_shared_release_then_finalize_forget(ctx: TestContext) -> None:
    from openagent.workflow.executor import (
        WorkflowExecutor,
        _RunCtx,
        _h_ai_prompt,
    )

    agent = _SpyAgent()
    executor = WorkflowExecutor(agent=agent, db=_StubDB())  # type: ignore[arg-type]
    run_ctx = _RunCtx(
        run_id="run-2",
        workflow_id="wf-1",
        inputs={},
        vars={},
    )
    for nid in ("n1", "n2"):
        await _h_ai_prompt(
            executor,
            {"id": nid, "type": "ai-prompt"},
            {"prompt": "hi", "session_policy": "shared"},
            run_ctx,
        )

    shared_sid = "workflow:wf-1:run-2"
    assert agent.run_calls == [
        (shared_sid, "hi"),
        (shared_sid, "hi"),
    ], agent.run_calls
    # shared: per-node is release (keeps resume id so nodes chain).
    assert agent.release_calls == [shared_sid, shared_sid], agent.release_calls
    # No forget yet — that happens at run finalization.
    assert agent.forget_calls == [], agent.forget_calls

    # _finalize_run should issue exactly one forget for the shared sid.
    await executor._finalize_run(run_ctx, status="success", outputs={})
    assert agent.forget_calls == [shared_sid], agent.forget_calls


@test("workflow_forget", "_finalize_run forgets shared session even on failure")
async def t_finalize_forget_on_failure(ctx: TestContext) -> None:
    from openagent.workflow.executor import WorkflowExecutor, _RunCtx

    agent = _SpyAgent()
    executor = WorkflowExecutor(agent=agent, db=_StubDB())  # type: ignore[arg-type]
    run_ctx = _RunCtx(
        run_id="run-3",
        workflow_id="wf-1",
        inputs={},
        vars={},
    )
    # Failure path — _finalize_run gets called with status="failed"; must
    # still wipe the shared sid so a retry of the workflow starts clean.
    await executor._finalize_run(run_ctx, status="failed", error="boom")
    assert agent.forget_calls == ["workflow:wf-1:run-3"], agent.forget_calls
