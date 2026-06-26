"""``ai-prompt`` nodes now run as durable child sessions.

A workflow ai-prompt node runs through ``core.child_session.run_child_session``
— a full child session (own row, two-layer prompt, navigable + continuable,
vision §8/§15) rather than a throwaway run that gets wiped. The per-run id is
unique (``workflow:{wf}:{run}:{node}`` for ephemeral, ``workflow:{wf}:{run}``
for shared), so a re-run of the same workflow can never inherit a prior run's
transcript — the issue-#5 root cause is removed structurally and the session
is kept, not forgotten.

These tests pin:
- ephemeral nodes get a per-node session, released (not forgotten) at node-end.
- shared nodes chain on one per-run session, released per-node, and run
  finalisation releases (not forgets) it — so the durable row survives.
"""
from __future__ import annotations

from ._framework import TestContext, test


class _SpyAgent:
    """Minimal Agent stub recording run / release / forget calls."""

    name = "spy"
    model = None

    def __init__(self) -> None:
        self.forget_calls: list[str] = []
        self.release_calls: list[str] = []
        self.run_calls: list[tuple[str, str]] = []

    async def run(
        self,
        *,
        message: str,
        user_id: str,
        session_id: str,
        model_override=None,
        author=None,
        on_status=None,
    ) -> str:
        self.run_calls.append((session_id, message))
        return "ok"

    async def forget_session(self, session_id: str) -> None:
        self.forget_calls.append(session_id)

    async def release_session(self, session_id: str, *, model_override=None) -> None:
        self.release_calls.append(session_id)


class _StubDB:
    """update_* no-ops; child_session metadata helpers absent (best-effort)."""

    async def update_workflow_run(self, run_id: str, **kwargs) -> None:
        return None

    async def update_workflow(self, workflow_id: str, **kwargs) -> None:
        return None


@test("workflow_forget", "ai-prompt ephemeral node = durable per-node session (release, not forget)")
async def t_ephemeral_durable(ctx: TestContext) -> None:
    from src.workflow.executor import WorkflowExecutor, _RunCtx, _h_ai_prompt

    agent = _SpyAgent()
    executor = WorkflowExecutor(agent=agent, db=_StubDB())  # type: ignore[arg-type]
    run_ctx = _RunCtx(run_id="run-1", workflow_id="wf-1", inputs={}, vars={})
    node = {"id": "n1", "type": "ai-prompt"}
    cfg = {"prompt": "hi", "session_policy": "ephemeral"}
    out = await _h_ai_prompt(executor, node, cfg, run_ctx)

    expected_sid = "workflow:wf-1:run-1:n1"
    assert agent.run_calls == [(expected_sid, "hi")], agent.run_calls
    # The node's child session is handed back so the run screen can deep-link.
    assert out["child_session_id"] == expected_sid, out
    # Durable: release (keeps the row), never forget.
    assert agent.release_calls == [expected_sid], agent.release_calls
    assert agent.forget_calls == [], agent.forget_calls


@test("workflow_forget", "ai-prompt shared nodes chain one durable session; finalize releases it")
async def t_shared_durable(ctx: TestContext) -> None:
    from src.workflow.executor import WorkflowExecutor, _RunCtx, _h_ai_prompt

    agent = _SpyAgent()
    executor = WorkflowExecutor(agent=agent, db=_StubDB())  # type: ignore[arg-type]
    run_ctx = _RunCtx(run_id="run-2", workflow_id="wf-1", inputs={}, vars={})
    for nid in ("n1", "n2"):
        await _h_ai_prompt(
            executor,
            {"id": nid, "type": "ai-prompt"},
            {"prompt": "hi", "session_policy": "shared"},
            run_ctx,
        )

    shared_sid = "workflow:wf-1:run-2"
    assert agent.run_calls == [(shared_sid, "hi"), (shared_sid, "hi")], agent.run_calls
    # Both nodes ran on the one shared session and released per-node.
    assert agent.release_calls == [shared_sid, shared_sid], agent.release_calls
    assert agent.forget_calls == [], agent.forget_calls

    # Finalisation releases (belt-and-suspenders) but does NOT forget — the
    # durable shared session must survive for navigation.
    await executor._finalize_run(run_ctx, status="success", outputs={})
    assert agent.release_calls == [shared_sid, shared_sid, shared_sid], agent.release_calls
    assert agent.forget_calls == [], agent.forget_calls


@test("workflow_forget", "_finalize_run releases (not forgets) the shared session on failure")
async def t_finalize_release_on_failure(ctx: TestContext) -> None:
    from src.workflow.executor import WorkflowExecutor, _RunCtx

    agent = _SpyAgent()
    executor = WorkflowExecutor(agent=agent, db=_StubDB())  # type: ignore[arg-type]
    run_ctx = _RunCtx(run_id="run-3", workflow_id="wf-1", inputs={}, vars={})
    await executor._finalize_run(run_ctx, status="failed", error="boom")
    # Durable: a failed run still keeps the session row (per-run id is unique,
    # so a retry starts clean without wiping).
    assert agent.release_calls == ["workflow:wf-1:run-3"], agent.release_calls
    assert agent.forget_calls == [], agent.forget_calls
