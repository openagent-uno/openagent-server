"""End-to-end: a workflow run-now must, while it runs,
  1. STREAM its ai-prompt node's frames through the broadcast sink — tagged
     with the node's child session id (``workflow:{wf}:{run}:{node}``) — so the
     run screen renders token-by-token and the in-chat run-launch card can link
     itself clickable mid-run; and
  2. ANNOUNCE the run as a ``workflow`` resource event the moment its row opens
     (and again when it finalizes) so the sidebar Recent feed surfaces it while
     it is still running — parity with how a scheduled-task firing announces
     itself via ``Scheduler.run_task``.

These pin the backend half of both reported bugs (workflow card not clickable
mid-run; workflow run absent from the sidebar while running). Uses a REAL
``MemoryDB`` so the full ``executor.run`` → ``run_child_session(stream=True)``
path executes exactly as in production; only the model is a stub.
"""
from __future__ import annotations

import uuid

from ._framework import TestContext, test


class _StreamSpyAgent:
    """The minimum ``run_child_session(stream=True)`` drives: a ``run_stream``
    async-generator yielding deltas, ``release_session``, a ``name``, and a
    ``model`` (None → no override). ``_db`` None → the executor uses its db."""

    name = "spy"
    model = None
    _db = None

    async def run_stream(self, *, message, user_id, session_id,
                         model_override=None, author=None, on_status=None):
        if on_status is not None:
            await on_status("working")
        for piece in ("Hel", "lo"):
            yield {"kind": "delta", "text": piece}
        yield {"kind": "done", "text": "Hello"}

    async def release_session(self, session_id, *, model_override=None) -> None:
        return None


# An ai-prompt node is the path that runs through run_child_session(stream=True).
# Shared between add_workflow (which mints the id) and the run dict below.
_AI_GRAPH = {
    "version": 1,
    "nodes": [{"id": "n1", "type": "ai-prompt", "config": {"prompt": "do it"}}],
    "edges": [],
    "variables": {},
}


@test(
    "workflow_live_stream",
    "run-now streams its ai-node child frames AND announces the run for the feed",
)
async def t_workflow_streams_and_announces(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.workflow.executor import WorkflowExecutor
    from src.stream.child_stream import set_child_broadcast_sink

    tmp = ctx.db_path.with_name(f"wf-live-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(tmp))
    await db.connect()
    try:
        # The workflow row must exist (FK target for the run row). add_workflow
        # mints the id; build the run dict around it.
        wf_id = await db.add_workflow(name="selftest", graph=_AI_GRAPH)

        # (2) capture the resource-event broadcasts (sidebar Recent feed refresh)
        broadcasts: list[tuple] = []

        def broadcast(resource, action, id=None):
            broadcasts.append((resource, action, id))

        # (1) capture the detached-child live frames (run screen + card link)
        frames: list[dict] = []

        async def sink(frame: dict) -> None:
            frames.append(frame)

        agent = _StreamSpyAgent()
        executor = WorkflowExecutor(agent=agent, db=db, broadcast=broadcast)
        set_child_broadcast_sink(sink)
        try:
            run_dict = {"id": wf_id, "name": "selftest", "graph": _AI_GRAPH}
            final = await executor.run(run_dict, trigger="ai")
        finally:
            set_child_broadcast_sink(None)

        assert final["status"] == "success", final
        run_id = final["id"]

        # (2) the run announced itself as a workflow resource event — at least
        # once (run row opened) so the Recent feed refetches while it runs.
        assert ("workflow", "updated", wf_id) in broadcasts, broadcasts

        # (1) the ai node actually streamed, tagged with ITS child sid, so the
        # app routes the frames to the run screen and links the card mid-run.
        child_sid = f"workflow:{wf_id}:{run_id}:n1"
        sids = {f["session_id"] for f in frames}
        assert child_sid in sids, (child_sid, sorted(sids))
        kinds = [(f["kind"], f.get("text")) for f in frames if f["session_id"] == child_sid]
        assert ("seed", "do it") in kinds, kinds            # Mission block, pre-delta
        assert ("delta", "Hel") in kinds, kinds              # token streaming
        assert any(k == "turn_complete" for k, _ in kinds), kinds
    finally:
        await db.close()
