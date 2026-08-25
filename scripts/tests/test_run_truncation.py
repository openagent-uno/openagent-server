"""A run stopped by its own budget must not report success.

The runtime does not raise when a run exhausts its tool-call budget: it feeds
the model "Tool call limit reached" and lets it write a final answer, so the
child run ends COMPLETED and the task recorded `success`. Measured on
clickup-task-quality-audit: three consecutive firings reported success with
0/4 lists audited, 0 tasks checked and 0 mutations. Only running the task by
hand revealed it — a scheduled task can look healthy for weeks this way.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("run_truncation", "the tool-call-limit marker is detected in a stored run")
async def t_marker_detected(ctx: TestContext) -> None:
    from src.core.scheduler import _run_was_truncated

    clean = {"status": "COMPLETED", "content": "did the work",
             "messages": [{"role": "tool", "content": "ok"}]}
    assert _run_was_truncated(clean) is False

    # The exact shape the provider writes when the budget is spent.
    truncated = {"status": "COMPLETED", "content": "here is my summary",
                 "messages": [{"role": "tool", "content":
                               "Tool call limit reached. Tool call "
                               "tool_search_call_tool not executed. "
                               "Don't try to execute it again."}]}
    assert _run_was_truncated(truncated) is True


@test("run_truncation", "detection never raises on an unserialisable run")
async def t_detection_is_safe(ctx: TestContext) -> None:
    """It runs on every firing: it must never be the thing that breaks one."""
    from src.core.scheduler import _run_was_truncated

    class _Hostile:
        def __repr__(self):
            raise RuntimeError("boom")

    assert _run_was_truncated({"content": _Hostile()}) in (True, False)
    assert _run_was_truncated({}) is False


@test("run_truncation", "a truncated child run overrides the task status to failed")
async def t_truncated_run_fails_the_task(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler

    class _DB:
        async def list_session_runs(self, session_id, limit=1):
            return [{
                "status": "COMPLETED",
                "content": "I could not audit the lists.",
                "messages": [{"role": "tool", "content":
                              "Tool call limit reached. Tool call x not executed."}],
            }]

    sch = Scheduler.__new__(Scheduler)
    sch.db = _DB()
    issue = await sch._child_session_terminal_issue("scheduler:t:r")
    assert issue is not None, "a truncated run still read as a success"
    status, error = issue
    assert status == "failed", status
    # The message has to name the budget, or the next person reads "failed"
    # and goes looking for a crash that never happened.
    assert "tool-call budget" in error, error


@test("run_truncation", "an ordinary completed run is still a success")
async def t_clean_run_unchanged(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler

    class _DB:
        async def list_session_runs(self, session_id, limit=1):
            return [{"status": "COMPLETED", "content": "audited 4/4 lists",
                     "messages": [{"role": "tool", "content": "ok"}]}]

    sch = Scheduler.__new__(Scheduler)
    sch.db = _DB()
    assert await sch._child_session_terminal_issue("scheduler:t:r") is None
