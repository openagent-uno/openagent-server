"""Dream-mode prompt sanity check."""
from __future__ import annotations

from ._framework import TestContext, test


@test("dream", "DREAM_MODE_PROMPT covers both missions (vault + logs)")
async def t_dream_prompt(ctx: TestContext) -> None:
    from src.core.server import DREAM_MODE_PROMPT, DREAM_MODE_TASK_NAME
    assert isinstance(DREAM_MODE_PROMPT, str)
    assert len(DREAM_MODE_PROMPT) > 100
    lower = DREAM_MODE_PROMPT.lower()
    # Mission 1 — memory vault evaluation and correction.
    assert "vault" in lower
    # Mission 2 — last-day log analysis and fixing broken tasks/workflows.
    assert "events.jsonl" in lower
    assert "workflow" in lower
    assert DREAM_MODE_TASK_NAME == "dream-mode"


@test("dream", "manager-review built-in is fully removed")
async def t_no_manager_review(ctx: TestContext) -> None:
    import src.core.server as server
    import src.core.builtin_tasks as bt

    assert not hasattr(server, "MANAGER_REVIEW_PROMPT")
    assert not hasattr(server.AgentServer, "_sync_manager_review")
    assert not hasattr(bt, "MANAGER_REVIEW_TASK_NAME")
    assert "manager-review" not in bt.BUILTIN_TASK_NAMES
    assert "manager_review" not in bt.CONFIG_SECTION_BY_TASK.values()
