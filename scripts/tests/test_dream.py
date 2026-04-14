"""Dream-mode prompt sanity check."""
from __future__ import annotations

from ._framework import TestContext, test


@test("dream", "DREAM_MODE_PROMPT is non-empty and mentions vault")
async def t_dream_prompt(ctx: TestContext) -> None:
    from openagent.core.server import DREAM_MODE_PROMPT, DREAM_MODE_TASK_NAME
    assert isinstance(DREAM_MODE_PROMPT, str)
    assert len(DREAM_MODE_PROMPT) > 100
    assert "vault" in DREAM_MODE_PROMPT.lower()
    assert DREAM_MODE_TASK_NAME == "dream-mode"
