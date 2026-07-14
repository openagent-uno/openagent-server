"""Dream-mode prompt sanity check."""
from __future__ import annotations

from ._framework import TestContext, test


@test("dream", "DREAM_MODE_PROMPT covers all three missions")
async def t_dream_prompt(ctx: TestContext) -> None:
    from src.core.server import DREAM_MODE_PROMPT, DREAM_MODE_TASK_NAME
    assert isinstance(DREAM_MODE_PROMPT, str)
    assert len(DREAM_MODE_PROMPT) > 100
    lower = DREAM_MODE_PROMPT.lower()

    # Mission 1 — memory vault evaluation and correction.
    assert "vault" in lower
    # Mission 2 — log analysis, fixing broken tasks/workflows.
    assert "workflow" in lower
    # Mission 3 — proactivity. Vision §15: "The agent is expected to be
    # proactive. It surfaces patterns it notices — recurring work that
    # could be scheduled ... and proposes automations rather than waiting
    # to be asked." This duty came from the retired ``manager-review``
    # task; folding that task into dream mode dropped it on the floor
    # (t_no_manager_review below proves the task is gone, and passed
    # happily while the CAPABILITY was gone too). Restored 2026-07-14.
    assert "recurring work" in lower, (
        "Mission 3 is missing: dream mode no longer tells the agent to "
        "notice work that should be automated. That is vision §15's "
        "proactivity, and nothing else in the system does it."
    )
    assert "pending-automation" in lower

    # The three missions must be numbered and ordered — Mission 3 widens
    # the same log window Mission 2 already paid for.
    for marker in ("## Mission 1", "## Mission 2", "## Mission 3"):
        assert marker in DREAM_MODE_PROMPT, f"{marker} missing"
    assert (
        DREAM_MODE_PROMPT.index("## Mission 1")
        < DREAM_MODE_PROMPT.index("## Mission 2")
        < DREAM_MODE_PROMPT.index("## Mission 3")
    )

    assert DREAM_MODE_TASK_NAME == "dream-mode"


@test("dream", "dream mode reads logs through the MCP, not the shell")
async def t_dream_uses_logs_mcp(ctx: TestContext) -> None:
    """The prompt used to hand the agent OS-specific paths and tell it to
    ``find ~ -name events.jsonl`` then ``tail -n 2000``. On the real
    production log that is ~77k tokens for 26% coverage, versus ~1.5k for
    100% via ``logs_summary``. The `logs` MCP (vision §14) now exists, so
    the shell workaround must not creep back.
    """
    from src.core.server import DREAM_MODE_PROMPT

    for tool in ("logs_summary", "logs_query", "logs_context"):
        assert tool in DREAM_MODE_PROMPT, f"Mission 2 no longer calls {tool}"

    # The old recipe, in any form.
    for anti in ("tail -n", "find ~", "Application Support/OpenAgent/logs"):
        assert anti not in DREAM_MODE_PROMPT, (
            f"the shell log-tail workaround is back ({anti!r}); the `logs` "
            "MCP resolves the path itself and summarises the whole window."
        )


@test("dream", "manager-review built-in is fully removed")
async def t_no_manager_review(ctx: TestContext) -> None:
    import src.core.server as server
    import src.core.builtin_tasks as bt

    assert not hasattr(server, "MANAGER_REVIEW_PROMPT")
    assert not hasattr(server.AgentServer, "_sync_manager_review")
    assert not hasattr(bt, "MANAGER_REVIEW_TASK_NAME")
    assert "manager-review" not in bt.BUILTIN_TASK_NAMES
    assert "manager_review" not in bt.CONFIG_SECTION_BY_TASK.values()
