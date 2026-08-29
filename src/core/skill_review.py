"""The post-turn review fork: what did this turn teach, if anything?

Borrowed from Hermes. After a turn ends well, a forked child session replays
what just happened and asks one question — *is there a skill to write or
update?* — with a tool whitelist narrowed to skills and memory. The parent
conversation is never touched: the fork reads a snapshot and answers into its
own transcript.

Why a fork per turn, when a nightly distiller already mines sessions. The
distiller works from what SURVIVED into storage, a day later, summarised. The
moment a procedure is actually legible is the moment it just worked: the
commands are there, the dead ends are there, and the thing that made it work
has not yet been compressed into "fixed the thing". The distiller finds
patterns across days; the fork catches the single good run before it fades.
They are not the same job and neither replaces the other.

Two properties keep this from being reckless.

**It proposes before it writes.** The default mode is ``propose``: the fork
may not mutate the library at all — enforced in
``src.mcp.servers.skills.provenance``, not asked for in a prompt. It writes
down what it would change and a person decides. Promotion to writing is a
config change once its notes have been read for a while.

**Cost follows the cache.** On the parent's own model the transcript is still
warm in the prefix cache, so replaying it in full is priced as cache reads —
the expensive-looking option is the cheap one. Routed to a different model
that cache cannot be reused, so the fork gets a compact digest instead of a
cold re-read of everything. Same reviewer, two shapes, chosen by whether the
cache is there.

Off by default (``skills.review_enabled``). A feature that fires after every
single turn has to be opted into deliberately.
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.core.logging import elog

# Only skills and memory. The reviewer's job is to notice a procedure worth
# keeping — it has no business running shell commands or talking to anyone,
# and a narrow grant is the difference between a reviewer and a second agent
# loose in the same session.
REVIEW_TOOL_FAMILIES = ("skills", "vault")

# How much of a digest is worth sending when the cache cannot be reused.
DIGEST_CHAR_BUDGET = 6_000

MODE_PROPOSE = "propose"
MODE_WRITE = "write"


def _mission(mode: str) -> str:
    proposing = mode != MODE_WRITE
    ending = (
        "You CANNOT write: this pass is proposal-only and the tools will "
        "refuse you. That is intended. Produce the proposal itself — the "
        "skill name, whether it is new or an edit, what is wrong with the "
        "current text if it exists, and the exact body you would write. A "
        "person reads it and decides."
        if proposing else
        "You may write, but only skills the agent authored and only ones that "
        "are not pinned; the tools enforce this. Prefer updating an existing "
        "skill over creating a near-duplicate."
    )
    return (
        "You are reviewing ONE conversation turn that just finished, to answer "
        "a single question: did it contain a procedure worth keeping?\n\n"
        "Most turns do not. Answering a question, reading a file, a chat — "
        "these leave nothing behind, and saying so is the correct outcome and "
        "the common one. Do not manufacture a skill to justify the pass; a "
        "library padded with near-misses is worse than a small one, because "
        "the next reader has to sort them.\n\n"
        "A turn is worth capturing when it shows a REPEATABLE procedure that "
        "was not obvious: a sequence that worked after something else failed, "
        "a constraint discovered the hard way, an exact incantation that is "
        "easy to get wrong. If an existing skill already covers it, say which "
        "one and stop — unless this turn CONTRADICTS it, which is the most "
        "valuable thing you can find, and then say exactly what changed.\n\n"
        "Search the existing skills before concluding anything is new.\n\n"
        + ending
    )


def _digest(transcript: str, budget: int = DIGEST_CHAR_BUDGET) -> str:
    """A head-and-tail digest of a transcript that does not fit the budget.

    Not a summary — a truncation that keeps both ends. The start carries what
    was asked; the end carries what finally worked. The middle is where the
    dead ends live, which matter less to "is this worth keeping" than to the
    turn itself. Cutting the middle and SAYING SO beats a smooth summary that
    hides which half is missing.
    """
    text = transcript or ""
    if len(text) <= budget:
        return text
    head = budget // 2
    tail = budget - head
    cut = len(text) - head - tail
    return (
        text[:head]
        + f"\n\n[...{cut} caratteri centrali omessi: la revisione gira su un "
          "modello diverso da quello del turno, quindi la cache del prefisso "
          "non e' riusabile e la trascrizione intera si pagherebbe a "
          "freddo...]\n\n"
        + text[-tail:]
    )


def review_payload(
    transcript: str,
    *,
    parent_model: str | None,
    review_model: str | None,
) -> tuple[str, bool]:
    """The transcript to send, and whether it went in whole.

    Same model → whole, because it is already warm in that model's prefix
    cache. Different model → digest, because there is no cache to reuse and a
    cold replay of everything is the one shape of this feature that would not
    pay for itself.
    """
    same_model = (not review_model) or (review_model == parent_model)
    if same_model:
        return transcript or "", True
    digested = _digest(transcript or "")
    return digested, digested == (transcript or "")


def should_review(settings: Any, *, reason: str) -> bool:
    """Whether this turn gets reviewed.

    Only turns that ENDED WELL. A cancelled turn is a turn the user walked
    away from, an errored one never reached its conclusion, and an empty one
    has nothing in it — reviewing any of them teaches the library from a
    procedure that did not actually work, which is precisely the failure this
    is supposed to prevent.
    """
    if not getattr(settings, "enabled", False):
        return False
    if not getattr(settings, "review_enabled", False):
        return False
    return reason == "completed"


async def run_review(
    *,
    agent: Any,
    db: Any,
    parent_session_id: str,
    transcript: str,
    settings: Any,
    parent_model: str | None = None,
) -> dict[str, Any] | None:
    """Spawn the review fork and return its verdict, or None if it did not run."""
    from src.core.child_session import run_child_session
    from src.mcp.servers.skills.provenance import (
        BACKGROUND, PROPOSE, reset_write_origin, set_write_origin,
    )

    mode = (getattr(settings, "review_mode", MODE_PROPOSE) or MODE_PROPOSE).lower()
    review_model = getattr(settings, "review_model", None)
    payload, whole = review_payload(
        transcript, parent_model=parent_model, review_model=review_model,
    )
    if not payload.strip():
        return None

    prompt = (
        _mission(mode)
        + "\n\n--- the turn ---\n"
        + payload
        + "\n--- end of the turn ---\n"
    )

    token = set_write_origin(PROPOSE if mode != MODE_WRITE else BACKGROUND)
    try:
        result = await run_child_session(
            agent=agent,
            db=db,
            parent_session_id=parent_session_id,
            origin="delegation",
            origin_ref={"kind": "skill-review"},
            title="Revisione skill del turno",
            prompt=prompt,
            model_id=review_model or None,
            allowed_tools=REVIEW_TOOL_FAMILIES,
        )
    finally:
        reset_write_origin(token)

    text = getattr(result, "text", None) or getattr(result, "output", None) or ""
    elog(
        "skill_review.done",
        mode=mode,
        model=review_model or "parent",
        transcript_whole=whole,
        chars=len(payload),
        preview=" ".join(str(text).split())[:400],
    )
    return {"mode": mode, "whole": whole, "text": text}


def schedule_review(**kwargs: Any) -> None:
    """Fire the review without making the turn wait for it.

    The turn is finished; the user is reading the answer. A reviewer that
    delays the next turn — or worse, breaks it — would have made the product
    worse in exchange for a library that is slightly better, so every failure
    here is logged and swallowed.
    """
    async def _guarded() -> None:
        try:
            await run_review(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            elog("skill_review.failed", level="warning", error=str(e))

    try:
        asyncio.get_running_loop().create_task(_guarded())
    except RuntimeError:
        # No loop (a sync caller in a test): nothing to schedule onto, and
        # refusing loudly here would break the turn for the sake of a review.
        elog("skill_review.no_loop", level="warning")
