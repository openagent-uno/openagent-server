"""Who is writing to the skill library right now — and what that permits.

The curator and the distiller are autonomous: they run on a schedule, with no
user present to stop them. Their whole safety story is a boundary — they may
only touch what the agent authored (``created_by: agent``) — and until now that
boundary lived in a PROMPT. A prompt is guidance a model follows most of the
time, which is the wrong strength for "may this process rewrite the playbook
eSound answers customers with".

So the boundary moves into code. A ContextVar records the write origin for the
duration of an autonomous pass; the skill tool consults it and refuses the
writes that origin is not allowed to make. Foreground turns — a human asking
the agent to write or fix a skill — are unrestricted, exactly as before.

Borrowed from Hermes (``tools/skill_provenance.py``), including the rule that
made it worth borrowing: a PINNED skill blocks the autonomous actor too. Being
autonomous is precisely why the pin applies to you.
"""

from __future__ import annotations

import contextvars

# "foreground" — a human is in the loop, directly or by having asked for this
# turn. "background" — an autonomous pass (curator / distiller) with nobody
# watching.
FOREGROUND = "foreground"
BACKGROUND = "background"
# "propose" — the post-turn review fork on its first setting: it reads the
# conversation and says what it WOULD change, and cannot change anything. The
# order matters and is deliberate. A reviewer that writes from day one gets to
# rewrite the library before anyone has seen what it wants to write; a
# reviewer that only proposes can be watched for a week at the cost of reading
# its notes. Promotion to BACKGROUND is a config change, not a code change.
PROPOSE = "propose"

_write_origin: contextvars.ContextVar[str] = contextvars.ContextVar(
    "openagent_skill_write_origin", default=FOREGROUND,
)


def set_write_origin(origin: str) -> contextvars.Token:
    """Mark the current context as ``origin``. Returns a token for reset()."""
    return _write_origin.set(
        origin if origin in (FOREGROUND, BACKGROUND, PROPOSE) else FOREGROUND,
    )


def reset_write_origin(token: contextvars.Token) -> None:
    try:
        _write_origin.reset(token)
    except (ValueError, LookupError):
        # Reset from a different context than the set — nothing to undo, and a
        # failed reset must never take down the pass that called it.
        pass


def current_write_origin() -> str:
    return _write_origin.get()


def is_background() -> bool:
    """True for any autonomous origin — the ones a mutation is checked against."""
    return current_write_origin() in (BACKGROUND, PROPOSE)


def is_propose_only() -> bool:
    return current_write_origin() == PROPOSE


def mutation_refusal(
    action: str,
    name: str,
    *,
    created_by: str | None,
    pinned: bool,
) -> str | None:
    """Why an autonomous pass may not perform ``action`` on this skill, or None.

    Two refusals, and the order matters — a pinned skill is refused even when
    the agent wrote it, because the pin is the user saying "this one is load
    bearing now, stop editing it".
    """
    if not is_background():
        return None  # a human asked; their library, their call
    if is_propose_only():
        # The reviewer's whole value in this mode is that its judgement is
        # visible before it is applied. Refusing here rather than in a prompt
        # is what makes "propose only" a property of the system instead of a
        # request the model usually honours.
        return (
            f"This pass is proposal-only: it may not {action} {name!r}, or "
            "anything else. Write what you would change and why — the name, "
            "what is wrong with it now, and the exact replacement text — and "
            "stop there. A person decides whether it lands."
        )
    if pinned:
        return (
            f"{name!r} is pinned. A pin blocks autonomous writes — including "
            "yours, and including skills you authored: pinning is how the user "
            "says this one is load-bearing now. Report what you would have "
            "changed and leave the file alone."
        )
    if (created_by or "").strip().lower() != "agent":
        return (
            f"{name!r} was not authored by the agent (no `created_by: agent` in "
            "its frontmatter), so it is seed or user content and off-limits to "
            f"an autonomous {action}. If it is wrong, say so in your log — do "
            "not edit it."
        )
    return None
