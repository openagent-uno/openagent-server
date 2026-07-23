"""Anti-fabrication pre-send reply guard.

WHY THIS EXISTS
---------------
Some models, despite an explicit system-prompt rule, still emit an UNBACKED
promise of human follow-up — "a teammate will personally verify", "our team
will get back to you", "flagged to the partnerships team", "un collega ti
ricontatterà" — when NO handoff/task was actually created this turn. That reply
reaches the customer as a fabricated commitment (the F9 failure the quality
scorer keeps flagging). Detecting it AFTER the fact does not help the customer
who already received the false promise; the model's instruction-following gap
needs a deterministic net on the reply path, not another sentence in the prompt.

WHAT IT DOES
------------
Runs SYNCHRONOUSLY just before the reply is returned to the channel. When the
reply promises human/team follow-up AND no "backing action" tool ran this turn,
it regenerates the reply ONCE to drop the unbacked promise (help the customer
with what the agent can actually do now). It never invents a handoff — an honest
"here is what I can do" beats a fabricated "someone will call you".

FAIL-OPEN, ALWAYS
-----------------
Disabled, no tool visibility, regex miss, regeneration error or empty result →
the ORIGINAL reply is returned unchanged. A guard bug must cost a rewrite, never
a turn. Grounding visibility comes from ``tool_trace`` (the same trace the
quality judge uses); without it the guard cannot tell a backed promise from an
unbacked one and no-ops.

ROLE-AGNOSTIC / §17
-------------------
The promise patterns and the "backing action" tool-name substrings are generic
and configurable; nothing here knows about any particular product, queue, or
MCP. Gated on ``OPENAGENT_REPLY_GUARD_ENABLED`` (off by default) so a deployment
that never enabled it is byte-identical.
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

from src.core.logging import elog

_ENABLED_ENV = "OPENAGENT_REPLY_GUARD_ENABLED"
_BACKING_TOOLS_ENV = "OPENAGENT_REPLY_GUARD_BACKING_TOOLS"

# Tool-name substrings (case-insensitive) that count as a REAL backing action
# for a "a human will follow up" promise — creating a handoff, task, ticket, or
# team notification. Generic on purpose; an operator extends the set per
# deployment via OPENAGENT_REPLY_GUARD_BACKING_TOOLS (comma-separated).
_DEFAULT_BACKING_TOOLS = (
    "mark_as_human", "mark_human", "markhuman", "human_review", "humanreview",
    "escalate", "escalation", "handoff", "hand_off",
    "create_task", "createtask", "create_ticket", "createticket", "add_task",
    "assign", "notify_team", "notifyteam", "forward_to", "forwardto",
    "flag_for", "flagfor", "route_to", "routeto", "open_ticket",
)

# Promise-of-human-follow-up patterns (English + Italian), drawn from the real
# scored-BAD replies. High-precision phrasings — a forward-looking commitment
# that a PERSON/TEAM will act — not generic empathy. Compiled once.
_PROMISE = re.compile(
    r"(?:"
    # EN: "a teammate / team member / colleague / specialist / human / agent /
    #      representative / someone (on|from) the team ... will <act>"
    r"\b(?:a|our|the|one of our)\s+(?:team\s?mate|team\s?member|colleague|"
    r"specialist|human|agent|representative|rep|engineer|technician|"
    r"support\s+(?:agent|member|specialist)|member of (?:our|the) team)\b"
    r"[^.?!\n]{0,60}?\b(?:will|is going to|'ll|shall)\b"
    r"[^.?!\n]{0,40}?\b(?:follow(?:\s|-)?up|get back|verify|check|look into|"
    r"reach out|contact|be in touch|review|investigate|take a look|assist|"
    r"help you|respond|reply)\b"
    r"|"
    # EN: "(our|the) team will <act>" / "someone (on|from) the team will"
    r"\b(?:our|the)\s+team\b[^.?!\n]{0,40}?\b(?:will|is going to|'ll)\b"
    r"[^.?!\n]{0,40}?\b(?:follow(?:\s|-)?up|get back|verify|check|look into|"
    r"reach out|contact|be in touch|review|investigate|respond|reply)\b"
    r"|"
    r"\bsomeone\s+(?:on|from)\s+(?:our|the)\s+team\b[^.?!\n]{0,40}?\bwill\b"
    r"|"
    # EN: "flagged/forwarded/passed/escalated/sent/routed ... to (the) ... team"
    r"\b(?:flag(?:ged)?|forward(?:ed)?|pass(?:ed)?|escalat(?:ed|ing)?|sent|"
    r"routed|rais(?:ed|ing)?)\b[^.?!\n]{0,50}?\bto\s+(?:the\s+|our\s+)?"
    r"[a-z]{0,20}\s?team\b"
    r"|"
    # EN: "your case is/has been flagged", "has your case flagged"
    r"\byour\s+(?:case|ticket|issue|request)\b[^.?!\n]{0,30}?"
    r"\b(?:is|has been|'s)\s+(?:flagged|escalated|forwarded|passed)\b"
    r"|"
    r"\b(?:has|have|have got|got|got a|has a)\s+your\s+"
    r"(?:case|ticket|issue|request)\b[^.?!\n]{0,20}?\bflagged\b"
    r"|"
    # IT: "un collega / membro del team / tecnico ... ti (ricontatterà|
    #      risponderà|scriverà|contatterà)"
    r"\b(?:un|il nostro|il)\s+(?:collega|membro del team|tecnico|"
    r"specialista|operatore|team)\b[^.?!\n]{0,60}?"
    r"\bti\s+(?:ricontatter|risponder|scriver|contatter|aggiorner)\w*"
    r"|"
    # IT: "il team (verificherà|controllerà|si occuperà|ti ricontatterà)"
    r"\bil\s+team\b[^.?!\n]{0,40}?\b(?:verificher|controller|si occuper|"
    r"ti ricontatter|ti risponder|dar[àa] un'?occhiata|esaminer)\w*"
    r"|"
    # IT: "girato/inoltrato/passato/segnalato al team"
    r"\b(?:girat|inoltrat|passat|segnalat|inviato|trasmess)[oaie]?\b"
    r"[^.?!\n]{0,30}?\bal\s+team\b"
    r")",
    re.IGNORECASE,
)


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    return _truthy(os.environ.get(_ENABLED_ENV, "0"))


def _backing_tool_substrings() -> tuple[str, ...]:
    raw = os.environ.get(_BACKING_TOOLS_ENV, "").strip()
    if not raw:
        return _DEFAULT_BACKING_TOOLS
    extra = tuple(
        s.strip().lower() for s in raw.split(",") if s.strip()
    )
    # The operator's list REPLACES the default only when non-empty; otherwise
    # fall back to the built-in set so a blank env never disables grounding.
    return extra or _DEFAULT_BACKING_TOOLS


def promises_followup(text: Optional[str]) -> bool:
    """True when *text* commits a human/team to follow up, verify, escalate, or
    get back to the customer. Never raises."""
    if not text:
        return False
    try:
        return bool(_PROMISE.search(text))
    except Exception:  # noqa: BLE001 — a regex miss must never break the turn
        return False


def _has_backing_action(tool_names: list[str]) -> bool:
    subs = _backing_tool_substrings()
    for name in tool_names:
        low = str(name).lower()
        if any(sub in low for sub in subs):
            return True
    return False


async def _regenerate(model: Any, user_message: str, draft: str) -> str:
    """One bounded rewrite that keeps the substance but drops the unbacked
    human-follow-up promise. Self-contained (no session history needed) — it
    revises the draft in place. Returns "" on any failure."""
    system = (
        "You are revising a customer-support reply before it is sent. The draft "
        "promises that a human, teammate, or team will follow up, verify, "
        "escalate, or get back to the customer — but NO such handoff was "
        "actually created, so that promise is false and must not be sent. "
        "Rewrite the reply to help the customer directly with what can actually "
        "be done now, and REMOVE every promise of human/team follow-up, "
        "verification, escalation, flagging, or being contacted later. Keep the "
        "rest of the substance, facts, and tone; do not invent new facts. "
        "Output ONLY the revised reply text — no preamble, no quotes."
    )
    prompt = (
        f"Customer message:\n{user_message or '(not available)'}\n\n"
        f"Draft reply (contains a forbidden unbacked promise):\n{draft}\n\n"
        "Revised reply:"
    )
    try:
        resp = await model.generate(
            [{"role": "user", "content": prompt}],
            system=system,
            session_id=None,  # never pollute the session being replied to
        )
    except Exception:  # noqa: BLE001 — regeneration failure = keep the original
        return ""
    return (getattr(resp, "content", "") or "").strip()


async def guard_reply(
    agent: Any, session_id: Optional[str], user_message: str, reply: str,
) -> str:
    """Return *reply*, or a rewrite of it with an unbacked human-follow-up
    promise removed. FAIL-OPEN: returns *reply* unchanged on disabled / no
    visibility / no promise / backed promise / regeneration failure / any error.
    """
    if not enabled() or not reply:
        return reply
    try:
        if not promises_followup(reply):
            return reply
        # Grounding visibility: without the tool trace we cannot tell a backed
        # promise from an unbacked one, so we do NOT touch the reply.
        from src.core import tool_trace

        if not tool_trace._enabled():
            return reply
        rows = tool_trace.peek(session_id)  # non-consuming — the judge still drains it
        tool_names = [name for (name, _excerpt) in rows] if rows else []
        if _has_backing_action(tool_names):
            return reply  # the promise is backed by a real handoff this turn — fine

        model = getattr(agent, "model", None)
        if model is None:
            return reply
        revised = await _regenerate(model, user_message, reply)
        if revised and not promises_followup(revised):
            elog(
                "reply_guard.rewrote",
                level="warning",
                session_id=session_id,
                reason="unbacked_human_followup_promise",
                tools=len(tool_names),
            )
            return revised
        # Rewrite failed or still promises — keep the original (fail-open) but
        # leave a breadcrumb so the scorer/operator sees it slipped through.
        elog(
            "reply_guard.unbacked_promise_kept",
            level="warning",
            session_id=session_id,
            reason="rewrite_empty_or_still_promising",
            tools=len(tool_names),
        )
        return reply
    except Exception as exc:  # noqa: BLE001 — the guard must never break the turn
        try:
            elog(
                "reply_guard.error",
                level="warning",
                session_id=session_id,
                error_type=type(exc).__name__,
                error=str(exc) or repr(exc),
            )
        except Exception:  # noqa: BLE001
            pass
        return reply
