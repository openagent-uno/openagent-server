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

FAILURE BEHAVIOUR
-----------------
Disabled, no tool visibility, regex miss, regeneration error or empty result →
the ORIGINAL reply is returned unchanged on normal turns. A lean local event is
stricter: high-precision future-release promises are checked without tool-trace
visibility, and a failed rewrite has the offending sentence removed
deterministically. Grounding visibility for human handoffs still comes from
``tool_trace`` (the same trace the quality judge uses).

ROLE-AGNOSTIC / §17
-------------------
The promise patterns and the "backing action" tool-name substrings are generic
and configurable; nothing here knows about any particular product, queue, or
MCP. Gated on ``OPENAGENT_REPLY_GUARD_ENABLED`` (off by default) so a deployment
that never enabled it is byte-identical.
"""
from __future__ import annotations

import json
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

# F11: a support agent may report a fix that is already verified as awaiting
# release, but it must not promise the next update/version/date. Keep this
# deliberately high precision so generic troubleshooting future tense is not
# rewritten.
_FUTURE_RELEASE = re.compile(
    r"(?:"
    r"\b(?:will|shall|'ll)\b[^.?!\n]{0,70}?\b(?:included|available|fixed|"
    r"resolved|shipped|released|delivered|land)\w*\b[^.?!\n]{0,45}?"
    r"\b(?:next|upcoming|future)\s+(?:app\s+)?(?:update|release|version)\b"
    r"|"
    r"\b(?:coming|planned|scheduled|targeted)\b[^.?!\n]{0,45}?"
    r"\b(?:next|upcoming|future)\s+(?:app\s+)?(?:update|release|version)\b"
    r"|"
    r"\b(?:sar[aà]|verr[aà])\b[^.?!\n]{0,70}?\b(?:inclus[oaie]|disponibile|"
    r"corrett[oaie]|risolt[oaie]|rilasciat[oaie])\b[^.?!\n]{0,45}?"
    r"\bprossim[oa]\s+(?:aggiornamento|release|versione)\b"
    r"|"
    r"\b(?:nel|con il)\s+prossimo\s+(?:aggiornamento|release|versione)\b"
    r")",
    re.IGNORECASE,
)

_COMPLETED_ACTION = re.compile(
    r"\b(?:i|we)(?:'ve|\s+have|\s+already)?\s+"
    r"(?:linked|attached|forwarded|escalated|flagged|submitted|opened|created|"
    r"filed|sent|notified|refunded|credited|activated|renewed|extended|updated|"
    r"fixed|resolved|corrected|changed|marked|assigned|routed|enabled)\b"
    r"|\b(?:ho|abbiamo)\s+(?:collegat|allegat|inoltrat|girat|segnalat|apert|"
    r"creat|inviat|rimborsat|accreditat|attivat|abilitat|rinnovat|estes|"
    r"aggiornat|corrett|risolt|assegnat)\w*\b",
    re.IGNORECASE,
)

# Diagnostics is the one claim family where the fabrication is not a verb the
# generic pattern above would ever catch: "I've enabled diagnostic logging on
# your account", then "we've received your logs" and "from your logs we can
# see". Every one of those is a statement about OUR side of the system that the
# customer cannot check, and each one was sent on a real Lyra thread
# (26-ago-2026) while no capture had ever been switched on and no log existed.
# The customer then performed the capture ritual for nothing, twice.
_DIAGNOSTIC_SUBJECT = (
    r"(?:diagnostic\w*|log\s?file\w*|logs?\b|telemetr\w+|"
    r"diagnostic\w*|registri|registrazion\w+|diagnostica|"
    r"registros?|journaux|protocolos?)"
)
_DIAGNOSTIC_CLAIM = re.compile(
    # "I enabled diagnostics", "we received your logs", "I've read the logs"
    r"\b(?:i|we)(?:'ve|'ll\s+have|\s+have|\s+already)?\s+"
    r"(?:enabled|turned\s+on|switched\s+on|activated|started|set\s+up|"
    r"received|got|collected|captured|pulled|read|reviewed|checked|"
    r"analy[sz]ed|inspected|examined)\b[^.?!\n]{0,60}?" + _DIAGNOSTIC_SUBJECT
    + r"|" + _DIAGNOSTIC_SUBJECT + r"[^.?!\n]{0,60}?\b(?:has|have|were|was|is|are)"
    r"\s+been\s+(?:enabled|activated|received|collected|captured|read|"
    r"reviewed|analy[sz]ed)\b"
    # "from your logs we can see", "your logs show" - a claim to have read them.
    + r"|\bfrom\s+(?:your|the)\s+" + _DIAGNOSTIC_SUBJECT
    + r"|" + _DIAGNOSTIC_SUBJECT + r"\s+(?:show|shows|indicate|indicates|tell\s+us)\b"
    r"|\b(?:ho|abbiamo)\s+(?:attivat|abilitat|acces|avviat|ricevut|ottenut|"
    r"raccolt|lett|esaminat|analizzat|controllat|verificat)\w*\b"
    r"[^.?!\n]{0,60}?(?:diagnostic\w*|log\b|logs\b|registri)"
    r"|\bdai\s+(?:tuoi\s+)?log\b|\bnei\s+(?:tuoi\s+)?log\b",
    re.IGNORECASE,
)

# The active-voice pattern above misses the form a model actually reaches for
# when it invents an outcome: the agentless passive. "Your refund request has
# been processed" names no actor, so nothing in _COMPLETED_ACTION matched it,
# and a fabricated refund reached the customer through the widened composer.
_COMPLETED_ACTION_PASSIVE = re.compile(
    r"\b(?:has|have|had|was|were|is|are)\s+been\s+"
    r"(?:processed|refunded|credited|issued|completed|approved|activated|"
    r"renewed|extended|updated|resolved|fixed|submitted|created|opened|"
    r"escalated|forwarded|cancelled|canceled|deleted|removed)\b"
    r"|\b(?:is|are)\s+being\s+"
    r"(?:processed|refunded|reviewed|escalated|investigated)\b"
    r"|\b(?:sono|è|e')\s+stat[oaie]\s+"
    r"(?:elaborat|rimborsat|accreditat|emess|complet|approvat|attivat|"
    r"rinnovat|estes|aggiornat|risolt|corrett|apert|creat|inoltrat|"
    r"annullat|cancellat|eliminat)\w*\b",
    re.IGNORECASE,
)

_ACCOUNT_STATE_CLAIM = re.compile(
    r"\byour\s+(?:premium\s+)?(?:subscription|account|plan)\s+"
    r"(?:is|shows?\s+as)\s+(?:active|inactive|expired|cancelled|canceled)\b"
    r"|\bil\s+tuo\s+(?:abbonamento|account|premium)\b[^.?!\n]{0,20}?"
    r"\b(?:[eè]\s+)?(?:attiv|inattiv|scadut|annullat)\w*\b",
    re.IGNORECASE,
)

_MONEY_AMOUNT = re.compile(
    r"[$€£]\s?\d[\d.,]*"
    r"|\b\d[\d.,]*\s?(?:usd|eur|gbp|euros?|dollars?|dollari|sterline)\b",
    re.IGNORECASE,
)

_COMMERCIAL_COMMITMENT = re.compile(
    r"\b(?:i|we)(?:'ll|\s+will|\s+can|\s+are going to)\s+"
    r"(?:refund|credit|extend|renew|activate|grant|give|offer|provide)\b"
    r"|\b(?:ti|le)\s+(?:rimborser|accrediter|attiver|rinnover|offriremo|"
    r"daremo|concederemo)\w*\b"
    # Agentless passive: names no actor, promises the same thing.
    r"|\b(?:refund|credit|reimbursement|rimborso|accredito)\b[^.?!\n]{0,40}"
    r"\b(?:will\s+be|is\s+going\s+to\s+be|sar[àa]|verr[àa])\s+"
    r"(?:issued|processed|credited|refunded|applied|sent|emess\w*|elaborat\w*|"
    r"accreditat\w*|rimborsat\w*|inviat\w*)\b"
    # A delivery window is a commitment even without a verb of granting.
    r"|\bwithin\s+\d+\s*(?:[-–]|to)?\s*\d*\s*(?:business\s+)?"
    r"(?:days?|weeks?|hours?|working\s+days?)\b"
    r"|\bentro\s+\d+\s*(?:[-–]|a)?\s*\d*\s*(?:giorni|settimane|ore)"
    r"(?:\s+lavorativ\w+)?\b",
    re.IGNORECASE,
)


# An identifier is the most load-bearing token a support reply can contain: the
# customer reads it as proof their report, order or refund exists. A local
# model invented "#86cavv98q" and presented it as a tracked issue.
#
# Deliberately tenant-agnostic: this must hold for Lyra and any other brand on
# the same agent, so it keys on the CONTEXT word rather than on one workspace's
# id shape. Matching bare alphanumerics instead would flag "iPhone16" in
# ordinary prose, and a guard that cries wolf gets turned off.
_IDENTIFIER_CLAIM = re.compile(
    r"\b(?:task|ticket|issue|case|order|ref|reference|id|subscription|invoice|"
    r"transaction|richiesta|ordine|fattura|abbonamento|pedido|commande)\b"
    # Short filler between the keyword and the token: "order ID is GPA-1",
    # "ticket number #42", "richiesta n. 7".
    r"(?:[\s:#\u00ba\u00b0.=()\[\]'\"\u00ab\u00bb-]|\b(?:id|number|num|no|ref|is|was|e|\u00e8|"
    r"numero|codice)\b){0,18}[\s:#.=()\[\]'\"-]\s*"
    r"(?P<named>[A-Za-z0-9][A-Za-z0-9._-]{3,})"
    r"|#\s?(?P<hashed>[A-Za-z0-9][A-Za-z0-9._-]{3,})",
    re.IGNORECASE,
)


# Stripping the invented id is not enough: the model then says "this is a known
# issue, a task already exists" with no id at all. To a customer that is the
# same promise - their report is being worked on - and it is just as unfounded.
# Deliberately NOT "known issue": that phrase can be legitimately grounded in a
# vault analysis note, and telling the two apart generically is unreliable - a
# guard that fires on honest replies gets disabled. What is never ambiguous is
# the claim that a TICKET EXISTS, which is also the claim a customer acts on.
_TRACKED_CLAIM = re.compile(
    r"\bbeing\s+tracked\b"
    r"|\b(?:we(?:'re| are)|i(?:'m| am))\s+(?:tracking|investigating)\b"
    r"|\ba\s+task\s+(?:already\s+)?exists\b"
    r"|\balready\s+(?:tracked|reported|logged|open)\b"
    r"|\b(?:una\s+)?(?:task|segnalazione|ticket)\s+(?:gi[\u00e0a]'?\s+)?(?:esiste|aperta?)\b"
    r"|\bgi[\u00e0a]'?\s+(?:segnalat|tracciat|apert)\w*\b"
    r"|\bstiamo\s+(?:tracciando|indagando)\b",
    re.IGNORECASE,
)

# Evidence that a real task exists: a task search or create that succeeded.
_TASK_TOOL = re.compile(
    r"(?:workspace_tasks|tasks_search|create_task|get_task|thread_link_task)",
    re.IGNORECASE,
)


def claims_issue_tracked(text: Optional[str]) -> bool:
    """True when a reply tells the customer their report is already tracked."""
    if not text:
        return False
    try:
        return bool(_TRACKED_CLAIM.search(text))
    except Exception:  # noqa: BLE001
        return False


def _trace_supports_tracking(rows: list[tuple[str, str]]) -> bool:
    return any(
        _TASK_TOOL.search(f"{name} {excerpt}")
        and _trace_result_succeeded(excerpt)
        # An empty result list proves the opposite: nothing was found.
        and not re.search(r'"tasks"\s*:\s*\[\s*\]', excerpt)
        for name, excerpt in rows or ()
    )


def _identifier_tokens(text: str) -> list[str]:
    """Identifier-shaped tokens a reply presents as a real reference."""
    out: list[str] = []
    for match in _IDENTIFIER_CLAIM.finditer(text or ""):
        token = match.group("named") or match.group("hashed") or ""
        token = token.strip().rstrip(".,;:)")
        # No digit means it is a word ("task management"), not a reference.
        if not token or not re.search(r"\d", token):
            continue
        # A version or a date is legitimate prose, never a ticket reference.
        if re.fullmatch(r"v?\d+(?:[.\-]\d+)+", token):
            continue
        out.append(token)
    return out


def _evidence_forms(rows: list[tuple[str, str]]) -> tuple[str, str]:
    """Trace text in two shapes: raw-lowercase, and alphanumeric-only."""
    raw = " ".join(f"{name} {excerpt}" for name, excerpt in rows or ()).lower()
    return raw, re.sub(r"[^a-z0-9]", "", raw)


def unbacked_identifiers(text: Optional[str], rows: list[tuple[str, str]]) -> list[str]:
    """Identifiers in *text* that this turn's tools never returned."""
    if not text:
        return []
    _raw, alnum = _evidence_forms(rows)
    out: list[str] = []
    for token in _identifier_tokens(text):
        needle = re.sub(r"[^a-z0-9]", "", token.lower())
        if needle and needle not in alnum:
            out.append(token)
    return out


# Kept as the original name used by the eSound controller and its tests.
unbacked_task_ids = unbacked_identifiers


def unbacked_money(text: Optional[str], rows: list[tuple[str, str]]) -> list[str]:
    """Monetary amounts in *text* absent from this turn's tool evidence.

    Compared as the literal number so "9.99", "9,99" and "€9.99" agree, and
    NOT as bare digits: matching "999" anywhere in a trace would ground an
    invented figure against a timestamp.
    """
    if not text:
        return []
    raw, _alnum = _evidence_forms(rows)
    out: list[str] = []
    for match in _MONEY_AMOUNT.findall(text):
        number = re.sub(r"[^\d.,]", "", str(match))
        if not number:
            continue
        variants = {number, number.replace(",", "."), number.replace(".", ",")}
        if not any(variant in raw for variant in variants):
            out.append(str(match).strip())
    return out

_FIX_STATUS_CLAIM = re.compile(
    r"\b(?:fix(?:es)?|correction|issue|bug)\b[^.?!\n]{0,55}?"
    r"\b(?:already\s+)?(?:implemented|fixed|corrected|resolved|done|complete|"
    r"completed|awaiting release|pending release)\b"
    r"|\b(?:has|have)\s+(?:already\s+)?been\s+"
    r"(?:implemented|fixed|corrected|resolved|completed)\b"
    r"|\b(?:correzion[ei]|fix|problema|bug)\b[^.?!\n]{0,55}?"
    r"\b(?:implementat|corrett|risolt|completat|fatt|rilasciat|"
    r"in attesa di (?:essere )?rilasciat)\w*\b"
    r"|\b(?:[eè]\s+stat[oaie]|sono\s+stat[ie])\b[^.?!\n]{0,55}?"
    r"\b(?:implementat|corrett|risolt|completat|fatt|rilasciat)\w*\b"
    r"|\b(?:implementat|corrett|risolt|completat|fatt|rilasciat)\w*\b[^.?!\n]{0,25}?"
    r"\b(?:nel|a livello di)\s+codice\b"
    r"|\b(?:the\s+)?fix\s+(?:is|was)\s+(?:already\s+)?(?:in\s+place|live)\b",
    re.IGNORECASE,
)

_NEGATIVE_FIX_EVIDENCE = re.compile(
    r"\b(?:no code modified|no code changes?|dry run only|status['\"=: ]+analysis|"
    r"proposed fix|proposal only|not (?:yet )?(?:fixed|implemented|merged)|"
    r"analysis only|fix not applied)\b",
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


def promises_future_release(text: Optional[str]) -> bool:
    """True for a concrete next-update/release commitment (F11)."""
    if not text:
        return False
    try:
        return bool(_FUTURE_RELEASE.search(text))
    except Exception:  # noqa: BLE001
        return False


def claims_completed_action(text: Optional[str]) -> bool:
    if not text:
        return False
    try:
        return bool(
            _COMPLETED_ACTION.search(text)
            or _COMPLETED_ACTION_PASSIVE.search(text)
        )
    except Exception:  # noqa: BLE001
        return False


def claims_diagnostics(text: Optional[str]) -> bool:
    """True when *text* claims a diagnostic capture was enabled, received or read."""
    if not text:
        return False
    try:
        return bool(_DIAGNOSTIC_CLAIM.search(text))
    except Exception:  # noqa: BLE001 — a regex miss must never break the turn
        return False


def quotes_money(text: Optional[str]) -> bool:
    """True when a reply names a concrete monetary amount.

    Support replies have no business quoting a figure that is not on a
    receipt. A local model asked to sound helpful invented "$4.99" for a
    refund that never happened.
    """
    if not text:
        return False
    try:
        return bool(_MONEY_AMOUNT.search(text))
    except Exception:  # noqa: BLE001
        return False


def claims_account_state(text: Optional[str]) -> bool:
    """True when a reply states a customer's current account/subscription state."""
    if not text:
        return False
    try:
        return bool(_ACCOUNT_STATE_CLAIM.search(text))
    except Exception:  # noqa: BLE001
        return False


def promises_commercial_value(text: Optional[str]) -> bool:
    if not text:
        return False
    try:
        return bool(_COMMERCIAL_COMMITMENT.search(text))
    except Exception:  # noqa: BLE001
        return False


def claims_completed_fix(text: Optional[str]) -> bool:
    if not text:
        return False
    try:
        return bool(_FIX_STATUS_CLAIM.search(text))
    except Exception:  # noqa: BLE001
        return False


def _trace_contradicts_completed_fix(rows: list[tuple[str, str]]) -> bool:
    return any(_NEGATIVE_FIX_EVIDENCE.search(excerpt or "") for _name, excerpt in rows)


def _trace_uses_historical_receipt(rows: list[tuple[str, str]]) -> bool:
    """A receipt can prove an older outcome, never the state of a new report."""
    return any("receipts/" in (excerpt or "").lower() for _name, excerpt in rows)


def _has_backing_action(tool_names: list[str]) -> bool:
    subs = _backing_tool_substrings()
    for name in tool_names:
        low = str(name).lower()
        if any(sub in low for sub in subs):
            return True
    return False


_TRACE_ERROR = re.compile(
    r"\b(?:error|failed|failure|forbidden|unauthorized|timeout|timed out|"
    r"not found|tool not found|cannot|could not)\b|\bhttp\s*[45]\d\d\b|"
    # ``success:false`` alone was not enough: MCP servers just as often answer
    # ``ok:false``, ``isError:true``, or a bare 4xx/5xx ``status``. A failed
    # call read as a successful one is the worst possible direction for this
    # check - it is what lets "the reply was sent" be claimed when it was not.
    r"[\"']?success[\"']?\s*[:=]\s*false\b|"
    r"[\"']?ok[\"']?\s*[:=]\s*false\b|"
    r"[\"']?is_?error[\"']?\s*[:=]\s*true\b|"
    r"[\"']?status[\"']?\s*[:=]\s*[45]\d\d\b",
    re.IGNORECASE,
)


def _trace_result_succeeded(excerpt: str) -> bool:
    """Conservative success check for a completed MCP call excerpt.

    The runtime records MCP protocol errors as ``Error from MCP tool``. Some
    successful servers return ``success:true`` while others return the updated
    object or a short ``queued``/``ok`` acknowledgement. A non-empty result is
    therefore acceptable only when it carries no explicit failure marker.
    """
    value = str(excerpt or "").strip()
    if " result=" in value:
        value = value.split(" result=", 1)[1].strip()
    return bool(value) and not bool(_TRACE_ERROR.search(value))


def _has_successful_backing_action(rows: list[tuple[str, str]]) -> bool:
    subs = _backing_tool_substrings()
    return any(
        any(sub in f"{name} {excerpt}".lower() for sub in subs)
        and _trace_result_succeeded(excerpt)
        for name, excerpt in rows
    )


_ACTION_FAMILIES = (
    (("refund", "rimbors", "accredit"), ("refund", "credit")),
    (("cancel", "annull", "disdett"), ("cancel", "revoke")),
    (("grant", "activat", "attivat", "reactivat", "riattivat"),
     ("grant", "activate", "reactivate", "entitlement")),
    (("deleted", "eliminat", "cancellato l'account"), ("delete",)),
    (("linked", "collegat"), ("link",)),
    (("sent", "replied", "inviat", "rispost"), ("send", "respond", "reply")),
    (("forward", "escalat", "flagged", "segnalat", "inoltrat", "girat"),
     ("forward", "escalat", "flag", "mark_for_human", "handoff")),
    (("updated", "changed", "marked", "assegnat", "aggiornat"),
     ("update", "patch", "set_", "mark", "assign", "tag")),
    (("diagnostic", "diagnostica", "log", "registri"),
     ("diagnostic", "diagnostics", "diagnostic_log", "diagnostic_stream")),
)


def _trace_supports_completed_action(
    reply: str, rows: list[tuple[str, str]],
) -> bool:
    low_reply = (reply or "").lower()
    for claim_words, tool_words in _ACTION_FAMILIES:
        if not any(word in low_reply for word in claim_words):
            continue
        return any(
            any(word in f"{name} {excerpt}".lower() for word in tool_words)
            and _trace_result_succeeded(excerpt)
            for name, excerpt in rows
        )
    return False


def _trace_supports_account_state(
    reply: str, rows: list[tuple[str, str]],
) -> bool:
    """Require a same-turn authoritative BillingBear/admin read for state."""
    low_reply = (reply or "").lower()
    wants_active = bool(re.search(r"\b(?:active|attiv[oa])\b", low_reply)) and not bool(
        re.search(r"\b(?:inactive|inattiv[oa])\b", low_reply)
    )
    wants_inactive = bool(re.search(
        r"\b(?:inactive|expired|cancelled|canceled|inattiv[oa]|scadut[oa]|annullat[oa])\b",
        low_reply,
    ))
    for name, excerpt in rows:
        tool_identity = f"{name} {excerpt}".lower()
        if "billingbear" not in tool_identity and "esound_admin" not in tool_identity:
            continue
        value = str(excerpt or "").lower()
        if not _trace_result_succeeded(value):
            continue
        if wants_active and re.search(
            r"[\"']?ispremium[\"']?\s*[:=]\s*true\b|"
            r"[\"']?status[\"']?\s*[:=]\s*[\"']active[\"']",
            value,
        ):
            return True
        if wants_inactive and re.search(
            r"[\"']?ispremium[\"']?\s*[:=]\s*false\b|"
            r"[\"']?status[\"']?\s*[:=]\s*[\"'](?:inactive|expired|cancelled|canceled)[\"']",
            value,
        ):
            return True
    return False


async def _regenerate(
    model: Any, user_message: str, draft: str, *, future_release: bool = False,
    completed_action: bool = False, commercial: bool = False,
    fix_status: bool = False, evidence_trace: str = "",
) -> str:
    """One bounded rewrite that keeps the substance but drops the unbacked
    human-follow-up promise. Self-contained (no session history needed) — it
    revises the draft in place. Returns "" on any failure."""
    future_rule = (
        " The draft also promises a next or upcoming app update, release, or "
        "version. REMOVE that commitment and do not substitute another date, "
        "version, timeline, or 'soon'. It is acceptable to say only that a "
        "verified completed fix is awaiting release."
        if future_release else ""
    )
    human_rule = (
        "The draft may promise that a human, teammate, or team will follow up, "
        "verify, escalate, or get back to the customer without a real handoff. "
        "REMOVE every unbacked promise of human/team follow-up, verification, "
        "escalation, flagging, or being contacted later. "
    )
    action_rule = (
        " The draft claims that an action was completed, but this is a dry run "
        "and no mutation occurred. REMOVE every claim that something was "
        "linked, forwarded, escalated, filed, changed, fixed, refunded, or "
        "otherwise performed."
        if completed_action else ""
    )
    commercial_rule = (
        " The draft offers or promises a refund, credit, discount, free "
        "Premium, extension, or other commercial value. REMOVE that commitment; "
        "do not replace it with a conditional offer."
        if commercial else ""
    )
    fix_rule = (
        " The draft claims that a fix is implemented, complete, or awaiting "
        "release, but the retrieved evidence includes analysis/proposal-only "
        "material and does not support that status for every issue. REMOVE the "
        "completion/release-status claim. You may call an issue known only when "
        "the evidence supports that narrower statement."
        if fix_status else ""
    )
    system = (
        "You are revising a customer-support reply before it is sent. "
        + human_rule
        + "Rewrite the reply to help the customer directly with what can "
        "actually be done now. Keep the "
        "rest of the substance, facts, and tone; do not invent new facts. "
        "Output ONLY the revised reply text — no preamble, no quotes."
        + future_rule
        + action_rule
        + commercial_rule
        + fix_rule
    )
    prompt = (
        f"Customer message:\n{user_message or '(not available)'}\n\n"
        f"Draft reply (contains a forbidden unbacked promise):\n{draft}\n\n"
        + (f"\n\nVerified tool-evidence excerpts:\n{evidence_trace}" if evidence_trace else "")
        + "\n\nRevised reply:"
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


_UNVERIFIED_FIX_STATUS = (
    "Available evidence confirms tracking only; no remediation state or "
    "release date is established."
)
_UNVERIFIED_FIX_STATUS_IT = (
    "Le fonti disponibili confermano soltanto la segnalazione; non è verificato "
    "lo stato della correzione né una data di rilascio."
)
_MISSING_ACCOUNT_EVIDENCE = (
    "Please provide the account email and the store receipt or order ID before "
    "the subscription status can be verified."
)
_MISSING_ACCOUNT_EVIDENCE_IT = (
    "Per verificare lo stato dell'abbonamento, servono l'email dell'account e "
    "la ricevuta dello store o l'ID dell'ordine."
)


def _unverified_fix_status_for(text: str) -> str:
    if re.search(
        r"\b(?:il|lo|la|gli|le|un|una|problema|correzione|rilascio|"
        r"segnalazione|stato)\b|[àèéìòù]",
        text or "", re.IGNORECASE,
    ):
        return _UNVERIFIED_FIX_STATUS_IT
    return _UNVERIFIED_FIX_STATUS


def _missing_account_evidence_for(text: str) -> str:
    return (
        _MISSING_ACCOUNT_EVIDENCE_IT
        if _unverified_fix_status_for(text) == _UNVERIFIED_FIX_STATUS_IT
        else _MISSING_ACCOUNT_EVIDENCE
    )


def _strip_plain_sentences(
    text: str, *, explain_unverified_fix: bool = False,
    explain_missing_account: bool = False,
    evidence: list[tuple[str, str]] | None = None,
) -> str:
    """Remove unsafe prose sentences without making assumptions about JSON."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    kept: list[str] = []
    stripped_fix_status = False
    stripped_account_state = False
    for sentence in sentences:
        if not sentence:
            continue
        unsafe_fix = claims_completed_fix(sentence)
        unsafe_account = claims_account_state(sentence)
        if unsafe_fix:
            stripped_fix_status = True
        if unsafe_account:
            stripped_account_state = True
        if (
            promises_followup(sentence)
            or promises_future_release(sentence)
            or claims_completed_action(sentence)
            or promises_commercial_value(sentence)
            or unsafe_fix
            or (
                evidence is not None
                and (
                    unbacked_identifiers(sentence, evidence)
                    or unbacked_money(sentence, evidence)
                    or (
                        claims_issue_tracked(sentence)
                        and not _trace_supports_tracking(evidence)
                    )
                )
            )
        ):
            continue
        kept.append(sentence)
    if explain_unverified_fix and stripped_fix_status:
        kept.append(_unverified_fix_status_for(text))
    if explain_missing_account and stripped_account_state:
        kept.append(_missing_account_evidence_for(text))
    return " ".join(kept).strip()


def _sanitize_json_value(
    value: Any, *, explain_unverified_fix: bool = False,
    explain_missing_account: bool = False,
    evidence: list[tuple[str, str]] | None = None,
) -> Any:
    """Recursively sanitize strings while preserving valid JSON structure."""
    if isinstance(value, str):
        return _strip_plain_sentences(
            value, explain_unverified_fix=explain_unverified_fix,
            explain_missing_account=explain_missing_account,
            evidence=evidence,
        )
    if isinstance(value, list):
        return [
            _sanitize_json_value(
                item, explain_unverified_fix=explain_unverified_fix,
                explain_missing_account=explain_missing_account,
                evidence=evidence,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: _sanitize_json_value(
                item, explain_unverified_fix=explain_unverified_fix,
                explain_missing_account=explain_missing_account,
                evidence=evidence,
            )
            for key, item in value.items()
        }
    return value


_JSON_FENCE = re.compile(r"```json\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _strip_forbidden_sentences(
    text: str, *, explain_unverified_fix: bool = False,
    explain_missing_account: bool = False,
    evidence: list[tuple[str, str]] | None = None,
) -> str:
    """Last-resort fail-closed cleanup used only by a lean local event.

    JSON payloads need special handling: splitting the entire response on
    punctuation can delete a quote or comma that happens to share a sentence
    with an unsafe claim. Parse fenced JSON, sanitize its string values, and
    serialize it again. If a model emitted malformed JSON, drop that optional
    block instead of returning a newly or already broken machine payload.
    """
    stripped = text.strip()

    # A response may itself be a raw JSON object rather than Markdown.
    try:
        raw_value = json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        raw_value = None
    if isinstance(raw_value, (dict, list)):
        return json.dumps(
            _sanitize_json_value(
                raw_value, explain_unverified_fix=explain_unverified_fix,
                explain_missing_account=explain_missing_account,
            evidence=evidence,
            ),
            ensure_ascii=False,
        )

    pieces: list[str] = []
    cursor = 0
    for match in _JSON_FENCE.finditer(stripped):
        prose = _strip_plain_sentences(
            stripped[cursor:match.start()],
            explain_unverified_fix=explain_unverified_fix,
            explain_missing_account=explain_missing_account,
            evidence=evidence,
        )
        if prose:
            pieces.append(prose)
        try:
            payload = json.loads(match.group(1).strip())
        except (TypeError, ValueError, json.JSONDecodeError):
            # Optional structured mirror is unsafe to preserve if it cannot be
            # parsed. The customer-facing prose remains available.
            cursor = match.end()
            continue
        sanitized = _sanitize_json_value(
            payload, explain_unverified_fix=explain_unverified_fix,
            explain_missing_account=explain_missing_account,
            evidence=evidence,
        )
        pieces.append(
            "```json\n"
            + json.dumps(sanitized, ensure_ascii=False, indent=2)
            + "\n```"
        )
        cursor = match.end()
    tail = _strip_plain_sentences(
        stripped[cursor:], explain_unverified_fix=explain_unverified_fix,
        explain_missing_account=explain_missing_account,
            evidence=evidence,
    )
    if tail:
        pieces.append(tail)

    cleaned = "\n\n".join(pieces).strip()
    return cleaned or (
        "I cannot verify the draft's proposed follow-up or release commitment, "
        "so I will not present it as confirmed."
    )


def _replace_structured_reply(text: str, replacement: str) -> str:
    """Replace only a structured payload's customer-facing ``reply`` field."""
    stripped = (text or "").strip()
    try:
        raw_value = json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        raw_value = None
    if isinstance(raw_value, dict) and "reply" in raw_value:
        raw_value["reply"] = replacement
        return json.dumps(raw_value, ensure_ascii=False)

    pieces: list[str] = []
    cursor = 0
    replaced = False
    for match in _JSON_FENCE.finditer(stripped):
        pieces.append(stripped[cursor:match.start()])
        try:
            payload = json.loads(match.group(1).strip())
        except (TypeError, ValueError, json.JSONDecodeError):
            pieces.append(match.group(0))
            cursor = match.end()
            continue
        if isinstance(payload, dict) and "reply" in payload:
            payload["reply"] = replacement
            replaced = True
        pieces.append(
            "```json\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n```"
        )
        cursor = match.end()
    pieces.append(stripped[cursor:])
    return "".join(pieces).strip() if replaced else replacement


_NO_EVIDENCE_FALLBACK = (
    "I don't have verified information to share on this yet. "
    "Could you send the details again so I can check them properly?"
)


async def guard_reply(
    agent: Any, session_id: Optional[str], user_message: str, reply: str,
    *, model_override: Any = None,
) -> str:
    """Return *reply*, or a rewrite of it with an unbacked human-follow-up
    promise removed. FAIL-OPEN: returns *reply* unchanged on disabled / no
    visibility / no promise / backed promise / regeneration failure / any error.
    """
    from src.core.execution_profile import lean_local_event_active

    strict_local = lean_local_event_active()
    if not (enabled() or strict_local) or not reply:
        return reply
    try:
        trace_rows: list[tuple[str, str]] = []
        if strict_local:
            from src.core import tool_trace
            trace_rows = list(tool_trace.peek(session_id) or [])
        from src.core.dry_run import is_dry_run
        dry_run = is_dry_run()

        has_human_promise = promises_followup(reply)
        has_future_promise = strict_local and promises_future_release(reply)
        has_action_claim = (
            strict_local
            and claims_completed_action(reply)
            and (dry_run or not _trace_supports_completed_action(reply, trace_rows))
        )
        has_unsupported_account_state = (
            strict_local
            and claims_account_state(reply)
            and (dry_run or not _trace_supports_account_state(reply, trace_rows))
        )
        has_commercial_promise = strict_local and promises_commercial_value(reply)
        # Grounding checks need trace visibility to be meaningful. Under the
        # strict-local profile capture is always on, so an empty trace is not
        # "cannot tell" - it means the reply names a task or an amount that no
        # tool ever returned, which is the fabrication itself.
        evidence = trace_rows if strict_local else None
        fabricated_ids = unbacked_task_ids(reply, trace_rows) if strict_local else []
        fabricated_money = unbacked_money(reply, trace_rows) if strict_local else []
        unbacked_tracking = (
            strict_local
            and claims_issue_tracked(reply)
            and not _trace_supports_tracking(trace_rows)
        )
        has_unsupported_fix_status = (
            strict_local
            and claims_completed_fix(reply)
            and (
                not trace_rows
                or _trace_contradicts_completed_fix(trace_rows)
                or _trace_uses_historical_receipt(trace_rows)
            )
        )
        if not any((
            has_human_promise, has_future_promise, has_action_claim,
            has_commercial_promise, has_unsupported_fix_status,
            has_unsupported_account_state, fabricated_ids, fabricated_money,
            unbacked_tracking,
        )):
            return reply
        if has_unsupported_account_state:
            cleaned = _replace_structured_reply(
                reply, _missing_account_evidence_for(reply),
            )
            elog(
                "reply_guard.stripped",
                level="warning",
                session_id=session_id,
                reason="dry_run_unverified_account_state",
                tools=len(trace_rows),
            )
            return cleaned
        # Receipt-only fix claims are a particularly common small-model error:
        # an old closed task is promoted to the status of today's recurrence.
        # Strip this deterministically so a second local generation cannot lose
        # a structured JSON envelope or repeat the same inference.
        if has_unsupported_fix_status and _trace_uses_historical_receipt(trace_rows):
            cleaned = _strip_forbidden_sentences(
                reply, explain_unverified_fix=True,
            )
            elog(
                "reply_guard.stripped",
                level="warning",
                session_id=session_id,
                reason="historical_receipt_not_current_fix_status",
                tools=len(trace_rows),
            )
            return cleaned
        if fabricated_ids or fabricated_money or unbacked_tracking:
            # Deterministic strip, no regeneration: asking the same model to
            # rewrite a sentence it invented tends to produce a differently
            # worded invention.
            cleaned = _strip_forbidden_sentences(reply, evidence=evidence)
            elog(
                "reply_guard.stripped",
                level="warning",
                session_id=session_id,
                reason="ungrounded_identifier_amount_or_tracking",
                ids=len(fabricated_ids),
                amounts=len(fabricated_money),
                tools=len(trace_rows),
            )
            return cleaned or _NO_EVIDENCE_FALLBACK
        tool_names: list[str] = []
        if has_human_promise:
            # Without trace visibility a normal turn cannot judge whether a
            # human promise is backed. A future-release violation is independent
            # and can still be removed from a strict local event.
            from src.core import tool_trace
            if not tool_trace._enabled():
                if not any((
                    has_future_promise, has_action_claim,
                    has_commercial_promise, has_unsupported_fix_status,
                    has_unsupported_account_state,
                )):
                    return reply
                has_human_promise = False
            else:
                rows = tool_trace.peek(session_id)
                tool_names = [name for (name, _excerpt) in rows] if rows else []
                if _has_successful_backing_action(rows or []):
                    has_human_promise = False
                    if not any((
                        has_future_promise, has_action_claim,
                        has_commercial_promise, has_unsupported_fix_status,
                        has_unsupported_account_state,
                    )):
                        return reply

        # The per-run override wins: it is the model that wrote this reply.
        model = model_override or getattr(agent, "model", None)
        if model is None:
            return reply
        revised = await _regenerate(
            model,
            user_message,
            reply,
            future_release=has_future_promise,
            completed_action=has_action_claim,
            commercial=has_commercial_promise,
            fix_status=has_unsupported_fix_status,
            evidence_trace=(
                "\n".join(f"{name}: {excerpt}" for name, excerpt in trace_rows)
            )[:3_000],
        )
        revised_is_safe = (
            revised
            and (not has_human_promise or not promises_followup(revised))
            # A rewrite can trade one violation for another (observed: it
            # removed "already implemented" but invented "next update"). A
            # strict-local rewrite therefore has to pass the whole policy, not
            # only the predicates that fired on the original draft.
            and (not strict_local or not promises_future_release(revised))
            and (
                not strict_local
                or not dry_run
                or not claims_completed_action(revised)
            )
            and (
                not strict_local
                or not promises_commercial_value(revised)
            )
            and (
                not strict_local
                or not claims_completed_fix(revised)
            )
            and (
                not strict_local
                or not claims_account_state(revised)
                or _trace_supports_account_state(revised, trace_rows)
            )
        )
        if revised_is_safe:
            elog(
                "reply_guard.rewrote",
                level="warning",
                session_id=session_id,
                reason=(
                    "local_event_policy_violation"
                    if any((
                        has_future_promise, has_action_claim,
                        has_commercial_promise, has_unsupported_fix_status,
                    ))
                    else "unbacked_human_followup_promise"
                ),
                tools=len(tool_names),
            )
            return revised
        if strict_local:
            cleaned = _strip_forbidden_sentences(
                reply, explain_unverified_fix=has_unsupported_fix_status,
            )
            elog(
                "reply_guard.stripped",
                level="warning",
                session_id=session_id,
                reason="rewrite_failed_strict_local_event",
                tools=len(tool_names),
            )
            return cleaned
        # Normal turns retain the existing fail-open behaviour.
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
