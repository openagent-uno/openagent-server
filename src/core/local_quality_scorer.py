"""Deterministic quality scorer for support replies.

The model-driven version of this task was measured against the real scheduler
and skipped the recording step in two firings out of three: it graded the
replies, printed the verdicts, and never called ``quality_record``. Once it
even reported ``RECORDING ok`` for a write the key had refused. Both failures
are invisible downstream - the log looks like a successful run and the table
stays empty.

So the shape is the one the support controller already proved: the code
fetches, the code computes the score, the code records and the code decides
whether to alert. The model is asked for exactly one thing - six sub-scores as
JSON - and anything it returns outside 0..1 is discarded rather than trusted.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from src.core.execution_profile import (
    stateless_completion_scope,
    strict_local_only_scope,
)
from src.core.logging import elog
from src.core.tool_scope import reset_tool_allowlist, set_tool_allowlist

# grounding and appropriateness are the two that decide whether a reply is
# safe; tone, language, length and attachment-reading decide whether it is
# good. The weights follow the note the Lyra scorer has used since July.
_WEIGHTS = {
    "grounding": 2.0,
    "appropriateness": 2.0,
    "tone": 1.0,
    "f7_read": 1.0,
    "language": 1.0,
    "length": 1.0,
}
_DIMENSIONS = tuple(_WEIGHTS)

_SYSTEM = (
    "You grade ONE customer-support reply. Output JSON only, exactly: "
    '{"grounding":0-1,"appropriateness":0-1,"tone":0-1,"f7_read":0-1,'
    '"language":0-1,"length":0-1}. Every value is a number between 0 and 1. '
    "No prose, no verdict, no explanation - the caller computes those.\n"
    "grounding: is every claim backed by the receipts given? A reply that says "
    "a bug is known or tracked while has_task is false AND escalated is false "
    "scores 0.\n"
    "appropriateness: is the advice right for the state actually shown?\n"
    "tone: warm and specific, especially to an angry customer.\n"
    "f7_read: 1 if there was no attachment, or the attachment was read; 0 if "
    "the thread had an attachment and attachment_read is false.\n"
    "language: 1 if the reply is in the customer's language. A Play review "
    "reaches us translated into English, so judge against reviewer_language "
    "when it is given.\n"
    "length: 1 if the reply is under 300 characters on a short channel."
)


def enabled() -> bool:
    return os.environ.get(
        "OPENAGENT_QUALITY_SCORER", "1",
    ).strip().lower() in {"1", "true", "yes", "on"}


def _clamp(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number < 0.0 or number > 1.0:
        return None
    return number


def weighted_score(dimensions: dict[str, float]) -> float:
    total = sum(_WEIGHTS[name] * dimensions[name] for name in _DIMENSIONS)
    return round(total / sum(_WEIGHTS.values()), 3)


def verdict_for(score: float, dimensions: dict[str, float] | None = None) -> str:
    # A reply that asserts something nothing backs is not "acceptable" however
    # warm and well-formed the rest of it is: the weighted average alone would
    # have called a fabricated tracking claim OK.
    if dimensions is not None and dimensions.get("grounding", 1.0) <= 0.0:
        return "BAD"
    if score >= 0.8:
        return "GOOD"
    return "OK" if score >= 0.5 else "BAD"


async def run(
    agent: Any, event: dict[str, Any], pool: Any, session_id: str,
    *, product: str = "esound", limit: int = 20,
) -> dict[str, Any]:
    """Fetch, grade, record and decide - all of it in code but the grading."""
    from src.core.local_support_controller import _call_first, _succeeded

    _tool, listed = await _call_first(
        pool, "replio", ("replio_replies_to_score", "replies_to_score"),
        {"product": product, "limit": limit}, required=False,
    )
    items = (listed or {}).get("items") if isinstance(listed, dict) else None
    if not isinstance(items, list) or not items:
        return {"scored": 0, "recorded": 0, "bad": 0, "read_only": False,
                "alerted": False, "rows": []}

    rows: list[dict[str, Any]] = []
    recorded = read_only = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        dimensions = await grade_one(agent, event, item, session_id)
        if dimensions is None:
            continue
        score = weighted_score(dimensions)
        verdict = verdict_for(score, dimensions)
        row = {
            "message_id": str(item.get("message_id") or ""),
            "thread_id": str(item.get("thread_id") or ""),
            "score": score, "verdict": verdict, "dimensions": dimensions,
        }
        rows.append(row)
        _rtool, receipt = await _call_first(
            pool, "replio", ("replio_quality_record", "quality_record"),
            {
                "message_id": row["message_id"], "thread_id": row["thread_id"],
                "product": product,
                "channel_kind": str(item.get("channel_kind") or ""),
                "score": score, "verdict": verdict, "dimensions": dimensions,
                "grader": "qwen3-moe-local/deterministic-scorer",
                "notes": _weakest(dimensions),
            },
            required=False,
        )
        if _succeeded(receipt):
            recorded += 1
        else:
            # The eSound key is read-only. That is a known state, not an
            # error to retry - but it must never be reported as recorded.
            if "read-only" in json.dumps(receipt, default=str).lower():
                read_only += 1

    # Write the correction the weak dimension earns, so the NEXT reply on this
    # thread carries it. Without this half the loop is open: the scorer
    # notices, and nothing changes.
    corrected = 0
    for row in rows:
        # Not gated on the verdict: a reply that scores well overall but went
        # out in the wrong language still earns the language correction, which
        # is exactly the rule the Lyra scorer has used since July.
        earned = correction_for(row["dimensions"])
        if earned is None or not row["thread_id"]:
            continue
        title, text = earned
        _ltool, receipt = await _call_first(
            pool, "replio",
            ("replio_thread_learning_add", "thread_learning_add"),
            {"thread_id": row["thread_id"], "title": title,
             "content": text, "kind": "correction"},
            required=False,
        )
        if _succeeded(receipt):
            corrected += 1

    bad = [row for row in rows if row["verdict"] == "BAD"]
    average = round(sum(r["score"] for r in rows) / len(rows), 3) if rows else 0.0
    alerted = False
    if rows and (average < 0.65 or len(bad) >= 2):
        lines = "\n".join(
            f"{r['message_id'][:8]} — {_weakest(r['dimensions'])}" for r in bad[:5]
        )
        _mtool, sent = await _call_first(
            pool, "messaging",
            ("messaging_send_telegram", "send_telegram"),
            {"chat_id": os.environ.get("OPENAGENT_OWNER_TELEGRAM", "7284821"),
             "text": (f"{product}: qualita' in calo — media {average}, "
                      f"{len(bad)} BAD su {len(rows)}.\n{lines}")},
            required=False,
        )
        alerted = _succeeded(sent)
    return {"scored": len(rows), "recorded": recorded, "bad": len(bad),
            "corrections_written": corrected,
            "average": average, "read_only": bool(read_only),
            "alerted": alerted, "rows": rows}


async def digest(
    pool: Any, *, product: str = "esound", window_days: int = 7,
) -> dict[str, Any]:
    """System-level self-improvement: which dimension is failing REPEATEDLY.

    A per-thread correction fixes one reply. A dimension that is weakest in
    half the graded replies is not a bad reply, it is a bug - and every defect
    worth finding in this system so far has had that shape: the fixed English
    fallback, Spanish read as Italian, the review language, the premium lookup.
    So the digest counts, and says which one to go and fix.
    """
    from src.core.local_support_controller import _call_first, _succeeded

    _tool, stats = await _call_first(
        pool, "replio", ("replio_quality_stats", "quality_stats"),
        {"window_days": window_days, "product": product}, required=False,
    )
    _tool, recent = await _call_first(
        pool, "replio", ("replio_learnings_list", "learnings_list"),
        {"limit": 100}, required=False,
    )
    counts: dict[str, int] = {}
    items = (recent or {}).get("items") if isinstance(recent, dict) else None
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("kind") or "").lower() != "correction":
            continue
        title = str(item.get("title") or "")
        name = title.split(":", 1)[1].strip() if ":" in title else title.strip()
        if name in _WEIGHTS:
            counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    systemic = [
        {"dimension": name, "count": n, "share": round(n / total, 3)}
        for name, n in ranked
        # A third of all corrections pointing at one dimension is a pattern,
        # not noise.
        if total >= 6 and n / total >= 0.33
    ]
    return {
        "product": product,
        "window_days": window_days,
        "quality": stats if _succeeded(stats) else None,
        "corrections_by_dimension": dict(ranked),
        "systemic": systemic,
    }


def _weakest(dimensions: dict[str, float]) -> str:
    name = min(dimensions, key=lambda key: dimensions[key])
    return f"{name}={dimensions[name]}"


# The correction a weak dimension earns. Fixed sentences on purpose: a
# correction is injected into every later reply on that thread, so it must be
# PROCEDURAL and it must not be written by a model. The scorer's own note says
# a correction may never carry a product fact - the surest way to honour that
# is to never let free text into one.
_CORRECTION_FOR = {
    "grounding": (
        "Do not state anything that this turn's receipts do not show. In "
        "particular, never say an issue is known, tracked or forwarded unless "
        "a task or a human hand-off actually succeeded in the same turn."
    ),
    "appropriateness": (
        "Do not suggest a fix before verifying the state that would make it "
        "the right fix. Ask for what is missing instead."
    ),
    "language": (
        "Answer this customer in the language they wrote in. On a store "
        "review the reviewer's declared language wins over the translated text."
    ),
    "f7_read": (
        "This thread carries an attachment. Open it before answering, or say "
        "plainly that it could not be read and ask for a description."
    ),
    "tone": (
        "Acknowledge what this customer said before asking anything, and greet "
        "them by name when the channel gives one."
    ),
    "length": (
        "Keep the reply short: under 300 characters, plain sentences, no lists."
    ),
}


def correction_for(dimensions: dict[str, float]) -> tuple[str, str] | None:
    """The (title, text) a graded reply earns, or nothing when it earns none."""
    weakest = min(dimensions, key=lambda key: dimensions[key])
    if dimensions[weakest] > 0.5:
        return None
    text = _CORRECTION_FOR.get(weakest)
    return (f"correction: {weakest}", text) if text else None


def _extract_json(text: str) -> dict[str, Any] | None:
    raw = str(text or "")
    start = raw.find("{")
    while start != -1:
        depth = 0
        for index in range(start, len(raw)):
            if raw[index] == "{":
                depth += 1
            elif raw[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(raw[start:index + 1])
                    except ValueError:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = raw.find("{", start + 1)
    return None


async def grade_one(agent: Any, event: dict[str, Any], item: dict[str, Any],
                    session_id: str) -> dict[str, float] | None:
    """Six numbers, or nothing. A partial grade is not a grade."""
    model = getattr(agent, "model", None)
    model_id = str((event or {}).get("model") or "").strip()
    if model_id and callable(getattr(model, "build_override_model", None)):
        model = model.build_override_model(model_id)
    if model is None:
        return None
    packet = {
        "customer_message": str(item.get("last_inbound") or "")[:2000],
        "reply": str(item.get("reply") or "")[:2000],
        "has_task": bool(item.get("has_task")),
        "escalated": bool(item.get("escalated")),
        "had_attachment": bool(item.get("inbound_attachments")),
        "attachment_read": bool(item.get("attachment_read")),
        "channel_kind": str(item.get("channel_kind") or ""),
    }
    token = set_tool_allowlist([])
    try:
        with strict_local_only_scope(True), stateless_completion_scope(True):
            response = await asyncio.wait_for(
                model.generate(
                    messages=[{"role": "user",
                               "content": json.dumps(packet, ensure_ascii=False)}],
                    system=_SYSTEM,
                    session_id=f"{session_id}:quality-grade",
                ),
                timeout=max(1.0, float(os.environ.get(
                    "OPENAGENT_QUALITY_GRADE_TIMEOUT_SECONDS", "20",
                ))),
            )
    except Exception as exc:  # noqa: BLE001 - one ungraded reply is not a failure
        elog("quality_scorer.grade_failed", level="warning", error=str(exc)[:200])
        return None
    finally:
        reset_tool_allowlist(token)
    payload = _extract_json(getattr(response, "content", "")) or {}
    graded: dict[str, float] = {}
    for name in _DIMENSIONS:
        value = _clamp(payload.get(name))
        if value is None:
            # A missing dimension makes the weighted score meaningless, and a
            # meaningless score in the table is worse than a gap.
            return None
        graded[name] = value
    return graded
