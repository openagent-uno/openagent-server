"""Validate cited, non-operational guidance before using a retrieved answer."""
from __future__ import annotations

import re
from typing import Any
from src.core import reply_guard


def excerpts(result: Any) -> list[str]:
    """Only document body fields; a title/path is not supporting evidence."""
    if isinstance(result, list):
        return [s for row in result for s in excerpts(row)][:6]
    if not isinstance(result, dict):
        return []
    for key in ("results", "documents", "items", "data"):
        if isinstance(result.get(key), list):
            return excerpts(result[key])
    for key in ("excerpt", "snippet", "body", "text", "content"):
        text = result.get(key)
        if isinstance(text, str) and len(text.strip()) >= 80:
            return [text[:5000]]
    return []


def validated_answer(packet: Any, sources: list[str]) -> str:
    if not isinstance(packet, dict) or packet.get("applicable") is not True:
        return ""
    answer = packet.get("answer")
    quotes = packet.get("quotes")
    if not isinstance(answer, str) or not 20 <= len(answer) <= 1000:
        return ""
    if not isinstance(quotes, list) or not 1 <= len(quotes) <= 4:
        return ""
    norm = lambda s: re.sub(r"\s+", " ", s).strip().casefold()
    if not all(isinstance(q, str) and len(q.strip()) >= 40 and
               any(norm(q) in norm(s) for s in sources) for q in quotes):
        return ""
    if any(check(answer) for check in (
        reply_guard.claims_human_identity, reply_guard.claims_profile_change,
        reply_guard.claims_completed_action, reply_guard.claims_account_state,
        reply_guard.promises_future_release, reply_guard.promises_followup,
    )):
        return ""
    return answer.strip()
