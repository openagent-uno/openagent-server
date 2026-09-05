"""Decode billing evidence without turning transport success into account state."""
from __future__ import annotations

import json
import re
from typing import Any


def billing_payload(value: Any, depth: int = 0) -> dict[str, Any]:
    if depth > 8:
        return {"ok": False, "error": "billing_envelope_depth"}
    if isinstance(value, str):
        text = value.strip()
        status = re.match(r"^HTTP\s+(\d{3})[^\n]*\n", text)
        if status:
            if not 200 <= int(status[1]) < 300:
                return {"ok": False, "status": int(status[1])}
            text = text[status.end():]
        try:
            return billing_payload(json.loads(text), depth + 1)
        except (ValueError, TypeError):
            return {"ok": False, "error": "billing_payload_unreadable"}
    if not isinstance(value, dict):
        return {"ok": False, "error": "billing_payload_not_object"}
    if any(value.get(key) is flag for key, flag in (
        ("isError", True), ("is_error", True), ("ok", False), ("success", False),
    )):
        return {"ok": False, "error": "billing_tool_failed"}
    status = value.get("status")
    if isinstance(status, int) and not 200 <= status < 300:
        return {"ok": False, "status": status}
    for key in ("structuredContent", "structured_content"):
        if key in value and value[key] is not None:
            return billing_payload(value[key], depth + 1)
    if isinstance(value.get("content"), list):
        blocks = value["content"]
        texts = [b.get("text") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        if len(texts) != 1 or len(blocks) != 1:
            return {"ok": False, "error": "billing_payload_ambiguous"}
        return billing_payload(texts[0], depth + 1)
    return value


def explicit_premium(value: dict[str, Any]) -> bool | None:
    """Missing/null/string values are unknown, including the string 'false'."""
    values = [value[k] for k in ("isPremium", "is_premium") if k in value]
    if not values or any(type(v) is not bool for v in values):
        return None
    return values[0] if all(v == values[0] for v in values) else None
