"""Provider-neutral execution envelopes for unattended agent runs.

Prompts describe the mission; this module enforces the resource and capability
boundary around it. The policy is deliberately model-agnostic and contains no
product concepts, so the same scheduled task can move between providers without
losing its timeout, tool budget, or MCP-family allowlist.
"""

from __future__ import annotations

import contextvars
import json
from collections.abc import Mapping
from typing import Any


_ALLOWED_KEYS = frozenset(
    {"max_tool_calls", "timeout_seconds", "allowed_tool_families"}
)
_policy_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "openagent_execution_policy", default=None,
)


def normalize_execution_policy(value: Any) -> dict[str, Any]:
    """Validate and canonicalise a task execution policy.

    ``None``/empty means the historical unrestricted defaults. Unknown fields
    fail closed so a misspelled safety control never appears to have landed.
    """
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("execution_policy must be valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("execution_policy must be an object")
    unknown = sorted(set(value) - _ALLOWED_KEYS)
    if unknown:
        raise ValueError(
            "unknown execution_policy fields: " + ", ".join(unknown)
        )

    out: dict[str, Any] = {}
    if value.get("max_tool_calls") is not None:
        raw = value["max_tool_calls"]
        if isinstance(raw, bool):
            raise ValueError("execution_policy.max_tool_calls must be an integer")
        try:
            limit = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "execution_policy.max_tool_calls must be an integer"
            ) from exc
        if limit < 1 or limit > 500:
            raise ValueError(
                "execution_policy.max_tool_calls must be between 1 and 500"
            )
        out["max_tool_calls"] = limit

    if value.get("timeout_seconds") is not None:
        raw = value["timeout_seconds"]
        if isinstance(raw, bool):
            raise ValueError("execution_policy.timeout_seconds must be a number")
        try:
            timeout = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "execution_policy.timeout_seconds must be a number"
            ) from exc
        if timeout < 5 or timeout > 86_400:
            raise ValueError(
                "execution_policy.timeout_seconds must be between 5 and 86400"
            )
        out["timeout_seconds"] = timeout

    if "allowed_tool_families" in value:
        raw = value.get("allowed_tool_families")
        if raw is None:
            pass
        elif not isinstance(raw, (list, tuple)):
            raise ValueError(
                "execution_policy.allowed_tool_families must be an array"
            )
        else:
            families: list[str] = []
            for item in raw:
                family = str(item).strip()
                if not family:
                    raise ValueError(
                        "execution_policy.allowed_tool_families cannot contain blanks"
                    )
                if len(family) > 100:
                    raise ValueError("execution policy tool family is too long")
                if family not in families:
                    families.append(family)
            if len(families) > 64:
                raise ValueError(
                    "execution_policy.allowed_tool_families supports at most 64 entries"
                )
            # An empty array is meaningful: no MCP tools at all.
            out["allowed_tool_families"] = families
    return out


def encode_execution_policy(value: Any) -> str | None:
    policy = normalize_execution_policy(value)
    return (
        json.dumps(policy, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if policy
        else None
    )


def task_execution_policy(task: Mapping[str, Any]) -> dict[str, Any]:
    if "execution_policy" in task:
        return normalize_execution_policy(task.get("execution_policy"))
    return normalize_execution_policy(task.get("execution_policy_json"))


def narrow_execution_policy(
    parent: Mapping[str, Any] | None,
    child: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Compose nested envelopes without allowing the child to expand them."""
    outer = normalize_execution_policy(parent)
    inner = normalize_execution_policy(child)
    out: dict[str, Any] = {}
    for key in ("max_tool_calls", "timeout_seconds"):
        values = [policy[key] for policy in (outer, inner) if key in policy]
        if values:
            out[key] = min(values)

    outer_tools = outer.get("allowed_tool_families")
    inner_tools = inner.get("allowed_tool_families")
    if outer_tools is not None and inner_tools is not None:
        from src.core.tool_scope import normalize_family

        inner_set = {normalize_family(item) for item in inner_tools}
        out["allowed_tool_families"] = [
            item for item in outer_tools
            if normalize_family(item) in inner_set
        ]
    elif outer_tools is not None:
        out["allowed_tool_families"] = list(outer_tools)
    elif inner_tools is not None:
        out["allowed_tool_families"] = list(inner_tools)
    return out


def event_execution_policy(event: Mapping[str, Any]) -> dict[str, Any]:
    if "execution_policy" in event:
        return normalize_execution_policy(event.get("execution_policy"))
    return normalize_execution_policy(event.get("execution_policy_json"))


def current_execution_policy() -> dict[str, Any] | None:
    policy = _policy_var.get()
    return dict(policy) if policy is not None else None


def current_max_tool_calls() -> int | None:
    policy = _policy_var.get() or {}
    value = policy.get("max_tool_calls")
    return int(value) if value is not None else None


def set_execution_policy(value: Any) -> contextvars.Token:
    policy = normalize_execution_policy(value)
    return _policy_var.set(policy or None)


def reset_execution_policy(token: contextvars.Token) -> None:
    _policy_var.reset(token)
