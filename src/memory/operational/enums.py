"""Canonical wire-enum normalization for operational records."""

from __future__ import annotations

from typing import Any


RUN_STATUSES = frozenset(
    {
        "pending",
        "queued",
        "received",
        "running",
        "success",
        "failed",
        "cancelled",
        "rejected",
        "interrupted",
        "skipped",
        "timed_out",
    }
)
MESSAGE_STATUSES = frozenset(
    {"streaming", "complete", "interrupted", "cancelled", "failed"}
)
TOOL_STATUSES = frozenset({"pending", "running", "success", "error", "cancelled"})
COMPLETENESS_VALUES = frozenset(
    {"complete", "partial", "legacy_compacted", "malformed_source", "unknown"}
)


class UnmappedStatusError(ValueError):
    """A source status has no reviewed lossless wire mapping."""


def _source_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip()


_RUN_STATUS_MAP = {
    "PENDING": "pending",
    "PAUSED": "pending",
    "QUEUED": "queued",
    "RECEIVED": "received",
    "RUNNING": "running",
    "CANCELLING": "running",
    "COMPLETED": "success",
    "COMPLETE": "success",
    "SUCCESS": "success",
    "SUCCEEDED": "success",
    "ERROR": "failed",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
    "CANCELED": "cancelled",
    "REJECTED": "rejected",
    "BLOCKED": "rejected",
    "DENIED": "rejected",
    "INTERRUPTED": "interrupted",
    "SKIPPED": "skipped",
    "TIMEOUT": "timed_out",
    "TIMED_OUT": "timed_out",
}


def normalize_run_status(value: Any) -> tuple[str, str]:
    raw = _source_value(value)
    normalized = _RUN_STATUS_MAP.get(raw.upper())
    if normalized is None:
        raise UnmappedStatusError(f"unmapped run status: {raw!r}")
    return normalized, raw


_TOOL_STATUS_MAP = {
    "REQUESTED": "pending",
    "PENDING": "pending",
    "STARTED": "running",
    "RUNNING": "running",
    "COMPLETED": "success",
    "SUCCEEDED": "success",
    "SUCCESS": "success",
    "FAILED": "error",
    "ERROR": "error",
    "DENIED": "error",
    "CANCELLED": "cancelled",
    "CANCELED": "cancelled",
    "INTERRUPTED": "cancelled",
}


def normalize_tool_status(
    value: Any,
    *,
    tool_call_error: bool | None = None,
    result_present: bool = False,
) -> tuple[str, str]:
    raw = _source_value(value)
    if raw:
        normalized = _TOOL_STATUS_MAP.get(raw.upper())
        if normalized is None:
            raise UnmappedStatusError(f"unmapped tool status: {raw!r}")
        if tool_call_error is True:
            normalized = "error"
        return normalized, raw
    if tool_call_error is True:
        return "error", "tool_call_error"
    if result_present:
        return "success", "completed"
    return "running", "running"

