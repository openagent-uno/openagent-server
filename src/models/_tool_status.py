"""API-native ToolExecution → wire JSON.

The universal app consumes the runtime's native ``ToolExecution.to_dict()``
shape directly. This module is the single conversion point so live
streaming and rehydration emit byte-identical JSON for the same
``ToolExecution``.

Phase (running / completed / error) is derived in the UI from the
``tool_call_error`` flag and the presence/absence of ``result`` — no
synthetic status enum on the wire.
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any, Awaitable, Callable

_logger = logging.getLogger(__name__)


async def emit_tool_status(
    on_status: Callable[[str], Awaitable[None]] | None,
    tool_exec: Any,
    *,
    error_text: str | None = None,
    phase: str | None = None,
) -> None:
    """Encode ``tool_exec`` and forward to ``on_status``.

    Shared by every runtime tool-status emitter (NativeProvider,
    dispatcher helpers) so the wire envelope and the defensive
    on_status error handling can't drift across call sites.
    """
    if on_status is None:
        return
    encoded = tool_exec_to_wire_json(tool_exec, error_text=error_text, phase=phase)
    if encoded is None:
        return
    try:
        await on_status(encoded)
    except Exception as e:  # noqa: BLE001
        _logger.debug("on_status callback raised: %s", e)


def tool_exec_to_wire_json(
    tool_exec: Any,
    *,
    error_text: str | None = None,
    phase: str | None = None,
) -> str | None:
    """Live-path encoder: runtime ``ToolExecution`` → JSON string.

    When ``error_text`` is given (``ToolCallErrorEvent``) the encoder
    sets ``tool_call_error=True`` and stashes the message in ``result``
    — the runtime's error event carries the text inline but the stored
    ``ToolExecution`` only keeps the bool, so ``result`` is the durable
    carrier the UI inspects.

    ``phase="started"`` forces ``result=None`` on the payload so the
    non-streaming ``generate()`` path can emit a "running" frame from
    a ToolExecution that already carries its final result. The UI's
    local phase derivation reads ``result`` absence as still-running.
    """
    if tool_exec is None or not getattr(tool_exec, "tool_name", None):
        return None
    execution_host = _ensure_execution_host(tool_exec)
    payload = _to_dict(tool_exec)
    # Runtime ToolExecution serializers enumerate declared fields and may omit
    # the trusted execution_host attribute stamped above.  Overlay only that
    # field on their native envelope so live status frames cannot lose host
    # attribution while every runtime-owned field remains byte-for-byte intact.
    payload["execution_host"] = execution_host
    if error_text is not None:
        payload["tool_call_error"] = True
        payload["result"] = error_text
    elif phase == "started":
        payload["result"] = None
    return _json.dumps(payload)


def stored_tool_to_wire(stored: dict[str, Any]) -> dict[str, Any] | None:
    """Rehydration-path encoder: stored ``ToolExecution`` dict → wire dict.

    Stored rows are already in the runtime's native shape; this just gates on
    ``tool_name`` presence so the UI doesn't render empty chips.
    """
    if not stored.get("tool_name"):
        return None
    payload = dict(stored)
    if not isinstance(payload.get("execution_host"), dict):
        payload["execution_host"] = _default_execution_host(payload.get("tool_args"))
    return payload


def _to_dict(tool_exec: Any) -> dict[str, Any]:
    """Prefer ``ToolExecution.to_dict()`` when available, else build manually."""
    fn = getattr(tool_exec, "to_dict", None)
    if callable(fn):
        try:
            d = fn()
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return {
        "tool_name": getattr(tool_exec, "tool_name", None),
        "tool_call_id": getattr(tool_exec, "tool_call_id", None),
        "tool_args": getattr(tool_exec, "tool_args", None),
        "tool_call_error": getattr(tool_exec, "tool_call_error", None),
        "result": getattr(tool_exec, "result", None),
        "execution_host": getattr(tool_exec, "execution_host", None),
    }


def _ensure_execution_host(tool_exec: Any) -> dict[str, Any]:
    """Stamp and return the trusted host for persistence and live encoding."""

    existing = getattr(tool_exec, "execution_host", None)
    if isinstance(existing, dict):
        return existing
    host = _default_execution_host(getattr(tool_exec, "tool_args", None))
    try:
        tool_exec.execution_host = host
    except Exception:  # noqa: BLE001 — duck-typed immutable fixture
        pass
    return host


def _default_execution_host(args: Any) -> dict[str, Any]:
    """Infer only the safe legacy default when a persisted row predates hosts."""

    server = args.get("server") if isinstance(args, dict) else None
    if isinstance(server, str) and server.startswith("client:"):
        try:
            from src.core.execution_origin import current_execution_origin

            origin = current_execution_origin()
        except Exception:  # noqa: BLE001
            origin = None
        return (
            origin.execution_host
            if origin is not None
            else {"kind": "client", "device_label": "Unavailable client"}
        )
    return {"kind": "server", "device_label": "Server OpenAgent"}


__all__ = ["emit_tool_status", "tool_exec_to_wire_json", "stored_tool_to_wire"]
