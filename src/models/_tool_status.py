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

    Shared by every runtime tool-status emitter (NativeProvider, dispatcher
    helpers, ClaudeCLI / CodexCLI) so the wire envelope and the
    defensive on_status error handling can't drift across call sites.
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
    payload = _to_dict(tool_exec)
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
    return dict(stored)


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
    }


__all__ = ["emit_tool_status", "tool_exec_to_wire_json", "stored_tool_to_wire"]
