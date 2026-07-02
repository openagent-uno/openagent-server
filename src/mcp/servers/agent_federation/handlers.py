"""Process-level runtime context for the in-process agent-federation builtin.

Holds the ONE running ``IrohNode`` + the agent DB, bound once at gateway
wire-up via ``set_agent_runtime``. The tools read them LAZILY at call time
(so build-time ordering vs the gateway doesn't matter). Fail loud if unset —
never a silent no-op.
"""

from __future__ import annotations

from typing import Any

_iroh_node: Any = None
_db: Any = None


def set_agent_runtime(iroh_node: Any, db: Any) -> None:
    """Bind the gateway's IrohNode + agent DB (called once at wire-up)."""
    global _iroh_node, _db
    _iroh_node = iroh_node
    _db = db


def get_iroh_node() -> Any:
    if _iroh_node is None:
        raise RuntimeError(
            "agent-federation: iroh runtime not attached "
            "(set_agent_runtime was not called at gateway wire-up)"
        )
    return _iroh_node


def get_db() -> Any:
    if _db is None:
        raise RuntimeError(
            "agent-federation: db not attached "
            "(set_agent_runtime was not called at gateway wire-up)"
        )
    return _db
