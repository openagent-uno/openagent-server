"""Per-message authorship context.

Carries "who is speaking this turn" through an agent run without threading
it as an explicit argument the vendored runner would have to know about —
the same contextvar trick the delegation MCP uses (see
``src/mcp/servers/delegation/handlers.py``).

The runtime stamps a single stable ``user_id="openagent"`` sentinel on
every session row (the tenancy/ownership invariant — see
``project_openagent_session_user_id_invariant``), so it CANNOT carry the
real human identity. Authorship lives instead on each ``Message`` (the
``author`` field, persisted in the runs JSON), and that field is filled
from whatever is bound here for the duration of the turn:

  - a **human** author — a network handle (vision §11), stable across all
    of that user's devices, so multi-user and bridge (Telegram/Discord)
    sessions attribute each message to the right person;
  - an **agent** author — an agent-self seed message: the first message of
    a delegated sub-agent, a scheduled-task firing, or a workflow node,
    where the agent (not a human) authored the task/mission/role prompt.

Author shape (a plain JSON-able dict so it round-trips through
``Message.to_dict``):
    {"kind": "human"|"agent", "handle"?: str, "display"?: str,
     "device_id"?: str, "agent_name"?: str}
"""

from __future__ import annotations

import contextvars
from typing import Any, Optional

# Bound for the duration of one agent turn by ``install_author_context``.
_author_var: contextvars.ContextVar[Optional[dict[str, Any]]] = contextvars.ContextVar(
    "openagent_message_author", default=None,
)


def install_author_context(author: Optional[dict[str, Any]]):
    """Bind ``author`` as the current turn's message author.

    Returns a reset token; pass it to :func:`reset_author_context`. Safe to
    call with ``None`` (a no-op binding) so callers don't need to branch.
    """
    return _author_var.set(author)


def reset_author_context(token) -> None:
    _author_var.reset(token)


def current_author() -> Optional[dict[str, Any]]:
    """The author bound for the current turn, or ``None``."""
    return _author_var.get()


def human_author(
    handle: Optional[str],
    *,
    display: Optional[str] = None,
    device_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Build a human author dict, or ``None`` when there is no handle to
    attribute (so a handle-less / coordinator-less deployment simply falls
    back to today's generic rendering)."""
    if not handle:
        return None
    out: dict[str, Any] = {"kind": "human", "handle": handle}
    if display:
        out["display"] = display
    if device_id:
        out["device_id"] = device_id
    return out


def owner_handle_of(author: Optional[dict[str, Any]]) -> Optional[str]:
    """The owning human's handle for a turn, or ``None``.

    Only a **human** author carries an owner handle for any child session
    spawned during the turn; agent-self / automation authors return ``None``
    so the child inherits the owner from its parent row instead.
    """
    if isinstance(author, dict) and author.get("kind") == "human":
        return author.get("handle")
    return None


def agent_author(display: str, *, agent_name: Optional[str] = None) -> dict[str, Any]:
    """Build an agent-self author dict for a seed prompt the agent gave
    itself — a delegated task, a scheduled mission, a workflow node.

    ``display`` is the human-facing label (e.g. the delegated model id, the
    task name, "Scheduler", "Workflow"). ``agent_name`` is this agent's own
    name when known."""
    out: dict[str, Any] = {"kind": "agent", "display": display}
    if agent_name:
        out["agent_name"] = agent_name
    return out
