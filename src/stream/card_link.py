"""Make a spawned child's card clickable WHILE it runs.

A delegation / run-now tool call renders in the parent chat as a navigable
card (``DelegationCard`` → the sub-agent's session, ``RunLaunchCard`` → the
run screen). The link target is the spawned ``child_session_id``, which is
minted the instant the child starts — but on the MCP delegation / scheduler /
workflow paths it only reaches the client in the tool's FINAL result, so the
card sits inert ("running…") for the whole child run and becomes clickable
only once it finishes.

The team runner already solves this for ``delegate_task_to_member`` by
re-streaming the in-flight tool's status frame carrying ``child_session_id``
once it's minted (``_emit_member_card_link``). This module is the same trick
for the IN-PROCESS spawn paths — MCP ``delegate_task`` / ``run_dream_mode``:

  * the provider stamps the IN-FLIGHT tool's ``(call_id, name)`` into a
    contextvar around each tool execution (``set_active_tool_call``). Because
    the deferred-tool dispatcher (``tool_search_call_tool``) invokes the inner
    handler in the SAME asyncio task, a dispatcher-wrapped delegation reads the
    OUTER (client-visible) tool's id — exactly the chip to flip clickable.
  * ``run_child_session`` calls ``emit_card_link(child_sid)`` right after
    minting; it re-streams a ``status`` frame for that in-flight tool carrying
    the child id, which the app merges onto the existing chip by
    ``tool_call_id``. The frame stays in the ``running`` phase (no result), so
    the eventual completion frame still matches and replaces it.

SCOPE — this fires only when ``run_child_session`` executes in the SAME
asyncio task that the provider stamped (an in-process delegation inside the
chat turn). A scheduled/workflow run-now does NOT qualify: its
``run_scheduled_task_now`` / ``run_workflow`` tool only drops a request row
(in an MCP subprocess), and the real run executes later on the Scheduler's own
background task — where neither contextvar is set, so ``emit_card_link``
no-ops. Those run-now cards are flipped clickable CLIENT-SIDE instead: the run
broadcasts its live frames to every client, which links the child session to
the matching in-flight run-now chip by parent id (see the app's
``linkRunNowChip``).

Best-effort throughout: a no-op when no tool is in flight (a detached run, an
autonomous cron fire) or no streaming client drives the turn. Never raises —
surfacing a link must not break the spawn it mirrors.
"""

from __future__ import annotations

import contextvars
import json
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# (tool_call_id, tool_name) of the tool the provider is currently executing.
# Read deep in the stack (inside an MCP handler, even one reached through the
# tool-search dispatcher) without threading it through every signature.
_active_tool: contextvars.ContextVar[Optional[Tuple[str, str]]] = contextvars.ContextVar(
    "openagent_active_tool_call", default=None,
)


def set_active_tool_call(tool_call_id: str, tool_name: str):
    """Bind the in-flight tool's id/name for the duration of its execution.
    Returns a reset token (pass to :func:`reset_active_tool_call`)."""
    return _active_tool.set((tool_call_id, tool_name))


def reset_active_tool_call(token) -> None:
    _active_tool.reset(token)


def current_active_tool_call() -> Optional[Tuple[str, str]]:
    """The ``(tool_call_id, tool_name)`` of the tool currently executing, or
    None when nothing is in flight."""
    return _active_tool.get()


async def emit_card_link(child_session_id: str) -> None:
    """Re-stream the in-flight tool's status frame carrying ``child_session_id``
    so the parent's delegation / run-launch card flips clickable mid-run.

    No-op (silent) when no tool is in flight, no chat session owns it, or no
    streaming client is bound. Never raises.
    """
    try:
        if not child_session_id:
            return
        active = current_active_tool_call()
        if active is None:
            return
        tool_call_id, tool_name = active
        if not tool_call_id:
            return
        # The chat session whose stream shows the in-flight tool (NOT
        # run_child_session's synthetic scheduler/workflow parent). The main
        # agent that triggers these spawns owns the delegation MCP, so its
        # context carries the chat session id here.
        from src.mcp.servers.delegation.handlers import current_parent_session_id
        sid = current_parent_session_id()
        if not sid:
            return
        payload = json.dumps({
            "tool_name": tool_name,
            "tool_call_id": tool_call_id,
            "child_session_id": child_session_id,
        })
        # A running status frame (no ``result``) — the app merges it onto the
        # existing chip by tool_call_id, flipping the card clickable while the
        # phase stays "running"; the later completion frame still matches.
        from src.stream.child_stream import emit_child_frame
        await emit_child_frame(sid, "status", text=payload)
    except Exception as e:  # noqa: BLE001 — card linkage is best-effort
        logger.debug("card_link: emit for %s failed: %s", child_session_id, e)
