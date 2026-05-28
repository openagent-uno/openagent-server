"""Delegation MCP — let any agent invoke another registered model.

Surfaces a single tool, ``delegate_task``, that takes a target model's
runtime id and a task and runs that model to completion, returning the
final assistant text.

This is the primitive that lets subscription-CLI agents (Claude SDK,
Codex SDK) act as team leaders. They cannot make an LLM tool-call to
``delegate_task_to_member`` the way the inlined ``TeamRunner`` expects,
but they CAN call an MCP tool from inside their SDK loop. This server
turns that MCP call into a real sub-agent run.

Vision §4 alignment: sub-agents share the parent's context — the same
session id, the same MCP pool, the same files attached to the turn.
The implementation defers as much as possible to the existing
``NativeProvider`` / ``ClaudeBackedAgent`` / ``CodexBackedAgent``
machinery so a delegated run looks identical to a user-initiated one.

The handler runs in-process (same Python interpreter as the agent that
called it) so it has direct read access to the model catalog and the
live ``MCPPool``. There is no subprocess and no IPC.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any, Optional

from src.core.logging import elog

logger = logging.getLogger(__name__)


# Set by the agent runtime before invoking the SDK loop so the in-process
# delegation handler can resolve "the current session" / "the current
# MCP pool" / "the model catalog" without those being passed as arguments
# the SDK would have to know about.
_session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "openagent_delegation_session_id", default=None,
)
_pool_var: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "openagent_delegation_pool", default=None,
)
_db_var: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "openagent_delegation_db", default=None,
)
_dispatcher_var: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "openagent_delegation_dispatcher", default=None,
)


def install_context(*, session_id, pool, db, dispatcher):
    """Bind the per-run context that delegation handlers will read.

    Returns four reset tokens; pass them back to :func:`reset_context`
    in the same order. The agent runtime wraps each turn in this binding
    so the in-process MCP handler sees the right session.
    """
    return (
        _session_id_var.set(session_id),
        _pool_var.set(pool),
        _db_var.set(db),
        _dispatcher_var.set(dispatcher),
    )


def reset_context(tokens) -> None:
    sid_tok, pool_tok, db_tok, disp_tok = tokens
    _session_id_var.reset(sid_tok)
    _pool_var.reset(pool_tok)
    _db_var.reset(db_tok)
    _dispatcher_var.reset(disp_tok)


async def delegate_task(model_id: str, task: str) -> dict[str, Any]:
    """Run ``task`` on the model named by ``model_id`` and return the answer.

    Args:
        model_id: runtime id of a model registered in the catalog, e.g.
            ``"anthropic:claude-3-5-sonnet-20240620"``,
            ``"openai:gpt-4o"``, ``"claude-cli:default"``.
        task: the prompt to send the delegated model.

    Returns:
        A dict with ``status`` (``"ok"`` / ``"error"``), the delegated
        ``model_id``, and either ``answer`` (final text) or ``error``
        (failure message).
    """
    parent_session_id = _session_id_var.get()
    pool = _pool_var.get()
    db = _db_var.get()
    dispatcher = _dispatcher_var.get()

    if dispatcher is None or db is None:
        return {
            "status": "error",
            "model_id": model_id,
            "error": (
                "delegate_task called outside an agent turn. "
                "This tool can only be invoked while a parent agent is "
                "running — the runtime didn't install a delegation context."
            ),
        }

    elog(
        "subagent.start",
        session_id=parent_session_id,
        delegated_to=model_id,
        task_preview=(task[:200] + "…") if len(task) > 200 else task,
    )

    try:
        result = await dispatcher.run_delegated(
            model_id=model_id,
            task=task,
            parent_session_id=parent_session_id,
            pool=pool,
            db=db,
        )
    except Exception as e:  # noqa: BLE001
        elog(
            "subagent.error",
            session_id=parent_session_id,
            delegated_to=model_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return {
            "status": "error",
            "model_id": model_id,
            "error": f"{type(e).__name__}: {e}",
        }

    elog(
        "subagent.complete",
        session_id=parent_session_id,
        delegated_to=model_id,
        chars=len(result or ""),
    )
    return {"status": "ok", "model_id": model_id, "answer": result}


async def list_delegatable_models() -> dict[str, Any]:
    """Return the runtime ids of every model available for delegation.

    The caller (the leader agent) needs the names before it can pick
    one. Reads the catalog through the shared ``MemoryDB`` instance.
    """
    db = _db_var.get()
    if db is None:
        return {"status": "error", "models": [], "error": "no db context"}
    try:
        rows = await db.list_enabled_models()
    except AttributeError:
        # MemoryDB API surface name may differ; fall back to a raw query.
        rows = await db.fetch_all(
            "SELECT runtime_id, display_name, framework FROM models WHERE enabled = 1"
        )
    return {
        "status": "ok",
        "models": [
            {
                "runtime_id": r["runtime_id"] if isinstance(r, dict) else r[0],
                "display_name": (
                    r["display_name"] if isinstance(r, dict) else r[1]
                ),
                "framework": (
                    r["framework"] if isinstance(r, dict) else r[2]
                ),
            }
            for r in (rows or [])
        ],
    }
