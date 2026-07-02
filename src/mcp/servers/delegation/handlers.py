"""Delegation MCP — let any agent invoke another registered model.

Surfaces a single tool, ``delegate_task``, that takes a target model's
runtime id and a task and runs that model to completion, returning the
final assistant text.

This is the primitive that lets any agent delegate to another registered
model — an alternative to the inlined ``TeamRunner``'s
``delegate_task_to_member`` tool-call that works from any agent loop.
This server turns the MCP call into a real sub-agent run.

Vision §4 alignment: sub-agents share the parent's context — the same
session id, the same MCP pool, the same files attached to the turn.
The implementation defers as much as possible to the existing
``NativeProvider`` machinery so a delegated run looks identical to a
user-initiated one.

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
# The live Agent driving the parent turn. ``delegate_task`` runs the
# sub-agent through ``core.child_session.run_child_session`` (which needs the
# Agent's full system prompt + tool set + ``run``), so the parent binds
# itself here for the duration of the turn.
_agent_var: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar(
    "openagent_delegation_agent", default=None,
)
# The owner handle driving the parent turn (the authenticated human). Threaded
# so a delegation's child session is stamped with the RIGHT owner even on a
# session's very first turn — before the gateway's fire-and-forget owner stamp
# on the parent row has committed (the race that hid sub-agent rows from the
# sidebar). None for agent-self / automation runs; child_session falls back to
# parent-row inheritance then the deployment's primary owner.
_owner_handle_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "openagent_delegation_owner_handle", default=None,
)


def install_context(*, session_id, pool, db, dispatcher, agent=None, owner_handle=None):
    """Bind the per-run context that delegation handlers will read.

    Returns reset tokens; pass them back to :func:`reset_context`
    in the same order. The agent runtime wraps each turn in this binding
    so the in-process MCP handler sees the right session and Agent.
    """
    return (
        _session_id_var.set(session_id),
        _pool_var.set(pool),
        _db_var.set(db),
        _dispatcher_var.set(dispatcher),
        _agent_var.set(agent),
        _owner_handle_var.set(owner_handle),
    )


def reset_context(tokens) -> None:
    sid_tok, pool_tok, db_tok, disp_tok, agent_tok, owner_tok = tokens
    _session_id_var.reset(sid_tok)
    _pool_var.reset(pool_tok)
    _db_var.reset(db_tok)
    _dispatcher_var.reset(disp_tok)
    _agent_var.reset(agent_tok)
    _owner_handle_var.reset(owner_tok)


def current_parent_session_id() -> Optional[str]:
    """The chat session driving the current turn (the session whose stream a
    spawned child's card lives on). Set for the whole turn by the main agent's
    ``install_context``; ``None`` in a headless / autonomous run with no chat
    turn. Used by ``src.stream.card_link`` to target the in-flight card."""
    return _session_id_var.get()


async def delegate_task(model_id: str, task: str) -> dict[str, Any]:
    """Run ``task`` on the model named by ``model_id`` and return the answer.

    Args:
        model_id: runtime id of a model registered in the catalog, e.g.
            ``"anthropic:claude-3-5-sonnet-20240620"``, ``"openai:gpt-4o"``.
        task: the prompt to send the delegated model.

    Returns:
        A dict with ``status`` (``"ok"`` / ``"error"``), the delegated
        ``model_id``, and either ``answer`` (final text) or ``error``
        (failure message).
    """
    parent_session_id = _session_id_var.get()
    db = _db_var.get()
    agent = _agent_var.get()
    owner_handle = _owner_handle_var.get()

    if agent is None or db is None:
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

    # Run the sub-agent as a FULL child session: its own durable row, the
    # same two-layer framework+persona system prompt and full tool set as
    # the parent (vision §15 — the old ``run_delegated`` path passed
    # ``system=None`` and persisted nothing), linked back to the parent and
    # surfaced as a clickable card in the leader's transcript.
    from src.core.child_session import run_child_session

    try:
        result = await run_child_session(
            agent=agent,
            db=db,
            parent_session_id=parent_session_id,
            origin="delegation",
            origin_ref={"model_id": model_id},
            title=f"Sub-agent · {model_id}",
            prompt=task,
            owner_client_id=owner_handle,
            model_id=model_id,
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
        child_session_id=result.session_id,
        chars=len(result.text or ""),
    )
    # ``child_session_id`` lets the app render the delegation tool call as a
    # card that deep-links into the sub-agent's full session (OpenCode-style).
    return {
        "status": "ok",
        "model_id": model_id,
        "answer": result.text,
        "child_session_id": result.session_id,
    }


async def run_dream_mode() -> dict[str, Any]:
    """Run a full DREAM-MODE memory-maintenance pass as a SEPARATE session.

    Unlike ``vault_dream`` (which runs the maintenance INLINE in the current
    chat), this spawns the real dream-mode routine as its own durable child
    session — it appears in the sidebar as a scheduled run and as a clickable
    card in this chat, exactly like the nightly automated firing. The agent's
    full reasoning lives in that separate session, keeping this conversation
    clean. Use this whenever the user asks you to "run dream mode".

    Returns ``status`` plus the spawned ``child_session_id`` (the card target).
    """
    import time
    import uuid as _uuid

    parent_session_id = _session_id_var.get()
    db = _db_var.get()
    agent = _agent_var.get()
    owner_handle = _owner_handle_var.get()

    if agent is None or db is None:
        return {
            "status": "error",
            "error": (
                "run_dream_mode called outside an agent turn — the runtime "
                "didn't install a delegation context."
            ),
        }

    from src.core.builtin_tasks import DREAM_MODE_TASK_NAME
    from src.core.child_session import mint_child_session_id, run_child_session
    from src.core.identity_context import agent_author
    from src.core.server import DREAM_MODE_PROMPT
    from src.stream.resource_events import emit_resource_event

    run_id = _uuid.uuid4().hex[:8]
    child_sid = mint_child_session_id(
        "scheduler", {"task_id": DREAM_MODE_TASK_NAME, "run_id": run_id},
    )

    # Best-effort: record a task_runs row so the firing also surfaces in the
    # task run-history screen. The FK needs the dream-mode scheduled_tasks row
    # to exist (it's a framework builtin, present when dream mode is enabled);
    # when absent we just skip the row — the child session alone already gives
    # the in-chat card and the sidebar entry.
    task_run_id: str | None = None
    dream_task_id: str | None = None
    try:
        tasks = await db.get_tasks()
        dream_task = next(
            (t for t in tasks if t.get("name") == DREAM_MODE_TASK_NAME), None,
        )
        if dream_task is None:
            # Self-heal: dream mode is "toggleable but not removable" (vision
            # §12), but the row is only seeded when the feature has been enabled
            # at least once. Create it DISABLED so a manual firing always has a
            # ``scheduled_tasks`` row to record its ``task_runs`` history on —
            # otherwise the run never surfaces in the app's "Recent" feed.
            # Disabled means the scheduler never auto-fires it.
            try:
                new_id = await db.add_task(
                    DREAM_MODE_TASK_NAME, "0 3 * * *", DREAM_MODE_PROMPT,
                )
                await db.update_task(new_id, enabled=0, next_run=None)
                dream_task = await db.get_task(new_id)
            except Exception as e:  # noqa: BLE001
                logger.debug("run_dream_mode: dream-mode row self-heal failed: %s", e)
        if dream_task is not None:
            dream_task_id = dream_task["id"]
            task_run_id = await db.add_task_run(
                task_id=dream_task_id, trigger="manual",
                run_id=run_id, session_id=child_sid,
            )
            # Flip the "Recent" feed live the moment the firing opens — the
            # same ``scheduled_task`` resource event the cron path broadcasts
            # from ``Scheduler.run_task``, so a manual run shows up in the
            # sidebar exactly like a nightly one (not only on the next reload).
            emit_resource_event("scheduled_task", "updated", dream_task_id)
    except Exception as e:  # noqa: BLE001
        logger.debug("run_dream_mode: task_run record skipped: %s", e)

    elog("dream.manual_start", session_id=parent_session_id,
         child_session_id=child_sid)

    try:
        result = await run_child_session(
            agent=agent,
            db=db,
            parent_session_id=parent_session_id,
            origin="scheduler",
            origin_ref={"task_id": DREAM_MODE_TASK_NAME, "run_id": run_id},
            title="Dream mode",
            prompt=DREAM_MODE_PROMPT,
            owner_client_id=owner_handle,
            author=agent_author(
                "Dream mode", agent_name=getattr(agent, "name", None),
            ),
            # Stream the firing live so its card / run screen fills in
            # token-by-token, exactly like the nightly cron firing
            # (``scheduler.run_task`` also passes ``stream=True``). Without
            # this the child ran through the non-streaming ``agent.run``
            # branch and its transcript only appeared once fully finished.
            stream=True,
        )
    except Exception as e:  # noqa: BLE001
        if task_run_id is not None:
            try:
                await db.update_task_run(
                    task_run_id, status="failed",
                    finished_at=time.time(), error=f"{type(e).__name__}: {e}",
                )
            except Exception:  # noqa: BLE001
                pass
            # Flip the feed's run status off "running".
            emit_resource_event("scheduled_task", "updated", dream_task_id)
        elog("dream.manual_error", level="error",
             session_id=parent_session_id, error=str(e))
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}

    if task_run_id is not None:
        try:
            await db.update_task_run(
                task_run_id, status="success",
                finished_at=time.time(), output=(result.text or "")[:2000],
            )
        except Exception:  # noqa: BLE001
            pass
        # Re-broadcast so the feed picks up the terminal status.
        emit_resource_event("scheduled_task", "updated", dream_task_id)

    elog("dream.manual_complete", session_id=parent_session_id,
         child_session_id=result.session_id, chars=len(result.text or ""))
    return {
        "status": "ok",
        "child_session_id": result.session_id,
        "answer": (
            "Dream mode is running as its own session — open the card above to "
            "follow along as it works through the vault."
        ),
    }


async def list_delegatable_models() -> dict[str, Any]:
    """Return the runtime ids of every model available for delegation.

    The caller (the leader agent) needs the names before it can pick
    one. Reads the catalog through the shared ``MemoryDB`` instance.
    """
    db = _db_var.get()
    if db is None:
        return {"status": "error", "models": [], "error": "no db context"}
    rows = await db.list_models_enriched(enabled_only=True, kind="llm")
    return {
        "status": "ok",
        "models": [
            {
                "runtime_id": r["runtime_id"],
                "display_name": r["display_name"],
                "framework": r["framework"],
            }
            for r in rows
        ],
    }
