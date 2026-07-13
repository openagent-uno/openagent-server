"""The one primitive every spawned (non-interactive) AI run funnels through.

A delegated sub-agent, a scheduled-task firing, and a workflow AI-prompt
node are all the same thing: a *full child session* — a real ``sessions``
row that runs through the ordinary ``Agent.run`` path (so it gets the
two-layer framework+persona system prompt, the full tool set, and durable
runs-JSON persistence; vision §15) whose first message is the task/mission
prompt the agent gave itself.

``run_child_session`` does only the things ``Agent.run`` does not:

  1. mint a UNIQUE durable child ``session_id`` (so hundreds of siblings
     never collide, and a re-fired schedule / re-run workflow never inherits
     a prior transcript — the root cause of the issue-#5 compaction bug);
  2. resolve the owner handle and pre-stamp the row's link metadata
     (``parent_session_id`` / ``origin`` / ``kind`` / ``client_id``) via
     ``MemoryDB.upsert_session`` so the app's flat session list surfaces it
     with an origin chip and a navigable parent breadcrumb. This is pure
     metadata — it NEVER writes ``user_id`` (the runtime-owned sentinel);
  3. stamp the agent-self author on the seed message (so the app can render
     it as a Mission/Role/Task block rather than a human "You" bubble);
  4. bound concurrency so a fan-out of hundreds doesn't exhaust the single
     SQLite writer / provider budget;
  5. on completion ``release_session`` (free live runtime resources, KEEP
     the durable row) instead of ``forget_session`` (which wipes it).

The pre-stamped metadata survives the runtime's runs write because
``read_or_create_session`` loads the existing row (matched by
``user_id IS NULL``) and writes its metadata back unchanged — the same
mechanism by which a gateway-stamped chat title survives today.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from src.core.identity_context import agent_author

logger = logging.getLogger(__name__)


# Child origins that are NOT surfaced as standalone rows in the flat session
# list / sidebar. Each is navigable only from where it was spawned — a
# delegation from its parent transcript's card, a scheduled firing / workflow
# node from its run's execution screen, an event-prompt session from its
# delivery in the Events feed — so listing it separately would just duplicate
# a row the user already reaches in context. Shared by the gateway's
# ``GET /api/sessions`` filter and its child-session-created broadcast skip so
# the two never drift. ``chat`` is the only origin that lists normally.
HIDDEN_CHILD_ORIGINS: tuple[str, ...] = ("delegation", "scheduler", "workflow", "event")


# Total concurrent child runs across the whole agent. The real backpressure
# valve: it bounds how many ``Agent.run`` loops (each holding an MCP runtime
# + writing runs JSON to the one SQLite file) execute at once, regardless of
# how many a single turn fans out. Env-overridable for big hosts.
_GLOBAL_CONCURRENCY = max(1, int(os.environ.get("OPENAGENT_CHILD_SESSION_CONCURRENCY", "16")))
# Per-parent fan-out cap: how many children one parent turn may run in
# parallel. Keeps a single runaway leader from starving every other session
# of the global pool. 0 disables the per-parent cap.
_PER_PARENT_CONCURRENCY = max(0, int(os.environ.get("OPENAGENT_CHILD_SESSION_PER_PARENT", "8")))


# asyncio primitives bind to the loop they are first used on; tests spin a
# fresh loop per ``asyncio.run``. Cache the semaphores per running loop so a
# semaphore created on a since-closed loop is never reused.
_global_sems: dict[int, asyncio.Semaphore] = {}
_parent_sems: dict[int, dict[str, asyncio.Semaphore]] = {}


# Optional best-effort "a child session was just created" hook. The gateway
# registers one (``broadcast_resource_sync("session", "created", sid)``) so a
# freshly-spawned child appears in connected clients' session lists live,
# without ``core`` having to import the gateway. Stays None in tests / headless
# runs. Signature: ``listener(session_id: str, info: dict) -> None``.
_listener: Optional[Any] = None


def set_child_session_listener(cb: Optional[Any]) -> None:
    """Register (or clear with None) the child-session-created listener."""
    global _listener
    _listener = cb


def _notify_created(session_id: str, info: dict[str, Any]) -> None:
    cb = _listener
    if cb is None:
        return
    try:
        cb(session_id, info)
    except Exception as e:  # noqa: BLE001
        logger.debug("child_session: created-listener failed for %s: %s", session_id, e)


def _global_semaphore() -> asyncio.Semaphore:
    loop_id = id(asyncio.get_running_loop())
    sem = _global_sems.get(loop_id)
    if sem is None:
        sem = asyncio.Semaphore(_GLOBAL_CONCURRENCY)
        _global_sems[loop_id] = sem
    return sem


def _parent_semaphore(parent_session_id: Optional[str]) -> Optional[asyncio.Semaphore]:
    if not parent_session_id or _PER_PARENT_CONCURRENCY <= 0:
        return None
    loop_id = id(asyncio.get_running_loop())
    per_loop = _parent_sems.setdefault(loop_id, {})
    sem = per_loop.get(parent_session_id)
    if sem is None:
        sem = asyncio.Semaphore(_PER_PARENT_CONCURRENCY)
        per_loop[parent_session_id] = sem
    return sem


@dataclass
class ChildSessionResult:
    """What a spawned child run returns to its caller.

    ``session_id`` is the durable row's id (the delegation card / run screen
    deep-links to it); ``text`` is the final assistant text (what a leader
    agent synthesises over, or a workflow node emits downstream)."""

    session_id: str
    text: str


def mint_child_session_id(origin: str, origin_ref: dict[str, Any]) -> str:
    """Build the durable child session_id for one spawn.

    Always unique per spawn so siblings never collide and a re-fired
    schedule / re-run workflow / repeated delegation never resumes a prior
    transcript. The id is also human-legible so logs and the sidebar reveal
    lineage at a glance.
    """
    if origin == "delegation":
        parent = origin_ref.get("parent_session_id") or "root"
        model_id = origin_ref.get("model_id") or "model"
        return f"{parent}::sub::{model_id}::{uuid.uuid4().hex[:8]}"
    if origin == "scheduler":
        task_id = origin_ref.get("task_id") or "task"
        run_id = origin_ref.get("run_id") or uuid.uuid4().hex[:8]
        return f"scheduler:{task_id}:{run_id}"
    if origin == "workflow":
        wf = origin_ref.get("workflow_id") or "wf"
        run = origin_ref.get("run_id") or "run"
        node = origin_ref.get("node_id")
        return f"workflow:{wf}:{run}:{node}" if node else f"workflow:{wf}:{run}"
    if origin == "event":
        ev = origin_ref.get("event_id") or "event"
        delivery = origin_ref.get("delivery_id") or uuid.uuid4().hex[:8]
        return f"event:{ev}:{delivery}"
    # Unknown origin — still produce a unique, namespaced id.
    return f"{origin}:{uuid.uuid4().hex}"


def _kind_label(origin: str, origin_ref: dict[str, Any], model_id: Optional[str]) -> Optional[str]:
    """The fine-grained ``kind`` tag stored in metadata (the delegated model
    id, the task id, ``workflow:node``) — what the app shows on the origin
    chip beyond the broad ``origin``."""
    if origin == "delegation":
        return model_id or origin_ref.get("model_id")
    if origin == "scheduler":
        return origin_ref.get("task_id")
    if origin == "workflow":
        wf = origin_ref.get("workflow_id")
        node = origin_ref.get("node_id")
        return f"{wf}:{node}" if (wf and node) else wf
    if origin == "event":
        return origin_ref.get("event_id")
    return None


async def run_child_session(
    *,
    agent: Any,
    db: Any,
    parent_session_id: Optional[str],
    origin: str,
    origin_ref: Optional[dict[str, Any]] = None,
    title: str,
    prompt: str,
    owner_client_id: Optional[str] = None,
    model_id: Optional[str] = None,
    author: Optional[dict[str, Any]] = None,
    on_status: Any = None,
    stream: bool = False,
    session_id: Optional[str] = None,
) -> ChildSessionResult:
    """Spawn ``prompt`` as a full durable child session and return its result.

    Args:
        agent: the live ``Agent`` (provides ``run`` + the system prompt + tools).
        db: the ``MemoryDB`` used to resolve the owner and stamp link metadata.
        parent_session_id: the spawning session (a chat session for delegation;
            a synthetic per-task / per-run root for scheduler / workflow). May
            be None.
        origin: ``delegation`` | ``scheduler`` | ``workflow``.
        origin_ref: identifiers used to mint the id and the ``kind`` label
            (e.g. ``{"model_id": ...}``, ``{"task_id", "run_id"}``,
            ``{"workflow_id", "run_id", "node_id"}``).
        title: human-facing label (sidebar title / card title).
        prompt: the task/mission/role prompt — becomes the seed first message.
        owner_client_id: owner handle to stamp so the row appears in that
            user's flat list. When omitted, inherited from the parent row.
        model_id: optional runtime id to run this child on (built into a
            model_override). Omit to use the agent's default/router model.
        author: optional author override for the seed message; defaults to an
            agent-self author labelled with ``title``.
        on_status: optional status callback forwarded to ``Agent.run``.
        session_id: optional durable child session id to reuse. Omit for the
            normal one-spawn/one-session path; pass an existing OpenAgent
            session id when a caller intentionally injects another turn into a
            durable child transcript (event payload binding).
        stream: when True, drive ``Agent.run_stream`` and forward each content
            delta / tool status as a child-tagged frame (``emit_child_frame``),
            so a DETACHED run (a scheduled firing, a workflow node) streams live
            into its run screen exactly like an interactive session. When no
            turn-scoped emitter is bound, the gateway's broadcast sink is
            installed for the duration so the frames reach connected clients.
            Defaults False (a single ``Agent.run``) — delegation keeps its
            existing behaviour (it streams via the team runner's own emitter).
    """
    origin_ref = origin_ref or {}
    if parent_session_id and "parent_session_id" not in origin_ref:
        origin_ref = {**origin_ref, "parent_session_id": parent_session_id}

    child_sid = session_id or mint_child_session_id(origin, origin_ref)

    # Resolve the owner handle so the row lands in the right flat list. An
    # explicit owner wins (scheduler / workflow have no human parent); else
    # inherit the parent chat session's owner.
    owner = owner_client_id
    if not owner and parent_session_id and db is not None:
        try:
            parent_row = await db.get_session(parent_session_id)
            if parent_row:
                owner = parent_row.get("client_id") or None
        except Exception as e:  # noqa: BLE001
            logger.debug("child_session: parent owner lookup failed: %s", e)
    # Last resort: the deployment's primary owner. Covers the race where a
    # delegation fires on a session's first turn before the gateway's
    # fire-and-forget owner stamp has committed (so the parent row has no
    # client_id yet) — without this the child, AND its unowned parent, both
    # fall out of every flat list. Single-owner deploys (the common case)
    # always resolve here; multi-user inherits the parent above first.
    if not owner and db is not None:
        try:
            owner = await db.primary_owner_handle()
        except Exception as e:  # noqa: BLE001
            logger.debug("child_session: primary-owner fallback failed: %s", e)

    # Pre-stamp link metadata so the row exists (and can be broadcast) before
    # the run populates it. Metadata-only — never writes user_id.
    if db is not None:
        try:
            await db.upsert_session(
                child_sid,
                client_id=owner,
                title=(title or "")[:200] or None,
                model=model_id,
                parent_session_id=parent_session_id,
                origin=origin,
                kind=_kind_label(origin, origin_ref, model_id),
            )
            # Announce the new child so connected clients add it to the
            # session list live (best-effort; no-op when no gateway listener).
            _notify_created(child_sid, {
                "owner": owner, "origin": origin,
                "parent_session_id": parent_session_id, "title": title,
            })
        except Exception as e:  # noqa: BLE001
            logger.warning("child_session: metadata pre-stamp failed for %s: %s", child_sid, e)

    # Flip the parent's delegation card clickable NOW (the child id is minted)
    # instead of only when this tool finally returns — re-streams the in-flight
    # chip carrying child_sid. This fires for an IN-PROCESS delegation (same
    # asyncio task as the chat turn's tool); it no-ops for a scheduler/workflow
    # run-now (executed later on the Scheduler's own task, with no in-flight
    # tool bound) — those cards are linked client-side from the broadcast.
    # See ``src.stream.card_link``.
    try:
        from src.stream.card_link import emit_card_link
        await emit_card_link(child_sid)
    except Exception as e:  # noqa: BLE001 — card linkage is best-effort
        logger.debug("child_session: card-link emit failed for %s: %s", child_sid, e)

    # Build a model override when a specific model was requested (mirrors the
    # workflow ai-prompt path). Falls back to the agent's default/router model.
    override = None
    if model_id:
        smart = getattr(agent, "model", None)
        if smart is not None and hasattr(smart, "build_override_model"):
            try:
                override = smart.build_override_model(model_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("child_session: model_override %r failed: %s", model_id, e)

    if author is None:
        author = agent_author(title or origin, agent_name=getattr(agent, "name", None))

    parent_sem = _parent_semaphore(parent_session_id)
    global_sem = _global_semaphore()

    async def _run_streamed() -> str:
        """Drive ``run_stream`` and forward each delta / status as a child frame
        so the run screen renders token-by-token, just like a chat turn."""
        from src.stream.child_stream import (
            emit_child_frame, current_child_stream_emitter,
            install_child_stream_emitter, reset_child_stream_emitter,
            broadcast_child_emitter,
        )
        # Detached runs (scheduler / workflow) have no turn-scoped emitter — bind
        # the gateway broadcast sink so their frames still reach clients. When an
        # emitter IS already bound (a delegation inside a chat turn), leave it.
        emit_tok = None
        if current_child_stream_emitter() is None:
            be = broadcast_child_emitter()
            if be is not None:
                emit_tok = install_child_stream_emitter(be)

        async def _status(msg: str) -> None:
            await emit_child_frame(child_sid, "status", text=msg)
            if on_status is not None:
                try:
                    await on_status(msg)
                except Exception:  # noqa: BLE001
                    pass

        # Echo the agent-self mission/seed at the TOP of the live transcript,
        # before any delta — so a detached run's screen shows the Mission block
        # *while it runs*, not only once the canonical transcript persists at
        # completion (mirrors a chat optimistically showing the user's message).
        await emit_child_frame(child_sid, "seed", text=prompt, author=author)

        parts: list[str] = []
        try:
            async for event in agent.run_stream(
                message=prompt,
                user_id=origin,
                session_id=child_sid,
                model_override=override,
                author=author,
                on_status=_status,
            ):
                kind = event.get("kind")
                if kind == "delta":
                    piece = event.get("text") or ""
                    if piece:
                        parts.append(piece)
                        await emit_child_frame(child_sid, "delta", text=piece)
                elif kind == "done":
                    tail = event.get("text") or ""
                    if tail and not parts:
                        parts.append(tail)
                        await emit_child_frame(child_sid, "delta", text=tail)
                    break
        finally:
            # A ``response`` frame commits the streamed bubble (the app
            # marker-strips it and clears its delta buffer) BEFORE turn_complete
            # — matching a normal turn, which ends OutTextFinal → TurnComplete.
            # turn_complete then lets the app swap in the canonical persisted
            # transcript (reconcile).
            final = "".join(parts)
            if final:
                await emit_child_frame(child_sid, "response", text=final)
            await emit_child_frame(child_sid, "turn_complete")
            if emit_tok is not None:
                reset_child_stream_emitter(emit_tok)
        return "".join(parts)

    async def _run() -> str:
        try:
            # Streaming needs ``run_stream`` (every real Agent has it); a
            # minimal stub without it degrades to a single ``run`` — the
            # transcript still lands, just not token-by-token.
            if stream and callable(getattr(agent, "run_stream", None)):
                return await _run_streamed()
            return await agent.run(
                message=prompt,
                user_id=origin,
                session_id=child_sid,
                model_override=override,
                author=author,
                on_status=on_status,
            )
        finally:
            try:
                await agent.release_session(child_sid, model_override=override)
            except Exception as e:  # noqa: BLE001
                logger.debug("child_session: release_session(%s) failed: %s", child_sid, e)

    async with global_sem:
        if parent_sem is not None:
            async with parent_sem:
                text = await _run()
            _maybe_prune_parent_sem(parent_session_id, parent_sem)
        else:
            text = await _run()

    return ChildSessionResult(session_id=child_sid, text=text or "")


def _maybe_prune_parent_sem(parent_session_id: Optional[str], sem: asyncio.Semaphore) -> None:
    """Drop a per-parent semaphore once it's fully released, so the cache
    doesn't grow without bound over a long-lived server."""
    if not parent_session_id:
        return
    try:
        if sem._value >= _PER_PARENT_CONCURRENCY:  # type: ignore[attr-defined]
            per_loop = _parent_sems.get(id(asyncio.get_running_loop()))
            if per_loop is not None:
                per_loop.pop(parent_session_id, None)
    except Exception:  # noqa: BLE001
        pass
