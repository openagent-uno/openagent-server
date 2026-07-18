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
     SQLite writer / provider budget, and bound DEPTH so a chain of sub-agents
     delegating to sub-agents terminates instead of recursing until it has
     eaten the whole pool (see ``_MAX_DEPTH`` / ``_tier_capacity``). Because
     this is the single spawn primitive for every non-interactive origin, it is
     the one place both bounds can be enforced for all of them at once;
  5. on completion ``release_session`` (free live runtime resources, KEEP
     the durable row) instead of ``forget_session`` (which wipes it).

The pre-stamped metadata survives the runtime's runs write because
``read_or_create_session`` loads the existing row (matched by
``user_id IS NULL``) and writes its metadata back unchanged — the same
mechanism by which a gateway-stamped chat title survives today.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from src.core.identity_context import agent_author
from src.core.logging import elog
from src.core.tool_scope import (
    current_tool_allowlist,
    normalize_family,
    reset_tool_allowlist,
    set_tool_allowlist,
)

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

# How deep a delegation CHAIN may go. A chat turn's sub-agent is depth 1; a
# sub-agent that delegates further is depth 2; and so on. This does NOT ban
# depth — vision §4 explicitly blesses it ("a sub-agent that delegates further
# spawns child sessions of its own, and the lineage is explicit"). It bounds a
# RUNAWAY: before this existed there was no ceiling of any kind, so an agent
# that kept delegating to itself recursed until it had eaten every slot in the
# global pool and queued every other session on the server — one misbehaving
# cron could freeze the whole agent.
#
# Why 5: the production crons on this runtime (eSound, Lyra, Spicysparks) fan
# out cron → per-item sub-agent → that sub-agent's own helper, so depth 2-3 is
# NORMAL traffic, not an anomaly. 5 leaves two clear levels of headroom above
# the deepest legitimate workload while still cutting an unbounded self-
# delegating loop off long before it can saturate anything. A limit that trips
# on real work would be worse than no limit at all: it converts a silent risk
# into a live outage on someone's nightly cron.
_MAX_DEPTH = max(1, int(os.environ.get("OPENAGENT_CHILD_SESSION_MAX_DEPTH", "5")))

# Per-CHAIN fan-out cap: how many children one delegation TREE (identified by
# its root, not by its immediate parent) may run in parallel at one depth.
#
# Replaces a per-IMMEDIATE-PARENT cap that could not throttle a chain at all:
# keyed on the immediate parent, a chain A→B→C→D never accumulated against any
# single key — every hop minted a fresh semaphore with a fresh full allowance —
# so the per-parent cap bounded breadth but was blind to depth, and the only
# real backstop left was the global pool. Keyed on the chain ROOT, every
# descendant of one spawn accounts to one budget, so a runaway tree throttles
# itself instead of the server.
#
# Defaults to the legacy OPENAGENT_CHILD_SESSION_PER_PARENT when an operator
# has tuned that, so existing deployments keep their configured fan-out width.
# 0 disables the per-chain cap.
_PER_CHAIN_CONCURRENCY = max(0, int(os.environ.get(
    "OPENAGENT_CHILD_SESSION_PER_CHAIN",
    os.environ.get("OPENAGENT_CHILD_SESSION_PER_PARENT", "8"),
)))


class DelegationDepthExceeded(RuntimeError):
    """Raised when a spawn would exceed :data:`_MAX_DEPTH`.

    Carries the chain's ``root`` / ``depth`` so callers can log the lineage
    that ran away. ``delegate_task`` turns this into a clean, actionable tool
    error (rather than an opaque crash) so the model understands it hit a
    structural limit and should do the remaining work itself.
    """

    def __init__(self, message: str, *, root: Optional[str], depth: int) -> None:
        super().__init__(message)
        self.root = root
        self.depth = depth


# The delegation depth / chain root of the run currently executing on this
# context. A chat turn (nobody's child) leaves the defaults: depth 0, no root.
#
# Threaded EXPLICITLY rather than parsed back out of the session id. Ids do
# encode lineage (``{parent}::sub::{model}::{uuid8}``), but parsing them is not
# sound: ``mint_child_session_id`` is bypassed entirely whenever a caller
# passes an explicit ``session_id`` (the event payload-binding path does, and
# reuses a durable id with no lineage in it at all), a model_id is free-form
# text that can itself contain the separator, and scheduler/workflow/event ids
# carry no delegation lineage by construction. An id is a display artifact;
# depth is control state, so it travels as control state.
#
# ContextVars propagate into ``asyncio.create_task`` children (the context is
# COPIED at task-creation time), which is exactly the spawn path here: the
# provider fans tool calls out through ``asyncio.gather`` (each gathered
# coroutine gets its own copy), so a value set around one child's run is seen
# by every tool call that run makes — including its own nested spawns — while
# remaining invisible to its siblings and to its parent. This is the same
# mechanism ``src.core.dry_run`` relies on to reach every MCP call of a run.
_depth_var: contextvars.ContextVar[int] = contextvars.ContextVar(
    "openagent_child_depth", default=0,
)
_root_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "openagent_child_chain_root", default=None,
)


# asyncio primitives bind to the loop they are first used on; tests spin a
# fresh loop per ``asyncio.run``. Cache the semaphores per running loop so a
# semaphore created on a since-closed loop is never reused. Both pools are
# keyed by DEPTH — see ``_tier_capacity`` for why that is load-bearing.
_global_sems: dict[int, dict[int, asyncio.Semaphore]] = {}
_chain_sems: dict[int, dict[tuple[str, int], asyncio.Semaphore]] = {}


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


def _tier_capacity(depth: int) -> int:
    """How many child runs may execute concurrently AT ``depth``.

    Both pools are keyed by depth, and that is what makes this scheme
    deadlock-FREE rather than merely deadlock-prone-but-lucky.

    The hazard is real and predates this module's limits. A child holds its
    slot for its WHOLE run — including the stretch where it is doing nothing
    but awaiting a grandchild, which needs a slot of its own. With ONE flat
    pool of N, N children each awaiting a grandchild hold all N slots, and the
    grandchildren queue on a pool that can only drain once their parents
    return: a textbook hold-and-wait cycle that hangs every one of them, plus
    every unrelated session behind them. With the old flat 16-slot global pool
    that needed just two ordinary 8-wide fan-outs to trigger. A per-root
    semaphore has exactly the same defect (a parent holds a root slot while
    awaiting a child that needs one from the same root pool), which is why
    "just key the existing semaphore on the root" is NOT the fix.

    Giving each depth its own pool imposes a resource hierarchy: a holder at
    depth d only ever waits on depth d+1, never on depth d or shallower. Waits
    therefore run strictly one way along a finite, acyclic ladder (depth is
    bounded by ``_MAX_DEPTH``), so the wait-for graph can never contain a
    cycle, and no cycle means no deadlock. Progress is guaranteed from the
    leaves inward: the deepest tier depends on nothing, drains, and releases
    its parents in turn.

    Depth 0 is the RESERVED tier: it holds "root" runs that begin a chain
    rather than continue one — a scheduled firing, a workflow node, an event
    delivery. Because delegation can only ever occupy tiers ≥1, a runaway
    delegation tree can never consume a depth-0 slot, so the nightly cron still
    fires while a chat session is busy melting down. That is the headroom
    reservation for non-delegation origins.

    Tiers 0 and 1 get the full configured budget — that is where all real work
    lives (automation roots, and the one-level fan-out that is the overwhelming
    majority of delegation), so the common path is not throttled at all
    relative to the old flat pool. Deeper tiers halve each step, which dampens
    exactly the shape a runaway has — the deeper a chain drills, the less
    parallelism it is handed — while still letting genuine multi-level
    decomposition through.

    The trade-off, stated plainly: per-tier budgets raise the theoretical
    ceiling on concurrent runs from a flat 16 to 16+16+8+4+2+1 = 47 at the
    defaults. That is deliberate. The old flat 16 was not a ceiling that held —
    it was a ceiling that DEADLOCKED on the way to being reached, so the safe
    capacity was never actually 16. Normal traffic never leaves tiers 0-1 (≤32,
    and typically ≤16); the deep tiers are narrow precisely because only a
    pathological tree ever occupies them, and filling all of them at once
    requires the runaway that ``_MAX_DEPTH`` already cuts off. Splitting one
    16-slot budget across tiers instead would have halved everyday depth-1
    fan-out to 8 — a throughput regression on real production crons, paid every
    night, to buy headroom against a case that cannot occur.
    """
    if depth <= 1:
        return _GLOBAL_CONCURRENCY
    return max(1, _GLOBAL_CONCURRENCY >> (depth - 1))


def _global_semaphore(depth: int) -> asyncio.Semaphore:
    loop_id = id(asyncio.get_running_loop())
    per_loop = _global_sems.setdefault(loop_id, {})
    sem = per_loop.get(depth)
    if sem is None:
        sem = asyncio.Semaphore(_tier_capacity(depth))
        per_loop[depth] = sem
    return sem


def _chain_semaphore(root: Optional[str], depth: int) -> Optional[asyncio.Semaphore]:
    """The per-chain cap for ``root`` at ``depth``.

    Keyed on ``(root, depth)`` rather than ``root`` alone for the reason in
    ``_tier_capacity``: a single per-root semaphore would let a parent hold a
    root slot while awaiting a child that needs one from the same pool.
    """
    if not root or _PER_CHAIN_CONCURRENCY <= 0:
        return None
    loop_id = id(asyncio.get_running_loop())
    per_loop = _chain_sems.setdefault(loop_id, {})
    key = (root, depth)
    sem = per_loop.get(key)
    if sem is None:
        sem = asyncio.Semaphore(_PER_CHAIN_CONCURRENCY)
        per_loop[key] = sem
    return sem


def _chain_of(parent_session_id: Optional[str], child_sid: str,
              origin: str) -> tuple[int, str]:
    """Resolve ``(depth, root)`` for a spawn about to happen.

    Reads the ambient chain (see ``_depth_var``) rather than the session id, so
    it is correct even for a caller that supplied its own ``session_id``.

    A non-delegation origin spawned from OUTSIDE any chain is a root: the
    Scheduler firing a cron on its own task, the executor running a workflow
    node. It starts a chain at depth 0 instead of continuing one.

    A non-delegation origin spawned from INSIDE a chain still counts as a hop.
    That closes an otherwise trivial bypass of the depth limit: ``run_dream_mode``
    spawns with ``origin="scheduler"``, so a sub-agent that called it would
    re-enter at depth 0 and could then delegate again — alternating the two
    origins to recurse forever underneath a limit that only watched delegation.
    """
    ambient_depth = _depth_var.get()
    ambient_root = _root_var.get()
    if origin == "delegation" or ambient_root is not None:
        depth = ambient_depth + 1
    else:
        depth = 0
    # Prefer the inherited root so every descendant of one spawn shares a key.
    # Falling back to the parent keeps two different chat sessions (or two
    # different cron tasks) on separate budgets; falling back to the child's own
    # id covers a parentless root.
    root = ambient_root or parent_session_id or child_sid
    return depth, root


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
    allowed_tools: Optional[Any] = None,
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
        allowed_tools: OPTIONAL opt-in per-child tool scoping. Omit (``None``,
            the default) and the child runs with the FULL toolset exactly as
            today — byte-identical, no contextvar touched. When an iterable of
            MCP tool-family / server names (e.g. ``["vault", "web"]``) is passed,
            the child's runtime is built with only those families for the
            duration of its run (see ``src.core.tool_scope`` +
            ``models.native_provider``). The child can therefore only ever be
            NARROWER than the parent's grant, never broader. This never changes
            what an unrestricted child receives.
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

    # Bound the chain BEFORE any side effect: minting an id is pure, but the
    # metadata pre-stamp below creates a durable row, and a run we are about to
    # refuse must not leave a ghost session in the user's list.
    depth, root = _chain_of(parent_session_id, child_sid, origin)
    if depth > _MAX_DEPTH:
        elog(
            "child_session.depth_exceeded",
            level="warning",
            session_id=parent_session_id,
            chain_root=root,
            depth=depth,
            max_depth=_MAX_DEPTH,
            origin=origin,
            delegated_to=model_id,
            title=title,
        )
        raise DelegationDepthExceeded(
            f"Delegation depth limit reached: this task is already {depth - 1} "
            f"levels deep in a chain of sub-agents (the limit is {_MAX_DEPTH}). "
            f"Spawning another would risk unbounded recursion, so it was "
            f"refused. Do this part of the work yourself in the current session "
            f"rather than delegating it again — or, if the task genuinely needs "
            f"deeper decomposition, report back to your parent so it can "
            f"restructure the work.",
            root=root,
            depth=depth,
        )

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

    chain_sem = _chain_semaphore(root, depth)
    global_sem = _global_semaphore(depth)

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

    # Bind the chain for the child's run. Every nested spawn the child makes
    # reads these, so this is what turns "my depth" into "my child's depth" one
    # hop down. The reset in ``finally`` is load-bearing rather than tidiness:
    # two delegations issued SEQUENTIALLY on one task (a model that calls
    # ``delegate_task`` twice in a row, unfanned) must each be measured from the
    # parent's depth — without the reset the second would inherit the first's
    # and the chain would appear to deepen when it had not.
    depth_tok = _depth_var.set(depth)
    root_tok = _root_var.set(root)
    # Opt-in per-child tool scoping. Default (allowed_tools is None) → the
    # contextvar is NOT touched, so an unrestricted child runs with the full
    # toolset exactly as before (byte-identical). When a subset IS requested we
    # additionally INTERSECT it with any allowlist already in force on this
    # context (a restricted ancestor), so a descendant can only ever NARROW its
    # parent's grant, never widen it — the single choke point where that
    # invariant holds for every spawn origin, not just ``delegate_task``.
    scope_tok = None
    if allowed_tools is not None:
        requested = frozenset(normalize_family(t) for t in allowed_tools)
        ambient = current_tool_allowlist()
        effective = requested if ambient is None else (requested & ambient)
        scope_tok = set_tool_allowlist(effective)
    try:
        if chain_sem is None:
            async with global_sem:
                text = await _run()
        else:
            # Chain slot BEFORE global slot — never the reverse, for two
            # reasons. (1) Starvation: a child queueing for its own chain's cap
            # must not sit on a global slot while it waits, or a runaway tree
            # parks the shared pool merely to stand in its own queue — which is
            # precisely how one bad session froze unrelated ones. Queueing on
            # the chain sem costs the rest of the server nothing. (2) Ordering:
            # every acquisition in the process takes these two in this one
            # order, so there is no AB-BA pair to deadlock on.
            if chain_sem.locked():
                # The tree is at its cap and this spawn will wait. Normal
                # backpressure, not a failure — but it is the signal that
                # distinguishes "slow" from "throttled" when a run looks stuck,
                # so it is worth a line. Only fires when it actually blocks.
                elog(
                    "child_session.chain_saturated",
                    level="warning",
                    session_id=parent_session_id,
                    chain_root=root,
                    depth=depth,
                    limit=_PER_CHAIN_CONCURRENCY,
                    origin=origin,
                    child_session_id=child_sid,
                )
            async with chain_sem:
                async with global_sem:
                    text = await _run()
            _maybe_prune_chain_sem(root, depth, chain_sem)
    finally:
        _depth_var.reset(depth_tok)
        _root_var.reset(root_tok)
        if scope_tok is not None:
            reset_tool_allowlist(scope_tok)

    return ChildSessionResult(session_id=child_sid, text=text or "")


def _maybe_prune_chain_sem(root: Optional[str], depth: int,
                           sem: asyncio.Semaphore) -> None:
    """Drop a per-chain semaphore once it's fully released, so the cache
    doesn't grow without bound over a long-lived server. Keyed ``(root, depth)``
    to match ``_chain_semaphore``: an entry is only dropped when that tier of
    that chain is completely idle, so a live sibling can never lose its cap."""
    if not root:
        return
    try:
        if sem._value >= _PER_CHAIN_CONCURRENCY:  # type: ignore[attr-defined]
            per_loop = _chain_sems.get(id(asyncio.get_running_loop()))
            if per_loop is not None:
                per_loop.pop((root, depth), None)
    except Exception:  # noqa: BLE001
        pass
