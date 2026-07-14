"""Bounding runaway delegation recursion without banning legitimate depth.

``run_child_session`` is the single spawn primitive for every non-interactive
origin, and a child runs the ordinary ``Agent.run`` path with the FULL tool set
— including ``delegate_task`` itself. Before these limits existed nothing
stopped a child spawning a grandchild forever, and the per-parent concurrency
cap could not see it: keyed on the IMMEDIATE parent, a chain A→B→C→D handed
every hop a fresh semaphore with a fresh full allowance, so the cap bounded
breadth and was blind to depth. The only real backstop was the 16-slot global
pool — so one runaway chain saturated it and queued every other session on the
server, healthy or not. A single misbehaving cron could freeze everything.

Worse, the global pool could not merely fill — it could DEADLOCK. A child holds
its slot for its whole run, including while it is doing nothing but awaiting a
grandchild that needs a slot from the same pool; enough children waiting on
children and the pool can only drain once its own waiters return. Two ordinary
8-wide fan-outs were enough to reach that with the old flat 16-slot pool.

What these tests lock down (vision §4 must keep working throughout — deep,
durable delegation is a FEATURE here, "scaling to hundreds of concurrent child
sessions"; the goal is bounding a runaway, not banning depth):

  - depth/root are threaded through the chain, and accounting attributes to the
    ROOT rather than the immediate parent;
  - a chain deeper than the limit is refused cleanly, with an actionable error
    and a structured elog — not an opaque crash;
  - a non-delegation origin cannot be used to reset depth and bypass the limit;
  - legitimate fan-out still runs, and still runs CONCURRENTLY;
  - the hold-and-wait shape that deadlocks a flat pool completes here (with a
    control proving the hazard is real and the test is not vacuous);
  - a runaway tree does not starve unrelated sessions of the global pool.
"""
from __future__ import annotations

import asyncio
import contextlib
import time

from ._framework import TestContext, test


# ── Stubs / helpers ──────────────────────────────────────────────────


class _Agent:
    """The minimum ``run_child_session`` drives: ``run`` + ``release_session``.

    ``body`` (optional) runs INSIDE the bound chain context, so a test can read
    the ambient depth/root there — or spawn again to build a real chain.
    """

    name = "spy"
    model = None

    def __init__(self, body=None) -> None:
        self.body = body
        self.runs: list[str] = []
        self.released: list[str] = []

    async def run(self, *, message, user_id, session_id,
                  model_override=None, author=None, on_status=None) -> str:
        self.runs.append(session_id)
        if self.body is not None:
            return await self.body(session_id)
        return "ok"

    async def release_session(self, session_id, *, model_override=None) -> None:
        self.released.append(session_id)


@contextlib.contextmanager
def _limits(*, max_depth=None, global_conc=None, per_chain=None):
    """Patch the module's limits and reset the per-loop semaphore caches.

    Capacity is read when a semaphore is CREATED, so the caches must be cleared
    on the way in (to build pools at the patched size) and on the way out (so a
    later test never inherits a tiny pool from this one).
    """
    from src.core import child_session as cs

    old = (cs._MAX_DEPTH, cs._GLOBAL_CONCURRENCY, cs._PER_CHAIN_CONCURRENCY)
    if max_depth is not None:
        cs._MAX_DEPTH = max_depth
    if global_conc is not None:
        cs._GLOBAL_CONCURRENCY = global_conc
    if per_chain is not None:
        cs._PER_CHAIN_CONCURRENCY = per_chain
    cs._global_sems.clear()
    cs._chain_sems.clear()
    try:
        yield cs
    finally:
        cs._MAX_DEPTH, cs._GLOBAL_CONCURRENCY, cs._PER_CHAIN_CONCURRENCY = old
        cs._global_sems.clear()
        cs._chain_sems.clear()


def _spawn(cs, agent, *, parent, origin="delegation", **kw):
    return cs.run_child_session(
        agent=agent, db=None, parent_session_id=parent, origin=origin,
        origin_ref=kw.pop("origin_ref", {}), title="t", prompt="p", **kw,
    )


# ── Depth / root accounting ──────────────────────────────────────────


@test("delegation_depth", "chain accounting attributes to the ROOT, not the immediate parent")
async def t_root_accounting(ctx: TestContext) -> None:
    """The containment hole this fixes: keyed on the immediate parent, every
    hop of A→B→C→D got a fresh budget, so a chain accumulated against nothing.
    Keyed on the root, the whole tree shares one — and depth counts hops.

    Also pins the cron shape that matters in production: a scheduled firing is a
    ROOT (depth 0, the reserved tier), its per-item sub-agents are depth 1, and
    their helpers depth 2 — all still accounted to the cron, not to whichever
    sub-agent happened to spawn them.
    """
    with _limits(max_depth=5) as cs:
        seen: list[tuple[int, str]] = []

        async def leaf(sid):
            seen.append((cs._depth_var.get(), cs._root_var.get()))
            return "ok"

        # A chat turn delegating: depth 1, rooted on the CHAT session.
        await _spawn(cs, _Agent(leaf), parent="chat-1")
        assert seen[-1] == (1, "chat-1"), seen

        # A scheduled firing spawned from outside any chain is a ROOT: depth 0.
        # It must not land on a delegation tier — that reservation is what keeps
        # the nightly cron firing while a chat session melts down.
        async def cron_body(sid):
            seen.append((cs._depth_var.get(), cs._root_var.get()))

            # The cron's per-item sub-agent: depth 1, still rooted on the CRON.
            async def item_body(_sid):
                seen.append((cs._depth_var.get(), cs._root_var.get()))

                # That sub-agent's own helper: depth 2, STILL rooted on the
                # cron. Under the old per-parent key this hop accounted to the
                # item sub-agent and the chain vanished from the books.
                async def helper_body(__sid):
                    seen.append((cs._depth_var.get(), cs._root_var.get()))
                    return "ok"

                await _spawn(cs, _Agent(helper_body), parent=_sid)
                return "ok"

            await _spawn(cs, _Agent(item_body), parent=sid)
            return "ok"

        await _spawn(cs, _Agent(cron_body), parent="scheduler:t1",
                     origin="scheduler")

    assert seen[1:] == [
        (0, "scheduler:t1"),   # the firing itself — reserved root tier
        (1, "scheduler:t1"),   # per-item sub-agent
        (2, "scheduler:t1"),   # its helper — depth 2 is NORMAL production traffic
    ], seen


@test("delegation_depth", "sequential delegations on one task each measure from the parent")
async def t_sequential_no_accumulation(ctx: TestContext) -> None:
    """Two ``delegate_task`` calls in a row (unfanned) run on the SAME task, so
    the contextvar set by the first must be reset before the second — otherwise
    depth would creep upward with breadth and a wide-but-flat turn would trip a
    limit meant for recursion."""
    with _limits(max_depth=5) as cs:
        seen: list[int] = []

        async def leaf(sid):
            seen.append(cs._depth_var.get())
            return "ok"

        agent = _Agent(leaf)
        await _spawn(cs, agent, parent="chat-1")
        await _spawn(cs, agent, parent="chat-1")
        await _spawn(cs, agent, parent="chat-1")

    assert seen == [1, 1, 1], f"depth accumulated across siblings: {seen}"


# ── The depth limit ──────────────────────────────────────────────────


@test("delegation_depth", "a chain deeper than the limit is refused cleanly")
async def t_chain_refused(ctx: TestContext) -> None:
    """A self-delegating agent — the runaway shape — must terminate at the
    limit, having run exactly ``_MAX_DEPTH`` sessions, and raise a typed error
    naming the lineage rather than crashing opaquely."""
    with _limits(max_depth=3) as cs:
        events: list[tuple] = []
        real_elog = cs.elog
        cs.elog = lambda ev, **kw: events.append((ev, kw))
        try:
            agent = _Agent()

            async def recurse(sid):
                # Every child immediately delegates again, forever.
                await _spawn(cs, agent, parent=sid)
                return "ok"

            agent.body = recurse

            try:
                await _spawn(cs, agent, parent="chat-runaway")
                raise AssertionError("runaway chain was never refused")
            except cs.DelegationDepthExceeded as e:
                err = e
        finally:
            cs.elog = real_elog

    # Depths 1..3 ran; the 4th hop was refused.
    assert len(agent.runs) == 3, f"expected 3 runs before the cut, got {agent.runs}"
    assert err.depth == 4, err.depth
    assert err.root == "chat-runaway", err.root
    # Actionable: the model must learn it hit a structural limit and should do
    # the work itself — not read it as a transient fault worth retrying.
    msg = str(err).lower()
    assert "depth limit" in msg and "yourself" in msg, str(err)

    # Observable: a limit that fires silently is a limit nobody can debug.
    trips = [kw for ev, kw in events if ev == "child_session.depth_exceeded"]
    assert len(trips) == 1, [e for e, _ in events]
    assert trips[0]["depth"] == 4 and trips[0]["max_depth"] == 3, trips[0]
    assert trips[0]["chain_root"] == "chat-runaway", trips[0]
    assert trips[0]["level"] == "warning", trips[0]


@test("delegation_depth", "a refused spawn leaves no durable ghost row")
async def t_no_ghost_row(ctx: TestContext) -> None:
    """The depth check runs before the metadata pre-stamp, so a refused spawn
    never leaves a session in the user's list that never ran."""
    upserts: list[str] = []

    class _DB:
        async def get_session(self, sid):
            return None

        async def primary_owner_handle(self):
            return "owner"

        async def upsert_session(self, sid, **kw):
            upserts.append(sid)

    with _limits(max_depth=1) as cs:
        agent = _Agent()

        async def recurse(sid):
            await cs.run_child_session(
                agent=agent, db=_DB(), parent_session_id=sid,
                origin="delegation", origin_ref={}, title="t", prompt="p",
            )
            return "ok"

        agent.body = recurse
        try:
            await cs.run_child_session(
                agent=agent, db=_DB(), parent_session_id="chat-1",
                origin="delegation", origin_ref={}, title="t", prompt="p",
            )
            raise AssertionError("not refused")
        except cs.DelegationDepthExceeded:
            pass

    # Only the depth-1 child was ever stamped; the refused depth-2 spawn wasn't.
    assert len(upserts) == 1, upserts


@test("delegation_depth", "a non-delegation origin cannot reset depth to bypass the limit")
async def t_no_origin_bypass(ctx: TestContext) -> None:
    """``run_dream_mode`` spawns with ``origin="scheduler"``. If a
    non-delegation origin restarted the count whenever it appeared, an agent
    could alternate the two origins and recurse forever underneath a limit that
    only watched delegation. Inside a chain, every origin is a hop."""
    with _limits(max_depth=2) as cs:
        agent = _Agent()

        async def scheduler_hop(sid):
            # A "scheduler" spawn from INSIDE a chain — must still count.
            await _spawn(cs, agent, parent=sid, origin="scheduler",
                         origin_ref={"task_id": "dream", "run_id": "r1"})
            return "ok"

        async def delegate_hop(sid):
            await _spawn(cs, agent, parent=sid, origin="delegation")
            return "ok"

        # chat → delegation(1) → scheduler(2) → delegation(3) > limit 2.
        depth1 = _Agent(scheduler_hop)
        agent.body = delegate_hop

        try:
            await _spawn(cs, depth1, parent="chat-1")
            raise AssertionError("origin-alternating bypass was not refused")
        except cs.DelegationDepthExceeded as e:
            assert e.depth == 3, e.depth
            # The root is carried through the origin switch, not reset by it.
            assert e.root == "chat-1", e.root


@test("delegation_depth", "delegate_task turns the limit into a clean tool error, not a crash")
async def t_delegate_task_error(ctx: TestContext) -> None:
    """The MCP handler must hand the model a plain, actionable explanation —
    an opaque traceback invites the retry that IS the runaway."""
    from src.mcp.servers.delegation import handlers as dh

    with _limits(max_depth=1) as cs:
        agent = _Agent()

        async def recurse(sid):
            # The child re-enters delegate_task exactly as a real sub-agent
            # would, through the same context the runtime installs per turn.
            return await dh.delegate_task("again", model_id="m")

        agent.body = recurse

        class _DB:
            async def get_session(self, sid):
                return None

            async def primary_owner_handle(self):
                return "owner"

            async def upsert_session(self, sid, **kw):
                return None

        tokens = dh.install_context(
            session_id="chat-1", pool=None, db=_DB(), dispatcher=None, agent=agent,
        )
        try:
            out = await dh.delegate_task("do the thing", model_id="m")
        finally:
            dh.reset_context(tokens)

    # The OUTER delegation succeeded; the inner one hit the limit and came back
    # as a structured error the model can act on, without killing the parent.
    assert out["status"] == "ok", out
    inner = out["answer"]
    assert inner["status"] == "error", inner
    assert inner.get("depth_limit_reached") is True, inner
    assert "depth limit" in inner["error"].lower(), inner


# ── Concurrency: fan-out, deadlock-freedom, fairness ─────────────────


@test("delegation_depth", "legitimate fan-out still runs CONCURRENTLY, not serialized")
async def t_fanout_concurrent(ctx: TestContext) -> None:
    """Vision §4: parallel fan-out within one turn is the common, blessed case
    ("sub-agents may run in parallel when the work is independent"). Wall-clock
    is the clearest signal: 8 × 0.3s siblings finish in ~0.3s parallel, ~2.4s
    serialized."""
    with _limits(max_depth=5, global_conc=16, per_chain=8) as cs:
        peak = 0
        live = 0

        async def work(sid):
            nonlocal peak, live
            live += 1
            peak = max(peak, live)
            try:
                await asyncio.sleep(0.3)
            finally:
                live -= 1
            return "ok"

        agent = _Agent(work)
        t0 = time.monotonic()
        await asyncio.gather(*(
            _spawn(cs, agent, parent="chat-fan") for _ in range(8)
        ))
        total = time.monotonic() - t0

    assert peak == 8, f"fan-out serialized; peak concurrency was {peak}"
    # Parallel ≈0.3s; serialized ≥2.4s. 0.9s crosses the boundary with room
    # for scheduler noise.
    assert total < 0.9, f"fan-out serialized; total={total:.3f}s"


async def _two_parents_each_awaiting_a_child(cs, timeout: float) -> None:
    """Drive the exact hold-and-wait shape, through the real spawn path.

    Both depth-1 children rendezvous at a barrier BEFORE either spawns, so both
    are provably holding global slots at the moment their grandchildren ask for
    one. The barrier is what makes this deterministic: a semaphore with a free
    slot never yields on acquire, so without it the first parent runs to
    completion before the second starts and the pool never actually fills — the
    hazard hides. Two parents that each fan out is ordinary traffic, not an
    exotic race.
    """
    barrier = asyncio.Barrier(2)

    async def grandchild_body(sid):
        await asyncio.sleep(0.05)
        return "ok"

    async def child_body(sid):
        await barrier.wait()   # both parents now hold depth-1 slots
        await _spawn(cs, _Agent(grandchild_body), parent=sid)
        return "ok"

    agent = _Agent(child_body)
    await asyncio.wait_for(
        asyncio.gather(*(
            _spawn(cs, agent, parent="chat-deep") for _ in range(2)
        )),
        timeout=timeout,
    )


@test("delegation_depth", "CONTROL: this shape DOES deadlock on the old flat pool")
async def t_flat_pool_control(ctx: TestContext) -> None:
    """Proves the hazard is real and the next test is not vacuous.

    Identical code path, identical shape — the ONLY difference is that the
    global pool is flat (one semaphore for every depth), as it was before this
    change. Both parents hold the pool's 2 slots and both then await a child
    that needs a slot only a parent can free: nobody can proceed, ever. If this
    ever stops hanging, the depth-keyed pools are no longer what is preventing
    the deadlock and both tests need rethinking.
    """
    with _limits(max_depth=5, global_conc=2, per_chain=8) as cs:
        flat = asyncio.Semaphore(2)
        real = cs._global_semaphore
        cs._global_semaphore = lambda depth: flat  # the old, un-tiered pool
        try:
            await _two_parents_each_awaiting_a_child(cs, timeout=1.0)
        except asyncio.TimeoutError:
            return  # expected: hold-and-wait cycle, no progress possible
        finally:
            cs._global_semaphore = real
    raise AssertionError(
        "the flat pool did NOT deadlock — this control is invalid, so the "
        "deadlock-freedom test below proves nothing"
    )


@test("delegation_depth", "the deadlock shape completes: depth-keyed pools cannot cycle")
async def t_no_deadlock(ctx: TestContext) -> None:
    """The shape the control just proved hangs on a flat pool must COMPLETE
    here, with the pool sized identically (2 slots, 2 parents each awaiting a
    child).

    Depth-keyed pools impose a resource hierarchy: a holder at depth d only ever
    waits on depth d+1, never on depth d or shallower, so waits run strictly one
    way along a finite ladder and the wait-for graph cannot contain a cycle.
    Progress is guaranteed from the leaves inward. The grandchildren serialize
    (tier 2 is deliberately narrower) — they do not hang.
    """
    with _limits(max_depth=5, global_conc=2, per_chain=8) as cs:
        await _two_parents_each_awaiting_a_child(cs, timeout=5.0)


@test("delegation_depth", "a runaway tree does not starve unrelated sessions")
async def t_no_starvation(ctx: TestContext) -> None:
    """The freeze this whole change exists to stop: one tree eating the shared
    pool and queueing everyone else.

    Discriminates the acquisition ORDER specifically. Children queueing for
    their own chain's cap must not sit on global slots while they wait — take
    the global slot first and a runaway parks the shared pool merely to stand in
    its own queue, and the unrelated session behind it waits out the whole tree.
    """
    with _limits(max_depth=5, global_conc=4, per_chain=2) as cs:
        t0 = time.monotonic()
        started: dict[str, float] = {}

        async def runaway(sid):
            await asyncio.sleep(0.4)
            return "ok"

        async def bystander(sid):
            started["bystander"] = time.monotonic() - t0
            return "ok"

        # A runaway tree wants 6 children at once but its chain cap is 2, so 4
        # sit in its own queue. They must queue WITHOUT holding global slots.
        tree = [
            asyncio.create_task(_spawn(cs, _Agent(runaway), parent="chat-runaway"))
            for _ in range(6)
        ]
        await asyncio.sleep(0.05)  # let the tree fill its cap and pile up

        # An unrelated session spawns one child. It must start ~immediately.
        await asyncio.wait_for(
            _spawn(cs, _Agent(bystander), parent="chat-healthy"), timeout=2.0,
        )
        await asyncio.wait_for(asyncio.gather(*tree), timeout=5.0)

    # Starved (old shape): the bystander waits out the tree's 3 waves ≈1.2s.
    # Contained: it starts as soon as the scheduler gets to it, ≈0.05s.
    assert started["bystander"] < 0.3, (
        f"bystander starved by the runaway tree; started at "
        f"{started['bystander']:.3f}s"
    )
