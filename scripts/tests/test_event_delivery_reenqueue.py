"""At-least-once event delivery: an orphaned webhook delivery is re-enqueued
and re-dispatched instead of being dropped as ``failed`` — and a replay never
double-messages the customer.

The bug (esound-openagent, 2026-07): ``webhook_site`` records a delivery
``claimed=True`` then dispatches the agent turn as a detached background task
and returns 202 — at-most-once. If the process died before the turn finished,
``reap_orphan_event_deliveries`` marked the in-flight row ``failed`` and never
re-fired it. On the live pod that dropped **1181 support tickets** (of 1267
``failed`` rows, all carrying the ``reaped: orphan …`` marker — not failures).

The fix makes delivery at-least-once: reap RE-ENQUEUES an orphan
(``claimed_at=NULL``, status → ``received``) so the Scheduler drain
re-dispatches it, bounded by ``reenqueue_count`` so a poison delivery can't
crash-loop.

Safety — why a replay is not a double-reply. An event child session id is the
deterministic ``event:{event_id}:{delivery_id}``, and the only external event
(``Replio inbound thread``) is session-bound on the thread id. So a replay of
the same delivery ALWAYS resumes the SAME session — the agent re-runs with its
own prior transcript, and the customer reply goes through Replio's thread-scoped
reply_guard, which suppresses a second outbound for an inbound already answered.
These tests pin the local half of that guarantee (session reuse on replay) and
model the thread-scoped guard end-to-end.
"""
from __future__ import annotations

import os
import time
import uuid

from ._framework import TestContext, test


class _ReplioSpyAgent:
    """A stand-in agent that models Replio's thread-scoped reply_guard.

    ``run`` is what ``run_child_session`` calls for an agent without
    ``run_stream``. It "sends a reply" for an inbound at most once per session:
    the second time the identical prompt arrives on the same session (a replay
    of the same inbound), the reply_guard suppresses it — exactly the
    outbound-newer-than-this-inbound rule enforced server-side in Replio.
    """

    name = "replio-spy"
    model = None

    def __init__(self) -> None:
        # session_id -> set of inbound prompts already answered on that thread.
        self._answered: dict[str, set[str]] = {}
        self.replies_sent: list[tuple[str, str]] = []  # (session_id, prompt)
        self.runs: list[tuple[str, str]] = []          # every turn (incl. no-ops)

    async def refresh_registries(self):
        return None

    async def run(self, *, message, user_id, session_id, model_override=None,
                  author=None, on_status=None):
        self.runs.append((session_id, message))
        answered = self._answered.setdefault(session_id, set())
        if message in answered:
            # reply_guard: this inbound was already answered on this thread.
            return "[reply_guard] already answered — no second reply"
        answered.add(message)
        self.replies_sent.append((session_id, message))
        return "reply sent"

    async def release_session(self, session_id, *, model_override=None):
        return None


def _fresh_db_path(ctx: TestContext) -> "os.PathLike":
    return ctx.db_path.with_name(f"evredel-{uuid.uuid4().hex[:8]}.db")


async def _make_db(path):
    from src.memory.db import MemoryDB
    db = MemoryDB(str(path))
    await db.connect()
    return db


async def _add_bound_event(db, *, slug: str, binding: bool = True):
    from src.core.event_secret import make_secret_material
    clear, enc, hint = make_secret_material(db_path=db.db_path)
    return await db.add_event(
        name=f"evt-{slug}", action_kind="prompt", slug=slug,
        secret_enc=enc, secret_hint=hint,
        prompt_template="Handle ticket {{payload.ticket.id}}: {{payload.ticket.body}}",
        session_binding_enabled=binding,
        session_binding_path="ticket.id",
    )


async def _dispatch(db, ev, agent, did, payload):
    """Run one delivery through the REAL dispatch path (as the scheduler drain
    does), returning the produced child session id."""
    from src.core.scheduler import Scheduler
    from src.core.event_dispatcher import dispatch_event
    os.environ.pop("OPENAGENT_SCHEDULER_DURABLE_SESSIONS", None)
    scheduler = Scheduler(db=db, agent=agent)  # type: ignore[arg-type]
    result = await dispatch_event(
        agent=agent, db=db, scheduler=scheduler, event=ev,
        payload=payload, delivery_id=did, source="webhook",
    )
    return result["session_id"]


# ── 1. Orphan is re-enqueued (not failed) and re-dispatches ───────────────


@test("event_delivery_reenqueue",
      "a claimed-but-incomplete orphan is re-enqueued, not failed, and re-dispatches")
async def t_orphan_reenqueued_and_redispatched(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        agent = _ReplioSpyAgent()
        eid = await _add_bound_event(db, slug="reap-basic")
        ev = await db.get_event(eid)

        # A delivery that was claimed and had started its turn (status=running)
        # when the process died: a classic orphan.
        did = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "T-1", "body": "help"}},
            claimed=True,
        )
        await db.update_event_delivery(did, status="running")

        n = await db.reap_orphan_event_deliveries(max_attempts=5)
        assert n == 1, f"expected 1 orphan acted on, got {n}"

        row = await db.get_event_delivery(did)
        assert row["status"] == "received", \
            f"orphan should be RE-ENQUEUED (received), got {row['status']!r}"
        assert row["claimed_at"] is None, "re-enqueued row must drop its claim"
        assert row["reenqueue_count"] == 1, row["reenqueue_count"]
        assert (row.get("error") or "").startswith("re-enqueued"), row.get("error")

        # The scheduler drain claims exactly this row (claimed_at IS NULL) …
        claimed = await db.claim_pending_event_deliveries()
        assert [c["id"] for c in claimed] == [did], claimed

        # … and re-dispatching it runs the turn and finalizes success.
        await _dispatch(db, ev, agent, did, {"ticket": {"id": "T-1", "body": "help"}})
        assert len(agent.replies_sent) == 1, agent.replies_sent
        final = await db.get_event_delivery(did)
        assert final["status"] == "success", final["status"]
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 2. A replay of an already-replied delivery does NOT reply twice ───────


@test("event_delivery_reenqueue",
      "replay resumes the SAME bound session and the reply_guard blocks a 2nd reply")
async def t_replay_no_double_reply(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        agent = _ReplioSpyAgent()
        eid = await _add_bound_event(db, slug="reap-idem")
        ev = await db.get_event(eid)

        payload1 = {"ticket": {"id": "T-1", "body": "first message"}}
        did1 = await db.add_event_delivery(event_id=eid, payload=payload1, claimed=True)
        sid_first = await _dispatch(db, ev, agent, did1, payload1)
        assert len(agent.replies_sent) == 1, "first turn should send exactly one reply"

        # The process died right after the reply was sent, before the delivery
        # was finalized: orphan it.
        await db.update_event_delivery(did1, status="running")
        await db.reap_orphan_event_deliveries(max_attempts=5)

        # Replay via the drain → dispatch again.
        claimed = await db.claim_pending_event_deliveries()
        assert did1 in [c["id"] for c in claimed]
        sid_replay = await _dispatch(db, ev, agent, did1, payload1)

        # SAFETY invariant: the replay resumed the SAME session (so Replio's
        # thread-scoped reply_guard applies), and NO second customer reply was
        # produced.
        assert sid_replay == sid_first, \
            f"replay must reuse the bound session: {sid_first} != {sid_replay}"
        assert len(agent.replies_sent) == 1, \
            f"replay double-messaged the customer: {agent.replies_sent}"
        assert len(agent.runs) == 2, "the turn did re-run (guard is at reply, not dispatch)"

        # And a genuinely-new inbound on the SAME thread still gets answered —
        # the guard is per-inbound, never a blanket thread mute (never-drop).
        payload2 = {"ticket": {"id": "T-1", "body": "second message"}}
        did2 = await db.add_event_delivery(event_id=eid, payload=payload2, claimed=True)
        sid2 = await _dispatch(db, ev, agent, did2, payload2)
        assert sid2 == sid_first, "same thread → same bound session"
        assert len(agent.replies_sent) == 2, \
            "a new inbound on the thread must still be answered"
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 3. A genuinely-new delivery still processes exactly once ──────────────


@test("event_delivery_reenqueue",
      "a fresh delivery processes once; reap with no orphans is a no-op")
async def t_new_delivery_processes_once(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        agent = _ReplioSpyAgent()
        eid = await _add_bound_event(db, slug="reap-fresh")
        ev = await db.get_event(eid)

        payload = {"ticket": {"id": "T-9", "body": "brand new"}}
        did = await db.add_event_delivery(event_id=eid, payload=payload, claimed=True)
        await _dispatch(db, ev, agent, did, payload)
        assert len(agent.replies_sent) == 1
        row = await db.get_event_delivery(did)
        assert row["status"] == "success" and row["reenqueue_count"] == 0, row

        # Nothing is in flight → reap touches nothing (no spurious re-fire).
        n = await db.reap_orphan_event_deliveries(max_attempts=5)
        assert n == 0, f"reap should be a no-op with no orphans, acted on {n}"
        assert len(agent.replies_sent) == 1, "reap must not re-fire a completed delivery"
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 4. Bounded replay: a delivery that keeps orphaning is parked terminal ─


@test("event_delivery_reenqueue",
      "the replay budget is bounded — a persistently-orphaning delivery is parked failed")
async def t_retry_budget_exhausted(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        eid = await _add_bound_event(db, slug="reap-budget")
        did = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "T-1", "body": "x"}}, claimed=True,
        )

        requeues = 0
        for _ in range(6):
            # Faithful cycle: the drain claims the re-enqueued row (sets
            # claimed_at), the turn starts (status=running), then the process
            # dies again → reap.
            row = await db.get_event_delivery(did)
            if row["claimed_at"] is None:
                await db.claim_pending_event_deliveries()
            await db.update_event_delivery(did, status="running")
            await db.reap_orphan_event_deliveries(max_attempts=2)
            row = await db.get_event_delivery(did)
            if row["status"] == "received":
                requeues += 1
            elif row["status"] == "failed":
                break

        row = await db.get_event_delivery(did)
        assert requeues == 2, f"expected exactly max_attempts=2 re-enqueues, got {requeues}"
        assert row["status"] == "failed", \
            f"an exhausted orphan must be parked terminal, got {row['status']!r}"
        assert "retry budget exhausted" in (row.get("error") or ""), row.get("error")
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 5. Genuine failures / historical backfill partitioning ────────────────


@test("event_delivery_reenqueue",
      "a genuine failure is never resurrected; a historical reaped-orphan is backfilled")
async def t_partition_failed_rows(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        eid = await _add_bound_event(db, slug="reap-partition")

        # (a) A genuine application failure (bad template etc.): terminal, no
        #     reap marker, claimed. It must NOT be re-enqueued.
        genuine = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "G", "body": "x"}}, claimed=True,
        )
        await db.update_event_delivery(
            genuine, status="failed", error="EventDispatchError: bad template",
        )

        # (b) A historical orphan the OLD reaper dropped: status=failed carrying
        #     the exact legacy marker. This is a dropped ticket to recover.
        historical = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "H", "body": "y"}}, claimed=True,
        )
        await db.update_event_delivery(
            historical, status="failed",
            error="reaped: orphan from prior process",
        )

        # Default (recover_failed=True): historical is re-enqueued, genuine is not.
        n = await db.reap_orphan_event_deliveries(max_attempts=5, recover_failed=True)
        assert n == 1, f"only the historical orphan should be recovered, acted on {n}"
        assert (await db.get_event_delivery(historical))["status"] == "received"
        assert (await db.get_event_delivery(genuine))["status"] == "failed", \
            "a genuine failure must never be resurrected"

        # With backfill disabled, a fresh historical orphan is left untouched
        # (go-forward-only mode).
        hist2 = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "H2", "body": "z"}}, claimed=True,
        )
        await db.update_event_delivery(
            hist2, status="failed", error="reaped: orphan from prior process",
        )
        n2 = await db.reap_orphan_event_deliveries(max_attempts=5, recover_failed=False)
        assert n2 == 0, f"backfill disabled should recover nothing, acted on {n2}"
        assert (await db.get_event_delivery(hist2))["status"] == "failed"
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ── 6. Kill-switch falls back to legacy at-most-once behaviour ─────────────


@test("event_delivery_reenqueue",
      "the kill-switch restores legacy mark-failed behaviour (no re-enqueue)")
async def t_kill_switch_legacy(ctx: TestContext) -> None:
    path = _fresh_db_path(ctx)
    db = await _make_db(path)
    try:
        eid = await _add_bound_event(db, slug="reap-killswitch")
        did = await db.add_event_delivery(
            event_id=eid, payload={"ticket": {"id": "K", "body": "x"}}, claimed=True,
        )
        await db.update_event_delivery(did, status="running")

        n = await db.reap_orphan_event_deliveries(enabled=False)
        assert n == 1, n
        row = await db.get_event_delivery(did)
        assert row["status"] == "failed", "kill-switch must mark orphans failed"
        assert row["claimed_at"] is not None, "legacy path does not drop the claim"
        assert "reaped: orphan from prior process" in (row.get("error") or "")
    finally:
        await db.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
