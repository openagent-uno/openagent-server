"""Skill-distiller — the automatic WRITER half of the self-improvement loop.

The curator half (``test_skill_curator``) proves consolidation; this proves the
authoring pass that gives the curator something to consolidate. All LLM-free
(the distillation itself is LLM-driven and deferred, exactly like the curator /
dream-mode live tests):

  * **Distiller gating (OFF by default)** — with ``skills.distiller_enabled``
    false the ``skill-distiller`` scheduled task is NOT seeded; with it true
    (and ``skills.enabled``) it IS. Asserted through the real seeding path
    (``AgentServer._sync_skill_distiller``), no live run. Byte-identical to a
    build without the feature at defaults: no row at all.
  * **Distiller and curator are DISTINCT tasks** — each flips on its own toggle,
    seeds its own row, and one may run without the other.
  * **The layering boundary** — the distiller prompt tells it to CREATE only
    (``skill_manage(action="create")``) and NEVER merge/archive; the curator
    prompt does the opposite. That separation is the whole reason the loop is
    clean, so it is pinned in the prompt text.

Pure-unit: a throwaway ``MemoryDB`` for the seeding gate (same shape as
test_skill_curator). No LLM, pool, or gateway.
"""
from __future__ import annotations

import uuid

from ._framework import TestContext, test


# ── helpers (mirror test_skill_curator) ───────────────────────────────

async def _bare_server(config: dict, db):
    """A minimally-constructed AgentServer wired just enough to drive
    ``_sync_skill_distiller`` / ``_sync_skill_curator``: it reads ``self.config``
    and ``self.agent._db``. ``__new__`` skips the heavy ``__init__``."""
    from src.core.server import AgentServer

    srv = AgentServer.__new__(AgentServer)
    srv.config = config

    class _Agent:
        pass

    agent = _Agent()
    agent._db = db
    srv.agent = agent
    return srv, agent


async def _row(db, name: str):
    tasks = await db.get_tasks()
    return next((t for t in tasks if t["name"] == name), None)


async def _distiller_row(db):
    from src.core.builtin_tasks import SKILL_DISTILLER_TASK_NAME
    return await _row(db, SKILL_DISTILLER_TASK_NAME)


async def _curator_row(db):
    from src.core.builtin_tasks import SKILL_CURATOR_TASK_NAME
    return await _row(db, SKILL_CURATOR_TASK_NAME)


def _fresh_db(ctx: TestContext, tag: str):
    from src.memory.db import MemoryDB
    return MemoryDB(str(ctx.db_path.with_name(f"distiller-{tag}-{uuid.uuid4().hex[:8]}.db")))


# ── 1. distiller gating — OFF by default, seeded only when opted in ───

@test("skill_distiller", "OFF by default: no scheduled task seeded; ON when opted in")
async def t_distiller_gating(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler

    # (a) DEFAULT config → nothing seeded (byte-identical to no feature).
    off_db = _fresh_db(ctx, "off")
    await off_db.connect()
    try:
        srv, agent = await _bare_server({}, off_db)
        sched = Scheduler(off_db, agent)
        await srv._sync_skill_distiller(sched)
        assert await _distiller_row(off_db) is None, (
            "the distiller task was seeded with default config — it must be "
            "invisible, not merely disabled"
        )
    finally:
        await off_db.close()

    # (b) skills.enabled but distiller_enabled still false → STILL nothing.
    half_db = _fresh_db(ctx, "half")
    await half_db.connect()
    try:
        srv, agent = await _bare_server({"skills": {"enabled": True}}, half_db)
        sched = Scheduler(half_db, agent)
        await srv._sync_skill_distiller(sched)
        assert await _distiller_row(half_db) is None, (
            "skills.enabled alone seeded the distiller — it needs the second "
            "distiller_enabled gate too"
        )
    finally:
        await half_db.close()

    # (c) BOTH gates on → seeded AND enabled, with the daily default cron.
    on_db = _fresh_db(ctx, "on")
    await on_db.connect()
    try:
        srv, agent = await _bare_server(
            {"skills": {"enabled": True, "distiller_enabled": True}}, on_db)
        sched = Scheduler(on_db, agent)
        await srv._sync_skill_distiller(sched)
        row = await _distiller_row(on_db)
        assert row is not None, "both gates on but the distiller was not seeded"
        assert row["enabled"], "the distiller was seeded but left disabled"
        assert row["prompt"], "the distiller task has no prompt"

        from src.core.server import SKILL_DISTILLER_DEFAULT_CRON
        assert row["cron_expression"] == SKILL_DISTILLER_DEFAULT_CRON, row

        # (d) Turning it back off at runtime disables the surviving row.
        srv.config = {"skills": {"enabled": True, "distiller_enabled": False}}
        await srv._sync_skill_distiller(sched)
        row2 = await _distiller_row(on_db)
        assert row2 is not None, "the row was deleted, not disabled"
        assert not row2["enabled"], "runtime-off did not disable the distiller row"
    finally:
        await on_db.close()

    # (e) A custom schedule is honoured.
    cron_db = _fresh_db(ctx, "cron")
    await cron_db.connect()
    try:
        srv, agent = await _bare_server(
            {"skills": {"enabled": True, "distiller_enabled": True,
                        "distiller_schedule": "30 2 * * 1"}}, cron_db)
        sched = Scheduler(cron_db, agent)
        await srv._sync_skill_distiller(sched)
        row = await _distiller_row(cron_db)
        assert row is not None and row["cron_expression"] == "30 2 * * 1", row
    finally:
        await cron_db.close()


# ── 2. distiller and curator are DISTINCT tasks (independent toggles) ─

@test("skill_distiller", "distiller and curator seed distinct rows on independent toggles")
async def t_distiller_curator_distinct(ctx: TestContext) -> None:
    from src.core.builtin_tasks import (
        SKILL_CURATOR_TASK_NAME,
        SKILL_DISTILLER_TASK_NAME,
    )
    from src.core.scheduler import Scheduler

    assert SKILL_DISTILLER_TASK_NAME != SKILL_CURATOR_TASK_NAME

    # distiller ON, curator OFF → only the distiller row exists.
    d_db = _fresh_db(ctx, "donly")
    await d_db.connect()
    try:
        srv, agent = await _bare_server(
            {"skills": {"enabled": True, "distiller_enabled": True}}, d_db)
        sched = Scheduler(d_db, agent)
        await srv._sync_skill_distiller(sched)
        await srv._sync_skill_curator(sched)
        assert await _distiller_row(d_db) is not None, "distiller not seeded"
        assert await _curator_row(d_db) is None, (
            "the curator was seeded by the distiller's toggle — the two halves "
            "must flip independently"
        )
    finally:
        await d_db.close()

    # curator ON, distiller OFF → only the curator row exists.
    c_db = _fresh_db(ctx, "conly")
    await c_db.connect()
    try:
        srv, agent = await _bare_server(
            {"skills": {"enabled": True, "curator_enabled": True}}, c_db)
        sched = Scheduler(c_db, agent)
        await srv._sync_skill_distiller(sched)
        await srv._sync_skill_curator(sched)
        assert await _curator_row(c_db) is not None, "curator not seeded"
        assert await _distiller_row(c_db) is None, (
            "the distiller was seeded by the curator's toggle"
        )
    finally:
        await c_db.close()

    # BOTH on → two distinct enabled rows.
    both_db = _fresh_db(ctx, "both")
    await both_db.connect()
    try:
        srv, agent = await _bare_server(
            {"skills": {"enabled": True, "curator_enabled": True,
                        "distiller_enabled": True}}, both_db)
        sched = Scheduler(both_db, agent)
        await srv._sync_skill_distiller(sched)
        await srv._sync_skill_curator(sched)
        d = await _distiller_row(both_db)
        c = await _curator_row(both_db)
        assert d is not None and c is not None, (d, c)
        assert d["id"] != c["id"], "distiller and curator collapsed into one row"
        assert d["enabled"] and c["enabled"]
        assert d["prompt"] != c["prompt"], "the two tasks share a prompt"
    finally:
        await both_db.close()


# ── 3. the distiller prompt encodes CREATE-only + the real signals ────

@test("skill_distiller", "SKILL_DISTILLER_PROMPT: creates (never merges), names its signals")
async def t_distiller_prompt(_ctx: TestContext) -> None:
    from src.core.server import (
        SKILL_CURATOR_PROMPT,
        SKILL_DISTILLER_DEFAULT_CRON,
        SKILL_DISTILLER_PROMPT,
    )

    p = SKILL_DISTILLER_PROMPT
    lower = p.lower()

    # It CREATES.
    assert 'skill_manage(action="create"' in p or "skill_manage" in p, (
        "the distiller prompt never calls skill_manage"
    )
    assert "create" in lower, "the distiller prompt does not say it creates"

    # It must NOT be told to consolidate — that is the curator's lane. The
    # prompt explicitly forbids the merge/archive/remove actions.
    for forbidden in ("archive", "remove", "update"):
        assert forbidden in lower, (
            f"the prompt should name {forbidden!r} to forbid it (layering)"
        )
    assert "curator" in lower, (
        "the distiller prompt never references the curator boundary"
    )

    # The real signals OpenAgent already has.
    assert "search_past_conversations" in p, "prompt omits the transcript-search signal"
    assert "vault_recall_stats" in p, "prompt omits the OUTCOME_OK ledger signal"
    assert "outcome_ok" in lower or "ok_rate" in lower or "outcome ledger" in lower, (
        "prompt omits the outcome-ledger framing"
    )
    # Overlap check before writing — don't duplicate an existing skill.
    assert "skill_search" in p, "prompt omits the overlap check via skill_search"
    assert "overlap" in lower or "already" in lower, (
        "prompt does not tell the distiller to skip existing coverage"
    )
    # Only NOVEL + RECURRING patterns become skills.
    assert "recurring" in lower, "prompt does not require the pattern to recur"

    # The two prompts are genuinely different passes.
    assert p != SKILL_CURATOR_PROMPT

    # Default cadence is daily (shorter than the weekly curator).
    assert SKILL_DISTILLER_DEFAULT_CRON == "53 3 * * *", SKILL_DISTILLER_DEFAULT_CRON


# ── 4. wiring: the task is a hidden built-in in the config-skills section ─

@test("skill_distiller", "skill-distiller is a built-in mapped to the skills config section")
async def t_distiller_is_builtin(_ctx: TestContext) -> None:
    import src.core.builtin_tasks as bt

    assert bt.SKILL_DISTILLER_TASK_NAME == "skill-distiller"
    assert bt.SKILL_DISTILLER_TASK_NAME in bt.BUILTIN_TASK_NAMES, (
        "the distiller must be a built-in so the gateway hides it from the "
        "user task list and rejects writes"
    )
    assert bt.CONFIG_SECTION_BY_TASK[bt.SKILL_DISTILLER_TASK_NAME] == "skills", (
        "the distiller toggle lives in the skills config section"
    )
