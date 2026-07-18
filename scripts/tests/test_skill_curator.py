"""Skill-curator — the self-improving "dream mode for skills".

Four load-bearing properties, all LLM-free (the consolidation itself is
LLM-driven and deferred, exactly like the dream-mode live tests):

  * **Provenance stamping** — ``skill_manage(create)`` stamps
    ``created_by: agent`` in the SKILL.md frontmatter; the registry's
    ``agent_authored`` filter returns only those, and a hand-written seed
    skill (no stamp) is excluded.
  * **Curator gating (OFF by default)** — with ``skills.curator_enabled``
    false the ``skill-curator`` scheduled task is NOT seeded; with it true
    (and ``skills.enabled``) it IS. Asserted through the real seeding path
    (``AgentServer._sync_skill_curator``), no live run.
  * **The provenance boundary** — the set of skills the curator may touch
    excludes seed/user skills, so a future consolidation can never clobber
    curated seed content.
  * **Archived skills leave the index** — an archived skill is dropped from
    ``render_skills_index`` so the frozen prompt stays clean.

Pure-unit: throwaway skills trees in a tmpdir + a throwaway ``MemoryDB`` for
the seeding gate (same shape as test_skills / test_bootstrap). No LLM, pool,
or gateway.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path

from ._framework import TestContext, test


# ── helpers ───────────────────────────────────────────────────────────

def _mkskills() -> Path:
    return Path(tempfile.mkdtemp(prefix="skillcurator-"))


def _write_raw_skill(root: Path, folder: str, raw: str) -> None:
    d = root / folder
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(raw)


def _seed_skill(root: Path, folder: str, name: str, *, category: str = "support",
                description: str = "a seed skill") -> None:
    """A hand-written seed skill — NO ``created_by`` stamp (off-limits)."""
    _write_raw_skill(
        root, folder,
        f"---\nname: {name}\ndescription: {description}\ncategory: {category}\n"
        f"---\n\nSeed body.\n",
    )


# ── 1. provenance stamping + agent_authored filter ────────────────────

@test("skill_curator", "create stamps created_by:agent; filter excludes seed skills")
async def t_provenance_stamp_and_filter(_ctx: TestContext) -> None:
    from src.mcp.servers.skills import handlers
    from src.mcp.servers.skills.registry import SkillsRegistry

    root = _mkskills()
    prev = os.environ.get("OPENAGENT_SKILLS_PATH")
    os.environ["OPENAGENT_SKILLS_PATH"] = str(root)
    try:
        # A hand-written seed skill (no provenance).
        _seed_skill(root, "triage", "support-triage")

        # The agent authors one via skill_manage.
        created = await handlers.skill_manage(
            "create", "escalation-playbook", body="Page the on-call.",
            description="Escalate to a human", category="support")
        assert created["ok"], created

        reg = SkillsRegistry(root)
        reg.load()

        # The created skill carries the stamp; the seed one does not.
        agent_skill = reg.get("escalation-playbook")
        seed_skill = reg.get("support-triage")
        assert agent_skill is not None and agent_skill.is_agent_authored, agent_skill
        assert seed_skill is not None and not seed_skill.is_agent_authored, seed_skill
        assert agent_skill.created_by == "agent", agent_skill.created_by

        # The frontmatter really contains the stamp on disk (not just parsed).
        raw = agent_skill.path.read_text()
        assert "created_by: agent" in raw, raw

        # agent_authored() returns ONLY the agent skill.
        authored = {s.name for s in reg.agent_authored()}
        assert authored == {"escalation-playbook"}, authored

        # An UPDATE preserves the provenance stamp (does not strip it).
        upd = await handlers.skill_manage(
            "update", "escalation-playbook", body="Page the on-call NOW.",
            description="Escalate fast", category="support")
        assert upd["ok"], upd
        reg.load()
        again = reg.get("escalation-playbook")
        assert again is not None and again.is_agent_authored, (
            "update stripped the created_by stamp — the skill fell out of the "
            "curator's reach"
        )

        # And an update to a SEED skill does NOT forge the agent stamp onto it.
        upd_seed = await handlers.skill_manage(
            "update", "support-triage", body="Edited by request.",
            description="a seed skill", category="support")
        assert upd_seed["ok"], upd_seed
        reg.load()
        seed_again = reg.get("support-triage")
        assert seed_again is not None and not seed_again.is_agent_authored, (
            "editing a seed skill mislabelled it as agent-authored — the "
            "curator could now consolidate it"
        )
    finally:
        if prev is None:
            os.environ.pop("OPENAGENT_SKILLS_PATH", None)
        else:
            os.environ["OPENAGENT_SKILLS_PATH"] = prev
        shutil.rmtree(root, ignore_errors=True)


# ── 2. curator gating — OFF by default, seeded only when opted in ─────

async def _bare_server(config: dict, db):
    """A minimally-constructed AgentServer wired just enough to drive
    ``_sync_skill_curator``: it reads ``self.config`` and ``self.agent._db``.
    ``__new__`` skips the heavy ``__init__`` (no pool, gateway, or model)."""
    from src.core.server import AgentServer

    srv = AgentServer.__new__(AgentServer)
    srv.config = config

    class _Agent:
        pass

    agent = _Agent()
    agent._db = db
    srv.agent = agent
    return srv, agent


async def _curator_row(db):
    from src.core.builtin_tasks import SKILL_CURATOR_TASK_NAME
    tasks = await db.get_tasks()
    return next((t for t in tasks if t["name"] == SKILL_CURATOR_TASK_NAME), None)


@test("skill_curator", "OFF by default: no scheduled task seeded; ON when opted in")
async def t_curator_gating(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler
    from src.memory.db import MemoryDB

    # (a) DEFAULT config → nothing seeded (byte-identical to no feature).
    off_db = MemoryDB(str(ctx.db_path.with_name(f"curator-off-{uuid.uuid4().hex[:8]}.db")))
    await off_db.connect()
    try:
        srv, agent = await _bare_server({}, off_db)
        sched = Scheduler(off_db, agent)
        await srv._sync_skill_curator(sched)
        assert await _curator_row(off_db) is None, (
            "the curator task was seeded with default config — it must be "
            "invisible, not merely disabled"
        )
    finally:
        await off_db.close()

    # (b) skills.enabled but curator_enabled still false → STILL nothing.
    half_db = MemoryDB(str(ctx.db_path.with_name(f"curator-half-{uuid.uuid4().hex[:8]}.db")))
    await half_db.connect()
    try:
        srv, agent = await _bare_server({"skills": {"enabled": True}}, half_db)
        sched = Scheduler(half_db, agent)
        await srv._sync_skill_curator(sched)
        assert await _curator_row(half_db) is None, (
            "skills.enabled alone seeded the curator — it needs the second "
            "curator_enabled gate too"
        )
    finally:
        await half_db.close()

    # (c) BOTH gates on → seeded AND enabled, on the configured/default cron.
    on_db = MemoryDB(str(ctx.db_path.with_name(f"curator-on-{uuid.uuid4().hex[:8]}.db")))
    await on_db.connect()
    try:
        srv, agent = await _bare_server(
            {"skills": {"enabled": True, "curator_enabled": True}}, on_db)
        sched = Scheduler(on_db, agent)
        await srv._sync_skill_curator(sched)
        row = await _curator_row(on_db)
        assert row is not None, "both gates on but the curator was not seeded"
        assert row["enabled"], "the curator was seeded but left disabled"
        assert row["prompt"], "the curator task has no prompt"

        # (d) Turning it back off at runtime disables the surviving row.
        srv.config = {"skills": {"enabled": True, "curator_enabled": False}}
        await srv._sync_skill_curator(sched)
        row2 = await _curator_row(on_db)
        assert row2 is not None, "the row was deleted, not disabled"
        assert not row2["enabled"], "runtime-off did not disable the curator row"
    finally:
        await on_db.close()


# ── 3. the provenance boundary — the curator's work set ───────────────

@test("skill_curator", "the curator's work set excludes seed and user skills")
async def t_provenance_boundary(_ctx: TestContext) -> None:
    """The single guard that stops a consolidation pass from clobbering
    curated seed content: the curator discovers its work through
    ``agent_authored()``, and that filter must never admit a skill lacking
    the ``created_by: agent`` stamp — no matter its category or contents."""
    from src.mcp.servers.skills.registry import SkillsRegistry

    root = _mkskills()
    try:
        # Two agent skills, two off-limits skills (a seed and a user one).
        _write_raw_skill(
            root, "agent-a",
            "---\nname: agent-a\ndescription: d\ncategory: support\n"
            "created_by: agent\n---\n\nbody\n")
        _write_raw_skill(
            root, "agent-b",
            "---\nname: agent-b\ndescription: d\ncategory: ops\n"
            "created_by: agent\n---\n\nbody\n")
        _seed_skill(root, "seed", "seed-skill")  # no stamp
        _write_raw_skill(
            root, "user",
            "---\nname: user-skill\ndescription: d\ncategory: support\n"
            "created_by: alice\n---\n\nbody\n")  # a human, not the agent

        reg = SkillsRegistry(root)
        reg.load()

        touchable = {s.name for s in reg.agent_authored()}
        assert touchable == {"agent-a", "agent-b"}, touchable
        # Explicitly: the off-limits skills are NOT in the set.
        assert "seed-skill" not in touchable
        assert "user-skill" not in touchable, (
            "a skill authored by a human (created_by != agent) slipped into "
            "the curator's work set"
        )
        # All four still exist in the registry — the filter narrows, it does
        # not hide.
        assert {s.name for s in reg.skills()} == {
            "agent-a", "agent-b", "seed-skill", "user-skill"}
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── 4. archived skills stay out of the frozen index ───────────────────

@test("skill_curator", "an archived skill is dropped from render_skills_index")
async def t_archived_excluded_from_index(_ctx: TestContext) -> None:
    from src.mcp.servers.skills import handlers
    from src.mcp.servers.skills.registry import SkillsRegistry

    root = _mkskills()
    prev = os.environ.get("OPENAGENT_SKILLS_PATH")
    os.environ["OPENAGENT_SKILLS_PATH"] = str(root)
    try:
        # Two agent skills; one will be archived via skill_manage.
        await handlers.skill_manage(
            "create", "keeper", body="Keep me.",
            description="a live skill", category="support")
        await handlers.skill_manage(
            "create", "retiree", body="Retire me.",
            description="a stale skill", category="support")

        reg = SkillsRegistry(root)
        reg.load()
        idx_before = reg.render_skills_index()
        assert "keeper" in idx_before and "retiree" in idx_before, idx_before

        # Archive one. The file must remain on disk (reversible), but its name
        # must leave the index.
        arch = await handlers.skill_manage("archive", "retiree")
        assert arch["ok"] and arch["status"] == "archived", arch

        reg.load()  # rescan
        retiree = reg.get("retiree")
        assert retiree is not None and retiree.is_archived, (
            "archive deleted the skill or failed to set status — it must "
            "retire in place"
        )
        assert (root / "retiree" / "SKILL.md").is_file(), "archive hard-deleted the file"

        idx_after = reg.render_skills_index()
        assert "keeper" in idx_after, idx_after
        assert "retiree" not in idx_after, (
            "an archived skill is still advertised in the frozen prompt index"
        )
    finally:
        if prev is None:
            os.environ.pop("OPENAGENT_SKILLS_PATH", None)
        else:
            os.environ["OPENAGENT_SKILLS_PATH"] = prev
        shutil.rmtree(root, ignore_errors=True)


# ── 5. the curator prompt encodes the provenance boundary ─────────────

@test("skill_curator", "SKILL_CURATOR_PROMPT names the provenance boundary and tools")
async def t_curator_prompt(_ctx: TestContext) -> None:
    from src.core.server import SKILL_CURATOR_DEFAULT_CRON, SKILL_CURATOR_PROMPT

    lower = SKILL_CURATOR_PROMPT.lower()
    # It must use the skill tools, not shell out.
    for tool in ("skill_view", "skill_search", "skill_manage"):
        assert tool in SKILL_CURATOR_PROMPT, f"the curator prompt never calls {tool}"
    # The whole point: the provenance boundary is explicit.
    assert "created_by: agent" in SKILL_CURATOR_PROMPT, (
        "the curator prompt never states the created_by:agent boundary"
    )
    assert "off-limits" in lower, "the prompt does not mark seed/user skills off-limits"
    # It archives rather than hard-deletes by default (reversible).
    assert "archive" in lower
    # Default cadence is weekly (Sunday).
    assert SKILL_CURATOR_DEFAULT_CRON == "0 4 * * 0", SKILL_CURATOR_DEFAULT_CRON
