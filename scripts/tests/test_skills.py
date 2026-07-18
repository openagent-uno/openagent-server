"""Native Skills subsystem — registry, index render, MCP tools, gating.

Hermes / Claude-Code SKILL.md progressive disclosure: only a small INDEX
(category → ``name: description``) is injected into the CACHED system
prefix; full bodies load on demand via ``skill_view``. The subsystem is
OFF BY DEFAULT and must be strictly additive — with ``skills.enabled``
false the framework prompt and the registered MCP set stay byte-identical
to a build without it.

These tests pin the four load-bearing properties:

  * the registry scans + renders an index grouped by category,
  * the index is BYTE-STABLE (it feeds the prompt cache — no volatile
    tokens, identical across calls and across registry instances),
  * the three tools (view / search / manage) round-trip on disk,
  * the disabled path is inert (empty index render + no MCP registration),
  * and a malformed SKILL.md is skipped, never fatal.

Pure-unit: throwaway skills trees in a temp dir, no LLM / pool / gateway
(one test opens a throwaway ``MemoryDB`` to prove the bootstrap gate, in
the same spirit as test_bootstrap).
"""
from __future__ import annotations

import re
import shutil
import tempfile
import uuid
from pathlib import Path

from ._framework import TestContext, test


# ── helpers ───────────────────────────────────────────────────────────

def _mkskills() -> Path:
    return Path(tempfile.mkdtemp(prefix="skills-"))


def _write_skill(root: Path, folder: str, *, name: str | None,
                 description: str = "", category: str | None = None,
                 body: str = "body text", raw: str | None = None) -> None:
    d = root / folder
    d.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        (d / "SKILL.md").write_text(raw)
        return
    fm = ["---"]
    if name is not None:
        fm.append(f"name: {name}")
    if description:
        fm.append(f"description: {description}")
    if category:
        fm.append(f"category: {category}")
    fm.append("---")
    (d / "SKILL.md").write_text("\n".join(fm) + f"\n\n{body}\n")


# ── 1. registry load + index shape ────────────────────────────────────

@test("skills", "registry loads every SKILL.md; index groups by category")
async def t_registry_loads_and_indexes(_ctx: TestContext) -> None:
    from src.mcp.servers.skills.registry import SkillsRegistry

    root = _mkskills()
    try:
        _write_skill(root, "triage", name="support-triage",
                     description="Triage a support ticket", category="support")
        _write_skill(root, "refund", name="issue-refund",
                     description="Refund a customer", category="support")
        _write_skill(root, "deploy", name="deploy-web",
                     description="Ship the web app", category="ops")

        reg = SkillsRegistry(root)
        reg.load()
        names = sorted(s.name for s in reg.skills())
        assert names == ["deploy-web", "issue-refund", "support-triage"], names

        idx = reg.render_skills_index()
        # Every name + its description must be present, and the two support
        # skills must sit under one category header.
        for nm, desc in [
            ("support-triage", "Triage a support ticket"),
            ("issue-refund", "Refund a customer"),
            ("deploy-web", "Ship the web app"),
        ]:
            assert nm in idx, f"{nm!r} missing from index:\n{idx}"
            assert desc in idx, f"{desc!r} missing from index:\n{idx}"
        assert "support" in idx and "ops" in idx, idx
        # Category grouping: 'support' header appears once for two skills.
        assert idx.count("**support**") == 1, idx

        # A skill with no ``category`` falls into the default bucket.
        _write_skill(root, "loose", name="loose-skill",
                     description="no category given")
        reg.load()
        assert "**general**" in reg.render_skills_index()
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── 2. byte-stability (guards the prompt cache) ───────────────────────

@test("skills", "render_skills_index is byte-stable with no volatile tokens")
async def t_index_byte_stable(_ctx: TestContext) -> None:
    """The index lands ABOVE ``<session-id>`` in the cached system prefix,
    so it MUST be identical across sessions/turns on a box. If it carried a
    timestamp or a session id, every session would rewrite the ~10.8k prefix
    — a silent per-session cache-write regression with no test symptom but
    the bill. Pin: two calls agree, a fresh registry over the same dir
    agrees, and the bytes carry no date/time/session token."""
    from src.mcp.servers.skills.registry import SkillsRegistry

    root = _mkskills()
    try:
        _write_skill(root, "a", name="alpha", description="first skill",
                     category="cat-a")
        _write_skill(root, "b", name="beta", description="second skill",
                     category="cat-b")

        reg = SkillsRegistry(root)
        reg.load()
        first = reg.render_skills_index()
        second = reg.render_skills_index()
        assert first == second, "index differs between two calls on one registry"

        # A brand-new registry instance over the same tree must produce the
        # exact same bytes — proves determinism, not just caching.
        reg2 = SkillsRegistry(root)
        reg2.load()
        assert reg2.render_skills_index() == first, (
            "two registries over the same dir rendered different bytes — the "
            "index is not a deterministic function of disk contents"
        )

        # No volatile tokens: no ISO date, no clock time, no 'session'.
        assert not re.search(r"\d{4}-\d{2}-\d{2}", first), first
        assert not re.search(r"\d{2}:\d{2}:\d{2}", first), first
        assert "session" not in first.lower(), first
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ── 3. the three tools round-trip on disk ─────────────────────────────

@test("skills", "skill_view / skill_search / skill_manage round-trip")
async def t_tools_round_trip(_ctx: TestContext) -> None:
    import os
    from src.mcp.servers.skills import handlers
    from src.mcp.servers.skills.registry import SkillsRegistry

    root = _mkskills()
    prev = os.environ.get("OPENAGENT_SKILLS_PATH")
    os.environ["OPENAGENT_SKILLS_PATH"] = str(root)
    try:
        _write_skill(root, "triage", name="support-triage",
                     description="Triage a ticket", category="support",
                     body="Step 1. Read the ticket carefully.")

        # skill_view returns the FULL body.
        v = await handlers.skill_view("support-triage")
        assert v["ok"] and "Read the ticket carefully" in v["body"], v
        assert v["category"] == "support"

        # Unknown name is a clean miss, not a crash.
        miss = await handlers.skill_view("does-not-exist")
        assert miss["ok"] is False and "support-triage" in miss["available"], miss

        # skill_search finds by name/description/body.
        by_body = await handlers.skill_search("carefully")
        assert by_body["count"] == 1 and by_body["results"][0]["name"] == "support-triage"
        assert "body" in by_body["results"][0]["matched_in"]
        by_desc = await handlers.skill_search("triage")
        assert by_desc["count"] == 1, by_desc

        # skill_manage: create → update → remove, verified on disk each step.
        created = await handlers.skill_manage(
            "create", "escalation", body="Page the on-call.",
            description="Escalate to a human", category="support")
        assert created["ok"], created
        reg = SkillsRegistry(root); reg.load()
        assert reg.get("escalation") is not None, "create did not land on disk"

        # create is idempotency-guarded: a second create is refused.
        dup = await handlers.skill_manage("create", "escalation", body="x")
        assert dup["ok"] is False, dup

        updated = await handlers.skill_manage(
            "update", "escalation", body="Page the on-call IMMEDIATELY.",
            description="Escalate fast", category="support")
        assert updated["ok"], updated
        v2 = await handlers.skill_view("escalation")
        assert "IMMEDIATELY" in v2["body"] and v2["description"] == "Escalate fast", v2

        removed = await handlers.skill_manage("remove", "escalation")
        assert removed["ok"], removed
        reg.load()
        assert reg.get("escalation") is None, "remove left the skill on disk"
    finally:
        if prev is None:
            os.environ.pop("OPENAGENT_SKILLS_PATH", None)
        else:
            os.environ["OPENAGENT_SKILLS_PATH"] = prev
        shutil.rmtree(root, ignore_errors=True)


# ── 4. gating / off-by-default (the whole point) ──────────────────────

@test("skills", "disabled by default: empty index render AND no MCP registration")
async def t_disabled_path_is_inert(ctx: TestContext) -> None:
    from src.core.config import skills_settings
    from src.core.prompts import FRAMEWORK_SYSTEM_PROMPT, build_skills_index
    from src.mcp.builtins import (
        BUILTIN_MCP_SPECS, DEFAULT_MCPS, config_gated_mcp_entries,
    )

    # Default OFF, opt-in ON.
    assert skills_settings({}).enabled is False
    assert skills_settings(None).enabled is False
    assert skills_settings({"skills": {"enabled": True}}).enabled is True

    # Disabled → the placeholder renders to "" and the framework prompt is
    # BYTE-IDENTICAL to a build without the feature (the placeholder sits
    # flush against the next header, so an empty render leaves no gap).
    assert build_skills_index(None) == ""
    assert "{{SKILLS_INDEX}}" in FRAMEWORK_SYSTEM_PROMPT
    disabled = FRAMEWORK_SYSTEM_PROMPT.replace(
        "{{SKILLS_INDEX}}", build_skills_index(None))
    assert "{{SKILLS_INDEX}}" not in disabled
    assert "\n\n## Builtin management MCPs (canonical paths)" in disabled
    assert "## Skills\n" not in disabled, (
        "the Skills section leaked into the DISABLED prompt — not byte-identical"
    )

    # The skills MCP is NEVER a default, and is only gated in when enabled.
    default_builtins = [e.get("builtin") for e in DEFAULT_MCPS]
    assert "skills" not in default_builtins, (
        "skills must NOT be in DEFAULT_MCPS — it would register unconditionally"
    )
    assert config_gated_mcp_entries({}) == []
    assert config_gated_mcp_entries(None) == []
    enabled_entries = config_gated_mcp_entries({"skills": {"enabled": True}})
    assert [e.get("builtin") for e in enabled_entries] == ["skills"], enabled_entries
    # But the SPEC exists so it is registerable once enabled.
    assert "skills" in BUILTIN_MCP_SPECS and BUILTIN_MCP_SPECS["skills"]["in_process"]

    # Bootstrap gate end-to-end: a fresh DB seeds NO skills row when disabled,
    # and exactly one when enabled.
    from src.memory.bootstrap import ensure_builtin_mcps
    from src.memory.db import MemoryDB

    off_db = MemoryDB(str(ctx.db_path.with_name(f"skills-off-{uuid.uuid4().hex[:8]}.db")))
    await off_db.connect()
    try:
        await ensure_builtin_mcps(off_db, config={})
        assert "skills" not in {r["name"] for r in await off_db.list_mcps()}
    finally:
        await off_db.close()

    on_db = MemoryDB(str(ctx.db_path.with_name(f"skills-on-{uuid.uuid4().hex[:8]}.db")))
    await on_db.connect()
    try:
        await ensure_builtin_mcps(on_db, config={"skills": {"enabled": True}})
        rows = {r["name"]: r for r in await on_db.list_mcps()}
        assert "skills" in rows, "enabled skills was not seeded"
        assert rows["skills"]["builtin_name"] == "skills"
        assert rows["skills"]["enabled"] is True
    finally:
        await on_db.close()


# ── 5. malformed SKILL.md is skipped, never fatal ─────────────────────

@test("skills", "malformed / nameless SKILL.md is skipped gracefully")
async def t_malformed_is_skipped(_ctx: TestContext) -> None:
    from src.mcp.servers.skills.registry import SkillsRegistry

    root = _mkskills()
    try:
        # One good skill.
        _write_skill(root, "good", name="good-skill",
                     description="works fine", category="ok")
        # Malformed YAML frontmatter.
        _write_skill(root, "badyaml", name=None,
                     raw="---\nname: [unterminated\ndescription: x\n---\n\nbody\n")
        # Frontmatter present but no ``name``.
        _write_skill(root, "noname", name=None,
                     raw="---\ndescription: nameless\n---\n\nbody\n")
        # No frontmatter at all.
        _write_skill(root, "plain", name=None, raw="just a markdown body\n")
        # Empty name.
        _write_skill(root, "emptyname", name=None,
                     raw="---\nname: \ndescription: blank\n---\n\nbody\n")

        reg = SkillsRegistry(root)
        reg.load()  # must not raise
        names = [s.name for s in reg.skills()]
        assert names == ["good-skill"], (
            f"only the well-formed skill should load, got {names}"
        )
        # The index still renders and still contains the good skill.
        idx = reg.render_skills_index()
        assert "good-skill" in idx and "works fine" in idx
    finally:
        shutil.rmtree(root, ignore_errors=True)
