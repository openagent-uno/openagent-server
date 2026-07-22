"""Intrinsic self-improvement loop + the per-LLM-call anti-wedge timeout.

Two features, both LLM-free (the grading/synthesis itself is LLM-driven and
deferred, exactly like the dream-mode / skill-curator live tests):

  * **Self-improvement (Feature A)** — the ``quality-scorer`` (every-2h grader)
    and ``quality-digest`` (daily synthesis) are INTRINSIC built-in scheduled
    tasks: ON by default, no per-agent config. Asserted through the real
    seeding path (``AgentServer._sync_quality_scorer`` /
    ``_sync_quality_digest``), no live run. The load-bearing property beyond
    "it seeds" is the DEDUP: an agent that already ships its own tuned,
    NON-builtin quality-scorer / quality-digest keeps it — the builtin defers
    rather than double-running (so eSound/Lyra don't get two).

  * **Anti-wedge timeout (Feature B)** — ``NativeProvider._construct_model``
    gives every model build a per-read socket timeout (default 90s, under the
    120s lease TTL) so a hung provider connection raises instead of wedging the
    worker, and ``_build_agent`` wires ``model.timeout_seconds`` → the env
    override.

Pure-unit: throwaway ``MemoryDB`` for the seeding gates (same shape as
test_skill_curator / test_skill_distiller) and tiny stub model classes for the
constructor. No LLM, pool, or gateway.
"""
from __future__ import annotations

import os
import uuid

from ._framework import TestContext, test


# ── helpers (mirror test_skill_distiller) ─────────────────────────────

async def _bare_server(config: dict, db):
    """A minimally-constructed AgentServer wired just enough to drive
    ``_sync_quality_scorer`` / ``_sync_quality_digest``: it reads
    ``self.config`` and ``self.agent._db``. ``__new__`` skips the heavy
    ``__init__`` (no pool, gateway, or model)."""
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


async def _scorer_row(db):
    from src.core.builtin_tasks import QUALITY_SCORER_TASK_NAME
    return await _row(db, QUALITY_SCORER_TASK_NAME)


async def _digest_row(db):
    from src.core.builtin_tasks import QUALITY_DIGEST_TASK_NAME
    return await _row(db, QUALITY_DIGEST_TASK_NAME)


async def _cost_row(db):
    from src.core.builtin_tasks import COST_OBSERVABILITY_TASK_NAME
    return await _row(db, COST_OBSERVABILITY_TASK_NAME)


async def _audit_row(db):
    from src.core.builtin_tasks import ESCALATION_AUDIT_TASK_NAME
    return await _row(db, ESCALATION_AUDIT_TASK_NAME)


def _fresh_db(ctx: TestContext, tag: str):
    from src.memory.db import MemoryDB
    return MemoryDB(str(ctx.db_path.with_name(f"selfimp-{tag}-{uuid.uuid4().hex[:8]}.db")))


def _set_env(key: str, value):
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


# ── 1. config settings — ON by default, overrides parse ───────────────

@test("self_improvement", "settings are ON by default and overrides parse")
async def t_settings_defaults(_ctx: TestContext) -> None:
    from src.core.config import SelfImprovementSettings, self_improvement_settings

    # Empty / missing stanza → the intrinsic ON default.
    d = self_improvement_settings({})
    assert d == SelfImprovementSettings(), d
    assert d.enabled and d.scorer_enabled and d.digest_enabled, (
        "the intrinsic quality loop must default ON — that is the whole point"
    )
    assert d.scorer_schedule is None and d.digest_schedule is None

    # A non-dict stanza is defensive, not fatal.
    assert self_improvement_settings({"self_improvement": "nope"}) == SelfImprovementSettings()

    # The CONSUMPTION arm (cost-observability) is part of the same intrinsic
    # loop — ON by default, its own independent gate + schedule.
    assert d.cost_observability_enabled, (
        "cost-observability must default ON — cache-aware cost monitoring is "
        "intrinsic, like the quality halves"
    )
    assert d.cost_observability_schedule is None
    # The HANDOFF arm (escalation-audit) is part of the same intrinsic loop.
    assert d.escalation_audit_enabled, (
        "escalation-audit must default ON — auditing your own handoffs is "
        "intrinsic, like the quality/cost arms"
    )
    assert d.escalation_audit_schedule is None

    # Explicit overrides parse, including the per-task gates and schedules.
    o = self_improvement_settings({
        "self_improvement": {
            "enabled": False,
            "scorer_enabled": False,
            "digest_enabled": True,
            "cost_observability_enabled": False,
            "escalation_audit_enabled": False,
            "scorer_schedule": "*/30 * * * *",
            "digest_schedule": "0 8 * * *",
            "cost_observability_schedule": "*/15 * * * *",
            "escalation_audit_schedule": "0 7 * * *",
        }
    })
    assert not o.enabled and not o.scorer_enabled and o.digest_enabled
    assert not o.cost_observability_enabled
    assert not o.escalation_audit_enabled
    assert o.scorer_schedule == "*/30 * * * *"
    assert o.digest_schedule == "0 8 * * *"
    assert o.cost_observability_schedule == "*/15 * * * *"
    assert o.escalation_audit_schedule == "0 7 * * *"


# ── 2. gating — ON by default, parked when disabled ───────────────────

@test("self_improvement", "ON by default: both tasks seeded+enabled; toggles park them")
async def t_gating(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler

    # (a) DEFAULT config → BOTH intrinsic tasks seeded AND enabled. This is the
    #     opposite of the skill builtins (OFF-by-default): intrinsic means on.
    on_db = _fresh_db(ctx, "on")
    await on_db.connect()
    try:
        srv, agent = await _bare_server({}, on_db)
        sched = Scheduler(on_db, agent)
        await srv._sync_quality_scorer(sched)
        await srv._sync_quality_digest(sched)
        s_row = await _scorer_row(on_db)
        d_row = await _digest_row(on_db)
        assert s_row is not None and s_row["enabled"], (
            "the quality-scorer must seed AND enable with default config — it is "
            "intrinsic, not opt-in"
        )
        assert d_row is not None and d_row["enabled"], "quality-digest not enabled by default"
        assert s_row["prompt"] and d_row["prompt"]
        assert s_row["cron_expression"] == "0 */2 * * *", s_row["cron_expression"]
        assert d_row["cron_expression"] == "0 9 * * *", d_row["cron_expression"]
    finally:
        await on_db.close()

    # (b) master switch off → BOTH parked disabled (row kept, not deleted).
    off_db = _fresh_db(ctx, "off")
    await off_db.connect()
    try:
        srv, agent = await _bare_server({}, off_db)
        sched = Scheduler(off_db, agent)
        await srv._sync_quality_scorer(sched)
        await srv._sync_quality_digest(sched)
        # Now flip the master switch off at runtime.
        srv.config = {"self_improvement": {"enabled": False}}
        await srv._sync_quality_scorer(sched)
        await srv._sync_quality_digest(sched)
        s_row = await _scorer_row(off_db)
        d_row = await _digest_row(off_db)
        assert s_row is not None and not s_row["enabled"], "scorer not parked on disable"
        assert d_row is not None and not d_row["enabled"], "digest not parked on disable"
    finally:
        await off_db.close()

    # (c) per-task gate: scorer off, digest on — independent halves.
    half_db = _fresh_db(ctx, "half")
    await half_db.connect()
    try:
        srv, agent = await _bare_server(
            {"self_improvement": {"scorer_enabled": False}}, half_db)
        sched = Scheduler(half_db, agent)
        await srv._sync_quality_scorer(sched)
        await srv._sync_quality_digest(sched)
        s_row = await _scorer_row(half_db)
        d_row = await _digest_row(half_db)
        assert s_row is None or not s_row["enabled"], (
            "scorer_enabled:false must not enable the scorer"
        )
        assert d_row is not None and d_row["enabled"], (
            "digest must still run when only the scorer is gated off"
        )
    finally:
        await half_db.close()


# ── 3. DEDUP — defer to an agent's own tuned custom task ───────────────

@test("self_improvement", "DEDUP: a custom non-builtin scorer/digest suppresses the builtin")
async def t_dedup(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler
    from src.core.builtin_tasks import QUALITY_SCORER_TASK_NAME, QUALITY_DIGEST_TASK_NAME

    # (a) A tuned custom task already exists (eSound/Lyra shape) → the builtin
    #     is NOT seeded; the agent keeps its own.
    db = _fresh_db(ctx, "dedup")
    await db.connect()
    try:
        srv, agent = await _bare_server({}, db)
        sched = Scheduler(db, agent)
        # Seed the agent's own tuned tasks (NON-builtin names).
        await sched.add_task("response-quality-scorer", "0 */2 * * *", "custom scorer prompt")
        await sched.add_task("quality-improvement-digest", "0 9 * * *", "custom digest prompt")

        await srv._sync_quality_scorer(sched)
        await srv._sync_quality_digest(sched)

        assert await _scorer_row(db) is None, (
            "the builtin quality-scorer was seeded despite a tuned custom one — "
            "eSound/Lyra would double-run their grader"
        )
        assert await _digest_row(db) is None, (
            "the builtin quality-digest was seeded despite a tuned custom one"
        )
        # The custom tasks are untouched.
        assert await _row(db, "response-quality-scorer") is not None
        assert await _row(db, "quality-improvement-digest") is not None
    finally:
        await db.close()

    # (b) A builtin row that pre-dates the custom one is PARKED when the custom
    #     appears (defer, don't keep firing two).
    db2 = _fresh_db(ctx, "dedup-park")
    await db2.connect()
    try:
        srv, agent = await _bare_server({}, db2)
        sched = Scheduler(db2, agent)
        # First boot: no custom task → builtin seeds+enables.
        await srv._sync_quality_scorer(sched)
        assert (await _scorer_row(db2))["enabled"], "builtin scorer should be enabled first"
        # A custom scorer appears; re-sync must park the builtin.
        await sched.add_task("live-reply-quality-score", "0 */2 * * *", "tuned")
        await srv._sync_quality_scorer(sched)
        row = await _scorer_row(db2)
        assert row is not None and not row["enabled"], (
            "the builtin scorer kept firing alongside a newly-added custom one"
        )
    finally:
        await db2.close()


@test("self_improvement", "dedup name-matchers: correct scope, no false positives")
async def t_dedup_name_matcher(_ctx: TestContext) -> None:
    from src.core.server import (
        _is_custom_quality_scorer_name, _is_custom_quality_digest_name,
    )

    # Scorer purpose.
    assert _is_custom_quality_scorer_name("response-quality-scorer")
    assert _is_custom_quality_scorer_name("Quality Score Pass")
    assert not _is_custom_quality_scorer_name("quality-improvement-digest")
    # A quality task that is NOT a scorer must not be swallowed.
    assert not _is_custom_quality_scorer_name("clickup-task-quality-audit")
    assert not _is_custom_quality_scorer_name("dream-mode")

    # Digest purpose.
    assert _is_custom_quality_digest_name("quality-improvement-digest")
    assert _is_custom_quality_digest_name("weekly-quality-digest")
    assert not _is_custom_quality_digest_name("response-quality-scorer")
    assert not _is_custom_quality_digest_name("auto-update")


# ── 4. prompts encode the intended discipline ─────────────────────────

@test("self_improvement", "scorer/digest prompts encode the steps, dimensions, and cadence")
async def t_prompts(_ctx: TestContext) -> None:
    from src.core.server import (
        QUALITY_SCORER_DEFAULT_CRON, QUALITY_SCORER_PROMPT,
        QUALITY_DIGEST_DEFAULT_CRON, QUALITY_DIGEST_PROMPT,
    )

    s = QUALITY_SCORER_PROMPT
    s_low = s.lower()
    # Role-agnostic: it must grade whatever the agent produced (replies OR ops).
    assert "role-agnostic" in s_low or "ops" in s_low, s_low[:200]
    # Grades against the agent's OWN vault rules FIRST (grounding).
    assert "vault" in s_low and ("own rules" in s_low or "own vault" in s_low)
    # The five grading dimensions.
    for dim in ("correctness", "grounding", "tone", "completeness"):
        assert dim in s_low, f"scorer prompt missing dimension {dim!r}"
    assert "follows-own-rules" in s_low
    # Writes a grounded correction for the weak ones.
    assert "correction" in s_low
    # Anti-fabrication + quiet-unless-regression discipline.
    assert "fabricat" in s_low
    assert "regression" in s_low
    # Verdict scale.
    for v in ("GOOD", "OK", "BAD"):
        assert v in s, f"scorer prompt missing verdict {v!r}"
    assert QUALITY_SCORER_DEFAULT_CRON == "0 */2 * * *", QUALITY_SCORER_DEFAULT_CRON

    d = QUALITY_DIGEST_PROMPT
    d_low = d.lower()
    # Auto-apply only SAFE grounded rule/doc updates; PROPOSE anything risky.
    assert "auto-apply" in d_low and "propose" in d_low
    assert "safe" in d_low and "risky" in d_low
    # One short recap; a quiet day = one line.
    assert "recap" in d_low and ("one line" in d_low or "single line" in d_low)
    # Never ship a release autonomously.
    assert "autonomous" in d_low or "release" in d_low
    assert QUALITY_DIGEST_DEFAULT_CRON == "0 9 * * *", QUALITY_DIGEST_DEFAULT_CRON


# ── 5. both are built-ins mapped to the self_improvement section ───────

@test("self_improvement", "quality-scorer/digest are built-ins in the self_improvement section")
async def t_is_builtin(_ctx: TestContext) -> None:
    from src.core.builtin_tasks import (
        BUILTIN_TASK_NAMES, CONFIG_SECTION_BY_TASK,
        QUALITY_SCORER_TASK_NAME, QUALITY_DIGEST_TASK_NAME,
        COST_OBSERVABILITY_TASK_NAME, ESCALATION_AUDIT_TASK_NAME,
    )

    assert QUALITY_SCORER_TASK_NAME == "quality-scorer"
    assert QUALITY_DIGEST_TASK_NAME == "quality-digest"
    assert COST_OBSERVABILITY_TASK_NAME == "cost-observability"
    assert ESCALATION_AUDIT_TASK_NAME == "escalation-audit"
    for name in (
        QUALITY_SCORER_TASK_NAME, QUALITY_DIGEST_TASK_NAME,
        COST_OBSERVABILITY_TASK_NAME, ESCALATION_AUDIT_TASK_NAME,
    ):
        assert name in BUILTIN_TASK_NAMES, f"{name} not registered as a built-in"
        assert CONFIG_SECTION_BY_TASK[name] == "self_improvement", (
            f"{name} not mapped to the self_improvement config section"
        )


# ── 5b. cost-observability — the CACHE-AWARE consumption arm ──────────

@test("self_improvement", "cost-observability: ON by default; its own gate parks it")
async def t_cost_gating(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler

    # (a) DEFAULT config → the cost watcher seeds AND enables, hourly.
    on_db = _fresh_db(ctx, "cost-on")
    await on_db.connect()
    try:
        srv, agent = await _bare_server({}, on_db)
        sched = Scheduler(on_db, agent)
        await srv._sync_cost_observability(sched)
        row = await _cost_row(on_db)
        assert row is not None and row["enabled"], (
            "cost-observability must seed AND enable by default — cache-aware "
            "cost monitoring is intrinsic, not opt-in"
        )
        assert row["cron_expression"] == "0 * * * *", row["cron_expression"]
        assert row["prompt"]
    finally:
        await on_db.close()

    # (b) its OWN gate off (master still on) → parked, independent of quality.
    gate_db = _fresh_db(ctx, "cost-gate")
    await gate_db.connect()
    try:
        srv, agent = await _bare_server(
            {"self_improvement": {"cost_observability_enabled": False}}, gate_db)
        sched = Scheduler(gate_db, agent)
        await srv._sync_cost_observability(sched)
        row = await _cost_row(gate_db)
        assert row is None or not row["enabled"], (
            "cost_observability_enabled:false must not enable the watcher"
        )
    finally:
        await gate_db.close()

    # (c) master switch off → parked too.
    off_db = _fresh_db(ctx, "cost-off")
    await off_db.connect()
    try:
        srv, agent = await _bare_server({}, off_db)
        sched = Scheduler(off_db, agent)
        await srv._sync_cost_observability(sched)
        srv.config = {"self_improvement": {"enabled": False}}
        await srv._sync_cost_observability(sched)
        row = await _cost_row(off_db)
        assert row is not None and not row["enabled"], "cost watcher not parked on master disable"
    finally:
        await off_db.close()


@test("self_improvement", "cost-observability DEDUP: a custom cost watcher suppresses the builtin")
async def t_cost_dedup(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler

    # (a) eSound/Lyra ship `*-cost-observability` → the builtin must NOT seed.
    db = _fresh_db(ctx, "cost-dedup")
    await db.connect()
    try:
        srv, agent = await _bare_server({}, db)
        sched = Scheduler(db, agent)
        await sched.add_task("esound-cost-observability", "0 * * * *", "custom cost prompt")
        await srv._sync_cost_observability(sched)
        assert await _cost_row(db) is None, (
            "the builtin cost-observability was seeded despite a tuned custom "
            "one — eSound/Lyra would double-run their cost watcher"
        )
        assert await _row(db, "esound-cost-observability") is not None
    finally:
        await db.close()

    # (b) a builtin that pre-dates the custom one is PARKED when it appears.
    db2 = _fresh_db(ctx, "cost-dedup-park")
    await db2.connect()
    try:
        srv, agent = await _bare_server({}, db2)
        sched = Scheduler(db2, agent)
        await srv._sync_cost_observability(sched)
        assert (await _cost_row(db2))["enabled"], "builtin cost watcher should enable first"
        await sched.add_task("lyra-cost-observability", "0 * * * *", "tuned")
        await srv._sync_cost_observability(sched)
        row = await _cost_row(db2)
        assert row is not None and not row["enabled"], (
            "the builtin cost watcher kept firing alongside a newly-added custom one"
        )
    finally:
        await db2.close()


@test("self_improvement", "cost dedup name-matcher: correct scope, no false positives")
async def t_cost_dedup_name_matcher(_ctx: TestContext) -> None:
    from src.core.server import _is_custom_cost_observability_name

    assert _is_custom_cost_observability_name("esound-cost-observability")
    assert _is_custom_cost_observability_name("lyra-cost-observability")
    assert _is_custom_cost_observability_name("Cost Anomaly Monitor")
    # Not a cost watcher → must not be swallowed.
    assert not _is_custom_cost_observability_name("response-quality-scorer")
    assert not _is_custom_cost_observability_name("repo-sync")
    assert not _is_custom_cost_observability_name("dream-mode")


@test("self_improvement", "cost-observability prompt is CACHE-AWARE and hourly")
async def t_cost_prompt(_ctx: TestContext) -> None:
    from src.core.server import (
        COST_OBSERVABILITY_DEFAULT_CRON, COST_OBSERVABILITY_PROMPT,
    )

    p = COST_OBSERVABILITY_PROMPT
    low = p.lower()
    # The cardinal rule: cache-aware, never alert on raw summed input.
    assert "cache-aware" in low, "prompt must state the cache-aware rule"
    assert "cache_read" in low or "non-cached" in low or "uncached" in low
    assert "fresh" in low, "prompt must key on FRESH / re-processed tokens"
    assert "never" in low and "raw" in low, (
        "prompt must forbid alerting on the raw summed input count"
    )
    # Sources the engine's own cache-aware signal, not a re-derived alarm.
    assert "router.cost_anomaly" in low
    # Silent-by-default discipline + real-cost framing.
    assert "silent" in low and ("cost_usd" in low or "real cost" in low)
    assert COST_OBSERVABILITY_DEFAULT_CRON == "0 * * * *", COST_OBSERVABILITY_DEFAULT_CRON


# ── 5c. escalation-audit — the ROLE-AGNOSTIC handoff arm ──────────────

@test("self_improvement", "escalation-audit: ON by default; its own gate parks it")
async def t_audit_gating(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler

    on_db = _fresh_db(ctx, "audit-on")
    await on_db.connect()
    try:
        srv, agent = await _bare_server({}, on_db)
        sched = Scheduler(on_db, agent)
        await srv._sync_escalation_audit(sched)
        row = await _audit_row(on_db)
        assert row is not None and row["enabled"], (
            "escalation-audit must seed AND enable by default — auditing your own "
            "handoffs is intrinsic, not opt-in"
        )
        assert row["cron_expression"] == "30 8 * * *", row["cron_expression"]
        assert row["prompt"]
    finally:
        await on_db.close()

    gate_db = _fresh_db(ctx, "audit-gate")
    await gate_db.connect()
    try:
        srv, agent = await _bare_server(
            {"self_improvement": {"escalation_audit_enabled": False}}, gate_db)
        sched = Scheduler(gate_db, agent)
        await srv._sync_escalation_audit(sched)
        row = await _audit_row(gate_db)
        assert row is None or not row["enabled"], (
            "escalation_audit_enabled:false must not enable the auditor"
        )
    finally:
        await gate_db.close()


@test("self_improvement", "escalation-audit DEDUP: a custom auditor suppresses the builtin")
async def t_audit_dedup(ctx: TestContext) -> None:
    from src.core.scheduler import Scheduler

    db = _fresh_db(ctx, "audit-dedup")
    await db.connect()
    try:
        srv, agent = await _bare_server({}, db)
        sched = Scheduler(db, agent)
        # eSound/Lyra ship `support-escalation-audit` → the builtin must NOT seed.
        await sched.add_task("support-escalation-audit", "30 8 * * *", "custom audit prompt")
        await srv._sync_escalation_audit(sched)
        assert await _audit_row(db) is None, (
            "the builtin escalation-audit was seeded despite a tuned custom one — "
            "eSound/Lyra would double-run their auditor"
        )
        assert await _row(db, "support-escalation-audit") is not None
    finally:
        await db.close()


@test("self_improvement", "escalation dedup name-matcher: correct scope, no false positives")
async def t_audit_name_matcher(_ctx: TestContext) -> None:
    from src.core.server import _is_custom_escalation_audit_name

    assert _is_custom_escalation_audit_name("support-escalation-audit")
    assert _is_custom_escalation_audit_name("Handoff Audit")
    assert _is_custom_escalation_audit_name("escalation-hygiene")
    # Not an escalation auditor → must not be swallowed.
    assert not _is_custom_escalation_audit_name("response-quality-scorer")
    assert not _is_custom_escalation_audit_name("repo-sync")
    assert not _is_custom_escalation_audit_name("dream-mode")


@test("self_improvement", "escalation-audit prompt is ROLE/TOOL-agnostic and window-aware")
async def t_audit_prompt(_ctx: TestContext) -> None:
    from src.core.server import (
        ESCALATION_AUDIT_DEFAULT_CRON, ESCALATION_AUDIT_PROMPT,
    )

    p = ESCALATION_AUDIT_PROMPT
    low = p.lower()
    norm = " ".join(low.split())  # collapse the block-scalar line wrapping
    # MUST be generic — OpenAgent is a general engine, not only a support runtime.
    assert "role-agnostic" in low and "tool-agnostic" in low
    assert "general engine" in low and "not only" in low
    assert "do not assume any particular product, queue, or mcp" in norm
    # Audits HANDOFFS / escalations, distinguishing justified vs over-escalated.
    assert "hand" in low and "escalat" in low
    assert "over-escalat" in low and "justified" in low
    # Must NOT cry wolf on platform-unreachable handoffs (the window lesson).
    assert "nobody can act" in low or "no human could act" in low
    # Silent unless regression; files a grounded correction.
    assert "regression" in low and "correction" in low
    assert ESCALATION_AUDIT_DEFAULT_CRON == "30 8 * * *", ESCALATION_AUDIT_DEFAULT_CRON


# ── 6. Feature B — anti-wedge per-LLM-call timeout ────────────────────

@test("model_timeout", "_construct_model applies a per-read timeout with the right precedence")
async def t_construct_model_timeout(_ctx: TestContext) -> None:
    from src.models.native_provider import NativeProvider

    class WithTimeout:
        def __init__(self, id=None, timeout=None, api_key=None):
            self.id, self.timeout, self.api_key = id, timeout, api_key

    class NoTimeout:
        def __init__(self, id=None, api_key=None):
            self.id, self.api_key = id, api_key

    np = NativeProvider.__new__(NativeProvider)
    prev = os.environ.get("OPENAGENT_MODEL_TIMEOUT_SECONDS")
    try:
        # Default 90s when the class accepts timeout and no env / kwarg is set.
        _set_env("OPENAGENT_MODEL_TIMEOUT_SECONDS", None)
        m = NativeProvider._construct_model(np, WithTimeout, id="x", api_key="k")
        assert m.timeout == 90.0, m.timeout
        # It must stay UNDER the 120s lease TTL, or a hung call still wedges.
        assert m.timeout < 120.0

        # Env override wins over the default.
        _set_env("OPENAGENT_MODEL_TIMEOUT_SECONDS", "45")
        assert NativeProvider._construct_model(np, WithTimeout, id="x").timeout == 45.0

        # An explicit kwarg wins over both (never clobbered).
        assert NativeProvider._construct_model(
            np, WithTimeout, id="x", timeout=12.5).timeout == 12.5

        # A class that does not accept `timeout` is left untouched (no crash).
        _set_env("OPENAGENT_MODEL_TIMEOUT_SECONDS", None)
        m = NativeProvider._construct_model(np, NoTimeout, id="x")
        assert not hasattr(m, "timeout")

        # Garbage env falls back to the 90s default, never crashes a build.
        _set_env("OPENAGENT_MODEL_TIMEOUT_SECONDS", "not-a-float")
        assert NativeProvider._construct_model(np, WithTimeout, id="x").timeout == 90.0
    finally:
        _set_env("OPENAGENT_MODEL_TIMEOUT_SECONDS", prev)


@test("model_timeout", "_build_agent wires model.timeout_seconds → the env override")
async def t_build_agent_wires_timeout(ctx: TestContext) -> None:
    from src.core.server import _build_agent

    prev = os.environ.get("OPENAGENT_MODEL_TIMEOUT_SECONDS")
    _set_env("OPENAGENT_MODEL_TIMEOUT_SECONDS", None)
    try:
        _build_agent({
            "name": "timeout-test",
            "memory": {"db_path": str(ctx.db_path)},
            "model": {"timeout_seconds": 45},
        })
        assert os.environ.get("OPENAGENT_MODEL_TIMEOUT_SECONDS") == "45.0", (
            os.environ.get("OPENAGENT_MODEL_TIMEOUT_SECONDS")
        )
    finally:
        _set_env("OPENAGENT_MODEL_TIMEOUT_SECONDS", prev)
