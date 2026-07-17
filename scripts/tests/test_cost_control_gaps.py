"""Cost-control regression tests — the three gaps that let background / agent
work silently burn the Claude subscription and the DeepSeek budget.

Pure-unit: throwaway MemoryDB, the REAL ModelDispatcher / TeamRouterProvider /
BudgetGuard, and hand-primed OpenRouter pricing. No pool / gateway / LLM.

C1 — the budget guard gates the Team LEADER but not its MEMBERS, so an over-cap
     provider still got delegated member work. We pin: a blocked provider is
     dropped from the member set, but filtering NEVER empties it.
C2 — a ``cost_usd`` budget on the $0 claude-sub-proxy computes 0 spend forever,
     so the cap can't trip. We pin: the guard WARNS loudly (log + usage-view
     surface) and never silently no-ops, but only once real pricing is loaded
     (no boot false-positive) and never for a genuinely-priced provider.
C3 — compaction + the quality judge defaulted to the full Team router when their
     env model is unset (a ~150k-token fold through the premium leader / paid
     DeepSeek). We pin: unset now routes to the cheapest enabled row as a
     toolkit-free NativeProvider, the set-path is unchanged, a NON-router
     fallback is preserved untouched, and the router fallback is logged.
"""
from __future__ import annotations

import contextlib
import os
import time
import uuid
from types import SimpleNamespace

from ._framework import TestContext, test


# ── shared fixtures ────────────────────────────────────────────────────


def _providers(*entries: tuple[str, list[str]]) -> list[dict]:
    """v0.12 providers_config from (provider_name, [model_ids]) pairs. Each
    provider carries an undialled api_key so a NativeProvider can be constructed
    (never invoked)."""
    out: list[dict] = []
    for i, (name, models) in enumerate(entries, start=1):
        out.append({
            "id": i,
            "name": name,
            "framework": "api-based",
            "api_key": "sk-test-not-dialled",
            "base_url": None,
            "enabled": True,
            "models": [{"id": i * 100 + j, "model": m} for j, m in enumerate(models)],
        })
    return out


async def _make(ctx: TestContext, providers_config: list[dict]):
    """Fresh isolated DB + a real ModelDispatcher (guard created + given the
    providers view in set_db). TTL pinned high so the sync hot path never spawns
    a background refresh mid-test."""
    from src.memory.db import MemoryDB
    from src.models.dispatcher import ModelDispatcher

    db = MemoryDB(str(ctx.test_dir / f"cost_gaps_{uuid.uuid4().hex}.db"))
    await db.connect()
    disp = ModelDispatcher(providers_config)
    disp.set_db(db)
    disp.budget_guard._ttl = 1e9
    return db, disp


@contextlib.contextmanager
def _capture_bg():
    """Capture ``budget_guard``'s structured events as (name, kwargs) tuples."""
    import src.core.budget_guard as bg

    events: list[tuple[str, dict]] = []
    orig = bg.elog
    bg.elog = lambda name, level="info", **kw: events.append((name, kw))
    try:
        yield events
    finally:
        bg.elog = orig


@contextlib.contextmanager
def _capture_mod(module):
    """Capture a module's ``elog`` events as (name, kwargs) tuples."""
    events: list[tuple[str, dict]] = []
    orig = module.elog
    module.elog = lambda name, level="info", **kw: events.append((name, kw))
    try:
        yield events
    finally:
        module.elog = orig


@contextlib.contextmanager
def _env(**kw):
    """Set env vars for the block; restore prior values (None = unset)."""
    saved = {k: os.environ.get(k) for k in kw}
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def _primed_pricing():
    """Prime OpenRouter pricing so ``deepseek:*`` is priced and any ``local:*``
    (sub-proxy) is absent → $0. ``openrouter_pricing_ready()`` then reports
    True."""
    import time as _t

    from src.models import discovery
    import src.models.catalog as catalog

    fake = [{"id": "deepseek/deepseek-v4-pro",
             "pricing": {"prompt": "0.000000435", "completion": "0.00000087"}}]
    saved_cache = getattr(discovery, "_OPENROUTER_CACHE", None)
    saved_index = catalog._OPENROUTER_INDEX
    discovery._OPENROUTER_CACHE = (_t.time() + 1e6, fake)  # far-future ts = never stale
    catalog._OPENROUTER_INDEX = None
    try:
        yield
    finally:
        discovery._OPENROUTER_CACHE = saved_cache
        catalog._OPENROUTER_INDEX = saved_index


@contextlib.contextmanager
def _cold_pricing():
    """Force the OpenRouter pricing cache empty (cold boot)."""
    from src.models import discovery
    import src.models.catalog as catalog

    saved_cache = getattr(discovery, "_OPENROUTER_CACHE", None)
    saved_index = catalog._OPENROUTER_INDEX
    discovery._OPENROUTER_CACHE = None
    catalog._OPENROUTER_INDEX = None
    try:
        yield
    finally:
        discovery._OPENROUTER_CACHE = saved_cache
        catalog._OPENROUTER_INDEX = saved_index


# ══ C1 — the guard must gate MEMBERS, not only the leader ══════════════


class _FakeAgent:
    """Minimal runtime-Agent stand-in: exposes ``.model`` and accepts a
    ``.metadata`` assignment (team-member-session linkage)."""

    def __init__(self, runtime_id: str) -> None:
        self.model = SimpleNamespace(id=runtime_id, provider=runtime_id.split(":", 1)[0])
        self.metadata = None


def _blocked_guard(*blocked_providers: str):
    """A BudgetGuard whose cached snapshot marks each provider over its cap NOW.
    No DB, no refresh — we drive the read-side gate directly."""
    from src.core.budget_guard import BudgetGuard, RuleState

    guard = BudgetGuard(db=None)
    guard._ttl = 1e9
    guard._last_refresh_monotonic = time.monotonic()  # never schedule a refresh
    now = time.time()
    guard._snapshot = [
        RuleState(
            rule_id=f"r-{p}", scope_kind="provider", scope_value=p,
            metric="cost_usd", window="day", amount=1.0, thresholds=(),
            webhook_url=None, window_start=now - 3600, window_end=now + 3600,
            spend=99.0, ratio=99.0, over=True,
        )
        for p in blocked_providers
    ]
    return guard


def _members_built_by_ensure_runtime(providers, entry_runtime_id, guard):
    """Drive the REAL ``TeamRouterProvider._ensure_runtime`` with the runtime
    Agent/Team construction stubbed out, and return the list of member
    runtime_ids the team was actually built with (leader excluded)."""
    import src.core._runner.team as team_mod
    from src.models.dispatcher import TeamRouterProvider

    provider = TeamRouterProvider(
        entry_runtime_id=entry_runtime_id,
        providers_config=providers,
        budget_guard=guard,
    )

    member_ids: list[str] = []

    def fake_build_api_agent_for(entry, *, name, role=None, system=None, db=None):
        return _FakeAgent(entry.runtime_id)  # leader

    def fake_build_agent_for(entry, *, name, role=None, system=None, db=None):
        member_ids.append(entry.runtime_id)
        return _FakeAgent(entry.runtime_id)

    provider._build_api_agent_for = fake_build_api_agent_for  # type: ignore[assignment]
    provider._build_agent_for = fake_build_agent_for  # type: ignore[assignment]

    captured: dict = {}

    class _StubTeam:
        def __init__(self, *args, **kwargs):
            captured["members"] = kwargs.get("members")

    saved_team = team_mod.Team
    team_mod.Team = _StubTeam  # bound at call time via the lazy import
    try:
        provider._ensure_runtime(f"s-{uuid.uuid4().hex[:8]}", system="SYS")
    finally:
        team_mod.Team = saved_team
    return member_ids


@test("cost_control_gaps", "C1: an over-cap provider is dropped from the Team member set")
async def t_c1_member_excludes_blocked(ctx: TestContext) -> None:
    # Leader = local sub ($0). Members would be deepseek + anthropic — but
    # deepseek is over its cap, so it must NOT be handed delegated member work.
    providers = _providers(
        ("local", ["claude-sub"]),
        ("deepseek", ["deepseek-v4-pro"]),
        ("anthropic", ["claude-opus-4-8"]),
    )
    guard = _blocked_guard("deepseek")
    members = _members_built_by_ensure_runtime(providers, "local:claude-sub", guard)
    assert "deepseek:deepseek-v4-pro" not in members, (
        f"an over-cap provider was still delegated member work: {members}")
    assert "anthropic:claude-opus-4-8" in members, (
        f"an unrelated provider was wrongly excluded from members: {members}")


@test("cost_control_gaps", "C1: filtering members NEVER empties the set (never-empty guarantee)")
async def t_c1_member_never_empty(ctx: TestContext) -> None:
    # BOTH members are over their own cap. Excluding both would empty the member
    # set; the guard's filter_catalog falls back instead (same never-empty rule
    # as the leader), so the team still has members to route to.
    providers = _providers(
        ("local", ["claude-sub"]),
        ("deepseek", ["deepseek-v4-pro"]),
        ("anthropic", ["claude-opus-4-8"]),
    )
    guard = _blocked_guard("deepseek", "anthropic")
    members = _members_built_by_ensure_runtime(providers, "local:claude-sub", guard)
    assert members, "member filtering emptied the set — never-empty guarantee violated"
    assert set(members) == {"deepseek:deepseek-v4-pro", "anthropic:claude-opus-4-8"}, (
        f"all-over-budget members should fall back to the full set, got {members}")


@test("cost_control_gaps", "C1: no guard / no rules leaves the member set byte-identical")
async def t_c1_member_no_guard_unchanged(ctx: TestContext) -> None:
    from src.core.budget_guard import BudgetGuard

    providers = _providers(
        ("local", ["claude-sub"]),
        ("deepseek", ["deepseek-v4-pro"]),
        ("anthropic", ["claude-opus-4-8"]),
    )
    empty_guard = BudgetGuard(db=None)  # no snapshot → nothing blocked
    empty_guard._ttl = 1e9
    empty_guard._last_refresh_monotonic = time.monotonic()
    members = _members_built_by_ensure_runtime(providers, "local:claude-sub", empty_guard)
    assert set(members) == {"deepseek:deepseek-v4-pro", "anthropic:claude-opus-4-8"}, (
        f"off-by-default filtering changed the member set: {members}")


# ══ C2 — a cost_usd budget cannot cap the $0-priced sub ════════════════


_ZERO_PRICE_EVENT = "budget.cost_metric_ineffective_zero_price"


@test("cost_control_gaps", "C2: a cost_usd rule on the $0 sub-proxy warns; a priced provider does not")
async def t_c2_warns_on_zero_priced(ctx: TestContext) -> None:
    db, disp = await _make(ctx, _providers(
        ("local", ["claude-sub"]), ("deepseek", ["deepseek-v4-pro"])))
    try:
        await db.add_budget(scope_kind="provider", scope_value="local",
                            metric="cost_usd", window="day", amount=10.0)
        await db.add_budget(scope_kind="provider", scope_value="deepseek",
                            metric="cost_usd", window="day", amount=10.0)
        with _primed_pricing(), _capture_bg() as events:
            await disp.budget_guard.refresh()
        warned = {kw.get("scope") for n, kw in events if n == _ZERO_PRICE_EVENT}
        assert "local" in warned, (
            f"the $0 sub-proxy cost_usd cap did not WARN (it silently no-ops): {events}")
        assert "deepseek" not in warned, (
            "a genuinely-priced provider was wrongly flagged as cost-ineffective")
    finally:
        await db.close()


@test("cost_control_gaps", "C2: the warning is loud-but-once-per-rule, not per refresh")
async def t_c2_warn_dedupe(ctx: TestContext) -> None:
    db, disp = await _make(ctx, _providers(("local", ["claude-sub"])))
    try:
        await db.add_budget(scope_kind="model", scope_value="local:claude-sub",
                            metric="cost_usd", window="day", amount=5.0)
        with _primed_pricing(), _capture_bg() as events:
            await disp.budget_guard.refresh()
            await disp.budget_guard.refresh()  # same rule, same window
        fired = [1 for n, _ in events if n == _ZERO_PRICE_EVENT]
        assert len(fired) == 1, f"zero-price warning fired {len(fired)}x (should be once)"
    finally:
        await db.close()


@test("cost_control_gaps", "C2: a COLD pricing cache must not false-warn at boot")
async def t_c2_no_boot_false_positive(ctx: TestContext) -> None:
    db, disp = await _make(ctx, _providers(("local", ["claude-sub"])))
    try:
        await db.add_budget(scope_kind="provider", scope_value="local",
                            metric="cost_usd", window="day", amount=10.0)
        with _cold_pricing(), _capture_bg() as events:
            await disp.budget_guard.refresh()
        assert not any(n == _ZERO_PRICE_EVENT for n, _ in events), (
            "warned on a cold pricing cache — a boot-time miss reads like a free "
            "model and must not fire")
    finally:
        await db.close()


@test("cost_control_gaps", "C2: the usage view surfaces cost_metric_ineffective per rule")
async def t_c2_usage_view_surface(ctx: TestContext) -> None:
    from src.core.budget_guard import compute_budget_usage

    providers = _providers(("local", ["claude-sub"]), ("deepseek", ["deepseek-v4-pro"]))
    db, disp = await _make(ctx, providers)
    try:
        await db.add_budget(scope_kind="provider", scope_value="local",
                            metric="cost_usd", window="day", amount=10.0)
        await db.add_budget(scope_kind="provider", scope_value="deepseek",
                            metric="cost_usd", window="day", amount=10.0)
        await db.add_budget(scope_kind="model", scope_value="local:claude-sub",
                            metric="tokens", window="day", amount=1000)
        with _primed_pricing():
            rows = await compute_budget_usage(db, providers_config=providers)
        by = {(r["scope_kind"], r["scope_value"], r["metric"]): r for r in rows}

        local_cost = by[("provider", "local", "cost_usd")]
        assert local_cost["cost_metric_ineffective"] is True and local_cost.get("warning"), (
            f"usage view did not surface the $0 cost_usd hazard: {local_cost}")
        # A tokens rule on the same $0 scope is the RIGHT tool → not flagged.
        assert by[("model", "local:claude-sub", "tokens")]["cost_metric_ineffective"] is False
        # A priced provider is fine.
        assert by[("provider", "deepseek", "cost_usd")]["cost_metric_ineffective"] is False
    finally:
        await db.close()


# ══ C3 — compaction + judge must default to a CHEAP path, not the router ══


class _PlainModel:
    """A non-router fallback (an already-cheap single model)."""

    model = "local:claude-sub"


def _router(providers) -> object:
    from src.models.dispatcher import ModelDispatcher

    return ModelDispatcher(providers)


@test("cost_control_gaps", "C3: compaction unset → cheapest NativeProvider, NOT the full router")
async def t_c3_compaction_cheap_default(ctx: TestContext) -> None:
    from src.core.compaction import _pick_summary_model

    providers = _providers(("deepseek", ["deepseek-v4-pro"]), ("local", ["claude-sub"]))
    agent = SimpleNamespace(_providers_config=providers, _db=None)
    router = _router(providers)
    with _env(OPENAGENT_COMPACTION_MODEL=None), _primed_pricing():
        picked = _pick_summary_model(agent, fallback=router)
    assert picked is not router, "unset env still returned the full Team router (C3 regression)"
    assert type(picked).__name__ == "NativeProvider", f"expected a NativeProvider, got {picked!r}"
    # Cheapest enabled row = the $0 sub (deepseek is priced), regardless of order.
    assert picked.model == "local:claude-sub", f"did not pick the cheapest row: {picked.model!r}"


@test("cost_control_gaps", "C3: a NON-router fallback is preserved untouched (old contract)")
async def t_c3_nonrouter_fallback_preserved(ctx: TestContext) -> None:
    from src.core.compaction import _pick_summary_model

    providers = _providers(("deepseek", ["deepseek-v4-pro"]), ("local", ["claude-sub"]))
    agent = SimpleNamespace(_providers_config=providers, _db=None)
    plain = _PlainModel()
    with _env(OPENAGENT_COMPACTION_MODEL=None), _primed_pricing():
        picked = _pick_summary_model(agent, fallback=plain)
    assert picked is plain, (
        "a non-router fallback must be returned unchanged — the cheap-path rewrite "
        "must ONLY replace the full dispatcher")


@test("cost_control_gaps", "C3: no enabled row → router fallback is used AND logged")
async def t_c3_no_cheap_row_warns(ctx: TestContext) -> None:
    import src.core.compaction as compaction

    agent = SimpleNamespace(_providers_config=[], _db=None)  # nothing to resolve
    router = _router([])
    with _env(OPENAGENT_COMPACTION_MODEL=None), _capture_mod(compaction) as events:
        picked = compaction._pick_summary_model(agent, fallback=router)
    assert picked is router, "with no cheap row resolvable, must fall back to the router"
    assert any(n == "runtime.compaction.summary_model_dispatcher_fallback" for n, _ in events), (
        f"the expensive dispatcher fallback was not logged: {[n for n, _ in events]}")


@test("cost_control_gaps", "C3: compaction with env SET is unchanged (configured cheap model)")
async def t_c3_compaction_env_set_unchanged(ctx: TestContext) -> None:
    from src.core.compaction import _pick_summary_model

    providers = _providers(("deepseek", ["deepseek-v4-pro"]), ("local", ["claude-sub"]))
    agent = SimpleNamespace(_providers_config=providers, _db=None)
    router = _router(providers)
    with _env(OPENAGENT_COMPACTION_MODEL="deepseek:deepseek-v4-pro"):
        picked = _pick_summary_model(agent, fallback=router)
    assert picked is not router and type(picked).__name__ == "NativeProvider"
    assert picked.model == "deepseek:deepseek-v4-pro", (
        f"env-set path must honour the configured model exactly: {picked.model!r}")


@test("cost_control_gaps", "C3: quality judge unset → cheapest NativeProvider, NOT the router")
async def t_c3_judge_cheap_default(ctx: TestContext) -> None:
    from src.core.quality_monitor import _pick_judge_model

    providers = _providers(("deepseek", ["deepseek-v4-pro"]), ("local", ["claude-sub"]))
    router = _router(providers)
    agent = SimpleNamespace(model=router, _providers_config=providers, _db=None, system_prompt="")
    with _env(OPENAGENT_QUALITY_MONITOR_MODEL=None, OPENAGENT_COMPACTION_MODEL=None), _primed_pricing():
        judge = _pick_judge_model(agent)
    assert judge is not router, "unset judge still returned the full Team router (C3 regression)"
    assert type(judge).__name__ == "NativeProvider", f"expected a NativeProvider, got {judge!r}"
    assert judge.model == "local:claude-sub", f"judge did not pick the cheapest row: {judge.model!r}"


@test("cost_control_gaps", "C3: quality judge with a NON-router model is preserved untouched")
async def t_c3_judge_nonrouter_preserved(ctx: TestContext) -> None:
    from src.core.quality_monitor import _pick_judge_model

    providers = _providers(("deepseek", ["deepseek-v4-pro"]), ("local", ["claude-sub"]))
    plain = _PlainModel()
    agent = SimpleNamespace(model=plain, _providers_config=providers, _db=None, system_prompt="")
    with _env(OPENAGENT_QUALITY_MONITOR_MODEL=None, OPENAGENT_COMPACTION_MODEL=None), _primed_pricing():
        judge = _pick_judge_model(agent)
    assert judge is plain, "a non-router agent.model must be returned unchanged for the judge"
