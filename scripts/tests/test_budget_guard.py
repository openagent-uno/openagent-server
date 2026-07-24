"""Budget enforcement gate tests — the brake that protects DeepSeek's $100.

These pin the SEAMS, not just the helper:

- off by default (no rules → ``_enabled_catalog`` byte-identical, old yaml boots);
- the gate excludes an over-cap scope, cost AND tokens;
- the JUNCTION — the real ``_resolve_entry_model`` routes around a blocked
  scope, proving ``_enabled_catalog`` actually consults the guard;
- a global cap blocks every model but never empties the catalog (logged);
- window rollover un-blocks from cached state with no new traffic;
- alerts fire once per threshold per window;
- the usage view reports spend vs limit, incl. non-enforced task/per_run rules;
- yaml seeding is additive and never clobbers an app edit.

Pure-unit: a throwaway MemoryDB per test, the REAL ModelDispatcher, no
pool/gateway/LLM. ``elog`` is monkeypatched to capture structured events so the
alert + never-empty assertions are deterministic.
"""
from __future__ import annotations

import contextlib
import time
import uuid

from ._framework import TestContext, test


def _providers(*entries: tuple[str, list[str]]) -> list[dict]:
    """Build a v0.12 providers_config from (provider_name, [model_ids]) pairs."""
    out: list[dict] = []
    for i, (name, models) in enumerate(entries, start=1):
        out.append({
            "id": i,
            "name": name,
            "framework": "api-based",
            "enabled": True,
            "models": [{"id": i * 100 + j, "model": m} for j, m in enumerate(models)],
        })
    return out


async def _make(ctx: TestContext, providers_config: list[dict]):
    """Fresh isolated DB + a real ModelDispatcher wired to it (guard created in
    set_db). TTL pinned high so the sync hot path never spawns a background
    refresh mid-test — every test drives ``refresh()`` explicitly."""
    from src.memory.db import MemoryDB
    from src.models.dispatcher import ModelDispatcher

    db = MemoryDB(str(ctx.test_dir / f"budget_guard_{uuid.uuid4().hex}.db"))
    await db.connect()
    disp = ModelDispatcher(providers_config)
    disp.set_db(db)
    disp.budget_guard._ttl = 1e9
    return db, disp


@contextlib.contextmanager
def _capture_elog():
    """Capture ``budget_guard``'s structured events as (name, kwargs) tuples."""
    import src.core.budget_guard as bg

    events: list[tuple[str, dict]] = []
    orig = bg.elog
    bg.elog = lambda name, **kw: events.append((name, kw))
    try:
        yield events
    finally:
        bg.elog = orig


def _ids(disp) -> list[str]:
    return [e.runtime_id for e in disp._enabled_catalog()]


async def _rec(db, model: str, *, cost: float = 0.0, tokens: int = 0):
    """Record one usage_log row. ``tokens`` splits across in/out."""
    await db.record_usage(
        model=model, input_tokens=tokens // 2, output_tokens=tokens - tokens // 2,
        cost=cost, session_id=f"s-{uuid.uuid4().hex[:6]}",
    )


# ── off by default ────────────────────────────────────────────────────


@test("budget_guard", "off by default: no rules → catalog byte-identical")
async def t_budget_off_by_default(ctx: TestContext) -> None:
    from src.models.catalog import iter_configured_models

    providers = _providers(("anthropic", ["claude-opus-4-8"]),
                           ("deepseek", ["deepseek-v4-pro"]))
    db, disp = await _make(ctx, providers)
    try:
        await disp.budget_guard.refresh()  # no rows → empty snapshot
        expected = [e.runtime_id for e in iter_configured_models(providers)
                    if not e.disabled]
        assert _ids(disp) == expected, "no-rules catalog diverged from unfiltered"
        # Byte-identical: with nothing blocked the gate returns its input object
        # unchanged — a deployment that never configured a budget is untouched.
        sample = disp._enabled_catalog()
        assert disp.budget_guard.filter_catalog(sample) is sample
    finally:
        await db.close()


# ── the gate ──────────────────────────────────────────────────────────


@test("budget_guard", "gate excludes a scope over its daily COST cap")
async def t_budget_gate_cost(ctx: TestContext) -> None:
    db, disp = await _make(ctx, _providers(
        ("anthropic", ["claude-opus-4-8"]), ("deepseek", ["deepseek-v4-pro"])))
    try:
        await db.add_budget(scope_kind="provider", scope_value="deepseek",
                            metric="cost_usd", window="day", amount=10.0)
        # Under the cap → still routable.
        await _rec(db, "deepseek:deepseek-v4-pro", cost=5.0)
        await disp.budget_guard.refresh()
        assert "deepseek:deepseek-v4-pro" in _ids(disp), "excluded while UNDER cap"
        # Push over the cap → excluded; the untouched provider stays.
        await _rec(db, "deepseek:deepseek-v4-pro", cost=6.0)  # total 11.0 > 10
        await disp.budget_guard.refresh()
        ids = _ids(disp)
        assert "deepseek:deepseek-v4-pro" not in ids, "over-cap scope NOT excluded"
        assert "anthropic:claude-opus-4-8" in ids, "unrelated scope wrongly excluded"
    finally:
        await db.close()


@test("budget_guard", "gate excludes on a TOKEN cap even when cost is zero")
async def t_budget_gate_tokens(ctx: TestContext) -> None:
    db, disp = await _make(ctx, _providers(
        ("anthropic", ["claude-opus-4-8"]), ("deepseek", ["deepseek-v4-pro"])))
    try:
        await db.add_budget(scope_kind="model",
                            scope_value="deepseek:deepseek-v4-pro",
                            metric="tokens", window="day", amount=1000)
        # cost=0 (pricing unavailable) but tokens still trip the cap.
        await _rec(db, "deepseek:deepseek-v4-pro", cost=0.0, tokens=1100)
        await disp.budget_guard.refresh()
        assert "deepseek:deepseek-v4-pro" not in _ids(disp), "token cap didn't block"
    finally:
        await db.close()


@test("budget_guard", "JUNCTION: _resolve_entry_model routes around a blocked scope")
async def t_budget_junction_entry_model(ctx: TestContext) -> None:
    # deepseek is FIRST in catalog order (would be the entry pick), so a working
    # gate must move entry resolution to anthropic — proving the filtered
    # catalog flows through the real dispatch path, not just the helper.
    db, disp = await _make(ctx, _providers(
        ("deepseek", ["deepseek-v4-pro"]), ("anthropic", ["claude-opus-4-8"])))
    try:
        await db.add_budget(scope_kind="provider", scope_value="deepseek",
                            metric="cost_usd", window="day", amount=1.0)
        await _rec(db, "deepseek:deepseek-v4-pro", cost=2.0)  # over
        await disp.budget_guard.refresh()
        decision = await disp._resolve_entry_model(None)
        assert decision.primary_model == "anthropic:claude-opus-4-8", (
            f"entry model did not route around the blocked scope: {decision}")
    finally:
        await db.close()


@test("budget_guard", "a session PIN survives an over-cap window — routed around, NOT unpinned")
async def t_budget_pin_survives_over_cap(ctx: TestContext) -> None:
    # The trap: a user pins deepseek; deepseek hits its daily cap; the pin must
    # be honoured again once the window rolls over. Auto-unpinning here would
    # turn one over-cap moment into a PERMANENT loss of the user's model choice.
    db, disp = await _make(ctx, _providers(
        ("deepseek", ["deepseek-v4-pro"]), ("anthropic", ["claude-opus-4-8"])))
    try:
        sid = f"pinned-{uuid.uuid4().hex[:8]}"
        await db.pin_session_model(sid, "deepseek:deepseek-v4-pro")
        await db.add_budget(scope_kind="provider", scope_value="deepseek",
                            metric="cost_usd", window="day", amount=1.0)
        await _rec(db, "deepseek:deepseek-v4-pro", cost=2.0)  # over cap
        await disp.budget_guard.refresh()

        decision = await disp._resolve_entry_model(sid)
        # This turn routes AWAY from the capped pin...
        assert decision.primary_model == "anthropic:claude-opus-4-8", (
            f"over-cap turn did not route around the pinned model: {decision}")
        # ...but the pin is STILL in the DB (temporary bypass, not auto-heal).
        assert await db.get_session_pin(sid) == "deepseek:deepseek-v4-pro", (
            "a budget exclusion permanently unpinned the user's model choice")
    finally:
        await db.close()


@test("budget_guard", "pin auto-heal still fires for a genuinely-gone model")
async def t_pin_auto_heal_on_deleted_model(ctx: TestContext) -> None:
    # The other side of the same branch: a pin to a model no longer in the
    # config (deleted/disabled) must still be cleaned up — else the session
    # keeps asking for a model that will never come back.
    db, disp = await _make(ctx, _providers(
        ("deepseek", ["deepseek-v4-pro"]), ("anthropic", ["claude-opus-4-8"])))
    try:
        sid = f"stale-{uuid.uuid4().hex[:8]}"
        await db.pin_session_model(sid, "openai:gpt-4o")  # not in this config
        await disp.budget_guard.refresh()

        decision = await disp._resolve_entry_model(sid)
        assert decision.primary_model == "deepseek:deepseek-v4-pro", (
            f"did not fall through to first-enabled after a stale pin: {decision}")
        assert await db.get_session_pin(sid) is None, (
            "a pin to a deleted model was NOT auto-healed")
    finally:
        await db.close()


# ── global + never-empty ──────────────────────────────────────────────


@test("budget_guard", "global cap blocks all but never empties the catalog (logged)")
async def t_budget_global_never_empty(ctx: TestContext) -> None:
    db, disp = await _make(ctx, _providers(
        ("deepseek", ["deepseek-v4-pro"]), ("anthropic", ["claude-opus-4-8"])))
    try:
        await db.add_budget(scope_kind="global", metric="tokens",
                            window="day", amount=100)
        await _rec(db, "deepseek:deepseek-v4-pro", tokens=120)  # global > 100
        with _capture_elog() as events:
            await disp.budget_guard.refresh()
            # The union logic: a tripped global rule targets EVERY runtime_id.
            blocked = disp.budget_guard.blocked_runtime_ids(_entries(disp))
            assert blocked == {"deepseek:deepseek-v4-pro", "anthropic:claude-opus-4-8"}
            # …but excluding all of them would empty the catalog, so the gate
            # refuses and keeps the agent online.
            ids = _ids(disp)
        assert set(ids) == {"deepseek:deepseek-v4-pro", "anthropic:claude-opus-4-8"}, (
            "never-empty invariant violated — a global cap took the agent offline")
        names = [n for n, _ in events]
        assert "budget.cap_not_enforced_would_empty_catalog" in names, (
            "would-empty refusal was not logged")
    finally:
        await db.close()


def _entries(disp):
    from src.models.catalog import iter_configured_models
    return [e for e in iter_configured_models(disp._providers_config) if not e.disabled]


@test("budget_guard", "a global cap must not mask a scope-specific brake")
async def t_budget_global_does_not_mask_provider(ctx: TestContext) -> None:
    # The operator's real setup: DeepSeek (paid) + Claude (subscription, $0), a
    # DeepSeek provider cap AND a global cap, both tripped. A naive union would
    # empty the catalog and refuse to enforce anything — silently disabling the
    # DeepSeek brake. The gate must instead exclude the individually-capped
    # DeepSeek and keep the only-global Claude online.
    db, disp = await _make(ctx, _providers(
        ("deepseek", ["deepseek-v4-pro"]), ("anthropic", ["claude-opus-4-8"])))
    try:
        await db.add_budget(scope_kind="provider", scope_value="deepseek",
                            metric="cost_usd", window="day", amount=10.0)
        await db.add_budget(scope_kind="global", metric="cost_usd",
                            window="day", amount=10.0)
        # DeepSeek spends $12 → over its own $10 provider cap AND over the $10
        # global cap. Claude ($0, subscription) adds nothing to the dollar total.
        await _rec(db, "deepseek:deepseek-v4-pro", cost=12.0)
        await disp.budget_guard.refresh()
        ids = _ids(disp)
        assert ids == ["anthropic:claude-opus-4-8"], (
            f"global cap masked the DeepSeek provider brake: {ids}")
    finally:
        await db.close()


# ── window rollover ───────────────────────────────────────────────────


@test("budget_guard", "window rollover un-blocks from cached state, no new traffic")
async def t_budget_rollover(ctx: TestContext) -> None:
    db, disp = await _make(ctx, _providers(
        ("deepseek", ["deepseek-v4-pro"]), ("anthropic", ["claude-opus-4-8"])))
    try:
        await db.add_budget(scope_kind="provider", scope_value="deepseek",
                            metric="cost_usd", window="day", amount=1.0)
        await _rec(db, "deepseek:deepseek-v4-pro", cost=5.0)  # over
        await disp.budget_guard.refresh()
        assert "deepseek:deepseek-v4-pro" not in _ids(disp), "not blocked pre-rollover"
        # Simulate the window elapsing: push every snapshot rule's window END
        # into the past. No refresh(), no new usage — the read-side gate must
        # drop the block purely from the cached snapshot + wall clock.
        for st in disp.budget_guard._snapshot:
            st.window_end = time.time() - 1.0
        assert "deepseek:deepseek-v4-pro" in _ids(disp), (
            "rollover did not un-block from cached state")
    finally:
        await db.close()


# ── alerts ────────────────────────────────────────────────────────────


@test("budget_guard", "alerts fire once per threshold per window, not per call")
async def t_budget_alerts_dedupe(ctx: TestContext) -> None:
    db, disp = await _make(ctx, _providers(("deepseek", ["deepseek-v4-pro"]),
                                          ("anthropic", ["claude-opus-4-8"])))
    try:
        await db.add_budget(scope_kind="provider", scope_value="deepseek",
                            metric="cost_usd", window="day", amount=10.0,
                            alert_thresholds=[0.5, 0.9])
        await _rec(db, "deepseek:deepseek-v4-pro", cost=6.0)  # 60% → crosses 0.5
        with _capture_elog() as events:
            await disp.budget_guard.refresh()
            await disp.budget_guard.refresh()  # same window, same spend
        alerts = [kw for n, kw in events if n == "budget.alert"]
        assert len(alerts) == 1, f"0.5 alert fired {len(alerts)}x (should be once)"
        assert abs(alerts[0]["threshold"] - 0.5) < 1e-9, alerts[0]

        await _rec(db, "deepseek:deepseek-v4-pro", cost=5.0)  # total 11 → 110%
        with _capture_elog() as events2:
            await disp.budget_guard.refresh()
            await disp.budget_guard.refresh()
        fired = sorted(kw["threshold"] for n, kw in events2 if n == "budget.alert")
        assert fired == [0.9, 1.0], f"expected 0.9 + cap once each, got {fired}"
        assert any(kw.get("capped") for n, kw in events2 if n == "budget.alert"), (
            "100% cap alert missing its capped=true marker")
    finally:
        await db.close()


# ── usage view (REST + MCP surface) ───────────────────────────────────


@test("budget_guard", "usage view reports spend vs limit; task/per_run not enforced")
async def t_budget_usage_view(ctx: TestContext) -> None:
    from src.core.budget_guard import compute_budget_usage

    db, disp = await _make(ctx, _providers(("deepseek", ["deepseek-v4-pro"])))
    try:
        await db.add_budget(scope_kind="provider", scope_value="deepseek",
                            metric="cost_usd", window="day", amount=10.0)
        await db.add_budget(scope_kind="task", scope_value="nightly",
                            metric="cost_usd", window="day", amount=1.0)
        await db.add_budget(scope_kind="model",
                            scope_value="deepseek:deepseek-v4-pro",
                            metric="cost_usd", window="per_run", amount=0.01)
        await _rec(db, "deepseek:deepseek-v4-pro", cost=3.0)
        rows = await compute_budget_usage(db)
        by = {(r["scope_kind"], r["window"]): r for r in rows}

        prov = by[("provider", "day")]
        assert prov["enforced"] is True
        assert abs(prov["spend"] - 3.0) < 1e-9, prov
        assert prov["amount"] == 10.0 and abs(prov["remaining"] - 7.0) < 1e-9, prov
        assert prov["over"] is False

        # task scope is reported (spend computed) but NOT enforced by the gate.
        assert by[("task", "day")]["enforced"] is False
        # per_run has no calendar window → spend None, not enforced.
        per_run = by[("model", "per_run")]
        assert per_run["enforced"] is False and per_run["spend"] is None, per_run
    finally:
        await db.close()


# ── yaml seeding ──────────────────────────────────────────────────────


@test("budget_guard", "yaml seed is additive and never clobbers an app edit")
async def t_budget_seed_reconcile(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.models.dispatcher import ModelDispatcher

    path = str(ctx.test_dir / f"budget_seed_{uuid.uuid4().hex}.db")
    seed = [{"scope_kind": "provider", "scope": "deepseek", "metric": "cost_usd",
             "window": "day", "amount": 10.0, "alert_thresholds": [0.5, 0.9]}]

    db = MemoryDB(path)
    await db.connect()
    try:
        disp = ModelDispatcher(_providers(("deepseek", ["deepseek-v4-pro"])))
        disp.set_budget_seed(seed)
        disp.set_db(db)
        await disp.budget_guard.warm()
        rows = await db.list_budgets()
        assert len(rows) == 1 and rows[0]["source"] == "yaml", rows
        assert rows[0]["amount"] == 10.0

        # Operator edits the amount in the app.
        await db.update_budget(rows[0]["id"], amount=25.0)

        # Reboot: a fresh dispatcher re-seeds from the SAME yaml. The edit must
        # survive (seed-only-if-absent) and no duplicate row appears.
        disp2 = ModelDispatcher(_providers(("deepseek", ["deepseek-v4-pro"])))
        disp2.set_budget_seed(seed)
        disp2.set_db(db)
        await disp2.budget_guard.warm()
        rows2 = await db.list_budgets()
        assert len(rows2) == 1, f"seed clobbered/duplicated the row: {rows2}"
        assert rows2[0]["amount"] == 25.0, "operator's app edit was clobbered on reboot"
    finally:
        await db.close()


# ── strict scopes (hard cap: enforce even if it empties the catalog) ────
# OPENAGENT_BUDGET_STRICT_SCOPES lets an operator say "for THIS scope, go
# offline rather than overspend" — overriding the never-empty safety for the
# named scope only.


@test("budget_guard", "_strict_scopes parses kind:value list + bare 'global'")
async def t_strict_scopes_parse(ctx: TestContext) -> None:
    import os
    from src.core.budget_guard import _strict_scopes
    os.environ["OPENAGENT_BUDGET_STRICT_SCOPES"] = "provider:deepseek, model:x:y , global, junk, task:t"
    try:
        got = _strict_scopes()
    finally:
        os.environ.pop("OPENAGENT_BUDGET_STRICT_SCOPES", None)
    assert ("provider", "deepseek") in got
    assert ("model", "x:y") in got          # value may itself contain a colon
    assert ("global", "") in got
    assert ("junk", "") not in got          # no colon, not 'global' → skipped
    assert ("task", "t") not in got         # task is not an enforceable scope
    # unset → empty
    os.environ.pop("OPENAGENT_BUDGET_STRICT_SCOPES", None)
    assert _strict_scopes() == frozenset()


@test("budget_guard", "STRICT scope cap enforces even if it empties the catalog (agent goes offline)")
async def t_budget_strict_offline(ctx: TestContext) -> None:
    import os
    db, disp = await _make(ctx, _providers(("deepseek", ["deepseek-v4-pro"])))
    os.environ["OPENAGENT_BUDGET_STRICT_SCOPES"] = "provider:deepseek"
    try:
        await db.add_budget(scope_kind="provider", scope_value="deepseek",
                            metric="tokens", window="day", amount=100)
        await _rec(db, "deepseek:deepseek-v4-pro", tokens=120)  # over 100
        with _capture_elog() as events:
            await disp.budget_guard.refresh()
            ids = _ids(disp)
        assert ids == [], f"strict cap must take the ONLY model offline, got {ids}"
        names = [n for n, _ in events]
        assert "budget.cap_enforced_strict" in names, "strict enforcement not logged"
        assert "budget.cap_not_enforced_would_empty_catalog" not in names, (
            "strict scope must NOT fall through to the never-empty refusal")
    finally:
        os.environ.pop("OPENAGENT_BUDGET_STRICT_SCOPES", None)
        await db.close()


@test("budget_guard", "STRICT drops its own models but keeps NON-strict models online")
async def t_budget_strict_partial(ctx: TestContext) -> None:
    import os
    db, disp = await _make(ctx, _providers(
        ("deepseek", ["deepseek-v4-pro"]), ("anthropic", ["claude-opus-4-8"])))
    os.environ["OPENAGENT_BUDGET_STRICT_SCOPES"] = "provider:deepseek"
    try:
        # both providers over their OWN token caps → both blocked → would-empty
        await db.add_budget(scope_kind="provider", scope_value="deepseek",
                            metric="tokens", window="day", amount=100)
        await db.add_budget(scope_kind="provider", scope_value="anthropic",
                            metric="tokens", window="day", amount=100)
        await _rec(db, "deepseek:deepseek-v4-pro", tokens=120)
        await _rec(db, "anthropic:claude-opus-4-8", tokens=120)
        await disp.budget_guard.refresh()
        ids = _ids(disp)
        # strict deepseek dropped (offline for it); non-strict anthropic kept
        # online by the never-empty safety.
        assert ids == ["anthropic:claude-opus-4-8"], (
            f"strict deepseek should drop, non-strict anthropic stay: {ids}")
    finally:
        os.environ.pop("OPENAGENT_BUDGET_STRICT_SCOPES", None)
        await db.close()


@test("budget_guard", "STRICT env naming a DIFFERENT scope → never-empty safety still holds")
async def t_budget_strict_nonmatching(ctx: TestContext) -> None:
    import os
    db, disp = await _make(ctx, _providers(("deepseek", ["deepseek-v4-pro"])))
    # strict names anthropic, but it is deepseek that is over cap → not strict
    os.environ["OPENAGENT_BUDGET_STRICT_SCOPES"] = "provider:anthropic"
    try:
        await db.add_budget(scope_kind="provider", scope_value="deepseek",
                            metric="tokens", window="day", amount=100)
        await _rec(db, "deepseek:deepseek-v4-pro", tokens=120)
        with _capture_elog() as events:
            await disp.budget_guard.refresh()
            ids = _ids(disp)
        assert ids == ["deepseek:deepseek-v4-pro"], (
            "a non-matching strict scope must not take the agent offline")
        names = [n for n, _ in events]
        assert "budget.cap_not_enforced_would_empty_catalog" in names
        assert "budget.cap_enforced_strict" not in names
    finally:
        os.environ.pop("OPENAGENT_BUDGET_STRICT_SCOPES", None)
        await db.close()
