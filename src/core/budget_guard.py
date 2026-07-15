"""Budget enforcement — the brake that lets DeepSeek's $100 PAYG be safe.

WHY THIS MODULE EXISTS
----------------------
``BudgetTracker`` (``src/models/budget.py``) only ever RECORDED cost into
``usage_log``. Nothing read a remaining balance before a call, and the one gate
that would have stopped a runaway was deleted with a yaml knob (see the
dispatcher comment at ``record_cost``). So a model on a metered API could loop
and burn its whole balance with no code path able to notice. This module is the
missing half: it reads spend per configured scope and, at the cap, EXCLUDES that
scope from the router's enabled catalog for the rest of the window.

THE BEHAVIOUR, DECIDED BY THE OPERATOR (do not re-litigate)
----------------------------------------------------------
At the cap the scope is DROPPED FROM ROUTING, not hard-stopped. DeepSeek over
its daily cap simply disappears from the enabled catalog; the agent keeps
working on whatever stays enabled (Claude via the subscription proxy costs $0
PAYG). The bot never stops — it stops *spending on that model*. The exclusion
lifts automatically when the window rolls over.

THE SYNC / ASYNC SPLIT (the hard architectural constraint)
----------------------------------------------------------
``ModelDispatcher._enabled_catalog`` is SYNCHRONOUS and on the hot path (entry
resolution, team-member selection). It cannot run a DB aggregation on every
call. So the guard keeps a CACHED SNAPSHOT of per-rule spend, recomputed
asynchronously (:meth:`refresh`) after each ``record_cost`` and on a short TTL
backstop. The gate the dispatcher calls (:meth:`filter_catalog`) is a pure,
allocation-cheap function of that snapshot plus the current wall-clock — no I/O.

WINDOW ROLLOVER WITH NO TRAFFIC (correctness, not theory)
---------------------------------------------------------
A scope blocked at 23:59 must un-block at 00:00 even if nothing records usage
after. The snapshot therefore stores each rule's window END, and the read-side
gate treats a rule as blocking ONLY while ``window_start <= now < window_end``.
Once the clock passes ``window_end`` the rule stops contributing — the block
lifts on the very next read, from cached state, with no query and no background
tick required. The TTL refresh then re-measures the fresh (near-zero) window
spend when traffic resumes. This "compute-on-read from a cached snapshot" is
what the brief asked for; the periodic refresh only keeps the numbers current.

NEVER EMPTY THE CATALOG
-----------------------
A budget cap must not take the agent fully offline. If excluding every tripped
scope would leave ZERO enabled models, the guard does NOT exclude — it returns
the full catalog and logs that it would have. A global TOKEN cap can trip even
the $0 subscription models, so this guard is real here, not decorative.

OFF BY DEFAULT
--------------
No rules → :meth:`filter_catalog` returns its input unchanged (same list), so a
deployment that never configured a budget behaves byte-identically to before.
No new env var, no config that does nothing.

SELF-HOSTED / §17
-----------------
No provider is structurally required. A malformed rule, a bad timezone, or a
failed query is skipped with an audit line — exactly like ``safety.py`` drops a
bad regex — and never crashes a turn or takes the agent offline.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone as _timezone
from typing import Any

from src.core.logging import elog

# The router gate acts on these only. ``task`` scopes and the ``per_run`` window
# are stored, validated, and REPORTED in the usage view, but their enforcement
# (scheduler skip + mid-run cancel) is phase 2 and lives at other hooks — a task
# is a CALLER, not a routing target, so excluding a model can't stop it.
ENFORCEABLE_SCOPE_KINDS = frozenset({"global", "provider", "model"})
ENFORCEABLE_WINDOWS = frozenset({"hour", "day", "month"})
ALL_SCOPE_KINDS = frozenset({"global", "provider", "model", "task"})
ALL_WINDOWS = frozenset({"hour", "day", "month", "per_run"})
VALID_METRICS = frozenset({"cost_usd", "tokens"})

DEFAULT_ALERT_THRESHOLDS = (0.5, 0.9)

# How stale the snapshot may get before a hot-path call opportunistically
# schedules a refresh. Short because the refresh is one cheap aggregation and
# because it is the backstop that re-measures a rolled-over window when traffic
# resumes. Cost recording triggers a refresh directly, so under load the
# snapshot is fresher than this.
_REFRESH_TTL_S = 30.0

# Outbound alert webhook timeout — a slow or dead endpoint must never wedge a
# refresh, so the POST is fire-and-forget with a tight bound.
_WEBHOOK_TIMEOUT_S = 8.0


def _agent_zone():
    """The agent's timezone for window boundaries, or UTC.

    Reads ``src.memory.schedule.default_timezone_name`` — the same
    "the operator's day, not UTC" source the scheduler uses. Degrades to UTC on
    any error (bad env var, missing tzdata) rather than raising: a budget window
    resolving to UTC is a defensible fallback; crashing the gate is not (§17).
    """
    try:
        from src.memory.schedule import default_timezone_name, resolve_timezone

        return resolve_timezone(default_timezone_name()) or _timezone.utc
    except Exception:  # noqa: BLE001 — any tz failure → UTC, never fatal
        return _timezone.utc


def _window_bounds(window: str, now: float, zone) -> tuple[float, float] | None:
    """(start_epoch, end_epoch) of the calendar window containing ``now``.

    Boundaries are wall-clock in ``zone`` — a "day" is midnight-to-midnight
    where the operator lives, not UTC. Aware-datetime + ``timedelta`` arithmetic
    is wall-clock (Python normalises the offset on ``.timestamp()``), so a day
    that is 23h/25h across a DST switch still spans exactly one local calendar
    day. Returns ``None`` for ``per_run`` / unknown windows (no calendar bound).
    """
    if window not in ENFORCEABLE_WINDOWS:
        return None
    dt = datetime.fromtimestamp(now, zone)
    if window == "hour":
        start = dt.replace(minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=1)
    elif window == "day":
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
    else:  # month
        start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = (
            start.replace(year=start.year + 1, month=1)
            if start.month == 12
            else start.replace(month=start.month + 1)
        )
    return start.timestamp(), end.timestamp()


def window_start_epoch(window: str, now: float, zone=None) -> float | None:
    bounds = _window_bounds(window, now, zone or _agent_zone())
    return bounds[0] if bounds else None


@dataclass
class RuleState:
    """A rule's evaluation for the current window — the snapshot unit."""

    rule_id: str
    scope_kind: str
    scope_value: str
    metric: str
    window: str
    amount: float
    thresholds: tuple[float, ...]
    webhook_url: str | None
    window_start: float
    window_end: float
    spend: float
    ratio: float
    over: bool


def _infer_scope_kind(scope_value: str | None) -> str:
    """Best-effort scope_kind for a convenience yaml/API ``scope`` string when
    ``scope_kind`` is not given explicitly. Explicit always wins upstream."""
    if not scope_value:
        return "global"
    return "model" if ":" in scope_value else "provider"


def normalize_rule_input(raw: Any) -> dict[str, Any] | None:
    """Normalise a yaml / API budget dict into ``add_budget``/``seed_budget``
    kwargs, or ``None`` if it is not a usable rule.

    Accepts ``scope`` or ``scope_value`` for the target; infers ``scope_kind``
    from the colon only when it is not stated (explicit wins). Raises
    ``ValueError`` on an out-of-vocabulary field so the caller can skip it with
    an audit line rather than writing a row the CHECK constraint would reject.
    """
    if not isinstance(raw, dict):
        return None
    scope_value = (
        raw.get("scope_value")
        if raw.get("scope_value") is not None
        else raw.get("scope")
    )
    scope_value = "" if scope_value is None else str(scope_value).strip()
    scope_kind = (raw.get("scope_kind") or "").strip() or _infer_scope_kind(scope_value)
    if scope_kind == "global":
        scope_value = ""
    metric = (raw.get("metric") or "cost_usd").strip()
    window = (raw.get("window") or "day").strip()
    if scope_kind not in ALL_SCOPE_KINDS:
        raise ValueError(f"unknown scope_kind {scope_kind!r}")
    if metric not in VALID_METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    if window not in ALL_WINDOWS:
        raise ValueError(f"unknown window {window!r}")
    if scope_kind != "global" and not scope_value:
        raise ValueError(f"scope is required for scope_kind={scope_kind}")
    try:
        amount = float(raw.get("amount"))
    except (TypeError, ValueError):
        raise ValueError("amount must be a number")
    thresholds = raw.get("alert_thresholds")
    if thresholds is None:
        thresholds = list(DEFAULT_ALERT_THRESHOLDS)
    else:
        thresholds = [float(t) for t in thresholds]
    return {
        "scope_kind": scope_kind,
        "scope_value": scope_value,
        "metric": metric,
        "window": window,
        "amount": amount,
        "alert_thresholds": thresholds,
        "webhook_url": (str(raw.get("webhook_url")).strip() or None)
        if raw.get("webhook_url")
        else None,
        "enabled": bool(raw.get("enabled", True)),
    }


class BudgetGuard:
    """Cached, timezone-aware budget gate consulted synchronously by the router.

    One instance per :class:`ModelDispatcher`, created in ``set_db`` and shared
    with every ``TeamRouterProvider`` so ``record_cost`` can nudge a refresh.
    """

    def __init__(self, db: Any):
        self._db = db
        self._snapshot: list[RuleState] = []
        self._has_rules = False
        # yaml seed rules (raw dicts); applied once, additively, at first refresh.
        self._seed_rules: list[Any] = []
        self._seeded = False
        self._last_refresh_monotonic = 0.0
        self._ttl = _REFRESH_TTL_S
        self._refresh_task: asyncio.Task | None = None
        self._refresh_pending = False
        # Alert de-dupe: {rule_id: {window_start: {thresholds already fired}}}.
        # In-memory: a threshold may re-fire once after a process restart, which
        # is not spam; window rollover changes the key and resets naturally.
        self._alerts_fired: dict[str, dict[float, set[float]]] = {}
        # Log throttles so the gate is not chatty per turn.
        self._last_excluded_key: frozenset[str] | None = None
        self._last_would_empty_key: frozenset[str] | None = None

    # ── seed (yaml) ───────────────────────────────────────────────────

    def set_seed_rules(self, rules: Any) -> None:
        self._seed_rules = list(rules) if isinstance(rules, (list, tuple)) else []

    async def _seed_once(self) -> None:
        """Additively seed yaml rules the first time we refresh. Idempotent at
        the DB level (``seed_budget`` inserts only when absent), so a partial
        failure simply retries next refresh."""
        if self._seeded or not self._seed_rules:
            self._seeded = True
            return
        seeded = 0
        for raw in self._seed_rules:
            try:
                norm = normalize_rule_input(raw)
            except ValueError as e:
                elog("budget.seed_rule_invalid", level="warning",
                     error=str(e), rule=str(raw)[:200])
                continue
            if norm is None:
                continue
            try:
                if await self._db.seed_budget(**norm):
                    seeded += 1
            except Exception as e:  # noqa: BLE001 — one bad row must not abort seeding
                elog("budget.seed_rule_error", level="warning",
                     error=str(e), rule=str(raw)[:200])
        self._seeded = True
        if seeded:
            elog("budget.seeded", count=seeded)

    async def warm(self) -> None:
        """Boot-time: seed yaml rules and prime the snapshot so the very first
        turn already routes around an over-cap scope. Never fatal."""
        try:
            await self.refresh()
        except Exception as e:  # noqa: BLE001
            elog("budget.warm_error", level="warning", error=str(e))

    # ── refresh (async) ───────────────────────────────────────────────

    async def refresh(self) -> None:
        """Recompute the per-rule snapshot and fire threshold alerts.

        Reloads all enabled rules every time, so a rule added out-of-process
        (the ``budget-manager`` MCP subprocess writes straight to the DB) is
        picked up within one TTL. Never raises."""
        if self._db is None:
            return
        try:
            await self._seed_once()
        except Exception as e:  # noqa: BLE001
            elog("budget.seed_error", level="warning", error=str(e))
        try:
            rules = await self._db.list_budgets(enabled_only=True)
        except Exception as e:  # noqa: BLE001
            elog("budget.refresh_error", level="warning", error=str(e))
            return
        now = time.time()
        zone = _agent_zone()
        snapshot: list[RuleState] = []
        for rule in rules:
            try:
                st = await self._evaluate(rule, now, zone)
            except Exception as e:  # noqa: BLE001 — skip a bad rule, keep the rest
                elog("budget.rule_skipped", level="warning",
                     budget_id=rule.get("id"), error=str(e))
                continue
            if st is None:
                continue
            snapshot.append(st)
            self._maybe_fire_alerts(st)
        self._snapshot = snapshot
        self._has_rules = bool(rules)
        self._last_refresh_monotonic = time.monotonic()
        self._prune_alert_state(snapshot)

    async def _evaluate(self, rule: dict, now: float, zone) -> RuleState | None:
        """Turn one enabled rule row into a :class:`RuleState`, or ``None`` when
        it is not something the router gate enforces (task / per_run / non-
        positive amount)."""
        scope_kind = rule.get("scope_kind")
        window = rule.get("window")
        metric = rule.get("metric")
        if scope_kind not in ALL_SCOPE_KINDS or window not in ALL_WINDOWS \
                or metric not in VALID_METRICS:
            raise ValueError(
                f"out-of-vocabulary rule scope_kind={scope_kind!r} "
                f"window={window!r} metric={metric!r}"
            )
        # Reported by the usage view, never routed on.
        if scope_kind not in ENFORCEABLE_SCOPE_KINDS or window not in ENFORCEABLE_WINDOWS:
            return None
        amount = float(rule.get("amount") or 0.0)
        if amount <= 0:
            return None  # a zero/negative cap can't meaningfully block
        bounds = _window_bounds(window, now, zone)
        if bounds is None:
            return None
        start, end = bounds
        scope_value = rule.get("scope_value") or ""
        spend = await self._db.get_scope_spend(
            scope_kind=scope_kind, scope_value=scope_value,
            metric=metric, since_epoch=start,
        )
        ratio = spend / amount if amount > 0 else 0.0
        thresholds = tuple(
            float(t) for t in (rule.get("alert_thresholds") or []) if 0 < float(t) < 1
        )
        return RuleState(
            rule_id=str(rule.get("id")),
            scope_kind=scope_kind,
            scope_value=scope_value,
            metric=metric,
            window=window,
            amount=amount,
            thresholds=thresholds,
            webhook_url=rule.get("webhook_url"),
            window_start=start,
            window_end=end,
            spend=spend,
            ratio=ratio,
            over=spend >= amount,
        )

    # ── read-side gate (sync, hot path) ───────────────────────────────

    def blocked_scope_keys(self) -> set[tuple[str, str]]:
        """(scope_kind, scope_value) pairs currently over cap, computed on read.

        A rule blocks ONLY while the clock is inside its snapshotted window;
        past ``window_end`` it drops out automatically, so a rolled-over window
        un-blocks with no refresh and no traffic."""
        now = time.time()
        keys: set[tuple[str, str]] = set()
        for st in self._snapshot:
            if st.over and st.window_start <= now < st.window_end:
                keys.add((st.scope_kind, st.scope_value))
        return keys

    def blocked_runtime_ids(self, entries) -> set[str]:
        """Concrete runtime_ids to exclude from ``entries`` — the UNION of what
        every tripped rule blocks (a global rule blocks all; a provider rule all
        of its models; a model rule just itself)."""
        keys = self.blocked_scope_keys()
        if not keys:
            return set()
        global_blocked = ("global", "") in keys
        blocked: set[str] = set()
        for e in entries:
            rid = getattr(e, "runtime_id", None)
            if not rid:
                continue
            provider = getattr(e, "provider", None)
            if (
                global_blocked
                or ("provider", provider) in keys
                or ("model", rid) in keys
            ):
                blocked.add(rid)
        return blocked

    def filter_catalog(self, entries: list) -> list:
        """The method ``_enabled_catalog`` delegates to. Returns ``entries``
        unchanged when nothing is blocked (off-by-default is byte-identical),
        drops over-cap models otherwise, and NEVER returns an empty list when
        the input was non-empty.

        The subtlety is a GLOBAL cap interacting with a scope-specific one. A
        tripped global rule targets every model, so the raw union always empties
        the catalog — a naive "if it would empty, keep everything" would then let
        a global cap MASK a provider/model cap (the operator's DeepSeek provider
        brake would silently stop working the moment a global cap also tripped).
        So when nothing survives, we still exclude the models that are over their
        OWN (provider/model) cap and keep only the ones caught *solely* by the
        global aggregate — the individually-capped model stays excluded (the
        brake holds) while a $0 subscription model keeps the agent online. Only
        when even that leaves nothing do we refuse to enforce and log it.
        """
        self.maybe_schedule_refresh()
        keys = self.blocked_scope_keys()
        if not keys:
            if self._last_excluded_key is not None:
                self._last_excluded_key = None
            return entries
        global_over = ("global", "") in keys

        def _specific(e) -> bool:
            rid = getattr(e, "runtime_id", None)
            provider = getattr(e, "provider", None)
            return ("provider", provider) in keys or ("model", rid) in keys

        def _blocked(e) -> bool:
            return _specific(e) or global_over

        remaining = [e for e in entries if not _blocked(e)]
        if remaining:
            # Only reachable when no global rule is over (else all are blocked),
            # so the excluded set here is exactly the individually-capped models.
            self._log_excluded({e.runtime_id for e in entries if _blocked(e)})
            return remaining

        # Every model is blocked. Prefer to still exclude the individually-capped
        # ones and keep the models blocked ONLY by the global aggregate, so a
        # global cap never masks a scope-specific brake.
        only_global = [e for e in entries if not _specific(e)]
        if 0 < len(only_global) < len(entries):
            self._log_excluded({e.runtime_id for e in entries if _specific(e)})
            return only_global

        # Nothing left to route to without going fully offline (all models are
        # individually capped, or the only cap is a global one that can't be
        # routed around). Stay online and log that the cap was not enforced.
        self._log_would_empty({e.runtime_id for e in entries if _blocked(e)})
        return entries

    def _log_excluded(self, blocked: set[str]) -> None:
        key = frozenset(blocked)
        if key == self._last_excluded_key:
            return
        self._last_excluded_key = key
        self._last_would_empty_key = None
        elog("budget.scope_excluded", level="warning", blocked=sorted(blocked))

    def _log_would_empty(self, blocked: set[str]) -> None:
        key = frozenset(blocked)
        if key == self._last_would_empty_key:
            return
        self._last_would_empty_key = key
        elog(
            "budget.cap_not_enforced_would_empty_catalog",
            level="warning",
            blocked=sorted(blocked),
            note="all enabled models are over budget; leaving them enabled so "
            "the agent stays online (never leave zero enabled models)",
        )

    # ── refresh scheduling ────────────────────────────────────────────

    def maybe_schedule_refresh(self) -> None:
        """TTL backstop: if the snapshot is stale, kick an async refresh. Cheap
        (a monotonic subtraction); safe to call on every hot-path pass."""
        if time.monotonic() - self._last_refresh_monotonic >= self._ttl:
            self.schedule_refresh()

    def schedule_refresh(self) -> None:
        """Fire a background refresh if a loop is running. Coalesces: a request
        arriving during an in-flight refresh sets a pending flag so exactly one
        more runs afterwards (picking up the latest DB state)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop (sync/boot context) — a later hot-path call retries
        if self._refresh_task is not None and not self._refresh_task.done():
            self._refresh_pending = True
            return
        self._refresh_task = loop.create_task(self._refresh_runner())

    async def _refresh_runner(self) -> None:
        try:
            await self.refresh()
            while self._refresh_pending:
                self._refresh_pending = False
                await self.refresh()
        finally:
            self._refresh_task = None

    # ── alerts ────────────────────────────────────────────────────────

    def _maybe_fire_alerts(self, st: RuleState) -> None:
        """Emit a ``budget.alert`` (and fire the webhook) at each threshold
        crossing, once per threshold per window. The 1.0 cap event always fires;
        configured thresholds are added below it."""
        thresholds = sorted(set(st.thresholds) | {1.0})
        fired_map = self._alerts_fired.setdefault(st.rule_id, {})
        fired = fired_map.setdefault(st.window_start, set())
        for t in thresholds:
            if st.ratio + 1e-9 >= t and t not in fired:
                fired.add(t)
                self._emit_alert(st, t)

    def _prune_alert_state(self, snapshot: list[RuleState]) -> None:
        """Drop alert bookkeeping for windows that are no longer current so the
        de-dupe map does not grow without bound."""
        current = {st.rule_id: st.window_start for st in snapshot}
        for rule_id in list(self._alerts_fired.keys()):
            if rule_id not in current:
                del self._alerts_fired[rule_id]
                continue
            win = current[rule_id]
            self._alerts_fired[rule_id] = {
                k: v for k, v in self._alerts_fired[rule_id].items() if k == win
            }

    def _emit_alert(self, st: RuleState, threshold: float) -> None:
        capped = threshold >= 1.0
        elog(
            "budget.alert",
            level="warning" if capped else "info",
            budget_id=st.rule_id,
            scope_kind=st.scope_kind,
            scope=st.scope_value or "*",
            metric=st.metric,
            window=st.window,
            threshold=threshold,
            spend=round(st.spend, 6),
            limit=st.amount,
            ratio=round(st.ratio, 4),
            capped=capped,
        )
        if st.webhook_url:
            self._spawn_webhook(st, threshold, capped)

    def _spawn_webhook(self, st: RuleState, threshold: float, capped: bool) -> None:
        payload = {
            "event": "budget.alert",
            "budget_id": st.rule_id,
            "scope_kind": st.scope_kind,
            "scope": st.scope_value or "*",
            "metric": st.metric,
            "window": st.window,
            "threshold": threshold,
            "spend": round(st.spend, 6),
            "limit": st.amount,
            "ratio": round(st.ratio, 4),
            "capped": capped,
        }
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._fire_webhook(st.webhook_url, payload))

    async def _fire_webhook(self, url: str | None, payload: dict) -> None:
        if not url:
            return
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=_WEBHOOK_TIMEOUT_S)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    await resp.read()
        except Exception as e:  # noqa: BLE001 — a dead webhook never breaks a turn
            elog("budget.webhook_error", level="warning",
                 url=url[:120], error=str(e))


# ── shared usage view (REST + MCP) ────────────────────────────────────


def _public_rule(rule: dict) -> dict:
    """The fields of a rule the usage view echoes back."""
    return {
        "id": rule.get("id"),
        "scope_kind": rule.get("scope_kind"),
        "scope": rule.get("scope_value") or ("*" if rule.get("scope_kind") == "global" else ""),
        "scope_value": rule.get("scope_value") or "",
        "metric": rule.get("metric"),
        "window": rule.get("window"),
        "amount": rule.get("amount"),
        "alert_thresholds": rule.get("alert_thresholds"),
        "webhook_url": rule.get("webhook_url"),
        "enabled": rule.get("enabled"),
        "source": rule.get("source"),
    }


async def compute_budget_usage(db: Any, *, enabled_only: bool = False) -> list[dict]:
    """Per-rule current spend vs limit — the meter the app and the agent read.

    Computed FRESH from the DB (not the guard's cached snapshot) so it is always
    authoritative regardless of when the guard last refreshed. ``per_run``
    windows have no calendar boundary → ``spend`` is ``None``; ``task`` scopes
    and ``per_run`` windows report ``enforced: false`` (phase 2). The single
    cost path (``usage_log`` via ``get_scope_spend``) is reused — no second
    accounting.
    """
    rules = await db.list_budgets(enabled_only=enabled_only)
    now = time.time()
    zone = _agent_zone()
    out: list[dict] = []
    for rule in rules:
        base = _public_rule(rule)
        scope_kind = rule.get("scope_kind")
        window = rule.get("window")
        metric = rule.get("metric")
        enforced = scope_kind in ENFORCEABLE_SCOPE_KINDS and window in ENFORCEABLE_WINDOWS
        base["enforced"] = enforced
        bounds = _window_bounds(window, now, zone)
        if bounds is None:
            # per_run (or unknown) — no window to sum over.
            base.update({"spend": None, "ratio": None, "over": False,
                         "remaining": None, "window_start": None, "window_end": None})
            out.append(base)
            continue
        start, end = bounds
        try:
            spend = await db.get_scope_spend(
                scope_kind=scope_kind, scope_value=rule.get("scope_value") or "",
                metric=metric, since_epoch=start,
            )
        except Exception as e:  # noqa: BLE001
            base.update({"spend": None, "error": str(e), "window_start": start,
                         "window_end": end})
            out.append(base)
            continue
        amount = float(rule.get("amount") or 0.0)
        ratio = (spend / amount) if amount > 0 else None
        base.update({
            "spend": round(spend, 6),
            "ratio": round(ratio, 4) if ratio is not None else None,
            "over": bool(amount > 0 and spend >= amount),
            "remaining": round(max(0.0, amount - spend), 6) if amount > 0 else None,
            "window_start": start,
            "window_end": end,
        })
        out.append(base)
    return out
