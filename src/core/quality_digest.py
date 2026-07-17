"""Scheduled quality DIGEST + ALERTING — the review-only-what-matters layer.

``quality_monitor.py`` records per-turn signals (``quality.score``,
``recall.metric``) and ``aggregate()`` sums them on read. This module adds the
periodic, push side: a background loop that every ``interval_hours``

  * emits a ``quality.digest`` event — avg score, verdict breakdown, recall
    used/hit-rate + timeout-rate, turn count, cost, AND the ``session_id``s that
    scored ``bad`` or were ``fabrication``-flagged, so a human reviews ONLY those
    instead of skimming every reply; and
  * emits ``quality.alert`` events (warning level) when a threshold trips: avg
    quality drops below a floor, any fabrication is flagged, the recall-timeout
    rate is too high, or the embedder looks DOWN — inferred from a spike in
    ``semantic.embed_error`` (so this doubles as remote-ollama/Mac-Mini health
    monitoring; no separate healthcheck); and, when
    ``OPENAGENT_QUALITY_DIGEST_ALERT_WEBHOOK_URL`` is set, ALSO POSTs each newly
    active alert to that generic webhook (Slack/Telegram/PagerDuty bridge) so a
    human actually gets paged instead of the alert dying in a log line. The POST
    mirrors ``budget_guard._fire_webhook`` — best-effort, tightly bounded, and it
    can never break the digest loop — and is edge-triggered so a persisting
    condition is not re-POSTed every cycle. Unset → byte-identical to before
    (``elog`` only).

Same shape as ``learning.curator`` / ``memory.semantic_index_builder``: a
``start()`` returning a task, a **no-op when disabled** (§17) — the digest
defaults ON whenever the quality monitor is on, so enabling the monitor gets you
the digest for free, and both are byte-identical to before when the monitor is
off. Reuses ``quality_monitor.aggregate()`` and the one-reverse-scan
``iter_events_reverse`` primitive, so there is no second store.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Optional
from urllib import request as _urllib_request

from src.core.logging import elog, iter_events_reverse
from src.core import quality_monitor

# ── config (env-driven, set by ``_build_agent`` from ``quality_monitor.digest.*``) ──
_ENABLED_ENV = "OPENAGENT_QUALITY_DIGEST_ENABLED"
_INTERVAL_ENV = "OPENAGENT_QUALITY_DIGEST_INTERVAL_HOURS"
_MIN_SCORE_ALERT_ENV = "OPENAGENT_QUALITY_DIGEST_MIN_SCORE_ALERT"
_RECALL_TIMEOUT_ALERT_ENV = "OPENAGENT_QUALITY_DIGEST_RECALL_TIMEOUT_ALERT"
_EMBED_ERROR_ALERT_ENV = "OPENAGENT_QUALITY_DIGEST_EMBED_ERROR_ALERT"
# Optional generic alert webhook. When set, each newly active ``quality.alert``
# is ALSO POSTed here (provider-neutral — a Slack/Telegram/PagerDuty bridge
# consumes it). Env-driven like every other knob above; unset → elog only.
_ALERT_WEBHOOK_ENV = "OPENAGENT_QUALITY_DIGEST_ALERT_WEBHOOK_URL"

_DEFAULT_INTERVAL_HOURS = 24.0
_DEFAULT_MIN_SCORE_ALERT = 0.70          # alert if avg judged score dips below this
_DEFAULT_RECALL_TIMEOUT_ALERT = 0.25     # alert if >25% of turns time out on recall
_DEFAULT_EMBED_ERROR_ALERT = 10          # alert if > this many embed errors in the window
_MIN_INTERVAL_SECONDS = 300.0            # floor so a misconfig can't tight-loop
_MAX_FLAGGED = 50                        # bound the flagged-session list in one event
# Outbound alert-webhook timeout — mirrors ``budget_guard._WEBHOOK_TIMEOUT_S``: a
# slow/dead endpoint must never wedge the digest, so the POST is tightly bounded.
_WEBHOOK_TIMEOUT_S = 8.0


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    """True when the digest should run. Defaults to the quality monitor's own
    switch (enabling the monitor gives you the digest), unless
    ``OPENAGENT_QUALITY_DIGEST_ENABLED`` explicitly overrides it."""
    raw = os.environ.get(_ENABLED_ENV)
    if raw is None or raw.strip() == "":
        return quality_monitor.enabled()
    return _truthy(raw)


def _interval_seconds() -> float:
    raw = os.environ.get(_INTERVAL_ENV, "").strip()
    if not raw:
        return _DEFAULT_INTERVAL_HOURS * 3600.0
    try:
        return max(_MIN_INTERVAL_SECONDS, float(raw) * 3600.0)
    except ValueError:
        return _DEFAULT_INTERVAL_HOURS * 3600.0


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _scan_extras(window_seconds: float, path: Any = None) -> dict[str, Any]:
    """One reverse scan for what ``aggregate()`` doesn't track: the bad/
    fabrication ``session_id``s (for the review list), the recall-timeout count
    (``auto_recall.hook_error``), the embed-error count (``semantic.embed_error``,
    the embedder-down signal), and the turn count (``event.done``)."""
    since = time.time() - max(0.0, window_seconds)
    flagged: list[dict] = []
    seen: set = set()
    recall_timeouts = 0
    embed_errors = 0
    turns = 0
    for e in iter_events_reverse(since=since, path=path):
        ev = e.get("event")
        if ev == "quality.score":
            if e.get("verdict") == "bad" or e.get("fabrication"):
                sid = e.get("session_id")
                if sid and sid not in seen and len(flagged) < _MAX_FLAGGED:
                    seen.add(sid)
                    flagged.append({
                        "session_id": sid,
                        "verdict": e.get("verdict"),
                        "fabrication": bool(e.get("fabrication")),
                        "score": e.get("score"),
                        "rationale": (e.get("rationale") or "")[:200],
                    })
        elif ev == "auto_recall.hook_error":
            recall_timeouts += 1
        elif ev == "semantic.embed_error":
            embed_errors += 1
        elif ev == "event.done":
            turns += 1
    return {
        "flagged": flagged,
        "recall_timeouts": recall_timeouts,
        "embed_errors": embed_errors,
        "turns": turns,
    }


def build_digest(window_seconds: float, path: Any = None) -> dict[str, Any]:
    """Assemble the digest dict (``aggregate()`` + the extra scan). Pure — no
    emit — so it can be tested and inspected directly."""
    agg = quality_monitor.aggregate(window_seconds, path=path)
    extras = _scan_extras(window_seconds, path=path)
    turns = extras["turns"] or agg["usage"]["turns"]
    timeout_rate = round(extras["recall_timeouts"] / turns, 3) if turns else 0.0
    return {
        "window_hours": round(window_seconds / 3600.0, 2),
        "quality": agg["quality"],
        "usage": agg["usage"],
        "recall": {
            **agg["recall"],
            "timeouts": extras["recall_timeouts"],
            "timeout_rate": timeout_rate,
        },
        "embed_errors": extras["embed_errors"],
        "flagged_sessions": extras["flagged"],
    }


def evaluate_alerts(digest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return an alert dict for each tripped threshold (empty when healthy). Pure."""
    alerts: list[dict] = []

    avg = digest["quality"]["avg_score"]
    floor = _float_env(_MIN_SCORE_ALERT_ENV, _DEFAULT_MIN_SCORE_ALERT)
    if avg is not None and avg < floor:
        alerts.append({"kind": "quality_low", "avg_score": avg, "threshold": floor,
                       "judged": digest["quality"]["judged"]})

    fab = digest["quality"]["fabrication_flagged"]
    if fab:
        alerts.append({"kind": "fabrication", "count": fab})

    rt = digest["recall"]["timeout_rate"]
    rt_thr = _float_env(_RECALL_TIMEOUT_ALERT_ENV, _DEFAULT_RECALL_TIMEOUT_ALERT)
    # Require an absolute floor of 3 timeouts so a 1/2-turn blip doesn't alert.
    if rt is not None and rt > rt_thr and digest["recall"]["timeouts"] >= 3:
        alerts.append({"kind": "recall_timeouts", "rate": rt, "threshold": rt_thr,
                       "count": digest["recall"]["timeouts"]})

    ee = digest["embed_errors"]
    ee_thr = _int_env(_EMBED_ERROR_ALERT_ENV, _DEFAULT_EMBED_ERROR_ALERT)
    if ee > ee_thr:
        alerts.append({"kind": "embedder_down", "embed_errors": ee, "threshold": ee_thr})

    return alerts


# ── alert webhook (optional; provider-neutral) ────────────────────────────
# Edge-trigger de-dupe: the alert ``kind``s we have already POSTed while they
# stay continuously active. A persisting condition (e.g. avg quality stuck low)
# thus pages ONCE, not every cycle; a kind that clears and later recurs pages
# again. The ``elog`` side is untouched — it still fires every cycle — so the
# unset-webhook behaviour is byte-identical to before.
_alerted_kinds: set[str] = set()


def _alert_webhook_url() -> Optional[str]:
    raw = os.environ.get(_ALERT_WEBHOOK_ENV, "").strip()
    return raw or None


def _alert_summary(a: dict[str, Any]) -> str:
    """One human-readable line per alert kind (for the webhook consumer)."""
    kind = a.get("kind")
    if kind == "quality_low":
        return (f"avg quality {a.get('avg_score')} below floor {a.get('threshold')} "
                f"({a.get('judged')} judged)")
    if kind == "fabrication":
        return f"{a.get('count')} fabrication-flagged repl(y/ies) this window"
    if kind == "recall_timeouts":
        return (f"recall timeout rate {a.get('rate')} ({a.get('count')} timeouts) "
                f"over threshold {a.get('threshold')}")
    if kind == "embedder_down":
        return (f"embedder likely DOWN: {a.get('embed_errors')} embed errors "
                f"(> {a.get('threshold')})")
    return f"quality alert: {kind}"


def _alert_payload(alert: dict[str, Any], digest: dict[str, Any]) -> dict[str, Any]:
    """Compact, provider-neutral JSON body: alert type, severity, summary,
    counts, and the timestamp source (wall clock + window + emitter)."""
    return {
        "event": "quality.alert",
        "kind": alert.get("kind"),
        "severity": "warning",
        "summary": _alert_summary(alert),
        "counts": {k: v for k, v in alert.items() if k != "kind"},
        "window_hours": digest.get("window_hours"),
        "ts": time.time(),
        "source": "quality_digest",
    }


def _post_webhook(url: str, payload: dict[str, Any]) -> None:
    """Best-effort POST of one alert. SYNCHRONOUS on purpose: ``run_once`` runs
    in a worker thread (``asyncio.to_thread``) with no running event loop, so
    ``budget_guard``'s ``loop.create_task(_fire_webhook)`` pattern does not apply
    here — a blocking POST with a tight timeout stalls only the digest worker
    thread (interval apart), never the event loop. A dead endpoint must NEVER
    break the digest, so every failure is swallowed and logged (like the budget
    one)."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req = _urllib_request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urllib_request.urlopen(req, timeout=_WEBHOOK_TIMEOUT_S) as resp:
            resp.read()
    except Exception as e:  # noqa: BLE001 — a dead webhook never breaks the digest
        elog("quality.webhook_error", level="warning", url=url[:120], error=str(e))


def _dispatch_alert_webhooks(alerts: list[dict[str, Any]], digest: dict[str, Any]) -> None:
    """POST the newly-active alerts to the configured webhook, if any. Unset →
    no POST and no state change (behaviour unchanged). Edge-triggered so a
    persisting condition is not re-POSTed every cycle."""
    url = _alert_webhook_url()
    if not url:
        return
    current = {a["kind"] for a in alerts}
    new_kinds = current - _alerted_kinds
    for a in alerts:
        if a["kind"] in new_kinds:
            _post_webhook(url, _alert_payload(a, digest))
    # Remember exactly what is active now: a cleared kind drops out (so it can
    # re-fire on recurrence), a still-active kind stays (so it is not re-POSTed).
    _alerted_kinds.clear()
    _alerted_kinds.update(current)


def run_once(window_seconds: Optional[float] = None,
             path: Any = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build + emit one ``quality.digest`` and any ``quality.alert`` events.
    Returns ``(digest, alerts)``. Safe to call ad-hoc (e.g. to verify without
    waiting for the interval)."""
    ws = window_seconds if window_seconds is not None else _interval_seconds()
    digest = build_digest(ws, path=path)
    q, u, r = digest["quality"], digest["usage"], digest["recall"]
    elog("quality.digest",
         window_hours=digest["window_hours"],
         judged=q["judged"], avg_score=q["avg_score"], verdicts=q["verdicts"],
         fabrication_flagged=q["fabrication_flagged"],
         turns=u["turns"], cost_usd=u["cost_usd"],
         recall_used_rate=r["used_rate"], recall_hit_rate=r["hit_rate"],
         recall_timeout_rate=r["timeout_rate"], embed_errors=digest["embed_errors"],
         flagged_count=len(digest["flagged_sessions"]),
         flagged_sessions=[f["session_id"] for f in digest["flagged_sessions"]])
    alerts = evaluate_alerts(digest)
    for a in alerts:
        elog("quality.alert", level="warning", **a)
    # In addition to the elog above, page a human via the optional webhook.
    _dispatch_alert_webhooks(alerts, digest)
    return digest, alerts


async def _loop() -> None:
    # Sleep first so the initial digest summarises a window with real traffic,
    # not an empty just-booted one.
    while True:
        try:
            await asyncio.sleep(_interval_seconds())
        except asyncio.CancelledError:
            break
        if not enabled():
            continue  # re-check each cycle so a live toggle takes effect
        try:
            await asyncio.to_thread(run_once)
        except asyncio.CancelledError:
            break
        except Exception as exc:  # noqa: BLE001 — a digest error must not kill the loop
            elog("quality.digest_error", level="warning",
                 error=str(exc) or type(exc).__name__)


def start() -> Optional[asyncio.Task]:
    """Start the periodic digest loop, or ``None`` when disabled / no loop.

    Cheap to call unconditionally: returns ``None`` (paying nothing) when the
    quality monitor is off, exactly like the curator / semantic-index builder.
    """
    if not enabled():
        return None
    try:
        return asyncio.create_task(_loop())
    except RuntimeError:
        return None  # no running loop (e.g. a unit test calling start() directly)
