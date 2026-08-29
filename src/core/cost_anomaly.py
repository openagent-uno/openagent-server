"""Per-run cost-anomaly alerting — page on REAL cost, not summed input tokens.

WHY THIS EXISTS
---------------
An agentic run makes many model calls (13 at worst on this runtime), and each
re-sends the growing conversation prefix. The dispatcher records ONE
``router.cost_recorded`` per run whose ``input_tokens`` is the SUM across those
calls — so a perfectly ordinary support turn shows a six-figure input-token
count (a real run logged 447,229) even though ~94% of it is prefix-cache READS
priced at a tiny fraction of a fresh token. Real cost was ~$0.018.

A cost alarm that reads that summed ``input_tokens`` as if it were one giant
prompt pages on nothing — a two-cent run trips a "447k-token prompt!" alert. So
this module bases the anomaly on signals that track ACTUAL spend:

* ``cost_usd`` — what the run really cost (cache-aware, per ``compute_cost``);
* non-cached input = ``input_tokens - cache_read_tokens`` — the tokens that were
  actually re-processed (a genuinely huge fresh prompt), NOT the cached prefix
  re-sent each step.

Either crossing its threshold pages; the raw summed ``input_tokens`` never does.
Defaults are set so a run costing a few cents can never page (see below).

ON BY DEFAULT (safe thresholds)
-------------------------------
``OPENAGENT_COST_ANOMALY_ENABLED`` defaults ON. The whole point of the fix is
that the thresholds are now safe: the known false-alarm run (447k summed input
tokens, ~94% cached, ~$0.018 real cost) is ~50x under the $1.00 cost floor and
well under the 200k non-cached-token floor, so it can never page — while a
genuinely expensive or genuinely uncached run does. ``evaluate`` on the hot path
is pure arithmetic and ``note_run`` swallows every error, so default-on costs a
turn nothing but a comparison. Set ``OPENAGENT_COST_ANOMALY_ENABLED=0`` to
disable, or raise the thresholds via env / config.

ALERT ROUTING
-------------
A genuine anomaly emits ``router.cost_anomaly`` (warning) to ``events.jsonl``
(the floor — always on when enabled) AND, when an alert webhook is configured,
POSTs a compact provider-neutral body to it so a human is actually paged. The
webhook is OPTIONAL (never hard-required): it uses the dedicated
``OPENAGENT_COST_ANOMALY_ALERT_WEBHOOK_URL`` or, unset, falls back to the shared
``OPENAGENT_QUALITY_DIGEST_ALERT_WEBHOOK_URL`` so an operator who already wired a
Slack/PagerDuty bridge for quality alerts gets cost anomalies on the same
channel. The POST mirrors ``budget_guard._fire_webhook`` — best-effort, tightly
bounded, fired off the turn as a background task, and swallowed on failure so a
dead endpoint can never break a turn. Everything is best-effort.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Optional
from urllib import request as _urllib_request

from src.core.execution_origin import create_server_only_task
from src.core.logging import elog

_ENABLED_ENV = "OPENAGENT_COST_ANOMALY_ENABLED"
_COST_USD_ENV = "OPENAGENT_COST_ANOMALY_COST_USD"
_UNCACHED_TOKENS_ENV = "OPENAGENT_COST_ANOMALY_UNCACHED_INPUT_TOKENS"
# Optional alert webhook. Dedicated var first; unset → the shared quality-digest
# paging webhook (reuse the same alert channel). Neither set → elog only.
_ALERT_WEBHOOK_ENV = "OPENAGENT_COST_ANOMALY_ALERT_WEBHOOK_URL"
_SHARED_ALERT_WEBHOOK_ENV = "OPENAGENT_QUALITY_DIGEST_ALERT_WEBHOOK_URL"
# Outbound alert-webhook timeout — mirrors ``budget_guard._WEBHOOK_TIMEOUT_S``: a
# slow/dead endpoint must never wedge the turn, so the POST is tightly bounded.
_WEBHOOK_TIMEOUT_S = 8.0

# Defaults, deliberately well ABOVE anything a normal cached agentic turn hits:
#   * $1.00 of REAL spend in a single run — a two-cent run (the false-alarm
#     case) is ~50x under this and can never page.
#   * 200,000 NON-CACHED input tokens — a genuinely large fresh prompt. The
#     447k-token/94%-cached run has only ~27k non-cached, comfortably under.
_DEFAULT_COST_USD = 1.00
_DEFAULT_UNCACHED_INPUT_TOKENS = 200_000


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    """True when per-run cost-anomaly alerting is switched on. Default ON — the
    thresholds are safe (the false-alarm run can't page), so a real runaway run
    surfaces out of the box. ``OPENAGENT_COST_ANOMALY_ENABLED=0`` disables it."""
    return _truthy(os.environ.get(_ENABLED_ENV, "1"))


def _cost_threshold() -> float:
    try:
        return max(0.0, float(os.environ.get(_COST_USD_ENV, "").strip()
                              or _DEFAULT_COST_USD))
    except ValueError:
        return _DEFAULT_COST_USD


def _uncached_threshold() -> int:
    try:
        return max(0, int(os.environ.get(_UNCACHED_TOKENS_ENV, "").strip()
                          or _DEFAULT_UNCACHED_INPUT_TOKENS))
    except ValueError:
        return _DEFAULT_UNCACHED_INPUT_TOKENS


def evaluate(
    *,
    cost_usd: Optional[float],
    input_tokens: int,
    cache_read_tokens: int = 0,
    output_tokens: int = 0,
    cost_threshold: Optional[float] = None,
    uncached_threshold: Optional[int] = None,
) -> Optional[dict[str, Any]]:
    """Return an anomaly dict when a run is genuinely expensive, else ``None``.

    Pure and side-effect-free (thresholds default from env only when not passed),
    so it is directly testable. A run trips when EITHER real ``cost_usd`` OR the
    NON-CACHED input crosses its threshold — never on raw summed ``input_tokens``,
    which is inflated by the cached prefix re-sent every agentic step.
    """
    cost_thr = _cost_threshold() if cost_threshold is None else max(0.0, float(cost_threshold))
    unc_thr = _uncached_threshold() if uncached_threshold is None else max(0, int(uncached_threshold))

    in_tok = int(input_tokens or 0)
    cache_tok = int(cache_read_tokens or 0)
    # Cache reads are a SUBSET of input; clamp so a mismatched counter can never
    # make "non-cached" go negative (or spuriously large).
    uncached = max(0, in_tok - min(cache_tok, in_tok))
    cost = float(cost_usd or 0.0)

    reasons: list[str] = []
    if cost >= cost_thr:
        reasons.append("cost_usd")
    if uncached >= unc_thr:
        reasons.append("uncached_input_tokens")
    if not reasons:
        return None
    return {
        "reasons": reasons,
        "cost_usd": round(cost, 6),
        "input_tokens": in_tok,
        "cache_read_tokens": cache_tok,
        "uncached_input_tokens": uncached,
        "output_tokens": int(output_tokens or 0),
        "cost_threshold": cost_thr,
        "uncached_input_threshold": unc_thr,
    }


# ── alert webhook (optional; provider-neutral) ────────────────────────────


def _alert_webhook_url() -> Optional[str]:
    """The alert webhook to page, or ``None``. The dedicated cost-anomaly var
    wins; unset, we reuse the quality-digest paging webhook so a single
    already-configured Slack/PagerDuty bridge receives cost anomalies too."""
    raw = os.environ.get(_ALERT_WEBHOOK_ENV, "").strip()
    if raw:
        return raw
    shared = os.environ.get(_SHARED_ALERT_WEBHOOK_ENV, "").strip()
    return shared or None


def _alert_summary(model: Optional[str], anomaly: dict[str, Any]) -> str:
    """One human-readable line for the webhook consumer."""
    return (
        f"cost anomaly on {model or 'unknown model'}: real cost "
        f"${anomaly.get('cost_usd')} / {anomaly.get('uncached_input_tokens')} "
        f"non-cached input tokens tripped {'+'.join(anomaly.get('reasons') or [])}"
    )


def _alert_payload(session_id: Optional[str], model: Optional[str],
                   anomaly: dict[str, Any]) -> dict[str, Any]:
    """Compact, provider-neutral JSON body — same shape as the quality-digest
    alert payload (event / severity / summary / counts / ts / source)."""
    return {
        "event": "router.cost_anomaly",
        "severity": "warning",
        "summary": _alert_summary(model, anomaly),
        "session_id": session_id,
        "model": model,
        "counts": dict(anomaly),
        "ts": time.time(),
        "source": "cost_anomaly",
    }


async def _fire_webhook(url: str, payload: dict[str, Any]) -> None:
    """POST one anomaly to the alert webhook. Best-effort, tightly bounded — a
    dead endpoint must never affect a turn. Mirrors ``budget_guard._fire_webhook``."""
    try:
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=_WEBHOOK_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                await resp.read()
    except Exception as e:  # noqa: BLE001 — a dead webhook never breaks a turn
        elog("router.cost_anomaly_webhook_error", level="warning",
             url=url[:120], error=str(e))


def _post_webhook_sync(url: str, payload: dict[str, Any]) -> None:
    """Blocking, tightly-bounded POST — used only when there is NO running loop
    (a sync/ad-hoc caller), so the anomaly still pages instead of being dropped."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req = _urllib_request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        with _urllib_request.urlopen(req, timeout=_WEBHOOK_TIMEOUT_S) as resp:
            resp.read()
    except Exception as e:  # noqa: BLE001 — a dead webhook never breaks a turn
        elog("router.cost_anomaly_webhook_error", level="warning",
             url=url[:120], error=str(e))


def _page_webhook(session_id: Optional[str], model: Optional[str],
                  anomaly: dict[str, Any]) -> None:
    """Fire the OPTIONAL alert webhook for a genuine anomaly. No-op when no URL
    is configured — the webhook is never hard-required; the ``elog`` is the floor.

    Fired off the turn as a background task on the running loop (``note_run`` is
    called synchronously on the event loop from the dispatcher); a blocking POST
    is used only when there is no loop at all."""
    url = _alert_webhook_url()
    if not url:
        return
    payload = _alert_payload(session_id, model, anomaly)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None:
        create_server_only_task(
            _fire_webhook(url, payload), name="cost-anomaly-webhook",
        )
    else:
        _post_webhook_sync(url, payload)


def note_run(
    *,
    session_id: Optional[str],
    model: Optional[str],
    cost_usd: Optional[float],
    input_tokens: int,
    cache_read_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Emit ``router.cost_anomaly`` (warning) when a run is genuinely expensive,
    and page the optional alert webhook.

    No-op when disabled. Never raises — a monitoring miss must cost an alert,
    never a turn. The ``elog`` is the always-on floor; the webhook POST is
    additive and best-effort (a dead/unset endpoint changes nothing)."""
    if not enabled():
        return
    try:
        anomaly = evaluate(
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            cache_read_tokens=cache_read_tokens,
            output_tokens=output_tokens,
        )
        if anomaly is None:
            return
        elog("router.cost_anomaly", level="warning",
             session_id=session_id, model=model, **anomaly)
        # Additionally page a human via the optional webhook (if one is set).
        _page_webhook(session_id, model, anomaly)
    except Exception:  # noqa: BLE001 — anomaly telemetry must never break a turn
        return
