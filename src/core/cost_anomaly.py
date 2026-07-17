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

OFF BY DEFAULT / §17
--------------------
``OPENAGENT_COST_ANOMALY_ENABLED`` defaults OFF: a deployment that never turns
it on is byte-identical (``note_run`` returns before doing anything). The pure
:func:`evaluate` is always callable for tests / ad-hoc checks. Everything is
best-effort — a monitoring miss must never break a turn.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from src.core.logging import elog

_ENABLED_ENV = "OPENAGENT_COST_ANOMALY_ENABLED"
_COST_USD_ENV = "OPENAGENT_COST_ANOMALY_COST_USD"
_UNCACHED_TOKENS_ENV = "OPENAGENT_COST_ANOMALY_UNCACHED_INPUT_TOKENS"

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
    """True when per-run cost-anomaly alerting is switched on. Default OFF."""
    return _truthy(os.environ.get(_ENABLED_ENV, "0"))


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


def note_run(
    *,
    session_id: Optional[str],
    model: Optional[str],
    cost_usd: Optional[float],
    input_tokens: int,
    cache_read_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Emit ``router.cost_anomaly`` (warning) when a run is genuinely expensive.

    No-op when disabled (§17). Never raises — a monitoring miss must cost an
    alert, never a turn."""
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
    except Exception:  # noqa: BLE001 — anomaly telemetry must never break a turn
        return
