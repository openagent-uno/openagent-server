"""Per-run cost-anomaly tests — page on REAL cost, never on summed input tokens.

Pin the fix for the misleading "447,229 input tokens!" alert: a run whose
input-token count is huge but is ~94% prefix-cache READS (real cost a couple of
cents) must NOT trip, while a genuinely expensive or genuinely uncached run
must. Pure-unit: the pure ``evaluate`` + the ``note_run`` emitter with ``elog``
captured. No dispatcher/pool/gateway.
"""
from __future__ import annotations

import contextlib
import os

from ._framework import TestContext, test


@contextlib.contextmanager
def _env(**kw):
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
def _capture():
    import src.core.cost_anomaly as ca

    events: list[tuple[str, dict]] = []
    orig = ca.elog
    ca.elog = lambda name, level="info", **kw: events.append((name, kw))
    try:
        yield events
    finally:
        ca.elog = orig


# The real false-alarm run, from production: 447,229 summed input tokens across
# ~13 agentic steps, 94% of them cached prefix reads, real cost ~$0.018.
_CACHED_INPUT = 447_229
_CACHED_READS = 420_395  # ~94%
_CACHED_COST = 0.018


@test("cost_anomaly", "a high-summed-token, mostly-cached, cheap run does NOT trip")
async def t_cached_run_no_alarm(ctx: TestContext) -> None:
    import src.core.cost_anomaly as ca

    # The exact shape that false-paged before: raw input_tokens is six figures,
    # but non-cached input is ~27k and real cost is under 2 cents.
    got = ca.evaluate(
        cost_usd=_CACHED_COST,
        input_tokens=_CACHED_INPUT,
        cache_read_tokens=_CACHED_READS,
        output_tokens=1200,
    )
    assert got is None, f"a $0.018 mostly-cached run tripped the anomaly alert: {got}"


@test("cost_anomaly", "a genuinely EXPENSIVE run trips (real cost over threshold)")
async def t_expensive_run_trips(ctx: TestContext) -> None:
    import src.core.cost_anomaly as ca

    got = ca.evaluate(cost_usd=2.50, input_tokens=120_000, cache_read_tokens=0)
    assert got is not None and "cost_usd" in got["reasons"], got
    assert got["cost_usd"] == 2.5


@test("cost_anomaly", "a genuinely UNCACHED large prompt trips on non-cached input")
async def t_uncached_run_trips(ctx: TestContext) -> None:
    import src.core.cost_anomaly as ca

    # Cheap per-token but a real 300k fresh prompt (no cache) — a genuine
    # oversized-context anomaly worth a look, unlike the cached case.
    got = ca.evaluate(cost_usd=0.30, input_tokens=300_000, cache_read_tokens=0)
    assert got is not None and "uncached_input_tokens" in got["reasons"], got
    assert got["uncached_input_tokens"] == 300_000


@test("cost_anomaly", "raw summed input_tokens alone never trips (the actual defect)")
async def t_summed_input_alone_never_trips(ctx: TestContext) -> None:
    import src.core.cost_anomaly as ca

    # A pathological 5M summed input tokens, but ALL of it cached and ~free:
    # the old signal would scream, the new one stays silent.
    got = ca.evaluate(cost_usd=0.05, input_tokens=5_000_000,
                      cache_read_tokens=5_000_000)
    assert got is None, f"raw summed input_tokens tripped the alert: {got}"


@test("cost_anomaly", "cache_read is clamped so non-cached can't go negative")
async def t_cache_read_clamp(ctx: TestContext) -> None:
    import src.core.cost_anomaly as ca

    # A mismatched counter reporting more cache reads than input must not
    # underflow non-cached input into a spurious trip (or a negative).
    got = ca.evaluate(cost_usd=0.01, input_tokens=1000, cache_read_tokens=999_999,
                      uncached_threshold=1)
    assert got is None or "uncached_input_tokens" not in got.get("reasons", []), got


@test("cost_anomaly", "note_run: OFF is a no-op; ON emits router.cost_anomaly only when anomalous")
async def t_note_run_gating(ctx: TestContext) -> None:
    import src.core.cost_anomaly as ca

    # Disabled → nothing, even for a genuinely expensive run (§17 byte-identical).
    with _env(OPENAGENT_COST_ANOMALY_ENABLED="0"), _capture() as ev:
        ca.note_run(session_id="s", model="deepseek:x", cost_usd=99.0,
                    input_tokens=10, cache_read_tokens=0)
    assert ev == [], f"disabled monitor emitted: {ev}"

    # Enabled + cheap cached run → still nothing.
    with _env(OPENAGENT_COST_ANOMALY_ENABLED="1"), _capture() as ev:
        ca.note_run(session_id="s", model="local:claude-sub", cost_usd=_CACHED_COST,
                    input_tokens=_CACHED_INPUT, cache_read_tokens=_CACHED_READS)
    assert ev == [], f"cheap cached run paged: {ev}"

    # Enabled + expensive run → exactly one router.cost_anomaly warning.
    with _env(OPENAGENT_COST_ANOMALY_ENABLED="1"), _capture() as ev:
        ca.note_run(session_id="s7", model="deepseek:x", cost_usd=3.0,
                    input_tokens=100_000, cache_read_tokens=0)
    anomalies = [kw for name, kw in ev if name == "router.cost_anomaly"]
    assert len(anomalies) == 1, ev
    assert anomalies[0]["session_id"] == "s7" and "cost_usd" in anomalies[0]["reasons"]


@test("cost_anomaly", "thresholds are configurable via env")
async def t_thresholds_configurable(ctx: TestContext) -> None:
    import src.core.cost_anomaly as ca

    # Tighten the cost floor to a cent — now the cached run's $0.018 DOES trip,
    # proving the knob is live (default 1.00 keeps it silent).
    with _env(OPENAGENT_COST_ANOMALY_COST_USD="0.01"):
        got = ca.evaluate(cost_usd=_CACHED_COST, input_tokens=_CACHED_INPUT,
                          cache_read_tokens=_CACHED_READS)
    assert got is not None and "cost_usd" in got["reasons"], got
    assert got["cost_threshold"] == 0.01
