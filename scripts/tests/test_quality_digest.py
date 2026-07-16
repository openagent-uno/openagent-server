"""Quality-digest tests — the scheduled push side of observability.

Pin the seams: the digest summarises quality+recall+cost AND surfaces the
bad/fabrication ``session_id``s to review; alerts fire on a real threshold breach
and stay silent when healthy; ``run_once`` emits ``quality.digest`` (+
``quality.alert``); and the layer defaults to the monitor's own switch and is a
no-op when off. Pure-unit: a temp events.jsonl, ``elog`` captured, no
scheduler/network.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import time
from pathlib import Path

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
def _events_file(rows):
    fd, name = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    p = Path(name)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    try:
        yield p
    finally:
        p.unlink(missing_ok=True)


@contextlib.contextmanager
def _capture():
    import src.core.quality_digest as qd

    events: list[tuple[str, dict]] = []
    orig = qd.elog
    qd.elog = lambda name, level="info", **kw: events.append((name, kw))
    try:
        yield events
    finally:
        qd.elog = orig


def _rows(now):
    """A window with 3 judged (good, bad, bad+fabrication), 2 recall.metric,
    4 completed turns, 1 recall timeout, 2 embed errors, cost."""
    return [
        {"ts": now, "event": "quality.score", "session_id": "s-good", "score": 0.9,
         "verdict": "good", "fabrication": False},
        {"ts": now, "event": "quality.score", "session_id": "s-bad", "score": 0.3,
         "verdict": "bad", "fabrication": False, "rationale": "quoted the wrong refund policy"},
        {"ts": now, "event": "quality.score", "session_id": "s-fab", "score": 0.4,
         "verdict": "bad", "fabrication": True, "rationale": "invented an order id"},
        {"ts": now, "event": "recall.metric", "used": True, "hits": 2, "top_score": 0.8},
        {"ts": now, "event": "recall.metric", "used": True, "hits": 1, "top_score": 0.7},
        {"ts": now, "event": "router.cost_recorded", "cost_usd": 0.02,
         "input_tokens": 200, "output_tokens": 40},
        {"ts": now, "event": "event.done"},
        {"ts": now, "event": "event.done"},
        {"ts": now, "event": "event.done"},
        {"ts": now, "event": "event.done"},
        {"ts": now, "event": "auto_recall.hook_error"},
        {"ts": now, "event": "semantic.embed_error"},
        {"ts": now, "event": "semantic.embed_error"},
    ]


@test("quality", "digest summarises quality/recall/cost + lists ONLY flagged sessions")
async def t_digest_summary(ctx: TestContext) -> None:
    import src.core.quality_digest as qd

    now = time.time()
    with _events_file(_rows(now)) as p:
        d = qd.build_digest(3600, path=p)
    assert d["quality"]["judged"] == 3, d
    assert d["quality"]["verdicts"] == {"good": 1, "warn": 0, "bad": 2}, d
    assert d["quality"]["fabrication_flagged"] == 1, d
    assert d["usage"]["turns"] == 1 and abs(d["usage"]["cost_usd"] - 0.02) < 1e-9, d
    assert d["recall"]["timeouts"] == 1 and d["recall"]["timeout_rate"] == 0.25, d  # 1/4 turns
    assert d["embed_errors"] == 2, d
    flagged = {f["session_id"] for f in d["flagged_sessions"]}
    assert flagged == {"s-bad", "s-fab"}, d  # the good turn is NOT in the review list
    fab = next(f for f in d["flagged_sessions"] if f["session_id"] == "s-fab")
    assert fab["fabrication"] is True and "order id" in fab["rationale"], fab


@test("quality", "alerts fire on breach, silent when healthy")
async def t_alerts(ctx: TestContext) -> None:
    import src.core.quality_digest as qd

    now = time.time()
    # avg (0.9+0.3+0.4)/3 = 0.533 < 0.7 floor; 1 fabrication; only 1 timeout
    # (< the 3-count floor) so NO recall alert; 2 embed errors (< 10) so NO embedder alert.
    with _events_file(_rows(now)) as p, _env(OPENAGENT_QUALITY_DIGEST_MIN_SCORE_ALERT="0.7"):
        d = qd.build_digest(3600, path=p)
        alerts = qd.evaluate_alerts(d)
    kinds = {a["kind"] for a in alerts}
    assert "quality_low" in kinds, alerts
    assert "fabrication" in kinds, alerts
    assert "recall_timeouts" not in kinds, "1 timeout must not alert (below the 3-count floor)"
    assert "embedder_down" not in kinds, alerts

    healthy = [
        {"ts": now, "event": "quality.score", "session_id": "s1", "score": 0.95,
         "verdict": "good", "fabrication": False},
        {"ts": now, "event": "event.done"},
    ]
    with _events_file(healthy) as p:
        d = qd.build_digest(3600, path=p)
        assert qd.evaluate_alerts(d) == [], d


@test("quality", "recall-timeout + embedder-down alerts trip at their thresholds")
async def t_alerts_recall_embed(ctx: TestContext) -> None:
    import src.core.quality_digest as qd

    now = time.time()
    rows = [{"ts": now, "event": "event.done"} for _ in range(10)]
    rows += [{"ts": now, "event": "auto_recall.hook_error"} for _ in range(5)]   # 5/10=0.5 >0.25, count 5>=3
    rows += [{"ts": now, "event": "semantic.embed_error"} for _ in range(15)]    # 15 > 10
    with _events_file(rows) as p:
        d = qd.build_digest(3600, path=p)
        alerts = qd.evaluate_alerts(d)
    kinds = {a["kind"] for a in alerts}
    assert "recall_timeouts" in kinds and "embedder_down" in kinds, alerts


@test("quality", "run_once emits quality.digest and any quality.alert")
async def t_run_once_emits(ctx: TestContext) -> None:
    import src.core.quality_digest as qd

    now = time.time()
    with _events_file(_rows(now)) as p, _capture() as ev, \
            _env(OPENAGENT_QUALITY_DIGEST_MIN_SCORE_ALERT="0.7"):
        qd.run_once(3600, path=p)
    names = [e[0] for e in ev]
    assert "quality.digest" in names, ev
    dg = next(kw for n, kw in ev if n == "quality.digest")
    assert dg["judged"] == 3 and dg["flagged_count"] == 2, dg
    assert set(dg["flagged_sessions"]) == {"s-bad", "s-fab"}, dg
    alert_kinds = {kw["kind"] for n, kw in ev if n == "quality.alert"}
    assert "quality_low" in alert_kinds and "fabrication" in alert_kinds, ev


@test("quality", "digest defaults to the monitor switch; start() no-op when off")
async def t_enabled_default_and_start(ctx: TestContext) -> None:
    import src.core.quality_digest as qd

    with _env(OPENAGENT_QUALITY_DIGEST_ENABLED=None, OPENAGENT_QUALITY_MONITOR_ENABLED="1"):
        assert qd.enabled() is True
    with _env(OPENAGENT_QUALITY_DIGEST_ENABLED=None, OPENAGENT_QUALITY_MONITOR_ENABLED="0"):
        assert qd.enabled() is False
        assert qd.start() is None  # off → no task, pays nothing
    with _env(OPENAGENT_QUALITY_DIGEST_ENABLED="0", OPENAGENT_QUALITY_MONITOR_ENABLED="1"):
        assert qd.enabled() is False  # explicit override wins over the monitor switch
    # enabled → a real task, which we cancel so the test leaks nothing.
    with _env(OPENAGENT_QUALITY_DIGEST_ENABLED="1"):
        t = qd.start()
        assert t is not None
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
