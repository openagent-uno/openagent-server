"""Quality-monitor tests — the correctness half of observability.

Pin the seams that make it safe to ship: OFF is a true no-op (an existing
deployment is byte-identical), sampling is deterministic and rate-bounded, the
LLM-judge parses tolerantly and emits ``quality.score``, gating skips trivial/
unsampled turns, and the aggregate sums quality+cost+recall out of a temp
events.jsonl. Pure-unit: a fake judge model, ``elog`` monkeypatched to capture
events, no gateway/pool/network.
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from ._framework import TestContext, test


@contextlib.contextmanager
def _env(**kw):
    """Set env vars for the block, restoring prior values (None = unset)."""
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
    """Capture quality_monitor's structured events as (name, kwargs) tuples."""
    import src.core.quality_monitor as qm

    events: list[tuple[str, dict]] = []
    orig = qm.elog
    qm.elog = lambda name, level="info", **kw: events.append((name, kw))
    try:
        yield events
    finally:
        qm.elog = orig


class _FakeModel:
    """Stand-in provider: records calls, returns a fixed ``.content``."""

    model = "fake:judge"

    def __init__(self, content: str):
        self._content = content
        self.calls: list[tuple] = []

    async def generate(self, messages, system=None, session_id=None):
        self.calls.append((messages, system, session_id))
        return SimpleNamespace(content=self._content)


def _agent(model):
    return SimpleNamespace(model=model, _providers_config=[], _db=None)


@test("quality", "OFF is a true no-op — no events, no judge call")
async def t_disabled_noop(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm

    fm = _FakeModel('{"score":1.0}')
    with _env(OPENAGENT_QUALITY_MONITOR_ENABLED="0"), _capture() as ev:
        qm.note_recall("s", used=True, hits=2, top_score=0.9)
        await qm.maybe_score_turn(_agent(fm), "s", "hi", "a" * 100)
        qm.spawn_scoring(_agent(fm), "s", "hi", "a" * 100)
    assert ev == [], f"disabled monitor emitted events: {ev}"
    assert fm.calls == [], "disabled monitor called the judge"


@test("quality", "sampling is deterministic + rate-bounded (0 never, 1 always)")
async def t_sampling(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm

    with _env(OPENAGENT_QUALITY_MONITOR_SAMPLE_RATE="0"):
        assert not qm.should_sample("s", "resp")
    with _env(OPENAGENT_QUALITY_MONITOR_SAMPLE_RATE="1"):
        assert qm.should_sample("s", "resp")
    with _env(OPENAGENT_QUALITY_MONITOR_SAMPLE_RATE="0.5"):
        a = qm.should_sample("sess", "resp")
        b = qm.should_sample("sess", "resp")
        assert a == b, "same (session, response) sampled differently — not deterministic"
        got = sum(qm.should_sample("sess", f"resp-{i}") for i in range(400))
        assert 120 < got < 280, f"rate 0.5 over 400 distinct turns sampled {got} (expected ~200)"


@test("quality", "note_recall emits recall.metric with the outcome fields")
async def t_note_recall(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm

    with _env(OPENAGENT_QUALITY_MONITOR_ENABLED="1"), _capture() as ev:
        qm.note_recall("sess", used=True, hits=3, top_score=0.812345)
    assert len(ev) == 1 and ev[0][0] == "recall.metric", ev
    kw = ev[0][1]
    assert kw["session_id"] == "sess" and kw["used"] is True and kw["hits"] == 3
    assert abs(kw["top_score"] - 0.8123) < 1e-3, kw


@test("quality", "_parse_verdict is tolerant (fenced/prose/derived), rejects junk")
async def t_parse_verdict(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm

    good = qm._parse_verdict('{"score":0.9,"verdict":"good","fabrication":false,"rationale":"ok"}')
    assert good and good["score"] == 0.9 and good["verdict"] == "good"
    prose = qm._parse_verdict('sure, here: {"score": 0.3, "rationale":"bad"} done')
    assert prose and prose["verdict"] == "bad", prose  # verdict derived from score
    fenced = qm._parse_verdict("```json\n{\"score\":0.6}\n```")
    assert fenced and fenced["verdict"] == "warn", fenced
    assert qm._parse_verdict("no json here at all") is None
    assert qm._parse_verdict('{"nope":1}') is None  # missing score
    assert qm._parse_verdict('{"score":"NaNish"}') is None  # unparseable score


@test("quality", "the judge scores a turn and emits quality.score")
async def t_judge_emits(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm

    fm = _FakeModel('{"score":0.9,"verdict":"good","fabrication":false,"rationale":"grounded"}')
    with _env(OPENAGENT_QUALITY_MONITOR_ENABLED="1",
              OPENAGENT_QUALITY_MONITOR_MODEL=None,
              OPENAGENT_COMPACTION_MODEL=None), _capture() as ev:
        await qm._judge(_agent(fm), "sess", "user question", "assistant answer")
    assert fm.calls, "judge model was not called"
    scores = [e for e in ev if e[0] == "quality.score"]
    assert len(scores) == 1, ev
    kw = scores[0][1]
    assert kw["score"] == 0.9 and kw["verdict"] == "good" and kw["fabrication"] is False
    assert kw["session_id"] == "sess" and kw["judge_model"] == "fake:judge"


@test("quality", "gating skips short + unsampled turns, judges long+sampled")
async def t_gating(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm

    with _env(OPENAGENT_QUALITY_MONITOR_ENABLED="1",
              OPENAGENT_QUALITY_MONITOR_SAMPLE_RATE="1",
              OPENAGENT_QUALITY_MONITOR_MIN_LEN="40",
              OPENAGENT_QUALITY_MONITOR_MODEL=None,
              OPENAGENT_COMPACTION_MODEL=None), _capture():
        fm = _FakeModel('{"score":1.0,"verdict":"good"}')
        await qm.maybe_score_turn(_agent(fm), "s", "q", "too short")
        assert not fm.calls, "judged a below-min_len turn"
        await qm.maybe_score_turn(_agent(fm), "s", "q", "x" * 60)
        assert fm.calls, "did not judge a long, fully-sampled turn"


@test("quality", "the judge unparseable case logs, never fabricates a score")
async def t_judge_unparseable(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm

    fm = _FakeModel("I cannot produce JSON, sorry.")
    with _env(OPENAGENT_QUALITY_MONITOR_ENABLED="1",
              OPENAGENT_QUALITY_MONITOR_MODEL=None,
              OPENAGENT_COMPACTION_MODEL=None), _capture() as ev:
        await qm._judge(_agent(fm), "s", "q", "a")
    assert not [e for e in ev if e[0] == "quality.score"], "fabricated a score"
    assert [e for e in ev if e[0] == "quality.judge_unparseable"], ev


@test("quality", "aggregate sums quality + cost + recall over the window")
async def t_aggregate(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm

    now = time.time()
    # events.jsonl is append-only and ts-ORDERED (oldest first); the reverse
    # scanner relies on that to stop early. Mirror it: the out-of-window row
    # sits at the TOP, in-window rows after.
    rows = [
        # older than the window — must be excluded
        {"ts": now - 10_000, "event": "quality.score", "score": 0.1, "verdict": "bad"},
        {"ts": now, "event": "quality.score", "score": 0.9, "verdict": "good", "fabrication": False},
        {"ts": now, "event": "quality.score", "score": 0.4, "verdict": "bad", "fabrication": True},
        {"ts": now, "event": "router.cost_recorded", "cost_usd": 0.01,
         "input_tokens": 100, "output_tokens": 50},
        {"ts": now, "event": "recall.metric", "used": True, "hits": 2, "top_score": 0.8},
        {"ts": now, "event": "recall.metric", "used": True, "hits": 0, "top_score": 0.0},
    ]
    fd, name = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    p = Path(name)
    try:
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        rep = qm.aggregate(3600, path=p)
    finally:
        p.unlink(missing_ok=True)

    assert rep["quality"]["judged"] == 2, rep  # the old one is excluded
    assert abs(rep["quality"]["avg_score"] - 0.65) < 1e-6, rep
    assert rep["quality"]["verdicts"] == {"good": 1, "warn": 0, "bad": 1}, rep
    assert rep["quality"]["fabrication_flagged"] == 1, rep
    assert rep["usage"]["turns"] == 1 and abs(rep["usage"]["cost_usd"] - 0.01) < 1e-9, rep
    assert rep["recall"]["turns"] == 2 and rep["recall"]["hit_rate"] == 0.5, rep
    assert rep["recall"]["avg_top_score"] == 0.4, rep


@test("quality", "spawn_scoring off the loop is a no-op when disabled")
async def t_spawn_disabled(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm

    with _env(OPENAGENT_QUALITY_MONITOR_ENABLED="0"):
        # Must not raise, must not schedule anything.
        qm.spawn_scoring(_agent(_FakeModel("{}")), "s", "q", "r" * 100)
    assert True
