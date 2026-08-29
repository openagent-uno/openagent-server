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


@test("quality", "both run AND run_stream wire spawn_scoring (real traffic streams)")
async def t_scoring_wired_on_both_paths(ctx: TestContext) -> None:
    """Regression for the 0.18.5→6 gap: the monitor was wired into ``run()`` only,
    but events go through ``run_stream`` (support turns log ``run_stream.done``),
    so ``quality.score`` never fired on real traffic. Both paths must schedule the
    judge — pin it structurally since the pure-unit fakes can't drive a real
    streaming turn."""
    import inspect
    from src.core.agent import Agent

    for meth in ("run", "run_stream"):
        src = inspect.getsource(getattr(Agent, meth))
        assert "spawn_scoring" in src, (
            f"Agent.{meth} must call quality_monitor.spawn_scoring — "
            f"without it the monitor never fires on that path"
        )
    # run_stream must schedule on the terminal 'done' event, not per-delta.
    stream_src = inspect.getsource(Agent.run_stream)
    assert 'kind") == "done"' in stream_src or "'done'" in stream_src, (
        "run_stream must gate spawn_scoring on the done event"
    )


@test("quality", "judge grounds on the agent's own operating rules when present")
async def t_judge_grounded(ctx: TestContext) -> None:
    """A generic rubric can't tell 'followed the refund policy' from 'sounded
    reasonable'. When the agent has a system_prompt (its playbook), the judge
    prompt must carry those rules and the score event must mark itself grounded."""
    import src.core.quality_monitor as qm

    model = _FakeModel('{"score":0.9,"verdict":"good","fabrication":false,"rationale":"ok"}')
    agent = _agent(model)
    agent.system_prompt = (
        "OPERATING RULES: never invent an order id; issue a refund ONLY after "
        "verifying the receipt; a reported bug MUST become a Replio task."
    )
    with _env(OPENAGENT_QUALITY_MONITOR_ENABLED="1"), _capture() as evs:
        await qm._judge(agent, "s", "I want a refund", "Sure — let me verify your receipt first.")

    assert model.calls, "judge model was never called"
    prompt = model.calls[0][0][0]["content"]
    assert "OPERATING RULES" in prompt and "verifying the receipt" in prompt, prompt[:200]
    scores = [kw for name, kw in evs if name == "quality.score"]
    assert scores and scores[0].get("grounded") is True, scores


@test("quality", "judge falls back to the generic rubric when no rules are available")
async def t_judge_generic_fallback(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm

    model = _FakeModel('{"score":0.7,"verdict":"warn","fabrication":false,"rationale":"x"}')
    agent = _agent(model)  # SimpleNamespace has no system_prompt
    with _env(OPENAGENT_QUALITY_MONITOR_ENABLED="1"), _capture() as evs:
        await qm._judge(agent, "s", "hi", "a sufficiently long assistant reply to grade")

    prompt = model.calls[0][0][0]["content"]
    assert "OPERATING RULES" not in prompt, prompt[:200]
    scores = [kw for name, kw in evs if name == "quality.score"]
    assert scores and scores[0].get("grounded") is False, scores


@test("quality", "grounding rules are length-capped to bound judge cost")
async def t_judge_rules_capped(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm

    agent = _agent(_FakeModel("{}"))
    agent.system_prompt = "R" * 5000
    with _env(OPENAGENT_QUALITY_MONITOR_RULES_CHARS="100"):
        rules = qm._agent_rules(agent)
    assert rules.endswith("…[truncated]") and len(rules) <= 100 + len(" …[truncated]"), len(rules)
    # 0 disables grounding entirely (pure generic rubric).
    with _env(OPENAGENT_QUALITY_MONITOR_RULES_CHARS="0"):
        assert qm._agent_rules(agent) == ""


# ══ Defect 1 — the judge must SEE the tool trace so it stops false-flagging a
#    tool-grounded id as fabrication ═══════════════════════════════════════════


class _GroundingJudge:
    """A fake judge that HONOURS the grounding rule: an id present in the TOOL
    TRACE section of the prompt is grounded (good), otherwise it is fabricated
    (bad). This is exactly the discrimination the real judge could not make
    before the trace was passed — so it pins that the fix reaches the outcome,
    not just the prompt text."""

    model = "fake:judge"

    def __init__(self, cited_id: str):
        self._id = cited_id
        self.calls: list[tuple] = []

    async def generate(self, messages, system=None, session_id=None):
        self.calls.append((messages, system, session_id))
        prompt = messages[0]["content"]
        # Inspect ONLY the tool-trace block (between its markers), never the
        # ASSISTANT reply that follows — otherwise the reply's own mention of
        # the id would read as grounding.
        trace = ""
        if "--- TOOL TRACE" in prompt and "--- END TOOL TRACE ---" in prompt:
            trace = prompt.split("--- TOOL TRACE", 1)[1].split("--- END TOOL TRACE ---", 1)[0]
        if self._id in trace:
            return SimpleNamespace(content=(
                '{"score":0.9,"verdict":"good","fabrication":false,'
                '"rationale":"id present in tool result"}'))
        return SimpleNamespace(content=(
            '{"score":0.3,"verdict":"bad","fabrication":true,'
            '"rationale":"ungrounded id / no tool calls shown"}'))


@test("quality", "a reply citing an id present in a tool RESULT is NOT flagged fabrication")
async def t_judge_grounded_by_tool_trace(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm

    fm = _GroundingJudge("86cat39x8")
    # The id and user were returned by a tool — a Replio thread brief — so the
    # reply that quotes them is grounded, not invented.
    rows = [("replio_thread_brief",
             "thread for user stav; linked ClickUp task 86cat39x8, priority high")]
    with _env(OPENAGENT_QUALITY_MONITOR_ENABLED="1",
              OPENAGENT_QUALITY_MONITOR_MODEL=None,
              OPENAGENT_COMPACTION_MODEL=None), _capture() as ev:
        await qm._judge(
            _agent(fm), "3243dae6", "what's the status of my ticket?",
            "Hi stav — your ticket 86cat39x8 is high priority and in progress.",
            rows,
        )
    prompt = fm.calls[0][0][0]["content"]
    assert "TOOL TRACE" in prompt and "86cat39x8" in prompt, prompt[:300]
    scores = [kw for name, kw in ev if name == "quality.score"]
    assert scores, ev
    assert scores[0]["verdict"] == "good" and scores[0]["fabrication"] is False, scores
    assert scores[0]["tool_calls"] == 1, scores  # the trace count is recorded


@test("quality", "a genuinely-absent id IS still flagged fabrication (trace lacks it)")
async def t_judge_absent_id_still_flagged(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm

    fm = _GroundingJudge("ZZ999NOTREAL")
    # The tool trace is present but does NOT contain the cited id — so it really
    # was invented, and the judge must still catch it.
    rows = [("replio_thread_brief", "thread for user stav; task 86cat39x8")]
    with _env(OPENAGENT_QUALITY_MONITOR_ENABLED="1",
              OPENAGENT_QUALITY_MONITOR_MODEL=None,
              OPENAGENT_COMPACTION_MODEL=None), _capture() as ev:
        await qm._judge(
            _agent(fm), "s", "status?",
            "Your order ZZ999NOTREAL shipped yesterday.",
            rows,
        )
    scores = [kw for name, kw in ev if name == "quality.score"]
    assert scores and scores[0]["verdict"] == "bad" and scores[0]["fabrication"] is True, scores


@test("quality", "the system rubric carries the tool-trace grounding rule")
async def t_judge_system_rubric_grounding(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm

    sysrub = qm._JUDGE_SYSTEM.lower()
    assert "tool trace" in sysrub and "grounded" in sysrub, qm._JUDGE_SYSTEM[:200]
    # No trace → the prompt must NOT invent a trace block, and the judge must not
    # be told tools ran.
    fm = _FakeModel('{"score":0.8,"verdict":"good"}')
    with _env(OPENAGENT_QUALITY_MONITOR_ENABLED="1",
              OPENAGENT_QUALITY_MONITOR_MODEL=None,
              OPENAGENT_COMPACTION_MODEL=None):
        await qm._judge(_agent(fm), "s", "hi",
                        "a sufficiently long assistant reply here", None)
    prompt = fm.calls[0][0][0]["content"]
    assert "TOOL TRACE" not in prompt, prompt[:200]


@test("quality", "spawn_scoring drains the run's captured tool trace into the judge")
async def t_spawn_scoring_drains_trace(ctx: TestContext) -> None:
    """End-to-end plumbing: tool_trace.publish (what the dispatcher does on a
    completed run) → spawn_scoring drains it synchronously → the judge prompt
    carries it. Pins that the capture-and-handoff actually reaches the judge."""
    import asyncio
    import src.core.quality_monitor as qm
    from src.core import tool_trace

    fm = _GroundingJudge("86cat39x8")
    with _env(OPENAGENT_QUALITY_MONITOR_ENABLED="1",
              OPENAGENT_QUALITY_MONITOR_SAMPLE_RATE="1",
              OPENAGENT_QUALITY_MONITOR_MIN_LEN="10",
              OPENAGENT_QUALITY_MONITOR_MODEL=None,
              OPENAGENT_COMPACTION_MODEL=None):
        # Simulate the dispatcher publishing this run's trace for the session.
        sink, tok = tool_trace.maybe_open()
        tool_trace.record("replio_thread_brief", "user stav; task 86cat39x8")
        tool_trace.publish("sessX", sink)
        tool_trace.close(tok)
        qm.spawn_scoring(_agent(fm), "sessX", "status?",
                         "Your task 86cat39x8 is in progress, stav.")
        # Drain the fire-and-forget judge task.
        if qm._INFLIGHT:
            await asyncio.gather(*list(qm._INFLIGHT))
    assert fm.calls, "spawn_scoring never reached the judge"
    prompt = fm.calls[0][0][0]["content"]
    assert "86cat39x8" in prompt and "TOOL TRACE" in prompt, prompt[:300]
    # And the trace was consumed (not left dangling for the next turn).
    assert tool_trace.take("sessX") is None


# ══ Defect 2 — a configured-but-unresolvable judge model must resolve to a
#    valid non-null cheap row, not silently fall to the router ═══════════════════


def _providers(*entries):
    """Minimal v0.12 providers_config from (name, [models]) pairs."""
    out = []
    for i, (name, models) in enumerate(entries, start=1):
        out.append({
            "id": i, "name": name, "framework": "api-based",
            "api_key": "sk-test-not-dialled", "base_url": None, "enabled": True,
            "models": [{"id": i * 100 + j, "model": m} for j, m in enumerate(models)],
        })
    return out


@contextlib.contextmanager
def _primed_pricing():
    """Prime OpenRouter pricing so deepseek is priced and local:* is $0."""
    import time as _t
    from src.models import discovery
    import src.models.catalog as catalog

    fake = [{"id": "deepseek/deepseek-v4-pro",
             "pricing": {"prompt": "0.000000435", "completion": "0.00000087"}}]
    saved_cache = getattr(discovery, "_OPENROUTER_CACHE", None)
    saved_index = catalog._OPENROUTER_INDEX
    discovery._OPENROUTER_CACHE = (_t.time() + 1e6, fake)
    catalog._OPENROUTER_INDEX = None
    try:
        yield
    finally:
        discovery._OPENROUTER_CACHE = saved_cache
        catalog._OPENROUTER_INDEX = saved_index


@test("quality", "configured-but-unresolvable judge → cheapest NativeProvider, not the null router")
async def t_judge_unresolved_resolves_cheap(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm
    from src.models.dispatcher import ModelDispatcher

    providers = _providers(("deepseek", ["deepseek-v4-pro"]), ("local", ["claude-sub"]))
    router = ModelDispatcher(providers)  # the OLD null-yielding fallback
    agent = SimpleNamespace(model=router, _providers_config=providers,
                            _db=None, system_prompt="")
    # A model id that matches NO configured api-based row (the production
    # misconfig): the judge must NOT fall to the router (whose .model/.id is
    # None → judge_model: null + weak verdicts).
    qm._JUDGE_UNRESOLVED_WARNED.discard("bogus:not-a-real-model")
    with _env(OPENAGENT_QUALITY_MONITOR_MODEL="bogus:not-a-real-model",
              OPENAGENT_COMPACTION_MODEL=None), _primed_pricing(), _capture() as ev:
        judge = qm._pick_judge_model(agent)
        judge2 = qm._pick_judge_model(agent)  # a second sampled turn
    assert judge is not router, "unresolvable config still fell through to the router"
    assert type(judge).__name__ == "NativeProvider", f"expected NativeProvider, got {judge!r}"
    judge_id = getattr(judge, "model", None) or getattr(judge, "id", None)
    assert judge_id == "local:claude-sub", f"judge model unresolved/null: {judge_id!r}"
    assert getattr(judge2, "model", None) == "local:claude-sub"
    # The unresolved WARNING is emitted at most once (not per sampled turn).
    warns = [1 for name, _ in ev if name == "quality.judge_model_unresolved"]
    assert len(warns) <= 1, f"unresolved warning spammed {len(warns)}x (should warn once)"


# ══ Defect 3 — the DEFAULT judge (env unset) must be deepseek:deepseek-chat,
#    an isolated cheap model, NOT the $0 claude-sub-proxy that competes with the
#    live agents for the shared Claude subscription ═══════════════════════════════


@contextlib.contextmanager
def _primed_pricing_rows(rows):
    """Prime OpenRouter pricing with arbitrary ``rows`` (each ``{"id": ...,
    "pricing": {"prompt","completion"}}``). Ids absent from ``rows`` price as $0
    (the local sub-proxy / self-hosted case)."""
    import time as _t
    from src.models import discovery
    import src.models.catalog as catalog

    saved_cache = getattr(discovery, "_OPENROUTER_CACHE", None)
    saved_index = catalog._OPENROUTER_INDEX
    discovery._OPENROUTER_CACHE = (_t.time() + 1e6, rows)
    catalog._OPENROUTER_INDEX = None
    try:
        yield
    finally:
        discovery._OPENROUTER_CACHE = saved_cache
        catalog._OPENROUTER_INDEX = saved_index


# deepseek-chat priced ABOVE $0 so pure "cheapest enabled" would NOT pick it
# (local:* and anthropic:* price $0 here) — the ONLY reason it wins is the
# explicit deepseek-default preference. That's the discrimination we want.
_DEEPSEEK_PRICED = [{"id": "deepseek/deepseek-chat",
                     "pricing": {"prompt": "0.0000002", "completion": "0.0000002"}}]


@test("quality", "UNSET env defaults the judge to deepseek:deepseek-chat when enabled")
async def t_default_judge_is_deepseek(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm
    from src.models.dispatcher import ModelDispatcher

    # local FIRST + $0 (so old cheapest-logic would pick it), an anthropic row
    # ALSO $0 (so 'never an Anthropic key' has teeth), deepseek LAST + priced.
    providers = _providers(("anthropic", ["claude-x"]),
                           ("local", ["claude-sub"]),
                           ("deepseek", ["deepseek-chat"]))
    router = ModelDispatcher(providers)
    agent = SimpleNamespace(model=router, _providers_config=providers,
                            _db=None, system_prompt="")
    with _env(OPENAGENT_QUALITY_MONITOR_MODEL=None, OPENAGENT_COMPACTION_MODEL=None), \
            _primed_pricing_rows(_DEEPSEEK_PRICED):
        judge = qm._pick_judge_model(agent)
    assert judge is not router, "default judge fell through to the null router"
    assert type(judge).__name__ == "NativeProvider", f"expected NativeProvider, got {judge!r}"
    judge_id = getattr(judge, "model", None)
    assert judge_id == "deepseek:deepseek-chat", (
        f"default judge resolved to {judge_id!r}, not the isolated deepseek default"
    )
    # never null, never an Anthropic key.
    assert judge_id is not None and not judge_id.startswith("anthropic:"), judge_id


@test("quality", "env override wins over the deepseek default")
async def t_env_override_beats_default(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm
    from src.models.dispatcher import ModelDispatcher

    providers = _providers(("local", ["claude-sub"]), ("deepseek", ["deepseek-chat"]))
    router = ModelDispatcher(providers)
    agent = SimpleNamespace(model=router, _providers_config=providers,
                            _db=None, system_prompt="")
    # An explicit configured judge must be honoured verbatim — NOT the default.
    with _env(OPENAGENT_QUALITY_MONITOR_MODEL="local:claude-sub",
              OPENAGENT_COMPACTION_MODEL=None):
        judge = qm._pick_judge_model(agent)
    assert type(judge).__name__ == "NativeProvider", judge
    assert getattr(judge, "model", None) == "local:claude-sub", (
        f"env override ignored — got {getattr(judge, 'model', None)!r}"
    )


@test("quality", "deepseek NOT enabled → default falls back to cheapest-enabled (never null)")
async def t_default_falls_back_when_no_deepseek(ctx: TestContext) -> None:
    import src.core.quality_monitor as qm
    from src.models.dispatcher import ModelDispatcher

    # No deepseek row at all — the default must fall back to the prior
    # cheapest-enabled logic (local:claude-sub, $0), never null, never anthropic.
    providers = _providers(("local", ["claude-sub"]), ("openai", ["gpt-x"]))
    router = ModelDispatcher(providers)
    agent = SimpleNamespace(model=router, _providers_config=providers,
                            _db=None, system_prompt="")
    with _env(OPENAGENT_QUALITY_MONITOR_MODEL=None, OPENAGENT_COMPACTION_MODEL=None), \
            _primed_pricing_rows([]):
        judge = qm._pick_judge_model(agent)
    assert judge is not router, "fallback fell through to the null router"
    assert type(judge).__name__ == "NativeProvider", judge
    judge_id = getattr(judge, "model", None)
    assert judge_id == "local:claude-sub", f"expected cheapest-enabled fallback, got {judge_id!r}"
    assert judge_id is not None and not judge_id.startswith("anthropic:"), judge_id


@test("quality", "session_scope classifies the id shapes the runtime emits")
async def t_session_scope_classifies(ctx: TestContext) -> None:
    """The four audiences, keyed off the ids production actually produced."""
    from src.core import quality_monitor as qm
    assert qm.session_scope(
        "event:fe4d5c37-ef40-410c-bfd8-03e3d022f084:f7e34886-1750") == qm.SCOPE_CUSTOMER
    assert qm.session_scope(
        "scheduler:aafb8e0c-b5d2-4cb4:6eabc30e-ebb5") == qm.SCOPE_SCHEDULED
    # Real ids from the production log — a sub-agent, and a per-model child.
    assert qm.session_scope("7e::sub::local:claude-opus-5::84851e72") == qm.SCOPE_INTERNAL
    assert qm.session_scope("nnet-5::4de52e03::sub::model::ed4d3e8f") == qm.SCOPE_INTERNAL
    assert qm.session_scope("plain-chat-session") == qm.SCOPE_INTERACTIVE
    assert qm.session_scope(None) == qm.SCOPE_INTERACTIVE


@test("quality", "the judge skips the agent's own sub-agent sessions by default")
async def t_internal_scope_not_judged(ctx: TestContext) -> None:
    """The production failure this closes: the judge graded a sub-agent whose
    output was itself a grading rubric, then flagged it as a bad customer
    reply. Internal child sessions are out of scope unless asked for."""
    from src.core import quality_monitor as qm
    fm = _FakeModel('{"score":0.9,"verdict":"good"}')
    reply = "a sufficiently long assistant reply here to clear the min-len gate"
    with _env(OPENAGENT_QUALITY_MONITOR_ENABLED="1",
              OPENAGENT_QUALITY_MONITOR_SAMPLE_RATE="1",
              OPENAGENT_QUALITY_MONITOR_SCOPES=None,
              OPENAGENT_QUALITY_MONITOR_MODEL=None,
              OPENAGENT_COMPACTION_MODEL=None):
        await qm.maybe_score_turn(_agent(fm), "7e::sub::local:claude-opus-5::84851e72",
                                  "compress this rule", reply)
        assert fm.calls == [], "a sub-agent session must not reach the judge"
        # A customer thread on the same settings still gets graded.
        await qm.maybe_score_turn(_agent(fm),
                                  "event:fe4d5c37:f7e34886", "no music plays", reply)
        assert len(fm.calls) == 1, "a customer thread must still be judged"
        # Opting internal back in is one env away.
        with _env(OPENAGENT_QUALITY_MONITOR_SCOPES="customer,internal"):
            await qm.maybe_score_turn(_agent(fm),
                                      "7e::sub::local:claude-opus-5::84851e72",
                                      "compress this rule", reply)
            assert len(fm.calls) == 2, "explicit opt-in must judge internal again"


@test("quality", "aggregate splits the score by audience instead of blending it")
async def t_aggregate_by_scope(ctx: TestContext) -> None:
    """A blended average hides a customer regression behind healthy internal
    work. Also pins that events written before the scope stamp are classified
    from their session id, so a window spanning an upgrade still splits."""
    from src.core import quality_monitor as qm
    now = time.time()
    rows = [
        # Customer replies: bad. Stamped (post-upgrade).
        {"ts": now, "event": "quality.score", "score": 0.2, "verdict": "bad",
         "fabrication": True, "scope": "customer", "session_id": "event:a:b"},
        {"ts": now, "event": "quality.score", "score": 0.4, "verdict": "bad",
         "scope": "customer", "session_id": "event:a:c"},
        # Internal sub-agent: good, and UNstamped (pre-upgrade event).
        {"ts": now, "event": "quality.score", "score": 1.0, "verdict": "good",
         "session_id": "7e::sub::local:claude-opus-5::84851e72"},
    ]
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "events.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        out = qm.aggregate(window_seconds=3600, path=p)
    q = out["quality"]
    assert q["judged"] == 3, q
    # The blended figure is the misleading one the operator used to read.
    assert q["avg_score"] == round((0.2 + 0.4 + 1.0) / 3, 3), q
    # The one that matters is separated out, and the pre-upgrade row landed in
    # the right bucket from its id alone.
    assert q["customer_avg_score"] == 0.3, q
    by = q["by_scope"]
    assert by["customer"]["judged"] == 2 and by["customer"]["verdicts"]["bad"] == 2, by
    assert by["customer"]["fabrication_flagged"] == 1, by
    assert by["internal"]["judged"] == 1 and by["internal"]["avg_score"] == 1.0, by
