"""Response-quality monitoring — the correctness half of observability.

OpenAgent already tracks *usage* well: every turn's spend lands in
``events.jsonl`` as ``router.cost_recorded`` (model, tokens, cost_usd), the
``budgets`` table caps it, and ``/api/budgets`` draws the meter. What it can't
answer is the other half — *are the answers any good?* A support agent can be
cheap and fast and still be confidently wrong: quoting the wrong refund policy,
inventing an order id, or "resolving" a bug ticket without opening a task. Cost
telemetry is blind to all of it.

This module adds that half, in the SAME shape as the cost half — structured
events on ``events.jsonl``, read back through one aggregator — so quality sits
beside spend in one place:

* ``quality.score`` — an LLM-as-judge grades a SAMPLED fraction of completed
  turns against a correctness rubric (policy followed? no fabrication? right
  action taken?), emitting a 0..1 score + verdict + short rationale.
* ``recall.metric`` — whether semantic recall fired for a turn, its top score,
  and hit/miss, recorded straight from the auto-recall path.

Three rules keep it honest and cheap, mirroring the rest of the codebase:

1. **Opt-in, no-op when off** (§17). Disabled → ``maybe_score_turn`` returns
   before doing anything and ``note_recall`` is a bare ``return``; an existing
   deployment that never sets ``quality_monitor.enabled`` is byte-identical.
2. **Sampled, off the turn path.** The judge is a SECOND model call; running it
   on every turn would double latency and cost. So it fires-and-forgets on a
   background task for a configurable fraction of turns, and a judge failure is
   swallowed — a broken monitor must never break a chat.
3. **Cheap judge by explicit config** (the ``_SUMMARY_MODEL_ENV`` pattern from
   ``core/compaction.py``): the operator NAMES the judge model, we never infer
   one; unset falls back to the compaction model, then the agent's own model.

The read side (``aggregate``) is a single reverse scan of ``events.jsonl`` — the
same ``iter_events_reverse`` primitive ``read_tail`` and the logs MCP sit on —
so quality/cost/recall are summarised without a second store.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from typing import Any, Optional

from src.core.logging import elog, iter_events_reverse

# ── config (env-driven, set by ``_build_agent`` from ``quality_monitor.*``) ──

_ENABLED_ENV = "OPENAGENT_QUALITY_MONITOR_ENABLED"
_RATE_ENV = "OPENAGENT_QUALITY_MONITOR_SAMPLE_RATE"
_MODEL_ENV = "OPENAGENT_QUALITY_MONITOR_MODEL"
_TIMEOUT_ENV = "OPENAGENT_QUALITY_MONITOR_TIMEOUT"
_MIN_LEN_ENV = "OPENAGENT_QUALITY_MONITOR_MIN_LEN"
_RULES_CHARS_ENV = "OPENAGENT_QUALITY_MONITOR_RULES_CHARS"


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def enabled() -> bool:
    """True when the monitor is switched on. Default OFF — a deployment that
    never configured it pays nothing (§17)."""
    return _truthy(os.environ.get(_ENABLED_ENV, "0"))


def _rate() -> float:
    """Fraction of turns to judge, clamped to [0, 1]. Default 0.1 (10%)."""
    try:
        return max(0.0, min(1.0, float(os.environ.get(_RATE_ENV, "0.1"))))
    except ValueError:
        return 0.1


def _timeout() -> float:
    try:
        return max(1.0, float(os.environ.get(_TIMEOUT_ENV, "30")))
    except ValueError:
        return 30.0


def _min_len() -> int:
    """Skip trivially short turns (a one-word ack isn't worth a judge call)."""
    try:
        return max(0, int(os.environ.get(_MIN_LEN_ENV, "40")))
    except ValueError:
        return 40


def _rules_chars() -> int:
    """Cap on the agent-rules grounding text spliced into the judge prompt.

    Grounding lets the judge grade against THIS agent's playbook instead of a
    generic standard, but the rules must be bounded or the judge prompt balloons
    (judge cost + latency matter — we've already seen judge timeouts). ``0``
    disables grounding entirely (pure generic rubric). Default 2000 chars — the
    operating principles fit; the exhaustive procedures live in the vault."""
    try:
        return max(0, int(os.environ.get(_RULES_CHARS_ENV, "2000")))
    except ValueError:
        return 2000


def should_sample(session_id: Optional[str], response: str) -> bool:
    """Deterministic sampling gate keyed on ``(session_id, response)``.

    Deterministic (a hash, not ``random``) so the same turn always samples the
    same way — tests are reproducible and a turn can't be judged twice on a
    retry — while a real stream of distinct turns still spreads evenly across
    the [0,1) bucket. ``rate`` 0 never samples, 1 always does.
    """
    rate = _rate()
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    key = f"{session_id or ''}\x00{response or ''}".encode("utf-8", "replace")
    bucket = int(hashlib.sha1(key).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < rate


# ── recall-quality metric (called from the auto-recall path) ─────────────

def note_recall(session_id: Optional[str], *, used: bool,
                hits: int, top_score: float) -> None:
    """Record one turn's recall outcome as a ``recall.metric`` event.

    No-op when the monitor is disabled. ``used`` is whether semantic recall was
    even attempted (an embedder is wired), ``hits``/``top_score`` describe what
    cleared the threshold — so the aggregate can report a hit-rate and a score
    distribution, the signal for tuning ``min_score``.
    """
    if not enabled():
        return
    try:
        elog("recall.metric", session_id=session_id, used=bool(used),
             hits=int(hits), top_score=round(float(top_score), 4))
    except Exception:  # noqa: BLE001 — a metric must never break a turn
        pass


# ── LLM-as-judge (sampled, off the turn path) ────────────────────────────

_JUDGE_SYSTEM = (
    "You are a strict QA reviewer for a customer-support/assistant agent. "
    "Given the USER message and the ASSISTANT reply, grade the reply for "
    "CORRECTNESS and SAFETY, not style. Judge four things: (1) did it stay "
    "grounded — no invented facts, ids, prices, or policies presented as real; "
    "(2) did it follow a sensible policy for the request; (3) did it take the "
    "right ACTION or next step (e.g. a reported bug should become a task, a "
    "billing dispute should be investigated not hand-waved); (4) is it actually "
    "responsive to what was asked. Reply with ONLY a JSON object: "
    '{\"score\": <0.0-1.0>, \"verdict\": \"good\"|\"warn\"|\"bad\", '
    '\"fabrication\": <bool>, \"rationale\": \"<one sentence>\"}. '
    "score >= 0.8 good, 0.5-0.8 warn, < 0.5 bad. Be terse."
)

_MAX_EXCERPT = 4000


def _excerpt(s: str, n: int = _MAX_EXCERPT) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + " …[truncated]"


def _agent_rules(agent: Any) -> str:
    """The agent's OWN operating rules for grounding the judge.

    Sourced from the operator-configured ``system_prompt`` — the actual playbook
    (refund policy, triage codes, anti-fabrication, "a bug becomes a task", …) —
    NOT the generic framework boilerplate. That is what makes "policy followed?"
    precise for THIS agent instead of a one-size rubric. Capped by
    ``_rules_chars`` so the judge stays cheap, and empty (→ generic-rubric
    fallback, never raises) when no ``system_prompt`` is set (§17)."""
    cap = _rules_chars()
    if cap <= 0:
        return ""
    try:
        rules = str(getattr(agent, "system_prompt", "") or "").strip()
    except Exception:  # noqa: BLE001 — grounding is best-effort, never fatal
        return ""
    if not rules:
        return ""
    return rules if len(rules) <= cap else rules[:cap] + " …[truncated]"


def _pick_judge_model(agent: Any) -> Any:
    """Resolve the judge model, cheap-by-config, never fatal.

    Order (explicit config, never inferred — the ``_SUMMARY_MODEL_ENV``
    pattern): ``OPENAGENT_QUALITY_MONITOR_MODEL`` → the compaction model
    (``OPENAGENT_COMPACTION_MODEL``, already chosen cheap for exactly this kind
    of throwaway call) → the agent's own primary model. Builds a toolkit-free
    ``NativeProvider`` for a named id so the judge pays no tool-schema overhead.
    Returns ``None`` only when even the fallback is missing.
    """
    configured = (os.environ.get(_MODEL_ENV, "").strip()
                  or os.environ.get("OPENAGENT_COMPACTION_MODEL", "").strip())
    if configured:
        try:
            from src.models.catalog import FRAMEWORK_API_BASED, iter_configured_models
            from src.models.native_provider import NativeProvider

            providers_config = getattr(agent, "_providers_config", None) or []
            match = next(
                (e for e in iter_configured_models(providers_config)
                 if e.runtime_id == configured and not e.disabled
                 and e.framework == FRAMEWORK_API_BASED),
                None,
            )
            if match is None:
                # The judge model is often intentionally DISABLED in the routable
                # catalog (e.g. local:claude-sonnet-4-6 is the $0 default but kept
                # out of the delegation set, so only deepseek is "enabled"). A
                # judge is an explicit operator choice AND a direct throwaway call
                # (a NativeProvider, not the router), so honour a disabled row too
                # as long as its provider row is api-based — otherwise the judge
                # falls back to agent.model and routes through the full router
                # (slow → the judge timeouts we saw, and it may bill DeepSeek).
                match = next(
                    (e for e in iter_configured_models(providers_config)
                     if e.runtime_id == configured
                     and e.framework == FRAMEWORK_API_BASED),
                    None,
                )
            if match is not None:
                db_path = getattr(getattr(agent, "_db", None), "db_path", None)
                return NativeProvider(
                    model=match.runtime_id,
                    providers_config=providers_config,
                    db_path=str(db_path) if db_path else None,
                )
            elog("quality.judge_model_unresolved", level="warning",
                 configured=configured, reason="no_enabled_api_based_row")
        except Exception as exc:  # noqa: BLE001
            elog("quality.judge_model_failed", level="warning",
                 configured=configured, error=str(exc) or type(exc).__name__)
        # Configured but unresolved/errored → historical fallback to the agent's
        # own model. Env IS set, so behaviour is unchanged (C3 keeps the set
        # path identical).
        return getattr(agent, "model", None)
    # UNSET (C3): no dedicated judge or compaction model configured. The old
    # behaviour returned ``agent.model`` — the full Team router — so a grader ran
    # up to a ~150k-token judge prompt through the premium leader (and could bill
    # a paid DeepSeek delegation) on every sampled turn. Default instead to the
    # cheapest enabled row as a toolkit-free NativeProvider. Only a *router*
    # fallback is rewritten (see ``_cheap_background_model``), so a non-router
    # ``agent.model`` — e.g. the fake model in unit tests — is untouched.
    from src.core.compaction import _cheap_background_model
    return _cheap_background_model(
        agent, getattr(agent, "model", None),
        picked_event="quality.judge_model",
        fallback_event="quality.judge_model_dispatcher_fallback",
        failed_event="quality.judge_model_failed",
        what="quality judge",
        env_hint=_MODEL_ENV,
    )


def _parse_verdict(text: str) -> Optional[dict]:
    """Pull the judge's JSON verdict out of its reply, tolerantly.

    Models wrap JSON in prose or fences; we grab the first balanced-looking
    object and validate the fields we depend on. Returns ``None`` when nothing
    usable is there — the caller logs a ``judge_unparseable`` rather than
    fabricating a score."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or "score" not in obj:
        return None
    try:
        score = max(0.0, min(1.0, float(obj["score"])))
    except (TypeError, ValueError):
        return None
    verdict = str(obj.get("verdict") or
                  ("good" if score >= 0.8 else "warn" if score >= 0.5 else "bad"))
    return {
        "score": round(score, 3),
        "verdict": verdict,
        "fabrication": bool(obj.get("fabrication", False)),
        "rationale": str(obj.get("rationale") or "")[:400],
    }


async def _judge(agent: Any, session_id: Optional[str],
                 user_message: str, response: str) -> None:
    """Run the judge on one turn and emit ``quality.score``. Never raises."""
    model = _pick_judge_model(agent)
    if model is None:
        return
    rules = _agent_rules(agent)
    if rules:
        # Grounded: grade compliance with THIS agent's own playbook, so
        # "policy followed?" / "right action?" are precise, not generic.
        prompt = (
            "The ASSISTANT operates under these OPERATING RULES. Grade whether "
            "the reply complied with THESE specific rules — the policy it must "
            "follow, the action/next-step it must take, and what it must NEVER "
            "do — not a generic standard:\n"
            f"--- OPERATING RULES ---\n{rules}\n--- END RULES ---\n\n"
            f"USER:\n{_excerpt(user_message)}\n\n"
            f"ASSISTANT:\n{_excerpt(response)}\n\n"
            "Grade the ASSISTANT reply against the RULES now."
        )
    else:
        prompt = (
            f"USER:\n{_excerpt(user_message)}\n\n"
            f"ASSISTANT:\n{_excerpt(response)}\n\n"
            "Grade the ASSISTANT reply now."
        )
    try:
        result = await asyncio.wait_for(
            model.generate([{"role": "user", "content": prompt}],
                           system=_JUDGE_SYSTEM, session_id=None),
            timeout=_timeout(),
        )
    except Exception as exc:  # noqa: BLE001 — a judge failure is not a turn failure
        elog("quality.judge_error", level="warning", session_id=session_id,
             error_type=type(exc).__name__, error=str(exc) or repr(exc))
        return
    verdict = _parse_verdict(getattr(result, "content", "") or "")
    judge_id = getattr(model, "model", None) or getattr(model, "id", None)
    if verdict is None:
        elog("quality.judge_unparseable", level="warning",
             session_id=session_id, judge_model=judge_id)
        return
    elog("quality.score", level=("warning" if verdict["verdict"] == "bad" else "info"),
         session_id=session_id, judge_model=judge_id, grounded=bool(rules), **verdict)


async def maybe_score_turn(agent: Any, session_id: Optional[str],
                           user_message: str, response: str) -> None:
    """Entry point from the turn-completion path. Cheap gate first, then judge.

    Designed to be scheduled fire-and-forget (``asyncio.create_task``) AFTER the
    turn returns to the user, so the judge's latency and cost never sit on the
    reply path. Returns immediately when disabled, unsampled, or the turn is too
    trivial to be worth grading.
    """
    if not enabled():
        return
    resp = (response or "").strip()
    if len(resp) < _min_len():
        return
    if not should_sample(session_id, resp):
        return
    try:
        await _judge(agent, session_id, user_message or "", resp)
    except Exception as exc:  # noqa: BLE001 — belt-and-suspenders around the task
        elog("quality.monitor_error", level="warning",
             session_id=session_id, error=str(exc) or type(exc).__name__)


def spawn_scoring(agent: Any, session_id: Optional[str],
                  user_message: str, response: str) -> None:
    """Fire-and-forget ``maybe_score_turn`` on the running loop, cheaply.

    The enabled-check happens BEFORE creating a task so the disabled path
    allocates nothing. Swallows the no-running-loop case (a sync test caller).
    """
    if not enabled():
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(maybe_score_turn(agent, session_id, user_message, response))
    # Keep a reference so the task isn't GC'd mid-flight; drop it on completion.
    _INFLIGHT.add(task)
    task.add_done_callback(_INFLIGHT.discard)


_INFLIGHT: set = set()


# ── read side: aggregate quality + cost + recall over a window ────────────

def aggregate(window_seconds: float = 86400.0, *,
              path: Any = None) -> dict[str, Any]:
    """Summarise quality/cost/recall over the last ``window_seconds``.

    One reverse scan of ``events.jsonl`` (the append-only, ts-ordered log): it
    stops as soon as it passes the window edge, so a day's summary reads only a
    day's tail, not the whole file. Returns counts and averages for
    ``quality.score``, spend from ``router.cost_recorded``, and recall hit-rate
    from ``recall.metric`` — the three signals side by side.
    """
    import time as _time
    since = _time.time() - max(0.0, window_seconds)

    q_scores: list[float] = []
    q_verdicts: dict[str, int] = {"good": 0, "warn": 0, "bad": 0}
    q_fabrication = 0
    cost_usd = 0.0
    turns = 0
    in_tok = 0
    out_tok = 0
    recall_total = 0
    recall_used = 0
    recall_hits = 0
    recall_top: list[float] = []

    for entry in iter_events_reverse(since=since, path=path):
        ev = entry.get("event")
        if ev == "quality.score":
            try:
                q_scores.append(float(entry.get("score")))
            except (TypeError, ValueError):
                pass
            v = entry.get("verdict")
            if v in q_verdicts:
                q_verdicts[v] += 1
            if entry.get("fabrication"):
                q_fabrication += 1
        elif ev == "router.cost_recorded":
            turns += 1
            try:
                cost_usd += float(entry.get("cost_usd") or 0.0)
                in_tok += int(entry.get("input_tokens") or 0)
                out_tok += int(entry.get("output_tokens") or 0)
            except (TypeError, ValueError):
                pass
        elif ev == "recall.metric":
            recall_total += 1
            if entry.get("used"):
                recall_used += 1
            if entry.get("hits"):
                recall_hits += 1
            try:
                recall_top.append(float(entry.get("top_score") or 0.0))
            except (TypeError, ValueError):
                pass

    n = len(q_scores)
    return {
        "window_seconds": window_seconds,
        "quality": {
            "judged": n,
            "avg_score": round(sum(q_scores) / n, 3) if n else None,
            "verdicts": q_verdicts,
            "fabrication_flagged": q_fabrication,
        },
        "usage": {
            "turns": turns,
            "cost_usd": round(cost_usd, 4),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        },
        "recall": {
            "turns": recall_total,
            "used_rate": round(recall_used / recall_total, 3) if recall_total else None,
            "hit_rate": round(recall_hits / recall_total, 3) if recall_total else None,
            "avg_top_score": round(sum(recall_top) / len(recall_top), 3) if recall_top else None,
        },
    }
