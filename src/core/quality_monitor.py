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

# The DEFAULT judge model when no judge/compaction model is configured.
# ``deepseek:deepseek-chat`` is cheap-but-capable AND isolated from the Claude
# subscription: the previous default resolved to the cheapest enabled row, which
# is the $0 ``local`` claude-sub-proxy — grading THROUGH the same subscription
# the live agents run on, competing with them for it. DeepSeek is a paid
# api-based provider off that sub, so the grader runs on its own budget. It is
# still only a DEFAULT — ``OPENAGENT_QUALITY_MONITOR_MODEL`` (and the compaction
# model) override it — and it is NEVER an Anthropic key: when deepseek is not an
# enabled api-based row we fall back to the existing cheapest-enabled logic.
_DEFAULT_JUDGE_MODEL = "deepseek:deepseek-chat"


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
    "Given the USER message, the run's TOOL TRACE, and the ASSISTANT reply, "
    "grade the reply for CORRECTNESS and SAFETY, not style. Judge four things: "
    "(1) did it stay grounded — no invented facts, ids, prices, or policies "
    "presented as real; (2) did it follow a sensible policy for the request; "
    "(3) did it take the right ACTION or next step (e.g. a reported bug should "
    "become a task, a billing dispute should be investigated not hand-waved); "
    "(4) is it actually responsive to what was asked. "
    "GROUNDING RULE — READ CAREFULLY: an id, name, ticket, price, URL or fact "
    "that appears in a TOOL TRACE result below (or in the OPERATING RULES) is "
    "GROUNDED: it came from a real tool call, so it is NOT fabricated. Only set "
    "fabrication=true, or fault the reply as 'ungrounded id' / 'no tool calls "
    "shown' / 'procedure skipped', when the value is absent from BOTH the tool "
    "results AND the rules. When a TOOL TRACE is present the reply did make tool "
    "calls — never say it made none. When the TOOL TRACE is empty or missing, "
    "judge grounding from the reply and rules alone and do NOT assume tools ran. "
    "Reply with ONLY a JSON object: "
    '{\"score\": <0.0-1.0>, \"verdict\": \"good\"|\"warn\"|\"bad\", '
    '\"fabrication\": <bool>, \"rationale\": \"<one sentence>\"}. '
    "score >= 0.8 good, 0.5-0.8 warn, < 0.5 bad. Be terse."
)

# Cap on the tool-trace text spliced into the judge prompt. Bounded for the same
# reason as the agent rules: the judge is a SECOND model call and its prompt must
# not balloon (we have already seen judge timeouts). Large tool results are
# truncated per-tool in ``tool_trace``; this caps the whole block.
_MAX_TRACE_CHARS = 3000

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


# Configured judge ids already warned as "resolves to no enabled api-based row".
# Warned ONCE each (a clear, actionable operator signal), NOT once per sampled
# turn — the old code re-emitted the warning on every score (37x in production).
# Keyed on the configured value so a fixed/changed config re-arms.
_JUDGE_UNRESOLVED_WARNED: set = set()


def _warn_judge_unresolved_once(configured: str) -> None:
    if configured in _JUDGE_UNRESOLVED_WARNED:
        return
    _JUDGE_UNRESOLVED_WARNED.add(configured)
    elog(
        "quality.judge_model_unresolved", level="warning",
        configured=configured, reason="no_api_based_row_using_cheapest_enabled",
        note=(f"{_MODEL_ENV} (or OPENAGENT_COMPACTION_MODEL) is set to "
              f"{configured!r} but it matches no enabled api-based model row; "
              "falling back to the cheapest enabled row (a valid non-null judge) "
              "instead of the router. Set it to an enabled api-based id "
              "(e.g. deepseek:deepseek-chat or a local: sub-proxy model) to "
              "silence this."),
    )


def _default_judge_model(agent: Any) -> Any:
    """Resolve the DEFAULT judge (no judge/compaction model configured).

    Prefer ``deepseek:deepseek-chat`` when it is an enabled api-based row — a
    cheap-but-capable grader isolated from the $0 claude-sub-proxy, so the judge
    does not compete with the live agents for the shared Claude subscription.
    When deepseek is NOT enabled, fall back to the existing cheapest-enabled
    default (``_cheap_background_model``). Never routes to an Anthropic key:
    both branches build a toolkit-free ``NativeProvider`` over an api-based row,
    and the fallback's own catalog-cheapest pick is the same $0-first path used
    before. Only a ROUTER ``agent.model`` is rewritten — a non-router model
    (e.g. a unit-test fake) is returned unchanged, preserving the old contract.
    """
    fallback = getattr(agent, "model", None)
    # Only a full Team-router fallback is worth overriding; an already-plain
    # model was chosen deliberately (or is a test fake) — leave it be, exactly
    # as ``_cheap_background_model`` does for non-router callers.
    from src.core.compaction import _cheap_background_model, _is_router

    if _is_router(fallback):
        try:
            from src.models.catalog import FRAMEWORK_API_BASED, iter_configured_models
            from src.models.native_provider import NativeProvider

            providers_config = getattr(agent, "_providers_config", None) or []
            match = next(
                (e for e in iter_configured_models(providers_config)
                 if e.runtime_id == _DEFAULT_JUDGE_MODEL and not e.disabled
                 and e.framework == FRAMEWORK_API_BASED),
                None,
            )
            if match is not None:
                db_path = getattr(getattr(agent, "_db", None), "db_path", None)
                provider = NativeProvider(
                    model=match.runtime_id,
                    providers_config=providers_config,
                    db_path=str(db_path) if db_path else None,
                )
                elog("quality.judge_model", model=match.runtime_id,
                     reason="deepseek_default")
                return provider
        except Exception as exc:  # noqa: BLE001 — a default pick must never break a turn
            elog("quality.judge_model_failed", level="warning",
                 configured=_DEFAULT_JUDGE_MODEL,
                 error_type=type(exc).__name__, error=str(exc) or repr(exc))
    # deepseek not enabled (or a non-router model) → the prior cheapest-enabled
    # default. Routing still only touches the local sub-proxy / deepseek, never
    # an Anthropic key.
    return _cheap_background_model(
        agent, fallback,
        picked_event="quality.judge_model",
        fallback_event="quality.judge_model_dispatcher_fallback",
        failed_event="quality.judge_model_failed",
        what="quality judge",
        env_hint=_MODEL_ENV,
    )


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
            _warn_judge_unresolved_once(configured)
        except Exception as exc:  # noqa: BLE001
            elog("quality.judge_model_failed", level="warning",
                 configured=configured, error=str(exc) or type(exc).__name__)
        # Configured but the id resolves to no enabled api-based row (a typo, a
        # disabled/renamed model, a provider whose framework isn't api-based).
        # The OLD behaviour returned ``agent.model`` — the full Team router —
        # whose ``.model``/``.id`` is None, so EVERY sampled ``quality.score``
        # logged ``judge_model: null`` AND re-emitted the unresolved warning
        # (37x in production), and the grader silently ran through the premium
        # leader instead of a cheap judge. Resolve DETERMINISTICALLY to the
        # cheapest enabled row (a toolkit-free NativeProvider with a non-null
        # ``.model``) instead — the same $0-first path the UNSET branch uses, so
        # routing still only touches the local sub-proxy / deepseek, never an
        # Anthropic key. A non-router ``agent.model`` (e.g. a unit-test fake) is
        # returned unchanged, preserving the old contract for that caller.
        from src.core.compaction import _cheap_background_model
        return _cheap_background_model(
            agent, getattr(agent, "model", None),
            picked_event="quality.judge_model",
            fallback_event="quality.judge_model_dispatcher_fallback",
            failed_event="quality.judge_model_failed",
            what="quality judge",
            env_hint=_MODEL_ENV,
        )
    # UNSET (C3): no dedicated judge or compaction model configured. The old
    # behaviour returned ``agent.model`` — the full Team router — so a grader ran
    # up to a ~150k-token judge prompt through the premium leader (and could bill
    # a paid DeepSeek delegation) on every sampled turn. Then it defaulted to the
    # cheapest enabled row — which is the $0 ``local`` claude-sub-proxy, i.e. the
    # grader ran THROUGH the Claude subscription, competing with the live agents.
    # Default instead to ``deepseek:deepseek-chat`` (isolated, cheap, capable)
    # when it is enabled, falling back to the cheapest-enabled row otherwise —
    # never an Anthropic key. Only a *router* ``agent.model`` is rewritten, so a
    # non-router fake in unit tests is untouched (see ``_default_judge_model``).
    return _default_judge_model(agent)


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


def _trace_block(tool_trace_rows: Any) -> str:
    """Render the run's tool trace for the judge prompt, or ``""`` when empty.

    The block is what lets the judge VERIFY grounding instead of guessing: an id
    the reply quotes that is present in a tool RESULT here is grounded, not
    fabricated. Bounded by ``_MAX_TRACE_CHARS`` so the judge call stays cheap.
    Best-effort — a rendering failure just drops the trace, never raises."""
    if not tool_trace_rows:
        return ""
    try:
        from src.core import tool_trace as _tt
        body = _tt.render(tool_trace_rows, max_chars=_MAX_TRACE_CHARS)
    except Exception:  # noqa: BLE001 — grounding aid is best-effort, never fatal
        return ""
    if not body:
        return ""
    return (
        "--- TOOL TRACE (tool calls this run made, with a truncated excerpt of "
        "each RESULT; ids/facts appearing here are GROUNDED) ---\n"
        f"{body}\n--- END TOOL TRACE ---\n\n"
    )


async def _judge(agent: Any, session_id: Optional[str],
                 user_message: str, response: str,
                 tool_trace_rows: Any = None) -> None:
    """Run the judge on one turn and emit ``quality.score``. Never raises.

    ``tool_trace_rows`` is the run's compact tool trace (``[(name, excerpt)]``)
    captured by ``src/core/tool_trace.py``. Passing it lets the judge check a
    cited id/fact against what the tools actually returned before calling it
    fabricated — the fix for the false ``bad``/fabrication verdicts on
    tool-grounded replies."""
    model = _pick_judge_model(agent)
    if model is None:
        return
    trace = _trace_block(tool_trace_rows)
    rules = _agent_rules(agent)
    # Only assert the grounding rule when a trace is actually present — otherwise
    # the judge would be told to check a TOOL TRACE that isn't there.
    if rules and trace:
        ground_note = (" An id/fact present in the TOOL TRACE or the RULES is "
                       "grounded, not fabricated.")
    elif trace:
        ground_note = (" An id/fact present in the TOOL TRACE is grounded, not "
                       "fabricated.")
    else:
        ground_note = ""
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
            f"{trace}"
            f"ASSISTANT:\n{_excerpt(response)}\n\n"
            f"Grade the ASSISTANT reply against the RULES now.{ground_note}"
        )
    else:
        prompt = (
            f"USER:\n{_excerpt(user_message)}\n\n"
            f"{trace}"
            f"ASSISTANT:\n{_excerpt(response)}\n\n"
            f"Grade the ASSISTANT reply now.{ground_note}"
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
         session_id=session_id, judge_model=judge_id, grounded=bool(rules),
         tool_calls=(len(tool_trace_rows) if tool_trace_rows else 0), **verdict)


async def maybe_score_turn(agent: Any, session_id: Optional[str],
                           user_message: str, response: str,
                           tool_trace_rows: Any = None) -> None:
    """Entry point from the turn-completion path. Cheap gate first, then judge.

    Designed to be scheduled fire-and-forget (``asyncio.create_task``) AFTER the
    turn returns to the user, so the judge's latency and cost never sit on the
    reply path. Returns immediately when disabled, unsampled, or the turn is too
    trivial to be worth grading.

    ``tool_trace_rows`` is the run's compact tool trace, captured synchronously
    by ``spawn_scoring`` (see ``src/core/tool_trace.py``) so the judge can verify
    grounding — an id cited from a tool result is not fabrication.
    """
    if not enabled():
        return
    resp = (response or "").strip()
    if len(resp) < _min_len():
        return
    if not should_sample(session_id, resp):
        return
    try:
        await _judge(agent, session_id, user_message or "", resp, tool_trace_rows)
    except Exception as exc:  # noqa: BLE001 — belt-and-suspenders around the task
        elog("quality.monitor_error", level="warning",
             session_id=session_id, error=str(exc) or type(exc).__name__)


def spawn_scoring(agent: Any, session_id: Optional[str],
                  user_message: str, response: str) -> None:
    """Fire-and-forget ``maybe_score_turn`` on the running loop, cheaply.

    The enabled-check happens BEFORE creating a task so the disabled path
    allocates nothing. Swallows the no-running-loop case (a sync test caller).

    The run's tool trace is DRAINED here, synchronously, before the task is
    scheduled — so the judge task carries exactly this turn's trace even if a
    later turn on the same session publishes a new one before the task runs.
    """
    if not enabled():
        return
    try:
        from src.core import tool_trace
        tool_trace_rows = tool_trace.take(session_id)
    except Exception:  # noqa: BLE001 — the trace is a grounding aid, never required
        tool_trace_rows = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(
        maybe_score_turn(agent, session_id, user_message, response, tool_trace_rows))
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
