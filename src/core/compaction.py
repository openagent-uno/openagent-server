"""In-session compaction — fold older turns into a recap when context fills up.

Vision §2 says it plain: *"When a conversation approaches a model's
context limit, the session compacts in place. The agent rewrites earlier
turns into a summary, preserves the salient details, and continues
without forcing the user to restart."* This module is the
implementation of that line.

The flow per turn, called from ``Agent._run_inner`` *before* the next
``model.generate()``:

1. :func:`should_compact` measures cumulative input tokens across the
   stored runs and asks "would the next call cross
   ``OPENAGENT_COMPACTION_THRESHOLD`` (default 0.75) of the model's max
   context?". The model's max context is best-effort — we try a few
   common attribute names on the provider, then fall back to a generous
   200k-token default. The whole comparison is cheap (sqlite read +
   tiktoken / char-count fallback), so we run it on every turn.

2. :func:`compact` summarises everything except the most recent
   ``OPENAGENT_COMPACTION_KEEP_RUNS`` (default 4) runs into one
   synthetic "session recap" run, then rewrites
   ``sessions.runs`` in place as ``[recap_run] + last_N_runs``.
   The recap run carries ``metadata={"compaction": True}`` so it is
   identifiable on inspection and so a future compaction pass doesn't
   try to re-summarise an already-compacted span.

3. Progress is announced through the ``on_status`` hook as a
   ``{"kind": "session.compacted", "phase": ...}`` envelope — once with
   ``phase="running"`` before the (slow) summariser call and once with
   ``phase="done"`` after the rewrite lands (or ``phase="error"`` if the
   summary came back empty). The turn runner
   (:mod:`src.stream.session`) lifts that envelope into a typed
   :class:`stream.events.SessionCompacted` frame, so the desktop app
   draws a compaction card, the CLI a step line, and the bridges a
   "Compacting conversation" → "Compacted conversation" message. A
   structured ``runtime.compaction`` row also lands in ``events.jsonl``
   so debugging and metrics queries have a stable trail.

The reactive ``ContextWindowExceededError`` fallback in
``src.models.providers.fallback`` stays — it's the safety net for the
case where compaction can't run (no DB-backed session, summarisation
failure, race condition with very large single messages). Compaction
runs first; fallback catches anything that slips through.

What compaction is NOT:

* It does **not** touch ``src.learning.user_profile`` — that subsystem
  summarises *across* sessions, this one folds turns *within* one.
* It does **not** delegate to the dropped ``CompressionManager`` subsystem
  stubs in ``src.core._runner._stubs``. The runtime there is a typing
  shim; the real work lives here, with raw SQL against
  ``sessions.runs`` so it works regardless of whether the
  underlying provider is api-based.
* It does **not** mutate the model's runtime caches. The next
  ``generate()`` re-reads ``sessions.runs`` via the runtime's
  ``add_history_to_context``, so the freshly rewritten row is what the
  next call sees.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

from src.core.logging import elog

# ── Tunables (env-driven so test/debug paths can override cleanly) ────

# Compaction triggers when cumulative input tokens for the *next* call
# would exceed this fraction of the model's max context window. 0.75
# leaves headroom for the new user turn plus a healthy response budget.
_DEFAULT_THRESHOLD = 0.75
# How many of the most recent runs survive the rewrite verbatim. The
# recap run replaces everything older than this. 4 keeps a stable
# "recent conversation" floor even on aggressive triggers.
_DEFAULT_KEEP_RUNS = 4
# Fallback assumption when we can't resolve a model's max context. 200k
# matches Claude / GPT-4o / modern long-context models; cheaper local
# models that expose a smaller window via ``context_window`` /
# ``max_context_length`` will short-circuit on the real value before
# we hit this constant.
_FALLBACK_MAX_CONTEXT = 200_000

# A COST ceiling on the history we let a session carry, independent of how much
# the model could technically swallow.
#
# The threshold below is a fraction of the model's context window, which is the
# right denominator for "will this overflow?" and the wrong one for "what will
# this cost?". A 1M-context model licenses ~786k tokens of accumulated history —
# and history is not paid once. The agentic loop re-sends the whole context on
# every step (3.3 on average, 13 at worst), and a session bound to a long-lived
# external thread replays it on every delivery. On 2026-07-13 that turned single
# support threads into 16M-input-token sessions.
#
# So we compact on whichever comes first: the model's window, or this. 150k
# tokens of history is a lot of conversation; what it is not is 786k.
_MAX_HISTORY_TOKENS = 150_000

# Runtime id (``<provider>:<model>``) of a cheap model to summarise with,
# instead of whatever model the user is talking to. Unset = use the primary.
#
# Why an env var, and NOT the ``is_classifier`` DB flag this module's docstring
# used to promise: ``is_classifier`` does not mean "cheap". It is read by
# ``dispatcher._resolve_entry_model`` as the user's persistent "default team
# leader" hint — the model ENTRY turns route to (per-session pin →
# is_classifier → first enabled). On a typical setup that is the user's *best*
# model, not their cheapest, so wiring compaction to it would summarise on the
# premium model exactly when the point was to stop doing that. It would also
# couple two unrelated subsystems: flipping the flag to move the team leader
# would silently move the summariser too.
#
# ``tier_hint`` is the other candidate and is also the wrong thing to switch
# on. It is free-form prose ("fast and cheap general-purpose chat"), and
# vision §3 is explicit: "Scopes over hardcoded routing. A model's role is a
# sentence the router reads, not a switch statement in code." Grepping
# /cheap|mini|fast/ over that sentence is the switch statement it forbids, and
# it would mis-fire on the first user who writes "not for cheap work".
#
# So: explicit config, in the same env-driven shape as every other tunable
# above. The operator names the model; we never guess one.
_SUMMARY_MODEL_ENV = "OPENAGENT_COMPACTION_MODEL"


def _cost_ceiling() -> int:
    raw = os.environ.get("OPENAGENT_COMPACTION_MAX_HISTORY_TOKENS", "").strip()
    if raw:
        try:
            val = int(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return _MAX_HISTORY_TOKENS


def _flag_enabled() -> bool:
    """Honour OPENAGENT_COMPACTION_ENABLED — default ON.

    Anything except an explicit false-ish string keeps the feature on,
    so the most common opt-out (``OPENAGENT_COMPACTION_ENABLED=false``
    in a test runner) works without ceremony. ``""`` (the unset
    sentinel after ``os.environ.get``) also reads as enabled.
    """
    val = os.environ.get("OPENAGENT_COMPACTION_ENABLED", "").strip().lower()
    if val in {"false", "0", "no", "off"}:
        return False
    return True


def _threshold() -> float:
    raw = os.environ.get("OPENAGENT_COMPACTION_THRESHOLD", "").strip()
    if not raw:
        return _DEFAULT_THRESHOLD
    try:
        v = float(raw)
    except ValueError:
        return _DEFAULT_THRESHOLD
    # Clamp into ``(0, 1]`` so a misconfigured env var can't make us
    # compact on every turn (≤0) or never compact (>1).
    if v <= 0:
        return _DEFAULT_THRESHOLD
    if v > 1.0:
        return 1.0
    return v


def _keep_runs() -> int:
    raw = os.environ.get("OPENAGENT_COMPACTION_KEEP_RUNS", "").strip()
    if not raw:
        return _DEFAULT_KEEP_RUNS
    try:
        v = int(raw)
    except ValueError:
        return _DEFAULT_KEEP_RUNS
    return max(1, v)


# When > 0, an oversized tool-result message in a KEPT-in-history run (every run
# the compaction keeps verbatim EXCEPT the most recent) is elided down to a
# re-fetch pointer. This is the second half of the bound-session bloat fix:
# ``src/core/tool_output.py`` caps ONE tool result at fetch time, but a long
# tool-loop's MANY (individually-capped) results still SUM into a kept run, and
# every kept run is re-sent on every following turn — the live symptom was lyra
# turns at 0.9–1.2M input tokens. Eliding stale tool output from history bounds
# that footprint WITHOUT touching the turn that fetched it (it already reasoned
# over the full result) and WITHOUT touching the most recent run (its tool
# output is still live context). 0 (default) = OFF, byte-identical to before.
_DEFAULT_HISTORY_TOOL_RESULT_CHARS = 0


def _history_tool_result_chars() -> int:
    """Char ceiling for a tool-result message kept in HISTORY (0 = off)."""
    raw = os.environ.get(
        "OPENAGENT_COMPACTION_HISTORY_TOOL_RESULT_CHARS", "").strip()
    if not raw:
        return _DEFAULT_HISTORY_TOOL_RESULT_CHARS
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_HISTORY_TOOL_RESULT_CHARS


# ── Token estimation ──────────────────────────────────────────────────


def _estimate_text_tokens(text: str, model_id: str | None) -> int:
    """Best-effort token count for *text* under *model_id*.

    Defers to the runtime's existing tiktoken / HuggingFace tokenizer
    selector (``src.core._runner.utils.tokens.count_text_tokens``) when
    available — it knows about Llama, Cohere, OpenAI variants, etc. and
    keeps the answer aligned with what providers see internally. Falls
    back to ``len(text) // 4`` (the same fallback the runtime uses) when
    tiktoken isn't installed.
    """
    if not text:
        return 0
    try:
        from src.core._runner.utils.tokens import count_text_tokens
        return count_text_tokens(text, model_id or "gpt-4o")
    except Exception:  # noqa: BLE001 — never let measurement block a turn
        return max(1, len(text) // 4)


def _extract_run_text(run: dict[str, Any]) -> str:
    """Flatten one stored RunOutput dict into plain text for measurement.

    Pulls ``content`` (assistant reply) and every ``messages[*].content``
    string. Tool calls / metrics / IDs are ignored — the measurement is
    about how much the next ``add_history_to_context=True`` call will
    serialise back into a prompt. Robust to legacy shapes (string vs
    dict content, missing keys) since older sessions may predate the
    current schema.
    """
    chunks: list[str] = []
    content = run.get("content")
    if isinstance(content, str) and content:
        chunks.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
            elif isinstance(part, str):
                chunks.append(part)
    for msg in run.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        mc = msg.get("content")
        if isinstance(mc, str) and mc:
            chunks.append(mc)
        elif isinstance(mc, list):
            for part in mc:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
                elif isinstance(part, str):
                    chunks.append(part)
    return "\n".join(chunks)


def _tool_result_pointer(tool_name: Any, orig_len: int, cap: int) -> str:
    """The data-safe placeholder that replaces an elided tool result in history.

    States exactly what was cut and how to get it back — never a silent drop."""
    name = str(tool_name or "the tool")
    return (
        f"[tool result from `{name}` elided from history: {orig_len} chars "
        f"exceeded the {cap}-char history ceiling and would otherwise be re-sent "
        f"on every following turn. The turn that ran this tool already used the "
        f"FULL result; re-run `{name}` with the same arguments to fetch it again "
        f"if you still need it.]"
    )


def _elide_tool_content(content: Any, tool_name: Any, cap: int) -> tuple[Any, int]:
    """Replace an oversized tool-result ``content`` with a pointer.

    Returns ``(new_content, chars_elided)`` (0 elided ⇒ untouched). Handles a
    plain string and the provider content-block list form; only the oversized
    text is collapsed — image/other non-text blocks are preserved, mirroring
    ``tool_output.cap_tool_output`` so nothing binary is dropped."""
    if isinstance(content, str):
        if len(content) <= cap:
            return content, 0
        return _tool_result_pointer(tool_name, len(content), cap), len(content)
    if isinstance(content, list):
        total = (
            sum(len(p) for p in content if isinstance(p, str))
            + sum(len(p["text"]) for p in content
                  if isinstance(p, dict) and isinstance(p.get("text"), str))
        )
        if total <= cap:
            return content, 0
        preserved = [
            p for p in content
            if not isinstance(p, str)
            and not (isinstance(p, dict) and isinstance(p.get("text"), str))
        ]
        pointer = {"type": "text", "text": _tool_result_pointer(tool_name, total, cap)}
        return [*preserved, pointer], total
    return content, 0


def _trim_kept_tool_results(
    kept: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Elide oversized tool-result messages from the KEPT runs — every one the
    compaction preserves verbatim EXCEPT the most recent (whose tool output is
    still live context). Returns ``(new_kept, results_elided, chars_elided)``.
    No-op (returns the input) when the knob is 0 or there is ≤1 kept run.

    Data-safety: the turn that produced each result already reasoned over it in
    full; only the copy retained for FUTURE turns is shrunk, and to a pointer
    that says how to re-fetch — never a silent tail-drop."""
    cap = _history_tool_result_chars()
    if cap <= 0 or len(kept) <= 1:
        return kept, 0, 0
    out: list[dict[str, Any]] = []
    n_elided = 0
    chars = 0
    for i, run in enumerate(kept):
        if i == len(kept) - 1 or not isinstance(run, dict):
            out.append(run)  # never touch the most recent run
            continue
        msgs = run.get("messages")
        if not isinstance(msgs, list):
            out.append(run)
            continue
        new_msgs: list[Any] = []
        touched = False
        for msg in msgs:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                new_c, elided = _elide_tool_content(
                    msg.get("content"), msg.get("tool_name"), cap)
                if elided:
                    new_msgs.append({**msg, "content": new_c})
                    n_elided += 1
                    chars += elided
                    touched = True
                    continue
            new_msgs.append(msg)
        out.append({**run, "messages": new_msgs} if touched else run)
    return out, n_elided, chars


def _resolve_model_id(model: Any) -> str | None:
    """Extract a human-readable model id from any provider shape.

    Providers expose this under a handful of attributes — ``self.model``
    (NativeProvider), ``effective_model_id(session_id)``
    (ModelDispatcher), ``id`` (runtime Model wrapper). We try them in order and
    accept the first stringy result. Returns ``None`` when nothing
    sticks; callers use that to fall back to the generic ``gpt-4o``
    tokenizer.
    """
    if model is None:
        return None
    for attr in ("model", "id", "model_id"):
        val = getattr(model, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    eff = getattr(model, "effective_model_id", None)
    if callable(eff):
        try:
            val = eff(None)
            if isinstance(val, str) and val.strip():
                return val.strip()
        except Exception:  # noqa: BLE001
            pass
    return None


def _resolve_max_context(model: Any) -> int:
    """Best-effort lookup of the model's max input context window.

    Providers in this tree don't (yet) carry an authoritative context
    limit on the provider object — the runtime's ``Model`` exposes one in some
    versions but our wrappers don't surface it. We try a few common
    names and otherwise fall back to ``_FALLBACK_MAX_CONTEXT``. The
    threshold check is intentionally conservative (default 0.75) so the
    fallback still triggers compaction comfortably before any of today's
    major models actually overflow.
    """
    for attr in ("context_window", "max_context", "max_context_length",
                 "max_input_tokens", "max_tokens_context"):
        val = getattr(model, attr, None)
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    # No provider attribute set — consult the shared catalog (OpenRouter's
    # live ``context_length``) so compaction and the /context panel agree
    # on the denominator. Only trust a real catalog hit; the catalog's own
    # fallback is the same 200k we return below, so a miss changes nothing.
    model_id = _resolve_model_id(model)
    if model_id:
        try:
            from src.models.catalog import get_model_context_window

            window, source = get_model_context_window(model_id)
            if source == "openrouter" and window > 0:
                return int(window)
        except Exception:  # noqa: BLE001 — never let a lookup block a turn
            pass
    return _FALLBACK_MAX_CONTEXT


# ── Session row I/O ───────────────────────────────────────────────────


def _resolve_db_path(agent: Any) -> str | None:
    """Find the SQLite file the agent's sessions live in.

    Mirrors the resolution order ``NativeProvider`` uses:
    ``MemoryDB.db_path`` first (the agent's own runtime DB), then the
    provider's ``_db_path`` if it cached one. Returns ``None`` when no
    DB is configured (pure in-memory tests, fresh imports) — callers
    treat that as "compaction unavailable" and skip silently.
    """
    db = getattr(agent, "_db", None)
    if db is not None:
        db_path = getattr(db, "db_path", None)
        if db_path:
            return str(db_path)
    model = getattr(agent, "model", None)
    if model is not None:
        provider_path = getattr(model, "_db_path", None)
        if provider_path:
            return str(provider_path)
    return None


def _load_runs(db_path: str, session_id: str) -> list[dict[str, Any]]:
    """Read ``sessions.runs`` for *session_id* as a list of dicts.

    Returns ``[]`` on any condition that means "nothing to compact":
    missing table, missing row, NULL/empty runs column, or a parse
    failure. We use raw sqlite3 (sync, short-lived connection) because
    this runs inside the per-turn hot path and the overhead is well
    under a millisecond on a small DB. The schema-defensive double
    json.loads matches ``MemoryDB.list_session_runs`` — the runtime's writer
    sometimes serialises the runs column as a JSON-encoded string of a
    JSON array.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=2.0)
    except sqlite3.Error:
        return []
    try:
        try:
            cursor = conn.execute(
                "SELECT runs FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
        except sqlite3.Error:
            # Table might not exist yet on a brand-new DB. Treat as no runs.
            return []
        if not row or not row[0]:
            return []
        try:
            runs = json.loads(row[0])
            if isinstance(runs, str):
                # the runtime's double-encoding edge case — unwrap once more.
                runs = json.loads(runs)
        except (TypeError, ValueError):
            return []
        return [r for r in runs if isinstance(r, dict)] if isinstance(runs, list) else []
    finally:
        conn.close()


def _save_runs(db_path: str, session_id: str, runs: list[dict[str, Any]]) -> None:
    """Persist *runs* back to ``sessions.runs`` for *session_id*.

    Touches the row's ``updated_at`` so the session-list UI orders the
    compacted session correctly and so the gateway's hot-reload probe
    sees the timestamp move. No-op when the row doesn't exist — that
    would mean the session got deleted between ``_load_runs`` and now;
    rewriting a phantom row would resurrect deleted history.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
    except sqlite3.Error as exc:
        elog("compaction.save_open_failed", level="warning",
             session_id=session_id, error=str(exc))
        return
    try:
        try:
            cursor = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            if cursor.fetchone() is None:
                return
            conn.execute(
                "UPDATE sessions SET runs = ?, updated_at = ? "
                "WHERE session_id = ?",
                (json.dumps(runs), int(time.time()), session_id),
            )
            conn.commit()
        except sqlite3.Error as exc:
            elog("compaction.save_failed", level="warning",
                 session_id=session_id, error=str(exc))
    finally:
        conn.close()


# ── Public surface ────────────────────────────────────────────────────


def should_compact(session_id: str | None, model: Any, *, agent: Any) -> bool:
    """Return ``True`` if the next call would breach the compaction threshold.

    *session_id* is the bus identifier the caller passes to
    ``model.generate``. *model* is the active provider (used to read the
    model id + max-context heuristic). *agent* gives us a handle to the
    runtime DB.

    Behaviour:

    * Returns ``False`` when the feature flag is off, no session id is
      set, no DB is configured, or the row has fewer runs than the
      keep-window (nothing to fold).
    * Returns ``True`` when ``cumulative_input_tokens / max_context >
      threshold``. ``cumulative_input_tokens`` is the sum of estimated
      tokens across every stored run, which is what the runtime will replay
      into the next prompt via ``add_history_to_context=True``.

    The estimate is deliberately conservative — we count every stored
    run, including the new user turn we're about to add (since the
    caller has not yet pushed it). Over-eager triggering is fine
    (compaction is cheap and idempotent on top of an already-compacted
    row); under-triggering is the failure mode that surfaces as
    ``ContextWindowExceededError``.
    """
    if not _flag_enabled():
        return False
    if not session_id:
        return False
    db_path = _resolve_db_path(agent)
    if not db_path:
        return False
    keep = _keep_runs()
    runs = _load_runs(db_path, session_id)
    if len(runs) <= keep:
        return False

    model_id = _resolve_model_id(model)
    cumulative = 0
    for run in runs:
        cumulative += _estimate_text_tokens(_extract_run_text(run), model_id)

    # Compact on whichever bites first: the model's context window (so we never
    # overflow) or the cost ceiling (so a 1M-window model can't quietly carry
    # 786k tokens of history into every step of every turn).
    max_context = _resolve_max_context(model)
    threshold = _threshold()
    budget = min(int(max_context * threshold), _cost_ceiling())
    breached = cumulative > budget
    if breached:
        elog(
            "runtime.compaction.threshold_breached",
            session_id=session_id,
            model=model_id,
            cumulative_tokens=cumulative,
            max_context=max_context,
            threshold=threshold,
            budget=budget,
            capped_by="cost_ceiling" if budget < int(max_context * threshold) else "context_window",
            runs=len(runs),
        )
    return breached


async def _summarize_runs(
    runs: list[dict[str, Any]], model: Any, agent: Any,
) -> str:
    """Ask a model to fold *runs* into a single recap paragraph.

    Prefers a small/cheap classifier-flagged model when one is enabled
    in the catalog (per vision §3 — "Cheap and fast classifier suitable
    for routing decisions" — the same routing pick that the dispatcher
    uses for cheap calls). Falls back to the agent's primary model when
    no classifier is configured. The summarisation call passes
    ``session_id=None`` so it doesn't pollute the very session history
    it's trying to fold.

    Returns an empty string on failure — the caller treats that as "skip
    this compaction pass" rather than blowing up the turn.
    """
    transcript_parts: list[str] = []
    for idx, run in enumerate(runs, start=1):
        block = _extract_run_text(run).strip()
        if not block:
            continue
        transcript_parts.append(f"[Turn {idx}]\n{block}")
    if not transcript_parts:
        return ""
    transcript = "\n\n".join(transcript_parts)

    system_prompt = (
        "You are compacting an ongoing conversation into a recap that "
        "REPLACES it — a future assistant will see only this recap and must "
        "continue seamlessly, as if it had read the whole thread. Capture: "
        "the user's goals and explicit requests; key decisions and their "
        "rationale; files, paths, identifiers, and commands mentioned; what "
        "was done and its outcome; the current state; and any open or "
        "pending tasks. Be thorough but compact — drop tool-call mechanics "
        "and verbatim rephrasing, keep the substance."
    )
    user_prompt = (
        "Compact the following conversation into a recap a future assistant "
        "can read to continue it without loss of important context:\n\n"
        + transcript
    )

    summariser = _pick_summary_model(agent, fallback=model)
    if summariser is None:
        return ""

    try:
        response = await summariser.generate(
            [{"role": "user", "content": user_prompt}],
            system=system_prompt,
            # Pass no session_id so the summariser doesn't append a row
            # to the very session we're compacting (that would defeat
            # the purpose).
            session_id=None,
        )
    except Exception as exc:  # noqa: BLE001 — never crash the turn
        elog(
            "runtime.compaction.summary_failed",
            level="warning",
            error_type=type(exc).__name__,
            error=str(exc) or repr(exc),
        )
        return ""
    summary = (getattr(response, "content", "") or "").strip()
    return summary


def _pick_summary_model(agent: Any, *, fallback: Any) -> Any:
    """Return the configured cheap summariser model, else *fallback*.

    Compaction is the one place OpenAgent knowingly ships up to
    ``_cost_ceiling()`` (150k) tokens of transcript into a model to get a few
    hundred tokens of prose back. Doing that on the primary means paying the
    user's premium reasoning model — the one they picked for the actual
    conversation — to do a summarisation job a cheap model does fine. Worse,
    it scales with exactly the sessions we already flagged as expensive: the
    long ones.

    Set ``OPENAGENT_COMPACTION_MODEL=<provider>:<model>`` (e.g.
    ``anthropic:claude-haiku-4-5``) to point that work at a cheap row. See
    ``_SUMMARY_MODEL_ENV`` above for why this is explicit config rather than
    an inference over ``is_classifier`` or ``tier_hint``.

    Resolution is strict but never fatal: the id must match an enabled,
    api-based row in the agent's hydrated providers config, and anything
    unresolvable logs and falls back to *fallback*. A compaction that
    summarises expensively is a cost bug; a compaction that raises is a broken
    chat — and this function is called on the turn's critical path.

    The summariser is deliberately built WITHOUT MCP toolkits: it takes one
    prompt and returns prose, so tool schemas would be pure token overhead on
    the very call we are trying to make cheap.
    """
    configured = os.environ.get(_SUMMARY_MODEL_ENV, "").strip()
    if not configured:
        return fallback
    try:
        from src.models.catalog import FRAMEWORK_API_BASED, iter_configured_models
        from src.models.native_provider import NativeProvider

        providers_config = getattr(agent, "_providers_config", None) or []
        match = next(
            (
                entry
                for entry in iter_configured_models(providers_config)
                if entry.runtime_id == configured
                and not entry.disabled
                and entry.framework == FRAMEWORK_API_BASED
            ),
            None,
        )
        if match is None:
            elog(
                "runtime.compaction.summary_model_unresolved",
                level="warning",
                configured=configured,
                reason="no_enabled_api_based_row",
            )
            return fallback
        db_path = getattr(getattr(agent, "_db", None), "db_path", None)
        summariser = NativeProvider(
            model=match.runtime_id,
            providers_config=providers_config,
            db_path=str(db_path) if db_path else None,
        )
        elog("runtime.compaction.summary_model", model=match.runtime_id)
        return summariser
    except Exception as exc:  # noqa: BLE001 — a cost win must never break a turn
        elog(
            "runtime.compaction.summary_model_failed",
            level="warning",
            configured=configured,
            error_type=type(exc).__name__,
            error=str(exc) or repr(exc),
        )
        return fallback


async def _emit_compaction_status(
    on_status: Any | None, session_id: str, fields: dict[str, Any],
) -> None:
    """Fire one ``session.compacted`` envelope down the *on_status* channel.

    The envelope rides the same hook tool progress uses; the turn runner
    (``src.stream.session``) and the bridges lift it out with
    ``channels.base.parse_compaction_status`` and render a first-class
    compaction affordance instead of a raw status line. Best-effort: a UI
    hint must never crash the turn, so a failing callback is logged and
    swallowed.
    """
    if on_status is None:
        return
    try:
        await on_status(json.dumps({"kind": "session.compacted", **fields}))
    except Exception as exc:  # noqa: BLE001 — UI hint is best-effort
        elog(
            "runtime.compaction.status_failed",
            level="warning",
            session_id=session_id,
            error=str(exc) or type(exc).__name__,
        )


async def compact(
    session_id: str, model: Any, agent: Any,
    *, on_status: Any | None = None, keep: int | None = None,
) -> dict[str, Any] | None:
    """Fold older runs of *session_id* into a single recap run.

    Reads the stored runs, summarises everything older than the last
    *keep* runs, and writes ``[recap_run] + last_keep_runs`` back to
    ``sessions.runs``. The recap run is tagged with
    ``metadata={"compaction": True}`` so it's identifiable on inspection
    and so future passes can detect "already compacted" without re-parsing.

    *keep* controls how many recent runs survive verbatim:

    * ``None`` (the default, used by the AUTOMATIC context-pressure path)
      → ``OPENAGENT_COMPACTION_KEEP_RUNS`` (default 4). Keeps a floor of
      recent turns intact while folding the older span.
    * ``0`` (the MANUAL ``/compact`` command, Claude-Code style) → fold
      the ENTIRE conversation into one recap and keep nothing verbatim;
      the conversation continues from the summary alone. This is why a
      hand-typed ``/compact`` compacts even a short chat, instead of
      no-opping until the conversation exceeds the keep window.

    Emits ``runtime.compaction.done`` (success) or
    ``runtime.compaction.skipped`` (no-op) to ``events.jsonl`` and fires
    the ``session.compacted`` progress envelope through *on_status* (see
    :func:`_emit_compaction_status`).

    Returns a dict summarising the rewrite when one occurred, ``None``
    when nothing changed. No-ops (and idempotently returns ``None`` +
    emits a terminal feedback frame) when there is nothing real to fold —
    an empty session, or one whose only foldable content is an existing
    recap.
    """
    if not _flag_enabled() or not session_id:
        return None
    db_path = _resolve_db_path(agent)
    if not db_path:
        return None
    keep_n = _keep_runs() if keep is None else max(0, int(keep))
    runs = _load_runs(db_path, session_id)

    # Split into the span to fold and the recent turns to keep verbatim.
    # ``keep_n == 0`` (manual /compact) folds everything; slicing with
    # ``runs[:-0]`` would wrongly yield ``[]``, so branch explicitly.
    if keep_n > 0:
        old_runs = runs[:-keep_n]
        kept = runs[-keep_n:]
    else:
        old_runs = list(runs)
        kept = []

    # Nothing worth folding? No old runs at all (automatic path below the
    # keep window), or the only thing older than the window is an
    # already-compacted recap (folding a recap into a recap gains
    # nothing). Emit a terminal ``done`` frame with ``folded_runs=0`` so a
    # manual /compact still shows feedback ("Already compact — nothing to
    # fold") instead of silence — the app reads the frame, not the command
    # result.
    real_old = [
        r for r in old_runs
        if not (isinstance(r, dict) and (r.get("metadata") or {}).get("compaction"))
    ]
    if not real_old:
        elog(
            "runtime.compaction.skipped",
            session_id=session_id,
            reason="nothing_to_fold",
            runs=len(runs),
            keep=keep_n,
        )
        await _emit_compaction_status(on_status, session_id, {
            "phase": "done",
            "folded_runs": 0,
            "kept_runs_count": len(kept),
        })
        return None

    # Measure the folded span up front so both the live "Compacting…"
    # hint and the persisted recap metadata can report how much context
    # the fold frees (``tokens_before − tokens_after``). Same estimation
    # path ``should_compact`` used to decide we were over the threshold.
    model_id = _resolve_model_id(model)
    tokens_before = sum(
        _estimate_text_tokens(_extract_run_text(r), model_id) for r in old_runs
    )

    # Announce the fold BEFORE the summariser round-trip. That call is a
    # full model generate() (a couple of seconds), so this ``running``
    # hint is the only thing that lets a client show "Compacting
    # conversation" instead of unexplained latency. The matching
    # ``done``/``error`` hint below always resolves it — a client that
    # showed a spinner never gets stranded on it.
    await _emit_compaction_status(on_status, session_id, {
        "phase": "running",
        "folded_runs": len(old_runs),
        "kept_runs_count": len(kept),
        "tokens_before": tokens_before,
    })

    summary = await _summarize_runs(old_runs, model, agent)
    if not summary:
        elog(
            "runtime.compaction.skipped",
            session_id=session_id,
            reason="empty_summary",
            runs=len(runs),
        )
        # We already told the client we were compacting; resolve that
        # notice so a spinner doesn't hang on "Compacting…" forever. The
        # reactive ContextWindowExceeded fallback still backstops the turn.
        await _emit_compaction_status(on_status, session_id, {"phase": "error"})
        return None

    tokens_after = _estimate_text_tokens(summary, model_id)

    recap_run: dict[str, Any] = {
        "run_id": f"compaction-{int(time.time())}",
        "session_id": session_id,
        "content": summary,
        "content_type": "str",
        "messages": [
            {"role": "user", "content": "[Previous conversation was compacted into this summary. Continue helping the user as if you had read the full conversation — all key context, decisions, and pending tasks are captured below.]"},
            {"role": "assistant", "content": summary},
        ],
        # Persist the same stats the live "done" hint carries so a
        # reopened / reconciled session rebuilds the identical compaction
        # card from the recap row (see gateway.api.sessions
        # ``_expand_run_messages``) — the boundary marker survives, it's
        # not a live-only affordance.
        "metadata": {
            "compaction": True,
            "folded_runs": len(old_runs),
            "kept_runs_count": len(kept),
            "summary_chars": len(summary),
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
        },
        "created_at": int(time.time()),
    }
    # Second-stage bound (opt-in): elide oversized tool results from the KEPT
    # runs (all but the most recent) so a long tool-loop's accumulated output
    # stops being re-sent on every following turn. No-op unless the operator
    # sets OPENAGENT_COMPACTION_HISTORY_TOOL_RESULT_CHARS.
    kept, results_elided, chars_elided = _trim_kept_tool_results(kept)
    new_runs = [recap_run, *kept]
    _save_runs(db_path, session_id, new_runs)

    if results_elided:
        elog(
            "runtime.compaction.history_tool_results_elided",
            session_id=session_id,
            results_elided=results_elided,
            chars_elided=chars_elided,
            cap=_history_tool_result_chars(),
        )

    elog(
        "runtime.compaction.done",
        session_id=session_id,
        folded_runs=len(old_runs),
        kept_runs=len(kept),
        summary_chars=len(summary),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
    )

    # Tell the UI the fold landed. The turn runner turns this envelope
    # into a typed ``SessionCompacted`` wire frame; the desktop app
    # renders a tool-style compaction card, the CLI a step line, and the
    # bridges flip "Compacting conversation" → "Compacted conversation".
    await _emit_compaction_status(on_status, session_id, {
        "phase": "done",
        "folded_runs": len(old_runs),
        "kept_runs_count": len(kept),
        "summary_chars": len(summary),
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
    })

    return {
        "folded_runs": len(old_runs),
        "kept_runs": len(kept),
        "summary_chars": len(summary),
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
    }


__all__ = ["should_compact", "compact"]
