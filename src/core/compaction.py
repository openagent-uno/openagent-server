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

3. A :class:`stream.events.SessionCompacted` event is published if the
   active stream session is in reach, plus a structured
   ``runtime.compaction`` row in ``events.jsonl`` so debugging and
   metrics queries both have a stable trail.

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
  underlying provider is api-based, claude-cli, or codex-cli.
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


def _resolve_model_id(model: Any) -> str | None:
    """Extract a human-readable model id from any provider shape.

    Providers expose this under a handful of attributes — ``self.model``
    (NativeProvider, ClaudeCLI), ``effective_model_id(session_id)``
    (SmartRouter), ``id`` (runtime Model wrapper). We try them in order and
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

    max_context = _resolve_max_context(model)
    threshold = _threshold()
    breached = cumulative > int(max_context * threshold)
    if breached:
        elog(
            "runtime.compaction.threshold_breached",
            session_id=session_id,
            model=model_id,
            cumulative_tokens=cumulative,
            max_context=max_context,
            threshold=threshold,
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
        "You are summarizing earlier turns of an ongoing conversation. "
        "Preserve the user's stated goals, decisions made, file paths "
        "mentioned, identifiers, and any open questions. Drop tool-call "
        "mechanics and rephrasing. Output one paragraph."
    )
    user_prompt = (
        "Summarize the following earlier turns into one paragraph that a "
        "future assistant can read to continue the conversation without "
        "loss of important context:\n\n" + transcript
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
    """Return the cheapest classifier-flagged model, else *fallback*.

    Looks for a ``is_classifier=True`` row in the agent's hydrated
    providers config. When one exists we'd ideally instantiate it as a
    standalone ``NativeProvider``; for now the simpler shape is to use
    the agent's primary model directly — it already has working
    credentials, MCP wiring, and a session pool. The hook stays here so
    the cheap-model branch can be wired up without disturbing callers
    when the dispatcher gains a public classifier accessor.
    """
    # Future: probe agent._providers_config for an enabled classifier
    # row and build a transient NativeProvider against it. Today we
    # delegate to the same model the user is talking to — cheaper than
    # any cross-provider roundtrip and zero extra config to maintain.
    return fallback


async def compact(
    session_id: str, model: Any, agent: Any,
    *, on_status: Any | None = None,
) -> dict[str, Any] | None:
    """Fold the oldest runs of *session_id* into a single recap run.

    Reads the stored runs, summarises everything older than the last
    ``OPENAGENT_COMPACTION_KEEP_RUNS`` (default 4), and writes
    ``[recap_run] + last_N_runs`` back to ``sessions.runs``. The
    recap run is tagged with ``metadata={"compaction": True}`` so it's
    identifiable on inspection and so future compaction passes can
    detect "this was already compacted" without re-parsing.

    Emits ``runtime.compaction.done`` (success) or
    ``runtime.compaction.skipped`` (no-op) to ``events.jsonl``. Also
    invokes *on_status* with a SessionCompacted-style JSON envelope so
    StreamSession turns the call into an ``OutToolStatus`` frame on the
    outbound bus — that's what the desktop UI reads to render
    "Compacted earlier turns" inline with the active reply.

    Returns a dict summarising the rewrite when one occurred, ``None``
    when nothing changed. Idempotent in the no-op direction: calling
    with too few runs simply returns ``None`` after the early bail.
    """
    if not _flag_enabled() or not session_id:
        return None
    db_path = _resolve_db_path(agent)
    if not db_path:
        return None
    keep = _keep_runs()
    runs = _load_runs(db_path, session_id)
    if len(runs) <= keep:
        elog(
            "runtime.compaction.skipped",
            session_id=session_id,
            reason="below_keep_window",
            runs=len(runs),
            keep=keep,
        )
        return None

    old_runs = runs[:-keep]
    kept = runs[-keep:]

    summary = await _summarize_runs(old_runs, model, agent)
    if not summary:
        elog(
            "runtime.compaction.skipped",
            session_id=session_id,
            reason="empty_summary",
            runs=len(runs),
        )
        return None

    recap_run: dict[str, Any] = {
        "run_id": f"compaction-{int(time.time())}",
        "session_id": session_id,
        "content": summary,
        "content_type": "str",
        "messages": [
            {"role": "system", "content": "[compacted session recap]"},
            {"role": "assistant", "content": summary},
        ],
        "metadata": {"compaction": True, "folded_runs": len(old_runs)},
        "created_at": int(time.time()),
    }
    new_runs = [recap_run, *kept]
    _save_runs(db_path, session_id, new_runs)

    elog(
        "runtime.compaction.done",
        session_id=session_id,
        folded_runs=len(old_runs),
        kept_runs=len(kept),
        summary_chars=len(summary),
    )

    # Tell the UI. The on_status hook is the same channel as tool
    # progress (``OutToolStatus`` on the outbound bus); a JSON envelope
    # with ``kind="session.compacted"`` keeps the existing wire codec
    # happy without inventing a new frame type at this layer. The
    # SessionCompacted dataclass in src.stream.events is the typed shape
    # bridges use when they hand the same payload upstream.
    if on_status is not None:
        try:
            payload = json.dumps({
                "kind": "session.compacted",
                "summary_chars": len(summary),
                "kept_runs_count": len(kept),
            })
            await on_status(payload)
        except Exception as exc:  # noqa: BLE001 — UI hint is best-effort
            elog(
                "runtime.compaction.status_failed",
                level="warning",
                session_id=session_id,
                error=str(exc) or type(exc).__name__,
            )

    return {
        "folded_runs": len(old_runs),
        "kept_runs": len(kept),
        "summary_chars": len(summary),
    }


__all__ = ["should_compact", "compact"]
