"""Per-session user profile that survives restarts.

Every ``flush_every_n_turns`` user turns we summon Groq Llama to compress
the last batch of turns into a small JSON profile capturing preferences,
ongoing work, communication style. The result lives in the ``user_profiles``
SQLite table keyed by ``session_id`` and is reinjected as a system-style
context message the next time the session is resumed.

Why a profile JSON rather than a free-form transcript:
  • Bounded size (~800 chars) keeps the inject cost negligible even on
    long-lived chats — token cost ≈ a single SMS per turn.
  • Structured keys (preferences / ongoing_projects / style / notes) let
    the matcher in ``skills`` cross-reference cleanly when both hooks
    are on without re-parsing prose.
  • Survives the SDK session being dropped (e.g. by the compression
    watchdog in ``models.claude_cli``), so the recap is independent of
    Claude's own resume mechanism.

Opt-in via ``memory.user_profile.enabled: true`` in ``openagent.yaml``
(see ``core/server.py`` for the env mapping). Default off.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Optional

from src.core.logging import elog
from src.learning._groq import complete as groq_complete

_DEFAULT_FLUSH_EVERY = 10
_PROFILE_CHAR_LIMIT = 1400  # generous cap so the prompt stays well below 500 tokens

_SYSTEM_PROMPT = (
    "You maintain a tiny rolling profile of a single user across chat "
    "sessions. Read the prior profile (may be empty) and the latest "
    "conversation excerpt, then return an UPDATED profile as JSON.\n\n"
    "JSON SCHEMA (strict — no other keys):\n"
    '{\n'
    '  "preferences": [string, …],       // short bullets, max 8\n'
    '  "ongoing_projects": [string, …],  // short bullets, max 5\n'
    '  "style": string,                  // one sentence describing how the user writes\n'
    '  "notes": [string, …]              // factual asides the bot should remember (max 8)\n'
    '}\n\n'
    f"HARD LIMIT: the serialised JSON must be ≤ {_PROFILE_CHAR_LIMIT} characters. "
    "If you would exceed that, drop the oldest/lowest-signal entries first. "
    "Return ONLY the JSON object — no prose, no markdown fences."
)


def _is_enabled() -> bool:
    return (
        os.environ.get("OPENAGENT_USER_PROFILE_ENABLED", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )


def _flush_every() -> int:
    try:
        n = int(os.environ.get("OPENAGENT_USER_PROFILE_FLUSH_EVERY", _DEFAULT_FLUSH_EVERY))
    except (TypeError, ValueError):
        return _DEFAULT_FLUSH_EVERY
    return max(2, n)


async def _load_state(db: Any, session_id: str) -> tuple[dict, int]:
    """Return (profile_dict, turn_count) for ``session_id``. New row
    if missing — returns ``({}, 0)`` so the first flush sees an empty
    baseline instead of a SQL error."""
    conn = getattr(db, "_conn", None)
    if conn is None:
        return ({}, 0)
    try:
        cur = await conn.execute(
            "SELECT profile_json, turn_count FROM user_profiles WHERE session_id = ?",
            (session_id,),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        return ({}, 0)
    if not row:
        return ({}, 0)
    try:
        profile = json.loads(row[0] or "{}")
    except Exception:
        profile = {}
    return (profile if isinstance(profile, dict) else {}, int(row[1] or 0))


async def _persist(db: Any, session_id: str, profile: dict, turn_count: int) -> None:
    conn = getattr(db, "_conn", None)
    if conn is None:
        return
    now = time.time()
    payload = json.dumps(profile, ensure_ascii=False)
    if len(payload) > _PROFILE_CHAR_LIMIT:
        # Defensive clamp — the model should have respected this but a
        # rogue response shouldn't bloat the DB row.
        payload = payload[:_PROFILE_CHAR_LIMIT]
    try:
        await conn.execute(
            """
            INSERT INTO user_profiles
                (session_id, profile_json, turn_count, last_flushed_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                turn_count = excluded.turn_count,
                last_flushed_at = excluded.last_flushed_at,
                updated_at = excluded.updated_at
            """,
            (session_id, payload, turn_count, now, now, now),
        )
        await conn.commit()
    except Exception as e:
        elog("user_profile.persist_error", session_id=session_id, error=str(e)[:200])


async def _bump_turn_count(db: Any, session_id: str) -> int:
    """Increment ``turn_count`` and return the new value. Lazy-init the
    row when needed."""
    profile, count = await _load_state(db, session_id)
    new_count = count + 1
    await _persist(db, session_id, profile, new_count)
    return new_count


async def _flush_now(
    db: Any,
    session_id: str,
    conversation_excerpt: str,
    current_profile: dict,
) -> None:
    user_prompt = (
        "## Prior profile (JSON, may be empty)\n"
        f"{json.dumps(current_profile or {}, ensure_ascii=False)}\n\n"
        "## Recent conversation excerpt\n"
        f"{conversation_excerpt[:8000]}"
    )
    raw = await groq_complete(
        db=db,
        prompt=user_prompt,
        system=_SYSTEM_PROMPT,
        max_tokens=600,
        timeout_s=15.0,
        log_event="user_profile.flush",
        session_id=session_id,
    )
    if not raw:
        return  # already logged by _groq.complete
    parsed: Any
    try:
        parsed = json.loads(raw)
    except Exception:
        # Sometimes Llama wraps the JSON in fences despite the prompt;
        # try a single rescue strip before giving up.
        cleaned = raw.strip().lstrip("`").rstrip("`").strip()
        for marker in ("json\n", "json\r\n"):
            if cleaned.lower().startswith(marker):
                cleaned = cleaned[len(marker):]
                break
        try:
            parsed = json.loads(cleaned)
        except Exception:
            elog(
                "user_profile.flush_parse_failed",
                session_id=session_id,
                preview=raw[:120],
            )
            return
    if not isinstance(parsed, dict):
        elog("user_profile.flush_parse_failed", session_id=session_id, reason="not_object")
        return
    # Keep the existing turn_count — the flush call increments it
    # separately so concurrent flushes never roll back the counter.
    _, current_count = await _load_state(db, session_id)
    await _persist(db, session_id, parsed, current_count)
    elog(
        "user_profile.flushed",
        session_id=session_id,
        keys=list(parsed.keys()),
        size=len(json.dumps(parsed, ensure_ascii=False)),
    )


async def maybe_flush_after_turn(
    db: Any,
    session_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    """Call this AFTER a turn is persisted. Increments the turn counter
    and triggers a Groq flush when the configured interval is reached.
    Safe to call unconditionally — early-exits when the feature is off.
    """
    if not _is_enabled() or not session_id or not db:
        return
    try:
        new_count = await _bump_turn_count(db, session_id)
    except Exception as e:
        elog("user_profile.bump_error", session_id=session_id, error=str(e)[:200])
        return
    if new_count % _flush_every() != 0:
        return
    # Pull a small excerpt — the last user + assistant pair is usually
    # enough signal because Groq compares it against the existing
    # profile and adds incrementally.
    excerpt = (
        f"USER: {(user_text or '')[:1500]}\n\n"
        f"ASSISTANT: {(assistant_text or '')[:1500]}"
    )
    profile, _ = await _load_state(db, session_id)
    # Fire as a task so the post-turn path returns immediately.
    asyncio.create_task(_flush_now(db, session_id, excerpt, profile))


async def render_profile_for_system_prompt(db: Any, session_id: str) -> Optional[str]:
    """Return a system-message-ready string describing the user, or
    ``None`` when nothing to render. Combines two independent recalls:
    the cross-session user profile (gated on ``user_profile.enabled``)
    AND a per-session compaction summary written by the compression
    watchdog (always rendered when present, since its mere existence
    means the operator opted into compression).
    """
    if not session_id or not db:
        return None
    profile, _ = await _load_state(db, session_id)
    if not profile:
        return None
    parts: list[str] = []
    # Cross-session profile — only when the user_profile feature is
    # explicitly enabled. Compaction is a separate concern below.
    if _is_enabled():
        user_parts: list[str] = ["[Recall what we know about this user across sessions:]"]
        prefs = profile.get("preferences") or []
        if prefs:
            user_parts.append("Preferences: " + "; ".join(str(p) for p in prefs[:8]))
        projs = profile.get("ongoing_projects") or []
        if projs:
            user_parts.append("Ongoing: " + "; ".join(str(p) for p in projs[:5]))
        style = profile.get("style")
        if style:
            user_parts.append(f"Style: {style}")
        notes = profile.get("notes") or []
        if notes:
            user_parts.append("Notes: " + "; ".join(str(n) for n in notes[:8]))
        if len(user_parts) > 1:
            parts.append("\n".join(user_parts))
    compaction = profile.get("_compaction")
    if compaction:
        parts.append(
            "[Summary of earlier turns in this conversation (older context was compressed to save tokens):]\n"
            f"{compaction}"
        )
    if not parts:
        return None
    return "\n\n".join(parts)


async def set_compaction(db: Any, session_id: str, summary: str) -> None:
    """Store a compaction summary on the user profile JSON under the
    ``_compaction`` key. Used by the compression watchdog when it
    summarises the recent turns before dropping the SDK client.
    """
    if not session_id or not db or not summary:
        return
    profile, count = await _load_state(db, session_id)
    profile = profile or {}
    profile["_compaction"] = summary.strip()[:_PROFILE_CHAR_LIMIT]
    await _persist(db, session_id, profile, count)
    elog("user_profile.compaction_stored", session_id=session_id, chars=len(summary))


async def clear_compaction(db: Any, session_id: str) -> None:
    """Drop the compaction summary — used when the model has clearly
    moved past the older context (manual /clear, session forget, etc.)."""
    if not session_id or not db:
        return
    profile, count = await _load_state(db, session_id)
    if not profile or "_compaction" not in profile:
        return
    profile.pop("_compaction", None)
    await _persist(db, session_id, profile, count)
