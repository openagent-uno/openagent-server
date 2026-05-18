"""Auto-skills — durable how-tos the AI distils from recurring requests.

Two halves:

  Post-turn DETECTOR. After each user/assistant turn we ask Groq Llama
  "is this a reusable pattern worth saving?" — if yes, the response
  carries a name, a short markdown body, a stable signature, and
  optional tags. We insert into ``skills`` keyed by ``signature_hash``
  so a near-duplicate detection (e.g. the user asks the same kind of
  task four times) collapses into a single row whose ``usage_count``
  increments.

  Pre-turn MATCHER. Before a new turn fires we score the user's text
  against the saved skills using a cheap tag-and-substring heuristic
  (no LLM call on the hot path), return the top ``MAX_INJECTED_SKILLS``
  matches, and surface them as a system-style hint that gets injected
  alongside the user profile in ``models/claude_cli._ensure_client``.

Why no LLM matcher: the detector already pays the Groq round-trip in
the background; layering another one on the user-visible path would
re-add 200-500 ms per Telegram message — the whole point of these
hooks is to make the bot *feel* faster while still learning. The
heuristic is good enough until you have many dozens of skills, at
which point a re-rank step can be bolted on without changing storage.

Opt-in via ``memory.skills.enabled: true`` in ``openagent.yaml`` (env
mapping in ``core/server.py``). Default off.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from typing import Any, Optional

from src.core.logging import elog
from src.learning._groq import complete as groq_complete


_MAX_INJECTED_SKILLS = 2          # cap so the system-prompt overlay stays small
_MIN_MATCH_SCORE = 2              # below this we don't inject — keep false positives low
_RECENT_SKILLS_FOR_DETECTOR = 12  # context for the detector so it skips near-duplicates
_DETECTOR_TIMEOUT_S = 18.0
_SKILL_MARKDOWN_CHAR_LIMIT = 1800


_DETECTOR_SYSTEM_PROMPT = (
    "You watch chat turns and decide if the user is teaching the assistant "
    "a *reusable* how-to. You only mark a turn as save-worthy when it "
    "encodes a procedure the assistant should follow next time a similar "
    "request arrives (preferences, command flags, naming conventions, "
    "validation steps, etc.). Pure factual answers, one-off Q&A, "
    "small-talk, and emotional support do NOT count.\n\n"
    "Return ONLY a JSON object on a single line, with EXACTLY these keys:\n"
    '{\n'
    '  "action": "save" | "skip",\n'
    '  "name": short kebab-case identifier (≤ 60 chars),\n'
    '  "markdown": the playbook, ≤ 1800 chars, second person ("When the user asks for X, do Y"),\n'
    '  "signature": stable phrase capturing the trigger so duplicates dedup ("user wants telegram bot replies in italian"),\n'
    '  "tags": [string, …] (lowercase keywords for the matcher, max 8)\n'
    '}\n\n'
    "On skip, all keys other than action may be empty strings / empty lists.\n"
    "Never wrap the JSON in markdown fences or prose."
)


def _is_enabled() -> bool:
    return (
        os.environ.get("OPENAGENT_SKILLS_ENABLED", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )


def _signature_hash(signature: str) -> str:
    """Normalise + hash the signature so trivial wording differences
    collapse to the same row. Lower, strip, collapse whitespace,
    drop trailing punctuation."""
    norm = re.sub(r"\s+", " ", (signature or "").strip().lower())
    norm = norm.rstrip(".!?,:;")
    return hashlib.sha256(norm.encode("utf-8", errors="ignore")).hexdigest()[:24]


_WORD_RE = re.compile(r"[a-zA-Z0-9_]{3,}")


def _tokenize(text: str) -> set[str]:
    """Cheap token bag for tag-matching. Drops sub-3-char tokens so
    "an" / "il" don't dominate scoring."""
    return {tok.lower() for tok in _WORD_RE.findall(text or "")}


# ── Storage ────────────────────────────────────────────────────────


async def _load_all_skills(db: Any) -> list[dict[str, Any]]:
    conn = getattr(db, "_conn", None)
    if conn is None:
        return []
    try:
        cur = await conn.execute(
            "SELECT id, name, signature_hash, markdown, tags_json, usage_count, last_used_at "
            "FROM skills ORDER BY usage_count DESC, last_used_at DESC NULLS LAST"
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for r in rows or []:
        try:
            tags = json.loads(r[4] or "[]")
        except Exception:
            tags = []
        out.append(
            {
                "id": r[0],
                "name": r[1],
                "signature_hash": r[2],
                "markdown": r[3],
                "tags": [str(t).lower() for t in tags if t],
                "usage_count": int(r[5] or 0),
                "last_used_at": r[6],
            }
        )
    return out


async def _load_recent_signatures(db: Any, limit: int = _RECENT_SKILLS_FOR_DETECTOR) -> list[str]:
    """Return the names + signatures of the most-recently-used skills
    so the detector can avoid creating near-duplicates."""
    conn = getattr(db, "_conn", None)
    if conn is None:
        return []
    try:
        cur = await conn.execute(
            "SELECT name, signature_hash FROM skills "
            "ORDER BY last_used_at DESC NULLS LAST, updated_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception:
        return []
    return [f"{r[0]} ({r[1][:8]})" for r in (rows or [])]


async def _upsert_skill(
    db: Any, *, name: str, markdown: str, signature: str, tags: list[str]
) -> None:
    conn = getattr(db, "_conn", None)
    if conn is None:
        return
    if len(markdown) > _SKILL_MARKDOWN_CHAR_LIMIT:
        markdown = markdown[:_SKILL_MARKDOWN_CHAR_LIMIT]
    sig_hash = _signature_hash(signature)
    now = time.time()
    tag_payload = json.dumps([str(t).lower() for t in tags[:8] if t])
    try:
        # Dedup by signature_hash: if it exists, bump usage_count and
        # refresh the markdown (the model may have phrased it better
        # this time around).
        await conn.execute(
            """
            INSERT INTO skills
                (id, name, signature_hash, markdown, tags_json,
                 usage_count, last_used_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(signature_hash) DO UPDATE SET
                name = excluded.name,
                markdown = excluded.markdown,
                tags_json = excluded.tags_json,
                usage_count = skills.usage_count + 1,
                last_used_at = excluded.last_used_at,
                updated_at = excluded.updated_at
            """,
            (uuid.uuid4().hex, name[:60], sig_hash, markdown, tag_payload, now, now, now),
        )
        await conn.commit()
        elog(
            "skills.saved",
            name=name[:60],
            sig_hash=sig_hash,
            tags=tag_payload,
            chars=len(markdown),
        )
    except Exception as e:
        elog("skills.persist_error", error=str(e)[:200])


async def _mark_skills_used(db: Any, skill_ids: list[str]) -> None:
    conn = getattr(db, "_conn", None)
    if conn is None or not skill_ids:
        return
    now = time.time()
    try:
        placeholders = ",".join("?" for _ in skill_ids)
        await conn.execute(
            f"UPDATE skills SET usage_count = usage_count + 1, last_used_at = ? "
            f"WHERE id IN ({placeholders})",
            (now, *skill_ids),
        )
        await conn.commit()
    except Exception:
        pass


# ── Detector (post-turn) ───────────────────────────────────────────


async def _detect_and_save(
    db: Any, user_text: str, assistant_text: str, session_id: str
) -> None:
    recents = await _load_recent_signatures(db)
    recents_block = (
        ("Already-saved skill names (don't recreate near-duplicates):\n- "
         + "\n- ".join(recents) + "\n\n")
        if recents else ""
    )
    user_prompt = (
        f"{recents_block}"
        "## Latest exchange\n"
        f"USER: {(user_text or '')[:1800]}\n\n"
        f"ASSISTANT: {(assistant_text or '')[:1800]}"
    )
    raw = await groq_complete(
        db=db,
        prompt=user_prompt,
        system=_DETECTOR_SYSTEM_PROMPT,
        max_tokens=600,
        timeout_s=_DETECTOR_TIMEOUT_S,
        log_event="skills.detect",
        session_id=session_id,
    )
    if not raw:
        return
    parsed: Any
    try:
        parsed = json.loads(raw)
    except Exception:
        cleaned = raw.strip().lstrip("`").rstrip("`").strip()
        for marker in ("json\n", "json\r\n"):
            if cleaned.lower().startswith(marker):
                cleaned = cleaned[len(marker):]
                break
        try:
            parsed = json.loads(cleaned)
        except Exception:
            elog("skills.detect_parse_failed", session_id=session_id, preview=raw[:120])
            return
    if not isinstance(parsed, dict):
        return
    action = (parsed.get("action") or "").strip().lower()
    if action != "save":
        elog("skills.detect_result", session_id=session_id, action=action or "skip")
        return
    name = (parsed.get("name") or "").strip()
    markdown = (parsed.get("markdown") or "").strip()
    signature = (parsed.get("signature") or "").strip()
    tags = parsed.get("tags") or []
    if not (name and markdown and signature):
        elog(
            "skills.detect_incomplete",
            session_id=session_id,
            has_name=bool(name),
            has_markdown=bool(markdown),
            has_signature=bool(signature),
        )
        return
    await _upsert_skill(
        db, name=name, markdown=markdown, signature=signature, tags=tags
    )


async def maybe_detect_after_turn(
    db: Any,
    session_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    """Schedule the detector as a background task. Safe to call
    unconditionally — early-exits when the feature is off."""
    if not _is_enabled() or not session_id or not db:
        return
    # Skip tiny exchanges — not enough signal to learn from.
    if len((user_text or "").strip()) < 8:
        return
    if len((assistant_text or "").strip()) < 8:
        return
    asyncio.create_task(_detect_and_save(db, user_text, assistant_text, session_id))


# ── Matcher (pre-turn, hot path) ───────────────────────────────────


def _score_skill(skill: dict[str, Any], user_tokens: set[str]) -> int:
    """Lightweight scoring: +1 for each tag present in the user message,
    +1 if any token of the skill name (split on '-') matches, capped so
    a single skill with many tags doesn't drown the others."""
    tags = skill.get("tags") or []
    hits = sum(1 for t in tags if t and t in user_tokens)
    name_parts = (skill.get("name") or "").split("-")
    name_hits = sum(
        1 for p in name_parts if len(p) >= 3 and p.lower() in user_tokens
    )
    return min(hits, 5) + min(name_hits, 2)


async def get_relevant_skills(db: Any, user_text: str) -> list[dict[str, Any]]:
    skills = await _load_all_skills(db)
    if not skills:
        return []
    user_tokens = _tokenize(user_text)
    if not user_tokens:
        return []
    scored = [(_score_skill(sk, user_tokens), sk) for sk in skills]
    scored.sort(key=lambda pair: (-pair[0], -(pair[1].get("usage_count") or 0)))
    out: list[dict[str, Any]] = []
    for score, sk in scored:
        if score < _MIN_MATCH_SCORE:
            break
        out.append(sk)
        if len(out) >= _MAX_INJECTED_SKILLS:
            break
    return out


async def render_skills_for_system_prompt(db: Any, user_text: str) -> Optional[str]:
    """Returns a markdown blob with the top-matching skills, or ``None``
    when nothing crosses the relevance bar. Side-effect: bumps the
    ``usage_count`` for whatever we inject so the matcher can prefer
    proven skills next time."""
    if not _is_enabled() or not db:
        return None
    matches = await get_relevant_skills(db, user_text)
    if not matches:
        return None
    await _mark_skills_used(db, [m["id"] for m in matches])
    parts = ["[Reusable how-tos relevant to this turn:]"]
    for sk in matches:
        parts.append(f"## {sk['name']}\n{sk['markdown']}")
    elog(
        "skills.injected",
        names=[m["name"] for m in matches],
        count=len(matches),
    )
    return "\n\n".join(parts)
