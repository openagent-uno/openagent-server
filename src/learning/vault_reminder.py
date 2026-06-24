"""Per-session vault-save reminder injected into the user prompt.

Every ``_every()`` user turns we prepend a short memory-checkpoint hint
to the outbound user text, asking the agent to review the conversation
and save anything relevant to the vault before replying. The reminder is
pure prompt text — no model call, no network hop — so it costs nothing
extra on the hot path.

Opt-in via ``memory.vault_reminder.enabled: true`` in ``openagent.yaml``
(see ``core/server.py`` for the env mapping). Default off.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from src.core.logging import elog

_DEFAULT_EVERY = 6

_REMINDER_TEMPLATE = (
    "[Memory checkpoint — turn {n}: before replying, review this "
    "conversation and SAVE to the memory vault (vault_write_note / "
    "vault_patch_note) anything relevant learned and not yet saved: "
    "preferences, decisions, facts, names, paths, gotchas, completed "
    "tasks. Notes must be ATOMIC (one topic), STRUCTURED (complete "
    "frontmatter: title, summary, tags, status, created/updated as "
    "YYYY-MM-DD) and well-linked with ≥3 real [[wikilinks]] to existing "
    "notes — search first with vault_search_notes to avoid duplicates "
    "and connect related notes, and never leave a broken link or an "
    "orphan. If you wrote several notes, run vault_gate to check them. "
    "If there is nothing new to save, ignore this reminder.]"
)


def _is_enabled() -> bool:
    return (
        os.environ.get("OPENAGENT_VAULT_REMINDER_ENABLED", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )


def _every() -> int:
    try:
        n = int(os.environ.get("OPENAGENT_VAULT_REMINDER_EVERY_N_TURNS", _DEFAULT_EVERY))
    except (TypeError, ValueError):
        return _DEFAULT_EVERY
    return max(2, n)


async def _bump_turn_count(db: Any, session_id: str) -> int:
    """Increment the turn counter for ``session_id`` and return the new
    value. Lazy-initialises the row on first call."""
    conn = getattr(db, "_conn", None)
    if conn is None:
        return 0
    now = time.time()
    try:
        cur = await conn.execute(
            "SELECT turn_count FROM vault_save_reminders WHERE session_id = ?",
            (session_id,),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        return 0
    new_count = (int(row[0]) + 1) if row else 1
    try:
        await conn.execute(
            """
            INSERT INTO vault_save_reminders
                (session_id, turn_count, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                turn_count = excluded.turn_count,
                updated_at = excluded.updated_at
            """,
            (session_id, new_count, now, now),
        )
        await conn.commit()
    except Exception as e:
        elog("vault_reminder.persist_error", session_id=session_id, error=str(e)[:200])
        return 0
    return new_count


async def maybe_render_reminder(db: Any, session_id: str) -> Optional[str]:
    """Increment the turn counter and return a reminder string when the
    configured interval is reached, otherwise ``None``.

    Safe to call unconditionally — early-exits when the feature is off,
    ``session_id`` is falsy, or ``db`` is unavailable.
    """
    if not _is_enabled() or not session_id or not db:
        return None
    try:
        new_count = await _bump_turn_count(db, session_id)
    except Exception as e:
        elog("vault_reminder.bump_error", session_id=session_id, error=str(e)[:200])
        return None
    if new_count == 0 or new_count % _every() != 0:
        return None
    # Update last_reminded_at.
    conn = getattr(db, "_conn", None)
    if conn is not None:
        try:
            await conn.execute(
                "UPDATE vault_save_reminders SET last_reminded_at = ? WHERE session_id = ?",
                (time.time(), session_id),
            )
            await conn.commit()
        except Exception:
            pass
    elog("vault_reminder.fired", session_id=session_id, turn_count=new_count)
    return _REMINDER_TEMPLATE.format(n=new_count)
