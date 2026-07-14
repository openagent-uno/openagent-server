"""Per-session vault-save reminder injected into the user prompt.

On the first user turn, and then every ``_every()`` turns, we prepend a
short memory-checkpoint hint to the outbound user text telling the agent to
ALWAYS save anything relevant to the memory vault before replying. The
reminder is pure prompt text — no model call, no network hop — so it costs
nothing extra on the hot path.

On by default. Tune via ``memory.vault_reminder`` in ``openagent.yaml``
(``enabled``, ``every_n_turns``); see ``core/server.py`` for the env mapping.

TWO RULES THIS MODULE LEARNED THE HARD WAY
------------------------------------------
1. **Only ``maybe_render_reminder`` gates the feature.** It is documented as
   safe to call unconditionally and early-exits when disabled. Until v0.15.11
   the sole call site (``bridges/base.py``) ALSO checked
   ``OPENAGENT_VAULT_REMINDER_ENABLED`` inline — but defaulted it to ``"0"``
   while this module defaults to ``"1"``. The call site's default won, so a
   feature documented as on-by-default here, and in
   ``guide/vault-quality.md``, was off on every deployment that had not
   explicitly opted *in*. Never re-implement the check at a call site: the
   default must live in exactly one place, and that place is ``_is_enabled``.
2. **It is wired on the shared run path, not per-channel.** It used to live in
   the bridge send path, so only Telegram/Discord/Slack/WhatsApp turns could
   ever see it — a desktop or gateway-only install got no nudge at all, and
   neither did any scheduled task, sub-agent, workflow AI block, or event run.
   Vision §15: "There is no reduced or alternate baseline for non-interactive
   execution paths — the agent is the same agent wherever it runs." §7 says
   the same of the scheduler ("chat with the user's seat empty… writes any
   results back to the vault"). It now hooks ``core/agent.py``'s
   ``_run_inner`` / ``_run_inner_stream``, which every origin funnels through.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

from src.core.logging import elog

_DEFAULT_EVERY = 3

_REMINDER_TEMPLATE = (
    "[Memory checkpoint (turn {n}) — ALWAYS save what matters. Before "
    "replying, review this conversation and SAVE to the memory vault "
    "(vault_write_note / vault_patch_note) EVERYTHING relevant that isn't "
    "already there: preferences, decisions, facts, names, paths, deadlines, "
    "gotchas, completed tasks. Saving is the DEFAULT — only skip when truly "
    "nothing was learned this turn. Notes must be ATOMIC (one idea), "
    "STRUCTURED (complete frontmatter: title, summary, tags, status, "
    "created/updated as YYYY-MM-DD) and densely linked with >=3 real "
    "[[wikilinks]] to existing notes — search first with vault_search_notes "
    "to avoid duplicates and connect related notes; never leave a broken "
    "link or an orphan. Every change is version-controlled automatically, so "
    "write freely. After a burst of writes, run vault_gate to check them.]"
)


def _is_enabled() -> bool:
    # On by default; opt out with memory.vault_reminder.enabled: false.
    return (
        os.environ.get("OPENAGENT_VAULT_REMINDER_ENABLED", "1").strip().lower()
        in ("1", "true", "yes", "on")
    )


def _every() -> int:
    try:
        n = int(os.environ.get("OPENAGENT_VAULT_REMINDER_EVERY_N_TURNS", _DEFAULT_EVERY))
    except (TypeError, ValueError):
        return _DEFAULT_EVERY
    return max(1, n)


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
    # Fire on the very first turn, then on every Nth turn after.
    if new_count == 0 or not (new_count == 1 or new_count % _every() == 0):
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
