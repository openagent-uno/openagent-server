"""Read-back for vault recall attribution — ``vault_recall_stats``.

The write half lives in ``src/core/vault_recall.py`` (the sink) and
``TeamRouterProvider.record_vault_recalls`` (the flush). This is the half that
makes it a policy rather than a diary with numbers: a score nobody consults
changes no behaviour.

WHY THIS TOOL LIVES IN ``vault-gate`` AND NOT IN A NEW MCP
----------------------------------------------------------
Measured, not assumed. ``prompts.py:_render_catalog_summary_lines`` foregrounds
the MCP catalog into the framework prompt on EVERY turn — and §15 means every
turn includes every sub-agent, every workflow AI block and every cron firing.
A new MCP server would therefore add a permanent catalog line
(``- ``vault-recall`` (1 tools): <description>.``) to a prompt that already
costs ~10.8k tokens, on every run, forever. Worse, ``_INLINE_TOOL_KEYS_SERVERS``
(the allowlist whose exact tool keys get inlined so the model copies rather
than guesses them) lives in ``prompts.py``, so a new server could not be added
to it here — the model would burn a ``list_tools`` round-trip on every use.

``vault-gate`` is already in that allowlist, already default-on, already
described as "the agent's handle on vault quality", and carries 10 of its 24
permitted inline keys. Adding one tool to it costs one tool key in a list that
is already rendered — and nothing else. The schema itself stays deferred (§6),
so it is paid only when the agent actually reaches for it.
"""
from __future__ import annotations

import os
import time
from typing import Any

# Cap the rows a single call can pull back. This lands in a model's context,
# and the vault is meant to be read by an agent deciding what to trust, not
# dumped wholesale.
_MAX_LIMIT = 100


def _db_path() -> str:
    """Resolve the same ``openagent.db`` the writer used.

    Mirrors ``src/mcp/servers/scheduler/server.py:_db_path`` — env override
    first, then the packaged default — so an in-process tool and a subprocess
    MCP can never disagree about which database is the agent's.
    """
    override = os.environ.get("OPENAGENT_DB_PATH")
    if override:
        return override
    from src.core.paths import default_db_path

    return str(default_db_path())


async def vault_recall_stats(
    limit: int = 20,
    since_days: int | None = 30,
    note_path: str | None = None,
) -> dict:
    """Per-note recall counts joined to how those runs ended.

    Returns counts, never a verdict — see the ``caveat`` in the payload and
    the module docstring for why that distinction is load-bearing.
    """
    from src.memory.db import MemoryDB

    limit = max(1, min(int(limit or 20), _MAX_LIMIT))
    since = None
    if since_days is not None and int(since_days) > 0:
        since = time.time() - int(since_days) * 86400

    db = MemoryDB(_db_path())
    try:
        await db.connect()
        rows = await db.get_vault_recall_stats(
            since=since, limit=limit, note_path=note_path,
        )
    finally:
        close = getattr(db, "close", None)
        if close is not None:
            try:
                await close()
            except Exception:  # noqa: BLE001
                pass

    notes: list[dict[str, Any]] = []
    for r in rows:
        notes.append({
            "note": r.get("note_path"),
            "recalls": int(r.get("recalls") or 0),
            "ok": int(r.get("ok") or 0),
            "errored": int(r.get("errored") or 0),
            # Surfaced, never scored. A barge-in is the user interrupting
            # (§2), which says nothing about the note.
            "cancelled_excluded": int(r.get("cancelled") or 0),
            "scorable": int(r.get("scorable") or 0),
            "ok_rate": r.get("ok_rate"),
            "cost_usd": round(float(r.get("cost") or 0.0), 6),
            "last_recalled": r.get("last_recalled"),
        })

    return {
        "notes": notes,
        "window_days": since_days,
        "total_notes": len(notes),
        # This text is the honesty of the feature, not decoration. Without it
        # a model reads "ok_rate: 0.9" as "this note is good" and starts
        # trusting a number that only ever measured co-occurrence.
        "caveat": (
            "ASSOCIATION, NOT CAUSATION. ok_rate is the share of runs that "
            "read this note and then finished without raising. It does NOT "
            "mean the note was correct, helpful, or responsible for the "
            "outcome — nothing here observes answer quality, and a run that "
            "read ten notes credits all ten equally. Barge-ins (the user "
            "interrupting) are counted in cancelled_excluded and left OUT of "
            "ok_rate entirely, because interrupting is normal use, not "
            "failure. Treat a low ok_rate on a well-used note as a prompt to "
            "go READ it, never as grounds to delete it; and treat any note "
            "with a small 'scorable' count as noise."
        ),
    }
