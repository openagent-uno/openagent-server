"""Async handler behind the ``vault_contradiction_candidates`` tool.

A thin wrapper over the deterministic generator in
``src.memory.vault.contradiction``. It lives in its own module (rather than in
``handlers.py``) for the same reason ``recall.py`` does: it is a distinct
concern with its own honesty contract, and it is the file to read when you want
to know what this tool will and will not claim.

The generator never calls a model, so this tool works with zero providers
configured (vision §17). Judging the candidates is the AI's job -- see the
tool docstring, which is deliberately blunt about the difference.
"""
from __future__ import annotations

import asyncio

from src.memory.vault.contradiction import (
    ContradictionConfig,
    find_contradiction_candidates,
)
from src.memory.vault.service import get_service


async def vault_contradiction_candidates(limit: int = 20,
                                         sync: bool = True) -> dict:
    """Note pairs that LOOK like they disagree about one technical subject.

    Vision §5: contradictions must be "flagged and reconciled rather than
    silently overwritten". This is the flagging half. It is CANDIDATE
    GENERATION, NOT DETECTION: code matched opposing wording (one note calls a
    subject deprecated/forbidden/absent, another uses it affirmatively) about a
    shared identifier. It did not read or understand either note.

    Expect roughly half to be false positives -- a different scope, a migration
    table, or a cue that belongs to another subject on the same line. Read BOTH
    notes in full before you touch anything, then reconcile by correcting or
    retiring whichever is actually stale.

    Absence of candidates is NOT proof the vault is consistent. This sees only
    explicit dead/deprecated/forbidden wording about identifier-shaped subjects
    in durable notes; event records (receipts, logs, dated snapshots) are
    excluded because two records of different events never contradict. A
    contradiction phrased any other way is invisible here.

    NEVER delete a note on this signal alone.
    """
    svc = get_service()
    if sync:
        await svc.sync()
    index = await svc._ensure_index()
    cfg = ContradictionConfig()
    report = await asyncio.to_thread(
        find_contradiction_candidates,
        index,
        svc.vault_root,
        cfg,
        journal_root=svc.journal_root,
    )
    payload = report.to_dict()
    shown = payload["candidates"][:limit]
    payload["candidates"] = shown
    payload["candidates_shown"] = len(shown)
    return payload
