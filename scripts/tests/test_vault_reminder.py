"""Vault-save reminder — unit tests.

Verifies the turn-counter mechanic and reminder text without any
external services (SQLite in-memory, env-var toggling).
"""
from __future__ import annotations

import os
import sys

from ._framework import TestContext, test


async def _make_db() -> object:
    """Return a fresh in-memory MemoryDB (connected)."""
    import aiosqlite

    class _Shim:
        def __init__(self, conn):
            self._conn = conn

    conn = await aiosqlite.connect(":memory:")
    # Bootstrap the schema so vault_save_reminders exists.
    from src.memory.db import SCHEMA_SQL
    await conn.executescript(SCHEMA_SQL)
    return _Shim(conn)


@test("vault_reminder", "default off — maybe_render_reminder returns None")
async def t_default_off(ctx: TestContext) -> None:
    old = os.environ.pop("OPENAGENT_VAULT_REMINDER_ENABLED", None)
    try:
        from src.learning import vault_reminder
        import importlib
        importlib.reload(vault_reminder)
        db = await _make_db()
        result = await vault_reminder.maybe_render_reminder(db, "sess-1")
        assert result is None, f"expected None when disabled, got: {result!r}"
    finally:
        if old is not None:
            os.environ["OPENAGENT_VAULT_REMINDER_ENABLED"] = old
        else:
            os.environ.pop("OPENAGENT_VAULT_REMINDER_ENABLED", None)


@test("vault_reminder", "enabled + every=3: fires at turn 3 and 6, None otherwise")
async def t_fires_at_every_n(ctx: TestContext) -> None:
    os.environ["OPENAGENT_VAULT_REMINDER_ENABLED"] = "1"
    os.environ["OPENAGENT_VAULT_REMINDER_EVERY_N_TURNS"] = "3"
    try:
        from src.learning import vault_reminder
        import importlib
        importlib.reload(vault_reminder)
        db = await _make_db()
        results = []
        for i in range(6):
            r = await vault_reminder.maybe_render_reminder(db, "sess-fire")
            results.append(r)
        # Turns are 1-indexed after bump: 1,2,3,4,5,6
        assert results[0] is None, f"turn 1 should be None, got {results[0]!r}"
        assert results[1] is None, f"turn 2 should be None, got {results[1]!r}"
        assert results[2] is not None, f"turn 3 should fire, got None"
        assert results[3] is None, f"turn 4 should be None, got {results[3]!r}"
        assert results[4] is None, f"turn 5 should be None, got {results[4]!r}"
        assert results[5] is not None, f"turn 6 should fire, got None"
    finally:
        os.environ.pop("OPENAGENT_VAULT_REMINDER_ENABLED", None)
        os.environ.pop("OPENAGENT_VAULT_REMINDER_EVERY_N_TURNS", None)


@test("vault_reminder", "reminder text contains 'vault' and 'wikilinks'")
async def t_reminder_text_content(ctx: TestContext) -> None:
    os.environ["OPENAGENT_VAULT_REMINDER_ENABLED"] = "1"
    os.environ["OPENAGENT_VAULT_REMINDER_EVERY_N_TURNS"] = "2"
    try:
        from src.learning import vault_reminder
        import importlib
        importlib.reload(vault_reminder)
        db = await _make_db()
        # Burn turn 1, get turn 2.
        await vault_reminder.maybe_render_reminder(db, "sess-text")
        reminder = await vault_reminder.maybe_render_reminder(db, "sess-text")
        assert reminder is not None, "expected reminder at turn 2"
        low = reminder.lower()
        assert "vault" in low, f"'vault' not found in reminder text: {reminder!r}"
        assert "wikilinks" in low, f"'wikilinks' not found in reminder text: {reminder!r}"
    finally:
        os.environ.pop("OPENAGENT_VAULT_REMINDER_ENABLED", None)
        os.environ.pop("OPENAGENT_VAULT_REMINDER_EVERY_N_TURNS", None)
