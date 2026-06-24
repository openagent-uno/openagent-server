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


@test("vault_reminder", "explicitly disabled — maybe_render_reminder returns None")
async def t_explicit_off(ctx: TestContext) -> None:
    os.environ["OPENAGENT_VAULT_REMINDER_ENABLED"] = "0"
    try:
        from src.learning import vault_reminder
        import importlib
        importlib.reload(vault_reminder)
        db = await _make_db()
        result = await vault_reminder.maybe_render_reminder(db, "sess-1")
        assert result is None, f"expected None when disabled, got: {result!r}"
    finally:
        os.environ.pop("OPENAGENT_VAULT_REMINDER_ENABLED", None)


@test("vault_reminder", "on by default — fires on the first turn")
async def t_default_on_first_turn(ctx: TestContext) -> None:
    # No env set → default ON. The first prompt must get a reminder.
    for k in ("OPENAGENT_VAULT_REMINDER_ENABLED", "OPENAGENT_VAULT_REMINDER_EVERY_N_TURNS"):
        os.environ.pop(k, None)
    from src.learning import vault_reminder
    import importlib
    importlib.reload(vault_reminder)
    db = await _make_db()
    first = await vault_reminder.maybe_render_reminder(db, "sess-default")
    assert first is not None, "expected a reminder on the first turn by default"


@test("vault_reminder", "every=3: fires at turn 1, 3, 6 (first prompt + every 3)")
async def t_fires_first_and_every_n(ctx: TestContext) -> None:
    os.environ["OPENAGENT_VAULT_REMINDER_ENABLED"] = "1"
    os.environ["OPENAGENT_VAULT_REMINDER_EVERY_N_TURNS"] = "3"
    try:
        from src.learning import vault_reminder
        import importlib
        importlib.reload(vault_reminder)
        db = await _make_db()
        fired = []
        for _ in range(7):  # turns 1..7
            r = await vault_reminder.maybe_render_reminder(db, "sess-fire")
            fired.append(r is not None)
        # 1-indexed: fire on 1 (first), 3, 6 (every 3); not 2,4,5,7
        assert fired == [True, False, True, False, False, True, False], fired
    finally:
        os.environ.pop("OPENAGENT_VAULT_REMINDER_ENABLED", None)
        os.environ.pop("OPENAGENT_VAULT_REMINDER_EVERY_N_TURNS", None)


@test("vault_reminder", "reminder text says ALWAYS save + has vault/wikilinks")
async def t_reminder_text_content(ctx: TestContext) -> None:
    os.environ["OPENAGENT_VAULT_REMINDER_ENABLED"] = "1"
    try:
        from src.learning import vault_reminder
        import importlib
        importlib.reload(vault_reminder)
        db = await _make_db()
        reminder = await vault_reminder.maybe_render_reminder(db, "sess-text")
        assert reminder is not None, "expected a reminder on turn 1"
        low = reminder.lower()
        assert "always" in low, f"'ALWAYS' not in reminder: {reminder!r}"
        assert "vault" in low, f"'vault' not found in reminder text: {reminder!r}"
        assert "wikilinks" in low, f"'wikilinks' not found in reminder text: {reminder!r}"
    finally:
        os.environ.pop("OPENAGENT_VAULT_REMINDER_ENABLED", None)
