"""Regression: ``MemoryDB._parse_metadata`` must always return a dict.

Production crash on mixout (3 occurrences) — ``upsert_session`` raised
``TypeError: 'NoneType' object does not support item assignment`` at the
``meta["client_id"] = client_id`` assignment. Root cause: the
``metadata`` column held the literal JSON ``"null"`` (decodes to
``None``), and ``_parse_metadata`` returned whatever ``json.loads``
produced without enforcing dict shape. Callers
(``upsert_session``, ``list_all_sessions``, ``get_session``) then
treated the result as a dict and crashed.

The fix forces non-dict JSON values to coerce to ``{}``.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("db_metadata_parse", "_parse_metadata coerces non-dict JSON to empty dict")
async def t_parse_metadata_coerces(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB

    # None and empty-ish inputs.
    assert MemoryDB._parse_metadata(None) == {}
    assert MemoryDB._parse_metadata("") == {}
    assert MemoryDB._parse_metadata("{}") == {}

    # Valid JSON that is NOT a dict — the production crash case.
    assert MemoryDB._parse_metadata("null") == {}
    assert MemoryDB._parse_metadata("[]") == {}
    assert MemoryDB._parse_metadata('"plain string"') == {}
    assert MemoryDB._parse_metadata("42") == {}

    # Invalid JSON still returns {}.
    assert MemoryDB._parse_metadata("not json") == {}

    # Dict input passes through.
    assert MemoryDB._parse_metadata('{"a": 1}') == {"a": 1}


@test("db_metadata_parse", "upsert_session survives a 'null' metadata row")
async def t_upsert_survives_null_metadata(ctx: TestContext) -> None:
    """Pre-seed an ``agno_sessions`` row whose ``metadata`` column is the
    literal ``'null'`` (decodes to ``None``), then call
    ``upsert_session``. Before the fix this raised
    ``TypeError: 'NoneType' object does not support item assignment``.
    """
    import sqlite3
    from src.memory.db import MemoryDB

    # Create the agno_sessions schema and seed a corrupt-metadata row.
    conn = sqlite3.connect(str(ctx.db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agno_sessions (
                session_id TEXT PRIMARY KEY,
                session_type TEXT,
                agent_id TEXT,
                team_id TEXT,
                workflow_id TEXT,
                user_id TEXT,
                session_data TEXT,
                agent_data TEXT,
                team_data TEXT,
                workflow_data TEXT,
                metadata TEXT,
                runs TEXT,
                summary TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO agno_sessions
              (session_id, session_type, user_id, metadata,
               created_at, updated_at)
            VALUES (?, 'agent', 'openagent', 'null', 100, 200)
            """,
            ("tg:bad-meta",),
        )
        conn.commit()
    finally:
        conn.close()

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        # The line that used to crash.
        await db.upsert_session(
            "tg:bad-meta",
            client_id="client-xyz",
            title="restored",
            model="anthropic:claude-opus-4-7",
            framework="claude_cli",
        )
        row = await db.get_session("tg:bad-meta")
        assert row is not None
        assert row["client_id"] == "client-xyz"
        assert row["title"] == "restored"
        assert row["model"] == "anthropic:claude-opus-4-7"
        assert row["framework"] == "claude_cli"
    finally:
        await db.close()
