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


@test("db_metadata_parse", "_parse_metadata unwraps double-JSON-encoded dicts")
async def t_parse_metadata_unwraps_double_encoded(ctx: TestContext) -> None:
    """Some writers (Agno's serialize_session_json_fields when handed a
    stringified metadata field) store metadata as ``json.dumps(json_str)``
    — a JSON-encoded string whose decoded value is itself a JSON object.
    The parser must recover the inner dict so the session-list filter in
    ``list_all_sessions`` sees the right ``client_id``.

    Observed in the wild on lyra: rows shaped like
    ``"{\\"client_id\\": \\"alessandro\\", ...}"`` caused every
    desktop-originated session to vanish from ``/api/sessions``.
    """
    from src.memory.db import MemoryDB
    import json as _json

    inner = {"client_id": "alessandro", "device_id": "deadbeef", "title": "hi"}
    double = _json.dumps(_json.dumps(inner))  # JSON-encoded JSON string
    assert MemoryDB._parse_metadata(double) == inner

    # Whitespace and odd-but-valid keys also survive.
    inner2 = {"client_id": "__bridge_telegram"}
    assert MemoryDB._parse_metadata(_json.dumps(_json.dumps(inner2))) == inner2

    # Double-encoded non-dict (e.g. ``json.dumps(json.dumps("foo"))``) is
    # still garbage — must fall back to ``{}`` rather than returning a str.
    assert MemoryDB._parse_metadata(_json.dumps(_json.dumps("foo"))) == {}


@test("db_metadata_parse", "list_all_sessions filter matches on double-encoded rows")
async def t_list_all_sessions_unwraps_double_encoded(ctx: TestContext) -> None:
    """End-to-end: a row whose ``metadata`` column was written in the
    double-encoded shape must still match the per-handle filter and show
    up in ``list_all_sessions``. Without the parser unwrap, the filter
    sees ``client_id == ""`` and drops the row."""
    import sqlite3
    import json as _json
    from src.memory.db import MemoryDB

    inner = {"client_id": "alessandro", "device_id": "feedface", "title": "broken row"}
    double = _json.dumps(_json.dumps(inner))  # the corrupt shape on disk

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
            "INSERT INTO agno_sessions "
            "(session_id, session_type, user_id, metadata, created_at, updated_at) "
            "VALUES (?, 'agent', 'openagent', ?, 100, 200)",
            ("session-broken-row", double),
        )
        conn.commit()
    finally:
        conn.close()

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        rows = await db.list_all_sessions("alessandro", limit=50)
        sids = [r["session_id"] for r in rows]
        assert "session-broken-row" in sids, (
            f"double-encoded row was filtered out (sids={sids})"
        )
        row = next(r for r in rows if r["session_id"] == "session-broken-row")
        assert row["client_id"] == "alessandro"
        assert row["title"] == "broken row"
    finally:
        await db.close()


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
