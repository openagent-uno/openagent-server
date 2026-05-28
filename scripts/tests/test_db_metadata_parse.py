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
            CREATE TABLE IF NOT EXISTS sessions (
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
            "INSERT INTO sessions "
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


@test("db_metadata_parse", "list_all_sessions LIMIT applies AFTER the client_id filter")
async def t_list_all_sessions_limit_post_filter(ctx: TestContext) -> None:
    """Seed 100 rows that DON'T belong to ``alessandro`` (with newer
    ``updated_at``) and a handful of rows that DO (mixed shapes:
    proper JSON-object and double-encoded). With ``limit=50``, the
    pre-fix code returned 0 of alessandro's rows because the 100
    newer noise rows filled the top 50. After the fix, all matching
    rows come back regardless of where they sit in the global order.
    """
    import sqlite3
    import json as _json
    from src.memory.db import MemoryDB

    conn = sqlite3.connect(str(ctx.db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
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
        # 100 newer rows for an unrelated client (workflow / classifier
        # / other-user noise). updated_at = 10_000 + i so they sort
        # above the alessandro rows below.
        for i in range(100):
            conn.execute(
                "INSERT INTO sessions "
                "(session_id, session_type, user_id, metadata, "
                " created_at, updated_at) "
                "VALUES (?, 'agent', 'openagent', ?, ?, ?)",
                (
                    f"workflow:noise:{i}",
                    _json.dumps({"client_id": "__bridge_telegram"}),
                    1000, 10_000 + i,
                ),
            )
        # 4 alessandro rows interleaved with older timestamps. Mix the
        # storage shape so the filter has to handle both:
        #   - proper JSON object (the "right" shape)
        #   - double-encoded JSON string (the wild shape)
        proper_meta = _json.dumps({"client_id": "alessandro", "title": "row A"})
        double_meta = _json.dumps(_json.dumps({"client_id": "alessandro", "title": "row B"}))
        proper2 = _json.dumps({"client_id": "alessandro", "title": "row C"})
        double2 = _json.dumps(_json.dumps({"client_id": "alessandro", "title": "row D"}))
        for sid, meta, ts in [
            ("session-a", proper_meta, 5_000),
            ("session-b", double_meta, 5_001),
            ("session-c", proper2, 5_002),
            ("session-d", double2, 5_003),
        ]:
            conn.execute(
                "INSERT INTO sessions "
                "(session_id, session_type, user_id, metadata, "
                " created_at, updated_at) "
                "VALUES (?, 'agent', 'openagent', ?, 1000, ?)",
                (sid, meta, ts),
            )
        conn.commit()
    finally:
        conn.close()

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        rows = await db.list_all_sessions("alessandro", limit=50)
        sids = {r["session_id"] for r in rows}
        # All four of OUR alessandro rows must be present — the 100
        # noise rows above must not push them off the page.
        for sid in ("session-a", "session-b", "session-c", "session-d"):
            assert sid in sids, f"missing {sid}; got {sorted(sids)}"
        # No noise rows should be in the result.
        assert not any(s.startswith("workflow:noise:") for s in sids), (
            f"noise rows leaked through filter: {sorted(sids)}"
        )
        # Titles survive the unwrap from both shapes.
        title_by_sid = {r["session_id"]: r["title"] for r in rows}
        assert title_by_sid["session-a"] == "row A"
        assert title_by_sid["session-b"] == "row B"
        assert title_by_sid["session-c"] == "row C"
        assert title_by_sid["session-d"] == "row D"
        # Ordering of OUR four rows must be updated_at DESC.
        our_in_order = [r["session_id"] for r in rows
                        if r["session_id"] in {"session-a", "session-b", "session-c", "session-d"}]
        assert our_in_order == ["session-d", "session-c", "session-b", "session-a"], (
            f"wrong ordering: {our_in_order}"
        )
    finally:
        await db.close()


@test("db_metadata_parse", "list_session_runs unwraps double-encoded runs column")
async def t_list_session_runs_unwraps(ctx: TestContext) -> None:
    """The runs column is JSON too, and Agno's serialize path will
    double-encode it the same way it does metadata. Without an
    unwrap, ``list_session_runs`` sees a str (not a list) and
    returns ``[]`` — clicking a session in the desktop shows no
    messages even though the conversation history is there on disk.
    """
    import sqlite3
    import json as _json
    from src.memory.db import MemoryDB

    real_runs = [{
        "run_id": "r1",
        "status": "completed",
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi back"},
        ],
        "created_at": 123,
    }]
    double_runs = _json.dumps(_json.dumps(real_runs))

    conn = sqlite3.connect(str(ctx.db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
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
            "INSERT INTO sessions "
            "(session_id, session_type, user_id, runs, "
            " created_at, updated_at) "
            "VALUES (?, 'agent', 'openagent', ?, 100, 200)",
            ("session-runs-double", double_runs),
        )
        conn.commit()
    finally:
        conn.close()

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        runs = await db.list_session_runs("session-runs-double", limit=10)
        assert len(runs) == 1, f"expected 1 run, got {len(runs)}: {runs}"
        run = runs[0]
        assert run["run_id"] == "r1"
        msgs = run["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user" and msgs[0]["content"] == "hello"
        assert msgs[1]["role"] == "assistant" and msgs[1]["content"] == "hi back"
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
            CREATE TABLE IF NOT EXISTS sessions (
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
            INSERT INTO sessions
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
