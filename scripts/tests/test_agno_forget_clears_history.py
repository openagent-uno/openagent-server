"""Regression: AgnoProvider.forget_session must wipe the stored
session row so the next generate() on that session_id doesn't
auto-load the prior transcript (and summary).

Before the fix, ``AgnoProvider`` only inherited BaseModel's default
``forget_session``, which called ``close_session`` — itself a no-op
for Agno (nothing to disconnect; history lives in an SqliteDb). So
``/clear`` and the scheduler's per-fire forget never actually erased
agno-backed history; with ``add_history_to_context=True`` the very
next turn loaded all prior messages plus the rolling session
summary back into context.

The fix: AgnoProvider implements a real ``forget_session`` that
calls ``SqliteDb.delete_session`` (Agno's native API), with a raw
SQL fallback against the ``agno_sessions`` table for API drift.

These tests cover both paths: the API path with a seeded real row,
and the fallback path with a pre-created table + the native API
artificially broken.
"""
from __future__ import annotations

import sqlite3

from ._framework import TestContext, TestSkip, test


def _agno_available() -> bool:
    try:
        from agno.db.sqlite import SqliteDb  # noqa: F401
        return True
    except ImportError:
        return False


def _seed_agno_sessions_table(db_path: str) -> None:
    """Create the agno_sessions schema and give it one seeded row.

    We use raw SQL because Agno's ORM upsert has NOT-NULL
    constraints tied to its own helper code that's awkward to stub.
    The column list mirrors Agno 2.x — enough for ``delete_session``
    to match and delete by session_id.
    """
    conn = sqlite3.connect(db_path)
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
            INSERT OR REPLACE INTO agno_sessions
              (session_id, session_type, agent_id, user_id, runs, summary,
               created_at, updated_at)
            VALUES (?, 'agent', 'a1', 'openagent',
                    '[{"messages":[{"role":"user","content":"secret word is banana"}]}]',
                    'prior summary', 100, 200)
            """,
            ("tg:agno-user",),
        )
        conn.commit()
    finally:
        conn.close()


def _row_count(db_path: str, session_id: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM agno_sessions WHERE session_id=?",
            (session_id,),
        ).fetchone()[0]
    finally:
        conn.close()


@test("agno_forget", "forget_session wipes row via Agno native API")
async def t_forget_via_native_api(ctx: TestContext) -> None:
    if not _agno_available():
        raise TestSkip("agno not installed")

    from openagent.models.agno_provider import AgnoProvider

    db_file = ctx.test_dir / "agno-forget.db"
    _seed_agno_sessions_table(str(db_file))
    assert _row_count(str(db_file), "tg:agno-user") == 1, "seed failed"

    provider = AgnoProvider(
        model="agno:openai/gpt-4o-mini",
        providers_config=[{"name": "openai", "framework": "agno"}],
        db_path=str(db_file),
    )
    await provider.forget_session("tg:agno-user")

    assert _row_count(str(db_file), "tg:agno-user") == 0, (
        "session row survived forget_session via native API"
    )


@test("agno_forget", "forget_session falls back to raw SQL when API fails")
async def t_forget_fallback_sql(ctx: TestContext) -> None:
    if not _agno_available():
        raise TestSkip("agno not installed")

    from openagent.models.agno_provider import AgnoProvider
    from agno.db.sqlite import SqliteDb

    db_file = ctx.test_dir / "agno-forget-fallback.db"
    _seed_agno_sessions_table(str(db_file))
    assert _row_count(str(db_file), "tg:agno-user") == 1, "seed failed"

    # Break the native API to force the fallback path.
    original = SqliteDb.delete_session

    def _broken(self, session_id, user_id=None):  # type: ignore[override]
        raise RuntimeError("simulated API drift")

    SqliteDb.delete_session = _broken  # type: ignore[assignment]
    try:
        provider = AgnoProvider(
            model="agno:openai/gpt-4o-mini",
            providers_config=[{"name": "openai", "framework": "agno"}],
            db_path=str(db_file),
        )
        await provider.forget_session("tg:agno-user")
    finally:
        SqliteDb.delete_session = original  # type: ignore[assignment]

    assert _row_count(str(db_file), "tg:agno-user") == 0, (
        "session row survived forget_session fallback path"
    )


@test("agno_forget", "forget_session on nonexistent session is a no-op")
async def t_forget_nonexistent(ctx: TestContext) -> None:
    if not _agno_available():
        raise TestSkip("agno not installed")

    from openagent.models.agno_provider import AgnoProvider

    db_file = ctx.test_dir / "agno-forget-noop.db"
    # Brand-new DB file; schema not yet created — fallback must cope.
    provider = AgnoProvider(
        model="agno:openai/gpt-4o-mini",
        providers_config=[{"name": "openai", "framework": "agno"}],
        db_path=str(db_file),
    )
    # Must not raise even though the DB has no agno_sessions table yet.
    await provider.forget_session("tg:nobody")
