"""One busy timeout for every writer on the agent DB.

SQLite in WAL mode has a single writer and this agent has many, several of
them in other processes. The lock errors we kept seeing — a workflow_run
stranded in 'running', a compaction thrown away, ``touch_device`` failing on
login — were never "the lock was held forever". They were connections given a
*different*, shorter patience than their neighbours: 10s in the MCP
subprocesses, 5s in compaction, and none at all in the runtime session store,
which is the engine that commits a big ``runs`` blob per step.

These tests pin the two halves of the fix: one number everyone reads, and the
runtime store no longer being the odd one out.
"""
from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import uuid

from ._framework import TestContext, test


class _Env:
    """Set/restore environment variables around a check."""

    def __init__(self, **values):
        self._values = values
        self._saved: dict[str, str | None] = {}

    def __enter__(self):
        for key, value in self._values.items():
            self._saved[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return self

    def __exit__(self, *_exc):
        for key, old in self._saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        return False


@test("sqlite_busy_timeout", "the shared budget is 60s and env-tunable")
async def t_shared_budget(ctx: TestContext) -> None:
    from src.memory.db import sqlite_busy_timeout_ms, sqlite_busy_timeout_s

    with _Env(OPENAGENT_SQLITE_BUSY_TIMEOUT_MS=None):
        assert sqlite_busy_timeout_ms() == 60_000
        assert sqlite_busy_timeout_s() == 60.0

    with _Env(OPENAGENT_SQLITE_BUSY_TIMEOUT_MS="15000"):
        assert sqlite_busy_timeout_ms() == 15_000
        assert sqlite_busy_timeout_s() == 15.0


@test("sqlite_busy_timeout", "a zero or unparseable budget falls back, never to 0")
async def t_no_instant_failure(ctx: TestContext) -> None:
    # "Fail instantly" is the bug this constant exists to remove, so neither a
    # typo nor a literal 0 may reintroduce it.
    from src.memory.db import sqlite_busy_timeout_ms

    for bad in ("0", "-1", "abc", ""):
        with _Env(OPENAGENT_SQLITE_BUSY_TIMEOUT_MS=bad):
            assert sqlite_busy_timeout_ms() == 60_000, bad


@test("sqlite_busy_timeout", "MemoryDB's connection carries the shared budget")
async def t_memorydb_pragma(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB, sqlite_busy_timeout_ms

    tmp = ctx.db_path.with_name(f"busy-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(tmp))
    try:
        await db.connect()
        conn = await db._ensure_connected()
        cursor = await conn.execute("PRAGMA busy_timeout")
        row = await cursor.fetchone()
        assert int(row[0]) == sqlite_busy_timeout_ms(), row[0]
        cursor = await conn.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()
        assert str(row[0]).lower() == "wal", row[0]
    finally:
        try:
            await db.close()
        except Exception:  # noqa: BLE001
            pass
        for suffix in ("", "-wal", "-shm"):
            try:
                tmp.with_name(tmp.name + suffix).unlink()
            except OSError:
                pass


@test("sqlite_busy_timeout", "the runtime session store waits by DEFAULT")
async def t_session_store_default_on(ctx: TestContext) -> None:
    # This is the regression that matters: the pragma hook shipped OFF, nobody
    # opted in, and the busiest writer on the file ran with SQLite's stock
    # settings — busy_timeout 0, i.e. give up on the first contended commit.
    from src.memory.store.sqlite.sqlite import (
        _make_session_store_engine, _session_store_pragma_enabled,
    )
    from src.memory.db import sqlite_busy_timeout_ms

    with _Env(
        OPENAGENT_SESSION_STORE_PRAGMA_ENABLED=None,
        OPENAGENT_SESSION_STORE_BUSY_TIMEOUT_SECONDS=None,
    ):
        assert _session_store_pragma_enabled() is True

        tmp = ctx.db_path.with_name(f"store-{uuid.uuid4().hex[:8]}.db")
        engine = _make_session_store_engine(f"sqlite:///{tmp}")
        try:
            from sqlalchemy.pool import NullPool

            assert isinstance(engine.pool, NullPool), type(engine.pool).__name__
            raw = engine.raw_connection()
            try:
                cur = raw.cursor()
                cur.execute("PRAGMA busy_timeout")
                busy = int(cur.fetchone()[0])
                cur.execute("PRAGMA journal_mode")
                journal = str(cur.fetchone()[0]).lower()
                cur.close()
            finally:
                raw.close()
        finally:
            engine.dispose()
            for suffix in ("", "-wal", "-shm"):
                try:
                    tmp.with_name(tmp.name + suffix).unlink()
                except OSError:
                    pass

    # Same patience as everyone else — a private, shorter one is how this
    # engine used to lose races it should have won.
    assert busy == sqlite_busy_timeout_ms(), busy
    assert journal == "wal", journal


@test("sqlite_busy_timeout", "single-row runtime reads finalize their WAL cursor")
async def t_runtime_store_single_row_read_releases_reader(ctx: TestContext) -> None:
    """Regression for the low-volume pin that survived after the boot fix.

    SQLAlchemy's ``fetchone`` leaves the result cursor open after returning a
    row; ``first`` closes it. Keep the Session and Result objects alive across
    a sibling write to prove the read mark itself has already been released.
    """
    from sqlalchemy import text
    from src.memory.store.sqlite import SqliteDb

    tmp = ctx.db_path.with_name(f"wal-reader-{uuid.uuid4().hex[:8]}.db")
    seed = sqlite3.connect(tmp)
    seed.execute("PRAGMA journal_mode=WAL")
    seed.execute("CREATE TABLE rows_for_reader (value INTEGER)")
    seed.executemany(
        "INSERT INTO rows_for_reader(value) VALUES (?)",
        ((value,) for value in range(256)),
    )
    seed.commit()
    seed.close()

    store = SqliteDb(db_file=str(tmp))
    session = store.Session()
    result = session.execute(text("SELECT value FROM rows_for_reader ORDER BY value"))
    assert result.first()[0] == 0

    writer = sqlite3.connect(tmp)
    try:
        writer.execute("INSERT INTO rows_for_reader(value) VALUES (999)")
        writer.commit()
        checkpoint = writer.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
        assert checkpoint[2] == checkpoint[1], checkpoint

        source = Path(__file__).resolve().parents[2] / "src/memory/store/sqlite/sqlite.py"
        assert ".fetchone()" not in source.read_text(), (
            "runtime single-row reads must use first() so CursorResult closes"
        )
    finally:
        writer.close()
        store.close()
        for suffix in ("", "-wal", "-shm"):
            try:
                tmp.with_name(tmp.name + suffix).unlink()
            except OSError:
                pass


@test("sqlite_busy_timeout", "an operator can still switch the store hook off")
async def t_session_store_opt_out(ctx: TestContext) -> None:
    from src.memory.store.sqlite.sqlite import _session_store_pragma_enabled

    for off in ("0", "false", "no", "off", ""):
        with _Env(OPENAGENT_SESSION_STORE_PRAGMA_ENABLED=off):
            assert _session_store_pragma_enabled() is False, off
    with _Env(OPENAGENT_SESSION_STORE_PRAGMA_ENABLED="1"):
        assert _session_store_pragma_enabled() is True


@test("sqlite_busy_timeout", "no writer is left with a private, shorter patience")
async def t_no_short_timeouts_left(ctx: TestContext) -> None:
    # A grep-style guard: the literals that caused this (a bare
    # ``aiosqlite.connect(path)`` or a hard-coded busy_timeout) must not come
    # back into the write paths that share the agent DB.
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    watched = [
        "src/mcp/servers/_common.py",
        "src/mcp/servers/scheduler/server.py",
        "src/mcp/servers/workflow_manager/server.py",
        "src/mcp/servers/events_manager/server.py",
    ]
    for rel in watched:
        text = (root / rel).read_text()
        assert "aiosqlite.connect(path)" not in text, f"{rel}: connect without a timeout"
        assert "busy_timeout = 10000" not in text, f"{rel}: hard-coded 10s busy_timeout"
        assert "sqlite_busy_timeout" in text, f"{rel}: not using the shared budget"
