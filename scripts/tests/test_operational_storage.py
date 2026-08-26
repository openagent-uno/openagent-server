"""Pure-unit invariants for additive operational-storage v2 bootstrap."""

from __future__ import annotations

import errno
import sqlite3
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import aiosqlite

from ._framework import TestContext, test


_LEGACY_SCHEMA = """
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    updated_at INTEGER
);
INSERT INTO sessions(session_id, updated_at) VALUES ('legacy-canary', 1);
"""


def _seed_legacy(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_LEGACY_SCHEMA)
    finally:
        conn.close()


async def _migration_state(path: Path) -> tuple[str, str, dict[str, int]]:
    from src.memory.operational.schema import MIGRATION_ID

    async with aiosqlite.connect(path) as conn:
        ledger = await (
            await conn.execute(
                "SELECT status FROM schema_migrations WHERE migration_id=?",
                (MIGRATION_ID,),
            )
        ).fetchone()
        phase = await (
            await conn.execute(
                "SELECT phase FROM storage_migration_state WHERE singleton_id=1"
            )
        ).fetchone()
        events = await (
            await conn.execute(
                "SELECT event_type, COUNT(*) FROM storage_migration_journal "
                "GROUP BY event_type"
            )
        ).fetchall()
        integrity = await (await conn.execute("PRAGMA integrity_check")).fetchone()
    assert ledger is not None and phase is not None
    assert integrity is not None and integrity[0] == "ok"
    return str(ledger[0]), str(phase[0]), {str(row[0]): int(row[1]) for row in events}


@test("operational_storage", "first migration backs up; repeat is journal-idempotent")
async def t_backup_and_idempotence(_ctx: TestContext) -> None:
    from src.memory.operational.schema import ensure_operational_storage

    with TemporaryDirectory(prefix="openagent-operational-") as directory:
        path = Path(directory) / "openagent.db"
        _seed_legacy(path)
        conn = await aiosqlite.connect(path)
        try:
            backup = await ensure_operational_storage(
                conn, str(path), app_version="test-beta"
            )
            assert backup is not None
            assert Path(backup.database_path).is_file()
            assert Path(backup.manifest_path).is_file()
            first = await _migration_state(path)

            repeated = await ensure_operational_storage(
                conn, str(path), app_version="test-beta"
            )
            second = await _migration_state(path)
        finally:
            await conn.close()

        assert repeated is None
        assert first == second
        status, phase, events = second
        assert (status, phase) == ("complete", "shadow")
        assert events == {
            "backup_verified": 1,
            "ddl_completed": 1,
            "ddl_started": 1,
            "phase_changed": 1,
        }


@test("operational_storage", "committed DDL + failed bridge resumes without false backup")
async def t_executescript_crash_recovery(_ctx: TestContext) -> None:
    import src.memory.operational.schema as schema

    with TemporaryDirectory(prefix="openagent-operational-recovery-") as directory:
        path = Path(directory) / "openagent.db"
        _seed_legacy(path)
        conn = await aiosqlite.connect(path)
        real_bridge = schema.legacy_bridge_sql
        schema.legacy_bridge_sql = lambda: "NOT VALID SQL;"
        try:
            try:
                await schema.ensure_operational_storage(
                    conn, str(path), app_version="test-beta"
                )
            except sqlite3.Error:
                pass
            else:
                raise AssertionError("the bridge failure was not injected")
            failed = await _migration_state(path)
        finally:
            schema.legacy_bridge_sql = real_bridge

        try:
            recovered_backup = await schema.ensure_operational_storage(
                conn, str(path), app_version="test-beta"
            )
            recovered = await _migration_state(path)
        finally:
            await conn.close()

        assert failed[0:2] == ("failed", "legacy")
        # The already-committed v2 DB must never receive a misleading
        # "pre-migration" backup during recovery.
        assert recovered_backup is None
        assert recovered[0:2] == ("complete", "shadow")
        assert recovered[2]["ddl_completed"] == 1
        assert recovered[2]["phase_changed"] == 1


@test("operational_storage", "non-contention lock errors fail immediately")
async def t_lock_error_classification(_ctx: TestContext) -> None:
    import fcntl

    from src.memory.operational.schema import (
        OperationalMigrationError,
        _MigrationFileLock,
    )

    with TemporaryDirectory(prefix="openagent-operational-lock-") as directory:
        lock = _MigrationFileLock(Path(directory) / "openagent.db")
        real_flock = fcntl.flock

        def _bad_descriptor(_fd: int, _operation: int) -> None:
            raise OSError(errno.EBADF, "synthetic bad descriptor")

        fcntl.flock = _bad_descriptor
        started = time.monotonic()
        try:
            try:
                lock.acquire(timeout_s=5.0)
            except OperationalMigrationError as exc:
                assert isinstance(exc.__cause__, OSError)
                assert exc.__cause__.errno == errno.EBADF
            else:
                raise AssertionError("EBADF was treated as successful lock acquisition")
        finally:
            fcntl.flock = real_flock
            lock.release()
        assert time.monotonic() - started < 0.5


@test("operational_storage", "failed backup leaves no temporary or orphan evidence")
async def t_backup_failure_cleanup(_ctx: TestContext) -> None:
    import src.memory.operational.schema as schema

    with TemporaryDirectory(prefix="openagent-operational-cleanup-") as directory:
        path = Path(directory) / "openagent.db"
        _seed_legacy(path)
        conn = await aiosqlite.connect(path)
        real_fsync_directory = schema._fsync_directory
        calls = 0

        def _fail_first_fsync(_path: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError(errno.EIO, "synthetic directory fsync failure")
            real_fsync_directory(_path)

        schema._fsync_directory = _fail_first_fsync
        try:
            try:
                await schema._verified_backup(
                    conn, path, app_version="test-beta"
                )
            except OSError as exc:
                assert exc.errno == errno.EIO
            else:
                raise AssertionError("the backup fsync failure was not injected")
        finally:
            schema._fsync_directory = real_fsync_directory
            await conn.close()

        backup_dir = path.parent / "backups"
        assert backup_dir.is_dir()
        assert list(backup_dir.iterdir()) == []


@test("operational_storage", "legacy session projects runs, messages, tools and outbox")
async def t_session_projection(_ctx: TestContext) -> None:
    import json

    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-operational-project-") as directory:
        path = Path(directory) / "openagent.db"
        db = MemoryDB(str(path))
        await db.connect()
        try:
            await db.upsert_session(
                "projection-canary",
                client_id="alice",
                title="Projection canary",
            )
            conn = db._conn
            assert conn is not None
            runs = [
                {
                    "run_id": "run-1",
                    "status": "COMPLETED",
                    "created_at": 1_700_000_000,
                    "content": "assistant canary",
                    "messages": [
                        {
                            "id": "message-user",
                            "role": "user",
                            "content": "user canary",
                        },
                        {
                            "id": "message-assistant",
                            "role": "assistant",
                            "content": "assistant canary",
                        },
                    ],
                    "tools": [
                        {
                            "tool_call_id": "tool-call-1",
                            "tool_name": "shell_execute",
                            "tool_args": {"cmd": "echo canary"},
                            "result": "canary result",
                            "status": "completed",
                        }
                    ],
                }
            ]
            await conn.execute(
                "UPDATE sessions SET runs=?, updated_at=? WHERE session_id=?",
                (json.dumps(runs), 1_700_000_002, "projection-canary"),
            )
            await db._project_operational_session("projection-canary")
            await conn.commit()

            session = await (
                await conn.execute(
                    "SELECT owner_principal_id, source_version, completeness "
                    "FROM sessions_v2 WHERE id='projection-canary'"
                )
            ).fetchone()
            messages = await (
                await conn.execute(
                    "SELECT role, text FROM session_messages "
                    "WHERE session_id='projection-canary' ORDER BY sequence"
                )
            ).fetchall()
            tools = await (
                await conn.execute(
                    "SELECT tool_name, status FROM tool_invocations "
                    "WHERE session_id='projection-canary'"
                )
            ).fetchall()
            pending = await (
                await conn.execute(
                    "SELECT COUNT(*) FROM legacy_session_changes "
                    "WHERE session_id='projection-canary' AND processed_at_ms IS NULL"
                )
            ).fetchone()
        finally:
            await db.close()

        assert tuple(session) == ("user:alice", 2, "complete")
        assert [tuple(row) for row in messages] == [
            ("user", "user canary"),
            ("assistant", "assistant canary"),
        ]
        assert [tuple(row) for row in tools] == [("shell_execute", "success")]
        assert pending is not None and pending[0] == 0


@test("operational_storage", "SQLAlchemy runtime write is query-ready before restart")
async def t_sqlalchemy_same_transaction_projection(_ctx: TestContext) -> None:
    from src.core._run_state.agent import RunOutput
    from src.core._run_state.base import RunStatus
    from src.memory.db import MemoryDB
    from src.memory.sessions import AgentSession
    from src.memory.store.sqlite import SqliteDb
    from src.models.providers.message import Message

    with TemporaryDirectory(prefix="openagent-operational-runtime-") as directory:
        path = Path(directory) / "openagent.db"
        bootstrap = MemoryDB(str(path))
        await bootstrap.connect()
        await bootstrap.close()

        now = int(time.time())
        runtime = SqliteDb(db_file=str(path), session_table="sessions")
        session = AgentSession(
            session_id="runtime-canary",
            agent_id="agent-canary",
            user_id="openagent",
            metadata={"client_id": "alice", "title": "Runtime canary"},
            created_at=now,
            updated_at=now,
        )
        session.upsert_run(
            RunOutput(
                run_id="runtime-run",
                agent_id="agent-canary",
                session_id="runtime-canary",
                user_id="openagent",
                content="runtime answer",
                messages=[
                    Message(role="user", content="runtime question"),
                    Message(role="assistant", content="runtime answer"),
                ],
                status=RunStatus.completed,
                created_at=now,
            )
        )
        try:
            assert runtime.upsert_session(session) is not None
        finally:
            runtime.close()

        # No MemoryDB reconnect/restart and no explicit reconciler call here:
        # the runtime transaction itself must leave history/search sources ready.
        conn = sqlite3.connect(path)
        try:
            projected = conn.execute(
                "SELECT owner_principal_id, completeness FROM sessions_v2 "
                "WHERE id='runtime-canary'"
            ).fetchone()
            message_count = conn.execute(
                "SELECT COUNT(*) FROM session_messages "
                "WHERE session_id='runtime-canary'"
            ).fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM legacy_session_changes "
                "WHERE session_id='runtime-canary' AND processed_at_ms IS NULL"
            ).fetchone()[0]
            outbox = conn.execute(
                "SELECT COUNT(*) FROM search_outbox "
                "WHERE source_id='runtime-canary' AND source_kind='session'"
            ).fetchone()[0]
        finally:
            conn.close()

        assert projected == ("user:alice", "complete")
        assert message_count == 2
        assert pending == 0
        assert outbox == 1


@test("operational_storage", "ownerless visibility is limited to installation/quarantine")
async def t_ownerless_visibility_policy(_ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.memory.operational.schema import operational_search_schema_sql

    with TemporaryDirectory(prefix="openagent-operational-acl-") as directory:
        path = Path(directory) / "openagent.db"
        db = MemoryDB(str(path))
        await db.connect()
        conn = db._conn
        assert conn is not None
        try:
            tenant = "installation:test"
            common = (
                tenant,
                "legacy-unattributed",
                "agent",
                "automation",
                "legacy",
                "active",
                "unknown",
                "{}",
                1,
                1,
                1,
            )
            await conn.execute(
                "INSERT INTO sessions_v2 "
                "(id, tenant_id, owner_principal_id, visibility, title, "
                "session_type, kind, origin, status, completeness, source_version, "
                "metadata_json, created_at_ms, updated_at_ms, last_activity_at_ms) "
                "VALUES ('ownerless-installation', ?, NULL, 'installation_shared', ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)",
                common,
            )
            try:
                await conn.execute(
                    "INSERT INTO sessions_v2 "
                    "(id, tenant_id, owner_principal_id, visibility, title, "
                    "session_type, kind, origin, status, completeness, source_version, "
                    "metadata_json, created_at_ms, updated_at_ms, last_activity_at_ms) "
                    "VALUES ('ownerless-private', ?, NULL, 'private', ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)",
                    common,
                )
            except sqlite3.IntegrityError:
                pass
            else:
                raise AssertionError("ownerless private session passed the ACL CHECK")
            await conn.rollback()
        finally:
            await db.close()

        search = sqlite3.connect(":memory:")
        try:
            search.executescript(operational_search_schema_sql())
            base = (
                "doc-installation",
                tenant,
                "installation_shared",
                "session_metadata",
                "session",
                "legacy-unattributed",
                "chat",
                "legacy-unattributed",
                "chat",
                1,
                1,
                "extractor-test",
                "redaction-test",
                "safe",
                "unknown",
                "0" * 64,
            )
            search.execute(
                "INSERT INTO search_documents "
                "(doc_id, tenant_id, owner_principal_id, visibility, acl_version, "
                "document_kind, resource_type, resource_id, root_kind, root_id, "
                "target_kind, session_id, title_safe, author_display_safe, "
                "occurred_at_ms, updated_at_ms, source_version, extractor_version, "
                "redaction_version, sensitivity, completeness, content_hash) "
                "VALUES (?, ?, NULL, ?, 1, ?, ?, ?, ?, ?, ?, 'legacy-unattributed', "
                "'', '', ?, ?, 1, ?, ?, ?, ?, ?)",
                base,
            )
        finally:
            search.close()
