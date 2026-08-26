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


def _reused_tool_call_runs() -> list[dict[str, object]]:
    """Two distinct runs that legitimately reuse one provider call id."""

    return [
        {
            "run_id": "run-a",
            "status": "COMPLETED",
            "created_at": 1_700_000_000,
            "messages": [
                {
                    "id": "tool-message-a",
                    "role": "tool",
                    "tool_call_id": "reused-call",
                    "content": "first distinct result",
                }
            ],
            "tools": [
                {
                    "tool_call_id": "reused-call",
                    "tool_name": "shell_execute",
                    "tool_args": {"cmd": "printf first"},
                    "result": "first distinct result",
                    "status": "completed",
                }
            ],
        },
        {
            "run_id": "run-b",
            "status": "COMPLETED",
            "created_at": 1_700_000_100,
            "messages": [
                {
                    "id": "tool-message-b",
                    "role": "tool",
                    "tool_call_id": "reused-call",
                    "content": "second distinct result",
                }
            ],
            "tools": [
                {
                    "tool_call_id": "reused-call",
                    "tool_name": "shell_execute",
                    "tool_args": {"cmd": "printf second"},
                    "result": "second distinct result",
                    "status": "completed",
                }
            ],
        },
    ]


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
            pending_after_repair = await (
                await conn.execute(
                    "SELECT COUNT(*) FROM legacy_session_changes "
                    "WHERE processed_at_ms IS NULL"
                )
            ).fetchone()
            first = await _migration_state(path)

            repeated = await ensure_operational_storage(
                conn, str(path), app_version="test-beta"
            )
            second = await _migration_state(path)
        finally:
            await conn.close()

        assert repeated is None
        # A first upgrade has no backfill checkpoint yet. The post-v2 repair
        # must not synchronously enumerate/queue the whole legacy corpus.
        assert pending_after_repair is not None
        assert int(pending_after_repair[0]) == 0
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


@test("operational_storage", "committed tool-context DDL resumes without duplicate retries")
async def t_tool_context_repair_crash_recovery(_ctx: TestContext) -> None:
    import json

    import src.memory.operational.schema as schema
    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-operational-repair-recovery-") as directory:
        path = Path(directory) / "openagent.db"
        real_ensure = schema._ensure_tool_call_context_repair

        async def _skip_repair(
            _conn: aiosqlite.Connection,
            *,
            app_version: str,
        ) -> bool:
            del app_version
            return False

        schema._ensure_tool_call_context_repair = _skip_repair
        db = MemoryDB(str(path))
        try:
            await db.connect()
        finally:
            schema._ensure_tool_call_context_repair = real_ensure
        conn = db._conn
        assert conn is not None
        try:
            await conn.execute(
                "INSERT INTO sessions "
                "(session_id, session_type, agent_id, user_id, metadata, runs, "
                "created_at, updated_at) "
                "VALUES ('repair-gap', 'agent', 'openagent', 'openagent', ?, "
                "'[]', 1700000000, 1700000100)",
                (json.dumps({"client_id": "alice", "title": "Repair gap"}),),
            )
            await conn.execute("DELETE FROM legacy_session_changes")
            await conn.execute(
                "UPDATE storage_migration_state SET checkpoint_updated_at=1700000100, "
                "checkpoint_session_id='repair-gap', failed_sessions=1 "
                "WHERE singleton_id=1"
            )
            await conn.commit()

            real_verify = schema._verify_tool_call_context_repair
            failed_once = False

            async def _fail_after_committed_ddl(
                raw_conn: aiosqlite.Connection,
            ) -> None:
                nonlocal failed_once
                await real_verify(raw_conn)
                if not failed_once:
                    failed_once = True
                    raise RuntimeError("synthetic crash after repair DDL commit")

            schema._verify_tool_call_context_repair = _fail_after_committed_ddl
            try:
                try:
                    await schema._ensure_tool_call_context_repair(
                        conn,
                        app_version="test-beta",
                    )
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("post-DDL repair crash was not injected")
            finally:
                schema._verify_tool_call_context_repair = real_verify

            failed_ledger = await (
                await conn.execute(
                    "SELECT status FROM schema_migrations WHERE migration_id=?",
                    (schema.TOOL_CALL_CONTEXT_REPAIR_MIGRATION_ID,),
                )
            ).fetchone()
            indexes_after_crash = {
                str(row[1])
                for row in await (
                    await conn.execute("PRAGMA index_list(tool_invocations)")
                ).fetchall()
            }
            retries_after_crash = await (
                await conn.execute(
                    "SELECT COUNT(*) FROM legacy_session_changes "
                    "WHERE session_id='repair-gap'"
                )
            ).fetchone()
            assert failed_ledger is not None and str(failed_ledger[0]) == "failed"
            assert "uq_tool_invocations_call_context" not in indexes_after_crash
            assert (
                "uq_tool_invocations_session_run_call_context"
                in indexes_after_crash
            )
            assert retries_after_crash is not None
            assert int(retries_after_crash[0]) == 1

            resumed = await schema._ensure_tool_call_context_repair(
                conn,
                app_version="test-beta",
            )
            complete_ledger = await (
                await conn.execute(
                    "SELECT status FROM schema_migrations WHERE migration_id=?",
                    (schema.TOOL_CALL_CONTEXT_REPAIR_MIGRATION_ID,),
                )
            ).fetchone()
            retries_after_resume = await (
                await conn.execute(
                    "SELECT COUNT(*) FROM legacy_session_changes "
                    "WHERE session_id='repair-gap'"
                )
            ).fetchone()
            assert resumed is True
            assert complete_ledger is not None
            assert str(complete_ledger[0]) == "complete"
            assert retries_after_resume is not None
            assert int(retries_after_resume[0]) == 1
        finally:
            await db.close()


@test("operational_storage", "legacy REAL timestamps are accepted by the strict change journal")
async def t_legacy_real_timestamp_journal(_ctx: TestContext) -> None:
    from src.memory.operational.schema import ensure_operational_storage

    with TemporaryDirectory(prefix="openagent-operational-real-time-") as directory:
        path = Path(directory) / "openagent.db"
        _seed_legacy(path)
        conn = await aiosqlite.connect(path)
        try:
            await ensure_operational_storage(conn, str(path), app_version="test-beta")
            await conn.execute(
                "UPDATE sessions SET updated_at=? WHERE session_id=?",
                (1_700_000_000.75, "legacy-canary"),
            )
            row = await (
                await conn.execute(
                    "SELECT legacy_updated_at, typeof(legacy_updated_at) "
                    "FROM legacy_session_changes WHERE session_id=? "
                    "ORDER BY seq DESC LIMIT 1",
                    ("legacy-canary",),
                )
            ).fetchone()
        finally:
            await conn.close()

        assert row is not None
        assert tuple(row) == (1_700_000_000, "integer")


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


@test("operational_storage", "empty and statusless legacy runs project without transcript loss")
async def t_legacy_empty_and_statusless_projection(_ctx: TestContext) -> None:
    import json

    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-operational-legacy-gaps-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        conn = db._conn
        assert conn is not None
        try:
            await db.upsert_session("empty-canary", client_id="alice")
            await conn.execute(
                "UPDATE sessions SET runs=NULL WHERE session_id='empty-canary'"
            )
            await db._project_operational_session("empty-canary")

            await db.upsert_session("statusless-canary", client_id="alice")
            await conn.execute(
                "UPDATE sessions SET runs=?, updated_at=? WHERE session_id=?",
                (
                    json.dumps([{
                        "run_id": "statusless-run",
                        "content": "durable final answer",
                        "messages": [
                            {"role": "user", "content": "durable question"},
                            {"role": "assistant", "content": "durable final answer"},
                        ],
                    }]),
                    1_700_000_002,
                    "statusless-canary",
                ),
            )
            await db._project_operational_session("statusless-canary")
            await conn.commit()

            empty = await (
                await conn.execute(
                    "SELECT completeness FROM sessions_v2 WHERE id='empty-canary'"
                )
            ).fetchone()
            empty_runs = await (
                await conn.execute(
                    "SELECT COUNT(*) FROM session_runs WHERE session_id='empty-canary'"
                )
            ).fetchone()
            statusless = await (
                await conn.execute(
                    "SELECT completeness FROM sessions_v2 "
                    "WHERE id='statusless-canary'"
                )
            ).fetchone()
            normalized_run = await (
                await conn.execute(
                    "SELECT status, status_raw, completeness, raw_envelope_json "
                    "FROM session_runs WHERE session_id='statusless-canary'"
                )
            ).fetchone()
            messages = await (
                await conn.execute(
                    "SELECT role, text FROM session_messages "
                    "WHERE session_id='statusless-canary' ORDER BY sequence"
                )
            ).fetchall()

            prefer = await db.set_operational_storage_phase("prefer_v2")
            assert prefer.verified_sessions == 2
            assert prefer.v2_eligible_sessions == 1
        finally:
            await db.close()

        assert empty is not None and str(empty[0]) == "complete"
        assert empty_runs is not None and int(empty_runs[0]) == 0
        assert statusless is not None and str(statusless[0]) == "partial"
        assert normalized_run is not None
        assert tuple(normalized_run[:3]) == (
            "success",
            "legacy_missing_inferred_success",
            "partial",
        )
        raw_envelope = json.loads(str(normalized_run[3]))
        assert "status" not in raw_envelope
        assert [tuple(row) for row in messages] == [
            ("user", "durable question"),
            ("assistant", "durable final answer"),
        ]


@test("operational_storage", "tool_call_id reuse is isolated by run and search message")
async def t_tool_call_id_reuse_across_runs(_ctx: TestContext) -> None:
    import json

    from src.memory.db import MemoryDB
    from src.memory.operational.search import _source_row

    with TemporaryDirectory(prefix="openagent-operational-tool-context-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        conn = db._conn
        assert conn is not None
        try:
            await db.upsert_session(
                "tool-context-canary",
                client_id="alice",
                title="Tool context canary",
            )
            await conn.execute(
                "UPDATE sessions SET runs=?, updated_at=? WHERE session_id=?",
                (
                    json.dumps(_reused_tool_call_runs()),
                    1_700_000_200,
                    "tool-context-canary",
                ),
            )
            await db._project_operational_session("tool-context-canary")
            await conn.commit()

            tools = await (
                await conn.execute(
                    "SELECT id, session_run_id, tool_call_id, result_text "
                    "FROM tool_invocations WHERE session_id=? "
                    "ORDER BY session_run_id",
                    ("tool-context-canary",),
                )
            ).fetchall()
            assert len(tools) == 2
            assert len({str(row[0]) for row in tools}) == 2
            assert [str(row[1]) for row in tools] == [
                "run:tool-context-canary:run-a",
                "run:tool-context-canary:run-b",
            ]
            assert [str(row[2]) for row in tools] == [
                "reused-call",
                "reused-call",
            ]
            assert [str(row[3]) for row in tools] == [
                "first distinct result",
                "second distinct result",
            ]

            message_ids: list[str] = []
            for row in tools:
                source = await _source_row(conn, "tool_invocation", str(row[0]))
                assert source is not None
                message_ids.append(str(source[0]["message_id"]))
            assert message_ids == [
                "msg:tool-context-canary:tool-message-a",
                "msg:tool-context-canary:tool-message-b",
            ]
        finally:
            await db.close()


@test("operational_storage", "same-run duplicate tool_call_id is rejected atomically")
async def t_same_run_duplicate_tool_call_id(_ctx: TestContext) -> None:
    import json

    from src.memory.db import MemoryDB
    from src.memory.operational.repository import project_legacy_session_async

    with TemporaryDirectory(prefix="openagent-operational-tool-duplicate-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        conn = db._conn
        assert conn is not None
        try:
            await db.upsert_session(
                "same-run-duplicate",
                client_id="alice",
                title="Same run duplicate",
            )
            run = _reused_tool_call_runs()[0]
            assert isinstance(run["tools"], list)
            run["tools"] = [run["tools"][0], dict(run["tools"][0])]
            await conn.execute(
                "UPDATE sessions SET runs=?, updated_at=? WHERE session_id=?",
                (json.dumps([run]), 1_700_000_200, "same-run-duplicate"),
            )

            await conn.execute("SAVEPOINT same_run_duplicate")
            try:
                await project_legacy_session_async(conn, "same-run-duplicate")
            except sqlite3.IntegrityError:
                await conn.execute("ROLLBACK TO SAVEPOINT same_run_duplicate")
                await conn.execute("RELEASE SAVEPOINT same_run_duplicate")
            else:
                raise AssertionError("same-run duplicate tool_call_id was accepted")

            projected_tools = await (
                await conn.execute(
                    "SELECT COUNT(*) FROM tool_invocations WHERE session_id=?",
                    ("same-run-duplicate",),
                )
            ).fetchone()
            pending = await (
                await conn.execute(
                    "SELECT COUNT(*) FROM legacy_session_changes "
                    "WHERE session_id=? AND processed_at_ms IS NULL",
                    ("same-run-duplicate",),
                )
            ).fetchone()
            assert projected_tools is not None and int(projected_tools[0]) == 0
            assert pending is not None and int(pending[0]) == 1
            await conn.commit()
        finally:
            await db.close()


@test("operational_storage", "failed backfill retries durably without starving new work")
async def t_backfill_retry_is_durable_and_fair(_ctx: TestContext) -> None:
    import json

    import src.memory.operational.repository as repository
    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-operational-retry-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        conn = db._conn
        assert conn is not None
        try:
            for session_id, updated_at in (
                ("retry-a", 1_700_000_100),
                ("retry-b", 1_700_000_200),
            ):
                await conn.execute(
                    "INSERT INTO sessions "
                    "(session_id, session_type, agent_id, user_id, metadata, "
                    "runs, created_at, updated_at) "
                    "VALUES (?, 'agent', 'openagent', 'openagent', ?, '[]', ?, ?)",
                    (
                        session_id,
                        json.dumps({"client_id": "alice", "title": session_id}),
                        updated_at - 1,
                        updated_at,
                    ),
                )
            await conn.execute("DELETE FROM legacy_session_changes")
            await conn.commit()

            real_project = repository.project_legacy_session

            def _fail_one(raw_conn, session_id: str, *, now_ms=None):
                if session_id == "retry-a":
                    raise RuntimeError("synthetic transient projection failure")
                return real_project(raw_conn, session_id, now_ms=now_ms)

            repository.project_legacy_session = _fail_one
            try:
                writes, complete = await repository.backfill_batch_async(
                    conn,
                    limit=100,
                )
                await conn.commit()
            finally:
                repository.project_legacy_session = real_project

            before = await repository.projection_coverage_async(conn)
            state_before = await (
                await conn.execute(
                    "SELECT failed_sessions FROM storage_migration_state "
                    "WHERE singleton_id=1"
                )
            ).fetchone()
            retry = await (
                await conn.execute(
                    "SELECT attempt_count, last_error_class "
                    "FROM legacy_session_changes WHERE session_id='retry-a' "
                    "AND processed_at_ms IS NULL"
                )
            ).fetchone()
            assert complete is True
            assert [write.session_id for write in writes] == ["retry-b"]
            assert before == {
                "legacy_sessions": 2,
                "projected_sessions": 1,
                "failed_sessions": 1,
                "pending_sessions": 1,
                "complete": False,
            }
            assert state_before is not None and int(state_before[0]) == 1
            assert retry is not None
            assert tuple(retry) == (1, "RuntimeError")

            # A new attempt-0 journal row must run before the older failed
            # attempt-1 row, so a poison session cannot monopolize a batch.
            await conn.execute(
                "INSERT INTO sessions "
                "(session_id, session_type, agent_id, user_id, metadata, runs, "
                "created_at, updated_at) "
                "VALUES ('retry-c', 'agent', 'openagent', 'openagent', ?, '[]', ?, ?)",
                (
                    json.dumps({"client_id": "alice", "title": "retry-c"}),
                    1_700_000_299,
                    1_700_000_300,
                ),
            )
            await conn.commit()
            first_retry = await repository.reconcile_pending_async(conn, limit=1)
            await conn.commit()
            assert [write.session_id for write in first_retry] == ["retry-c"]
            still_failed = await (
                await conn.execute(
                    "SELECT 1 FROM legacy_session_changes "
                    "WHERE session_id='retry-a' AND processed_at_ms IS NULL"
                )
            ).fetchone()
            assert still_failed is not None

            recovered = await repository.reconcile_pending_async(conn, limit=10)
            await conn.commit()
            assert [write.session_id for write in recovered] == ["retry-a"]
            after = await repository.projection_coverage_async(conn)
            state_after = await (
                await conn.execute(
                    "SELECT failed_sessions FROM storage_migration_state "
                    "WHERE singleton_id=1"
                )
            ).fetchone()
            assert after == {
                "legacy_sessions": 3,
                "projected_sessions": 3,
                "failed_sessions": 0,
                "pending_sessions": 0,
                "complete": True,
            }
            assert state_after is not None and int(state_after[0]) == 0
        finally:
            await db.close()


@test("operational_storage", "beta3 ledger repair retries missing backfill and stays downgrade-readable")
async def t_beta3_tool_context_repair(_ctx: TestContext) -> None:
    import json

    import src.memory.operational.schema as schema
    from src.memory.db import MemoryDB
    from src.memory.operational.repository import (
        project_legacy_session_async,
        projection_coverage_async,
    )

    with TemporaryDirectory(prefix="openagent-operational-beta3-repair-") as directory:
        path = Path(directory) / "openagent.db"
        real_repair = schema._ensure_tool_call_context_repair

        async def _old_beta3_no_repair(
            _conn: aiosqlite.Connection,
            *,
            app_version: str,
        ) -> bool:
            del app_version
            return False

        # Build the exact already-ledgerized beta3 shape: v2 is complete and
        # still owns the old session-global unique index.
        schema._ensure_tool_call_context_repair = _old_beta3_no_repair
        beta3 = MemoryDB(str(path))
        try:
            await beta3.connect()
        finally:
            schema._ensure_tool_call_context_repair = real_repair
        conn = beta3._conn
        assert conn is not None
        legacy_runs = json.dumps(_reused_tool_call_runs(), separators=(",", ":"))
        try:
            await conn.executemany(
                "INSERT INTO sessions "
                "(session_id, session_type, agent_id, user_id, metadata, runs, "
                "created_at, updated_at) VALUES (?, 'agent', 'openagent', "
                "'openagent', ?, ?, ?, ?)",
                (
                    (
                        "beta3-missing",
                        json.dumps(
                            {"client_id": "alice", "title": "Beta3 missing"}
                        ),
                        legacy_runs,
                        1_700_000_000,
                        1_700_000_200,
                    ),
                    (
                        "beta3-pending",
                        json.dumps(
                            {"client_id": "alice", "title": "Beta3 pending"}
                        ),
                        "[]",
                        1_700_000_000,
                        1_700_000_300,
                    ),
                ),
            )
            await conn.execute("DELETE FROM legacy_session_changes")

            # Prove that the already-shipped index rejects this valid source,
            # then construct beta3's persisted aggregate-only failure state.
            await conn.execute("SAVEPOINT beta3_old_projection")
            try:
                await project_legacy_session_async(conn, "beta3-missing")
            except sqlite3.IntegrityError:
                await conn.execute("ROLLBACK TO SAVEPOINT beta3_old_projection")
                await conn.execute("RELEASE SAVEPOINT beta3_old_projection")
            else:
                raise AssertionError("old beta3 index accepted cross-run call reuse")
            await conn.execute(
                "INSERT INTO legacy_session_changes "
                "(session_id, operation, legacy_updated_at) "
                "VALUES ('beta3-pending', 'update', 1700000300)"
            )
            await conn.execute(
                "UPDATE storage_migration_state SET checkpoint_updated_at=?, "
                "checkpoint_session_id=?, failed_sessions=2 WHERE singleton_id=1",
                (1_700_000_300, "beta3-pending"),
            )
            await conn.commit()
            before = await projection_coverage_async(conn)
            state_before = await (
                await conn.execute(
                    "SELECT failed_sessions FROM storage_migration_state "
                    "WHERE singleton_id=1"
                )
            ).fetchone()
            base_ledger = await (
                await conn.execute(
                    "SELECT checksum, status FROM schema_migrations "
                    "WHERE migration_id=?",
                    (schema.MIGRATION_ID,),
                )
            ).fetchone()
            old_index = await (
                await conn.execute(
                    "SELECT 1 FROM sqlite_schema WHERE type='index' AND name=?",
                    ("uq_tool_invocations_call_context",),
                )
            ).fetchone()
            assert before == {
                "legacy_sessions": 2,
                "projected_sessions": 0,
                "failed_sessions": 0,
                "pending_sessions": 1,
                "complete": False,
            }
            assert state_before is not None and int(state_before[0]) == 2
            assert base_ledger is not None
            assert tuple(base_ledger) == (schema.operational_schema_checksum(), "complete")
            assert schema.operational_schema_checksum() == (
                "ce406057aec3d3b0076e3750045ae24ab50f7829396809ae8f72defdd2111863"
            )
            assert old_index is not None
        finally:
            await beta3.close()

        # A current binary applies only the separate repair ledger, queues the
        # missing source and lets normal startup reconciliation finish it.
        repaired = MemoryDB(str(path))
        await repaired.connect()
        conn = repaired._conn
        assert conn is not None
        try:
            after = await projection_coverage_async(conn)
            base_ledger = await (
                await conn.execute(
                    "SELECT checksum, status FROM schema_migrations "
                    "WHERE migration_id=?",
                    (schema.MIGRATION_ID,),
                )
            ).fetchone()
            repair_ledger = await (
                await conn.execute(
                    "SELECT checksum, status FROM schema_migrations "
                    "WHERE migration_id=?",
                    (schema.TOOL_CALL_CONTEXT_REPAIR_MIGRATION_ID,),
                )
            ).fetchone()
            tools = await (
                await conn.execute(
                    "SELECT session_run_id, tool_call_id FROM tool_invocations "
                    "WHERE session_id=? ORDER BY session_run_id",
                    ("beta3-missing",),
                )
            ).fetchall()
            preserved = await (
                await conn.execute(
                    "SELECT runs FROM sessions WHERE session_id=?",
                    ("beta3-missing",),
                )
            ).fetchone()
            state_after = await (
                await conn.execute(
                    "SELECT failed_sessions FROM storage_migration_state "
                    "WHERE singleton_id=1"
                )
            ).fetchone()
            pending_journal_rows = await (
                await conn.execute(
                    "SELECT session_id, COUNT(*) FROM legacy_session_changes "
                    "GROUP BY session_id ORDER BY session_id"
                )
            ).fetchall()
            assert after == {
                "legacy_sessions": 2,
                "projected_sessions": 2,
                "failed_sessions": 0,
                "pending_sessions": 0,
                "complete": True,
            }
            assert state_after is not None and int(state_after[0]) == 0
            assert base_ledger is not None
            assert tuple(base_ledger) == (schema.operational_schema_checksum(), "complete")
            assert repair_ledger is not None
            assert tuple(repair_ledger) == (
                schema.tool_call_context_repair_checksum(),
                "complete",
            )
            assert [tuple(row) for row in tools] == [
                ("run:beta3-missing:run-a", "reused-call"),
                ("run:beta3-missing:run-b", "reused-call"),
            ]
            assert preserved is not None and str(preserved[0]) == legacy_runs
            assert [tuple(row) for row in pending_journal_rows] == [
                ("beta3-missing", 1),
                ("beta3-pending", 1),
            ]

            # Re-entry is ledger-idempotent and does not fabricate retry rows.
            repeated = await schema._ensure_tool_call_context_repair(
                conn,
                app_version="test-beta",
            )
            pending = await (
                await conn.execute(
                    "SELECT COUNT(*) FROM legacy_session_changes "
                    "WHERE processed_at_ms IS NULL"
                )
            ).fetchone()
            assert repeated is False
            assert pending is not None and int(pending[0]) == 0
        finally:
            await repaired.close()

        # Simulate the previous beta's bootstrap after rollback. It knows only
        # the immutable v2 ledger; the additive indexes and projected rows do
        # not prevent it from opening and reading the canonical legacy blob.
        schema._ensure_tool_call_context_repair = _old_beta3_no_repair
        downgraded = MemoryDB(str(path))
        try:
            await downgraded.connect()
            assert downgraded._conn is not None
            row = await (
                await downgraded._conn.execute(
                    "SELECT runs FROM sessions WHERE session_id=?",
                    ("beta3-missing",),
                )
            ).fetchone()
            assert row is not None and str(row[0]) == legacy_runs
        finally:
            schema._ensure_tool_call_context_repair = real_repair
            await downgraded.close()


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


@test("operational_storage", "append mutates only the normalized tail and preserves prefix rowids")
async def t_incremental_projection_append_cost(_ctx: TestContext) -> None:
    import json

    from src.memory.db import MemoryDB
    from src.memory.operational.repository import project_legacy_session_async

    def run(number: int) -> dict[str, object]:
        return {
            "run_id": f"run-{number}",
            "status": "COMPLETED",
            "created_at": 1_700_000_000 + number,
            "content": f"assistant {number}",
            "messages": [
                {"id": f"user-{number}", "role": "user", "content": f"question {number}"},
                {"id": f"assistant-{number}", "role": "assistant", "content": f"assistant {number}"},
            ],
            "tools": [{
                "tool_call_id": f"call-{number}",
                "tool_name": "shell_execute",
                "tool_args": {"cmd": f"printf {number}"},
                "result": f"result {number}",
                "status": "completed",
            }],
        }

    with TemporaryDirectory(prefix="openagent-operational-incremental-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        conn = db._conn
        assert conn is not None
        try:
            await db.upsert_session(
                "incremental-canary", client_id="alice", title="Incremental"
            )
            initial_runs = [run(number) for number in range(40)]
            await conn.execute(
                "UPDATE sessions SET runs=?, updated_at=? WHERE session_id=?",
                (json.dumps(initial_runs), 1_700_000_040, "incremental-canary"),
            )
            initial = await project_legacy_session_async(conn, "incremental-canary")
            await conn.commit()
            assert initial.delta.runs_inserted == 40
            assert initial.delta.messages_inserted == 80
            assert initial.delta.tools_inserted == 40

            before: dict[str, list[tuple[object, ...]]] = {}
            for table in ("session_runs", "session_messages", "tool_invocations"):
                rows = await (
                    await conn.execute(
                        f"SELECT id, rowid, source_version FROM {table} "
                        "WHERE session_id=? ORDER BY id",
                        ("incremental-canary",),
                    )
                ).fetchall()
                before[table] = [tuple(row) for row in rows]

            await conn.execute(
                "UPDATE sessions SET runs=?, updated_at=? WHERE session_id=?",
                (json.dumps([*initial_runs, run(40)]), 1_700_000_041, "incremental-canary"),
            )
            appended = await project_legacy_session_async(conn, "incremental-canary")
            await conn.commit()

            assert appended.delta.runs_inserted == 1
            assert appended.delta.messages_inserted == 2
            assert appended.delta.tools_inserted == 1
            assert appended.delta.nested_writes == 4
            assert appended.delta.nested_deletes == 0
            assert appended.delta.runs_updated == 0
            assert appended.delta.messages_updated == 0
            assert appended.delta.tools_updated == 0

            for table in ("session_runs", "session_messages", "tool_invocations"):
                prefix_ids = [str(row[0]) for row in before[table]]
                placeholders = ",".join("?" for _ in prefix_ids)
                rows = await (
                    await conn.execute(
                        f"SELECT id, rowid, source_version FROM {table} "
                        f"WHERE id IN ({placeholders}) ORDER BY id",
                        tuple(prefix_ids),
                    )
                ).fetchall()
                assert [tuple(row) for row in rows] == before[table]
        finally:
            await db.close()


@test("operational_storage", "cumulative run append and streaming edit keep stable anchors")
async def t_incremental_cumulative_run(_ctx: TestContext) -> None:
    import json

    from src.memory.db import MemoryDB
    from src.memory.operational.projection import NESTED_LAYOUT_KEY
    from src.memory.operational.repository import project_legacy_session_async

    with TemporaryDirectory(prefix="openagent-operational-cumulative-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        conn = db._conn
        assert conn is not None
        try:
            await db.upsert_session("cumulative-canary", client_id="alice")
            run = {
                "run_id": "one-long-run",
                "status": "RUNNING",
                "created_at": 1_700_000_000,
                "messages": [{"role": "user", "content": "question"}],
            }
            await conn.execute(
                "UPDATE sessions SET runs=?, updated_at=? WHERE session_id=?",
                (json.dumps([run]), 1_700_000_001, "cumulative-canary"),
            )
            await project_legacy_session_async(conn, "cumulative-canary")
            await conn.commit()
            user_before = await (
                await conn.execute(
                    "SELECT id, rowid, source_version FROM session_messages "
                    "WHERE session_id='cumulative-canary' AND ordinal=0"
                )
            ).fetchone()
            raw_run = await (
                await conn.execute(
                    "SELECT raw_envelope_json FROM session_runs "
                    "WHERE session_id='cumulative-canary'"
                )
            ).fetchone()
            assert user_before is not None and raw_run is not None
            bounded = json.loads(str(raw_run[0]))
            assert "messages" not in bounded
            assert bounded[NESTED_LAYOUT_KEY]["messages"]["passthrough"] == {}

            run["messages"].append(
                {"role": "assistant", "content": "partial answer"}
            )
            await conn.execute(
                "UPDATE sessions SET runs=?, updated_at=? WHERE session_id=?",
                (json.dumps([run]), 1_700_000_002, "cumulative-canary"),
            )
            appended = await project_legacy_session_async(
                conn, "cumulative-canary"
            )
            await conn.commit()
            assert appended.delta.messages_inserted == 1
            assert appended.delta.messages_updated == 0
            assert appended.delta.runs_updated == 0
            user_after = await (
                await conn.execute(
                    "SELECT id, rowid, source_version FROM session_messages "
                    "WHERE session_id='cumulative-canary' AND ordinal=0"
                )
            ).fetchone()
            assistant_before = await (
                await conn.execute(
                    "SELECT id, rowid FROM session_messages "
                    "WHERE session_id='cumulative-canary' AND ordinal=1"
                )
            ).fetchone()
            assert tuple(user_after) == tuple(user_before)
            assert assistant_before is not None

            run["status"] = "COMPLETED"
            run["messages"][1]["content"] = "final answer"
            await conn.execute(
                "UPDATE sessions SET runs=?, updated_at=? WHERE session_id=?",
                (json.dumps([run]), 1_700_000_003, "cumulative-canary"),
            )
            completed = await project_legacy_session_async(
                conn, "cumulative-canary"
            )
            await conn.commit()
            assistant_after = await (
                await conn.execute(
                    "SELECT id, rowid, text FROM session_messages "
                    "WHERE session_id='cumulative-canary' AND ordinal=1"
                )
            ).fetchone()
            assert completed.delta.messages_inserted == 0
            assert completed.delta.messages_updated == 1
            assert completed.delta.messages_deleted == 0
            assert assistant_after is not None
            assert tuple(assistant_after[:2]) == tuple(assistant_before)
            assert str(assistant_after[2]) == "final answer"
        finally:
            await db.close()


@test("operational_storage", "promotion relinks sessions and tools projected before targets")
async def t_projection_dependency_relink(_ctx: TestContext) -> None:
    import json

    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-operational-relink-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        conn = db._conn
        assert conn is not None
        try:
            await db.upsert_session(
                "child-first",
                client_id="alice",
                parent_session_id="late-parent",
                origin="delegation",
            )
            await conn.execute(
                "UPDATE sessions SET runs=?, updated_at=? "
                "WHERE session_id='child-first'",
                (
                    json.dumps([{
                        "run_id": "dependency-run",
                        "status": "COMPLETED",
                        "tools": [{
                            "tool_call_id": "dependency-call",
                            "tool_name": "delegate_task",
                            "status": "completed",
                            "result": "synthetic result",
                            "child_session_id": "late-tool-child",
                        }],
                    }]),
                    1_700_000_200,
                ),
            )
            await db._project_operational_session("child-first")
            await conn.commit()
            before = await (
                await conn.execute(
                    "SELECT s.parent_session_id, s.root_session_id, "
                    "t.child_session_id FROM sessions_v2 s "
                    "JOIN tool_invocations t ON t.session_id=s.id "
                    "WHERE s.id='child-first'"
                )
            ).fetchone()
            assert before is not None and before[0] is None
            assert str(before[1]) == "child-first"
            assert before[2] is None

            await db.upsert_session("late-parent", client_id="alice")
            await db.upsert_session("late-tool-child", client_id="alice")
            promoted = await db.set_operational_storage_phase("prefer_v2")
            after = await (
                await conn.execute(
                    "SELECT s.parent_session_id, s.root_session_id, "
                    "t.child_session_id FROM sessions_v2 s "
                    "JOIN tool_invocations t ON t.session_id=s.id "
                    "WHERE s.id='child-first'"
                )
            ).fetchone()
            assert after is not None and tuple(after) == (
                "late-parent",
                "late-parent",
                "late-tool-child",
            )
            assert promoted.verified_sessions == 3
        finally:
            await db.close()


@test("operational_storage", "guarded phases verify v2 reads, fall back, recover and roll back")
async def t_guarded_phase_cutover(_ctx: TestContext) -> None:
    import json

    from src.memory.db import MemoryDB
    from src.memory.operational.phase import (
        guard_storage_phase_async,
        preferred_session_source_async,
    )
    from src.memory.operational.repository import reconcile_pending_async

    with TemporaryDirectory(prefix="openagent-operational-phase-") as directory:
        db = MemoryDB(str(Path(directory) / "openagent.db"))
        await db.connect()
        conn = db._conn
        assert conn is not None
        try:
            await db.upsert_session(
                "phase-canary", client_id="alice", title="Phase canary"
            )
            runs = [{
                "run_id": "phase-run-1",
                "status": "COMPLETED",
                "created_at": 1_700_000_000,
                "messages": [
                    {"id": "phase-user-1", "role": "user", "content": "one"},
                    {"id": "phase-agent-1", "role": "assistant", "content": "two"},
                ],
                "content": "two",
            }]
            await conn.execute(
                "UPDATE sessions SET runs=?, updated_at=? WHERE session_id=?",
                (json.dumps(runs), 1_700_000_001, "phase-canary"),
            )
            await db._project_operational_session("phase-canary")
            await conn.commit()

            prefer = await db.set_operational_storage_phase("prefer_v2")
            assert (prefer.from_phase, prefer.to_phase) == ("shadow", "prefer_v2")
            assert prefer.verified_sessions == prefer.v2_eligible_sessions == 1
            assert await preferred_session_source_async(conn, "phase-canary") == "v2"

            promoted = await db.set_operational_storage_phase("v2")
            assert (promoted.from_phase, promoted.to_phase) == ("prefer_v2", "v2")

            # A direct/pre-v2 compatibility write immediately invalidates this
            # session and the global guard rolls back before reconciliation.
            changed_runs = [*runs, {
                "run_id": "phase-run-2",
                "status": "COMPLETED",
                "created_at": 1_700_000_010,
                "messages": [
                    {"id": "phase-user-2", "role": "user", "content": "three"},
                    {"id": "phase-agent-2", "role": "assistant", "content": "four"},
                ],
                "content": "four",
            }]
            await conn.execute(
                "UPDATE sessions SET runs=?, updated_at=? WHERE session_id=?",
                (json.dumps(changed_runs), 1_700_000_011, "phase-canary"),
            )
            assert await preferred_session_source_async(conn, "phase-canary") == "legacy"
            guarded = await guard_storage_phase_async(conn, writer_version="test-beta")
            await conn.commit()
            assert (guarded.from_phase, guarded.to_phase, guarded.reason) == (
                "v2", "shadow", "pending_legacy_change"
            )

            writes = await reconcile_pending_async(conn, limit=10)
            await conn.commit()
            assert [write.session_id for write in writes] == ["phase-canary"]
            await db.set_operational_storage_phase("prefer_v2")
            await db.set_operational_storage_phase("v2")
            rolled_back = await db.set_operational_storage_phase("shadow")
            assert (rolled_back.from_phase, rolled_back.to_phase) == ("v2", "shadow")
            legacy = await (
                await conn.execute(
                    "SELECT runs FROM sessions WHERE session_id='phase-canary'"
                )
            ).fetchone()
            normalized = await (
                await conn.execute(
                    "SELECT COUNT(*) FROM session_runs WHERE session_id='phase-canary'"
                )
            ).fetchone()
            assert legacy is not None and json.loads(str(legacy[0])) == changed_runs
            assert normalized is not None and int(normalized[0]) == 2
        finally:
            await db.close()


@test("operational_storage", "runtime prefer_v2 rehydrates full runs without selecting legacy blob")
async def t_runtime_v2_read_fidelity(_ctx: TestContext) -> None:
    from sqlalchemy import event

    from src.core._run_state.agent import RunOutput
    from src.core._run_state.base import RunStatus
    from src.memory.db import MemoryDB
    from src.memory.sessions import AgentSession
    from src.memory.store.base import SessionType
    from src.memory.store.sqlite import SqliteDb
    from src.models.providers.message import Message

    with TemporaryDirectory(prefix="openagent-operational-v2-read-") as directory:
        path = Path(directory) / "openagent.db"
        db = MemoryDB(str(path))
        await db.connect()
        now = int(time.time())
        runtime = SqliteDb(db_file=str(path), session_table="sessions")
        session = AgentSession(
            session_id="v2-read-canary",
            agent_id="agent-canary",
            user_id="openagent",
            metadata={"client_id": "alice", "title": "V2 read"},
            created_at=now,
            updated_at=now,
        )
        session.upsert_run(
            RunOutput(
                run_id="v2-read-run",
                agent_id="agent-canary",
                session_id="v2-read-canary",
                user_id="openagent",
                content="full fidelity answer",
                messages=[
                    Message(role="user", content="full fidelity question"),
                    Message(role="assistant", content="full fidelity answer"),
                ],
                status=RunStatus.completed,
                created_at=now,
            )
        )
        try:
            assert runtime.upsert_session(session) is not None
        finally:
            runtime.close()

        try:
            await db.set_operational_storage_phase("prefer_v2")
            runtime = SqliteDb(db_file=str(path), session_table="sessions")
            statements: list[str] = []

            @event.listens_for(runtime.db_engine, "before_cursor_execute")
            def _capture(_conn, _cursor, statement, _parameters, _context, _many):
                statements.append(str(statement))

            try:
                loaded = runtime.get_session(
                    "v2-read-canary",
                    SessionType.AGENT,
                    user_id="openagent",
                )
            finally:
                runtime.close()
            assert isinstance(loaded, AgentSession)
            assert loaded.runs is not None and len(loaded.runs) == 1
            assert loaded.runs[0].content == "full fidelity answer"
            normalized = [" ".join(statement.lower().split()) for statement in statements]
            assert not any(
                " sessions.runs" in statement and " from sessions" in statement
                for statement in normalized
            ), normalized
        finally:
            await db.close()


@test("operational_storage", "prefer_v2 is per-session and v2 rejects partial or corrupted projections")
async def t_phase_readiness_and_corruption_recovery(_ctx: TestContext) -> None:
    import json

    from src.memory.db import MemoryDB
    from src.memory.operational.phase import StoragePhaseError

    with TemporaryDirectory(prefix="openagent-operational-phase-gates-") as directory:
        path = Path(directory) / "openagent.db"
        db = MemoryDB(str(path))
        await db.connect()
        conn = db._conn
        assert conn is not None
        try:
            await db.upsert_session("partial-canary", client_id="alice")
            await conn.execute(
                "UPDATE sessions SET runs=?, updated_at=? WHERE session_id=?",
                (
                    json.dumps([{"run_id": "bad", "status": "NOT_A_STATUS"}]),
                    1_700_000_001,
                    "partial-canary",
                ),
            )
            await db._project_operational_session("partial-canary")
            await conn.commit()
            prefer = await db.set_operational_storage_phase("prefer_v2")
            assert prefer.verified_sessions == 1
            assert prefer.v2_eligible_sessions == 0
            try:
                await db.set_operational_storage_phase("v2")
            except StoragePhaseError:
                pass
            else:
                raise AssertionError("v2 accepted a malformed legacy session")
            phase = await (
                await conn.execute(
                    "SELECT phase FROM storage_migration_state WHERE singleton_id=1"
                )
            ).fetchone()
            assert phase is not None and str(phase[0]) == "prefer_v2"
            await db.set_operational_storage_phase("shadow")

            await db.delete_session("partial-canary")
            await db.upsert_session("corrupt-canary", client_id="alice")
            await conn.execute(
                "UPDATE sessions SET runs=?, updated_at=? WHERE session_id=?",
                (
                    json.dumps([{
                        "run_id": "good",
                        "status": "COMPLETED",
                        "messages": [{
                            "id": "good-message",
                            "role": "user",
                            "content": "canonical text",
                        }],
                    }]),
                    1_700_000_010,
                    "corrupt-canary",
                ),
            )
            await db._project_operational_session("corrupt-canary")
            await conn.commit()
            await db.set_operational_storage_phase("prefer_v2")
            await conn.execute(
                "UPDATE session_messages SET text='corrupted text' "
                "WHERE id='msg:corrupt-canary:good-message'"
            )
            await conn.commit()
            try:
                await db.set_operational_storage_phase("v2")
            except StoragePhaseError:
                pass
            else:
                raise AssertionError("v2 accepted a corrupted normalized row")
            # Model an old binary/table replacement that lost one downgrade
            # trigger. A clean boot trusts the durable journal and stays O(1)
            # in transcript size; missing trigger evidence forces the exact
            # audit which detects and repairs this mismatch.
            await conn.execute(
                "DROP TRIGGER trg_legacy_sessions_update_journal"
            )
            await conn.commit()
        finally:
            await db.close()

        # Promoted boot audit demotes, queues the exact mismatch, and startup
        # reconciliation repairs it from the still-canonical legacy source.
        recovered = MemoryDB(str(path))
        await recovered.connect()
        conn = recovered._conn
        assert conn is not None
        try:
            phase = await (
                await conn.execute(
                    "SELECT phase FROM storage_migration_state WHERE singleton_id=1"
                )
            ).fetchone()
            message = await (
                await conn.execute(
                    "SELECT text FROM session_messages "
                    "WHERE id='msg:corrupt-canary:good-message'"
                )
            ).fetchone()
            pending = await (
                await conn.execute(
                    "SELECT COUNT(*) FROM legacy_session_changes "
                    "WHERE processed_at_ms IS NULL"
                )
            ).fetchone()
            assert phase is not None and str(phase[0]) == "shadow"
            assert message is not None and str(message[0]) == "canonical text"
            assert pending is not None and int(pending[0]) == 0
        finally:
            await recovered.close()


@test("operational_storage", "lost legacy triggers repair missing and extra v2 ids")
async def t_lost_trigger_set_drift_recovery(_ctx: TestContext) -> None:
    import json

    from src.memory.db import MemoryDB

    with TemporaryDirectory(prefix="openagent-operational-set-drift-") as directory:
        path = Path(directory) / "openagent.db"
        db = MemoryDB(str(path))
        await db.connect()
        conn = db._conn
        assert conn is not None
        try:
            for session_id in ("missing-v2-canary", "extra-v2-canary"):
                await db.upsert_session(session_id, client_id="alice")
                await conn.execute(
                    "UPDATE sessions SET runs=?, updated_at=? WHERE session_id=?",
                    (
                        json.dumps([{
                            "run_id": f"run-{session_id}",
                            "status": "COMPLETED",
                            "messages": [{
                                "id": f"message-{session_id}",
                                "role": "user",
                                "content": session_id,
                            }],
                        }]),
                        1_700_000_100,
                        session_id,
                    ),
                )
                await db._project_operational_session(session_id)
            await conn.commit()
            await db.set_operational_storage_phase("prefer_v2")

            # Simulate an old binary replacing the compatibility table: trigger
            # evidence disappears and neither set mutation reaches the journal.
            await conn.execute("DROP TRIGGER trg_legacy_sessions_update_journal")
            await conn.execute("DROP TRIGGER trg_legacy_sessions_delete_journal")
            await conn.execute(
                "UPDATE sessions_v2 SET status='deleted', "
                "deleted_at_ms=MAX(created_at_ms, updated_at_ms, last_activity_at_ms) "
                "WHERE id='missing-v2-canary'"
            )
            await conn.execute(
                "DELETE FROM sessions WHERE session_id='extra-v2-canary'"
            )
            await conn.commit()
        finally:
            await db.close()

        recovered = MemoryDB(str(path))
        await recovered.connect()
        conn = recovered._conn
        assert conn is not None
        try:
            phase = await (
                await conn.execute(
                    "SELECT phase FROM storage_migration_state WHERE singleton_id=1"
                )
            ).fetchone()
            missing = await (
                await conn.execute(
                    "SELECT deleted_at_ms FROM sessions_v2 "
                    "WHERE id='missing-v2-canary'"
                )
            ).fetchone()
            extra = await (
                await conn.execute(
                    "SELECT deleted_at_ms FROM sessions_v2 "
                    "WHERE id='extra-v2-canary'"
                )
            ).fetchone()
            pending = await (
                await conn.execute(
                    "SELECT COUNT(*) FROM legacy_session_changes "
                    "WHERE processed_at_ms IS NULL"
                )
            ).fetchone()
            assert phase is not None and str(phase[0]) == "shadow"
            assert missing is not None and missing[0] is None
            assert extra is not None and extra[0] is not None
            assert pending is not None and int(pending[0]) == 0
        finally:
            await recovered.close()


@test("operational_storage", "clean promoted boot trusts intact downgrade journal")
async def t_clean_promoted_boot_skips_transcript_audit(_ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.memory.operational import phase as phase_module

    with TemporaryDirectory(prefix="openagent-operational-clean-boot-") as directory:
        path = Path(directory) / "openagent.db"
        db = MemoryDB(str(path))
        await db.connect()
        try:
            await db.upsert_session("clean-boot-canary", client_id="alice")
            await db.set_operational_storage_phase("prefer_v2")
        finally:
            await db.close()

        original_verify = phase_module.verify_session_projection

        def unexpected_full_audit(*_args, **_kwargs):
            raise AssertionError("clean promoted boot reparsed transcript")

        phase_module.verify_session_projection = unexpected_full_audit
        reopened = MemoryDB(str(path))
        try:
            await reopened.connect()
            assert reopened._conn is not None
            row = await (
                await reopened._conn.execute(
                    "SELECT phase FROM storage_migration_state WHERE singleton_id=1"
                )
            ).fetchone()
            assert row is not None and str(row[0]) == "prefer_v2"
        finally:
            phase_module.verify_session_projection = original_verify
            await reopened.close()
