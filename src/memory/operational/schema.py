"""Rollback-safe bootstrap for additive operational-storage v2 tables.

The shipped legacy schema is still opened first.  Before the first v2 DDL this
module takes a SQLite-consistent backup, verifies it, applies the normative DDL,
installs the version-gated legacy change journal, and enters ``shadow``.  No
legacy column or row is removed, so a beta binary can return to legacy reads.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any
from uuid import uuid4

import aiosqlite


OPERATIONAL_SCHEMA_VERSION = 2
MIGRATION_ID = "operational-storage-v2"
MIGRATION_DESCRIPTION = "add normalized operational history and projections"
TOOL_CALL_CONTEXT_REPAIR_MIGRATION_ID = "operational-tool-call-context-v1"
TOOL_CALL_CONTEXT_REPAIR_DESCRIPTION = (
    "scope session tool-call uniqueness to one run and retry missing projections"
)
_MIN_SQLITE_VERSION = (3, 38, 0)
_LOCK_CONTENTION_ERRNOS = {
    errno.EACCES,
    errno.EAGAIN,
    getattr(errno, "EWOULDBLOCK", errno.EAGAIN),
    getattr(errno, "EDEADLK", errno.EAGAIN),
}


class OperationalMigrationError(RuntimeError):
    """The additive migration could not be proven safe."""


@dataclass(frozen=True)
class BackupEvidence:
    database_path: str
    manifest_path: str
    sha256: str
    size_bytes: int
    created_at_ms: int


def _resource_text(name: str) -> str:
    return files("src.memory.operational.sql").joinpath(name).read_text(encoding="utf-8")


def operational_schema_sql() -> str:
    return _resource_text("operational_storage_v2.sql")


def legacy_bridge_sql() -> str:
    return _resource_text("legacy_session_change_triggers.sql")


def automation_bridge_sql() -> str:
    return _resource_text("legacy_automation_change_triggers.sql")


def operational_search_schema_sql() -> str:
    return _resource_text("operational_search_v1.sql")


def tool_call_context_repair_sql() -> str:
    return _resource_text("operational_tool_call_context_v1.sql")


def operational_schema_checksum() -> str:
    return hashlib.sha256(operational_schema_sql().encode("utf-8")).hexdigest()


def tool_call_context_repair_checksum() -> str:
    return hashlib.sha256(
        tool_call_context_repair_sql().encode("utf-8")
    ).hexdigest()


class _MigrationFileLock:
    """Small cross-platform advisory lock beside the canonical database."""

    def __init__(self, db_path: Path) -> None:
        self.path = db_path.with_name(f"{db_path.name}.operational-v2.lock")
        self._fd: int | None = None

    def acquire(self, timeout_s: float = 60.0) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        deadline = time.monotonic() + max(0.1, timeout_s)
        while True:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    if os.fstat(fd).st_size == 0:
                        os.write(fd, b"0")
                        os.fsync(fd)
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno == errno.EINTR:
                    continue
                if exc.errno not in _LOCK_CONTENTION_ERRNOS:
                    os.close(fd)
                    raise OperationalMigrationError(
                        f"cannot acquire migration lock {self.path}: {exc}"
                    ) from exc
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise OperationalMigrationError(
                        f"timed out acquiring migration lock {self.path}"
                    ) from exc
                time.sleep(0.05)
                continue
            break

        # Failures after the OS lock is held (for example EBADF/EIO while
        # writing diagnostics) are not contention and must surface at once.
        self._fd = fd
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode("ascii"))
            os.fsync(fd)
        except OSError as exc:
            self.release()
            raise OperationalMigrationError(
                f"cannot initialize migration lock {self.path}: {exc}"
            ) from exc

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
            self._fd = None


async def _table_exists(conn: aiosqlite.Connection, table: str) -> bool:
    row = await (
        await conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?", (table,)
        )
    ).fetchone()
    return row is not None


async def _validate_runtime(conn: aiosqlite.Connection) -> None:
    row = await (await conn.execute("SELECT sqlite_version()" )).fetchone()
    raw = str(row[0] if row else "0.0.0")
    try:
        version = tuple(int(part) for part in raw.split(".")[:3])
    except ValueError as exc:
        raise OperationalMigrationError(f"unparseable SQLite version {raw!r}") from exc
    if version < _MIN_SQLITE_VERSION:
        raise OperationalMigrationError(
            f"operational storage requires SQLite >= 3.38; found {raw}"
        )
    row = await (await conn.execute("SELECT json_valid('{}')" )).fetchone()
    if not row or int(row[0]) != 1:
        raise OperationalMigrationError("SQLite JSON functions are unavailable")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes made by ``os.replace``.

    Windows does not expose a portable directory fsync.  SQLite's own backup
    remains durable there; POSIX platforms additionally make both renames
    crash-durable before migration proceeds.
    """

    if sys.platform == "win32":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _cleanup_paths(*paths: Path) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Preserve the original migration error. A later housekeeping
            # pass can remove a uniquely named orphan if the filesystem is
            # temporarily read-only.
            pass


async def _verified_backup(
    conn: aiosqlite.Connection,
    db_path: Path,
    *,
    app_version: str,
) -> BackupEvidence:
    """Create and rehearse a consistent pre-DDL SQLite backup."""

    await conn.commit()
    try:
        await conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
    except sqlite3.OperationalError:
        # A non-WAL fresh fixture is still safe to back up through the API.
        pass

    backup_dir = db_path.parent / "backups"
    backup_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(backup_dir, 0o700)
    except OSError:
        pass

    now_ms = int(time.time() * 1000)
    stem = f"{db_path.stem}-pre-operational-v2-{now_ms}-{uuid4().hex[:8]}"
    temp_path = backup_dir / f".{stem}.db.tmp"
    backup_path = backup_dir / f"{stem}.db"
    manifest_path = backup_dir / f"{stem}.manifest.json"
    manifest_tmp = backup_dir / f".{stem}.manifest.json.tmp"

    try:
        destination = sqlite3.connect(str(temp_path))
        try:
            await conn.backup(destination)
            destination.commit()
            destination.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            integrity = destination.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise OperationalMigrationError(
                    f"pre-migration backup integrity_check failed: {integrity!r}"
                )
        finally:
            destination.close()

        with temp_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, backup_path)
        _fsync_directory(backup_dir)

        # Restore rehearsal: copy through SQLite's backup API into memory and
        # run integrity_check there. Merely opening the snapshot is not proof
        # that it can be restored.
        source = sqlite3.connect(f"file:{backup_path}?mode=ro", uri=True)
        rehearsal = sqlite3.connect(":memory:")
        try:
            source.backup(rehearsal)
            restored = rehearsal.execute("PRAGMA integrity_check").fetchone()
            if not restored or restored[0] != "ok":
                raise OperationalMigrationError(
                    f"pre-migration restore rehearsal failed: {restored!r}"
                )
        finally:
            rehearsal.close()
            source.close()

        digest = await asyncio.to_thread(_sha256_file, backup_path)
        manifest: dict[str, Any] = {
            "kind": "openagent-operational-storage-pre-migration",
            "migration_id": MIGRATION_ID,
            "source_database": str(db_path.resolve()),
            "backup_database": str(backup_path.resolve()),
            "server_version": app_version,
            "schema_target": OPERATIONAL_SCHEMA_VERSION,
            "created_at_ms": now_ms,
            "size_bytes": backup_path.stat().st_size,
            "sha256": digest,
            "integrity_check": "ok",
            "restore_rehearsal": "ok",
        }
        fd = os.open(manifest_tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            payload = json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode()
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(manifest_tmp, manifest_path)
        _fsync_directory(backup_dir)

        return BackupEvidence(
            database_path=str(backup_path),
            manifest_path=str(manifest_path),
            sha256=digest,
            size_bytes=backup_path.stat().st_size,
            created_at_ms=now_ms,
        )
    except Exception:
        _cleanup_paths(temp_path, manifest_tmp, manifest_path, backup_path)
        try:
            _fsync_directory(backup_dir)
        except OSError:
            pass
        raise


async def _legacy_bridge_compatible(conn: aiosqlite.Connection) -> bool:
    if not await _table_exists(conn, "sessions"):
        return False
    rows = await (await conn.execute("PRAGMA table_info(sessions)" )).fetchall()
    columns = {str(row[1]) for row in rows}
    return {"session_id", "updated_at"}.issubset(columns)


async def _automation_bridge_compatible(conn: aiosqlite.Connection) -> bool:
    tables = {
        "workflow_tasks",
        "workflow_runs",
        "scheduled_tasks",
        "task_runs",
        "events",
        "event_deliveries",
    }
    rows = await (
        await conn.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' "
            f"AND name IN ({','.join('?' for _ in tables)})",
            tuple(sorted(tables)),
        )
    ).fetchall()
    return {str(row[0]) for row in rows} == tables


async def _journal_once(
    conn: aiosqlite.Connection,
    *,
    event_type: str,
    app_version: str,
    from_phase: str | None,
    to_phase: str | None,
    details: dict[str, Any] | None = None,
) -> None:
    """Append one lifecycle event per migration and event type.

    The normative journal is deliberately append-only.  Idempotence therefore
    belongs in the writer, not in an UPDATE or a mutable uniqueness marker.
    """

    now_ms = int(time.time() * 1000)
    await conn.execute(
        "INSERT INTO storage_migration_journal "
        "(migration_id, event_type, from_phase, to_phase, writer_version, "
        "details_json, occurred_at_ms) "
        "SELECT ?, ?, ?, ?, ?, ?, ? "
        "WHERE NOT EXISTS ("
        "SELECT 1 FROM storage_migration_journal "
        "WHERE migration_id=? AND event_type=?"
        ")",
        (
            MIGRATION_ID,
            event_type,
            from_phase,
            to_phase,
            app_version,
            json.dumps(details or {}, sort_keys=True, separators=(",", ":")),
            now_ms,
            MIGRATION_ID,
            event_type,
        ),
    )


async def _migration_row(
    conn: aiosqlite.Connection,
) -> tuple[str, str] | None:
    checksum = operational_schema_checksum()
    existing = await (
        await conn.execute(
            "SELECT checksum, status FROM schema_migrations WHERE migration_id=?",
            (MIGRATION_ID,),
        )
    ).fetchone()
    if existing is not None and str(existing[0]) != checksum:
        raise OperationalMigrationError(
            "operational-storage-v2 checksum differs from the migration ledger"
        )
    if existing is None:
        return None
    return str(existing[0]), str(existing[1])


async def _verify_installed_schema(conn: aiosqlite.Connection) -> None:
    required = {
        "schema_migrations",
        "storage_migration_state",
        "storage_migration_journal",
        "operational_storage_state",
        "sessions_v2",
        "session_runs",
        "session_messages",
        "tool_invocations",
        "domain_events",
        "activity_items",
        "operational_resource_owners",
        "operational_automation_projection",
        "operational_automation_changes",
        "search_outbox",
        "legacy_session_changes",
    }
    rows = await (
        await conn.execute(
            "SELECT name FROM sqlite_schema WHERE type='table' "
            f"AND name IN ({','.join('?' for _ in required)})",
            tuple(sorted(required)),
        )
    ).fetchall()
    present = {str(row[0]) for row in rows}
    missing = sorted(required - present)
    if missing:
        raise OperationalMigrationError(
            f"operational storage schema is incomplete: {', '.join(missing)}"
        )
    state = await (
        await conn.execute(
            "SELECT schema_version FROM operational_storage_state WHERE singleton_id=1"
        )
    ).fetchone()
    if state is None or int(state[0]) < OPERATIONAL_SCHEMA_VERSION:
        raise OperationalMigrationError("operational storage state is missing or stale")


async def _verify_tool_call_context_repair(
    conn: aiosqlite.Connection,
) -> None:
    tool_rows = await (
        await conn.execute("PRAGMA index_list(tool_invocations)")
    ).fetchall()
    tool_indexes = {
        str(row[1]): (int(row[2]), int(row[4]))
        for row in tool_rows
    }
    if "uq_tool_invocations_call_context" in tool_indexes:
        raise OperationalMigrationError(
            "obsolete session-global tool-call index is still installed"
        )
    message_rows = await (
        await conn.execute("PRAGMA index_list(session_messages)")
    ).fetchall()
    message_indexes = {
        str(row[1]): (int(row[2]), int(row[4]))
        for row in message_rows
    }
    expected = {
        "uq_tool_invocations_session_run_call_context": (
            tool_indexes,
            1,
            ("session_run_id", "tool_call_id"),
            "where root_kind = 'session' and session_run_id is not null "
            "and tool_call_id is not null",
        ),
        "uq_tool_invocations_session_root_call_context": (
            tool_indexes,
            1,
            ("root_kind", "root_id", "tool_call_id"),
            "where root_kind = 'session' and session_run_id is null "
            "and tool_call_id is not null",
        ),
        "uq_tool_invocations_non_session_call_context": (
            tool_indexes,
            1,
            ("root_kind", "root_id", "tool_call_id"),
            "where root_kind <> 'session' and tool_call_id is not null",
        ),
        "idx_session_messages_run_tool_call": (
            message_indexes,
            0,
            ("session_id", "run_id", "tool_call_id"),
            "where run_id is not null and tool_call_id is not null",
        ),
    }
    definitions = {
        str(row[0]): " ".join(str(row[1] or "").lower().split())
        for row in await (
            await conn.execute(
                "SELECT name, sql FROM sqlite_schema WHERE type='index' "
                f"AND name IN ({','.join('?' for _ in expected)})",
                tuple(expected),
            )
        ).fetchall()
    }
    for name, (indexes, expected_unique, expected_columns, predicate) in expected.items():
        if name not in indexes:
            raise OperationalMigrationError(
                f"tool-call context repair is missing index {name}"
            )
        unique, partial = indexes[name]
        if unique != expected_unique or partial != 1:
            raise OperationalMigrationError(
                f"tool-call context repair index {name} has invalid flags"
            )
        columns = await (
            await conn.execute(f"PRAGMA index_info({name})")
        ).fetchall()
        actual = tuple(str(row[2]) for row in columns)
        if actual != expected_columns:
            raise OperationalMigrationError(
                f"tool-call context repair index {name} has columns {actual!r}"
            )
        if predicate not in definitions.get(name, ""):
            raise OperationalMigrationError(
                f"tool-call context repair index {name} has an invalid predicate"
            )


async def _tool_call_context_repair_row(
    conn: aiosqlite.Connection,
) -> tuple[str, str] | None:
    checksum = tool_call_context_repair_checksum()
    existing = await (
        await conn.execute(
            "SELECT checksum, status FROM schema_migrations WHERE migration_id=?",
            (TOOL_CALL_CONTEXT_REPAIR_MIGRATION_ID,),
        )
    ).fetchone()
    if existing is not None and str(existing[0]) != checksum:
        raise OperationalMigrationError(
            "operational-tool-call-context-v1 checksum differs from the migration ledger"
        )
    if existing is None:
        return None
    return str(existing[0]), str(existing[1])


async def _mark_tool_call_context_repair_failed(
    conn: aiosqlite.Connection,
    exc: BaseException,
) -> None:
    try:
        existing = await (
            await conn.execute(
                "SELECT status FROM schema_migrations WHERE migration_id=?",
                (TOOL_CALL_CONTEXT_REPAIR_MIGRATION_ID,),
            )
        ).fetchone()
        if existing is None or str(existing[0]) == "complete":
            return
        now_ms = int(time.time() * 1000)
        await conn.execute(
            "UPDATE schema_migrations SET status='failed', completed_at_ms=?, "
            "error_class=?, updated_at_ms=? WHERE migration_id=?",
            (
                now_ms,
                type(exc).__name__[:200],
                now_ms,
                TOOL_CALL_CONTEXT_REPAIR_MIGRATION_ID,
            ),
        )
        await conn.commit()
    except Exception:
        await conn.rollback()


async def _ensure_tool_call_context_repair(
    conn: aiosqlite.Connection,
    *,
    app_version: str,
) -> bool:
    """Apply the post-v2 repair without mutating the v2 SQL checksum.

    Returns ``True`` only when this invocation completed or resumed the repair.
    The DDL and retry queue update own one SQLite transaction; a crash leaves a
    resumable ``running`` ledger row and never changes ``sessions.runs``.
    """

    existing = await _tool_call_context_repair_row(conn)
    if existing is not None and existing[1] == "complete":
        await _verify_tool_call_context_repair(conn)
        return False

    now_ms = int(time.time() * 1000)
    checksum = tool_call_context_repair_checksum()
    if existing is None:
        await conn.execute(
            "INSERT INTO schema_migrations "
            "(migration_id, checksum, description, status, started_at_ms, "
            "completed_at_ms, app_version, runner_id, error_class, "
            "created_at_ms, updated_at_ms) "
            "VALUES (?, ?, ?, 'running', ?, NULL, ?, ?, NULL, ?, ?)",
            (
                TOOL_CALL_CONTEXT_REPAIR_MIGRATION_ID,
                checksum,
                TOOL_CALL_CONTEXT_REPAIR_DESCRIPTION,
                now_ms,
                app_version,
                f"pid:{os.getpid()}",
                now_ms,
                now_ms,
            ),
        )
    else:
        await conn.execute(
            "UPDATE schema_migrations SET status='running', "
            "started_at_ms=COALESCE(started_at_ms, ?), completed_at_ms=NULL, "
            "app_version=?, runner_id=?, error_class=NULL, updated_at_ms=? "
            "WHERE migration_id=? AND status!='complete'",
            (
                now_ms,
                app_version,
                f"pid:{os.getpid()}",
                now_ms,
                TOOL_CALL_CONTEXT_REPAIR_MIGRATION_ID,
            ),
        )
    await conn.commit()

    try:
        await conn.executescript(tool_call_context_repair_sql())
        await _verify_tool_call_context_repair(conn)
        completed_at_ms = int(time.time() * 1000)
        await conn.execute(
            "UPDATE schema_migrations SET status='complete', completed_at_ms=?, "
            "error_class=NULL, updated_at_ms=? "
            "WHERE migration_id=? AND status!='complete'",
            (
                completed_at_ms,
                completed_at_ms,
                TOOL_CALL_CONTEXT_REPAIR_MIGRATION_ID,
            ),
        )
        await conn.commit()
        return True
    except Exception as exc:
        await conn.rollback()
        await _mark_tool_call_context_repair_failed(conn, exc)
        raise


async def _begin_or_resume_migration(
    conn: aiosqlite.Connection,
    *,
    app_version: str,
    backup: BackupEvidence | None,
    ddl_applied_this_run: bool,
) -> bool:
    """Persist a resumable running ledger after the DDL transaction.

    ``sqlite3.Connection.executescript`` commits any active transaction before
    it starts the script.  The normative DDL also owns ``BEGIN IMMEDIATE`` and
    ``COMMIT``.  Consequently a process can die after all tables exist but
    before this ledger write.  On restart we recognize the complete table set,
    do *not* mislabel a post-DDL backup as pre-DDL, and resume from ``running``.

    Returns ``True`` when the migration was already complete.
    """

    now_ms = int(time.time() * 1000)
    checksum = operational_schema_checksum()
    existing = await _migration_row(conn)
    if existing is not None and existing[1] == "complete":
        return True

    if existing is None:
        await conn.execute(
            "INSERT INTO schema_migrations "
            "(migration_id, checksum, description, status, started_at_ms, "
            "completed_at_ms, app_version, runner_id, error_class, "
            "created_at_ms, updated_at_ms) "
            "VALUES (?, ?, ?, 'running', ?, NULL, ?, ?, NULL, ?, ?)",
            (
                MIGRATION_ID,
                checksum,
                MIGRATION_DESCRIPTION,
                now_ms,
                app_version,
                f"pid:{os.getpid()}",
                now_ms,
                now_ms,
            ),
        )
    else:
        # pending/failed rows are explicitly recoverable.  Keep their original
        # start time for observability and clear only terminal failure fields.
        await conn.execute(
            "UPDATE schema_migrations SET status='running', "
            "started_at_ms=COALESCE(started_at_ms, ?), completed_at_ms=NULL, "
            "app_version=?, runner_id=?, error_class=NULL, updated_at_ms=? "
            "WHERE migration_id=? AND status!='complete'",
            (now_ms, app_version, f"pid:{os.getpid()}", now_ms, MIGRATION_ID),
        )

    phase_row = await (
        await conn.execute(
            "SELECT phase FROM storage_migration_state WHERE singleton_id=1"
        )
    ).fetchone()
    phase = str(phase_row[0]) if phase_row else "legacy"
    details = {
        "sqlite_executescript_implicit_commit": True,
        "ddl_applied_this_run": ddl_applied_this_run,
        "recovered_committed_ddl": not ddl_applied_this_run,
    }
    await _journal_once(
        conn,
        event_type="ddl_started",
        app_version=app_version,
        from_phase=phase,
        to_phase=phase,
        details=details,
    )
    if backup is not None:
        await _journal_once(
            conn,
            event_type="backup_verified",
            app_version=app_version,
            from_phase=phase,
            to_phase=phase,
            details={
                "backup_file": Path(backup.database_path).name,
                "manifest_file": Path(backup.manifest_path).name,
                "sha256": backup.sha256,
                "size_bytes": backup.size_bytes,
            },
        )
    await _journal_once(
        conn,
        event_type="ddl_completed",
        app_version=app_version,
        from_phase=phase,
        to_phase=phase,
        details=details,
    )
    await conn.commit()
    return False


async def _complete_migration(
    conn: aiosqlite.Connection,
    *,
    app_version: str,
) -> None:
    existing = await _migration_row(conn)
    if existing is None:
        raise OperationalMigrationError("operational migration ledger is missing")
    if existing[1] == "complete":
        phase_row = await (
            await conn.execute(
                "SELECT phase FROM storage_migration_state WHERE singleton_id=1"
            )
        ).fetchone()
        if phase_row is None or str(phase_row[0]) == "legacy":
            raise OperationalMigrationError(
                "completed operational migration has an invalid legacy phase"
            )
        return

    now_ms = int(time.time() * 1000)
    phase_row = await (
        await conn.execute(
            "SELECT phase FROM storage_migration_state WHERE singleton_id=1"
        )
    ).fetchone()
    from_phase = str(phase_row[0]) if phase_row else "legacy"
    to_phase = "shadow" if from_phase == "legacy" else from_phase
    if from_phase == "legacy":
        await conn.execute(
            "UPDATE storage_migration_state SET phase='shadow', "
            "state_version=state_version+1, last_writer_version=?, "
            "leader_id=NULL, leader_acquired_at_ms=NULL, updated_at_ms=? "
            "WHERE singleton_id=1",
            (app_version, now_ms),
        )
        await _journal_once(
            conn,
            event_type="phase_changed",
            app_version=app_version,
            from_phase="legacy",
            to_phase="shadow",
        )
    await conn.execute(
        "UPDATE schema_migrations SET status='complete', completed_at_ms=?, "
        "error_class=NULL, updated_at_ms=? "
        "WHERE migration_id=? AND status!='complete'",
        (now_ms, now_ms, MIGRATION_ID),
    )
    await conn.commit()


async def _mark_migration_failed(
    conn: aiosqlite.Connection,
    exc: BaseException,
) -> None:
    """Best-effort terminal marker for a resumable, already-created ledger."""

    try:
        if not await _table_exists(conn, "schema_migrations"):
            return
        existing = await (
            await conn.execute(
                "SELECT status FROM schema_migrations WHERE migration_id=?",
                (MIGRATION_ID,),
            )
        ).fetchone()
        if existing is None or str(existing[0]) == "complete":
            return
        now_ms = int(time.time() * 1000)
        await conn.execute(
            "UPDATE schema_migrations SET status='failed', completed_at_ms=?, "
            "error_class=?, updated_at_ms=? WHERE migration_id=?",
            (now_ms, type(exc).__name__[:200], now_ms, MIGRATION_ID),
        )
        await conn.commit()
    except Exception:
        await conn.rollback()


async def ensure_operational_storage(
    conn: aiosqlite.Connection,
    db_path: str,
    *,
    app_version: str,
) -> BackupEvidence | None:
    """Ensure the additive v2 schema and legacy journal are installed.

    Returns backup evidence only when this invocation performed the first DDL.
    Existing v2 databases are checksum-verified and left in their current phase.
    """

    await _validate_runtime(conn)
    memory_database = db_path == ":memory:" or db_path.startswith("file::memory:")
    resolved = Path(db_path).expanduser().resolve() if not memory_database else None
    lock = _MigrationFileLock(resolved) if resolved is not None else None
    if lock is not None:
        await asyncio.to_thread(lock.acquire)
    try:
        installed = await _table_exists(conn, "operational_storage_state")
        backup: BackupEvidence | None = None
        if not installed:
            if resolved is not None:
                backup = await _verified_backup(conn, resolved, app_version=app_version)
            # executescript commits an active transaction before running this
            # self-transactional DDL. Recovery is therefore ledger-driven,
            # not based on an impossible outer rollback.
            await conn.executescript(operational_schema_sql())
        await _verify_installed_schema(conn)
        await _begin_or_resume_migration(
            conn,
            app_version=app_version,
            backup=backup,
            ddl_applied_this_run=not installed,
        )
        if await _legacy_bridge_compatible(conn):
            await conn.executescript(legacy_bridge_sql())
        if await _automation_bridge_compatible(conn):
            await conn.executescript(automation_bridge_sql())
        await _complete_migration(
            conn,
            app_version=app_version,
        )
        await _ensure_tool_call_context_repair(
            conn,
            app_version=app_version,
        )
        return backup
    except Exception as exc:
        await conn.rollback()
        await _mark_migration_failed(conn, exc)
        raise
    finally:
        if lock is not None:
            await asyncio.to_thread(lock.release)
