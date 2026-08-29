"""Checksummed additive migration for normalized ordered message parts."""

from __future__ import annotations

import hashlib
import os
import time
from importlib.resources import files
from typing import Any


MIGRATION_ID = "message-parts-v1"
MIGRATION_DESCRIPTION = "add canonical ordered text, attachment, and Custom View message parts"
REQUIRED_TABLE = "session_message_parts"


class MessagePartsMigrationError(RuntimeError):
    pass


def migration_sql() -> str:
    return files("src.memory.operational.sql").joinpath("message_parts_v1.sql").read_text(
        encoding="utf-8"
    )


def migration_checksum() -> str:
    return hashlib.sha256(migration_sql().encode("utf-8")).hexdigest()


async def _verify(conn: Any) -> None:
    row = await (
        await conn.execute(
            "SELECT sql FROM sqlite_schema WHERE type='table' AND name=?",
            (REQUIRED_TABLE,),
        )
    ).fetchone()
    if row is None:
        raise MessagePartsMigrationError("message-parts-v1 is missing its table")
    foreign_keys = await (await conn.execute("PRAGMA foreign_key_check")).fetchall()
    if foreign_keys:
        raise MessagePartsMigrationError("message-parts-v1 foreign key verification failed")


async def ensure_message_parts_storage(conn: Any, *, app_version: str) -> bool:
    """Install or verify the immutable ``message-parts-v1`` ledger entry."""

    checksum = migration_checksum()
    existing = await (
        await conn.execute(
            "SELECT checksum, status FROM schema_migrations WHERE migration_id=?",
            (MIGRATION_ID,),
        )
    ).fetchone()
    if existing is not None and str(existing[0]) != checksum:
        raise MessagePartsMigrationError(
            "message-parts-v1 checksum differs from the migration ledger"
        )
    if existing is not None and str(existing[1]) == "complete":
        await _verify(conn)
        return False

    now_ms = int(time.time() * 1000)
    runner_id = f"pid:{os.getpid()}"
    if existing is None:
        await conn.execute(
            "INSERT INTO schema_migrations "
            "(migration_id, checksum, description, status, started_at_ms, completed_at_ms, "
            "app_version, runner_id, error_class, created_at_ms, updated_at_ms) "
            "VALUES (?, ?, ?, 'running', ?, NULL, ?, ?, NULL, ?, ?)",
            (
                MIGRATION_ID,
                checksum,
                MIGRATION_DESCRIPTION,
                now_ms,
                app_version,
                runner_id,
                now_ms,
                now_ms,
            ),
        )
    else:
        await conn.execute(
            "UPDATE schema_migrations SET status='running', "
            "started_at_ms=COALESCE(started_at_ms, ?), completed_at_ms=NULL, "
            "app_version=?, runner_id=?, error_class=NULL, updated_at_ms=? "
            "WHERE migration_id=? AND status<>'complete'",
            (now_ms, app_version, runner_id, now_ms, MIGRATION_ID),
        )
    await conn.commit()
    try:
        await conn.executescript(migration_sql())
        await _verify(conn)
        done_ms = int(time.time() * 1000)
        await conn.execute(
            "UPDATE schema_migrations SET status='complete', completed_at_ms=?, "
            "error_class=NULL, updated_at_ms=? WHERE migration_id=? AND status<>'complete'",
            (done_ms, done_ms, MIGRATION_ID),
        )
        await conn.commit()
        return True
    except Exception as exc:
        await conn.rollback()
        failed_ms = int(time.time() * 1000)
        try:
            await conn.execute(
                "UPDATE schema_migrations SET status='failed', completed_at_ms=?, "
                "error_class=?, updated_at_ms=? WHERE migration_id=? AND status<>'complete'",
                (failed_ms, type(exc).__name__[:200], failed_ms, MIGRATION_ID),
            )
            await conn.commit()
        except Exception:
            await conn.rollback()
        raise


__all__ = [
    "MIGRATION_ID",
    "MessagePartsMigrationError",
    "ensure_message_parts_storage",
    "migration_checksum",
    "migration_sql",
]
