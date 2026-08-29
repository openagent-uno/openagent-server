"""Checksummed additive migration for Custom Views v1."""

from __future__ import annotations

import hashlib
import os
import time
from importlib.resources import files
from typing import Any


MIGRATION_ID = "custom-views-v1"
MIGRATION_DESCRIPTION = "add revisioned custom views, live data, sources, actions, and chat links"
REQUIRED_TABLES = frozenset({
    "ui_views", "ui_view_revisions", "ui_data_sources", "ui_data_state",
    "ui_actions", "ui_action_runs", "ui_message_links",
})


class CustomViewMigrationError(RuntimeError):
    pass


def migration_sql() -> str:
    return files("src.memory.operational.sql").joinpath("custom_views_v1.sql").read_text(encoding="utf-8")


def migration_checksum() -> str:
    return hashlib.sha256(migration_sql().encode("utf-8")).hexdigest()


async def _verify(conn: Any) -> None:
    placeholders = ",".join("?" for _ in REQUIRED_TABLES)
    rows = await (
        await conn.execute(
            f"SELECT name FROM sqlite_schema WHERE type='table' AND name IN ({placeholders})",
            tuple(sorted(REQUIRED_TABLES)),
        )
    ).fetchall()
    present = {str(row[0]) for row in rows}
    missing = REQUIRED_TABLES - present
    if missing:
        raise CustomViewMigrationError("custom-views-v1 is missing required tables")
    # Scoped to the tables THIS migration owns, one PRAGMA per table.
    #
    # The unscoped `PRAGMA foreign_key_check` checks the whole database, so a
    # dangling row anywhere — an old `task_runs` row whose `scheduled_tasks`
    # parent was deleted, an `event_deliveries` row for a removed event —
    # failed this migration and, because `connect()` raises, stopped the agent
    # from booting AT ALL. Measured on a three-month-old production agent: 8
    # such rows, none of them in a `ui_*` table, and the process died in
    # `_serve` before the HTTP server ever came up. A migration that adds seven
    # tables must not adjudicate the integrity of the rest of the database, and
    # certainly must not make old orphan rows an unbootable condition.
    for table in sorted(REQUIRED_TABLES):
        fk = await (
            await conn.execute(f"PRAGMA foreign_key_check({table})")
        ).fetchall()
        if fk:
            raise CustomViewMigrationError(
                f"custom-views-v1 foreign key verification failed on {table}"
            )


async def ensure_custom_views_storage(conn: Any, *, app_version: str) -> bool:
    """Install or verify the immutable `custom-views-v1` ledger entry."""

    checksum = migration_checksum()
    existing = await (
        await conn.execute(
            "SELECT checksum, status FROM schema_migrations WHERE migration_id=?",
            (MIGRATION_ID,),
        )
    ).fetchone()
    if existing is not None and str(existing[0]) != checksum:
        raise CustomViewMigrationError("custom-views-v1 checksum differs from the migration ledger")
    if existing is not None and str(existing[1]) == "complete":
        await _verify(conn)
        return False

    now_ms = int(time.time() * 1000)
    if existing is None:
        await conn.execute(
            "INSERT INTO schema_migrations "
            "(migration_id, checksum, description, status, started_at_ms, completed_at_ms, "
            "app_version, runner_id, error_class, created_at_ms, updated_at_ms) "
            "VALUES (?, ?, ?, 'running', ?, NULL, ?, ?, NULL, ?, ?)",
            (
                MIGRATION_ID, checksum, MIGRATION_DESCRIPTION, now_ms, app_version,
                f"pid:{os.getpid()}", now_ms, now_ms,
            ),
        )
    else:
        await conn.execute(
            "UPDATE schema_migrations SET status='running', "
            "started_at_ms=COALESCE(started_at_ms, ?), completed_at_ms=NULL, "
            "app_version=?, runner_id=?, error_class=NULL, updated_at_ms=? "
            "WHERE migration_id=? AND status!='complete'",
            (now_ms, app_version, f"pid:{os.getpid()}", now_ms, MIGRATION_ID),
        )
    await conn.commit()
    try:
        await conn.executescript(migration_sql())
        await _verify(conn)
        done_ms = int(time.time() * 1000)
        await conn.execute(
            "UPDATE schema_migrations SET status='complete', completed_at_ms=?, "
            "error_class=NULL, updated_at_ms=? WHERE migration_id=? AND status!='complete'",
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
                "error_class=?, updated_at_ms=? WHERE migration_id=? AND status!='complete'",
                (failed_ms, type(exc).__name__[:200], failed_ms, MIGRATION_ID),
            )
            await conn.commit()
        except Exception:
            await conn.rollback()
        raise
