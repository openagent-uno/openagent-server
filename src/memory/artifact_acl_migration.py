"""Checksummed migration for owner-private, link-authorized artifacts."""

from __future__ import annotations

import hashlib
import os
import time
from importlib.resources import files
from typing import Any

MIGRATION_ID = "artifact-acl-v1"
MIGRATION_DESCRIPTION = (
    "normalize artifacts to owner-private metadata and inherit sharing from links"
)
REQUIRED_TRIGGERS = frozenset(
    {
        "trg_artifacts_owner_private_insert",
        "trg_artifacts_owner_private_update",
        "trg_resource_acl_no_artifact_insert",
        "trg_resource_acl_no_artifact_update",
    }
)


class ArtifactAclMigrationError(RuntimeError):
    """Artifact ACL invariants or the immutable migration ledger are invalid."""


def migration_sql() -> str:
    return files("src.memory.operational.sql").joinpath("artifact_acl_v1.sql").read_text(
        encoding="utf-8"
    )


def migration_checksum() -> str:
    return hashlib.sha256(migration_sql().encode("utf-8")).hexdigest()


async def _verify(conn: Any) -> None:
    invalid = await (
        await conn.execute(
            "SELECT COUNT(*) FROM artifacts WHERE visibility<>'private' "
            "OR owner_principal_id IS NULL OR owner_principal_id=''"
        )
    ).fetchone()
    if invalid is None or int(invalid[0]) != 0:
        raise ArtifactAclMigrationError("artifact-acl-v1 left non-private artifacts")
    direct_grants = await (
        await conn.execute(
            "SELECT COUNT(*) FROM resource_acl WHERE resource_type='artifact'"
        )
    ).fetchone()
    if direct_grants is None or int(direct_grants[0]) != 0:
        raise ArtifactAclMigrationError("artifact-acl-v1 left direct artifact grants")
    placeholders = ",".join("?" for _ in REQUIRED_TRIGGERS)
    rows = await (
        await conn.execute(
            f"SELECT name FROM sqlite_schema WHERE type='trigger' "
            f"AND name IN ({placeholders})",
            tuple(sorted(REQUIRED_TRIGGERS)),
        )
    ).fetchall()
    if {str(row[0]) for row in rows} != REQUIRED_TRIGGERS:
        raise ArtifactAclMigrationError("artifact-acl-v1 is missing privacy triggers")


async def ensure_artifact_acl_storage(conn: Any, *, app_version: str) -> bool:
    """Install or verify the immutable ``artifact-acl-v1`` ledger entry."""

    checksum = migration_checksum()
    existing = await (
        await conn.execute(
            "SELECT checksum, status FROM schema_migrations WHERE migration_id=?",
            (MIGRATION_ID,),
        )
    ).fetchone()
    if existing is not None and str(existing[0]) != checksum:
        raise ArtifactAclMigrationError(
            "artifact-acl-v1 checksum differs from the migration ledger"
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
    "ArtifactAclMigrationError",
    "ensure_artifact_acl_storage",
    "migration_checksum",
    "migration_sql",
]
