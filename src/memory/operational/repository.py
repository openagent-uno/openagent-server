"""Transactional projection repository for operational-storage v2.

The legacy ``sessions`` row remains the rollback source in ``shadow``. Every
legacy writer is observed by SQLite triggers; reviewed writers additionally
call this repository in the same transaction for read-after-write parity. The
reconciler consumes any trigger entries left by older/direct writers.

This module intentionally uses the Python DB-API surface only. It can therefore
run inside SQLAlchemy's existing transaction and, through aiosqlite's serialized
connection worker, inside ``MemoryDB``'s transaction without opening a second
writer or committing on behalf of the caller.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import aiosqlite

from .projection import (
    NESTED_LAYOUT_KEY,
    SessionProjection,
    build_session_projection,
)


@dataclass(frozen=True)
class ProjectionDelta:
    """Physical normalized-row mutations made by one projection write.

    Keeping this evidence on the write result makes the append-cost contract
    testable without relying on SQLite's connection-global ``total_changes``.
    An append must report only the new/changed tail rows; unchanged history is
    neither updated nor deleted.
    """

    runs_inserted: int = 0
    runs_updated: int = 0
    runs_deleted: int = 0
    messages_inserted: int = 0
    messages_updated: int = 0
    messages_deleted: int = 0
    tools_inserted: int = 0
    tools_updated: int = 0
    tools_deleted: int = 0

    @property
    def nested_writes(self) -> int:
        return (
            self.runs_inserted
            + self.runs_updated
            + self.messages_inserted
            + self.messages_updated
            + self.tools_inserted
            + self.tools_updated
        )

    @property
    def nested_deletes(self) -> int:
        return self.runs_deleted + self.messages_deleted + self.tools_deleted


@dataclass(frozen=True)
class ProjectionWrite:
    session_id: str
    changed: bool
    deleted: bool
    source_version: int | None
    completeness: str | None
    history_revision: int | None
    delta: ProjectionDelta = ProjectionDelta()


@dataclass(frozen=True)
class ProjectionVerification:
    session_id: str
    matches: bool
    eligible_for_v2: bool
    source_hash: str | None
    source_version: int | None
    reason: str | None = None
    mismatched_fields: tuple[str, ...] = ()


def _row_dict(cursor: Any, row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if hasattr(row, "keys"):
        return {str(key): row[key] for key in row.keys()}
    return {
        str(column[0]): value
        for column, value in zip(cursor.description or (), row, strict=False)
    }


def _table_exists(conn: Any, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def operational_storage_available(conn: Any) -> bool:
    return _table_exists(conn, "sessions_v2") and _table_exists(
        conn, "legacy_session_changes"
    )


def _tenant_id(conn: Any) -> str:
    if _table_exists(conn, "network"):
        cursor = conn.execute("SELECT network_id FROM network LIMIT 1")
        row = cursor.fetchone()
        if row is not None and row[0] is not None and str(row[0]).strip():
            return str(row[0]).strip()
    row = conn.execute(
        "SELECT db_instance_id FROM operational_storage_state WHERE singleton_id=1"
    ).fetchone()
    if row is None:
        raise RuntimeError("operational_storage_state is missing")
    return f"installation:{row[0]}"


def _fetch_legacy(conn: Any, session_id: str) -> dict[str, Any] | None:
    cursor = conn.execute(
        "SELECT session_id, session_type, agent_id, team_id, workflow_id, "
        "user_id, session_data, agent_data, team_data, workflow_data, "
        "metadata, runs, summary, created_at, updated_at "
        "FROM sessions WHERE session_id=?",
        (session_id,),
    )
    row = cursor.fetchone()
    return _row_dict(cursor, row) if row is not None else None


def _allocate_revision(conn: Any, now_ms: int) -> int:
    row = conn.execute(
        "UPDATE operational_storage_state "
        "SET history_revision=history_revision+1, updated_at_ms=? "
        "WHERE singleton_id=1 RETURNING history_revision",
        (now_ms,),
    ).fetchone()
    if row is None:
        raise RuntimeError("failed to allocate operational history revision")
    return int(row[0])


def _mark_legacy_changes(
    conn: Any,
    session_id: str,
    *,
    now_ms: int,
    source_hash: str | None,
    error_class: str | None = None,
) -> None:
    had_failed_retry = conn.execute(
        "SELECT 1 FROM legacy_session_changes WHERE session_id=? "
        "AND processed_at_ms IS NULL AND last_error_class IS NOT NULL LIMIT 1",
        (session_id,),
    ).fetchone() is not None
    conn.execute(
        "UPDATE legacy_session_changes SET processed_at_ms=?, source_hash=?, "
        "last_error_class=?, claimed_by=NULL, claimed_at_ms=NULL "
        "WHERE session_id=? AND processed_at_ms IS NULL",
        (now_ms, source_hash, error_class, session_id),
    )
    if had_failed_retry:
        conn.execute(
            "UPDATE storage_migration_state SET "
            "failed_sessions=MAX(0, failed_sessions-1), updated_at_ms=? "
            "WHERE singleton_id=1",
            (now_ms,),
        )


def _record_projection_failure(
    conn: Any,
    session_id: str,
    *,
    legacy_updated_at: int | None,
    error_class: str,
    now_ms: int,
) -> bool:
    """Persist a fair retry and report whether it is a newly failed session."""

    pending = conn.execute(
        "SELECT 1 FROM legacy_session_changes WHERE session_id=? "
        "AND processed_at_ms IS NULL LIMIT 1",
        (session_id,),
    ).fetchone() is not None
    already_failed = conn.execute(
        "SELECT 1 FROM legacy_session_changes WHERE session_id=? "
        "AND processed_at_ms IS NULL AND last_error_class IS NOT NULL LIMIT 1",
        (session_id,),
    ).fetchone() is not None
    if pending:
        conn.execute(
            "UPDATE legacy_session_changes SET attempt_count=attempt_count+1, "
            "last_error_class=?, claimed_by=NULL, claimed_at_ms=NULL "
            "WHERE session_id=? AND processed_at_ms IS NULL",
            (error_class[:200], session_id),
        )
    else:
        conn.execute(
            "INSERT INTO legacy_session_changes "
            "(session_id, operation, legacy_updated_at, observed_at_ms, "
            "attempt_count, last_error_class) "
            "VALUES (?, 'update', ?, ?, 1, ?)",
            (
                session_id,
                legacy_updated_at,
                now_ms,
                error_class[:200],
            ),
        )
    return not already_failed


def _insert_outbox(
    conn: Any,
    *,
    tenant_id: str,
    source_kind: str,
    source_id: str,
    operation: str,
    source_version: int,
    acl_version: int,
    now_ms: int,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO search_outbox "
        "(tenant_id, source_kind, source_id, operation, source_version, "
        "acl_version, committed_at_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            tenant_id,
            source_kind,
            source_id,
            operation,
            source_version,
            acl_version,
            now_ms,
        ),
    )


def _event_id(session_id: str, event_type: str, source_version: int) -> str:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:24]
    return f"evt:session:{digest}:{event_type}:v{source_version}"


def _insert_domain_event(
    conn: Any,
    *,
    session_id: str,
    tenant_id: str,
    owner_principal_id: str | None,
    event_type: str,
    source_version: int,
    now_ms: int,
    metadata: dict[str, Any],
) -> None:
    principal_type = None
    if owner_principal_id:
        principal_type = owner_principal_id.split(":", 1)[0]
    conn.execute(
        "INSERT OR IGNORE INTO domain_events "
        "(event_id, tenant_id, actor_principal_type, actor_principal_id, "
        "resource_type, resource_id, session_id, event_type, occurred_at_ms, "
        "schema_version, sensitivity, metadata_json) "
        "VALUES (?, ?, ?, ?, 'session', ?, ?, ?, ?, 1, 'normal', ?)",
        (
            _event_id(session_id, event_type, source_version),
            tenant_id,
            principal_type,
            owner_principal_id,
            session_id,
            session_id,
            event_type,
            now_ms,
            json.dumps(metadata, sort_keys=True, separators=(",", ":")),
        ),
    )


def _existing_id(conn: Any, table: str, resource_id: str, tenant_id: str) -> bool:
    row = conn.execute(
        f"SELECT 1 FROM {table} WHERE id=? AND tenant_id=?",  # table is internal
        (resource_id, tenant_id),
    ).fetchone()
    return row is not None


def _lineage_for_write(
    conn: Any,
    projection: SessionProjection,
    tenant_id: str,
) -> tuple[str | None, str | None]:
    row = projection.session
    session_id = str(row["id"])
    desired_parent = row.get("parent_session_id")
    desired_root = row.get("root_session_id")
    parent = (
        str(desired_parent)
        if desired_parent
        and _existing_id(conn, "sessions_v2", str(desired_parent), tenant_id)
        else None
    )
    if parent is None:
        return None, session_id
    root = str(desired_root or parent)
    if root != session_id and not _existing_id(conn, "sessions_v2", root, tenant_id):
        root = parent
    return parent, root


_SESSION_COLUMNS = (
    "id",
    "tenant_id",
    "owner_principal_id",
    "owner_handle_snapshot",
    "visibility",
    "acl_version",
    "title",
    "session_type",
    "kind",
    "origin",
    "parent_session_id",
    "root_session_id",
    "agent_id",
    "team_id",
    "workflow_id",
    "model",
    "framework",
    "status",
    "completeness",
    "source_version",
    "legacy_source_hash",
    "metadata_json",
    "created_at_ms",
    "updated_at_ms",
    "last_activity_at_ms",
    "deleted_at_ms",
)


def _upsert_session_row(conn: Any, row: dict[str, Any]) -> None:
    placeholders = ",".join("?" for _ in _SESSION_COLUMNS)
    updates = ",".join(
        f"{column}=excluded.{column}"
        for column in _SESSION_COLUMNS
        if column != "id"
    )
    conn.execute(
        f"INSERT INTO sessions_v2 ({','.join(_SESSION_COLUMNS)}) "
        f"VALUES ({placeholders}) ON CONFLICT(id) DO UPDATE SET {updates}",
        tuple(row.get(column) for column in _SESSION_COLUMNS),
    )


_RUN_COLUMNS = (
    "id", "tenant_id", "session_id", "ordinal", "idempotency_key",
    "parent_run_id", "runner_kind", "agent_id", "team_id", "workflow_id",
    "workflow_step_id", "status", "status_raw", "model", "model_provider",
    "input_json", "output_json", "metrics_json", "metadata_json",
    "source_version", "completeness", "raw_envelope_json",
    "raw_envelope_schema", "legacy_raw_json", "created_at_ms", "finished_at_ms",
)

_MESSAGE_COLUMNS = (
    "id", "tenant_id", "session_id", "run_id", "sequence", "ordinal",
    "idempotency_key", "role", "status", "author_kind",
    "author_principal_id", "author_handle_snapshot", "author_display",
    "author_device_id", "name", "text", "content_json",
    "compressed_content", "reasoning_content", "redacted_reasoning_content",
    "tool_call_id", "visibility", "source_version", "completeness",
    "raw_envelope_json", "raw_envelope_schema", "legacy_inferred",
    "created_at_ms", "updated_at_ms", "completed_at_ms",
)

_TOOL_COLUMNS = (
    "id", "tenant_id", "owner_principal_id", "visibility", "acl_version",
    "root_kind", "root_id", "session_id", "session_run_id",
    "workflow_run_id", "workflow_step_id", "task_run_id", "event_delivery_id",
    "ordinal", "idempotency_key", "tool_call_id", "tool_server", "tool_name",
    "status", "status_raw", "args_json", "result_json", "result_text",
    "error_json", "error_text", "approval_json", "sensitivity",
    "child_run_id", "child_session_id", "result_sha256", "result_size_bytes",
    "result_complete", "source_version", "completeness", "raw_envelope_json",
    "raw_envelope_schema", "legacy_inferred", "created_at_ms", "finished_at_ms",
)

# Search documents for messages/tools hydrate root metadata from ``sessions_v2``
# when an outbox event is consumed. Replay unchanged children when one of those
# inherited title/lineage/ACL fields changes, or the derived index would retain
# stale display data and, more importantly, stale access-control metadata.
_SESSION_SEARCH_HEADER_COLUMNS = (
    "owner_principal_id",
    "visibility",
    "acl_version",
    "title",
    "kind",
    "origin",
    "parent_session_id",
)


def _run_values(
    row: dict[str, Any], *, tenant_id: str, session_id: str, source_version: int
) -> dict[str, Any]:
    values = dict(row)
    values.update(
        tenant_id=tenant_id,
        session_id=session_id,
        source_version=source_version,
        raw_envelope_schema=1,
        legacy_raw_json=None,
    )
    return values


def _message_values(
    row: dict[str, Any], *, tenant_id: str, session_id: str, source_version: int
) -> dict[str, Any]:
    values = dict(row)
    values.update(
        tenant_id=tenant_id,
        session_id=session_id,
        source_version=source_version,
        raw_envelope_schema=1,
    )
    return values


def _tool_values(
    conn: Any,
    row: dict[str, Any],
    *,
    tenant_id: str,
    session_id: str,
    source_version: int,
    owner_principal_id: str | None,
    visibility: str,
    acl_version: int,
) -> dict[str, Any]:
    values = dict(row)
    values.update(
        tenant_id=tenant_id,
        session_id=session_id,
        source_version=source_version,
        owner_principal_id=owner_principal_id,
        visibility=visibility,
        acl_version=acl_version,
        workflow_run_id=None,
        workflow_step_id=None,
        task_run_id=None,
        event_delivery_id=None,
        raw_envelope_schema=1,
    )
    child_run = values.get("child_run_id")
    if child_run and not _existing_id(conn, "session_runs", str(child_run), tenant_id):
        values["child_run_id"] = None
    child_session = values.get("child_session_id")
    if child_session and not _existing_id(
        conn, "sessions_v2", str(child_session), tenant_id
    ):
        values["child_session_id"] = None
    return values


def _insert_row(
    conn: Any, table: str, columns: Sequence[str], values: dict[str, Any]
) -> None:
    conn.execute(
        f"INSERT INTO {table} ({','.join(columns)}) VALUES "  # internal names
        f"({','.join('?' for _ in columns)})",
        tuple(values.get(column) for column in columns),
    )


def _update_row(
    conn: Any, table: str, columns: Sequence[str], values: dict[str, Any]
) -> None:
    mutable = tuple(column for column in columns if column != "id")
    conn.execute(
        f"UPDATE {table} SET "  # internal names
        + ",".join(f"{column}=?" for column in mutable)
        + " WHERE id=?",
        tuple(values.get(column) for column in mutable) + (values["id"],),
    )


def _fetch_nested_rows(
    conn: Any, table: str, columns: Sequence[str], session_id: str
) -> dict[str, dict[str, Any]]:
    cursor = conn.execute(
        f"SELECT {','.join(columns)} FROM {table} WHERE session_id=?",  # internal names
        (session_id,),
    )
    return {
        str(row["id"]): row
        for raw in cursor.fetchall()
        for row in (_row_dict(cursor, raw),)
    }


def _row_changed(
    existing: dict[str, Any], desired: dict[str, Any], columns: Sequence[str]
) -> bool:
    # ``source_version`` identifies the revision that last changed this record,
    # not the current parent-session revision. Advancing it on every append
    # would turn an otherwise O(new content) write back into O(transcript).
    return any(
        existing.get(column) != desired.get(column)
        for column in columns
        if column != "source_version"
    )


def _changed_columns(
    existing: dict[str, Any], desired: dict[str, Any], columns: Sequence[str]
) -> tuple[str, ...]:
    return tuple(
        column
        for column in columns
        if column != "source_version"
        and existing.get(column) != desired.get(column)
    )


def _ids_requiring_reinsert(
    existing: dict[str, dict[str, Any]],
    desired: dict[str, dict[str, Any]],
    structural_columns: Sequence[str],
) -> set[str]:
    """Rows whose uniqueness/FK identity moved must be deleted before insert.

    Ordinary content/status updates retain their rowid.  Sequence/ordinal/run
    moves can collide with a stale row's UNIQUE slot, so those rare rewrites
    deliberately take the delete+insert path while append-only tails do not.
    """

    return {
        resource_id
        for resource_id in existing.keys() & desired.keys()
        if any(
            existing[resource_id].get(column) != desired[resource_id].get(column)
            for column in structural_columns
        )
    }


def _sync_desired_rows(
    conn: Any,
    *,
    table: str,
    columns: Sequence[str],
    existing: dict[str, dict[str, Any]],
    desired: dict[str, dict[str, Any]],
    predeleted: set[str],
) -> tuple[set[str], set[str]]:
    inserted: set[str] = set()
    updated: set[str] = set()
    for resource_id, values in desired.items():
        prior = existing.get(resource_id)
        if prior is None or resource_id in predeleted:
            _insert_row(conn, table, columns, values)
            inserted.add(resource_id)
        elif _row_changed(prior, values, columns):
            _update_row(conn, table, columns, values)
            updated.add(resource_id)
    return inserted, updated


def _projection_schema(metadata_json: Any) -> int:
    try:
        parsed = json.loads(str(metadata_json or "{}"))
    except (TypeError, ValueError):
        return 0
    if not isinstance(parsed, dict):
        return 0
    try:
        return int(parsed.get("projection_schema") or 0)
    except (TypeError, ValueError):
        return 0


def _has_deferred_projection_links(projection: SessionProjection) -> bool:
    """Whether a previously missing target can change this projection.

    The first pass deliberately nulls cross-session/run FKs whose target has
    not been projected yet. Such a projection cannot use the source-hash fast
    path forever: replay after the target arrives must get a chance to relink.
    """

    if projection.session.get("parent_session_id"):
        return True
    return any(
        row.get("child_session_id") or row.get("child_run_id")
        for row in projection.tools
    )


def _assert_unique_projection_ids(
    rows: Iterable[dict[str, Any]], *, resource: str
) -> None:
    seen: set[str] = set()
    for row in rows:
        resource_id = str(row["id"])
        if resource_id in seen:
            # Match the database contract callers already handle: duplicated
            # provider ids are an integrity failure, never last-one-wins.
            raise sqlite3.IntegrityError(
                f"duplicate normalized {resource} id {resource_id!r}"
            )
        seen.add(resource_id)


def _upsert_activity(
    conn: Any,
    *,
    session: dict[str, Any],
    source_version: int,
    revision: int,
) -> None:
    session_id = str(session["id"])
    origin = str(session.get("origin") or "chat").lower()
    parent_id = session.get("parent_session_id")
    # Causal workflow/scheduler/event child sessions are persisted as ordinary
    # activity rows so ``include_children=true`` can expose them without a
    # second storage path. The default history query filters these origins and
    # therefore still absorbs them beneath their execution root.

    activity_id = f"activity:session:{session_id}"
    existing = conn.execute(
        "SELECT created_revision FROM activity_items WHERE activity_id=?",
        (activity_id,),
    ).fetchone()
    created_revision = int(existing[0]) if existing is not None else revision
    latest = conn.execute(
        "SELECT status FROM session_runs WHERE session_id=? "
        "ORDER BY ordinal DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    status = str(latest[0]) if latest is not None else None
    kind = "delegated_session" if parent_id else "chat"
    occurred_at = int(session["last_activity_at_ms"])
    conn.execute(
        "INSERT INTO activity_items "
        "(activity_id, kind, resource_type, resource_id, parent_type, parent_id, "
        "session_id, tenant_id, owner_principal_id, visibility, acl_version, "
        "status, title, origin, occurred_at_ms, updated_at_ms, source_version, "
        "created_revision, updated_revision, deleted_revision, deleted_at_ms) "
        "VALUES (?, ?, 'session', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL) "
        "ON CONFLICT(activity_id) DO UPDATE SET "
        "kind=excluded.kind, parent_type=excluded.parent_type, parent_id=excluded.parent_id, "
        "session_id=excluded.session_id, tenant_id=excluded.tenant_id, "
        "owner_principal_id=excluded.owner_principal_id, visibility=excluded.visibility, "
        "acl_version=excluded.acl_version, status=excluded.status, title=excluded.title, "
        "origin=excluded.origin, occurred_at_ms=excluded.occurred_at_ms, "
        "updated_at_ms=excluded.updated_at_ms, source_version=excluded.source_version, "
        "updated_revision=excluded.updated_revision, deleted_revision=NULL, deleted_at_ms=NULL",
        (
            activity_id,
            kind,
            session_id,
            "session" if parent_id else None,
            parent_id,
            session_id,
            session["tenant_id"],
            session.get("owner_principal_id"),
            session["visibility"],
            session["acl_version"],
            status,
            session.get("title"),
            session.get("origin"),
            occurred_at,
            int(session["updated_at_ms"]),
            source_version,
            created_revision,
            revision,
        ),
    )


def _old_resource_ids(conn: Any, session_id: str) -> tuple[set[str], set[str]]:
    messages = {
        str(row[0])
        for row in conn.execute(
            "SELECT id FROM session_messages WHERE session_id=?", (session_id,)
        ).fetchall()
    }
    tools = {
        str(row[0])
        for row in conn.execute(
            "SELECT id FROM tool_invocations WHERE session_id=?", (session_id,)
        ).fetchall()
    }
    return messages, tools


def project_legacy_session(
    conn: Any,
    session_id: str,
    *,
    now_ms: int | None = None,
    source_row: dict[str, Any] | None = None,
) -> ProjectionWrite:
    """Incrementally apply one session source without committing the caller.

    Reviewed runtime writers pass the row they already persisted, avoiding a
    second read of the large compatibility blob. Reconciliation/backfill omit
    it and fetch the durable legacy source by id.
    """

    effective_now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if not operational_storage_available(conn):
        return ProjectionWrite(session_id, False, False, None, None, None)
    legacy = source_row if source_row is not None else _fetch_legacy(conn, session_id)
    if legacy is None:
        return tombstone_session(conn, session_id, now_ms=effective_now)
    if str(legacy.get("session_id") or "") != session_id:
        raise ValueError("runtime session source row does not match requested id")

    tenant_id = _tenant_id(conn)
    projection = build_session_projection(
        legacy,
        tenant_id=tenant_id,
        now_ms=effective_now,
    )
    existing_cursor = conn.execute(
        "SELECT tenant_id, source_version, legacy_source_hash, metadata_json, "
        "deleted_at_ms, owner_principal_id, visibility, acl_version, title, "
        "kind, origin, parent_session_id "
        "FROM sessions_v2 WHERE id=?",
        (session_id,),
    )
    existing_row = existing_cursor.fetchone()
    existing = _row_dict(existing_cursor, existing_row) if existing_row else None
    if existing is not None and str(existing["tenant_id"]) != tenant_id:
        raise RuntimeError("refusing to move an operational session across tenants")
    if (
        existing is not None
        and existing.get("deleted_at_ms") is None
        and str(existing.get("legacy_source_hash") or "") == projection.source_hash
        and _projection_schema(existing.get("metadata_json")) >= 3
        and not _has_deferred_projection_links(projection)
    ):
        _mark_legacy_changes(
            conn,
            session_id,
            now_ms=effective_now,
            source_hash=projection.source_hash,
        )
        return ProjectionWrite(
            session_id,
            False,
            False,
            int(existing["source_version"]),
            str(projection.session["completeness"]),
            None,
        )

    source_version = int(existing["source_version"]) + 1 if existing else 1
    session_row = dict(projection.session)
    parent_id, root_id = _lineage_for_write(conn, projection, tenant_id)
    session_row.update(
        parent_session_id=parent_id,
        root_session_id=root_id,
        source_version=source_version,
        deleted_at_ms=None,
    )
    search_header_changed = existing is not None and any(
        existing.get(column) != session_row.get(column)
        for column in _SESSION_SEARCH_HEADER_COLUMNS
    )

    # The parent run FK is deferred, but provider data sometimes references a
    # run not present in the persisted envelope. Keep that raw reference in the
    # envelope and null only the relational projection.
    run_ids = {str(row["id"]) for row in projection.runs}
    run_rows: list[dict[str, Any]] = []
    for raw in projection.runs:
        row = dict(raw)
        if row.get("parent_run_id") not in run_ids:
            row["parent_run_id"] = None
        run_rows.append(row)

    _assert_unique_projection_ids(run_rows, resource="run")
    _assert_unique_projection_ids(projection.messages, resource="message")
    _assert_unique_projection_ids(projection.tools, resource="tool")

    existing_runs = _fetch_nested_rows(
        conn, "session_runs", _RUN_COLUMNS, session_id
    )
    existing_messages = _fetch_nested_rows(
        conn, "session_messages", _MESSAGE_COLUMNS, session_id
    )
    existing_tools = _fetch_nested_rows(
        conn, "tool_invocations", _TOOL_COLUMNS, session_id
    )

    desired_runs = {
        str(row["id"]): _run_values(
            row,
            tenant_id=tenant_id,
            session_id=session_id,
            source_version=source_version,
        )
        for row in run_rows
    }
    desired_messages = {
        str(row["id"]): _message_values(
            row,
            tenant_id=tenant_id,
            session_id=session_id,
            source_version=source_version,
        )
        for row in projection.messages
    }

    # Delete only stale or structurally moved rows.  Tool/message rows that
    # move to another run must go before the old run FK can be removed.  This
    # rare compaction/rewrite path is intentionally distinct from the normal
    # append path, which leaves every prefix row and rowid untouched.
    stale_tools = set(existing_tools) - {str(row["id"]) for row in projection.tools}
    stale_messages = set(existing_messages) - set(desired_messages)
    stale_runs = set(existing_runs) - set(desired_runs)
    moved_messages = _ids_requiring_reinsert(
        existing_messages,
        desired_messages,
        ("run_id", "sequence", "ordinal"),
    )
    moved_runs = _ids_requiring_reinsert(
        existing_runs,
        desired_runs,
        ("session_id", "ordinal"),
    )
    moved_messages |= {
        resource_id
        for resource_id, row in desired_messages.items()
        if row.get("run_id") in moved_runs
    }
    if stale_tools:
        conn.execute(
            f"DELETE FROM tool_invocations WHERE id IN ({','.join('?' for _ in stale_tools)})",
            tuple(sorted(stale_tools)),
        )
    if stale_messages | moved_messages:
        deleted = sorted(stale_messages | moved_messages)
        conn.execute(
            f"DELETE FROM session_messages WHERE id IN ({','.join('?' for _ in deleted)})",
            tuple(deleted),
        )

    # Tools can retain an id while moving between runs/ordinals. Materialize
    # them after current runs exist; for the pre-delete decision only their
    # deterministic projection fields are needed.
    provisional_tools = {
        str(row["id"]): dict(row)
        for row in projection.tools
    }
    moved_tools = _ids_requiring_reinsert(
        existing_tools,
        provisional_tools,
        ("root_kind", "root_id", "session_run_id", "ordinal"),
    )
    moved_tools |= {
        resource_id
        for resource_id, row in provisional_tools.items()
        if row.get("session_run_id") in moved_runs
    }
    if moved_tools - stale_tools:
        deleted = sorted(moved_tools - stale_tools)
        conn.execute(
            f"DELETE FROM tool_invocations WHERE id IN ({','.join('?' for _ in deleted)})",
            tuple(deleted),
        )
    if stale_runs | moved_runs:
        deleted = sorted(stale_runs | moved_runs)
        conn.execute(
            f"DELETE FROM session_runs WHERE id IN ({','.join('?' for _ in deleted)})",
            tuple(deleted),
        )

    _upsert_session_row(conn, session_row)
    inserted_runs, updated_runs = _sync_desired_rows(
        conn,
        table="session_runs",
        columns=_RUN_COLUMNS,
        existing=existing_runs,
        desired=desired_runs,
        predeleted=moved_runs,
    )
    inserted_messages, updated_messages = _sync_desired_rows(
        conn,
        table="session_messages",
        columns=_MESSAGE_COLUMNS,
        existing=existing_messages,
        desired=desired_messages,
        predeleted=moved_messages,
    )
    desired_tools = {
        str(row["id"]): _tool_values(
            conn,
            row,
            tenant_id=tenant_id,
            session_id=session_id,
            source_version=source_version,
            owner_principal_id=session_row.get("owner_principal_id"),
            visibility=str(session_row["visibility"]),
            acl_version=int(session_row["acl_version"]),
        )
        for row in projection.tools
    }
    inserted_tools, updated_tools = _sync_desired_rows(
        conn,
        table="tool_invocations",
        columns=_TOOL_COLUMNS,
        existing=existing_tools,
        desired=desired_tools,
        predeleted=moved_tools,
    )

    delta = ProjectionDelta(
        runs_inserted=len(inserted_runs),
        runs_updated=len(updated_runs),
        runs_deleted=len(stale_runs | moved_runs),
        messages_inserted=len(inserted_messages),
        messages_updated=len(updated_messages),
        messages_deleted=len(stale_messages | moved_messages),
        tools_inserted=len(inserted_tools),
        tools_updated=len(updated_tools),
        tools_deleted=len(stale_tools | moved_tools),
    )

    revision = _allocate_revision(conn, effective_now)
    _upsert_activity(
        conn,
        session=session_row,
        source_version=source_version,
        revision=revision,
    )
    current_messages = set(desired_messages)
    current_tools = set(desired_tools)
    message_outbox_upserts = inserted_messages | updated_messages
    tool_outbox_upserts = inserted_tools | updated_tools
    if search_header_changed:
        message_outbox_upserts |= current_messages
        tool_outbox_upserts |= current_tools
    _insert_outbox(
        conn,
        tenant_id=tenant_id,
        source_kind="session",
        source_id=session_id,
        operation="upsert",
        source_version=source_version,
        acl_version=int(session_row["acl_version"]),
        now_ms=effective_now,
    )
    for resource_id in sorted(message_outbox_upserts):
        _insert_outbox(
            conn,
            tenant_id=tenant_id,
            source_kind="message",
            source_id=resource_id,
            operation="upsert",
            source_version=source_version,
            acl_version=int(session_row["acl_version"]),
            now_ms=effective_now,
        )
    for resource_id in sorted(stale_messages):
        _insert_outbox(
            conn,
            tenant_id=tenant_id,
            source_kind="message",
            source_id=resource_id,
            operation="delete",
            source_version=source_version,
            acl_version=int(session_row["acl_version"]),
            now_ms=effective_now,
        )
    for resource_id in sorted(tool_outbox_upserts):
        _insert_outbox(
            conn,
            tenant_id=tenant_id,
            source_kind="tool_invocation",
            source_id=resource_id,
            operation="upsert",
            source_version=source_version,
            acl_version=int(session_row["acl_version"]),
            now_ms=effective_now,
        )
    for resource_id in sorted(stale_tools):
        _insert_outbox(
            conn,
            tenant_id=tenant_id,
            source_kind="tool_invocation",
            source_id=resource_id,
            operation="delete",
            source_version=source_version,
            acl_version=int(session_row["acl_version"]),
            now_ms=effective_now,
        )
    _insert_domain_event(
        conn,
        session_id=session_id,
        tenant_id=tenant_id,
        owner_principal_id=session_row.get("owner_principal_id"),
        event_type="session.projected",
        source_version=source_version,
        now_ms=effective_now,
        metadata={
            "source_version": source_version,
            "completeness": session_row["completeness"],
            "run_count": len(run_rows),
            "message_count": len(projection.messages),
            "tool_count": len(projection.tools),
            "nested_writes": delta.nested_writes,
            "nested_deletes": delta.nested_deletes,
        },
    )
    _mark_legacy_changes(
        conn,
        session_id,
        now_ms=effective_now,
        source_hash=projection.source_hash,
    )
    phase_row = conn.execute(
        "SELECT phase, COALESCE(last_writer_version, 'operational-v2'), "
        "last_writer_epoch FROM storage_migration_state WHERE singleton_id=1"
    ).fetchone()
    if (
        phase_row is not None
        and str(phase_row[0]) in {"prefer_v2", "v2"}
        and str(session_row["completeness"]) == "complete"
    ):
        # This transaction built the normalized rows from the exact legacy
        # value it also marked processed, so a successful commit is itself a
        # verification boundary. Explicit phase promotion still performs the
        # expensive byte-for-byte comparison for the historical corpus.
        record_session_verification(
            conn,
            ProjectionVerification(
                session_id=session_id,
                matches=True,
                eligible_for_v2=True,
                source_hash=projection.source_hash,
                source_version=source_version,
            ),
            writer_version=str(phase_row[1]),
            writer_epoch=int(phase_row[2]),
            now_ms=effective_now,
            method="native_incremental_write",
        )
    return ProjectionWrite(
        session_id,
        True,
        False,
        source_version,
        str(session_row["completeness"]),
        revision,
        delta,
    )


def tombstone_session(
    conn: Any,
    session_id: str,
    *,
    now_ms: int | None = None,
) -> ProjectionWrite:
    effective_now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if not operational_storage_available(conn):
        return ProjectionWrite(session_id, False, True, None, None, None)
    cursor = conn.execute(
        "SELECT tenant_id, owner_principal_id, visibility, acl_version, "
        "source_version, deleted_at_ms, created_at_ms, updated_at_ms, "
        "last_activity_at_ms FROM sessions_v2 WHERE id=?",
        (session_id,),
    )
    raw = cursor.fetchone()
    if raw is None:
        _mark_legacy_changes(
            conn, session_id, now_ms=effective_now, source_hash=None
        )
        return ProjectionWrite(session_id, False, True, None, None, None)
    row = _row_dict(cursor, raw)
    if row.get("deleted_at_ms") is not None:
        _mark_legacy_changes(
            conn, session_id, now_ms=effective_now, source_hash=None
        )
        return ProjectionWrite(
            session_id, False, True, int(row["source_version"]), None, None
        )

    source_version = int(row["source_version"]) + 1
    activity_boundary = conn.execute(
        "SELECT MAX(occurred_at_ms) FROM activity_items WHERE tenant_id=? "
        "AND resource_type='session' AND resource_id=?",
        (row["tenant_id"], session_id),
    ).fetchone()
    deleted_at = max(
        effective_now,
        int(row["created_at_ms"]),
        int(row["updated_at_ms"]),
        int(row["last_activity_at_ms"]),
        int(activity_boundary[0]) if activity_boundary and activity_boundary[0] is not None else 0,
    )
    old_messages, old_tools = _old_resource_ids(conn, session_id)
    revision = _allocate_revision(conn, effective_now)
    conn.execute(
        "UPDATE sessions_v2 SET status='deleted', source_version=?, "
        "updated_at_ms=MAX(updated_at_ms, ?), last_activity_at_ms=MAX(last_activity_at_ms, ?), "
        "deleted_at_ms=? WHERE id=?",
        (source_version, deleted_at, deleted_at, deleted_at, session_id),
    )
    conn.execute(
        "UPDATE activity_items SET source_version=?, updated_revision=?, "
        "deleted_revision=?, deleted_at_ms=?, updated_at_ms=MAX(updated_at_ms, ?) "
        "WHERE tenant_id=? AND resource_type='session' AND resource_id=? "
        "AND deleted_at_ms IS NULL",
        (
            source_version,
            revision,
            revision,
            deleted_at,
            deleted_at,
            row["tenant_id"],
            session_id,
        ),
    )
    for kind, resource_ids in (
        ("session", {session_id}),
        ("message", old_messages),
        ("tool_invocation", old_tools),
    ):
        for resource_id in sorted(resource_ids):
            _insert_outbox(
                conn,
                tenant_id=str(row["tenant_id"]),
                source_kind=kind,
                source_id=resource_id,
                operation="delete",
                source_version=source_version,
                acl_version=int(row["acl_version"]),
                now_ms=effective_now,
            )
    _insert_domain_event(
        conn,
        session_id=session_id,
        tenant_id=str(row["tenant_id"]),
        owner_principal_id=row.get("owner_principal_id"),
        event_type="session.deleted",
        source_version=source_version,
        now_ms=effective_now,
        metadata={"source_version": source_version},
    )
    _mark_legacy_changes(conn, session_id, now_ms=effective_now, source_hash=None)
    return ProjectionWrite(
        session_id, True, True, source_version, None, revision
    )


def reconcile_pending(
    conn: Any,
    *,
    limit: int = 100,
    worker_id: str = "operational-reconciler",
    now_ms: int | None = None,
) -> list[ProjectionWrite]:
    """Consume durable trigger entries, isolating failures per session."""

    if not operational_storage_available(conn):
        return []
    effective_now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    rows = conn.execute(
        "SELECT session_id, MIN(seq) AS first_seq, "
        "MIN(attempt_count) AS fewest_attempts FROM legacy_session_changes "
        "WHERE processed_at_ms IS NULL GROUP BY session_id "
        "ORDER BY fewest_attempts, first_seq LIMIT ?",
        (max(1, min(int(limit), 1000)),),
    ).fetchall()
    results: list[ProjectionWrite] = []
    for ordinal, row in enumerate(rows):
        session_id = str(row[0])
        savepoint = f"operational_reconcile_{ordinal}"
        conn.execute(f"SAVEPOINT {savepoint}")
        try:
            conn.execute(
                "UPDATE legacy_session_changes SET claimed_by=?, claimed_at_ms=?, "
                "attempt_count=attempt_count+1 WHERE session_id=? "
                "AND processed_at_ms IS NULL",
                (worker_id, effective_now, session_id),
            )
            result = project_legacy_session(
                conn, session_id, now_ms=effective_now
            )
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            results.append(result)
        except Exception as exc:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            newly_failed = _record_projection_failure(
                conn,
                session_id,
                legacy_updated_at=None,
                error_class=type(exc).__name__,
                now_ms=effective_now,
            )
            if newly_failed:
                conn.execute(
                    "UPDATE storage_migration_state SET "
                    "failed_sessions=failed_sessions+1, updated_at_ms=? "
                    "WHERE singleton_id=1",
                    (effective_now,),
                )
    return results


def backfill_batch(
    conn: Any,
    *,
    limit: int = 100,
    now_ms: int | None = None,
) -> tuple[list[ProjectionWrite], bool]:
    """Project a bounded legacy keyset page; return (writes, complete)."""

    if not operational_storage_available(conn):
        return [], True
    effective_now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    state = conn.execute(
        "SELECT checkpoint_updated_at, checkpoint_session_id "
        "FROM storage_migration_state WHERE singleton_id=1"
    ).fetchone()
    checkpoint_time = state[0] if state else None
    checkpoint_id = state[1] if state else None
    page_limit = max(1, min(int(limit), 1000))
    if checkpoint_time is None:
        rows = conn.execute(
            "SELECT session_id, updated_at FROM sessions "
            "ORDER BY updated_at, session_id LIMIT ?",
            (page_limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT session_id, updated_at FROM sessions "
            "WHERE updated_at > ? OR (updated_at = ? AND session_id > ?) "
            "ORDER BY updated_at, session_id LIMIT ?",
            (checkpoint_time, checkpoint_time, checkpoint_id, page_limit),
        ).fetchall()
    if not rows:
        return [], True

    results: list[ProjectionWrite] = []
    newly_failed = 0
    for ordinal, row in enumerate(rows):
        session_id = str(row[0])
        savepoint = f"operational_backfill_{ordinal}"
        conn.execute(f"SAVEPOINT {savepoint}")
        try:
            result = project_legacy_session(
                conn, session_id, now_ms=effective_now
            )
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            results.append(result)
        except Exception as exc:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            if _record_projection_failure(
                conn,
                session_id,
                legacy_updated_at=int(row[1]),
                error_class=type(exc).__name__,
                now_ms=effective_now,
            ):
                newly_failed += 1
    last_id, last_updated = str(rows[-1][0]), int(rows[-1][1])
    conn.execute(
        "UPDATE storage_migration_state SET checkpoint_updated_at=?, "
        "checkpoint_session_id=?, migrated_sessions=migrated_sessions+?, "
        "failed_sessions=failed_sessions+?, last_writer_version=?, updated_at_ms=? "
        "WHERE singleton_id=1",
        (
            last_updated,
            last_id,
            sum(1 for result in results if result.changed),
            newly_failed,
            "operational-v2",
            effective_now,
        ),
    )
    complete = len(rows) < page_limit
    return results, complete


def projection_coverage(conn: Any) -> dict[str, int | bool]:
    legacy = int(conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])
    projected = int(
        conn.execute(
            "SELECT COUNT(*) FROM sessions_v2 WHERE deleted_at_ms IS NULL"
        ).fetchone()[0]
    )
    missing = int(
        conn.execute(
            "SELECT COUNT(*) FROM sessions legacy WHERE NOT EXISTS ("
            "SELECT 1 FROM sessions_v2 projected "
            "WHERE projected.id=legacy.session_id "
            "AND projected.deleted_at_ms IS NULL)"
        ).fetchone()[0]
    )
    failed = int(
        conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM legacy_session_changes "
            "WHERE processed_at_ms IS NULL AND last_error_class IS NOT NULL"
        ).fetchone()[0]
    )
    pending = int(
        conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM legacy_session_changes "
            "WHERE processed_at_ms IS NULL"
        ).fetchone()[0]
    )
    return {
        "legacy_sessions": legacy,
        "projected_sessions": projected,
        "failed_sessions": failed,
        "pending_sessions": pending,
        "complete": missing == 0 and failed == 0 and pending == 0,
    }


def verify_session_projection(
    conn: Any,
    session_id: str,
    *,
    now_ms: int | None = None,
) -> ProjectionVerification:
    """Compare one normalized session byte-for-byte with its legacy source.

    This is intentionally more expensive than the steady-state writer and is
    used only by an explicit phase promotion.  The promotion therefore proves
    nested row identity/content, not merely matching row counts or a source
    hash copied into ``sessions_v2``.
    """

    effective_now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    legacy = _fetch_legacy(conn, session_id)
    if legacy is None:
        return ProjectionVerification(
            session_id, False, False, None, None, "legacy_source_missing"
        )
    tenant_id = _tenant_id(conn)
    stored_time = conn.execute(
        "SELECT created_at_ms FROM sessions_v2 WHERE id=?",
        (session_id,),
    ).fetchone()
    # Legacy rows created before timestamps were mandatory used the projector's
    # write time as their deterministic fallback. Reuse that persisted value
    # during later exact audits instead of comparing against the audit clock.
    projection_now = (
        int(stored_time[0])
        if stored_time is not None and legacy.get("created_at") is None
        else effective_now
    )
    projection = build_session_projection(
        legacy,
        tenant_id=tenant_id,
        now_ms=projection_now,
    )
    _assert_unique_projection_ids(projection.runs, resource="run")
    _assert_unique_projection_ids(projection.messages, resource="message")
    _assert_unique_projection_ids(projection.tools, resource="tool")
    session_cursor = conn.execute(
        f"SELECT {','.join(_SESSION_COLUMNS)} FROM sessions_v2 "
        "WHERE id=? AND deleted_at_ms IS NULL",
        (session_id,),
    )
    raw_session = session_cursor.fetchone()
    if raw_session is None:
        return ProjectionVerification(
            session_id,
            False,
            False,
            projection.source_hash,
            None,
            "normalized_session_missing",
        )
    stored_session = _row_dict(session_cursor, raw_session)
    source_version = int(stored_session["source_version"])
    expected_session = dict(projection.session)
    parent_id, root_id = _lineage_for_write(conn, projection, tenant_id)
    expected_session.update(
        parent_session_id=parent_id,
        root_session_id=root_id,
        source_version=source_version,
        deleted_at_ms=None,
    )
    session_mismatches = _changed_columns(
        stored_session, expected_session, _SESSION_COLUMNS
    )
    mismatch_fields = {
        f"session.{column}" for column in session_mismatches
    }
    mismatch_labels = {"session"} if session_mismatches else set()

    run_ids = {str(row["id"]) for row in projection.runs}
    expected_runs: dict[str, dict[str, Any]] = {}
    for raw in projection.runs:
        row = dict(raw)
        if row.get("parent_run_id") not in run_ids:
            row["parent_run_id"] = None
        expected_runs[str(row["id"])] = _run_values(
            row,
            tenant_id=tenant_id,
            session_id=session_id,
            source_version=source_version,
        )
    expected_messages = {
        str(row["id"]): _message_values(
            row,
            tenant_id=tenant_id,
            session_id=session_id,
            source_version=source_version,
        )
        for row in projection.messages
    }
    expected_tools = {
        str(row["id"]): _tool_values(
            conn,
            row,
            tenant_id=tenant_id,
            session_id=session_id,
            source_version=source_version,
            owner_principal_id=expected_session.get("owner_principal_id"),
            visibility=str(expected_session["visibility"]),
            acl_version=int(expected_session["acl_version"]),
        )
        for row in projection.tools
    }
    actual_sets = (
        (_fetch_nested_rows(conn, "session_runs", _RUN_COLUMNS, session_id), expected_runs, _RUN_COLUMNS, "run"),
        (_fetch_nested_rows(conn, "session_messages", _MESSAGE_COLUMNS, session_id), expected_messages, _MESSAGE_COLUMNS, "message"),
        (_fetch_nested_rows(conn, "tool_invocations", _TOOL_COLUMNS, session_id), expected_tools, _TOOL_COLUMNS, "tool"),
    )
    for actual, expected, columns, label in actual_sets:
        if set(actual) != set(expected):
            return ProjectionVerification(
                session_id,
                False,
                False,
                projection.source_hash,
                source_version,
                f"{label}_identity_mismatch",
            )
        content_mismatches = tuple(sorted({
            column
            for resource_id in expected
            for column in _changed_columns(
                actual[resource_id], expected[resource_id], columns
            )
        }))
        if content_mismatches:
            mismatch_labels.add(label)
            mismatch_fields.update(
                f"{label}.{column}" for column in content_mismatches
            )

    if mismatch_fields:
        reason = (
            "session_header_mismatch"
            if mismatch_labels == {"session"}
            else f"{next(iter(mismatch_labels))}_content_mismatch"
            if len(mismatch_labels) == 1
            else "projection_content_mismatch"
        )
        return ProjectionVerification(
            session_id,
            False,
            False,
            projection.source_hash,
            source_version,
            reason,
            tuple(sorted(mismatch_fields)),
        )

    complete = str(expected_session.get("completeness")) == "complete"
    header_ready = _projection_schema(expected_session.get("metadata_json")) >= 3
    return ProjectionVerification(
        session_id,
        True,
        complete and header_ready,
        projection.source_hash,
        source_version,
        None if complete and header_ready else "session_not_complete",
    )


def record_session_verification(
    conn: Any,
    verification: ProjectionVerification,
    *,
    writer_version: str,
    writer_epoch: int,
    now_ms: int | None = None,
    method: str = "exact_compare",
) -> None:
    if not verification.matches or verification.source_version is None:
        raise ValueError("cannot record a failed projection verification")
    effective_now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    conn.execute(
        "INSERT INTO storage_migration_journal "
        "(migration_id, event_type, from_phase, to_phase, session_id, "
        "source_version, source_hash, writer_version, writer_epoch, "
        "details_json, occurred_at_ms) "
        "VALUES ('operational-storage-v2', 'verification_completed', NULL, "
        "NULL, ?, ?, ?, ?, ?, ?, ?)",
        (
            verification.session_id,
            verification.source_version,
            verification.source_hash,
            writer_version,
            writer_epoch,
            json.dumps(
                {
                    "eligible_for_v2": verification.eligible_for_v2,
                    "method": method,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            effective_now,
        ),
    )


def session_v2_is_verified(conn: Any, session_id: str) -> bool:
    """Whether the current complete projection passed a promotion comparison."""

    row = conn.execute(
        "SELECT source_version, legacy_source_hash, completeness, metadata_json "
        "FROM sessions_v2 WHERE id=? AND deleted_at_ms IS NULL",
        (session_id,),
    ).fetchone()
    if (
        row is None
        or str(row[2]) != "complete"
        or _projection_schema(row[3]) < 3
    ):
        return False
    pending = conn.execute(
        "SELECT 1 FROM legacy_session_changes WHERE session_id=? "
        "AND processed_at_ms IS NULL LIMIT 1",
        (session_id,),
    ).fetchone()
    if pending is not None:
        return False
    verified = conn.execute(
        "SELECT 1 FROM storage_migration_journal "
        "WHERE migration_id='operational-storage-v2' "
        "AND event_type='verification_completed' AND session_id=? "
        "AND source_version=? AND source_hash=? "
        "AND json_extract(details_json, '$.eligible_for_v2')=1 "
        "ORDER BY seq DESC LIMIT 1",
        (session_id, int(row[0]), str(row[1] or "")),
    ).fetchone()
    return verified is not None


def load_v2_legacy_session(conn: Any, session_id: str) -> dict[str, Any] | None:
    """Rehydrate the runtime Session wire shape without reading ``sessions.runs``.

    Bounded provider run headers plus raw per-message/tool envelopes are the
    fidelity boundary. Returning the legacy in-memory shape keeps the runtime
    API compatible while nested transcript reads stay normalized and pageable.
    """

    cursor = conn.execute(
        "SELECT id, metadata_json FROM sessions_v2 "
        "WHERE id=? AND deleted_at_ms IS NULL",
        (session_id,),
    )
    raw = cursor.fetchone()
    if raw is None:
        return None
    try:
        metadata = json.loads(str(raw[1] or "{}"))
    except (TypeError, ValueError):
        return None
    if not isinstance(metadata, dict) or int(metadata.get("projection_schema") or 0) < 3:
        return None
    header = metadata.get("legacy_header")
    if not isinstance(header, dict):
        return None
    run_rows = conn.execute(
        "SELECT id, ordinal, raw_envelope_json FROM session_runs "
        "WHERE session_id=? ORDER BY ordinal",
        (session_id,),
    ).fetchall()
    message_groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for row in conn.execute(
        "SELECT run_id, ordinal, raw_envelope_json FROM session_messages "
        "WHERE session_id=? ORDER BY sequence",
        (session_id,),
    ).fetchall():
        try:
            message = json.loads(str(row[2]))
        except (TypeError, ValueError):
            return None
        if not isinstance(message, dict):
            return None
        # This row materializes run.content for search/history when the
        # provider did not include an assistant message. It was never part of
        # the runtime run.messages array and must not be synthesized into it.
        if message == {"source": "run.content"}:
            continue
        message_groups.setdefault(str(row[0]), []).append((int(row[1]), message))
    tool_groups: dict[str, list[dict[str, Any]]] = {}
    for row in conn.execute(
        "SELECT session_run_id, raw_envelope_json FROM tool_invocations "
        "WHERE session_id=? AND root_kind='session' ORDER BY ordinal",
        (session_id,),
    ).fetchall():
        try:
            tool = json.loads(str(row[1]))
        except (TypeError, ValueError):
            return None
        if not isinstance(tool, dict):
            return None
        tool_groups.setdefault(str(row[0]), []).append(tool)

    def restore_list(
        spec: Any,
        normalized: list[Any],
        *,
        positioned: bool,
    ) -> list[Any] | None:
        if not isinstance(spec, dict):
            return None
        passthrough = spec.get("passthrough") or {}
        if not isinstance(passthrough, dict):
            return None
        try:
            passthrough_ordinals = [int(value) for value in passthrough]
        except (TypeError, ValueError):
            return None
        normalized_ordinals = (
            [int(item[0]) for item in normalized] if positioned else []
        )
        if positioned:
            count = 1 + max([-1, *passthrough_ordinals, *normalized_ordinals])
        else:
            count = len(passthrough) + len(normalized)
        missing = object()
        restored: list[Any] = [missing] * count
        try:
            for raw_ordinal, value in passthrough.items():
                ordinal = int(raw_ordinal)
                if ordinal < 0 or ordinal >= count:
                    return None
                restored[ordinal] = value
        except (TypeError, ValueError):
            return None
        if positioned:
            for item in normalized:
                ordinal, value = item
                if ordinal < 0:
                    return None
                if ordinal >= len(restored):
                    restored.extend([missing] * (ordinal + 1 - len(restored)))
                restored[ordinal] = value
        else:
            available = (
                ordinal
                for ordinal, value in enumerate(restored)
                if value is missing
            )
            for value in normalized:
                ordinal = next(available, len(restored))
                if ordinal == len(restored):
                    restored.append(value)
                else:
                    restored[ordinal] = value
        if any(value is missing for value in restored):
            return None
        return restored

    runs: list[dict[str, Any]] = []
    for expected_ordinal, row in enumerate(run_rows):
        run_id = str(row[0])
        if int(row[1]) != expected_ordinal:
            return None
        try:
            envelope = json.loads(str(row[2]))
        except (TypeError, ValueError):
            return None
        if not isinstance(envelope, dict):
            return None
        layout = envelope.pop(NESTED_LAYOUT_KEY, None)
        if not isinstance(layout, dict):
            return None
        if "provider_reserved_value" in layout:
            envelope[NESTED_LAYOUT_KEY] = layout["provider_reserved_value"]
        if "messages" in layout:
            messages = restore_list(
                layout["messages"],
                message_groups.get(run_id, []),
                positioned=True,
            )
            if messages is None:
                return None
            envelope["messages"] = messages
        if "tools" in layout:
            tools = restore_list(
                layout["tools"],
                tool_groups.get(run_id, []),
                positioned=False,
            )
            if tools is None:
                return None
            envelope["tools"] = tools
        runs.append(envelope)
    result = dict(header)
    result["session_id"] = session_id
    result["runs"] = runs
    return result


def _aiosqlite_raw(conn: aiosqlite.Connection) -> sqlite3.Connection:
    raw = getattr(conn, "_conn", None)
    if raw is None:
        raise RuntimeError("unsupported aiosqlite connection implementation")
    return raw


async def project_legacy_session_async(
    conn: aiosqlite.Connection,
    session_id: str,
    *,
    now_ms: int | None = None,
) -> ProjectionWrite:
    """Run the sync repository on aiosqlite's own serialized worker thread."""

    runner = getattr(conn, "_execute", None)
    if runner is None:
        raise RuntimeError("unsupported aiosqlite connection implementation")
    return await runner(
        project_legacy_session,
        _aiosqlite_raw(conn),
        session_id,
        now_ms=now_ms,
    )


async def reconcile_pending_async(
    conn: aiosqlite.Connection,
    *,
    limit: int = 100,
    worker_id: str = "operational-reconciler",
) -> list[ProjectionWrite]:
    runner = getattr(conn, "_execute", None)
    if runner is None:
        raise RuntimeError("unsupported aiosqlite connection implementation")
    return await runner(
        reconcile_pending,
        _aiosqlite_raw(conn),
        limit=limit,
        worker_id=worker_id,
    )


async def backfill_batch_async(
    conn: aiosqlite.Connection,
    *,
    limit: int = 100,
) -> tuple[list[ProjectionWrite], bool]:
    runner = getattr(conn, "_execute", None)
    if runner is None:
        raise RuntimeError("unsupported aiosqlite connection implementation")
    return await runner(
        backfill_batch,
        _aiosqlite_raw(conn),
        limit=limit,
    )


async def projection_coverage_async(
    conn: aiosqlite.Connection,
) -> dict[str, int | bool]:
    runner = getattr(conn, "_execute", None)
    if runner is None:
        raise RuntimeError("unsupported aiosqlite connection implementation")
    return await runner(projection_coverage, _aiosqlite_raw(conn))


def reconcile_database(
    db_path: str | Path,
    *,
    pending_limit: int = 100,
    backfill_limit: int = 100,
    busy_timeout_ms: int = 60_000,
) -> tuple[list[ProjectionWrite], bool]:
    """Bounded standalone worker used by startup and maintenance loops."""

    conn = sqlite3.connect(str(db_path), timeout=busy_timeout_ms / 1000)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("BEGIN IMMEDIATE")
        pending = reconcile_pending(conn, limit=pending_limit)
        backfilled, complete = backfill_batch(conn, limit=backfill_limit)
        conn.commit()
        return [*pending, *backfilled], complete
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def sqlalchemy_driver_connection(session: Any) -> Any:
    """Return the DB-API connection owned by an active SQLAlchemy session."""

    proxied = session.connection().connection
    return getattr(proxied, "driver_connection", proxied)
