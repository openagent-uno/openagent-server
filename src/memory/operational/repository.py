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

from .projection import SessionProjection, build_session_projection


@dataclass(frozen=True)
class ProjectionWrite:
    session_id: str
    changed: bool
    deleted: bool
    source_version: int | None
    completeness: str | None
    history_revision: int | None


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
    conn.execute(
        "UPDATE legacy_session_changes SET processed_at_ms=?, source_hash=?, "
        "last_error_class=?, claimed_by=NULL, claimed_at_ms=NULL "
        "WHERE session_id=? AND processed_at_ms IS NULL",
        (now_ms, source_hash, error_class, session_id),
    )


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


def _insert_run(conn: Any, row: dict[str, Any], *, tenant_id: str, session_id: str, source_version: int) -> None:
    columns = (
        "id", "tenant_id", "session_id", "ordinal", "idempotency_key",
        "parent_run_id", "runner_kind", "agent_id", "team_id", "workflow_id",
        "workflow_step_id", "status", "status_raw", "model", "model_provider",
        "input_json", "output_json", "metrics_json", "metadata_json",
        "source_version", "completeness", "raw_envelope_json",
        "raw_envelope_schema", "legacy_raw_json", "created_at_ms", "finished_at_ms",
    )
    values = dict(row)
    values.update(
        tenant_id=tenant_id,
        session_id=session_id,
        source_version=source_version,
        raw_envelope_schema=1,
        legacy_raw_json=None,
    )
    conn.execute(
        f"INSERT INTO session_runs ({','.join(columns)}) VALUES "
        f"({','.join('?' for _ in columns)})",
        tuple(values.get(column) for column in columns),
    )


def _insert_message(conn: Any, row: dict[str, Any], *, tenant_id: str, session_id: str, source_version: int) -> None:
    columns = (
        "id", "tenant_id", "session_id", "run_id", "sequence", "ordinal",
        "idempotency_key", "role", "status", "author_kind",
        "author_principal_id", "author_handle_snapshot", "author_display",
        "author_device_id", "name", "text", "content_json",
        "compressed_content", "reasoning_content", "redacted_reasoning_content",
        "tool_call_id", "visibility", "source_version", "completeness",
        "raw_envelope_json", "raw_envelope_schema", "legacy_inferred",
        "created_at_ms", "updated_at_ms", "completed_at_ms",
    )
    values = dict(row)
    values.update(
        tenant_id=tenant_id,
        session_id=session_id,
        source_version=source_version,
        raw_envelope_schema=1,
    )
    conn.execute(
        f"INSERT INTO session_messages ({','.join(columns)}) VALUES "
        f"({','.join('?' for _ in columns)})",
        tuple(values.get(column) for column in columns),
    )


def _insert_tool(
    conn: Any,
    row: dict[str, Any],
    *,
    tenant_id: str,
    session_id: str,
    source_version: int,
    owner_principal_id: str | None,
    visibility: str,
    acl_version: int,
) -> None:
    columns = (
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
    conn.execute(
        f"INSERT INTO tool_invocations ({','.join(columns)}) VALUES "
        f"({','.join('?' for _ in columns)})",
        tuple(values.get(column) for column in columns),
    )


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
) -> ProjectionWrite:
    """Project one legacy row without committing the caller's transaction."""

    effective_now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    if not operational_storage_available(conn):
        return ProjectionWrite(session_id, False, False, None, None, None)
    legacy = _fetch_legacy(conn, session_id)
    if legacy is None:
        return tombstone_session(conn, session_id, now_ms=effective_now)

    tenant_id = _tenant_id(conn)
    projection = build_session_projection(
        legacy,
        tenant_id=tenant_id,
        now_ms=effective_now,
    )
    existing_cursor = conn.execute(
        "SELECT tenant_id, source_version, legacy_source_hash, deleted_at_ms "
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
    old_messages, old_tools = _old_resource_ids(conn, session_id)
    session_row = dict(projection.session)
    parent_id, root_id = _lineage_for_write(conn, projection, tenant_id)
    session_row.update(
        parent_session_id=parent_id,
        root_session_id=root_id,
        source_version=source_version,
        deleted_at_ms=None,
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

    # Replace one session's nested projection atomically. Full-fidelity raw
    # envelopes remain in the new rows; only the rebuildable search DB redacts.
    conn.execute("DELETE FROM tool_invocations WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM session_messages WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM session_runs WHERE session_id=?", (session_id,))
    _upsert_session_row(conn, session_row)
    for row in run_rows:
        _insert_run(
            conn,
            row,
            tenant_id=tenant_id,
            session_id=session_id,
            source_version=source_version,
        )
    for row in projection.messages:
        _insert_message(
            conn,
            row,
            tenant_id=tenant_id,
            session_id=session_id,
            source_version=source_version,
        )
    for row in projection.tools:
        _insert_tool(
            conn,
            row,
            tenant_id=tenant_id,
            session_id=session_id,
            source_version=source_version,
            owner_principal_id=session_row.get("owner_principal_id"),
            visibility=str(session_row["visibility"]),
            acl_version=int(session_row["acl_version"]),
        )

    revision = _allocate_revision(conn, effective_now)
    _upsert_activity(
        conn,
        session=session_row,
        source_version=source_version,
        revision=revision,
    )
    current_messages = {str(row["id"]) for row in projection.messages}
    current_tools = {str(row["id"]) for row in projection.tools}
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
    for resource_id in sorted(current_messages):
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
    for resource_id in sorted(old_messages - current_messages):
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
    for resource_id in sorted(current_tools):
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
    for resource_id in sorted(old_tools - current_tools):
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
        },
    )
    _mark_legacy_changes(
        conn,
        session_id,
        now_ms=effective_now,
        source_hash=projection.source_hash,
    )
    return ProjectionWrite(
        session_id,
        True,
        False,
        source_version,
        str(session_row["completeness"]),
        revision,
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
        "source_version, deleted_at_ms, created_at_ms FROM sessions_v2 WHERE id=?",
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
    deleted_at = max(effective_now, int(row["created_at_ms"]))
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
        "SELECT session_id, MIN(seq) AS first_seq FROM legacy_session_changes "
        "WHERE processed_at_ms IS NULL GROUP BY session_id "
        "ORDER BY first_seq LIMIT ?",
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
            conn.execute(
                "UPDATE legacy_session_changes SET attempt_count=attempt_count+1, "
                "last_error_class=?, claimed_by=NULL, claimed_at_ms=NULL "
                "WHERE session_id=? AND processed_at_ms IS NULL",
                (type(exc).__name__[:200], session_id),
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
    failed = 0
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
        except Exception:
            conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            conn.execute(f"RELEASE SAVEPOINT {savepoint}")
            failed += 1
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
            failed,
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
    failed = int(
        conn.execute(
            "SELECT failed_sessions FROM storage_migration_state WHERE singleton_id=1"
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
        "complete": projected >= legacy and failed == 0 and pending == 0,
    }


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
