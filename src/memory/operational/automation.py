"""Idempotent projection of legacy automation definitions and executions.

The existing workflow/scheduler/event tables remain canonical during shadow.
They predate ownership columns, so explicit certificate claims live in
``operational_resource_owners``. Unclaimed legacy rows retain the historical
same-installation visibility through the documented ``installation_shared``
policy; no synthetic human owner is invented.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any

from .enums import UnmappedStatusError, normalize_run_status


_RESOURCE_TABLES: tuple[tuple[str, str], ...] = (
    ("workflow_definition", "workflow_tasks"),
    ("workflow_run", "workflow_runs"),
    ("scheduled_definition", "scheduled_tasks"),
    ("scheduled_run", "task_runs"),
    ("event_definition", "events"),
    ("event_delivery", "event_deliveries"),
)


def _hash_row(row: Any) -> str:
    payload = {str(key): row[key] for key in row.keys()}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _ms(value: Any, fallback: int) -> int:
    if value is None:
        return fallback
    number = float(value)
    return int(number if number > 10_000_000_000 else number * 1000)


def _status(value: Any) -> tuple[str, bool]:
    try:
        return normalize_run_status(value)[0], True
    except UnmappedStatusError:
        return "failed", False


async def _tenant_id(conn: Any) -> str:
    row = await (await conn.execute("SELECT network_id FROM network LIMIT 1")).fetchone()
    if row is not None and row[0]:
        return str(row[0])
    row = await (
        await conn.execute("SELECT db_instance_id FROM operational_storage_state WHERE singleton_id=1")
    ).fetchone()
    return f"installation:{row[0]}"


async def claim_resource(
    conn: Any,
    *,
    tenant_id: str,
    resource_type: str,
    resource_id: str,
    owner_principal_id: str,
    now_ms: int | None = None,
) -> None:
    if resource_type not in {kind for kind, _ in _RESOURCE_TABLES}:
        raise ValueError("unsupported operational resource type")
    effective_now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    await conn.execute(
        "INSERT INTO operational_resource_owners "
        "(tenant_id, resource_type, resource_id, owner_principal_id, visibility, "
        "acl_version, provenance, created_at_ms, updated_at_ms) "
        "VALUES (?, ?, ?, ?, 'private', 1, 'certificate', ?, ?) "
        "ON CONFLICT(tenant_id, resource_type, resource_id) DO UPDATE SET "
        "owner_principal_id=excluded.owner_principal_id, visibility='private', "
        "acl_version=operational_resource_owners.acl_version+1, "
        "provenance='certificate', updated_at_ms=excluded.updated_at_ms",
        (tenant_id, resource_type, resource_id, owner_principal_id, effective_now, effective_now),
    )
    # Ownership is part of the search/history projection.  The legacy source
    # row itself did not change, so enqueue an explicit refresh alongside the
    # table triggers.
    await conn.execute(
        "INSERT INTO operational_automation_changes "
        "(resource_type, resource_id, operation, observed_at_ms) "
        "VALUES (?, ?, 'upsert', ?)",
        (resource_type, resource_id, effective_now),
    )


async def _owner(conn: Any, tenant: str, resource_type: str, resource_id: str) -> tuple[str | None, str, int, str]:
    row = await (
        await conn.execute(
            "SELECT owner_principal_id, visibility, acl_version, provenance "
            "FROM operational_resource_owners WHERE tenant_id=? AND resource_type=? AND resource_id=?",
            (tenant, resource_type, resource_id),
        )
    ).fetchone()
    if row is None:
        return None, "installation_shared", 1, "legacy_unattributed"
    return (str(row[0]) if row[0] else None, str(row[1]), int(row[2]), str(row[3]))


async def _effective_owner(
    conn: Any, tenant: str, resource_type: str, resource_id: str, row: Any
) -> tuple[str | None, str, int, str]:
    direct = await _owner(conn, tenant, resource_type, resource_id)
    if direct[3] != "legacy_unattributed":
        return direct
    parent: tuple[str, str] | None = None
    if resource_type == "workflow_run":
        parent = ("workflow_definition", str(row["workflow_id"]))
    elif resource_type == "scheduled_run":
        parent = ("scheduled_definition", str(row["task_id"]))
    elif resource_type == "event_delivery":
        parent = ("event_definition", str(row["event_id"]))
    if parent is None:
        return direct
    inherited = await _owner(conn, tenant, *parent)
    return inherited if inherited[3] != "legacy_unattributed" else direct


async def _allocate_revision(conn: Any, now_ms: int) -> int:
    row = await (
        await conn.execute(
            "UPDATE operational_storage_state SET history_revision=history_revision+1, "
            "updated_at_ms=? WHERE singleton_id=1 RETURNING history_revision",
            (now_ms,),
        )
    ).fetchone()
    return int(row[0])


async def _outbox(
    conn: Any,
    *,
    tenant: str,
    resource_type: str,
    resource_id: str,
    operation: str,
    source_version: int,
    acl_version: int,
    now_ms: int,
) -> None:
    await conn.execute(
        "INSERT OR IGNORE INTO search_outbox "
        "(tenant_id, source_kind, source_id, operation, source_version, acl_version, committed_at_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (tenant, resource_type, resource_id, operation, source_version, acl_version, now_ms),
    )


def _activity(resource_type: str, row: Any, now_ms: int) -> dict[str, Any] | None:
    if resource_type == "workflow_run":
        status, mapped = _status(row["status"])
        return {
            "kind": "workflow_run", "parent_type": "workflow", "parent_id": row["workflow_id"],
            "session_id": None, "status": status,
            "title": f"Workflow run {str(row['id'])[:8]}", "origin": row["trigger"],
            "occurred": _ms(row["started_at"], now_ms),
            "updated": _ms(row["finished_at"], _ms(row["started_at"], now_ms)),
            "complete": mapped,
        }
    if resource_type == "scheduled_run":
        status, mapped = _status(row["status"])
        return {
            "kind": "scheduled_run", "parent_type": "scheduled_task", "parent_id": row["task_id"],
            "session_id": row["session_id"], "status": status,
            "title": f"Scheduled run {str(row['id'])[:8]}", "origin": row["trigger"],
            "occurred": _ms(row["started_at"], now_ms),
            "updated": _ms(row["finished_at"], _ms(row["started_at"], now_ms)),
            "complete": mapped,
        }
    if resource_type == "event_delivery":
        status, mapped = _status(row["status"])
        return {
            "kind": "event_delivery", "parent_type": "event", "parent_id": row["event_id"],
            "session_id": row["session_id"], "status": status,
            "title": f"Event delivery {str(row['id'])[:8]}", "origin": row["source"],
            "occurred": _ms(row["started_at"], now_ms),
            "updated": _ms(row["finished_at"], _ms(row["started_at"], now_ms)),
            "complete": mapped,
        }
    return None


async def _upsert_activity(
    conn: Any,
    *,
    tenant: str,
    resource_type: str,
    resource_id: str,
    source_version: int,
    owner: str | None,
    visibility: str,
    acl_version: int,
    activity: dict[str, Any],
    now_ms: int,
) -> None:
    activity_id = f"activity:{resource_type}:{resource_id}"
    existing = await (
        await conn.execute("SELECT created_revision FROM activity_items WHERE activity_id=?", (activity_id,))
    ).fetchone()
    revision = await _allocate_revision(conn, now_ms)
    created_revision = int(existing[0]) if existing else revision
    await conn.execute(
        "INSERT INTO activity_items "
        "(activity_id, kind, resource_type, resource_id, parent_type, parent_id, session_id, "
        "tenant_id, owner_principal_id, visibility, acl_version, status, title, origin, "
        "occurred_at_ms, updated_at_ms, source_version, created_revision, updated_revision) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(activity_id) DO UPDATE SET owner_principal_id=excluded.owner_principal_id, "
        "visibility=excluded.visibility, acl_version=excluded.acl_version, status=excluded.status, "
        "title=excluded.title, origin=excluded.origin, session_id=excluded.session_id, "
        "updated_at_ms=excluded.updated_at_ms, source_version=excluded.source_version, "
        "updated_revision=excluded.updated_revision, deleted_revision=NULL, deleted_at_ms=NULL",
        (
            activity_id, activity["kind"], resource_type, resource_id,
            activity["parent_type"], activity["parent_id"], activity["session_id"], tenant,
            owner, visibility, acl_version, activity["status"], activity["title"], activity["origin"],
            activity["occurred"], max(activity["occurred"], activity["updated"]), source_version,
            created_revision, revision,
        ),
    )


async def _project_one(
    conn: Any,
    *,
    tenant: str,
    resource_type: str,
    resource_id: str,
    effective_now: int,
) -> int:
    table = dict(_RESOURCE_TABLES)[resource_type]
    row = await (
        await conn.execute(f"SELECT * FROM {table} WHERE id=?", (resource_id,))
    ).fetchone()
    ledger = await (
        await conn.execute(
            "SELECT source_hash, source_version FROM operational_automation_projection "
            "WHERE resource_type=? AND resource_id=?",
            (resource_type, resource_id),
        )
    ).fetchone()
    if row is None:
        if ledger is None:
            return 0
        _owner_id, _visibility, acl_version, _provenance = await _owner(
            conn, tenant, resource_type, resource_id
        )
        revision = await _allocate_revision(conn, effective_now)
        await conn.execute(
            "UPDATE activity_items SET deleted_revision=?, deleted_at_ms=?, updated_revision=?, "
            "updated_at_ms=MAX(updated_at_ms, ?) WHERE resource_type=? AND resource_id=? "
            "AND deleted_at_ms IS NULL",
            (revision, effective_now, revision, effective_now, resource_type, resource_id),
        )
        await _outbox(
            conn,
            tenant=tenant,
            resource_type=resource_type,
            resource_id=resource_id,
            operation="delete",
            source_version=int(ledger[1]) + 1,
            acl_version=acl_version,
            now_ms=effective_now,
        )
        await conn.execute(
            "DELETE FROM operational_automation_projection "
            "WHERE resource_type=? AND resource_id=?",
            (resource_type, resource_id),
        )
        return 1

    owner, visibility, acl_version, provenance = await _effective_owner(
        conn, tenant, resource_type, resource_id, row
    )
    # ACL changes must invalidate an otherwise identical legacy row.
    source_hash = hashlib.sha256(
        "\x00".join(
            (
                _hash_row(row),
                owner or "",
                visibility,
                str(acl_version),
                provenance,
            )
        ).encode()
    ).hexdigest()
    if ledger is not None and str(ledger[0]) == source_hash:
        return 0
    source_version = int(ledger[1]) + 1 if ledger else 1
    activity = _activity(resource_type, row, effective_now)
    if activity is not None:
        await _upsert_activity(
            conn,
            tenant=tenant,
            resource_type=resource_type,
            resource_id=resource_id,
            source_version=source_version,
            owner=owner,
            visibility=visibility,
            acl_version=acl_version,
            activity=activity,
            now_ms=effective_now,
        )
    await _outbox(
        conn,
        tenant=tenant,
        resource_type=resource_type,
        resource_id=resource_id,
        operation="upsert",
        source_version=source_version,
        acl_version=acl_version,
        now_ms=effective_now,
    )
    await conn.execute(
        "INSERT INTO operational_automation_projection "
        "(resource_type, resource_id, source_hash, source_version, projected_at_ms) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(resource_type, resource_id) DO UPDATE SET "
        "source_hash=excluded.source_hash, source_version=excluded.source_version, "
        "projected_at_ms=excluded.projected_at_ms",
        (resource_type, resource_id, source_hash, source_version, effective_now),
    )
    return 1


async def project_automation_pending(
    conn: Any,
    *,
    limit: int = 1000,
    now_ms: int | None = None,
) -> tuple[int, bool]:
    """Project one bounded page from the trigger-driven legacy journal."""

    effective_now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    tenant = await _tenant_id(conn)
    entries = await (
        await conn.execute(
            "SELECT seq, resource_type, resource_id FROM operational_automation_changes "
            "WHERE processed_at_ms IS NULL ORDER BY seq LIMIT ?",
            (max(1, min(int(limit), 10_000)),),
        )
    ).fetchall()
    changed = 0
    processed: list[int] = []
    # Coalesce repeated writes inside the page, while still acknowledging each
    # journal row only after the canonical projection succeeds.
    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (str(entry[1]), str(entry[2]))
        processed.append(int(entry[0]))
        if key not in seen:
            keys.append(key)
            seen.add(key)
    try:
        for resource_type, resource_id in keys:
            changed += await _project_one(
                conn,
                tenant=tenant,
                resource_type=resource_type,
                resource_id=resource_id,
                effective_now=effective_now,
            )
    except Exception as exc:
        if processed:
            await conn.execute(
                f"UPDATE operational_automation_changes SET last_error_class=? "
                f"WHERE seq IN ({','.join('?' for _ in processed)})",
                (type(exc).__name__[:200], *processed),
            )
        raise
    if processed:
        await conn.execute(
            f"UPDATE operational_automation_changes SET processed_at_ms=?, last_error_class=NULL "
            f"WHERE seq IN ({','.join('?' for _ in processed)})",
            (effective_now, *processed),
        )
    remaining = int(
        (
            await (
                await conn.execute(
                    "SELECT COUNT(*) FROM operational_automation_changes "
                    "WHERE processed_at_ms IS NULL"
                )
            ).fetchone()
        )[0]
    )
    return changed, remaining == 0


async def project_automation(conn: Any, *, now_ms: int | None = None) -> int:
    """Drain the incremental automation journal to a complete boundary."""

    changed = 0
    while True:
        page_changed, complete = await project_automation_pending(
            conn, limit=1000, now_ms=now_ms
        )
        changed += page_changed
        if complete:
            return changed
        await asyncio.sleep(0)
