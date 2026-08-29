"""Canonical ordered content for normalized conversation messages.

New turns write text, AttachmentRefs and revision-pinned Custom Views to the
``session_message_parts`` table.  The provider/session JSON and response
markers remain readable during beta, but are never the source of truth for a
new message once a canonical part set exists.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import aiosqlite


MAX_MESSAGE_PARTS = 512
MAX_TEXT_PART_BYTES = 2 * 1024 * 1024
PROJECTION_WAIT_SECONDS = 5.0


class MessagePartsError(RuntimeError):
    pass


def _db_path(db: Any) -> str:
    value = str(getattr(db, "db_path", "") or "")
    if not value:
        raise MessagePartsError("message parts require a canonical database")
    return value


@asynccontextmanager
async def _connection(db: Any):
    raw = _db_path(db)
    if raw == ":memory:" or raw.startswith("file::memory:"):
        yield await db._ensure_connected()
        return
    conn = await aiosqlite.connect(raw, timeout=60.0)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA busy_timeout=60000")
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA synchronous=FULL")
    await conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        await conn.close()


def _access_context(principal: Any) -> Any | None:
    if principal is None:
        return None
    from src.memory.operational.access import AccessContext

    if isinstance(principal, AccessContext):
        return principal
    try:
        return AccessContext.from_on_behalf_identity(principal)
    except PermissionError:
        return None


async def _wait_for_message(
    conn: Any,
    session_id: str,
    role: str,
    after_sequence: int,
    *,
    timeout_seconds: float,
) -> Any | None:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    delay = 0.02
    while True:
        row = await (
            await conn.execute(
                "SELECT id, tenant_id, sequence FROM session_messages "
                "WHERE session_id=? AND role=? AND visibility='user_visible' "
                "AND sequence>? ORDER BY sequence DESC, ordinal DESC, "
                "created_at_ms DESC LIMIT 1",
                (session_id, role, int(after_sequence)),
            )
        ).fetchone()
        if row is not None:
            return row
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(delay, remaining))
        delay = min(delay * 1.7, 0.25)


async def _assert_session_visible(
    conn: Any,
    *,
    session_id: str,
    tenant_id: str,
    principal: Any,
) -> None:
    session = await (
        await conn.execute(
            "SELECT *, id AS resource_id, 'session' AS resource_type "
            "FROM sessions_v2 WHERE id=? AND tenant_id=? AND deleted_at_ms IS NULL",
            (session_id, tenant_id),
        )
    ).fetchone()
    if session is None:
        raise MessagePartsError("normalized session is unavailable")
    access = _access_context(principal)
    if access is not None:
        from src.memory.operational.access import resource_is_visible

        if not await resource_is_visible(conn, session, access):
            raise MessagePartsError("normalized session is unavailable")


async def _attachment_link(
    conn: Any,
    *,
    tenant_id: str,
    session_id: str,
    message_id: str,
    role: str,
    attachment_ordinal: int,
    attachment: Mapping[str, Any],
) -> str | None:
    from src.memory.artifacts import (
        attachment_kind,
        safe_attachment_filename,
    )

    relation = f"{'input' if role == 'user' else 'output'}_attachment"
    existing = await (
        await conn.execute(
            "SELECT id, artifact_id FROM artifact_links WHERE tenant_id=? "
            "AND resource_type='message' AND resource_id=? AND relation=? "
            "AND ordinal=?",
            (tenant_id, message_id, relation, attachment_ordinal),
        )
    ).fetchone()
    artifact_id = str(
        attachment.get("artifact_id") or attachment.get("artifactId") or ""
    ).strip()
    if existing is not None:
        return str(existing[0]) if not artifact_id or str(existing[1]) == artifact_id else None
    if not artifact_id:
        return None
    artifact = await (
        await conn.execute(
            "SELECT * FROM artifacts WHERE id=? AND tenant_id=? "
            "AND deleted_at_ms IS NULL AND storage_state='available'",
            (artifact_id, tenant_id),
        )
    ).fetchone()
    if artifact is None:
        return None
    inherited = await (
        await conn.execute(
            "SELECT 1 FROM artifact_links WHERE tenant_id=? AND artifact_id=? "
            "AND resource_type='session' AND resource_id=? LIMIT 1",
            (tenant_id, artifact_id, session_id),
        )
    ).fetchone()
    if inherited is None:
        return None
    link_id = f"alink_{uuid4().hex}"
    filename = safe_attachment_filename(
        attachment.get("filename") or artifact["original_filename"],
        fallback=f"attachment-{attachment_ordinal + 1}",
    )
    metadata = json.dumps(
        {
            "kind": attachment_kind(
                str(artifact["mime"] or ""),
                str(attachment.get("type") or attachment.get("kind") or artifact["kind"]),
            ),
            "mime_type": str(artifact["mime"] or "application/octet-stream"),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    await conn.execute(
        "INSERT INTO artifact_links "
        "(id, tenant_id, artifact_id, resource_type, resource_id, relation, "
        "ordinal, display_name, metadata_json, created_at_ms) "
        "VALUES (?, ?, ?, 'message', ?, ?, ?, ?, ?, ?)",
        (
            link_id,
            tenant_id,
            artifact_id,
            message_id,
            relation,
            attachment_ordinal,
            filename,
            metadata,
            int(time.time() * 1000),
        ),
    )
    return link_id


async def _ui_link(
    conn: Any,
    *,
    tenant_id: str,
    session_id: str,
    message_id: str,
    view_id: str,
    revision: int,
    principal: Any,
) -> bool:
    row = await (
        await conn.execute(
            "SELECT v.*, v.id AS resource_id, 'ui_view' AS resource_type "
            "FROM ui_views v JOIN ui_view_revisions r ON r.view_id=v.id "
            "AND r.tenant_id=v.tenant_id WHERE v.id=? AND v.tenant_id=? "
            "AND r.revision=? AND v.surface='inline' AND v.session_id=? "
            "AND v.status<>'deleted'",
            (view_id, tenant_id, revision, session_id),
        )
    ).fetchone()
    if row is None:
        return False
    access = _access_context(principal)
    if access is not None:
        from src.memory.operational.access import resource_is_visible

        if not await resource_is_visible(conn, row, access):
            return False
    existing = await (
        await conn.execute(
            "SELECT 1 FROM ui_message_links WHERE tenant_id=? AND view_id=? "
            "AND revision=? AND session_id=? AND message_id=? LIMIT 1",
            (tenant_id, view_id, revision, session_id, message_id),
        )
    ).fetchone()
    if existing is None:
        await conn.execute(
            "INSERT INTO ui_message_links "
            "(id, tenant_id, view_id, revision, session_id, message_id, linked_at_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                uuid4().hex,
                tenant_id,
                view_id,
                revision,
                session_id,
                message_id,
                int(time.time() * 1000),
            ),
        )
    return True


async def persist_parts_for_latest_message(
    db: Any,
    session_id: str,
    *,
    role: str,
    parts: Sequence[Mapping[str, Any]],
    principal: Any = None,
    after_sequence: int = -1,
    timeout_seconds: float = PROJECTION_WAIT_SECONDS,
) -> str | None:
    """Persist one immutable ordered part set after normalized projection.

    Projection is intentionally asynchronous on recovery paths.  This helper
    waits for the first message beyond the captured turn boundary, so a slow
    projection cannot attach a View or file to the preceding response.
    """

    if role not in {"user", "assistant"}:
        raise MessagePartsError("role must be user or assistant")
    if len(parts) > MAX_MESSAGE_PARTS:
        raise MessagePartsError("message has too many content parts")
    async with _connection(db) as conn:
        message = await _wait_for_message(
            conn,
            session_id,
            role,
            after_sequence,
            timeout_seconds=timeout_seconds,
        )
        if message is None:
            return None
        message_id = str(message["id"])
        tenant_id = str(message["tenant_id"])
        await _assert_session_visible(
            conn,
            session_id=session_id,
            tenant_id=tenant_id,
            principal=principal,
        )

        dedicated = not (
            _db_path(db) == ":memory:" or _db_path(db).startswith("file::memory:")
        )
        if dedicated:
            await conn.execute("BEGIN IMMEDIATE")
        rows: list[tuple[str, str | None, str | None, str | None, int | None]] = []
        attachment_ordinal = 0
        try:
            existing_rows = await (
                await conn.execute(
                    "SELECT p.kind, p.text_content, l.artifact_id, "
                    "p.ui_view_id, p.ui_revision FROM session_message_parts p "
                    "LEFT JOIN artifact_links l ON l.id=p.artifact_link_id "
                    "WHERE p.message_id=? ORDER BY p.ordinal",
                    (message_id,),
                )
            ).fetchall()
            if existing_rows:
                # Canonical message content is append-once.  Projection retries
                # may repeat the exact payload, but a later retry must never
                # rewrite history or manufacture new artifact/View links.
                requested: list[
                    tuple[str, str | None, str | None, str | None, int | None]
                ] = []
                for raw in parts:
                    kind = str(raw.get("kind") or raw.get("type") or "")
                    if kind == "text":
                        value = str(raw.get("text") or "")
                        if not value:
                            continue
                        if len(value.encode("utf-8")) > MAX_TEXT_PART_BYTES:
                            raise MessagePartsError("message text part is too large")
                        requested.append(("text", value, None, None, None))
                    elif kind == "attachment":
                        attachment = raw.get("attachment")
                        if not isinstance(attachment, Mapping):
                            continue
                        artifact_id = str(
                            attachment.get("artifact_id")
                            or attachment.get("artifactId")
                            or ""
                        ).strip()
                        if artifact_id:
                            requested.append(
                                ("attachment", None, artifact_id, None, None)
                            )
                    elif kind == "ui_view" and role == "assistant":
                        view_id = str(raw.get("view_id") or raw.get("viewId") or "")
                        try:
                            revision = int(raw.get("revision") or 0)
                        except (TypeError, ValueError):
                            continue
                        if view_id and revision > 0:
                            requested.append(
                                ("ui_view", None, None, view_id, revision)
                            )
                existing = [
                    (
                        str(row["kind"]),
                        str(row["text_content"])
                        if row["text_content"] is not None else None,
                        str(row["artifact_id"])
                        if row["artifact_id"] is not None else None,
                        str(row["ui_view_id"])
                        if row["ui_view_id"] is not None else None,
                        int(row["ui_revision"])
                        if row["ui_revision"] is not None else None,
                    )
                    for row in existing_rows
                ]
                if requested != existing:
                    raise MessagePartsError(
                        "canonical message parts conflict with immutable history"
                    )
                await conn.commit()
                return message_id

            for raw in parts:
                kind = str(raw.get("kind") or raw.get("type") or "")
                if kind == "text":
                    value = str(raw.get("text") or "")
                    if not value:
                        continue
                    if len(value.encode("utf-8")) > MAX_TEXT_PART_BYTES:
                        raise MessagePartsError("message text part is too large")
                    rows.append(("text", value, None, None, None))
                    continue
                if kind == "attachment":
                    attachment = raw.get("attachment")
                    if not isinstance(attachment, Mapping):
                        continue
                    link_id = await _attachment_link(
                        conn,
                        tenant_id=tenant_id,
                        session_id=session_id,
                        message_id=message_id,
                        role=role,
                        attachment_ordinal=attachment_ordinal,
                        attachment=attachment,
                    )
                    attachment_ordinal += 1
                    if link_id is not None:
                        rows.append(("attachment", None, link_id, None, None))
                    continue
                if kind == "ui_view" and role == "assistant":
                    view_id = str(raw.get("view_id") or raw.get("viewId") or "")
                    try:
                        revision = int(raw.get("revision") or 0)
                    except (TypeError, ValueError):
                        continue
                    if view_id and revision > 0 and await _ui_link(
                        conn,
                        tenant_id=tenant_id,
                        session_id=session_id,
                        message_id=message_id,
                        view_id=view_id,
                        revision=revision,
                        principal=principal,
                    ):
                        rows.append(("ui_view", None, None, view_id, revision))

            now_ms = int(time.time() * 1000)
            for ordinal, (kind, text, artifact_link_id, view_id, revision) in enumerate(rows):
                await conn.execute(
                    "INSERT INTO session_message_parts "
                    "(id, tenant_id, session_id, message_id, ordinal, kind, text_content, "
                    "artifact_link_id, ui_view_id, ui_revision, created_at_ms) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"mpart_{uuid4().hex}",
                        tenant_id,
                        session_id,
                        message_id,
                        ordinal,
                        kind,
                        text,
                        artifact_link_id,
                        view_id,
                        revision,
                        now_ms,
                    ),
                )
            await conn.commit()
            return message_id
        except Exception:
            await conn.rollback()
            raise


async def canonical_parts_for_messages_on_connection(
    conn: Any,
    message_ids: Sequence[str],
    *,
    access: Any,
) -> dict[str, list[dict[str, Any]]]:
    """Batch hydrate canonical parts with current artifact/View ACL checks."""

    identifiers = [str(value) for value in message_ids if str(value)]
    if not identifiers:
        return {}
    from src.memory.artifacts import public_attachment_ref, safe_attachment_filename
    from src.memory.operational.access import resource_is_visible

    output: dict[str, list[dict[str, Any]]] = {}
    for start in range(0, len(identifiers), 300):
        chunk = identifiers[start : start + 300]
        placeholders = ",".join("?" for _ in chunk)
        rows = await (
            await conn.execute(
                "SELECT p.*, l.id AS link_id, l.display_name, l.metadata_json, "
                "a.id AS artifact_id, a.kind AS artifact_kind, a.mime, "
                "a.original_filename, a.sha256, a.size_bytes, a.storage_state, "
                "a.deleted_at_ms AS artifact_deleted_at, "
                "v.owner_principal_id AS view_owner_principal_id, "
                "v.owner_handle_snapshot AS view_owner_handle_snapshot, "
                "v.visibility AS view_visibility, v.acl_version AS view_acl_version, "
                "v.title AS view_title, v.status AS view_status, v.frozen AS view_frozen, "
                "v.expires_at_ms AS view_expires_at "
                "FROM session_message_parts p "
                "LEFT JOIN artifact_links l ON l.id=p.artifact_link_id "
                "LEFT JOIN artifacts a ON a.id=l.artifact_id AND a.tenant_id=l.tenant_id "
                "LEFT JOIN ui_views v ON v.id=p.ui_view_id AND v.tenant_id=p.tenant_id "
                f"WHERE p.message_id IN ({placeholders}) "
                "ORDER BY p.message_id, p.ordinal",
                tuple(chunk),
            )
        ).fetchall()
        for row in rows:
            message_id = str(row["message_id"])
            kind = str(row["kind"])
            if kind == "text":
                output.setdefault(message_id, []).append(
                    {"kind": "text", "text": str(row["text_content"] or "")}
                )
                continue
            if kind == "attachment":
                if (
                    row["artifact_id"] is None
                    or row["artifact_deleted_at"] is not None
                    or str(row["storage_state"] or "") != "available"
                ):
                    continue
                # The parent session ACL was checked by the messages endpoint;
                # a message link is therefore sufficient and does not expose an
                # artifact outside that already-authorized conversation.
                try:
                    metadata = json.loads(str(row["metadata_json"] or "{}"))
                except (TypeError, ValueError):
                    metadata = {}
                artifact_id = str(row["artifact_id"])
                ref = public_attachment_ref(
                    {
                        "type": str(metadata.get("kind") or row["artifact_kind"] or "file"),
                        "kind": str(metadata.get("kind") or row["artifact_kind"] or "file"),
                        "filename": safe_attachment_filename(
                            row["display_name"] or row["original_filename"]
                        ),
                        "mime_type": str(row["mime"] or "application/octet-stream"),
                        "size_bytes": int(row["size_bytes"]),
                        "sha256": str(row["sha256"]),
                        "artifact_id": artifact_id,
                        "artifact_link_id": str(row["link_id"]),
                        "url": f"/api/artifacts/{artifact_id}/content",
                    }
                )
                output.setdefault(message_id, []).append(
                    {"kind": "attachment", "attachment": ref}
                )
                continue
            if kind == "ui_view" and row["view_title"] is not None:
                acl_row = {
                    "tenant_id": row["tenant_id"],
                    "owner_principal_id": row["view_owner_principal_id"],
                    "owner_handle_snapshot": row["view_owner_handle_snapshot"],
                    "visibility": row["view_visibility"],
                    "acl_version": row["view_acl_version"],
                    "resource_type": "ui_view",
                    "resource_id": row["ui_view_id"],
                }
                if not await resource_is_visible(conn, acl_row, access):
                    continue
                output.setdefault(message_id, []).append(
                    {
                        "kind": "ui_view",
                        "view_id": str(row["ui_view_id"]),
                        "revision": int(row["ui_revision"]),
                        "title": str(row["view_title"]),
                        "status": "stale" if bool(row["view_frozen"]) else str(row["view_status"]),
                        "expires_at": int(row["view_expires_at"])
                        if row["view_expires_at"] is not None
                        else None,
                    }
                )
    return output


__all__ = [
    "MAX_MESSAGE_PARTS",
    "MessagePartsError",
    "canonical_parts_for_messages_on_connection",
    "persist_parts_for_latest_message",
]
