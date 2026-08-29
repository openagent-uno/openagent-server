"""Durable content-addressed storage for chat attachments.

The operational-storage v2 schema already defines ``artifacts`` and
``artifact_links`` as the canonical metadata layer.  This module supplies the
missing byte repository and the small compatibility envelope used by the
stream protocol.  Bytes are stored once by SHA-256; links carry the
conversation-specific filename/order and are what grant access through a
session.

``AttachmentRef`` deliberately remains a plain JSON dictionary on the wire.
Old clients continue to consume ``type/path/filename`` while new clients can
prefer ``artifact_id`` + the ACL-checked ``url``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import aiosqlite


class ArtifactError(RuntimeError):
    """Base error for the durable attachment repository."""


class AttachmentTooLarge(ArtifactError):
    """Raised before an oversized attachment is copied into the CAS."""

    def __init__(self, size_bytes: int, limit_bytes: int) -> None:
        super().__init__(
            f"attachment is too large ({size_bytes} bytes; limit {limit_bytes} bytes)"
        )
        self.size_bytes = int(size_bytes)
        self.limit_bytes = int(limit_bytes)


class ArtifactNotFound(ArtifactError):
    """The artifact row or its CAS bytes no longer exist."""


class ArtifactIntegrityError(ArtifactError):
    """Canonical metadata exists but its immutable bytes were corrupted."""


class UntrustedAttachmentPath(ArtifactError):
    """A remote principal attempted to make the server read a local path."""


_SAFE_NAME_RE = re.compile(r"[\x00-\x1f\x7f]+")
_KIND_VALUES = frozenset({"image", "file", "voice", "video"})


def attachment_limit_bytes(*, direction: str = "input") -> int:
    """Configured attachment cap, shared by uploads and chat bridges.

    Input defaults to 64 MiB: enough for ordinary documents/media without
    letting one platform update make the model pipeline read hundreds of MiB
    into memory.  Output keeps the historical 256 MiB file-serving ceiling.
    A value of 0 disables the cap for operators who explicitly need it.
    """

    env = (
        "OPENAGENT_MAX_INBOUND_ATTACHMENT_MB"
        if direction == "input"
        else "OPENAGENT_MAX_ATTACHMENT_MB"
    )
    default = 64 if direction == "input" else 256
    try:
        mb = int(os.environ.get(env, str(default)) or str(default))
    except ValueError:
        mb = default
    return max(0, mb) * 1024 * 1024


def safe_attachment_filename(value: Any, *, fallback: str = "attachment") -> str:
    """Return a basename safe to join below a staging directory."""

    raw = str(value or "").replace("\\", "/")
    name = raw.rsplit("/", 1)[-1]
    name = _SAFE_NAME_RE.sub("", name).strip().strip(".")
    if not name:
        name = fallback
    # Filesystems generally cap one component at 255 *bytes*, not Unicode
    # code points.  Keep a little headroom and preserve the suffix while
    # trimming on UTF-8 boundaries.  Character-count truncation lets a name
    # made of emoji/CJK exceed NAME_MAX even when ``len(name)`` is below 240.
    suffix = Path(name).suffix[:24]
    suffix_bytes = suffix.encode("utf-8")
    max_bytes = 240
    encoded = name.encode("utf-8")
    if len(encoded) > max_bytes:
        stem_budget = max(1, max_bytes - len(suffix_bytes))
        stem = name[:-len(suffix)] if suffix else name
        stem_bytes = stem.encode("utf-8")[:stem_budget]
        while stem_bytes:
            try:
                stem = stem_bytes.decode("utf-8")
                break
            except UnicodeDecodeError:
                stem_bytes = stem_bytes[:-1]
        else:
            stem = fallback[:1] or "a"
        name = stem + suffix
    return name


def safe_attachment_staging_name(prefix: Any, filename: Any) -> str:
    """Build one NAME_MAX-safe private staging basename.

    Channel bridges prepend platform ids so duplicate user-facing filenames
    cannot overwrite each other.  The combined component must be sanitised a
    second time: sanitising each half independently can still yield a 480-byte
    path component.
    """

    safe_prefix = safe_attachment_filename(prefix, fallback="attachment-id")
    safe_name = safe_attachment_filename(filename)
    return safe_attachment_filename(f"{safe_prefix}-{safe_name}")


def _sniff_mime(path: Path) -> str | None:
    try:
        with path.open("rb") as fh:
            head = fh.read(4096)
    except OSError:
        return None
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return "image/webp"
    if head.startswith(b"%PDF-"):
        return "application/pdf"
    if head.startswith(b"PK\x03\x04"):
        return "application/zip"
    stripped = head.lstrip()
    if stripped.startswith((b"{", b"[")):
        try:
            json.loads(stripped.decode("utf-8"))
            return "application/json"
        except Exception:
            pass
    return None


def infer_attachment_mime(
    path: str | Path,
    *,
    filename: str | None = None,
    declared: str | None = None,
) -> str:
    """Prefer a useful declared MIME, then extension and conservative sniff."""

    declared_clean = str(declared or "").split(";", 1)[0].strip().lower()
    guessed = mimetypes.guess_type(filename or str(path))[0]
    sniffed = _sniff_mime(Path(path))
    # ``application/octet-stream`` conveys no information; a magic-byte or
    # extension result is strictly better.  For image-as-document Telegram
    # updates, a declared image MIME correctly upgrades the kind below.
    if declared_clean and declared_clean != "application/octet-stream":
        return declared_clean
    return sniffed or guessed or "application/octet-stream"


def attachment_kind(mime: str | None, declared_kind: str | None = None) -> str:
    """Classify by MIME, retaining a valid explicit kind as fallback."""

    value = str(mime or "").lower()
    if value.startswith("image/"):
        return "image"
    if value.startswith("video/"):
        return "video"
    if value.startswith("audio/"):
        return "voice"
    explicit = str(declared_kind or "").lower()
    return explicit if explicit in _KIND_VALUES else "file"


def _db_path(db: Any) -> str:
    value = str(getattr(db, "db_path", "") or "")
    if not value:
        raise ArtifactError("artifact repository requires a MemoryDB path")
    return value


def artifact_store_root(db: Any) -> Path:
    raw = _db_path(db)
    if raw == ":memory:" or raw.startswith("file::memory:"):
        from src.core.paths import data_dir

        root = data_dir() / "artifacts"
    else:
        root = Path(raw).expanduser().resolve().parent / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


@asynccontextmanager
async def _connection(db: Any):
    """Use a dedicated WAL writer on disk; share the sole in-memory handle."""

    raw = _db_path(db)
    memory = raw == ":memory:" or raw.startswith("file::memory:")
    if memory:
        conn = await db._ensure_connected()
        yield conn
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


async def _tenant_id(conn: aiosqlite.Connection) -> str:
    try:
        row = await (await conn.execute("SELECT network_id FROM network LIMIT 1")).fetchone()
        if row is not None and row[0] is not None and str(row[0]).strip():
            return str(row[0]).strip()
    except Exception:
        pass
    row = await (
        await conn.execute(
            "SELECT db_instance_id FROM operational_storage_state WHERE singleton_id=1"
        )
    ).fetchone()
    if row is None:
        raise ArtifactError("operational artifact tables are unavailable")
    return f"installation:{row[0]}"


def _principal_fields(principal: Any) -> tuple[str | None, str | None, str | None]:
    if principal is None:
        return None, None, None
    getter = principal.get if isinstance(principal, Mapping) else lambda k, d=None: getattr(principal, k, d)
    tenant = str(getter("tenant_id", "") or "").strip() or None
    principal_id = str(getter("principal_id", "") or "").strip() or None
    handle = str(getter("handle", "") or "").strip() or None
    if principal_id is None and handle:
        principal_type = str(getter("principal_type", "user") or "user").strip()
        if principal_type not in {"user", "agent"}:
            principal_type = "user"
        principal_id = f"{principal_type}:{handle}"
    return tenant, principal_id, handle


def _access_context(principal: Any) -> Any | None:
    """Normalize request and server-bound identities for ACL evaluation."""

    if principal is None:
        return None
    from src.memory.operational.access import AccessContext

    if isinstance(principal, AccessContext):
        return principal
    try:
        return AccessContext.from_on_behalf_identity(principal)
    except PermissionError:
        return None


async def _ownership(
    conn: aiosqlite.Connection,
    *,
    session_id: str,
    principal: Any,
    direction: str,
) -> dict[str, Any]:
    row = None
    if session_id:
        row = await (
            await conn.execute(
                "SELECT tenant_id, owner_principal_id, owner_handle_snapshot, "
                "visibility, acl_version FROM sessions_v2 WHERE id=? "
                "AND deleted_at_ms IS NULL",
                (session_id,),
            )
        ).fetchone()
    p_tenant, p_id, p_handle = _principal_fields(principal)
    access = _access_context(principal)
    if row is not None and p_tenant is not None and str(row[0]) != p_tenant:
        raise ArtifactNotFound(session_id)
    if row is not None and access is not None:
        from src.memory.operational.access import resource_is_visible

        session_acl = {
            "tenant_id": row[0],
            "owner_principal_id": row[1],
            "owner_handle_snapshot": row[2],
            "visibility": row[3],
            "acl_version": row[4],
            "resource_type": "session",
            "resource_id": session_id,
        }
        if not await resource_is_visible(conn, session_acl, access):
            # A caller-controlled session id is never an ACL grant.
            raise ArtifactNotFound(session_id)
    tenant = str(row[0]) if row is not None else (p_tenant or await _tenant_id(conn))
    owner = str(row[1]) if row is not None and row[1] else p_id
    handle = str(row[2]) if row is not None and row[2] else p_handle
    visibility = str(row[3]) if row is not None else "private"
    acl_version = int(row[4]) if row is not None else 1
    if not owner:
        # Internal model output still needs a private canonical owner.  Access
        # through a linked session is checked separately by the REST handler.
        owner = "agent:openagent" if direction == "output" else "user:local"
    if visibility == "quarantined":
        visibility = "private"
    return {
        "tenant_id": tenant,
        "owner_principal_id": owner,
        "owner_handle_snapshot": handle,
        "visibility": visibility,
        "acl_version": acl_version,
        "session_exists": row is not None,
    }


def _copy_into_cas(source: Path, destination: Path, limit_bytes: int) -> tuple[str, int]:
    source = source.expanduser().resolve(strict=True)
    if not source.is_file():
        raise ArtifactNotFound(str(source))
    stated_size = int(source.stat().st_size)
    if limit_bytes and stated_size > limit_bytes:
        raise AttachmentTooLarge(stated_size, limit_bytes)

    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    copied = 0
    fd, temp_name = tempfile.mkstemp(prefix=".staged-", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as inp:
            while True:
                chunk = inp.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if limit_bytes and copied > limit_bytes:
                    raise AttachmentTooLarge(copied, limit_bytes)
                digest.update(chunk)
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        sha = digest.hexdigest()
        final = destination.parent.parent / sha[:2] / sha
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists() and _path_matches_digest(final, sha, copied):
            os.unlink(temp_name)
        else:
            os.replace(temp_name, final)
            with suppress(OSError):
                os.chmod(final, 0o600)
        return sha, copied
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise

def _cas_path(root: Path, sha256: str) -> Path:
    return root / "sha256" / sha256[:2] / sha256


def _path_matches_digest(path: Path, sha256: str, size_bytes: int) -> bool:
    """Verify both CAS address and length without trusting file metadata."""

    try:
        if not path.is_file() or int(path.stat().st_size) != int(size_bytes):
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest() == sha256
    except OSError:
        return False


async def _ensure_artifact_row(
    conn: aiosqlite.Connection,
    *,
    ownership: Mapping[str, Any],
    sha256: str,
    size_bytes: int,
    mime: str,
    filename: str,
    direction: str,
    kind: str,
) -> aiosqlite.Row:
    tenant = str(ownership["tenant_id"])
    row = await (
        await conn.execute(
            "SELECT * FROM artifacts WHERE tenant_id=? AND sha256=? "
            "AND deleted_at_ms IS NULL AND storage_state='available' "
            "ORDER BY created_at_ms LIMIT 1",
            (tenant, sha256),
        )
    ).fetchone()
    if row is not None:
        return row
    artifact_id = f"art_{uuid4().hex}"
    storage_key = f"sha256/{sha256[:2]}/{sha256}"
    now = int(time.time() * 1000)
    try:
        await conn.execute(
            "INSERT INTO artifacts "
            "(id, tenant_id, owner_principal_id, owner_handle_snapshot, visibility, "
            "acl_version, direction, kind, mime, original_filename, storage_key, "
            "sha256, size_bytes, storage_state, metadata_json, retention_class, "
            "ref_count, created_at_ms, updated_at_ms, deleted_at_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'available', '{}', "
            "'session', 0, ?, ?, NULL)",
            (
                artifact_id,
                tenant,
                ownership["owner_principal_id"],
                ownership["owner_handle_snapshot"],
                # Artifact visibility never snapshots the visibility of the
                # session that first referenced the bytes.  The canonical row
                # is owner-private; every other reader is authorized through
                # a live artifact_link and the linked resource's *current*
                # ACL.  Otherwise public -> private and ACL revocation on a
                # conversation would leave the attachment publicly readable.
                "private",
                1,
                direction,
                kind,
                mime,
                filename,
                storage_key,
                sha256,
                int(size_bytes),
                now,
                now,
            ),
        )
    except aiosqlite.IntegrityError:
        # Another turn ingested the same bytes while the copy ran.
        pass
    row = await (
        await conn.execute(
            "SELECT * FROM artifacts WHERE tenant_id=? AND storage_key=? "
            "AND deleted_at_ms IS NULL",
            (tenant, storage_key),
        )
    ).fetchone()
    if row is None:
        raise ArtifactError("artifact metadata could not be materialized")
    return row


async def _ensure_link(
    conn: aiosqlite.Connection,
    *,
    artifact: aiosqlite.Row,
    session_id: str,
    relation: str,
    display_name: str,
    existing_link_id: str | None,
) -> str | None:
    if not session_id:
        return None
    # ``artifact_links`` is deliberately polymorphic and has no FK to its
    # parent resource.  Keep the session link even when normalized projection
    # is a few milliseconds behind attachment ingestion: it grants nothing
    # until a same-tenant, non-deleted session actually exists, and then tracks
    # that session's current visibility/ACL.  This also lets a second uploader
    # reuse tenant-deduplicated bytes without a persistent artifact-level ACL.
    if existing_link_id:
        row = await (
            await conn.execute(
                "SELECT id FROM artifact_links WHERE id=? AND tenant_id=? "
                "AND artifact_id=? AND resource_type='session' AND resource_id=? "
                "AND relation=?",
                (
                    existing_link_id,
                    artifact["tenant_id"],
                    artifact["id"],
                    session_id,
                    relation,
                ),
            )
        ).fetchone()
        if row is not None:
            return str(row[0])
    row = await (
        await conn.execute(
            "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM artifact_links "
            "WHERE tenant_id=? AND resource_type='session' AND resource_id=? "
            "AND relation=?",
            (artifact["tenant_id"], session_id, relation),
        )
    ).fetchone()
    ordinal = int(row[0] if row is not None else 0)
    link_id = f"alink_{uuid4().hex}"
    await conn.execute(
        "INSERT INTO artifact_links "
        "(id, tenant_id, artifact_id, resource_type, resource_id, relation, "
        "ordinal, display_name, metadata_json, created_at_ms) "
        "VALUES (?, ?, ?, 'session', ?, ?, ?, ?, '{}', ?)",
        (
            link_id,
            artifact["tenant_id"],
            artifact["id"],
            session_id,
            relation,
            ordinal,
            display_name,
            int(time.time() * 1000),
        ),
    )
    return link_id


async def _artifact_visible_on_connection(
    conn: aiosqlite.Connection,
    artifact: Mapping[str, Any],
    access: Any,
) -> bool:
    visible, _display_name = await _artifact_access_on_connection(
        conn, artifact, access,
    )
    return visible


async def _artifact_access_on_connection(
    conn: aiosqlite.Connection,
    artifact: Mapping[str, Any],
    access: Any,
) -> tuple[bool, str | None]:
    """Evaluate owner-private and link-inherited visibility in one txn.

    Direct ``resource_acl`` rows for artifacts are intentionally ignored.  A
    grant belongs on the resource that references the bytes, so changing that
    resource's visibility, ACL version, grant set, or deletion state takes
    effect on the very next content request.  The returned display name comes
    from the authorizing link so tenant-level CAS dedup never leaks the first
    uploader's private filename into another context.
    """

    from src.memory.operational.access import resource_is_visible

    if str(artifact["tenant_id"]) != access.tenant_id:
        return False, None
    owner = str(artifact["owner_principal_id"] or "")
    if owner and owner in access.principal_ids:
        return True, str(artifact["original_filename"] or "") or None
    links = await (
        await conn.execute(
            "SELECT resource_type, resource_id, display_name, created_at_ms, id "
            "FROM artifact_links WHERE tenant_id=? AND artifact_id=? "
            "ORDER BY created_at_ms, id",
            (artifact["tenant_id"], artifact["id"]),
        )
    ).fetchall()
    best: tuple[int, str] | None = None
    for resource_type, resource_id, display_name, _created_at_ms, _link_id in links:
        parent = None
        if resource_type == "session":
            parent = await (
                await conn.execute(
                    "SELECT *, 'session' AS resource_type, id AS resource_id "
                    "FROM sessions_v2 WHERE id=? AND tenant_id=? "
                    "AND deleted_at_ms IS NULL",
                    (resource_id, artifact["tenant_id"]),
                )
            ).fetchone()
        elif resource_type == "message":
            parent = await (
                await conn.execute(
                    "SELECT s.*, 'session' AS resource_type, s.id AS resource_id "
                    "FROM session_messages m JOIN sessions_v2 s ON s.id=m.session_id "
                    "AND s.tenant_id=m.tenant_id WHERE m.id=? AND m.tenant_id=? "
                    "AND s.deleted_at_ms IS NULL",
                    (resource_id, artifact["tenant_id"]),
                )
            ).fetchone()
        elif resource_type == "tool_invocation":
            parent = await (
                await conn.execute(
                    "SELECT s.*, 'session' AS resource_type, s.id AS resource_id "
                    "FROM tool_invocations t JOIN sessions_v2 s ON s.id=t.session_id "
                    "AND s.tenant_id=t.tenant_id WHERE t.id=? AND t.tenant_id=? "
                    "AND s.deleted_at_ms IS NULL",
                    (resource_id, artifact["tenant_id"]),
                )
            ).fetchone()
        if parent is None or not await resource_is_visible(conn, parent, access):
            continue
        parent_owner = str(parent["owner_principal_id"] or "")
        # A link from a resource the caller owns is a better presentation
        # context than a coincidentally-visible public/shared link.  Stable SQL
        # ordering resolves the rare case of several equally strong links.
        priority = 2 if parent_owner in access.principal_ids else 1
        candidate = (priority, str(display_name or ""))
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is not None:
        return True, best[1] or None
    return False, None


def _attachment_ref(
    artifact: Mapping[str, Any],
    *,
    path: Path,
    filename: str,
    kind: str,
    link_id: str | None,
) -> dict[str, Any]:
    ref: dict[str, Any] = {
        # ``type`` is the established app/bridge field; ``kind`` is the typed
        # stream spelling.  Emitting both makes migration additive.
        "type": kind,
        "kind": kind,
        "path": str(path),
        "filename": filename,
        "mime_type": str(artifact["mime"] or "application/octet-stream"),
        "size_bytes": int(artifact["size_bytes"]),
        "sha256": str(artifact["sha256"]),
        "artifact_id": str(artifact["id"]),
        "url": f"/api/artifacts/{artifact['id']}/content",
    }
    if link_id:
        ref["artifact_link_id"] = link_id
    return ref


def public_attachment_ref(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical path-free AttachmentRef safe for wire/storage."""

    allowed = (
        "type", "kind", "filename", "mime_type", "size_bytes", "sha256",
        "artifact_id", "artifact_link_id", "url", "caption",
    )
    return {key: value[key] for key in allowed if value.get(key) is not None}


async def _persist_attachments(
    db: Any,
    attachments: Sequence[Mapping[str, Any]] | None,
    *,
    session_id: str,
    principal: Any,
    direction: str,
    allow_local_paths: bool,
) -> tuple[dict[str, Any], ...]:
    if not attachments:
        return ()
    root = artifact_store_root(db)
    limit = attachment_limit_bytes(direction=direction)
    out: list[dict[str, Any]] = []
    async with _connection(db) as conn:
        ownership = await _ownership(
            conn, session_id=session_id, principal=principal, direction=direction
        )
        # Serialize ordinal allocation and artifact dedup for this small batch.
        dedicated = not (
            _db_path(db) == ":memory:" or _db_path(db).startswith("file::memory:")
        )
        if dedicated:
            await conn.execute("BEGIN IMMEDIATE")
        try:
            for raw in attachments:
                if not isinstance(raw, Mapping):
                    continue
                source_value = str(raw.get("path") or "")
                artifact_id = str(raw.get("artifact_id") or "")
                existing = None
                if artifact_id:
                    existing = await (
                        await conn.execute(
                            "SELECT * FROM artifacts WHERE id=? AND tenant_id=? "
                            "AND deleted_at_ms IS NULL AND storage_state='available'",
                            (artifact_id, ownership["tenant_id"]),
                        )
                    ).fetchone()
                if existing is not None:
                    access = _access_context(principal)
                    if access is not None and not await _artifact_visible_on_connection(
                        conn, existing, access,
                    ):
                        # An opaque artifact id is not a bearer token. Without
                        # this check, guessing an id and linking it to one's own
                        # session would grant read access to another user's CAS
                        # bytes in the same tenant.
                        raise ArtifactNotFound(artifact_id)
                    source = _cas_path(root, str(existing["sha256"]))
                    if not await asyncio.to_thread(
                        _path_matches_digest,
                        source,
                        str(existing["sha256"]),
                        int(existing["size_bytes"]),
                    ):
                        raise ArtifactIntegrityError(str(existing["id"]))
                    filename = safe_attachment_filename(
                        raw.get("filename") or existing["original_filename"]
                    )
                    kind = attachment_kind(
                        str(existing["mime"] or ""),
                        str(raw.get("type") or raw.get("kind") or existing["kind"]),
                    )
                    link_id = await _ensure_link(
                        conn,
                        artifact=existing,
                        session_id=session_id,
                        relation=f"{direction}_attachment",
                        display_name=filename,
                        existing_link_id=str(raw.get("artifact_link_id") or "") or None,
                    )
                    out.append(
                        _attachment_ref(
                            existing,
                            path=source,
                            filename=filename,
                            kind=kind,
                            link_id=link_id,
                        )
                    )
                    continue
                if not source_value:
                    if artifact_id:
                        raise ArtifactNotFound(artifact_id)
                    continue
                if not allow_local_paths:
                    # Authenticated gateway members can upload bytes through
                    # the bounded multipart endpoint, but they cannot name a
                    # path on the *server* host.  Only a verified in-process
                    # bridge/internal staging path crosses this boundary.
                    raise UntrustedAttachmentPath(
                        "local attachment paths require a trusted bridge or upload"
                    )
                source = Path(source_value).expanduser()
                filename = safe_attachment_filename(
                    raw.get("filename") or source.name,
                    fallback=f"attachment-{uuid4().hex[:8]}",
                )
                mime = infer_attachment_mime(
                    source,
                    filename=filename,
                    declared=str(raw.get("mime_type") or raw.get("mime") or "") or None,
                )
                kind = attachment_kind(
                    mime, str(raw.get("type") or raw.get("kind") or "file")
                )
                # The worker determines the digest while copying.  Its dummy
                # destination only supplies the CAS root; final placement is
                # derived from the digest inside the worker.
                dummy = root / "sha256" / "00" / "pending"
                sha256, size_bytes = await asyncio.to_thread(
                    _copy_into_cas, source, dummy, limit
                )
                artifact = await _ensure_artifact_row(
                    conn,
                    ownership=ownership,
                    sha256=sha256,
                    size_bytes=size_bytes,
                    mime=mime,
                    filename=filename,
                    direction=direction,
                    kind=kind,
                )
                link_id = await _ensure_link(
                    conn,
                    artifact=artifact,
                    session_id=session_id,
                    relation=f"{direction}_attachment",
                    display_name=filename,
                    existing_link_id=None,
                )
                out.append(
                    _attachment_ref(
                        artifact,
                        path=_cas_path(root, sha256),
                        filename=filename,
                        kind=kind,
                        link_id=link_id,
                    )
                )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
    return tuple(out)


async def normalize_inbound_attachments(
    db: Any,
    attachments: Sequence[Mapping[str, Any]] | None,
    *,
    session_id: str,
    principal: Any = None,
    allow_local_paths: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Persist inbound refs and return ordered, wire-compatible refs.

    ``allow_local_paths`` is a trust-boundary decision made by the gateway,
    never a client capability.  Ordinary remote clients must upload bytes
    first and subsequently reference the returned ``artifact_id``.
    """

    return await _persist_attachments(
        db,
        attachments,
        session_id=session_id,
        principal=principal,
        direction="input",
        allow_local_paths=allow_local_paths,
    )


async def persist_output_attachments(
    db: Any,
    session_id: str,
    attachments: Sequence[Mapping[str, Any]] | None,
    *,
    principal: Any = None,
) -> tuple[dict[str, Any], ...]:
    """Persist marker/native output paths and return ordered AttachmentRefs."""

    return await _persist_attachments(
        db,
        attachments,
        session_id=session_id,
        principal=principal,
        direction="output",
        allow_local_paths=True,
    )


async def link_attachments_to_latest_message(
    db: Any,
    session_id: str,
    attachments: Sequence[Mapping[str, Any]] | None,
    *,
    role: str,
    principal: Any = None,
    retries: int = 6,
    after_sequence: int = -1,
) -> str | None:
    """Attach durable refs to the latest normalized user/assistant message.

    Ingestion happens before the legacy provider run is projected, so its
    first durable link is session-scoped.  Once the run has been persisted,
    this helper adds the precise message link used by transcript hydration.
    It never trusts a path or accepts an artifact that was not already linked
    to this same session.
    """

    if not attachments or role not in {"user", "assistant"}:
        return None
    async with _connection(db) as conn:
        message = None
        attempts = max(1, min(int(retries), 10))
        for attempt in range(attempts):
            message = await (
                await conn.execute(
                    "SELECT id, tenant_id FROM session_messages WHERE session_id=? "
                    "AND role=? AND visibility='user_visible' AND sequence>? "
                    "ORDER BY sequence DESC, ordinal DESC, created_at_ms DESC LIMIT 1",
                    (session_id, role, int(after_sequence)),
                )
            ).fetchone()
            if message is not None:
                break
            if attempt + 1 < attempts:
                await asyncio.sleep(min(0.02 * (2**attempt), 0.25))
        if message is None:
            return None
        message_id, tenant_id = str(message[0]), str(message[1])
        access = _access_context(principal)
        session = await (
            await conn.execute(
                "SELECT *, 'session' AS resource_type, id AS resource_id "
                "FROM sessions_v2 WHERE id=? AND tenant_id=? AND deleted_at_ms IS NULL",
                (session_id, tenant_id),
            )
        ).fetchone()
        if session is None:
            return None
        if access is not None:
            from src.memory.operational.access import resource_is_visible

            if not await resource_is_visible(conn, session, access):
                return None
        relation = f"{'input' if role == 'user' else 'output'}_attachment"
        for ordinal, raw in enumerate(attachments):
            if not isinstance(raw, Mapping):
                continue
            artifact_id = str(raw.get("artifact_id") or raw.get("artifactId") or "").strip()
            if not artifact_id:
                continue
            artifact = await (
                await conn.execute(
                    "SELECT * FROM artifacts WHERE id=? AND tenant_id=? AND deleted_at_ms IS NULL "
                    "AND storage_state='available'",
                    (artifact_id, tenant_id),
                )
            ).fetchone()
            if artifact is None:
                continue
            if access is not None and not await _artifact_visible_on_connection(
                conn, artifact, access,
            ):
                continue
            inherited = await (
                await conn.execute(
                    "SELECT 1 FROM artifact_links WHERE tenant_id=? AND artifact_id=? "
                    "AND resource_type='session' AND resource_id=? LIMIT 1",
                    (tenant_id, artifact_id, session_id),
                )
            ).fetchone()
            if inherited is None:
                await _ensure_link(
                    conn,
                    artifact=artifact,
                    session_id=session_id,
                    relation=relation,
                    display_name=safe_attachment_filename(
                        raw.get("filename") or artifact["original_filename"]
                    ),
                    existing_link_id=None,
                )
            existing = await (
                await conn.execute(
                    "SELECT artifact_id FROM artifact_links WHERE tenant_id=? "
                    "AND resource_type='message' AND resource_id=? AND relation=? "
                    "AND ordinal=?",
                    (tenant_id, message_id, relation, ordinal),
                )
            ).fetchone()
            if existing is not None:
                # Idempotent retries are expected; a different artifact at the
                # same immutable ordinal is never silently replaced.
                continue
            filename = safe_attachment_filename(
                raw.get("filename") or artifact["original_filename"],
                fallback=f"attachment-{ordinal + 1}",
            )
            metadata = json.dumps(
                {
                    "kind": attachment_kind(
                        str(artifact["mime"] or ""),
                        str(raw.get("type") or raw.get("kind") or artifact["kind"]),
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
                    f"alink_{uuid4().hex}", tenant_id, artifact_id, message_id,
                    relation, ordinal, filename, metadata, int(time.time() * 1000),
                ),
            )
        await conn.commit()
        return message_id


async def latest_message_sequence(db: Any, session_id: str, *, role: str) -> int:
    """Capture a turn boundary so delayed projection cannot link an old row."""

    if role not in {"user", "assistant"}:
        raise ValueError("role must be user or assistant")
    async with _connection(db) as conn:
        row = await (
            await conn.execute(
                "SELECT COALESCE(MAX(sequence), -1) FROM session_messages "
                "WHERE session_id=? AND role=?",
                (session_id, role),
            )
        ).fetchone()
        return int(row[0] if row is not None else -1)


async def attachment_refs_for_messages_on_connection(
    conn: Any,
    message_ids: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Hydrate public, path-free AttachmentRefs for normalized messages."""

    identifiers = [str(value) for value in message_ids if str(value)]
    if not identifiers:
        return {}
    output: dict[str, list[dict[str, Any]]] = {}
    for start in range(0, len(identifiers), 400):
        chunk = identifiers[start : start + 400]
        placeholders = ",".join("?" for _ in chunk)
        rows = await (
            await conn.execute(
                "SELECT l.resource_id AS message_id, l.id AS artifact_link_id, "
                "l.ordinal, l.display_name, l.metadata_json, a.id AS artifact_id, "
                "a.kind, a.mime, a.original_filename, a.sha256, a.size_bytes "
                "FROM artifact_links l JOIN artifacts a ON a.id=l.artifact_id "
                "AND a.tenant_id=l.tenant_id WHERE l.resource_type='message' "
                f"AND l.resource_id IN ({placeholders}) AND a.deleted_at_ms IS NULL "
                "AND a.storage_state='available' ORDER BY l.resource_id, l.relation, l.ordinal",
                tuple(chunk),
            )
        ).fetchall()
        for row in rows:
            try:
                meta = json.loads(str(row["metadata_json"] or "{}"))
            except (TypeError, ValueError):
                meta = {}
            kind = attachment_kind(
                str(row["mime"] or ""),
                str(meta.get("kind") or row["kind"] or "file"),
            )
            artifact_id = str(row["artifact_id"])
            output.setdefault(str(row["message_id"]), []).append(
                {
                    "type": kind,
                    "kind": kind,
                    "filename": safe_attachment_filename(
                        row["display_name"] or row["original_filename"]
                    ),
                    "mime_type": str(row["mime"] or "application/octet-stream"),
                    "size_bytes": int(row["size_bytes"]),
                    "sha256": str(row["sha256"]),
                    "artifact_id": artifact_id,
                    "artifact_link_id": str(row["artifact_link_id"]),
                    "url": f"/api/artifacts/{artifact_id}/content",
                }
            )
    return output


async def artifact_for_legacy_path(
    db: Any,
    requested_path: str | Path,
    access: Any,
) -> tuple[dict[str, Any], Path]:
    """Resolve a deprecated path URL to one visible canonical CAS artifact.

    Old clients sent ``AttachmentRef.path`` back through ``GET /api/files``.
    A path must never be treated as authorization: only a real path resolving
    to this agent's exact CAS layout is considered, and the matching artifact
    is re-authorized through its current owner/resource links.  Arbitrary
    agent-host files therefore fail closed even for an authenticated member.
    """

    normalized_access = _access_context(access)
    if normalized_access is None:
        raise ArtifactNotFound(str(requested_path))
    try:
        path = Path(requested_path).expanduser().resolve(strict=True)
        root = artifact_store_root(db).resolve(strict=True)
        relative = path.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        raise ArtifactNotFound(str(requested_path)) from None
    parts = relative.parts
    if len(parts) != 3 or parts[0] != "sha256":
        raise ArtifactNotFound(str(requested_path))
    digest = parts[2].lower()
    if (
        len(digest) != 64
        or parts[1].lower() != digest[:2]
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise ArtifactNotFound(str(requested_path))
    storage_key = f"sha256/{digest[:2]}/{digest}"
    async with _connection(db) as conn:
        artifact = await (
            await conn.execute(
                "SELECT * FROM artifacts WHERE tenant_id=? AND storage_key=? "
                "AND sha256=? AND deleted_at_ms IS NULL "
                "AND storage_state='available'",
                (normalized_access.tenant_id, storage_key, digest),
            )
        ).fetchone()
        if artifact is None:
            raise ArtifactNotFound(str(requested_path))
        visible, display_name = await _artifact_access_on_connection(
            conn, artifact, normalized_access,
        )
        if not visible:
            raise ArtifactNotFound(str(requested_path))
        result = dict(artifact)
        result["authorized_filename"] = safe_attachment_filename(
            display_name or result["original_filename"] or result["id"]
        )
    canonical = _cas_path(root, digest)
    if path != canonical or not await asyncio.to_thread(
        _path_matches_digest,
        canonical,
        digest,
        int(result["size_bytes"]),
    ):
        raise ArtifactIntegrityError(str(result["id"]))
    return result, canonical


async def artifact_authorized_row(
    db: Any,
    artifact_id: str,
    access: Any,
) -> tuple[dict[str, Any], Path]:
    """Return ACL-checked metadata, presentation filename, and CAS path."""

    normalized_access = _access_context(access)
    if normalized_access is None:
        raise ArtifactNotFound(artifact_id)
    async with _connection(db) as conn:
        artifact = await (
            await conn.execute(
                "SELECT * FROM artifacts WHERE id=? AND deleted_at_ms IS NULL "
                "AND storage_state='available'",
                (artifact_id,),
            )
        ).fetchone()
        if artifact is None:
            raise ArtifactNotFound(artifact_id)
        visible, display_name = await _artifact_access_on_connection(
            conn, artifact, normalized_access,
        )
        if not visible:
            raise ArtifactNotFound(artifact_id)
        result = dict(artifact)
        result["authorized_filename"] = safe_attachment_filename(
            display_name or result["original_filename"] or artifact_id
        )
    path = _cas_path(artifact_store_root(db), str(result["sha256"]))
    if not await asyncio.to_thread(
        _path_matches_digest,
        path,
        str(result["sha256"]),
        int(result["size_bytes"]),
    ):
        raise ArtifactIntegrityError(artifact_id)
    return result, path


async def artifact_row(db: Any, artifact_id: str) -> tuple[dict[str, Any], Path]:
    """Return canonical metadata and CAS path, failing closed on missing bytes."""

    async with _connection(db) as conn:
        row = await (
            await conn.execute(
                "SELECT * FROM artifacts WHERE id=? AND deleted_at_ms IS NULL "
                "AND storage_state='available'",
                (artifact_id,),
            )
        ).fetchone()
        if row is None:
            raise ArtifactNotFound(artifact_id)
        result = dict(row)
    path = _cas_path(artifact_store_root(db), str(result["sha256"]))
    if not await asyncio.to_thread(
        _path_matches_digest,
        path,
        str(result["sha256"]),
        int(result["size_bytes"]),
    ):
        raise ArtifactIntegrityError(artifact_id)
    return result, path


async def artifact_is_visible(db: Any, artifact_id: str, access: Any) -> bool:
    """Authorize an artifact directly or through one visible linked resource."""

    async with _connection(db) as conn:
        artifact = await (
            await conn.execute(
                "SELECT * FROM artifacts WHERE id=? AND deleted_at_ms IS NULL "
                "AND storage_state='available'",
                (artifact_id,),
            )
        ).fetchone()
        if artifact is None:
            return False
        return await _artifact_visible_on_connection(conn, artifact, access)


__all__ = [
    "ArtifactError",
    "ArtifactIntegrityError",
    "ArtifactNotFound",
    "AttachmentTooLarge",
    "UntrustedAttachmentPath",
    "artifact_authorized_row",
    "artifact_for_legacy_path",
    "artifact_is_visible",
    "artifact_row",
    "artifact_store_root",
    "attachment_kind",
    "attachment_limit_bytes",
    "attachment_refs_for_messages_on_connection",
    "infer_attachment_mime",
    "latest_message_sequence",
    "link_attachments_to_latest_message",
    "normalize_inbound_attachments",
    "persist_output_attachments",
    "public_attachment_ref",
    "safe_attachment_filename",
    "safe_attachment_staging_name",
]
