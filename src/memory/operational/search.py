"""Rebuildable, redacted operational FTS5 index.

The index is intentionally a different SQLite file and code path from both the
Markdown Memory Vault index and any transcript/provider index.  Canonical rows
remain in ``openagent.db``; this module consumes ``search_outbox`` and indexes
only user-visible/redacted projections.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .schema import operational_search_schema_sql


CONSUMER_ID = "operational-fts-v1"
EXTRACTOR_VERSION = "operational-extractor-v2"
REDACTION_VERSION = "fail-closed-redaction-v2"
ACL_PROJECTION_VERSION = "canonical-recheck-v2"
TOKENIZER_VERSION = "unicode61-v1"
_TERM_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SECRET_LABEL = (
    r"api[_-]?key|authorization|secret|token|access[_-]?token|refresh[_-]?token|"
    r"password|passwd|client[_-]?secret|signature|cookies?|set[_-]?cookie|"
    r"headers?|credentials?|private[_-]?key"
)
_LABELED_SECRET_RE = re.compile(
    rf"(?P<prefix>[\"']?(?<![\w-])(?:[A-Za-z0-9_-]*(?:{_SECRET_LABEL})"
    r"[A-Za-z0-9_-]*|sig)[\"']?\s*[:=]\s*)"
    r"(?P<value>(?!\s*\[REDACTED\])(?:\"(?:\\.|[^\"\\])*\"|"
    r"'(?:\\.|[^'\\])*'|[^\r\n,;}\]]+))",
    re.IGNORECASE,
)
_SENSITIVE_CONTAINER_RE = re.compile(
    r"(?P<prefix>[\"']?(?<![\w-])(?:headers?|cookies?)[\"']?\s*[:=]\s*)"
    r"\{[^{}\r\n]{0,16384}\}",
    re.IGNORECASE,
)
_SENSITIVE_HEADER_RE = re.compile(
    r"(?P<prefix>(?<![\w-])(?:authorization|proxy-authorization|cookie|set-cookie|"
    r"x-api-key|x-auth-token)\s*:\s*)[^\r\n]*",
    re.IGNORECASE,
)
_PRIVATE_PATH_RE = re.compile(
    r"(?i)(?:file://)?(?:/Users/[^/\s\"'<>]+|/home/[^/\s\"'<>]+|/root(?:/|\b)|"
    r"[A-Z]:\\Users\\[^\\\s\"'<>]+)(?:[/\\][^\s\"'<>]*)?"
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}")
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_PROVIDER_TOKEN_RE = re.compile(
    r"\b(?:sk-(?:ant-)?[A-Za-z0-9_-]{16,}|gh[opusr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{16,}|AKIA[A-Z0-9]{16})\b"
)
_PEM_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*(?:PRIVATE KEY|CERTIFICATE)-----.*?-----END [A-Z0-9 ]*(?:PRIVATE KEY|CERTIFICATE)-----",
    re.DOTALL,
)
_SIGNED_QUERY_RE = re.compile(
    r"(?i)([?&](?:token|api[_-]?key|key|signature|sig|x-amz-signature|x-goog-signature|access_token)=)[^&#\s]+"
)
_OPAQUE_TOKEN_RE = re.compile(r"(?<![\w])([A-Za-z0-9_+/=-]{32,})(?![\w])")
_SYNC_LOCKS: dict[str, asyncio.Lock] = {}


@dataclass(frozen=True)
class SearchIndexStatus:
    ready: bool
    state: str
    generation: str
    seq: int
    documents: int
    pending: int
    indexed_through_ms: int | None
    path: str


def operational_search_path(canonical_path: str | Path) -> Path:
    source = Path(canonical_path).resolve()
    target = source.with_name(f"{source.stem}.operational-search-v1.db")
    lowered = target.name.lower()
    if target == source or "vault" in lowered or "transcript_index" in lowered:
        raise RuntimeError("operational search path collides with a canonical or vault index")
    return target


def _harden_index_files(path: Path) -> None:
    """Keep the redacted index and transient WAL files owner-only."""

    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if not candidate.exists():
            continue
        try:
            os.chmod(candidate, 0o600)
        except OSError as exc:
            raise RuntimeError("operational search files cannot be made owner-only") from exc
        # POSIX mode bits are authoritative on Friday/Linux and macOS.  Windows
        # chmod only exposes the read-only bit; its effective ACL is inherited
        # from the already-private agent directory instead.
        if os.name != "nt" and candidate.stat().st_mode & 0o077:
            raise RuntimeError("operational search files are not owner-only")


def _open_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Index bootstrap may be dispatched to the default executor while the
    # single owner subsequently consumes outbox rows on the event-loop thread.
    # The connection is never shared concurrently; disabling the affinity
    # check permits that deliberate hand-off.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA trusted_schema=OFF")
    conn.execute("PRAGMA journal_mode=WAL")
    # Rebuildable derived index: NORMAL preserves WAL atomicity without paying
    # canonical-database FULL fsync cost on every outbox batch.
    conn.execute("PRAGMA synchronous=NORMAL")
    exists = conn.execute(
        "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='search_index_state'"
    ).fetchone()
    if exists is None:
        conn.executescript(operational_search_schema_sql())
    _harden_index_files(path)
    # Fail closed when Python's SQLite lacks FTS5.
    conn.execute("SELECT count(*) FROM search_fts WHERE search_fts MATCH 'canary'").fetchone()
    _harden_index_files(path)
    return conn


def _source_fingerprint(instance_id: str, schema_version: int) -> str:
    return hashlib.sha256(f"{instance_id}:{schema_version}".encode()).hexdigest()


def _index_is_incompatible(
    state: sqlite3.Row,
    *,
    instance_id: str,
    schema_version: int,
    fingerprint: str,
    outbox_head: int,
) -> bool:
    """Reject a ready generation that belongs to different canonical bytes.

    ``None`` compatibility fields are valid only for a brand-new, uninitialized
    index. Once a source fingerprint exists, every versioned extractor boundary
    and the monotonic outbox position must agree before any old content is read.
    """

    existing = state["source_fingerprint"]
    if existing is None:
        return False
    expected = {
        "source_db_instance_id": instance_id,
        "source_schema_version": schema_version,
        "source_fingerprint": fingerprint,
        "extractor_version": EXTRACTOR_VERSION,
        "redaction_version": REDACTION_VERSION,
        "acl_projection_version": ACL_PROJECTION_VERSION,
        "tokenizer_version": TOKENIZER_VERSION,
    }
    for field, value in expected.items():
        if str(state[field]) != str(value):
            return True
    # A restored canonical backup can carry the same logical instance id but a
    # shorter outbox. Never let a later derived snapshot survive that rewind.
    return int(state["last_indexed_seq"]) > int(outbox_head)


def _index_has_rowid_mismatch(conn: sqlite3.Connection) -> bool:
    """Verify that chunk metadata and contentful FTS expose identical rowids."""

    missing_fts = conn.execute(
        "SELECT 1 FROM search_chunks c WHERE NOT EXISTS ("
        "SELECT 1 FROM search_fts f WHERE f.rowid=c.chunk_rowid) LIMIT 1"
    ).fetchone()
    if missing_fts is not None:
        return True
    orphan_fts = conn.execute(
        "SELECT 1 FROM search_fts f WHERE NOT EXISTS ("
        "SELECT 1 FROM search_chunks c WHERE c.chunk_rowid=f.rowid) LIMIT 1"
    ).fetchone()
    return orphan_fts is not None


def _reset_index_generation(
    conn: sqlite3.Connection,
    path: Path,
    *,
    instance_id: str,
    schema_version: int,
    fingerprint: str,
) -> None:
    """Physically purge an incompatible derived generation before replay."""

    now_ms = int(time.time() * 1000)
    conn.execute("BEGIN IMMEDIATE")
    try:
        # FTS5 delete markers can retain the old term bytes in live segment
        # pages even after VACUUM. Drop the virtual table (and its shadow
        # tables) before compacting so a redaction/extractor boundary is also
        # a physical confidentiality boundary.
        conn.execute("DROP TABLE search_fts")
        conn.execute("DELETE FROM search_documents")
        conn.execute("DELETE FROM query_snapshots")
        conn.execute(
            "UPDATE search_index_state SET source_db_instance_id=?, "
            "source_schema_version=?, source_fingerprint=?, index_generation=lower(hex(randomblob(16))), "
            "extractor_version=?, redaction_version=?, acl_projection_version=?, "
            "tokenizer_version=?, coverage_state='building', last_indexed_seq=0, "
            "indexed_documents=0, indexed_chunks=0, pending_estimate=NULL, "
            "indexed_through_ms=NULL, build_started_at_ms=?, build_completed_at_ms=NULL, "
            "last_error_class=NULL, updated_at_ms=? WHERE singleton_id=1",
            (
                instance_id,
                schema_version,
                fingerprint,
                EXTRACTOR_VERSION,
                REDACTION_VERSION,
                ACL_PROJECTION_VERSION,
                TOKENIZER_VERSION,
                now_ms,
                now_ms,
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    # DELETE alone can leave old redacted-version bytes in freelist/WAL pages.
    # This derived database is small enough to compact only on incompatible
    # generations; normal incremental batches never pay this cost.
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("VACUUM")
    conn.executescript(operational_search_schema_sql())
    _harden_index_files(path)


def _safe_structure_tokens(raw: str | None) -> str:
    """Describe keys/types/sizes without retaining scalar values."""

    if not raw:
        return ""
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return f"opaque type:text size:{len(str(raw))}"
    tokens: list[str] = []

    def visit(item: Any, path: str, depth: int) -> None:
        if depth > 8 or len(tokens) >= 512:
            return
        if isinstance(item, dict):
            tokens.append(f"{path or 'root'} type:object keys:{len(item)}")
            for key, child in list(item.items())[:100]:
                safe_key = re.sub(r"[^\w.-]+", "_", str(key))[:128]
                child_path = f"{path}.{safe_key}" if path else safe_key
                tokens.append(f"key:{safe_key} path:{child_path}")
                visit(child, child_path, depth + 1)
        elif isinstance(item, list):
            tokens.append(f"{path or 'root'} type:array size:{len(item)}")
            for child in item[:10]:
                visit(child, path, depth + 1)
        elif isinstance(item, str):
            tokens.append(f"{path or 'value'} type:string size:{len(item)}")
        elif item is None:
            tokens.append(f"{path or 'value'} type:null")
        elif isinstance(item, bool):
            tokens.append(f"{path or 'value'} type:boolean")
        elif isinstance(item, (int, float)):
            tokens.append(f"{path or 'value'} type:number")
        else:
            tokens.append(f"{path or 'value'} type:unknown")

    visit(value, "", 0)
    return " ".join(tokens)


def _looks_high_entropy(token: str) -> bool:
    # UUIDs remain searchable. Every other long, unbroken multi-class token
    # fails closed; this may mask a random-looking non-secret identifier, which
    # is preferable to persisting a credential in a rebuildable index.
    if len(token) < 32 or re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", token):
        return False
    classes = sum(
        bool(re.search(pattern, token))
        for pattern in (r"[a-z]", r"[A-Z]", r"[0-9]", r"[_+/=-]")
    )
    if classes < 2:
        return False
    return classes >= 2


def _replace_labeled_secret(match: re.Match[str]) -> str:
    value = match.group("value")
    if len(value) >= 2 and value[0] in {"\"", "'"} and value[-1] == value[0]:
        replacement = f"{value[0]}[REDACTED]{value[0]}"
    else:
        replacement = "[REDACTED]"
    return f"{match.group('prefix')}{replacement}"


def redact_search_text(value: Any, *, limit: int = 98_304) -> str:
    """Single fail-closed redactor used by every operational extractor."""

    text = str(value or "")
    text = _PEM_RE.sub("[REDACTED_PEM]", text)
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _JWT_RE.sub("[REDACTED_JWT]", text)
    text = _PROVIDER_TOKEN_RE.sub("[REDACTED_TOKEN]", text)
    text = _SIGNED_QUERY_RE.sub(lambda match: f"{match.group(1)}[REDACTED]", text)
    text = _SENSITIVE_CONTAINER_RE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]", text
    )
    text = _SENSITIVE_HEADER_RE.sub(
        lambda match: f"{match.group('prefix')}[REDACTED]", text
    )
    text = _LABELED_SECRET_RE.sub(_replace_labeled_secret, text)
    text = _PRIVATE_PATH_RE.sub("[REDACTED_PATH]", text)
    text = _OPAQUE_TOKEN_RE.sub(
        lambda match: "[REDACTED_OPAQUE]" if _looks_high_entropy(match.group(1)) else match.group(1),
        text,
    )
    return text[:limit]


def _redact_text(value: Any, *, limit: int = 98_304) -> str:
    return redact_search_text(value, limit=limit)


def _workflow_definition_text(raw: str | None) -> tuple[str, str | None]:
    try:
        graph = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return "", None
    parts: list[str] = []
    first_node: str | None = None
    safe_fields = {"label", "name", "description", "prompt", "instructions", "template", "cron_expression"}
    for node in (graph.get("nodes") or [])[:1000]:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "")
        first_node = first_node or node_id or None
        parts.extend((f"node {node_id}", f"type {node.get('type') or ''}"))
        # Runtime-authored graphs use ``config``. ``data`` remains a read-only
        # compatibility fallback for early client prototypes.
        data: dict[str, Any] = {}
        if isinstance(node.get("data"), dict):
            data.update(node["data"])
        if isinstance(node.get("config"), dict):
            data.update(node["config"])
        for key in safe_fields:
            value = data.get(key, node.get(key))
            if isinstance(value, (str, int, float, bool)) and value not in {"", None}:
                parts.append(f"{key} {_redact_text(value, limit=8192)}")
    return " ".join(parts), first_node


def _workflow_trace_text(raw: str | None, run_id: str) -> tuple[str, str | None]:
    try:
        trace = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return "", None
    parts: list[str] = []
    first_step: str | None = None
    attempts: dict[str, int] = {}
    for step in trace[:500] if isinstance(trace, list) else []:
        if not isinstance(step, dict):
            continue
        node_id = str(step.get("node_id") or step.get("id") or "unknown")
        attempt = int(step.get("attempt") or attempts.get(node_id, 0))
        attempts[node_id] = attempt + 1
        step_id = str(step.get("id") or step.get("trace_step_id") or f"trace:{run_id}:{node_id}:{attempt}")
        first_step = first_step or step_id
        parts.extend(
            (
                f"step {node_id}", f"type {step.get('type') or 'unknown'}",
                f"status {step.get('status') or 'unknown'}",
            )
        )
        if step.get("error"):
            parts.append(f"error {_redact_text(step['error'], limit=8192)}")
        # Preserve only tool identity/anchor metadata; never trace input/output.
        for key in ("tool_name", "tool_server", "tool_call_id", "tool_invocation_id"):
            if step.get(key):
                parts.append(f"{key} {_redact_text(step[key], limit=512)}")
        for tool_id in (step.get("tool_invocation_ids") or [])[:100]:
            parts.append(f"tool_invocation_id {_redact_text(tool_id, limit=512)}")
    return " ".join(parts), first_step


async def _automation_acl(conn: Any, resource_type: str, resource_id: str, parent: tuple[str, str] | None = None) -> tuple[str, str | None, str, int, str, int]:
    tenant_row = await (await conn.execute("SELECT network_id FROM network LIMIT 1")).fetchone()
    if tenant_row is not None and tenant_row[0]:
        tenant = str(tenant_row[0])
    else:
        state = await (await conn.execute("SELECT db_instance_id FROM operational_storage_state WHERE singleton_id=1")).fetchone()
        tenant = f"installation:{state[0]}"
    for kind, identifier in ((resource_type, resource_id), parent or ("", "")):
        if not kind:
            continue
        owner = await (
            await conn.execute(
                "SELECT owner_principal_id, visibility, acl_version, provenance "
                "FROM operational_resource_owners WHERE tenant_id=? AND resource_type=? AND resource_id=?",
                (tenant, kind, identifier),
            )
        ).fetchone()
        if owner is not None:
            ledger = await (
                await conn.execute(
                    "SELECT source_version FROM operational_automation_projection WHERE resource_type=? AND resource_id=?",
                    (resource_type, resource_id),
                )
            ).fetchone()
            return tenant, (str(owner[0]) if owner[0] else None), str(owner[1]), int(owner[2]), str(owner[3]), int(ledger[0] if ledger else 1)
    ledger = await (
        await conn.execute(
            "SELECT source_version FROM operational_automation_projection WHERE resource_type=? AND resource_id=?",
            (resource_type, resource_id),
        )
    ).fetchone()
    return tenant, None, "installation_shared", 1, "legacy_unattributed", int(ledger[0] if ledger else 1)


def literal_fts_query(query: str) -> str:
    if len(query.encode("utf-8")) > 16_384:
        raise ValueError("query byte budget exceeded")
    terms = _TERM_RE.findall(query.casefold())
    if len(terms) > 64:
        raise ValueError("query term budget exceeded")
    # User input is never interpreted as MATCH syntax. Quotes/operators are
    # discarded by the tokenizer and every normalized term is quoted.
    return " AND ".join(f'"{term.replace(chr(34), chr(34) * 2)}"*' for term in terms)


def _delete_doc(conn: sqlite3.Connection, doc_id: str) -> None:
    rows = conn.execute(
        "SELECT c.chunk_rowid FROM search_chunks c JOIN search_documents d "
        "ON d.document_rowid=c.document_rowid WHERE d.doc_id=?",
        (doc_id,),
    ).fetchall()
    for row in rows:
        conn.execute("DELETE FROM search_fts WHERE rowid=?", (int(row[0]),))
    conn.execute("DELETE FROM search_documents WHERE doc_id=?", (doc_id,))


def _put_document(
    conn: sqlite3.Connection,
    *,
    metadata: dict[str, Any],
    match_kind: str,
    source_field: str,
    body_safe: str,
    keywords_safe: str = "",
    identifiers_safe: str = "",
) -> None:
    doc_id = str(metadata["doc_id"])
    _delete_doc(conn, doc_id)
    metadata = dict(metadata)
    metadata["title_safe"] = _redact_text(metadata.get("title_safe") or "", limit=4096)
    metadata["author_display_safe"] = _redact_text(
        metadata.get("author_display_safe") or "", limit=1024
    )
    body_safe = _redact_text(body_safe)
    keywords_safe = _redact_text(keywords_safe, limit=8192)
    identifiers_safe = _redact_text(identifiers_safe, limit=8192)
    content = "\n".join(
        (str(metadata.get("title_safe") or ""), keywords_safe, identifiers_safe, body_safe)
    )
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    columns = (
        "doc_id", "tenant_id", "owner_principal_id", "visibility", "acl_version",
        "document_kind", "resource_type", "resource_id", "root_kind", "root_id",
        "parent_type", "parent_id", "session_id", "session_run_id", "target_kind",
        "message_id", "tool_invocation_id", "workflow_id", "workflow_run_id",
        "workflow_node_id", "workflow_trace_step_id", "scheduled_task_id",
        "scheduled_run_id", "event_id", "event_delivery_id", "definition_field",
        "caused_by_event_id", "caused_by_delivery_id", "status", "origin",
        "author_principal_id", "title_safe", "author_display_safe", "occurred_at_ms",
        "updated_at_ms", "source_version", "extractor_version", "redaction_version",
        "sensitivity", "completeness", "content_hash", "deleted_at_ms",
    )
    values = dict(metadata)
    values.update(
        extractor_version=EXTRACTOR_VERSION,
        redaction_version=REDACTION_VERSION,
        content_hash=content_hash,
        deleted_at_ms=None,
    )
    conn.execute(
        f"INSERT INTO search_documents ({','.join(columns)}) VALUES "
        f"({','.join('?' for _ in columns)})",
        tuple(values.get(column) for column in columns),
    )
    document_rowid = int(
        conn.execute("SELECT document_rowid FROM search_documents WHERE doc_id=?", (doc_id,)).fetchone()[0]
    )
    chunk_id = f"{doc_id}:0"
    conn.execute(
        "INSERT INTO search_chunks "
        "(chunk_id, document_rowid, ordinal, match_kind, source_field, indexed_chars, content_hash) "
        "VALUES (?, ?, 0, ?, ?, ?, ?)",
        (chunk_id, document_rowid, match_kind, source_field, min(len(body_safe), 98_304), content_hash),
    )
    chunk_rowid = int(
        conn.execute("SELECT chunk_rowid FROM search_chunks WHERE chunk_id=?", (chunk_id,)).fetchone()[0]
    )
    conn.execute(
        "INSERT INTO search_fts(rowid, title_safe, author_search_safe, keywords_safe, identifiers_safe, body_safe) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            chunk_rowid,
            str(metadata.get("title_safe") or "")[:4096],
            str(metadata.get("author_display_safe") or "")[:1024],
            keywords_safe,
            identifiers_safe,
            body_safe[:98_304],
        ),
    )


async def _source_row(conn: Any, source_kind: str, source_id: str) -> tuple[dict[str, Any], str, str, str, str] | None:
    if source_kind == "session":
        row = await (
            await conn.execute(
                "SELECT s.*, s.id AS resource_id FROM sessions_v2 s WHERE s.id=? AND s.deleted_at_ms IS NULL",
                (source_id,),
            )
        ).fetchone()
        if row is None:
            return None
        kind = "delegated_session" if row["parent_session_id"] else "chat"
        metadata = dict(row)
        metadata.update(
            doc_id=f"session:{source_id}", document_kind="session_metadata",
            resource_type="session", root_kind=kind, root_id=source_id,
            parent_type="session" if row["parent_session_id"] else None,
            parent_id=row["parent_session_id"], session_id=source_id,
            target_kind="chat", title_safe=_redact_text(row["title"] or "Untitled chat", limit=4096),
            author_display_safe=_redact_text(row["owner_handle_snapshot"] or "", limit=1024),
            occurred_at_ms=int(row["created_at_ms"]),
        )
        return metadata, "title", "title", str(metadata["title_safe"]), str(row["id"])
    if source_kind == "message":
        row = await (
            await conn.execute(
                "SELECT m.*, s.owner_principal_id, s.visibility AS session_visibility, "
                "s.acl_version, s.title, s.kind AS session_kind, s.parent_session_id, "
                "s.origin, s.deleted_at_ms FROM session_messages m JOIN sessions_v2 s "
                "ON s.id=m.session_id WHERE m.id=? AND s.deleted_at_ms IS NULL",
                (source_id,),
            )
        ).fetchone()
        if row is None or str(row["visibility"]) != "user_visible":
            return None
        root_kind = "delegated_session" if row["parent_session_id"] else "chat"
        # ``redacted_reasoning_content`` is provider-private opaque
        # redacted_thinking, not reasoning sanitized for display/search.
        body = _redact_text(row["text"] or "")
        metadata = dict(row)
        metadata.update(
            doc_id=f"message:{source_id}", tenant_id=row["tenant_id"],
            owner_principal_id=row["owner_principal_id"], visibility=row["session_visibility"],
            document_kind="message", resource_type="session", resource_id=row["session_id"],
            root_kind=root_kind, root_id=row["session_id"],
            parent_type="session" if row["parent_session_id"] else None,
            parent_id=row["parent_session_id"], target_kind="chat_message",
            message_id=source_id, title_safe=_redact_text(row["title"] or "Untitled chat", limit=4096),
            author_display_safe=_redact_text(row["author_display"] or row["author_handle_snapshot"] or "", limit=1024),
            occurred_at_ms=int(row["created_at_ms"]),
        )
        return metadata, "message", "text", body, str(row["id"])
    if source_kind == "tool_invocation":
        row = await (
            await conn.execute(
                "SELECT t.*, s.title, s.parent_session_id, s.origin, "
                "COALESCE((SELECT m.id FROM session_messages m WHERE m.session_id=t.session_id "
                "AND t.root_kind='session' AND m.run_id=t.session_run_id "
                "AND m.tool_call_id=t.tool_call_id "
                "ORDER BY m.sequence, m.id LIMIT 1), "
                "(SELECT m.id FROM session_messages m WHERE m.session_id=t.session_id "
                "AND m.run_id=t.session_run_id ORDER BY m.sequence DESC LIMIT 1)) AS message_id "
                "FROM tool_invocations t JOIN sessions_v2 s ON s.id=t.session_id "
                "WHERE t.id=? AND t.root_kind='session' "
                "AND s.deleted_at_ms IS NULL",
                (source_id,),
            )
        ).fetchone()
        if row is None or row["message_id"] is None:
            return None
        root_kind = "delegated_session" if row["parent_session_id"] else "chat"
        structure = " ".join((_safe_structure_tokens(row["args_json"]), _safe_structure_tokens(row["result_json"])))
        body = " ".join(
            part for part in (
                f"tool {row['tool_server']} {row['tool_name']} status {row['status']}",
                structure,
                "tool error" if row["error_json"] or row["error_text"] else "",
            ) if part
        )
        metadata = dict(row)
        metadata.update(
            doc_id=f"tool:{source_id}", document_kind="tool_invocation",
            resource_type="tool_invocation", resource_id=source_id,
            root_kind=root_kind, root_id=row["session_id"], target_kind="chat_tool",
            message_id=row["message_id"], tool_invocation_id=source_id,
            title_safe=_redact_text(row["title"] or row["tool_name"], limit=4096),
            author_display_safe="", occurred_at_ms=int(row["created_at_ms"]),
            sensitivity="redacted",
        )
        return metadata, "tool_name", "tool_redacted", body, str(row["tool_name"])
    if source_kind == "workflow_definition":
        row = await (await conn.execute("SELECT * FROM workflow_tasks WHERE id=?", (source_id,))).fetchone()
        if row is None:
            return None
        tenant, owner, visibility, acl_version, provenance, source_version = await _automation_acl(conn, source_kind, source_id)
        graph_text, first_node = _workflow_definition_text(row["graph_json"])
        body = " ".join(
            part for part in (
                _redact_text(row["name"]), _redact_text(row["description"]), graph_text,
            ) if part
        )
        created, updated = int(float(row["created_at"]) * 1000), int(float(row["updated_at"]) * 1000)
        metadata = {
            "doc_id": f"workflow_definition:{source_id}", "tenant_id": tenant,
            "owner_principal_id": owner, "visibility": visibility, "acl_version": acl_version,
            "document_kind": "workflow_definition", "resource_type": source_kind,
            "resource_id": source_id, "root_kind": "workflow_definition", "root_id": source_id,
            "target_kind": "workflow_definition", "workflow_id": source_id,
            "title_safe": _redact_text(row["name"], limit=4096), "author_display_safe": "",
            "occurred_at_ms": created, "updated_at_ms": max(created, updated),
            "source_version": source_version, "sensitivity": "safe",
            "completeness": "complete" if provenance != "legacy_unattributed" else "unknown",
        }
        return metadata, "description", "workflow_definition", body, str(row["name"])
    if source_kind == "workflow_run":
        row = await (
            await conn.execute(
                "SELECT r.*, w.name, w.description FROM workflow_runs r JOIN workflow_tasks w "
                "ON w.id=r.workflow_id WHERE r.id=?",
                (source_id,),
            )
        ).fetchone()
        if row is None:
            return None
        tenant, owner, visibility, acl_version, provenance, source_version = await _automation_acl(
            conn, source_kind, source_id, ("workflow_definition", str(row["workflow_id"]))
        )
        trace_text, trace_step_id = _workflow_trace_text(row["trace_json"], source_id)
        body = " ".join(
            part for part in (
                _redact_text(row["name"]), f"trigger {_redact_text(row['trigger'])}",
                f"status {_redact_text(row['status'])}", trace_text,
                f"error {_redact_text(row['error'], limit=8192)}" if row["error"] else "",
            ) if part
        )
        try:
            from .enums import normalize_run_status
            status = normalize_run_status(row["status"])[0]
            completeness = "complete" if provenance != "legacy_unattributed" else "unknown"
        except Exception:
            status, completeness = "failed", "malformed_source"
        started = int(float(row["started_at"]) * 1000)
        finished = int(float(row["finished_at"] or row["started_at"]) * 1000)
        metadata = {
            "doc_id": f"workflow_run:{source_id}", "tenant_id": tenant,
            "owner_principal_id": owner, "visibility": visibility, "acl_version": acl_version,
            "document_kind": "workflow_run", "resource_type": source_kind,
            "resource_id": source_id, "root_kind": "workflow_run", "root_id": source_id,
            "parent_type": "workflow", "parent_id": row["workflow_id"],
            "target_kind": "workflow_run", "workflow_id": row["workflow_id"],
            "workflow_run_id": source_id,
            "status": status, "origin": row["trigger"],
            "title_safe": _redact_text(row["name"], limit=4096), "author_display_safe": "",
            "occurred_at_ms": started, "updated_at_ms": max(started, finished),
            "source_version": source_version, "sensitivity": "redacted" if row["error"] else "safe",
            "completeness": completeness,
        }
        return metadata, "workflow_step", "trace", body, source_id
    if source_kind == "scheduled_definition":
        row = await (await conn.execute("SELECT * FROM scheduled_tasks WHERE id=?", (source_id,))).fetchone()
        if row is None:
            return None
        tenant, owner, visibility, acl_version, provenance, source_version = await _automation_acl(conn, source_kind, source_id)
        body = " ".join((
            _redact_text(row["name"]), _redact_text(row["prompt"]),
            f"schedule {_redact_text(row['cron_expression'])}",
            f"timezone {_redact_text(row['timezone'] or 'UTC')}",
        ))
        created, updated = int(float(row["created_at"]) * 1000), int(float(row["updated_at"]) * 1000)
        metadata = {
            "doc_id": f"scheduled_definition:{source_id}", "tenant_id": tenant,
            "owner_principal_id": owner, "visibility": visibility, "acl_version": acl_version,
            "document_kind": "scheduled_definition", "resource_type": source_kind,
            "resource_id": source_id, "root_kind": "scheduled_definition", "root_id": source_id,
            "target_kind": "scheduled_definition", "scheduled_task_id": source_id,
            "title_safe": _redact_text(row["name"], limit=4096),
            "author_display_safe": "", "occurred_at_ms": created, "updated_at_ms": max(created, updated),
            "source_version": source_version, "sensitivity": "safe",
            "completeness": "complete" if provenance != "legacy_unattributed" else "unknown",
        }
        return metadata, "prompt", "prompt", body, str(row["name"])
    if source_kind == "scheduled_run":
        row = await (
            await conn.execute(
                "SELECT r.*, t.name FROM task_runs r JOIN scheduled_tasks t ON t.id=r.task_id WHERE r.id=?",
                (source_id,),
            )
        ).fetchone()
        if row is None:
            return None
        tenant, owner, visibility, acl_version, provenance, source_version = await _automation_acl(
            conn, source_kind, source_id, ("scheduled_definition", str(row["task_id"]))
        )
        try:
            from .enums import normalize_run_status
            status = normalize_run_status(row["status"])[0]
            completeness = "complete" if provenance != "legacy_unattributed" else "unknown"
        except Exception:
            status, completeness = "failed", "malformed_source"
        body = " ".join(
            part for part in (
                _redact_text(row["name"]), f"status {status}",
                _redact_text(row["output"]),
                f"error {_redact_text(row['error'], limit=8192)}" if row["error"] else "",
            ) if part
        )
        started = int(float(row["started_at"]) * 1000)
        finished = int(float(row["finished_at"] or row["started_at"]) * 1000)
        metadata = {
            "doc_id": f"scheduled_run:{source_id}", "tenant_id": tenant,
            "owner_principal_id": owner, "visibility": visibility, "acl_version": acl_version,
            "document_kind": "scheduled_run", "resource_type": source_kind,
            "resource_id": source_id, "root_kind": "scheduled_run", "root_id": source_id,
            "parent_type": "scheduled_task", "parent_id": row["task_id"], "session_id": row["session_id"],
            "target_kind": "scheduled_run", "scheduled_task_id": row["task_id"],
            "scheduled_run_id": source_id, "status": status, "origin": row["trigger"],
            "title_safe": _redact_text(row["name"], limit=4096), "author_display_safe": "",
            "occurred_at_ms": started, "updated_at_ms": max(started, finished),
            "source_version": source_version, "sensitivity": "redacted" if row["error"] else "safe",
            "completeness": completeness,
        }
        return metadata, "message", "output", body, source_id
    if source_kind == "event_definition":
        row = await (await conn.execute("SELECT * FROM events WHERE id=?", (source_id,))).fetchone()
        if row is None:
            return None
        tenant, owner, visibility, acl_version, provenance, source_version = await _automation_acl(conn, source_kind, source_id)
        try:
            schema = json.loads(row["input_schema_json"] or "[]")
        except (TypeError, ValueError):
            schema = []
        fields = " ".join(
            f"field {_redact_text(item.get('name'))} {_redact_text(item.get('description'))}"
            for item in schema[:100] if isinstance(item, dict)
        )
        # secret_enc, precondition_json, endpoint payloads and auth material are
        # intentionally absent from this extractor.
        body = " ".join(
            part for part in (
                _redact_text(row["name"]), _redact_text(row["description"]),
                _redact_text(row["prompt_template"]), fields,
                f"action {_redact_text(row['action_kind'])}",
            ) if part
        )
        created, updated = int(float(row["created_at"]) * 1000), int(float(row["updated_at"]) * 1000)
        metadata = {
            "doc_id": f"event_definition:{source_id}", "tenant_id": tenant,
            "owner_principal_id": owner, "visibility": visibility, "acl_version": acl_version,
            "document_kind": "event_definition", "resource_type": source_kind,
            "resource_id": source_id, "root_kind": "event_definition", "root_id": source_id,
            "target_kind": "event_definition", "event_id": source_id,
            "title_safe": _redact_text(row["name"], limit=4096),
            "author_display_safe": "", "occurred_at_ms": created, "updated_at_ms": max(created, updated),
            "source_version": source_version, "sensitivity": "safe",
            "completeness": "complete" if provenance != "legacy_unattributed" else "unknown",
        }
        return metadata, "description", "event_definition", body, str(row["name"])
    if source_kind == "event_delivery":
        row = await (
            await conn.execute(
                "SELECT d.*, e.name FROM event_deliveries d JOIN events e ON e.id=d.event_id WHERE d.id=?",
                (source_id,),
            )
        ).fetchone()
        if row is None:
            return None
        tenant, owner, visibility, acl_version, provenance, source_version = await _automation_acl(
            conn, source_kind, source_id, ("event_definition", str(row["event_id"]))
        )
        try:
            from .enums import normalize_run_status
            status = normalize_run_status(row["status"])[0]
            completeness = "complete" if provenance != "legacy_unattributed" else "unknown"
        except Exception:
            status, completeness = "failed", "malformed_source"
        # payload_json/external_id are never read here. Output is a user-visible
        # execution result; common credential forms are masked before indexing.
        body = " ".join(
            part for part in (
                _redact_text(row["name"]), f"source {_redact_text(row['source'])}",
                f"status {status}", _redact_text(row["output"]),
                f"error {_redact_text(row['error'], limit=8192)}" if row["error"] else "",
            ) if part
        )
        started = int(float(row["started_at"]) * 1000)
        finished = int(float(row["finished_at"] or row["started_at"]) * 1000)
        metadata = {
            "doc_id": f"event_delivery:{source_id}", "tenant_id": tenant,
            "owner_principal_id": owner, "visibility": visibility, "acl_version": acl_version,
            "document_kind": "event_delivery", "resource_type": source_kind,
            "resource_id": source_id, "root_kind": "event_delivery", "root_id": source_id,
            "parent_type": "event", "parent_id": row["event_id"], "session_id": row["session_id"],
            "target_kind": "event_delivery", "event_id": row["event_id"],
            "event_delivery_id": source_id, "status": status, "origin": row["source"],
            "title_safe": _redact_text(row["name"], limit=4096), "author_display_safe": "",
            "occurred_at_ms": started, "updated_at_ms": max(started, finished),
            "source_version": source_version, "sensitivity": "redacted" if row["error"] else "safe",
            "completeness": completeness,
        }
        return metadata, "message", "output", body, source_id
    return None


async def _sync_operational_search_unlocked(
    db: Any,
    *,
    limit: int = 10_000,
    source_conn: Any | None = None,
) -> SearchIndexStatus:
    source = source_conn if source_conn is not None else await db._ensure_connected()
    db_path = Path(str(db.db_path)).resolve()
    target = operational_search_path(db_path)
    source_state = await (
        await source.execute(
            "SELECT db_instance_id, schema_version FROM operational_storage_state WHERE singleton_id=1"
        )
    ).fetchone()
    if source_state is None:
        raise RuntimeError("operational storage is unavailable")
    instance_id, schema_version = str(source_state[0]), int(source_state[1])
    fingerprint = _source_fingerprint(instance_id, schema_version)
    outbox_head = int(
        (
            await (
                await source.execute("SELECT COALESCE(MAX(seq), 0) FROM search_outbox")
            ).fetchone()
        )[0]
    )
    index = await asyncio.to_thread(_open_index, target)
    try:
        state = index.execute("SELECT * FROM search_index_state WHERE singleton_id=1").fetchone()
        if (
            _index_is_incompatible(
                state,
                instance_id=instance_id,
                schema_version=schema_version,
                fingerprint=fingerprint,
                outbox_head=outbox_head,
            )
            or _index_has_rowid_mismatch(index)
        ):
            await asyncio.to_thread(
                _reset_index_generation,
                index,
                target,
                instance_id=instance_id,
                schema_version=schema_version,
                fingerprint=fingerprint,
            )
            state = index.execute(
                "SELECT * FROM search_index_state WHERE singleton_id=1"
            ).fetchone()
        generation = str(state["index_generation"])
        last_seq = int(state["last_indexed_seq"])
        outbox = await (
            await source.execute(
                "SELECT seq, source_kind, source_id, operation, committed_at_ms FROM search_outbox "
                "WHERE seq>? ORDER BY seq LIMIT ?",
                (last_seq, max(1, min(int(limit), 50_000))),
            )
        ).fetchall()
        now_ms = int(time.time() * 1000)
        index.execute("BEGIN IMMEDIATE")
        try:
            for item in outbox:
                seq, kind, source_id, operation = int(item[0]), str(item[1]), str(item[2]), str(item[3])
                doc_prefix = {
                    "session": "session",
                    "message": "message",
                    "tool_invocation": "tool",
                    "workflow_definition": "workflow_definition",
                    "workflow_run": "workflow_run",
                    "scheduled_definition": "scheduled_definition",
                    "scheduled_run": "scheduled_run",
                    "event_definition": "event_definition",
                    "event_delivery": "event_delivery",
                }.get(kind)
                if doc_prefix:
                    _delete_doc(index, f"{doc_prefix}:{source_id}")
                if operation != "delete":
                    projected = await _source_row(source, kind, source_id)
                    if projected is not None:
                        metadata, match_kind, field, body, identifier = projected
                        metadata.setdefault("source_version", 1)
                        metadata.setdefault("sensitivity", "safe")
                        metadata.setdefault("completeness", "unknown")
                        metadata.setdefault("updated_at_ms", metadata["occurred_at_ms"])
                        _put_document(
                            index,
                            metadata=metadata,
                            match_kind=match_kind,
                            source_field=field,
                            body_safe=body,
                            keywords_safe=str(metadata.get("origin") or ""),
                            identifiers_safe=identifier,
                        )
                last_seq = seq
            remaining = int(
                (await (await source.execute("SELECT COUNT(*) FROM search_outbox WHERE seq>?", (last_seq,))).fetchone())[0]
            )
            documents = int(index.execute("SELECT COUNT(*) FROM search_documents WHERE deleted_at_ms IS NULL").fetchone()[0])
            chunks = int(index.execute("SELECT COUNT(*) FROM search_chunks").fetchone()[0])
            fts_rows = int(index.execute("SELECT COUNT(*) FROM search_fts").fetchone()[0])
            rowids_match = not _index_has_rowid_mismatch(index)
            coverage = (
                "ready"
                if remaining == 0 and chunks == fts_rows and rowids_match
                else ("invalid" if chunks != fts_rows or not rowids_match else "building")
            )
            completed = now_ms if coverage == "ready" else None
            index.execute(
                "UPDATE search_index_state SET source_db_instance_id=?, source_schema_version=?, "
                "source_fingerprint=?, extractor_version=?, redaction_version=?, acl_projection_version=?, "
                "tokenizer_version=?, coverage_state=?, last_indexed_seq=?, indexed_documents=?, indexed_chunks=?, "
                "pending_estimate=?, indexed_through_ms=?, build_started_at_ms=COALESCE(build_started_at_ms, ?), "
                "build_completed_at_ms=?, last_error_class=NULL, updated_at_ms=? WHERE singleton_id=1",
                (
                    instance_id, schema_version, fingerprint, EXTRACTOR_VERSION,
                    REDACTION_VERSION, ACL_PROJECTION_VERSION, TOKENIZER_VERSION,
                    coverage, last_seq,
                    documents, chunks, remaining,
                    max((int(row[4]) for row in outbox), default=state["indexed_through_ms"]),
                    now_ms, completed, now_ms,
                ),
            )
            index.commit()
        except Exception:
            index.rollback()
            raise
        await source.execute(
            "INSERT INTO search_outbox_consumers "
            "(consumer_id, last_seq, index_generation, last_error_class, updated_at_ms) "
            "VALUES (?, ?, ?, NULL, ?) ON CONFLICT(consumer_id) DO UPDATE SET "
            "last_seq=excluded.last_seq, index_generation=excluded.index_generation, "
            "last_error_class=NULL, updated_at_ms=excluded.updated_at_ms",
            (CONSUMER_ID, last_seq, generation, now_ms),
        )
        checkpoint = await (
            await source.execute(
                "SELECT MIN(last_seq) FROM search_outbox_consumers"
            )
        ).fetchone()
        if checkpoint is not None and checkpoint[0] is not None:
            through = int(checkpoint[0])
            # Retain exactly the latest consumed row for each source. That row
            # is sufficient to replay a brand-new generation, while removing
            # the quadratic history produced by full legacy-session snapshots.
            await source.execute(
                "DELETE FROM search_outbox WHERE seq IN ("
                "SELECT old.seq FROM search_outbox AS old "
                "WHERE old.seq<=? AND EXISTS ("
                "SELECT 1 FROM search_outbox AS newer WHERE "
                "newer.tenant_id=old.tenant_id AND newer.source_kind=old.source_kind "
                "AND newer.source_id=old.source_id AND newer.seq>old.seq AND newer.seq<=?"
                "))",
                (through, through),
            )
        await source.commit()
        return _status_from_conn(index, target)
    finally:
        _harden_index_files(target)
        index.close()


async def sync_operational_search(
    db: Any,
    *,
    limit: int = 10_000,
    source_conn: Any | None = None,
) -> SearchIndexStatus:
    """Serialize every index consumer for one canonical database path.

    Gateway warm-up, the persistent consumer, and the in-process agent tool
    can all request freshness.  They share one owner lock so two stale state
    reads cannot race into independent FTS transactions.
    """

    key = str(Path(str(db.db_path)).resolve())
    lock = _SYNC_LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        return await _sync_operational_search_unlocked(
            db,
            limit=limit,
            source_conn=source_conn,
        )


def _status_from_conn(conn: sqlite3.Connection, path: Path) -> SearchIndexStatus:
    row = conn.execute("SELECT * FROM search_index_state WHERE singleton_id=1").fetchone()
    state = str(row["coverage_state"])
    return SearchIndexStatus(
        ready=state == "ready",
        state={"ready": "ready", "building": "warming", "uninitialized": "warming", "degraded": "degraded", "invalid": "degraded"}.get(state, "unavailable"),
        generation=str(row["index_generation"]),
        seq=int(row["last_indexed_seq"]),
        documents=int(row["indexed_documents"]),
        pending=int(row["pending_estimate"] or 0),
        indexed_through_ms=int(row["indexed_through_ms"]) if row["indexed_through_ms"] is not None else None,
        path=str(path),
    )


async def operational_search_status(db: Any) -> dict[str, Any]:
    path = operational_search_path(db.db_path)
    if not path.exists():
        return {"ready": False, "state": "unavailable", "generation": "unavailable", "seq": 0, "documents": 0, "pending": 0, "path": str(path)}
    source = await db._ensure_connected()
    source_state = await (
        await source.execute(
            "SELECT db_instance_id, schema_version FROM operational_storage_state WHERE singleton_id=1"
        )
    ).fetchone()
    if source_state is None:
        return {"ready": False, "state": "unavailable", "generation": "unavailable", "seq": 0, "documents": 0, "pending": 0, "path": str(path)}
    instance_id, schema_version = str(source_state[0]), int(source_state[1])
    fingerprint = _source_fingerprint(instance_id, schema_version)
    outbox_head = int(
        (
            await (
                await source.execute("SELECT COALESCE(MAX(seq), 0) FROM search_outbox")
            ).fetchone()
        )[0]
    )
    conn = await asyncio.to_thread(_open_index, path)
    try:
        state = conn.execute(
            "SELECT * FROM search_index_state WHERE singleton_id=1"
        ).fetchone()
        status = _status_from_conn(conn, path)
        payload = dict(status.__dict__)
        incompatible = _index_is_incompatible(
            state,
            instance_id=instance_id,
            schema_version=schema_version,
            fingerprint=fingerprint,
            outbox_head=outbox_head,
        )
        rowid_mismatch = _index_has_rowid_mismatch(conn)
        if incompatible:
            payload.update(
                ready=False,
                state="warming",
                pending=max(int(payload.get("pending") or 0), 1),
            )
        elif rowid_mismatch:
            payload.update(
                ready=False,
                state="degraded",
                pending=max(int(payload.get("pending") or 0), 1),
            )
        return payload
    finally:
        conn.close()


async def warm_operational_search(db: Any, *, batch_size: int = 10_000) -> SearchIndexStatus:
    """Drain the durable outbox to a verified ready boundary.

    Capability discovery uses this on fresh installs so clients never need to
    call an unadvertised endpoint to advance a capped one-shot warm-up. Each
    transaction remains bounded; yielding between batches keeps cancellation
    responsive even for Friday-scale imports.
    """

    while True:
        status = await sync_operational_search(db, limit=batch_size)
        if status.ready:
            return status
        await asyncio.sleep(0)


def read_search_rows(
    path: str | Path,
    *,
    fts_query: str,
    scopes: Iterable[str],
    sort: str,
    tenant_id: str,
    principal_ids: Iterable[str],
    granted_resources: Iterable[tuple[str, str, int]] = (),
    filters: dict[str, Any] | None = None,
    max_candidates: int = 5000,
) -> list[dict[str, Any]]:
    conn = _open_index(Path(path))
    try:
        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS request_search_owners "
            "(principal_id TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS request_search_grants "
            "(resource_type TEXT, resource_id TEXT, acl_version INTEGER, "
            "PRIMARY KEY(resource_type, resource_id, acl_version)) WITHOUT ROWID"
        )
        conn.execute("DELETE FROM request_search_owners")
        conn.execute("DELETE FROM request_search_grants")
        conn.executemany(
            "INSERT OR IGNORE INTO request_search_owners(principal_id) VALUES (?)",
            ((str(value),) for value in principal_ids),
        )
        conn.executemany(
            "INSERT OR IGNORE INTO request_search_grants "
            "(resource_type, resource_id, acl_version) VALUES (?, ?, ?)",
            ((str(kind), str(identifier), int(version)) for kind, identifier, version in granted_resources),
        )
        scope_kinds = {
            "chats": {"session_metadata", "message"},
            "tools": {"tool_invocation"},
            "workflows": {"workflow_definition", "workflow_run", "workflow_step"},
            "scheduled": {"scheduled_definition", "scheduled_run"},
            "events": {"event_definition", "event_delivery"},
        }
        kinds = sorted(set().union(*(scope_kinds.get(scope, set()) for scope in scopes)))
        if not kinds:
            return []
        placeholders = ",".join("?" for _ in kinds)
        clauses = [
            "d.tenant_id=?",
            f"d.document_kind IN ({placeholders})",
            "d.deleted_at_ms IS NULL",
            "d.visibility<>'quarantined'",
            "(d.visibility IN ('installation_shared','public') "
            "OR EXISTS (SELECT 1 FROM request_search_owners o "
            "WHERE o.principal_id=d.owner_principal_id) "
            "OR EXISTS (SELECT 1 FROM request_search_grants g "
            "WHERE ((g.resource_type=d.resource_type AND g.resource_id=d.resource_id) "
            "OR (d.document_kind='tool_invocation' AND g.resource_type='session' "
            "AND g.resource_id=d.session_id)) "
            "AND g.acl_version=d.acl_version))",
        ]
        params: list[Any] = [tenant_id, *kinds]
        effective_filters = filters or {}
        statuses = list(effective_filters.get("status") or ())
        if statuses:
            clauses.append(f"d.status IN ({','.join('?' for _ in statuses)})")
            params.extend(statuses)
        if effective_filters.get("from_ms") is not None:
            clauses.append("d.occurred_at_ms>=?")
            params.append(int(effective_filters["from_ms"]))
        if effective_filters.get("to_ms") is not None:
            clauses.append("d.occurred_at_ms<?")
            params.append(int(effective_filters["to_ms"]))
        for field in ("origin", "parent_type", "parent_id"):
            if effective_filters.get(field) is not None:
                clauses.append(f"d.{field}=?")
                params.append(str(effective_filters[field]))
        if effective_filters.get("session_id") is not None:
            clauses.append("d.session_id=?")
            params.append(str(effective_filters["session_id"]))
        root = effective_filters.get("root")
        if root:
            clauses.extend(("d.root_kind=?", "d.root_id=?"))
            params.extend((str(root["kind"]), str(root["id"])))
        where = " AND ".join(clauses)
        if fts_query:
            order = (
                "rank, d.occurred_at_ms DESC, d.document_rowid DESC"
                if sort == "relevance"
                else "d.occurred_at_ms DESC, d.document_rowid DESC"
            )
            rows = conn.execute(
                "SELECT d.*, c.chunk_id, c.match_kind, c.source_field, c.chunk_rowid, "
                "bm25(search_fts, 8.0, 2.0, 4.0, 12.0, 1.0) AS rank "
                "FROM search_fts JOIN search_chunks c ON c.chunk_rowid=search_fts.rowid "
                "JOIN search_documents d ON d.document_rowid=c.document_rowid "
                f"WHERE search_fts MATCH ? AND {where} ORDER BY {order} LIMIT ?",
                (fts_query, *params, max_candidates),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT d.*, c.chunk_id, c.match_kind, c.source_field, c.chunk_rowid, 0.0 AS rank "
                "FROM search_documents d JOIN search_chunks c ON c.document_rowid=d.document_rowid "
                f"WHERE {where} ORDER BY d.occurred_at_ms DESC, d.document_rowid DESC LIMIT ?",
                (*params, max_candidates),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def indexed_target_kinds(path: str | Path) -> frozenset[str]:
    conn = _open_index(Path(path))
    try:
        return frozenset(
            str(row[0])
            for row in conn.execute(
                "SELECT DISTINCT target_kind FROM search_documents WHERE deleted_at_ms IS NULL"
            ).fetchall()
        )
    finally:
        conn.close()


def read_authorized_highlight(
    path: str | Path,
    chunk_id: str,
    expected_source_version: int,
    expected_content_hash: str,
    fts_query: str,
) -> str | None:
    conn = _open_index(Path(path))
    try:
        if fts_query:
            row = conn.execute(
                "SELECT highlight(search_fts, 4, char(1), char(2)) "
                "FROM search_fts JOIN search_chunks c ON c.chunk_rowid=search_fts.rowid "
                "JOIN search_documents d ON d.document_rowid=c.document_rowid "
                "WHERE c.chunk_id=? AND d.source_version=? AND d.content_hash=? "
                "AND search_fts MATCH ?",
                (chunk_id, int(expected_source_version), expected_content_hash, fts_query),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT search_fts.body_safe FROM search_fts "
                "JOIN search_chunks c ON c.chunk_rowid=search_fts.rowid "
                "JOIN search_documents d ON d.document_rowid=c.document_rowid "
                "WHERE c.chunk_id=? AND d.source_version=? AND d.content_hash=?",
                (chunk_id, int(expected_source_version), expected_content_hash),
            ).fetchone()
        return str(row[0])[:400] if row is not None else None
    finally:
        conn.close()
