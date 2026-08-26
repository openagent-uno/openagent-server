"""Unified operational history and durable transcript REST handlers.

This module deliberately contains no vault/search-index imports.  History and
message navigation read canonical operational tables and remain available while
the rebuildable FTS index is warming or degraded.
"""

from __future__ import annotations

import base64
import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import aiosqlite
from aiohttp import web

from src.memory.operational.access import AccessContext, resource_is_visible
from src.memory.operational.repository import (
    backfill_batch_async,
    projection_coverage_async,
    reconcile_pending_async,
)

from ._common import gateway_db


logger = logging.getLogger(__name__)
_HISTORY_KINDS = frozenset(
    {"chat", "delegated_session", "workflow_run", "scheduled_run", "event_delivery"}
)
_RUN_STATUSES = frozenset(
    {
        "pending", "queued", "received", "running", "success", "failed",
        "cancelled", "rejected", "interrupted", "skipped", "timed_out",
    }
)
_SNAPSHOT_TTL_MS = 5 * 60 * 1000
_MAX_SNAPSHOT_ITEMS = 50_000
_MAX_SNAPSHOTS_PER_PRINCIPAL = 4
_MAX_SNAPSHOTS_GLOBAL = 32
_MAX_SNAPSHOT_ROWS_GLOBAL = 100_000


class ApiProblem(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(code)
        self.status = status
        self.code = code
        self.message = message
        self.reason = reason
        self.details = dict(details or {})


def _problem(problem: ApiProblem) -> web.Response:
    error: dict[str, Any] = {
        "code": problem.code,
        "message": problem.message,
        "retryable": problem.status in {409, 429, 503},
    }
    details = dict(problem.details)
    if problem.reason:
        details["reason"] = problem.reason
    if details:
        error["details"] = details
    return web.json_response(
        {"error": error}, status=problem.status, headers={"Cache-Control": "no-store"}
    )


def _json(payload: Any, *, status: int = 200) -> web.Response:
    return web.json_response(payload, status=status, headers={"Cache-Control": "no-store"})


def _iso(epoch_ms: int | float | None) -> str | None:
    if epoch_ms is None:
        return None
    return datetime.fromtimestamp(float(epoch_ms) / 1000, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _parse_iso(value: str | None, *, name: str) -> int | None:
    if not value:
        return None
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            raise ValueError
        return int(parsed.timestamp() * 1000)
    except (ValueError, OverflowError) as exc:
        raise ApiProblem(400, "invalid_request", f"{name} must be an ISO UTC timestamp") from exc


def _bounded_int(raw: str | None, *, name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = default if raw is None else int(raw)
    except (TypeError, ValueError) as exc:
        raise ApiProblem(400, "invalid_request", f"{name} must be an integer") from exc
    if value < minimum or value > maximum:
        raise ApiProblem(400, "invalid_request", f"{name} is outside the supported range")
    return value


def _bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"0", "false", "no"}:
        return False
    raise ApiProblem(400, "invalid_request", "boolean query parameter is invalid")


async def _canonical_status(gateway: Any, db: Any, conn: Any) -> bool:
    """Cheap, read-only readiness barrier for request handlers."""

    if not bool(getattr(gateway, "_operational_history_ready", False)):
        return False
    # Writers can append journals after the worker reached ready. These
    # indexed EXISTS probes do not wait on the maintenance connection/lock and
    # prevent a stale-ready cache from advertising history prematurely.
    pending_session = await (
        await conn.execute(
            "SELECT 1 FROM legacy_session_changes WHERE processed_at_ms IS NULL LIMIT 1"
        )
    ).fetchone()
    pending_automation = await (
        await conn.execute(
            "SELECT 1 FROM operational_automation_changes "
            "WHERE processed_at_ms IS NULL LIMIT 1"
        )
    ).fetchone()
    if pending_session is not None or pending_automation is not None:
        setattr(gateway, "_operational_history_ready", False)
        return False
    return True


async def _bounded_prepare(request: web.Request):
    db = gateway_db(request)
    if db is None:
        raise ApiProblem(501, "unsupported", "Operational history is not supported by this server")
    conn = await db._ensure_connected()
    complete = await _canonical_status(request.app["gateway"], db, conn)
    return db, conn, complete


async def _background_bootstrap(gateway: Any, db: Any) -> None:
    shared_conn = await db._ensure_connected()
    raw_path = str(db.db_path)
    owns_conn = raw_path != ":memory:" and not raw_path.startswith("file::memory:")
    if owns_conn:
        conn = await aiosqlite.connect(raw_path, timeout=60.0)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA busy_timeout=60000")
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=FULL")
        await conn.execute("PRAGMA foreign_keys=ON")
    else:
        conn = shared_conn
    ready_event = getattr(gateway, "_operational_ready_event", None)
    if ready_event is None:
        ready_event = asyncio.Event()
        setattr(gateway, "_operational_ready_event", ready_event)
    backoff_s = 1.0
    try:
        while True:
            try:
                while True:
                    writes = await reconcile_pending_async(
                        conn, limit=100, worker_id="gateway-background-bootstrap"
                    )
                    await conn.commit()
                    coverage = await projection_coverage_async(conn)
                    if int(coverage["pending_sessions"]) == 0:
                        break
                    if not writes:
                        raise RuntimeError("session reconciliation made no progress")
                    await asyncio.sleep(0)
                while True:
                    _writes, complete = await backfill_batch_async(conn, limit=100)
                    await conn.commit()
                    if complete:
                        break
                    await asyncio.sleep(0)
                from src.memory.operational.automation import project_automation_pending
                from src.memory.operational.search import operational_search_status, sync_operational_search

                while True:
                    _changed, complete = await project_automation_pending(conn, limit=100)
                    await conn.commit()
                    if complete:
                        break
                    await asyncio.sleep(0)
                coverage = await projection_coverage_async(conn)
                setattr(gateway, "_operational_history_ready", bool(coverage["complete"]))
                final_search_status: dict[str, Any] = {}
                while True:
                    raw_search = await operational_search_status(db)
                    outbox_head = int(
                        (
                            await (
                                await conn.execute(
                                    "SELECT COALESCE(MAX(seq), 0) FROM search_outbox"
                                )
                            ).fetchone()
                        )[0]
                    )
                    if bool(raw_search.get("ready")) and int(raw_search.get("seq") or 0) >= outbox_head:
                        final_search_status = dict(raw_search)
                        break
                    search_status = await sync_operational_search(
                        db, limit=1000, source_conn=conn
                    )
                    final_search_status = dict(search_status.__dict__)
                    if search_status.ready and search_status.seq >= outbox_head:
                        break
                    await asyncio.sleep(0)
                setattr(gateway, "_operational_search_status", final_search_status)
                setattr(gateway, "_operational_bootstrap_error", False)
                ready_event.set()
                backoff_s = 1.0
                # Stay alive as the single outbox consumer. New session writers
                # and automation triggers become searchable without requiring a
                # capability/search request to perform maintenance.
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                raise
            except Exception:
                try:
                    await conn.rollback()
                except Exception:
                    pass
                ready_event.clear()
                setattr(gateway, "_operational_search_status", {
                    "ready": False, "state": "degraded", "generation": "unavailable",
                    "seq": 0, "documents": 0, "pending": 0,
                    "indexed_through_ms": None,
                })
                already_failed = bool(getattr(gateway, "_operational_bootstrap_error", False))
                setattr(gateway, "_operational_bootstrap_error", True)
                if not already_failed:
                    # Do not format exceptions: source rows may contain private content.
                    logger.error("operational background bootstrap failed (details suppressed)")
                await asyncio.sleep(backoff_s)
                backoff_s = min(30.0, backoff_s * 2.0)
    finally:
        if owns_conn:
            await conn.close()


def start_background_maintenance(gateway: Any, db: Any | None = None) -> asyncio.Task | None:
    if db is None:
        db = getattr(getattr(gateway, "agent", None), "memory_db", None)
    if db is None:
        return None
    if getattr(gateway, "_operational_ready_event", None) is None:
        setattr(gateway, "_operational_ready_event", asyncio.Event())
    if not hasattr(gateway, "_operational_history_ready"):
        setattr(gateway, "_operational_history_ready", False)
    task = getattr(gateway, "_operational_bootstrap_task", None)
    if task is None or task.done():
        task = asyncio.create_task(_background_bootstrap(gateway, db))
        setattr(gateway, "_operational_bootstrap_task", task)
    return task


def _start_background(request: web.Request, db: Any) -> None:
    start_background_maintenance(request.app["gateway"], db)


async def stop_background_maintenance(gateway: Any) -> None:
    task = getattr(gateway, "_operational_bootstrap_task", None)
    if task is not None and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    setattr(gateway, "_operational_bootstrap_task", None)
    ready_event = getattr(gateway, "_operational_ready_event", None)
    if ready_event is not None:
        ready_event.clear()
    setattr(gateway, "_operational_ready_event", None)
    state = getattr(gateway, "_operational_cursor_state", None)
    if state is not None:
        state.history.clear()
        state.search.clear()
    setattr(gateway, "_operational_cursor_state", None)


async def _canonical_conn(request: web.Request):
    db, conn, complete = await _bounded_prepare(request)
    if not complete:
        _start_background(request, db)
        raise ApiProblem(503, "warming", "Operational history is still being prepared")
    return db, conn


async def _current_search_status(
    db: Any,
    conn: Any,
    *,
    gateway: Any | None = None,
    require_outbox_head: bool = True,
) -> dict[str, Any]:
    """Read derived-index status and compare it with canonical outbox head."""

    from src.memory.operational.search import operational_search_status

    cached = getattr(gateway, "_operational_search_status", None) if gateway else None
    status = dict(cached) if isinstance(cached, dict) else dict(await operational_search_status(db))
    outbox_head = int(
        (
            await (
                await conn.execute("SELECT COALESCE(MAX(seq), 0) FROM search_outbox")
            ).fetchone()
        )[0]
    )
    indexed_seq = int(status.get("seq") or 0)
    # The persistent consumer commits the derived index before publishing its
    # latest in-memory status.  A request can therefore observe a newer outbox
    # head while holding the previous loop's cache entry.  Refresh that stale
    # entry from the authoritative derived-index metadata before deciding that
    # search is warming; keep the caught-up cache as the hot path.
    if isinstance(cached, dict) and (
        not bool(status.get("ready"))
        or (require_outbox_head and indexed_seq < outbox_head)
    ):
        status = dict(await operational_search_status(db))
        indexed_seq = int(status.get("seq") or 0)
    if status.get("state") == "unavailable":
        status.update(ready=False, state="warming")
    if require_outbox_head and indexed_seq < outbox_head:
        status.update(
            ready=False,
            state="warming",
            pending=max(int(status.get("pending") or 0), outbox_head - indexed_seq),
        )
    return status


async def claim_created_resource(request: web.Request, resource_type: str, resource_id: str) -> None:
    """Persist certificate-derived ownership for a newly created automation."""

    access = AccessContext.from_request(request)
    db = gateway_db(request)
    if db is None:
        return
    conn = await db._ensure_connected()
    from src.memory.operational.automation import claim_resource

    await claim_resource(
        conn,
        tenant_id=access.tenant_id,
        resource_type=resource_type,
        resource_id=resource_id,
        owner_principal_id=access.principal_id,
    )
    await conn.commit()


@dataclass
class _HistorySnapshot:
    snapshot_id: str
    tenant_id: str
    principal_id: str
    request_digest: str
    revision: str
    activity_ids: tuple[str, ...]
    expires_at_ms: int


@dataclass
class _SearchSnapshot:
    snapshot_id: str
    tenant_id: str
    principal_id: str
    request_digest: str
    index_generation: str
    indexed_seq: int
    rows: tuple[dict[str, Any], ...]
    expires_at_ms: int


class _CursorState:
    def __init__(self) -> None:
        self.secret = secrets.token_bytes(32)
        self.history: dict[str, _HistorySnapshot] = {}
        self.search: dict[str, _SearchSnapshot] = {}
        self._serial = 0
        self._order: dict[tuple[str, str], int] = {}

    def _put(self, kind: str, snapshot: _HistorySnapshot | _SearchSnapshot) -> None:
        target = self.history if kind == "history" else self.search
        target[snapshot.snapshot_id] = snapshot  # type: ignore[assignment]
        self._serial += 1
        self._order[(kind, snapshot.snapshot_id)] = self._serial
        self._enforce_quotas(snapshot.principal_id)

    def put_history(self, snapshot: _HistorySnapshot) -> None:
        self._put("history", snapshot)

    def put_search(self, snapshot: _SearchSnapshot) -> None:
        self._put("search", snapshot)

    def _entries(self) -> list[tuple[str, str, Any]]:
        return [
            *(("history", key, value) for key, value in self.history.items()),
            *(("search", key, value) for key, value in self.search.items()),
        ]

    @staticmethod
    def _rows(kind: str, snapshot: Any) -> int:
        if kind == "history":
            return len(snapshot.activity_ids)
        # Root grouping stores compact match rows below each root. Count those
        # actual allocations, not only the number of outer roots, or one chat
        # with tens of thousands of matches bypasses the global memory quota.
        return sum(
            max(1, len(row.get("_matches") or ())) for row in snapshot.rows
        )

    def _enforce_quotas(self, principal_id: str) -> None:
        while True:
            entries = self._entries()
            principal_count = sum(
                1 for _kind, _key, item in entries if item.principal_id == principal_id
            )
            total_rows = sum(self._rows(kind, item) for kind, _key, item in entries)
            if (
                len(entries) <= _MAX_SNAPSHOTS_GLOBAL
                and principal_count <= _MAX_SNAPSHOTS_PER_PRINCIPAL
                and total_rows <= _MAX_SNAPSHOT_ROWS_GLOBAL
            ):
                return
            eligible = [
                (kind, key, item)
                for kind, key, item in entries
                if principal_count <= _MAX_SNAPSHOTS_PER_PRINCIPAL
                or item.principal_id == principal_id
            ]
            kind, key, _item = min(
                eligible,
                key=lambda entry: self._order.get((entry[0], entry[1]), 0),
            )
            (self.history if kind == "history" else self.search).pop(key, None)
            self._order.pop((kind, key), None)

    def encode(self, payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        sig = hmac.new(self.secret, raw, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(raw + sig).decode().rstrip("=")

    def decode(self, token: str) -> dict[str, Any]:
        try:
            raw_sig = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
            raw, signature = raw_sig[:-32], raw_sig[-32:]
            if len(signature) != 32 or not hmac.compare_digest(
                signature, hmac.new(self.secret, raw, hashlib.sha256).digest()
            ):
                raise ValueError
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError
            return value
        except Exception as exc:  # cursor is untrusted; never reflect it
            raise ApiProblem(
                409,
                "cursor_stale",
                "The result snapshot is no longer valid; refresh to continue",
                reason="invalid_signature",
            ) from exc

    def prune(self, now_ms: int) -> None:
        self.history = {
            key: value for key, value in self.history.items() if value.expires_at_ms > now_ms
        }
        self.search = {
            key: value for key, value in self.search.items() if value.expires_at_ms > now_ms
        }
        live = {
            *(("history", key) for key in self.history),
            *(("search", key) for key in self.search),
        }
        self._order = {key: value for key, value in self._order.items() if key in live}


def _state(request: web.Request) -> _CursorState:
    gateway = request.app["gateway"]
    value = getattr(gateway, "_operational_cursor_state", None)
    if value is None:
        value = _CursorState()
        setattr(gateway, "_operational_cursor_state", value)
    return value


def _history_filters(request: web.Request) -> tuple[dict[str, Any], int]:
    raw_kinds = request.query.get("kinds")
    raw_status = request.query.get("status")
    kinds = sorted({part for part in (raw_kinds or "").split(",") if part})
    statuses = sorted({part for part in (raw_status or "").split(",") if part})
    if any(kind not in _HISTORY_KINDS for kind in kinds):
        raise ApiProblem(400, "invalid_request", "kinds contains an unsupported value")
    if any(status not in _RUN_STATUSES for status in statuses):
        raise ApiProblem(400, "invalid_request", "status contains an unsupported value")
    parent_type = request.query.get("parent_type") or None
    parent_id = request.query.get("parent_id") or None
    if bool(parent_type) != bool(parent_id):
        raise ApiProblem(400, "invalid_request", "parent_type and parent_id must be used together")
    filters = {
        "kinds": kinds,
        "status": statuses,
        "origin": request.query.get("origin") or None,
        "parent_type": parent_type,
        "parent_id": parent_id,
        "from": _parse_iso(request.query.get("from"), name="from"),
        "to": _parse_iso(request.query.get("to"), name="to"),
        "include_children": _bool(request.query.get("include_children"), False),
    }
    if filters["from"] is not None and filters["to"] is not None and filters["from"] >= filters["to"]:
        raise ApiProblem(400, "invalid_request", "from must be earlier than to")
    limit = _bounded_int(request.query.get("limit"), name="limit", default=50, minimum=1, maximum=100)
    return filters, limit


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


async def _activity_rows(conn: Any, access: AccessContext, filters: dict[str, Any]) -> list[Any]:
    clauses = [
        "a.tenant_id=?",
        "a.deleted_at_ms IS NULL",
        "a.visibility<>'quarantined'",
    ]
    params: list[Any] = [access.tenant_id]
    owners = sorted(access.principal_ids)
    identities = sorted(access.grant_identities)
    authorization = ["a.visibility IN ('installation_shared','public')"]
    if owners:
        authorization.append(
            f"a.owner_principal_id IN ({','.join('?' for _ in owners)})"
        )
        params.extend(owners)
    if identities:
        identity_sql = " OR ".join(
            "(r.principal_type=? AND r.principal_id=?)" for _ in identities
        )
        authorization.append(
            "EXISTS (SELECT 1 FROM resource_acl r WHERE r.tenant_id=a.tenant_id "
            "AND r.resource_type=a.resource_type AND r.resource_id=a.resource_id "
            "AND r.permission IN ('view','admin') AND r.acl_version=a.acl_version "
            f"AND ({identity_sql}))"
        )
        for principal_type, principal_id in identities:
            params.extend((principal_type, principal_id))
    clauses.append(f"({' OR '.join(authorization)})")
    if filters["kinds"]:
        clauses.append(f"a.kind IN ({','.join('?' for _ in filters['kinds'])})")
        params.extend(filters["kinds"])
    if filters["status"]:
        clauses.append(f"a.status IN ({','.join('?' for _ in filters['status'])})")
        params.extend(filters["status"])
    for field in ("origin", "parent_type", "parent_id"):
        if filters[field] is not None:
            clauses.append(f"a.{field}=?")
            params.append(filters[field])
    if filters["from"] is not None:
        clauses.append("a.occurred_at_ms>=?")
        params.append(filters["from"])
    if filters["to"] is not None:
        clauses.append("a.occurred_at_ms<?")
        params.append(filters["to"])
    if not filters["include_children"]:
        clauses.append("NOT (a.resource_type='session' AND lower(COALESCE(a.origin, '')) IN ('workflow','scheduler','scheduled','event'))")
    cursor = await conn.execute(
        "SELECT a.*, COALESCE(s.completeness, 'unknown') AS completeness "
        "FROM activity_items a LEFT JOIN sessions_v2 s "
        "ON a.resource_type='session' AND s.id=a.resource_id AND s.tenant_id=a.tenant_id "
        f"WHERE {' AND '.join(clauses)} "
        "ORDER BY a.occurred_at_ms DESC, a.kind, a.activity_id LIMIT ?",
        (*params, _MAX_SNAPSHOT_ITEMS + 1),
    )
    rows = await cursor.fetchall()
    if len(rows) > _MAX_SNAPSHOT_ITEMS:
        raise ApiProblem(503, "degraded", "History snapshot is temporarily too large")
    visible: list[Any] = []
    for row in rows:
        if await resource_is_visible(conn, row, access):
            visible.append(row)
    return visible


def _activity_json(row: Any) -> dict[str, Any]:
    parent = None
    if row["parent_type"] and row["parent_id"]:
        parent = {
            "kind": str(row["parent_type"]),
            "id": str(row["parent_id"]),
            "title": str(row["parent_id"]),
        }
    status = row["status"]
    return {
        "id": str(row["activity_id"]),
        "kind": str(row["kind"]),
        "resource_id": str(row["resource_id"]),
        "title": str(row["title"] or "Untitled activity"),
        "status": str(status) if status is not None else None,
        "origin": str(row["origin"]) if row["origin"] is not None else None,
        "occurred_at": _iso(row["occurred_at_ms"]),
        "updated_at": _iso(row["updated_at_ms"]),
        "parent": parent,
        "session_id": str(row["session_id"]) if row["session_id"] else None,
        "live": status in {"pending", "queued", "received", "running"},
        "completeness": str(row["completeness"]),
    }


async def handle_capabilities(request: web.Request) -> web.Response:
    try:
        _access = AccessContext.from_request(request)
        db, conn, canonical_ready = await _bounded_prepare(request)
        state = await (
            await conn.execute(
                "SELECT schema_version, history_revision FROM operational_storage_state WHERE singleton_id=1"
            )
        ).fetchone()
        phase = await (
            await conn.execute("SELECT phase FROM storage_migration_state WHERE singleton_id=1")
        ).fetchone()
        pending_row = await (
            await conn.execute(
                "SELECT "
                "MAX(0, (SELECT COUNT(*) FROM sessions) - "
                "       (SELECT COUNT(*) FROM sessions_v2 WHERE deleted_at_ms IS NULL)) + "
                "(SELECT COUNT(DISTINCT session_id) FROM legacy_session_changes "
                " WHERE processed_at_ms IS NULL) + "
                "(SELECT COUNT(*) FROM operational_automation_changes "
                " WHERE processed_at_ms IS NULL)"
            )
        ).fetchone()
        history_pending = int(pending_row[0] if pending_row else 0)
        search_status = {"ready": False, "state": "warming", "generation": "unavailable", "seq": 0}
        if canonical_ready:
            try:
                search_status = await _current_search_status(
                    db, conn, gateway=request.app["gateway"]
                )
            except Exception:
                search_status["state"] = "degraded"
        if bool(getattr(request.app["gateway"], "_operational_bootstrap_error", False)):
            search_status.update(ready=False, state="degraded")
        if not canonical_ready or not search_status["ready"]:
            _start_background(request, db)
        # Realtime names are intentionally absent until the gateway emits
        # principal-filtered events.  Capability discovery must not promise a
        # transport feature merely because its wire schema exists in docs.
        features: dict[str, Any] = {}
        if canonical_ready:
            features.update({
                "history": {
                "version": 2,
                "kinds": sorted(_HISTORY_KINDS),
                "snapshot_pagination": True,
                "max_page_size": 100,
                },
                "session_messages": {"version": 1, "around": True, "bidirectional": True, "max_page_size": 100},
                "detail_resolvers": {
                "version": 1,
                "tool_invocation": True,
                "workflow_run": True,
                "scheduled_run": True,
                "event_delivery": True,
                "definition_field_anchors": False,
                },
            })
        if canonical_ready and search_status["ready"]:
            features["global_search"] = {
                "version": 1,
                "scopes": ["chats", "tools", "workflows", "scheduled", "events"],
                "sorts": ["relevance", "recent"],
                "query_modes": ["keyword"],
                "tool_content": "redacted",
                "targets": [
                    "chat", "chat_message", "chat_tool", "workflow_definition",
                    "workflow_run", "scheduled_definition", "scheduled_run",
                    "event_definition", "event_delivery",
                ],
                "snapshot_pagination": True,
                "max_page_size": 100,
            }
        return _json(
            {
                "api_revision": 2,
                "features": features,
                "storage": {
                    "phase": str(phase[0] if phase else "legacy"),
                    "schema_version": int(state[0] if state else 1),
                    "history_ready": bool(canonical_ready),
                    "history_pending": history_pending,
                    "search_state": str(search_status["state"]),
                    "search_ready": bool(search_status["ready"]),
                    "index_generation": str(search_status["generation"]),
                    "indexed_seq": int(search_status["seq"]),
                },
            }
        )
    except PermissionError:
        return _problem(ApiProblem(401, "unauthorized", "Authentication required"))
    except ApiProblem as exc:
        return _problem(exc)
    except Exception:
        logger.error("operational capabilities failed (exception class suppressed)")
        return _problem(ApiProblem(500, "internal_error", "Capabilities are temporarily unavailable"))


async def handle_history(request: web.Request) -> web.Response:
    try:
        access = AccessContext.from_request(request)
        _db, conn = await _canonical_conn(request)
        filters, limit = _history_filters(request)
        request_digest = _digest(filters)
        state = _state(request)
        now_ms = int(time.time() * 1000)
        state.prune(now_ms)
        cursor_token = request.query.get("cursor")
        if cursor_token:
            cursor = state.decode(cursor_token)
            snapshot = state.history.get(str(cursor.get("s", "")))
            if snapshot is None:
                raise ApiProblem(409, "cursor_stale", "The history snapshot expired; refresh to continue", reason="snapshot_missing")
            if snapshot.expires_at_ms <= now_ms:
                state.history.pop(snapshot.snapshot_id, None)
                raise ApiProblem(409, "cursor_stale", "The history snapshot expired; refresh to continue", reason="expired")
            if snapshot.tenant_id != access.tenant_id or snapshot.principal_id != access.principal_id:
                raise ApiProblem(409, "cursor_stale", "The history snapshot is not valid for this account", reason="acl_changed")
            if snapshot.request_digest != request_digest:
                raise ApiProblem(409, "cursor_stale", "History filters changed; refresh to continue", reason="filter_mismatch")
            offset = int(cursor.get("o", -1))
            if offset < 0:
                raise ApiProblem(409, "cursor_stale", "The history cursor is invalid", reason="invalid_signature")
        else:
            rows = await _activity_rows(conn, access, filters)
            revision_row = await (
                await conn.execute("SELECT history_revision FROM operational_storage_state WHERE singleton_id=1")
            ).fetchone()
            revision = str(int(revision_row[0] if revision_row else 0))
            snapshot_id = secrets.token_hex(16)
            snapshot = _HistorySnapshot(
                snapshot_id,
                access.tenant_id,
                access.principal_id,
                request_digest,
                revision,
                tuple(str(row["activity_id"]) for row in rows),
                now_ms + _SNAPSHOT_TTL_MS,
            )
            state.put_history(snapshot)
            offset = 0

        selected_ids = snapshot.activity_ids[offset : offset + limit]
        selected: list[Any] = []
        for activity_id in selected_ids:
            row = await (
                await conn.execute(
                    "SELECT a.*, COALESCE(s.completeness, 'unknown') AS completeness "
                    "FROM activity_items a LEFT JOIN sessions_v2 s "
                    "ON a.resource_type='session' AND s.id=a.resource_id AND s.tenant_id=a.tenant_id "
                    "WHERE a.activity_id=? AND a.deleted_at_ms IS NULL",
                    (activity_id,),
                )
            ).fetchone()
            # ACL is deliberately rechecked on continuation, after revocation.
            if row is not None and await resource_is_visible(conn, row, access):
                selected.append(row)
        next_offset = offset + len(selected_ids)
        has_more = next_offset < len(snapshot.activity_ids)
        next_cursor = state.encode({"k": "history", "s": snapshot.snapshot_id, "o": next_offset}) if has_more else None
        return _json(
            {
                "items": [_activity_json(row) for row in selected],
                "next_cursor": next_cursor,
                "has_more": has_more,
                "revision": snapshot.revision,
                "snapshot": {
                    "snapshot_id": snapshot.snapshot_id,
                    "revision": snapshot.revision,
                    "expires_at": _iso(snapshot.expires_at_ms),
                },
            }
        )
    except PermissionError:
        return _problem(ApiProblem(401, "unauthorized", "Authentication required"))
    except ApiProblem as exc:
        return _problem(exc)
    except Exception:
        logger.error("operational history failed (exception class suppressed)")
        return _problem(ApiProblem(500, "internal_error", "History is temporarily unavailable"))


async def _session_acl_row(conn: Any, session_id: str):
    return await (
        await conn.execute(
            "SELECT id AS resource_id, 'session' AS resource_type, tenant_id, "
            "owner_principal_id, visibility, acl_version FROM sessions_v2 "
            "WHERE id=? AND deleted_at_ms IS NULL",
            (session_id,),
        )
    ).fetchone()


def _message_json(row: Any) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "session_id": str(row["session_id"]),
        "run_id": str(row["run_id"]) if row["run_id"] else None,
        "ordinal": int(row["sequence"]),
        "role": str(row["role"]),
        "status": str(row["status"]),
        "author": {
            "kind": str(row["author_kind"]),
            "principal_id": str(row["author_principal_id"]) if row["author_principal_id"] else None,
            "handle": str(row["author_handle_snapshot"]) if row["author_handle_snapshot"] else None,
            "display": str(row["author_display"]) if row["author_display"] else None,
        },
        "text": str(row["text"] or ""),
        # Provider ``redacted_thinking`` is opaque/provider-private. There is
        # currently no explicitly public reasoning column in the canonical
        # projection, so fail closed rather than mislabeling that blob.
        "visible_reasoning": None,
        "tool_invocation_id": str(row["resolved_tool_id"]) if row["resolved_tool_id"] else None,
        "attachments": [],
        "created_at": _iso(row["created_at_ms"]),
        "completeness": str(row["completeness"]),
    }


async def handle_session_messages(request: web.Request) -> web.Response:
    try:
        access = AccessContext.from_request(request)
        _db, conn = await _canonical_conn(request)
        session_id = str(request.match_info.get("session_id") or request.match_info.get("sessionId") or "")
        session = await _session_acl_row(conn, session_id)
        if session is None or not await resource_is_visible(conn, session, access):
            raise ApiProblem(404, "target_not_found", "This result is no longer available")
        around = request.query.get("around")
        cursor_token = request.query.get("cursor")
        direction = request.query.get("direction")
        if around and cursor_token:
            raise ApiProblem(400, "invalid_request", "around cannot be combined with cursor")
        if cursor_token and direction not in {"before", "after"}:
            raise ApiProblem(400, "invalid_request", "direction is required with cursor")
        if not cursor_token and direction:
            raise ApiProblem(400, "invalid_request", "direction requires cursor")
        if not around and ("before" in request.query or "after" in request.query):
            raise ApiProblem(400, "invalid_request", "before and after require around")
        state = _state(request)
        base_select = (
            "SELECT m.*, (SELECT t.id FROM tool_invocations t "
            "WHERE t.root_kind='session' AND t.session_id=m.session_id "
            "AND t.session_run_id=m.run_id AND t.tool_call_id=m.tool_call_id "
            "LIMIT 1) AS resolved_tool_id "
            "FROM session_messages m WHERE m.session_id=? AND m.visibility='user_visible' "
        )
        anchor_found: bool | None = None
        anchor_id: str | None = None
        if around:
            before = _bounded_int(request.query.get("before"), name="before", default=30, minimum=0, maximum=100)
            after = _bounded_int(request.query.get("after"), name="after", default=30, minimum=0, maximum=100)
            anchor = await (
                await conn.execute(
                    "SELECT sequence FROM session_messages WHERE session_id=? AND id=? AND visibility='user_visible'",
                    (session_id, around),
                )
            ).fetchone()
            if anchor is None:
                rows = []
                anchor_found = False
            else:
                anchor_found, anchor_id = True, str(around)
                seq = int(anchor[0])
                prior = await (await conn.execute(base_select + "AND m.sequence<=? ORDER BY m.sequence DESC LIMIT ?", (session_id, seq, before + 1))).fetchall()
                later = await (await conn.execute(base_select + "AND m.sequence>? ORDER BY m.sequence ASC LIMIT ?", (session_id, seq, after))).fetchall()
                rows = [*reversed(prior), *later]
        elif cursor_token:
            payload = state.decode(cursor_token)
            if payload.get("k") != "messages" or payload.get("s") != session_id or payload.get("p") != access.principal_id:
                raise ApiProblem(409, "cursor_stale", "The message cursor is not valid for this account", reason="acl_changed")
            boundary = int(payload.get("q", -1))
            limit = _bounded_int(request.query.get("limit"), name="limit", default=100, minimum=1, maximum=100)
            operator, ordering = ("<", "DESC") if direction == "before" else (">", "ASC")
            rows = await (await conn.execute(base_select + f"AND m.sequence{operator}? ORDER BY m.sequence {ordering} LIMIT ?", (session_id, boundary, limit))).fetchall()
            if direction == "before":
                rows = list(reversed(rows))
        else:
            limit = _bounded_int(request.query.get("limit"), name="limit", default=100, minimum=1, maximum=100)
            rows = await (await conn.execute(base_select + "ORDER BY m.sequence DESC LIMIT ?", (session_id, limit))).fetchall()
            rows = list(reversed(rows))

        min_seq = int(rows[0]["sequence"]) if rows else None
        max_seq = int(rows[-1]["sequence"]) if rows else None
        has_before = bool(min_seq is not None and (await (await conn.execute("SELECT 1 FROM session_messages WHERE session_id=? AND visibility='user_visible' AND sequence<? LIMIT 1", (session_id, min_seq))).fetchone()))
        has_after = bool(max_seq is not None and (await (await conn.execute("SELECT 1 FROM session_messages WHERE session_id=? AND visibility='user_visible' AND sequence>? LIMIT 1", (session_id, max_seq))).fetchone()))
        before_cursor = state.encode({"k": "messages", "s": session_id, "p": access.principal_id, "q": min_seq}) if has_before else None
        after_cursor = state.encode({"k": "messages", "s": session_id, "p": access.principal_id, "q": max_seq}) if has_after else None
        revision = await (await conn.execute("SELECT history_revision FROM operational_storage_state WHERE singleton_id=1")).fetchone()
        return _json(
            {
                "session_id": session_id,
                "messages": [_message_json(row) for row in rows],
                "anchor_found": anchor_found,
                "anchor_message_id": anchor_id,
                "before_cursor": before_cursor,
                "after_cursor": after_cursor,
                "has_more_before": has_before,
                "has_more_after": has_after,
                "revision": str(int(revision[0] if revision else 0)),
            }
        )
    except PermissionError:
        return _problem(ApiProblem(401, "unauthorized", "Authentication required"))
    except ApiProblem as exc:
        return _problem(exc)
    except Exception:
        logger.error("operational messages failed (exception class suppressed)")
        return _problem(ApiProblem(500, "internal_error", "Messages are temporarily unavailable"))


def _safe_shape(raw: str | None) -> Any:
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {"type": "opaque", "size": len(str(raw))}

    def shape(item: Any, depth: int = 0) -> Any:
        if depth >= 6:
            return {"type": "nested", "redacted": True}
        if isinstance(item, dict):
            return {str(key)[:128]: shape(value, depth + 1) for key, value in list(item.items())[:100]}
        if isinstance(item, list):
            return {"type": "array", "size": len(item), "items": [shape(value, depth + 1) for value in item[:10]]}
        if isinstance(item, str):
            return {"type": "string", "size": len(item)}
        if item is None:
            return None
        return {"type": type(item).__name__}

    return shape(value)


async def handle_tool_invocation(request: web.Request) -> web.Response:
    try:
        access = AccessContext.from_request(request)
        _db, conn = await _canonical_conn(request)
        tool_id = str(request.match_info.get("tool_id") or request.match_info.get("toolInvocationId") or "")
        row = await (
            await conn.execute(
                "SELECT t.*, 'tool_invocation' AS resource_type, t.id AS resource_id, "
                "s.tenant_id AS parent_tenant_id, "
                "s.owner_principal_id AS parent_owner_principal_id, "
                "s.visibility AS parent_visibility, s.acl_version AS parent_acl_version, "
                "(SELECT m.id FROM session_messages m WHERE m.session_id=t.session_id "
                "AND t.root_kind='session' AND m.run_id=t.session_run_id "
                "AND m.tool_call_id=t.tool_call_id "
                "ORDER BY m.sequence, m.id LIMIT 1) AS message_id "
                "FROM tool_invocations t JOIN sessions_v2 s ON s.id=t.session_id "
                "AND s.tenant_id=t.tenant_id WHERE t.id=? "
                "AND t.root_kind='session' AND s.deleted_at_ms IS NULL",
                (tool_id,),
            )
        ).fetchone()
        parent_acl = None
        if row is not None:
            parent_acl = {
                "tenant_id": row["parent_tenant_id"],
                "owner_principal_id": row["parent_owner_principal_id"],
                "visibility": row["parent_visibility"],
                "acl_version": row["parent_acl_version"],
                "resource_type": "session",
                "resource_id": row["session_id"],
            }
        if row is None or not await resource_is_visible(conn, parent_acl, access):
            raise ApiProblem(404, "target_not_found", "This result is no longer available")
        root_kind = "delegated_session" if row["root_kind"] == "session" and row["child_session_id"] else ("chat" if row["root_kind"] == "session" else str(row["root_kind"]))
        artifacts = await (
            await conn.execute(
                "SELECT a.id, a.kind, a.original_filename, a.mime, a.size_bytes "
                "FROM artifact_links l JOIN artifacts a ON a.id=l.artifact_id "
                "WHERE l.resource_type='tool_invocation' AND l.resource_id=? "
                "AND a.deleted_at_ms IS NULL",
                (tool_id,),
            )
        ).fetchall()
        return _json(
            {
                "id": tool_id,
                "tool_call_id": str(row["tool_call_id"]) if row["tool_call_id"] else None,
                "root_kind": root_kind,
                "root_id": str(row["root_id"]),
                "session_id": str(row["session_id"]) if row["session_id"] else None,
                "message_id": str(row["message_id"]) if row["message_id"] else None,
                "workflow_run_id": str(row["workflow_run_id"]) if row["workflow_run_id"] else None,
                "trace_step_id": str(row["workflow_step_id"]) if row["workflow_step_id"] else None,
                "scheduled_run_id": str(row["task_run_id"]) if row["task_run_id"] else None,
                "event_delivery_id": str(row["event_delivery_id"]) if row["event_delivery_id"] else None,
                "tool_server": str(row["tool_server"]) if row["tool_server"] else None,
                "tool_name": str(row["tool_name"]),
                "status": str(row["status"]),
                "args_safe": _safe_shape(row["args_json"]),
                "result_safe": _safe_shape(row["result_json"]),
                "error_safe": "Tool execution failed; sensitive details are redacted" if row["error_json"] or row["error_text"] else None,
                "child_session_id": str(row["child_session_id"]) if row["child_session_id"] else None,
                "sensitivity": "safe" if row["sensitivity"] == "normal" else "redacted",
                "completeness": str(row["completeness"]),
                "artifacts": [
                    {"id": str(a[0]), "kind": str(a[1]), "filename": str(a[2] or "artifact"), "mime": str(a[3] or "application/octet-stream"), "size_bytes": int(a[4])}
                    for a in artifacts
                ],
                "created_at": _iso(row["created_at_ms"]),
                "finished_at": _iso(row["finished_at_ms"]),
            }
        )
    except PermissionError:
        return _problem(ApiProblem(401, "unauthorized", "Authentication required"))
    except ApiProblem as exc:
        return _problem(exc)
    except Exception:
        logger.error("operational tool resolver failed (exception class suppressed)")
        return _problem(ApiProblem(500, "internal_error", "Tool detail is temporarily unavailable"))


def _search_target(row: dict[str, Any]) -> dict[str, Any]:
    from src.memory.operational.service import search_target

    try:
        return search_target(row)
    except RuntimeError as exc:
        raise ApiProblem(
            500,
            "internal_error",
            "Search target is unavailable",
        ) from exc


def _fragments(highlighted: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    buffer: list[str] = []
    marked = False
    for char in highlighted:
        if char in {"\x01", "\x02"}:
            if buffer:
                parts.append({"text": "".join(buffer), "highlight": marked})
                buffer = []
            marked = char == "\x01"
        else:
            buffer.append(char)
    if buffer:
        parts.append({"text": "".join(buffer), "highlight": marked})
    return parts or [{"text": "", "highlight": False}]


def _search_match(row: dict[str, Any], highlighted: str) -> dict[str, Any]:
    author = None
    if row.get("author_principal_id") or row.get("author_display_safe"):
        principal = str(row.get("author_principal_id") or "")
        author = {
            "kind": "agent" if principal.startswith("agent:") else "user",
            "principal_id": principal or None,
            "handle": None,
            "display": str(row.get("author_display_safe") or "") or None,
        }
    return {
        "kind": str(row["match_kind"]),
        "id": str(row["chunk_id"]),
        "field": str(row["source_field"]),
        "author": author,
        "occurred_at": _iso(row["occurred_at_ms"]),
        "fragments": _fragments(highlighted),
        "sensitivity": str(row["sensitivity"]),
        "completeness": str(row["completeness"]),
        "target": _search_target(row),
    }


_SEARCH_SNAPSHOT_FIELDS = (
    "tenant_id", "owner_principal_id", "visibility", "acl_version",
    "document_kind", "resource_type", "resource_id", "root_kind", "root_id",
    "parent_type", "parent_id", "session_id", "session_run_id", "target_kind",
    "message_id", "tool_invocation_id", "workflow_id", "workflow_run_id",
    "workflow_node_id", "workflow_trace_step_id", "scheduled_task_id",
    "scheduled_run_id", "event_id", "event_delivery_id", "definition_field",
    "caused_by_event_id", "caused_by_delivery_id", "status", "origin",
    "author_principal_id", "title_safe", "author_display_safe", "occurred_at_ms",
    "source_version", "sensitivity", "completeness", "content_hash", "chunk_id",
    "match_kind", "source_field", "_candidate_truncated",
)


def _compact_search_row(row: dict[str, Any]) -> dict[str, Any]:
    """Store only resolver/ACL/version metadata, never snippets or FTS rowids."""

    return {field: row.get(field) for field in _SEARCH_SNAPSHOT_FIELDS}


async def _search_row_visible(conn: Any, row: dict[str, Any], access: AccessContext) -> bool:
    from src.memory.operational.service import search_row_visible

    return await search_row_visible(conn, row, access)


async def _granted_search_resources(
    conn: Any, access: AccessContext
) -> tuple[tuple[str, str, int], ...]:
    from src.memory.operational.service import granted_search_resources

    return await granted_search_resources(conn, access)


async def handle_search(request: web.Request) -> web.Response:
    """POST search with fail-closed logging and principal-bound snapshots."""

    try:
        access = AccessContext.from_request(request)
        if request.content_length is not None and request.content_length > 65_536:
            raise ApiProblem(413, "request_too_large", "The request exceeds the supported size")
        try:
            body = await request.json()
        except Exception as exc:
            raise ApiProblem(400, "invalid_request", "The search body must be valid JSON") from exc
        if not isinstance(body, dict):
            raise ApiProblem(400, "invalid_request", "The search body must be an object")
        allowed_keys = {"query", "scopes", "filters", "sort", "grouping", "limit", "cursor"}
        if set(body) - allowed_keys:
            raise ApiProblem(400, "invalid_request", "The search body contains unsupported fields")
        if not isinstance(body.get("query", ""), str):
            raise ApiProblem(400, "invalid_request", "Search query must be a string")
        query = body.get("query", "")
        scopes = body.get("scopes")
        if not isinstance(scopes, list) or not scopes or len(scopes) > 5 or any(
            not isinstance(scope, str)
            or scope not in {"chats", "tools", "workflows", "scheduled", "events"}
            for scope in scopes
        ):
            raise ApiProblem(400, "invalid_request", "At least one supported search scope is required")
        scopes = sorted(set(scopes))
        sort = str(body.get("sort") or "relevance")
        grouping = str(body.get("grouping") or "root")
        if sort not in {"relevance", "recent"} or grouping not in {"root", "match"}:
            raise ApiProblem(400, "invalid_request", "Search sort or grouping is invalid")
        if sort == "relevance" and not query.strip():
            raise ApiProblem(422, "unprocessable_query", "A query is required for relevance search")
        filters = body.get("filters") or {}
        if not isinstance(filters, dict):
            raise ApiProblem(400, "invalid_request", "Search filters must be an object")
        if set(filters) - {"status", "from", "to", "parent_type", "parent_id", "origin", "root"}:
            raise ApiProblem(400, "invalid_request", "Search filters contain unsupported fields")
        raw_statuses = filters.get("status")
        if raw_statuses is not None and (
            not isinstance(raw_statuses, list)
            or len(raw_statuses) > len(_RUN_STATUSES)
            or any(not isinstance(value, str) or value not in _RUN_STATUSES for value in raw_statuses)
        ):
            raise ApiProblem(400, "invalid_request", "filters.status must be an array of run statuses")
        if raw_statuses is not None:
            filters = dict(filters)
            filters["status"] = sorted(set(raw_statuses))
        parent_type, parent_id = filters.get("parent_type"), filters.get("parent_id")
        if (parent_type is None) != (parent_id is None):
            raise ApiProblem(400, "invalid_request", "filters.parent_type and filters.parent_id must be used together")
        if parent_type is not None and (
            not isinstance(parent_type, str)
            or parent_type not in {"session", "workflow", "scheduled_task", "event"}
        ):
            raise ApiProblem(400, "invalid_request", "filters.parent_type is invalid")
        if parent_id is not None and (not isinstance(parent_id, str) or not 1 <= len(parent_id) <= 512):
            raise ApiProblem(400, "invalid_request", "filters.parent_id is invalid")
        origin_filter = filters.get("origin")
        if origin_filter is not None and (not isinstance(origin_filter, str) or not 1 <= len(origin_filter) <= 128):
            raise ApiProblem(400, "invalid_request", "filters.origin is invalid")
        root_filter = filters.get("root")
        if root_filter is not None:
            if (
                not isinstance(root_filter, dict)
                or set(root_filter) != {"kind", "id"}
                or root_filter.get("kind") not in {
                    "chat", "delegated_session", "workflow_definition", "workflow_run",
                    "scheduled_definition", "scheduled_run", "event_definition", "event_delivery",
                }
                or not isinstance(root_filter.get("id"), str)
                or not 1 <= len(root_filter["id"]) <= 512
            ):
                raise ApiProblem(400, "invalid_request", "filters.root is invalid")
        for date_key in ("from", "to"):
            if filters.get(date_key) is not None and not isinstance(filters[date_key], str):
                raise ApiProblem(400, "invalid_request", f"filters.{date_key} must be a timestamp string")
        parsed_from = _parse_iso(filters.get("from"), name="filters.from")
        parsed_to = _parse_iso(filters.get("to"), name="filters.to")
        if parsed_from is not None and parsed_to is not None and parsed_from >= parsed_to:
            raise ApiProblem(400, "invalid_request", "filters.from must be earlier than filters.to")
        limit = body.get("limit", 40)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > 100:
            raise ApiProblem(400, "invalid_request", "Search limit is outside the supported range")
        raw_cursor = body.get("cursor")
        if raw_cursor is not None and (
            not isinstance(raw_cursor, str) or not 16 <= len(raw_cursor) <= 8192
        ):
            raise ApiProblem(400, "invalid_request", "Search cursor is invalid")
        try:
            from src.memory.operational.search import (
                literal_fts_query,
                read_authorized_highlight,
                read_search_rows,
            )

            fts_query = literal_fts_query(query)
        except ValueError as exc:
            raise ApiProblem(422, "unprocessable_query", "The query exceeds the supported complexity") from exc
        if sort == "relevance" and not fts_query:
            raise ApiProblem(
                422,
                "unprocessable_query",
                "The query must contain at least one searchable term",
            )
        db, conn = await _canonical_conn(request)
        cursor_token = body.get("cursor")
        try:
            raw_status = await _current_search_status(
                db,
                conn,
                gateway=request.app["gateway"],
                require_outbox_head=not bool(cursor_token),
            )
            index_status = type("IndexStatus", (), raw_status)()
        except Exception:
            # Do not log exception text: SQLite parser failures may contain a
            # fragment derived from the private query.
            logger.error("operational search indexing failed (details suppressed)")
            raise ApiProblem(503, "degraded", "Search is temporarily degraded")
        if not index_status.ready:
            _start_background(request, db)
            raise ApiProblem(503, "warming", "Search is still indexing authorized history")

        normalized_request = {
            "query_digest": hashlib.sha256(query.encode()).hexdigest(),
            "scopes": scopes,
            "filters": filters,
            "sort": sort,
            "grouping": grouping,
        }
        request_digest = _digest(normalized_request)
        state = _state(request)
        now_ms = int(time.time() * 1000)
        state.prune(now_ms)
        if cursor_token:
            cursor = state.decode(str(cursor_token))
            snapshot = state.search.get(str(cursor.get("s", "")))
            if snapshot is None:
                raise ApiProblem(409, "cursor_stale", "The search snapshot expired; refresh to continue", reason="snapshot_missing")
            if snapshot.expires_at_ms <= now_ms:
                raise ApiProblem(409, "cursor_stale", "The search snapshot expired; refresh to continue", reason="expired")
            if snapshot.principal_id != access.principal_id or snapshot.tenant_id != access.tenant_id:
                raise ApiProblem(409, "cursor_stale", "The search snapshot is not valid for this account", reason="acl_changed")
            if snapshot.request_digest != request_digest:
                raise ApiProblem(409, "cursor_stale", "Search filters changed; refresh to continue", reason="filter_mismatch")
            if snapshot.index_generation != index_status.generation:
                raise ApiProblem(409, "cursor_stale", "The search index changed incompatibly", reason="generation_changed")
            offset = int(cursor.get("o", -1))
            if offset < 0:
                raise ApiProblem(409, "cursor_stale", "The search cursor is invalid", reason="invalid_signature")
        else:
            granted_resources = await _granted_search_resources(conn, access)
            candidates = await __import__("asyncio").to_thread(
                read_search_rows,
                index_status.path,
                fts_query=fts_query,
                scopes=scopes,
                sort=sort,
                tenant_id=access.tenant_id,
                principal_ids=access.principal_ids,
                granted_resources=granted_resources,
                filters={
                    "status": tuple(filters.get("status") or ()),
                    "from_ms": parsed_from,
                    "to_ms": parsed_to,
                    "origin": filters.get("origin"),
                    "parent_type": filters.get("parent_type"),
                    "parent_id": filters.get("parent_id"),
                    "root": filters.get("root"),
                },
                max_candidates=50_001,
            )
            if len(candidates) > 50_000:
                raise ApiProblem(
                    422, "unprocessable_query",
                    "The query is too broad for a complete safe snapshot; add a filter",
                    reason="authorized_candidate_limit",
                    details={"estimated_total": ">50000", "candidate_limit": 50000},
                )
            candidate_truncated = False
            from_ms = parsed_from
            to_ms = parsed_to
            wanted_status = set(filters.get("status") or [])
            root_filter = filters.get("root") or None
            authorized: list[dict[str, Any]] = []
            for row in candidates:
                if row["tenant_id"] != access.tenant_id:
                    continue
                if wanted_status and row.get("status") not in wanted_status:
                    continue
                if from_ms is not None and int(row["occurred_at_ms"]) < from_ms:
                    continue
                if to_ms is not None and int(row["occurred_at_ms"]) >= to_ms:
                    continue
                if filters.get("origin") and row.get("origin") != filters["origin"]:
                    continue
                if filters.get("parent_type") and (row.get("parent_type"), row.get("parent_id")) != (filters.get("parent_type"), filters.get("parent_id")):
                    continue
                if root_filter and (row.get("root_kind"), row.get("root_id")) != (root_filter.get("kind"), root_filter.get("id")):
                    continue
                if await _search_row_visible(conn, row, access):
                    row["_candidate_truncated"] = candidate_truncated
                    authorized.append(_compact_search_row(row))
            if grouping == "root":
                grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
                for row in authorized:
                    grouped.setdefault((str(row["root_kind"]), str(row["root_id"])), []).append(row)
                snapshot_rows: list[dict[str, Any]] = []
                for rows in grouped.values():
                    best = dict(rows[0])
                    best["_matches"] = [dict(row) for row in rows]
                    best["_match_count"] = len(rows)
                    snapshot_rows.append(best)
            else:
                snapshot_rows = []
                for row in authorized:
                    item = dict(row)
                    item["_matches"] = [dict(row)]
                    item["_match_count"] = 1
                    snapshot_rows.append(item)
            snapshot = _SearchSnapshot(
                secrets.token_hex(16), access.tenant_id, access.principal_id,
                request_digest, index_status.generation, index_status.seq,
                tuple(snapshot_rows), now_ms + _SNAPSHOT_TTL_MS,
            )
            state.put_search(snapshot)
            offset = 0

        output: list[dict[str, Any]] = []
        visible_page_matches: list[dict[str, Any]] = []
        next_offset = offset
        while next_offset < len(snapshot.rows) and len(output) < limit:
            row = snapshot.rows[next_offset]
            next_offset += 1
            current_matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
            visible_match_count = 0
            for match_row in row["_matches"]:
                if not await _search_row_visible(conn, match_row, access):
                    continue
                visible_match_count += 1
                # The wire contract exposes at most two snippets per root. Keep
                # counting authorized matches for an exact match_count, but do
                # not open/highlight the FTS database tens of thousands of times
                # for content that will never be serialized.
                if len(current_matches) >= 2:
                    continue
                highlighted = await asyncio.to_thread(
                    read_authorized_highlight,
                    index_status.path,
                    str(match_row["chunk_id"]),
                    int(match_row["source_version"]),
                    str(match_row["content_hash"]),
                    fts_query,
                )
                if highlighted is None:
                    continue
                current_matches.append((match_row, _search_match(match_row, highlighted)))
            if not current_matches:
                continue
            visible_page_matches.extend(item[0] for item in current_matches)
            root_row = current_matches[0][0]
            matches = [item[1] for item in current_matches[:2]]
            root = {
                "kind": str(root_row["root_kind"]),
                "id": str(root_row["root_id"]),
                "title": str(root_row.get("title_safe") or "Untitled result"),
                "status": str(root_row["status"]) if root_row.get("status") else None,
                "occurred_at": _iso(root_row["occurred_at_ms"]),
                "session_id": str(root_row["session_id"]) if root_row.get("session_id") else None,
                "parent": None,
                "completeness": str(root_row["completeness"]),
            }
            output.append(
                {
                    "result_id": f"{root_row['root_kind']}:{root_row['root_id']}" if grouping == "root" else str(root_row["chunk_id"]),
                    "root": root,
                    "matches": matches,
                    "match_count": visible_match_count,
                    "target": matches[0]["target"],
                    "caused_by": None,
                }
            )
        has_more = next_offset < len(snapshot.rows)
        next_cursor = state.encode({"k": "search", "s": snapshot.snapshot_id, "o": next_offset}) if has_more else None
        corpus_map = {"chats": {"session_metadata", "message"}, "tools": {"tool_invocation"}, "workflows": {"workflow_definition", "workflow_run", "workflow_step"}, "scheduled": {"scheduled_definition", "scheduled_run"}, "events": {"event_definition", "event_delivery"}}
        per_corpus: dict[str, Any] = {}
        truncated = any(bool(row.get("_candidate_truncated")) for row in visible_page_matches)
        for scope in scopes:
            count = sum(1 for row in visible_page_matches if row["document_kind"] in corpus_map[scope])
            per_corpus[scope] = {"complete": not truncated, "indexed_documents": count, "estimated_total": count, "pending": 1 if truncated else 0, "lag_ms": 0}
        coverage = {
            "state": "ready", "complete": not truncated,
            "indexed_documents": len(visible_page_matches), "estimated_total": len(visible_page_matches),
            "pending": 1 if truncated else 0, "indexed_through": _iso(index_status.indexed_through_ms),
            "lag_ms": max(0, now_ms - int(index_status.indexed_through_ms or now_ms)),
            "last_error": None, "per_corpus": per_corpus,
        }
        return _json(
            {
                "items": output,
                "next_cursor": next_cursor,
                "has_more": has_more,
                "snapshot": {
                    "search_session_id": snapshot.snapshot_id,
                    "index_generation": snapshot.index_generation,
                    "indexed_seq": snapshot.indexed_seq,
                    "expires_at": _iso(snapshot.expires_at_ms),
                },
                "index_generation": snapshot.index_generation,
                "indexed_seq": snapshot.indexed_seq,
                "coverage": coverage,
                "query_mode": "keyword",
            }
        )
    except PermissionError:
        return _problem(ApiProblem(401, "unauthorized", "Authentication required"))
    except ApiProblem as exc:
        return _problem(exc)
    except Exception:
        # This is deliberately not logger.exception and contains no exception
        # formatting: request bodies and query terms cannot reach the global
        # CORS/error middleware or logs.
        logger.error("operational search request failed (details suppressed)")
        return _problem(ApiProblem(500, "internal_error", "Search is temporarily unavailable"))


async def _automation_acl_row(
    conn: Any,
    access: AccessContext,
    resource_type: str,
    resource_id: str,
    parent: tuple[str, str] | None = None,
) -> dict[str, Any]:
    owner = await (
        await conn.execute(
            "SELECT owner_principal_id, visibility, acl_version, provenance "
            "FROM operational_resource_owners WHERE tenant_id=? AND resource_type=? AND resource_id=?",
            (access.tenant_id, resource_type, resource_id),
        )
    ).fetchone()
    if owner is None and parent is not None:
        owner = await (
            await conn.execute(
                "SELECT owner_principal_id, visibility, acl_version, provenance "
                "FROM operational_resource_owners WHERE tenant_id=? AND resource_type=? AND resource_id=?",
                (access.tenant_id, parent[0], parent[1]),
            )
        ).fetchone()
    return {
        "tenant_id": access.tenant_id,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "owner_principal_id": owner[0] if owner else None,
        "visibility": str(owner[1]) if owner else "installation_shared",
        "acl_version": int(owner[2]) if owner else 1,
        "provenance": str(owner[3]) if owner else "legacy_unattributed",
    }


def _normalize_public_status(raw: Any) -> tuple[str, bool]:
    try:
        from src.memory.operational.enums import normalize_run_status

        return normalize_run_status(raw)[0], True
    except Exception:
        return "failed", False


def _workflow_trace_steps(run_id: str, trace: Any, *, run_started_ms: int) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    attempts: dict[str, int] = {}
    if not isinstance(trace, list):
        return steps
    for raw in trace:
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("node_id") or raw.get("id") or "unknown")
        attempt = int(raw.get("attempt") or attempts.get(node_id, 0))
        attempts[node_id] = attempt + 1
        stable_id = str(raw.get("id") or raw.get("trace_step_id") or f"trace:{run_id}:{node_id}:{attempt}")
        status, _mapped = _normalize_public_status(raw.get("status") or "pending")
        started = raw.get("started_at")
        finished = raw.get("finished_at")
        tool_ids = raw.get("tool_invocation_ids") or []
        if raw.get("tool_invocation_id"):
            tool_ids = [*tool_ids, raw["tool_invocation_id"]]
        steps.append(
            {
                "id": stable_id,
                "node_id": node_id,
                "attempt": attempt,
                "type": str(raw.get("type") or "unknown"),
                "status": status,
                "child_session_id": str(raw["child_session_id"]) if raw.get("child_session_id") else None,
                "error_safe": "Workflow step failed; sensitive details are redacted" if raw.get("error") else None,
                "started_at": _iso(int(float(started) * 1000)) if started is not None else _iso(run_started_ms),
                "finished_at": _iso(int(float(finished) * 1000)) if finished is not None else None,
                "tool_invocation_ids": [str(value) for value in tool_ids],
            }
        )
    return steps


async def decorate_workflow_run_detail(request: web.Request, row: dict[str, Any]) -> dict[str, Any] | None:
    access = AccessContext.from_request(request)
    _db, conn = await _canonical_conn(request)
    acl = await _automation_acl_row(
        conn, access, "workflow_run", str(row["id"]),
        ("workflow_definition", str(row["workflow_id"])),
    )
    if not await resource_is_visible(conn, acl, access):
        return None
    workflow = await (
        await conn.execute("SELECT name FROM workflow_tasks WHERE id=?", (row["workflow_id"],))
    ).fetchone()
    status, mapped = _normalize_public_status(row.get("status"))
    out = dict(row)  # legacy ``trace``/inputs/outputs remain additive-compatible
    started_ms = int(float(row["started_at"]) * 1000)
    finished_ms = int(float(row["finished_at"]) * 1000) if row.get("finished_at") else None
    out["started_at_epoch"] = row["started_at"]
    out["finished_at_epoch"] = row.get("finished_at")
    out["status_raw"] = row.get("status")
    out["status"] = status
    out["title"] = str(workflow[0] if workflow else f"Workflow run {str(row['id'])[:8]}")
    out["trace_steps"] = _workflow_trace_steps(str(row["id"]), row.get("trace"), run_started_ms=started_ms)
    # Existing app releases do numeric duration arithmetic on the base epoch
    # keys. Keep them stable and expose canonical ISO mirrors additively.
    out["started_at_iso"] = _iso(started_ms)
    out["finished_at_iso"] = _iso(finished_ms)
    out["completeness"] = "complete" if mapped and acl["provenance"] != "legacy_unattributed" else ("unknown" if mapped else "malformed_source")
    return out


async def handle_scheduled_run(request: web.Request) -> web.Response:
    try:
        access = AccessContext.from_request(request)
        _db, conn = await _canonical_conn(request)
        run_id = str(request.match_info.get("run_id") or request.match_info.get("runId") or "")
        row = await (
            await conn.execute(
                "SELECT r.*, t.name FROM task_runs r JOIN scheduled_tasks t ON t.id=r.task_id WHERE r.id=?",
                (run_id,),
            )
        ).fetchone()
        if row is None:
            raise ApiProblem(404, "target_not_found", "This result is no longer available")
        acl = await _automation_acl_row(
            conn, access, "scheduled_run", run_id,
            ("scheduled_definition", str(row["task_id"])),
        )
        if not await resource_is_visible(conn, acl, access):
            raise ApiProblem(404, "target_not_found", "This result is no longer available")
        status, mapped = _normalize_public_status(row["status"])
        return _json(
            {
                "id": run_id,
                "task_id": str(row["task_id"]),
                "title": str(row["name"]),
                "status": status,
                "trigger": str(row["trigger"]) if row["trigger"] else None,
                "session_id": str(row["session_id"]) if row["session_id"] else None,
                "output_summary_safe": _redacted_summary(row["output"]),
                "error_safe": "Scheduled execution failed; sensitive details are redacted" if row["error"] else None,
                "caused_by": None,
                "started_at": _iso(int(float(row["started_at"]) * 1000)),
                "finished_at": _iso(int(float(row["finished_at"]) * 1000)) if row["finished_at"] else None,
                "completeness": "complete" if mapped and acl["provenance"] != "legacy_unattributed" else ("unknown" if mapped else "malformed_source"),
            }
        )
    except PermissionError:
        return _problem(ApiProblem(401, "unauthorized", "Authentication required"))
    except ApiProblem as exc:
        return _problem(exc)
    except Exception:
        logger.error("scheduled run resolver failed (details suppressed)")
        return _problem(ApiProblem(500, "internal_error", "Scheduled detail is temporarily unavailable"))


def _redacted_summary(value: Any) -> str | None:
    if value is None:
        return None
    from src.memory.operational.search import redact_search_text

    return redact_search_text(value, limit=65_536)


async def decorate_event_delivery_detail(request: web.Request, row: dict[str, Any]) -> dict[str, Any] | None:
    access = AccessContext.from_request(request)
    _db, conn = await _canonical_conn(request)
    acl = await _automation_acl_row(
        conn, access, "event_delivery", str(row["id"]),
        ("event_definition", str(row["event_id"])),
    )
    if not await resource_is_visible(conn, acl, access):
        return None
    event = await (await conn.execute("SELECT name FROM events WHERE id=?", (row["event_id"],))).fetchone()
    status, mapped = _normalize_public_status(row.get("status"))
    downstream = None
    if row.get("workflow_run_id"):
        workflow_id = await (
            await conn.execute("SELECT workflow_id FROM workflow_runs WHERE id=?", (row["workflow_run_id"],))
        ).fetchone()
        if workflow_id:
            downstream = {"kind": "workflow_run", "run_id": str(row["workflow_run_id"]), "workflow_id": str(workflow_id[0])}
    elif row.get("task_run_id"):
        task_id = await (
            await conn.execute("SELECT task_id FROM task_runs WHERE id=?", (row["task_run_id"],))
        ).fetchone()
        if task_id:
            downstream = {"kind": "scheduled_run", "run_id": str(row["task_run_id"]), "task_id": str(task_id[0])}
    elif row.get("session_id"):
        downstream = {"kind": "chat", "session_id": str(row["session_id"])}
    out = dict(row)  # authorized legacy callers retain payload_json and links
    out["started_at_epoch"] = row["started_at"]
    out["finished_at_epoch"] = row.get("finished_at")
    out["status_raw"] = row.get("status")
    out["status"] = status
    out["title"] = str(event[0] if event else f"Event delivery {str(row['id'])[:8]}")
    out["downstream_target"] = downstream
    out["error_safe"] = "Event delivery failed; sensitive details are redacted" if row.get("error") else None
    out["occurred_at"] = _iso(int(float(row["started_at"]) * 1000))
    out["finished_at_iso"] = _iso(int(float(row["finished_at"]) * 1000)) if row.get("finished_at") else None
    out["completeness"] = "complete" if mapped and acl["provenance"] != "legacy_unattributed" else ("unknown" if mapped else "malformed_source")
    return out
