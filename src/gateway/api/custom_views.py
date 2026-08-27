"""Authenticated REST surface for OA-UI Custom Views.

Route registration intentionally lives in ``gateway.server``; this module is
kept side-effect free so tests and the in-process ui-manager can share the
same service without starting a web server or a source process.
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from aiohttp import web
from aiohttp.helpers import content_disposition_header

from src.custom_views.compiler import COMPONENTS, MAX_DEPTH, MAX_NODES
from src.custom_views.bundles import (
    MAX_ASSET_BYTES,
    MAX_BUNDLE_BYTES,
    MAX_BUNDLE_FILES,
    MAX_SCRIPT_BYTES,
    safe_relative_path,
)
from src.custom_views.repository import (
    CustomViewConflict,
    CustomViewError,
    CustomViewImmutable,
    CustomViewInputError,
    CustomViewNotFound,
    CustomViewRateLimited,
)
from src.custom_views.service import CustomViewService, service_for_gateway
from src.memory.operational.access import AccessContext


# A binary bundle can contain up to 64 MiB, and base64 expands it by 4/3.
# Keep the allowance route-scoped: ordinary REST/action bodies retain a much
# smaller ceiling instead of raising the gateway-wide aiohttp default.
MAX_CUSTOM_VIEW_REQUEST_BYTES = 96 * 1024 * 1024
MAX_CUSTOM_VIEW_SMALL_REQUEST_BYTES = 2 * 1024 * 1024


def _request_body_limit(request: web.Request) -> int | None:
    if request.method not in {"POST", "PUT", "DELETE"}:
        return None
    parts = tuple(part for part in request.path.split("/") if part)
    if parts[:3] != ("api", "ui", "views"):
        return None
    accepts_bundle = (
        (request.method == "POST" and len(parts) == 3)
        or (request.method == "PUT" and len(parts) == 4)
        or (
            request.method == "PUT"
            and len(parts) == 6
            and parts[4] == "sources"
        )
    )
    return (
        MAX_CUSTOM_VIEW_REQUEST_BYTES
        if accepts_bundle
        else MAX_CUSTOM_VIEW_SMALL_REQUEST_BYTES
    )


@web.middleware
async def body_limit_middleware(request: web.Request, handler: Callable) -> web.StreamResponse:
    """Apply Custom View JSON limits without weakening every gateway route."""

    limit = _request_body_limit(request)
    if limit is not None:
        request = request.clone(client_max_size=limit)
    return await handler(request)


def _problem(status: int, code: str, message: str) -> web.Response:
    return web.json_response(
        {"error": {"code": code, "message": message, "retryable": status in {429, 503}}},
        status=status,
        headers={"Cache-Control": "no-store"},
    )


def _map_error(exc: Exception) -> web.Response:
    if isinstance(exc, web.HTTPRequestEntityTooLarge):
        return _problem(413, "request_too_large", "Custom View request body is too large")
    if isinstance(exc, CustomViewNotFound):
        return _problem(404, "ui_view_not_found", "Custom View is not available")
    if isinstance(exc, CustomViewRateLimited):
        return _problem(429, "ui_action_rate_limited", str(exc))
    if isinstance(exc, CustomViewConflict):
        return _problem(409, "revision_conflict", str(exc))
    if isinstance(exc, CustomViewImmutable):
        return _problem(409, "immutable_view", str(exc))
    if isinstance(exc, CustomViewInputError):
        return _problem(400, "invalid_ui_view", str(exc))
    if isinstance(exc, PermissionError) and getattr(exc, "errno", None) is None:
        return _problem(401, "unauthorized", "Authentication is required")
    if isinstance(exc, OSError):
        return _problem(503, "ui_storage_unavailable", "Custom View storage is unavailable")
    if isinstance(exc, RuntimeError) and not isinstance(exc, CustomViewError):
        return _problem(503, "ui_runtime_unavailable", "Custom View runtime is unavailable")
    return _problem(500, "ui_view_error", "Custom View operation failed")


def _context(request: web.Request) -> tuple[CustomViewService, AccessContext]:
    return service_for_gateway(request.app["gateway"]), AccessContext.from_request(request)


async def _body(request: web.Request) -> dict[str, Any]:
    if not request.can_read_body:
        return {}
    try:
        value = await request.json()
    except web.HTTPRequestEntityTooLarge:
        raise
    except Exception as exc:
        raise CustomViewInputError("request body must be a JSON object") from exc
    if not isinstance(value, dict):
        raise CustomViewInputError("request body must be a JSON object")
    return value


async def _guard(
    operation: Callable[[], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    try:
        return await operation()
    except Exception as exc:  # handler boundary deliberately redacts unknowns
        return _map_error(exc)


def _optional_int(value: Any, *, field: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise CustomViewInputError(f"{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise CustomViewInputError(f"{field} must be an integer") from exc
    return parsed


def _etag(revision: int) -> str:
    return f'"ui-view-{int(revision)}"'


def _if_match_revision(request: web.Request) -> int | None:
    raw = request.headers.get("If-Match")
    if raw is None:
        return None
    value = raw.strip()
    if value.startswith("W/"):
        value = value[2:].strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        value = value[1:-1]
    if value.startswith("ui-view-"):
        value = value[len("ui-view-"):]
    return _optional_int(value, field="If-Match")


def _expected_revision(
    request: web.Request,
    body: Mapping[str, Any],
    *,
    required: bool = True,
) -> int | None:
    body_revision = _optional_int(
        body.get("expectedRevision", body.get("expected_revision")),
        field="expectedRevision",
    )
    header_revision = _if_match_revision(request)
    if (
        body_revision is not None and header_revision is not None
        and body_revision != header_revision
    ):
        raise CustomViewInputError("If-Match and expectedRevision disagree")
    expected = body_revision if body_revision is not None else header_revision
    if expected is not None and expected < 1:
        raise CustomViewInputError("expectedRevision must be a positive integer")
    if required and expected is None:
        raise CustomViewInputError("expectedRevision or If-Match is required")
    return expected


def _scripts(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or len(value) > MAX_BUNDLE_FILES:
        raise CustomViewInputError("scripts must map relative paths to UTF-8 source strings")
    result: dict[str, str] = {}
    total = 0
    for raw_name, payload in value.items():
        try:
            name = safe_relative_path(raw_name)
        except (TypeError, ValueError) as exc:
            raise CustomViewInputError("script path is invalid") from exc
        if not isinstance(payload, str) or len(payload) > MAX_SCRIPT_BYTES:
            raise CustomViewInputError("script exceeds the supported size")
        raw = payload.encode("utf-8")
        if len(raw) > MAX_SCRIPT_BYTES:
            raise CustomViewInputError("script exceeds the supported size")
        total += len(raw)
        if total > MAX_BUNDLE_BYTES:
            raise CustomViewInputError("scripts exceed the supported bundle size")
        result[name] = payload
    return result


def _assets(value: Any) -> dict[str, bytes] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or len(value) > MAX_BUNDLE_FILES:
        raise CustomViewInputError("assets must map relative paths to base64 payloads")
    max_encoded = 4 * ((MAX_ASSET_BYTES + 2) // 3)
    prepared: list[tuple[str, str]] = []
    total = 0
    for raw_name, raw in value.items():
        try:
            name = safe_relative_path(raw_name)
        except (TypeError, ValueError) as exc:
            raise CustomViewInputError("asset path is invalid") from exc
        encoded = raw.get("base64") if isinstance(raw, dict) else raw
        if not isinstance(encoded, str) or len(encoded) > max_encoded:
            raise CustomViewInputError("assets must map relative paths to base64 payloads")
        try:
            encoded.encode("ascii")
        except UnicodeEncodeError as exc:
            raise CustomViewInputError("asset base64 payload is invalid") from exc
        if len(encoded) % 4:
            raise CustomViewInputError("asset base64 payload is invalid")
        padding = 2 if encoded.endswith("==") else 1 if encoded.endswith("=") else 0
        predicted_size = (len(encoded) // 4) * 3 - padding
        if predicted_size > MAX_ASSET_BYTES:
            raise CustomViewInputError("asset exceeds the supported size")
        total += predicted_size
        if total > MAX_BUNDLE_BYTES:
            raise CustomViewInputError("assets exceed the supported bundle size")
        prepared.append((name, encoded))

    decoded: dict[str, bytes] = {}
    for name, encoded in prepared:
        try:
            payload = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise CustomViewInputError("asset base64 payload is invalid") from exc
        if len(payload) > MAX_ASSET_BYTES:
            raise CustomViewInputError("asset exceeds the supported size")
        decoded[name] = payload
    return decoded


def _cursor(offset: int) -> str:
    raw = json.dumps({"v": 1, "offset": offset}, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _offset(value: str | None) -> int:
    if not value:
        return 0
    try:
        padded = value + "=" * (-len(value) % 4)
        parsed = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
        offset = int(parsed["offset"])
    except Exception as exc:
        raise CustomViewInputError("cursor is invalid") from exc
    if parsed.get("v") != 1 or offset < 0:
        raise CustomViewInputError("cursor is invalid")
    return offset


async def handle_capabilities(request: web.Request) -> web.StreamResponse:
    async def run() -> web.StreamResponse:
        service, _access = _context(request)
        capabilities = {
            **service.capabilities,
            "version": 1,
            "customUiVersion": 1,
            "schemaVersions": [1],
            "realtime": True,
            "maxNodes": MAX_NODES,
            "maxDepth": MAX_DEPTH,
            "componentTypes": sorted(COMPONENTS),
        }
        return web.json_response(
            {"capabilities": capabilities}, headers={"Cache-Control": "private, max-age=60"},
        )

    return await _guard(run)


async def handle_list(request: web.Request) -> web.StreamResponse:
    async def run() -> web.StreamResponse:
        service, access = _context(request)
        limit = _optional_int(request.query.get("limit", "50"), field="limit") or 50
        offset = _offset(request.query.get("cursor"))
        views, has_more = await service.list(
            access,
            surface=request.query.get("surface") or None,
            session_id=request.query.get("sessionId") or request.query.get("session_id") or None,
            query=request.query.get("q") or request.query.get("query") or None,
            limit=limit,
            offset=offset,
        )
        next_cursor = _cursor(offset + len(views)) if has_more else None
        return web.json_response(
            {"views": views, "nextCursor": next_cursor},
            headers={"Cache-Control": "no-store"},
        )

    return await _guard(run)


async def handle_create(request: web.Request) -> web.StreamResponse:
    async def run() -> web.StreamResponse:
        service, access = _context(request)
        body = await _body(request)
        view = await service.create(
            access,
            surface=body.get("surface"),
            title=body.get("title"),
            description=body.get("description", ""),
            icon=body.get("icon"),
            markup=body.get("markup"),
            spec=body.get("spec"),
            session_id=body.get("sessionId", body.get("session_id")),
            expires_at=body.get("expiresAt", body.get("expires_at")),
            visibility=body.get("visibility", "private"),
            sources=body.get("sources"),
            actions=body.get("actions"),
            initial_data=body.get("initialData", body.get("initial_data")),
            scripts=_scripts(body.get("scripts")),
            assets=_assets(body.get("assets")),
            sidebar_order=body.get("sidebarOrder", body.get("sidebar_order", 0)),
            sidebar_group=body.get("sidebarGroup", body.get("sidebar_group")),
            frozen=bool(body.get("frozen", False)),
        )
        return web.json_response(
            {"view": view}, status=201,
            headers={"Cache-Control": "no-store", "ETag": _etag(view["revision"])},
        )

    return await _guard(run)


async def handle_get(request: web.Request) -> web.StreamResponse:
    async def run() -> web.StreamResponse:
        service, access = _context(request)
        revision = _optional_int(request.query.get("revision"), field="revision")
        view = await service.get(request.match_info["id"], access, revision=revision)
        return web.json_response(
            {"view": view},
            headers={"Cache-Control": "no-store", "ETag": _etag(view["revision"])},
        )

    return await _guard(run)


async def handle_update(request: web.Request) -> web.StreamResponse:
    async def run() -> web.StreamResponse:
        service, access = _context(request)
        body = await _body(request)
        expected = _expected_revision(request, body)
        assert expected is not None
        kwargs: dict[str, Any] = {"expected_revision": expected}
        aliases = {
            "title": "title", "description": "description", "icon": "icon",
            "markup": "markup", "spec": "spec", "visibility": "visibility",
            "actions": "actions", "frozen": "frozen",
            "expiresAt": "expires_at", "expires_at": "expires_at",
            "sidebarOrder": "sidebar_order", "sidebar_order": "sidebar_order",
            "sidebarGroup": "sidebar_group", "sidebar_group": "sidebar_group",
            "scripts": "scripts", "assets": "assets",
        }
        for source_key, target_key in aliases.items():
            if source_key in body and target_key not in kwargs:
                kwargs[target_key] = body[source_key]
        if "scripts" in kwargs:
            kwargs["scripts"] = _scripts(kwargs["scripts"])
        if "assets" in kwargs:
            kwargs["assets"] = _assets(kwargs["assets"])
        view = await service.update(request.match_info["id"], access, **kwargs)
        return web.json_response(
            {"view": view},
            headers={"Cache-Control": "no-store", "ETag": _etag(view["revision"])},
        )

    return await _guard(run)


async def handle_delete(request: web.Request) -> web.StreamResponse:
    async def run() -> web.StreamResponse:
        service, access = _context(request)
        body = await _body(request)
        if "expectedRevision" not in body and request.query.get("expectedRevision") is not None:
            body["expectedRevision"] = request.query.get("expectedRevision")
        expected = _expected_revision(request, body)
        assert expected is not None
        await service.delete(request.match_info["id"], access, expected_revision=expected)
        return web.json_response({"ok": True, "id": request.match_info["id"]})

    return await _guard(run)


async def handle_reactivate(request: web.Request) -> web.StreamResponse:
    async def run() -> web.StreamResponse:
        service, access = _context(request)
        body = await _body(request)
        view_id = request.match_info["id"]
        expected = _expected_revision(request, body, required=False)
        if expected is None:
            current = await service.get(view_id, access, include_deleted=True)
            expected = int(current["revision"])
        view = await service.reactivate(
            view_id,
            access,
            expected_revision=expected,
            expires_at=body.get("expiresAt", body.get("expires_at")),
        )
        return web.json_response(
            {"view": view},
            headers={"Cache-Control": "no-store", "ETag": _etag(view["revision"])},
        )

    return await _guard(run)


async def handle_set_data(request: web.Request) -> web.StreamResponse:
    async def run() -> web.StreamResponse:
        service, access = _context(request)
        body = await _body(request)
        if "value" not in body:
            raise CustomViewInputError("value is required")
        item = await service.set_data(
            request.match_info["id"],
            request.match_info["key"],
            body["value"],
            access,
            expected_version=_optional_int(
                body.get("expectedVersion", body.get("expected_version")),
                field="expectedVersion",
            ),
            expires_at=body.get("expiresAt", body.get("expires_at")),
            mode=body.get("mode", "replace"),
            max_items=body.get("maxItems", body.get("max_items", 1000)),
        )
        return web.json_response({"data": item}, headers={"Cache-Control": "no-store"})

    return await _guard(run)


async def handle_configure_source(request: web.Request) -> web.StreamResponse:
    async def run() -> web.StreamResponse:
        service, access = _context(request)
        body = await _body(request)
        expected = _expected_revision(request, body)
        assert expected is not None
        source = await service.configure_source(
            request.match_info["id"], request.match_info["key"],
            {
                "driver": body.get("driver"),
                "activation": body.get("activation"),
                "config": body.get("config"),
                "outputSchema": body.get("outputSchema", body.get("output_schema")),
                "enabled": body.get("enabled", True),
                "expiresAt": body.get("expiresAt", body.get("expires_at")),
            },
            access, expected_revision=expected,
            scripts=_scripts(body.get("scripts")),
        )
        return web.json_response(
            {"source": source},
            headers={"Cache-Control": "no-store", "ETag": _etag(source["revision"])},
        )

    return await _guard(run)


async def handle_delete_source(request: web.Request) -> web.StreamResponse:
    async def run() -> web.StreamResponse:
        service, access = _context(request)
        body = await _body(request)
        expected = _expected_revision(request, body)
        assert expected is not None
        revision = await service.delete_source(
            request.match_info["id"], request.match_info["key"], access,
            expected_revision=expected,
        )
        return web.json_response({"ok": True}, headers={"ETag": _etag(revision)})

    return await _guard(run)


async def handle_refresh_source(request: web.Request) -> web.StreamResponse:
    async def run() -> web.StreamResponse:
        service, access = _context(request)
        result = await service.refresh_source(
            request.match_info["id"], request.match_info["key"], access,
        )
        return web.json_response(result, status=202, headers={"Cache-Control": "no-store"})

    return await _guard(run)


async def handle_action(request: web.Request) -> web.StreamResponse:
    async def run() -> web.StreamResponse:
        service, access = _context(request)
        body = await _body(request)
        idempotency_key = body.get("idempotencyKey", body.get("idempotency_key"))
        if not isinstance(idempotency_key, str):
            raise CustomViewInputError("idempotencyKey is required")
        result = await service.invoke_action(
            request.match_info["id"],
            request.match_info["action_id"],
            access,
            input_value=body.get("input"),
            idempotency_key=idempotency_key,
            revision=_optional_int(body.get("revision"), field="revision"),
        )
        return web.json_response({"actionRun": result}, headers={"Cache-Control": "no-store"})

    return await _guard(run)


async def handle_list_grants(request: web.Request) -> web.StreamResponse:
    async def run() -> web.StreamResponse:
        service, access = _context(request)
        result = await service.list_grants(request.match_info["id"], access)
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    return await _guard(run)


async def handle_set_grant(request: web.Request) -> web.StreamResponse:
    async def run() -> web.StreamResponse:
        service, access = _context(request)
        body = await _body(request)
        expected = _optional_int(
            body.get("expectedAclVersion"), field="expectedAclVersion",
        )
        if expected is None:
            raise CustomViewInputError("expectedAclVersion is required")
        permissions = body.get("permissions")
        if not isinstance(permissions, list):
            raise CustomViewInputError("permissions must be an array")
        result = await service.set_grant(
            request.match_info["id"], access,
            principal_type=body.get("principalType"),
            principal_id=body.get("principalId"),
            permissions=permissions,
            expected_acl_version=expected,
        )
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    return await _guard(run)


async def handle_delete_grant(request: web.Request) -> web.StreamResponse:
    async def run() -> web.StreamResponse:
        service, access = _context(request)
        body = await _body(request)
        expected = _optional_int(
            body.get("expectedAclVersion"), field="expectedAclVersion",
        )
        if expected is None:
            raise CustomViewInputError("expectedAclVersion is required")
        result = await service.delete_grant(
            request.match_info["id"], access,
            principal_type=body.get("principalType"),
            principal_id=body.get("principalId"),
            expected_acl_version=expected,
        )
        return web.json_response(result, headers={"Cache-Control": "no-store"})

    return await _guard(run)


_INLINE_MIME = frozenset({
    "image/png", "image/jpeg", "image/gif", "image/webp", "image/avif",
    "font/woff", "font/woff2", "application/font-woff",
})


def _read_asset_bytes(path: Path) -> bytes:
    """Read one already-verified bundle asset without an unbounded allocation."""

    with path.open("rb") as handle:
        payload = handle.read(MAX_ASSET_BYTES + 1)
    if len(payload) > MAX_ASSET_BYTES:
        # Bundles enforce this at publication time. Reaching this branch means
        # storage changed underneath the immutable evidence check.
        raise OSError("Custom View asset exceeds its storage limit")
    return payload


async def handle_asset(request: web.Request) -> web.StreamResponse:
    async def run() -> web.StreamResponse:
        service, access = _context(request)
        revision = _optional_int(request.match_info.get("revision"), field="revision")
        if revision is None:
            raise CustomViewInputError("revision is required")
        relative = request.match_info.get("path") or request.match_info.get("asset_path") or ""
        path = await service.repository.revision_asset_path(
            request.match_info["id"], revision, relative, access,
        )
        guessed_mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        mime = guessed_mime if guessed_mime in _INLINE_MIME else "application/octet-stream"
        payload = await asyncio.to_thread(_read_asset_bytes, path)
        response = web.Response(body=payload, content_type=mime, headers={
            # Asset access is revision-scoped but authorization is not immutable:
            # grants can be revoked while the revision still exists.  Prevent a
            # browser cache from continuing to serve bytes after that revocation.
            "Cache-Control": "private, no-store",
            "Content-Security-Policy": "default-src 'none'; sandbox",
            "X-Content-Type-Options": "nosniff",
            "Cross-Origin-Resource-Policy": "same-origin",
        })
        disposition = "inline" if mime in _INLINE_MIME else "attachment"
        response.headers["Content-Disposition"] = content_disposition_header(
            disposition, filename=Path(path).name,
        )
        return response

    return await _guard(run)


# Naming aliases keep gateway registration readable across beta revisions.
handle_put = handle_update
handle_invoke_action = handle_action
handle_get_asset = handle_asset


__all__ = [name for name in globals() if name.startswith("handle_")]
