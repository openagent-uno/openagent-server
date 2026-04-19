"""/api/providers — LLM provider rows (v0.12 schema).

The ``providers`` SQLite table holds one row per ``(name, framework)``
pair. Provider edits hot-reload on the next message via
``Agent.refresh_registries`` → ``_hydrate_providers_from_db``.

Endpoints:

  GET    /api/providers             — list (masked keys)
  GET    /api/providers/{id}        — fetch one (masked key)
  POST   /api/providers             — create/upsert {name, framework, api_key?, base_url?}
  PUT    /api/providers/{id}        — update
  DELETE /api/providers/{id}        — delete; FK cascade wipes its models
  POST   /api/providers/{id}/enable   — flip enabled=1
  POST   /api/providers/{id}/disable  — flip enabled=0
  POST   /api/providers/{id}/test   — smoke-test a round-trip
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiohttp import web

from openagent.gateway.api._common import gateway_db as _db


def _mask_key(key: str | None) -> str:
    """Show only last 4 chars of an API key."""
    if not key or len(key) <= 4:
        return "****"
    return "****" + key[-4:]


def _shape_provider(row: dict[str, Any]) -> dict[str, Any]:
    """Public response shape for a provider row — masked api_key."""
    out = {
        "id": row["id"],
        "name": row["name"],
        "framework": row["framework"],
        "base_url": row.get("base_url"),
        "enabled": bool(row.get("enabled", True)),
        "metadata": row.get("metadata") or {},
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }
    key = row.get("api_key")
    if key is None:
        out["api_key_display"] = "—"
    elif isinstance(key, str) and key.startswith("${"):
        out["api_key_display"] = key
    else:
        out["api_key_display"] = _mask_key(key)
    return out


def _parse_provider_id(request: "web.Request") -> int | None:
    raw = request.match_info.get("id")
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


async def handle_list(request: web.Request) -> web.Response:
    from aiohttp import web as _web
    db = _db(request)
    if db is None:
        return _web.json_response({"providers": []})
    rows = await db.list_providers()
    return _web.json_response(
        {"providers": [_shape_provider(r) for r in rows]},
    )


async def handle_get(request: web.Request) -> web.Response:
    from aiohttp import web as _web
    db = _db(request)
    pid = _parse_provider_id(request)
    if db is None:
        return _web.json_response({"error": "DB not available"}, status=500)
    if pid is None:
        return _web.json_response({"error": "invalid provider id"}, status=400)
    row = await db.get_provider(pid)
    if row is None:
        return _web.json_response({"error": f"Provider id={pid} not found"}, status=404)
    return _web.json_response({"provider": _shape_provider(row)})


async def handle_create(request: web.Request) -> web.Response:
    """POST body: {name, framework, api_key?, base_url?, enabled?, metadata?}."""
    from aiohttp import web as _web
    from openagent.core.logging import elog
    from openagent.memory.db import VALID_FRAMEWORKS

    db = _db(request)
    if db is None:
        return _web.json_response({"error": "DB not available"}, status=500)
    body = await request.json() if request.can_read_body else {}
    name = str(body.get("name") or "").strip()
    framework = str(body.get("framework") or "").strip()
    if not name:
        return _web.json_response({"error": "name is required"}, status=400)
    if framework not in VALID_FRAMEWORKS:
        return _web.json_response(
            {"error": f"framework must be one of {list(VALID_FRAMEWORKS)}"},
            status=400,
        )
    try:
        pid = await db.upsert_provider(
            name=name,
            framework=framework,
            api_key=(body.get("api_key") or None),
            base_url=(body.get("base_url") or None),
            enabled=bool(body.get("enabled", True)),
            metadata=body.get("metadata") or None,
        )
    except ValueError as e:
        return _web.json_response({"error": str(e)}, status=400)
    elog("provider.created", provider_id=pid, name=name, framework=framework)
    row = await db.get_provider(pid)
    return _web.json_response({"ok": True, "provider": _shape_provider(row)}, status=201)


async def handle_update(request: web.Request) -> web.Response:
    """PUT body merges into the existing row — omitted fields stay.

    ``framework`` is immutable; attempts to change it return 400.
    """
    from aiohttp import web as _web

    db = _db(request)
    pid = _parse_provider_id(request)
    if db is None:
        return _web.json_response({"error": "DB not available"}, status=500)
    if pid is None:
        return _web.json_response({"error": "invalid provider id"}, status=400)
    existing = await db.get_provider(pid)
    if existing is None:
        return _web.json_response({"error": f"Provider id={pid} not found"}, status=404)
    body = await request.json() if request.can_read_body else {}
    # Framework is immutable — change requires delete + recreate.
    if "framework" in body and body["framework"] != existing["framework"]:
        return _web.json_response(
            {"error": "framework is immutable; delete + recreate the provider instead"},
            status=400,
        )
    try:
        await db.upsert_provider(
            name=body.get("name", existing["name"]),
            framework=existing["framework"],
            api_key=body.get("api_key", existing.get("api_key")),
            base_url=body.get("base_url", existing.get("base_url")),
            enabled=bool(body.get("enabled", existing.get("enabled", True))),
            metadata=body.get("metadata", existing.get("metadata") or None),
        )
    except ValueError as e:
        return _web.json_response({"error": str(e)}, status=400)
    row = await db.get_provider(pid)
    return _web.json_response({"ok": True, "provider": _shape_provider(row)})


async def handle_delete(request: web.Request) -> web.Response:
    """Delete provider row. FK cascade wipes every model that referenced it."""
    from aiohttp import web as _web
    from openagent.core.logging import elog

    db = _db(request)
    pid = _parse_provider_id(request)
    if db is None:
        return _web.json_response({"error": "DB not available"}, status=500)
    if pid is None:
        return _web.json_response({"error": "invalid provider id"}, status=400)
    existing = await db.get_provider(pid)
    if existing is None:
        return _web.json_response({"error": f"Provider id={pid} not found"}, status=404)
    # Count how many models are about to be cascade-deleted so the caller
    # can surface the side effect.
    models = await db.list_models(provider_id=pid)
    await db.delete_provider(pid)
    elog(
        "provider.deleted", provider_id=pid, name=existing["name"],
        framework=existing["framework"], models_purged=len(models),
    )
    return _web.json_response({"ok": True, "models_purged": len(models)})


async def _handle_toggle(request: web.Request, enabled: bool) -> web.Response:
    from aiohttp import web as _web
    db = _db(request)
    pid = _parse_provider_id(request)
    if db is None or pid is None:
        return _web.json_response({"error": "invalid provider id"}, status=400)
    if await db.get_provider(pid) is None:
        return _web.json_response({"error": f"Provider id={pid} not found"}, status=404)
    await db.set_provider_enabled(pid, enabled)
    row = await db.get_provider(pid)
    return _web.json_response({"ok": True, "provider": _shape_provider(row)})


async def handle_enable(request: web.Request) -> web.Response:
    return await _handle_toggle(request, True)


async def handle_disable(request: web.Request) -> web.Response:
    return await _handle_toggle(request, False)


async def handle_test(request: web.Request) -> web.Response:
    """Round-trip a short prompt through the configured provider.

    The provider row is looked up by the URL's ``{id}`` — unambiguous
    when the same vendor is registered under both frameworks. Body is
    optional ``{model: <runtime_id or bare>}``.
    """
    from aiohttp import web as _web
    from openagent.models.runtime import run_provider_smoke_test

    body = await request.json() if request.can_read_body else {}
    db = _db(request)
    pid = _parse_provider_id(request)
    if db is None or pid is None:
        return _web.json_response({"error": "invalid provider id"}, status=400)
    provider_row = await db.get_provider(pid)
    if provider_row is None:
        return _web.json_response({"error": f"Provider id={pid} not found"}, status=404)

    # Build a fresh providers_config snapshot from the DB so freshly-added
    # keys are usable without waiting for the next message's hot-reload.
    providers_config = await db.materialise_providers_config()

    try:
        runtime_model, resp = await run_provider_smoke_test(
            provider_row["name"],
            providers_config,
            model_id=body.get("model") or body.get("model_id"),
            framework=provider_row["framework"],
            session_id="provider-test",
        )
        return _web.json_response({"ok": True, "model": runtime_model, "response": resp.content})
    except Exception as e:  # noqa: BLE001 — surfaced to caller
        return _web.json_response({"ok": False, "error": str(e)}, status=400)
