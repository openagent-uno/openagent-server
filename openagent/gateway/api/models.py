"""/api/models — DB-backed model catalog (v0.11.0).

Pre-v0.11 this module held yaml-based provider CRUD at ``/api/models``.
Provider management moved to :mod:`openagent.gateway.api.providers`; the
endpoints here now exclusively drive the ``models`` SQLite table.

GET    /api/models              → list model rows
POST   /api/models              → add a model row
GET    /api/models/{runtime_id} → fetch one
PUT    /api/models/{runtime_id} → update one
DELETE /api/models/{runtime_id} → delete one
POST   /api/models/{runtime_id}/enable|disable

GET    /api/models/catalog      → iter_configured_models view w/ pricing
GET    /api/models/active       → active model yaml section
PUT    /api/models/active       → set active model yaml section
GET    /api/models/available    → discovery-driven per-provider catalog
GET    /api/models/providers    → supported provider list
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web

from openagent.gateway.api._common import gateway_db as _db
from openagent.gateway.api.config import _read_raw, _read_resolved, _write_raw
from openagent.gateway.api.providers import _mask_key


async def handle_get_active(request: web.Request) -> web.Response:
    """Return the active model config (masked key)."""
    from aiohttp import web as _web

    raw = _read_raw(request)
    active = dict(raw.get("model", {}))
    if "api_key" in active:
        k = active.pop("api_key")
        if isinstance(k, str) and k.startswith("${"):
            active["api_key_display"] = k
        else:
            active["api_key_display"] = _mask_key(k)

    return _web.json_response({"active": active})


async def handle_set_active(request: web.Request) -> web.Response:
    """Replace the active model config (yaml ``model:`` section)."""
    from aiohttp import web as _web
    from openagent.core.logging import elog

    body = await request.json()
    raw = _read_raw(request)
    raw["model"] = body
    _write_raw(request, raw)
    elog("config.update", section="model")

    return _web.json_response({"ok": True, "restart_required": False})


async def handle_available_providers(request: web.Request) -> web.Response:
    """GET /api/models/providers — provider catalog exposed by OpenAgent."""
    from aiohttp import web as _web
    from openagent.models.catalog import supported_providers

    providers_cfg = _read_resolved(request).get("providers", {})
    return _web.json_response({"providers": supported_providers(providers_cfg)})


async def handle_catalog(request: web.Request) -> web.Response:
    """GET /api/models/catalog?provider=openai — configured models with live pricing."""
    from aiohttp import web as _web
    from openagent.models.catalog import get_model_pricing, iter_configured_models

    provider_filter = request.query.get("provider", "")
    providers_cfg = _read_resolved(request).get("providers", {})
    results = []
    for entry in iter_configured_models(providers_cfg):
        if provider_filter and entry.provider != provider_filter:
            continue
        pricing = get_model_pricing(entry.runtime_id)
        results.append(
            {
                "provider": entry.provider,
                "model_id": entry.model_id,
                "runtime_id": entry.runtime_id,
                "history_mode": entry.history_mode,
                "tier_hint": entry.tier_hint,
                "notes": entry.notes,
                "input_cost_per_million": round(float(pricing["input_cost_per_million"] or 0.0), 4),
                "output_cost_per_million": round(float(pricing["output_cost_per_million"] or 0.0), 4),
            }
        )
    results.sort(key=lambda item: (item["provider"], item["input_cost_per_million"], item["model_id"]))
    return _web.json_response({"models": results})


# ──────────────────────────────────────────────────────────────────────
# DB-backed model catalog. These endpoints hit the ``models`` table the
# model-manager MCP writes to; the gateway's hot-reload loop picks up
# changes on the next message.
# ──────────────────────────────────────────────────────────────────────


async def handle_list_db(request: web.Request) -> web.Response:
    """GET /api/models/db — list every configured model row with live pricing."""
    from aiohttp import web as _web
    from openagent.models.catalog import get_model_pricing

    db = _db(request)
    if db is None:
        return _web.json_response({"error": "memory DB not available"}, status=500)
    provider = request.query.get("provider") or None
    enabled_only = request.query.get("enabled_only", "").lower() in ("1", "true", "yes")
    rows = await db.list_models(provider=provider, enabled_only=enabled_only)
    for row in rows:
        pricing = get_model_pricing(row["runtime_id"])
        row["input_cost_per_million"] = round(float(pricing["input_cost_per_million"] or 0.0), 4)
        row["output_cost_per_million"] = round(float(pricing["output_cost_per_million"] or 0.0), 4)
    return _web.json_response({"models": rows})


async def handle_get_db(request: web.Request) -> web.Response:
    from aiohttp import web as _web
    from openagent.models.catalog import get_model_pricing

    db = _db(request)
    runtime_id = request.match_info["runtime_id"]
    row = await db.get_model(runtime_id)
    if row is None:
        return _web.json_response({"error": f"model {runtime_id!r} not found"}, status=404)
    pricing = get_model_pricing(row["runtime_id"])
    row["input_cost_per_million"] = round(float(pricing["input_cost_per_million"] or 0.0), 4)
    row["output_cost_per_million"] = round(float(pricing["output_cost_per_million"] or 0.0), 4)
    return _web.json_response({"model": row})


async def handle_create_db(request: web.Request) -> web.Response:
    from aiohttp import web as _web
    from openagent.models.catalog import (
        FRAMEWORK_AGNO, FRAMEWORK_CLAUDE_CLI, SUPPORTED_FRAMEWORKS,
        build_runtime_model_id,
    )

    db = _db(request)
    body = await request.json() if request.can_read_body else {}
    provider = str(body.get("provider") or "").strip()
    model_id = str(body.get("model_id") or "").strip()
    framework = str(body.get("framework") or FRAMEWORK_AGNO).strip()
    if not provider or not model_id:
        return _web.json_response(
            {"error": "provider and model_id are required"}, status=400
        )
    # Legacy shorthand: caller passed provider="claude-cli" (pre-v0.10
    # vocabulary, when it was a pseudo-provider). Rewrite to the new
    # shape so there's exactly one way to represent claude-cli rows in
    # the DB.
    if provider == FRAMEWORK_CLAUDE_CLI:
        provider = "anthropic"
        framework = FRAMEWORK_CLAUDE_CLI
    if framework not in SUPPORTED_FRAMEWORKS:
        return _web.json_response(
            {"error": f"invalid framework {framework!r}; expected {SUPPORTED_FRAMEWORKS}"},
            status=400,
        )
    runtime_id = build_runtime_model_id(provider, model_id, framework)
    if not runtime_id:
        return _web.json_response(
            {"error": f"could not build runtime_id from provider={provider!r} model_id={model_id!r}"},
            status=400,
        )
    await db.upsert_model(
        runtime_id,
        provider=provider,
        framework=framework,
        model_id=model_id,
        display_name=body.get("display_name"),
        tier_hint=body.get("tier_hint"),
        notes=body.get("notes"),
        enabled=bool(body.get("enabled", True)),
        metadata=body.get("metadata") or None,
    )
    return _web.json_response(
        {"ok": True, "model": await db.get_model(runtime_id)}, status=201
    )


async def handle_update_db(request: web.Request) -> web.Response:
    from aiohttp import web as _web

    db = _db(request)
    runtime_id = request.match_info["runtime_id"]
    existing = await db.get_model(runtime_id)
    if existing is None:
        return _web.json_response({"error": f"model {runtime_id!r} not found"}, status=404)
    body = await request.json() if request.can_read_body else {}
    await db.upsert_model(
        runtime_id,
        provider=body.get("provider", existing["provider"]),
        framework=body.get("framework", existing.get("framework", "agno")),
        model_id=body.get("model_id", existing["model_id"]),
        display_name=body.get("display_name", existing.get("display_name")),
        tier_hint=body.get("tier_hint", existing.get("tier_hint")),
        notes=body.get("notes", existing.get("notes")),
        enabled=bool(body.get("enabled", existing.get("enabled", True))),
        metadata=body.get("metadata", existing.get("metadata") or None),
    )
    return _web.json_response({"ok": True, "model": await db.get_model(runtime_id)})


async def handle_delete_db(request: web.Request) -> web.Response:
    from aiohttp import web as _web

    db = _db(request)
    runtime_id = request.match_info["runtime_id"]
    existing = await db.get_model(runtime_id)
    if existing is None:
        return _web.json_response({"error": f"model {runtime_id!r} not found"}, status=404)
    # Deleting the last enabled row is allowed — the rejection gate in
    # gateway/server.py will then surface a clear "No models are enabled"
    # error on the next message, which is what the user wants when they
    # intentionally empty the catalog.
    await db.delete_model(runtime_id)
    return _web.json_response({"ok": True})


async def handle_enable_db(request: web.Request) -> web.Response:
    from aiohttp import web as _web

    db = _db(request)
    runtime_id = request.match_info["runtime_id"]
    if await db.get_model(runtime_id) is None:
        return _web.json_response({"error": f"model {runtime_id!r} not found"}, status=404)
    await db.set_model_enabled(runtime_id, True)
    return _web.json_response({"ok": True, "model": await db.get_model(runtime_id)})


async def handle_disable_db(request: web.Request) -> web.Response:
    from aiohttp import web as _web

    db = _db(request)
    runtime_id = request.match_info["runtime_id"]
    if await db.get_model(runtime_id) is None:
        return _web.json_response({"error": f"model {runtime_id!r} not found"}, status=404)
    await db.set_model_enabled(runtime_id, False)
    return _web.json_response({"ok": True, "model": await db.get_model(runtime_id)})


async def handle_available_models(request: web.Request) -> web.Response:
    """GET /api/models/available?provider=openai

    Dynamic provider catalog: tries the provider's /v1/models endpoint
    with the configured API key, falls back to the bundled catalog.
    Fires the live fetch and the DB lookup in parallel — the former can
    take seconds on a cold OpenRouter cache, the latter is always fast.
    """
    import asyncio
    from aiohttp import web as _web
    from openagent.models.catalog import build_runtime_model_id
    from openagent.models.discovery import list_provider_models

    provider = (request.query.get("provider") or "").strip()
    if not provider:
        return _web.json_response({"error": "provider query param is required"}, status=400)
    providers_cfg = _read_resolved(request).get("providers", {}) or {}
    cfg = providers_cfg.get(provider, {}) or {}

    db = _db(request)
    db_task = db.list_models(provider=provider) if db is not None else None
    discovery_task = list_provider_models(
        provider, api_key=cfg.get("api_key"), base_url=cfg.get("base_url"),
    )
    if db_task is not None:
        models_list, db_rows = await asyncio.gather(discovery_task, db_task)
        configured = {r["runtime_id"] for r in db_rows}
        for m in models_list:
            m["runtime_id"] = build_runtime_model_id(provider, m["id"])
            m["added"] = m["runtime_id"] in configured
    else:
        models_list = await discovery_task
    return _web.json_response({"provider": provider, "models": models_list})
