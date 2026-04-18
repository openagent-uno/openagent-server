"""/api/providers — LLM provider credentials, DB-backed.

The ``providers`` SQLite table is the source of truth for LLM provider
credentials. Provider edits hot-reload on the next message via
``Agent.refresh_registries`` → ``_hydrate_providers_from_db``.

Endpoints:

  GET    /api/providers              — list (masked keys)
  GET    /api/providers/{name}       — fetch one (masked key)
  POST   /api/providers              — create/upsert
  PUT    /api/providers/{name}       — update
  DELETE /api/providers/{name}       — delete, cascades to models
  POST   /api/providers/{name}/enable   — flip enabled=1
  POST   /api/providers/{name}/disable  — flip enabled=0
  POST   /api/providers/{name}/test  — smoke-test a round-trip
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


def _mask_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    key = out.pop("api_key", None)
    if isinstance(key, str) and key.startswith("${"):
        out["api_key_display"] = key
    else:
        out["api_key_display"] = _mask_key(key)
    return out


# Retained for backwards-compat with code paths that still pass a
# providers dict (e.g. cost resolution in usage APIs).
def mask_provider_entry(cfg: dict) -> dict:
    entry = dict(cfg)
    if "api_key" in entry:
        raw = entry.pop("api_key")
        if isinstance(raw, str) and raw.startswith("${"):
            entry["api_key_display"] = raw
        else:
            entry["api_key_display"] = _mask_key(raw)
    return entry


def mask_providers(providers: dict) -> dict:
    return {name: mask_provider_entry(cfg) for name, cfg in providers.items()}


async def handle_list(request: web.Request) -> web.Response:
    from aiohttp import web as _web
    db = _db(request)
    if db is None:
        return _web.json_response({"providers": {}})
    rows = await db.list_providers()
    return _web.json_response(
        {"providers": {r["name"]: _mask_row(r) for r in rows}},
    )


async def handle_get(request: web.Request) -> web.Response:
    from aiohttp import web as _web
    db = _db(request)
    name = request.match_info["name"]
    if db is None:
        return _web.json_response({"error": "DB not available"}, status=500)
    row = await db.get_provider(name)
    if row is None:
        return _web.json_response({"error": f"Provider {name!r} not found"}, status=404)
    return _web.json_response({"provider": _mask_row(row)})


async def handle_create(request: web.Request) -> web.Response:
    """Upsert a provider row. POST body: {name, api_key?, base_url?, enabled?}."""
    from aiohttp import web as _web
    from openagent.core.logging import elog

    db = _db(request)
    if db is None:
        return _web.json_response({"error": "DB not available"}, status=500)
    body = await request.json() if request.can_read_body else {}
    name = str(body.get("name") or "").strip()
    if not name:
        return _web.json_response({"error": "name is required"}, status=400)
    await db.upsert_provider(
        name,
        api_key=body.get("api_key") or None,
        base_url=body.get("base_url") or None,
        enabled=bool(body.get("enabled", True)),
        metadata=body.get("metadata") or None,
    )
    elog("provider.created", name=name)
    row = await db.get_provider(name)
    return _web.json_response({"ok": True, "provider": _mask_row(row)}, status=201)


async def handle_update(request: web.Request) -> web.Response:
    """PUT body merges into the existing row — omitted fields stay."""
    from aiohttp import web as _web

    db = _db(request)
    name = request.match_info["name"]
    if db is None:
        return _web.json_response({"error": "DB not available"}, status=500)
    existing = await db.get_provider(name)
    if existing is None:
        return _web.json_response({"error": f"Provider {name!r} not found"}, status=404)
    body = await request.json() if request.can_read_body else {}
    await db.upsert_provider(
        name,
        api_key=body.get("api_key", existing.get("api_key")),
        base_url=body.get("base_url", existing.get("base_url")),
        enabled=bool(body.get("enabled", existing.get("enabled", True))),
        metadata=body.get("metadata", existing.get("metadata") or None),
    )
    row = await db.get_provider(name)
    return _web.json_response({"ok": True, "provider": _mask_row(row)})


async def handle_delete(request: web.Request) -> web.Response:
    """Delete provider row + cascade-delete every model owned by it."""
    from aiohttp import web as _web
    from openagent.core.logging import elog

    db = _db(request)
    name = request.match_info["name"]
    if db is None:
        return _web.json_response({"error": "DB not available"}, status=500)
    if await db.get_provider(name) is None:
        return _web.json_response({"error": f"Provider {name!r} not found"}, status=404)
    purged = await db.delete_models_by_provider(name)
    await db.delete_provider(name)
    elog("provider.deleted", name=name, models_purged=purged)
    return _web.json_response({"ok": True, "models_purged": purged})


async def _handle_toggle(request: web.Request, enabled: bool) -> web.Response:
    from aiohttp import web as _web
    db = _db(request)
    name = request.match_info["name"]
    if db is None or await db.get_provider(name) is None:
        return _web.json_response({"error": f"Provider {name!r} not found"}, status=404)
    await db.set_provider_enabled(name, enabled)
    return _web.json_response({"ok": True, "provider": _mask_row(await db.get_provider(name))})


async def handle_enable(request: web.Request) -> web.Response:
    return await _handle_toggle(request, True)


async def handle_disable(request: web.Request) -> web.Response:
    return await _handle_toggle(request, False)


async def handle_test(request: web.Request) -> web.Response:
    """Round-trip a short prompt through the configured provider.

    Body: ``{"provider": "openai", "model_id": "gpt-4o-mini"?}`` or with
    ``provider`` in the URL via a ``{name}/test`` route — both supported.
    Uses the hydrated ``self.config["providers"]`` dict so live key
    edits are picked up without a message-level refresh.
    """
    from aiohttp import web as _web
    from openagent.models.runtime import run_provider_smoke_test

    body = await request.json() if request.can_read_body else {}
    provider_name = request.match_info.get("name") or body.get("provider") or ""
    if not provider_name:
        return _web.json_response({"error": "provider name required"}, status=400)

    # Read DB snapshot directly so a freshly-added key is usable
    # without waiting for the next message's hot-reload tick.
    db = _db(request)
    providers_materialised: dict[str, dict[str, Any]] = {}
    if db is not None:
        for r in await db.list_providers():
            entry: dict[str, Any] = {}
            if r.get("api_key"):
                entry["api_key"] = r["api_key"]
            if r.get("base_url"):
                entry["base_url"] = r["base_url"]
            providers_materialised[r["name"]] = entry

    try:
        runtime_model, resp = await run_provider_smoke_test(
            provider_name,
            providers_materialised,
            model_id=body.get("model_id"),
            session_id="provider-test",
        )
        return _web.json_response({"ok": True, "model": runtime_model, "response": resp.content})
    except Exception as e:  # noqa: BLE001 — surfaced to caller
        return _web.json_response({"ok": False, "error": str(e)}, status=400)
