"""CRUD /api/models — manage LLM providers (API keys) and active model.

GET    /api/models              → list providers (masked keys) + active model
POST   /api/models              → add a provider
GET    /api/models/active       → get active model config
PUT    /api/models/active       → set active model config
PUT    /api/models/{name}       → update a provider
DELETE /api/models/{name}       → remove a provider
POST   /api/models/{name}/test  → test provider connectivity
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web

from openagent.gateway.api.config import _read_raw, _write_raw
from openagent.gateway.api.providers import _mask_key


def _mask_providers(providers: dict) -> dict:
    """Return providers dict with API keys masked."""
    masked = {}
    for name, cfg in providers.items():
        entry = dict(cfg)
        if "api_key" in entry:
            raw = entry.pop("api_key")
            if isinstance(raw, str) and raw.startswith("${"):
                entry["api_key_display"] = raw
            else:
                entry["api_key_display"] = _mask_key(raw)
        masked[name] = entry
    return masked


async def handle_list(request: web.Request) -> web.Response:
    """List all configured providers + the active model."""
    from aiohttp import web as _web

    raw = _read_raw(request)
    providers = raw.get("providers", {})
    active = raw.get("model", {})

    # Mask the active model's api_key too
    active_masked = dict(active)
    if "api_key" in active_masked:
        k = active_masked.pop("api_key")
        if isinstance(k, str) and k.startswith("${"):
            active_masked["api_key_display"] = k
        else:
            active_masked["api_key_display"] = _mask_key(k)

    return _web.json_response({
        "models": _mask_providers(providers),
        "active": active_masked,
    })


async def handle_create(request: web.Request) -> web.Response:
    """Add a new provider entry."""
    from aiohttp import web as _web

    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        return _web.json_response({"error": "name is required"}, status=400)

    raw = _read_raw(request)
    if "providers" not in raw:
        raw["providers"] = {}

    entry: dict = {}
    if body.get("api_key"):
        entry["api_key"] = body["api_key"]
    if body.get("base_url"):
        entry["base_url"] = body["base_url"]

    raw["providers"][name] = entry
    _write_raw(request, raw)

    return _web.json_response({"ok": True, "name": name})


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
    """Replace the active model config."""
    from aiohttp import web as _web
    from openagent.core.logging import elog

    body = await request.json()
    raw = _read_raw(request)
    raw["model"] = body
    _write_raw(request, raw)
    elog("config.update", section="model")

    return _web.json_response({"ok": True, "restart_required": True})


async def handle_update(request: web.Request) -> web.Response:
    """Update a provider entry."""
    from aiohttp import web as _web

    name = request.match_info["name"]
    body = await request.json()

    raw = _read_raw(request)
    providers = raw.get("providers", {})
    if name not in providers:
        return _web.json_response({"error": f"Provider '{name}' not found"}, status=404)

    entry = providers[name]
    if "api_key" in body:
        entry["api_key"] = body["api_key"]
    if "base_url" in body:
        if body["base_url"]:
            entry["base_url"] = body["base_url"]
        else:
            entry.pop("base_url", None)
    if "disabled_models" in body:
        if body["disabled_models"]:
            entry["disabled_models"] = body["disabled_models"]
        else:
            entry.pop("disabled_models", None)

    raw["providers"][name] = entry
    _write_raw(request, raw)

    return _web.json_response({"ok": True})


async def handle_delete(request: web.Request) -> web.Response:
    """Remove a provider entry."""
    from aiohttp import web as _web

    name = request.match_info["name"]
    raw = _read_raw(request)
    providers = raw.get("providers", {})

    if name not in providers:
        return _web.json_response({"error": f"Provider '{name}' not found"}, status=404)

    del providers[name]
    raw["providers"] = providers
    _write_raw(request, raw)

    return _web.json_response({"ok": True})


async def handle_available_providers(request: web.Request) -> web.Response:
    """GET /api/models/providers — available LLM providers from litellm."""
    from aiohttp import web as _web

    try:
        from litellm import model_cost
    except ImportError:
        return _web.json_response({"providers": ["anthropic"]})

    # Extract unique provider prefixes from model_cost keys
    provider_set: set[str] = set()
    for model_id, info in model_cost.items():
        # Skip models with no pricing
        if not (info.get("input_cost_per_token") or info.get("output_cost_per_token")):
            continue
        # Provider is the prefix before the first dot or slash
        for sep in (".", "/"):
            if sep in model_id:
                provider_set.add(model_id.split(sep, 1)[0])
                break

    # Always include anthropic (has special CLI adapter)
    provider_set.add("anthropic")

    # Sort and return
    providers = sorted(provider_set)
    return _web.json_response({"providers": providers})


async def handle_catalog(request: web.Request) -> web.Response:
    """GET /api/models/catalog?provider=anthropic — available models with pricing."""
    from aiohttp import web as _web

    provider_filter = request.query.get("provider", "")

    try:
        from litellm import model_cost
    except ImportError:
        return _web.json_response({"models": []})

    results = []
    for model_id, info in model_cost.items():
        # Filter by provider prefix if specified
        if provider_filter and not model_id.startswith(provider_filter):
            continue
        input_cost = (info.get("input_cost_per_token", 0) or 0) * 1_000_000
        output_cost = (info.get("output_cost_per_token", 0) or 0) * 1_000_000
        if input_cost == 0 and output_cost == 0:
            continue  # skip models with no pricing info
        results.append({
            "model_id": model_id,
            "input_cost_per_million": round(input_cost, 4),
            "output_cost_per_million": round(output_cost, 4),
        })

    # Sort by input cost ascending
    results.sort(key=lambda x: x["input_cost_per_million"])

    return _web.json_response({"models": results})


async def handle_test(request: web.Request) -> web.Response:
    """Test a provider by sending a simple prompt via litellm."""
    from aiohttp import web as _web
    from openagent.core.config import _resolve_env_vars

    name = request.match_info["name"]
    body = await request.json() if request.can_read_body else {}

    raw = _read_raw(request)
    providers = _resolve_env_vars(raw.get("providers", {}))
    cfg = providers.get(name)
    if not cfg:
        return _web.json_response(
            {"ok": False, "error": f"Provider '{name}' not configured"}, status=400
        )

    from openagent.models.litellm_provider import get_cheapest_model
    model_id = body.get("model_id") or get_cheapest_model(name) or f"{name}/default"

    try:
        import litellm
        resp = await litellm.acompletion(
            model=model_id,
            messages=[{"role": "user", "content": "Say 'ok' and nothing else."}],
            max_tokens=5,
            api_key=cfg.get("api_key"),
            api_base=cfg.get("base_url"),
        )
        return _web.json_response({"ok": True, "model": model_id})
    except Exception as e:
        return _web.json_response({"ok": False, "error": str(e)}, status=400)
