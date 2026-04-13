"""GET/POST /api/providers — LLM provider management."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web

from openagent.gateway.api.config import _read_raw, _read_resolved


def _mask_key(key: str) -> str:
    """Show only last 4 chars of an API key."""
    if not key or len(key) <= 4:
        return "****"
    return "****" + key[-4:]


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
    """Return providers dict with API keys masked."""
    return {name: mask_provider_entry(cfg) for name, cfg in providers.items()}


async def handle_list(request: web.Request) -> web.Response:
    """List configured providers with masked keys."""
    from aiohttp import web as _web
    raw = _read_raw(request)
    providers = raw.get("providers", {})
    return _web.json_response({"providers": mask_providers(providers)})


async def handle_test(request: web.Request) -> web.Response:
    """Test a provider by sending a simple prompt."""
    from aiohttp import web as _web
    from openagent.models.runtime import run_provider_smoke_test

    body = await request.json()
    provider_name = body.get("provider", "")

    providers = _read_resolved(request).get("providers", {})

    try:
        runtime_model, resp = await run_provider_smoke_test(
            provider_name,
            providers,
            model_id=body.get("model_id"),
            session_id="provider-test",
        )
        return _web.json_response({"ok": True, "model": runtime_model, "response": resp.content})
    except Exception as e:
        return _web.json_response({"ok": False, "error": str(e)}, status=400)
