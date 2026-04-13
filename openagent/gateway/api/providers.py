"""GET/POST /api/providers — LLM provider management."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web


def _mask_key(key: str) -> str:
    """Show only last 4 chars of an API key."""
    if not key or len(key) <= 4:
        return "****"
    return "****" + key[-4:]


async def handle_list(request: web.Request) -> web.Response:
    """List configured providers with masked keys."""
    from aiohttp import web as _web

    gw = request.app["gateway"]
    config_path = gw.config_path
    if not config_path:
        return _web.json_response({"providers": {}})

    from pathlib import Path
    import yaml
    path = Path(config_path)
    if not path.exists():
        return _web.json_response({"providers": {}})

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    providers = raw.get("providers", {})
    masked = {}
    for name, cfg in providers.items():
        entry = dict(cfg)
        if "api_key" in entry:
            # Don't mask env var references
            if entry["api_key"].startswith("${"):
                entry["api_key_display"] = entry["api_key"]
            else:
                entry["api_key_display"] = _mask_key(entry["api_key"])
            del entry["api_key"]
        masked[name] = entry

    return _web.json_response({"providers": masked})


async def handle_test(request: web.Request) -> web.Response:
    """Test a provider by sending a simple prompt."""
    from aiohttp import web as _web
    from openagent.models.agno_provider import AgnoProvider
    from openagent.models.catalog import get_default_model_for_provider, normalize_runtime_model_id

    body = await request.json()
    provider_name = body.get("provider", "")

    # Load providers config
    gw = request.app["gateway"]
    config_path = gw.config_path
    if not config_path:
        return _web.json_response({"ok": False, "error": "No config file"}, status=400)

    from pathlib import Path
    import yaml
    from openagent.core.config import _resolve_env_vars

    path = Path(config_path)
    if not path.exists():
        return _web.json_response({"ok": False, "error": "Config not found"}, status=400)

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    providers = _resolve_env_vars(raw.get("providers", {}))
    cfg = providers.get(provider_name)
    if not cfg:
        return _web.json_response({"ok": False, "error": f"Provider '{provider_name}' not configured"}, status=400)

    runtime_model = body.get("model_id") or get_default_model_for_provider(provider_name, providers)
    if not runtime_model:
        return _web.json_response({"ok": False, "error": f"No models configured for provider '{provider_name}'"}, status=400)
    runtime_model = normalize_runtime_model_id(runtime_model, providers)

    try:
        provider = AgnoProvider(
            model=runtime_model,
            api_key=cfg.get("api_key"),
            base_url=cfg.get("base_url"),
            providers_config=providers,
        )
        resp = await provider.generate(
            messages=[{"role": "user", "content": "Say 'ok' and nothing else."}],
            session_id="provider-test",
        )
        return _web.json_response({"ok": True, "model": runtime_model, "response": resp.content})
    except Exception as e:
        return _web.json_response({"ok": False, "error": str(e)}, status=400)
