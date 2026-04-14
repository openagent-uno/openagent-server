"""Config REST API — read/write openagent.yaml.

GET   /api/config              → full config as JSON (env vars NOT resolved)
PUT   /api/config              → replace entire config
PATCH /api/config/{section}    → update one section
"""

from __future__ import annotations

from pathlib import Path

from openagent.core.logging import elog
from .vault import _sanitize  # reuse datetime sanitizer


def _resolve_config_path(request) -> Path:
    gw = request.app["gateway"]
    if gw.config_path:
        return Path(gw.config_path).expanduser().resolve()
    from openagent.core.paths import default_config_path
    return default_config_path()


def _load_raw_config(path: Path) -> dict:
    import yaml

    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_resolved_config(path: Path) -> dict:
    from openagent.core.config import _resolve_env_vars

    return _resolve_env_vars(_load_raw_config(path))


def _read_raw(request) -> dict:
    return _load_raw_config(_resolve_config_path(request))


def _read_resolved(request) -> dict:
    return _load_resolved_config(_resolve_config_path(request))


def _write_raw(request, config: dict) -> None:
    import yaml
    p = _resolve_config_path(request)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


async def handle_get(request):
    from aiohttp import web
    return web.json_response(_sanitize(_read_raw(request)))


async def handle_put(request):
    from aiohttp import web
    data = await request.json()
    _write_raw(request, data)
    elog("config.update", section="full")
    return web.json_response({"ok": True, "restart_required": True})


async def handle_patch(request):
    from aiohttp import web
    section = request.match_info["section"]
    patch = await request.json()
    config = _read_raw(request)
    config[section] = patch
    _write_raw(request, config)
    elog("config.update", section=section)
    # model/providers are hot-reloaded by the Gateway on the next message;
    # other sections (mcp, scheduler, channels, system_prompt) still need
    # a restart to take effect.
    hot_reloadable = section in ("model", "providers")
    return web.json_response({
        "ok": True,
        "restart_required": not hot_reloadable,
        section: _sanitize(patch),
    })
