"""Config REST API — read/write openagent.yaml.

GET   /api/config              → full config as JSON (env vars NOT resolved)
PUT   /api/config              → replace entire config
PATCH /api/config/{section}    → update one section
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.logging import elog
from .vault import _sanitize  # reuse datetime sanitizer


def _resolve_config_path(request) -> Path:
    gw = request.app["gateway"]
    if gw.config_path:
        return Path(gw.config_path).expanduser().resolve()
    from src.core.paths import default_config_path
    return default_config_path()


def _load_raw_config(path: Path) -> dict:
    import yaml

    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_resolved_config(path: Path) -> dict:
    from src.core.config import _resolve_env_vars

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


#: Sections owned by the SQLite DB. They must never be written to the
#: yaml — creating them there is a silent no-op at runtime and confuses
#: users who expect edits to take effect.
DB_OWNED_SECTIONS = frozenset({"providers", "models", "mcp", "mcps", "scheduled", "scheduled_tasks"})
IDENTITY_SECTIONS = frozenset({"name", "system_prompt"})


def _strip_db_owned(data: dict) -> dict:
    """Drop any DB-owned keys from a yaml payload before persisting."""
    return {k: v for k, v in data.items() if k not in DB_OWNED_SECTIONS}


async def handle_put(request):
    from aiohttp import web
    data = await request.json()
    if not isinstance(data, dict):
        return web.json_response({"error": "config body must be an object"}, status=400)
    current = _read_raw(request)
    # A full replacement has no revision field and historically bypassed all
    # identity ACL/validation. Keep it for non-identity configuration, but
    # require the dedicated owner-scoped PATCH surface for these two fields.
    missing = object()
    identity_changed = any(
        data.get(section, missing) != current.get(section, missing)
        for section in IDENTITY_SECTIONS
    )
    if identity_changed:
        return web.json_response({
            "error": (
                "name and system_prompt cannot be changed through full config PUT; "
                "use PATCH /api/agent/identity"
            ),
        }, status=400)
    _write_raw(request, _strip_db_owned(data))
    elog("config.update", section="full")
    return web.json_response({"ok": True, "restart_required": True})


def _merge_section(existing: Any, patch: Any) -> Any:
    """RFC 7386 merge of ``patch`` into ``existing``.

    ``null`` deletes a key (the only way to remove one, now that omission
    means "leave it alone"); dicts merge recursively; everything else
    replaces. A non-dict patch replaces outright — that is what a caller
    sending a scalar or a list can only mean.
    """
    if not isinstance(patch, dict):
        return patch
    out = dict(existing) if isinstance(existing, dict) else {}
    for key, value in patch.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict):
            out[key] = _merge_section(out.get(key), value)
        else:
            out[key] = value
    return out


async def handle_patch(request):
    from aiohttp import web
    section = request.match_info["section"]
    if section in IDENTITY_SECTIONS:
        # Compatibility route for released clients. It delegates to the exact
        # same owner-bound service as /api/agent/identity, so old clients do
        # not become an ACL/CAS/hot-apply bypass on an upgraded gateway.
        from . import agent_identity as agent_identity_api

        try:
            try:
                value = await request.json()
            except Exception as exc:  # noqa: BLE001
                from src.core.agent_identity import AgentIdentityInputError

                raise AgentIdentityInputError(
                    "request body must contain valid JSON",
                ) from exc
            kwargs = {section: value}
            result = await agent_identity_api.service_for_request(request).update(
                agent_identity_api.actor_for_request(request),
                **kwargs,
            )
            return web.json_response({
                "ok": True,
                "restart_required": False,
                section: result[section],
                "revision": result["revision"],
            })
        except Exception as exc:  # noqa: BLE001
            return agent_identity_api.error_response(exc)
    if section in DB_OWNED_SECTIONS:
        return web.json_response(
            {
                "error": (
                    f"section {section!r} is managed by the SQLite database; "
                    "use the corresponding /api/* endpoint instead of the "
                    "yaml config."
                )
            },
            status=400,
        )
    patch = await request.json()
    config = _read_raw(request)
    # MERGE, don't replace. This endpoint is a PATCH and was assigning the
    # body over the whole section: sending ``{"distiller_schedule": "..."}``
    # to ``/api/config/skills`` took ``enabled``, ``path`` and both toggles
    # with it — silently turning the skills subsystem off on a live agent
    # (found exactly that way, 2026-08-25). Merge semantics per RFC 7386: a
    # key set to ``null`` is a deletion, everything else is an upsert, and
    # nested objects merge instead of clobbering.
    merged = _merge_section(config.get(section), patch)
    config[section] = merged
    _write_raw(request, config)
    elog("config.update", section=section)

    # Sections with a registered live-reaction hook (dream_mode,
    # auto_update) take effect immediately — the
    # AgentServer registered a closure that re-syncs the matching
    # scheduled-task row to the new state without a process restart.
    gw = request.app.get("gateway")
    handled_live = False
    if gw is not None and section in getattr(gw, "_config_change_callbacks", {}):
        await gw.on_config_change(section, merged)
        handled_live = True
    # Other resource screens may want to refresh derived values.
    if gw is not None:
        await gw.broadcast_resource("config", "updated", section)

    return web.json_response({
        "ok": True,
        "restart_required": not handled_live,
        section: _sanitize(patch),
    })
