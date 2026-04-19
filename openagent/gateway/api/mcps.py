"""CRUD /api/mcps — manage the MCP server registry stored in SQLite.

The ``mcps`` SQLite table is the sole source of truth. These endpoints
drive the same table the mcp-manager MCP writes to, and every change is
picked up by the gateway's hot-reload loop on the next message.

GET    /api/mcps                → list all rows
GET    /api/mcps/{name}         → fetch one
POST   /api/mcps                → add a custom or builtin entry
PUT    /api/mcps/{name}         → partial update
DELETE /api/mcps/{name}         → remove permanently
POST   /api/mcps/{name}/enable  → flip enabled=1
POST   /api/mcps/{name}/disable → flip enabled=0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web


from openagent.gateway.api._common import gateway_db as _db  # noqa: E402


async def handle_list(request):
    from aiohttp import web

    db = _db(request)
    if db is None:
        return web.json_response({"error": "memory DB not available"}, status=500)
    rows = await db.list_mcps()
    return web.json_response({"mcps": rows})


async def handle_get(request):
    from aiohttp import web

    db = _db(request)
    name = request.match_info["name"]
    row = await db.get_mcp(name)
    if row is None:
        return web.json_response({"error": f"MCP {name!r} not found"}, status=404)
    return web.json_response({"mcp": row})


async def handle_create(request):
    from aiohttp import web

    db = _db(request)
    body = await request.json() if request.can_read_body else {}
    name = str(body.get("name") or "").strip()
    if not name:
        return web.json_response({"error": "name is required"}, status=400)
    builtin_name = body.get("builtin_name") or body.get("builtin")
    if builtin_name:
        # Builtins are defined in code and auto-seeded at boot; adding one
        # at runtime never produces new behaviour, only a duplicate row.
        # Callers should enable/disable the existing row instead.
        return web.json_response(
            {
                "error": (
                    "Builtin MCPs cannot be added at runtime — they are "
                    "defined in openagent.mcp.builtins.BUILTIN_MCP_SPECS "
                    "and auto-seeded on boot. Use POST "
                    f"/api/mcps/{builtin_name}/enable to turn the existing "
                    "row on, or PUT /api/mcps/{name} to change its env."
                ),
            },
            status=400,
        )
    else:
        command = body.get("command") or None
        url = body.get("url") or None
        if not command and not url:
            return web.json_response(
                {"error": "either command (argv list) or url is required"},
                status=400,
            )
        await db.upsert_mcp(
            name,
            kind="custom",
            command=list(command) if command else None,
            args=list(body.get("args") or []),
            url=url,
            env=dict(body.get("env") or {}),
            headers=dict(body.get("headers") or {}),
            oauth=bool(body.get("oauth", False)),
            enabled=bool(body.get("enabled", True)),
            source="api",
        )
    return web.json_response({"ok": True, "mcp": await db.get_mcp(name)}, status=201)


async def handle_update(request):
    from aiohttp import web

    db = _db(request)
    name = request.match_info["name"]
    existing = await db.get_mcp(name)
    if existing is None:
        return web.json_response({"error": f"MCP {name!r} not found"}, status=404)
    body = await request.json() if request.can_read_body else {}

    # Merge: body overrides existing field-by-field. Missing keys leave the
    # old values in place so callers can PATCH-style hit the endpoint.
    await db.upsert_mcp(
        name,
        kind=body.get("kind", existing["kind"]),
        builtin_name=body.get("builtin_name", existing.get("builtin_name")),
        command=body.get("command", existing.get("command")),
        args=body.get("args", existing.get("args")),
        url=body.get("url", existing.get("url")),
        env=body.get("env", existing.get("env")),
        headers=body.get("headers", existing.get("headers")),
        oauth=bool(body.get("oauth", existing.get("oauth", False))),
        enabled=bool(body.get("enabled", existing.get("enabled", True))),
        source=body.get("source", existing.get("source", "api")),
    )
    return web.json_response({"ok": True, "mcp": await db.get_mcp(name)})


async def handle_delete(request):
    from aiohttp import web

    db = _db(request)
    name = request.match_info["name"]
    existing = await db.get_mcp(name)
    if existing is None:
        return web.json_response({"error": f"MCP {name!r} not found"}, status=404)
    # Builtins are defined in code and can't be meaningfully "deleted";
    # their row is the control surface for env / enabled state. Force
    # disable-only for everything that isn't a user-added custom entry.
    if existing.get("kind") != "custom":
        return web.json_response(
            {
                "error": (
                    f"Refusing to delete builtin MCP {name!r} "
                    f"(kind={existing.get('kind')!r}). Disable it instead — "
                    "builtins can be toggled but not removed."
                ),
            },
            status=400,
        )
    await db.delete_mcp(name)
    return web.json_response({"ok": True})


async def handle_enable(request):
    from aiohttp import web

    db = _db(request)
    name = request.match_info["name"]
    existing = await db.get_mcp(name)
    if existing is None:
        return web.json_response({"error": f"MCP {name!r} not found"}, status=404)
    await db.set_mcp_enabled(name, True)
    return web.json_response({"ok": True, "mcp": await db.get_mcp(name)})


async def handle_disable(request):
    from aiohttp import web

    db = _db(request)
    name = request.match_info["name"]
    existing = await db.get_mcp(name)
    if existing is None:
        return web.json_response({"error": f"MCP {name!r} not found"}, status=404)
    await db.set_mcp_enabled(name, False)
    return web.json_response({"ok": True, "mcp": await db.get_mcp(name)})
