"""Skills REST API — the file-backed skill library, over the gateway.

GET    /api/skills                  → { "skills": [...] } metadata only
GET    /api/skills/search?q=&limit= → { "results": [...] }
GET    /api/skills/{name}           → one skill, full body + bundled files
POST   /api/skills                  → create {name, description?, category?, body}
PUT    /api/skills/{name}           → update the same fields
DELETE /api/skills/{name}           → remove the folder from disk
POST   /api/skills/{name}/archive   → retire without deleting

Same device-cert auth middleware as the rest of ``/api/*``. §10 says the
gateway is the only public surface, and until now skills were reachable
only through the agent's own ``skill_*`` MCP tools — so a user could read
what the agent had learned only by asking the agent.

Every handler DELEGATES to :mod:`src.mcp.servers.skills.handlers`. That is
deliberate: those functions already resolve the skills root through
``paths.default_skills_path()`` (honouring ``OPENAGENT_SKILLS_PATH``),
generate frontmatter, and preserve provenance. Re-implementing any of it
here would give the REST surface and the MCP tools two ways to disagree
about where a skill lives and what its frontmatter should say — the same
drift ``_resolve_vault`` exists to prevent on the vault side.

One semantic the API must not hide: a write lands on DISK, and the skills
index inside the cached system prompt is a frozen snapshot. A new or edited
skill reaches the agent's prompt on the next boot/reload, not mid-session.
Responses carry ``index_refreshed: false`` to say so out loud, so a client
can tell the user rather than leaving them wondering why the agent does not
know about the skill they just wrote.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web

from src.core.logging import elog

# Fields a create/update accepts. The frontmatter is GENERATED from these —
# a caller never supplies its own ``---`` block (the handler rejects that by
# construction, since it writes the fence itself).
_WRITABLE = ("description", "category", "body")


def _meta_to_dict(meta) -> dict:
    """One skill's metadata, without its body.

    ``created_by`` / ``status`` are surfaced raw AND as the two booleans the
    registry derives from them, because the distinction is load-bearing for a
    client: an archived skill still exists on disk but is absent from the
    prompt index, and an agent-authored skill is the only kind the curator is
    allowed to consolidate.
    """
    return {
        "name": meta.name,
        "description": meta.description,
        "category": meta.category,
        "path": str(meta.path),
        "created_by": meta.created_by,
        "status": meta.status,
        "agent_authored": meta.is_agent_authored,
        "archived": meta.is_archived,
        "from_hub": meta.is_hub,
    }


def _registry():
    from src.mcp.servers.skills.handlers import _registry as build

    return build()


async def handle_list(request):
    """GET /api/skills?include_archived= — metadata for every skill.

    Archived skills are excluded by default, matching what the prompt index
    does: the default view is "what the agent can actually reach".
    """
    from aiohttp import web

    include_archived = request.query.get("include_archived", "").lower() in ("1", "true", "yes")
    skills = _registry().skills()
    rows = [
        _meta_to_dict(m) for m in skills
        if include_archived or not m.is_archived
    ]
    rows.sort(key=lambda r: (r["category"] or "", r["name"] or ""))
    return web.json_response({"skills": rows})


async def handle_search(request):
    """GET /api/skills/search?q=&limit= — semantic when an embedding model is
    configured, plain substring otherwise. Registered BEFORE ``/{name}`` so
    ``search`` is not captured as a skill name."""
    from aiohttp import web

    from src.mcp.servers.skills.handlers import skill_search

    query = (request.query.get("q") or "").strip()
    if not query:
        return web.json_response({"error": "q is required"}, status=400)
    try:
        limit = max(1, min(int(request.query.get("limit", "20")), 100))
    except (TypeError, ValueError):
        limit = 20
    return web.json_response(await skill_search(query, limit))


async def handle_get(request):
    """GET /api/skills/{name} — the full SKILL.md body plus bundled files."""
    from aiohttp import web

    from src.mcp.servers.skills.handlers import skill_view

    result = await skill_view(request.match_info["name"])
    if not result.get("ok"):
        # ``skill_view`` answers a miss with the list of names that DO exist;
        # keep that in the 404 body — it is the useful half of the error.
        return web.json_response(result, status=404)
    return web.json_response(result)


async def _read_body(request):
    """Parse the JSON payload once, or return an error response.

    The payload must be read by exactly ONE caller: aiohttp flips
    ``can_read_body`` to False once the body has been consumed, so a handler
    that parses it and then delegates to another that parses it again gets an
    empty dict the second time — silently, as a 400 about a missing field
    that the client did send.
    """
    from aiohttp import web

    if not request.can_read_body:
        return {}, None
    try:
        payload = await request.json()
    except Exception:
        return None, web.json_response({"error": "Invalid JSON body"}, status=400)
    if not isinstance(payload, dict):
        return None, web.json_response({"error": "body must be an object"}, status=400)
    return payload, None


async def _write(request, action: str, name: str, payload: dict):
    """Shared create/update path: validate, delegate, broadcast.

    ``payload`` is passed in already parsed — see :func:`_read_body`.
    """
    from aiohttp import web

    from src.mcp.servers.skills.handlers import skill_manage

    body = payload
    fields = {k: body[k] for k in _WRITABLE if k in body}
    markdown = (fields.get("body") or "").strip()
    if action == "create" and not markdown:
        return web.json_response({"error": "body is required to create a skill"}, status=400)

    result = await skill_manage(
        action=action,
        name=name,
        body=fields.get("body"),
        description=fields.get("description"),
        category=fields.get("category"),
    )
    if not result.get("ok"):
        return web.json_response(result, status=400)

    elog(f"skill.{action}", name=name)
    gw = request.app.get("gateway")
    if gw is not None:
        await gw.broadcast_resource("skill", f"{action}d" if action != "create" else "created", name)
    # The registry the agent holds is a frozen snapshot for the cached
    # prompt; this write does not touch it. Say so rather than letting the
    # client assume the agent already knows.
    result["index_refreshed"] = False
    return web.json_response(result, status=201 if action == "create" else 200)


async def handle_create(request):
    """POST /api/skills — create a skill. ``name`` comes from the body."""
    from aiohttp import web

    payload, err = await _read_body(request)
    if err is not None:
        return err
    name = str(payload.get("name") or "").strip()
    if not name:
        return web.json_response({"error": "name is required"}, status=400)
    if _registry().get(name) is not None:
        return web.json_response(
            {"error": f"a skill named {name!r} already exists"}, status=409)
    return await _write(request, "create", name, payload)


async def handle_update(request):
    """PUT /api/skills/{name} — update description / category / body."""
    from aiohttp import web

    name = request.match_info["name"]
    if _registry().get(name) is None:
        return web.json_response({"error": f"No skill named {name!r}."}, status=404)
    payload, err = await _read_body(request)
    if err is not None:
        return err
    return await _write(request, "update", name, payload)


async def handle_delete(request):
    """DELETE /api/skills/{name} — remove the folder from disk.

    Irreversible. ``archive`` is the reversible retirement and is what a
    client should offer first.
    """
    from aiohttp import web

    from src.mcp.servers.skills.handlers import skill_manage

    name = request.match_info["name"]
    result = await skill_manage(action="remove", name=name)
    if not result.get("ok"):
        return web.json_response(result, status=404)
    elog("skill.remove", name=name)
    gw = request.app.get("gateway")
    if gw is not None:
        await gw.broadcast_resource("skill", "deleted", name)
    result["index_refreshed"] = False
    return web.json_response(result)


async def handle_archive(request):
    """POST /api/skills/{name}/archive — retire in place.

    The file stays on disk (auditable, reversible) but the registry drops it
    from the prompt index on the next reload.
    """
    from aiohttp import web

    from src.mcp.servers.skills.handlers import skill_manage

    name = request.match_info["name"]
    result = await skill_manage(action="archive", name=name)
    if not result.get("ok"):
        return web.json_response(result, status=404)
    elog("skill.archive", name=name)
    gw = request.app.get("gateway")
    if gw is not None:
        await gw.broadcast_resource("skill", "updated", name)
    result["index_refreshed"] = False
    return web.json_response(result)
