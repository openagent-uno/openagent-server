"""Owner-managed native session grants; embedding hosts keep their own policy."""

import re
import time

import aiosqlite
from aiohttp import web

from src.memory.operational.access import AccessContext, resource_is_visible


async def handle_members(request):
    gateway = request.app["gateway"]
    if getattr(gateway, "collaboration_authorizer", None) is not None:
        return web.json_response(
            {"error": "Session sharing is managed by the host"}, status=409
        )
    sid = request.match_info["session_id"]
    cert = request.get("device_cert")
    if not cert or not await gateway._request_device_still_authorized(request, cert):
        return web.json_response({"error": "Authentication required"}, status=401)
    access = AccessContext.from_request(request)
    db = gateway.agent.memory_db
    async with aiosqlite.connect(db.db_path, timeout=10) as conn:
        conn.row_factory = aiosqlite.Row
        if request.method == "PUT":
            await conn.execute("BEGIN IMMEDIATE")
        row = await (
            await conn.execute(
                "SELECT id AS resource_id, 'session' AS resource_type, tenant_id, owner_principal_id, visibility, acl_version "
                "FROM sessions_v2 WHERE id=? AND deleted_at_ms IS NULL",
                (sid,),
            )
        ).fetchone()
        if row is None or not await resource_is_visible(conn, row, access):
            return web.json_response({"error": "Session unavailable"}, status=404)
        owner = (
            row["owner_principal_id"] in access.principal_ids
            and row["tenant_id"] == access.tenant_id
        )
        if request.method == "PUT":
            if not owner:
                return web.json_response(
                    {"error": "Only the session owner can change sharing"}, status=403
                )
            try:
                body = await request.json()
                if not isinstance(body, dict) or set(body) != {"handle", "permission"}:
                    raise ValueError()
                handle, permission = body["handle"], body["permission"]
                if not isinstance(handle, str) or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", handle
                ):
                    raise ValueError()
                if permission not in {None, "view", "admin"}:
                    raise ValueError()
            except (ValueError, TypeError):
                return web.json_response(
                    {"error": "Expected handle and view/admin/null permission"},
                    status=400,
                )
            if not await gateway._request_device_still_authorized(request, cert):
                return web.json_response(
                    {"error": "Authentication changed"}, status=401
                )
            await conn.execute(
                "DELETE FROM resource_acl WHERE tenant_id=? AND resource_type='session' AND resource_id=? AND principal_type='user' AND principal_id IN (?,?)",
                (access.tenant_id, sid, handle, "user:" + handle),
            )
            # Advance all still-valid grants atomically with the resource.
            # Updating one member must neither revoke the others nor revive
            # historical grants invalidated by a previous ACL generation.
            await conn.execute(
                "UPDATE resource_acl SET acl_version=acl_version+1 "
                "WHERE tenant_id=? AND resource_type='session' AND resource_id=? AND acl_version=?",
                (access.tenant_id, sid, row["acl_version"]),
            )
            if permission:
                await conn.execute(
                    "INSERT INTO resource_acl (tenant_id,resource_type,resource_id,principal_type,principal_id,permission,acl_version,granted_by_principal_id,granted_at_ms) "
                    "VALUES (?, 'session', ?, 'user', ?, ?, ?, ?, ?)",
                    (
                        access.tenant_id,
                        sid,
                        handle,
                        permission,
                        row["acl_version"] + 1,
                        access.principal_id,
                        int(time.time() * 1000),
                    ),
                )
            await conn.execute(
                "UPDATE sessions_v2 SET acl_version=acl_version+1 WHERE id=?", (sid,)
            )
            await conn.commit()
        members = await (
            await conn.execute(
                "SELECT principal_id AS handle, permission FROM resource_acl WHERE tenant_id=? AND resource_type='session' AND resource_id=? AND principal_type='user' "
                "AND acl_version=(SELECT acl_version FROM sessions_v2 WHERE id=resource_id) ORDER BY principal_id",
                (access.tenant_id, sid),
            )
        ).fetchall()
    if request.method == "PUT":
        collaboration = gateway._collaboration
        collaboration.hub.wake()
        runtime = collaboration.runtimes.get(sid)
        if runtime:
            async with runtime.admission:
                if runtime.active:
                    try:
                        await collaboration._check(runtime.active["request"], sid)
                    except Exception:
                        await collaboration._interrupt(runtime)
        await gateway.broadcast_resource("session", "updated", sid)
    if not await gateway._request_device_still_authorized(request, cert):
        return web.json_response({"error": "Authentication changed"}, status=401)
    return web.json_response(
        {
            "owner": row["owner_principal_id"],
            "can_manage": owner,
            "members": [dict(m) for m in members],
        }
    )
