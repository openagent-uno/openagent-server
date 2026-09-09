"""Certificate-authenticated policy for live collaboration.

An embedding host may install ``gateway.collaboration_authorizer``, an async
callable ``(request, targets, permission) -> {userId, name, allowed}``. This is
an in-process extension, never a client-selected identity or network endpoint.
The native policy checks the same canonical ACL rows as operational REST.
"""

from __future__ import annotations

from src.memory.operational.access import AccessContext, resource_is_visible
from src.core.on_behalf_context import OnBehalfIdentity


async def authorize(request, targets, permission="view"):
    from .collaboration_hub import target_key

    gateway = request.app["gateway"]
    cert = request.get("device_cert")
    if cert is None or not await gateway._request_device_still_authorized(
        request, cert
    ):
        raise PermissionError("Device authorization changed")
    if permission not in {"view", "admin"}:
        raise PermissionError("Invalid permission")
    OnBehalfIdentity.from_certificate(
        cert, auth_kind=str(request.get("auth_kind") or "")
    )
    requested = {target_key(target) for target in targets}
    policy = getattr(gateway, "collaboration_authorizer", None)
    if policy is not None:
        result = await policy(request, targets, permission)
    else:
        access = AccessContext.from_request(request)
        conn = await gateway.agent.memory_db._ensure_connected()
        allowed = []
        resource_types = {
            "workflow": "workflow_definition",
            "scheduled_task": "scheduled_definition",
            "event": "event_definition",
        }
        resource_tables = {
            "workflow": "workflow_tasks",
            "scheduled_task": "scheduled_tasks",
            "event": "events",
        }
        for target in targets:
            kind, resource_id = target_key(target)
            if kind == "session":
                row = await (
                    await conn.execute(
                        "SELECT id AS resource_id, 'session' AS resource_type, tenant_id, "
                        "owner_principal_id, visibility, acl_version FROM sessions_v2 "
                        "WHERE id=? AND deleted_at_ms IS NULL",
                        (resource_id,),
                    )
                ).fetchone()
            else:
                # Definitions are still stored in the legacy tables. Owner
                # records may outlive deletion for audit/history purposes.
                exists = await (
                    await conn.execute(
                        f"SELECT id FROM {resource_tables[kind]} WHERE id=?",
                        (resource_id,),
                    )
                ).fetchone()
                if exists is None:
                    continue
                row = await (
                    await conn.execute(
                        "SELECT * FROM operational_resource_owners WHERE tenant_id=? "
                        "AND resource_type=? AND resource_id=?",
                        (access.tenant_id, resource_types[kind], resource_id),
                    )
                ).fetchone()
                if row is None:
                    # Same documented read-only default as the operational
                    # automation projector for unclaimed legacy definitions.
                    row = {
                        "tenant_id": access.tenant_id,
                        "resource_type": resource_types[kind],
                        "resource_id": resource_id,
                        "owner_principal_id": None,
                        "visibility": "installation_shared",
                        "acl_version": 1,
                    }
            if row is not None and await resource_is_visible(
                conn, row, access, permission=permission
            ):
                allowed.append(target)
        result = {
            "userId": access.principal_id,
            "name": access.handle,
            "allowed": allowed,
        }
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("userId"), str)
        or not result["userId"]
        or len(result["userId"]) > 1024
        or not isinstance(result.get("name"), str)
        or len(result["name"]) > 1024
        or not isinstance(result.get("allowed"), list)
        or len(result["allowed"]) > len(targets)
        or any(target_key(target) not in requested for target in result["allowed"])
    ):
        raise PermissionError("Invalid collaboration scope")
    # A policy/DB read may yield across a device revocation.
    if not await gateway._request_device_still_authorized(request, cert):
        raise PermissionError("Device authorization changed")
    return result
