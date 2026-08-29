"""Canonical principal and ACL checks for operational history/search.

Callers never supply a principal.  The gateway builds :class:`AccessContext`
from the authenticated device certificate and every API/search candidate is
rechecked against the canonical operational database before serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AccessContext:
    tenant_id: str
    principal_id: str
    principal_type: str
    handle: str
    device_id: str
    principal_ids: frozenset[str]
    grant_identities: frozenset[tuple[str, str]]

    @classmethod
    def from_request(cls, request: Any) -> "AccessContext":
        cert = request.get("device_cert")
        tenant = str(request.get("network_id") or getattr(cert, "network_id", "")).strip()
        handle = str(request.get("user_handle") or getattr(cert, "handle", "")).strip()
        device = str(request.get("client_id") or getattr(cert, "device_pubkey_hex", "")).strip()
        if cert is None or not tenant or not handle or not device:
            raise PermissionError("authenticated device context is required")
        capabilities = set(getattr(cert, "capabilities", ()) or ())
        principal_type = "agent" if "agent" in capabilities else "user"
        principal_id = f"{principal_type}:{handle}"
        # Compatibility aliases retain their principal type. Coordinator
        # storage permits a human and an agent to have the same handle, so an
        # ambiguous raw handle can never be an authorization identity.
        typed_handle = f"{principal_type}:{handle}"
        typed_device = f"{principal_type}:{device}"
        ids = frozenset(
            {
                typed_handle,
                typed_device,
                f"device:{device}",
            }
        )
        grants = frozenset(
            {
                (principal_type, handle),
                (principal_type, typed_handle),
                ("device", device),
                ("device", f"device:{device}"),
                ("installation", tenant),
            }
        )
        return cls(tenant, principal_id, principal_type, handle, device, ids, grants)

    @classmethod
    def from_on_behalf_identity(cls, identity: Any) -> "AccessContext":
        """Build the same ACL identity from a server-bound turn context.

        The operational-search MCP deliberately exposes no identity fields in
        its tool schema.  Its in-process adapter reads an
        :class:`~src.core.on_behalf_context.OnBehalfIdentity` that the gateway
        copied from the verified certificate and converts it here, preserving
        exact parity with REST authorization.
        """

        if identity is None:
            raise PermissionError("authenticated on-behalf-of context is required")
        tenant = str(getattr(identity, "tenant_id", "") or "").strip()
        handle = str(getattr(identity, "handle", "") or "").strip()
        device = str(getattr(identity, "device_id", "") or "").strip()
        principal_type = str(getattr(identity, "principal_type", "") or "").strip()
        if principal_type not in {"user", "agent"} or not tenant or not handle or not device:
            raise PermissionError("authenticated on-behalf-of context is incomplete")
        principal_id = f"{principal_type}:{handle}"
        ids = frozenset(
            {
                principal_id,
                f"{principal_type}:{device}",
                f"device:{device}",
            }
        )
        grants = frozenset(
            {
                (principal_type, handle),
                (principal_type, principal_id),
                ("device", device),
                ("device", f"device:{device}"),
                ("installation", tenant),
            }
        )
        return cls(
            tenant,
            principal_id,
            principal_type,
            handle,
            device,
            ids,
            grants,
        )


def row_is_visible_without_grant(row: Any, access: AccessContext) -> bool:
    if str(row["tenant_id"]) != access.tenant_id:
        return False
    visibility = str(row["visibility"])
    if visibility == "quarantined":
        return False
    if visibility in {"installation_shared", "public"}:
        return True
    owner = row["owner_principal_id"]
    return owner is not None and str(owner) in access.principal_ids


async def resource_is_visible(
    conn: Any,
    row: Any,
    access: AccessContext,
    *,
    permission: str = "view",
) -> bool:
    """Recheck one canonical resource, including explicit ACL grants.

    Installation/public visibility grants read/search access only.  Mutating
    permissions such as ``admin`` still require ownership or an explicit ACL;
    otherwise a readable shared dashboard would also be writable/executable by
    every principal in the tenant.
    """

    if str(row["tenant_id"]) != access.tenant_id or str(row["visibility"]) == "quarantined":
        return False
    owner = row["owner_principal_id"]
    if owner is not None and str(owner) in access.principal_ids:
        return True
    if (
        permission in {"view", "search"}
        and str(row["visibility"]) in {"installation_shared", "public"}
    ):
        return True
    for principal_type, principal_id in access.grant_identities:
        match = await (
            await conn.execute(
                "SELECT 1 FROM resource_acl WHERE tenant_id=? AND resource_type=? "
                "AND resource_id=? AND principal_type=? AND principal_id=? "
                "AND permission IN (?, 'admin') AND acl_version=? LIMIT 1",
                (
                    access.tenant_id,
                    str(row["resource_type"]),
                    str(row["resource_id"]),
                    principal_type,
                    principal_id,
                    permission,
                    int(row["acl_version"]),
                ),
            )
        ).fetchone()
        if match is not None:
            return True
    return False
