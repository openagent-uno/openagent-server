"""Authenticated principal carried through one agent turn.

Operational-history search is an in-process capability because the model must
never be allowed to choose the tenant or principal it searches as.  The
gateway binds the verified device certificate here before entering the agent
runtime; :mod:`src.mcp.servers.memory_search.adapters` reads the value at tool
execution time.  ``ContextVar`` propagation also gives delegated child tasks
the same subject without serialising a reusable credential.

This is deliberately separate from ``identity_context``.  Message authorship
may be supplied by a trusted bridge for display and multi-author transcripts;
authorization must come only from the gateway certificate.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OnBehalfIdentity:
    """Immutable identity copied from an already-verified device cert."""

    tenant_id: str
    principal_type: str
    handle: str
    device_id: str

    @classmethod
    def from_certificate(cls, cert: Any) -> "OnBehalfIdentity":
        if cert is None:
            raise PermissionError("authenticated device context is required")
        tenant = str(getattr(cert, "network_id", "") or "").strip()
        handle = str(getattr(cert, "handle", "") or "").strip()
        device = str(getattr(cert, "device_pubkey_hex", "") or "").strip()
        capabilities = set(getattr(cert, "capabilities", ()) or ())
        principal_type = "agent" if "agent" in capabilities else "user"
        if not tenant or not handle or not device:
            raise PermissionError("authenticated device context is incomplete")
        if any(len(value) > 1024 for value in (tenant, handle, device)):
            raise PermissionError("authenticated device context is invalid")
        return cls(tenant, principal_type, handle, device)


_identity_var: contextvars.ContextVar[OnBehalfIdentity | None] = contextvars.ContextVar(
    "openagent_on_behalf_identity",
    default=None,
)


def install_on_behalf_identity(identity: OnBehalfIdentity | None):
    """Bind a server-authenticated identity for the duration of one turn."""

    return _identity_var.set(identity)


def reset_on_behalf_identity(token: contextvars.Token) -> None:
    _identity_var.reset(token)


def current_on_behalf_identity() -> OnBehalfIdentity | None:
    """Return the authenticated turn subject, never a model-authored value."""

    return _identity_var.get()
