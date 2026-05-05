"""aiohttp middleware that gates every request on a valid device cert.

Replaces the legacy ``_check_bearer_token`` (which compared a single
shared token across every endpoint) with per-device, signed,
expiring credentials. The cert wire bytes are pulled from the
contextvar set by ``IrohSite`` for each accepted stream — they are
*not* sourced from request headers, which a peer could forge.

On success the verified ``DeviceCert`` is placed at
``request['device_cert']`` and the device's pubkey hex at
``request['client_id']`` (used by the gateway to scope sessions).

On failure we return ``401 unauthorized`` with a short reason. The
old WS auth-frame handshake is gone — by the time a WS upgrade lands
here, the cert has already been verified at the HTTP layer.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from aiohttp import web

from openagent.core.logging import elog
from openagent.network.auth.device_cert import (
    CertVerificationError,
    DeviceCert,
    verify_cert,
)
from openagent.network.transport.aiohttp_iroh_site import (
    current_device_cert_wire,
    current_peer_node_id,
)

if TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

logger = logging.getLogger(__name__)


class NetworkAuthState:
    """Per-gateway pinned state used by the middleware.

    ``coordinator_pubkey`` is the verify key for our home network's
    coordinator (loaded from the ``network`` row at gateway startup).

    ``revoked_pubkeys`` is a fast in-memory set populated from the
    ``network_devices`` table on coordinator-managed agents — checked
    after sig+expiry verification because the cert itself doesn't
    carry revocation state. Member-only agents leave this empty and
    rely on the cert's TTL for liveness.
    """

    def __init__(
        self,
        *,
        coordinator_pubkey: Ed25519PublicKey,
        network_id: str,
        revoked_pubkeys: set[bytes] | None = None,
    ) -> None:
        self.coordinator_pubkey = coordinator_pubkey
        self.network_id = network_id
        self.revoked_pubkeys = revoked_pubkeys or set()


def make_auth_middleware(state: NetworkAuthState):
    """Build the aiohttp middleware closure bound to *state*.

    We use a closure rather than reading from request.app because the
    state is gateway-wide, not per-request — and re-reading from the
    DB on every request would be slow.
    """

    @web.middleware
    async def auth_middleware(request: web.Request, handler):
        # OPTIONS / preflight passes through. The CORS middleware (set
        # up earlier in the chain) handles the actual response shape.
        if request.method == "OPTIONS":
            return await handler(request)

        wire = current_device_cert_wire()
        if not wire:
            elog(
                "auth.fail",
                level="warning",
                reason="no_cert",
                path=request.path,
                peer=current_peer_node_id() or "unknown",
            )
            return web.Response(status=401, text="missing device cert")

        try:
            cert = verify_cert(
                wire,
                coordinator_pubkey=state.coordinator_pubkey,
                expected_network_id=state.network_id,
            )
        except CertVerificationError as e:
            elog(
                "auth.fail",
                level="warning",
                reason=str(e),
                path=request.path,
                peer=current_peer_node_id() or "unknown",
            )
            return web.Response(status=401, text=f"cert rejected: {e}")

        if cert.device_pubkey in state.revoked_pubkeys:
            elog(
                "auth.fail",
                level="warning",
                reason="revoked",
                handle=cert.handle,
                device=cert.device_pubkey_hex,
            )
            return web.Response(status=401, text="device revoked")

        # Annotate the request so handlers + the WS auth path see the
        # authenticated identity. ``client_id`` was a freely-chosen
        # string in the legacy protocol; locking it to the device
        # pubkey hex prevents a reconnect from impersonating someone
        # else's open StreamSessions.
        request["device_cert"] = cert
        request["client_id"] = cert.device_pubkey_hex
        request["network_id"] = cert.network_id
        return await handler(request)

    return auth_middleware


def device_cert_or_401(request: web.Request) -> DeviceCert:
    """Convenience for ad-hoc handlers that want the cert directly.

    Most code paths can read ``request['device_cert']`` after the
    middleware has run; this helper wraps the lookup with a clean
    ``HTTPUnauthorized`` raise so callers don't need to handle a
    missing key as a special case.
    """
    cert = request.get("device_cert")
    if cert is None:
        raise web.HTTPUnauthorized(text="middleware did not set device_cert — wiring bug")
    if not isinstance(cert, DeviceCert):
        raise web.HTTPUnauthorized(text="device cert payload corrupted")
    return cert


def is_cert_due_for_refresh(cert: DeviceCert, *, now: float | None = None) -> bool:
    """Return True if this cert has crossed the 50% TTL refresh threshold."""
    n = now or time.time()
    midpoint = cert.issued_at + (cert.expires_at - cert.issued_at) * 0.5
    return n >= midpoint
