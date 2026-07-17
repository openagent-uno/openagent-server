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
import os
import secrets
import time
from typing import TYPE_CHECKING

from aiohttp import web

from src.core.logging import elog
from src.network.auth.device_cert import (
    CertVerificationError,
    DeviceCert,
    verify_cert,
)
from src.network.auth.peer_policy import check_peer_request
from src.network.transport.aiohttp_iroh_site import (
    current_device_cert_wire,
    current_is_authenticated_agent,
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

    When ``OPENAGENT_HTTP_TOKEN`` is set in the environment, requests
    carrying a matching ``X-OpenAgent-Token`` header skip device-cert
    verification entirely. This is the fast-path used by deployments
    that front the gateway with a trusted reverse proxy (Virgil's
    orchestrator, ngrok tunnels for local dev, etc.) where the Iroh
    transport isn't reachable. The token is captured at middleware
    construction time, not per-request, so changing the env var
    requires a gateway restart.

    An OPTIONAL second token, ``OPENAGENT_LLM_TOKEN``, is a
    least-privilege credential scoped to the LLM gateway ONLY: a holder
    of that token (and NOT the full ``OPENAGENT_HTTP_TOKEN``) may reach
    ``/api/llm/*`` and is rejected on every other route. This lets a
    caller that only needs the LLM gateway (e.g. Replio's reply-guard
    calling ``/api/llm/chat/completions``) be handed a token that cannot
    reach vault, config, scheduled-tasks, terminal-backed chat, etc. It
    is fully backward compatible: when ``OPENAGENT_LLM_TOKEN`` is unset,
    behaviour is identical to today, and the full token keeps working
    everywhere.
    """

    http_token = os.environ.get("OPENAGENT_HTTP_TOKEN", "").strip()
    # Optional SECOND, least-privilege token scoped to the LLM gateway.
    # Captured at construction time like the full token above. Unset ⇒
    # ``accepted`` below is just ``[http_token]`` on every path, exactly
    # as before this token existed.
    llm_token = os.environ.get("OPENAGENT_LLM_TOKEN", "").strip()

    @web.middleware
    async def auth_middleware(request: web.Request, handler):
        # OPTIONS / preflight passes through. The CORS middleware (set
        # up earlier in the chain) handles the actual response shape.
        if request.method == "OPTIONS":
            return await handler(request)

        # HTTP token bypass: trusted reverse proxies present a shared
        # secret in lieu of a device cert. We synthesize a minimal
        # request['device_cert'] so downstream handlers don't need to
        # branch on transport. The synthesized ``device_pubkey`` is 32
        # random bytes per connection — that's what the gateway uses as
        # the ``client_id`` dict key (StreamSession lookup, last-write-
        # wins kick on collision), and a stable key would cause
        # legitimate reconnects to kick each other in a loop.
        if http_token or llm_token:
            # Two equivalent ways to present a shared secret, both compared
            # with ``secrets.compare_digest`` — this only ADDS accepted
            # headers, it does not weaken anything:
            #   1. ``X-OpenAgent-Token: <token>`` — the original bridge
            #      header (Virgil's orchestrator, ngrok dev tunnels).
            #   2. ``Authorization: Bearer <token>`` — the standard header
            #      every OpenAI-compatible client sends, so the whole
            #      gateway (incl. /api/llm/chat/completions) is reachable
            #      by an off-the-shelf OpenAI SDK with no custom headers.
            # We gather whatever tokens the request presented (both header
            # forms are honoured for BOTH tokens) and accept if ANY of them
            # matches ANY token valid for THIS request's path; a valid
            # X-OpenAgent-Token carrying the full token keeps working exactly
            # as before.
            candidates: list[str] = []
            x_token = request.headers.get("X-OpenAgent-Token", "").strip()
            if x_token:
                candidates.append(x_token)
            authz = request.headers.get("Authorization", "").strip()
            if authz[:7].lower() == "bearer ":  # strip prefix case-insensitively
                bearer = authz[7:].strip()
                if bearer:
                    candidates.append(bearer)

            # Least-privilege scoping by request PATH. The full
            # ``OPENAGENT_HTTP_TOKEN`` is accepted on EVERY route (unchanged).
            # The optional ``OPENAGENT_LLM_TOKEN`` is accepted ONLY on the LLM
            # gateway (paths under ``/api/llm/``) — so a caller holding only
            # that token reaches the LLM routes and is rejected everywhere
            # else (falling through to the device-cert path → 401). When
            # ``OPENAGENT_LLM_TOKEN`` is unset this list is just
            # ``[http_token]`` on every path, i.e. behaviour is unchanged.
            path = request.rel_url.path
            accepted: list[str] = []
            if http_token:
                accepted.append(http_token)
            if llm_token and path.startswith("/api/llm/"):
                accepted.append(llm_token)

            if candidates and accepted and any(
                secrets.compare_digest(c, tok)
                for c in candidates
                for tok in accepted
            ):
                handle_hint = request.headers.get("X-OpenAgent-Handle", "") or "http-bridge"
                synthetic = DeviceCert(
                    handle=handle_hint[:64],
                    device_pubkey=secrets.token_bytes(32),
                    network_id=state.network_id,
                    issued_at=time.time(),
                    expires_at=time.time() + 365 * 24 * 3600,
                    capabilities=[],
                )
                request["device_cert"] = synthetic
                request["client_id"] = synthetic.device_pubkey_hex
                request["user_handle"] = synthetic.handle
                request["network_id"] = synthetic.network_id
                return await handler(request)

        # Agent ALPN bypass: the Iroh QUIC handshake proves node_id
        # ownership — no coordinator cert is needed. Synthesise a minimal
        # DeviceCert so downstream handlers don't need to branch.
        if current_is_authenticated_agent():
            import hashlib as _hashlib
            peer_id = current_peer_node_id() or "unknown-agent"
            # Phase-0 security: record every first-contact agent node_id so the
            # allowlist can be built/audited and unexpected dialers spotted.
            # This line is why ``network.peers.allowlist`` is enable-able on a
            # live mesh: it has been accumulating the operator's real peer list
            # since the ALPN shipped, so building the allowlist is a grep, not
            # a maintenance window. Kept unconditional for exactly that reason.
            elog("agent.contact", level="info", node_id=peer_id, path=request.path)

            # The enforcement that line was staged for. Proving node_id
            # ownership is not enrolment: nothing here consumed an invite or
            # checked a coordinator cert, so without this gate any node that
            # can dial reaches EVERY route on the shared app — see
            # ``peer_policy`` for what that really opens. It is not just
            # ``/api/events/{id}/trigger``: ``POST /api/mcps`` takes arbitrary
            # argv and the pool spawns it. ``capabilities=["agent"]`` below is
            # a provenance label, not a scope — the only two readers of it
            # (events.py, chat.py) use it to pick a source string and a session
            # prefix, and no ``require_capability`` helper exists in src/.
            #
            # Both toggles default OFF and ``check_peer_request`` returns
            # before reading any list when they are — an allowlist that armed
            # itself on upgrade would cut a running mesh dead with no warning.
            denial = check_peer_request(
                node_id=peer_id, method=request.method, path=request.path,
            )
            if denial is not None:
                return web.Response(status=403, text=denial)
            # Derive a stable synthetic device_pubkey from the peer's node_id.
            # sha256 gives us a deterministic 32-byte key that's unique per
            # peer without requiring the coordinator.
            device_pubkey = _hashlib.sha256(f"agent:{peer_id}".encode()).digest()
            handle = f"agent:{peer_id[:16]}"
            synthetic = DeviceCert(
                handle=handle,
                device_pubkey=device_pubkey,
                network_id=state.network_id,
                issued_at=time.time(),
                expires_at=time.time() + 3600,
                capabilities=["agent"],
            )
            request["device_cert"] = synthetic
            request["client_id"] = synthetic.device_pubkey_hex
            request["user_handle"] = synthetic.handle
            request["network_id"] = state.network_id
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
        # else's open StreamSessions. ``user_handle`` is the same user
        # across all of their devices — used by the session list so
        # device A's chats show up on device B after re-login.
        request["device_cert"] = cert
        request["client_id"] = cert.device_pubkey_hex
        request["user_handle"] = cert.handle
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
