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

import asyncio
import logging
import os
import secrets
import threading
import time
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from aiohttp import web

from src.core.logging import elog
from src.network.auth.device_cert import (
    CertVerificationError,
    DeviceCert,
    verify_cert,
)
from src.network.auth.peer_policy import check_peer_request
from src.network.identity import iroh_node_id_public_bytes
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
    after sig+expiry verification because the cert itself doesn't carry
    revocation state. Member gateways use an authenticated coordinator RPC
    for the same live check and periodically revalidate open transports.
    """

    def __init__(
        self,
        *,
        coordinator_pubkey: Ed25519PublicKey,
        network_id: str,
        revoked_pubkeys: set[bytes] | None = None,
        device_lookup: Callable[[bytes], Awaitable[Any]] | None = None,
    ) -> None:
        self.coordinator_pubkey = coordinator_pubkey
        self.network_id = network_id
        self.revoked_pubkeys = revoked_pubkeys or set()
        self._device_lookup = device_lookup
        self._revocation_listeners: list[Callable[[bytes], Any]] = []
        self._observed_active_devices: set[bytes] = set()
        # Every disconnect (hard revoke, suspension, roster uncertainty)
        # advances a device-local epoch synchronously, before connection-owner
        # callbacks are scheduled.  Long-lived WebSocket handlers carry the
        # epoch that was authenticated by this middleware and can therefore
        # reject a late registration even when their HTTP upgrade raced the
        # asynchronous close callback.  The lock keeps the counter safe if a
        # coordinator watcher invokes ``disconnect`` from another thread.
        self._device_epochs: dict[bytes, int] = {}
        self._device_epoch_lock = threading.Lock()

    def device_epoch(self, device_pubkey: bytes) -> int:
        """Return the current transport-authorization epoch for *device*."""

        key = bytes(device_pubkey)
        with self._device_epoch_lock:
            return self._device_epochs.get(key, 0)

    async def device_is_active(self, device_pubkey: bytes) -> bool:
        """Return live coordinator membership for a device.

        Coordinator gateways query their local roster. Member gateways query
        the same authoritative roster over an agent-authenticated Iroh RPC.
        """
        key = bytes(device_pubkey)
        if self._device_lookup is None:
            active = key not in self.revoked_pubkeys
        else:
            row = await self._device_lookup(key)
            active = row is not None and getattr(row, "status", None) == "active"
        if active:
            self._observed_active_devices.add(key)
        else:
            self._observed_active_devices.discard(key)
        return active

    async def revalidate_observed_devices(self) -> list[bytes]:
        """Fail closed and disconnect devices no longer live in the roster.

        A lookup failure closes existing transports but does not add a
        permanent local revocation: once the coordinator is reachable again,
        an actually-active device may establish a fresh authenticated stream.
        """
        if self._device_lookup is None or not self._observed_active_devices:
            return []
        keys = tuple(self._observed_active_devices)

        async def check(key: bytes) -> tuple[bytes, bool]:
            try:
                row = await self._device_lookup(key)
                active = row is not None and getattr(row, "status", None) == "active"
            except Exception:  # noqa: BLE001 - roster uncertainty is fail-closed
                active = False
            return key, active

        disconnected: list[bytes] = []
        for key, active in await asyncio.gather(*(check(key) for key in keys)):
            if active:
                continue
            self._observed_active_devices.discard(key)
            self.disconnect(key)
            disconnected.append(key)
        return disconnected

    def add_revocation_listener(self, callback: Callable[[bytes], Any]) -> None:
        if callback not in self._revocation_listeners:
            self._revocation_listeners.append(callback)

    def revoke(self, device_pubkey: bytes) -> None:
        """Update the live deny-set and notify connection owners immediately."""
        key = bytes(device_pubkey)
        self.revoked_pubkeys.add(key)
        self._observed_active_devices.discard(key)
        self.disconnect(key)

    def disconnect(self, device_pubkey: bytes) -> None:
        """Close live transports without permanently revoking the pairing.

        Account suspension uses this path. The live roster rejects new
        streams while suspended, but reactivation can make the same device
        usable again. Device deletion/revocation uses :meth:`revoke` instead.
        """
        key = bytes(device_pubkey)
        with self._device_epoch_lock:
            self._device_epochs[key] = self._device_epochs.get(key, 0) + 1
        for callback in tuple(self._revocation_listeners):
            try:
                callback(key)
            except Exception:  # noqa: BLE001 - revocation must continue fan-out
                logger.exception("device revocation listener failed")


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

        # Liveness-probe exemption. ``GET /api/health`` is the ONE route that
        # skips auth entirely, so a k8s liveness/readiness probe can issue a
        # PLAIN httpGet with no device cert and no bearer token — the earlier
        # attempt to smuggle a token into an exec probe mangled its shell
        # quoting and crash-looped a pod. The match is EXACT on the resolved
        # path (``rel_url.path`` has query string + params stripped), NOT a
        # prefix, so it can never be widened to reach ``/api/health/ingest`` or
        # any other ``/api/*`` route. The handler itself returns only basic
        # liveness status (agent name, version, connected-client count) — no
        # secrets, config, or session data — so exposing it unauthenticated is
        # safe. Everything else below still requires a cert/token.
        if request.rel_url.path == "/api/health":
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
                request["auth_kind"] = "http_token"
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
            request["auth_kind"] = "agent"
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

        # The device certificate and the Iroh QUIC peer are two views of the
        # same Ed25519 key.  Signature verification alone proves only that the
        # coordinator once issued the certificate; this comparison proves the
        # current caller still owns the corresponding private key.
        peer_node_id = current_peer_node_id() or ""
        try:
            peer_pubkey = iroh_node_id_public_bytes(peer_node_id)
        except ValueError:
            elog(
                "auth.fail",
                level="warning",
                reason="invalid_peer_identity",
                path=request.path,
                peer=peer_node_id or "unknown",
            )
            return web.Response(status=401, text="device cert peer identity unavailable")
        if not secrets.compare_digest(peer_pubkey, cert.device_pubkey):
            elog(
                "auth.fail",
                level="warning",
                reason="peer_key_mismatch",
                path=request.path,
                peer=peer_node_id,
                device=cert.device_pubkey_hex,
            )
            return web.Response(status=401, text="device cert does not belong to Iroh peer")

        if cert.device_pubkey in state.revoked_pubkeys:
            elog(
                "auth.fail",
                level="warning",
                reason="revoked",
                handle=cert.handle,
                device=cert.device_pubkey_hex,
            )
            return web.Response(status=401, text="device revoked")

        # In-process channel bridges have coordinator-signed, PoP-bound
        # identities but deliberately are not user devices in the coordinator
        # roster.  Every real human device must still be present and active.
        # Snapshot before the potentially-suspending roster lookup.  If a
        # revoke/suspension lands while the lookup is in flight, its result is
        # stale even if it says "active" and this request must fail closed.
        device_auth_epoch = state.device_epoch(cert.device_pubkey)
        try:
            device_active = (
                True
                if "bridge" in cert.capabilities
                else await state.device_is_active(cert.device_pubkey)
            )
        except Exception as exc:  # noqa: BLE001 - auth must fail closed
            elog(
                "auth.fail",
                level="warning",
                reason="device_liveness_unavailable",
                handle=cert.handle,
                device=cert.device_pubkey_hex,
                error_type=type(exc).__name__,
            )
            return web.Response(status=503, text="device membership unavailable")
        if not device_active:
            # The authoritative roster remains the persistent source. Do not
            # turn a reversible account suspension into a permanent local
            # revocation; actual revoke events call ``state.revoke`` directly.
            state.disconnect(cert.device_pubkey)
            elog(
                "auth.fail",
                level="warning",
                reason="device_inactive_or_missing",
                handle=cert.handle,
                device=cert.device_pubkey_hex,
            )
            return web.Response(status=401, text="device inactive or removed")
        if state.device_epoch(cert.device_pubkey) != device_auth_epoch:
            elog(
                "auth.fail",
                level="warning",
                reason="device_authorization_changed",
                handle=cert.handle,
                device=cert.device_pubkey_hex,
            )
            return web.Response(status=401, text="device authorization changed")

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
        request["device_auth_epoch"] = device_auth_epoch
        # Set only after signature, network, live-roster and cert↔Iroh peer
        # proof-of-possession checks have all succeeded. Handlers must use
        # this provenance marker rather than infer trust from a synthetic
        # DeviceCert shape or a non-empty transport context variable.
        request["auth_kind"] = "device_cert"
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
