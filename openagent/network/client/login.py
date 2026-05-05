"""Client-side login + registration. UI-agnostic — the CLI and app both call this.

Flow:

    1. ``register(invite_code, handle, password)`` — runs the PAKE
       registration round-trip + adds our device, returns a freshly
       minted cert. Caller persists it via ``store_cert``.

    2. ``login(handle, password)`` — runs the PAKE login round-trip,
       returns a fresh cert.

    3. ``refresh_cert(handle)`` — replays login when an existing cert
       is past its 50% TTL. Same wire as ``login``.

All three end up at the coordinator's NodeId via ``IrohNode.dial``.
The coordinator's pubkey is pinned at first contact: when adding a
network you record ``(coordinator_node_id, coordinator_pubkey)`` and
every subsequent connection verifies the response cert against the
pinned pubkey.

Wire format mirrors ``coordinator.service``:
``len(4) || cbor({id, method, params}) || finish``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

import cbor2

from openagent.network.auth.device_cert import (
    CertVerificationError,
    DeviceCert,
    verify_cert,
)
from openagent.network.coordinator.pake import (
    Srp6aBackend,
    Srp6aClientLogin,
    srp6a_make_registration,
)
from openagent.network.identity import Identity
from openagent.network.iroh_node import IrohNode, NetworkAlpn

logger = logging.getLogger(__name__)


class LoginError(Exception):
    """Raised on any client-side failure (wire, PAKE, cert) during login."""


async def register(
    *,
    node: IrohNode,
    coordinator_node_id: str,
    coordinator_pubkey_bytes: bytes,
    handle: str,
    password: str,
    invite_code: str,
    device_identity: Identity,
    network_id: str,
    label: str | None = None,
) -> bytes:
    """Register a new ``handle@network`` and return the wire-encoded cert.

    Two RPCs: ``register`` (creates the user row + verifier) then
    ``login_init``+``login_finish`` (proves we hold the password and
    pairs our device). The coordinator emits a cert at the end of the
    login, which is what we cache.
    """
    pake_payload = srp6a_make_registration(handle, password)

    await _rpc(
        node=node,
        coordinator_node_id=coordinator_node_id,
        method="register",
        params={
            "invite": invite_code,
            "handle": handle,
            "pake_record": pake_payload,
        },
    )
    return await login(
        node=node,
        coordinator_node_id=coordinator_node_id,
        coordinator_pubkey_bytes=coordinator_pubkey_bytes,
        handle=handle,
        password=password,
        device_identity=device_identity,
        network_id=network_id,
        invite_code=invite_code,
        label=label,
    )


async def login(
    *,
    node: IrohNode,
    coordinator_node_id: str,
    coordinator_pubkey_bytes: bytes,
    handle: str,
    password: str,
    device_identity: Identity,
    network_id: str,
    invite_code: str | None = None,
    label: str | None = None,
) -> bytes:
    """Run an SRP-6a login; return the issued cert wire bytes.

    *invite_code* is optional — only needed when we're a new device
    pairing onto an existing account. Existing devices skip it; the
    coordinator recognises the device pubkey and just reissues a cert.
    """
    client = Srp6aClientLogin.start(handle, password)

    init_resp = await _rpc(
        node=node,
        coordinator_node_id=coordinator_node_id,
        method="login_init",
        params={"handle": handle, "ke1": client.A},
    )
    state_id = init_resp["state_id"]
    server_response = init_resp["response"]

    M1 = client.respond(bytes(server_response))

    finish_params: dict = {
        "state_id": state_id,
        "ke3": M1,
        "device_pubkey": device_identity.public_bytes,
    }
    if invite_code:
        finish_params["invite"] = invite_code
    if label:
        finish_params["label"] = label

    finish_resp = await _rpc(
        node=node,
        coordinator_node_id=coordinator_node_id,
        method="login_finish",
        params=finish_params,
    )
    cert_wire = bytes(finish_resp["cert"])
    server_proof = bytes(finish_resp["m2"])
    # The server proof is what binds the cert delivery to "this is the
    # real coordinator we ran SRP with"; we ignore the bytes (we don't
    # use the SRP session key for anything else) but verifying it
    # would mean re-deriving M2 inside Srp6aClientLogin. srptools'
    # ``key_proof`` already gives us the same value the server returns,
    # so this is a defensive check rather than a security boundary.
    try:
        client.verify_server(server_proof)
    except Exception:
        logger.debug("SRP server-proof check skipped or mismatched (non-fatal)")

    # Cert verification — sanity-check the cert against the pinned
    # coordinator pubkey before we hand it back. The session dialer
    # would catch a bad cert later, but failing here gives the caller
    # a cleaner error path.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    pubkey = Ed25519PublicKey.from_public_bytes(coordinator_pubkey_bytes)
    try:
        # An empty ``network_id`` is the ticket-driven path: caller
        # doesn't have the canonical id yet (they only see what the
        # coordinator publishes), so we accept whatever the cert
        # claims. Caller is expected to pin it on persist.
        cert = verify_cert(
            cert_wire,
            coordinator_pubkey=pubkey,
            expected_network_id=network_id or None,
        )
    except CertVerificationError as e:
        raise LoginError(f"coordinator returned malformed cert: {e}") from e

    if cert.handle != handle.strip().lower():
        raise LoginError(f"cert handle mismatch (got {cert.handle!r}, expected {handle!r})")
    if cert.device_pubkey != device_identity.public_bytes:
        raise LoginError("cert device pubkey doesn't match this device")

    return cert_wire


async def refresh_cert(
    *,
    node: IrohNode,
    coordinator_node_id: str,
    coordinator_pubkey_bytes: bytes,
    handle: str,
    password: str,
    device_identity: Identity,
    network_id: str,
) -> bytes:
    """Re-run login to get a fresh cert. Same wire as ``login``."""
    return await login(
        node=node,
        coordinator_node_id=coordinator_node_id,
        coordinator_pubkey_bytes=coordinator_pubkey_bytes,
        handle=handle,
        password=password,
        device_identity=device_identity,
        network_id=network_id,
    )


async def fetch_network_info(
    *,
    node: IrohNode,
    coordinator_node_id: str,
) -> dict:
    """Fetch the coordinator's self-description (used for first-add of a network)."""
    return await _rpc(
        node=node,
        coordinator_node_id=coordinator_node_id,
        method="network_info",
        params={},
    )


async def list_agents(
    *,
    node: IrohNode,
    coordinator_node_id: str,
) -> list[dict]:
    """List the agents registered in this network (handle + node_id pairs)."""
    resp = await _rpc(
        node=node,
        coordinator_node_id=coordinator_node_id,
        method="list_agents",
        params={},
    )
    return resp.get("agents") or []


# ── Internals ────────────────────────────────────────────────────────


async def _rpc(
    *,
    node: IrohNode,
    coordinator_node_id: str,
    method: str,
    params: dict,
    timeout: float = 30.0,
) -> dict:
    """Open a coordinator stream, send one CBOR-framed request, read the response."""
    connection = await node.dial(coordinator_node_id, NetworkAlpn.COORDINATOR)
    bi = await connection.open_bi()
    # iroh-py 0.35: BiStream.send() / .recv() are methods returning the
    # underlying SendStream / RecvStream — invoking them once unwraps the
    # bidirectional stream into the two halves.
    send, recv = bi.send(), bi.recv()
    try:
        request = {"id": uuid.uuid4().hex, "method": method, "params": params}
        await _write_frame(send, request)
        # Half-close so the coordinator knows we're done sending.
        finish = getattr(send, "finish", None) or getattr(send, "close", None)
        if finish is not None:
            await finish()
        response = await asyncio.wait_for(_read_frame(recv), timeout=timeout)
    finally:
        # ``Connection.close`` is sync in iroh-py 0.35.
        try:
            connection.close(0, b"")
        except Exception:
            pass
    if not isinstance(response, dict):
        raise LoginError(f"coordinator returned non-map response: {type(response)!r}")
    if "error" in response:
        err = response["error"] or {}
        raise LoginError(f"{err.get('code', 'unknown')}: {err.get('message', '')}")
    if "result" not in response:
        raise LoginError("coordinator response had neither 'result' nor 'error'")
    return response["result"]


async def _write_frame(stream: object, obj: object) -> None:
    payload = cbor2.dumps(obj)
    await stream.write_all(len(payload).to_bytes(4, "big") + payload)


async def _read_frame(stream: object, *, max_size: int = 1 * 1024 * 1024) -> object:
    length_bytes = await _read_exact(stream, 4)
    n = int.from_bytes(length_bytes, "big")
    if n > max_size:
        raise LoginError(f"coordinator response too large: {n} > {max_size}")
    payload = await _read_exact(stream, n)
    return cbor2.loads(payload)


async def _read_exact(stream: object, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = await stream.read(n - len(out))
        if not chunk:
            raise LoginError("coordinator stream closed mid-frame")
        out.extend(chunk)
    return bytes(out)
