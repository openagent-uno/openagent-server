"""Peer-network management: an agent acting as a CLIENT of other networks.

The local agent's *home* network sits in the singleton ``network`` row.
``peer_networks`` is the registry of OTHER networks this agent joins
to talk to peer agents (federation). Each row pairs a network_id with
the coordinator's pinned NodeId/pubkey and the handle this agent uses
to authenticate there.

REST surface (``/api/peers``):

    GET    /api/peers                    -> list rows + cached cert status
    POST   /api/peers                    -> add a peer (interactive: handle + invite + password)
    DELETE /api/peers/{network_id}       -> drop a peer membership
    POST   /api/peers/{network_id}/refresh -> force a cert refresh
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import aiosqlite
from aiohttp import web

from openagent.core.logging import elog
from openagent.memory.db import MemoryDB
from openagent.network.auth.device_cert import (
    CertVerificationError,
    verify_cert,
)
from openagent.network.client.login import (
    LoginError,
    fetch_network_info,
    list_agents as coord_list_agents,
    login as coord_login,
    refresh_cert as coord_refresh_cert,
)
from openagent.network.client.session import NetworkBinding, SessionDialer
from openagent.network.identity import Identity, load_or_create_identity
from openagent.network.iroh_node import IrohNode

logger = logging.getLogger(__name__)


@dataclass
class PeerNetworkRow:
    network_id: str
    name: str
    coordinator_node_id: str
    coordinator_pubkey: bytes
    our_handle: str
    status: str
    added_at: float
    last_seen: float | None


class PeerStore:
    """Async helpers around the ``peer_networks`` + ``device_certs`` tables."""

    def __init__(self, db: MemoryDB) -> None:
        self._db = db

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self._db._conn is None:
            raise RuntimeError("MemoryDB.connect() must be called before PeerStore use")
        return self._db._conn

    async def list_peers(self) -> list[PeerNetworkRow]:
        cur = await self._conn.execute(
            "SELECT network_id, name, coordinator_node_id, coordinator_pubkey, our_handle, "
            "status, added_at, last_seen FROM peer_networks ORDER BY added_at",
        )
        return [PeerNetworkRow(**dict(row)) for row in await cur.fetchall()]

    async def get_peer(self, network_id: str) -> PeerNetworkRow | None:
        cur = await self._conn.execute(
            "SELECT network_id, name, coordinator_node_id, coordinator_pubkey, our_handle, "
            "status, added_at, last_seen FROM peer_networks WHERE network_id=?",
            (network_id,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return PeerNetworkRow(**dict(row))

    async def add_peer(
        self,
        *,
        network_id: str,
        name: str,
        coordinator_node_id: str,
        coordinator_pubkey: bytes,
        our_handle: str,
    ) -> None:
        await self._conn.execute(
            "INSERT INTO peer_networks (network_id, name, coordinator_node_id, "
            "coordinator_pubkey, our_handle, status, added_at) "
            "VALUES (?, ?, ?, ?, ?, 'active', ?) "
            "ON CONFLICT(network_id) DO UPDATE SET name=excluded.name, "
            "coordinator_node_id=excluded.coordinator_node_id, "
            "coordinator_pubkey=excluded.coordinator_pubkey, "
            "our_handle=excluded.our_handle, status='active'",
            (network_id, name, coordinator_node_id, coordinator_pubkey, our_handle, time.time()),
        )
        await self._conn.commit()

    async def remove_peer(self, network_id: str) -> bool:
        cur = await self._conn.execute(
            "DELETE FROM peer_networks WHERE network_id=?",
            (network_id,),
        )
        await self._conn.execute(
            "DELETE FROM device_certs WHERE network_id=?",
            (network_id,),
        )
        await self._conn.commit()
        return cur.rowcount > 0

    async def store_cert(self, *, network_id: str, handle: str, cert_wire: bytes, expires_at: float) -> None:
        await self._conn.execute(
            "INSERT INTO device_certs (network_id, handle, cert, expires_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(network_id, handle) DO UPDATE SET cert=excluded.cert, "
            "expires_at=excluded.expires_at",
            (network_id, handle, cert_wire, expires_at),
        )
        await self._conn.commit()

    async def get_cert(self, *, network_id: str, handle: str) -> tuple[bytes, float] | None:
        cur = await self._conn.execute(
            "SELECT cert, expires_at FROM device_certs WHERE network_id=? AND handle=?",
            (network_id, handle),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return bytes(row[0]), float(row[1])


# ── Builders ────────────────────────────────────────────────────────────


async def make_dialer_for_peer(
    *,
    db: MemoryDB,
    peer: PeerNetworkRow,
    node: IrohNode,
    refresh_password: str | None = None,
    device_identity: Identity | None = None,
) -> SessionDialer:
    """Build a SessionDialer for an existing peer-network membership.

    Looks up the cached cert; if it's expired or missing and a password
    was supplied, attempts a fresh login. If we don't have the means
    to refresh (no password, no device identity), raises ``LoginError``.
    """
    store = PeerStore(db)
    cached = await store.get_cert(network_id=peer.network_id, handle=peer.our_handle)
    cert_wire: bytes | None = cached[0] if cached else None
    expires_at = cached[1] if cached else 0.0

    if cert_wire is None or expires_at <= time.time():
        if refresh_password is None or device_identity is None:
            raise LoginError(
                f"no valid cert for {peer.our_handle}@{peer.name} and no credentials to refresh",
            )
        cert_wire = await coord_refresh_cert(
            node=node,
            coordinator_node_id=peer.coordinator_node_id,
            coordinator_pubkey_bytes=peer.coordinator_pubkey,
            handle=peer.our_handle,
            password=refresh_password,
            device_identity=device_identity,
            network_id=peer.network_id,
        )
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            pubkey = Ed25519PublicKey.from_public_bytes(peer.coordinator_pubkey)
            cert = verify_cert(
                cert_wire,
                coordinator_pubkey=pubkey,
                expected_network_id=peer.network_id,
            )
        except CertVerificationError as e:
            raise LoginError(f"refreshed cert failed verification: {e}") from e
        await store.store_cert(
            network_id=peer.network_id,
            handle=peer.our_handle,
            cert_wire=cert_wire,
            expires_at=cert.expires_at,
        )

    binding = NetworkBinding(
        network_id=peer.network_id,
        network_name=peer.name,
        coordinator_node_id=peer.coordinator_node_id,
        coordinator_pubkey_bytes=peer.coordinator_pubkey,
        our_handle=peer.our_handle,
    )
    return SessionDialer(node=node, binding=binding, cert_wire=cert_wire)


# ── REST handlers ──────────────────────────────────────────────────────


async def handle_list(request: web.Request) -> web.Response:
    """GET /api/peers — list peer networks this agent has joined."""
    gw = request.app["gateway"]
    db = gw.agent._db
    if db is None:
        return web.json_response({"error": "no DB attached"}, status=503)
    store = PeerStore(db)
    rows = await store.list_peers()
    out = []
    for r in rows:
        cert = await store.get_cert(network_id=r.network_id, handle=r.our_handle)
        out.append({
            "network_id": r.network_id,
            "name": r.name,
            "coordinator_node_id": r.coordinator_node_id,
            "our_handle": r.our_handle,
            "status": r.status,
            "added_at": r.added_at,
            "last_seen": r.last_seen,
            "cert_expires_at": cert[1] if cert else None,
        })
    return web.json_response({"peers": out})


async def handle_create(request: web.Request) -> web.Response:
    """POST /api/peers — add a new peer-network membership.

    Body: ``{coordinator_node_id, handle, password, invite?, label?}``.
    Performs a fresh login, pins the coordinator pubkey, persists the
    membership + cert. The handler is synchronous-ish (login is one
    Iroh round-trip) so the UI gets a single OK/fail response.
    """
    gw = request.app["gateway"]
    db = gw.agent._db
    if db is None:
        return web.json_response({"error": "no DB attached"}, status=503)
    body = await request.json()
    coordinator_node_id = body.get("coordinator_node_id")
    handle = (body.get("handle") or "").strip().lower()
    password = body.get("password") or ""
    invite = body.get("invite")
    if not (coordinator_node_id and handle and password):
        return web.json_response(
            {"error": "coordinator_node_id, handle, password are required"},
            status=400,
        )

    state = getattr(gw, "_network_state", None)
    if state is None:
        return web.json_response({"error": "gateway has no network state"}, status=500)

    info = await fetch_network_info(node=state.iroh_node, coordinator_node_id=coordinator_node_id)
    network_id = info["network_id"]
    network_name = info.get("name") or network_id
    coord_pubkey_bytes = bytes(coordinator_node_id_to_pubkey_bytes(coordinator_node_id))

    user_identity_path_value = body.get("device_identity_path") or None
    if user_identity_path_value:
        device_identity = load_or_create_identity(user_identity_path_value)
    else:
        # Federation: peer agent uses its own agent identity for inbound
        # auth at the peer network. Fine because the agent IS a "device"
        # in that network's eyes.
        device_identity = state.identity

    try:
        cert_wire = await coord_login(
            node=state.iroh_node,
            coordinator_node_id=coordinator_node_id,
            coordinator_pubkey_bytes=coord_pubkey_bytes,
            handle=handle,
            password=password,
            device_identity=device_identity,
            network_id=network_id,
            invite_code=invite,
            label=body.get("label"),
        )
    except LoginError as e:
        elog("peers.add_failed", level="warning", error=str(e), handle=handle, network=network_name)
        return web.json_response({"error": str(e)}, status=400)

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        pubkey = Ed25519PublicKey.from_public_bytes(coord_pubkey_bytes)
        cert = verify_cert(
            cert_wire, coordinator_pubkey=pubkey, expected_network_id=network_id,
        )
    except CertVerificationError as e:
        return web.json_response({"error": f"cert verification failed: {e}"}, status=502)

    store = PeerStore(db)
    await store.add_peer(
        network_id=network_id,
        name=network_name,
        coordinator_node_id=coordinator_node_id,
        coordinator_pubkey=coord_pubkey_bytes,
        our_handle=handle,
    )
    await store.store_cert(
        network_id=network_id, handle=handle,
        cert_wire=cert_wire, expires_at=cert.expires_at,
    )
    elog("peers.added", network=network_name, handle=handle)
    return web.json_response({
        "ok": True, "network_id": network_id, "name": network_name,
        "handle": handle, "expires_at": cert.expires_at,
    })


async def handle_delete(request: web.Request) -> web.Response:
    """DELETE /api/peers/{network_id} — drop a peer-network membership."""
    gw = request.app["gateway"]
    db = gw.agent._db
    if db is None:
        return web.json_response({"error": "no DB attached"}, status=503)
    network_id = request.match_info["network_id"]
    store = PeerStore(db)
    ok = await store.remove_peer(network_id)
    return web.json_response({"ok": ok})


async def handle_list_agents(request: web.Request) -> web.Response:
    """GET /api/peers/{network_id}/agents — list agents in a peer network."""
    gw = request.app["gateway"]
    state = getattr(gw, "_network_state", None)
    db = gw.agent._db
    if db is None or state is None:
        return web.json_response({"error": "gateway misconfigured"}, status=503)
    store = PeerStore(db)
    network_id = request.match_info["network_id"]
    peer = await store.get_peer(network_id)
    if peer is None:
        return web.json_response({"error": "unknown peer"}, status=404)
    agents = await coord_list_agents(
        node=state.iroh_node,
        coordinator_node_id=peer.coordinator_node_id,
    )
    return web.json_response({"agents": agents})


# ── Helpers ─────────────────────────────────────────────────────────────


def coordinator_node_id_to_pubkey_bytes(node_id: str) -> bytes:
    """Decode an Iroh NodeId string into raw 32 bytes.

    Iroh NodeIds are an encoded form of the Ed25519 public key bytes.
    iroh-py 0.35 exposes ``PublicKey.from_string(s).as_bytes()``; we
    defer the import so this module doesn't pull iroh into agents that
    never join a peer network.
    """
    import iroh  # noqa: WPS433

    pk = iroh.PublicKey.from_string(node_id)
    raw = pk.to_bytes()
    if len(raw) != 32:
        raise ValueError(f"NodeId pubkey is not 32 bytes: {len(raw)}")
    return bytes(raw)
