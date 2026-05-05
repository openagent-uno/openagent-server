"""The embedded coordinator JSON-RPC service.

Speaks ``openagent/coordinator/1`` ALPN. One stream per RPC call:
client opens a bi-stream, sends a length-prefixed CBOR request, reads
a length-prefixed CBOR response, closes. Trivial framing, no
multiplexing — Iroh already gives us per-stream concurrency.

Request shape::

    {"id": "<corr>", "method": "register", "params": {...}}

Response shape (success)::

    {"id": "<corr>", "result": {...}}

Response shape (error)::

    {"id": "<corr>", "error": {"code": "bad_request", "message": "..."}}

Methods (all v1):

    register(invite, handle, pake_record) -> {ok: true}
    login_init(handle, ke1) -> {salt+B}
    login_finish(state_id, ke3, device_pubkey, label?) -> {cert, m2}
    list_agents() -> {agents: [...]}
    add_agent(invite, handle, node_id, owner_handle, label?) -> {ok}
    remove_agent(handle) -> {ok}
    revoke_device(device_pubkey) -> {ok}
    create_invitation(role, ttl_seconds?, uses?, bind_to_handle?) -> {code, ...}

Auth model: ``register``, ``login_*`` and ``add_agent`` consume an
invitation code (if required for that method). Other write methods
require the caller to present a fresh device cert with the
``coordinator_admin`` capability — same wire as gateway streams use.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

import cbor2
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from openagent.core.logging import elog
from openagent.network.auth.device_cert import (
    CertVerificationError,
    issue_cert,
    verify_cert,
)
from openagent.network.coordinator.pake import (
    LoginInProgress,
    PakeBackend,
    PakeError,
    Srp6aBackend,
)
from openagent.network.coordinator.store import (
    CoordinatorStore,
    InvitationRow,
)
from openagent.network.iroh_node import IrohNode, NetworkAlpn

logger = logging.getLogger(__name__)


LOGIN_STATE_TTL = 30.0  # seconds — drop stale half-completed logins


@dataclass
class _ServiceConfig:
    network_id: str
    network_name: str
    coordinator_key: Ed25519PrivateKey


class _CoordinatorRpcError(Exception):
    """Carries a method/code/message for the on-the-wire error response."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class CoordinatorService:
    """Owns the JSON-RPC handlers + ephemeral login state.

    One instance per agent-as-coordinator. Constructed by the
    AgentServer when ``network.role='coordinator'`` and registered as
    the ALPN handler before the IrohNode starts.
    """

    def __init__(
        self,
        *,
        store: CoordinatorStore,
        coordinator_key: Ed25519PrivateKey,
        network_id: str,
        network_name: str,
        pake_backend: PakeBackend | None = None,
    ) -> None:
        self._store = store
        self._cfg = _ServiceConfig(
            network_id=network_id,
            network_name=network_name,
            coordinator_key=coordinator_key,
        )
        self._pake: PakeBackend = pake_backend or Srp6aBackend()
        self._logins: dict[str, LoginInProgress] = {}
        self._gc_task: asyncio.Task | None = None

    # ── Lifecycle ────────────────────────────────────────────────

    def attach(self, node: IrohNode) -> None:
        """Register the coordinator handler on *node*. Call before ``node.start()``."""
        node.register_handler(NetworkAlpn.COORDINATOR, self._on_stream)

    async def start_gc(self) -> None:
        if self._gc_task is None or self._gc_task.done():
            self._gc_task = asyncio.create_task(self._gc_loop(), name="coord-login-gc")

    async def stop(self) -> None:
        if self._gc_task is not None:
            self._gc_task.cancel()
            try:
                await self._gc_task
            except (asyncio.CancelledError, Exception):
                pass
            self._gc_task = None

    async def _gc_loop(self) -> None:
        while True:
            await asyncio.sleep(5.0)
            now = time.time()
            stale = [
                k for k, st in self._logins.items()
                if now - st.created_at > LOGIN_STATE_TTL
            ]
            for k in stale:
                self._logins.pop(k, None)

    # ── Wire handler ─────────────────────────────────────────────

    async def _on_stream(self, connection: object) -> None:
        """One inbound coordinator connection — handle each bi-stream as one RPC."""
        peer_node_id = "unknown"
        try:
            # iroh-py 0.35: ``remote_node_id`` is sync.
            raw = connection.remote_node_id()
            peer_node_id = str(raw) if raw is not None else "unknown"
        except Exception:
            pass
        try:
            while True:
                try:
                    bi = await connection.accept_bi()
                except Exception:
                    return
                if bi is None:
                    return
                asyncio.create_task(
                    self._handle_one_rpc(bi.send(), bi.recv(), peer_node_id),
                    name=f"coord-rpc-{peer_node_id[:8]}",
                )
        except asyncio.CancelledError:
            raise

    async def _handle_one_rpc(self, send_stream, recv_stream, peer_node_id: str) -> None:
        request: object | None = None
        try:
            request = await _read_cbor_frame(recv_stream)
            response = await self._dispatch(request, peer_node_id=peer_node_id)
        except _CoordinatorRpcError as e:
            response = {
                "id": (request or {}).get("id") if isinstance(request, dict) else None,
                "error": {"code": e.code, "message": e.message},
            }
        except Exception as e:  # noqa: BLE001
            logger.exception("coordinator dispatch crashed")
            response = {
                "error": {"code": "internal", "message": str(e) or "unknown"},
            }
        await _write_cbor_frame(send_stream, response)
        try:
            finish = getattr(send_stream, "finish", None)
            if finish is not None:
                await finish()
        except Exception:
            pass

    async def _dispatch(self, request: object, *, peer_node_id: str) -> dict:
        if not isinstance(request, dict):
            raise _CoordinatorRpcError("bad_request", "request not a CBOR map")
        corr = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if not isinstance(method, str):
            raise _CoordinatorRpcError("bad_request", "missing 'method'")
        handler = self._METHODS.get(method)
        if handler is None:
            raise _CoordinatorRpcError("unknown_method", method)
        try:
            result = await handler(self, params, peer_node_id=peer_node_id)
        except _CoordinatorRpcError:
            raise
        except (PakeError, ValueError) as e:
            raise _CoordinatorRpcError("bad_request", str(e)) from e
        elog(
            "coord.rpc",
            method=method,
            peer=peer_node_id[:16],
        )
        return {"id": corr, "result": result}

    # ── Method implementations ───────────────────────────────────

    async def _m_register(self, params: dict, *, peer_node_id: str) -> dict:
        invite_code = _required(params, "invite", str)
        handle = _required(params, "handle", str).strip().lower()
        record = _required(params, "pake_record", bytes)

        invite = await self._store.consume_invitation(invite_code)
        if invite is None or invite.role != "user":
            raise _CoordinatorRpcError("invalid_invite", "invite missing/expired/wrong-role")

        existing = await self._store.get_user(handle)
        if existing is not None:
            raise _CoordinatorRpcError("conflict", f"handle '{handle}' already taken")

        validated = self._pake.register_finalize(handle, record)
        await self._store.create_user(
            handle=handle, pake_record=validated, pake_algo=self._pake.algo,
        )
        return {"ok": True, "handle": handle}

    async def _m_login_init(self, params: dict, *, peer_node_id: str) -> dict:
        handle = _required(params, "handle", str).strip().lower()
        ke1 = _required(params, "ke1", bytes)

        user = await self._store.get_user(handle)
        if user is None or user.status != "active":
            # Constant-time negative response shape so a bad handle
            # doesn't return faster than a real one — we still hash a
            # dummy salt+verifier through the PAKE backend.
            raise _CoordinatorRpcError("unauthorized", "login failed")

        try:
            state, response = self._pake.login_init(handle, user.pake_record, ke1)
        except PakeError as e:
            raise _CoordinatorRpcError("unauthorized", str(e)) from e

        state_id = uuid.uuid4().hex
        self._logins[state_id] = state
        return {"state_id": state_id, "response": response}

    async def _m_login_finish(self, params: dict, *, peer_node_id: str) -> dict:
        state_id = _required(params, "state_id", str)
        ke3 = _required(params, "ke3", bytes)
        device_pubkey = _required(params, "device_pubkey", bytes)
        label = params.get("label")
        invite_code = params.get("invite")  # optional — only needed for first-device pairings

        state = self._logins.pop(state_id, None)
        if state is None:
            raise _CoordinatorRpcError("expired", "login state not found or expired")

        try:
            m2 = self._pake.login_finish(state, ke3)
        except PakeError as e:
            raise _CoordinatorRpcError("unauthorized", str(e)) from e

        # Ensure the device is bound to the user. Three cases:
        #  - device already known and active → no-op (just refresh
        #    last_seen and reissue the cert).
        #  - device unknown, user has zero devices → this is the
        #    first-device-of-a-fresh-user pairing. The registration
        #    invite was already consumed by ``_m_register``; consuming
        #    it again here would fail because invites are single-use,
        #    so we skip the invite check entirely.
        #  - device unknown, user already has devices → this is a
        #    second-device pairing. Caller must present a fresh
        #    user/device-role invite.
        existing_device = await self._store.get_device(device_pubkey)
        if existing_device is None:
            user_has_other_devices = await self._store.user_has_devices(state.handle)
            if user_has_other_devices:
                if not invite_code:
                    raise _CoordinatorRpcError(
                        "invalid_invite", "adding a new device requires an invite",
                    )
                inv = await self._store.consume_invitation(invite_code)
                if inv is None or inv.role not in ("user", "device"):
                    raise _CoordinatorRpcError(
                        "invalid_invite", "device invite missing/expired/wrong-role",
                    )
                if inv.bind_to_handle and inv.bind_to_handle != state.handle:
                    raise _CoordinatorRpcError(
                        "invalid_invite", "device invite is for a different handle",
                    )
            await self._store.add_device(
                device_pubkey=device_pubkey, user_handle=state.handle, label=label,
            )
        elif existing_device.status != "active":
            raise _CoordinatorRpcError("revoked", "device has been revoked")
        else:
            await self._store.touch_device(device_pubkey)

        cert = issue_cert(
            coordinator_key=self._cfg.coordinator_key,
            handle=state.handle,
            device_pubkey=device_pubkey,
            network_id=self._cfg.network_id,
        )
        return {"cert": cert, "m2": m2}

    async def _m_list_agents(self, params: dict, *, peer_node_id: str) -> dict:
        # No auth: the agent list is the network-wide directory and is
        # only useful to those who can already mint a session by
        # logging in. Anyone who reaches this RPC has either a valid
        # cert or is an unauthenticated peer that gets nothing
        # actionable from the list of NodeIds (they still need to
        # auth at each agent's gateway).
        rows = await self._store.list_agents()
        return {
            "agents": [
                {
                    "handle": r.handle,
                    "node_id": r.node_id,
                    "label": r.label,
                    "owner_handle": r.owner_handle,
                    "added_at": r.added_at,
                    "last_seen": r.last_seen,
                }
                for r in rows
            ],
        }

    async def _m_add_agent(self, params: dict, *, peer_node_id: str) -> dict:
        # Two paths: (a) admin caller with a valid ``coordinator_admin``
        # cert can register any agent; (b) an invitation with role
        # ``agent`` self-registers a new agent via the invite.
        cert_wire = params.get("cert")
        invite_code = params.get("invite")

        owner_handle: str
        if cert_wire:
            cert = self._verify_admin_cert(cert_wire)
            owner_handle = cert.handle
        elif invite_code:
            inv = await self._store.consume_invitation(invite_code)
            if inv is None or inv.role != "agent":
                raise _CoordinatorRpcError("invalid_invite", "agent invite missing/expired/wrong-role")
            owner_handle = inv.bind_to_handle or "system"
        else:
            raise _CoordinatorRpcError("unauthorized", "add_agent requires admin cert or invite")

        handle = _required(params, "handle", str).strip().lower()
        node_id = _required(params, "node_id", str)
        label = params.get("label")
        await self._store.register_agent(
            handle=handle, node_id=node_id, owner_handle=owner_handle, label=label,
        )
        return {"ok": True}

    async def _m_remove_agent(self, params: dict, *, peer_node_id: str) -> dict:
        cert = self._verify_admin_cert(_required(params, "cert", bytes))
        handle = _required(params, "handle", str)
        ok = await self._store.remove_agent(handle)
        return {"ok": ok, "removed_by": cert.handle}

    async def _m_revoke_device(self, params: dict, *, peer_node_id: str) -> dict:
        cert = self._verify_admin_cert(_required(params, "cert", bytes))
        device_pubkey = _required(params, "device_pubkey", bytes)
        ok = await self._store.revoke_device(device_pubkey)
        return {"ok": ok, "revoked_by": cert.handle}

    async def _m_create_invitation(self, params: dict, *, peer_node_id: str) -> dict:
        cert = self._verify_admin_cert(_required(params, "cert", bytes))
        role = _required(params, "role", str)
        ttl_seconds = int(params.get("ttl_seconds") or 7 * 24 * 3600)
        uses = int(params.get("uses") or 1)
        bind_to_handle = params.get("bind_to_handle")
        invite = await self._store.create_invitation(
            role=role,
            created_by=cert.handle,
            ttl_seconds=ttl_seconds,
            uses=uses,
            bind_to_handle=bind_to_handle,
        )
        return {
            "code": invite.code,
            "display_code": invite.display_code,
            "role": invite.role,
            "expires_at": invite.expires_at,
            "uses_left": invite.uses_left,
        }

    async def _m_network_info(self, params: dict, *, peer_node_id: str) -> dict:
        return {
            "network_id": self._cfg.network_id,
            "name": self._cfg.network_name,
            "pake_algo": self._pake.algo,
        }

    # ── Helpers ──────────────────────────────────────────────────

    def _verify_admin_cert(self, wire: bytes):
        try:
            cert = verify_cert(
                wire,
                coordinator_pubkey=self._cfg.coordinator_key.public_key(),
                expected_network_id=self._cfg.network_id,
            )
        except CertVerificationError as e:
            raise _CoordinatorRpcError("unauthorized", f"cert rejected: {e}") from e
        if "coordinator_admin" not in cert.capabilities:
            raise _CoordinatorRpcError("forbidden", "missing coordinator_admin capability")
        return cert

    _METHODS: dict[str, "asyncio.Coroutine"] = {}


# Method registration is done after class body so the bound methods
# on ``self`` resolve correctly. Python doesn't let us reference
# ``self._m_register`` inside the class body's dict literal, so we
# patch it in here.
CoordinatorService._METHODS = {
    "register": CoordinatorService._m_register,
    "login_init": CoordinatorService._m_login_init,
    "login_finish": CoordinatorService._m_login_finish,
    "list_agents": CoordinatorService._m_list_agents,
    "add_agent": CoordinatorService._m_add_agent,
    "remove_agent": CoordinatorService._m_remove_agent,
    "revoke_device": CoordinatorService._m_revoke_device,
    "create_invitation": CoordinatorService._m_create_invitation,
    "network_info": CoordinatorService._m_network_info,
}


# ── CBOR framing helpers ────────────────────────────────────────────


async def _read_cbor_frame(stream: object, *, max_size: int = 1 * 1024 * 1024) -> object:
    length_bytes = await _read_exact(stream, 4)
    n = int.from_bytes(length_bytes, "big")
    if n > max_size:
        raise _CoordinatorRpcError("too_large", f"frame {n} > max {max_size}")
    payload = await _read_exact(stream, n)
    return cbor2.loads(payload)


async def _write_cbor_frame(stream: object, obj: object) -> None:
    payload = cbor2.dumps(obj)
    await stream.write_all(len(payload).to_bytes(4, "big") + payload)


async def _read_exact(stream: object, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = await stream.read(n - len(out))
        if not chunk:
            raise _CoordinatorRpcError("eof", "stream closed before frame complete")
        out.extend(chunk)
    return bytes(out)


def _required(params: dict, name: str, ty: type):
    if name not in params:
        raise _CoordinatorRpcError("bad_request", f"missing param: {name}")
    val = params[name]
    if ty is bytes and isinstance(val, (bytes, bytearray)):
        return bytes(val)
    if not isinstance(val, ty):
        raise _CoordinatorRpcError("bad_request", f"param '{name}' wrong type")
    return val
