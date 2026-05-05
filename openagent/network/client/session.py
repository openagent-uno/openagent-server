"""Authenticated dialer: opens cert-prefixed Iroh streams to a target NodeId.

Built on top of ``IrohNode.dial`` — adds the cert framing the gateway
expects (4-byte length prefix + cert wire) and a small connection
pool so multiple concurrent HTTP requests share one Iroh connection.

Used by the CLI's ``GatewayClient``, the desktop-app sidecar, and any
agent acting as a federation client of another network.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import aiohttp

from openagent.network.auth.device_cert import (
    DeviceCert,
    verify_cert,
)
from openagent.network.iroh_node import IrohNode, NetworkAlpn

logger = logging.getLogger(__name__)


@dataclass
class NetworkBinding:
    """Everything the dialer needs to talk to one network."""

    network_id: str
    network_name: str
    coordinator_node_id: str
    coordinator_pubkey_bytes: bytes
    our_handle: str


class SessionDialer:
    """Holds a cert + opens authed gateway streams to a target agent.

    The cert is mutable — ``update_cert`` swaps in a freshly-refreshed
    one without dropping in-flight HTTP keep-alives. All actively-used
    connections continue to use whatever cert was current at the time
    they were opened; new connections pick up the new cert.
    """

    def __init__(
        self,
        *,
        node: IrohNode,
        binding: NetworkBinding,
        cert_wire: bytes,
    ) -> None:
        self._node = node
        self._binding = binding
        self._cert_wire = cert_wire
        self._cert_lock = asyncio.Lock()
        self._connections: dict[str, object] = {}  # node_id -> iroh.Connection
        self._connections_lock = asyncio.Lock()

    @property
    def binding(self) -> NetworkBinding:
        return self._binding

    @property
    def cert_wire(self) -> bytes:
        return self._cert_wire

    async def update_cert(self, cert_wire: bytes) -> None:
        async with self._cert_lock:
            self._cert_wire = cert_wire

    def parsed_cert(self) -> DeviceCert:
        """Return the parsed cert (verified against the pinned coordinator pubkey)."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        pubkey = Ed25519PublicKey.from_public_bytes(self._binding.coordinator_pubkey_bytes)
        return verify_cert(
            self._cert_wire,
            coordinator_pubkey=pubkey,
            expected_network_id=self._binding.network_id,
        )

    # ── Connection / stream pool ─────────────────────────────────

    async def open_gateway_stream(self, target_node_id: str) -> "GatewayStream":
        """Open one bi-stream to *target_node_id* with the cert prefix attached.

        Reuses the underlying Iroh connection if we already have one
        to the same NodeId; opens a fresh one otherwise.
        """
        connection = await self._get_or_open_connection(target_node_id)
        bi = await connection.open_bi()
        send, recv = bi.send(), bi.recv()
        # Send cert prefix immediately. The gateway's IrohSite reads
        # exactly these bytes off the wire before handing the stream
        # to aiohttp.
        async with self._cert_lock:
            cert = self._cert_wire
        await send.write_all(len(cert).to_bytes(4, "big") + cert)
        return GatewayStream(send=send, recv=recv, target_node_id=target_node_id)

    async def _get_or_open_connection(self, node_id: str) -> object:
        async with self._connections_lock:
            conn = self._connections.get(node_id)
            if conn is not None:
                # In a long-running CLI the same connection is reused
                # for many requests; drop it eagerly if Iroh marks it
                # closed. iroh-py 0.35 exposes ``Connection.closed()``
                # which is async and resolves with the close reason
                # once the connection is actually closed — we don't
                # want to await that here (it'd block forever on a
                # healthy connection). Skip the proactive check; let
                # the next ``open_bi`` fail-and-retry path handle it.
                pass
            if conn is None:
                conn = await self._node.dial(node_id, NetworkAlpn.GATEWAY)
                self._connections[node_id] = conn
            return conn

    async def close(self) -> None:
        async with self._connections_lock:
            for conn in self._connections.values():
                try:
                    # ``Connection.close`` is sync in iroh-py 0.35.
                    conn.close(0, b"")
                except Exception:
                    pass
            self._connections.clear()


@dataclass
class GatewayStream:
    """One open bi-stream after the cert prefix has been written.

    Pass the ``send`` / ``recv`` objects to ``IrohStreamReader`` /
    ``IrohStreamWriter`` to wrap them in asyncio streams that aiohttp's
    HTTP client can consume.
    """

    send: object
    recv: object
    target_node_id: str

    async def close(self) -> None:
        try:
            finish = getattr(self.send, "finish", None) or getattr(self.send, "close", None)
            if finish is not None:
                await finish()
        except Exception:
            pass


# ── aiohttp ClientSession bound to a SessionDialer ────────────────────────
#
# The CLI / app talk to the gateway via HTTP+WS over Iroh. aiohttp's
# ``ClientSession`` is hard to retarget without monkey-patching: it
# wants a TCP connector. The cleanest path is to expose an HTTP/1.1
# endpoint over a Unix-domain socket (or, on Windows, a named pipe)
# that proxies bytes onto the Iroh stream — a "loopback adapter".
#
# We implement this proxy here so the CLI's existing aiohttp code keeps
# working unchanged. ``serve_loopback`` listens on an OS-assigned local
# port; every accepted TCP connection is wrapped in a fresh GatewayStream
# from the dialer.


class LoopbackProxy:
    """Tiny TCP↔Iroh proxy so aiohttp can talk to the gateway via HTTP.

    ``start`` returns the (host, port) we bound on. Hand
    ``http://host:port`` to aiohttp's ``ClientSession`` — every request
    gets a fresh Iroh stream with the cert prefix already attached.
    """

    def __init__(self, *, dialer: SessionDialer, target_node_id: str) -> None:
        self._dialer = dialer
        self._target = target_node_id
        self._server: asyncio.AbstractServer | None = None
        self._sockname: tuple[str, int] | None = None

    @property
    def base_url(self) -> str:
        if self._sockname is None:
            raise RuntimeError("LoopbackProxy.start() not awaited")
        host, port = self._sockname
        return f"http://{host}:{port}"

    @property
    def ws_url(self) -> str:
        if self._sockname is None:
            raise RuntimeError("LoopbackProxy.start() not awaited")
        host, port = self._sockname
        return f"ws://{host}:{port}/ws"

    async def start(self) -> tuple[str, int]:
        self._server = await asyncio.start_server(
            self._handle_local, host="127.0.0.1", port=0,
        )
        sock = self._server.sockets[0].getsockname()
        # ``getsockname`` returns (host, port[, ...]) — keep the first 2.
        self._sockname = (sock[0], sock[1])
        return self._sockname

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_local(
        self,
        local_reader: asyncio.StreamReader,
        local_writer: asyncio.StreamWriter,
    ) -> None:
        """Pump bytes between the local TCP socket and a fresh Iroh stream."""
        try:
            stream = await self._dialer.open_gateway_stream(self._target)
        except Exception as e:  # noqa: BLE001
            logger.warning("loopback dial failed: %s", e)
            local_writer.close()
            return

        async def local_to_iroh() -> None:
            try:
                while True:
                    data = await local_reader.read(64 * 1024)
                    if not data:
                        break
                    await stream.send.write_all(data)
            except Exception as e:  # noqa: BLE001
                logger.debug("local->iroh pump ended: %s", e)
            finally:
                try:
                    finish = getattr(stream.send, "finish", None)
                    if finish:
                        await finish()
                except Exception:
                    pass

        async def iroh_to_local() -> None:
            try:
                while True:
                    chunk = await stream.recv.read(64 * 1024)
                    if not chunk:
                        break
                    local_writer.write(chunk)
                    await local_writer.drain()
            except Exception as e:  # noqa: BLE001
                logger.debug("iroh->local pump ended: %s", e)
            finally:
                try:
                    local_writer.close()
                except Exception:
                    pass

        await asyncio.gather(local_to_iroh(), iroh_to_local(), return_exceptions=True)
