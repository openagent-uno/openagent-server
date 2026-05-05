"""In-process iroh-shaped transport for co-located gateway clients.

The gateway's iroh transport (``IrohSite``) drives aiohttp from a stream
of QUIC bi-streams fed in by ``ProtocolHandler.accept`` callbacks. For
in-process clients (the in-process bridges built in
``openagent.core.server._build_bridges``) we don't want the round-trip
through iroh's local QUIC stack — and iroh-py 0.35 can't dial its own
NodeId anyway.

Instead we hand ``IrohSite._handle_stream`` a synthetic connection
backed by ``asyncio.Queue`` pairs. To the site it walks-and-quacks
exactly like an iroh ``Connection``: ``accept_bi()`` blocks until the
client opens a stream, ``open_bi()`` returns a bi-stream pair, each
half exposing ``write_all`` / ``read`` / ``finish`` the same way.

The client side is wired to a ``LoopbackProxy`` via
``InProcDialer.open_gateway_stream`` — drop-in replacement for the
iroh-backed ``SessionDialer`` from ``client.session``.

Cert handling stays identical: each new stream opens with a
length-prefixed cert wire that the IrohSite reads before injecting it
into the auth contextvar. So the gateway's auth middleware path is
unchanged — it never knows the bytes came in over an in-process pipe.
"""

from __future__ import annotations

import asyncio
import logging

from openagent.network.client.session import GatewayStream

logger = logging.getLogger(__name__)


_EOF = object()  # sentinel pushed onto a queue to signal write-half closed


class _InProcSendStream:
    """Mimics iroh-py's ``SendStream`` — async ``write_all`` / ``finish``."""

    def __init__(self, queue: asyncio.Queue) -> None:
        self._q = queue
        self._closed = False

    async def write_all(self, data: bytes) -> None:
        if self._closed:
            raise RuntimeError("send stream closed")
        if not data:
            return
        await self._q.put(data)

    async def finish(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._q.put(_EOF)

    async def close(self) -> None:
        await self.finish()


class _InProcRecvStream:
    """Mimics iroh-py's ``RecvStream`` — async ``read(n)`` honoring EOF."""

    def __init__(self, queue: asyncio.Queue) -> None:
        self._q = queue
        self._buf = bytearray()
        self._eof = False

    async def read(self, n: int = -1) -> bytes:
        # Honor any leftover from the previous chunk (read may be
        # smaller than the chunk size).
        if self._buf:
            if n < 0 or n >= len(self._buf):
                out = bytes(self._buf)
                self._buf.clear()
                return out
            out = bytes(self._buf[:n])
            del self._buf[:n]
            return out

        if self._eof:
            return b""

        item = await self._q.get()
        if item is _EOF:
            self._eof = True
            return b""

        if n < 0 or n >= len(item):
            return item
        self._buf.extend(item[n:])
        return item[:n]


class _InProcBiStream:
    """Mimics iroh-py's ``BiStream`` (``send()`` / ``recv()`` accessors)."""

    def __init__(self, send_q: asyncio.Queue, recv_q: asyncio.Queue) -> None:
        self._send = _InProcSendStream(send_q)
        self._recv = _InProcRecvStream(recv_q)

    def send(self) -> _InProcSendStream:
        return self._send

    def recv(self) -> _InProcRecvStream:
        return self._recv


class InProcConnection:
    """In-process equivalent of iroh-py's ``Connection``.

    One object plays both roles. The gateway side calls ``accept_bi``;
    the bridge side calls ``open_bi``. Each ``open_bi`` enqueues a
    matching server-side ``BiStream`` for ``accept_bi`` to yield, and
    returns the client-side ``BiStream`` directly.
    """

    def __init__(self, *, peer_node_id: str = "inproc:bridge") -> None:
        self._peer_node_id = peer_node_id
        self._stream_queue: asyncio.Queue = asyncio.Queue()
        self._closed = False
        # Track every queue we've ever handed out so ``close`` can
        # signal EOF to in-flight readers. Without this, a pending
        # ``RecvStream.read`` blocks forever and any caller waiting
        # on stream cleanup (e.g. ``LoopbackProxy.stop``'s
        # ``server.wait_closed``) hangs.
        self._all_queues: list[asyncio.Queue] = []

    def remote_node_id(self) -> str:
        # IrohSite.handle_stream calls this once at connection accept
        # time for logging. Return a stable identifier so the logs are
        # readable.
        return self._peer_node_id

    async def accept_bi(self) -> _InProcBiStream | None:
        if self._closed:
            return None
        item = await self._stream_queue.get()
        if item is None:
            return None
        return item

    async def open_bi(self) -> _InProcBiStream:
        if self._closed:
            raise RuntimeError("inproc connection closed")
        c2s: asyncio.Queue = asyncio.Queue()
        s2c: asyncio.Queue = asyncio.Queue()
        self._all_queues.append(c2s)
        self._all_queues.append(s2c)
        client_bi = _InProcBiStream(send_q=c2s, recv_q=s2c)
        server_bi = _InProcBiStream(send_q=s2c, recv_q=c2s)
        await self._stream_queue.put(server_bi)
        return client_bi

    def close(self, *_args, **_kwargs) -> None:
        if self._closed:
            return
        self._closed = True
        # Wake any pending accept_bi so the IrohSite handler exits its loop.
        try:
            self._stream_queue.put_nowait(None)
        except Exception:  # noqa: BLE001
            pass
        # Push EOF onto every stream we ever opened so pending reads
        # unblock and the byte-pump tasks (LoopbackProxy._handle_local,
        # IrohSite._handle_one_stream) can finish.
        for q in self._all_queues:
            try:
                q.put_nowait(_EOF)
            except Exception:  # noqa: BLE001
                pass


class InProcDialer:
    """``LoopbackProxy``-compatible dialer that opens streams over an InProcConnection.

    The cert prefix is written here exactly like ``SessionDialer`` does
    over a real iroh stream, so the gateway's IrohSite path doesn't need
    to know it isn't talking to a remote peer.
    """

    def __init__(self, *, connection: InProcConnection, cert_wire: bytes) -> None:
        self._conn = connection
        self._cert_wire = cert_wire
        self._cert_lock = asyncio.Lock()

    async def update_cert(self, cert_wire: bytes) -> None:
        async with self._cert_lock:
            self._cert_wire = cert_wire

    @property
    def cert_wire(self) -> bytes:
        return self._cert_wire

    async def open_gateway_stream(self, _target_node_id: str) -> GatewayStream:
        bi = await self._conn.open_bi()
        send = bi.send()
        recv = bi.recv()
        async with self._cert_lock:
            cert = self._cert_wire
        await send.write_all(len(cert).to_bytes(4, "big") + cert)
        return GatewayStream(send=send, recv=recv, target_node_id="inproc")

    async def close(self) -> None:
        # Connection lifecycle is owned by ``BridgeSession``; this
        # method only exists so callers can ``await dialer.close()``
        # symmetrically with ``SessionDialer``.
        return None
