"""aiohttp ``BaseSite`` subclass that drives an ``IrohNode`` instead of TCP.

Handing the existing ``web.Application`` to ``IrohSite(app, node)`` is
the *only* change needed to put the gateway behind Iroh. Every route,
middleware, WebSocket handler etc. is reused verbatim.

Wire format on each accepted bi-stream:

  - First 4 bytes (big-endian): ``cert_len``.
  - Next ``cert_len`` bytes: the device cert wire (CBOR payload + sig).
  - Remaining bytes: a regular HTTP/1.1 request (or a request that
    upgrades to WebSocket — same as a TCP request would carry).

We strip the cert prefix here and stash it on ``request['device_cert_wire']``
via a contextvar so the auth middleware can verify it without
re-reading from the stream. Stream framing stays HTTP/1.1 unchanged
after that point.

QUIC streams give us per-request multiplexing for free — one QUIC
connection can carry many concurrent streams, each = one HTTP request
or one upgraded WebSocket. So we don't need our own request multiplexer.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import Any

from aiohttp import web
from aiohttp.web_protocol import RequestHandler

from src.network.iroh_node import IrohNode, NetworkAlpn
from src.network.transport.asyncio_bridge import (
    IrohStreamReader,
    IrohStreamWriter,
)

logger = logging.getLogger(__name__)

# A contextvar so the cert reaches the middleware without leaking
# through aiohttp's request/response APIs. Set right before
# ``RequestHandler.data_received`` for each HTTP request, read by the
# auth middleware via ``current_device_cert_wire()``.
_current_cert_wire: contextvars.ContextVar[bytes | None] = contextvars.ContextVar(
    "openagent_cert_wire", default=None,
)
_current_peer_node_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "openagent_peer_node_id", default=None,
)
# Set to True by AgentSite (openagent/agent/1 ALPN) to signal that this
# stream is authenticated by Iroh node_id rather than a coordinator cert.
_is_authenticated_agent: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "openagent_is_authenticated_agent", default=False,
)


def current_device_cert_wire() -> bytes | None:
    """Return the cert wire bytes for the in-flight request, if any."""
    return _current_cert_wire.get()


def current_peer_node_id() -> str | None:
    """Return the dialing peer's NodeId for the in-flight request."""
    return _current_peer_node_id.get()


def current_is_authenticated_agent() -> bool:
    """Return True when the in-flight request arrived via the AGENT ALPN.

    The auth middleware checks this to skip cert verification and
    synthesise a DeviceCert for the peer agent instead.
    """
    return _is_authenticated_agent.get()


CERT_LEN_BYTES = 4
MAX_CERT_LEN = 16 * 1024  # generous bound; a real cert is ~500 bytes
# Handshake timeout: a peer that connects but never sends the cert prefix
# must not hold a task forever. Do not wrap the whole stream with this
# value: upgraded WebSockets are expected to stay open beyond it.
_STREAM_TIMEOUT = 120.0


class IrohSite(web.BaseSite):
    """A ``BaseSite`` that doesn't bind a TCP listener.

    Plugs into ``web.AppRunner`` exactly like ``web.TCPSite`` would: the
    runner thinks it owns the lifecycle. The accept loop is owned by
    the underlying ``IrohNode`` — we just register a handler on it.
    """

    __slots__ = ("_app", "_node", "_started")

    def __init__(self, runner: web.AppRunner, node: IrohNode) -> None:
        super().__init__(runner, shutdown_timeout=60.0)
        self._node = node
        self._started = False
        # NB: the ALPN handler is registered earlier (in
        # ``Gateway._prepare_iroh_site``, called before the iroh node
        # binds) because iroh-py 0.35 bakes the ALPN dict into
        # ``NodeOptions`` at construction time. The pre-registered
        # handler is a thin shim that delegates back to this site's
        # ``_handle_stream`` once construction completes. By the time
        # any inbound stream lands, ``Gateway._site`` has been set.

    @property
    def name(self) -> str:
        return f"iroh://{self._node.identity.public_hex[:8]}"

    async def start(self) -> None:
        if self._started:
            return
        await super().start()
        self._started = True

    async def _handle_stream(self, connection: object) -> None:
        """Handle one accepted Iroh connection.

        Drains bi-streams off the connection, dispatching each to its
        own task. Each stream is treated as one HTTP/1.1 request (or
        one WebSocket upgrade) by the existing aiohttp ``RequestHandler``.
        Iroh multiplexes streams over QUIC so concurrent in-flight
        requests don't block one another.
        """
        peer_node_id = "unknown"
        try:
            # ``remote_node_id`` is sync in iroh-py 0.35.
            raw = connection.remote_node_id()
            peer_node_id = str(raw) if raw is not None else "unknown"
        except Exception:
            pass

        _tasks: set[asyncio.Task] = set()
        try:
            while True:
                try:
                    bi = await connection.accept_bi()
                except Exception as e:  # noqa: BLE001
                    logger.debug("iroh-gw: connection ended: %s", e)
                    break
                if bi is None:
                    break
                send_stream = bi.send()
                recv_stream = bi.recv()
                # Spawn per-stream task so a slow request doesn't block
                # other streams on this same connection.
                task = asyncio.create_task(
                    self._handle_one_stream(peer_node_id, send_stream, recv_stream),
                    name=f"iroh-gw-stream-{peer_node_id[:8]}",
                )
                _tasks.add(task)
                task.add_done_callback(_tasks.discard)
        except asyncio.CancelledError:
            raise
        finally:
            for t in list(_tasks):
                if not t.done():
                    t.cancel()
            if _tasks:
                await asyncio.gather(*_tasks, return_exceptions=True)

    async def _handle_one_stream(
        self,
        peer_node_id: str,
        send_stream: object,
        recv_stream: object,
    ) -> None:
        """Handle one bi-stream: cert prefix → HTTP/1 dispatch → response."""
        try:
            await self._handle_one_stream_inner(peer_node_id, send_stream, recv_stream)
        except asyncio.TimeoutError:
            logger.debug(
                "iroh-gw: stream handshake timed out after %.0fs from %s",
                _STREAM_TIMEOUT,
                peer_node_id[:16],
            )
        except asyncio.CancelledError:
            raise

    async def _handle_one_stream_inner(
        self,
        peer_node_id: str,
        send_stream: object,
        recv_stream: object,
    ) -> None:
        """Inner handler: cert prefix → HTTP/1 dispatch → response."""
        # ── Strip the cert prefix ──
        try:
            length_bytes = await _read_exact(
                recv_stream,
                CERT_LEN_BYTES,
                timeout=_STREAM_TIMEOUT,
            )
        except _StreamClosed:
            return
        cert_len = int.from_bytes(length_bytes, "big")
        if cert_len < 0 or cert_len > MAX_CERT_LEN:
            logger.warning(
                "rejecting stream from %s: cert_len out of range (%d)",
                peer_node_id, cert_len,
            )
            return
        cert_wire: bytes | None = None
        if cert_len > 0:
            try:
                cert_wire = await _read_exact(
                    recv_stream,
                    cert_len,
                    timeout=_STREAM_TIMEOUT,
                )
            except _StreamClosed:
                return

        # ── Wire the streams into aiohttp ──
        # ``RequestHandler`` is an asyncio.Protocol. Normally
        # ``loop.create_server`` instantiates it and feeds it bytes via
        # ``data_received``. We have no listening socket here, so we
        # construct it directly and pump bytes from the iroh recv
        # stream into ``data_received`` ourselves. The IrohStreamWriter
        # impersonates the outgoing transport so aiohttp's writes go
        # back over the same iroh bi-stream.
        loop = asyncio.get_event_loop()
        writer = IrohStreamWriter(
            send_stream,
            peer_node_id=peer_node_id,
            alpn=NetworkAlpn.GATEWAY,
        )
        # Set the cert contextvar BEFORE ``connection_made`` — that
        # call spawns the request-handler task, which copies the
        # ambient contextvars at task-creation time. Setting after
        # would mean the auth middleware (running inside that task)
        # sees ``None`` and rejects every request as "missing cert".
        cert_token = _current_cert_wire.set(cert_wire)
        node_token = _current_peer_node_id.set(peer_node_id)

        protocol = RequestHandler(
            self._runner.server,
            loop=loop,
            access_log=None,
        )
        protocol.connection_made(_ProtocolTransport(writer))

        async def _pump_iroh_to_protocol() -> None:
            try:
                while True:
                    chunk = await recv_stream.read(64 * 1024)
                    if not chunk:
                        break
                    protocol.data_received(chunk)
            except Exception as e:  # noqa: BLE001
                logger.debug("iroh-gw: read pump ended: %s", e)
            finally:
                # Tell aiohttp the client side has gone away. Without
                # this its parser keeps waiting for headers and
                # ``task_handler`` never completes.
                try:
                    protocol.connection_lost(None)
                except Exception:
                    pass

        pump_task = asyncio.create_task(
            _pump_iroh_to_protocol(),
            name=f"iroh-gw-pump-{peer_node_id[:8]}",
        )

        try:
            task_handler = getattr(protocol, "_task_handler", None)
            try:
                if task_handler is None:
                    # Older / newer aiohttp may name this differently.
                    # Fall back to polling writer.is_closing().
                    while not writer.is_closing():
                        await asyncio.sleep(0.05)
                else:
                    await task_handler
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.debug("aiohttp task_handler ended: %s", e)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("aiohttp protocol crashed on iroh stream: %s", e)
        finally:
            _current_cert_wire.reset(cert_token)
            _current_peer_node_id.reset(node_token)
            if not pump_task.done():
                pump_task.cancel()
                try:
                    await pump_task
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                writer.close()
            except Exception:
                pass


class _ProtocolTransport:
    """Minimal asyncio.Transport wrapping our IrohStreamWriter.

    aiohttp's RequestHandler stores the transport and queries it for
    peername / write / close. Routing every call through the writer
    keeps both sides consistent.
    """

    __slots__ = ("_writer",)

    def __init__(self, writer: IrohStreamWriter) -> None:
        self._writer = writer

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        return self._writer.get_extra_info(name, default)

    def write(self, data: bytes) -> None:
        self._writer.write(data)

    def writelines(self, data) -> None:
        self._writer.writelines(data)

    def is_closing(self) -> bool:
        return self._writer.is_closing()

    def close(self) -> None:
        self._writer.close()

    def can_write_eof(self) -> bool:
        return self._writer.can_write_eof()

    def write_eof(self) -> None:
        # Sync version — schedule the async finish.
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return
        loop.create_task(self._writer.write_eof())

    def pause_reading(self) -> None:
        # Backpressure isn't wired through to Iroh's flow control — the
        # pump task already buffers chunks. Real flow control would
        # need ``recv_stream.stop()``/``resume()`` which iroh-py 0.35
        # doesn't expose on Python.
        return

    def resume_reading(self) -> None:
        return

    def set_write_buffer_limits(self, high: int | None = None, low: int | None = None) -> None:
        return


# ── Helpers ──────────────────────────────────────────────────────────────


class _StreamClosed(Exception):
    """The Iroh stream was closed before we read all expected bytes."""


async def _read_exact(stream: object, n: int, *, timeout: float | None = None) -> bytes:
    """Read exactly *n* bytes off an Iroh recv stream or raise."""
    out = bytearray()
    while len(out) < n:
        try:
            read_coro = stream.read(n - len(out))
            if timeout is None:
                chunk = await read_coro
            else:
                chunk = await asyncio.wait_for(read_coro, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            if isinstance(e, asyncio.TimeoutError):
                raise
            raise _StreamClosed(str(e)) from e
        if not chunk:
            raise _StreamClosed("eof before frame complete")
        out.extend(chunk)
    return bytes(out)
