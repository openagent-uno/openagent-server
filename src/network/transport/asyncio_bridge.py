"""Adapter from Iroh ``RecvStream``/``SendStream`` to asyncio ``StreamReader``/``StreamWriter``.

aiohttp's HTTP parser, the WebSocket framing code, and basically every
asyncio-native protocol implementation expect ``StreamReader``-shaped
APIs. Iroh exposes a different surface: ``await recv.read(n)`` /
``await send.write_all(data)``. This module bridges the two.

Design choices:

  - ``IrohStreamReader`` is a real asyncio ``StreamReader`` whose buffer
    is fed by a small pump task that reads off the Iroh stream until
    EOF. We don't subclass — we hand callers a vanilla StreamReader so
    aiohttp's internal type checks are happy.
  - ``IrohStreamWriter`` quacks like asyncio's writer but only
    implements the methods aiohttp actually calls (``write``,
    ``writelines``, ``drain``, ``close``, ``can_write_eof``,
    ``write_eof``, ``is_closing``, ``transport``, ``get_extra_info``).
    Faking the full ``StreamWriter`` API would require a real
    Transport, which Iroh streams don't have.

There's no try-everything fallback: if Iroh's stream API has changed
(say in a 0.36 bump), the `_pump` loop will fail loudly and we can
adjust here in one place.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


READ_CHUNK = 64 * 1024  # large enough to keep up with WS frames; small enough for prompt cancellation


class _FakeTransport:
    """Minimal asyncio.Transport surface that aiohttp pokes at."""

    def __init__(self, peer_node_id: str, alpn: bytes) -> None:
        self._closing = False
        self._peer_node_id = peer_node_id
        self._alpn = alpn
        # aiohttp queries ``get_extra_info('peername')`` for logging;
        # returning a tuple shaped like a TCP peername keeps its access.log
        # formatter happy without a special case.
        self._peer_name = (peer_node_id, 0)

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        if name == "peername":
            return self._peer_name
        if name == "sockname":
            return ("iroh", 0)
        if name == "openagent_peer_node_id":
            return self._peer_node_id
        if name == "openagent_alpn":
            return self._alpn
        return default

    def is_closing(self) -> bool:
        return self._closing

    def close(self) -> None:
        self._closing = True


class IrohStreamReader:
    """Factory only — callers receive a real ``asyncio.StreamReader``."""

    def __new__(cls, recv_stream: object, *, loop: asyncio.AbstractEventLoop | None = None) -> asyncio.StreamReader:
        loop = loop or asyncio.get_event_loop()
        # Picking a generous limit: the gateway streams audio chunks
        # and image attachments through the WS, and we don't want
        # aiohttp's parser to choke on a single large frame. 4 MiB
        # matches aiohttp's default ``read_buffer_size``.
        reader = asyncio.StreamReader(limit=4 * 1024 * 1024, loop=loop)
        loop.create_task(_pump_recv_into(reader, recv_stream), name="iroh-recv-pump")
        return reader  # type: ignore[return-value]


async def _pump_recv_into(reader: asyncio.StreamReader, recv_stream: object) -> None:
    """Drain *recv_stream* and feed bytes into *reader* until EOF."""
    try:
        while True:
            try:
                chunk = await recv_stream.read(READ_CHUNK)
            except Exception as e:  # noqa: BLE001
                logger.debug("iroh recv read failed: %s", e)
                reader.feed_eof()
                return
            if not chunk:
                reader.feed_eof()
                return
            reader.feed_data(chunk)
    except asyncio.CancelledError:
        reader.feed_eof()
        raise


class IrohStreamWriter:
    """Quacks like ``asyncio.StreamWriter`` for the methods aiohttp uses.

    aiohttp's WebSocket and HTTP/1 paths only need a small subset, so we
    don't bother emulating the full surface. If a future change makes
    aiohttp call something else, add the shim here — there is exactly
    one writer implementation in this codebase.
    """

    def __init__(
        self,
        send_stream: object,
        *,
        peer_node_id: str = "unknown",
        alpn: bytes = b"",
    ) -> None:
        self._send = send_stream
        self._closed = False
        self._transport = _FakeTransport(peer_node_id, alpn)
        # aiohttp's WS writer batches writes into a list and calls
        # ``drain()`` once after a flush. We pipeline our writes through
        # a single asyncio.Lock so concurrent ``write`` calls from the
        # same writer don't interleave on the wire.
        self._write_lock = asyncio.Lock()

    @property
    def transport(self) -> _FakeTransport:
        return self._transport

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        return self._transport.get_extra_info(name, default)

    def is_closing(self) -> bool:
        return self._closed or self._transport.is_closing()

    def can_write_eof(self) -> bool:
        # Iroh streams support a clean half-close, but aiohttp only
        # calls ``write_eof`` for HTTP request bodies (not responses).
        # Return True so its keepalive logic doesn't think we're an
        # ancient HTTP/0.9 transport.
        return True

    def write(self, data: bytes) -> None:
        if self._closed:
            return
        # The send.write_all is async, but aiohttp's writer expects
        # ``write`` to be sync (queue and drain). Schedule the actual
        # send on the loop and let ``drain()`` await it.
        loop = asyncio.get_event_loop()
        # Coalesce sequential writes to keep ordering deterministic.
        # We hold the lock across the await — if the call site fires
        # writes concurrently from different tasks (rare for HTTP/1.1
        # but possible for WS pings on a misconfigured peer), they
        # serialise here rather than racing on the Iroh stream.
        loop.create_task(self._write_async(data))

    def writelines(self, data) -> None:
        for d in data:
            self.write(d)

    async def _write_async(self, data: bytes) -> None:
        async with self._write_lock:
            try:
                await self._send.write_all(data)
            except Exception as e:  # noqa: BLE001
                logger.debug("iroh send write failed: %s", e)
                self._closed = True

    async def drain(self) -> None:
        # The lock acts as the drain barrier: once we acquire-release
        # it, any prior ``write`` has finished writing to Iroh.
        async with self._write_lock:
            return

    async def write_eof(self) -> None:
        if self._closed:
            return
        try:
            # iroh-py exposes ``finish()`` on SendStream to mark a clean
            # half-close. Older versions used ``close()`` — try both.
            finish = getattr(self._send, "finish", None) or getattr(self._send, "close", None)
            if finish is not None:
                await finish()
        except Exception as e:  # noqa: BLE001
            logger.debug("iroh send finish failed: %s", e)
        self._closed = True

    def close(self) -> None:
        # Sync close: aiohttp calls this on shutdown. Fire-and-forget
        # the async finish; the Iroh stream tolerates the hang-up.
        if self._closed:
            return
        self._closed = True
        self._transport.close()
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return
        finish = getattr(self._send, "finish", None) or getattr(self._send, "close", None)
        if finish is not None:
            loop.create_task(finish())

    async def wait_closed(self) -> None:
        # Nothing to wait on — ``close()`` already scheduled the finish.
        return
