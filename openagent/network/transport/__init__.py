"""Iroh-based transport for aiohttp.

Two pieces:

  - ``asyncio_bridge``: wraps an Iroh ``RecvStream``/``SendStream`` pair
    in an asyncio ``StreamReader``/``StreamWriter`` so anything that
    speaks asyncio streams can ride on top.
  - ``aiohttp_iroh_site``: a custom aiohttp ``BaseSite`` that drives
    an ``IrohNode`` instead of a TCP listener. Each accepted bi-stream
    becomes one HTTP/1.1 request (or one upgraded WebSocket session)
    handled by aiohttp's existing ``RequestHandler``.

The transport layer knows nothing about authentication. The cert
prefix is stripped by ``aiohttp_iroh_site`` before handing the stream
to aiohttp, then placed on ``request["device_cert_wire"]`` for the
auth middleware to verify.
"""

from openagent.network.transport.aiohttp_iroh_site import IrohSite
from openagent.network.transport.asyncio_bridge import (
    IrohStreamReader,
    IrohStreamWriter,
)

__all__ = ["IrohSite", "IrohStreamReader", "IrohStreamWriter"]
