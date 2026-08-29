"""GET /api/terminals — list the caller's live PTY terminals.

The interactive-terminal feature is driven entirely over the WebSocket
(see :mod:`src.gateway.terminals` and the ``terminal_*`` protocol
frames). This one REST endpoint exists so another window attached to the
same client websocket can paint that connection's sessions without waiting
for a WS event.

Terminal ownership is the opaque ``connection_id`` returned in ``auth_ok``;
the device certificate still authenticates the request. The endpoint verifies
that the requested connection belongs to that certificate before reading it.
For old clients that do not send a connection id, the only active websocket
for that device is inferred when it is unambiguous. With Desktop and CLI online
simultaneously, omission safely returns an empty list instead of mixing PTYs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web


async def handle_list(request: "web.Request") -> "web.Response":
    from aiohttp import web as _web

    gw = request.app["gateway"]
    cert = request.get("device_cert")
    if cert is None:
        return _web.json_response({"error": "unauthorized"}, status=401)
    device_id = cert.device_pubkey_hex

    connection_id = str(
        request.query.get("connection_id")
        or request.headers.get("X-OpenAgent-Connection-Id")
        or ""
    ).strip()
    if connection_id:
        if gw._chat_client_devices.get(connection_id) != device_id:
            # Do not reveal whether an opaque connection exists for a
            # different certificate.
            return _web.json_response({"terminals": []})
    else:
        candidates = [
            owner_connection_id
            for owner_connection_id, owner_device_id
            in gw._chat_client_devices.items()
            if owner_device_id == device_id
        ]
        connection_id = candidates[0] if len(candidates) == 1 else ""

    sessions = (
        gw.terminals.list_for_connection(connection_id)
        if connection_id
        else []
    )
    return _web.json_response({
        "connection_id": connection_id or None,
        "terminals": [
            {
                "terminal_id": s.terminal_id,
                "pid": s.pid,
                "shell": s.shell,
                "cwd": s.cwd,
                "cols": s.cols,
                "rows": s.rows,
                "running": s.is_running,
            }
            for s in sessions
        ],
    })
