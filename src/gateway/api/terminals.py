"""GET /api/terminals — list the caller's live PTY terminals.

The interactive-terminal feature is driven entirely over the WebSocket
(see :mod:`src.gateway.terminals` and the ``terminal_*`` protocol
frames). This one REST endpoint exists so a freshly-opened client (or a
detached terminal window that just resumed the connection) can paint the
list of sessions it already has running without waiting for a WS event.

Terminals are scoped to the calling *device* — the cert pubkey on the
request, which is the same ``client_id`` the WS handler keys terminals
under — so a user only ever sees their own shells, never another
member's.
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
    client_id = cert.device_pubkey_hex

    sessions = gw.terminals.list_for_client(client_id)
    return _web.json_response({
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
