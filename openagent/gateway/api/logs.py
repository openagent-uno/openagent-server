"""Event log REST API — read and clear the unified event log.

GET    /api/logs?lines=100&event=tool.error  → recent log entries
DELETE /api/logs                              → clear the log file
"""

from __future__ import annotations


async def handle_get(request):
    """Return the last N log entries, optionally filtered by event prefix."""
    from aiohttp import web
    from openagent.core.logging import read_tail

    lines = int(request.query.get("lines", "100"))
    event_filter = request.query.get("event")

    entries = read_tail(lines=lines, event_filter=event_filter)
    return web.json_response(entries)


async def handle_delete(request):
    """Clear the event log file."""
    from aiohttp import web
    from openagent.core.logging import clear

    clear()
    return web.json_response({"ok": True})
