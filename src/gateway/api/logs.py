"""Event log REST API — read and clear the unified event log.

GET    /api/logs?lines=100&event=tool.error  → recent log entries
DELETE /api/logs                              → clear the log file
"""

from __future__ import annotations

import asyncio


async def handle_get(request):
    """Return the last N log entries, optionally filtered by event prefix.

    ``read_tail`` is plain blocking file I/O and this handler is async, so the
    read goes to a thread. It used to run inline while slurping the *entire*
    events.jsonl (measured 728 KB / 5365 entries on a light install; dream
    mode trims the file by age, not size) — on the gateway's event loop, the
    same loop carrying live WebSocket streams and voice audio, where a stall
    is something the user hears mid-sentence. ``read_tail`` now reads
    backwards only as far as it needs, but that bounds the work rather than
    moving it: only the thread hop keeps the loop free while the file is
    touched, which still matters for a filter that has to reach far back.
    """
    from aiohttp import web
    from src.core.logging import read_tail

    lines = int(request.query.get("lines", "100"))
    event_filter = request.query.get("event")

    entries = await asyncio.to_thread(read_tail, lines, event_filter)
    return web.json_response(entries)


async def handle_delete(request):
    """Clear the event log file."""
    from aiohttp import web
    from src.core.logging import clear

    clear()
    return web.json_response({"ok": True})
