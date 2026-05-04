"""Control REST API — update and restart OpenAgent.

POST /api/update  → trigger upgrade (pip or executable) + restart if updated
POST /api/restart → restart OpenAgent processes
"""

from __future__ import annotations

import asyncio

from openagent.core.logging import elog


def _schedule_bridge_offset_flush(gateway) -> None:
    """Proactively ACK pending platform updates *before* the restart fires.

    Without this, the exact Update that triggered /restart can stay in
    Telegram's delivery queue: library shutdown inside
    ``Updater.stop()`` runs ``_get_updates_cleanup`` which is itself a
    ``getUpdates`` POST, and that POST can block or be cancelled as the
    event loop winds down. When launchd restarts us, ``getUpdates`` on
    the next boot still advertises the same Update and the command
    re-fires → crash loop (observed on lyra-agent 2026-04-20).

    We schedule the flush as a background task on the current loop so
    the restart path isn't blocked by network I/O. The bridge's
    ``flush_updates_offset`` swallows its own errors/cancellation.
    """
    bridges = getattr(gateway, "_bridges", None) or []
    for bridge in bridges:
        flush = getattr(bridge, "flush_updates_offset", None)
        if flush is None:
            continue
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (unit test context). Nothing to do.
            return
        try:
            loop.create_task(flush(), name=f"bridge:{bridge.name}:flush-updates")
        except Exception as e:  # noqa: BLE001 — best-effort
            elog(
                "bridge.flush_schedule_error",
                level="warning",
                bridge=getattr(bridge, "name", "?"),
                error=str(e),
            )


def request_restart(gateway, *, source: str) -> None:
    """Set the restart exit code and ask the server loop to stop.

    Before signalling stop we kick off a best-effort bridge offset flush
    so any command that came in via a platform (Telegram today) gets
    ACKed on the platform side, preventing replay after restart.
    """
    from openagent.core.server import RESTART_EXIT_CODE

    elog("server.restart", source=source)
    _schedule_bridge_offset_flush(gateway)
    gateway.agent._restart_exit_code = RESTART_EXIT_CODE
    if getattr(gateway, "_stop_event", None):
        gateway._stop_event.set()


def perform_update(gateway) -> dict:
    """Run the package update flow and return a structured result.

    Note: this function is synchronous and BLOCKS for the duration of
    the download + apply (potentially minutes for a 200 MB archive).
    Callers on the event loop must run it via ``asyncio.to_thread`` or
    the entire gateway becomes unresponsive — observed serialising 5
    concurrent ``/api/update`` calls at ~310 ms each.

    The restart is NOT triggered here. The async caller is responsible
    for scheduling :func:`request_restart` AFTER the HTTP response has
    been flushed; otherwise ``stop_event`` racing the response writer
    leaves the client with "Empty reply from server" while the update
    actually succeeded (observed on performa-agent 2026-05-04).
    """
    from openagent.core.server import run_upgrade

    try:
        old, new = run_upgrade()
    except Exception as exc:
        elog("update.error", error=str(exc))
        return {"ok": False, "error": str(exc)}

    if old == new:
        elog("update.check", version=old, updated=False)
        return {"ok": True, "updated": False, "version": old}

    elog("update.installed", old=old, new=new)
    return {"ok": True, "updated": True, "old": old, "new": new, "restart_needed": True}


async def handle_update(request):
    """Check for updates and install if available."""
    from aiohttp import web

    gw = request.app["gateway"]
    # Run the (blocking) upgrade flow off the event loop. Without this
    # the gateway is unresponsive for the entire download — concurrent
    # /api/health, WS frames, and bridge polling all stall.
    result = await asyncio.to_thread(perform_update, gw)
    if not result["ok"]:
        return web.json_response({"error": result["error"]}, status=500)

    payload = {k: v for k, v in result.items() if k not in ("ok", "restart_needed")}
    response = web.json_response(payload)

    if result.get("restart_needed"):
        # Schedule the restart on the loop so the response writer flushes
        # FIRST. ``request_restart`` sets ``stop_event`` which the server
        # loop picks up immediately; if we triggered it inside the same
        # tick as the response, the client could see "Empty reply from
        # server" while the update actually succeeded.
        async def _delayed_restart():
            # Short sleep gives aiohttp time to drain the response onto
            # the socket and for the kernel to push the bytes. 0.5 s is
            # negligible against the 5-30 s shutdown that follows.
            await asyncio.sleep(0.5)
            request_restart(gw, source="update")
        asyncio.get_running_loop().create_task(
            _delayed_restart(), name="control:delayed-restart"
        )

    return response


async def handle_restart(request):
    """Restart OpenAgent processes."""
    from aiohttp import web

    gw = request.app["gateway"]
    request_restart(gw, source="api")
    return web.json_response({"ok": True})
