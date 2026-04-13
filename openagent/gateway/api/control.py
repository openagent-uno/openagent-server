"""Control REST API — update and restart OpenAgent.

POST /api/update  → trigger upgrade (pip or executable) + restart if updated
POST /api/restart → restart OpenAgent processes
"""

from __future__ import annotations

from openagent.core.logging import elog


async def handle_update(request):
    """Check for updates and install if available."""
    from aiohttp import web
    from openagent.core.server import run_upgrade, RESTART_EXIT_CODE

    gw = request.app["gateway"]
    try:
        old, new = run_upgrade()
    except Exception as exc:
        elog("update.error", error=str(exc))
        return web.json_response({"error": str(exc)}, status=500)

    if old == new:
        elog("update.check", version=old, updated=False)
        return web.json_response({"updated": False, "version": old})

    elog("update.installed", old=old, new=new)
    # Signal restart
    gw.agent._restart_exit_code = RESTART_EXIT_CODE
    if hasattr(gw, "_stop_event") and gw._stop_event:
        gw._stop_event.set()
    return web.json_response({"updated": True, "old": old, "new": new})


async def handle_restart(request):
    """Restart OpenAgent processes."""
    from aiohttp import web
    from openagent.core.server import RESTART_EXIT_CODE

    gw = request.app["gateway"]
    elog("server.restart", source="api")
    gw.agent._restart_exit_code = RESTART_EXIT_CODE
    if hasattr(gw, "_stop_event") and gw._stop_event:
        gw._stop_event.set()
    return web.json_response({"ok": True})
