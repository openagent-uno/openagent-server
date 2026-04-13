"""Control REST API — update and restart OpenAgent.

POST /api/update  → trigger upgrade (pip or executable) + restart if updated
POST /api/restart → restart OpenAgent processes
"""

from __future__ import annotations

from openagent.core.logging import elog


def request_restart(gateway, *, source: str) -> None:
    """Set the restart exit code and ask the server loop to stop."""
    from openagent.core.server import RESTART_EXIT_CODE

    elog("server.restart", source=source)
    gateway.agent._restart_exit_code = RESTART_EXIT_CODE
    if getattr(gateway, "_stop_event", None):
        gateway._stop_event.set()


def perform_update(gateway) -> dict:
    """Run the package update flow and return a structured result."""
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
    request_restart(gateway, source="update")
    return {"ok": True, "updated": True, "old": old, "new": new}


async def handle_update(request):
    """Check for updates and install if available."""
    from aiohttp import web

    gw = request.app["gateway"]
    result = perform_update(gw)
    if not result["ok"]:
        return web.json_response({"error": result["error"]}, status=500)
    payload = dict(result)
    payload.pop("ok", None)
    return web.json_response(payload)


async def handle_restart(request):
    """Restart OpenAgent processes."""
    from aiohttp import web

    gw = request.app["gateway"]
    request_restart(gw, source="api")
    return web.json_response({"ok": True})
