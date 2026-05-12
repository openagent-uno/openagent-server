"""REST API handlers for Claude Code CLI setup, auth status, and login.

Endpoints:
    GET  /api/claude/status   — binary location + auth state
    POST /api/claude/install  — idempotent auto-install
    POST /api/claude/auth/login — launch OAuth login in a subprocess
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from typing import TYPE_CHECKING

from openagent.models.claude_install import (
    check_claude_auth,
    ensure_claude_binary,
    find_claude_binary,
    full_status,
    launch_claude_login,
)

if TYPE_CHECKING:
    from aiohttp import web

logger = logging.getLogger(__name__)


async def handle_status(request: web.Request) -> web.Response:
    """GET /api/claude/status — binary + auth status.

    Returns::

        {
            "binary_ok": true,
            "binary_path": "/usr/local/bin/claude",
            "auth_ok": true,
            "auth_email": "user@example.com",
            "auth_type": "pro"
        }
    """
    from aiohttp import web as _web

    status = full_status()
    return _web.json_response(
        {
            "binary_ok": status.binary_ok,
            "binary_path": status.binary_path,
            "auth_ok": status.auth_ok,
            "auth_email": status.auth_email,
            "auth_type": status.auth_type,
        }
    )


async def handle_install(request: web.Request) -> web.Response:
    """POST /api/claude/install — idempotent auto-install.

    Runs in a thread to avoid blocking the event loop (the install
    script can take 30-120s). Returns::

        {
            "binary_ok": true,
            "binary_path": "/usr/local/bin/claude",
            "auth_ok": false
        }
    """
    from aiohttp import web as _web

    path = await asyncio.to_thread(ensure_claude_binary)
    if path is None:
        return _web.json_response(
            {
                "binary_ok": False,
                "error": (
                    "Could not install Claude Code CLI. "
                    "Install manually: curl -fsSL https://claude.ai/install.sh | bash"
                ),
            },
            status=500,
        )

    auth = await asyncio.to_thread(check_claude_auth, binary_path=path)
    return _web.json_response(
        {
            "binary_ok": True,
            "binary_path": path,
            "auth_ok": auth.logged_in,
            "auth_email": auth.email,
            "auth_type": auth.account_type,
        }
    )


async def handle_auth_login(request: web.Request) -> web.Response:
    """POST /api/claude/auth/login — launch OAuth login.

    Accepts optional JSON body::

        {"email": "user@example.com", "sso": false, "console": false}

    Spawns ``claude auth login`` as a subprocess which opens the
    browser for OAuth. Returns immediately; the caller should poll
    ``GET /api/claude/status`` to know when auth completes.
    """
    from aiohttp import web as _web

    body = {}
    try:
        if request.can_read_body:
            raw = await request.text()
            if raw:
                body = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        pass

    cli = find_claude_binary()
    if cli is None:
        return _web.json_response(
            {
                "ok": False,
                "error": "Claude Code CLI not found. Run POST /api/claude/install first.",
            },
            status=400,
        )

    email = body.get("email") if isinstance(body, dict) else None
    sso = bool(body.get("sso")) if isinstance(body, dict) else False
    console = bool(body.get("console")) if isinstance(body, dict) else False

    proc = launch_claude_login(
        email=email,
        sso=sso,
        console=console,
        binary_path=cli,
    )

    if proc is None:
        return _web.json_response(
            {"ok": False, "error": "Failed to launch claude auth login"},
            status=500,
        )

    return _web.json_response(
        {
            "ok": True,
            "pid": proc.pid,
            "detail": "Browser launched for OAuth. Poll GET /api/claude/status.",
        }
    )
