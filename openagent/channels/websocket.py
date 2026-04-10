"""WebSocket channel for the OpenAgent desktop/web app.

Exposes a WebSocket endpoint for real-time bidirectional chat and a small
REST API for non-chat operations (vault, config, MCPs). Both share the
same ``aiohttp`` HTTP server so there's a single port to configure.

Protocol — JSON over WebSocket
-------------------------------

Client → Server::

    {"type": "auth",    "token": "..."}
    {"type": "message", "text": "...", "session_id": "default"}
    {"type": "command", "name": "stop|new|status|queue|help"}
    {"type": "ping"}

Server → Client::

    {"type": "auth_ok",    "agent_name": "...", "version": "..."}
    {"type": "auth_error", "reason": "..."}
    {"type": "status",     "text": "...",  "session_id": "..."}
    {"type": "response",   "text": "...",  "session_id": "..."}
    {"type": "error",      "text": "..."}
    {"type": "queued",     "position": N}
    {"type": "command_result", "text": "..."}
    {"type": "pong"}

REST API (same port)::

    GET /api/health  →  {"status": "ok", "agent": "...", "version": "..."}
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from openagent.channels.base import BaseChannel, parse_response_markers
from openagent.channels.commands import CommandDispatcher
from openagent.channels.formatting import markdown_to_telegram_html
from openagent.channels.queue import UserQueueManager

if TYPE_CHECKING:
    from openagent.core.agent import Agent

logger = logging.getLogger(__name__)


class WebSocketChannel(BaseChannel):
    """WebSocket + REST channel powered by aiohttp."""

    name = "websocket"

    def __init__(
        self,
        agent: Agent,
        host: str = "0.0.0.0",
        port: int = 8765,
        token: str | None = None,
        allowed_origins: list[str] | None = None,
    ):
        super().__init__(agent)
        self.host = host
        self.port = port
        self.token = token
        self.allowed_origins = set(allowed_origins) if allowed_origins else None
        self._queue = UserQueueManager(platform="websocket", agent_name=agent.name)
        self._commands = CommandDispatcher(agent, self._queue)
        self._app = None
        self._runner = None
        self._clients: dict[str, object] = {}  # client_id → aiohttp.WebSocketResponse

    async def _run(self) -> None:
        try:
            from aiohttp import web
        except ImportError:
            raise ImportError(
                "aiohttp is required for the WebSocket channel. "
                "Install it with: pip install openagent-framework[websocket]"
            )

        app = web.Application()
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_get("/api/health", self._handle_health)
        self._app = app

        runner = web.AppRunner(app)
        await runner.setup()
        self._runner = runner

        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info(
            "WebSocket channel listening on ws://%s:%d/ws",
            self.host, self.port,
        )

        assert self._stop_event is not None
        await self._stop_event.wait()

    async def _shutdown(self) -> None:
        try:
            await self._queue.shutdown()
        except Exception:
            pass
        if self._runner:
            try:
                await self._runner.cleanup()
            finally:
                self._runner = None
                self._app = None
        self._clients.clear()

    # ── WebSocket handler ──────────────────────────────────────────────

    async def _handle_ws(self, request):
        from aiohttp import web, WSMsgType

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        client_id: str | None = None
        authenticated = self.token is None  # no token = open access

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await ws.send_json({"type": "error", "text": "Invalid JSON"})
                        continue

                    msg_type = data.get("type", "")

                    # ── auth ──
                    if msg_type == "auth":
                        if self.token and data.get("token") != self.token:
                            await ws.send_json({"type": "auth_error", "reason": "Invalid token"})
                            await ws.close()
                            return ws
                        client_id = data.get("client_id") or f"ws-{id(ws)}"
                        authenticated = True
                        self._clients[client_id] = ws
                        import openagent
                        await ws.send_json({
                            "type": "auth_ok",
                            "agent_name": self.agent.name,
                            "version": getattr(openagent, "__version__", "?"),
                        })
                        continue

                    if not authenticated:
                        await ws.send_json({"type": "auth_error", "reason": "Not authenticated"})
                        continue

                    if client_id is None:
                        client_id = f"ws-{id(ws)}"
                        self._clients[client_id] = ws

                    # ── ping ──
                    if msg_type == "ping":
                        await ws.send_json({"type": "pong"})
                        continue

                    # ── command ──
                    if msg_type == "command":
                        cmd_name = data.get("name", "")
                        result = await self._commands.dispatch(f"/{cmd_name}", client_id)
                        if result:
                            await ws.send_json({"type": "command_result", "text": result.text})
                        else:
                            await ws.send_json({"type": "error", "text": f"Unknown command: {cmd_name}"})
                        continue

                    # ── message ──
                    if msg_type == "message":
                        text = data.get("text", "").strip()
                        session_id = data.get("session_id", "default")
                        if not text:
                            continue

                        async def handler(
                            _text=text,
                            _sid=session_id,
                            _cid=client_id,
                            _ws=ws,
                        ):
                            await self._process_message(_ws, _cid, _text, _sid)

                        position = await self._queue.enqueue(client_id, handler)
                        if position > 0:
                            await ws.send_json({"type": "queued", "position": position})
                        continue

                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break

        except Exception as e:
            logger.error("WebSocket error for %s: %s", client_id, e)
        finally:
            if client_id and client_id in self._clients:
                del self._clients[client_id]

        return ws

    async def _process_message(self, ws, client_id: str, text: str, session_id: str) -> None:
        """Run agent and stream status updates + final response back."""
        try:
            async def on_status(status: str) -> None:
                try:
                    await ws.send_json({
                        "type": "status",
                        "text": status,
                        "session_id": session_id,
                    })
                except Exception:
                    pass

            response = await self.agent.run(
                message=text,
                user_id=client_id,
                session_id=self._queue.get_session_id(client_id),
                on_status=on_status,
            )

            clean_text, attachments = parse_response_markers(response)
            att_list = [
                {"type": a.type, "path": a.path, "filename": a.filename}
                for a in attachments
            ]

            await ws.send_json({
                "type": "response",
                "text": clean_text,
                "session_id": session_id,
                "attachments": att_list if att_list else None,
            })

        except Exception as e:
            logger.error("WebSocket process error for %s: %s", client_id, e)
            try:
                await ws.send_json({"type": "error", "text": str(e)})
            except Exception:
                pass

    # ── REST endpoints ────────────────────────────────────────────────

    async def _handle_health(self, request):
        from aiohttp import web
        import openagent
        return web.json_response({
            "status": "ok",
            "agent": self.agent.name,
            "version": getattr(openagent, "__version__", "?"),
            "connected_clients": len(self._clients),
        })
