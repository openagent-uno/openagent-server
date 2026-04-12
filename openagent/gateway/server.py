"""Gateway server — the single public interface for OpenAgent.

Hosts a WebSocket endpoint for real-time chat and REST endpoints for
vault, config, and health. All clients (Electron app, CLI, bridges)
connect through this server.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from openagent.gateway import protocol as P
from openagent.gateway.sessions import SessionManager
from openagent.gateway.api import vault, config, health

if TYPE_CHECKING:
    from openagent.core.agent import Agent

logger = logging.getLogger(__name__)


class Gateway:
    """WebSocket + REST gateway powered by aiohttp."""

    def __init__(
        self,
        agent: Agent,
        host: str = "0.0.0.0",
        port: int = 8765,
        token: str | None = None,
        vault_path: str | None = None,
        config_path: str | None = None,
    ):
        self.agent = agent
        self.host = host
        self.port = port
        self.token = token
        self.vault_path = vault_path
        self.config_path = config_path
        self.sessions = SessionManager(agent_name=agent.name)
        self.clients: dict[str, object] = {}  # client_id → WebSocketResponse
        self._runner = None

    async def start(self) -> None:
        from aiohttp import web
        from aiohttp.web import middleware

        @middleware
        async def cors(request, handler):
            if request.method == "OPTIONS":
                resp = web.Response(status=204)
            else:
                try:
                    resp = await handler(request)
                except web.HTTPException as ex:
                    resp = ex
                except Exception as exc:
                    logger.exception("REST error: %s", exc)
                    resp = web.Response(status=500, text=str(exc))
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Methods"] = "GET, PUT, PATCH, DELETE, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            return resp

        app = web.Application(middlewares=[cors])
        app["gateway"] = self  # accessible in handlers via request.app["gateway"]

        # WebSocket
        app.router.add_get("/ws", self._handle_ws)
        # REST
        app.router.add_get("/api/health", health.handle_health)
        app.router.add_get("/api/vault/notes", vault.handle_list)
        app.router.add_get("/api/vault/graph", vault.handle_graph)
        app.router.add_get("/api/vault/search", vault.handle_search)
        app.router.add_get("/api/vault/notes/{path:.+}", vault.handle_read)
        app.router.add_put("/api/vault/notes/{path:.+}", vault.handle_write)
        app.router.add_delete("/api/vault/notes/{path:.+}", vault.handle_delete)
        app.router.add_get("/api/config", config.handle_get)
        app.router.add_put("/api/config", config.handle_put)
        app.router.add_patch("/api/config/{section}", config.handle_patch)
        app.router.add_post("/api/upload", self._handle_upload)
        app.router.add_route("OPTIONS", "/{path:.*}", self._handle_options)

        runner = web.AppRunner(app)
        await runner.setup()
        self._runner = runner

        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info("Gateway listening on ws://%s:%d/ws", self.host, self.port)

    async def stop(self) -> None:
        await self.sessions.shutdown()
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self.clients.clear()

    # ── File upload ──

    async def _handle_upload(self, request):
        """POST /api/upload — save file, auto-transcribe if audio.

        Returns {path, filename, transcription?}. If the file is audio
        (webm, ogg, mp3, wav, m4a), it's transcribed via faster-whisper
        or OpenAI Whisper and the text is returned in `transcription`.
        """
        from aiohttp import web
        import tempfile

        reader = await request.multipart()
        field = await reader.next()
        if not field:
            return web.json_response({"error": "No file"}, status=400)

        filename = field.filename or "upload"
        tmp = tempfile.mkdtemp(prefix="oa_upload_")
        path = f"{tmp}/{filename}"
        with open(path, "wb") as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)

        result: dict = {"path": path, "filename": filename}

        # Auto-transcribe audio files
        audio_exts = ('.webm', '.ogg', '.mp3', '.wav', '.m4a', '.opus', '.flac')
        if any(filename.lower().endswith(ext) for ext in audio_exts):
            try:
                from openagent.channels.voice import transcribe
                text = await transcribe(path)
                if text:
                    result["transcription"] = text
            except Exception as e:
                logger.warning("Voice transcription failed: %s", e)

        return web.json_response(result)

    # ── WebSocket ──

    async def _handle_options(self, request):
        from aiohttp import web
        return web.Response(status=204)

    async def _handle_ws(self, request):
        from aiohttp import web, WSMsgType

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        client_id: str | None = None
        authed = self.token is None

        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    break
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    await ws.send_json({"type": P.ERROR, "text": "Invalid JSON"})
                    continue

                t = data.get("type", "")

                # Auth
                if t == P.AUTH:
                    if self.token and data.get("token") != self.token:
                        await ws.send_json({"type": P.AUTH_ERROR, "reason": "Invalid token"})
                        await ws.close()
                        return ws
                    client_id = data.get("client_id") or f"ws-{id(ws)}"
                    authed = True
                    self.clients[client_id] = ws
                    import openagent
                    await ws.send_json({
                        "type": P.AUTH_OK,
                        "agent_name": self.agent.name,
                        "version": getattr(openagent, "__version__", "?"),
                    })
                    continue

                if not authed:
                    await ws.send_json({"type": P.AUTH_ERROR, "reason": "Not authenticated"})
                    continue
                if client_id is None:
                    client_id = f"ws-{id(ws)}"
                    self.clients[client_id] = ws

                # Ping
                if t == P.PING:
                    await ws.send_json({"type": P.PONG})

                # Command
                elif t == P.COMMAND:
                    await self._handle_command(ws, client_id, data.get("name", ""))

                # Message
                elif t == P.MESSAGE:
                    text = data.get("text", "").strip()
                    session_id = data.get("session_id", "default")
                    if text:
                        sid = self.sessions.get_or_create_session(client_id, session_id)

                        async def handler(_t=text, _s=sid, _c=client_id, _w=ws):
                            await self._process_message(_w, _c, _t, _s)

                        pos = await self.sessions.enqueue(client_id, handler)
                        if pos > 0:
                            await ws.send_json({"type": P.QUEUED, "position": pos})

        except Exception as e:
            logger.error("WS error for %s: %s", client_id, e)
        finally:
            if client_id and client_id in self.clients:
                del self.clients[client_id]
        return ws

    async def _handle_command(self, ws, client_id: str, name: str) -> None:
        sm = self.sessions
        if name in ("new", "reset"):
            sid = sm.create_session(client_id)
            text = f"New session: {sid[-8:]}"
        elif name == "stop":
            stopped = sm.stop_current(client_id)
            text = "Stopped." if stopped else "Nothing running."
        elif name == "status":
            busy = sm.is_busy(client_id)
            depth = sm.queue_depth(client_id)
            sessions = sm.list_sessions(client_id)
            text = f"{'Busy' if busy else 'Idle'} | Queue: {depth} | Sessions: {len(sessions)}"
        elif name == "queue":
            text = f"Queue depth: {sm.queue_depth(client_id)}"
        elif name == "clear":
            n = sm.clear_queue(client_id)
            text = f"Queue cleared ({n} messages removed)." if n else "Queue already empty."
        elif name == "help":
            text = (
                "Available commands:\n"
                "• /new — start a fresh conversation (clears context)\n"
                "• /stop — cancel the current operation\n"
                "• /status — show agent status and queue depth\n"
                "• /queue — show pending messages\n"
                "• /clear — clear the message queue\n"
                "• /help — show this help message"
            )
        else:
            text = f"Unknown command: {name}"
        await ws.send_json({"type": P.COMMAND_RESULT, "text": text})

    async def _process_message(self, ws, client_id: str, text: str, session_id: str) -> None:
        try:
            async def on_status(status: str) -> None:
                try:
                    await ws.send_json({"type": P.STATUS, "text": status, "session_id": session_id})
                except Exception:
                    pass

            response = await self.agent.run(
                message=text,
                user_id=client_id,
                session_id=session_id,
                on_status=on_status,
            )

            from openagent.channels.base import parse_response_markers
            clean, attachments = parse_response_markers(response)
            att_list = [{"type": a.type, "path": a.path, "filename": a.filename} for a in attachments]

            await ws.send_json({
                "type": P.RESPONSE,
                "text": clean,
                "session_id": session_id,
                "attachments": att_list or None,
            })
        except Exception as e:
            logger.error("Process error for %s: %s", client_id, e)
            try:
                await ws.send_json({"type": P.ERROR, "text": str(e)})
            except Exception:
                pass
