"""Gateway server — the single public interface for OpenAgent.

Hosts a WebSocket endpoint for real-time chat and REST endpoints for
vault, config, and health. All clients (Electron app, CLI, bridges)
connect through this server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from pathlib import Path
from typing import TYPE_CHECKING

from openagent.gateway import protocol as P
from openagent.gateway.commands import command_help_text
from openagent.gateway.sessions import SessionManager
from openagent.gateway.api import vault, config, health, logs, control, usage, providers, models

if TYPE_CHECKING:
    from openagent.core.agent import Agent

from openagent.core.logging import elog

logger = logging.getLogger(__name__)


def _find_available_port(preferred: int, host: str = "0.0.0.0") -> int:
    """Try the preferred port, then scan +1..+99 for an available one."""
    for port in range(preferred, preferred + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                return port
        except OSError:
            continue
    raise RuntimeError(
        f"No available port found in range {preferred}–{preferred + 99}"
    )


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
        stop_event: asyncio.Event | None = None,
    ):
        self.agent = agent
        self.host = host
        self.port = _find_available_port(port, host)
        if self.port != port:
            logger.info("Port %d busy, using %d instead", port, self.port)
        self.token = token
        self.vault_path = vault_path
        self.config_path = config_path
        self._stop_event = stop_event
        self.sessions = SessionManager(agent_name=agent.name)
        self.clients: dict[str, object] = {}  # client_id → WebSocketResponse
        self._runner = None
        self._port_file = None
        self._model_cache: dict[str, object] = {}  # model_spec → BaseModel instance

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
                    elog("gateway.rest_error", path=request.path, method=request.method, error=str(exc))
                    resp = web.Response(status=500, text=str(exc))
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            return resp

        app = web.Application(middlewares=[cors])
        app["gateway"] = self  # accessible in handlers via request.app["gateway"]
        self._register_routes(app)

        runner = web.AppRunner(app)
        await runner.setup()
        self._runner = runner

        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info("Gateway listening on ws://%s:%d/ws", self.host, self.port)
        elog("gateway.start", host=self.host, port=self.port)

        # Write .port file for agent discovery
        self._write_port_file()

    async def stop(self) -> None:
        await self.sessions.shutdown()
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self.clients.clear()
        self._remove_port_file()

    def _register_routes(self, app) -> None:
        """Register the gateway WebSocket endpoint and REST API routes."""
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_post("/api/upload", self._handle_upload)
        app.router.add_get("/api/agent-info", self._handle_agent_info)

        routes = (
            ("GET", "/api/health", health.handle_health),
            ("GET", "/api/vault/notes", vault.handle_list),
            ("GET", "/api/vault/graph", vault.handle_graph),
            ("GET", "/api/vault/search", vault.handle_search),
            ("GET", "/api/vault/notes/{path:.+}", vault.handle_read),
            ("PUT", "/api/vault/notes/{path:.+}", vault.handle_write),
            ("DELETE", "/api/vault/notes/{path:.+}", vault.handle_delete),
            ("GET", "/api/config", config.handle_get),
            ("PUT", "/api/config", config.handle_put),
            ("PATCH", "/api/config/{section}", config.handle_patch),
            ("GET", "/api/logs", logs.handle_get),
            ("DELETE", "/api/logs", logs.handle_delete),
            ("GET", "/api/usage", usage.handle_get),
            ("GET", "/api/usage/daily", usage.handle_daily),
            ("GET", "/api/usage/pricing", usage.handle_pricing),
            ("GET", "/api/providers", providers.handle_list),
            ("POST", "/api/providers/test", providers.handle_test),
            ("GET", "/api/models/catalog", models.handle_catalog),
            ("GET", "/api/models/providers", models.handle_available_providers),
            ("GET", "/api/models", models.handle_list),
            ("POST", "/api/models", models.handle_create),
            ("GET", "/api/models/active", models.handle_get_active),
            ("PUT", "/api/models/active", models.handle_set_active),
            ("PUT", "/api/models/{name}", models.handle_update),
            ("DELETE", "/api/models/{name}", models.handle_delete),
            ("POST", "/api/models/{name}/test", models.handle_test),
            ("POST", "/api/update", control.handle_update),
            ("POST", "/api/restart", control.handle_restart),
        )
        for method, path, handler in routes:
            app.router.add_route(method, path, handler)
        app.router.add_route("OPTIONS", "/{path:.*}", self._handle_options)

    def _write_port_file(self) -> None:
        """Write a .port file to the agent dir for discovery by CLI/app."""
        from openagent.core.paths import get_agent_dir
        agent_dir = get_agent_dir()
        if agent_dir is not None:
            port_file = agent_dir / ".port"
            port_file.write_text(str(self.port))
            self._port_file = port_file

    def _remove_port_file(self) -> None:
        """Remove the .port file on shutdown."""
        if self._port_file and self._port_file.exists():
            try:
                self._port_file.unlink()
            except OSError:
                pass

    def runtime_info(self) -> dict:
        """Return shared gateway/agent metadata exposed by REST endpoints."""
        import openagent
        from openagent.core.paths import get_agent_dir

        agent_dir = get_agent_dir()
        return {
            "agent": self.agent.name,
            "agent_dir": str(agent_dir) if agent_dir else None,
            "port": self.port,
            "version": getattr(openagent, "__version__", "?"),
        }

    async def _handle_agent_info(self, request):
        """GET /api/agent-info — agent name, dir, port, version."""
        from aiohttp import web

        info = self.runtime_info()
        return web.json_response({
            "name": info["agent"],
            "agent_dir": info["agent_dir"],
            "port": info["port"],
            "version": info["version"],
        })

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
        elog("upload.received", filename=filename)
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
        from openagent.channels.voice import is_audio_file

        if is_audio_file(filename):
            try:
                from openagent.channels.voice import transcribe
                text = await transcribe(path)
                if text:
                    result["transcription"] = text
                    elog("upload.transcribed", filename=filename, chars=len(text))
            except Exception as e:
                logger.warning("Voice transcription failed: %s", e)
                elog("upload.transcribe_error", filename=filename, error=str(e))

        elog("upload.saved", filename=filename, path=path, transcribed=bool(result.get("transcription")))
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
                        elog("auth.fail", client_id=data.get("client_id"))
                        await ws.send_json({"type": P.AUTH_ERROR, "reason": "Invalid token"})
                        await ws.close()
                        return ws
                    client_id = data.get("client_id") or f"ws-{id(ws)}"
                    authed = True
                    self.clients[client_id] = ws
                    import openagent
                    elog("gateway.client_connect", client_id=client_id)
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
                    elog("command.received", client_id=client_id, name=data.get("name", ""))
                    await self._handle_command(ws, client_id, data.get("name", ""))

                # Message
                elif t == P.MESSAGE:
                    text = data.get("text", "").strip()
                    session_id = data.get("session_id", "default")
                    if text:
                        sid = self.sessions.get_or_create_session(client_id, session_id)

                        async def handler(_t=text, _s=sid, _c=client_id, _w=ws):
                            await self._process_message(_w, _c, _t, _s)

                        pos = await self.sessions.enqueue(client_id, handler, session_id=sid)
                        if pos < 0:
                            await ws.send_json({"type": P.ERROR, "text": "Too many messages queued. Please wait.", "session_id": sid})
                        elif pos > 0:
                            elog("message.queued", client_id=client_id, session_id=sid, position=pos)
                            await ws.send_json({"type": P.QUEUED, "position": pos})

        except Exception as e:
            logger.error("WS error for %s: %s", client_id, e)
            elog("gateway.ws_error", client_id=client_id, error=str(e))
        finally:
            if client_id and client_id in self.clients:
                del self.clients[client_id]
                elog("gateway.client_disconnect", client_id=client_id)
        return ws

    async def _handle_command(self, ws, client_id: str, name: str) -> None:
        sm = self.sessions
        if name in ("new", "reset"):
            sid = sm.create_session(client_id)
            text = f"New session: {sid[-8:]}"
        elif name == "stop":
            stopped = sm.stop_current(client_id)
            cleared = sm.clear_queue(client_id)
            parts = []
            if stopped:
                parts.append("Stopped current operation")
            if cleared:
                parts.append(f"cleared {cleared} queued message{'s' if cleared != 1 else ''}")
            text = ". ".join(parts) + "." if parts else "Nothing running."
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
        elif name == "usage":
            from openagent.gateway.api.usage import _usage_summary_for_agent

            summary = await _usage_summary_for_agent(self.agent)
            spend = float(summary.get("monthly_spend", 0) or 0)
            budget = summary.get("monthly_budget")
            by_model = summary.get("by_model", {}) or {}
            if budget:
                text = f"Usage: ${spend:.4f} / ${float(budget):.4f} this month across {len(by_model)} model(s)."
            else:
                text = f"Usage tracking available for {len(by_model)} model(s); monthly spend is ${spend:.4f}."
        elif name == "update":
            result = control.perform_update(self)
            if not result["ok"]:
                text = f"Update failed: {result['error']}"
            elif result["updated"]:
                text = f"Updated: v{result['old']} → v{result['new']}. Restarting..."
            else:
                text = f"Already up-to-date (v{result['version']})."
        elif name == "restart":
            text = "Restarting..."
            control.request_restart(self, source="ws_command")
        elif name == "help":
            text = command_help_text()
        else:
            text = f"Unknown command: {name}"
        elog("command.result", client_id=client_id, name=name, text=text)
        await ws.send_json({"type": P.COMMAND_RESULT, "text": text})

    def _resolve_channel_model(self, client_id: str):
        """Resolve per-channel model override from config."""
        channel = client_id.split(":", 1)[1] if client_id.startswith("bridge:") else "websocket"
        logger.debug("Resolving model for channel=%s (client_id=%s)", channel, client_id)

        # Read channels config from YAML
        if not self.config_path:
            return None
        try:
            from openagent.gateway.api.config import _load_resolved_config

            raw = _load_resolved_config(Path(self.config_path))
            channels_cfg = raw.get("channels", {})
            model_spec = channels_cfg.get(channel, {}).get("model")
            if not model_spec:
                return None
            elog("channel.model_override", client_id=client_id, channel=channel, spec=model_spec)
            return self._get_or_create_model(model_spec, raw.get("providers", {}))
        except Exception as e:
            logger.debug("Channel model resolution failed: %s", e)
            elog("channel.model_override_error", client_id=client_id, channel=channel, error=str(e))
            return None

    def _get_or_create_model(self, spec: str, providers_config: dict = None):
        """Get or create a cached model instance for a spec string."""
        if spec in self._model_cache:
            elog("model.override_cache_hit", spec=spec)
            return self._model_cache[spec]

        from openagent.models.runtime import create_model_from_spec

        model = create_model_from_spec(
            spec,
            providers_config=providers_config or {},
            db=self.agent._db,
            mcp_registry=self.agent._mcp,
            mcp_servers=getattr(self.agent._mcp, "_servers", None),
        )

        self._model_cache[spec] = model
        elog("model.override_create", spec=spec, kind=type(model).__name__)
        return model

    async def _process_message(self, ws, client_id: str, text: str, session_id: str) -> None:
        try:
            elog("message.received", client_id=client_id, session_id=session_id, length=len(text))

            async def on_status(status: str) -> None:
                try:
                    await ws.send_json({"type": P.STATUS, "text": status, "session_id": session_id})
                except Exception:
                    pass

            channel_model = self._resolve_channel_model(client_id)
            active_model = channel_model or self.agent.model
            history_mode = getattr(active_model, "history_mode", None)
            try:
                self.sessions.bind_history_mode(client_id, session_id, history_mode)
            except ValueError as e:
                elog(
                    "session.history_mode_conflict",
                    client_id=client_id,
                    session_id=session_id,
                    history_mode=history_mode,
                    error=str(e),
                )
                await ws.send_json({"type": P.ERROR, "text": str(e), "session_id": session_id})
                return
            elog(
                "message.process_start",
                client_id=client_id,
                session_id=session_id,
                model_class=type(active_model).__name__,
                model_override=bool(channel_model),
            )
            response = await self.agent.run(
                message=text,
                user_id=client_id,
                session_id=session_id,
                on_status=on_status,
                model_override=channel_model,
            )

            from openagent.channels.base import parse_response_markers
            clean, attachments = parse_response_markers(response)
            att_list = [{"type": a.type, "path": a.path, "filename": a.filename} for a in attachments]
            response_meta = self.agent.last_response_meta(session_id)

            elog("message.response", client_id=client_id, session_id=session_id, length=len(clean))
            await ws.send_json({
                "type": P.RESPONSE,
                "text": clean,
                "session_id": session_id,
                "attachments": att_list or None,
                "model": response_meta.get("model"),
            })
        except Exception as e:
            logger.error("Process error for %s: %s", client_id, e)
            elog("message.error", client_id=client_id, session_id=session_id, error=str(e))
            try:
                await ws.send_json({"type": P.ERROR, "text": str(e), "session_id": session_id})
            except Exception:
                pass  # WS is dead — bridge timeout will handle cleanup
