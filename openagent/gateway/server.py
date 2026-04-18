"""Gateway server — the single public interface for OpenAgent.

Hosts a WebSocket endpoint for real-time chat and REST endpoints for
vault, config, and health. All clients (Electron app, CLI, bridges)
connect through this server.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import socket
from pathlib import Path
from typing import TYPE_CHECKING

from openagent.gateway import protocol as P
from openagent.gateway.commands import command_help_text
from openagent.gateway.sessions import SessionManager
from openagent.gateway.api import vault, config, health, logs, control, usage, providers, models, scheduled_tasks, mcps

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

        # Bound by AgentServer after Scheduler.start(); None when the scheduler
        # is disabled in config. Handlers in api/scheduled_tasks.py check this
        # and return 503 when it's absent.
        self._scheduler = None

        # Hot-reload state. Fingerprint = (mtime_ns, first 8 bytes of sha256).
        # Recomputed on each message in _process_message.
        self._config_fingerprint: tuple[int, bytes] | None = None
        self._reload_lock = asyncio.Lock()

    @staticmethod
    async def _safe_ws_send_json(ws, payload: dict) -> bool:
        """Best-effort websocket send that tolerates closing transports."""
        if ws is None or getattr(ws, "closed", False):
            return False
        try:
            await ws.send_json(payload)
            return True
        except Exception as e:
            if "closing transport" in str(e).lower():
                logger.debug("WS send skipped on closing transport")
                return False
            if getattr(ws, "closed", False):
                return False
            raise

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
                    elog("gateway.rest_error", level="error", exc_info=True,
                         path=request.path, method=request.method, error=str(exc))
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
        elog("gateway.start", host=self.host, port=self.port)

        # Write .port file for agent discovery
        self._write_port_file()

        # Seed the config fingerprint so the first message doesn't trigger
        # a spurious reload just because we've never sampled the file before.
        self._config_fingerprint = self._compute_config_fingerprint()

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
        app.router.add_get("/api/files", self._handle_files)
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
            ("GET", "/api/scheduled-tasks", scheduled_tasks.handle_list),
            ("POST", "/api/scheduled-tasks", scheduled_tasks.handle_create),
            ("GET", "/api/scheduled-tasks/{id}", scheduled_tasks.handle_get),
            ("PATCH", "/api/scheduled-tasks/{id}", scheduled_tasks.handle_update),
            ("DELETE", "/api/scheduled-tasks/{id}", scheduled_tasks.handle_delete),
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
            # DB-backed model catalog (the models table). Ordered before the
            # legacy yaml-providers CRUD above so that a request to a literal
            # ``/api/models/db`` path matches here instead of being captured
            # by the ``{name}`` catch-all.
            ("GET", "/api/models/available", models.handle_available_models),
            ("GET", "/api/models/db", models.handle_list_db),
            ("POST", "/api/models/db", models.handle_create_db),
            ("GET", "/api/models/db/{runtime_id:.+}", models.handle_get_db),
            ("PUT", "/api/models/db/{runtime_id:.+}", models.handle_update_db),
            ("DELETE", "/api/models/db/{runtime_id:.+}", models.handle_delete_db),
            ("POST", "/api/models/db/{runtime_id:.+}/enable", models.handle_enable_db),
            ("POST", "/api/models/db/{runtime_id:.+}/disable", models.handle_disable_db),
            # DB-backed MCP registry.
            ("GET", "/api/mcps", mcps.handle_list),
            ("POST", "/api/mcps", mcps.handle_create),
            ("GET", "/api/mcps/{name}", mcps.handle_get),
            ("PUT", "/api/mcps/{name}", mcps.handle_update),
            ("DELETE", "/api/mcps/{name}", mcps.handle_delete),
            ("POST", "/api/mcps/{name}/enable", mcps.handle_enable),
            ("POST", "/api/mcps/{name}/disable", mcps.handle_disable),
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

    def _compute_config_fingerprint(self) -> tuple[int, bytes] | None:
        """Return ``(mtime_ns, sha256[:8])`` or ``None`` if the file is missing."""
        if not self.config_path:
            return None
        try:
            st = os.stat(self.config_path, follow_symlinks=True)
        except FileNotFoundError:
            return None
        except OSError:
            return None
        try:
            with open(self.config_path, "rb") as f:
                digest = hashlib.sha256(f.read()).digest()[:8]
        except OSError:
            return None
        return (st.st_mtime_ns, digest)

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
        import os
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

        # On macOS ``tempfile.mkdtemp()`` returns a path under
        # ``/var/folders/...`` — a symlink to ``/private/var/folders/...``.
        # The reference ``@modelcontextprotocol/server-filesystem`` compares
        # tool-call paths to its allowlist by string-prefix against
        # realpaths, so a caller who hands the logical ``/var/folders/...``
        # path to ``read_text_file`` gets "Access denied — path outside
        # allowed directories" even though the realpath IS allowed. Resolve
        # here so the returned path matches what filesystem MCP will accept.
        path = os.path.realpath(path)
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
                elog("upload.transcribe_error", level="warning", filename=filename, error=str(e))

        elog("upload.saved", filename=filename, path=path, transcribed=bool(result.get("transcription")))
        return web.json_response(result)

    # ── File serving (agent → client) ──

    async def _handle_files(self, request):
        """GET /api/files?path=<abs>&token=<gateway-token>

        Serve a local file off the agent server's filesystem so remote
        clients (desktop app, CLI) can fetch attachments the agent
        emitted via ``[IMAGE:/path]`` / ``[FILE:/path]`` / ``[VOICE:/path]``
        / ``[VIDEO:/path]`` markers in a response.

        The agent runs with broad filesystem access and already returns
        the absolute path to the client in the WS ``response`` message's
        ``attachments`` array. For local installs the client can read
        the path directly; for remote installs (app on your laptop,
        agent on a VPS) this endpoint ferries the bytes over HTTP.

        **Authentication**: requires ``token`` query param matching the
        gateway token (same token clients use for WS auth). Without a
        configured token, reads are unauthenticated — this matches the
        existing ``/api/*`` endpoints which also rely on the gateway
        binding to localhost for single-user deploys.

        **Path safety**: we use ``os.path.realpath`` before checking
        ``isfile`` so symlinks resolve, and we reject paths that don't
        resolve to an actual file. Since the gateway token is required,
        we don't further restrict to specific directories — the agent
        has full FS access anyway, so any allow-listing would be
        theater against a caller who already holds the token.
        """
        from aiohttp import web
        import os

        if self.token:
            token = request.query.get("token") or (
                request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
                if request.headers.get("Authorization", "").startswith("Bearer ")
                else ""
            )
            if token != self.token:
                return web.Response(status=401, text="Unauthorized")

        path = request.query.get("path", "")
        if not path:
            return web.Response(status=400, text="path required")

        real = os.path.realpath(path)
        if not os.path.isfile(real):
            return web.Response(status=404, text="not found")

        # Let aiohttp pick the Content-Type from the extension and stream
        # the file from disk instead of buffering the whole thing in RAM.
        # Expose a sensible Content-Disposition so browsers download with
        # the original filename rather than a random hash.
        filename = os.path.basename(real)
        return web.FileResponse(
            real,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

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
                    await self._safe_ws_send_json(ws, {"type": P.ERROR, "text": "Invalid JSON"})
                    continue

                t = data.get("type", "")

                # Auth
                if t == P.AUTH:
                    if self.token and data.get("token") != self.token:
                        elog("auth.fail", client_id=data.get("client_id"))
                        await self._safe_ws_send_json(ws, {"type": P.AUTH_ERROR, "reason": "Invalid token"})
                        await ws.close()
                        return ws
                    client_id = data.get("client_id") or f"ws-{id(ws)}"
                    authed = True
                    self.clients[client_id] = ws
                    import openagent
                    elog("gateway.client_connect", client_id=client_id)
                    await self._safe_ws_send_json(ws, {
                        "type": P.AUTH_OK,
                        "agent_name": self.agent.name,
                        "version": getattr(openagent, "__version__", "?"),
                    })
                    continue

                if not authed:
                    await self._safe_ws_send_json(ws, {"type": P.AUTH_ERROR, "reason": "Not authenticated"})
                    continue
                if client_id is None:
                    client_id = f"ws-{id(ws)}"
                    self.clients[client_id] = ws

                # Ping
                if t == P.PING:
                    await self._safe_ws_send_json(ws, {"type": P.PONG})

                # Command
                elif t == P.COMMAND:
                    cmd_name = data.get("name", "")
                    cmd_sid = data.get("session_id")
                    elog(
                        "command.received",
                        client_id=client_id,
                        name=cmd_name,
                        session_id=cmd_sid,
                    )
                    await self._handle_command(ws, client_id, cmd_name, cmd_sid)

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
                            await self._safe_ws_send_json(
                                ws,
                                {"type": P.ERROR, "text": "Too many messages queued. Please wait.", "session_id": sid},
                            )
                        elif pos > 0:
                            elog("message.queued", client_id=client_id, session_id=sid, position=pos)
                            await self._safe_ws_send_json(ws, {"type": P.QUEUED, "position": pos})

        except Exception as e:
            elog("gateway.ws_error", level="error", client_id=client_id, error=str(e))
        finally:
            if client_id and client_id in self.clients:
                del self.clients[client_id]
                elog("gateway.client_disconnect", client_id=client_id)
        return ws

    async def _handle_command(
        self, ws, client_id: str, name: str, session_id: str | None = None
    ) -> None:
        """Dispatch a WS command.

        When ``session_id`` is provided, scope-sensitive commands (``stop``,
        ``clear``, ``new``, ``reset``) act only on that conversation. Bridges
        that multiplex many users onto one ``client_id`` (telegram, discord,
        whatsapp) and UI clients that host many independent chat tabs on one
        websocket (desktop app) MUST pass this — otherwise a ``/clear`` from
        one user/tab wipes everyone else on the same ``client_id``.
        """
        sm = self.sessions
        if name in ("new", "reset", "clear"):
            # /new, /reset, /clear: full wipe — stop anything running, drop
            # the queue, AND forget provider-native resume state. Scoped to
            # ``session_id`` when given; falls back to client-wide wipe
            # otherwise.
            if session_id:
                stopped = sm.stop_current(client_id, session_id=session_id)
                cleared = sm.clear_queue_for_session(client_id, session_id)
                forgotten = await self._forget_one_session(session_id)
            else:
                stopped = sm.stop_current(client_id)
                cleared = sm.clear_queue(client_id)
                forgotten = await self._forget_all_client_sessions(client_id)
            fresh_sid = sm.create_session(client_id)
            parts = []
            if stopped:
                parts.append("stopped current operation")
            if cleared:
                parts.append(f"cleared {cleared} queued message{'s' if cleared != 1 else ''}")
            if forgotten:
                parts.append(f"forgot {forgotten} prior conversation{'s' if forgotten != 1 else ''}")
            parts.append(f"fresh session: {fresh_sid[-8:]}")
            text = ". ".join(p.capitalize() if i == 0 else p for i, p in enumerate(parts)) + "."
        elif name == "stop":
            if session_id:
                stopped = sm.stop_current(client_id, session_id=session_id)
                cleared = sm.clear_queue_for_session(client_id, session_id)
            else:
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
        await self._safe_ws_send_json(ws, {"type": P.COMMAND_RESULT, "text": text})

    # Prefix used by each bridge when naming its per-user session ids.
    # Used ONLY for the legacy, unscoped fallback path of /clear (no
    # ``session_id`` in the command payload). Keep in sync with the
    # bridge sources:
    #   - bridges/telegram.py: ``f"tg:{uid}"``
    #   - bridges/discord.py: ``f"dc:{uid}"``
    #   - bridges/whatsapp.py: ``f"wa:{uid}"``
    _BRIDGE_SESSION_PREFIXES: dict[str, str] = {
        "bridge:telegram": "tg:",
        "bridge:discord": "dc:",
        "bridge:whatsapp": "wa:",
    }

    async def _forget_one_session(self, session_id: str) -> int:
        """Forget just one session. Returns 1 on success, 0 on failure."""
        try:
            await self.agent.forget_session(session_id)
        except Exception as e:
            elog("session.forget_one", session_id=session_id, forgotten=0, error=str(e))
            return 0
        elog("session.forget_one", session_id=session_id, forgotten=1)
        return 1

    async def _forget_all_client_sessions(self, client_id: str) -> int:
        """Erase provider-native resume state for every session tied to ``client_id``.

        Uses two sources because SessionManager is RAM-only and starts empty
        after every restart: any session attached before the latest restart
        would otherwise be invisible here, and /clear would silently keep
        the prior transcript alive (the model rehydrates ``_sdk_sessions``
        from sqlite on startup and ``--resume`` keeps reconstituting it).

        Sources:
          1. ``SessionManager.list_sessions`` — what the gateway has seen
             since the current process started.
          2. The model's own ``known_session_ids()`` filtered by the bridge
             prefix for this client (``tg:`` for telegram, ``discord:`` for
             discord, ``whatsapp:`` for whatsapp). Catches any resume state
             that outlived the restart.

        Returns the number of sessions whose resume state was dropped.
        """
        sids: set[str] = set(self.sessions.list_sessions(client_id))
        prefix = self._BRIDGE_SESSION_PREFIXES.get(client_id)
        if prefix:
            for sid in self.agent.known_model_session_ids():
                if sid.startswith(prefix):
                    sids.add(sid)
        forgotten = 0
        for sid in sids:
            try:
                await self.agent.forget_session(sid)
                forgotten += 1
            except Exception as e:
                logger.debug("forget_session(%s) failed: %s", sid, e)
        elog(
            "session.forget_all",
            client_id=client_id,
            forgotten=forgotten,
            total=len(sids),
        )
        return forgotten

    async def _maybe_reload_agent_model(self) -> None:
        """If the YAML fingerprint changed since the last check, rebuild the
        primary agent model and swap it onto the agent.

        Scope: only the ``model:`` and ``providers:`` sections drive
        which model we rebuild. Other sections (channels, mcp, scheduler,
        system_prompt) are not reread here and still require a restart.

        Safe to call from concurrent messages — the lock serializes the
        actual rebuild.
        """
        if not self.config_path or not self.agent._initialized:
            return

        current = self._compute_config_fingerprint()
        if current is None or current == self._config_fingerprint:
            return

        async with self._reload_lock:
            # Double-check under lock: another task may have just reloaded.
            current = self._compute_config_fingerprint()
            if current is None or current == self._config_fingerprint:
                return

            try:
                from openagent.core.config import load_config
                new_config = load_config(self.config_path)
            except Exception as e:
                elog("gateway.config_reload_parse_error", level="warning", error=str(e))
                return

            try:
                from openagent.models.runtime import create_model_from_config, wire_model_runtime
                new_model = create_model_from_config(new_config)
                wire_model_runtime(new_model, db=self.agent._db, mcp_pool=self.agent._mcp)
            except Exception as e:
                elog("gateway.config_reload_build_error", level="warning", error=str(e))
                return

            old_model, drain_event = self.agent.swap_model(new_model)

            # Invalidate per-channel override cache — specs may resolve
            # to different provider configs after the reload.
            self._model_cache.clear()
            self._config_fingerprint = current

            elog(
                "gateway.config_reload",
                old_class=type(old_model).__name__ if old_model else None,
                new_class=type(new_model).__name__,
            )

            if old_model is not None and old_model is not new_model:
                asyncio.create_task(self._drain_and_shutdown_model(old_model, drain_event))

    async def _drain_and_shutdown_model(self, model, drain_event) -> None:
        """Wait for *model*'s last in-flight call to finish, then shut it down."""
        try:
            await drain_event.wait()
        except Exception:
            pass
        try:
            shutdown = getattr(model, "shutdown", None)
            if callable(shutdown):
                await shutdown()
        except Exception as e:
            logger.debug("Drain shutdown error (ignored): %s", e)
        finally:
            self.agent._unregister_runtime_model(model)
            elog("gateway.config_reload_drained", model_class=type(model).__name__)

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
            # Debug-level: channel override is a fallback, failing is non-fatal.
            elog("channel.model_override_error", level="debug",
                 client_id=client_id, channel=channel, error=str(e))
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
            await self._maybe_reload_agent_model()

            # Hot-reload MCPs/models if the registry tables changed, and
            # get the enabled-model count for the rejection gate — one
            # round-trip to the DB. ``-1`` means no DB is wired.
            try:
                _, enabled_count = await self.agent.refresh_registries()
            except Exception as e:  # noqa: BLE001 — inner method already guards
                elog("hot_reload.error", error=str(e))
                enabled_count = -1
            if enabled_count == 0:
                await self._safe_ws_send_json(ws, {
                    "type": P.ERROR,
                    "text": (
                        "No models are enabled. Add one via /models or ask "
                        "the agent to add an openai/anthropic/google model."
                    ),
                    "session_id": session_id,
                })
                elog("session.rejected_no_models", session_id=session_id)
                return

            async def on_status(status: str) -> None:
                try:
                    await self._safe_ws_send_json(
                        ws,
                        {"type": P.STATUS, "text": status, "session_id": session_id},
                    )
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
                await self._safe_ws_send_json(ws, {"type": P.ERROR, "text": str(e), "session_id": session_id})
                return
            elog(
                "message.process_start",
                client_id=client_id,
                session_id=session_id,
                model_class=type(active_model).__name__,
                model_override=bool(channel_model),
            )
            try:
                response = await self.agent.run(
                    message=text,
                    user_id=client_id,
                    session_id=session_id,
                    on_status=on_status,
                    model_override=channel_model,
                )
            except asyncio.CancelledError:
                # Server is shutting down (restart for config update, launchd
                # stop, etc.). Send a user-facing message so the client knows
                # it wasn't a silent failure, log the cancellation, and
                # re-raise so the surrounding cancel scope propagates.
                elog(
                    "message.cancelled",
                    client_id=client_id,
                    session_id=session_id,
                    reason="server_shutdown",
                )
                try:
                    await self._safe_ws_send_json(ws, {
                        "type": P.ERROR,
                        "text": "Server is restarting, please try your message again in a moment.",
                        "session_id": session_id,
                    })
                except Exception:
                    pass  # WS may already be half-closed
                raise

            from openagent.channels.base import parse_response_markers
            clean, attachments = parse_response_markers(response)
            att_list = [{"type": a.type, "path": a.path, "filename": a.filename} for a in attachments]
            response_meta = self.agent.last_response_meta(session_id)

            elog("message.response", client_id=client_id, session_id=session_id, length=len(clean))
            await self._safe_ws_send_json(ws, {
                "type": P.RESPONSE,
                "text": clean,
                "session_id": session_id,
                "attachments": att_list or None,
                "model": response_meta.get("model"),
            })
        except asyncio.CancelledError:
            raise  # already handled above or a separate cancel scope
        except Exception as e:
            elog(
                "message.error",
                level="error",
                client_id=client_id,
                session_id=session_id,
                error_type=type(e).__name__,
                error=str(e) or repr(e),
            )
            try:
                await self._safe_ws_send_json(
                    ws,
                    {"type": P.ERROR, "text": str(e) or type(e).__name__, "session_id": session_id},
                )
            except Exception:
                pass  # WS is dead — bridge timeout will handle cleanup
