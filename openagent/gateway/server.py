"""Gateway server — the single public interface for OpenAgent.

Hosts a WebSocket endpoint for real-time chat and REST endpoints for
vault, config, and health. All clients (Electron app, CLI, bridges)
connect through this server.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from openagent.gateway import protocol as P
from openagent.gateway.commands import command_help_text
from openagent.gateway.sessions import SessionManager
from openagent.gateway.api import vault, config, health, logs, control, usage, providers, models, scheduled_tasks, workflow_tasks, mcps, marketplace, sessions as sessions_api, system as system_api
from openagent.network import peers as peers_api
from openagent.network.auth.middleware import make_auth_middleware
from openagent.network.transport.aiohttp_iroh_site import IrohSite

if TYPE_CHECKING:
    from openagent.core.agent import Agent
    from openagent.network.state import NetworkState

from openagent.core.logging import elog

logger = logging.getLogger(__name__)


@dataclass
class _StreamHolder:
    """A live stream session attached to a client WS."""

    session: "StreamSession"
    channel: "RealtimeChannel"


class Gateway:
    """aiohttp gateway tunneled over Iroh — handle@network auth, no IPs/ports.

    The legacy ``host:port + token`` constructor is gone. The Gateway now
    binds to the Iroh endpoint owned by ``NetworkState`` and authenticates
    every inbound stream via the device-cert middleware.
    """

    def __init__(
        self,
        agent: Agent,
        network_state: NetworkState,
        vault_path: str | None = None,
        config_path: str | None = None,
        stop_event: asyncio.Event | None = None,
    ):
        self.agent = agent
        self._network_state = network_state
        self.vault_path = vault_path
        self.config_path = config_path
        self._stop_event = stop_event
        self.sessions = SessionManager(agent_name=agent.name)
        self.clients: dict[str, object] = {}  # client_id → WebSocketResponse
        self._runner = None
        self._site: IrohSite | None = None

        # Per (client_id, session_id) StreamSession + RealtimeChannel
        # pair. Created on demand from inbound stream frames; closed on
        # ``session_close`` or WS drop.
        self._stream_sessions: dict[tuple[str, str], _StreamHolder] = {}

        # Cross-platform host-telemetry sampler (psutil). Filled in by
        # ``start()``; the /api/system handler and the broadcast loop
        # both read off this single instance so the network-rate
        # deltas come from one continuous time series.
        self._system_telemetry: system_api.SystemTelemetry | None = None
        self._system_broadcast_task: asyncio.Task | None = None

        # Bound by AgentServer after Scheduler.start(); None when the agent
        # was constructed without a DB. Handlers in api/scheduled_tasks.py
        # check this and return 503 when it's absent.
        self._scheduler = None

        # Bound by AgentServer.start() once bridges are instantiated. Used by
        # control.request_restart so /restart can proactively ACK pending
        # Telegram updates before the restart fires (so a queued /restart
        # can't replay on the next boot and produce a crash loop).
        self._bridges: list = []

        # Per-section live-reaction hooks, populated by AgentServer when it
        # spins up the scheduler. ``config.handle_patch`` calls
        # ``on_config_change(section, patch)`` after writing the yaml so
        # toggles (dream_mode, manager_review, auto_update) take effect
        # without a restart. Keyed by config section name.
        self._config_change_callbacks: dict[
            str, Callable[[dict], Awaitable[None]]
        ] = {}

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

    async def broadcast(self, payload: dict) -> None:
        """Best-effort fan-out to every authenticated client.

        Resource-change pings travel here so the desktop app's list
        screens can refetch without polling. Never raises — a single
        flaky client must not interrupt the producer (a REST handler
        or the scheduler tick loop).
        """
        if not self.clients:
            return
        # Snapshot keys: a slow client closing during the loop would
        # otherwise mutate self.clients underneath us.
        for client_id, ws in list(self.clients.items()):
            try:
                await self._safe_ws_send_json(ws, payload)
            except Exception as e:  # noqa: BLE001
                logger.debug("broadcast skipped for %s: %s", client_id, e)

    async def broadcast_resource(
        self,
        resource: str,
        action: str,
        id: str | None = None,
    ) -> None:
        """Emit a ``resource_event`` to all connected clients."""
        payload: dict[str, Any] = {
            "type": P.RESOURCE_EVENT,
            "resource": resource,
            "action": action,
        }
        if id is not None:
            payload["id"] = id
        await self.broadcast(payload)

    def broadcast_resource_sync(
        self,
        resource: str,
        action: str,
        id: str | None = None,
    ) -> None:
        """Schedule a resource broadcast from sync context.

        Used by the Scheduler tick (it's running inside an asyncio task
        already, so ``create_task`` works) and by any other producer
        that doesn't want to ``await``. Silently no-ops outside an
        event loop so unit tests that drive a Scheduler without a live
        gateway aren't forced to mock everything.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self.broadcast_resource(resource, action, id))

    async def on_config_change(self, section: str, patch: dict) -> None:
        """Notify a registered side-effect that a yaml section changed.

        ``AgentServer`` registers closures here for ``dream_mode``,
        ``manager_review`` and ``auto_update`` so toggles flow straight
        into the scheduler without a restart.
        """
        cb = self._config_change_callbacks.get(section)
        if cb is None:
            return
        try:
            await cb(patch)
        except Exception as e:  # noqa: BLE001
            logger.warning("config-change callback for %r failed: %s", section, e)

    def _prepare_iroh_site(self) -> None:
        """Register the gateway ALPN handler on the IrohNode.

        iroh-py 0.35 bakes the protocol handler dict into NodeOptions
        at node-construction time, so this MUST run before
        ``IrohNode.start``. The AgentServer calls this between
        building the NetworkState and starting it. We don't construct
        the IrohSite here yet — that needs the aiohttp runner — but
        we DO register the per-ALPN callback that the IrohSite will
        eventually own. Storing the unbound method works because
        register_handler captures it as a callable.
        """
        # The actual IrohSite is created in ``start`` once the runner
        # exists; until then we register a thin shim that defers to
        # ``self._site`` when it lands.
        async def _gateway_handler(connection):
            site = self._site
            if site is None:
                # Site hasn't been wired yet — drain and drop.
                try:
                    connection.close(0, b"gateway not ready")
                except Exception:
                    pass
                return
            await site._handle_stream(connection)

        from openagent.network.iroh_node import NetworkAlpn as _Alpn
        self._network_state.iroh_node.register_handler(_Alpn.GATEWAY, _gateway_handler)

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

        # Order matters: cors first (handles OPTIONS preflight without
        # going through auth), then auth (every other request needs a
        # valid device cert). Both run for every route added below.
        auth_middleware = make_auth_middleware(self._network_state.auth_state)
        app = web.Application(middlewares=[cors, auth_middleware])
        app["gateway"] = self  # accessible in handlers via request.app["gateway"]
        self._register_routes(app)

        runner = web.AppRunner(app)
        await runner.setup()
        self._runner = runner

        # The IrohSite was built in ``__init__`` (it registers the ALPN
        # handler with the IrohNode, which has to happen *before*
        # ``IrohNode.start``). Here we just attach it to the runner
        # and start the lifecycle. The runner-aware bits of BaseSite
        # need a constructed runner, hence the deferred wiring.
        self._site = IrohSite(runner, self._network_state.iroh_node)
        await self._site.start()
        node_id = await self._network_state.node_id()
        elog(
            "gateway.start",
            transport="iroh",
            node_id=node_id,
            network=self._network_state.network_name,
            role=self._network_state.role,
        )

        # Spin up host telemetry. Broadcast loop primes psutil's CPU
        # baseline on first tick, then emits one ``system_snapshot``
        # every ``BROADCAST_INTERVAL_S`` seconds — but only when at
        # least one client is listening, so an idle gateway never
        # iterates processes for nobody.
        self._system_telemetry = system_api.SystemTelemetry()
        self._system_broadcast_task = asyncio.create_task(
            self._system_broadcast_loop(), name="gateway-system-broadcast"
        )

    async def stop(self) -> None:
        await self.sessions.shutdown()
        if self._system_broadcast_task is not None:
            self._system_broadcast_task.cancel()
            try:
                await self._system_broadcast_task
            except (asyncio.CancelledError, Exception):
                pass
            self._system_broadcast_task = None
        if self._site is not None:
            try:
                await self._site.stop()
            except Exception as e:  # noqa: BLE001
                logger.debug("iroh site stop failed: %s", e)
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self.clients.clear()

    async def _system_broadcast_loop(self) -> None:
        """Push a ``system_snapshot`` to all clients on a fixed cadence.

        Skips the iter-processes call when ``self.clients`` is empty —
        an idle gateway shouldn't burn CPU sampling its own host. Errors
        are logged and swallowed so a transient psutil hiccup (e.g. a
        process that vanished mid-iteration) doesn't kill the loop.
        """
        interval = system_api.BROADCAST_INTERVAL_S
        telemetry = self._system_telemetry
        assert telemetry is not None
        while True:
            try:
                await asyncio.sleep(interval)
                if not self.clients:
                    continue
                snap = await telemetry.snapshot()
                await self.broadcast({
                    "type": P.SYSTEM_SNAPSHOT,
                    "snapshot": snap,
                })
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.debug("system broadcast skipped: %s", e)

    def _register_routes(self, app) -> None:
        """Register the gateway WebSocket endpoint and REST API routes."""
        app.router.add_get("/ws", self._handle_ws)
        app.router.add_post("/api/upload", self._handle_upload)
        app.router.add_get("/api/files", self._handle_files)
        app.router.add_get("/api/agent-info", self._handle_agent_info)
        app.router.add_post("/api/tts/synthesize", self._handle_tts_synthesize)
        app.router.add_post("/api/stt/transcribe", self._handle_stt_transcribe)

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
            # Workflow engine (n8n-style multi-block pipelines). Same
            # scheduler 503 invariant as scheduled-tasks — handlers
            # return 503 when no Scheduler is attached.
            ("GET", "/api/workflows", workflow_tasks.handle_list),
            ("POST", "/api/workflows", workflow_tasks.handle_create),
            ("GET", "/api/workflows/{id}", workflow_tasks.handle_get),
            ("PATCH", "/api/workflows/{id}", workflow_tasks.handle_update),
            ("DELETE", "/api/workflows/{id}", workflow_tasks.handle_delete),
            ("POST", "/api/workflows/{id}/run", workflow_tasks.handle_run),
            ("GET", "/api/workflows/{id}/runs", workflow_tasks.handle_runs_list),
            ("GET", "/api/workflows/{id}/stats", workflow_tasks.handle_stats),
            ("GET", "/api/workflow-runs/{run_id}", workflow_tasks.handle_run_get),
            ("GET", "/api/workflow-block-types", workflow_tasks.handle_block_types),
            ("GET", "/api/mcp-tools", workflow_tasks.handle_mcp_tools),
            ("GET", "/api/cron/describe", workflow_tasks.handle_cron_describe),
            ("GET", "/api/logs", logs.handle_get),
            ("DELETE", "/api/logs", logs.handle_delete),
            ("GET", "/api/usage", usage.handle_get),
            ("GET", "/api/usage/daily", usage.handle_daily),
            ("GET", "/api/usage/pricing", usage.handle_pricing),
            # DB-backed provider CRUD. The ``providers`` SQLite table is
            # canonical. Rows are keyed on surrogate integer ``id`` so the
            # same vendor can coexist under both frameworks.
            ("GET", "/api/providers", providers.handle_list),
            ("POST", "/api/providers", providers.handle_create),
            ("GET", r"/api/providers/{id:\d+}", providers.handle_get),
            ("PUT", r"/api/providers/{id:\d+}", providers.handle_update),
            ("DELETE", r"/api/providers/{id:\d+}", providers.handle_delete),
            ("POST", r"/api/providers/{id:\d+}/enable", providers.handle_enable),
            ("POST", r"/api/providers/{id:\d+}/disable", providers.handle_disable),
            ("POST", r"/api/providers/{id:\d+}/test", providers.handle_test),
            # Models. ``/api/models`` is the DB-backed catalog.
            ("GET", "/api/models/catalog", models.handle_catalog),
            ("GET", "/api/models/providers", models.handle_available_providers),
            ("GET", "/api/models/available", models.handle_available_models),
            ("GET", "/api/models", models.handle_list_db),
            ("POST", "/api/models", models.handle_create_db),
            ("GET", r"/api/models/{id:\d+}", models.handle_get_db),
            ("PUT", r"/api/models/{id:\d+}", models.handle_update_db),
            ("DELETE", r"/api/models/{id:\d+}", models.handle_delete_db),
            ("POST", r"/api/models/{id:\d+}/enable", models.handle_enable_db),
            ("POST", r"/api/models/{id:\d+}/disable", models.handle_disable_db),
            # DB-backed MCP registry.
            ("GET", "/api/mcps", mcps.handle_list),
            ("POST", "/api/mcps", mcps.handle_create),
            ("GET", "/api/mcps/{name}", mcps.handle_get),
            ("PUT", "/api/mcps/{name}", mcps.handle_update),
            ("DELETE", "/api/mcps/{name}", mcps.handle_delete),
            ("POST", "/api/mcps/{name}/enable", mcps.handle_enable),
            ("POST", "/api/mcps/{name}/disable", mcps.handle_disable),
            # MCP marketplace — proxy + installer for the official registry.
            ("GET", "/api/marketplace/search", marketplace.handle_search),
            ("GET", "/api/marketplace/servers", marketplace.handle_server_detail),
            ("POST", "/api/marketplace/install", marketplace.handle_install),
            # Per-session model pin.
            ("GET", "/api/sessions/{session_id}/model", sessions_api.handle_get),
            ("PUT", "/api/sessions/{session_id}/model", sessions_api.handle_pin),
            ("DELETE", "/api/sessions/{session_id}/model", sessions_api.handle_unpin),
            ("POST", "/api/update", control.handle_update),
            ("POST", "/api/restart", control.handle_restart),
            # Cross-platform host telemetry (psutil-backed). Live
            # updates flow over the WS as ``system_snapshot`` events;
            # this REST handler exists for the initial paint and any
            # client that doesn't speak the WS feed.
            ("GET", "/api/system", system_api.handle_get),
            # Network membership: peer networks (federation), and
            # info about this agent's home network. ``network/info``
            # works on both coordinator and member agents; ``peers``
            # is per-agent so federation state can differ between
            # peers in the same home network.
            ("GET", "/api/network/info", self._handle_network_info),
            ("GET", "/api/peers", peers_api.handle_list),
            ("POST", "/api/peers", peers_api.handle_create),
            ("DELETE", "/api/peers/{network_id}", peers_api.handle_delete),
            ("GET", "/api/peers/{network_id}/agents", peers_api.handle_list_agents),
        )
        for method, path, handler in routes:
            app.router.add_route(method, path, handler)
        app.router.add_route("OPTIONS", "/{path:.*}", self._handle_options)

    def runtime_info(self) -> dict:
        """Return shared gateway/agent metadata exposed by REST endpoints."""
        import openagent
        from openagent.core.paths import get_agent_dir

        agent_dir = get_agent_dir()
        return {
            "agent": self.agent.name,
            "agent_dir": str(agent_dir) if agent_dir else None,
            "node_id": self._network_state.identity.public_hex,
            "network": self._network_state.network_name,
            "role": self._network_state.role,
            "version": getattr(openagent, "__version__", "?"),
        }

    async def _handle_agent_info(self, request):
        """GET /api/agent-info — agent name, network, node_id, version."""
        from aiohttp import web

        info = self.runtime_info()
        return web.json_response({
            "name": info["agent"],
            "agent_dir": info["agent_dir"],
            "node_id": info["node_id"],
            "network": info["network"],
            "role": info["role"],
            "version": info["version"],
        })

    async def _handle_network_info(self, request):
        """GET /api/network/info — describe this agent's network membership."""
        from aiohttp import web

        ns = self._network_state
        return web.json_response({
            "role": ns.role,
            "network_id": ns.network_id,
            "network_name": ns.network_name,
            "node_id": ns.identity.public_hex,
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
            # Hint downstream that the next chat message originated from
            # voice — clients tag the next ``text_final`` with
            # ``source="stt"`` so the StreamSession applies the mirror-
            # modality rule (instant barge-in + spoken reply when TTS
            # is configured) regardless of the session-level speak
            # toggle.
            result["transcribed_from_voice"] = True
            # Optional ISO-639-1 hint from the client (?lang=it). Auto-
            # detect on small Whisper models is unreliable for short
            # utterances and has misidentified Italian as Cyrillic.
            lang = (request.query.get("lang") or "").strip().lower() or None
            try:
                from openagent.channels.voice import transcribe
                text = await transcribe(
                    path,
                    db=getattr(self.agent, "db", None),
                    language=lang,
                )
                if text:
                    result["transcription"] = text
                    elog(
                        "upload.transcribed",
                        filename=filename, chars=len(text),
                        language=lang or "auto",
                    )
                else:
                    elog(
                        "upload.transcribed_empty",
                        level="warning",
                        filename=filename, language=lang or "auto",
                    )
            except Exception as e:
                elog(
                    "upload.transcribe_error",
                    level="warning",
                    filename=filename, error=str(e), language=lang or "auto",
                )

        elog("upload.saved", filename=filename, path=path, transcribed=bool(result.get("transcription")))
        return web.json_response(result)

    def _check_bearer_token(self, request) -> bool:
        """Legacy compat — returns True iff the auth middleware approved this request.

        The ``Gateway.token`` field and inline token check are gone;
        every authed handler now sees ``request['device_cert']`` set
        by ``make_auth_middleware``. Some pre-existing handlers still
        call this helper for clarity; it just confirms the middleware
        attached an identity.
        """
        return request.get("device_cert") is not None

    async def _handle_stt_transcribe(self, request):
        """POST /api/stt/transcribe — transcribe an audio upload.

        Accepts multipart form ``file`` and returns ``{text}``. Used by
        bridges (Telegram, Discord, WhatsApp) so they all share the
        same DB-configured STT route — no per-bridge Whisper install.

        Resolution order matches :func:`channels.voice.transcribe`:
        DB-configured LiteLLM row → local faster-whisper → OpenAI
        Whisper API (env-driven) → ``404`` if every backend fails.
        """
        from aiohttp import web
        from openagent.channels.voice import transcribe, is_audio_file
        import tempfile

        # Auth handled by middleware; cert is on request["device_cert"].
        reader = await request.multipart()
        field = await reader.next()
        if not field:
            return web.json_response({"error": "no file"}, status=400)
        filename = field.filename or "upload"
        if not is_audio_file(filename):
            return web.json_response({"error": "not an audio file"}, status=400)
        tmp = tempfile.mkdtemp(prefix="oa_stt_")
        path = f"{tmp}/{filename}"
        with open(path, "wb") as f:
            while True:
                chunk = await field.read_chunk()
                if not chunk:
                    break
                f.write(chunk)
        db = getattr(self.agent, "db", None)
        lang = (request.query.get("lang") or "").strip().lower() or None
        try:
            text = await transcribe(path, db=db, language=lang)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"error": str(e)}, status=500)
        if not text:
            return web.json_response({"error": "no STT backend produced text"}, status=404)
        return web.json_response({"text": text})

    async def _handle_tts_synthesize(self, request):
        """POST /api/tts/synthesize — synthesise text to audio bytes for bridges.

        Body: ``{"text": "..."}`` → audio bytes (MIME varies by vendor,
        usually ``audio/mpeg``); ``404`` if no TTS provider configured,
        ``400`` on missing text. Bridges (Telegram, Discord, WhatsApp)
        call this so the ElevenLabs/OpenAI/Azure key only lives in the
        SQLite providers table next to the gateway, never in each
        bridge process.
        """
        from aiohttp import web
        from openagent.channels.tts import resolve_tts_provider, synthesize_full

        # Auth handled by middleware; cert is on request["device_cert"].
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        text = (body.get("text") or "").strip()
        if not text:
            return web.json_response({"error": "text is required"}, status=400)

        db = getattr(self.agent, "db", None)
        cfg = await resolve_tts_provider(db)
        if cfg is None:
            return web.json_response({"error": "no TTS provider configured"}, status=404)

        audio = await synthesize_full(text, cfg)
        if not audio:
            return web.json_response({"error": "synthesis failed"}, status=502)
        return web.Response(body=audio, content_type="audio/mpeg")

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

        # Auth is enforced by ``make_auth_middleware`` for every
        # ``/api/*`` route — by the time the handler runs the cert is
        # already verified. Defensive sanity check kept so a misrouted
        # request without a cert can't bypass FS access.
        if request.get("device_cert") is None:
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

        # The auth middleware already verified the device cert before
        # the WS upgrade ran — by the time we get here, the request
        # carries a valid identity. The legacy ``token`` AUTH frame is
        # gone; the first AUTH frame just carries an optional
        # ``client_kind`` for telemetry.
        cert = request.get("device_cert")
        if cert is None:
            return web.Response(status=401, text="Unauthorized")

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # ``client_id`` is bound to the device's pubkey. This means a
        # reconnect from the same device picks up its existing
        # StreamSessions; a different device gets a fresh slot. The
        # client cannot pick its own ID anymore — that was a footgun
        # the old protocol allowed (one client could clobber another's
        # sessions by guessing the ID).
        client_id: str = cert.device_pubkey_hex
        old_ws = self.clients.get(client_id)
        self.clients[client_id] = ws
        if old_ws is not None and old_ws is not ws:
            self._adopt_sessions_to_ws(client_id, ws)
            elog("gateway.client_reconnect", client_id=client_id)
            if not getattr(old_ws, "closed", True):
                try:
                    await old_ws.close()
                except Exception as e:  # noqa: BLE001
                    logger.debug("old ws close failed: %s", e)

        # Greet the client. Old clients sent an AUTH frame and waited
        # for AUTH_OK; the new wire skips the AUTH frame but keeps the
        # AUTH_OK greeting for backward-compat in client code that
        # waits for it as a "ready" signal.
        import openagent
        elog("gateway.client_connect", client_id=client_id, handle=cert.handle)
        await self._safe_ws_send_json(ws, {
            "type": P.AUTH_OK,
            "agent_name": self.agent.name,
            "version": getattr(openagent, "__version__", "?"),
            "handle": cert.handle,
            # Human-readable name (``agent-personal``) so the renderer
            # can pass it back through as the ``network`` segment of
            # ``handle@network`` on re-login. ``network_id`` is the
            # internal UUID; older clients that only kept ``network``
            # would otherwise stash the UUID and break next sign-in.
            "network": self._network_state.network_name,
            "network_id": cert.network_id,
        })

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

                # Legacy AUTH frame: ignored — the middleware already
                # authenticated us. Tolerate it for old clients that
                # still send one before their first session_open.
                if t == P.AUTH:
                    continue

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

                # Stream protocol — typed event frames. Every text/voice/
                # video/attachment message goes through here now (the
                # legacy ``MESSAGE`` handler was retired once bridges,
                # the universal app and the CLI all migrated to
                # ``session_open`` + ``text_final``). Decoded via
                # :mod:`openagent.stream.wire` and dispatched to a
                # per-(client, session) :class:`StreamSession`.
                elif t in (
                    P.SESSION_OPEN, P.SESSION_CLOSE,
                    P.TEXT_DELTA_IN, P.TEXT_FINAL_IN,
                    P.AUDIO_CHUNK_IN, P.AUDIO_END_IN,
                    P.VIDEO_FRAME_IN, P.ATTACHMENT_IN, P.INTERRUPT,
                ):
                    await self._handle_stream_frame(ws, client_id, data)

        except Exception as e:
            elog("gateway.ws_error", level="error", client_id=client_id, error=str(e))
        finally:
            # Identity-aware cleanup: a reconnected ws may have already
            # taken this client_id's slot via ``_adopt_sessions_to_ws``.
            # Touching ``clients[client_id]`` or
            # ``_close_stream_sessions_for`` here would tear down the
            # *new* connection's live sessions and leave its UI stuck.
            if client_id and self.clients.get(client_id) is ws:
                del self.clients[client_id]
                elog("gateway.client_disconnect", client_id=client_id)
                # Tear down any stream sessions belonging to this client
                # so the agent's per-session resources (claude-cli
                # subprocesses, agno session rows) get a clean release.
                await self._close_stream_sessions_for(client_id)
            elif client_id:
                elog(
                    "gateway.client_replaced",
                    client_id=client_id,
                )
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

    async def _handle_stream_frame(
        self, ws, client_id: str, frame: dict
    ) -> None:
        """Decode a stream-protocol wire frame and dispatch into the
        matching :class:`StreamSession`.

        Sessions are created on demand on the first frame for a given
        ``(client_id, session_id)`` pair. ``session_close`` (or the
        client WS dropping) tears them down.
        """
        from openagent.stream.session import StreamSession
        from openagent.stream.channel import RealtimeChannel
        from openagent.stream.wire import wire_to_event
        from openagent.stream.events import SessionClose, SessionOpen

        session_id = (frame.get("session_id") or "default").strip() or "default"
        sid = self.sessions.get_or_create_session(client_id, session_id)
        key = (client_id, sid)
        evt = wire_to_event(frame)
        if evt is None:
            return

        if isinstance(evt, SessionClose):
            await self._close_stream_session(key)
            return

        holder = self._stream_sessions.get(key)
        if holder is None:
            language: str | None = None
            profile = "realtime"
            # ``None`` lets ``StreamSession`` pick its own default
            # (currently 500 ms — the OpenAI-Realtime-style merged-burst
            # UX). The wire decoder hands us ``None`` whenever the client
            # didn't carry the field, and an explicit ``0`` whenever the
            # client opted out.
            coalesce_window_ms: int | None = None
            speak_enabled = True
            if isinstance(evt, SessionOpen):
                language = evt.language
                profile = evt.profile
                coalesce_window_ms = evt.coalesce_window_ms
                speak_enabled = bool(evt.speak)
            session = StreamSession(
                self.agent,
                client_id=client_id,
                session_id=sid,
                profile=profile,
                language=language,
                coalesce_window_ms=coalesce_window_ms,
                speak_enabled=speak_enabled,
            )
            # Install gateway hooks: pre-dispatch enforces the same
            # "no enabled models" + "history-mode binding" guards the
            # legacy MESSAGE handler did per turn; post-turn fires the
            # same MCP resource broadcasts. Both close over ``self``
            # and ``client_id`` / ``sid`` so each session gets its own
            # scoped pair.
            session.pre_dispatch_hook = self._make_stream_pre_dispatch_hook(
                client_id, sid,
            )
            session.post_turn_hook = self._make_stream_post_turn_hook()
            await session.start()
            channel = RealtimeChannel(
                session,
                lambda payload, _ws=ws: self._safe_ws_send_json(_ws, payload),
                on_unrecoverable=self._make_unrecoverable_callback(key),
            )
            await channel.start()
            holder = _StreamHolder(session=session, channel=channel)
            self._stream_sessions[key] = holder
            elog(
                "stream.session.attach",
                client_id=client_id,
                session_id=sid,
                profile=profile,
            )

        if isinstance(evt, SessionOpen):
            return  # session already created above; SessionOpen is metadata-only

        await holder.session.push_in(evt)

    async def _close_stream_session(self, key: tuple[str, str]) -> None:
        """Pop and close one stream session. Idempotent + crash-safe."""
        holder = self._stream_sessions.pop(key, None)
        if holder is None:
            return
        try:
            await holder.channel.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("stream session close failed: %s", e)

    async def _close_stream_sessions_for(self, client_id: str | None) -> None:
        """Close any stream sessions belonging to a disconnecting client."""
        if not client_id:
            return
        for key in [k for k in self._stream_sessions if k[0] == client_id]:
            await self._close_stream_session(key)

    def _adopt_sessions_to_ws(self, client_id: str, ws) -> None:
        """Rebind every live channel for ``client_id`` onto ``ws``.

        Called from the AUTH path when a previous ws under the same
        ``client_id`` is being replaced. The channel pump retries the
        in-flight frame on every iteration, so a frame stuck on the
        dead transport completes delivery as soon as we swap the send
        callable.
        """
        for key, holder in self._stream_sessions.items():
            if key[0] != client_id:
                continue
            holder.channel.rebind(
                lambda payload, _ws=ws: self._safe_ws_send_json(_ws, payload),
            )

    def _make_unrecoverable_callback(self, key: tuple[str, str]):
        """Build the ``on_unrecoverable`` callback for a channel.

        Fires when the channel pump has spent
        :attr:`RealtimeChannel.UNRECOVERABLE_AFTER_S` waiting for any ws
        to come back. Reaping the session releases the StreamSession's
        agent resources rather than leaking them.
        """
        async def _on_unrecoverable() -> None:
            elog(
                "stream.session.unrecoverable",
                client_id=key[0],
                session_id=key[1],
            )
            await self._close_stream_session(key)
        return _on_unrecoverable

    def _make_stream_pre_dispatch_hook(self, client_id: str, session_id: str):
        """Build a per-session pre-dispatch hook for the stream path.

        Mirrors the per-turn checks the legacy MESSAGE handler did:
        hot-reload registries, refuse the turn if no models are
        enabled, then bind the session's history mode for SmartRouter.
        Returns a non-None error string to reject the turn — the
        StreamSession publishes ``OutError`` + ``TurnComplete`` so the
        client gets a clean error frame.
        """
        async def _pre_dispatch(_msg) -> str | None:
            # Hot-reload MCPs/models if the registry tables changed,
            # and get the enabled-model count for the rejection gate —
            # one round-trip to the DB. ``-1`` means no DB is wired.
            try:
                _, enabled_count = await self.agent.refresh_registries()
            except Exception as e:  # noqa: BLE001 — inner method already guards
                elog("hot_reload.error", error=str(e))
                enabled_count = -1
            if enabled_count == 0:
                elog("session.rejected_no_models", session_id=session_id)
                return (
                    "No models are enabled. Add one via /models or ask "
                    "the agent to add an openai/anthropic/google model."
                )

            # SmartRouter handles per-session binding internally; it
            # reports ``history_mode = None`` so this is a no-op for
            # the common case. Direct-provider models (legacy/test
            # paths) get the SessionManager pre-bind enforcement.
            active_model = self.agent.model
            history_mode = getattr(active_model, "history_mode", None)
            try:
                self.sessions.bind_history_mode(
                    client_id, session_id, history_mode,
                )
            except ValueError as e:
                elog(
                    "session.history_mode_conflict",
                    client_id=client_id,
                    session_id=session_id,
                    history_mode=history_mode,
                    error=str(e),
                )
                return str(e)

            elog(
                "stream.turn.pre_dispatch",
                client_id=client_id,
                session_id=session_id,
                model_class=type(active_model).__name__,
            )
            return None

        return _pre_dispatch

    def _make_stream_post_turn_hook(self):
        """Build a post-turn hook that fans out MCP resource broadcasts.

        Mirrors the legacy MESSAGE handler's tail: after the turn
        completes, send one ``resource_event`` per MCP namespace
        whose tools fired during the turn, so the desktop app's
        MCPs/Tasks/Workflows screens refresh. ``StreamSession``
        accumulates the set via ``OutToolStatus`` taps.
        """
        async def _post_turn(seen_resources: set[str]) -> None:
            if not seen_resources:
                return
            results = await asyncio.gather(
                *(self.broadcast_resource(r, "changed") for r in seen_resources),
                return_exceptions=True,
            )
            for r, outcome in zip(seen_resources, results):
                if isinstance(outcome, Exception):
                    logger.debug("broadcast_resource(%s) failed: %s", r, outcome)

        return _post_turn

