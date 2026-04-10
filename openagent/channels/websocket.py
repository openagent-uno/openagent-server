"""WebSocket channel for the OpenAgent desktop/web app.

Exposes a WebSocket endpoint for real-time bidirectional chat and a REST
API for vault operations (notes CRUD, graph data, search). Both share the
same ``aiohttp`` HTTP server on a single port.

REST API::

    GET  /api/health                → agent status
    GET  /api/vault/notes           → list all notes
    GET  /api/vault/notes/{path}    → read note content
    PUT  /api/vault/notes/{path}    → write/update note
    DELETE /api/vault/notes/{path}  → delete note
    GET  /api/vault/graph           → graph data {nodes, edges}
    GET  /api/vault/search?q=...    → search notes
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from openagent.channels.base import BaseChannel, parse_response_markers
from openagent.channels.commands import CommandDispatcher
from openagent.channels.queue import UserQueueManager

if TYPE_CHECKING:
    from openagent.core.agent import Agent

logger = logging.getLogger(__name__)

# Wikilink pattern: [[target]] or [[target|alias]]
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


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
        vault_path: str | None = None,
    ):
        super().__init__(agent)
        self.host = host
        self.port = port
        self.token = token
        self.allowed_origins = set(allowed_origins) if allowed_origins else None
        self.vault_path = vault_path
        self._queue = UserQueueManager(platform="websocket", agent_name=agent.name)
        self._commands = CommandDispatcher(agent, self._queue)
        self._app = None
        self._runner = None
        self._clients: dict[str, object] = {}

    def _resolve_vault(self) -> Path:
        if self.vault_path:
            return Path(self.vault_path).expanduser().resolve()
        from openagent.core.paths import default_vault_path
        return default_vault_path()

    async def _run(self) -> None:
        try:
            from aiohttp import web
            from aiohttp.web import middleware
        except ImportError:
            raise ImportError(
                "aiohttp is required for the WebSocket channel. "
                "Install it with: pip install openagent-framework[websocket]"
            )

        # CORS middleware — must handle both normal responses AND HTTP
        # exceptions (404, 500, etc.) which aiohttp raises as exceptions
        # that bypass the normal return path.
        @middleware
        async def cors_middleware(request, handler):
            # Handle preflight immediately
            if request.method == 'OPTIONS':
                resp = web.Response(status=204)
            else:
                try:
                    resp = await handler(request)
                except web.HTTPException as ex:
                    resp = ex
                except Exception:
                    resp = web.Response(status=500, text="Internal Server Error")
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Methods"] = "GET, PUT, DELETE, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
            return resp

        app = web.Application(middlewares=[cors_middleware])
        app.router.add_get("/ws", self._handle_ws)
        # REST
        app.router.add_get("/api/health", self._handle_health)
        app.router.add_get("/api/vault/notes", self._handle_vault_list)
        app.router.add_get("/api/vault/graph", self._handle_vault_graph)
        app.router.add_get("/api/vault/search", self._handle_vault_search)
        app.router.add_get("/api/vault/notes/{path:.+}", self._handle_vault_read)
        app.router.add_put("/api/vault/notes/{path:.+}", self._handle_vault_write)
        app.router.add_delete("/api/vault/notes/{path:.+}", self._handle_vault_delete)
        # OPTIONS preflight
        app.router.add_route("OPTIONS", "/{path:.*}", self._handle_options)
        self._app = app

        runner = web.AppRunner(app)
        await runner.setup()
        self._runner = runner

        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        logger.info("WebSocket channel on ws://%s:%d/ws", self.host, self.port)

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
        authenticated = self.token is None

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        await ws.send_json({"type": "error", "text": "Invalid JSON"})
                        continue

                    msg_type = data.get("type", "")

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

                    if msg_type == "ping":
                        await ws.send_json({"type": "pong"})
                    elif msg_type == "command":
                        cmd_name = data.get("name", "")
                        result = await self._commands.dispatch(f"/{cmd_name}", client_id)
                        text = result.text if result else f"Unknown command: {cmd_name}"
                        await ws.send_json({"type": "command_result", "text": text})
                    elif msg_type == "message":
                        text = data.get("text", "").strip()
                        session_id = data.get("session_id", "default")
                        if text:
                            async def handler(_t=text, _s=session_id, _c=client_id, _w=ws):
                                await self._process_message(_w, _c, _t, _s)
                            position = await self._queue.enqueue(client_id, handler)
                            if position > 0:
                                await ws.send_json({"type": "queued", "position": position})

                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                    break
        except Exception as e:
            logger.error("WebSocket error for %s: %s", client_id, e)
        finally:
            if client_id and client_id in self._clients:
                del self._clients[client_id]
        return ws

    async def _process_message(self, ws, client_id, text, session_id):
        try:
            async def on_status(status):
                try:
                    await ws.send_json({"type": "status", "text": status, "session_id": session_id})
                except Exception:
                    pass

            response = await self.agent.run(
                message=text, user_id=client_id,
                session_id=self._queue.get_session_id(client_id),
                on_status=on_status,
            )
            clean_text, attachments = parse_response_markers(response)
            att_list = [{"type": a.type, "path": a.path, "filename": a.filename} for a in attachments]
            await ws.send_json({
                "type": "response", "text": clean_text,
                "session_id": session_id,
                "attachments": att_list or None,
            })
        except Exception as e:
            logger.error("WS process error for %s: %s", client_id, e)
            try:
                await ws.send_json({"type": "error", "text": str(e)})
            except Exception:
                pass

    # ── REST: health ───────────────────────────────────────────────────

    async def _handle_options(self, request):
        from aiohttp import web
        return web.Response(status=204)

    async def _handle_health(self, request):
        from aiohttp import web
        import openagent
        return web.json_response({
            "status": "ok",
            "agent": self.agent.name,
            "version": getattr(openagent, "__version__", "?"),
            "connected_clients": len(self._clients),
        })

    # ── REST: vault ────────────────────────────────────────────────────

    def _parse_frontmatter(self, content: str) -> tuple[dict, str]:
        """Extract YAML frontmatter from markdown. Returns (meta, body)."""
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                import yaml
                try:
                    meta = yaml.safe_load(parts[1]) or {}
                except Exception:
                    meta = {}
                return meta, parts[2].strip()
        return {}, content

    def _scan_wikilinks(self, content: str) -> list[str]:
        return _WIKILINK_RE.findall(content)

    async def _handle_vault_list(self, request):
        """List all .md notes with metadata."""
        from aiohttp import web
        vault = self._resolve_vault()
        if not vault.exists():
            return web.json_response({"notes": []})

        notes = []
        for md in sorted(vault.rglob("*.md")):
            rel = str(md.relative_to(vault))
            content = md.read_text(errors="replace")
            meta, body = self._parse_frontmatter(content)
            stat = md.stat()
            notes.append({
                "path": rel,
                "title": meta.get("title", md.stem),
                "tags": meta.get("tags", []),
                "type": meta.get("type", ""),
                "modified": stat.st_mtime,
                "size": stat.st_size,
            })
        return web.json_response({"notes": notes})

    async def _handle_vault_read(self, request):
        """Read a single note's full content + frontmatter."""
        from aiohttp import web
        vault = self._resolve_vault()
        note_path = request.match_info["path"]
        full = vault / note_path
        if not full.exists() or not full.is_file():
            return web.json_response({"error": "Not found"}, status=404)

        content = full.read_text(errors="replace")
        meta, body = self._parse_frontmatter(content)
        links = self._scan_wikilinks(content)
        return web.json_response({
            "path": note_path,
            "content": content,
            "frontmatter": meta,
            "body": body,
            "links": links,
            "modified": full.stat().st_mtime,
        })

    async def _handle_vault_write(self, request):
        """Write or update a note."""
        from aiohttp import web
        vault = self._resolve_vault()
        note_path = request.match_info["path"]
        full = vault / note_path
        full.parent.mkdir(parents=True, exist_ok=True)
        data = await request.json()
        content = data.get("content", "")
        full.write_text(content)
        return web.json_response({"ok": True, "path": note_path})

    async def _handle_vault_delete(self, request):
        """Delete a note."""
        from aiohttp import web
        vault = self._resolve_vault()
        note_path = request.match_info["path"]
        full = vault / note_path
        if full.exists():
            full.unlink()
            return web.json_response({"ok": True})
        return web.json_response({"error": "Not found"}, status=404)

    async def _handle_vault_search(self, request):
        """Search notes by content or title."""
        from aiohttp import web
        vault = self._resolve_vault()
        query = (request.query.get("q") or "").lower().strip()
        if not query:
            return web.json_response({"results": []})

        results = []
        for md in vault.rglob("*.md"):
            content = md.read_text(errors="replace")
            if query in content.lower() or query in md.stem.lower():
                meta, _ = self._parse_frontmatter(content)
                rel = str(md.relative_to(vault))
                results.append({
                    "path": rel,
                    "title": meta.get("title", md.stem),
                    "tags": meta.get("tags", []),
                })
        return web.json_response({"results": results})

    async def _handle_vault_graph(self, request):
        """Build graph data: nodes (notes) and edges (wikilinks)."""
        from aiohttp import web
        vault = self._resolve_vault()
        if not vault.exists():
            return web.json_response({"nodes": [], "edges": []})

        nodes = []
        edges = []
        # Map stem → relative path for link resolution
        stem_map: dict[str, str] = {}
        note_data: dict[str, dict] = {}

        for md in vault.rglob("*.md"):
            rel = str(md.relative_to(vault))
            content = md.read_text(errors="replace")
            meta, body = self._parse_frontmatter(content)
            stem_map[md.stem.lower()] = rel
            note_data[rel] = {
                "meta": meta,
                "links": self._scan_wikilinks(content),
            }

        for rel, data in note_data.items():
            meta = data["meta"]
            tags = meta.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            nodes.append({
                "id": rel,
                "label": meta.get("title", Path(rel).stem),
                "tags": tags,
                "type": meta.get("type", ""),
            })
            for link_target in data["links"]:
                target_key = link_target.lower().strip()
                target_path = stem_map.get(target_key)
                if target_path and target_path != rel:
                    edges.append({"source": rel, "target": target_path})

        return web.json_response({"nodes": nodes, "edges": edges})
