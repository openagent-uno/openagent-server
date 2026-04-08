"""OAuth support for MCP servers that require authentication.

Handles the full OAuth 2.1 + PKCE flow:
1. First connection attempt returns 401
2. Discovers OAuth server metadata
3. Registers client dynamically
4. Opens browser for user login (one-time)
5. Receives callback, exchanges for tokens
6. Saves tokens to disk for future use
7. Auto-refreshes expired tokens

Tokens are stored in ~/.openagent/oauth/<server_name>.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import urlparse, parse_qs

from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientMetadata, OAuthClientInformationFull, OAuthToken

logger = logging.getLogger(__name__)

OAUTH_DIR = Path.home() / ".openagent" / "oauth"
CALLBACK_PORT_RANGE = (18700, 18720)  # ports to try for OAuth callback


class FileTokenStorage:
    """Stores OAuth tokens and client info in a JSON file per server."""

    def __init__(self, server_name: str):
        self._dir = OAUTH_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{server_name}.json"
        self._data: dict[str, Any] = {}
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2, default=str))

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._data.get("tokens")
        if raw:
            return OAuthToken(**raw)
        return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._data["tokens"] = tokens.model_dump()
        self._save()

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._data.get("client_info")
        if raw:
            return OAuthClientInformationFull(**raw)
        return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._data["client_info"] = client_info.model_dump()
        self._save()


class _CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""
    auth_code: str | None = None
    state: str | None = None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        _CallbackHandler.auth_code = params.get("code", [None])[0]
        _CallbackHandler.state = params.get("state", [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html><body><h1>Authenticated!</h1><p>You can close this window.</p></body></html>")

    def log_message(self, format, *args):
        pass  # suppress logs


def _find_free_port() -> int:
    """Find a free port for the OAuth callback server."""
    import socket
    for port in range(*CALLBACK_PORT_RANGE):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError("No free port for OAuth callback")


def create_oauth_provider(server_name: str, server_url: str) -> OAuthClientProvider:
    """Create an OAuth provider for an MCP server.

    On first use, opens the browser for login. Subsequent uses reuse saved tokens.
    """
    storage = FileTokenStorage(server_name)
    port = _find_free_port()
    redirect_uri = f"http://127.0.0.1:{port}/callback"

    client_metadata = OAuthClientMetadata(
        redirect_uris=[redirect_uri],
        client_name=f"OpenAgent ({server_name})",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",
    )

    # Callback server + handler
    _CallbackHandler.auth_code = None
    _CallbackHandler.state = None
    callback_server: HTTPServer | None = None
    server_thread: Thread | None = None

    async def redirect_handler(auth_url: str) -> None:
        """Open browser for OAuth login."""
        nonlocal callback_server, server_thread

        # Start callback server
        callback_server = HTTPServer(("127.0.0.1", port), _CallbackHandler)
        server_thread = Thread(target=callback_server.serve_forever, daemon=True)
        server_thread.start()

        logger.info(f"OAuth: opening browser for {server_name} authentication...")
        logger.info(f"OAuth URL: {auth_url}")

        # Try to open browser
        try:
            webbrowser.open(auth_url)
        except Exception:
            logger.warning(f"Could not open browser. Please visit: {auth_url}")

    async def callback_handler() -> tuple[str, str | None]:
        """Wait for OAuth callback with auth code."""
        # Wait for the callback
        for _ in range(300):  # 5 minutes timeout
            if _CallbackHandler.auth_code:
                break
            await asyncio.sleep(1)

        code = _CallbackHandler.auth_code
        state = _CallbackHandler.state

        # Cleanup
        if callback_server:
            callback_server.shutdown()

        if not code:
            raise RuntimeError(f"OAuth timeout: no callback received for {server_name}")

        logger.info(f"OAuth: received callback for {server_name}")
        return code, state

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
        timeout=300.0,
    )
