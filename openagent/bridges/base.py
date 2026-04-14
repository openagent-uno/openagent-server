"""Base bridge — connects to the Gateway via WS and translates messages.

Subclasses implement platform-specific polling (Telegram, Discord, etc.)
and call `self.send_message()` / `self.send_command()` to route through
the Gateway. Responses arrive via `on_response()` / `on_status()` callbacks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Awaitable

from openagent.gateway import protocol as P

from openagent.core.logging import elog

logger = logging.getLogger(__name__)

# Retry cooldown between bridge crashes
BRIDGE_RETRY_SECONDS = 30
# Maximum time to wait for a gateway response before giving up.
# Generous to accommodate tool-heavy queries (Claude SDK timeout is 300s × 2 retries).
BRIDGE_RESPONSE_TIMEOUT = 660


def format_tool_status(raw: str) -> str:
    """Convert a raw status string (possibly JSON tool event) into a
    human-readable line suitable for Telegram/Discord/WhatsApp.

    Structured events look like: ``{"tool":"bash","status":"running",...}``
    Plain strings like ``"Thinking..."`` are returned unchanged.
    """
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or "tool" not in data:
            return raw
    except (json.JSONDecodeError, TypeError):
        return raw

    tool = data["tool"]
    status = data.get("status", "running")

    if status == "running":
        return f"Using {tool}..."
    if status == "error":
        err = data.get("error", "unknown error")
        return f"✗ {tool} failed: {err}"
    # done
    return f"✓ {tool} done"


class BaseBridge:
    """Abstract base for platform bridges."""

    name: str = "bridge"

    def __init__(self, gateway_url: str = "ws://localhost:8765/ws", gateway_token: str | None = None):
        self.gateway_url = gateway_url
        self.gateway_token = gateway_token
        self._ws = None
        self._ws_session = None  # aiohttp.ClientSession — must be closed
        self._listener_task: asyncio.Task | None = None
        self._should_stop = False
        self._pending: dict[str, asyncio.Future] = {}  # session_id → response future
        self._command_future: asyncio.Future | None = None
        self._command_lock = asyncio.Lock()
        self._status_callbacks: dict[str, Callable] = {}  # session_id → on_status
        self._session_locks: dict[str, asyncio.Lock] = {}  # per-session serialization

    async def start(self) -> None:
        """Connect to Gateway and start the platform polling loop with retry."""
        self._should_stop = False
        elog("bridge.start", name=self.name)
        while not self._should_stop:
            try:
                await self._connect_gateway()
                await self._run()
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._should_stop:
                    break
                logger.error("%s bridge crashed: %s, retrying in %ds...", self.name, e, BRIDGE_RETRY_SECONDS)
                elog("bridge.error", name=self.name, error=str(e))
                await asyncio.sleep(BRIDGE_RETRY_SECONDS)

    async def stop(self) -> None:
        elog("bridge.stop", name=self.name)
        self._should_stop = True
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except (asyncio.CancelledError, Exception):
                pass
            self._listener_task = None
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._ws_session:
            await self._ws_session.close()
            self._ws_session = None

    def _resolve_orphaned_futures(self, reason: str) -> None:
        """Resolve all pending futures with an error so callers don't hang."""
        orphaned = list(self._pending.items())
        self._pending.clear()
        self._status_callbacks.clear()
        for sid, future in orphaned:
            if not future.done():
                future.set_result({"type": "error", "text": reason})
                logger.warning("Resolved orphaned future for %s: %s", sid, reason)
        if self._command_future and not self._command_future.done():
            self._command_future.set_result({"type": "error", "text": reason})
        self._command_future = None

    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a per-session lock to serialize message sending."""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]

    @staticmethod
    def append_model_feedback(text: str, model: str | None) -> str:
        """Append a compact model footer to a response body."""
        if not model:
            return text
        footer = f"Model: {model}"
        return f"{text}\n\n{footer}" if text else footer

    async def _connect_gateway(self) -> None:
        """Connect to the Gateway WebSocket and authenticate."""
        import aiohttp

        # Clean up stale state from any previous connection
        self._resolve_orphaned_futures("Reconnecting to gateway")
        self._session_locks.clear()

        # Close any previous session/ws from a prior connection attempt
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._ws_session:
            await self._ws_session.close()
            self._ws_session = None

        session = aiohttp.ClientSession()
        self._ws_session = session
        self._ws = await session.ws_connect(self.gateway_url)

        # Authenticate
        auth_msg = {"type": P.AUTH, "token": self.gateway_token or "", "client_id": f"bridge:{self.name}"}
        await self._ws.send_json(auth_msg)

        # Wait for auth response
        resp = await self._ws.receive_json()
        if resp.get("type") == P.AUTH_ERROR:
            raise ConnectionError(f"Gateway auth failed: {resp.get('reason')}")
        logger.info("%s bridge connected to Gateway", self.name)

        # Start response listener — store the task so exceptions are not lost
        self._listener_task = asyncio.create_task(
            self._listen_gateway(), name=f"{self.name}:gw-listener"
        )

    async def _listen_gateway(self) -> None:
        """Listen for Gateway responses and dispatch to pending futures."""
        import aiohttp
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    t = data.get("type")
                    sid = data.get("session_id")

                    if t == P.STATUS and sid in self._status_callbacks:
                        try:
                            await self._status_callbacks[sid](data.get("text", ""))
                        except Exception:
                            pass
                    elif t == P.RESPONSE and sid in self._pending:
                        self._pending[sid].set_result(data)
                        del self._pending[sid]
                        self._status_callbacks.pop(sid, None)
                    elif t == P.ERROR:
                        # Errors may or may not have a session_id.  Try to
                        # route to the matching pending future; if no match,
                        # just log it.
                        if sid and sid in self._pending:
                            self._pending[sid].set_result(data)
                            del self._pending[sid]
                            self._status_callbacks.pop(sid, None)
                    elif t == P.COMMAND_RESULT and self._command_future and not self._command_future.done():
                        self._command_future.set_result(data)
                        self._command_future = None

                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        finally:
            # Resolve any futures still waiting so callers don't hang forever
            self._resolve_orphaned_futures("Gateway connection lost")

    async def send_message(
        self,
        text: str,
        session_id: str,
        on_status: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict:
        """Send a message through the Gateway and wait for the response.

        Per-session locking ensures only one message is in-flight per user,
        preventing the ``_pending`` dict from being overwritten by a second
        concurrent message for the same session.  A generous timeout prevents
        the caller from hanging forever if the gateway drops.
        """
        async with self._get_session_lock(session_id):
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            self._pending[session_id] = future
            if on_status:
                self._status_callbacks[session_id] = on_status

            await self._ws.send_json({
                "type": P.MESSAGE,
                "text": text,
                "session_id": session_id,
            })

            try:
                return await asyncio.wait_for(future, timeout=BRIDGE_RESPONSE_TIMEOUT)
            except asyncio.TimeoutError:
                self._pending.pop(session_id, None)
                self._status_callbacks.pop(session_id, None)
                logger.error("Bridge response timeout for %s after %ds", session_id, BRIDGE_RESPONSE_TIMEOUT)
                return {"type": "error", "text": "Request timed out. Please try again."}

    async def send_command(self, name: str) -> str:
        """Send a command and wait for the result."""
        async with self._command_lock:
            future: asyncio.Future = asyncio.get_running_loop().create_future()
            self._command_future = future
            await self._ws.send_json({"type": P.COMMAND, "name": name})
            try:
                result = await asyncio.wait_for(future, timeout=BRIDGE_RESPONSE_TIMEOUT)
            except asyncio.TimeoutError:
                if self._command_future is future:
                    self._command_future = None
                logger.error("Bridge command timeout for %s after %ds", name, BRIDGE_RESPONSE_TIMEOUT)
                return "Command timed out. Please try again."
            finally:
                if self._command_future is future:
                    self._command_future = None
            return result.get("text", "")

    async def _run(self) -> None:
        """Platform-specific polling loop. Override in subclass."""
        raise NotImplementedError
