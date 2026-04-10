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

logger = logging.getLogger(__name__)

# Retry cooldown between bridge crashes
BRIDGE_RETRY_SECONDS = 30


class BaseBridge:
    """Abstract base for platform bridges."""

    name: str = "bridge"

    def __init__(self, gateway_url: str = "ws://localhost:8765/ws", gateway_token: str | None = None):
        self.gateway_url = gateway_url
        self.gateway_token = gateway_token
        self._ws = None
        self._should_stop = False
        self._pending: dict[str, asyncio.Future] = {}  # session_id → response future
        self._status_callbacks: dict[str, Callable] = {}  # session_id → on_status

    async def start(self) -> None:
        """Connect to Gateway and start the platform polling loop with retry."""
        self._should_stop = False
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
                await asyncio.sleep(BRIDGE_RETRY_SECONDS)

    async def stop(self) -> None:
        self._should_stop = True
        if self._ws:
            await self._ws.close()

    async def _connect_gateway(self) -> None:
        """Connect to the Gateway WebSocket and authenticate."""
        import aiohttp
        session = aiohttp.ClientSession()
        self._ws = await session.ws_connect(self.gateway_url)

        # Authenticate
        auth_msg = {"type": P.AUTH, "token": self.gateway_token or "", "client_id": f"bridge:{self.name}"}
        await self._ws.send_json(auth_msg)

        # Wait for auth response
        resp = await self._ws.receive_json()
        if resp.get("type") == P.AUTH_ERROR:
            raise ConnectionError(f"Gateway auth failed: {resp.get('reason')}")
        logger.info("%s bridge connected to Gateway", self.name)

        # Start response listener
        asyncio.create_task(self._listen_gateway())

    async def _listen_gateway(self) -> None:
        """Listen for Gateway responses and dispatch to pending futures."""
        import aiohttp
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
                elif t == P.ERROR and sid in self._pending:
                    self._pending[sid].set_result(data)
                    del self._pending[sid]
                elif t == P.COMMAND_RESULT and "__cmd__" in self._pending:
                    self._pending["__cmd__"].set_result(data)
                    del self._pending["__cmd__"]

            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break

    async def send_message(
        self,
        text: str,
        session_id: str,
        on_status: Callable[[str], Awaitable[None]] | None = None,
    ) -> dict:
        """Send a message through the Gateway and wait for the response."""
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[session_id] = future
        if on_status:
            self._status_callbacks[session_id] = on_status

        await self._ws.send_json({
            "type": P.MESSAGE,
            "text": text,
            "session_id": session_id,
        })

        return await future

    async def send_command(self, name: str) -> str:
        """Send a command and wait for the result."""
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending["__cmd__"] = future
        await self._ws.send_json({"type": P.COMMAND, "name": name})
        result = await future
        return result.get("text", "")

    async def _run(self) -> None:
        """Platform-specific polling loop. Override in subclass."""
        raise NotImplementedError
