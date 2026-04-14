"""WebSocket session tests — round-trip and isolation.

Both tests connect to the running gateway from ``ctx.extras["gateway_port"]``
and exercise the WS protocol the same way clients would. Isolation uses a
unique tag the model has no reason to persist anywhere so session B
genuinely has no way to know session A's value.
"""
from __future__ import annotations

import asyncio
import json
import uuid

from ._framework import TestContext, TestSkip, test


@test("sessions", "WebSocket round-trip: send message, get response")
async def t_ws_roundtrip(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.ws_connect(f"ws://127.0.0.1:{port}/ws") as ws:
            await ws.send_json({"type": "auth", "client_id": "test-client"})
            await asyncio.wait_for(ws.receive(), timeout=10)
            sid = f"ws-roundtrip-{uuid.uuid4().hex[:8]}"
            await ws.send_json({"type": "message", "text": "Reply with literally PING_RESP",
                                "session_id": sid})
            response_text = None
            async with asyncio.timeout(60):
                async for msg in ws:
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    payload = json.loads(msg.data)
                    if payload.get("type") == "response":
                        response_text = payload.get("text", "")
                        break
            assert response_text is not None, "no response message received"
            assert "PING_RESP" in response_text.upper(), f"got: {response_text!r}"


@test("sessions", "session isolation: B can't see A's conversation history")
async def t_session_isolation(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp

    async def _send(client_id: str, sid: str, text: str) -> str:
        async with aiohttp.ClientSession() as http:
            async with http.ws_connect(f"ws://127.0.0.1:{port}/ws") as ws:
                await ws.send_json({"type": "auth", "client_id": client_id})
                await asyncio.wait_for(ws.receive(), timeout=10)
                await ws.send_json({"type": "message", "text": text, "session_id": sid})
                async with asyncio.timeout(60):
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        payload = json.loads(msg.data)
                        if payload.get("type") == "response":
                            return payload.get("text", "")
        return ""

    sid_a = f"isol-a-{uuid.uuid4().hex[:8]}"
    sid_b = f"isol-b-{uuid.uuid4().hex[:8]}"
    tag = f"zorpgleep_{uuid.uuid4().hex[:6]}"
    await _send("client-a", sid_a,
                f"My favorite test word for today is {tag}. Reply with just OK_NOTED — "
                "do NOT call any tools, do NOT write to vault, do NOT save anything.")
    resp_b = await _send("client-b", sid_b,
                         "What was my favorite test word for today? "
                         "If you don't know from THIS conversation, say NO_INFO. "
                         "Do NOT search vault.")
    assert tag.lower() not in resp_b.lower(), \
        f"session B knew session A's tag {tag!r}: {resp_b[:200]}"
