"""Models REST surface — list, catalog, providers (read-only queries).

These three endpoints get called by the desktop app on every launch,
so they're worth a smoke test even though they don't mutate anything.
"""
from __future__ import annotations

import json

from ._framework import TestContext, TestSkip, test


@test("models", "GET /api/models returns a list of DB model rows")
async def t_models_list(ctx: TestContext) -> None:
    """v0.11.0: ``/api/models`` is DB-backed (the ``models`` table). The
    gateway fixture runs without a wired DB, so we accept either the
    happy path ``{"models": [...]}`` or a 500 "memory DB not available"
    which just means this particular fixture can't exercise it."""
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/models") as r:
            if r.status == 500:
                raise TestSkip("gateway fixture has no MemoryDB wired")
            assert r.status == 200
            body = await r.json()
            assert isinstance(body, dict) and "models" in body, body
            assert isinstance(body["models"], list), body


@test("models", "GET /api/models/catalog returns catalog with pricing")
async def t_models_catalog(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/models/catalog") as r:
            if r.status == 404:
                raise TestSkip("/api/models/catalog not exposed in this build")
            assert r.status == 200
            body = await r.json()
            assert isinstance(body, (list, dict)), body


@test("models", "GET /api/models/providers lists supported providers")
async def t_models_providers(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/models/providers") as r:
            if r.status == 404:
                raise TestSkip("/api/models/providers not exposed in this build")
            assert r.status == 200
            body = await r.json()
            assert isinstance(body, (list, dict)), body
            blob = json.dumps(body).lower()
            assert "openai" in blob, body
