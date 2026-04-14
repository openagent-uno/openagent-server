"""Usage + pricing endpoints — spend summary + pricing table."""
from __future__ import annotations

from ._framework import TestContext, TestSkip, test


@test("pricing", "GET /api/usage/pricing returns model prices")
async def t_pricing_endpoint(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/usage/pricing") as r:
            if r.status == 404:
                raise TestSkip("/api/usage/pricing not exposed in this build")
            assert r.status == 200, f"status {r.status}"
            body = await r.json()
            assert isinstance(body, (list, dict)), body


@test("usage", "GET /api/usage returns spend summary")
async def t_usage_endpoint(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/usage") as r:
            assert r.status == 200
            body = await r.json()
            assert any(k in body for k in ("monthly_spend", "spend", "by_model", "monthly_budget")), body
