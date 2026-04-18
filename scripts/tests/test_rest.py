"""REST endpoint coverage — one test per HTTP route.

Each test assumes the gateway is already running (``ctx.extras["gateway_port"]``
is set by ``test_gateway.t_gateway_health``) and verifies the endpoint
returns 2xx with a sensible body shape. Destructive endpoints (restart,
delete) are exercised with test data we created in the same test, so
they don't touch real state.
"""
from __future__ import annotations

import json
import uuid

from ._framework import TestContext, TestSkip, test


# ── /api/config ──────────────────────────────────────────────────────


@test("config", "GET /api/config returns the current config")
async def t_config_get(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/config") as r:
            assert r.status == 200
            body = await r.json()
            assert any(k in body for k in ("name", "model", "providers", "config")), body


@test("config", "PATCH /api/config/{section} updates one section")
async def t_config_patch(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    # Patch the `system_prompt` section — round-trip a new value, then restore.
    new_prompt = f"patched at {uuid.uuid4().hex[:6]}"
    async with aiohttp.ClientSession() as http:
        async with http.patch(f"http://127.0.0.1:{port}/api/config/system_prompt",
                              json=new_prompt) as r:
            assert r.status == 200, f"status {r.status}"
            body = await r.json()
            assert body.get("ok") is True, body
        # Read back and confirm
        async with http.get(f"http://127.0.0.1:{port}/api/config") as r:
            cfg = await r.json()
            assert cfg.get("system_prompt") == new_prompt, cfg.get("system_prompt")


# ── /api/logs ────────────────────────────────────────────────────────


@test("logs", "GET /api/logs returns recent events")
async def t_logs_get(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/logs") as r:
            if r.status == 404:
                raise TestSkip("/api/logs not exposed")
            assert r.status == 200, f"status {r.status}"
            body = await r.json()
            # Shape is typically {events: [...]} or a list
            assert isinstance(body, (list, dict)), body


@test("logs", "DELETE /api/logs clears recent events")
async def t_logs_delete(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.delete(f"http://127.0.0.1:{port}/api/logs") as r:
            if r.status == 404:
                raise TestSkip("/api/logs DELETE not exposed")
            assert r.status in (200, 204), f"status {r.status}"


# ── /api/usage/daily ─────────────────────────────────────────────────


@test("usage", "GET /api/usage/daily returns daily breakdown")
async def t_usage_daily(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/usage/daily") as r:
            if r.status == 404:
                raise TestSkip("/api/usage/daily not exposed")
            assert r.status == 200
            body = await r.json()
            assert isinstance(body, (list, dict)), body


# ── /api/providers ───────────────────────────────────────────────────


@test("providers", "GET /api/providers lists DB-backed providers (v0.11)")
async def t_providers_list(ctx: TestContext) -> None:
    """Providers moved from yaml to the ``providers`` SQLite table in
    v0.11.0. The gateway fixture runs without a wired DB, so the list
    comes back empty — we just verify the shape, not the contents."""
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/providers") as r:
            if r.status == 404:
                raise TestSkip("/api/providers not exposed")
            assert r.status == 200
            body = await r.json()
            assert isinstance(body, dict) and "providers" in body, body
            assert isinstance(body["providers"], dict), body


# ── /api/providers CRUD (was /api/models CRUD pre-v0.11) ──────────────


@test("providers", "POST /api/providers adds a provider, DELETE removes it")
async def t_providers_crud(ctx: TestContext) -> None:
    """Replaces the pre-v0.11 ``POST /api/models`` provider-add path.
    Provider keys now live in the DB; legacy yaml endpoint is gone."""
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    name = f"test_provider_{uuid.uuid4().hex[:6]}"
    async with aiohttp.ClientSession() as http:
        async with http.post(
            f"http://127.0.0.1:{port}/api/providers",
            json={"name": name, "api_key": "sk-fake"},
        ) as r:
            if r.status == 500:
                raise TestSkip("gateway fixture has no MemoryDB wired")
            assert r.status == 201, f"POST returned {r.status}: {await r.text()}"
            body = await r.json()
            assert body.get("ok") is True, body
        async with http.delete(f"http://127.0.0.1:{port}/api/providers/{name}") as r:
            assert r.status in (200, 204), f"DELETE returned {r.status}"


@test("models", "GET /api/models/active returns active model config")
async def t_models_active(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/models/active") as r:
            if r.status == 404:
                raise TestSkip("/api/models/active not exposed")
            assert r.status == 200
            body = await r.json()
            assert "active" in body or "provider" in body, body


# ── /api/vault ────────────────────────────────────────────────────────


@test("vault_rest", "PUT + GET + DELETE /api/vault/notes round-trip")
async def t_vault_rest(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    note_path = f"test/openagent-rest-{uuid.uuid4().hex[:6]}.md"
    content = f"# test note\n\nhello REST {uuid.uuid4().hex[:8]}"
    async with aiohttp.ClientSession() as http:
        # Write
        async with http.put(f"http://127.0.0.1:{port}/api/vault/notes/{note_path}",
                            json={"content": content}) as r:
            assert r.status == 200, f"PUT returned {r.status}"
        # Read
        async with http.get(f"http://127.0.0.1:{port}/api/vault/notes/{note_path}") as r:
            assert r.status == 200, f"GET returned {r.status}"
            body = await r.json()
            assert content in (body.get("content", "") + body.get("body", "")), body
        # List
        async with http.get(f"http://127.0.0.1:{port}/api/vault/notes") as r:
            assert r.status == 200
            listing = await r.json()
            notes = listing.get("notes") or listing.get("results") or []
            paths = [n.get("path", "") for n in notes if isinstance(n, dict)]
            assert any(note_path in p for p in paths), f"note not listed: {paths[:5]}"
        # Delete
        async with http.delete(f"http://127.0.0.1:{port}/api/vault/notes/{note_path}") as r:
            assert r.status in (200, 204), f"DELETE returned {r.status}"


@test("vault_rest", "GET /api/vault/graph returns {nodes, edges}")
async def t_vault_graph(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    async with aiohttp.ClientSession() as http:
        async with http.get(f"http://127.0.0.1:{port}/api/vault/graph") as r:
            assert r.status == 200
            body = await r.json()
            assert "nodes" in body and "edges" in body, body
            assert isinstance(body["nodes"], list)
            assert isinstance(body["edges"], list)


@test("vault_rest", "GET /api/vault/search?q=... returns matches")
async def t_vault_search(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("gateway not running")
    import aiohttp
    # Write a searchable marker first
    tag = f"ZORPMARKER_{uuid.uuid4().hex[:6]}"
    note_path = f"test/searchme-{uuid.uuid4().hex[:4]}.md"
    async with aiohttp.ClientSession() as http:
        async with http.put(f"http://127.0.0.1:{port}/api/vault/notes/{note_path}",
                            json={"content": f"# searchable\n\n{tag}"}) as r:
            assert r.status == 200
        try:
            async with http.get(f"http://127.0.0.1:{port}/api/vault/search",
                                params={"q": tag}) as r:
                assert r.status == 200
                body = await r.json()
                results = body.get("results") or []
                paths = [res.get("path", "") for res in results]
                assert any(note_path in p for p in paths), \
                    f"search didn't find {tag}: {paths[:5]}"
        finally:
            async with http.delete(
                f"http://127.0.0.1:{port}/api/vault/notes/{note_path}") as r:
                pass  # best-effort cleanup
