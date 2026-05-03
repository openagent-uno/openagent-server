"""REST — /api/mcps CRUD and /api/models/db CRUD.

Uses the same gateway fixture the existing rest tests use. Verifies the
new endpoints return the right shapes and that writes land in the DB.

Skipped when the gateway fixture has no ``MemoryDB`` wired — the current
test harness doesn't pass one (the DB-level unit tests cover that layer
independently; exercising the full DB-backed REST path would require a
standalone integration-test fixture we don't have yet).
"""
from __future__ import annotations

import aiohttp

from ._framework import TestContext, TestSkip, test


def _agent_has_db(ctx: TestContext) -> bool:
    agent = ctx.extras.get("agent")
    return getattr(agent, "memory_db", None) is not None if agent else False


@test("mcps_rest", "GET /api/mcps lists rows from the mcps table")
async def t_list_mcps(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("requires gateway fixture")
    if not _agent_has_db(ctx):
        raise TestSkip("gateway fixture has no MemoryDB wired")

    async with aiohttp.ClientSession() as sess:
        async with sess.get(f"http://127.0.0.1:{port}/api/mcps") as resp:
            assert resp.status == 200
            data = await resp.json()
    assert "mcps" in data
    assert isinstance(data["mcps"], list)


@test("mcps_rest", "DELETE refuses builtin (kind != 'custom'), allows disable")
async def t_delete_builtin_refused(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB
    import uuid as _uuid

    tmp = ctx.db_path.with_name(f"mcps-builtin-{_uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp))
        await db.connect()
        # Simulate a default row written by the bootstrap.
        await db.upsert_mcp(
            "shell",
            kind="default",
            builtin_name="shell",
            enabled=True,
            source="yaml-default",
        )
        # Direct DB delete IS possible (belt and braces); the guard lives
        # one level up in the REST + MCP manager surfaces. Test the REST
        # path here since it's the one users hit.
        from openagent.gateway.api import mcps as mcps_rest

        class _FakeRequest:
            match_info = {"name": "shell"}
            class _FakeApp:
                def __init__(self, agent_db):
                    self._gw = type("GW", (), {"agent": type("A", (), {"memory_db": agent_db})()})()
                def __getitem__(self, key):
                    return self._gw if key == "gateway" else None
            def __init__(self, agent_db):
                self.app = self._FakeApp(agent_db)

        fake = _FakeRequest(db)
        resp = await mcps_rest.handle_delete(fake)
        assert resp.status == 400, resp.status
        # aiohttp.web.Response.text is a property (str), not a coroutine.
        body = resp.text or ""
        assert "builtin" in body.lower() or "refusing" in body.lower(), body
        # Row still present
        assert await db.get_mcp("shell") is not None
        # Disable should still work
        await db.set_mcp_enabled("shell", False)
        assert (await db.get_mcp("shell"))["enabled"] is False
        await db.close()
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


@test("mcps_rest", "POST /api/mcps creates, DELETE removes")
async def t_create_delete_mcp(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("requires gateway fixture")
    if not _agent_has_db(ctx):
        raise TestSkip("gateway fixture has no MemoryDB wired")

    name = "rest-test-mcp"
    body = {"name": name, "command": ["/bin/true"], "enabled": True}
    async with aiohttp.ClientSession() as sess:
        async with sess.post(f"http://127.0.0.1:{port}/api/mcps", json=body) as resp:
            assert resp.status == 201, await resp.text()

        async with sess.get(f"http://127.0.0.1:{port}/api/mcps/{name}") as resp:
            assert resp.status == 200
            row = (await resp.json())["mcp"]
            assert row["name"] == name
            assert row["command"] == ["/bin/true"]

        async with sess.post(
            f"http://127.0.0.1:{port}/api/mcps/{name}/disable"
        ) as resp:
            assert resp.status == 200
            assert (await resp.json())["mcp"]["enabled"] is False

        async with sess.delete(f"http://127.0.0.1:{port}/api/mcps/{name}") as resp:
            assert resp.status == 200


@test("mcps_rest", "POST /api/models writes a row (v0.12: provider_id + model)")
async def t_create_db_model(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("requires gateway fixture")
    if not _agent_has_db(ctx):
        raise TestSkip("gateway fixture has no MemoryDB wired")

    # Create an agno openai provider row first so models can FK to it.
    async with aiohttp.ClientSession() as sess:
        async with sess.post(
            f"http://127.0.0.1:{port}/api/providers",
            json={"name": "openai", "framework": "agno", "api_key": "sk-fake"},
        ) as resp:
            assert resp.status == 201, await resp.text()
            provider_id = (await resp.json())["provider"]["id"]

        try:
            body = {
                "provider_id": provider_id,
                "model": "gpt-rest-test",
                "display_name": "REST Test",
            }
            async with sess.post(
                f"http://127.0.0.1:{port}/api/models", json=body,
            ) as resp:
                assert resp.status == 201, await resp.text()
                created = (await resp.json())["model"]
                model_id = created["id"]
                assert created["runtime_id"] == "openai:gpt-rest-test"

            async with sess.get(f"http://127.0.0.1:{port}/api/models") as resp:
                assert resp.status == 200
                rows = (await resp.json())["models"]
                assert any(m["id"] == model_id for m in rows)

            async with sess.delete(
                f"http://127.0.0.1:{port}/api/models/{model_id}",
            ) as resp:
                # Since v0.10.3 the guardrail that refused "last enabled model"
                # is gone — the rejection gate in StreamSession's pre-dispatch
                # hook surfaces the zero-model state explicitly, so DELETE
                # always succeeds.
                assert resp.status == 200, await resp.text()
        finally:
            async with sess.delete(
                f"http://127.0.0.1:{port}/api/providers/{provider_id}",
            ) as resp:
                pass  # FK cascade handles leftover models


@test("mcps_rest", "GET /api/models/available?provider=openai returns fallback when no key")
async def t_available_openai(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("requires gateway fixture")
    if not _agent_has_db(ctx):
        raise TestSkip("gateway fixture has no MemoryDB wired")

    # Provider row must exist — v0.12 discovery resolves either by
    # provider_id (preferred) or by the legacy ``provider=<name>`` query.
    async with aiohttp.ClientSession() as sess:
        async with sess.post(
            f"http://127.0.0.1:{port}/api/providers",
            json={"name": "openai", "framework": "agno", "api_key": "sk-fake"},
        ) as resp:
            provider_id = None
            if resp.status == 201:
                provider_id = (await resp.json())["provider"]["id"]
            elif resp.status != 400:
                raise TestSkip(f"POST /api/providers returned {resp.status}")

        try:
            async with sess.get(
                f"http://127.0.0.1:{port}/api/models/available?provider=openai"
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
            assert data["provider"] == "openai"
            assert isinstance(data["models"], list)
            if data["models"]:
                assert "id" in data["models"][0]
        finally:
            if provider_id is not None:
                async with sess.delete(
                    f"http://127.0.0.1:{port}/api/providers/{provider_id}"
                ) as resp:
                    pass
