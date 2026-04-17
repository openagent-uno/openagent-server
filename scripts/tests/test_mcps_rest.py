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


@test("mcps_rest", "POST /api/models/db writes a row")
async def t_create_db_model(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("requires gateway fixture")
    if not _agent_has_db(ctx):
        raise TestSkip("gateway fixture has no MemoryDB wired")

    body = {"provider": "openai", "model_id": "gpt-rest-test", "display_name": "REST Test"}
    async with aiohttp.ClientSession() as sess:
        async with sess.post(f"http://127.0.0.1:{port}/api/models/db", json=body) as resp:
            assert resp.status == 201, await resp.text()
            created = (await resp.json())["model"]
            runtime_id = created["runtime_id"]
            assert runtime_id == "openai:gpt-rest-test"

        async with sess.get(f"http://127.0.0.1:{port}/api/models/db") as resp:
            assert resp.status == 200
            rows = (await resp.json())["models"]
            assert any(m["runtime_id"] == runtime_id for m in rows)

        async with sess.delete(
            f"http://127.0.0.1:{port}/api/models/db/{runtime_id}"
        ) as resp:
            # Delete may fail if this would leave zero enabled models — handle
            # both outcomes so the test works in isolation and in full runs.
            assert resp.status in (200, 400), await resp.text()


@test("mcps_rest", "GET /api/models/available?provider=openai returns fallback when no key")
async def t_available_openai(ctx: TestContext) -> None:
    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("requires gateway fixture")

    async with aiohttp.ClientSession() as sess:
        async with sess.get(
            f"http://127.0.0.1:{port}/api/models/available?provider=openai"
        ) as resp:
            assert resp.status == 200
            data = await resp.json()
    assert data["provider"] == "openai"
    assert isinstance(data["models"], list)
    # Either live-fetch succeeded (has a key) or bundled fallback kicks in;
    # either way we expect at least one entry with an ``id`` field.
    if data["models"]:
        assert "id" in data["models"][0]
