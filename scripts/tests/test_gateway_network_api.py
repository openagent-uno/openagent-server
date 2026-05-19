"""Gateway ``/api/network/*`` endpoints: list users/agents/invites + mint.

Unit-level tests that drive the handler functions directly against a
real MemoryDB + a mocked aiohttp request. Avoids the gateway-lifecycle
overhead (no live socket, no auth middleware, no Iroh node) — those
are exercised by the live-smoke script and the higher-level gateway
tests. Here we pin:

  - GET /users + /agents return what's in the coordinator store
  - GET /invitations filters out spent/expired rows
  - POST /invitations auto-picks the role from whether HANDLE exists
    (existing → device-bound, new → user-role)
  - POST /invitations honours an explicit ``role`` override
  - DELETE /invitations/{code} is idempotent (200 on stale code)
  - Member-mode agents return 404 (the network row's role != coordinator)
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from aiohttp.test_utils import make_mocked_request

from ._framework import TestContext, test


def _make_request(memory_db, *, method: str = "GET", path: str = "/x",
                  body: bytes | None = None,
                  user_handle: str = "alessandro",
                  match_info: dict | None = None):
    """Build a minimal aiohttp request the handlers can read from.

    Handlers only look at ``request.app['gateway'].agent.memory_db``
    (via the shared ``gateway_db`` helper), ``request['user_handle']``,
    ``request.match_info``, and (for POST) ``await request.json()``.
    aiohttp's ``make_mocked_request`` accepts a dict-like ``app=``
    argument and exposes it via ``request.app[...]`` — no monkey-
    patching needed."""
    from aiohttp.streams import StreamReader

    headers = {}
    if body is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body))

    fake_app = {
        "gateway": SimpleNamespace(
            agent=SimpleNamespace(memory_db=memory_db),
        ),
    }

    kwargs = {
        "method": method,
        "path": path,
        "headers": headers,
        "match_info": match_info or {},
        "app": fake_app,
    }
    if body is not None:
        # ``StreamReader`` is what ``request.json()`` reads from. We
        # feed the whole body in one shot and mark EOF.
        from unittest.mock import Mock
        protocol = Mock(_reading_paused=False)
        reader = StreamReader(protocol, limit=2**16, loop=None)
        reader.feed_data(body)
        reader.feed_eof()
        kwargs["payload"] = reader

    req = make_mocked_request(**kwargs)
    req["user_handle"] = user_handle
    return req


async def _open_db(db_path: Path):
    from src.memory.db import MemoryDB
    db = MemoryDB(str(db_path))
    await db.connect()
    return db


async def _seed_coord(db_path: Path, *, users: list[str] | None = None,
                     agents: list[tuple[str, str]] | None = None,
                     invites: list[dict] | None = None,
                     network_name: str = "lyra-agent"):
    """Spin up a coordinator-mode network row + optional users/agents/invites."""
    from src.network.coordinator.store import CoordinatorStore
    from src.core import paths as core_paths

    db = await _open_db(db_path)
    try:
        store = CoordinatorStore(db)
        coord_node = "cb" + "0" * 62
        await store.set_network_role(
            role="coordinator",
            network_id=uuid.uuid4().hex,
            name=network_name,
            coordinator_node_id=coord_node,
            coordinator_pubkey=b"\x42" * 32,
        )
        for u in users or []:
            await store.create_user(handle=u, pake_record=b"\x00" * 64, pake_algo="srp6a")
        for handle, node_id in agents or []:
            await store.register_agent(handle=handle, node_id=node_id, owner_handle="system")
        for inv in invites or []:
            await store.create_invitation(
                role=inv.get("role", "user"),
                created_by=inv.get("created_by", "cli"),
                ttl_seconds=inv.get("ttl", 3600),
                uses=inv.get("uses", 1),
                bind_to_handle=inv.get("bind_to"),
            )
    finally:
        await db.close()

    # The mint handler needs ``get_agent_dir`` set so it can find the
    # identity.key for the ticket's NodeId. Seed it now.
    agent_dir = db_path.parent
    core_paths.set_agent_dir(agent_dir)


@test("gateway_network", "GET /users returns the registered handles")
async def t_list_users(ctx: TestContext) -> None:
    from src.gateway.api.network import handle_list_users

    agent_dir = ctx.test_dir / f"gwnet-users-{uuid.uuid4().hex[:6]}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    db_path = agent_dir / "agent.db"
    await _seed_coord(db_path, users=["alessandro", "marco", "virgil"])

    db = await _open_db(db_path)
    try:
        req = _make_request(db)
        resp = await handle_list_users(req)
        assert resp.status == 200, f"got {resp.status}: {resp.text!r}"
        import json
        body = json.loads(resp.body)
        handles = {u["handle"] for u in body["users"]}
        assert handles == {"alessandro", "marco", "virgil"}, handles
        for u in body["users"]:
            assert u["status"] == "active"
            assert u["pake_algo"] == "srp6a"
            assert isinstance(u["created_at"], (int, float))
    finally:
        await db.close()


@test("gateway_network", "GET /agents returns the coordinator's own agent first")
async def t_list_agents(ctx: TestContext) -> None:
    """Sanity-check the v0.13.17 ordering fix flows through the HTTP
    layer too: ``agents[0]`` must be the agent whose NodeId matches
    the coordinator's, otherwise default-picking clients would dial
    a foreign-network gateway and 1006-out."""
    from src.gateway.api.network import handle_list_agents

    agent_dir = ctx.test_dir / f"gwnet-agents-{uuid.uuid4().hex[:6]}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    db_path = agent_dir / "agent.db"

    # Use the same coord NodeId the seeder set on the network row.
    coord_node = "cb" + "0" * 62
    foreign = "ff" + "8" * 62
    await _seed_coord(db_path, agents=[
        ("foreign-agent", foreign),  # registered first
        ("lyra-agent",    coord_node),
    ])

    db = await _open_db(db_path)
    try:
        req = _make_request(db)
        resp = await handle_list_agents(req)
        assert resp.status == 200
        import json
        body = json.loads(resp.body)
        assert body["agents"][0]["handle"] == "lyra-agent", \
            f"coord agent must be first: {[a['handle'] for a in body['agents']]}"
        assert {a["handle"] for a in body["agents"]} == {"lyra-agent", "foreign-agent"}
    finally:
        await db.close()


@test("gateway_network", "GET /invitations skips spent + expired entries")
async def t_list_invitations(ctx: TestContext) -> None:
    from src.gateway.api.network import handle_list_invitations

    agent_dir = ctx.test_dir / f"gwnet-invs-{uuid.uuid4().hex[:6]}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    db_path = agent_dir / "agent.db"

    await _seed_coord(db_path, invites=[
        {"role": "user", "ttl": 600},   # fresh
        {"role": "device", "bind_to": "alessandro", "ttl": 600},  # fresh
        {"role": "user", "ttl": -10},   # already expired (negative TTL → past expires_at)
    ])

    db = await _open_db(db_path)
    try:
        # Also burn one of the live invites so we can check it's hidden.
        # ``list_invitations(include_expired=False)`` is what the handler
        # uses; matching its filter keeps the test honest.
        from src.network.coordinator.store import CoordinatorStore
        store = CoordinatorStore(db)
        live = await store.list_invitations(include_expired=False)
        assert len(live) == 2, f"setup wrong: expected 2 live, got {len(live)}"
        burned = await store.consume_invitation(live[0].code)
        assert burned is not None, "consume should have succeeded on a fresh invite"

        req = _make_request(db)
        resp = await handle_list_invitations(req)
        assert resp.status == 200
        import json
        body = json.loads(resp.body)
        codes = {i["code"] for i in body["invitations"]}
        # Exactly the remaining fresh invite should show up.
        assert len(codes) == 1, f"expected 1 live invite, got {body}"
    finally:
        await db.close()


@test("gateway_network", "POST /invitations: new handle → user-role")
async def t_mint_new_handle(ctx: TestContext) -> None:
    from src.gateway.api.network import handle_mint_invitation

    agent_dir = ctx.test_dir / f"gwnet-mintnew-{uuid.uuid4().hex[:6]}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    db_path = agent_dir / "agent.db"
    await _seed_coord(db_path, users=["alessandro"])

    db = await _open_db(db_path)
    try:
        req = _make_request(
            db, method="POST", path="/api/network/invitations",
            body=b'{"handle":"marco"}',
        )
        resp = await handle_mint_invitation(req)
        assert resp.status == 201, f"got {resp.status}: {resp.body!r}"
        import json
        body = json.loads(resp.body)
        assert body["role"] == "user", body
        assert body["bind_to"] == "", body
        assert body["intent"].startswith("onboard marco"), body
        assert body["ticket"].startswith("oa1"), body
        assert len(body["code"]) >= 16, body
    finally:
        await db.close()


@test("gateway_network", "POST /invitations: existing handle → device-bound")
async def t_mint_existing_handle(ctx: TestContext) -> None:
    from src.gateway.api.network import handle_mint_invitation

    agent_dir = ctx.test_dir / f"gwnet-mintexist-{uuid.uuid4().hex[:6]}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    db_path = agent_dir / "agent.db"
    await _seed_coord(db_path, users=["alessandro"])

    db = await _open_db(db_path)
    try:
        req = _make_request(
            db, method="POST", path="/api/network/invitations",
            body=b'{"handle":"alessandro"}',
        )
        resp = await handle_mint_invitation(req)
        assert resp.status == 201
        import json
        body = json.loads(resp.body)
        assert body["role"] == "device", body
        assert body["bind_to"] == "alessandro", body
        assert "alessandro" in body["intent"], body
    finally:
        await db.close()


@test("gateway_network", "POST /invitations: explicit role overrides auto-detect")
async def t_mint_explicit_role(ctx: TestContext) -> None:
    from src.gateway.api.network import handle_mint_invitation

    agent_dir = ctx.test_dir / f"gwnet-mintexp-{uuid.uuid4().hex[:6]}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    db_path = agent_dir / "agent.db"
    await _seed_coord(db_path, users=["alessandro"])

    db = await _open_db(db_path)
    try:
        req = _make_request(
            db, method="POST", path="/api/network/invitations",
            body=b'{"handle":"alessandro","role":"agent"}',
        )
        resp = await handle_mint_invitation(req)
        assert resp.status == 201
        import json
        body = json.loads(resp.body)
        # Explicit role=agent must not be remapped to device.
        assert body["role"] == "agent", body
    finally:
        await db.close()


@test("gateway_network", "POST /invitations: invalid role → 400")
async def t_mint_invalid_role(ctx: TestContext) -> None:
    from src.gateway.api.network import handle_mint_invitation

    agent_dir = ctx.test_dir / f"gwnet-mintbad-{uuid.uuid4().hex[:6]}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    db_path = agent_dir / "agent.db"
    await _seed_coord(db_path)

    db = await _open_db(db_path)
    try:
        req = _make_request(
            db, method="POST", path="/api/network/invitations",
            body=b'{"role":"superadmin"}',
        )
        resp = await handle_mint_invitation(req)
        assert resp.status == 400, f"expected 400, got {resp.status}: {resp.body!r}"
    finally:
        await db.close()


@test("gateway_network", "DELETE /invitations/{code} is idempotent")
async def t_revoke(ctx: TestContext) -> None:
    from src.gateway.api.network import handle_mint_invitation, handle_revoke_invitation

    agent_dir = ctx.test_dir / f"gwnet-revoke-{uuid.uuid4().hex[:6]}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    db_path = agent_dir / "agent.db"
    await _seed_coord(db_path)

    db = await _open_db(db_path)
    try:
        # Mint, revoke, revoke again. First revoke truthy, second falsy.
        mint_req = _make_request(
            db, method="POST", path="/api/network/invitations",
            body=b'{"handle":"new-friend"}',
        )
        mint_resp = await handle_mint_invitation(mint_req)
        import json
        code = json.loads(mint_resp.body)["code"]

        first = await handle_revoke_invitation(_make_request(
            db, method="DELETE",
            path=f"/api/network/invitations/{code}",
            match_info={"code": code},
        ))
        assert first.status == 200
        assert json.loads(first.body)["revoked"] is True

        second = await handle_revoke_invitation(_make_request(
            db, method="DELETE",
            path=f"/api/network/invitations/{code}",
            match_info={"code": code},
        ))
        assert second.status == 200
        assert json.loads(second.body)["revoked"] is False, \
            "second revoke must be idempotent (already-revoked → revoked=False)"
    finally:
        await db.close()


@test("gateway_network", "member-mode agent returns 404 on /users")
async def t_member_mode_404(ctx: TestContext) -> None:
    """A member-mode agent doesn't own the user table — the HTTP
    surface must say so clearly instead of 500ing or returning empty.
    The desktop app uses the 404 to route the user back to the
    coordinator's own gateway for these operations."""
    from src.gateway.api.network import handle_list_users
    from src.memory.db import MemoryDB
    from src.network.coordinator.store import CoordinatorStore

    agent_dir = ctx.test_dir / f"gwnet-member-{uuid.uuid4().hex[:6]}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    db_path = agent_dir / "agent.db"

    db = MemoryDB(str(db_path))
    await db.connect()
    try:
        store = CoordinatorStore(db)
        await store.set_network_role(
            role="member",
            network_id=uuid.uuid4().hex,
            name="lyra-agent",
            coordinator_node_id="cb" + "0" * 62,
            coordinator_pubkey=b"\x42" * 32,
        )
        req = _make_request(db)
        resp = await handle_list_users(req)
        assert resp.status == 404, f"expected 404, got {resp.status}: {resp.body!r}"
    finally:
        await db.close()
