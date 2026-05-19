"""``list_agents`` returns the coordinator's own agent FIRST.

Real-world failure on lyra (2026-05-19): the lyra-agent network had
``virgil`` registered as an agent (NodeId pointing at a *different*
network's coordinator). ``list_agents`` ordered by ``added_at``, so
``virgil`` (registered earlier) came back first. The Electron app
picks ``agents[0]`` for its WS target, dialed virgil's gateway with
a cert for the lyra-agent network, and got a 401 — surfacing as
``WebSocket closed before authentication (code=1006)`` on the
client side.

The fix sorts the coordinator's own agent (``node_id ==
coordinator_node_id``) to the front. Foreign-network rows stay
visible (operator inspection still works via the same API) but
they can't accidentally become the default pick.
"""
from __future__ import annotations

import uuid

from ._framework import TestContext, test


@test("list_agents_ordering",
      "agent whose NodeId == coordinator NodeId is returned first")
async def t_coord_agent_first(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.network.coordinator.store import CoordinatorStore

    tmp_db = ctx.db_path.with_name(f"list-agents-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(tmp_db))
    try:
        await db.connect()
        store = CoordinatorStore(db)

        coord_node = "cb359e6415fdd19fa781666f37995b747aa540a99eec13e9e2fa27dcf295d1ff"
        foreign_node = "ff8973688fc639cf430d717dcdd97b6f1d74444b2197be8c7a2fbecdfc36f0aa"

        await store.set_network_role(
            role="coordinator",
            network_id="net-uuid-1",
            name="lyra-agent",
            coordinator_node_id=coord_node,
            coordinator_pubkey=b"\x00" * 32,
        )

        # Register the FOREIGN agent first (lyra's bad state on 2026-05-19).
        await store.register_agent(
            handle="virgil", node_id=foreign_node,
            owner_handle="system", label="Virgil",
        )
        # Then the coordinator's own agent.
        await store.register_agent(
            handle="lyra-agent", node_id=coord_node,
            owner_handle="system", label="Lyra Agent",
        )

        agents = await store.list_agents()
        handles = [a.handle for a in agents]
        assert handles[0] == "lyra-agent", (
            f"coordinator's own agent must be first, got order: {handles}"
        )
        assert set(handles) == {"lyra-agent", "virgil"}, \
            f"both rows must remain visible: {handles}"
    finally:
        try:
            await db.close()
        except Exception:
            pass
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("list_agents_ordering",
      "single-agent network: only row returned, no ordering hazard")
async def t_only_one_agent(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.network.coordinator.store import CoordinatorStore

    tmp_db = ctx.db_path.with_name(f"single-agent-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(tmp_db))
    try:
        await db.connect()
        store = CoordinatorStore(db)

        coord_node = "abc" * 21 + "x"  # 64 chars
        await store.set_network_role(
            role="coordinator", network_id="net2", name="solo",
            coordinator_node_id=coord_node, coordinator_pubkey=b"\x11" * 32,
        )
        await store.register_agent(
            handle="solo", node_id=coord_node, owner_handle="system",
        )

        agents = await store.list_agents()
        assert len(agents) == 1
        assert agents[0].handle == "solo"
        assert agents[0].node_id == coord_node
    finally:
        try:
            await db.close()
        except Exception:
            pass
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("list_agents_ordering",
      "no network row yet: list_agents still works, no preferred ordering")
async def t_no_network_row(ctx: TestContext) -> None:
    """If somehow ``list_agents`` is called before ``set_network_role``,
    we fall back to a stable order rather than crashing. The 'coord
    NodeId' is the empty string, so no row matches the priority clause."""
    from src.memory.db import MemoryDB
    from src.network.coordinator.store import CoordinatorStore

    tmp_db = ctx.db_path.with_name(f"no-net-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(tmp_db))
    try:
        await db.connect()
        store = CoordinatorStore(db)

        await store.register_agent(
            handle="a", node_id="node-a", owner_handle="system",
        )
        await store.register_agent(
            handle="b", node_id="node-b", owner_handle="system",
        )

        agents = await store.list_agents()
        assert {a.handle for a in agents} == {"a", "b"}, \
            "all rows must come back even with no network row set"
    finally:
        try:
            await db.close()
        except Exception:
            pass
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass
