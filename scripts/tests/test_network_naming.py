"""Per-agent network name has no decorative suffix; rename is cosmetic.

Two related pins:

1. ``auto_init_if_standalone`` produces a network named after the
   agent itself (``<agent_name>``) — earlier installs landed on
   ``<agent>-personal`` because the bootstrap was framed as a
   "personal network", but the network is just per-agent; the suffix
   was misleading and is now dropped.

2. ``CoordinatorStore.rename_network`` updates only the ``name``
   column. ``network_id`` and the coordinator identity columns must
   stay put — otherwise every paired device would be invalidated.
   This pins that contract so a future "make rename clean up X too"
   refactor can't silently break existing pairings.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from ._framework import TestContext, test


@test("network_naming",
      "auto_init_if_standalone names the network <agent_name> (no -personal suffix)")
async def t_auto_init_no_suffix(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.network.cli_commands import auto_init_if_standalone
    from src.network.coordinator.store import CoordinatorStore

    tmp_dir = ctx.test_dir / f"netname-{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_db = tmp_dir / "agent.db"
    config = {"name": "marco", "memory": {"db_path": str(tmp_db)}}

    row = await auto_init_if_standalone(
        agent_dir=tmp_dir, config=config, quiet=True,
    )
    assert row is not None, "auto_init should have created a network row"
    assert row["role"] == "coordinator", f"role={row['role']!r}"
    assert row["name"] == "marco", (
        f"expected bare agent name, got {row['name']!r} — the -personal "
        "suffix has been removed from the auto-bootstrap path"
    )
    assert row["network_id"], "network_id must be set"

    # Belt-and-braces: re-open the DB and confirm what was persisted.
    db = MemoryDB(str(tmp_db))
    await db.connect()
    try:
        persisted = await CoordinatorStore(db).get_network_role()
        assert persisted is not None
        assert persisted["name"] == "marco"
    finally:
        await db.close()


@test("network_naming",
      "auto_init_if_standalone is idempotent — second call returns the same row, no rename")
async def t_auto_init_idempotent(ctx: TestContext) -> None:
    """If the agent was previously bootstrapped under the old name
    (``<agent>-personal``), a second call must NOT silently rename it
    to ``<agent>`` — that would change the label out from under any
    device whose UI shows the cached network name."""
    from src.memory.db import MemoryDB
    from src.network.cli_commands import auto_init_if_standalone
    from src.network.coordinator.store import CoordinatorStore

    tmp_dir = ctx.test_dir / f"netidem-{uuid.uuid4().hex[:8]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_db = tmp_dir / "agent.db"
    config = {"name": "marco", "memory": {"db_path": str(tmp_db)}}

    # Pre-seed with the OLD naming convention.
    db = MemoryDB(str(tmp_db))
    await db.connect()
    try:
        store = CoordinatorStore(db)
        await store.set_network_role(
            role="coordinator",
            network_id="legacy-id",
            name="marco-personal",
            coordinator_node_id="legacy-node",
            coordinator_pubkey=b"\x00" * 32,
        )
    finally:
        await db.close()

    row = await auto_init_if_standalone(
        agent_dir=tmp_dir, config=config, quiet=True,
    )
    assert row is not None
    assert row["name"] == "marco-personal", (
        f"existing network must not be silently renamed, got {row['name']!r}"
    )
    assert row["network_id"] == "legacy-id", \
        "network_id must be preserved across idempotent re-init"


@test("network_naming",
      "rename_network updates ``name`` only — network_id and identity preserved")
async def t_rename_preserves_identity(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.network.coordinator.store import CoordinatorStore

    tmp_db = ctx.db_path.with_name(f"rename-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(tmp_db))
    try:
        await db.connect()
        store = CoordinatorStore(db)

        original_id = "abc123"
        original_node = "node-xyz"
        original_pubkey = b"\x42" * 32
        await store.set_network_role(
            role="coordinator",
            network_id=original_id,
            name="marco-personal",
            coordinator_node_id=original_node,
            coordinator_pubkey=original_pubkey,
        )

        updated = await store.rename_network("marco")
        assert updated is True, "rename should report a row was updated"

        row = await store.get_network_role()
        assert row is not None
        assert row["name"] == "marco", f"name not updated: {row['name']!r}"
        # Critical: every other column is byte-for-byte identical.
        assert row["network_id"] == original_id, \
            "network_id must NOT change on rename (would invalidate pairings)"
        assert row["coordinator_node_id"] == original_node, \
            "coordinator_node_id must NOT change on rename"
        assert row["coordinator_pubkey"] == original_pubkey, \
            "coordinator_pubkey must NOT change on rename (would invalidate certs)"
        assert row["role"] == "coordinator", \
            f"role must not flip, got {row['role']!r}"
    finally:
        try:
            await db.close()
        except Exception:
            pass
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("network_naming",
      "rename_network on a brand-new DB returns False (nothing to rename)")
async def t_rename_no_network(ctx: TestContext) -> None:
    """The CLI catches this case with a friendly error before calling
    the store, but the store itself must also signal cleanly so a
    direct caller (e.g. an API endpoint) can do the same check."""
    from src.memory.db import MemoryDB
    from src.network.coordinator.store import CoordinatorStore

    tmp_db = ctx.db_path.with_name(f"rename-empty-{uuid.uuid4().hex[:8]}.db")
    db = MemoryDB(str(tmp_db))
    try:
        await db.connect()
        store = CoordinatorStore(db)
        updated = await store.rename_network("anything")
        assert updated is False, \
            "rename on empty network table should return False, not raise"
    finally:
        try:
            await db.close()
        except Exception:
            pass
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass
