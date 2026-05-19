"""``openagent invite [HANDLE]`` auto-picks the right protocol role.

User-facing UX simplification (2026-05-19): the CLI no longer exposes
``--role user|device|agent`` as the primary surface. The operator
just types ``openagent invite [HANDLE]`` and the CLI picks under the
hood:

  - no HANDLE                 → user-role, open invite
  - HANDLE that doesn't exist → user-role, onboarding invite
  - HANDLE that exists        → device-role bound to HANDLE

``--role`` is still accepted (hidden, advanced) for the federated-
agent registration case and any future power-user need. Existing
device invites already in the wild keep working — the change is in
how new invites are minted, not how they're consumed.

This test exercises the auto-pick logic by driving the underlying
``_run_invite`` function the CLI delegates to.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path

from ._framework import TestContext, test


@dataclass
class _StubCtx:
    """Minimal stand-in for Click's ``ctx`` carrying just ``obj['config']``.

    ``_run_invite`` only needs ``ctx.obj["config"]`` (and inside,
    ``config['memory']['db_path']``); everything else on the real
    Click ctx is unused here."""
    obj: dict


def _make_ctx(db_path: Path, agent_dir: Path) -> _StubCtx:
    """Build a ctx + activate the agent dir. ``_run_invite`` reads
    the agent dir via ``core_paths.get_agent_dir()`` (so identity.key
    + addr cache live in the right place); the public setter handles
    the bookkeeping without us reaching into module-private state."""
    from src.core import paths as core_paths

    core_paths.set_agent_dir(agent_dir)
    return _StubCtx(obj={
        "config": {"memory": {"db_path": str(db_path)}},
    })


async def _seed_network_and_user(
    db_path: Path, *, existing_user: str | None = None,
):
    """Set up a coordinator-mode network and optionally pre-register
    one user — so we can drive both the 'handle exists' and 'handle
    doesn't exist' branches."""
    from src.memory.db import MemoryDB
    from src.network.coordinator.store import CoordinatorStore

    db = MemoryDB(str(db_path))
    await db.connect()
    try:
        store = CoordinatorStore(db)
        await store.set_network_role(
            role="coordinator",
            network_id=uuid.uuid4().hex,
            name="lyra-agent",
            coordinator_node_id="cb" + "0" * 62,
            coordinator_pubkey=b"\x00" * 32,
        )
        if existing_user:
            await store.create_user(
                handle=existing_user,
                pake_record=b"\x00" * 64,
                pake_algo="srp6a",
            )
    finally:
        await db.close()


async def _read_last_invite(db_path: Path) -> dict | None:
    """Pull the most recent invite row from the test DB so we can
    assert what the CLI just minted."""
    import aiosqlite

    async with aiosqlite.connect(str(db_path)) as conn:
        cur = await conn.execute(
            "SELECT code, role, bind_to_handle FROM network_invitations "
            "ORDER BY created_at DESC LIMIT 1"
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return {"code": row[0], "role": row[1], "bind_to_handle": row[2]}


@test("invite_smart", "no handle → user role, no bind_to (open invite)")
async def t_no_handle(ctx: TestContext) -> None:
    from src.network.cli_commands import _run_invite

    agent_dir = ctx.test_dir / f"invsmart-no-{uuid.uuid4().hex[:6]}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    db_path = agent_dir / "agent.db"
    await _seed_network_and_user(db_path)

    cli_ctx = _make_ctx(db_path, agent_dir)
    await _run_invite(cli_ctx, handle=None, role=None, ttl=600, uses=1)

    inv = await _read_last_invite(db_path)
    assert inv is not None, "invite row should have been written"
    assert inv["role"] == "user", f"expected user role, got {inv['role']!r}"
    assert inv["bind_to_handle"] in (None, ""), \
        f"open invite must not be bound: {inv['bind_to_handle']!r}"


@test("invite_smart", "new handle → user role, no bind_to (onboarding invite)")
async def t_new_handle(ctx: TestContext) -> None:
    from src.network.cli_commands import _run_invite

    agent_dir = ctx.test_dir / f"invsmart-new-{uuid.uuid4().hex[:6]}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    db_path = agent_dir / "agent.db"
    await _seed_network_and_user(db_path)  # no existing user

    cli_ctx = _make_ctx(db_path, agent_dir)
    await _run_invite(cli_ctx, handle="marco", role=None, ttl=600, uses=1)

    inv = await _read_last_invite(db_path)
    assert inv is not None
    assert inv["role"] == "user", \
        f"non-existing handle should mint user invite, got {inv['role']!r}"
    # bind_to=marco is *advisory* (the new user just picks marco when
    # they register); the protocol role is user, and the user-role
    # register path doesn't honor bind_to. We don't pin bind_to here
    # because it's not load-bearing for user-role invites — but the
    # role MUST be user, never device, because marco doesn't exist
    # and a device invite for a non-existent handle is unusable.


@test("invite_smart", "existing handle → device role bound to it (pairing invite)")
async def t_existing_handle(ctx: TestContext) -> None:
    from src.network.cli_commands import _run_invite

    agent_dir = ctx.test_dir / f"invsmart-exist-{uuid.uuid4().hex[:6]}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    db_path = agent_dir / "agent.db"
    await _seed_network_and_user(db_path, existing_user="alessandro")

    cli_ctx = _make_ctx(db_path, agent_dir)
    await _run_invite(cli_ctx, handle="alessandro", role=None, ttl=600, uses=1)

    inv = await _read_last_invite(db_path)
    assert inv is not None
    assert inv["role"] == "device", (
        f"existing handle should mint device invite, got {inv['role']!r}"
    )
    assert inv["bind_to_handle"] == "alessandro", (
        f"device invite must bind to the existing handle, "
        f"got {inv['bind_to_handle']!r}"
    )


@test("invite_smart", "explicit --role overrides auto-detect (power-user path)")
async def t_explicit_role(ctx: TestContext) -> None:
    """``--role agent`` (or any explicit role) must NOT be remapped
    by the auto-detect — the operator is signalling intent."""
    from src.network.cli_commands import _run_invite

    agent_dir = ctx.test_dir / f"invsmart-explicit-{uuid.uuid4().hex[:6]}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    db_path = agent_dir / "agent.db"
    await _seed_network_and_user(db_path, existing_user="alessandro")

    cli_ctx = _make_ctx(db_path, agent_dir)
    # ``role=agent`` with handle=alessandro should be honored even
    # though auto-detect would have picked device. The operator is
    # explicitly registering an agent owned by alessandro.
    await _run_invite(
        cli_ctx, handle="alessandro", role="agent", ttl=600, uses=1,
    )

    inv = await _read_last_invite(db_path)
    assert inv is not None
    assert inv["role"] == "agent", (
        f"explicit --role=agent must not be auto-remapped, got {inv['role']!r}"
    )


@test("invite_smart", "handle case + whitespace are normalised before lookup")
async def t_handle_normalisation(ctx: TestContext) -> None:
    """Operators paste handles with mixed case / trailing whitespace
    from the users list ("Alessandro\\n", "ALESSANDRO"). The lookup
    must normalise so ``alessandro`` is found and the device branch
    fires. Otherwise we'd silently downgrade to a user invite for
    every non-lowercase paste."""
    from src.network.cli_commands import _run_invite

    agent_dir = ctx.test_dir / f"invsmart-norm-{uuid.uuid4().hex[:6]}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    db_path = agent_dir / "agent.db"
    await _seed_network_and_user(db_path, existing_user="alessandro")

    cli_ctx = _make_ctx(db_path, agent_dir)
    await _run_invite(
        cli_ctx, handle="  ALESSANDRO  ", role=None, ttl=600, uses=1,
    )

    inv = await _read_last_invite(db_path)
    assert inv is not None
    assert inv["role"] == "device", \
        f"case/whitespace-bearing handle must still match: {inv}"
    assert inv["bind_to_handle"] == "alessandro"
