"""``user_store`` is keyed by ``(name, handle)`` — two handles can join
the same network from one machine.

Regression pin for the invite/network bug: a user-role invite is meant
to be redeemable by anyone at signup, but the store used to be keyed by
network *name* alone. Redeeming the invite under a new handle on a
machine that already joined the network found the prior handle's row
and was rejected client-side ("network X is bound to A, not B") before
the request ever reached the coordinator.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from ._framework import TestContext, test


def _fake_home(ctx: TestContext) -> tuple[Path, str | None]:
    """Point ``Path.home()`` at a throwaway dir so the store writes
    under the test sandbox. Returns the dir + the prior $HOME to
    restore."""
    home = ctx.test_dir / f"userstore-home-{uuid.uuid4().hex[:8]}"
    home.mkdir(parents=True, exist_ok=True)
    old_home = os.environ.get("HOME")
    os.environ["HOME"] = str(home)
    return home, old_home


def _restore_home(old_home: str | None) -> None:
    if old_home is None:
        os.environ.pop("HOME", None)
    else:
        os.environ["HOME"] = old_home


@test("user_store",
      "two handles coexist on one network name — invite redeemable by anyone")
async def t_two_handles_coexist(ctx: TestContext) -> None:
    from src.network import user_store

    _, old_home = _fake_home(ctx)
    try:
        store = user_store.UserStore()
        # Alessandro joins lyra-agent.
        user_store.add_or_update(
            store, name="lyra-agent", network_id="netid-1",
            coordinator_node_id="node-1", coordinator_pubkey_hex="aa" * 32,
            handle="alessandro",
        )
        # Marco redeems the user invite for the SAME network — a new
        # handle, NOT a re-login. Pre-fix this clobbered Alessandro's row.
        user_store.add_or_update(
            store, name="lyra-agent", network_id="netid-1",
            coordinator_node_id="node-1", coordinator_pubkey_hex="aa" * 32,
            handle="marco",
        )
        assert len(store.networks) == 2, (
            f"a distinct handle must be a distinct row, got {len(store.networks)}"
        )

        # Each handle resolves to its own row.
        a = user_store.find(store, "lyra-agent", "alessandro")
        m = user_store.find(store, "lyra-agent", "marco")
        assert a is not None and a.handle == "alessandro"
        assert m is not None and m.handle == "marco"

        # A non-member handle is not found even though the name matches.
        assert user_store.find(store, "lyra-agent", "nobody") is None

        # Handle-less find still works — returns the first matching row.
        first = user_store.find(store, "lyra-agent")
        assert first is not None and first.handle == "alessandro"

        # Survives a save + reload round-trip.
        user_store.save(store)
        reloaded = user_store.load()
        assert len(reloaded.networks) == 2
        assert user_store.find(reloaded, "lyra-agent", "alessandro") is not None
        assert user_store.find(reloaded, "lyra-agent", "marco") is not None
    finally:
        _restore_home(old_home)


@test("user_store",
      "add_or_update on the same (name, handle) updates in place, keeps added_at")
async def t_same_pair_updates_in_place(ctx: TestContext) -> None:
    from src.network import user_store

    _, old_home = _fake_home(ctx)
    try:
        store = user_store.UserStore()
        r1 = user_store.add_or_update(
            store, name="home", network_id="net1",
            coordinator_node_id="n1", coordinator_pubkey_hex="aa" * 32,
            handle="alice",
        )
        added_at = r1.added_at
        r2 = user_store.add_or_update(
            store, name="home", network_id="net2",
            coordinator_node_id="n2", coordinator_pubkey_hex="bb" * 32,
            handle="alice",
        )
        assert len(store.networks) == 1, "same (name, handle) must update in place"
        assert r2.added_at == added_at, "added_at preserved across update"
        assert r2.network_id == "net2", "other fields bumped on update"
    finally:
        _restore_home(old_home)


@test("user_store",
      "remove targets one (name, handle) pair, leaving the other intact")
async def t_remove_targets_handle(ctx: TestContext) -> None:
    from src.network import user_store

    _, old_home = _fake_home(ctx)
    try:
        store = user_store.UserStore()
        for h in ("alessandro", "marco"):
            user_store.add_or_update(
                store, name="lyra-agent", network_id="netid-1",
                coordinator_node_id="node-1", coordinator_pubkey_hex="aa" * 32,
                handle=h,
            )
        assert user_store.remove(store, "lyra-agent", "alessandro") is True
        assert len(store.networks) == 1
        assert store.networks[0].handle == "marco"
        # Removing a (name, handle) pair that isn't there is a no-op.
        assert user_store.remove(store, "lyra-agent", "ghost") is False
    finally:
        _restore_home(old_home)
