"""Coordinator ``_m_login_finish`` is resilient to non-critical DB locks.

Real-world failure mode (2026-05-18, lyra-agent): heavy runtime session
writes held the SQLite writer lock for an extended period. The
coordinator's ``login_finish`` happens to call ``store.touch_device``
to update ``last_seen`` — a purely cosmetic field — and the lock made
that write raise ``sqlite3.OperationalError: database is locked``,
which propagated and broke login for every device returning to an
already-registered account. The cert is mintable from in-memory state
alone, so a transient SQLite error on the touch should never take
auth down.

This test pins that contract: when ``touch_device`` raises a lock
error, login_finish still returns a valid cert. As a counter-test, we
also verify the new-device path still propagates the same error (we
can't quietly skip recording the device — that would issue a cert
against an unknown device row).
"""
from __future__ import annotations

import sqlite3
import time
import uuid

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ._framework import TestContext, test


class _FakeStore:
    """Hand-rolled stand-in for CoordinatorStore.

    Each method is a tiny coroutine that returns canned values. The
    flags / hooks make it easy to drive specific branches in
    ``_m_login_finish`` (existing active device vs. new device) and to
    simulate SQLite lock failures on the writes that matter.
    """

    def __init__(
        self,
        *,
        existing_device,
        touch_device_raises: Exception | None = None,
        add_device_raises: Exception | None = None,
        user_has_other_devices: bool = False,
    ) -> None:
        self._existing_device = existing_device
        self._touch_device_raises = touch_device_raises
        self._add_device_raises = add_device_raises
        self._user_has_other_devices = user_has_other_devices
        self.touch_device_called = False
        self.add_device_called = False

    async def get_device(self, device_pubkey):
        return self._existing_device

    async def user_has_devices(self, handle):
        return self._user_has_other_devices

    async def touch_device(self, device_pubkey):
        self.touch_device_called = True
        if self._touch_device_raises is not None:
            raise self._touch_device_raises

    async def add_device(self, *, device_pubkey, user_handle, label=None):
        self.add_device_called = True
        if self._add_device_raises is not None:
            raise self._add_device_raises

    async def consume_invitation(self, code):
        return None  # never reached in the paths we test


class _FakePake:
    """Stub PakeBackend that accepts any proof and returns canned M2."""

    algo = "srp6a"

    def login_finish(self, state, ke3):
        return b"m2-fake-server-proof"


def _make_login_state(handle: str):
    from src.network.coordinator.pake import LoginInProgress

    # ``SRPServerSession`` isn't easily mockable, but our fake pake
    # doesn't touch ``state.session`` at all, so any sentinel works.
    return LoginInProgress(handle=handle, session=object(), created_at=time.time())


def _make_service(store):
    from src.network.coordinator.service import CoordinatorService
    from src.network.coordinator.store import DeviceRow

    coord_key = Ed25519PrivateKey.generate()
    svc = CoordinatorService(
        store=store,
        coordinator_key=coord_key,
        network_id="test-net-" + uuid.uuid4().hex[:8],
        network_name="test-net",
        pake_backend=_FakePake(),
    )
    return svc, coord_key


@test("coord_login_resilience", "touch_device DB-lock doesn't break login for existing device")
async def t_touch_device_locked_login_still_succeeds(ctx: TestContext) -> None:
    from src.network.auth.device_cert import verify_cert
    from src.network.coordinator.store import DeviceRow

    device_pubkey = b"\x00" * 32
    handle = "alessandro"
    existing = DeviceRow(
        device_pubkey=device_pubkey,
        user_handle=handle,
        label=None,
        status="active",
        added_at=time.time() - 1000,
        last_seen=time.time() - 500,
    )
    store = _FakeStore(
        existing_device=existing,
        touch_device_raises=sqlite3.OperationalError("database is locked"),
    )
    svc, coord_key = _make_service(store)

    state_id = "state-" + uuid.uuid4().hex
    svc._logins[state_id] = _make_login_state(handle)

    params = {
        "state_id": state_id,
        "ke3": b"\x00" * 64,
        "device_pubkey": device_pubkey,
    }
    result = await svc._m_login_finish(params, peer_node_id="inproc:test")

    assert store.touch_device_called, "touch_device should have been attempted"
    assert "cert" in result, "no cert returned despite touch_device being non-critical"
    assert result["m2"] == b"m2-fake-server-proof"

    cert = verify_cert(
        result["cert"],
        coordinator_pubkey=coord_key.public_key(),
        expected_network_id=svc._cfg.network_id,
    )
    assert cert.handle == handle
    assert cert.device_pubkey == device_pubkey


@test("coord_login_resilience", "add_device DB-lock still fails login (critical write)")
async def t_add_device_locked_login_fails(ctx: TestContext) -> None:
    """Counter-test: we deliberately do NOT swallow lock errors on
    ``add_device`` — without that row the device isn't actually paired
    and the cert would point at a phantom record. The error must
    propagate so the client retries cleanly."""

    handle = "marco"
    device_pubkey = b"\x11" * 32
    store = _FakeStore(
        existing_device=None,  # first-device pairing
        user_has_other_devices=False,
        add_device_raises=sqlite3.OperationalError("database is locked"),
    )
    svc, _coord_key = _make_service(store)

    state_id = "state-" + uuid.uuid4().hex
    svc._logins[state_id] = _make_login_state(handle)

    params = {
        "state_id": state_id,
        "ke3": b"\x00" * 64,
        "device_pubkey": device_pubkey,
    }
    raised: Exception | None = None
    try:
        await svc._m_login_finish(params, peer_node_id="inproc:test")
    except sqlite3.OperationalError as e:
        raised = e
    assert raised is not None, "add_device lock error must propagate"
    assert store.add_device_called, "add_device should have been attempted"
