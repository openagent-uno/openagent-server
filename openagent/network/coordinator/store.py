"""SQLite-backed store for the coordinator: users, devices, agents, invitations.

Reuses the agent's existing ``MemoryDB`` connection — we don't open a
second SQLite handle. All methods are async and take/return plain
dicts so the JSON-RPC service layer can serialise them without
adapter glue.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass

import aiosqlite

from openagent.memory.db import MemoryDB


INVITE_CODE_BYTES = 12  # 96 bits — plenty of entropy, ~20 base32 chars


def _gen_invite_code() -> str:
    # base32, no padding, lowercase. Hyphens added every 4 chars purely
    # for human display — the DB stores the raw form.
    import base64

    raw = secrets.token_bytes(INVITE_CODE_BYTES)
    return base64.b32encode(raw).decode("ascii").rstrip("=").lower()


def _format_invite_for_display(code: str) -> str:
    return "-".join(code[i:i + 4] for i in range(0, len(code), 4))


def _normalize_invite_code(maybe_pretty: str) -> str:
    return maybe_pretty.strip().lower().replace("-", "").replace(" ", "")


@dataclass
class UserRow:
    handle: str
    pake_record: bytes
    pake_algo: str
    status: str
    created_at: float


@dataclass
class DeviceRow:
    device_pubkey: bytes
    user_handle: str
    label: str | None
    status: str
    added_at: float
    last_seen: float | None


@dataclass
class AgentRow:
    handle: str
    node_id: str
    label: str | None
    owner_handle: str
    added_at: float
    last_seen: float | None


@dataclass
class InvitationRow:
    code: str
    role: str
    created_by: str | None
    bind_to_handle: str | None
    uses_left: int
    expires_at: float
    created_at: float
    used_at: float | None

    @property
    def display_code(self) -> str:
        return _format_invite_for_display(self.code)


class CoordinatorStore:
    """All coordinator-side persistence in one place.

    Methods are pretty thin SQL wrappers — the service module owns the
    business logic (PAKE protocol state, cert minting, rate limiting).
    Keeping the store dumb makes each piece testable in isolation.
    """

    def __init__(self, db: MemoryDB) -> None:
        self._db = db

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self._db._conn is None:
            raise RuntimeError("MemoryDB.connect() must be called before CoordinatorStore use")
        return self._db._conn

    # ── network row (singleton) ────────────────────────────────────

    async def get_network_role(self) -> dict | None:
        cur = await self._conn.execute(
            "SELECT role, network_id, name, coordinator_node_id, coordinator_pubkey, created_at "
            "FROM network WHERE singleton=1",
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return dict(row)

    async def set_network_role(
        self,
        *,
        role: str,
        network_id: str | None = None,
        name: str | None = None,
        coordinator_node_id: str | None = None,
        coordinator_pubkey: bytes | None = None,
    ) -> None:
        await self._conn.execute(
            "INSERT INTO network (singleton, role, network_id, name, coordinator_node_id, "
            "coordinator_pubkey, created_at) VALUES (1, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(singleton) DO UPDATE SET role=excluded.role, "
            "network_id=excluded.network_id, name=excluded.name, "
            "coordinator_node_id=excluded.coordinator_node_id, "
            "coordinator_pubkey=excluded.coordinator_pubkey",
            (role, network_id, name, coordinator_node_id, coordinator_pubkey, time.time()),
        )
        await self._conn.commit()

    # ── users ──────────────────────────────────────────────────────

    async def get_user(self, handle: str) -> UserRow | None:
        cur = await self._conn.execute(
            "SELECT handle, pake_record, pake_algo, status, created_at "
            "FROM network_users WHERE handle=?",
            (handle,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return UserRow(**dict(row))

    async def create_user(self, *, handle: str, pake_record: bytes, pake_algo: str) -> None:
        await self._conn.execute(
            "INSERT INTO network_users (handle, pake_record, pake_algo, status, created_at) "
            "VALUES (?, ?, ?, 'active', ?)",
            (handle, pake_record, pake_algo, time.time()),
        )
        await self._conn.commit()

    async def list_users(self) -> list[UserRow]:
        cur = await self._conn.execute(
            "SELECT handle, pake_record, pake_algo, status, created_at "
            "FROM network_users ORDER BY created_at",
        )
        return [UserRow(**dict(row)) for row in await cur.fetchall()]

    # ── devices ────────────────────────────────────────────────────

    async def add_device(
        self,
        *,
        device_pubkey: bytes,
        user_handle: str,
        label: str | None = None,
    ) -> None:
        await self._conn.execute(
            "INSERT INTO network_devices (device_pubkey, user_handle, label, status, added_at) "
            "VALUES (?, ?, ?, 'active', ?) "
            "ON CONFLICT(device_pubkey) DO UPDATE SET user_handle=excluded.user_handle, "
            "label=excluded.label, status='active'",
            (device_pubkey, user_handle, label, time.time()),
        )
        await self._conn.commit()

    async def user_has_devices(self, user_handle: str) -> bool:
        """True if at least one device is bound to *user_handle*.

        Used by login_finish to distinguish a freshly-registered user
        (no devices yet → first device pairs without re-spending the
        registration invite) from a returning user adding a second
        device (must present a fresh device-role invite).
        """
        cur = await self._conn.execute(
            "SELECT 1 FROM network_devices WHERE user_handle=? LIMIT 1",
            (user_handle,),
        )
        return await cur.fetchone() is not None

    async def get_device(self, device_pubkey: bytes) -> DeviceRow | None:
        cur = await self._conn.execute(
            "SELECT device_pubkey, user_handle, label, status, added_at, last_seen "
            "FROM network_devices WHERE device_pubkey=?",
            (device_pubkey,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        return DeviceRow(**dict(row))

    async def revoke_device(self, device_pubkey: bytes) -> bool:
        cur = await self._conn.execute(
            "UPDATE network_devices SET status='revoked' WHERE device_pubkey=? AND status='active'",
            (device_pubkey,),
        )
        await self._conn.commit()
        return cur.rowcount > 0

    async def list_revoked_pubkeys(self) -> set[bytes]:
        cur = await self._conn.execute(
            "SELECT device_pubkey FROM network_devices WHERE status='revoked'",
        )
        return {bytes(row[0]) for row in await cur.fetchall()}

    async def touch_device(self, device_pubkey: bytes) -> None:
        await self._conn.execute(
            "UPDATE network_devices SET last_seen=? WHERE device_pubkey=?",
            (time.time(), device_pubkey),
        )
        await self._conn.commit()

    # ── agents ─────────────────────────────────────────────────────

    async def register_agent(
        self,
        *,
        handle: str,
        node_id: str,
        owner_handle: str,
        label: str | None = None,
    ) -> None:
        await self._conn.execute(
            "INSERT INTO network_agents (handle, node_id, label, owner_handle, added_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(handle) DO UPDATE SET node_id=excluded.node_id, "
            "label=excluded.label, owner_handle=excluded.owner_handle",
            (handle, node_id, label, owner_handle, time.time()),
        )
        await self._conn.commit()

    async def list_agents(self) -> list[AgentRow]:
        cur = await self._conn.execute(
            "SELECT handle, node_id, label, owner_handle, added_at, last_seen "
            "FROM network_agents ORDER BY added_at",
        )
        return [AgentRow(**dict(row)) for row in await cur.fetchall()]

    async def remove_agent(self, handle: str) -> bool:
        cur = await self._conn.execute(
            "DELETE FROM network_agents WHERE handle=?",
            (handle,),
        )
        await self._conn.commit()
        return cur.rowcount > 0

    # ── invitations ────────────────────────────────────────────────

    async def create_invitation(
        self,
        *,
        role: str,
        created_by: str | None,
        ttl_seconds: int = 7 * 24 * 3600,
        uses: int = 1,
        bind_to_handle: str | None = None,
    ) -> InvitationRow:
        if role not in ("user", "device", "agent"):
            raise ValueError(f"unknown invite role: {role}")
        code = _gen_invite_code()
        now = time.time()
        await self._conn.execute(
            "INSERT INTO network_invitations (code, role, created_by, bind_to_handle, "
            "uses_left, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (code, role, created_by, bind_to_handle, uses, now + ttl_seconds, now),
        )
        await self._conn.commit()
        return InvitationRow(
            code=code, role=role, created_by=created_by, bind_to_handle=bind_to_handle,
            uses_left=uses, expires_at=now + ttl_seconds, created_at=now, used_at=None,
        )

    async def consume_invitation(self, code_input: str) -> InvitationRow | None:
        """Decrement uses_left on a valid invite. Returns the row if redeemed."""
        code = _normalize_invite_code(code_input)
        cur = await self._conn.execute(
            "SELECT code, role, created_by, bind_to_handle, uses_left, expires_at, "
            "created_at, used_at FROM network_invitations WHERE code=?",
            (code,),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        invite = InvitationRow(**dict(row))
        if invite.uses_left <= 0:
            return None
        if invite.expires_at < time.time():
            return None
        await self._conn.execute(
            "UPDATE network_invitations SET uses_left=uses_left-1, "
            "used_at=COALESCE(used_at, ?) WHERE code=?",
            (time.time(), code),
        )
        await self._conn.commit()
        invite.uses_left -= 1
        invite.used_at = invite.used_at or time.time()
        return invite

    async def list_invitations(self, *, include_expired: bool = False) -> list[InvitationRow]:
        if include_expired:
            cur = await self._conn.execute(
                "SELECT code, role, created_by, bind_to_handle, uses_left, expires_at, "
                "created_at, used_at FROM network_invitations ORDER BY created_at DESC",
            )
        else:
            cur = await self._conn.execute(
                "SELECT code, role, created_by, bind_to_handle, uses_left, expires_at, "
                "created_at, used_at FROM network_invitations "
                "WHERE expires_at>=? ORDER BY created_at DESC",
                (time.time(),),
            )
        return [InvitationRow(**dict(row)) for row in await cur.fetchall()]
