"""Cross-device session visibility — handle-keyed listing + legacy fallback.

Pre-fix the gateway filtered ``agno_sessions`` by the WebSocket's
``client_id`` = device pubkey hex. Every device a user owned had a
different pubkey, so device B logged in as the same user could never
see device A's chats. The fix has two halves:

1. ``upsert_session`` now writes the user handle into ``metadata.client_id``
   so the row's primary owner is user-scoped.
2. ``list_all_sessions`` accepts rows whose ``metadata.client_id`` is a
   device pubkey bound to the same handle in ``network_devices`` —
   a soft-fallback for sessions persisted before the fix landed.
"""
from __future__ import annotations

import time as _time
import uuid

from ._framework import TestContext, test


async def _seed_device_binding(db, handle: str, pubkey_hex: str) -> None:
    """Insert a ``network_devices`` row binding a device pubkey to handle."""
    conn = await db._ensure_connected()
    await conn.execute(
        "INSERT OR REPLACE INTO network_users "
        "(handle, pake_record, pake_algo, status, created_at) "
        "VALUES (?, ?, 'srp6a', 'active', ?)",
        (handle, b"", _time.time()),
    )
    await conn.execute(
        "INSERT OR REPLACE INTO network_devices "
        "(device_pubkey, user_handle, label, status, added_at) "
        "VALUES (?, ?, ?, 'active', ?)",
        (bytes.fromhex(pubkey_hex), handle, "test-device", _time.time()),
    )
    await conn.commit()


@test("sessions_cross_device", "list_all_sessions matches by handle")
async def t_match_by_handle(ctx: TestContext) -> None:
    from openagent.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"xd-handle-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        sid = f"xd-{uuid.uuid4().hex[:8]}"
        await db.upsert_session(sid, client_id="alice", title="Hi", framework="agno")
        rows = await db.list_all_sessions("alice", limit=50)
        assert any(r["session_id"] == sid for r in rows), rows
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("sessions_cross_device", "legacy device-pubkey rows resolve via network_devices")
async def t_legacy_pubkey_via_devices(ctx: TestContext) -> None:
    """A pre-fix row carries ``metadata.client_id = <pubkey_hex_B>``. The
    user later pairs device B (so ``network_devices`` binds pubkey_B to
    handle_A). Listing by handle_A must surface that row even though
    its ``client_id`` is still the pubkey form."""
    from openagent.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"xd-legacy-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        # Pubkey bytes printed as 64 hex chars — what the auth layer
        # stamps onto requests.
        pubkey_b = uuid.uuid4().hex + uuid.uuid4().hex
        await _seed_device_binding(db, "handle_A", pubkey_b)
        # Legacy row: client_id is the device pubkey, not the handle.
        sid_legacy = f"legacy-{uuid.uuid4().hex[:8]}"
        await db.upsert_session(
            sid_legacy, client_id=pubkey_b, title="legacy", framework="claude-cli",
        )
        # Foreign row: a pubkey NOT bound to handle_A. Must NOT leak.
        pubkey_other = uuid.uuid4().hex + uuid.uuid4().hex
        sid_foreign = f"foreign-{uuid.uuid4().hex[:8]}"
        await db.upsert_session(
            sid_foreign, client_id=pubkey_other, title="foreign", framework="agno",
        )
        rows = await db.list_all_sessions("handle_A", limit=50)
        ids = {r["session_id"] for r in rows}
        assert sid_legacy in ids, ids
        assert sid_foreign not in ids, ids
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass


@test("sessions_cross_device", "upsert_session keeps device_id alongside handle owner")
async def t_device_id_preserved(ctx: TestContext) -> None:
    """Per-device routing (sticky retries, WS reconnect) still needs the
    device pubkey. The fix stores the user handle in
    ``metadata.client_id`` AND the originating device in
    ``metadata.device_id`` so both pieces of routing information
    survive a process restart."""
    from openagent.memory.db import MemoryDB

    tmp_db = ctx.db_path.with_name(f"xd-devid-{uuid.uuid4().hex[:8]}.db")
    try:
        db = MemoryDB(str(tmp_db))
        await db.connect()
        sid = f"sid-{uuid.uuid4().hex[:8]}"
        pubkey = uuid.uuid4().hex + uuid.uuid4().hex
        await db.upsert_session(
            sid, client_id="alice", device_id=pubkey, framework="agno",
        )
        s = await db.get_session(sid)
        assert s is not None
        assert s["client_id"] == "alice", s
        # ``get_session`` only surfaces a small projection; read the raw
        # metadata to confirm device_id landed.
        conn = await db._ensure_connected()
        cur = await conn.execute(
            "SELECT metadata FROM agno_sessions WHERE session_id = ?",
            (sid,),
        )
        row = await cur.fetchone()
        meta = db._parse_metadata(row[0])
        assert meta.get("device_id") == pubkey, meta
        await db.close()
    finally:
        try:
            tmp_db.unlink()
        except FileNotFoundError:
            pass
