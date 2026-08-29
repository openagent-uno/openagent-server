"""Durable content-addressed attachment storage and ACL tests."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace

from ._framework import TestContext, test


class _DiskDB:
    def __init__(self, path: Path) -> None:
        self.db_path = str(path)


def _artifact_db(root: Path) -> tuple[_DiskDB, str]:
    db_path = root / "agent.db"
    schema = (
        Path(__file__).parents[2]
        / "src/memory/operational/sql/operational_storage_v2.sql"
    ).read_text(encoding="utf-8")
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema)
        row = conn.execute(
            "SELECT db_instance_id FROM operational_storage_state WHERE singleton_id=1"
        ).fetchone()
        conn.commit()
    finally:
        conn.close()
    assert row is not None
    return _DiskDB(db_path), f"installation:{row[0]}"


def _access(tenant: str, handle: str):
    from src.memory.operational.access import AccessContext

    principal = f"user:{handle}"
    return AccessContext(
        tenant_id=tenant,
        principal_id=principal,
        principal_type="user",
        handle=handle,
        device_id=f"device-{handle}",
        principal_ids=frozenset({principal, f"device:device-{handle}"}),
        grant_identities=frozenset({("user", handle)}),
    )


def _seed_session(
    db: _DiskDB,
    tenant: str,
    session_id: str,
    handle: str,
    *,
    visibility: str = "private",
) -> None:
    now = 1_700_000_000_000
    conn = sqlite3.connect(db.db_path)
    try:
        conn.execute(
            "INSERT INTO sessions_v2 "
            "(id,tenant_id,owner_principal_id,owner_handle_snapshot,visibility,acl_version,"
            "session_type,kind,status,completeness,source_version,metadata_json,created_at_ms,"
            "updated_at_ms,last_activity_at_ms) VALUES (?,?,?,?,?,1,"
            "'agent','chat','active','complete',1,'{}',?,?,?)",
            (
                session_id, tenant, f"user:{handle}", handle,
                visibility,
                now, now, now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_message(
    db: _DiskDB,
    tenant: str,
    session_id: str,
    message_id: str,
    *,
    sequence: int,
    role: str = "user",
) -> None:
    now = 1_700_000_000_000 + sequence
    author_kind = "user" if role == "user" else "agent"
    principal = "user:alice" if role == "user" else "agent:openagent"
    conn = sqlite3.connect(db.db_path)
    try:
        conn.execute(
            "INSERT INTO session_messages "
            "(id,tenant_id,session_id,run_id,sequence,ordinal,role,status,author_kind,"
            "author_principal_id,text,visibility,source_version,completeness,raw_envelope_json,"
            "raw_envelope_schema,legacy_inferred,created_at_ms,updated_at_ms,completed_at_ms) "
            "VALUES (?,?,?,NULL,?,0,?,'complete',?,?,?,'user_visible',1,'complete','{}',1,0,?,?,?)",
            (
                message_id, tenant, session_id, sequence, role, author_kind,
                principal, f"message {sequence}", now, now, now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


@test("artifacts", "CAS deduplicates bytes while preserving attachment order and links")
async def t_artifact_cas_dedup_and_order(ctx: TestContext) -> None:
    from src.memory.artifacts import (
        artifact_row,
        normalize_inbound_attachments,
        persist_output_attachments,
    )

    with tempfile.TemporaryDirectory(prefix="oa-artifacts-test-") as raw:
        root = Path(raw)
        db, tenant = _artifact_db(root)
        first = root / "one.json"
        second = root / "two.json"
        payload = b'{"hello":"world"}'
        first.write_bytes(payload)
        second.write_bytes(payload)
        alice = _access(tenant, "alice")
        _seed_session(db, tenant, "tg:alice", "alice")

        refs = await normalize_inbound_attachments(
            db,
            (
                {"path": str(first), "filename": "../one.json"},
                {"path": str(second), "filename": "two.json"},
            ),
            session_id="tg:alice",
            principal=alice,
            allow_local_paths=True,
        )
        assert [item["filename"] for item in refs] == ["one.json", "two.json"]
        assert refs[0]["artifact_id"] == refs[1]["artifact_id"]
        assert refs[0]["artifact_link_id"] != refs[1]["artifact_link_id"]
        assert Path(refs[0]["path"]).read_bytes() == payload

        # Re-normalising a canonical ref is idempotent for the same relation.
        again = await normalize_inbound_attachments(
            db, (refs[0],), session_id="tg:alice", principal=alice,
        )
        assert again[0]["artifact_link_id"] == refs[0]["artifact_link_id"]

        output = await persist_output_attachments(
            db, "tg:alice", (refs[0],), principal=alice,
        )
        assert output[0]["artifact_id"] == refs[0]["artifact_id"]
        assert output[0]["artifact_link_id"] != refs[0]["artifact_link_id"]
        row, cas_path = await artifact_row(db, refs[0]["artifact_id"])
        assert row["ref_count"] == 3, row
        assert cas_path == Path(refs[0]["path"])


@test("artifacts", "artifact ACL is owner-only and fails closed across principals")
async def t_artifact_acl(ctx: TestContext) -> None:
    from src.memory.artifacts import (
        ArtifactNotFound,
        artifact_is_visible,
        normalize_inbound_attachments,
    )

    with tempfile.TemporaryDirectory(prefix="oa-artifacts-acl-") as raw:
        root = Path(raw)
        db, tenant = _artifact_db(root)
        source = root / "report.pdf"
        source.write_bytes(b"%PDF-1.7\nprobe")
        alice = _access(tenant, "alice")
        bob = _access(tenant, "bob")
        ref = (
            await normalize_inbound_attachments(
                db,
                ({"path": str(source), "filename": "report.pdf"},),
                session_id="",
                principal=alice,
                allow_local_paths=True,
            )
        )[0]
        assert await artifact_is_visible(db, ref["artifact_id"], alice)
        assert not await artifact_is_visible(db, ref["artifact_id"], bob)
        assert not await artifact_is_visible(db, "art_missing", alice)
        try:
            await normalize_inbound_attachments(
                db,
                ({"artifact_id": ref["artifact_id"]},),
                session_id="tg:bob",
                principal=bob,
            )
        except ArtifactNotFound:
            pass
        else:
            raise AssertionError("opaque artifact id granted cross-principal access")

        # Supplying bytes is proof of possession and creates a pending resource
        # link, never a durable artifact-level grant.  The link grants nothing
        # until the caller's normalized session actually exists.
        bob_copy = root / "bob-report.pdf"
        bob_copy.write_bytes(source.read_bytes())
        bob_ref = (
            await normalize_inbound_attachments(
                db,
                ({"path": str(bob_copy), "filename": "bob-report.pdf"},),
                session_id="not-yet-projected",
                principal=bob,
                allow_local_paths=True,
            )
        )[0]
        assert bob_ref["artifact_id"] == ref["artifact_id"]
        assert "artifact_link_id" in bob_ref
        assert not await artifact_is_visible(db, bob_ref["artifact_id"], bob)
        _seed_session(db, tenant, "not-yet-projected", "bob")
        assert await artifact_is_visible(db, bob_ref["artifact_id"], bob)

        _seed_session(db, tenant, "alice-private", "alice")
        try:
            await normalize_inbound_attachments(
                db,
                ({"path": str(bob_copy), "filename": "forged.pdf"},),
                session_id="alice-private",
                principal=bob,
                allow_local_paths=True,
            )
        except ArtifactNotFound:
            pass
        else:
            raise AssertionError("attachment linked to an unauthorized session")


@test("artifacts", "linked ACL changes, deletion, and cross-context dedup are immediate")
async def t_artifact_link_acl_lifecycle(ctx: TestContext) -> None:
    from src.memory.artifacts import (
        artifact_is_visible,
        artifact_row,
        normalize_inbound_attachments,
    )

    with tempfile.TemporaryDirectory(prefix="oa-artifacts-link-acl-") as raw:
        root = Path(raw)
        db, tenant = _artifact_db(root)
        alice = _access(tenant, "alice")
        bob = _access(tenant, "bob")
        _seed_session(db, tenant, "alice-session", "alice", visibility="public")
        _seed_session(db, tenant, "bob-session", "bob")
        source = root / "shared.txt"
        source.write_text("same immutable bytes", encoding="utf-8")

        alice_ref = (
            await normalize_inbound_attachments(
                db,
                ({"path": str(source), "filename": "alice.txt"},),
                session_id="alice-session",
                principal=alice,
                allow_local_paths=True,
            )
        )[0]
        row, _path = await artifact_row(db, alice_ref["artifact_id"])
        assert row["visibility"] == "private"
        assert row["owner_principal_id"] == "user:alice"
        assert await artifact_is_visible(db, alice_ref["artifact_id"], bob)

        conn = sqlite3.connect(db.db_path)
        try:
            conn.execute(
                "UPDATE sessions_v2 SET visibility='private', updated_at_ms=updated_at_ms+1 "
                "WHERE id='alice-session'"
            )
            conn.commit()
        finally:
            conn.close()
        assert not await artifact_is_visible(db, alice_ref["artifact_id"], bob)

        conn = sqlite3.connect(db.db_path)
        try:
            conn.execute(
                "UPDATE sessions_v2 SET visibility='shared', updated_at_ms=updated_at_ms+1 "
                "WHERE id='alice-session'"
            )
            conn.execute(
                "INSERT INTO resource_acl "
                "(tenant_id,resource_type,resource_id,principal_type,principal_id,permission,"
                "acl_version,granted_by_principal_id,granted_at_ms) "
                "VALUES (?, 'session', 'alice-session', 'user', 'bob', 'view', 1, "
                "'user:alice', ?)",
                (tenant, 1_700_000_000_100),
            )
            conn.commit()
        finally:
            conn.close()
        assert await artifact_is_visible(db, alice_ref["artifact_id"], bob)

        conn = sqlite3.connect(db.db_path)
        try:
            conn.execute(
                "DELETE FROM resource_acl WHERE tenant_id=? AND resource_type='session' "
                "AND resource_id='alice-session'",
                (tenant,),
            )
            conn.commit()
        finally:
            conn.close()
        assert not await artifact_is_visible(db, alice_ref["artifact_id"], bob)

        # Same bytes, different private context: one CAS/artifact identity, two
        # independently authorized resource links, no direct artifact grant.
        bob_copy = root / "bob-copy.txt"
        bob_copy.write_bytes(source.read_bytes())
        bob_ref = (
            await normalize_inbound_attachments(
                db,
                ({"path": str(bob_copy), "filename": "bob.txt"},),
                session_id="bob-session",
                principal=bob,
                allow_local_paths=True,
            )
        )[0]
        assert bob_ref["artifact_id"] == alice_ref["artifact_id"]
        assert await artifact_is_visible(db, bob_ref["artifact_id"], bob)

        conn = sqlite3.connect(db.db_path)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM artifacts WHERE tenant_id=?",
                (tenant,),
            ).fetchone()
            grants = conn.execute(
                "SELECT COUNT(*) FROM resource_acl WHERE tenant_id=? "
                "AND resource_type='artifact'",
                (tenant,),
            ).fetchone()
            assert count == (1,)
            assert grants == (0,)
            conn.execute(
                "UPDATE sessions_v2 SET deleted_at_ms=updated_at_ms+10 "
                "WHERE id='alice-session'"
            )
            conn.commit()
        finally:
            conn.close()
        assert await artifact_is_visible(db, bob_ref["artifact_id"], bob)

        conn = sqlite3.connect(db.db_path)
        try:
            conn.execute(
                "UPDATE sessions_v2 SET deleted_at_ms=updated_at_ms+10 "
                "WHERE id='bob-session'"
            )
            # Even a stale direct grant must not bypass the link lifecycle.
            conn.execute(
                "INSERT INTO resource_acl "
                "(tenant_id,resource_type,resource_id,principal_type,principal_id,permission,"
                "acl_version,granted_by_principal_id,granted_at_ms) "
                "VALUES (?, 'artifact', ?, 'user', 'bob', 'view', 1, 'user:alice', ?)",
                (tenant, bob_ref["artifact_id"], 1_700_000_000_200),
            )
            conn.commit()
        finally:
            conn.close()
        assert not await artifact_is_visible(db, bob_ref["artifact_id"], bob)


@test("artifacts", "checksummed artifact ACL migration normalizes old beta rows")
async def t_artifact_acl_migration(ctx: TestContext) -> None:
    import aiosqlite

    from src.memory.artifact_acl_migration import (
        MIGRATION_ID,
        ensure_artifact_acl_storage,
        migration_checksum,
    )

    with tempfile.TemporaryDirectory(prefix="oa-artifacts-acl-migration-") as raw:
        root = Path(raw)
        db, tenant = _artifact_db(root)
        digest = "a" * 64
        conn = sqlite3.connect(db.db_path)
        try:
            conn.execute(
                "INSERT INTO artifacts "
                "(id,tenant_id,owner_principal_id,owner_handle_snapshot,visibility,acl_version,"
                "direction,kind,mime,original_filename,storage_key,sha256,size_bytes,"
                "storage_state,metadata_json,retention_class,ref_count,created_at_ms,"
                "updated_at_ms,deleted_at_ms) VALUES "
                "('art_legacy',?,NULL,NULL,'installation_shared',1,'input','file',"
                "'text/plain','legacy.txt',?, ?,0,'available','{}','session',0,?,?,NULL)",
                (tenant, f"sha256/{digest[:2]}/{digest}", digest, 100, 100),
            )
            conn.execute(
                "INSERT INTO resource_acl "
                "(tenant_id,resource_type,resource_id,principal_type,principal_id,permission,"
                "acl_version,granted_by_principal_id,granted_at_ms) "
                "VALUES (?, 'artifact', 'art_legacy', 'user', 'bob', 'view', 1, "
                "'user:alice', 100)",
                (tenant,),
            )
            conn.execute(
                "CREATE TRIGGER force_artifact_acl_migration_failure "
                "BEFORE DELETE ON resource_acl WHEN OLD.resource_type='artifact' "
                "BEGIN SELECT RAISE(ABORT, 'forced migration failure'); END"
            )
            conn.commit()
        finally:
            conn.close()

        async_conn = await aiosqlite.connect(db.db_path)
        async_conn.row_factory = aiosqlite.Row
        try:
            try:
                await ensure_artifact_acl_storage(async_conn, app_version="test")
            except Exception as exc:
                assert "forced migration failure" in str(exc)
            else:
                raise AssertionError("forced migration failure did not abort")
            # BEGIN IMMEDIATE keeps the row normalization and grant deletion in
            # the same transaction: neither mutation survives a mid-script
            # failure, while the ledger records a retryable failed attempt.
            rolled_back = await (
                await async_conn.execute(
                    "SELECT visibility,owner_principal_id FROM artifacts "
                    "WHERE id='art_legacy'"
                )
            ).fetchone()
            assert rolled_back is not None
            assert tuple(rolled_back) == ("installation_shared", None)
            old_grant = await (
                await async_conn.execute(
                    "SELECT COUNT(*) FROM resource_acl WHERE resource_type='artifact'"
                )
            ).fetchone()
            assert old_grant is not None and int(old_grant[0]) == 1
            failed = await (
                await async_conn.execute(
                    "SELECT status FROM schema_migrations WHERE migration_id=?",
                    (MIGRATION_ID,),
                )
            ).fetchone()
            assert failed is not None and failed[0] == "failed"
            await async_conn.execute("DROP TRIGGER force_artifact_acl_migration_failure")
            await async_conn.commit()

            assert await ensure_artifact_acl_storage(async_conn, app_version="test")
            assert not await ensure_artifact_acl_storage(async_conn, app_version="test")
            row = await (
                await async_conn.execute(
                    "SELECT visibility, owner_principal_id, acl_version FROM artifacts "
                    "WHERE id='art_legacy'"
                )
            ).fetchone()
            assert row is not None
            assert tuple(row) == ("private", "agent:openagent", 2)
            grant_count = await (
                await async_conn.execute(
                    "SELECT COUNT(*) FROM resource_acl WHERE resource_type='artifact'"
                )
            ).fetchone()
            assert grant_count is not None and int(grant_count[0]) == 0
            ledger = await (
                await async_conn.execute(
                    "SELECT checksum,status FROM schema_migrations WHERE migration_id=?",
                    (MIGRATION_ID,),
                )
            ).fetchone()
            assert ledger is not None
            assert tuple(ledger) == (migration_checksum(), "complete")
            try:
                await async_conn.execute(
                    "UPDATE artifacts SET visibility='public' WHERE id='art_legacy'"
                )
            except aiosqlite.IntegrityError:
                await async_conn.rollback()
            else:
                raise AssertionError("privacy trigger accepted a public artifact")
            try:
                await async_conn.execute(
                    "INSERT INTO resource_acl "
                    "(tenant_id,resource_type,resource_id,principal_type,principal_id,"
                    "permission,acl_version,granted_by_principal_id,granted_at_ms) "
                    "VALUES (?, 'artifact', 'art_legacy', 'user', 'bob', 'view', 2, "
                    "'agent:openagent', 200)",
                    (tenant,),
                )
            except aiosqlite.IntegrityError:
                await async_conn.rollback()
            else:
                raise AssertionError("migration accepted a direct artifact grant")
        finally:
            await async_conn.close()


@test("artifacts", "gateway accepts local paths only from verified bridges")
async def t_gateway_attachment_path_trust_boundary(ctx: TestContext) -> None:
    from src.core.on_behalf_context import OnBehalfIdentity
    from src.gateway.server import Gateway, _StreamHolder
    from src.gateway.sessions import SessionManager
    from src.stream.events import Attachment, TextFinal

    with tempfile.TemporaryDirectory(prefix="oa-artifacts-gateway-trust-") as raw:
        root = Path(raw)
        db, tenant = _artifact_db(root)
        _seed_session(db, tenant, "chat-secure", "alice")
        uploaded_by_bridge = root / "bridge-photo.png"
        payload = b"\x89PNG\r\n\x1a\ntrusted-bridge"
        uploaded_by_bridge.write_bytes(payload)
        identity = OnBehalfIdentity(
            tenant_id=tenant,
            principal_type="user",
            handle="alice",
            device_id="device-alice",
        )

        class _Session:
            def __init__(self) -> None:
                self.pushed = []
                self.on_behalf_identity = None
                self.allow_local_attachment_paths = False

            async def push_in(self, event, **_trusted) -> None:
                self.pushed.append(event)

            def update_client_capabilities(self, *_args) -> None:
                pass

            def has_active_turn(self) -> bool:
                return False

        class _Channel:
            def rebind(self, _send) -> None:
                pass

            async def start(self) -> None:
                pass

        class _WS:
            def __init__(self) -> None:
                self.sent = []

        session = _Session()
        gw = Gateway.__new__(Gateway)
        gw.agent = SimpleNamespace(memory_db=db, db=db, model=None)
        gw.sessions = SessionManager(agent_name="test")
        gw._stream_sessions = {
            ("alice", "chat-secure"): _StreamHolder(
                session=session,
                channel=_Channel(),
            )
        }
        gw._live_replays = {}

        async def _not_stale(_key, _holder) -> bool:
            return False

        async def _capture_send(ws, value) -> bool:
            ws.sent.append(value)
            return True

        gw._stream_holder_is_stale_for_attach = _not_stale
        gw._safe_ws_send_json = _capture_send
        ws = _WS()

        # A normal authenticated client cannot turn the agent into an
        # arbitrary-file oracle through either attachment wire shape.
        await gw._handle_stream_frame(
            ws,
            "device-alice",
            {
                "type": "text_final",
                "session_id": "chat-secure",
                "text": "steal",
                "attachments": [{"path": "/etc/passwd", "filename": "passwd"}],
            },
            handle="alice",
            on_behalf_identity=identity,
            trusted_bridge=False,
        )
        assert not session.pushed
        assert any(item.get("type") == "error" for item in ws.sent)

        before = len(ws.sent)
        await gw._handle_stream_frame(
            ws,
            "device-alice",
            {
                "type": "attachment",
                "session_id": "chat-secure",
                "path": "/etc/passwd",
                "filename": "passwd",
            },
            handle="alice",
            on_behalf_identity=identity,
            trusted_bridge=False,
        )
        assert not session.pushed
        assert len(ws.sent) == before + 2
        conn = sqlite3.connect(db.db_path)
        try:
            assert conn.execute("SELECT COUNT(*) FROM artifacts").fetchone() == (0,)
        finally:
            conn.close()

        # A bridge path is trusted by its certificate capability, copied into
        # CAS, and replaced before the StreamSession sees it.
        await gw._handle_stream_frame(
            ws,
            "device-alice",
            {
                "type": "text_final",
                "session_id": "chat-secure",
                "text": "describe",
                "attachments": [{
                    "path": str(uploaded_by_bridge),
                    "filename": "bridge-photo.png",
                    "mime_type": "image/png",
                }],
            },
            handle="alice",
            on_behalf_identity=identity,
            trusted_bridge=True,
        )
        trusted = session.pushed[-1]
        assert isinstance(trusted, TextFinal)
        ref = trusted.attachments[0]
        assert ref["artifact_id"].startswith("art_")
        assert Path(ref["path"]).read_bytes() == payload
        assert Path(ref["path"]).resolve() != uploaded_by_bridge.resolve()

        # Once uploaded, a normal client may use artifact_id.  Any accompanying
        # hostile path is ignored and reconstructed from canonical metadata.
        await gw._handle_stream_frame(
            ws,
            "device-alice",
            {
                "type": "attachment",
                "session_id": "chat-secure",
                "artifact_id": ref["artifact_id"],
                "artifact_link_id": ref["artifact_link_id"],
                "path": "/etc/passwd",
                "filename": "bridge-photo.png",
            },
            handle="alice",
            on_behalf_identity=identity,
            trusted_bridge=False,
        )
        canonical = session.pushed[-1]
        assert isinstance(canonical, Attachment)
        assert canonical.artifact_id == ref["artifact_id"]
        assert canonical.path == ref["path"]
        assert canonical.path != "/etc/passwd"


@test("artifacts", "inbound byte cap rejects oversized files before CAS publication")
async def t_artifact_size_limit(ctx: TestContext) -> None:
    from src.memory.artifacts import AttachmentTooLarge, normalize_inbound_attachments

    old = os.environ.get("OPENAGENT_MAX_INBOUND_ATTACHMENT_MB")
    os.environ["OPENAGENT_MAX_INBOUND_ATTACHMENT_MB"] = "1"
    try:
        with tempfile.TemporaryDirectory(prefix="oa-artifacts-limit-") as raw:
            root = Path(raw)
            db, tenant = _artifact_db(root)
            source = root / "large.bin"
            source.write_bytes(b"x" * (1024 * 1024 + 1))
            try:
                await normalize_inbound_attachments(
                    db,
                    ({"path": str(source), "filename": "large.bin"},),
                    session_id="",
                    principal=_access(tenant, "alice"),
                    allow_local_paths=True,
                )
            except AttachmentTooLarge as exc:
                assert exc.limit_bytes == 1024 * 1024
            else:
                raise AssertionError("oversized attachment was accepted")
            published = list((root / "artifacts/sha256").glob("*/*"))
            assert published == [], published
    finally:
        if old is None:
            os.environ.pop("OPENAGENT_MAX_INBOUND_ATTACHMENT_MB", None)
        else:
            os.environ["OPENAGENT_MAX_INBOUND_ATTACHMENT_MB"] = old


@test("artifacts", "message linking waits past an old row and history refs stay path-free")
async def t_artifact_message_link_projection_race(ctx: TestContext) -> None:
    import aiosqlite

    from src.memory.artifacts import (
        attachment_refs_for_messages_on_connection,
        link_attachments_to_latest_message,
        normalize_inbound_attachments,
    )

    with tempfile.TemporaryDirectory(prefix="oa-artifacts-link-race-") as raw:
        root = Path(raw)
        db, tenant = _artifact_db(root)
        alice = _access(tenant, "alice")
        _seed_session(db, tenant, "chat-a", "alice")
        _seed_message(db, tenant, "chat-a", "old-user", sequence=0)
        source = root / "note.txt"
        source.write_text("projected later", encoding="utf-8")
        ref = (
            await normalize_inbound_attachments(
                db,
                ({"path": str(source), "filename": "note.txt"},),
                session_id="chat-a",
                principal=alice,
                allow_local_paths=True,
            )
        )[0]

        linking = asyncio.create_task(
            link_attachments_to_latest_message(
                db,
                "chat-a",
                (ref,),
                role="user",
                principal=alice,
                after_sequence=0,
            )
        )
        await asyncio.sleep(0.03)
        _seed_message(db, tenant, "chat-a", "new-user", sequence=1)
        assert await linking == "new-user"

        conn = await aiosqlite.connect(db.db_path)
        conn.row_factory = aiosqlite.Row
        try:
            hydrated = await attachment_refs_for_messages_on_connection(
                conn, ("old-user", "new-user"),
            )
        finally:
            await conn.close()
        assert "old-user" not in hydrated
        assert hydrated["new-user"][0]["artifact_id"] == ref["artifact_id"]
        assert "path" not in hydrated["new-user"][0]


@test("artifacts", "artifact content endpoint enforces ACL and serves canonical bytes")
async def t_artifact_endpoint_acl(ctx: TestContext) -> None:
    from aiohttp import FormData, web
    from aiohttp.test_utils import TestClient, TestServer

    from src.gateway.api import artifacts as artifacts_api
    from src.memory.artifacts import normalize_inbound_attachments

    with tempfile.TemporaryDirectory(prefix="oa-artifacts-http-") as raw:
        root = Path(raw)
        db, tenant = _artifact_db(root)
        source = root / "pixel.png"
        payload = b"\x89PNG\r\n\x1a\nendpoint"
        source.write_bytes(payload)
        alice = _access(tenant, "alice")
        ref = (
            await normalize_inbound_attachments(
                db,
                ({"path": str(source), "filename": "pixel.png"},),
                session_id="",
                principal=alice,
                allow_local_paths=True,
            )
        )[0]

        @web.middleware
        async def _identity(request, handler):
            handle = request.headers.get("X-Test-Handle", "alice")
            cert = SimpleNamespace(
                network_id=tenant,
                handle=handle,
                device_pubkey_hex=f"device-{handle}",
                capabilities=[],
            )
            request["device_cert"] = cert
            request["network_id"] = tenant
            request["user_handle"] = handle
            request["client_id"] = cert.device_pubkey_hex
            return await handler(request)

        app = web.Application(middlewares=[_identity])
        app["gateway"] = SimpleNamespace(
            agent=SimpleNamespace(memory_db=db),
        )
        app.router.add_get(
            "/api/artifacts/{artifact_id}/content",
            artifacts_api.handle_content,
        )
        app.router.add_get(
            "/api/artifacts/{artifact_id}",
            artifacts_api.handle_metadata,
        )
        app.router.add_post("/api/artifacts", artifacts_api.handle_upload)
        from src.gateway.server import Gateway

        async def _legacy_file(request):
            return await Gateway._handle_files(app["gateway"], request)

        app.router.add_get("/api/files", _legacy_file)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.get(
                f"/api/artifacts/{ref['artifact_id']}/content",
            )
            assert response.status == 200
            assert await response.read() == payload
            assert response.headers["Content-Type"].startswith("image/png")

            denied = await client.get(
                f"/api/artifacts/{ref['artifact_id']}/content",
                headers={"X-Test-Handle": "bob"},
            )
            assert denied.status == 404

            form = FormData()
            form.add_field(
                "file", b"fresh upload", filename="fresh.txt", content_type="text/plain",
            )
            uploaded = await client.post("/api/artifacts", data=form)
            assert uploaded.status == 201, await uploaded.text()
            uploaded_ref = (await uploaded.json())["attachment"]
            assert uploaded_ref["artifact_id"].startswith("art_")
            assert uploaded_ref["url"].endswith("/content")
            assert "path" not in uploaded_ref

            # Tenant-level byte dedup must not deduplicate presentation
            # metadata across private contexts. Alice owns the canonical row;
            # Bob is authorized by his own link and sees only its display name.
            _seed_session(db, tenant, "filename-alice", "alice")
            _seed_session(db, tenant, "filename-bob", "bob")
            alice_file = root / "alice-private-name.txt"
            bob_file = root / "bob-private-name.txt"
            alice_file.write_bytes(b"same bytes, context-specific filename")
            bob_file.write_bytes(alice_file.read_bytes())
            alice_named = (
                await normalize_inbound_attachments(
                    db,
                    ({"path": str(alice_file), "filename": alice_file.name},),
                    session_id="filename-alice",
                    principal=alice,
                    allow_local_paths=True,
                )
            )[0]
            bob_named = (
                await normalize_inbound_attachments(
                    db,
                    ({"path": str(bob_file), "filename": bob_file.name},),
                    session_id="filename-bob",
                    principal=_access(tenant, "bob"),
                    allow_local_paths=True,
                )
            )[0]
            assert alice_named["artifact_id"] == bob_named["artifact_id"]
            artifact_url = f"/api/artifacts/{alice_named['artifact_id']}"

            alice_meta = await client.get(artifact_url)
            assert alice_meta.status == 200
            assert (await alice_meta.json())["artifact"]["filename"] == alice_file.name
            bob_meta = await client.get(
                artifact_url,
                headers={"X-Test-Handle": "bob"},
            )
            assert bob_meta.status == 200
            assert (await bob_meta.json())["artifact"]["filename"] == bob_file.name

            alice_content = await client.get(f"{artifact_url}/content")
            assert alice_file.name in alice_content.headers["Content-Disposition"]
            assert await alice_content.read() == alice_file.read_bytes()
            bob_content = await client.get(
                f"{artifact_url}/content",
                headers={"X-Test-Handle": "bob"},
            )
            assert bob_file.name in bob_content.headers["Content-Disposition"]
            assert await bob_content.read() == bob_file.read_bytes()

            alice_legacy = await client.get(
                "/api/files",
                params={"path": alice_named["path"]},
            )
            assert alice_file.name in alice_legacy.headers["Content-Disposition"]
            assert await alice_legacy.read() == alice_file.read_bytes()
            bob_legacy = await client.get(
                "/api/files",
                params={"path": bob_named["path"]},
                headers={"X-Test-Handle": "bob"},
            )
            assert bob_file.name in bob_legacy.headers["Content-Disposition"]
            assert await bob_legacy.read() == bob_file.read_bytes()

            # Same-length tampering must not be masked by metadata size or a
            # stale ETag. The visible artifact remains distinguishable from an
            # inaccessible id, but its bytes fail closed.
            from src.memory.artifacts import artifact_row

            _row, cas_path = await artifact_row(db, ref["artifact_id"])
            cas_path.write_bytes(b"x" * len(payload))
            corrupted = await client.get(
                f"/api/artifacts/{ref['artifact_id']}/content",
            )
            assert corrupted.status == 503
            assert (await corrupted.json())["error"]["code"] == "artifact_unavailable"
        finally:
            await client.close()
