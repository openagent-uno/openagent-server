"""Normalized ordered message parts and delayed-projection regression tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile

from ._framework import TestContext, test


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


async def _seed_session(conn, tenant: str, session_id: str) -> None:
    now = 1_800_000_000_000
    await conn.execute(
        "INSERT INTO sessions_v2 "
        "(id,tenant_id,owner_principal_id,owner_handle_snapshot,visibility,acl_version,"
        "session_type,kind,status,completeness,source_version,metadata_json,created_at_ms,"
        "updated_at_ms,last_activity_at_ms) VALUES (?,?,?,?, 'private',1,"
        "'agent','chat','active','complete',1,'{}',?,?,?)",
        (session_id, tenant, "user:alice", "alice", now, now, now),
    )
    await conn.commit()


async def _seed_message(
    conn,
    tenant: str,
    session_id: str,
    message_id: str,
    *,
    sequence: int,
    role: str,
    text: str,
) -> None:
    now = 1_800_000_000_000 + sequence
    author_kind = "agent" if role == "assistant" else "user"
    principal = "agent:openagent" if role == "assistant" else "user:alice"
    await conn.execute(
        "INSERT INTO session_messages "
        "(id,tenant_id,session_id,run_id,sequence,ordinal,role,status,author_kind,"
        "author_principal_id,text,visibility,source_version,completeness,raw_envelope_json,"
        "raw_envelope_schema,legacy_inferred,created_at_ms,updated_at_ms,completed_at_ms) "
        "VALUES (?,?,?,NULL,?,0,?,'complete',?,?,?,'user_visible',1,'complete','{}',1,0,?,?,?)",
        (
            message_id,
            tenant,
            session_id,
            sequence,
            role,
            author_kind,
            principal,
            text,
            now,
            now,
            now,
        ),
    )
    await conn.commit()


@test("message_parts", "checksummed ordered-parts migration is additive and idempotent")
async def t_message_parts_migration(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.memory.message_parts_migration import (
        MIGRATION_ID,
        ensure_message_parts_storage,
        migration_checksum,
    )

    with tempfile.TemporaryDirectory(prefix="oa-message-parts-schema-") as raw:
        db = MemoryDB(str(Path(raw) / "agent.db"))
        await db.connect()
        try:
            conn = await db._ensure_connected()
            row = await (
                await conn.execute(
                    "SELECT checksum, status FROM schema_migrations WHERE migration_id=?",
                    (MIGRATION_ID,),
                )
            ).fetchone()
            assert row is not None
            assert row["status"] == "complete"
            assert row["checksum"] == migration_checksum()
            assert not await ensure_message_parts_storage(conn, app_version="test")
            table = await (
                await conn.execute(
                    "SELECT 1 FROM sqlite_schema WHERE type='table' "
                    "AND name='session_message_parts'"
                )
            ).fetchone()
            assert table is not None
        finally:
            await db.close()


@test("message_parts", "failed ordered-parts DDL rolls back atomically")
async def t_message_parts_migration_rollback(ctx: TestContext) -> None:
    import src.memory.message_parts_migration as migration
    from src.memory.db import MemoryDB

    with tempfile.TemporaryDirectory(prefix="oa-message-parts-rollback-") as raw:
        db = MemoryDB(str(Path(raw) / "agent.db"))
        await db.connect()
        original_sql = migration.migration_sql
        original_checksum_fn = migration.migration_checksum
        try:
            conn = await db._ensure_connected()
            await conn.execute(
                "UPDATE schema_migrations SET status='failed', error_class='injected', "
                "completed_at_ms=updated_at_ms WHERE migration_id=?",
                (migration.MIGRATION_ID,),
            )
            await conn.commit()
            completed_checksum = original_checksum_fn()
            migration.migration_sql = lambda: (
                "BEGIN IMMEDIATE; "
                "CREATE TABLE message_parts_rollback_probe(id INTEGER PRIMARY KEY); "
                "SELECT * FROM table_that_must_not_exist; "
                "COMMIT;"
            )
            migration.migration_checksum = lambda: completed_checksum
            try:
                await migration.ensure_message_parts_storage(conn, app_version="test")
            except Exception:
                pass
            else:
                raise AssertionError("intentionally broken migration unexpectedly completed")

            probe = await (
                await conn.execute(
                    "SELECT 1 FROM sqlite_schema WHERE type='table' "
                    "AND name='message_parts_rollback_probe'"
                )
            ).fetchone()
            assert probe is None
            ledger = await (
                await conn.execute(
                    "SELECT status, error_class FROM schema_migrations WHERE migration_id=?",
                    (migration.MIGRATION_ID,),
                )
            ).fetchone()
            assert ledger is not None and ledger["status"] == "failed"
            assert ledger["error_class"]
        finally:
            migration.migration_sql = original_sql
            migration.migration_checksum = original_checksum_fn
            await db.close()


@test("message_parts", "mixed text, attachment, and View order survives delayed projection")
async def t_message_parts_projection_and_hydration(ctx: TestContext) -> None:
    from src.custom_views.repository import CustomViewRepository
    from src.memory.artifacts import persist_output_attachments
    from src.memory.db import MemoryDB
    from src.memory.message_parts import (
        canonical_parts_for_messages_on_connection,
        persist_parts_for_latest_message,
    )

    with tempfile.TemporaryDirectory(prefix="oa-message-parts-order-") as raw:
        root = Path(raw)
        db = MemoryDB(str(root / "agent.db"))
        await db.connect()
        tenant = "tenant-parts"
        access = _access(tenant, "alice")
        session_id = "chat-parts"
        try:
            conn = await db._ensure_connected()
            await _seed_session(conn, tenant, session_id)
            await _seed_message(
                conn,
                tenant,
                session_id,
                "old-assistant",
                sequence=0,
                role="assistant",
                text="old",
            )
            source = root / "report.txt"
            source.write_text("canonical attachment", encoding="utf-8")
            attachment = (
                await persist_output_attachments(
                    db,
                    session_id,
                    ({"path": str(source), "filename": "report.txt"},),
                    principal=access,
                )
            )[0]
            view = await CustomViewRepository(db).create(
                access,
                surface="inline",
                session_id=session_id,
                title="Inline status",
                spec={
                    "schemaVersion": 1,
                    "root": {"type": "text", "props": {"text": "Ready"}},
                },
            )
            ordered = [
                {"kind": "text", "text": "Before "},
                {"kind": "attachment", "attachment": dict(attachment)},
                {"kind": "text", "text": " middle "},
                {
                    "kind": "ui_view",
                    "view_id": view["id"],
                    "revision": int(view["revision"]),
                },
                {"kind": "text", "text": " after"},
            ]
            pending = asyncio.create_task(
                persist_parts_for_latest_message(
                    db,
                    session_id,
                    role="assistant",
                    parts=ordered,
                    principal=access,
                    after_sequence=0,
                    timeout_seconds=2.0,
                )
            )
            # Regression: the old 100 ms lookup window lost this View forever.
            await asyncio.sleep(0.35)
            await _seed_message(
                conn,
                tenant,
                session_id,
                "new-assistant",
                sequence=1,
                role="assistant",
                text=(
                    "Before [FILE:/tmp/legacy-report.txt] middle "
                    f"[OPENAGENT_UI:{view['id']}@1] after"
                ),
            )
            assert await pending == "new-assistant"

            hydrated = await canonical_parts_for_messages_on_connection(
                conn,
                ("old-assistant", "new-assistant"),
                access=access,
            )
            assert "old-assistant" not in hydrated
            parts = hydrated["new-assistant"]
            assert [part["kind"] for part in parts] == [
                "text",
                "attachment",
                "text",
                "ui_view",
                "text",
            ]
            ref = parts[1]["attachment"]
            assert ref["artifact_id"] == attachment["artifact_id"]
            assert ref["url"].endswith("/content")
            assert "path" not in ref
            assert parts[3]["view_id"] == view["id"]

            # The UI carrier is represented only by normalized rows. No new
            # message-part payload is written into provider/session JSON.
            raw_envelope = await (
                await conn.execute(
                    "SELECT raw_envelope_json FROM session_messages WHERE id='new-assistant'"
                )
            ).fetchone()
            assert "OPENAGENT_UI" not in str(raw_envelope[0])
        finally:
            await db.close()


@test("message_parts", "canonical parts are insert-once and retries cannot leave new links")
async def t_message_parts_are_immutable(ctx: TestContext) -> None:
    from src.custom_views.repository import CustomViewRepository
    from src.memory.artifacts import persist_output_attachments
    from src.memory.db import MemoryDB
    from src.memory.message_parts import MessagePartsError, persist_parts_for_latest_message

    with tempfile.TemporaryDirectory(prefix="oa-message-parts-immutable-") as raw:
        root = Path(raw)
        db = MemoryDB(str(root / "agent.db"))
        await db.connect()
        tenant = "tenant-immutable"
        access = _access(tenant, "alice")
        session_id = "chat-immutable"
        try:
            conn = await db._ensure_connected()
            await _seed_session(conn, tenant, session_id)
            await _seed_message(
                conn, tenant, session_id, "assistant-immutable",
                sequence=1, role="assistant", text="Ready",
            )
            first_file = root / "first.txt"
            first_file.write_text("first", encoding="utf-8")
            first_attachment = (
                await persist_output_attachments(
                    db, session_id,
                    ({"path": str(first_file), "filename": "first.txt"},),
                    principal=access,
                )
            )[0]
            repo = CustomViewRepository(db)
            first_view = await repo.create(
                access,
                surface="inline",
                session_id=session_id,
                title="First",
                spec={
                    "schemaVersion": 1,
                    "root": {"type": "text", "props": {"text": "First"}},
                },
            )
            original = [
                {"kind": "text", "text": "Ready "},
                {"kind": "attachment", "attachment": dict(first_attachment)},
                {
                    "kind": "ui_view",
                    "view_id": first_view["id"],
                    "revision": first_view["revision"],
                },
            ]
            assert await persist_parts_for_latest_message(
                db, session_id, role="assistant", parts=original,
                principal=access, after_sequence=0,
            ) == "assistant-immutable"
            # Exact delivery retries are idempotent.
            assert await persist_parts_for_latest_message(
                db, session_id, role="assistant", parts=original,
                principal=access, after_sequence=0,
            ) == "assistant-immutable"

            before_parts = await (
                await conn.execute(
                    "SELECT kind, text_content, artifact_link_id, ui_view_id, ui_revision "
                    "FROM session_message_parts WHERE message_id=? ORDER BY ordinal",
                    ("assistant-immutable",),
                )
            ).fetchall()
            before_message_links = int((await (
                await conn.execute(
                    "SELECT COUNT(*) FROM artifact_links WHERE resource_type='message' "
                    "AND resource_id=?",
                    ("assistant-immutable",),
                )
            ).fetchone())[0])
            before_ui_links = int((await (
                await conn.execute(
                    "SELECT COUNT(*) FROM ui_message_links WHERE message_id=?",
                    ("assistant-immutable",),
                )
            ).fetchone())[0])

            second_view = await repo.create(
                access,
                surface="inline",
                session_id=session_id,
                title="Second",
                spec={
                    "schemaVersion": 1,
                    "root": {"type": "text", "props": {"text": "Second"}},
                },
            )
            conflicting = [
                {"kind": "text", "text": "Changed "},
                {"kind": "attachment", "attachment": dict(first_attachment)},
                {
                    "kind": "ui_view",
                    "view_id": second_view["id"],
                    "revision": second_view["revision"],
                },
            ]
            try:
                await persist_parts_for_latest_message(
                    db, session_id, role="assistant", parts=conflicting,
                    principal=access, after_sequence=0,
                )
            except MessagePartsError:
                pass
            else:
                raise AssertionError("conflicting canonical retry was accepted")

            after_parts = await (
                await conn.execute(
                    "SELECT kind, text_content, artifact_link_id, ui_view_id, ui_revision "
                    "FROM session_message_parts WHERE message_id=? ORDER BY ordinal",
                    ("assistant-immutable",),
                )
            ).fetchall()
            assert [tuple(row) for row in after_parts] == [tuple(row) for row in before_parts]
            after_message_links = int((await (
                await conn.execute(
                    "SELECT COUNT(*) FROM artifact_links WHERE resource_type='message' "
                    "AND resource_id=?",
                    ("assistant-immutable",),
                )
            ).fetchone())[0])
            after_ui_links = int((await (
                await conn.execute(
                    "SELECT COUNT(*) FROM ui_message_links WHERE message_id=?",
                    ("assistant-immutable",),
                )
            ).fetchone())[0])
            assert after_message_links == before_message_links
            assert after_ui_links == before_ui_links
        finally:
            await db.close()
