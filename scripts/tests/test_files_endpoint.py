"""Fail-closed compatibility tests for deprecated ``GET /api/files``."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from ._framework import TestContext, test


class _DiskDB:
    def __init__(self, path: Path) -> None:
        self.db_path = str(path)


def _database(root: Path) -> tuple[_DiskDB, str]:
    db_path = root / "agent.db"
    schema = (
        Path(__file__).parents[2]
        / "src/memory/operational/sql/operational_storage_v2.sql"
    ).read_text(encoding="utf-8")
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema)
        state = conn.execute(
            "SELECT db_instance_id FROM operational_storage_state WHERE singleton_id=1"
        ).fetchone()
        assert state is not None
        tenant = f"installation:{state[0]}"
        now = 1_700_000_000_000
        conn.execute(
            "INSERT INTO sessions_v2 "
            "(id,tenant_id,owner_principal_id,owner_handle_snapshot,visibility,acl_version,"
            "session_type,kind,status,completeness,source_version,metadata_json,created_at_ms,"
            "updated_at_ms,last_activity_at_ms) VALUES "
            "('legacy-file-session',?,'user:alice','alice','public',1,'agent','chat',"
            "'active','complete',1,'{}',?,?,?)",
            (tenant, now, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return _DiskDB(db_path), tenant


def _access(tenant: str, handle: str):
    from src.memory.operational.access import AccessContext

    return AccessContext(
        tenant_id=tenant,
        principal_id=f"user:{handle}",
        principal_type="user",
        handle=handle,
        device_id=f"device-{handle}",
        principal_ids=frozenset(
            {f"user:{handle}", f"user:device-{handle}", f"device:device-{handle}"}
        ),
        grant_identities=frozenset({("user", handle)}),
    )


@asynccontextmanager
async def _legacy_server(root: Path):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    from src.gateway.server import Gateway
    from src.memory.artifacts import normalize_inbound_attachments

    db, tenant = _database(root)
    source = root / "agent-report.txt"
    payload = b"authorized legacy attachment bytes"
    source.write_bytes(payload)
    ref = (
        await normalize_inbound_attachments(
            db,
            ({"path": str(source), "filename": "agent-report.txt"},),
            session_id="legacy-file-session",
            principal=_access(tenant, "alice"),
            allow_local_paths=True,
        )
    )[0]

    @web.middleware
    async def _identity(request, handler):
        handle = request.headers.get("X-Test-Handle", "bob")
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

    fake_gateway = SimpleNamespace(agent=SimpleNamespace(memory_db=db))

    async def _handle(request):
        return await Gateway._handle_files(fake_gateway, request)

    app = web.Application(middlewares=[_identity])
    app.router.add_get("/api/files", _handle)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        yield client, db, tenant, ref, payload
    finally:
        await client.close()


@test("files_endpoint", "legacy path serves only a currently visible CAS link")
async def t_files_authorized_cas(ctx: TestContext) -> None:
    with tempfile.TemporaryDirectory(prefix="oa-files-authorized-") as raw:
        root = Path(raw)
        async with _legacy_server(root) as (client, db, _tenant, ref, payload):
            response = await client.get("/api/files", params={"path": ref["path"]})
            assert response.status == 200, await response.text()
            assert await response.read() == payload
            assert "agent-report.txt" in response.headers["Content-Disposition"]
            assert response.headers["Cache-Control"] == "private, no-store"
            assert response.headers["Deprecation"] == "true"
            assert ref["artifact_id"] in response.headers["Link"]

            # Resolving a symlink to the same authorized immutable CAS object is
            # compatible; authorization is still based on the canonical row.
            alias = root / "legacy-alias"
            os.symlink(ref["path"], alias)
            symlinked = await client.get("/api/files", params={"path": str(alias)})
            assert symlinked.status == 200
            assert await symlinked.read() == payload

            conn = sqlite3.connect(db.db_path)
            try:
                conn.execute(
                    "UPDATE sessions_v2 SET visibility='private', "
                    "updated_at_ms=updated_at_ms+1 WHERE id='legacy-file-session'"
                )
                conn.commit()
            finally:
                conn.close()
            revoked = await client.get("/api/files", params={"path": ref["path"]})
            assert revoked.status == 404


@test("files_endpoint", "legacy path never reads arbitrary host files")
async def t_files_arbitrary_paths_fail_closed(ctx: TestContext) -> None:
    with tempfile.TemporaryDirectory(prefix="oa-files-denied-") as raw:
        root = Path(raw)
        async with _legacy_server(root) as (client, db, _tenant, _ref, _payload):
            secret = root / "config-secret.yaml"
            secret.write_text("api_key: should-never-leave-host", encoding="utf-8")
            targets = [str(secret), db.db_path, "/etc/passwd"]
            for target in targets:
                response = await client.get("/api/files", params={"path": target})
                body = await response.read()
                assert response.status == 404, (target, response.status, body[:100])
                assert b"should-never-leave-host" not in body


@test("files_endpoint", "legacy path validates required and missing targets")
async def t_files_invalid_requests(ctx: TestContext) -> None:
    with tempfile.TemporaryDirectory(prefix="oa-files-invalid-") as raw:
        async with _legacy_server(Path(raw)) as (client, *_rest):
            missing = await client.get("/api/files")
            assert missing.status == 400
            absent = await client.get(
                "/api/files",
                params={"path": "/tmp/openagent-does-not-exist-xyz123.bin"},
            )
            assert absent.status == 404
