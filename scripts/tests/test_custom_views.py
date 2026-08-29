"""OA-UI compiler, durable bundles, repository, runtime, and action tests."""

from __future__ import annotations

import asyncio
import base64
import sys
import tempfile
import time
from pathlib import Path

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
        principal_ids=frozenset({principal, f"user:device-{handle}", f"device:device-{handle}"}),
        grant_identities=frozenset({("user", handle), ("device", f"device-{handle}")}),
    )


def _spec(text: str = "Hello") -> dict:
    return {
        "schemaVersion": 1,
        "root": {
            "type": "stack",
            "id": "root",
            "children": [
                {"type": "text", "props": {"text": text}},
                {
                    "type": "sub-view",
                    "props": {"name": "details"},
                    "children": [
                        {
                            "type": "text-input",
                            "props": {
                                "name": "query",
                                "value": "{{state.query}}",
                                "placeholder": "Filter",
                                "multiline": False,
                                "inputType": "search",
                                "submitLabel": "Apply",
                                "maxLength": 200,
                            },
                        },
                        {
                            "type": "bar-chart",
                            "props": {"data": "{{data.metrics.points}}", "stacked": True},
                        },
                    ],
                },
                {
                    "type": "sub-view",
                    "props": {
                        "name": "referenced",
                        "viewId": "details-view",
                        "revision": 2,
                    },
                },
            ],
        },
        "states": {
            "loading": {"type": "loading-state", "props": {"text": "Loading"}},
            "empty": {"type": "empty-state", "props": {"text": "No data"}},
            "stale": {"type": "stale-state", "props": {"text": "Paused"}},
            "error": {"type": "error-state", "props": {"text": "Unavailable"}},
        },
    }


async def _db(root: Path):
    from src.memory.db import MemoryDB

    db = MemoryDB(str(root / "agent.db"))
    await db.connect()
    return db


@test("custom_views", "OA-UI v1 compiles bindings and rejects executable or remote media")
async def t_oaui_compiler(ctx: TestContext) -> None:
    from src.custom_views.compiler import OAUIValidationError, compile_oaui

    compiled = compile_oaui(spec=_spec())
    input_node = compiled["root"]["children"][1]["children"][0]
    chart = compiled["root"]["children"][1]["children"][1]
    assert input_node["props"]["value"] == {"$bind": {"source": "state", "path": "/query"}}
    assert chart["props"]["data"] == {"$bind": {"source": "data", "path": "/metrics/points"}}
    assert chart["props"]["stacked"] is True
    markup = compile_oaui(
        markup='<sub-view name="main"><text-input name="q" inputType="search" /></sub-view>',
    )
    assert markup["root"]["type"] == "sub-view"
    for unsafe in (
        {"schemaVersion": 1, "root": {"type": "iframe"}},
        {"schemaVersion": 1, "root": {"type": "image", "props": {"src": "https://example.test/x.png"}}},
        {"schemaVersion": 1, "root": {"type": "text", "props": {"href": "javascript:alert(1)"}}},
        {"schemaVersion": 1, "root": {"type": "text", "props": {"href": "//example.test/path"}}},
        {"schemaVersion": 1, "root": {"type": "text", "props": {"href": "https://user:pass@example.test/"}}},
        {"schemaVersion": 1, "root": {"type": "image", "props": {"src": "asset:ok/../secret"}}},
        {"schemaVersion": 1, "root": {"type": "text", "props": {"target": "popup"}}},
        {
            "schemaVersion": 1,
            "root": {"type": "sub-view", "props": {"viewId": "mutable-latest"}},
        },
        {
            "schemaVersion": 1,
            "root": {
                "type": "sub-view",
                "props": {"viewId": "ambiguous", "revision": 1},
                "children": [{"type": "text", "props": {"text": "ignored"}}],
            },
        },
    ):
        try:
            compile_oaui(spec=unsafe)
        except OAUIValidationError:
            pass
        else:
            raise AssertionError(f"unsafe OA-UI was accepted: {unsafe}")


@test("custom_views", "checksummed migration and immutable bundle layout are idempotent")
async def t_custom_view_migration_bundle(ctx: TestContext) -> None:
    from src.custom_views.migration import MIGRATION_ID, ensure_custom_views_storage, migration_checksum
    from src.custom_views.repository import CustomViewRepository

    with tempfile.TemporaryDirectory(prefix="oa-ui-schema-") as raw:
        root = Path(raw)
        db = await _db(root)
        try:
            conn = await db._ensure_connected()
            row = await (
                await conn.execute(
                    "SELECT checksum, status FROM schema_migrations WHERE migration_id=?",
                    (MIGRATION_ID,),
                )
            ).fetchone()
            assert row is not None and row["status"] == "complete"
            assert row["checksum"] == migration_checksum()
            assert not await ensure_custom_views_storage(conn, app_version="test")
            repo = CustomViewRepository(db)
            view = await repo.create(
                _access("tenant-a", "alice"),
                surface="sidebar",
                title="Operations",
                spec=_spec(),
                actions={
                    "private-call": {
                        "kind": "mcp_tool",
                        "config": {
                            "server": "example",
                            "tool": "lookup",
                            "args": {"token": "do-not-copy-to-readable-bundle"},
                        },
                    },
                },
            )
            revision = root / "ui" / "views" / view["id"] / "revisions" / "1"
            assert {item.name for item in revision.iterdir()} == {
                "view.oaui", "compiled.json", "manifest.json", "scripts", "assets",
            }
            assert (revision / "scripts").is_dir() and (revision / "assets").is_dir()
            manifest_text = (revision / "manifest.json").read_text(encoding="utf-8")
            assert "do-not-copy-to-readable-bundle" not in manifest_text
            assert '"actions"' not in manifest_text and '"sources"' not in manifest_text
            evidence_row = await (
                await conn.execute(
                    "SELECT bundle_path, bundle_sha256, bundle_size_bytes "
                    "FROM ui_view_revisions WHERE view_id=? AND revision=1",
                    (view["id"],),
                )
            ).fetchone()
            from src.custom_views.bundles import BundleEvidence

            evidence = BundleEvidence(
                evidence_row["bundle_path"], evidence_row["bundle_sha256"],
                evidence_row["bundle_size_bytes"],
            )
            assert repo.bundles.verify(evidence)
            assert not (revision.stat().st_mode & 0o222)
            assert not ((revision / "compiled.json").stat().st_mode & 0o222)

            table_info = await (
                await conn.execute("PRAGMA table_info(ui_message_links)")
            ).fetchall()
            message_column = next(row for row in table_info if row[1] == "message_id")
            assert int(message_column[3]) == 1
            foreign_keys = await (
                await conn.execute("PRAGMA foreign_key_list(ui_message_links)")
            ).fetchall()
            assert any(
                row[2] == "session_messages"
                and row[3] == "message_id"
                and row[4] == "id"
                and str(row[6]).upper() == "CASCADE"
                for row in foreign_keys
            )
        finally:
            await db.close()


@test("custom_views", "failed Custom Views DDL rolls back atomically")
async def t_custom_view_migration_rollback(ctx: TestContext) -> None:
    import src.custom_views.migration as migration

    with tempfile.TemporaryDirectory(prefix="oa-ui-schema-rollback-") as raw:
        root = Path(raw)
        db = await _db(root)
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
                "CREATE TABLE migration_rollback_probe(id INTEGER PRIMARY KEY); "
                "SELECT * FROM table_that_must_not_exist; "
                "COMMIT;"
            )
            migration.migration_checksum = lambda: completed_checksum
            try:
                await migration.ensure_custom_views_storage(conn, app_version="test")
            except Exception:
                pass
            else:
                raise AssertionError("intentionally broken migration unexpectedly completed")

            probe = await (
                await conn.execute(
                    "SELECT 1 FROM sqlite_schema WHERE type='table' "
                    "AND name='migration_rollback_probe'"
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


@test("custom_views", "repository pins revisions, enforces ACL, and bounds append data")
async def t_custom_view_repository(ctx: TestContext) -> None:
    from src.custom_views.repository import CustomViewNotFound, CustomViewRepository

    with tempfile.TemporaryDirectory(prefix="oa-ui-repo-") as raw:
        db = await _db(Path(raw))
        try:
            repo = CustomViewRepository(db)
            alice = _access("tenant-a", "alice")
            bob = _access("tenant-a", "bob")
            first = await repo.create(
                alice,
                surface="sidebar",
                title="First",
                description="static searchable metadata",
                spec=_spec("one"),
                initial_data={"points": [1]},
                sidebar_order=7,
                sidebar_group="Ops",
            )
            assert first["canExecute"] is True
            try:
                await repo.get(first["id"], bob)
            except CustomViewNotFound:
                pass
            else:
                raise AssertionError("owner-only View leaked across principals")
            second = await repo.update(
                first["id"], alice, expected_revision=1,
                title="Second", spec=_spec("two"),
            )
            assert second["revision"] == 2 and second["title"] == "Second"
            await repo.set_data(
                first["id"], "points", [2, 3], alice, mode="append", max_items=3,
            )
            await repo.set_data(
                first["id"], "points", 4, alice, mode="append", max_items=3,
            )
            pinned = await repo.get(first["id"], alice, revision=1)
            assert pinned["revision"] == 1 and pinned["title"] == "First"
            assert pinned["data"]["points"]["value"] == [2, 3, 4]
            matches, more = await repo.list(alice, query="searchable metadata")
            assert not more and [row["id"] for row in matches] == [first["id"]]
            assert matches[0]["canExecute"] is True

            shared = await repo.create(
                alice,
                surface="sidebar",
                title="Shared read-only",
                visibility="installation_shared",
                spec=_spec("readable"),
                actions={
                    "set": {
                        "kind": "set_data",
                        "config": {"key": "answer", "value": 42},
                    },
                },
            )
            bob_view = await repo.get(shared["id"], bob)
            assert bob_view["title"] == "Shared read-only"
            assert bob_view["canExecute"] is False
            bob_list, _more = await repo.list(bob, query="Shared read-only")
            assert bob_list[0]["canExecute"] is False
            try:
                await repo.action_definition(shared["id"], "set", bob)
            except CustomViewNotFound:
                pass
            else:
                raise AssertionError("read-only viewer received action execution access")
            for mutation in (
                repo.update(shared["id"], bob, expected_revision=1, title="hijacked"),
                repo.set_data(shared["id"], "value", 1, bob),
                repo.delete(shared["id"], bob, expected_revision=1),
            ):
                try:
                    await mutation
                except CustomViewNotFound:
                    pass
                else:
                    raise AssertionError("shared visibility granted mutation access")

            view_grant = await repo.set_grant(
                shared["id"], alice,
                principal_type="user", principal_id="bob",
                permissions=["view"], expected_acl_version=1,
            )
            assert view_grant["aclVersion"] == 2
            assert (await repo.get(shared["id"], bob))["canExecute"] is False
            admin_grant = await repo.set_grant(
                shared["id"], alice,
                principal_type="user", principal_id="bob",
                permissions=["admin"], expected_acl_version=2,
            )
            assert admin_grant["aclVersion"] == 3
            assert (await repo.get(shared["id"], bob))["canExecute"] is True
            assert (await repo.action_definition(shared["id"], "set", bob))["id"] == "set"
        finally:
            await db.close()


@test("custom_views", "inactive inline Views freeze stale and reactivate at the same revision")
async def t_custom_view_freeze(ctx: TestContext) -> None:
    from src.custom_views.repository import CustomViewRepository

    with tempfile.TemporaryDirectory(prefix="oa-ui-freeze-") as raw:
        db = await _db(Path(raw))
        try:
            repo = CustomViewRepository(db)
            alice = _access("tenant-a", "alice")
            view = await repo.create(
                alice,
                surface="inline",
                session_id="chat-1",
                title="Snapshot",
                spec=_spec(),
                initial_data={"metrics": {"points": [1]}},
            )
            conn = await db._ensure_connected()
            old = int((time.time() - 8 * 24 * 60 * 60) * 1000)
            await conn.execute(
                "UPDATE ui_views SET last_viewed_at_ms=? WHERE id=?", (old, view["id"]),
            )
            await conn.commit()
            frozen = await repo.freeze_inactive_inline(inactive_before_ms=int(time.time() * 1000))
            assert frozen == [(view["id"], 1)]
            snapshot = await repo.get(view["id"], alice)
            assert snapshot["frozen"] is True
            assert snapshot["data"]["metrics"]["status"] == "stale"
            active = await repo.reactivate(
                view["id"], alice, expected_revision=1,
            )
            assert active["revision"] == 1 and active["frozen"] is False
            assert active["data"]["metrics"]["status"] == "ready"
        finally:
            await db.close()


@test("custom_views", "runtime starts while-visible sources on demand and actions are idempotent")
async def t_custom_view_runtime_action(ctx: TestContext) -> None:
    from src.custom_views.service import CustomViewService

    class _WS:
        closed = False

        def __init__(self) -> None:
            self.frames: list[dict] = []

        async def send_json(self, frame: dict) -> None:
            self.frames.append(frame)

    with tempfile.TemporaryDirectory(prefix="oa-ui-runtime-") as raw:
        root = Path(raw)
        watched = root / "source.json"
        watched.write_text('{"file": 7}', encoding="utf-8")
        db = await _db(root)
        service = CustomViewService(db)
        try:
            alice = _access("tenant-a", "alice")
            view = await service.create(
                alice,
                surface="sidebar",
                title="Live",
                spec={
                    "schemaVersion": 1,
                    "root": {
                        "type": "stack",
                        "children": [
                            {"type": "text", "props": {"value": "{{data.feed.value}}"}},
                            {"type": "button", "props": {"action": "append"}},
                        ],
                    },
                },
                sources={
                    "feed": {
                        "driver": "static",
                        "activation": "while_visible",
                        "config": {"value": {"value": 42}},
                    },
                    "file": {
                        "driver": "file_watch",
                        "activation": "manual",
                        "config": {"path": str(watched), "intervalMs": 1000},
                    },
                    "poll": {
                        "driver": "command_poll",
                        "activation": "manual",
                        "config": {
                            "argv": [sys.executable, "-c", "print('{\"poll\": 8}')"],
                            "intervalMs": 1000,
                        },
                    },
                    "stream": {
                        "driver": "command_stream",
                        "activation": "manual",
                        "config": {
                            "argv": [sys.executable, "-c", "print('{\"stream\": 9}', flush=True)"],
                            "timeoutMs": 2000,
                        },
                    },
                },
                actions={
                    "append": {
                        "kind": "set_data",
                        "config": {"key": "points", "mode": "append", "maxItems": 2},
                    },
                },
            )
            # No subscriber means no while-visible task/value publication.
            assert not service.runtime._source_tasks
            ws = _WS()
            await service.subscribe(
                ws,
                subscription_id="sub-1",
                view_id=view["id"],
                access=alice,
            )
            for _ in range(50):
                if any(frame.get("type") == "ui_data" for frame in ws.frames):
                    break
                await asyncio.sleep(0.01)
            assert any(frame.get("type") == "ui_snapshot" for frame in ws.frames)
            assert any(frame.get("type") == "ui_data" and frame.get("key") == "feed" for frame in ws.frames)
            for key in ("file", "poll", "stream"):
                await service.refresh_source(view["id"], key, alice)
            for _ in range(100):
                seen = {
                    frame.get("key") for frame in ws.frames
                    if frame.get("type") == "ui_data"
                }
                if {"file", "poll", "stream"}.issubset(seen):
                    break
                await asyncio.sleep(0.01)
            values = {
                frame["key"]: frame["value"] for frame in ws.frames
                if frame.get("type") == "ui_data" and frame.get("key") in {"file", "poll", "stream"}
            }
            assert values == {
                "file": {"file": 7},
                "poll": {"poll": 8},
                "stream": {"stream": 9},
            }, values
            first = await service.invoke_action(
                view["id"], "append", alice, input_value=1, idempotency_key="same",
            )
            duplicate = await service.invoke_action(
                view["id"], "append", alice, input_value=2, idempotency_key="same",
            )
            assert first["id"] == duplicate["id"] and duplicate["status"] == "completed"
            stored = await service.get(view["id"], alice)
            assert stored["data"]["points"]["value"] == [1]
            await service.unsubscribe(ws, "sub-1")
        finally:
            await service.close()
            await db.close()


@test("custom_views", "frozen and expired Views cannot mutate data or execute work")
async def t_custom_view_lifecycle_blocks_work(ctx: TestContext) -> None:
    from src.custom_views.repository import CustomViewImmutable
    from src.custom_views.service import CustomViewService

    async def expect_blocked(awaitable) -> None:
        try:
            await awaitable
        except CustomViewImmutable:
            return
        raise AssertionError("frozen/expired Custom View executed server-side work")

    with tempfile.TemporaryDirectory(prefix="oa-ui-lifecycle-") as raw:
        db = await _db(Path(raw))
        service = CustomViewService(db)
        try:
            alice = _access("tenant-a", "alice")
            view = await service.create(
                alice,
                surface="inline",
                session_id="chat-lifecycle",
                title="Lifecycle",
                markup='<stack><button action="set">Set</button></stack>',
                sources={
                    "sample": {
                        "driver": "static",
                        "activation": "manual",
                        "config": {"value": 1},
                    },
                },
                actions={
                    "set": {
                        "kind": "set_data",
                        "config": {"key": "answer", "value": 42},
                    },
                },
            )
            frozen = await service.set_frozen(view["id"], alice, frozen=True)
            assert frozen["frozen"] is True and frozen["canExecute"] is False
            await expect_blocked(service.set_data(view["id"], "manual", 1, alice))
            await expect_blocked(service.refresh_source(view["id"], "sample", alice))
            await expect_blocked(
                service.invoke_action(
                    view["id"], "set", alice,
                    revision=1, idempotency_key="frozen-action",
                )
            )

            active = await service.reactivate(
                view["id"], alice, expected_revision=1,
            )
            assert active["revision"] == 1 and active["canExecute"] is True
            completed = await service.invoke_action(
                view["id"], "set", alice,
                revision=1, idempotency_key="active-inline-action",
            )
            assert completed["status"] == "completed"

            # Simulate the wall clock crossing expiresAt without a timer
            # rewriting status. Execution boundaries must evaluate the TTL.
            conn = await db._ensure_connected()
            await conn.execute(
                "UPDATE ui_views SET status='active', expires_at_ms=? WHERE id=?",
                (int(time.time() * 1000) - 1, view["id"]),
            )
            await conn.commit()
            expired = await service.get(view["id"], alice)
            assert expired["status"] == "expired" and expired["canExecute"] is False
            await expect_blocked(service.set_data(view["id"], "manual", 2, alice))
            await expect_blocked(service.refresh_source(view["id"], "sample", alice))
            await expect_blocked(
                service.invoke_action(
                    view["id"], "set", alice,
                    revision=1, idempotency_key="expired-action",
                )
            )

            resumed = await service.reactivate(
                view["id"], alice, expected_revision=1,
            )
            assert resumed["revision"] == 1 and resumed["canExecute"] is True
        finally:
            await service.close()
            await db.close()


@test("custom_views", "local E2E startup never resumes persistent sources")
async def t_custom_view_local_e2e_does_not_resume_always(ctx: TestContext) -> None:
    from src.custom_views.service import CustomViewService

    with tempfile.TemporaryDirectory(prefix="oa-ui-e2e-runtime-") as raw:
        db = await _db(Path(raw))
        service = CustomViewService(db)
        try:
            alice = _access("tenant-a", "alice")
            view = await service.create(
                alice,
                surface="sidebar",
                title="Persistent",
                spec=_spec(),
                sources={
                    "persistent": {
                        "driver": "command_stream",
                        "activation": "always",
                        "config": {
                            "argv": [sys.executable, "-c", "import time; time.sleep(60)"],
                            "timeoutMs": 30_000,
                        },
                    },
                },
            )
            await service.start(resume_always=False)
            await asyncio.sleep(0)
            assert not service.runtime._source_tasks
            # An explicit, authenticated refresh remains available to a
            # controlled fixture even though automatic resume is parked.
            await service.refresh_source(view["id"], "persistent", alice)
            assert (view["id"], "persistent") in service.runtime._source_tasks
        finally:
            await service.close()
            await db.close()


@test("custom_views", "subscriptions are snapshot-first, bounded, and cleaned after send failure")
async def t_custom_view_subscription_boundaries(ctx: TestContext) -> None:
    from src.custom_views.repository import CustomViewRateLimited, CustomViewRepository
    from src.custom_views.runtime import CustomViewRuntime
    from src.custom_views.service import CustomViewService

    class _WS:
        closed = False

        def __init__(self) -> None:
            self.frames: list[dict] = []

        async def send_json(self, frame: dict) -> None:
            self.frames.append(frame)

    with tempfile.TemporaryDirectory(prefix="oa-ui-subscriptions-") as raw:
        db = await _db(Path(raw))
        repo = CustomViewRepository(db)
        send_enabled = True

        async def sender(ws, frame) -> bool:
            if not send_enabled:
                return False
            await ws.send_json(frame)
            return True

        runtime = CustomViewRuntime(
            repo,
            send_json=sender,
            max_subscriptions_per_socket=2,
            max_subscriptions_total=3,
        )
        service = CustomViewService(db, repository=repo, runtime=runtime)
        try:
            alice = _access("tenant-a", "alice")
            views = []
            for index in range(5):
                views.append(
                    await service.create(
                        alice,
                        surface="sidebar",
                        title=f"Bounded {index}",
                        spec=_spec(str(index)),
                        sources=(
                            {
                                "instant": {
                                    "driver": "static",
                                    "activation": "while_visible",
                                    "config": {"value": index},
                                },
                            }
                            if index in {0, 4}
                            else None
                        ),
                    )
                )

            first_ws = _WS()
            await service.subscribe(
                first_ws, subscription_id="one", view_id=views[0]["id"], access=alice,
            )
            for _ in range(50):
                if any(frame.get("type") == "ui_data" for frame in first_ws.frames[1:]):
                    break
                await asyncio.sleep(0.01)
            assert first_ws.frames[0]["type"] == "ui_snapshot", first_ws.frames
            assert any(frame.get("type") == "ui_data" for frame in first_ws.frames[1:])

            await service.subscribe(
                first_ws, subscription_id="two", view_id=views[1]["id"], access=alice,
            )
            try:
                await service.subscribe(
                    first_ws, subscription_id="three", view_id=views[2]["id"], access=alice,
                )
            except CustomViewRateLimited:
                pass
            else:
                raise AssertionError("per-socket subscription cap was not enforced")

            second_ws = _WS()
            await service.subscribe(
                second_ws, subscription_id="three", view_id=views[2]["id"], access=alice,
            )
            try:
                await service.subscribe(
                    _WS(), subscription_id="four", view_id=views[3]["id"], access=alice,
                )
            except CustomViewRateLimited:
                pass
            else:
                raise AssertionError("global subscription cap was not enforced")

            # A failed replacement send invalidates every subscription owned by
            # that dead socket; it must not retain a producer capability.
            send_enabled = False
            await service.subscribe(
                first_ws, subscription_id="one", view_id=views[3]["id"], access=alice,
            )
            assert service.runtime.subscription(first_ws, "one") is None
            assert service.runtime.subscription(first_ws, "two") is None

            # There is room after cleanup. A brand-new failed snapshot must not
            # register the subscriber nor start its while-visible source.
            failed_ws = _WS()
            await service.subscribe(
                failed_ws, subscription_id="failed", view_id=views[4]["id"], access=alice,
            )
            assert service.runtime.subscription(failed_ws, "failed") is None
            assert (views[4]["id"], "instant") not in service.runtime._source_tasks
        finally:
            await service.close()
            await db.close()


@test("custom_views", "cancelled and oversized command actions terminate their process trees")
async def t_custom_view_action_process_cleanup(ctx: TestContext) -> None:
    import psutil

    from src.custom_views.repository import CustomViewError
    from src.custom_views.service import CustomViewService

    async def wait_for_file(path: Path) -> tuple[int, int]:
        for _ in range(200):
            if path.exists() and path.read_text(encoding="utf-8").strip():
                parent, child = path.read_text(encoding="utf-8").split()
                return int(parent), int(child)
            await asyncio.sleep(0.01)
        raise AssertionError("command action did not publish its process ids")

    async def assert_reaped(*pids: int) -> None:
        for _ in range(200):
            alive = []
            for pid in pids:
                try:
                    process = psutil.Process(pid)
                    if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                        alive.append(pid)
                except psutil.Error:
                    pass
            if not alive:
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"command process tree survived cleanup: {alive}")

    with tempfile.TemporaryDirectory(prefix="oa-ui-action-cleanup-") as raw:
        root = Path(raw)
        cancelled_pids = root / "cancelled.pids"
        oversized_pids = root / "oversized.pids"
        sleeper = (
            "import os,subprocess,sys,time;"
            "child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)']);"
            "time.sleep(.1);"
            "open(sys.argv[1],'w').write(f'{os.getpid()} {child.pid}');"
            "time.sleep(60)"
        )
        oversized = (
            "import os,subprocess,sys,time;"
            "child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)']);"
            "time.sleep(.1);"
            "open(sys.argv[1],'w').write(f'{os.getpid()} {child.pid}');"
            "print('x'*(1024*1024+1),flush=True);time.sleep(60)"
        )
        db = await _db(root)
        service = CustomViewService(db)
        try:
            alice = _access("tenant-a", "alice")
            view = await service.create(
                alice,
                surface="sidebar",
                title="Command cleanup",
                markup=(
                    '<stack><button action="cancel">Cancel</button>'
                    '<button action="oversized">Oversized</button></stack>'
                ),
                actions={
                    "cancel": {
                        "kind": "command",
                        "config": {
                            "argv": [sys.executable, "-c", sleeper, str(cancelled_pids)],
                            "timeoutMs": 30_000,
                        },
                    },
                    "oversized": {
                        "kind": "command",
                        "config": {
                            "argv": [sys.executable, "-c", oversized, str(oversized_pids)],
                            "timeoutMs": 30_000,
                        },
                    },
                },
            )
            cancelled = asyncio.create_task(
                service.invoke_action(
                    view["id"], "cancel", alice, idempotency_key="cancel-once",
                )
            )
            cancel_parent, cancel_child = await wait_for_file(cancelled_pids)
            cancelled.cancel()
            try:
                await cancelled
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("action cancellation was swallowed")
            await assert_reaped(cancel_parent, cancel_child)
            terminal = await service.invoke_action(
                view["id"], "cancel", alice, idempotency_key="cancel-once",
            )
            assert terminal["status"] == "failed"
            assert terminal["error"] == "CancelledError"
            assert terminal["completedAt"] is not None

            try:
                await service.invoke_action(
                    view["id"], "oversized", alice, idempotency_key="oversized-once",
                )
            except CustomViewError:
                pass
            else:
                raise AssertionError("oversized command output was accepted")
            oversize_parent, oversize_child = await wait_for_file(oversized_pids)
            await assert_reaped(oversize_parent, oversize_child)
        finally:
            await service.close()
            await db.close()


@test("custom_views", "live ACL revocation invalidates the subscribed client cache")
async def t_custom_view_live_acl_revoke(ctx: TestContext) -> None:
    from src.custom_views.service import CustomViewService

    class _WS:
        closed = False

        def __init__(self) -> None:
            self.frames: list[dict] = []

        async def send_json(self, frame: dict) -> None:
            self.frames.append(frame)

    with tempfile.TemporaryDirectory(prefix="oa-ui-revoke-") as raw:
        db = await _db(Path(raw))
        service = CustomViewService(db)
        try:
            alice = _access("tenant-a", "alice")
            bob = _access("tenant-a", "bob")
            view = await service.create(
                alice,
                surface="sidebar",
                title="Revocable",
                spec=_spec(),
                sources={
                    "feed": {
                        "driver": "static",
                        "activation": "while_visible",
                        "config": {"value": 1},
                    },
                },
            )
            grant = await service.set_grant(
                view["id"],
                alice,
                principal_type="user",
                principal_id="bob",
                permissions=["view"],
                expected_acl_version=1,
            )
            assert grant["aclVersion"] == 2
            ws = _WS()
            await service.subscribe(
                ws,
                subscription_id="bob-sub",
                view_id=view["id"],
                access=bob,
            )
            assert service.runtime.subscription(ws, "bob-sub") is not None
            await service.delete_grant(
                view["id"],
                alice,
                principal_type="user",
                principal_id="bob",
                expected_acl_version=2,
            )
            revoked = [frame for frame in ws.frames if frame.get("type") == "ui_error"]
            assert revoked and revoked[-1] == {
                "type": "ui_error",
                "code": "access_revoked",
                "message": "Custom View access was revoked",
                "subscriptionId": "bob-sub",
                "viewId": view["id"],
            }
            assert service.runtime.subscription(ws, "bob-sub") is None
            frame_count = len(ws.frames)
            await service.set_data(view["id"], "feed", 2, alice)
            assert len(ws.frames) == frame_count
        finally:
            await service.close()
            await db.close()


@test("custom_views", "historical sidebar layouts are renderable but actions are revoked")
async def t_custom_view_revision_runtime(ctx: TestContext) -> None:
    from src.custom_views.repository import CustomViewNotFound
    from src.custom_views.service import CustomViewService

    class _WS:
        closed = False

        def __init__(self) -> None:
            self.frames: list[dict] = []

        async def send_json(self, frame: dict) -> None:
            self.frames.append(frame)

    with tempfile.TemporaryDirectory(prefix="oa-ui-revisions-") as raw:
        db = await _db(Path(raw))
        service = CustomViewService(db)
        try:
            alice = _access("tenant-a", "alice")
            await service.start(resume_always=False)
            markup = '<stack><button action="set">Set</button></stack>'
            view = await service.create(
                alice,
                surface="sidebar",
                title="Revision one",
                markup=markup,
                sources={
                    "poll": {
                        "driver": "command_poll",
                        "activation": "while_visible",
                        "config": {
                            "argv": [sys.executable, "-c", "print('{\"revision\": 1}')"],
                            "intervalMs": 1000,
                        },
                    },
                },
                actions={
                    "set": {"kind": "set_data", "config": {"key": "chosen", "value": "v1"}},
                },
            )
            old_ws = _WS()
            old_snapshot = await service.subscribe(
                old_ws,
                subscription_id="old",
                view_id=view["id"],
                access=alice,
            )
            assert old_snapshot["view"]["revision"] == 1
            for _ in range(50):
                if (view["id"], "poll") in service.runtime._source_tasks:
                    break
                await asyncio.sleep(0.01)
            assert (view["id"], "poll") in service.runtime._source_tasks

            updated = await service.update(
                view["id"],
                alice,
                expected_revision=1,
                title="Revision two",
                actions={
                    "set": {"kind": "set_data", "config": {"key": "chosen", "value": "v2"}},
                },
            )
            assert updated["revision"] == 2
            # The only subscriber is pinned to revision 1, so the revision-2
            # while-visible source must remain stopped.
            assert (view["id"], "poll") not in service.runtime._source_tasks
            historical = await service.get(view["id"], alice, revision=1)
            assert historical["canExecute"] is False
            try:
                await service.handle_ws_frame(
                    old_ws,
                    {
                        "type": "ui_action",
                        "subscriptionId": "old",
                        "actionId": "set",
                        "idempotencyKey": "old-action",
                    },
                    alice,
                )
            except CustomViewNotFound:
                pass
            else:
                raise AssertionError("a superseded sidebar action remained executable")
            assert "chosen" not in (await service.get(view["id"], alice))["data"]

            latest_ws = _WS()
            latest_snapshot = await service.subscribe(
                latest_ws,
                subscription_id="latest-two",
                view_id=view["id"],
                access=alice,
            )
            assert latest_snapshot["view"]["revision"] == 2
            assert latest_snapshot["view"]["latestRevision"] == 2
            for _ in range(50):
                if (view["id"], "poll") in service.runtime._source_tasks:
                    break
                await asyncio.sleep(0.01)
            assert (view["id"], "poll") in service.runtime._source_tasks
            await service.handle_ws_frame(
                latest_ws,
                {
                    "type": "ui_action",
                    "subscriptionId": "latest-two",
                    "actionId": "set",
                    "idempotencyKey": "new-action",
                },
                alice,
            )
            assert (await service.get(view["id"], alice))["data"]["chosen"]["value"] == "v2"
            # The data plane is shared even though layouts and action
            # definitions are revision-pinned.
            assert any(
                frame.get("type") == "ui_data"
                and frame.get("key") == "chosen"
                and frame.get("value") == "v2"
                for frame in old_ws.frames
            )

            source = await service.configure_source(
                view["id"],
                "poll",
                {
                    "driver": "command_poll",
                    "activation": "while_visible",
                    "config": {
                        "argv": [sys.executable, "-c", "print('{\"revision\": 3}')"],
                        "intervalMs": 1000,
                    },
                },
                alice,
                expected_revision=2,
            )
            assert source["revision"] == 3
            assert (view["id"], "poll") not in service.runtime._source_tasks
            historical_ws = _WS()
            await service.subscribe(
                historical_ws,
                subscription_id="historical-two",
                view_id=view["id"],
                revision=2,
                access=alice,
            )
            assert (view["id"], "poll") not in service.runtime._source_tasks
            newest_ws = _WS()
            await service.subscribe(
                newest_ws,
                subscription_id="latest-three",
                view_id=view["id"],
                access=alice,
            )
            for _ in range(50):
                if (view["id"], "poll") in service.runtime._source_tasks:
                    break
                await asyncio.sleep(0.01)
            assert (view["id"], "poll") in service.runtime._source_tasks
        finally:
            await service.close()
            await db.close()


@test("custom_views", "checkpoint serialization persists manual writes and source generations")
async def t_custom_view_checkpoint_generation(ctx: TestContext) -> None:
    from src.custom_views.service import CustomViewService

    with tempfile.TemporaryDirectory(prefix="oa-ui-checkpoint-") as raw:
        db = await _db(Path(raw))
        alice = _access("tenant-a", "alice")
        service = CustomViewService(db)
        try:
            view = await service.create(
                alice,
                surface="sidebar",
                title="Checkpoint",
                spec=_spec(),
                sources={
                    "sample": {
                        "driver": "static",
                        "activation": "manual",
                        "config": {"value": {"source": 1}},
                    },
                },
            )
            await service.start(resume_always=False)
            await service.refresh_source(view["id"], "sample", alice)
            for _ in range(50):
                first = await service.repository.data_state_for_runtime(view["id"], "sample")
                if first and first["generation"] == 1:
                    break
                await asyncio.sleep(0.01)
            assert first is not None and first["generation"] == 1

            await service.runtime._publish_source_value(
                view["id"],
                "sample",
                tenant_id=alice.tenant_id,
                value={"pending": True},
                generation=1,
                expires_at=None,
            )
            await asyncio.gather(
                service.set_data(view["id"], "sample", {"manual": True}, alice),
                service.runtime._flush_checkpoint((view["id"], "sample")),
            )
            stored = await service.repository.data_state_for_runtime(view["id"], "sample")
            assert stored is not None and stored["value"] == {"manual": True}
        finally:
            await service.close()

        restarted = CustomViewService(db)
        try:
            await restarted.start(resume_always=False)
            await restarted.refresh_source(view["id"], "sample", alice)
            for _ in range(50):
                second = await restarted.repository.data_state_for_runtime(view["id"], "sample")
                if second and second["generation"] >= 2:
                    break
                await asyncio.sleep(0.01)
            assert second is not None and second["generation"] == 2
        finally:
            await restarted.close()
            await db.close()


@test("custom_views", "source outputSchema rejects invalid static, push, and runtime values")
async def t_custom_view_output_schema(ctx: TestContext) -> None:
    from src.custom_views.repository import CustomViewInputError
    from src.custom_views.service import CustomViewService

    with tempfile.TemporaryDirectory(prefix="oa-ui-output-schema-") as raw:
        db = await _db(Path(raw))
        service = CustomViewService(db)
        try:
            alice = _access("tenant-a", "alice")
            try:
                await service.create(
                    alice,
                    surface="sidebar",
                    title="Invalid static",
                    spec=_spec(),
                    sources={
                        "typed": {
                            "driver": "static",
                            "outputSchema": {"type": "integer"},
                            "config": {"value": "not-an-integer"},
                        },
                    },
                )
            except CustomViewInputError as exc:
                assert "outputSchema" in str(exc)
            else:
                raise AssertionError("invalid static output bypassed outputSchema")

            view = await service.create(
                alice,
                surface="sidebar",
                title="Typed sources",
                spec=_spec(),
                sources={
                    "pushed": {
                        "driver": "push",
                        "activation": "manual",
                        "outputSchema": {
                            "type": "object",
                            "required": ["count"],
                            "properties": {"count": {"type": "integer"}},
                            "additionalProperties": False,
                        },
                        "config": {},
                    },
                    "produced": {
                        "driver": "command_poll",
                        "activation": "manual",
                        "outputSchema": {"type": "integer"},
                        "config": {
                            "argv": [sys.executable, "-c", "print('\\\"wrong\\\"')"],
                            "intervalMs": 1000,
                        },
                    },
                },
            )
            try:
                await service.set_data(view["id"], "pushed", {"count": "wrong"}, alice)
            except CustomViewInputError as exc:
                assert "outputSchema" in str(exc)
            else:
                raise AssertionError("invalid pushed data bypassed outputSchema")
            accepted = await service.set_data(
                view["id"], "pushed", {"count": 3}, alice,
            )
            assert accepted["value"] == {"count": 3}

            await service.start(resume_always=False)
            await service.refresh_source(view["id"], "produced", alice)
            for _ in range(100):
                produced = await service.repository.data_state_for_runtime(
                    view["id"], "produced",
                )
                if produced and produced["status"] == "error":
                    break
                await asyncio.sleep(0.01)
            assert produced is not None and produced["status"] == "error"
            assert produced["value"] is None
        finally:
            await service.close()
            await db.close()


@test("custom_views", "command actions treat clicked input as JSON data, never shell source")
async def t_custom_view_action_injection(ctx: TestContext) -> None:
    from src.custom_views.service import CustomViewService

    with tempfile.TemporaryDirectory(prefix="oa-ui-action-") as raw:
        root = Path(raw)
        db = await _db(root)
        service = CustomViewService(db)
        try:
            alice = _access("tenant-a", "alice")
            escaped_target = root / "must-not-exist"
            view = await service.create(
                alice,
                surface="sidebar",
                title="Safe action",
                markup='<button action="echo">Echo</button>',
                actions={
                    "echo": {
                        "kind": "command",
                        "config": {
                            "argv": [
                                sys.executable,
                                "-c",
                                "import json,sys; value=json.load(sys.stdin); "
                                "print(json.dumps({'value': value}))",
                            ],
                        },
                    },
                },
            )
            malicious = f"'; open({str(escaped_target)!r}, 'w').write('owned'); #"
            result = await service.invoke_action(
                view["id"],
                "echo",
                alice,
                input_value=malicious,
                idempotency_key="literal-input",
            )
            assert result["status"] == "completed"
            assert result["result"] == {"value": malicious}
            assert not escaped_target.exists()
        finally:
            await service.close()
            await db.close()


@test("custom_views", "ui-manager preflights bundle paths and sizes before decoding")
async def t_ui_manager_bundle_preflight(ctx: TestContext) -> None:
    from types import SimpleNamespace

    from src.mcp.servers.ui_manager import adapters

    with tempfile.TemporaryDirectory(prefix="oa-ui-toolkit-") as raw:
        db = await _db(Path(raw))
        pool = SimpleNamespace(_db=db)
        original_asset_limit = adapters.MAX_ASSET_BYTES
        original_bundle_limit = adapters.MAX_BUNDLE_BYTES
        original_script_limit = adapters.MAX_SCRIPT_BYTES
        original_decode = adapters.base64.b64decode
        adapters.MAX_ASSET_BYTES = 8
        adapters.MAX_BUNDLE_BYTES = 8
        adapters.MAX_SCRIPT_BYTES = 8
        try:
            toolkit = adapters.build_runtime_toolkit(pool)
            create = toolkit.async_functions["ui_create_view"].entrypoint
            assert create is not None
            decode_calls = 0

            def observed_decode(*args, **kwargs):
                nonlocal decode_calls
                decode_calls += 1
                return original_decode(*args, **kwargs)

            adapters.base64.b64decode = observed_decode
            aggregate = await create(
                title="Rejected aggregate",
                markup="<text>safe</text>",
                assets_base64={
                    "one.bin": base64.b64encode(b"123456").decode("ascii"),
                    "two.bin": base64.b64encode(b"abcdef").decode("ascii"),
                },
            )
            assert aggregate["ok"] is False and aggregate["error"] == "invalid_view"
            assert decode_calls == 0

            oversized = "sensitive-asset-payload"
            result = await create(
                title="Rejected",
                markup="<text>safe</text>",
                assets_base64={
                    "large.bin": base64.b64encode(oversized.encode()).decode("ascii"),
                },
            )
            assert result["ok"] is False and result["error"] == "invalid_view"
            assert oversized not in str(result)
            traversal = await create(
                title="Rejected path",
                markup="<text>safe</text>",
                scripts={"../escape.py": "print(1)"},
            )
            assert traversal["ok"] is False and traversal["error"] == "invalid_view"
        finally:
            adapters.MAX_ASSET_BYTES = original_asset_limit
            adapters.MAX_BUNDLE_BYTES = original_bundle_limit
            adapters.MAX_SCRIPT_BYTES = original_script_limit
            adapters.base64.b64decode = original_decode
            service = getattr(db, "_custom_view_service", None)
            if service is not None:
                await service.close()
            await db.close()


@test("custom_views", "REST contract is ACL checked from create through asset delivery")
async def t_custom_view_rest(ctx: TestContext) -> None:
    from types import SimpleNamespace

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

    from src.gateway.api import custom_views as api

    class _Gateway:
        def __init__(self, db) -> None:
            self.agent = SimpleNamespace(memory_db=db, _mcp=None)
            self._scheduler = None
            self.events: list[tuple[str, str, str]] = []

        async def broadcast_resource(self, resource: str, action: str, resource_id: str) -> None:
            self.events.append((resource, action, resource_id))

    with tempfile.TemporaryDirectory(prefix="oa-ui-rest-") as raw:
        root = Path(raw)
        db = await _db(root)
        gateway = _Gateway(db)
        tenant = "tenant-rest"

        @web.middleware
        async def identity(request, handler):
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

        app = web.Application(middlewares=[api.body_limit_middleware, identity])
        app["gateway"] = gateway
        app.router.add_get("/api/ui/capabilities", api.handle_capabilities)
        app.router.add_get("/api/ui/views", api.handle_list)
        app.router.add_post("/api/ui/views", api.handle_create)
        app.router.add_get("/api/ui/views/{id}", api.handle_get)
        app.router.add_put("/api/ui/views/{id}", api.handle_update)
        app.router.add_delete("/api/ui/views/{id}", api.handle_delete)
        app.router.add_put("/api/ui/views/{id}/data/{key}", api.handle_set_data)
        app.router.add_put("/api/ui/views/{id}/sources/{key}", api.handle_configure_source)
        app.router.add_delete("/api/ui/views/{id}/sources/{key}", api.handle_delete_source)
        app.router.add_post("/api/ui/views/{id}/actions/{action_id}", api.handle_action)
        app.router.add_get("/api/ui/views/{id}/grants", api.handle_list_grants)
        app.router.add_put("/api/ui/views/{id}/grants", api.handle_set_grant)
        app.router.add_delete("/api/ui/views/{id}/grants", api.handle_delete_grant)
        app.router.add_get(
            "/api/ui/views/{id}/revisions/{revision}/assets/{path:.+}",
            api.handle_asset,
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            pixel = b"\x89PNG\r\n\x1a\nrest"
            caps = await client.get("/api/ui/capabilities")
            assert caps.status == 200
            assert (await caps.json())["capabilities"]["schemaVersions"] == [1]
            created_response = await client.post(
                "/api/ui/views",
                json={
                    "surface": "sidebar",
                    "title": "REST board",
                    "spec": {
                        "schemaVersion": 1,
                        "root": {
                            "type": "button",
                            "props": {"label": "Set", "action": "set"},
                        },
                    },
                    "actions": {
                        "set": {"kind": "set_data", "config": {"key": "answer", "value": 42}},
                    },
                    "assets": {"pixel.png": base64.b64encode(pixel).decode("ascii")},
                },
            )
            assert created_response.status == 201, await created_response.text()
            assert created_response.headers["ETag"] == '"ui-view-1"'
            created = (await created_response.json())["view"]
            view_id = created["id"]

            # aiohttp defaults to 1 MiB. A maximally-sized valid script plus
            # its JSON envelope crosses that threshold and proves the
            # route-scoped bundle allowance is actually active.
            large_script = "#" * (1024 * 1024)
            large_response = await client.post(
                "/api/ui/views",
                json={
                    "surface": "sidebar",
                    "title": "Large valid bundle",
                    "markup": "<text>large</text>",
                    "scripts": {"large.py": large_script},
                },
            )
            assert large_response.status == 201, await large_response.text()

            original_request_limit = api.MAX_CUSTOM_VIEW_REQUEST_BYTES
            api.MAX_CUSTOM_VIEW_REQUEST_BYTES = 1024
            try:
                oversized_response = await client.post(
                    "/api/ui/views",
                    json={
                        "surface": "sidebar",
                        "title": "Too large",
                        "description": "x" * 2048,
                        "markup": "<text>never parsed</text>",
                    },
                )
                assert oversized_response.status == 413
                assert (await oversized_response.json())["error"]["code"] == "request_too_large"
            finally:
                api.MAX_CUSTOM_VIEW_REQUEST_BYTES = original_request_limit

            listed = await client.get("/api/ui/views?surface=sidebar&limit=10")
            assert listed.status == 200
            assert view_id in [row["id"] for row in (await listed.json())["views"]]
            denied = await client.get(
                f"/api/ui/views/{view_id}", headers={"X-Test-Handle": "bob"},
            )
            assert denied.status == 404
            action = await client.post(
                f"/api/ui/views/{view_id}/actions/set",
                json={"idempotencyKey": "rest-once"},
            )
            assert action.status == 200, await action.text()
            loaded = await client.get(f"/api/ui/views/{view_id}")
            assert (await loaded.json())["view"]["data"]["answer"]["value"] == 42

            # Assets are immutable-revision scoped and re-use the View ACL.
            asset = root / "ui" / "views" / view_id / "revisions" / "1" / "assets" / "pixel.png"
            served = await client.get(
                f"/api/ui/views/{view_id}/revisions/1/assets/pixel.png",
            )
            assert served.status == 200 and await served.read() == pixel == asset.read_bytes()
            assert served.headers["Cache-Control"] == "private, no-store"
            assert served.headers["X-Content-Type-Options"] == "nosniff"
            # The Iroh aiohttp transport cannot deliver FileResponse/sendfile
            # bodies. Keep this handler on the ordinary buffered Response path.
            direct_request = make_mocked_request(
                "GET",
                f"/api/ui/views/{view_id}/revisions/1/assets/pixel.png",
                match_info={
                    "id": view_id,
                    "revision": "1",
                    "path": "pixel.png",
                },
                app=app,
            )
            direct_cert = SimpleNamespace(
                network_id=tenant,
                handle="alice",
                device_pubkey_hex="device-alice",
                capabilities=[],
            )
            direct_request["device_cert"] = direct_cert
            direct_request["network_id"] = tenant
            direct_request["user_handle"] = "alice"
            direct_request["client_id"] = direct_cert.device_pubkey_hex
            direct = await api.handle_asset(direct_request)
            assert type(direct) is web.Response and direct.body == pixel
            denied_asset = await client.get(
                f"/api/ui/views/{view_id}/revisions/1/assets/pixel.png",
                headers={"X-Test-Handle": "bob"},
            )
            assert denied_asset.status == 404

            missing_precondition = await client.put(
                f"/api/ui/views/{view_id}", json={"title": "No lock"},
            )
            assert missing_precondition.status == 400
            invalid_markup = await client.put(
                f"/api/ui/views/{view_id}",
                headers={"If-Match": '"ui-view-1"'},
                json={"markup": "<script />"},
            )
            assert invalid_markup.status == 400
            updated_response = await client.put(
                f"/api/ui/views/{view_id}",
                headers={"If-Match": '"ui-view-1"'},
                json={"title": "REST board v2"},
            )
            assert updated_response.status == 200, await updated_response.text()
            assert updated_response.headers["ETag"] == '"ui-view-2"'
            stale = await client.put(
                f"/api/ui/views/{view_id}",
                headers={"If-Match": '"ui-view-1"'},
                json={"title": "stale"},
            )
            assert stale.status == 409
            pinned = await client.get(f"/api/ui/views/{view_id}?revision=1")
            assert pinned.status == 200
            assert pinned.headers["ETag"] == '"ui-view-1"'
            assert (await pinned.json())["view"]["title"] == "REST board"

            source = await client.put(
                f"/api/ui/views/{view_id}/sources/pushed",
                headers={"If-Match": '"ui-view-2"'},
                json={
                    "driver": "push",
                    "activation": "manual",
                    "config": {},
                },
            )
            assert source.status == 200, await source.text()
            assert source.headers["ETag"] == '"ui-view-3"'
            stale_source_delete = await client.delete(
                f"/api/ui/views/{view_id}/sources/pushed",
                headers={"If-Match": '"ui-view-2"'},
            )
            assert stale_source_delete.status == 409
            source_delete = await client.delete(
                f"/api/ui/views/{view_id}/sources/pushed",
                headers={"If-Match": '"ui-view-3"'},
            )
            assert source_delete.status == 200
            assert source_delete.headers["ETag"] == '"ui-view-4"'

            grant = await client.put(
                f"/api/ui/views/{view_id}/grants",
                json={
                    "principalType": "user",
                    "principalId": "bob",
                    "permissions": ["view"],
                    "expectedAclVersion": 1,
                },
            )
            assert grant.status == 200, await grant.text()
            assert (await grant.json())["aclVersion"] == 2
            allowed = await client.get(
                f"/api/ui/views/{view_id}", headers={"X-Test-Handle": "bob"},
            )
            assert allowed.status == 200
            allowed_asset = await client.get(
                f"/api/ui/views/{view_id}/revisions/1/assets/pixel.png",
                headers={"X-Test-Handle": "bob"},
            )
            assert allowed_asset.status == 200 and await allowed_asset.read() == pixel
            revoke = await client.delete(
                f"/api/ui/views/{view_id}/grants",
                json={
                    "principalType": "user",
                    "principalId": "bob",
                    "expectedAclVersion": 2,
                },
            )
            assert revoke.status == 200
            denied_again = await client.get(
                f"/api/ui/views/{view_id}", headers={"X-Test-Handle": "bob"},
            )
            assert denied_again.status == 404
            denied_asset_again = await client.get(
                f"/api/ui/views/{view_id}/revisions/1/assets/pixel.png",
                headers={"X-Test-Handle": "bob"},
            )
            assert denied_asset_again.status == 404

            deleted = await client.delete(
                f"/api/ui/views/{view_id}", headers={"If-Match": '"ui-view-4"'},
            )
            assert deleted.status == 200
            assert (await client.get(f"/api/ui/views/{view_id}")).status == 404
        finally:
            service = getattr(db, "_custom_view_service", None)
            if service is not None:
                await service.close()
            await client.close()
            await db.close()


@test("custom_views", "an old orphan row elsewhere is not an unbootable condition")
async def t_migration_fk_check_is_scoped(ctx: TestContext) -> None:
    """The verify used an unscoped `PRAGMA foreign_key_check`.

    A three-month-old production agent had 8 dangling rows - `task_runs` whose
    `scheduled_tasks` parent was deleted, `event_deliveries` for a removed
    event - none of them in a `ui_*` table. The whole-database check failed the
    migration, `connect()` raised, and the process died inside `_serve` before
    the HTTP server came up: the agent could not boot at all.
    """
    from src.custom_views.migration import (
        CustomViewMigrationError,
        REQUIRED_TABLES,
        ensure_custom_views_storage,
    )

    with tempfile.TemporaryDirectory(prefix="oa-ui-fk-") as raw:
        root = Path(raw)
        db = await _db(root)
        try:
            conn = await db._ensure_connected()
            # A dangling child in a table this migration does not own. Written
            # with the enforcement pragma off, exactly as the rows that
            # accumulated in production did.
            await conn.execute("PRAGMA foreign_keys=OFF")
            await conn.execute(
                "INSERT INTO task_runs (id, task_id, status, started_at) "
                "VALUES ('run-orphan', 'task-that-was-deleted', 'done', 0)"
            )
            await conn.commit()
            violations = await (
                await conn.execute("PRAGMA foreign_key_check")
            ).fetchall()
            assert violations, "fixture must actually violate a foreign key"
            assert all(
                str(row[0]) not in REQUIRED_TABLES for row in violations
            ), violations

            # Re-verifying the already-complete migration must still pass.
            assert not await ensure_custom_views_storage(conn, app_version="test")

            # A violation INSIDE the migration's own tables is still fatal.
            await conn.execute(
                "INSERT INTO ui_view_revisions "
                "(view_id, tenant_id, revision, bundle_path, bundle_sha256, "
                " bundle_size_bytes, created_by_principal_id, created_at_ms) "
                "VALUES ('view-that-does-not-exist', 'tenant-a', 1, 'p', ?, 0, 'user:alice', 0)",
                ("0" * 64,)
            )
            await conn.commit()
            try:
                await ensure_custom_views_storage(conn, app_version="test")
            except CustomViewMigrationError as exc:
                assert "ui_view_revisions" in str(exc), exc
            else:
                raise AssertionError("an orphan ui_* row must still fail verification")
        finally:
            await db.close()


@test("custom_views", "message-parts verification is scoped to its own table too")
async def t_message_parts_fk_check_is_scoped(ctx: TestContext) -> None:
    """The same unscoped check, in the migration that runs right after.

    Fixing only `custom-views-v1` moved the production crash one migration
    down: the Lyra agent then died on `message-parts-v1 foreign key
    verification failed`, from the identical whole-database PRAGMA.
    """
    from src.memory.message_parts_migration import (
        MessagePartsMigrationError,
        REQUIRED_TABLE,
        ensure_message_parts_storage,
    )

    with tempfile.TemporaryDirectory(prefix="oa-parts-fk-") as raw:
        root = Path(raw)
        db = await _db(root)
        try:
            conn = await db._ensure_connected()
            await conn.execute("PRAGMA foreign_keys=OFF")
            await conn.execute(
                "INSERT INTO task_runs (id, task_id, status, started_at) "
                "VALUES ('run-orphan-2', 'task-that-was-deleted', 'done', 0)"
            )
            await conn.commit()
            violations = await (
                await conn.execute("PRAGMA foreign_key_check")
            ).fetchall()
            assert violations and all(
                str(row[0]) != REQUIRED_TABLE for row in violations
            ), violations

            assert not await ensure_message_parts_storage(conn, app_version="test")
        finally:
            await db.close()
