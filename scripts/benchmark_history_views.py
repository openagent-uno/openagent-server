#!/usr/bin/env python3
"""Local performance smoke for normalized history, search and Custom Views.

The benchmark creates a disposable canonical database; it never opens the
operator's agent.  It is intentionally deterministic and reports JSON so beta
release checks can compare runs over time.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import tempfile
import time
from typing import Awaitable, Callable, TypeVar


T = TypeVar("T")


def _access(tenant: str):
    from src.memory.operational.access import AccessContext

    return AccessContext(
        tenant_id=tenant,
        principal_id="user:benchmark",
        principal_type="user",
        handle="benchmark",
        device_id="benchmark-device",
        principal_ids=frozenset({"user:benchmark", "device:benchmark-device"}),
        grant_identities=frozenset({("user", "benchmark")}),
    )


async def _measure(fn: Callable[[], Awaitable[T]], repeats: int = 5) -> tuple[float, T]:
    samples: list[float] = []
    result: T | None = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = await fn()
        samples.append((time.perf_counter() - started) * 1000)
    assert result is not None
    return statistics.median(samples), result


async def run(session_count: int, view_count: int) -> dict:
    from src.custom_views.repository import CustomViewRepository
    from src.memory.db import MemoryDB
    from src.memory.message_parts import canonical_parts_for_messages_on_connection
    from src.memory.operational.access import resource_is_visible
    from src.memory.operational.search import sync_operational_search
    from src.memory.operational.service import OperationalSearchService

    with tempfile.TemporaryDirectory(prefix="oa-history-benchmark-") as raw:
        db = MemoryDB(str(Path(raw) / "agent.db"))
        await db.connect()
        try:
            conn = await db._ensure_connected()
            tenant = "tenant-benchmark"
            access = _access(tenant)
            base = 1_900_000_000_000
            sessions = []
            activities = []
            outbox = []
            for index in range(session_count):
                session_id = f"bench-session-{index:08d}"
                title = (
                    f"benchmarkneedle history {index}"
                    if index == session_count - 1
                    else f"Conversation {index}"
                )
                stamp = base + index
                sessions.append(
                    (
                        session_id,
                        tenant,
                        "user:benchmark",
                        "benchmark",
                        title,
                        stamp,
                    )
                )
                activities.append(
                    (
                        f"activity-{index:08d}",
                        session_id,
                        tenant,
                        title,
                        stamp,
                    )
                )
                outbox.append((tenant, session_id, stamp))
            await conn.executemany(
                "INSERT INTO sessions_v2 "
                "(id,tenant_id,owner_principal_id,owner_handle_snapshot,visibility,acl_version,"
                "title,session_type,kind,origin,status,completeness,source_version,metadata_json,"
                "created_at_ms,updated_at_ms,last_activity_at_ms) "
                "VALUES (?,?,?,?, 'private',1,?,'agent','chat','benchmark','active','complete',1,'{}',?,?,?)",
                [row + (row[-1], row[-1]) for row in sessions],
            )
            await conn.executemany(
                "INSERT INTO activity_items "
                "(activity_id,kind,resource_type,resource_id,parent_type,parent_id,session_id,"
                "tenant_id,owner_principal_id,visibility,acl_version,status,title,origin,"
                "occurred_at_ms,updated_at_ms,source_version,created_revision,updated_revision,"
                "deleted_revision,deleted_at_ms) "
                "VALUES (?,'chat','session',?,NULL,NULL,?,?,'user:benchmark','private',1,NULL,?,"
                "'benchmark',?,?,1,1,1,NULL,NULL)",
                [
                    (activity_id, session_id, session_id, tenant_id, title, stamp, stamp)
                    for activity_id, session_id, tenant_id, title, stamp in activities
                ],
            )
            await conn.executemany(
                "INSERT INTO search_outbox "
                "(tenant_id,source_kind,source_id,operation,source_version,acl_version,committed_at_ms) "
                "VALUES (?,'session',?,'upsert',1,1,?)",
                outbox,
            )

            now = base + session_count + 1
            for index in range(view_count):
                view_id = f"bench-view-{index:08d}"
                title = (
                    f"benchmarkneedle view {index}"
                    if index == view_count - 1
                    else f"Dashboard {index}"
                )
                await conn.execute(
                    "INSERT INTO ui_views "
                    "(id,tenant_id,owner_principal_id,owner_handle_snapshot,visibility,acl_version,"
                    "surface,session_id,title,description,icon,status,schema_version,latest_revision,"
                    "search_text,sidebar_order,sidebar_group,last_viewed_at_ms,frozen,frozen_at_ms,"
                    "expires_at_ms,created_at_ms,updated_at_ms,deleted_at_ms) "
                    "VALUES (?,?, 'user:benchmark','benchmark','private',1,'sidebar',NULL,?,?,'chart',"
                    "'active',1,1,?,?,NULL,?,0,NULL,NULL,?,?,NULL)",
                    (
                        view_id,
                        tenant,
                        title,
                        "Static dashboard benchmark content",
                        f"{title} static dashboard benchmark content",
                        index,
                        now + index,
                        now + index,
                        now + index,
                    ),
                )

            # One dense conversation window for normalized-part hydration.
            dense_session = sessions[-1][0]
            messages = []
            parts = []
            for index in range(200):
                message_id = f"bench-message-{index:04d}"
                stamp = now + view_count + index
                messages.append(
                    (
                        message_id,
                        tenant,
                        dense_session,
                        session_count + index,
                        index,
                        "assistant" if index % 2 else "user",
                        "agent" if index % 2 else "user",
                        "agent:openagent" if index % 2 else "user:benchmark",
                        f"normalized content {index}",
                        stamp,
                    )
                )
                parts.append(
                    (
                        f"mpart-bench-{index:04d}",
                        tenant,
                        dense_session,
                        message_id,
                        0,
                        f"normalized content {index}",
                        stamp,
                    )
                )
            await conn.executemany(
                "INSERT INTO session_messages "
                "(id,tenant_id,session_id,run_id,sequence,ordinal,role,status,author_kind,"
                "author_principal_id,text,visibility,source_version,completeness,raw_envelope_json,"
                "raw_envelope_schema,legacy_inferred,created_at_ms,updated_at_ms,completed_at_ms) "
                "VALUES (?,?,?,NULL,?,?,?,'complete',?,?,?,'user_visible',1,'complete','{}',1,0,?,?,?)",
                [row[:5] + row[5:9] + (row[9], row[9], row[9]) for row in messages],
            )
            await conn.executemany(
                "INSERT INTO session_message_parts "
                "(id,tenant_id,session_id,message_id,ordinal,kind,text_content,artifact_link_id,"
                "ui_view_id,ui_revision,created_at_ms) "
                "VALUES (?,?,?,?,?,'text',?,NULL,NULL,NULL,?)",
                parts,
            )
            await conn.commit()

            # Cold index construction is reported separately; interaction
            # timings below are warm-state client-visible paths.
            cold_started = time.perf_counter()
            while True:
                status = await sync_operational_search(db, limit=10_000)
                if status.ready:
                    break
            cold_index_ms = (time.perf_counter() - cold_started) * 1000

            async def history_page():
                rows = await (
                    await conn.execute(
                        "SELECT * FROM activity_items WHERE tenant_id=? AND deleted_at_ms IS NULL "
                        "ORDER BY occurred_at_ms DESC, kind, activity_id LIMIT 100",
                        (tenant,),
                    )
                ).fetchall()
                visible = []
                for row in rows:
                    if await resource_is_visible(conn, row, access):
                        visible.append(row)
                return visible

            async def sidebar_page():
                return await CustomViewRepository(db).list(
                    access,
                    surface="sidebar",
                    limit=50,
                    offset=0,
                )

            async def global_search():
                return await OperationalSearchService(db).search(
                    access=access,
                    query="benchmarkneedle",
                    scopes=("chats", "views"),
                    limit=10,
                )

            window_ids = [row[0] for row in messages[-100:]]

            async def hydrate_parts():
                return await canonical_parts_for_messages_on_connection(
                    conn,
                    window_ids,
                    access=access,
                )

            history_ms, history = await _measure(history_page)
            sidebar_ms, sidebar = await _measure(sidebar_page)
            search_ms, search = await _measure(global_search)
            hydrate_ms, hydrated = await _measure(hydrate_parts)
            metrics = {
                "seed": {
                    "sessions": session_count,
                    "views": view_count,
                    "hydrated_messages": len(window_ids),
                },
                "cold_index_ms": round(cold_index_ms, 2),
                "median_ms": {
                    "history_page_100": round(history_ms, 2),
                    "sidebar_views_50": round(sidebar_ms, 2),
                    "global_search": round(search_ms, 2),
                    "hydrate_parts_100": round(hydrate_ms, 2),
                },
                "results": {
                    "history": len(history),
                    "views": len(sidebar[0]),
                    "search_hits": len(search.get("hits") or []),
                    "hydrated": len(hydrated),
                },
            }
            limits = {
                "history_page_100": 250.0,
                "sidebar_views_50": 250.0,
                "global_search": 500.0,
                "hydrate_parts_100": 250.0,
            }
            failures = {
                key: {"actual_ms": metrics["median_ms"][key], "limit_ms": limit}
                for key, limit in limits.items()
                if metrics["median_ms"][key] > limit
            }
            metrics["thresholds_ms"] = limits
            metrics["ok"] = not failures and metrics["results"]["search_hits"] >= 2
            metrics["failures"] = failures
            return metrics
        finally:
            await db.close()


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=10_000)
    parser.add_argument("--views", type=int, default=500)
    args = parser.parse_args()
    if not 1 <= args.sessions <= 1_000_000:
        parser.error("--sessions must be between 1 and 1000000")
    if not 1 <= args.views <= 100_000:
        parser.error("--views must be between 1 and 100000")
    report = await run(args.sessions, args.views)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
