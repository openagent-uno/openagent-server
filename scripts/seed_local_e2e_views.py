#!/usr/bin/env python3
"""Seed deterministic Custom Views into a marked disposable local-E2E agent."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import tempfile
import time


def _inside_temp(path: Path) -> bool:
    root = Path(tempfile.gettempdir()).resolve()
    try:
        return path.resolve(strict=True).relative_to(root) != Path(".")
    except (OSError, ValueError):
        return False


def _access(tenant: str, handle: str):
    from src.memory.operational.access import AccessContext

    principal = f"user:{handle}"
    device = f"local-e2e-{handle}"
    return AccessContext(
        tenant_id=tenant,
        principal_id=principal,
        principal_type="user",
        handle=handle,
        device_id=device,
        principal_ids=frozenset({principal, f"user:{device}", f"device:{device}"}),
        grant_identities=frozenset({
            ("user", handle),
            ("user", principal),
            ("device", device),
            ("device", f"device:{device}"),
            ("installation", tenant),
        }),
    )


def _states() -> dict:
    return {
        "loading": {"type": "loading-state", "props": {"text": "Loading live data…"}},
        "empty": {"type": "empty-state", "props": {"text": "No rows in this period"}},
        "stale": {"type": "stale-state", "props": {"text": "Updates are paused"}},
        "error": {"type": "error-state", "props": {"text": "Data source unavailable"}},
    }


def _dashboard_spec() -> dict:
    return {
        "schemaVersion": 1,
        "root": {
            "type": "stack",
            "id": "dashboard-root",
            "props": {"gap": 14},
            "children": [
                {"type": "heading", "props": {"level": 2, "text": "Local agent health"}},
                {
                    "type": "grid",
                    "props": {"columns": 3, "minColumnWidth": 150, "gap": 10},
                    "children": [
                        {"type": "metric", "props": {"label": "CPU", "value": "{{data.metrics.cpu}}", "unit": "%", "trend": "+2.1%"}},
                        {"type": "metric", "props": {"label": "Memory", "value": "{{data.metrics.memory}}", "unit": "%", "detail": "12.8 / 24 GB"}},
                        {"type": "metric", "props": {"label": "Active jobs", "value": "{{data.metrics.jobs}}", "detail": "2 scheduled"}},
                    ],
                },
                {
                    "type": "tabs",
                    "children": [
                        {
                            "type": "sub-view",
                            "id": "trend-tab",
                            "props": {"label": "Trend"},
                            "children": [
                                {"type": "line-chart", "props": {"data": "{{data.metrics.points}}", "xKey": "time", "yKey": "value", "legend": True}},
                            ],
                        },
                        {
                            "type": "sub-view",
                            "id": "services-tab",
                            "props": {"label": "Services"},
                            "children": [
                                {"type": "table", "props": {"rows": "{{data.metrics.services}}", "columns": ["name", "status", "latency"], "pageSize": 6}},
                            ],
                        },
                        {
                            "type": "sub-view",
                            "id": "controls-tab",
                            "props": {"label": "Controls"},
                            "children": [
                                {"type": "toggle", "id": "alerts", "props": {"label": "Local alerts", "description": "Pure client-side state", "checked": True}},
                                {"type": "segmented", "id": "range", "props": {"label": "Range", "options": [{"label": "1h", "value": "1h"}, {"label": "24h", "value": "24h"}, {"label": "7d", "value": "7d"}], "selected": "24h"}},
                                {"type": "button", "props": {"text": "Inject realtime sample", "action": "inject-sample"}},
                                {"type": "status", "props": {"label": "{{data.action.message}}", "status": "success"}},
                            ],
                        },
                    ],
                },
            ],
        },
        "states": _states(),
    }


async def seed(agent_dir: Path, handle: str) -> dict:
    from src.custom_views.repository import CustomViewRepository
    from src.memory.db import MemoryDB
    from src.memory.message_parts import persist_parts_for_latest_message

    marker = agent_dir / "openagent.yaml"
    if not _inside_temp(agent_dir) or agent_dir.is_symlink() or not marker.is_file():
        raise RuntimeError("local E2E seeding only accepts a real agent directory under the OS temp root")
    if "local_e2e_fixture: true" not in marker.read_text(encoding="utf-8"):
        raise RuntimeError("agent directory is not marked as a local E2E fixture")

    db = MemoryDB(str(agent_dir / "openagent.db"))
    await db.connect()
    try:
        conn = await db._ensure_connected()
        network = await (await conn.execute(
            "SELECT network_id FROM network WHERE singleton=1"
        )).fetchone()
        tenant = str(network[0] if network is not None else "").strip()
        if not tenant:
            raise RuntimeError("fixture has no initialized network id")
        access = _access(tenant, handle.strip().lower())
        repo = CustomViewRepository(db)
        now = int(time.time() * 1000)

        session_id = "local-e2e-custom-view-chat"
        run_id = "local-e2e-custom-view-run"
        message_id = "local-e2e-custom-view-message"
        tool_invocation_id = "local-e2e-search-tool-invocation"
        await conn.execute(
            "INSERT OR IGNORE INTO sessions_v2 "
            "(id,tenant_id,owner_principal_id,owner_handle_snapshot,visibility,acl_version,"
            "title,session_type,kind,origin,status,completeness,source_version,metadata_json,"
            "created_at_ms,updated_at_ms,last_activity_at_ms) "
            "VALUES (?,?,?,?, 'private',1,?,'agent','chat','local_e2e','active','complete',1,'{}',?,?,?)",
            (session_id, tenant, access.principal_id, access.handle, "Custom View inline E2E", now, now, now),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO session_runs "
            "(id,tenant_id,session_id,ordinal,runner_kind,status,status_raw,metadata_json,"
            "source_version,completeness,raw_envelope_json,raw_envelope_schema,created_at_ms,finished_at_ms) "
            "VALUES (?,?,?,0,'agent','success','success','{}',1,'complete','{}',1,?,?)",
            (run_id, tenant, session_id, now, now),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO session_messages "
            "(id,tenant_id,session_id,run_id,sequence,ordinal,role,status,author_kind,"
            "author_principal_id,author_handle_snapshot,text,content_json,visibility,source_version,"
            "completeness,raw_envelope_json,raw_envelope_schema,legacy_inferred,created_at_ms,"
            "updated_at_ms,completed_at_ms) "
            "VALUES (?,?,?,?,0,0,'assistant','complete','agent','agent:openagent','openagent',?,?,'user_visible',1,'complete','{}',1,0,?,?,?)",
            (
                message_id,
                tenant,
                session_id,
                run_id,
                "The agent generated an inline dashboard. It remains useful as plain text in older clients.",
                json.dumps("The agent generated an inline dashboard. It remains useful as plain text in older clients."),
                now,
                now,
                now,
            ),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO tool_invocations "
            "(id,tenant_id,owner_principal_id,visibility,acl_version,root_kind,root_id,"
            "session_id,session_run_id,ordinal,tool_call_id,tool_server,tool_name,status,status_raw,"
            "args_json,result_json,result_text,sensitivity,result_complete,source_version,"
            "completeness,raw_envelope_json,raw_envelope_schema,legacy_inferred,created_at_ms,finished_at_ms) "
            "VALUES (?,?,?,'private',1,'session',?,?,?,?,'local-e2e-tool-call','local_e2e',"
            "'local_e2e_search_tool','success','success',?,?,?,'normal',1,1,'complete','{}',1,0,?,?)",
            (
                tool_invocation_id,
                tenant,
                access.principal_id,
                session_id,
                session_id,
                run_id,
                0,
                json.dumps({"query": "deterministic local E2E tool search"}),
                json.dumps({"status": "indexed"}),
                "Deterministic local E2E tool invocation indexed successfully.",
                now,
                now,
            ),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO activity_items "
            "(activity_id,kind,resource_type,resource_id,parent_type,parent_id,session_id,tenant_id,"
            "owner_principal_id,visibility,acl_version,status,title,origin,occurred_at_ms,updated_at_ms,"
            "source_version,created_revision,updated_revision,deleted_revision,deleted_at_ms) "
            "VALUES (?,'chat','session',?,NULL,NULL,?,?,?,'private',1,NULL,?,'local_e2e',?,?,1,1,1,NULL,NULL)",
            (f"activity:{session_id}", session_id, session_id, tenant, access.principal_id, "Custom View inline E2E", now, now),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO search_outbox "
            "(tenant_id,source_kind,source_id,operation,source_version,acl_version,committed_at_ms) "
            "VALUES (?,'session',?,'upsert',1,1,?)",
            (tenant, session_id, now),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO search_outbox "
            "(tenant_id,source_kind,source_id,operation,source_version,acl_version,committed_at_ms) "
            "VALUES (?,'message',?,'upsert',1,1,?)",
            (tenant, message_id, now),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO search_outbox "
            "(tenant_id,source_kind,source_id,operation,source_version,acl_version,committed_at_ms) "
            "VALUES (?,'tool_invocation',?,'upsert',1,1,?)",
            (tenant, tool_invocation_id, now),
        )
        await conn.commit()

        metrics = {
            "cpu": 42,
            "memory": 53,
            "jobs": 7,
            "points": [
                {"time": "09:00", "value": 29},
                {"time": "10:00", "value": 37},
                {"time": "11:00", "value": 32},
                {"time": "12:00", "value": 48},
                {"time": "13:00", "value": 42},
            ],
            "services": [
                {"name": "gateway", "status": "healthy", "latency": "18 ms"},
                {"name": "scheduler", "status": "parked (E2E)", "latency": "—"},
                {"name": "search", "status": "healthy", "latency": "31 ms"},
                {"name": "memory vault", "status": "healthy", "latency": "12 ms"},
            ],
        }
        common_sources = {
            "metrics": {
                "driver": "static",
                "activation": "while_visible",
                "config": {"value": metrics},
            },
            "action": {
                "driver": "push",
                "activation": "manual",
                "config": {"mode": "replace"},
            },
        }
        common_actions = {
            "inject-sample": {
                "kind": "set_data",
                "label": "Inject realtime sample",
                "config": {
                    "key": "action",
                    "value": {"message": "Sample injected through an authenticated action"},
                },
            },
        }
        initial = {"metrics": metrics, "action": {"message": "Ready for an action"}}

        dashboard = await repo.create(
            access,
            surface="sidebar",
            title="E2E Agent health",
            description="Charts, tabs, controls and live action data",
            icon="activity",
            spec=_dashboard_spec(),
            sources=common_sources,
            actions=common_actions,
            initial_data=initial,
            sidebar_order=-100,
            sidebar_group="Local E2E",
        )
        inline = await repo.create(
            access,
            surface="inline",
            session_id=session_id,
            title="Inline agent health",
            description="Revision-pinned inside a chat message",
            icon="bar-chart-2",
            spec=_dashboard_spec(),
            sources=common_sources,
            actions=common_actions,
            initial_data=initial,
        )
        empty = await repo.create(
            access,
            surface="sidebar",
            title="E2E Empty state",
            icon="inbox",
            spec={
                "schemaVersion": 1,
                "root": {"type": "list", "props": {"items": "{{data.rows}}"}},
                "states": _states(),
            },
            sources={"rows": {"driver": "static", "config": {"value": []}}},
            sidebar_order=-90,
            sidebar_group="Local E2E",
        )
        stale = await repo.create(
            access,
            surface="sidebar",
            title="E2E Stale state",
            icon="pause-circle",
            spec={
                "schemaVersion": 1,
                "root": {"type": "text", "props": {"text": "Last known snapshot"}},
                "states": _states(),
            },
            initial_data={"snapshot": {"ok": True}},
            frozen=True,
            sidebar_order=-80,
            sidebar_group="Local E2E",
        )
        error = await repo.create(
            access,
            surface="sidebar",
            title="E2E Error state",
            icon="alert-triangle",
            spec={
                "schemaVersion": 1,
                "root": {"type": "text", "props": {"text": "This is replaced by the error state"}},
                "states": _states(),
            },
            sources={
                "broken": {
                    "driver": "command_poll",
                    "activation": "while_visible",
                    "config": {"argv": ["/usr/bin/false"], "intervalMs": 1000, "timeoutMs": 1000},
                },
            },
            sidebar_order=-70,
            sidebar_group="Local E2E",
        )
        await persist_parts_for_latest_message(
            db,
            session_id,
            role="assistant",
            parts=(
                {"kind": "text", "text": "The agent generated an inline dashboard. "},
                {"kind": "ui_view", "view_id": inline["id"], "revision": inline["revision"]},
                {"kind": "text", "text": "It remains useful as plain text in older clients."},
            ),
            principal=access,
            after_sequence=-1,
        )
        # ``--local-e2e`` deliberately starts no background index writer.
        # Build the derived search sidecar explicitly so the packaged-client
        # scenario can prove that message text and static View text are
        # searchable without weakening that hermetic server contract.
        from src.memory.operational.search import warm_operational_search

        await warm_operational_search(db)
        return {
            "tenant": tenant,
            "handle": access.handle,
            "session_id": session_id,
            "tool_invocation": tool_invocation_id,
            "dashboard": dashboard["id"],
            "inline": inline["id"],
            "empty": empty["id"],
            "stale": stale["id"],
            "error": error["id"],
        }
    finally:
        await db.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", type=Path, required=True)
    parser.add_argument("--handle", required=True)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(seed(args.agent_dir, args.handle)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
