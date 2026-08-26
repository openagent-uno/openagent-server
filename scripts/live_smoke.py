#!/usr/bin/env python3
"""Live E2E smoke test against a deployed OpenAgent coordinator.

Two modes:

    # 1. Register a fresh smoke user (consumes a USER invite)
    python scripts/live_smoke.py register \\
        --ticket <oa1...> \\
        --password <pw>

       Prints the handle on success. Use the handle + password for
       the pair step below.

    # 2. Pair a NEW device for an existing user (consumes a DEVICE invite)
    python scripts/live_smoke.py pair \\
        --ticket <oa1...> \\
        --handle <handle> \\
        --password <pw>

Exit code: 0 = success, 1 = failure.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

# Repo root → import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


async def _gateway_ws_handshake(
    node, dialer, target_node_id: str, timeout: float = 15.0,
) -> str:
    """Open a cert-prefixed iroh gateway stream + WS upgrade + read AUTH_OK.

    Returns the JSON-decoded AUTH_OK frame on success. Raises on any
    failure, with a message indicating which step broke. This is the
    EXACT path the Electron app takes after a successful SRP login.
    """
    import aiohttp

    proxy = __import__("src.network.client.session", fromlist=["LoopbackProxy"]).LoopbackProxy(
        dialer=dialer, target_node_id=target_node_id,
    )
    host, port = await proxy.start()
    try:
        ws_url = f"ws://{host}:{port}/ws"
        async with aiohttp.ClientSession() as session:
            ws = await asyncio.wait_for(
                session.ws_connect(ws_url, heartbeat=None),
                timeout=timeout,
            )
            try:
                first = await asyncio.wait_for(ws.receive(), timeout=timeout)
                if first.type != aiohttp.WSMsgType.TEXT:
                    raise RuntimeError(
                        f"expected TEXT frame for AUTH_OK, got {first.type.name} "
                        f"(data={first.data!r})",
                    )
                return first.data
            finally:
                await ws.close()
    finally:
        await proxy.stop()


async def _do_register(
    ticket_str: str,
    password: str,
    *,
    operational_db: str | None = None,
    agent_handle: str | None = None,
) -> int:
    from src.network.auth.device_cert import verify_cert
    from src.network.client.login import LoginError, login, register
    from src.network.client.session import NetworkBinding, SessionDialer
    from src.network.identity import Identity
    from src.network.iroh_node import IrohNode
    from src.network.client.login import list_agents as coord_list_agents
    from src.network.peers import coordinator_node_id_to_pubkey_bytes
    from src.network.ticket import InviteTicket

    ut = InviteTicket.decode(ticket_str)
    if ut.role != "user":
        print(f"  ✗ ticket role is {ut.role!r}, expected 'user'", flush=True)
        return 1

    coord = ut.coordinator_node_id
    coord_pub = coordinator_node_id_to_pubkey_bytes(coord)
    handle = f"e2esmoke-{uuid.uuid4().hex[:6]}"

    print(f"  coordinator : {coord}", flush=True)
    print(f"  network     : {ut.network_name} / {ut.network_id}", flush=True)
    print(f"  invite code : {ut.code}", flush=True)
    print(f"  handle      : {handle}", flush=True)
    print(f"  password    : {password}", flush=True)

    dev = Identity.generate()
    node = IrohNode(dev)
    await node.start()
    try:
        print("\n[1/2] register + first login", flush=True)
        try:
            cert_wire = await register(
                node=node,
                coordinator_node_id=coord,
                coordinator_pubkey_bytes=coord_pub,
                handle=handle, password=password,
                invite_code=ut.code,
                device_identity=dev,
                network_id=ut.network_id,
                label=f"smoke-register",
            )
        except LoginError as e:
            print(f"  ✗ register failed: {e}", flush=True)
            return 1
        cert = verify_cert(
            cert_wire,
            coordinator_pubkey=Ed25519PublicKey.from_public_bytes(coord_pub),
        )
        print(f"  ✓ registered + cert minted ({cert.handle}@{cert.network_id[:8]}…)",
              flush=True)

        print("\n[2/3] returning-device login (touch_device path)", flush=True)
        try:
            cert_wire2 = await login(
                node=node,
                coordinator_node_id=coord,
                coordinator_pubkey_bytes=coord_pub,
                handle=handle, password=password,
                device_identity=dev,
                network_id=ut.network_id,
            )
        except LoginError as e:
            print(f"  ✗ returning-device login failed: {e}", flush=True)
            return 1
        verify_cert(
            cert_wire2,
            coordinator_pubkey=Ed25519PublicKey.from_public_bytes(coord_pub),
        )
        print("  ✓ cert renewed", flush=True)

        # ── 3. Gateway WS handshake (exact path the Electron app takes) ──
        print("\n[3/4] gateway WS auth (the Electron-app failure path)",
              flush=True)
        try:
            agents = await coord_list_agents(
                node=node, coordinator_node_id=coord,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ list_agents failed: {e}", flush=True)
            return 1
        if not agents:
            print("  ✗ no agents registered in network", flush=True)
            return 1
        selected = next(
            (
                agent for agent in agents
                if agent_handle and agent.get("handle") == agent_handle
            ),
            None,
        ) if agent_handle else agents[0]
        if selected is None:
            print(f"  ✗ agent {agent_handle!r} is not registered in this network", flush=True)
            return 1
        agent_node_id = selected["node_id"]
        print(f"  agent: {selected.get('handle','?')} @ {agent_node_id[:24]}…",
              flush=True)

        binding = NetworkBinding(
            network_id=ut.network_id,
            network_name=ut.network_name,
            coordinator_node_id=coord,
            coordinator_pubkey_bytes=coord_pub,
            our_handle=handle,
        )
        dialer = SessionDialer(node=node, binding=binding, cert_wire=cert_wire2)
        try:
            auth_ok_payload = await _gateway_ws_handshake(node, dialer, agent_node_id)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ gateway WS handshake failed: {type(e).__name__}: {e}",
                  flush=True)
            await dialer.close()
            return 1
        print(f"  ✓ AUTH_OK received: {auth_ok_payload[:120]}…", flush=True)

        # ── 4. /api/network/* — members & invitations through the gateway ──
        print("\n[4/4] /api/network/* (members + mint via gateway HTTP)",
              flush=True)
        try:
            await _exercise_network_api(dialer, agent_node_id)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ network api failed: {type(e).__name__}: {e}",
                  flush=True)
            await dialer.close()
            return 1

        if operational_db:
            print("\n[5/5] operational history/search canary", flush=True)
            try:
                await _exercise_operational_api(
                    dialer,
                    agent_node_id,
                    db_path=Path(operational_db).expanduser().resolve(),
                    network_id=ut.network_id,
                    handle=handle,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"  ✗ operational api failed: {type(e).__name__}: {e}",
                    flush=True,
                )
                await dialer.close()
                return 1
        await dialer.close()

    finally:
        await node.stop()

    print(f"\nREGISTER + WS + NETWORK API PASSED. Handle: {handle}", flush=True)
    return 0


async def _exercise_network_api(dialer, target_node_id: str) -> None:
    """Drive list/mint/revoke against the live gateway's /api/network/*.

    Mirrors what the desktop Members tab + the openagent-cli `users`/
    `invite` commands do, end-to-end over Iroh+HTTP."""
    import aiohttp

    from src.network.client.session import LoopbackProxy

    proxy = LoopbackProxy(dialer=dialer, target_node_id=target_node_id)
    host, port = await proxy.start()
    try:
        async with aiohttp.ClientSession() as s:
            base = f"http://{host}:{port}"
            async with s.get(f"{base}/api/network/users") as r:
                assert r.status == 200, f"GET /users → {r.status}"
                users = (await r.json()).get("users", [])
                print(f"  ✓ GET /users → {len(users)} user(s)",
                      flush=True)

            async with s.get(f"{base}/api/network/agents") as r:
                assert r.status == 200, f"GET /agents → {r.status}"
                agents = (await r.json()).get("agents", [])
                print(f"  ✓ GET /agents → {len(agents)} agent(s) (first: "
                      f"{agents[0]['handle'] if agents else '-'})",
                      flush=True)

            async with s.get(f"{base}/api/network/invitations") as r:
                assert r.status == 200, f"GET /invitations → {r.status}"
                pre = (await r.json()).get("invitations", [])
                print(f"  ✓ GET /invitations → {len(pre)} active",
                      flush=True)

            # Smart mint — handle absent → user role, open invite.
            async with s.post(
                f"{base}/api/network/invitations",
                json={"handle": "smoke-friend"},
            ) as r:
                assert r.status == 201, \
                    f"POST /invitations → {r.status}: {await r.text()}"
                minted = await r.json()
                assert minted["role"] == "user", minted
                assert minted["ticket"].startswith("oa1"), minted
                print(f"  ✓ POST /invitations → role={minted['role']}, "
                      f"code={minted['code']}, intent={minted['intent']!r}",
                      flush=True)

            # Idempotent revoke.
            async with s.delete(
                f"{base}/api/network/invitations/{minted['code']}",
            ) as r:
                assert r.status == 200, f"DELETE → {r.status}"
                payload = await r.json()
                assert payload.get("revoked") is True, payload
                print(f"  ✓ DELETE /invitations/{minted['code'][:10]}… "
                      f"→ revoked=True", flush=True)

            # PATCH /agents — relabel the coordinator's own agent and
            # immediately revert. Cosmetic-only so this is safe to
            # exercise against a live deployment.
            agents_first = agents[0]
            original_label = agents_first.get("label") or ""
            async with s.patch(
                f"{base}/api/network/agents/{agents_first['handle']}",
                json={"label": f"{original_label} (smoked)"},
            ) as r:
                assert r.status == 200, f"PATCH → {r.status}: {await r.text()}"
                print(f"  ✓ PATCH /agents/{agents_first['handle']} → label updated",
                      flush=True)
            async with s.patch(
                f"{base}/api/network/agents/{agents_first['handle']}",
                json={"label": original_label},
            ) as r:
                assert r.status == 200

            # DELETE /agents on the coord's own row must REFUSE.
            async with s.delete(
                f"{base}/api/network/agents/{agents_first['handle']}",
            ) as r:
                assert r.status == 409, (
                    f"deleting coord's own agent should 409, got {r.status}"
                )
                print(f"  ✓ DELETE /agents/{agents_first['handle']} → 409 "
                      f"(coord agent protected)", flush=True)
    finally:
        await proxy.stop()


async def _seed_operational_canary(
    db_path: Path,
    *,
    network_id: str,
    handle: str,
) -> dict[str, object]:
    """Insert an isolated fixture through the live DB's existing journals.

    This mode is intentionally opt-in because the smoke client may connect to
    a remote agent.  The caller supplies the local DB path of the selected
    agent; the network id is checked before any write.  Every inserted key has
    a random prefix and cleanup only addresses those exact keys.
    """
    import aiosqlite

    prefix = f"e2eop-{uuid.uuid4().hex[:12]}"
    term = f"opsmoke{uuid.uuid4().hex[:12]}"
    forbidden_term = f"forbiddensmoke{uuid.uuid4().hex[:12]}"
    ids = {
        "session": f"{prefix}-chat",
        "forbidden_session": f"{prefix}-forbidden-chat",
        "run": f"{prefix}-chat-run",
        "user_message": f"{prefix}-user-message",
        "tool_message": f"{prefix}-tool-message",
        "assistant_message": f"{prefix}-assistant-message",
        "tool_call": f"{prefix}-tool-call",
        "workflow": f"{prefix}-workflow",
        "workflow_run": f"{prefix}-workflow-run",
        "scheduled": f"{prefix}-scheduled",
        "scheduled_run": f"{prefix}-scheduled-run",
        "event": f"{prefix}-event",
        "event_delivery": f"{prefix}-event-delivery",
    }
    required_tables = {
        "sessions", "workflow_tasks", "workflow_runs", "scheduled_tasks",
        "task_runs", "events", "event_deliveries", "sessions_v2",
        "legacy_session_changes", "operational_automation_changes",
    }
    conn = await aiosqlite.connect(db_path, timeout=10)
    try:
        await conn.execute("PRAGMA busy_timeout=10000")
        tables = {
            str(row[0])
            for row in await (
                await conn.execute("SELECT name FROM sqlite_schema WHERE type='table'")
            ).fetchall()
        }
        missing = sorted(required_tables - tables)
        if missing:
            raise RuntimeError(f"selected DB lacks operational beta tables: {missing}")
        network = await (
            await conn.execute("SELECT network_id FROM network LIMIT 1")
        ).fetchone()
        if network is None or str(network[0]) != network_id:
            raise RuntimeError("--operational-db does not belong to the ticket network")

        now = int(time.time())
        runs = [{
            "run_id": ids["run"],
            "status": "COMPLETED",
            "created_at": now,
            "messages": [
                {
                    "id": ids["user_message"],
                    "role": "user",
                    "content": f"{term} user-visible chat canary",
                },
                {
                    "id": ids["tool_message"],
                    "role": "tool",
                    "tool_call_id": ids["tool_call"],
                    "content": f"{term} user-visible tool canary",
                },
                {
                    "id": ids["assistant_message"],
                    "role": "assistant",
                    "content": f"{term} assistant canary",
                },
            ],
            "tools": [{
                "tool_call_id": ids["tool_call"],
                "tool_name": f"{term}_tool",
                "tool_args": {"canary": term},
                "result": {"canary": term},
                "status": "completed",
            }],
        }]
        await conn.execute(
            "INSERT INTO sessions "
            "(session_id,session_type,user_id,metadata,runs,created_at,updated_at) "
            "VALUES(?, 'agent', ?, ?, ?, ?, ?)",
            (
                ids["session"], handle,
                json.dumps({"client_id": handle, "title": f"{term} chat"}),
                json.dumps(runs), now, now,
            ),
        )
        await conn.execute(
            "INSERT INTO sessions "
            "(session_id,session_type,user_id,metadata,runs,created_at,updated_at) "
            "VALUES(?, 'agent', 'acl-other-user', ?, '[]', ?, ?)",
            (
                ids["forbidden_session"],
                json.dumps({
                    "client_id": "acl-other-user",
                    "title": f"{forbidden_term} private chat",
                }),
                now, now,
            ),
        )
        graph = {
            "version": 1,
            "nodes": [{
                "id": f"{prefix}-node",
                "type": "tool",
                "data": {"label": f"{term} workflow node"},
            }],
            "edges": [],
        }
        await conn.execute(
            "INSERT INTO workflow_tasks(id,name,description,graph_json,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?)",
            (
                ids["workflow"], f"{term} workflow", f"{term} definition",
                json.dumps(graph), now, now,
            ),
        )
        await conn.execute(
            "INSERT INTO workflow_runs"
            "(id,workflow_id,trigger,status,started_at,finished_at,trace_json,error) "
            "VALUES(?,?, 'manual', 'success', ?, ?, ?, NULL)",
            (
                ids["workflow_run"], ids["workflow"], now, now + 1,
                json.dumps([{
                    "node_id": f"{prefix}-node", "type": "tool",
                    "status": "success", "tool_name": f"{term}_workflow_tool",
                    "started_at": now, "finished_at": now + 1,
                }]),
            ),
        )
        await conn.execute(
            "INSERT INTO scheduled_tasks"
            "(id,name,cron_expression,prompt,enabled,created_at,updated_at) "
            "VALUES(?,?, '0 0 1 1 *', ?, 0, ?, ?)",
            (ids["scheduled"], f"{term} schedule", f"{term} scheduled prompt", now, now),
        )
        await conn.execute(
            "INSERT INTO task_runs"
            "(id,task_id,trigger,status,started_at,finished_at,output,error) "
            "VALUES(?,?, 'manual', 'success', ?, ?, ?, NULL)",
            (
                ids["scheduled_run"], ids["scheduled"], now, now + 1,
                f"{term} scheduled output",
            ),
        )
        await conn.execute(
            "INSERT INTO events"
            "(id,name,slug,type,secret_enc,input_schema_json,action_kind,"
            "description,prompt_template,enabled,created_at,updated_at) "
            "VALUES(?,?,?, 'generic', 'smoke-secret-not-indexed', '[]', 'prompt',"
            "?,?,0,?,?)",
            (
                ids["event"], f"{term} event", ids["event"],
                f"{term} event definition", f"{term} event prompt", now, now,
            ),
        )
        await conn.execute(
            "INSERT INTO event_deliveries"
            "(id,event_id,source,status,payload_json,started_at,finished_at,output,error) "
            "VALUES(?,?, 'manual', 'success', '{}', ?, ?, ?, NULL)",
            (
                ids["event_delivery"], ids["event"], now, now + 1,
                f"{term} event delivery output",
            ),
        )
        await conn.commit()
    except Exception:
        await conn.rollback()
        raise
    finally:
        await conn.close()
    return {
        "prefix": prefix,
        "term": term,
        "forbidden_term": forbidden_term,
        "ids": ids,
    }


async def _remove_operational_canary_sources(
    db_path: Path,
    fixture: dict[str, object],
) -> None:
    import aiosqlite

    ids = fixture["ids"]
    assert isinstance(ids, dict)
    conn = await aiosqlite.connect(db_path, timeout=10)
    try:
        await conn.execute("PRAGMA busy_timeout=10000")
        for table, key in (
            ("event_deliveries", "event_delivery"),
            ("events", "event"),
            ("task_runs", "scheduled_run"),
            ("scheduled_tasks", "scheduled"),
            ("workflow_runs", "workflow_run"),
            ("workflow_tasks", "workflow"),
            ("sessions", "forbidden_session"),
            ("sessions", "session"),
        ):
            await conn.execute(f"DELETE FROM {table} WHERE id=?" if table != "sessions" else f"DELETE FROM {table} WHERE session_id=?", (ids[key],))
        await conn.commit()
    finally:
        await conn.close()


async def _wait_operational_canary_cleanup(
    db_path: Path,
    fixture: dict[str, object],
    *,
    timeout_s: float = 60,
) -> int:
    """Wait until journals projected deletes and the derived FTS consumed them."""
    import aiosqlite
    import sqlite3

    from src.memory.operational.search import operational_search_path

    ids = fixture["ids"]
    assert isinstance(ids, dict)
    automation_ids = tuple(
        str(ids[key]) for key in (
            "workflow", "workflow_run", "scheduled", "scheduled_run",
            "event", "event_delivery",
        )
    )
    deadline = time.monotonic() + timeout_s
    conn = await aiosqlite.connect(db_path, timeout=10)
    try:
        await conn.execute("PRAGMA busy_timeout=10000")
        while time.monotonic() < deadline:
            session_pending = int((await (
                await conn.execute(
                    "SELECT COUNT(*) FROM legacy_session_changes "
                    "WHERE session_id IN (?,?) AND processed_at_ms IS NULL",
                    (ids["session"], ids["forbidden_session"]),
                )
            ).fetchone())[0])
            placeholders = ",".join("?" for _ in automation_ids)
            automation_pending = int((await (
                await conn.execute(
                    f"SELECT COUNT(*) FROM operational_automation_changes "
                    f"WHERE resource_id IN ({placeholders}) AND processed_at_ms IS NULL",
                    automation_ids,
                )
            ).fetchone())[0])
            if session_pending == 0 and automation_pending == 0:
                head = int((await (
                    await conn.execute("SELECT COALESCE(MAX(seq),0) FROM search_outbox")
                ).fetchone())[0])
                index_path = operational_search_path(db_path)
                if index_path.exists():
                    try:
                        index = sqlite3.connect(index_path, timeout=1)
                        try:
                            indexed = index.execute(
                                "SELECT last_indexed_seq FROM search_index_state WHERE singleton_id=1"
                            ).fetchone()
                        finally:
                            index.close()
                        if indexed is not None and int(indexed[0]) >= head:
                            return head
                    except sqlite3.Error:
                        pass
            await asyncio.sleep(0.25)
    finally:
        await conn.close()
    raise RuntimeError("canary delete journals/FTS did not drain within 60s")


async def _purge_operational_canary_metadata(
    db_path: Path,
    fixture: dict[str, object],
) -> None:
    """Remove canary-only normalized/audit rows after FTS consumed deletes."""
    import aiosqlite

    ids = fixture["ids"]
    assert isinstance(ids, dict)
    resource_ids = tuple(str(value) for value in ids.values())
    conn = await aiosqlite.connect(db_path, timeout=10)
    try:
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.execute("PRAGMA busy_timeout=10000")
        tool_rows = await (
            await conn.execute(
                "SELECT id FROM tool_invocations WHERE session_id IN (?,?)",
                (ids["session"], ids["forbidden_session"]),
            )
        ).fetchall()
        message_rows = await (
            await conn.execute(
                "SELECT id FROM session_messages WHERE session_id IN (?,?)",
                (ids["session"], ids["forbidden_session"]),
            )
        ).fetchall()
        indexed_ids = (*resource_ids, *(str(row[0]) for row in tool_rows), *(str(row[0]) for row in message_rows))
        placeholders = ",".join("?" for _ in indexed_ids)
        await conn.execute(f"DELETE FROM search_outbox WHERE source_id IN ({placeholders})", indexed_ids)
        await conn.execute(f"DELETE FROM domain_events WHERE resource_id IN ({placeholders}) OR session_id IN (?,?)", (*indexed_ids, ids["session"], ids["forbidden_session"]))
        await conn.execute(f"DELETE FROM activity_items WHERE resource_id IN ({placeholders})", indexed_ids)
        await conn.execute(f"DELETE FROM resource_acl WHERE resource_id IN ({placeholders})", indexed_ids)
        await conn.execute(f"DELETE FROM operational_resource_owners WHERE resource_id IN ({placeholders})", indexed_ids)
        await conn.execute(f"DELETE FROM operational_automation_projection WHERE resource_id IN ({placeholders})", indexed_ids)
        await conn.execute(f"DELETE FROM operational_automation_changes WHERE resource_id IN ({placeholders})", indexed_ids)
        await conn.execute("DELETE FROM tool_invocations WHERE session_id IN (?,?)", (ids["session"], ids["forbidden_session"]))
        await conn.execute("DELETE FROM session_messages WHERE session_id IN (?,?)", (ids["session"], ids["forbidden_session"]))
        await conn.execute("DELETE FROM session_runs WHERE session_id IN (?,?)", (ids["session"], ids["forbidden_session"]))
        await conn.execute("DELETE FROM sessions_v2 WHERE id IN (?,?)", (ids["session"], ids["forbidden_session"]))
        await conn.execute("DELETE FROM legacy_session_changes WHERE session_id IN (?,?)", (ids["session"], ids["forbidden_session"]))
        await conn.commit()
    except Exception:
        await conn.rollback()
        raise
    finally:
        await conn.close()


async def _exercise_operational_api(
    dialer,
    target_node_id: str,
    *,
    db_path: Path,
    network_id: str,
    handle: str,
) -> None:
    import aiohttp
    from src.network.client.session import LoopbackProxy

    fixture = await _seed_operational_canary(
        db_path, network_id=network_id, handle=handle,
    )
    ids = fixture["ids"]
    assert isinstance(ids, dict)
    proxy = LoopbackProxy(dialer=dialer, target_node_id=target_node_id)
    host, port = await proxy.start()
    cleanup_seq = 0
    sources_removed = False
    cleanup_consumed = False
    try:
        async with aiohttp.ClientSession() as session:
            base = f"http://{host}:{port}"
            deadline = time.monotonic() + 60
            found: dict | None = None
            while time.monotonic() < deadline:
                async with session.get(f"{base}/api/capabilities") as response:
                    assert response.status == 200, f"capabilities → {response.status}"
                    capabilities = await response.json()
                search_feature = capabilities.get("features", {}).get("global_search")
                if search_feature:
                    async with session.post(
                        f"{base}/api/search",
                        json={
                            "query": fixture["term"],
                            "scopes": ["chats", "tools", "workflows", "scheduled", "events"],
                            "filters": {}, "sort": "relevance", "grouping": "match",
                            "limit": 100, "cursor": None,
                        },
                    ) as response:
                        if response.status == 200:
                            found = await response.json()
                            kinds = {item["target"]["kind"] for item in found.get("items", [])}
                            if kinds == {
                                "chat", "chat_message", "chat_tool",
                                "workflow_definition", "workflow_run",
                                "scheduled_definition", "scheduled_run",
                                "event_definition", "event_delivery",
                            }:
                                break
                await asyncio.sleep(0.25)
            else:
                raise RuntimeError("operational fixture did not become searchable within 60s")

            assert found is not None
            print("  ✓ capabilities + all 5 scopes / 9 typed targets", flush=True)
            async with session.get(f"{base}/api/history?limit=100&include_children=true") as response:
                assert response.status == 200, f"history → {response.status}"
                history = await response.json()
            assert any(item.get("resource_id") == ids["session"] for item in history["items"])
            print("  ✓ unified history contains the owned chat", flush=True)

            async with session.get(
                f"{base}/api/sessions/{ids['session']}/messages",
                params={"around": ids["user_message"], "before": 1, "after": 2},
            ) as response:
                assert response.status == 200, f"messages-around → {response.status}"
                around = await response.json()
            assert around["anchor_found"] is True
            assert {message["id"] for message in around["messages"]} >= {
                ids["user_message"], ids["tool_message"], ids["assistant_message"],
            }
            print("  ✓ messages-around resolves the exact canary anchor", flush=True)

            tool_target = next(
                item["target"] for item in found["items"]
                if item["target"]["kind"] == "chat_tool"
            )
            async with session.get(
                f"{base}/api/tool-invocations/{tool_target['tool_invocation_id']}"
            ) as response:
                assert response.status == 200, f"tool detail → {response.status}"
                tool = await response.json()
            assert tool["session_id"] == ids["session"]

            async with session.post(
                f"{base}/api/search",
                json={
                    "query": fixture["forbidden_term"], "scopes": ["chats"],
                    "filters": {}, "sort": "relevance", "grouping": "match",
                    "limit": 10, "cursor": None,
                },
            ) as response:
                assert response.status == 200, f"ACL search → {response.status}"
                hidden = await response.json()
            assert hidden["items"] == []
            async with session.get(
                f"{base}/api/sessions/{ids['forbidden_session']}/messages"
            ) as response:
                assert response.status == 404, f"ACL messages expected 404, got {response.status}"
            print("  ✓ private canary owned by another principal is invisible", flush=True)

            await _remove_operational_canary_sources(db_path, fixture)
            sources_removed = True
            cleanup_seq = await _wait_operational_canary_cleanup(db_path, fixture)
            cleanup_consumed = True
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                async with session.get(f"{base}/api/capabilities") as response:
                    caps = await response.json()
                if int(caps.get("storage", {}).get("indexed_seq", 0)) >= cleanup_seq:
                    async with session.post(
                        f"{base}/api/search",
                        json={
                            "query": fixture["term"],
                            "scopes": ["chats", "tools", "workflows", "scheduled", "events"],
                            "filters": {}, "sort": "relevance", "grouping": "match",
                            "limit": 10, "cursor": None,
                        },
                    ) as response:
                        if response.status == 200 and not (await response.json())["items"]:
                            break
                await asyncio.sleep(0.25)
            else:
                raise RuntimeError("operational canary cleanup was not consumed within 60s")
            print("  ✓ canary source rows removed and FTS tombstones consumed", flush=True)
    finally:
        if not sources_removed:
            try:
                await _remove_operational_canary_sources(db_path, fixture)
                sources_removed = True
            except Exception as cleanup_error:  # noqa: BLE001
                print(f"  ! source cleanup failed: {cleanup_error}", flush=True)
        if sources_removed and not cleanup_consumed:
            try:
                await _wait_operational_canary_cleanup(db_path, fixture)
                cleanup_consumed = True
            except Exception as cleanup_error:  # noqa: BLE001
                print(f"  ! cleanup drain incomplete; metadata retained safely: {cleanup_error}", flush=True)
        if cleanup_consumed:
            try:
                await _purge_operational_canary_metadata(db_path, fixture)
            except Exception as cleanup_error:  # noqa: BLE001
                print(f"  ! normalized canary cleanup failed: {cleanup_error}", flush=True)
        await proxy.stop()


async def _do_pair(ticket_str: str, handle: str, password: str) -> int:
    from src.network.auth.device_cert import verify_cert
    from src.network.client.login import LoginError, login
    from src.network.identity import Identity
    from src.network.iroh_node import IrohNode
    from src.network.peers import coordinator_node_id_to_pubkey_bytes
    from src.network.ticket import InviteTicket

    dt = InviteTicket.decode(ticket_str)
    if dt.role != "device":
        print(f"  ✗ ticket role is {dt.role!r}, expected 'device'", flush=True)
        return 1
    if dt.bind_to and dt.bind_to != handle:
        print(f"  ✗ ticket bind_to={dt.bind_to!r}, but --handle={handle!r}",
              flush=True)
        return 1

    coord = dt.coordinator_node_id
    coord_pub = coordinator_node_id_to_pubkey_bytes(coord)
    print(f"  coordinator : {coord}", flush=True)
    print(f"  network     : {dt.network_name} / {dt.network_id}", flush=True)
    print(f"  invite code : {dt.code} (bind_to={dt.bind_to!r})", flush=True)
    print(f"  handle      : {handle}", flush=True)

    new_dev = Identity.generate()  # fresh device pubkey
    node = IrohNode(new_dev)
    await node.start()
    try:
        print("\n[1/1] login with device invite (new device pairing)", flush=True)
        try:
            cert_wire = await login(
                node=node,
                coordinator_node_id=coord,
                coordinator_pubkey_bytes=coord_pub,
                handle=handle, password=password,
                device_identity=new_dev,
                network_id=dt.network_id,
                invite_code=dt.code,
                label="smoke-pair",
            )
        except LoginError as e:
            print(f"  ✗ device pair failed: {e}", flush=True)
            return 1
        verify_cert(
            cert_wire,
            coordinator_pubkey=Ed25519PublicKey.from_public_bytes(coord_pub),
        )
        print("  ✓ new device paired + cert minted", flush=True)
    finally:
        await node.stop()

    print("\nPAIR PHASE PASSED.", flush=True)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)

    pr = sub.add_parser("register", help="Register a smoke user (consumes user invite)")
    pr.add_argument("--ticket", required=True)
    pr.add_argument("--password", required=True)
    pr.add_argument(
        "--operational-db",
        help="Opt in to history/search canaries using the selected agent's local SQLite DB",
    )
    pr.add_argument(
        "--agent-handle",
        help="Target this agent handle (recommended when the network has multiple agents)",
    )

    pp = sub.add_parser("pair", help="Pair a new device for an existing user (consumes device invite)")
    pp.add_argument("--ticket", required=True)
    pp.add_argument("--handle", required=True)
    pp.add_argument("--password", required=True)

    args = p.parse_args()
    if args.mode == "register":
        rc = asyncio.run(_do_register(
            args.ticket,
            args.password,
            operational_db=args.operational_db,
            agent_handle=args.agent_handle,
        ))
    else:
        rc = asyncio.run(_do_pair(args.ticket, args.handle, args.password))
    sys.exit(rc)


if __name__ == "__main__":
    main()
