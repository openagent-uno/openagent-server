"""Hermetic collaboration contracts: real aiohttp, SQLite and StreamSession.

Run independently with ``python -m unittest scripts.tests.test_collaboration``.
Only model output and the certificate transport are fixtures.
"""

import asyncio
import contextlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer, make_mocked_request

from src.core.on_behalf_context import current_on_behalf_identity
from src.gateway.collaboration import Collaboration, CommandReply
from src.gateway.collaboration_access import authorize
from src.memory.db import MemoryDB


class CollaborationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db = MemoryDB(str(Path(self.directory.name) / "test.db"))
        await self.db.connect()
        self.addAsyncCleanup(self.db.close)
        await self.db.upsert_session("chat", client_id="alice")
        conn = await self.db._ensure_connected()
        row = await (
            await conn.execute("SELECT tenant_id FROM sessions_v2 WHERE id='chat'")
        ).fetchone()
        self.tenant = row[0]
        await conn.execute(
            "INSERT INTO resource_acl (tenant_id,resource_type,resource_id,principal_type,principal_id,"
            "permission,acl_version,granted_by_principal_id,granted_at_ms) VALUES(?, 'session','chat','user','bob','admin',1,'user:alice',?)",
            (self.tenant, int(time.time() * 1000)),
        )
        await conn.commit()
        self.revoked = set()
        self.started = asyncio.Queue()
        self.finish = asyncio.Event()
        self.seen = []
        self.attachments = []
        parent = self

        class Agent:
            memory_db = parent.db

            async def run_stream(self, **kwargs):
                self.cancel_current = asyncio.Event()
                parent.seen.append((kwargs["author"], current_on_behalf_identity()))
                parent.attachments.append(kwargs.get("attachments"))
                yield {"kind": "delta", "text": kwargs["message"] + " partial"}
                parent.started.put_nowait(kwargs["message"])
                waits = [
                    asyncio.create_task(parent.finish.wait()),
                    asyncio.create_task(self.cancel_current.wait()),
                ]
                try:
                    await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for task in waits:
                        task.cancel()
                    await asyncio.gather(*waits, return_exceptions=True)
                yield {"kind": "done"}

            async def request_cancel(self, sid):
                self.cancel_current.set()
                return True

        async def command(ws, client_id, name, sid, **kwargs):
            self.commands.append((name, current_on_behalf_identity().handle))
            await ws.send_json(
                {"type": "command_result", "text": "Compacted conversation."}
            )

        self.commands = []
        self.gateway = SimpleNamespace(
            agent=Agent(),
            _stream_sessions={},
            _request_device_still_authorized=AsyncMock(
                side_effect=lambda req, cert: cert.handle not in self.revoked
            ),
            _make_stream_pre_dispatch_hook=lambda *args: None,
            _make_stream_post_turn_hook=lambda: None,
            _handle_command=command,
        )
        self.service = self.gateway._collaboration = Collaboration(self.gateway)
        self.gateway.broadcast_resource = AsyncMock(
            side_effect=lambda *args: self.service.hub.resource(*args)
        )
        self.addAsyncCleanup(self.service.close)
        # The deterministic model emits wire events, not canonical provider
        # runs. Avoid waiting for a projection that this fixture never writes;
        # ACLs and model-pin persistence still use the real SQLite database.
        parts = patch(
            "src.memory.message_parts.persist_parts_for_latest_message", AsyncMock()
        )
        parts.start()
        self.addCleanup(parts.stop)
        for name in ("resolve_stt", "resolve_tts"):
            patched = patch("src.stream.session." + name, AsyncMock(return_value=None))
            patched.start()
            self.addCleanup(patched.stop)

        @web.middleware
        async def identity(request, handler):
            handle = request.headers.get("Test-User", "alice")
            request["user_handle"] = handle
            request["client_id"] = handle + "-device"
            request["network_id"] = self.tenant
            request["auth_kind"] = "device_cert"
            request["device_cert"] = SimpleNamespace(
                handle=handle,
                device_pubkey_hex=handle + "-device",
                network_id=self.tenant,
                capabilities=[],
            )
            return await handler(request)

        app = web.Application(middlewares=[identity])
        app["gateway"] = self.gateway
        app.router.add_get("/ws/collaboration", self.service.hub.handle)
        app.router.add_post("/api/collaboration/turns", self.service.handle_chat)
        app.router.add_post("/api/collaboration/stop", self.service.handle_stop)
        app.router.add_get("/api/collaboration", self.service.handle_info)
        app.router.add_get(
            "/api/collaboration/{session_id}/commands", self.service.handle_commands
        )
        from src.gateway.collaboration_members import handle_members

        app.router.add_get("/api/collaboration/{session_id}/members", handle_members)
        app.router.add_put("/api/collaboration/{session_id}/members", handle_members)
        self.server = TestServer(app)
        await self.server.start_server()
        self.addAsyncCleanup(self.server.close)
        self.http = ClientSession()
        self.addAsyncCleanup(self.http.close)

    def request(self, user="alice", **body):
        request = make_mocked_request(
            "POST", "/api/collaboration/turns", app={"gateway": self.gateway}
        )
        request["user_handle"] = user
        request["client_id"] = user + "-device"
        request["network_id"] = self.tenant
        request["auth_kind"] = "device_cert"
        request["device_cert"] = SimpleNamespace(
            handle=user,
            device_pubkey_hex=user + "-device",
            network_id=self.tenant,
            capabilities=[],
        )
        request.json = AsyncMock(return_value={"session_id": "chat", **body})
        return request

    async def connect(self, user="alice", sid="chat"):
        ws = await self.http.ws_connect(
            self.server.make_url("/ws/collaboration"), headers={"Test-User": user}
        )
        self.addAsyncCleanup(ws.close)
        self.assertTrue((await ws.receive_json())["shared"])
        await ws.send_json(
            {
                "type": "observe",
                "sessions": [sid],
                "focus": {"kind": "session", "id": sid},
            }
        )
        return ws

    async def receive(self, ws, kind, predicate=lambda frame: True):
        async with asyncio.timeout(3):
            while True:
                frame = await ws.receive_json()
                if frame["type"] == kind and predicate(frame):
                    return frame

    async def send(self, user, request_id, text="Hello", delivery="queue"):
        response = await self.http.post(
            self.server.make_url("/api/collaboration/turns"),
            headers={"Test-User": user},
            json={
                "session_id": "chat",
                "request_id": request_id,
                "message": text,
                "delivery": delivery,
            },
        )
        self.addAsyncCleanup(response.release)
        return response.status, await response.json()

    async def test_attachment_refs_use_authenticated_artifact_acl(self):
        from src.core.on_behalf_context import OnBehalfIdentity
        from src.memory.artifacts import (
            normalize_inbound_attachments,
            public_attachment_ref,
        )

        source = Path(self.directory.name) / "attachment.txt"
        source.write_text("shared artifact")
        principal = OnBehalfIdentity.from_certificate(
            self.request().get("device_cert"), auth_kind="device_cert"
        )
        ref = (
            await normalize_inbound_attachments(
                self.db,
                [{"path": str(source), "filename": source.name}],
                session_id="",
                principal=principal,
                allow_local_paths=True,
            )
        )[0]

        async def send_ref(user, request_id):
            return await self.http.post(
                self.server.make_url("/api/collaboration/turns"),
                headers={"Test-User": user},
                json={
                    "session_id": "chat",
                    "request_id": request_id,
                    "message": "read attachment",
                    "attachments": [public_attachment_ref(ref)],
                },
            )

        # Access to the conversation does not grant another user's private upload.
        denied = await send_ref("bob", "private-ref")
        # A refused attachment is a client error, not a gateway fault, and the
        # body must not confirm that the artifact id exists.
        self.assertEqual(denied.status, 404)
        self.assertNotIn(ref["artifact_id"], await denied.text())
        self.assertEqual(self.seen, [])
        self.finish.set()
        accepted = await send_ref("alice", "owned-ref")
        self.assertEqual(accepted.status, 200, await accepted.text())
        self.assertEqual(self.attachments[-1][0]["artifact_id"], ref["artifact_id"])
        public = self.service.hub.sessions["chat"]["turns"][-1]["messages"][0][
            "attachments"
        ][0]
        self.assertNotIn("path", public)

    async def test_unknown_attachment_is_refused_without_dispatch(self):
        response = await self.http.post(
            self.server.make_url("/api/collaboration/turns"),
            json={
                "session_id": "chat",
                "request_id": "ghost",
                "message": "read",
                "attachments": [{"artifact_id": "artifact-that-does-not-exist"}],
            },
        )
        self.addAsyncCleanup(response.release)
        # 404 rather than a generic 500: the turn never reached the runner and
        # the caller can distinguish a bad reference from a gateway failure.
        self.assertEqual(response.status, 404)
        self.assertEqual(self.seen, [])
        self.assertEqual(self.service.hub.sessions.get("chat", {}).get("turns", []), [])

    async def test_revoked_member_reconnect_never_replays_cached_transcript(self):
        bob = await self.connect("bob")
        self.service.hub.begin("chat", "Confidential prompt")
        state = await self.receive(bob, "shared_state")
        self.assertEqual(
            state["turns"][-1]["messages"][0]["text"], "Confidential prompt"
        )
        conn = await self.db._ensure_connected()
        await conn.execute("DELETE FROM resource_acl WHERE principal_id='bob'")
        await conn.commit()
        self.service.hub.wake()
        await self.receive(bob, "shared_revoked")
        await bob.close()
        # A fresh socket must re-authorize from the database, not from replay
        # state the hub still holds for the remaining members.
        again = await self.http.ws_connect(
            self.server.make_url("/ws/collaboration"), headers={"Test-User": "bob"}
        )
        self.addAsyncCleanup(again.close)
        self.assertTrue((await again.receive_json())["shared"])
        await again.send_json(
            {
                "type": "observe",
                "sessions": ["chat"],
                "focus": {"kind": "session", "id": "chat"},
            }
        )
        async with asyncio.timeout(3):
            while True:
                frame = await again.receive_json()
                self.assertNotEqual(frame["type"], "shared_state")
                if frame["type"] == "shared_revoked":
                    break
        self.assertEqual(
            (
                await self.http.get(
                    self.server.make_url("/api/collaboration/chat/commands"),
                    headers={"Test-User": "bob"},
                )
            ).status,
            403,
        )

    async def test_idle_handoff_refuses_while_another_turn_is_queued(self):
        first = asyncio.create_task(self.send("alice", "first"))
        await asyncio.wait_for(self.started.get(), 3)
        second = asyncio.create_task(self.send("bob", "second"))
        async with asyncio.timeout(3):
            while ("chat", "second") not in self.service.requests:
                await asyncio.sleep(0.01)
        # The queued turn has no runner yet, so `active` is unset; releasing
        # here would hand the session to a second writer mid-queue.
        self.assertFalse(await self.service.release_idle("chat"))
        self.finish.set()
        self.assertEqual((await first)[0], 200)
        self.assertEqual((await second)[0], 200)
        self.assertTrue(await self.service.release_idle("chat"))

    async def test_owner_sharing_and_immediate_revoke_stop(self):
        url = self.server.make_url("/api/collaboration/chat/members")
        response = await self.http.put(
            url,
            headers={"Test-User": "bob"},
            json={"handle": "mallory", "permission": "admin"},
        )
        self.assertEqual(response.status, 403)
        first = asyncio.create_task(self.send("bob", "member-run"))
        await asyncio.wait_for(self.started.get(), 3)
        response = await self.http.put(url, json={"handle": "bob", "permission": None})
        self.assertEqual(response.status, 200)
        self.assertIn((await asyncio.wait_for(first, 3))[0], (200, 403))
        self.assertEqual((await self.send("bob", "denied"))[0], 403)
        response = await self.http.put(
            url, json={"handle": "bob", "permission": "view"}
        )
        self.assertEqual(response.status, 200)
        self.assertEqual((await self.send("bob", "view-only"))[0], 403)
        response = await self.http.put(
            url, json={"handle": "bob", "permission": "admin"}
        )
        self.assertEqual(response.status, 200)

    async def test_durable_commands_remain_after_replay_expiry(self):
        self.assertEqual((await self.send("alice", "compact", "/compact"))[0], 200)
        self.service.hub.sessions.clear()
        response = await self.http.get(
            self.server.make_url("/api/collaboration/chat/commands")
        )
        data = await response.json()
        self.assertEqual(data["turns"][0]["messages"][0]["text"], "/compact")
        self.assertEqual(data["turns"][0]["messages"][0]["author"]["handle"], "alice")
        self.assertEqual(
            data["turns"][0]["messages"][1]["text"], "Compacted conversation."
        )
        response = await self.http.get(
            self.server.make_url("/api/collaboration/chat/commands"),
            headers={"Test-User": "mallory"},
        )
        self.assertEqual(response.status, 403)

    async def test_idle_handoff_never_detaches_running_turn(self):
        first = asyncio.create_task(self.send("alice", "handoff"))
        await asyncio.wait_for(self.started.get(), 3)
        self.assertFalse(await self.service.release_idle("chat"))
        self.finish.set()
        self.assertEqual((await first)[0], 200)
        self.assertTrue(await self.service.release_idle("chat"))
        self.assertFalse(self.service.owns("chat"))

    async def test_shared_input_rejects_local_paths_and_invalid_instances(self):
        for extra in (
            {"attachments": [{"path": "/etc/passwd"}]},
            {"client_instance_id": "../device"},
        ):
            response = await self.service.handle_chat(
                self.request(message="x", request_id="invalid", **extra)
            )
            self.assertEqual(response.status, 400)

    async def test_two_users_replay_presence_and_steering_keep_authors(self):
        alice, bob = await self.connect(), await self.connect("bob")
        first = asyncio.create_task(self.send("alice", "first", "Original"))
        self.assertEqual(await asyncio.wait_for(self.started.get(), 3), "Original")
        has_response = (
            lambda f: bool(f["turns"]) and len(f["turns"][-1]["messages"]) == 2
        )
        one = await self.receive(alice, "shared_state", has_response)
        two = await self.receive(bob, "shared_state", has_response)
        self.assertEqual(one, two)
        await bob.close()
        bob = await self.connect("bob")
        replay = await self.receive(bob, "shared_state", has_response)
        self.assertEqual(one["turns"], replay["turns"])
        second = asyncio.create_task(self.send("bob", "second", "Correction", "steer"))
        status, result = await asyncio.wait_for(first, 3)
        self.assertEqual(status, 200)
        self.assertTrue(result["interrupted"])
        self.assertEqual(result["response"], "Original partial")
        self.assertEqual(await asyncio.wait_for(self.started.get(), 3), "Correction")
        compact = asyncio.create_task(self.send("alice", "compact", "/compact"))
        await asyncio.sleep(0.05)
        self.assertFalse(compact.done())
        self.finish.set()
        self.assertEqual((await second)[0], 200)
        self.assertEqual((await compact)[1]["response"], "Compacted conversation.")
        self.assertEqual(self.commands, [("compact", "alice")])
        self.assertEqual(
            [(author["handle"], identity.handle) for author, identity in self.seen],
            [("alice", "alice"), ("bob", "bob")],
        )
        self.assertEqual(
            [turn["runId"] for turn in self.service.hub.sessions["chat"]["turns"]],
            ["first", "second", "compact"],
        )

    async def test_same_account_tabs_disconnect_independently(self):
        first, second, bob = (
            await self.connect(),
            await self.connect(),
            await self.connect("bob"),
        )
        roster = await self.receive(
            bob, "shared_presence", lambda f: len(f["people"]) == 2
        )
        self.assertEqual(
            {p["userId"] for p in roster["people"]}, {"user:alice", "user:bob"}
        )
        await first.close()
        self.service.hub.begin("chat", "Still here", run_id="next")
        self.assertEqual(
            (await self.receive(second, "shared_state", lambda f: bool(f["turns"])))[
                "turns"
            ][0]["runId"],
            "next",
        )

    async def test_acl_revocation_clears_live_state_and_prevents_steering(self):
        bob = await self.connect("bob")
        self.service.hub.begin("chat", "Secret")
        await self.receive(bob, "shared_state")
        conn = await self.db._ensure_connected()
        await conn.execute("DELETE FROM resource_acl WHERE principal_id='bob'")
        await conn.commit()
        self.service.hub.wake()
        await self.receive(bob, "shared_revoked")
        self.assertEqual(
            (await self.send("bob", "denied", "Interrupt", "steer"))[0], 403
        )

    async def test_device_revocation_closes_existing_observer(self):
        bob = await self.connect("bob")
        await self.receive(bob, "shared_state")
        self.revoked.add("bob")
        self.service.hub.wake()
        await self.receive(bob, "auth_error")

    async def test_view_permission_never_grants_write(self):
        conn = await self.db._ensure_connected()
        await conn.execute(
            "UPDATE resource_acl SET permission='view' WHERE principal_id='bob'"
        )
        await conn.commit()
        self.assertEqual(
            (await authorize(self.request("bob"), [{"kind": "session", "id": "chat"}]))[
                "allowed"
            ],
            [{"kind": "session", "id": "chat"}],
        )
        self.assertEqual((await self.send("bob", "read-only"))[0], 403)

    async def test_unknown_and_cross_tenant_sessions_are_invisible(self):
        request = self.request()
        self.assertEqual(
            (await authorize(request, [{"kind": "session", "id": "unknown"}]))[
                "allowed"
            ],
            [],
        )
        request["network_id"] = "other-network"
        self.assertEqual(
            (await authorize(request, [{"kind": "session", "id": "chat"}]))["allowed"],
            [],
        )

    async def test_policy_cannot_expand_scope_or_bypass_revocation(self):
        self.gateway.collaboration_authorizer = AsyncMock(
            return_value={
                "userId": "external",
                "name": "External",
                "allowed": [{"kind": "session", "id": "other"}],
            }
        )
        with self.assertRaises(PermissionError):
            await authorize(self.request(), [])
        self.revoked.add("alice")
        with self.assertRaises(PermissionError):
            await authorize(self.request(), [])

    async def test_http_disconnect_deduplication_and_reauthorization(self):
        request = self.request(message="Hello", request_id="dedup")
        call = asyncio.create_task(self.service.handle_chat(request))
        await asyncio.wait_for(self.started.get(), 3)
        call.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await call
        self.assertFalse(self.service.requests[("chat", "dedup")][1].cancelled())
        self.finish.set()
        response = await self.service.handle_chat(request)
        self.assertEqual(response.status, 200)
        self.assertEqual(len(self.seen), 1)
        conflict = self.request("bob", message="Hello", request_id="dedup")
        self.assertEqual((await self.service.handle_chat(conflict)).status, 409)
        self.revoked.add("alice")
        self.assertEqual((await self.service.handle_chat(request)).status, 403)

    async def test_stop_exact_queued_request_does_not_stop_other_user(self):
        first = asyncio.create_task(self.send("alice", "running"))
        await asyncio.wait_for(self.started.get(), 3)
        second = asyncio.create_task(self.send("bob", "queued"))
        async with asyncio.timeout(3):
            while ("chat", "queued") not in self.service.requests:
                await asyncio.sleep(0.01)
        response = await self.service.handle_stop(
            self.request("bob", request_id="queued")
        )
        self.assertTrue(json.loads(response.body)["stopped"])
        self.assertTrue((await second)[1]["interrupted"])
        self.assertFalse(first.done())
        self.finish.set()
        self.assertFalse((await first)[1]["interrupted"])

    async def test_model_pin_auto_and_disabled_provider_use_native_acl(self):
        provider = await self.db.upsert_provider(name="qa", framework="api-based")
        await self.db.upsert_model(provider_id=provider, model="shared")
        collector = CommandReply(self.service.hub, "chat")
        await self.service._model(self.request("bob"), "chat", "qa:shared", collector)
        self.assertFalse(collector.errored)
        self.assertEqual(await self.db.get_session_pin("chat"), "qa:shared")
        await self.service._model(self.request("bob"), "chat", "auto", collector)
        self.assertIsNone(await self.db.get_session_pin("chat"))
        await self.db.upsert_provider(name="qa", framework="api-based", enabled=False)
        await self.service._model(self.request(), "chat", "qa:shared", collector)
        self.assertTrue(collector.errored)

    async def test_invalid_frames_and_oversized_inputs_fail_closed(self):
        for frame in [
            {"type": "text_final"},
            {"type": "observe", "sessions": ["../x"]},
            {"type": "observe", "focus": {"kind": [], "id": "x"}},
        ]:
            ws = await self.connect()
            await ws.send_json(frame)
            async with asyncio.timeout(3):
                while not ws.closed:
                    await ws.receive()
            self.assertEqual(ws.close_code, 1008)
        for body in [
            {"message": "x", "request_id": "bad", "delivery": "surprise"},
            {"message": "x" * 131073, "request_id": "long"},
            {"message": "x", "request_id": "bad/id"},
        ]:
            self.assertEqual(
                (await self.service.handle_chat(self.request(**body))).status, 400
            )

    async def test_revocation_while_queued_prevents_dispatch(self):
        first = asyncio.create_task(self.send("alice", "first"))
        await asyncio.wait_for(self.started.get(), 3)
        second = asyncio.create_task(self.send("bob", "second"))
        async with asyncio.timeout(3):
            while ("chat", "second") not in self.service.requests:
                await asyncio.sleep(0.01)
        self.revoked.add("bob")
        self.finish.set()
        self.assertEqual((await first)[0], 200)
        self.assertEqual((await second)[0], 403)
        self.assertEqual(len(self.seen), 1)

    async def test_native_transport_cannot_overlap_or_mutate_shared_turn(self):
        from src.gateway.collaboration import guard_mutation

        self.gateway._stream_sessions[("alice", "chat")] = SimpleNamespace(
            session=SimpleNamespace(has_active_turn=lambda: True)
        )
        self.assertEqual((await self.send("alice", "busy"))[0], 409)
        self.assertEqual(self.seen, [])
        self.gateway._stream_sessions.clear()
        self.finish.set()
        self.assertEqual((await self.send("alice", "shared"))[0], 200)
        self.assertEqual(guard_mutation(self.request(), "chat").status, 409)
        self.assertIsNone(guard_mutation(self.request(), "other"))

    async def test_device_revocation_stops_only_its_active_author(self):
        first = asyncio.create_task(self.send("alice", "first"))
        await asyncio.wait_for(self.started.get(), 3)
        await self.service.revoke_device("bob-device")
        self.assertFalse(first.done())
        await self.service.revoke_device("alice-device")
        self.assertTrue((await asyncio.wait_for(first, 3))[1]["interrupted"])

    async def test_nonterminal_cancellation_releases_collector_and_fences_zombie(self):
        from src.stream.channel import BatchedChannel

        started = asyncio.Event()

        async def stalled(*args, **kwargs):
            self.service.hub.publish(
                {"type": "delta", "session_id": "chat", "text": "Saved partial"}
            )
            started.set()
            await asyncio.Event().wait()

        with patch.object(BatchedChannel, "run_one_shot", side_effect=stalled):
            first = asyncio.create_task(self.send("alice", "first"))
            await asyncio.wait_for(started.wait(), 3)
            runtime = self.service.runtimes["chat"]
            # Simulate a provider detached by StreamSession's bounded cancel
            # path, with no terminal event delivered to the batch collector.
            zombie = asyncio.create_task(asyncio.sleep(60))
            runtime.session._detached_turns.add(zombie)
            try:
                second = asyncio.create_task(
                    self.send("bob", "second", "Steer", "steer")
                )
                self.assertEqual(
                    (await asyncio.wait_for(first, 3))[1]["response"], "Saved partial"
                )
                self.assertEqual((await asyncio.wait_for(second, 3))[0], 409)
                self.assertFalse(
                    self.service.hub.sessions["chat"]["turns"][-1]["active"]
                )
            finally:
                zombie.cancel()
                await asyncio.gather(zombie, return_exceptions=True)
                runtime.session._detached_turns.clear()

    async def test_switching_views_during_authorization_drops_old_snapshot(self):
        ws = await self.connect()
        await self.receive(ws, "shared_state")
        entered, release = asyncio.Event(), asyncio.Event()

        async def policy(request, targets, permission):
            entered.set()
            await release.wait()
            return {"userId": "user:alice", "name": "alice", "allowed": targets}

        self.gateway.collaboration_authorizer = policy
        self.service.hub.begin("chat", "Old view")
        await asyncio.wait_for(entered.wait(), 3)
        await ws.send_json({"type": "observe", "sessions": ["next"], "focus": None})
        await asyncio.sleep(0.02)
        release.set()
        state = await self.receive(ws, "shared_state")
        self.assertEqual(state["session_id"], "next")

    async def test_bounded_replay_and_idle_runtime_cleanup(self):
        self.service.hub.begin("chat", "prompt")
        for _ in range(1000):
            self.service.hub.publish(
                {"type": "delta", "session_id": "chat", "text": "x" * 200}
            )
        turn = self.service.hub.sessions["chat"]["turns"][-1]
        self.assertEqual(len(turn["messages"][-1]["text"]), 131072)
        self.assertTrue(turn["truncated"])
        self.finish.set()
        self.assertEqual((await self.send("alice", "short"))[0], 200)
        fingerprint, task, _ = self.service.requests[("chat", "short")]
        self.service.requests[("chat", "short")] = (
            fingerprint,
            task,
            time.monotonic() - 61,
        )
        await self.service.prune()
        self.assertFalse(self.service.owns("chat"))


if __name__ == "__main__":
    unittest.main()
