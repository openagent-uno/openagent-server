"""Server-owned shared text turns and read-only live observation.

This additive API is independent of any embedding product. It uses the normal
StreamSession runner, canonical session ACLs and authenticated turn identities.
The legacy audio/video and local-capability transport is unchanged; a session
cannot have two different execution transports active at the same time.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import time
from dataclasses import dataclass, field

from aiohttp import web

from .collaboration_access import authorize
from .collaboration_hub import IDENTIFIER, SharedAgentHub

COMMANDS = {"compact", "model", "context", "status", "queue", "help", "usage", "stop"}


class BusyError(Exception):
    pass


@dataclass
class Runtime:
    session: object
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    admission: asyncio.Lock = field(default_factory=asyncio.Lock)
    active: dict | None = None


class CommandReply:
    closed = False

    def __init__(self, hub, sid):
        self.hub, self.sid = hub, sid
        self.text, self.errored = "", False

    async def send_json(self, frame):
        if frame.get("type") == "command_result":
            self.text = frame.get("text", "")
        elif frame.get("type") == "session_compacted":
            self.hub.resource("session", "updated", self.sid)


class Collaboration:
    def __init__(self, gateway):
        self.gateway = gateway
        self.hub = SharedAgentHub()
        self.runtimes = {}
        self.requests = {}
        self.registry_lock = asyncio.Lock()
        self.reaper = None
        self.closed = False

    def owns(self, sid):
        return sid in self.runtimes

    async def handle_info(self, request):
        return web.json_response(
            {"version": 1, "attachments": True, "client_capabilities": True}
        )

    async def handle_commands(self, request):
        sid = request.match_info["session_id"]
        target = {"kind": "session", "id": sid}
        try:
            access = await authorize(request, [target])
            if target not in access["allowed"]:
                raise PermissionError()
            conn = await self.gateway.agent.memory_db._ensure_connected()
            rows = await (
                await conn.execute(
                    "SELECT seq, ts_ms, data FROM session_events WHERE session_id=? AND type='command/result' ORDER BY seq DESC LIMIT 64",
                    (sid,),
                )
            ).fetchall()
            commands = []
            for row in reversed(rows):
                data = json.loads(row["data"])
                commands.append(
                    {
                        "id": data.get("turn_id") or f"command:{row['seq']}",
                        "runId": data.get("request_id"),
                        "active": False,
                        "startedAt": row["ts_ms"] / 1000,
                        "messages": [
                            {
                                "id": "input",
                                "role": "user",
                                "text": data.get("command", ""),
                                "timestamp": row["ts_ms"] / 1000,
                                "author": data.get("author"),
                            },
                            {
                                "id": "response",
                                "role": "assistant",
                                "text": data.get("text", ""),
                                "timestamp": row["ts_ms"] / 1000,
                            },
                        ],
                    }
                )
            access = await authorize(request, [target])
            if target not in access["allowed"]:
                raise PermissionError()
            return web.json_response({"turns": commands})
        except PermissionError:
            return web.json_response({"error": "Session unavailable"}, status=403)

    async def release_idle(self, sid):
        """Allow an idle text session to move back to its native media transport."""
        async with self.registry_lock:
            runtime = self.runtimes.get(sid)
            if runtime is None:
                return True
            if (
                runtime.active
                or runtime.session.has_active_turn()
                or runtime.session._detached_turns
            ):
                return False
            if any(
                key[0] == sid and not entry[1].done()
                for key, entry in self.requests.items()
            ):
                return False
            await runtime.session.close()
            self.runtimes.pop(sid, None)
            return True

    async def revoke_device(self, device_id):
        self.hub.wake()
        for runtime in list(self.runtimes.values()):
            async with runtime.admission:
                active = runtime.active
                if active and active["device_id"] == device_id:
                    await self._interrupt(runtime)

    async def close(self):
        self.closed = True
        if self.reaper is not None:
            self.reaper.cancel()
            await asyncio.gather(self.reaper, return_exceptions=True)
        tasks = [entry[1] for entry in self.requests.values()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(
            *(rt.session.close() for rt in self.runtimes.values()),
            return_exceptions=True,
        )
        await self.hub.close()
        self.requests.clear()
        self.runtimes.clear()

    async def _reap(self):
        while self.requests or self.runtimes:
            await asyncio.sleep(10)
            await self.prune()

    async def prune(self):
        async with self.registry_lock:
            for key, entry in list(self.requests.items()):
                if entry[1].done() and time.monotonic() - entry[2] > 60:
                    del self.requests[key]
            retained = {key[0] for key in self.requests}
            for sid, runtime in list(self.runtimes.items()):
                if (
                    sid not in retained
                    and not runtime.active
                    and not runtime.session._detached_turns
                ):
                    await runtime.session.close()
                    self.runtimes.pop(sid, None)
        self.hub.prune()

    async def _runtime(self, sid, request):
        from src.stream.session import StreamSession
        from src.stream.wire import event_to_wire
        from .api import chat

        async with self.registry_lock, chat._sessions_registry_lock:
            if self.closed:
                raise BusyError("The gateway is stopping")
            if sid in self.runtimes:
                return self.runtimes[sid]
            # Adopting an idle native session is safe. An active legacy turn
            # must finish before switching transports; never start a second
            # runner over the same durable/provider-native history.
            holders = [
                (key, holder)
                for key, holder in self.gateway._stream_sessions.items()
                if key[1] == sid
            ]
            rest = [
                (key, entry)
                for key, entry in chat._sessions.items()
                if key[1] == sid
                and getattr(entry[0], "_agent", None) is self.gateway.agent
            ]
            if any(holder.session.has_active_turn() for _, holder in holders) or any(
                entry[1].locked() or entry[0].has_active_turn() for _, entry in rest
            ):
                raise BusyError("An existing transport is still running this session")
            if len(self.runtimes) >= 128:
                raise BusyError("Live session capacity reached")
            # Reserve synchronously before closing transports (which awaits).
            session = StreamSession(
                self.gateway.agent,
                client_id=request.get("client_id") or "shared",
                session_id=sid,
                profile="batched",
                speak_enabled=False,
                coalesce_window_ms=0,
                handle=request.get("user_handle"),
            )
            runtime = Runtime(session)
            self.runtimes[sid] = runtime
            try:
                for key, _ in holders:
                    await self.gateway._close_stream_session(key)
                for key, entry in rest:
                    chat._sessions.pop(key, None)
                    await entry[0].close()
                session.on_event = lambda evt: self.hub.publish(event_to_wire(evt))
                session.pre_dispatch_hook = self.gateway._make_stream_pre_dispatch_hook(
                    session.client_id, sid
                )
                session.post_turn_hook = self.gateway._make_stream_post_turn_hook()
                await session.start()
            except BaseException:
                self.runtimes.pop(sid, None)
                await session.close()
                raise
            return runtime

    async def _check(self, request, sid):
        target = {"kind": "session", "id": sid}
        access = await authorize(request, [target], "admin")
        if target not in access["allowed"]:
            raise PermissionError("Session unavailable")
        return access

    async def _interrupt(self, runtime):
        active = runtime.active
        if not active or active["command"]:
            return False
        active["interrupted"] = True
        async with runtime.session._dispatch_lock:
            await runtime.session._cancel_active_turn(reason="manual")
        # A cancellation can interrupt the runner's final persistence await
        # before it emits TurnComplete. Once cancellation drained (or fenced
        # a detached provider), release the collector from our saved replay;
        # waiting only for a terminal frame would hold the turn lock forever.
        if runtime.active is active and not active["task"].done():
            active["task"].cancel()
        return True

    async def _model(self, request, sid, arg, collector):
        from .api import sessions

        # Preserve the original verified certificate, not a fabricated owner.
        class PinRequest(dict):
            app = request.app
            match_info = {"session_id": sid}
            can_read_body = True

            async def json(self):
                return {"runtime_id": arg}

        pin_request = PinRequest(request)
        pin_request["_collaboration_command"] = True
        response = await (
            sessions.handle_unpin(pin_request)
            if arg.lower() in {"auto", "default", "none", "reset"}
            else sessions.handle_pin(pin_request)
        )
        result = json.loads(response.body)
        collector.errored = response.status != 200
        collector.text = (
            result.get("error", "Model change failed")
            if collector.errored
            else "Model: " + (result.get("runtime_id") or "Auto")
        )

    async def _run(
        self, request, sid, text, request_id, delivery, attachments=(), instance_id=None
    ):
        from src.core.on_behalf_context import (
            OnBehalfIdentity,
            install_on_behalf_identity,
            reset_on_behalf_identity,
        )
        from src.stream.channel import BatchedChannel

        await self._check(request, sid)
        runtime = await self._runtime(sid, request)
        command = (
            text.split(maxsplit=1)[0][1:].lower() if text.startswith("/") else None
        )
        stopped = False
        async with runtime.admission:
            if (delivery == "steer" and command is None) or command == "stop":
                # Reauthorize after waiting for another admission/cancellation.
                await self._check(request, sid)
                stopped = await self._interrupt(runtime)
        async with runtime.lock:
            access = await self._check(request, sid)
            if runtime.session._detached_turns:
                raise BusyError("The previous provider is still stopping")
            principal = OnBehalfIdentity.from_certificate(
                request.get("device_cert"),
                auth_kind=str(request.get("auth_kind") or ""),
            )
            author = {
                "kind": "agent" if principal.principal_type == "agent" else "human",
                "userId": access["userId"],
                "display": access["name"],
                "handle": principal.handle,
            }
            normalized_attachments = []
            if attachments:
                from src.memory.artifacts import normalize_inbound_attachments

                normalized_attachments = await normalize_inbound_attachments(
                    self.gateway.agent.memory_db,
                    attachments,
                    session_id=sid,
                    principal=principal,
                    allow_local_paths=False,
                )
                await self._check(request, sid)
            if not self.hub.begin(sid, text, author, request_id):
                raise BusyError("Live replay capacity reached")
            live = self.hub.sessions[sid]["turns"][-1]
            state = {
                "id": request_id,
                "task": asyncio.current_task(),
                "command": command is not None,
                "interrupted": False,
                "device_id": request.get("device_cert").device_pubkey_hex,
                "request": request,
            }
            runtime.active = state
            # Set only inside the turn lock, before push_in. The StreamSession
            # snapshots this immutable principal into each dispatched runner.
            runtime.session.on_behalf_identity = principal
            from src.core.execution_origin import (
                TrustedIngressIdentity,
                TrustedTurnContext,
            )

            ingress = TrustedIngressIdentity(
                device_id=request.get("client_id"),
                connection_id="shared:" + request_id,
                client_instance_id=instance_id,
                turn_context=TrustedTurnContext(
                    on_behalf_identity=principal,
                    client_kind="shared-chat",
                    client_capabilities=(
                        ("attachments", True),
                        ("ordered_parts", True),
                        ("inline_ui", True),
                        ("custom_ui_version", 1),
                    ),
                ),
            )
            registry = getattr(self.gateway, "capabilities", None)
            origin = (
                registry.origin_for(request.get("client_id"), instance_id)
                if registry
                else None
            )
            if attachments:
                live["messages"][0]["attachments"] = list(attachments)
            memory_db = getattr(self.gateway.agent, "memory_db", None)
            try:
                prior_runs = (
                    await memory_db.list_session_runs(sid, limit=1)
                    if not command and memory_db
                    else []
                )
            except Exception:
                prior_runs = []
            prior_run_id = prior_runs[0].get("run_id") if prior_runs else None
            state["prior_run_id"] = prior_run_id

            async def identify_run():
                # Resolve once per run without delaying the synchronous token tee.
                for _ in range(100):
                    await asyncio.sleep(0.1)
                    try:
                        runs = await memory_db.list_session_runs(sid, limit=1)
                        if runs and runs[0].get("run_id") != prior_run_id:
                            live["providerRunId"] = f"run:{sid}:{runs[0]['run_id']}"
                            self.hub.changed(self.hub.sessions[sid])
                            return
                    except Exception:
                        return

            projection = (
                asyncio.create_task(identify_run())
                if not command and memory_db
                else None
            )
            identity_token = install_on_behalf_identity(
                runtime.session.on_behalf_identity
            )
            try:
                if command:
                    collector = CommandReply(self.hub, sid)
                    arg = (
                        text.split(maxsplit=1)[1].strip()
                        if len(text.split(maxsplit=1)) > 1
                        else None
                    )
                    if command == "stop":
                        collector.text = (
                            "Stopped the active turn." if stopped else "No active turn."
                        )
                    elif command in {"queue", "status"}:
                        queued = sum(
                            key[0] == sid
                            and key[1] != request_id
                            and not entry[1].done()
                            for key, entry in self.requests.items()
                        )
                        collector.text = (
                            f"Queue depth: {queued}"
                            if command == "queue"
                            else f"Idle | Queue: {queued}"
                        )
                    elif command == "help":
                        collector.text = "Available session commands: " + ", ".join(
                            "/" + name for name in sorted(COMMANDS)
                        )
                    elif command == "model" and arg:
                        await self._model(request, sid, arg, collector)
                    elif command in COMMANDS:
                        await self.gateway._handle_command(
                            collector,
                            runtime.session.client_id,
                            command,
                            sid,
                            handle=request.get("user_handle"),
                            arg=arg,
                        )
                    else:
                        collector.errored = True
                        collector.text = "Available session commands: " + ", ".join(
                            "/" + name for name in sorted(COMMANDS)
                        )
                    await runtime.session._journal(
                        "command/result",
                        {
                            "command": text,
                            "author": author,
                            "text": collector.text,
                            "request_id": request_id,
                            "turn_id": live["id"],
                        },
                    )
                    self.hub.publish(
                        {"type": "response", "session_id": sid, "text": collector.text}
                    )
                    self.hub.publish({"type": "turn_complete", "session_id": sid})
                    return {
                        "response": collector.text,
                        "model": "",
                        "errored": collector.errored,
                        "request_id": request_id,
                        "session_id": sid,
                    }
                reply = await BatchedChannel(runtime.session).run_one_shot(
                    text,
                    author=author,
                    attachments=list(normalized_attachments),
                    execution_origin=origin,
                    ingress_identity=ingress,
                )
                return {
                    "response": reply.text,
                    "model": reply.model,
                    "errored": reply.errored,
                    "interrupted": state["interrupted"],
                    "request_id": request_id,
                    "session_id": sid,
                }
            except asyncio.CancelledError:
                if not state["interrupted"]:
                    raise
                partial = next(
                    (m["text"] for m in live["messages"] if m["role"] == "assistant"),
                    "",
                )
                return {
                    "response": partial,
                    "errored": False,
                    "interrupted": True,
                    "request_id": request_id,
                    "session_id": sid,
                }
            finally:
                reset_on_behalf_identity(identity_token)
                if projection:
                    projection.cancel()
                    await asyncio.gather(projection, return_exceptions=True)
                if not command and memory_db:
                    try:
                        runs = await memory_db.list_session_runs(sid, limit=1)
                        if runs and runs[0].get("run_id") != prior_run_id:
                            live["providerRunId"] = f"run:{sid}:{runs[0]['run_id']}"
                            self.hub.changed(self.hub.sessions[sid])
                    except Exception:
                        pass  # projection failures cannot retain the execution lock
                if live["active"]:
                    self.hub.publish({"type": "turn_complete", "session_id": sid})
                runtime.active = None
                self.hub.resource("session", "updated", sid)

    async def handle_chat(self, request):
        if self.closed:
            return web.json_response({"error": "The gateway is stopping"}, status=503)
        from src.memory.artifacts import (
            ArtifactError,
            ArtifactIntegrityError,
            ArtifactNotFound,
            AttachmentTooLarge,
        )

        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) - {
                "session_id",
                "message",
                "request_id",
                "delivery",
                "attachments",
                "client_instance_id",
            }:
                raise ValueError()
            sid, text, request_id = (
                body.get("session_id"),
                body.get("message"),
                body.get("request_id"),
            )
            if not all(
                isinstance(v, str) and IDENTIFIER.fullmatch(v)
                for v in (sid, request_id)
            ):
                raise ValueError()
            attachments = body.get("attachments") or []
            if not isinstance(attachments, list) or len(attachments) > 32:
                raise ValueError()
            # Only public content-addressed references cross the shared API.
            # Resolution and authorization remain in the native stream runner.
            from src.memory.artifacts import public_attachment_ref

            if any(
                not isinstance(a, dict)
                or not isinstance(a.get("artifact_id"), str)
                or len(json.dumps(a)) > 8192
                for a in attachments
            ):
                raise ValueError()
            attachments = [public_attachment_ref(a) for a in attachments]
            instance_id = body.get("client_instance_id")
            if instance_id is not None and (
                not isinstance(instance_id, str)
                or not IDENTIFIER.fullmatch(instance_id)
            ):
                raise ValueError()
            if (
                not isinstance(text, str)
                or (not text.strip() and not attachments)
                or len(text) > 131072
            ):
                raise ValueError()
            delivery = body.get("delivery", "queue")
            if delivery not in ("queue", "steer"):
                raise ValueError()
        except (ValueError, TypeError):
            return web.json_response(
                {
                    "error": "Expected session_id, request_id, message and queue/steer delivery"
                },
                status=400,
            )
        try:
            access = await self._check(request, sid)
            await self.prune()
        except Exception:
            return web.json_response({"error": "Session unavailable"}, status=403)
        key = (sid, request_id)
        fingerprint = (
            access["userId"],
            text,
            delivery,
            json.dumps(attachments, sort_keys=True),
            instance_id,
        )
        # No await between deduplication and task insertion.
        entry = self.requests.get(key)
        if entry and entry[0] != fingerprint:
            return web.json_response({"error": "Request identity conflict"}, status=409)
        if not entry:
            if len(self.requests) >= 256:
                return web.json_response(
                    {"error": "Too many pending turns"}, status=429
                )
            task = asyncio.create_task(
                self._run(
                    request,
                    sid,
                    text.strip() or "[Attachments]",
                    request_id,
                    delivery,
                    attachments,
                    instance_id,
                ),
                context=contextvars.Context(),
            )

            def completed(done):
                error = None if done.cancelled() else done.exception()
                current = self.requests.get(key)
                if current is None or current[1] is not done:
                    return
                if isinstance(error, BusyError):
                    self.requests.pop(
                        key, None
                    )  # rejected before admission; safe to retry
                else:
                    self.requests[key] = (current[0], done, time.monotonic())

            task.add_done_callback(completed)
            entry = (fingerprint, task, float("inf"))
            self.requests[key] = entry
            if self.reaper is None or self.reaper.done():
                self.reaper = asyncio.create_task(
                    self._reap(), context=contextvars.Context()
                )
        try:
            result = await asyncio.shield(entry[1])
            await self._check(request, sid)  # revocation also applies to cached results
            return web.json_response(result)
        except PermissionError:
            return web.json_response({"error": "Session unavailable"}, status=403)
        except BusyError as exc:
            return web.json_response({"error": str(exc)}, status=409)
        # Attachment refusals are client errors, not gateway faults. The
        # message never names the offending id or path: an opaque artifact id
        # is not a bearer token and must not become an existence oracle.
        except ArtifactNotFound:
            return web.json_response(
                {"error": "Attachment is not available"}, status=404
            )
        except ArtifactIntegrityError:
            return web.json_response(
                {"error": "Attachment bytes failed integrity checks"}, status=503
            )
        except AttachmentTooLarge:
            return web.json_response({"error": "Attachment is too large"}, status=413)
        except ArtifactError:
            return web.json_response({"error": "Attachment rejected"}, status=400)
        except asyncio.CancelledError:
            if entry[1].cancelled():
                return web.json_response(
                    {
                        "interrupted": True,
                        "response": "",
                        "request_id": request_id,
                        "session_id": sid,
                    }
                )
            raise
        except Exception:
            return web.json_response({"error": "Shared turn unavailable"}, status=500)

    async def handle_stop(self, request):
        try:
            body = await request.json()
            if not isinstance(body, dict) or set(body) != {"session_id", "request_id"}:
                raise ValueError()
            sid, request_id = body["session_id"], body["request_id"]
            if not all(
                isinstance(v, str) and IDENTIFIER.fullmatch(v)
                for v in (sid, request_id)
            ):
                raise ValueError()
        except (ValueError, TypeError):
            return web.json_response(
                {"error": "Expected session_id and request_id"}, status=400
            )
        try:
            await self._check(request, sid)
            runtime = self.runtimes.get(sid)
            entry = self.requests.get((sid, request_id))
            stopped = False
            if runtime and entry and not entry[1].done():
                async with runtime.admission:
                    await self._check(request, sid)
                    active = runtime.active
                    if active and active["id"] == request_id:
                        if active["command"]:
                            # Do not cancel a compaction halfway through a DB
                            # rewrite. It finishes under the session lock.
                            raise BusyError("A session command is completing")
                        stopped = await self._interrupt(runtime)
                    else:
                        entry[1].cancel()  # exact queued request, never its neighbour
                        stopped = True
            elif entry and not entry[1].done():
                entry[1].cancel()
                stopped = True
            return web.json_response({"stopped": stopped, "request_id": request_id})
        except BusyError as exc:
            return web.json_response({"error": str(exc)}, status=409)
        except Exception:
            return web.json_response({"error": "Session unavailable"}, status=403)


def service(gateway):
    current = getattr(gateway, "_collaboration", None)
    if current is None or current.closed:
        current = gateway._collaboration = Collaboration(gateway)
    return current


def guard_mutation(request, sid):
    """Prevent legacy mutation APIs from bypassing an active shared queue."""
    current = getattr(request.app["gateway"], "_collaboration", None)
    if (
        current is not None
        and current.owns(sid)
        and not request.get("_collaboration_command")
    ):
        return web.json_response(
            {
                "error": "Use shared session commands, or wait for the shared transport to detach"
            },
            status=409,
        )
    return None
