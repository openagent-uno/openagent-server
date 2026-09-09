"""Read-only, authorized multi-client observation of the shared agent.

Producers never await clients or authorization. Each observer coalesces changes
for 40 ms, reauthorizes the exact views, then sends a replaceable snapshot. A
reconnection has no cursor to lose and never starts/stops an agent turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import copy
import json
import re
import time
import uuid
from dataclasses import dataclass, field

from .collaboration_access import authorize

KINDS = {"session", "workflow", "scheduled_task", "event"}
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}\Z")


def target_key(target):
    if not isinstance(target, dict) or set(target) != {"kind", "id"}:
        raise ValueError("Invalid view")
    if (
        not isinstance(target["kind"], str)
        or target["kind"] not in KINDS
        or not isinstance(target["id"], str)
        or not IDENTIFIER.fullmatch(target["id"])
    ):
        raise ValueError("Invalid view")
    return target["kind"], target["id"]


@dataclass(eq=False)
class Observer:
    ws: object
    actor: object
    user: dict
    sessions: tuple = ()
    focus: dict | None = None
    requested_focus: dict | None = None
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    sent: dict = field(default_factory=dict)
    roster: str = ""
    resources: dict = field(default_factory=dict)
    generation: int = 0


class SharedAgentHub:
    def __init__(self):
        self.closed = False
        self.sessions = {}
        self.clients = set()
        self.revision = 0
        self.authorization_limit = asyncio.Semaphore(8)

    def wake(self):
        for client in self.clients:
            client.wake.set()

    def begin(self, sid, text, author=None, run_id=None):
        self.prune()
        if sid not in self.sessions and len(self.sessions) >= 128:
            return False
        state = self.sessions.setdefault(sid, {"turns": [], "revision": 0})
        if state["turns"] and state["turns"][-1]["active"]:
            state["turns"][-1]["active"] = False
            state["turns"][-1]["finishedAt"] = time.time()
        state["turns"] = state["turns"][-7:]
        state["turns"].append(
            {
                "id": uuid.uuid4().hex,
                "runId": run_id,
                "active": True,
                "startedAt": time.time(),
                "messages": (
                    [
                        {
                            "id": "input",
                            "role": "user",
                            "text": text,
                            "timestamp": time.time(),
                            "author": author or {"kind": "agent", "display": "Agent"},
                        }
                    ]
                    if text
                    else []
                ),
            }
        )
        if text and len(text) > 131072:
            state["turns"][-1]["messages"][0]["text"] = text[:131072]
            state["turns"][-1]["truncated"] = True
        self.changed(state)
        return True

    def changed(self, state):
        self.revision += 1
        state["revision"] = self.revision
        state["updatedAt"] = time.time()
        self.wake()

    def publish(self, frame):
        sid, kind = frame.get("session_id"), frame.get("type")
        if not sid or kind not in {
            "delta",
            "response",
            "status",
            "reasoning",
            "turn_complete",
            "error",
            "seed",
        }:
            return
        state = self.sessions.get(sid)
        if kind == "seed":
            self.begin(sid, frame.get("text", ""), frame.get("author"))
            return
        if state is None or not state["turns"][-1]["active"]:
            if kind in {"turn_complete", "response", "error", "reasoning"}:
                return
            self.begin(sid, "")
            state = self.sessions.get(sid)
        if state is None:
            return
        turn = state["turns"][-1]
        messages = turn["messages"]
        text = frame.get("text", "")
        if kind in {"delta", "response"}:
            message = next((m for m in messages if m["id"] == "response"), None)
            if message is None:
                message = {
                    "id": "response",
                    "role": "assistant",
                    "text": "",
                    "timestamp": time.time(),
                }
                messages.append(message)
            value = message["text"] + text if kind == "delta" else text
            message["text"] = value[:131072]
            turn["truncated"] = len(value) > 131072 or turn.get("truncated", False)
            if frame.get("model"):
                message["model"] = frame["model"]
            if kind == "response":
                for field in ("attachments", "parts"):
                    if frame.get(field) and len(json.dumps(frame[field])) <= 262144:
                        message[field] = frame[field]
        elif kind == "status":
            # Tool status is a JSON envelope in OpenAgent; plain statuses are
            # separate from transcript messages (no fabricated tool results).
            turn["status"] = text[:16384]
        elif kind == "reasoning":
            turn["reasoning"] = frame.get("active") is True
        elif kind == "error":
            turn["error"] = text[:16384]
        elif kind == "turn_complete":
            turn["active"] = False
            turn["finishedAt"] = time.time()
        self.changed(state)

    def resource(self, resource, action, resource_id=None):
        frame = {"type": "resource_event", "resource": resource, "action": action}
        if resource_id:
            frame["id"] = resource_id
        for client in self.clients:
            # Invalidation is bounded/coalesced; content is fetched separately.
            client.resources[(resource, resource_id)] = frame
            if len(client.resources) > 64:
                client.resources = {
                    (resource, None): {
                        "type": "resource_event",
                        "resource": resource,
                        "action": "updated",
                    }
                }
        self.wake()

    def prune(self):
        now = time.time()
        for sid, state in list(self.sessions.items()):
            if not state["turns"][-1]["active"] and now - state["updatedAt"] > 60:
                del self.sessions[sid]
                self.revision += 1

    async def deliver(self, client):
        self.prune()
        generation = client.generation
        targets = [{"kind": "session", "id": sid} for sid in client.sessions]
        for peer in list(self.clients):
            if peer.focus and peer.focus not in targets:
                targets.append(peer.focus)
        if client.requested_focus and client.requested_focus not in targets:
            targets.append(client.requested_focus)
        # Resource IDs can reveal private activity too; authorize invalidations.
        for frame in client.resources.values():
            if frame["resource"] in KINDS and frame.get("id"):
                target = {"kind": frame["resource"], "id": frame["id"]}
                if target not in targets and len(targets) < 64:
                    targets.append(target)
        async with self.authorization_limit:
            access = await authorize(client.actor, targets)
        if client.generation != generation:
            client.wake.set()
            return
        user = {"userId": access["userId"], "name": access["name"]}
        if user != client.user:
            client.user = user
            self.wake()
        allowed = {target_key(t) for t in access["allowed"]}
        next_focus = (
            client.requested_focus
            if client.requested_focus and target_key(client.requested_focus) in allowed
            else None
        )
        if client.focus != next_focus:
            client.focus = next_focus
            self.wake()
        roster = []
        seen = set()
        for peer in list(self.clients):
            if peer.focus and target_key(peer.focus) in allowed:
                key = (peer.user["userId"], *target_key(peer.focus))
                if key not in seen:
                    seen.add(key)
                    roster.append({**peer.user, "target": peer.focus})
        roster.sort(key=lambda p: (p["target"]["kind"], p["target"]["id"], p["userId"]))
        encoded = json.dumps(roster, sort_keys=True)
        if encoded != client.roster:
            client.roster = encoded
            await client.ws.send_json({"type": "shared_presence", "people": roster})
        for sid in client.sessions:
            # Each send below yields to socket I/O. A revocation that arrives
            # after the roster/previous session must win before another
            # session's content is serialized on this socket.
            async with self.authorization_limit:
                current_access = await authorize(
                    client.actor, [{"kind": "session", "id": sid}]
                )
            if client.generation != generation:
                client.wake.set()
                return
            if {"kind": "session", "id": sid} not in current_access["allowed"]:
                allowed.discard(("session", sid))
            else:
                allowed.add(("session", sid))
            if ("session", sid) not in allowed:
                await client.ws.send_json({"type": "shared_revoked", "session_id": sid})
                client.sent.pop(sid, None)
                continue
            state = self.sessions.get(sid, {"revision": self.revision, "turns": []})
            if client.sent.get(sid) != state["revision"]:
                # Copy synchronously after authorization. Producers can run
                # while socket I/O is suspended without mutating this payload.
                payload = copy.deepcopy(state)
                client.sent[sid] = state["revision"]
                await client.ws.send_json(
                    {"type": "shared_state", "session_id": sid, **payload}
                )
        resources, client.resources = client.resources, {}
        for frame in resources.values():
            if frame["resource"] in KINDS and frame.get("id"):
                target = {"kind": frame["resource"], "id": frame["id"]}
                async with self.authorization_limit:
                    current_access = await authorize(client.actor, [target])
                if target not in current_access["allowed"]:
                    continue
            if (
                frame["resource"] in KINDS
                and frame.get("id")
                and (frame["resource"], frame["id"]) not in allowed
            ):
                continue
            await client.ws.send_json(frame)

    async def pump(self, client):
        try:
            while True:
                try:
                    await asyncio.wait_for(client.wake.wait(), 10)
                except asyncio.TimeoutError:
                    pass
                await asyncio.sleep(0.04)
                client.wake.clear()
                await asyncio.wait_for(self.deliver(client), 10)
        except asyncio.CancelledError:
            raise
        except Exception:
            with contextlib.suppress(Exception):
                await client.ws.send_json(
                    {
                        "type": "auth_error",
                        "text": "Shared session access is unavailable. Reconnecting…",
                    }
                )
                await client.ws.close(code=4001)

    async def handle(self, request):
        from aiohttp import web, WSMsgType

        if self.closed:
            return web.Response(status=503)
        if request.get("device_cert") is None:
            return web.Response(status=403)
        try:
            actor = request
            access = await authorize(actor, [])
        except Exception:
            return web.Response(status=403)
        if len(self.clients) >= 32:
            return web.Response(status=503)
        ws = web.WebSocketResponse(heartbeat=10, max_msg_size=16384)
        await ws.prepare(request)
        client = Observer(
            ws, actor, {"userId": access["userId"], "name": access["name"]}
        )
        try:
            await authorize(actor, [])  # recheck after the upgrade
        except Exception:
            await ws.close(code=4003)
            return ws
        if self.closed or len(self.clients) >= 32:
            await ws.close(code=1013)
            return ws
        self.clients.add(client)
        await ws.send_json({"type": "auth_ok", "shared": True})
        task = asyncio.create_task(
            self.pump(client), context=__import__("contextvars").Context()
        )
        try:
            async for message in ws:
                if message.type != WSMsgType.TEXT:
                    break
                try:
                    frame = json.loads(message.data)
                    if not isinstance(frame, dict):
                        raise ValueError("Invalid observation")
                    if frame.get("type") == "ping":
                        client.wake.set()
                        continue
                    if frame.get("type") != "observe" or set(frame) - {
                        "type",
                        "sessions",
                        "focus",
                    }:
                        raise ValueError("This channel is read-only")
                    sessions = frame.get("sessions", [])
                    if not isinstance(sessions, list) or len(sessions) > 16:
                        raise ValueError("Too many sessions")
                    for sid in sessions:
                        target_key({"kind": "session", "id": sid})
                    focus = frame.get("focus")
                    if focus is not None:
                        target_key(focus)
                    client.sessions = tuple(dict.fromkeys(sessions))
                    client.sent = {
                        sid: rev
                        for sid, rev in client.sent.items()
                        if sid in client.sessions
                    }
                    client.requested_focus = focus
                    if client.focus != focus:
                        client.focus = None
                    client.generation += 1
                    self.wake()
                except (ValueError, TypeError):
                    await ws.close(code=1008, message=b"Invalid observation frame")
                    break
        finally:
            self.clients.discard(client)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            self.wake()
        return ws

    async def close(self):
        self.closed = True
        await asyncio.gather(
            *(client.ws.close(code=1001) for client in list(self.clients)),
            return_exceptions=True,
        )
        self.sessions.clear()
