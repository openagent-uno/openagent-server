"""Webhook Events channel — DB, secret handling, webhook auth, isolation,
dispatch, and resource-event surfacing.

The Events channel adds an inbound doorway: an external service (or a peer)
calls a per-event webhook and the agent runs a bound action (a workflow, a
scheduled task, or a chat prompt). These tests cover the security-critical
surface (the secret is never stored in clear; the webhook listener never
exposes /api/*; bad secrets / bad signatures / oversized / duplicate requests
are all rejected) and the three dispatch paths.
"""
from __future__ import annotations

import hashlib
import hmac
import json

from ._framework import TestContext, test, free_port


# ── DB round-trip + secret hygiene ────────────────────────────────────


@test("events", "add_event round-trips; the secret is never stored in clear")
async def t_event_db_roundtrip(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.core.event_secret import make_secret_material, slugify, decrypt_secret

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        clear, enc, hint = make_secret_material(db_path=str(ctx.db_path))
        # The stored (encrypted) form must NOT contain the clear secret.
        assert clear not in enc, "clear secret leaked into the ciphertext"
        eid = await db.add_event(
            name="GitHub Push", action_kind="prompt", slug=slugify("GitHub Push"),
            secret_enc=enc, secret_hint=hint, event_type="github",
            prompt_template="Push by {{payload.pusher.name}}",
            input_schema=[{"name": "pusher", "path": "pusher.name"}],
            session_binding_enabled=True,
            session_binding_path="repository.id",
        )
        # A public read never returns the encrypted secret.
        pub = await db.get_event(eid)
        assert "secret_enc" not in pub, pub
        assert pub["secret_hint"] == hint
        assert pub["input_schema"][0]["name"] == "pusher"
        assert pub["enabled"] is True
        assert pub["session_binding_enabled"] is True
        assert pub["session_binding_path"] == "repository.id"
        # The private read (webhook-auth path) does — and it decrypts back.
        priv = await db.get_event(eid, include_secret=True)
        assert decrypt_secret(priv["secret_enc"], db_path=str(ctx.db_path)) == clear
        # slug is unique + queryable.
        assert (await db.get_event_by_slug("github-push"))["id"] == eid
        assert await db.slug_exists("github-push")
    finally:
        await db.close()


@test("events", "delivery de-dupe + rate-limit backstop counters")
async def t_event_deliveries_db(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.core.event_secret import make_secret_material

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        clear, enc, hint = make_secret_material(db_path=str(ctx.db_path))
        eid = await db.add_event(
            name="dedup", action_kind="prompt", slug="dedup",
            secret_enc=enc, secret_hint=hint, prompt_template="x",
        )
        did = await db.add_event_delivery(event_id=eid, external_id="gh-1", payload={"a": 1})
        # Same provider delivery id → found (caller returns 200 duplicate).
        dup = await db.find_delivery_by_external_id(eid, "gh-1")
        assert dup and dup["id"] == did
        assert await db.find_delivery_by_external_id(eid, "gh-2") is None
        # Recent-delivery counter powers the rate-limit backstop.
        assert await db.count_recent_deliveries(eid) == 1
        # Linking + finalising.
        await db.update_event_delivery(did, status="success", session_id="event:%s:%s" % (eid, did))
        assert (await db.list_event_deliveries(eid))[0]["status"] == "success"
    finally:
        await db.close()


# ── Webhook auth (pure functions) ─────────────────────────────────────


@test("events", "github HMAC verifies; a tampered signature is rejected")
async def t_webhook_auth_github(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.core.event_secret import make_secret_material
    from src.gateway.webhook_auth import authenticate, WebhookAuthError, extract_external_id

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        clear, enc, hint = make_secret_material(db_path=str(ctx.db_path))
        eid = await db.add_event(
            name="gh", action_kind="prompt", slug="gh",
            secret_enc=enc, secret_hint=hint, event_type="github", prompt_template="x",
        )
        ev = await db.get_event(eid, include_secret=True)
        body = b'{"zen":"Keep it simple."}'
        good = "sha256=" + hmac.new(clear.encode(), body, hashlib.sha256).hexdigest()
        # Valid signature + delivery id passes and yields the de-dupe id.
        authenticate(event=ev, raw_body=body,
                     headers={"X-Hub-Signature-256": good, "X-GitHub-Delivery": "d1"},
                     db_path=str(ctx.db_path))
        assert extract_external_id(event=ev, headers={"X-GitHub-Delivery": "d1"}, payload={}) == "d1"
        # Tampered signature is rejected.
        try:
            authenticate(event=ev, raw_body=body,
                         headers={"X-Hub-Signature-256": "sha256=deadbeef"},
                         db_path=str(ctx.db_path))
            raise AssertionError("bad signature should have been rejected")
        except WebhookAuthError:
            pass
        # A body swap under the same signature is rejected (integrity).
        try:
            authenticate(event=ev, raw_body=b'{"zen":"tampered"}',
                         headers={"X-Hub-Signature-256": good},
                         db_path=str(ctx.db_path))
            raise AssertionError("tampered body should have been rejected")
        except WebhookAuthError:
            pass
    finally:
        await db.close()


@test("events", "generic type verifies the bearer secret")
async def t_webhook_auth_generic(ctx: TestContext) -> None:
    from src.memory.db import MemoryDB
    from src.core.event_secret import make_secret_material
    from src.gateway.webhook_auth import authenticate, WebhookAuthError

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        clear, enc, hint = make_secret_material(db_path=str(ctx.db_path))
        eid = await db.add_event(
            name="g", action_kind="prompt", slug="g",
            secret_enc=enc, secret_hint=hint, event_type="generic", prompt_template="x",
        )
        ev = await db.get_event(eid, include_secret=True)
        authenticate(event=ev, raw_body=b"{}",
                     headers={"X-OpenAgent-Event-Secret": clear}, db_path=str(ctx.db_path))
        authenticate(event=ev, raw_body=b"{}",
                     headers={"Authorization": f"Bearer {clear}"}, db_path=str(ctx.db_path))
        for bad in ({}, {"X-OpenAgent-Event-Secret": "nope"}):
            try:
                authenticate(event=ev, raw_body=b"{}", headers=bad, db_path=str(ctx.db_path))
                raise AssertionError(f"should reject {bad!r}")
            except WebhookAuthError:
                pass
    finally:
        await db.close()


# ── Listener isolation + HTTP status codes ────────────────────────────


@test("events", "the webhook listener serves /hooks but NEVER /api/*")
async def t_webhook_listener_isolation(ctx: TestContext) -> None:
    import asyncio
    import urllib.request
    import urllib.error
    from src.memory.db import MemoryDB
    from src.core.event_secret import make_secret_material, slugify
    from src.gateway.webhook_site import WebhookSite

    class _Agent:
        def __init__(self, db):
            self.memory_db = db
            self.name = "test"

    class _Gateway:
        def __init__(self, db):
            self.agent = _Agent(db)
            self._scheduler = None

        def broadcast_resource_sync(self, *a, **k):
            pass

    def _get(url):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    def _post(url, data, headers):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    clear, enc, hint = make_secret_material(db_path=str(ctx.db_path))
    eid = await db.add_event(
        name="Hook", action_kind="workflow", slug=slugify("Hook iso"),
        secret_enc=enc, secret_hint=hint, event_type="generic", action_ref="missing-wf",
    )
    slug = (await db.get_event(eid))["slug"]
    site = WebhookSite(_Gateway(db))
    port = free_port()
    await site.start({"host": "127.0.0.1", "port": port, "public_url": None})
    base = f"http://127.0.0.1:{port}"
    try:
        assert await asyncio.to_thread(_get, base + "/hooks/health") == 200
        # The gateway API is structurally absent from this port.
        assert await asyncio.to_thread(_get, base + "/api/config") == 404
        assert await asyncio.to_thread(_get, base + "/api/events") == 404
        # Auth + status codes.
        assert await asyncio.to_thread(
            _post, f"{base}/hooks/{slug}", b'{"x":1}',
            {"X-OpenAgent-Event-Secret": clear}) == 202
        assert await asyncio.to_thread(
            _post, f"{base}/hooks/{slug}", b"{}",
            {"X-OpenAgent-Event-Secret": "wrong"}) == 401
        assert await asyncio.to_thread(
            _post, f"{base}/hooks/does-not-exist", b"{}",
            {"X-OpenAgent-Event-Secret": clear}) == 404
        # Oversized: tighten the cap and resend.
        await db.update_event(eid, max_payload_bytes=4)
        assert await asyncio.to_thread(
            _post, f"{base}/hooks/{slug}", b'{"aaaaaaaaaa":"bbbbbbbbbb"}',
            {"X-OpenAgent-Event-Secret": clear}) == 413
    finally:
        await site.stop()
        await db.close()


@test("events", "a disabled event's webhook returns 404 (no enumeration)")
async def t_webhook_disabled_404(ctx: TestContext) -> None:
    import asyncio
    import urllib.request
    import urllib.error
    from src.memory.db import MemoryDB
    from src.core.event_secret import make_secret_material
    from src.gateway.webhook_site import WebhookSite

    class _Agent:
        def __init__(self, db):
            self.memory_db = db
            self.name = "t"

    class _Gateway:
        def __init__(self, db):
            self.agent = _Agent(db)
            self._scheduler = None

        def broadcast_resource_sync(self, *a, **k):
            pass

    def _post(url, data, headers):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    clear, enc, hint = make_secret_material(db_path=str(ctx.db_path))
    eid = await db.add_event(
        name="off", action_kind="prompt", slug="off-event",
        secret_enc=enc, secret_hint=hint, prompt_template="x", enabled=False,
    )
    site = WebhookSite(_Gateway(db))
    port = free_port()
    await site.start({"host": "127.0.0.1", "port": port, "public_url": None})
    try:
        code = await asyncio.to_thread(
            _post, f"http://127.0.0.1:{port}/hooks/off-event", b"{}",
            {"X-OpenAgent-Event-Secret": clear})
        assert code == 404, code
    finally:
        await site.stop()
        await db.close()


# ── Dispatch: the three action kinds ──────────────────────────────────


class _SpyAgent:
    name = "spy"
    model = None

    def __init__(self):
        self.prompts: list[str] = []

    async def refresh_registries(self):
        return None

    async def run(self, *, message, user_id, session_id, model_override=None,
                  author=None, on_status=None):
        self.prompts.append(message)
        return "done"

    async def release_session(self, session_id, *, model_override=None):
        return None


@test("events", "prompt action renders the payload template into a visible child session")
async def t_dispatch_prompt(ctx: TestContext) -> None:
    import os
    from src.memory.db import MemoryDB
    from src.core.scheduler import Scheduler
    from src.core.event_dispatcher import dispatch_event
    from src.core.event_secret import make_secret_material
    from src.core.child_session import HIDDEN_CHILD_ORIGINS

    os.environ.pop("OPENAGENT_SCHEDULER_DURABLE_SESSIONS", None)
    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        agent = _SpyAgent()
        scheduler = Scheduler(db=db, agent=agent)  # type: ignore[arg-type]
        clear, enc, hint = make_secret_material(db_path=str(ctx.db_path))
        eid = await db.add_event(
            name="Notify", action_kind="prompt", slug="notify",
            secret_enc=enc, secret_hint=hint,
            prompt_template="Handle push by {{payload.pusher.name}}",
        )
        ev = await db.get_event(eid)
        did = await db.add_event_delivery(event_id=eid, payload={"pusher": {"name": "ale"}})
        result = await dispatch_event(
            agent=agent, db=db, scheduler=scheduler, event=ev,
            payload={"pusher": {"name": "ale"}}, delivery_id=did, source="webhook",
        )
        assert result["status"] == "success", result
        # The rendered payload reached the agent (injection-guarded but present).
        assert any("ale" in p for p in agent.prompts), agent.prompts
        assert any("untrusted" in p.lower() for p in agent.prompts), "missing injection guard"
        # The delivery links the produced child session.
        row = await db.get_event_delivery(did)
        assert row["status"] == "success" and row["session_id"], row
        # The child session is event-origin and therefore hidden as a standalone
        # row (surfaced via the delivery, not double-listed under chats).
        assert "event" in HIDDEN_CHILD_ORIGINS
    finally:
        await db.close()


@test("events", "prompt action can bind a payload id to one internal event run session")
async def t_dispatch_prompt_session_binding(ctx: TestContext) -> None:
    import os
    from src.memory.db import MemoryDB
    from src.core.scheduler import Scheduler
    from src.core.event_dispatcher import dispatch_event
    from src.core.event_secret import make_secret_material

    os.environ.pop("OPENAGENT_SCHEDULER_DURABLE_SESSIONS", None)
    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        agent = _SpyAgent()
        scheduler = Scheduler(db=db, agent=agent)  # type: ignore[arg-type]
        clear, enc, hint = make_secret_material(db_path=str(ctx.db_path))
        eid = await db.add_event(
            name="Ticket", action_kind="prompt", slug="ticket-bound",
            secret_enc=enc, secret_hint=hint,
            prompt_template="Handle ticket {{payload.ticket.id}}",
            session_binding_enabled=True,
            session_binding_path="ticket.id",
        )
        ev = await db.get_event(eid)

        async def fire(payload: dict) -> str:
            did = await db.add_event_delivery(event_id=eid, payload=payload)
            result = await dispatch_event(
                agent=agent, db=db, scheduler=scheduler, event=ev,
                payload=payload, delivery_id=did, source="webhook",
            )
            row = await db.get_event_delivery(did)
            assert row["session_id"] == result["session_id"], row
            return result["session_id"]

        sid1 = await fire({"ticket": {"id": "T-1", "body": "first"}})
        sid2 = await fire({"ticket": {"id": "T-1", "body": "follow-up"}})
        sid3 = await fire({"ticket": {"id": "T-2", "body": "other"}})
        assert sid1 == sid2, (sid1, sid2)
        assert sid3 != sid1, (sid1, sid3)
        binding = await db.get_event_session_binding(eid, "T-1")
        assert binding and binding["session_id"] == sid1, binding

        # Binding disabled keeps the old one-delivery/one-session behaviour,
        # even if the same payload id appears repeatedly.
        eid2 = await db.add_event(
            name="Ticket unbound", action_kind="prompt", slug="ticket-unbound",
            secret_enc=enc, secret_hint=hint,
            prompt_template="Handle {{payload.ticket.id}}",
            session_binding_enabled=False,
            session_binding_path="ticket.id",
        )
        ev2 = await db.get_event(eid2)

        async def fire_unbound() -> str:
            payload = {"ticket": {"id": "T-1"}}
            did = await db.add_event_delivery(event_id=eid2, payload=payload)
            result = await dispatch_event(
                agent=agent, db=db, scheduler=scheduler, event=ev2,
                payload=payload, delivery_id=did, source="webhook",
            )
            return result["session_id"]

        sid4 = await fire_unbound()
        sid5 = await fire_unbound()
        assert sid4 != sid5, (sid4, sid5)
    finally:
        await db.close()


@test("events", "scheduled-task action appends the payload as injection-guarded context")
async def t_dispatch_task_context(ctx: TestContext) -> None:
    import os
    from src.memory.db import MemoryDB
    from src.core.scheduler import Scheduler
    from src.core.event_dispatcher import dispatch_event
    from src.core.event_secret import make_secret_material

    os.environ.pop("OPENAGENT_SCHEDULER_DURABLE_SESSIONS", None)
    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        agent = _SpyAgent()
        scheduler = Scheduler(db=db, agent=agent)  # type: ignore[arg-type]
        tid = await db.add_task("responder", "0 9 * * *", "Respond to the ticket")
        clear, enc, hint = make_secret_material(db_path=str(ctx.db_path))
        eid = await db.add_event(
            name="Ticket", action_kind="scheduled_task", slug="ticket",
            secret_enc=enc, secret_hint=hint, action_ref=tid,
        )
        ev = await db.get_event(eid)
        did = await db.add_event_delivery(event_id=eid, payload={"subject": "urgent"})
        result = await dispatch_event(
            agent=agent, db=db, scheduler=scheduler, event=ev,
            payload={"subject": "urgent"}, delivery_id=did, source="webhook",
        )
        # The task fired with the base prompt PLUS the payload block.
        assert agent.prompts, "task did not fire"
        fired = agent.prompts[-1]
        assert "Respond to the ticket" in fired
        assert "Event payload" in fired and "urgent" in fired, fired
        row = await db.get_event_delivery(did)
        assert row["task_run_id"], row
    finally:
        await db.close()


@test("events", "run_task without context is unchanged (back-compat)")
async def t_run_task_no_context(ctx: TestContext) -> None:
    import os
    from src.memory.db import MemoryDB
    from src.core.scheduler import Scheduler

    os.environ.pop("OPENAGENT_SCHEDULER_DURABLE_SESSIONS", None)
    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        agent = _SpyAgent()
        scheduler = Scheduler(db=db, agent=agent)  # type: ignore[arg-type]
        task = {"id": "t1", "name": "Plain", "prompt": "just the prompt"}
        await scheduler.run_task(task)
        assert agent.prompts == ["just the prompt"], agent.prompts
    finally:
        await db.close()


# ── Resource-event surfacing ──────────────────────────────────────────


@test("events", "dispatch emits an 'event' resource event for the live UI")
async def t_dispatch_emits_resource_event(ctx: TestContext) -> None:
    import os
    from src.memory.db import MemoryDB
    from src.core.scheduler import Scheduler
    from src.core.event_dispatcher import dispatch_event
    from src.core.event_secret import make_secret_material
    from src.stream.resource_events import set_resource_event_sink

    os.environ.pop("OPENAGENT_SCHEDULER_DURABLE_SESSIONS", None)
    seen: list[tuple] = []
    set_resource_event_sink(lambda resource, action, id=None: seen.append((resource, action, id)))
    db = MemoryDB(str(ctx.db_path))
    await db.connect()
    try:
        agent = _SpyAgent()
        scheduler = Scheduler(db=db, agent=agent)  # type: ignore[arg-type]
        clear, enc, hint = make_secret_material(db_path=str(ctx.db_path))
        eid = await db.add_event(
            name="R", action_kind="prompt", slug="r-event",
            secret_enc=enc, secret_hint=hint, prompt_template="hi",
        )
        ev = await db.get_event(eid)
        did = await db.add_event_delivery(event_id=eid, payload={})
        await dispatch_event(agent=agent, db=db, scheduler=scheduler, event=ev,
                             payload={}, delivery_id=did, source="manual")
        assert any(r == "event" for (r, _a, _i) in seen), seen
    finally:
        set_resource_event_sink(None)
        await db.close()
