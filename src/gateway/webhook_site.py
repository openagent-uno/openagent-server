"""The dedicated webhook listener — the external-facing doorway of the
Events channel.

This is a SEPARATE ``aiohttp.web.Application`` on its own TCP port, serving
ONLY ``/hooks/{slug}`` (+ a bodyless ``/hooks/health`` probe). It shares NO
routes and NO middleware with the gateway application, so the gateway's
``/api/*`` surface is structurally unreachable from this port even if a future
edit slips up — there is simply nothing else mounted here.

Auth is per-event (a secret, plus an optional provider HMAC), enforced inside
the handler via ``webhook_auth`` — NOT the gateway's device-cert middleware,
which an external SaaS can't satisfy.

Config (``channels.webhook`` in openagent.yaml):

    channels:
      webhook:
        enabled: true
        host: 0.0.0.0        # bind address (default 0.0.0.0)
        port: 8899           # TCP port
        public_url: https://hooks.example.com   # shown in the UI, optional

Disabled by default: no ``enabled: true`` → no socket is opened.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from typing import Any, Optional

from aiohttp import web

from src.core.logging import elog
from src.gateway.webhook_auth import (
    WebhookAuthError, authenticate, extract_external_id,
)

logger = logging.getLogger(__name__)

# Absolute ceiling on a request body, regardless of the per-event cap — a first
# line of defense before we even look the event up. Per-event
# ``max_payload_bytes`` (default 256 KB) refines this downward.
ABSOLUTE_MAX_BODY = 5 * 1024 * 1024


class _RateLimiter:
    """Tiny per-event sliding-window limiter kept in memory. Backstopped by the
    DB (``count_recent_deliveries``) so a restart / multi-process deploy still
    bounds abuse; this just avoids a DB round-trip on the hot path."""

    def __init__(self) -> None:
        self._hits: dict[str, deque] = {}

    def allow(self, event_id: str, limit_per_min: int) -> bool:
        if limit_per_min <= 0:
            return True
        now = time.monotonic()
        dq = self._hits.setdefault(event_id, deque())
        cutoff = now - 60.0
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= limit_per_min:
            return False
        dq.append(now)
        return True


class WebhookSite:
    """Owns the dedicated webhook aiohttp app + its TCP listener."""

    def __init__(self, gateway: Any) -> None:
        self._gw = gateway
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._rate = _RateLimiter()

    @staticmethod
    def config_from(config: dict | None) -> dict | None:
        """Extract + normalise the ``channels.webhook`` block, or None when the
        channel is disabled / absent."""
        wh = ((config or {}).get("channels") or {}).get("webhook") or {}
        if not wh or not wh.get("enabled"):
            return None
        try:
            port = int(wh.get("port", 8899))
        except (TypeError, ValueError):
            port = 8899
        return {
            "host": (wh.get("host") or "0.0.0.0"),
            "port": port,
            "public_url": (wh.get("public_url") or "").strip() or None,
        }

    async def start(self, cfg: dict) -> None:
        app = web.Application(client_max_size=ABSOLUTE_MAX_BODY)
        app.router.add_post("/hooks/{slug}", self._handle_hook)
        app.router.add_get("/hooks/health", self._handle_health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, cfg["host"], cfg["port"])
        await site.start()
        self._runner = runner
        self._site = site
        elog("gateway.webhook_listener.start", host=cfg["host"], port=cfg["port"])

    async def stop(self) -> None:
        if self._site is not None:
            try:
                await self._site.stop()
            except Exception as e:  # noqa: BLE001
                logger.debug("webhook site stop failed: %s", e)
            self._site = None
        if self._runner is not None:
            try:
                await self._runner.cleanup()
            except Exception as e:  # noqa: BLE001
                logger.debug("webhook runner cleanup failed: %s", e)
            self._runner = None

    # ── handlers ──

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def _handle_hook(self, request: web.Request) -> web.Response:
        gw = self._gw
        db = getattr(gw.agent, "memory_db", None)
        if db is None:
            return web.json_response({"error": "unavailable"}, status=503)

        slug = request.match_info["slug"]
        # Fetch WITH the encrypted secret so auth can verify. A missing /
        # disabled event returns an identical 404 — no enumeration.
        event = await db.get_event_by_slug(slug, include_secret=True)
        if event is None or not event.get("enabled"):
            elog("event.received", slug=slug, result="not_found")
            return web.json_response({"error": "not found"}, status=404)

        event_id = event["id"]

        # Read the raw body up to the per-event cap. We need the RAW bytes for
        # HMAC verification, so read before parsing.
        max_bytes = int(event.get("max_payload_bytes") or 262144)
        raw = await request.read()
        if len(raw) > max_bytes:
            elog("event.received", id=event_id, result="too_large", size=len(raw))
            return web.json_response({"error": "payload too large"}, status=413)

        headers = dict(request.headers)

        # Auth (secret + optional provider HMAC). Constant-time; identical 401
        # on any failure so a probe can't tell secret-vs-signature.
        try:
            authenticate(
                event=event, raw_body=raw, headers=headers,
                db_path=getattr(db, "db_path", None),
            )
        except WebhookAuthError as e:
            elog("event.auth_fail", id=event_id, reason=e.reason)
            return web.json_response({"error": "unauthorized"}, status=401)

        # Parse JSON (fallback to a wrapper for non-JSON bodies so a form/text
        # sender still delivers something templatable).
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(payload, dict):
                payload = {"_body": payload}
        except (ValueError, UnicodeDecodeError):
            payload = {"_raw": raw.decode("utf-8", "replace")[:max_bytes]}

        # Rate limit (in-memory + DB backstop).
        limit = int(event.get("rate_limit_per_min") or 60)
        if not self._rate.allow(event_id, limit):
            elog("event.rate_limited", id=event_id)
            return web.json_response({"error": "rate limited"}, status=429)
        if limit > 0 and await db.count_recent_deliveries(event_id) >= limit:
            elog("event.rate_limited", id=event_id, backstop=True)
            return web.json_response({"error": "rate limited"}, status=429)

        # Idempotent de-dupe on the provider delivery id.
        external_id = extract_external_id(event=event, headers=headers, payload=payload)
        if external_id:
            existing = await db.find_delivery_by_external_id(event_id, external_id)
            if existing is not None:
                elog("event.duplicate", id=event_id, external_id=external_id)
                return web.json_response(
                    {"duplicate": True, "delivery_id": existing["id"]}, status=200,
                )

        # Record the delivery (already claimed — we dispatch it in-process) and
        # respond fast; the LLM run happens in the background.
        #
        # AT-LEAST-ONCE CONTRACT: the row is durable BEFORE we return 202, and
        # the background turn is best-effort within this process. If the
        # process dies after this insert but before the turn finishes, the row
        # is left ``received``/``running`` with ``claimed_at`` set — an orphan.
        # ``MemoryDB.reap_orphan_event_deliveries`` re-enqueues such orphans on
        # the next startup (``claimed_at=NULL``, status back to ``received``) so
        # the Scheduler's drain re-dispatches them, instead of dropping them as
        # ``failed``. Do NOT "optimise" this into a fire-and-forget with no DB
        # row, and do NOT assume the in-process dispatch is the only run: replay
        # resumes the SAME child session (bound thread session, or the
        # deterministic ``event:{event_id}:{delivery_id}`` id), so the customer
        # reply stays idempotent through Replio's reply_guard.
        delivery_id = await db.add_event_delivery(
            event_id=event_id, source="webhook",
            external_id=external_id, payload=payload, claimed=True,
        )
        elog("event.received", id=event_id, result="accepted", delivery=delivery_id)

        # Per-event circuit breaker: if this event's breaker is OPEN, park the
        # delivery durably as ``blocked`` (visible in history) and do NOT run the
        # turn. Inert by default — ``is_event_breaker_tripped`` returns False
        # unless OPENAGENT_EVENT_BREAKER_ENABLED is set — so this is a no-op on
        # today's behaviour. The row is still recorded above so nothing is lost.
        breaker_tripped = False
        try:
            breaker_tripped = await db.is_event_breaker_tripped(event_id)
        except Exception:  # noqa: BLE001
            breaker_tripped = False
        if breaker_tripped:
            await db.update_event_delivery(
                delivery_id, status="blocked",
                error="event circuit breaker open", finished_at=time.time(),
            )
            elog("event.blocked", id=event_id, delivery=delivery_id)
            return web.json_response(
                {"delivery_id": delivery_id, "status": "blocked"}, status=202,
            )

        scheduler = getattr(gw, "_scheduler", None)
        from src.core.event_dispatcher import dispatch_event
        coro = dispatch_event(
            agent=gw.agent, db=db, scheduler=scheduler,
            event=await db.get_event(event_id),  # public dict (no secret) for dispatch
            payload=payload, delivery_id=delivery_id, source="webhook",
            broadcast=gw.broadcast_resource_sync,
        )
        if scheduler is not None:
            scheduler._spawn_workflow(coro)
        else:
            # No scheduler (degraded) — run detached so we still respond fast.
            import asyncio
            asyncio.get_event_loop().create_task(coro)

        return web.json_response({"delivery_id": delivery_id, "status": "accepted"}, status=202)
