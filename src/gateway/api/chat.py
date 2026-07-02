"""POST /api/chat — single-turn REST endpoint for iOS and external clients.

Wraps a :class:`~src.stream.session.StreamSession` +
:class:`~src.stream.channel.BatchedChannel` so any HTTP client
can send one user message and receive the full agent reply in a single
request/response cycle — no WebSocket required.

Auth: inherits the gateway's standard device-cert middleware for
Iroh connections. Plain HTTP callers (iOS app via Cloudflare Tunnel)
reach this endpoint under the same rules as every other /api/* route.

Concurrency: each (client_id, session_id) pair gets its own
``StreamSession`` kept alive between calls so the agent retains
conversation history. A per-session asyncio.Lock serialises concurrent
requests against the same session (the agent processes one turn at a
time anyway).
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

from aiohttp import web

logger = logging.getLogger(__name__)

# Max wall-clock for one /api/chat turn. Generous by default so long peer jobs
# (delegation, workflows, deep research) are not cut off — the old hardcoded
# 120s cap was the main cause of dropped long-job results over federation.
# Even when this fires, the session worker keeps running and persists the
# result to the (first-class) session, so the caller can resume by session_id.
TURN_TIMEOUT = float(os.environ.get("OPENAGENT_CHAT_TURN_TIMEOUT", "900"))

# ---------------------------------------------------------------------------
# In-process session cache
# ---------------------------------------------------------------------------
# Keyed by (client_id, session_id) → (StreamSession, asyncio.Lock)
_sessions: dict[tuple[str, str], tuple[Any, asyncio.Lock]] = {}
_sessions_registry_lock = asyncio.Lock()


async def _get_or_create_session(
    gateway,
    client_id: str,
    session_id: str,
    handle: str | None = None,
):
    """Return the cached StreamSession for this (client_id, session_id).

    Creates and starts a fresh one on first call. Serialised by
    ``_sessions_registry_lock`` so two concurrent first-calls don't
    double-start the same session. ``handle`` (the authenticated user
    identity) is recorded so per-message human authorship is captured.
    """
    from src.stream.session import StreamSession

    key = (client_id, session_id)
    async with _sessions_registry_lock:
        entry = _sessions.get(key)
        if entry is None:
            session = StreamSession(
                gateway.agent,
                client_id=client_id,
                session_id=session_id,
                profile="batched",
                speak_enabled=False,
                handle=handle,
            )
            # Wire gateway hooks so MCP resource broadcasts and model
            # guards work the same as for WebSocket sessions.
            session.pre_dispatch_hook = gateway._make_stream_pre_dispatch_hook(
                client_id, session_id
            )
            session.post_turn_hook = gateway._make_stream_post_turn_hook()
            await session.start()
            lock = asyncio.Lock()
            _sessions[key] = (session, lock)
            logger.debug("chat: created session %s/%s", client_id, session_id)
            return session, lock
        return entry


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

async def handle_chat(request: web.Request) -> web.Response:
    """POST /api/chat

    Request body (JSON)::

        {
            "message":    "Ciao Virgil, come sto?",   // required
            "session_id": "ios:device-uuid",           // optional, default "default"
            "client_id":  "ios-rest"                   // optional, overridden by cert
        }

    Response (JSON)::

        {
            "response": "...",   // agent reply text
            "model":    "...",   // model used
            "errored":  false
        }

    On error::

        {"error": "message required"}   HTTP 400
        {"error": "timeout"}            HTTP 504
        {"error": "..."}                HTTP 500
    """
    from src.stream.channel import BatchedChannel

    gateway = request.app["gateway"]

    # ── Parse body ──────────────────────────────────────────────────────────
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    text = (body.get("message") or "").strip()
    if not text:
        return web.json_response({"error": "message required"}, status=400)

    # client_id: prefer the device cert set by auth middleware, then
    # body field, then a generic fallback so iOS works without cert.
    client_id: str = (
        request.get("client_id")           # set by make_auth_middleware on Iroh
        or body.get("client_id", "")
        or "ios-rest"
    )
    user_handle = request.get("user_handle")

    # Peer detection: an agent-ALPN caller (another OpenAgent agent) carries a
    # synthetic DeviceCert with capabilities=["agent"] and a handle
    # "agent:<node16>". For those, a peer conversation must be a FIRST-CLASS
    # session on this agent — owned by the human, visible in the sidebar,
    # resumable by session_id — instead of colliding on the literal "default".
    device_cert = request.get("device_cert")
    is_peer = (
        getattr(device_cert, "capabilities", None) == ["agent"]
        or str(user_handle or "").startswith("agent:")
    )

    raw_sid = (body.get("session_id") or "").strip()
    created_new = False
    if is_peer:
        # peerShort derives from client_id, which the middleware sets to
        # sha256("agent:" + FULL node_id) — stable per peer and collision-safe.
        peer_short = (client_id or "")[:16] or "peer"
        prefix = f"peer:{peer_short}:"
        if not raw_sid:
            session_id = f"{prefix}{uuid.uuid4().hex}"      # new thread
            created_new = True
        elif raw_sid.startswith(prefix):
            session_id = raw_sid                            # resume this peer's thread
        else:
            session_id = f"{prefix}{raw_sid}"               # namespace-enforce (peer-scoped)
    else:
        session_id = raw_sid or "default"

    # ── Get or create StreamSession ──────────────────────────────────────────
    session, turn_lock = await _get_or_create_session(
        gateway, client_id, session_id,
        handle=user_handle,
    )

    # If a previous (long-running) turn is still in flight for this session, a
    # peer resume must NOT push a new message — that barges in and cancels the
    # running turn, destroying the very result the caller is trying to fetch.
    # Tell them to poll again; the worker finishes and persists it.
    if is_peer:
        cur = getattr(session, "_current_turn", None)
        if cur is not None and not cur.done():
            return web.json_response({
                "error": "a previous turn is still running for this session — "
                         "retry shortly to fetch its result",
                "session_id": session_id,
                "still_running": True,
            }, status=409)

    # A peer conversation is first-class: stamp the HUMAN owner + origin="chat"
    # (identical affordances to any chat) + the peer handle as ``kind``.
    # Metadata only — never touches ``user_id`` (owned by the runtime). Awaited
    # BEFORE the turn so the runtime reads and round-trips the stamp instead of
    # overwriting it.
    if is_peer:
        db = getattr(gateway.agent, "_db", None)
        if db is not None:
            try:
                owner = await db.primary_owner_handle()
                if owner:
                    await db.upsert_session(
                        session_id, client_id=owner,
                        origin="chat", kind=user_handle,
                    )
                    if created_new:
                        gateway.broadcast_session("created", session_id)
            except Exception:
                logger.warning(
                    "chat: peer session stamp failed for %s", session_id,
                    exc_info=True,
                )

    # ── Run one turn (serialised per session) ────────────────────────────────
    async with turn_lock:
        channel = BatchedChannel(session)
        try:
            reply = await asyncio.wait_for(
                channel.run_one_shot(text, source="user_typed"),
                timeout=TURN_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "chat: turn exceeded %.0fs for %s/%s — the worker keeps running "
                "and persists the result; caller can resume by session_id",
                TURN_TIMEOUT, client_id, session_id,
            )
            return web.json_response({
                "error": "timeout waiting for agent reply",
                "session_id": session_id,
                "still_running": True,
            }, status=504)
        except Exception:
            logger.exception("chat: unhandled error for session %s/%s", client_id, session_id)
            # Invalidate the session so next call gets a fresh one
            async with _sessions_registry_lock:
                _sessions.pop((client_id, session_id), None)
            try:
                await session.close()
            except Exception:
                pass
            return web.json_response({"error": "internal error"}, status=500)

    return web.json_response({
        "response":   reply.text or "",
        "model":      reply.model or "",
        "errored":    reply.errored,
        "session_id": session_id,
        "created":    created_new,
    })


# ---------------------------------------------------------------------------
# Health ingest (Apple HealthKit data from iOS app)
# ---------------------------------------------------------------------------

async def handle_health_ingest(request: web.Request) -> web.Response:
    """POST /api/health/ingest — receive HealthKit metrics from the iOS app.

    Stores the payload in the vault under ``health/ios-latest.json`` and
    appends a one-line entry to ``health/ios-log.md`` so the agent can
    read health data via the vault MCP.

    Request body (JSON)::

        {
            "source":    "ios-healthkit",
            "timestamp": "2026-05-21T...",
            "metrics": {
                "steps_7d_avg":       8432,
                "hrv_ms":             48.2,
                "resting_hr_bpm":     58,
                "weight_kg":          78.5,
                "active_kcal_7d_avg": 620,
                "spo2_pct":           98
            }
        }
    """
    import json
    from pathlib import Path
    from datetime import datetime, timezone

    gateway = request.app["gateway"]

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)

    # Write to vault
    try:
        vault_path: Path | None = getattr(gateway, "_gateway_vault_path", None)
        if vault_path:
            vault_path = Path(vault_path)
            health_dir = vault_path / "health"
            health_dir.mkdir(parents=True, exist_ok=True)

            # Latest snapshot (overwrite)
            (health_dir / "ios-latest.json").write_text(
                json.dumps(body, indent=2, ensure_ascii=False)
            )

            # Append to log
            now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            metrics = body.get("metrics", {})
            line = (
                f"- {now}: "
                f"steps={metrics.get('steps_7d_avg', '?')} "
                f"hrv={metrics.get('hrv_ms', '?')}ms "
                f"rhr={metrics.get('resting_hr_bpm', '?')}bpm "
                f"weight={metrics.get('weight_kg', '?')}kg "
                f"kcal={metrics.get('active_kcal_7d_avg', '?')} "
                f"spo2={metrics.get('spo2_pct', '?')}%\n"
            )
            log_file = health_dir / "ios-log.md"
            with log_file.open("a") as f:
                f.write(line)

            logger.info("health/ingest: stored metrics from %s", body.get("source", "unknown"))
        else:
            logger.warning("health/ingest: no vault path configured, skipping write")

    except Exception:
        logger.exception("health/ingest: failed to write to vault")
        return web.json_response({"error": "failed to store health data"}, status=500)

    return web.json_response({"ok": True})
