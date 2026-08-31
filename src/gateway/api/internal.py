"""POST /api/internal/skill-data — write-back endpoint for central skill runners.

Central skills (runtime: central) run as shared multi-tenant pods in
virgil-system. After each ingest tick they push results here so each
user's agent has data in their local SQLite — no direct DB mount needed.

Auth: ``X-OpenAgent-Token`` header matching ``OPENAGENT_HTTP_TOKEN``
(handled by the gateway's ``make_auth_middleware`` before this handler
is reached; requests without a valid token get a 401 from middleware).

Payload (JSON)::

    {
        "skill_id": "telegram-ingester",   // required
        "events":   [...],                 // optional list of event dicts
        "entities": [...]                  // optional list of entity dicts
    }

Event dict fields::

    id, source, external_id, entity_id, timestamp,
    direction, channel, content, metadata_json

Entity dict fields::

    id, skill_id (overridden by body skill_id), type,
    display_name, identifiers_json, metadata_json

Both lists are upserted — idempotent on (skill_id, external_id) for
events and on ``id`` for entities.
"""

from __future__ import annotations

import logging
import time

import aiosqlite
from aiohttp import web

logger = logging.getLogger(__name__)


async def handle_skill_data(request: web.Request) -> web.Response:
    """POST /api/internal/skill-data"""
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    skill_id: str = (body.get("skill_id") or "").strip()
    if not skill_id:
        return web.json_response({"error": "skill_id required"}, status=400)

    events: list[dict] = body.get("events") or []
    entities: list[dict] = body.get("entities") or []

    if not events and not entities:
        return web.json_response({"ok": True, "inserted_events": 0, "inserted_entities": 0})

    # Reach into the agent's MemoryDB for the raw aiosqlite connection.
    gateway = request.app["gateway"]
    mem_db = gateway.agent.memory_db
    conn: aiosqlite.Connection | None = getattr(mem_db, "_conn", None)
    if conn is None:
        logger.error("skill-data: MemoryDB not connected")
        return web.json_response({"error": "database not ready"}, status=503)

    now = time.time()
    inserted_events = 0
    inserted_entities = 0

    try:
        async with conn.cursor() as cur:
            # ── Events ────────────────────────────────────────────────────────
            for ev in events:
                try:
                    await cur.execute(
                        """
                        INSERT OR IGNORE INTO skill_events
                            (id, skill_id, source, external_id, entity_id,
                             timestamp, direction, channel, content,
                             metadata_json, received_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            ev.get("id"),
                            skill_id,
                            ev.get("source", skill_id),
                            ev.get("external_id"),
                            ev.get("entity_id"),
                            ev.get("timestamp", ""),
                            ev.get("direction"),
                            ev.get("channel"),
                            ev.get("content"),
                            ev.get("metadata_json") or ev.get("metadata"),
                            now,
                        ),
                    )
                    inserted_events += cur.rowcount
                except Exception:
                    logger.exception("skill-data: failed to insert event %s", ev.get("id"))

            # ── Entities ──────────────────────────────────────────────────────
            for ent in entities:
                try:
                    await cur.execute(
                        """
                        INSERT INTO skill_entities
                            (id, skill_id, type, display_name,
                             identifiers_json, metadata_json, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            display_name     = excluded.display_name,
                            identifiers_json = excluded.identifiers_json,
                            metadata_json    = excluded.metadata_json,
                            updated_at       = excluded.updated_at
                        """,
                        (
                            ent.get("id"),
                            skill_id,
                            ent.get("type"),
                            ent.get("display_name"),
                            ent.get("identifiers_json") or ent.get("identifiers"),
                            ent.get("metadata_json") or ent.get("metadata"),
                            now,
                        ),
                    )
                    inserted_entities += cur.rowcount
                except Exception:
                    logger.exception("skill-data: failed to upsert entity %s", ent.get("id"))

        await conn.commit()

    except Exception:
        logger.exception("skill-data: transaction failed for skill %s", skill_id)
        return web.json_response({"error": "db write failed"}, status=500)

    logger.info(
        "skill-data: skill=%s events=%d entities=%d",
        skill_id, inserted_events, inserted_entities,
    )
    return web.json_response({
        "ok": True,
        "inserted_events": inserted_events,
        "inserted_entities": inserted_entities,
    })
