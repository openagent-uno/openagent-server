"""Skill-data MCP server.

Exposes read-only query tools over ``skill_events`` + ``skill_entities``
— the two tables that central skill runners (telegram-ingester, etc.)
populate via ``POST /api/internal/skill-data``.

Every skill that pushes data lands in the same two tables, so a single
generic query surface works for all of them. The agent picks rows by
``skill_id`` to scope to one source ("show me my recent Telegram
messages") or queries across all skills ("any event mentioning Tunisia
last week").

Connection: reads ``OPENAGENT_DB_PATH`` (injected by the MCP launcher
at spawn time, same as memory-search). No writes — central runners
write through the HTTP endpoint so there's only one write path.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import aiosqlite
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("skill-data")


def _db_path() -> str:
    """Same resolution as every other builtin MCP."""
    return os.environ.get("OPENAGENT_DB_PATH", "./openagent.db")


async def _connect() -> aiosqlite.Connection:
    conn = await aiosqlite.connect(_db_path(), timeout=5.0)
    conn.row_factory = aiosqlite.Row
    return conn


def _row_to_event(row: aiosqlite.Row) -> dict:
    d = dict(row)
    # Best-effort parse of metadata so the agent sees structured data
    if d.get("metadata_json"):
        try:
            d["metadata"] = json.loads(d["metadata_json"])
        except Exception:
            d["metadata"] = d["metadata_json"]
        d.pop("metadata_json", None)
    return d


def _row_to_entity(row: aiosqlite.Row) -> dict:
    d = dict(row)
    for js_key, target in (
        ("identifiers_json", "identifiers"),
        ("metadata_json", "metadata"),
    ):
        if d.get(js_key):
            try:
                d[target] = json.loads(d[js_key])
            except Exception:
                d[target] = d[js_key]
            d.pop(js_key, None)
    return d


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_skills() -> dict[str, Any]:
    """List every skill that has pushed data into this agent.

    Returns one row per ``skill_id`` with the count of events and the
    timestamp of the most recently received event. Useful when the
    agent wants to know "what data sources do I have?".
    """
    try:
        conn = await _connect()
        try:
            async with conn.execute(
                """
                SELECT skill_id,
                       COUNT(*)      AS event_count,
                       MAX(received_at) AS last_received_at,
                       MAX(timestamp)   AS last_event_at
                FROM skill_events
                GROUP BY skill_id
                ORDER BY last_received_at DESC NULLS LAST
                """
            ) as cur:
                rows = await cur.fetchall()
            return {"ok": True, "skills": [dict(r) for r in rows]}
        finally:
            await conn.close()
    except Exception as e:
        logger.exception("list_skills failed")
        return {"ok": False, "skills": [], "hint": str(e)}


@mcp.tool()
async def search_events(
    query: str,
    skill_id: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Full-text search across event content.

    Args:
        query: Substring to match against the ``content`` column
            (case-insensitive ``LIKE %query%``).
        skill_id: Restrict to one skill (e.g. ``"telegram-ingester"``).
            ``None`` searches every skill.
        limit: Max rows to return (default 20, hard cap 200).
    """
    if not query or not query.strip():
        return {"ok": False, "events": [], "hint": "empty query"}
    capped = max(1, min(int(limit), 200))

    sql = (
        "SELECT id, skill_id, source, external_id, entity_id, timestamp, "
        "direction, channel, content, metadata_json, received_at "
        "FROM skill_events WHERE content LIKE ?"
    )
    params: list[Any] = [f"%{query}%"]
    if skill_id:
        sql += " AND skill_id = ?"
        params.append(skill_id)
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(capped)

    try:
        conn = await _connect()
        try:
            async with conn.execute(sql, params) as cur:
                rows = await cur.fetchall()
            return {"ok": True, "events": [_row_to_event(r) for r in rows]}
        finally:
            await conn.close()
    except Exception as e:
        logger.exception("search_events failed")
        return {"ok": False, "events": [], "hint": str(e)}


@mcp.tool()
async def list_events_by_entity(
    entity_id: str,
    limit: int = 50,
) -> dict[str, Any]:
    """Recent events involving one entity (contact, group, channel).

    Args:
        entity_id: The ULID stored on ``skill_events.entity_id`` (also
            the PK of ``skill_entities``). Use ``list_entities`` first
            to look it up by name.
        limit: Max rows (default 50, hard cap 500).
    """
    capped = max(1, min(int(limit), 500))
    try:
        conn = await _connect()
        try:
            async with conn.execute(
                """
                SELECT id, skill_id, source, external_id, entity_id, timestamp,
                       direction, channel, content, metadata_json, received_at
                FROM skill_events
                WHERE entity_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (entity_id, capped),
            ) as cur:
                rows = await cur.fetchall()
            return {"ok": True, "events": [_row_to_event(r) for r in rows]}
        finally:
            await conn.close()
    except Exception as e:
        logger.exception("list_events_by_entity failed")
        return {"ok": False, "events": [], "hint": str(e)}


@mcp.tool()
async def list_entities(
    name_like: str | None = None,
    skill_id: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List contacts/groups/channels resolved by skills.

    Args:
        name_like: Substring match on ``display_name`` (e.g. "Mario").
        skill_id: Restrict to one skill.
        limit: Max rows (default 50, hard cap 500).
    """
    capped = max(1, min(int(limit), 500))
    sql = (
        "SELECT id, skill_id, type, display_name, identifiers_json, "
        "metadata_json, updated_at FROM skill_entities WHERE 1=1"
    )
    params: list[Any] = []
    if name_like:
        sql += " AND display_name LIKE ?"
        params.append(f"%{name_like}%")
    if skill_id:
        sql += " AND skill_id = ?"
        params.append(skill_id)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(capped)

    try:
        conn = await _connect()
        try:
            async with conn.execute(sql, params) as cur:
                rows = await cur.fetchall()
            return {"ok": True, "entities": [_row_to_entity(r) for r in rows]}
        finally:
            await conn.close()
    except Exception as e:
        logger.exception("list_entities failed")
        return {"ok": False, "entities": [], "hint": str(e)}


@mcp.tool()
async def recent_events(
    skill_id: str | None = None,
    limit: int = 30,
) -> dict[str, Any]:
    """Newest events, optionally scoped to one skill. Useful for a
    "what happened today?" lead-in to the agent's reply.
    """
    capped = max(1, min(int(limit), 200))
    sql = (
        "SELECT id, skill_id, source, external_id, entity_id, timestamp, "
        "direction, channel, content, metadata_json, received_at "
        "FROM skill_events"
    )
    params: list[Any] = []
    if skill_id:
        sql += " WHERE skill_id = ?"
        params.append(skill_id)
    sql += " ORDER BY timestamp DESC LIMIT ?"
    params.append(capped)

    try:
        conn = await _connect()
        try:
            async with conn.execute(sql, params) as cur:
                rows = await cur.fetchall()
            return {"ok": True, "events": [_row_to_event(r) for r in rows]}
        finally:
            await conn.close()
    except Exception as e:
        logger.exception("recent_events failed")
        return {"ok": False, "events": [], "hint": str(e)}


def main() -> None:
    """Entry point matched by ``builtins.py`` python_module pattern."""
    mcp.run()


if __name__ == "__main__":
    main()
