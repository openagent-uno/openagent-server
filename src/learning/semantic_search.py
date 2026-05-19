"""Semantic search over persisted conversation turns.

Two halves: a *writer* path that fires from ``claude_cli`` after every
turn to embed + store the user/assistant pair, and a *reader* path that
the ``memory-search`` MCP server uses to answer cross-session recall
queries ("remember when we discussed the launch deadline?").

Embedding model is OpenAI's ``text-embedding-3-small`` — 1536 dims,
~$0.02 per million tokens. A typical turn embeds in ~250-400 ms so the
write hook runs fire-and-forget; the reader path is invoked only on
explicit user queries via the MCP tool, so its latency lives on the
agent's regular tool-call timeline (where 300 ms is invisible next to
the model's own thinking).

Both halves are no-ops when ``OPENAGENT_SEMANTIC_SEARCH`` isn't truthy
or when ``OPENAI_API_KEY`` is unset. The feature degrades gracefully —
older deployments without OpenAI access just stop accruing rows and the
MCP returns an empty result with a hint about enabling the key.

Why JSON-in-TEXT for the vector instead of a real vector type:
SQLite ships no native vector ext; sqlite-vss would be the right move
once we cross ~50k turns, but for the typical personal/SaaS-per-tenant
deployment a Python cosine loop over ~5k rows is faster than installing
a binary extension and well below the cost floor of the embedding call
itself. When the deployment outgrows this, swap the ``search`` impl
without touching the schema.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from typing import Any, Optional

from src.core.logging import elog


_DEFAULT_EMBED_MODEL = "text-embedding-3-small"
_DEFAULT_TOP_K = 5
# Below this size, embedding mostly captures noise — skip so we don't
# pollute the index with "ok" / "thanks" / single emojis.
_MIN_TEXT_CHARS = 20
_MAX_EMBED_INPUT_CHARS = 8000
_MAX_STORED_TEXT_CHARS = 4000


def _is_enabled() -> bool:
    return (
        os.environ.get("OPENAGENT_SEMANTIC_SEARCH", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )


# Module-level singleton — instantiating ``AsyncOpenAI`` opens an httpx
# session + pays a TLS handshake; cache it across the many calls the
# writer path makes. Keyed by api_key so a key rotation lands cleanly.
_openai_clients: dict[str, Any] = {}


async def _get_openai_client() -> Optional[Any]:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        return None
    if api_key in _openai_clients:
        return _openai_clients[api_key]
    try:
        from openai import AsyncOpenAI  # type: ignore
        _openai_clients[api_key] = AsyncOpenAI(api_key=api_key)
        return _openai_clients[api_key]
    except Exception as e:
        elog("semantic_search.openai_import_failed", error=str(e)[:200])
        return None


async def embed_text(text: str) -> Optional[list[float]]:
    """Return the embedding for ``text`` or ``None`` on any failure.
    Caller treats ``None`` as "skip this row" rather than raising —
    the index is best-effort, not load-bearing for any other feature.
    """
    if not text:
        return None
    client = await _get_openai_client()
    if client is None:
        return None
    try:
        resp = await client.embeddings.create(
            model=_DEFAULT_EMBED_MODEL,
            input=text[:_MAX_EMBED_INPUT_CHARS],
        )
        return list(resp.data[0].embedding)
    except Exception as e:
        elog("semantic_search.embed_error", error=str(e)[:200])
        return None


async def _store_row(
    conn: Any,
    session_id: str,
    role: str,
    text: str,
    embedding: list[float],
) -> None:
    try:
        await conn.execute(
            """
            INSERT INTO conversation_embeddings
                (session_id, role, text, embedding_json, model, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                text[:_MAX_STORED_TEXT_CHARS],
                json.dumps(embedding),
                _DEFAULT_EMBED_MODEL,
                time.time(),
            ),
        )
        await conn.commit()
    except Exception as e:
        elog("semantic_search.persist_error", error=str(e)[:200])


async def store_turn(
    db: Any,
    session_id: str,
    user_text: str,
    assistant_text: str,
) -> None:
    """Embed + persist both halves of a turn. Fires as a background
    task so the writer never blocks the agent's reply path."""
    if not _is_enabled() or not session_id or not db:
        return
    conn = getattr(db, "_conn", None)
    if conn is None:
        return

    async def _embed_and_store_both() -> None:
        # Sequential rather than parallel: OpenAI rate-limits per-key,
        # and the writer is fire-and-forget so latency doesn't matter.
        for role, text in (("user", user_text or ""), ("assistant", assistant_text or "")):
            cleaned = (text or "").strip()
            if len(cleaned) < _MIN_TEXT_CHARS:
                continue
            emb = await embed_text(cleaned)
            if emb is None:
                continue
            await _store_row(conn, session_id, role, cleaned, emb)

    asyncio.create_task(_embed_and_store_both())


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Plain Python cosine. Vectorising via numpy here saves ~3× but
    numpy isn't a core dep; for the row counts a typical chat builds
    over a year (~5-10k) the difference is sub-50 ms total per query,
    which the embedding API call dwarfs anyway."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


async def search(
    db: Any,
    query: str,
    top_k: int = _DEFAULT_TOP_K,
    session_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return the ``top_k`` most semantically similar turn rows for
    ``query``. ``session_id`` optionally narrows to one conversation —
    None reads across all sessions on this agent. Empty list on any
    failure (feature off, no OPENAI_API_KEY, embedding miss, etc.) so
    the caller can render a "no matches" message gracefully.
    """
    if not _is_enabled() or not query or not db:
        return []
    conn = getattr(db, "_conn", None)
    if conn is None:
        return []
    q_emb = await embed_text(query)
    if q_emb is None:
        return []
    try:
        if session_id:
            cur = await conn.execute(
                "SELECT id, session_id, role, text, embedding_json, timestamp "
                "FROM conversation_embeddings WHERE session_id = ?",
                (session_id,),
            )
        else:
            cur = await conn.execute(
                "SELECT id, session_id, role, text, embedding_json, timestamp "
                "FROM conversation_embeddings"
            )
        rows = await cur.fetchall()
        await cur.close()
    except Exception as e:
        elog("semantic_search.read_error", error=str(e)[:200])
        return []

    scored: list[tuple[float, Any]] = []
    for r in rows or []:
        try:
            emb = json.loads(r[4])
        except Exception:
            continue
        score = _cosine_similarity(q_emb, emb)
        if score > 0.0:
            scored.append((score, r))
    scored.sort(key=lambda p: -p[0])
    out: list[dict[str, Any]] = []
    for score, r in scored[: max(1, int(top_k))]:
        out.append(
            {
                "id": int(r[0]),
                "session_id": r[1],
                "role": r[2],
                "text": r[3],
                "timestamp": r[5],
                "score": round(score, 4),
            }
        )
    return out
