"""Memory-search MCP server — full-text recall over the session transcript.

Exposes ``search_past_conversations``: "remember when we discussed X?" across
every session this agent has stored. The index lives in
:mod:`src.memory.transcript_index` (a rebuildable FTS5 cache over
``sessions.runs``); this server is a thin wrapper that keeps it fresh, bounds
the result, and — the part that carries the most weight — tells the truth
about what it did and did not look at.

**Always on.** No feature flag, no API key, no provider. FTS5 ships inside
SQLite and the data is already on disk, so there is nothing to enable and
nothing whose removal could break it (§17). The predecessor was gated on
``OPENAGENT_SEMANTIC_SEARCH`` *and* ``OPENAI_API_KEY`` and, underneath both,
was reading a table that no code path had ever written — see the history in
``memory/transcript_index.py``.

**Sync before query, deliberately.** Every call reconciles the cache first.
That is not just freshness bookkeeping, it is how the delete invariant holds:
``purge_session`` drops the ``sessions`` row, the reconcile sees the id is
gone, and the cached messages go with it — so a deleted conversation is
unfindable from the very first query after the purge, with no notification
path and nothing in ``db.py`` needing to know this index exists. The pull
direction is what makes it safe; a push would be one forgotten call site away
from resurfacing a chat the user deleted.

The sweep is affordable because the freshness gate is one query over
``(session_id, updated_at, length(runs))`` — no transcript blob is read
unless it actually changed — and because a first-run cold build is capped
per call. It is also a subprocess MCP, which is the point: the parse runs in
this process, not the gateway's, so it cannot stall a live voice stream the
way an unbounded read on the event loop would.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP("memory-search")

# Row caps. The default is a page a model can reason about; the hard cap is
# what a determined ``limit=1000`` collapses to. Modelled on the ``logs`` MCP,
# which bounds at several independent levels rather than trusting one.
_DEFAULT_LIMIT = 5
_MAX_LIMIT = 25

# Total payload bound. A hit is a snippet plus a little metadata (~250 chars,
# ~65 tokens), so the default page costs ~350 tokens and a maxed-out one
# ~1.7k. This backstop only bites if snippets run long.
_MAX_RESULT_CHARS = 12_000

_index: Any = None
_index_lock: Optional[asyncio.Lock] = None


def _db_path() -> str:
    """Resolve the agent DB path the way every other in-tree MCP subprocess
    does — env var first (injected by ``MCPPool`` at spawn), then a relative
    fallback so the module still runs when invoked directly."""
    return os.environ.get("OPENAGENT_DB_PATH", "./openagent.db")


async def _get_index() -> Any:
    """Lazily open the index. Built on first use so importing the module (or
    booting the MCP) never touches the disk."""
    global _index, _index_lock
    if _index_lock is None:
        _index_lock = asyncio.Lock()
    async with _index_lock:
        if _index is None:
            from src.memory.transcript_index import TranscriptIndex

            override = os.environ.get("OPENAGENT_TRANSCRIPT_INDEX_PATH") or None
            _index = await asyncio.to_thread(TranscriptIndex, _db_path(), override)
        return _index


def _clamp(value: Any, default: int, lo: int, hi: int) -> int:
    """A model that asks for ``limit=1000`` wants "lots" and should get the
    cap, not a failed call that costs a whole round-trip to discover."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(n, hi))


def _fit_budget(hits: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Drop lowest-ranked hits until the payload fits ``_MAX_RESULT_CHARS``."""
    total = 0
    kept: list[dict[str, Any]] = []
    for h in hits:
        size = sum(len(str(v)) for v in h.values()) + 32
        if kept and total + size > _MAX_RESULT_CHARS:
            return kept, True
        total += size
        kept.append(h)
    return kept, False


@mcp.tool()
async def search_past_conversations(
    query: str,
    limit: int = _DEFAULT_LIMIT,
    offset: int = 0,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Keyword search over what was actually SAID in this agent's past sessions.

    Use for "remember when we discussed X?", "have we talked about Y before?",
    "what did I say about Z last month?" — questions about conversation
    history. For what you have deliberately LEARNED and written down, search
    the vault instead (``vault_search_notes``); the two are complements, and
    the vault is the better first stop for facts, preferences and decisions.

    HOW IT MATCHES: full text, not meaning. Words (and word prefixes) must
    actually appear — "launch deadline" will NOT find "the ship date we
    agreed". If a search comes back empty, try the user's own likely wording,
    distinctive nouns, or fewer terms before concluding anything. Wrap a
    phrase in double quotes for an exact match.

    WHAT IT COVERS: user and assistant messages from every stored session
    (including sub-agent, scheduled and workflow sessions). NOT covered: tool
    calls and their output (use the ``logs`` MCP), attachments, file
    contents, system prompts, or text that has been folded away by
    compaction. A message that was compacted into a recap is searchable only
    as the recap's wording.

    Args:
        query: Words likely to appear in the conversation. Quote a phrase for
            an exact match.
        limit: Hits to return (default 5, hard cap 25).
        offset: Skip this many hits — page with ``offset += limit``.
        session_id: Restrict to one conversation. ``None`` searches all.

    Returns ``{ok, hits, index, hint?}``. Each hit is ``{session_id, title,
    origin, role, ts, snippet}`` — a snippet with the matched terms bracketed,
    not the full message. Read a full transcript with
    ``GET /api/sessions/{session_id}/runs``.

    IMPORTANT — an empty ``hits`` does NOT mean the conversation never
    happened. Check ``index.sessions``: if it is 0, nothing is indexed yet
    and this tool knows nothing. If ``index.warming`` is true, older sessions
    are not in the index YET and are still being added. Either way, say what
    you actually know and fall back to the vault or ``GET /api/sessions``
    rather than telling the user it was never discussed.
    """
    if not query or not query.strip():
        return {"ok": False, "hits": [], "hint": "empty query"}

    n_limit = _clamp(limit, _DEFAULT_LIMIT, 1, _MAX_LIMIT)
    n_offset = _clamp(offset, 0, 0, 100_000)

    try:
        index = await _get_index()
        # Both legs run off the event loop: the sync can parse transcripts and
        # the search touches the disk, and this process also has to keep its
        # stdio MCP channel answering.
        stats = await asyncio.to_thread(index.sync)
        hits = await asyncio.to_thread(
            index.search, query, limit=n_limit, offset=n_offset,
            session_id=session_id,
        )
        totals = await asyncio.to_thread(index.stats)
    except Exception as e:  # noqa: BLE001 — a recall miss must not fail a turn
        logger.exception("search_past_conversations failed")
        return {"ok": False, "hits": [], "hint": f"error: {e}"}

    hits, truncated = _fit_budget(hits)

    out: dict[str, Any] = {
        "ok": True,
        "hits": hits,
        "index": {
            "sessions": totals["sessions"],
            "messages": totals["messages"],
            "warming": stats.pending > 0,
        },
    }

    hints: list[str] = []
    if stats.pending:
        hints.append(
            f"Index still warming — {stats.pending} older session(s) not "
            "indexed yet (newest are indexed first). Run this search again "
            "to continue building; do not conclude anything from a miss yet."
        )
    if not hits:
        if totals["sessions"] == 0:
            hints.append(
                "The index holds no sessions, so this tool cannot tell you "
                "whether something was discussed. This is NOT evidence that "
                "it wasn't. Check GET /api/sessions and the vault."
            )
        else:
            hints.append(
                f"No message matched these terms across {totals['sessions']} "
                f"indexed session(s). Matching is literal — try the user's "
                "own wording, distinctive nouns, or fewer terms. A miss here "
                "means these WORDS are absent, not that the topic is."
            )
    if truncated:
        out["result_truncated"] = True
        hints.append(
            f"Payload exceeded {_MAX_RESULT_CHARS} chars; lowest-ranked hits "
            "were dropped. Narrow the query or lower `limit`."
        )
    if len(hits) == n_limit:
        hints.append(f"More hits may exist — page with offset={n_offset + n_limit}.")
    if hints:
        out["hint"] = " ".join(hints)
    return out


def main() -> None:
    """Entry point matched by ``builtins.py`` python_module pattern."""
    mcp.run()


if __name__ == "__main__":
    main()
