"""Transcript index + memory-search MCP — the feature, end to end.

The thing this file exists to prove is embarrassing but load-bearing: that
``search_past_conversations`` RETURNS SOMETHING. Its predecessor could not.
It read ``conversation_embeddings``, whose only writer (``store_turn``) had
zero callers on every deployment that ever existed, so the table was empty by
construction and the tool was a permanent "no matches". Nothing failed —
there was no test that indexed a real session and demanded a real hit, so an
inert feature looked healthy for its entire life.

So the headline test seeds REAL-shaped session rows and asserts a NON-EMPTY
result, and the rest pin the three ways this class of index goes quietly
wrong:

  - it indexes the system prompt and every session matches everything;
  - it keeps serving text the source deleted (purge, compaction rewrite);
  - it can't recover from its own cache being removed.

Pure-unit: temp SQLite DBs, the real ``MemoryDB`` (so ``purge_session`` is
the actual shipped code), and the real tool function. No pool, no gateway,
no network, no provider.
"""
from __future__ import annotations

import json
import os
import time as _time
import uuid
from pathlib import Path
from typing import Any, Optional

from ._framework import TestContext, test


# A stand-in for the framework system prompt: long, and IDENTICAL across every
# session — which is exactly what makes it poison. The runtime really does put
# this in ``run.messages`` (``core/_runner/agent/_messages.py`` builds
# ``Message(role="system", …)`` at position 0), so any indexer that trusts
# ``messages`` wholesale inherits it on every row.
_FRAMEWORK_PROMPT = (
    "You are OpenAgent. You have a memory vault, MCP capabilities, a "
    "scheduler, a workflow engine, and sub-agent delegation. "
) * 20


def _run(pairs: list[tuple[str, str]], *, ts: int | None = None,
         with_system: bool = True, recap: bool = False) -> dict:
    """One RunOutput-shaped dict, as the runtime persists it into
    ``sessions.runs``: a system message, then the turn's messages, plus
    ``content`` duplicating the assistant reply."""
    msgs: list[dict] = []
    if with_system:
        msgs.append({"role": "system", "content": _FRAMEWORK_PROMPT})
    for role, text in pairs:
        msgs.append({"role": role, "content": text})
    last_assistant = next(
        (t for r, t in reversed(pairs) if r == "assistant"), None
    )
    run: dict[str, Any] = {
        "run_id": f"run-{uuid.uuid4().hex[:8]}",
        "messages": msgs,
        "created_at": ts if ts is not None else int(_time.time()),
    }
    if last_assistant:
        run["content"] = last_assistant
    if recap:
        run["metadata"] = {"compaction": True, "folded_runs": 3}
    return run


async def _open_db(db_path: Path):
    from src.memory.db import MemoryDB

    db = MemoryDB(str(db_path))
    await db.connect()
    return db


async def _seed(db, sid: str, runs: list[dict], *, title: str = "Chat",
                origin: str = "chat", double: bool = False) -> None:
    """Insert a session row carrying ``runs``. ``double`` reproduces the
    runtime's double-encoded column (a JSON string of the JSON array)."""
    conn = await db._ensure_connected()
    now = int(_time.time())
    payload = json.dumps(runs)
    if double:
        payload = json.dumps(payload)
    meta = json.dumps({"title": title, "origin": origin, "client_id": "alice"})
    await conn.execute(
        "INSERT INTO sessions (session_id, session_type, user_id, metadata, "
        "runs, created_at, updated_at) VALUES (?, 'agent', 'alice', ?, ?, ?, ?)",
        (sid, meta, payload, now, now),
    )
    await conn.commit()


def _fresh_tool(db_path: Path, index_path: Path):
    """Point the MCP at these paths and reset its cached index singleton, so
    each test drives the real tool against its own DB."""
    os.environ["OPENAGENT_DB_PATH"] = str(db_path)
    os.environ["OPENAGENT_TRANSCRIPT_INDEX_PATH"] = str(index_path)
    from src.mcp.servers.memory_search import server

    if server._index is not None:
        try:
            server._index.close()
        except Exception:
            pass
    server._index = None
    server._index_lock = None
    return server


def _cleanup(*paths: Optional[Path]) -> None:
    for p in paths:
        if p is None:
            continue
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(p) + suffix).unlink()
            except (FileNotFoundError, OSError):
                pass
    os.environ.pop("OPENAGENT_DB_PATH", None)
    os.environ.pop("OPENAGENT_TRANSCRIPT_INDEX_PATH", None)


def _paths(ctx: TestContext, tag: str) -> tuple[Path, Path]:
    stem = f"ti-{tag}-{uuid.uuid4().hex[:8]}"
    return (ctx.db_path.with_name(f"{stem}.db"),
            ctx.db_path.with_name(f"{stem}-idx.db"))


# ── the headline ─────────────────────────────────────────────────────


@test("transcript_index", "search_past_conversations returns a REAL hit (never true before)")
async def t_end_to_end_non_empty(ctx: TestContext) -> None:
    """Index real session rows, search a phrase, get a non-empty result.

    This is the assertion the old implementation could not have passed at any
    point in its existence.
    """
    db_path, idx_path = _paths(ctx, "e2e")
    try:
        db = await _open_db(db_path)
        await _seed(db, "chat-a", [
            _run([("user", "Can we settle the launch deadline for the React migration?"),
                  ("assistant", "Let's target the fifteenth of March for the migration.")]),
        ], title="Migration planning")
        await _seed(db, "chat-b", [
            _run([("user", "What is the best way to bake sourdough bread?"),
                  ("assistant", "Start with a mature starter and a long cold proof.")]),
        ], title="Baking")
        await db.close()

        server = _fresh_tool(db_path, idx_path)
        res = await server.search_past_conversations("launch deadline")

        assert res["ok"] is True, res
        assert res["hits"], f"THE bug: empty hits over indexed sessions — {res}"
        assert res["index"]["sessions"] == 2, res["index"]
        assert res["index"]["messages"] > 0, res["index"]

        top = res["hits"][0]
        assert top["session_id"] == "chat-a", f"wrong session ranked first: {res['hits']}"
        assert top["title"] == "Migration planning", top
        assert "deadline" in top["snippet"].lower(), top
        assert "[" in top["snippet"], f"expected bracketed match markers: {top}"

        # The unrelated session must not be dragged in by a shared stopword.
        assert all(h["session_id"] == "chat-a" for h in res["hits"]), res["hits"]

        # Both directions of the conversation are searchable, not just one.
        res_u = await server.search_past_conversations("sourdough")
        assert res_u["hits"] and res_u["hits"][0]["session_id"] == "chat-b", res_u
        roles = {h["role"] for h in
                 (await server.search_past_conversations("migration"))["hits"]}
        assert "user" in roles or "assistant" in roles, roles
    finally:
        _cleanup(db_path, idx_path)


# ── the three quiet failures ─────────────────────────────────────────


@test("transcript_index", "the framework system prompt is not indexed")
async def t_system_prompt_not_indexed(ctx: TestContext) -> None:
    """``run.messages`` starts with the ~10.8k-token framework prompt, byte-
    identical on every session. Indexing it would make every session match
    every query — a search for "sub-agent delegation" would return the whole
    database, ranked meaninglessly. Only user/assistant roles are indexed."""
    db_path, idx_path = _paths(ctx, "sys")
    try:
        db = await _open_db(db_path)
        for i in range(3):
            await _seed(db, f"chat-{i}", [
                _run([("user", f"Question number {i} about zucchini."),
                      ("assistant", f"Answer number {i} about zucchini.")]),
            ])
        await db.close()

        server = _fresh_tool(db_path, idx_path)
        # A distinctive phrase that appears ONLY in the system prompt.
        res = await server.search_past_conversations('"sub-agent delegation"')
        assert res["hits"] == [], (
            "system-prompt text is searchable — the index is poisoned and "
            f"every session will match every query: {res}"
        )
        # Sanity: the real conversation text IS indexed on those same rows.
        ok = await server.search_past_conversations("zucchini")
        assert len(ok["hits"]) >= 3, ok
    finally:
        _cleanup(db_path, idx_path)


@test("transcript_index", "a purged session is not findable")
async def t_purged_not_findable(ctx: TestContext) -> None:
    """The invariant ``db.py``'s ``_SESSION_SATELLITE_TABLES`` protects for the
    old table ("a deleted chat would keep resurfacing through memory-search"),
    now honoured by reconcile instead of notification: ``purge_session`` drops
    the ``sessions`` row, and the sync before the next query drops the cached
    messages with it. Uses the REAL ``purge_session``, which knows nothing
    about this index — that's the point."""
    db_path, idx_path = _paths(ctx, "purge")
    try:
        db = await _open_db(db_path)
        await _seed(db, "chat-secret", [
            _run([("user", "The passphrase is hyacinth marmalade."),
                  ("assistant", "Noted, I will remember hyacinth marmalade.")]),
        ])
        await _seed(db, "chat-keep", [
            _run([("user", "Unrelated chatter about hyacinth flowers."),
                  ("assistant", "Hyacinths bloom in spring.")]),
        ])
        await db.close()

        server = _fresh_tool(db_path, idx_path)
        before = await server.search_past_conversations("marmalade")
        assert before["hits"], f"precondition failed — nothing to purge: {before}"

        db = await _open_db(db_path)
        await db.purge_session("chat-secret")
        await db.close()

        after = await server.search_past_conversations("marmalade")
        assert after["hits"] == [], (
            f"DELETED CONVERSATION STILL FINDABLE: {after}"
        )
        assert after["index"]["sessions"] == 1, after["index"]
        # The bystander survives — the reconcile deletes precisely, not broadly.
        keep = await server.search_past_conversations("hyacinth")
        assert keep["hits"] and all(
            h["session_id"] == "chat-keep" for h in keep["hits"]), keep
    finally:
        _cleanup(db_path, idx_path)


@test("transcript_index", "compaction-rewritten text does not resurface")
async def t_compaction_rewrite(ctx: TestContext) -> None:
    """``core/compaction.py`` folds old runs into a recap and rewrites
    ``sessions.runs`` IN PLACE. An index keyed on append-only assumptions
    would keep serving the folded-away text forever — quoting back to the user
    paragraphs the system already decided to drop. Drives the real
    ``compaction._save_runs``, so this test tracks the actual writer."""
    from src.core import compaction

    db_path, idx_path = _paths(ctx, "compact")
    try:
        db = await _open_db(db_path)
        original = [
            _run([("user", "Tell me about the peregrine falcon nesting site."),
                  ("assistant", "The peregrine nests on the cathedral tower.")]),
            _run([("user", "And the kestrel?"),
                  ("assistant", "Kestrels prefer open farmland.")]),
        ]
        await _seed(db, "chat-c", original)
        await db.close()

        server = _fresh_tool(db_path, idx_path)
        before = await server.search_past_conversations("peregrine")
        assert before["hits"], f"precondition failed: {before}"

        # Exactly what compact() persists: [recap] + last N runs, with the
        # folded text GONE from the column.
        recap = _run([("assistant", "Earlier we discussed raptor habitats.")],
                     recap=True)
        kept = original[-1:]
        # updated_at is second-granular; make sure the rewrite is not mistaken
        # for "unchanged" merely because the test ran fast. The runs length
        # also changes here, which is the other half of the invalidation pair.
        _time.sleep(1.05)
        compaction._save_runs(str(db_path), "chat-c", [recap] + kept)

        after = await server.search_past_conversations("peregrine")
        assert after["hits"] == [], (
            f"COMPACTED-AWAY TEXT STILL SEARCHABLE: {after}"
        )
        # The recap that replaced it IS searchable, and the kept run survives.
        rec = await server.search_past_conversations("raptor habitats")
        assert rec["hits"], f"recap text not indexed after rewrite: {rec}"
        keep = await server.search_past_conversations("kestrel")
        assert keep["hits"], f"kept run lost in the re-index: {keep}"
        assert after["index"]["sessions"] == 1, after["index"]
    finally:
        _cleanup(db_path, idx_path)


@test("transcript_index", "the index rebuilds after deletion")
async def t_rebuilds_after_deletion(ctx: TestContext) -> None:
    """The vault index's contract, restated here: "delete it and the next sync
    rebuilds it". It is a cache, not a store — nothing is lost by removing it,
    which is what keeps it from being a second source of truth."""
    db_path, idx_path = _paths(ctx, "rebuild")
    try:
        db = await _open_db(db_path)
        await _seed(db, "chat-r", [
            _run([("user", "Remind me about the tungsten shipment."),
                  ("assistant", "The tungsten arrives on Thursday.")]),
        ])
        await db.close()

        server = _fresh_tool(db_path, idx_path)
        first = await server.search_past_conversations("tungsten")
        assert first["hits"], first
        assert idx_path.exists(), "index file was never created"

        # Nuke the cache exactly as an operator would.
        server._index.close()
        server._index = None
        server._index_lock = None
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(idx_path) + suffix).unlink()
            except FileNotFoundError:
                pass
        assert not idx_path.exists()

        again = await server.search_past_conversations("tungsten")
        assert again["hits"], f"index did not rebuild after deletion: {again}"
        assert again["hits"][0]["session_id"] == "chat-r", again
    finally:
        _cleanup(db_path, idx_path)


# ── mechanics ────────────────────────────────────────────────────────


@test("transcript_index", "sync is incremental and re-parses only changed sessions")
async def t_incremental(ctx: TestContext) -> None:
    """The ``(mtime, size)`` gate of ``vault/index.py``, as
    ``(updated_at, length(runs))``. A quiet agent must not re-parse
    transcripts on every query — that is what makes sync-before-query
    affordable enough to be correct."""
    from src.memory.transcript_index import TranscriptIndex

    db_path, idx_path = _paths(ctx, "incr")
    try:
        db = await _open_db(db_path)
        for i in range(5):
            await _seed(db, f"chat-{i}", [
                _run([("user", f"Message {i} about badgers."),
                      ("assistant", f"Reply {i} about badgers.")]),
            ])
        await db.close()

        idx = TranscriptIndex(db_path, idx_path)
        first = idx.sync()
        assert first.added == 5 and first.unchanged == 0, first.to_dict()

        second = idx.sync()
        assert second.unchanged == 5, second.to_dict()
        assert second.added == 0 and second.updated == 0, second.to_dict()
        assert second.messages == 0, (
            f"re-parsed unchanged sessions: {second.to_dict()}"
        )

        # force= re-parses everything (the rebuild escape hatch).
        forced = idx.sync(force=True)
        assert forced.updated == 5 and forced.unchanged == 0, forced.to_dict()
        idx.close()
    finally:
        _cleanup(db_path, idx_path)


@test("transcript_index", "a cold build is bounded per sync and reports the remainder")
async def t_bounded_sync(ctx: TestContext) -> None:
    """A first query on a large existing agent must not slurp every transcript
    in one call. Newest first, cap the rest, and SAY so — an incomplete index
    that reports itself complete is how "no hits" becomes a lie."""
    from src.memory.transcript_index import TranscriptIndex

    db_path, idx_path = _paths(ctx, "bound")
    try:
        db = await _open_db(db_path)
        for i in range(6):
            await _seed(db, f"chat-{i}", [
                _run([("user", f"Note {i} about pangolins."),
                      ("assistant", f"Reply {i} about pangolins.")], ts=1000 + i),
            ])
        # Distinct updated_at so "newest first" is well-defined.
        conn = await db._ensure_connected()
        for i in range(6):
            await conn.execute("UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                               (2000 + i, f"chat-{i}"))
        await conn.commit()
        await db.close()

        idx = TranscriptIndex(db_path, idx_path)
        s1 = idx.sync(max_sessions=2)
        assert s1.added == 2, s1.to_dict()
        assert s1.pending == 4, s1.to_dict()
        # Newest indexed first.
        assert {r["session_id"] for r in idx._conn.execute(
            "SELECT session_id FROM indexed_sessions")} == {"chat-5", "chat-4"}

        s2 = idx.sync(max_sessions=2)
        assert s2.added == 2 and s2.pending == 2, s2.to_dict()
        s3 = idx.sync(max_sessions=10)
        assert s3.added == 2 and s3.pending == 0, s3.to_dict()
        assert idx.stats()["sessions"] == 6
        idx.close()
    finally:
        _cleanup(db_path, idx_path)


@test("transcript_index", "double-encoded runs are indexed")
async def t_double_encoded(ctx: TestContext) -> None:
    """The runtime stores ``runs`` double-encoded on some paths. A reader that
    doesn't unwrap twice sees an empty transcript on exactly those rows —
    the bug that made every session render as an empty message list."""
    db_path, idx_path = _paths(ctx, "dbl")
    try:
        db = await _open_db(db_path)
        await _seed(db, "chat-plain", [
            _run([("user", "Plain row mentions armadillo.")]),
        ])
        await _seed(db, "chat-double", [
            _run([("user", "Double row mentions armadillo.")]),
        ], double=True)
        await db.close()

        server = _fresh_tool(db_path, idx_path)
        res = await server.search_past_conversations("armadillo")
        found = {h["session_id"] for h in res["hits"]}
        assert found == {"chat-plain", "chat-double"}, (
            f"double-encoded runs not indexed: {res}"
        )
    finally:
        _cleanup(db_path, idx_path)


@test("transcript_index", "results are capped, paginated, and deduped")
async def t_caps_and_paging(ctx: TestContext) -> None:
    db_path, idx_path = _paths(ctx, "caps")
    try:
        db = await _open_db(db_path)
        for i in range(30):
            await _seed(db, f"chat-{i}", [
                _run([("user", f"Entry {i} concerning wombats."),
                      ("assistant", f"Response {i} concerning wombats.")]),
            ])
        await db.close()

        server = _fresh_tool(db_path, idx_path)
        # limit is clamped, not rejected.
        big = await server.search_past_conversations("wombats", limit=1000)
        assert len(big["hits"]) <= 25, len(big["hits"])
        assert big["ok"] is True

        # Default page size.
        default = await server.search_past_conversations("wombats")
        assert len(default["hits"]) == 5, len(default["hits"])
        assert "offset=5" in default.get("hint", ""), default.get("hint")

        # Pagination advances.
        p1 = await server.search_past_conversations("wombats", limit=3, offset=0)
        p2 = await server.search_past_conversations("wombats", limit=3, offset=3)
        k1 = [(h["session_id"], h["snippet"]) for h in p1["hits"]]
        k2 = [(h["session_id"], h["snippet"]) for h in p2["hits"]]
        assert k1 and k2 and k1 != k2, (k1, k2)

        # ``content`` duplicates the assistant message; one reply = one hit.
        one = await server.search_past_conversations(
            "wombats", limit=25, session_id="chat-0")
        assert len(one["hits"]) == 2, (
            f"expected exactly user+assistant, got {len(one['hits'])}: {one['hits']}"
        )

        # session_id scoping.
        assert all(h["session_id"] == "chat-0" for h in one["hits"]), one
    finally:
        _cleanup(db_path, idx_path)


@test("transcript_index", "an empty or missing index never reads as 'never discussed'")
async def t_honest_when_empty(ctx: TestContext) -> None:
    """The exact failure ``core/prompts.py`` calls out. A tool that cannot
    answer must say it cannot answer — "no hits" from an empty index is not
    evidence of absence, and the model has no way to know that unless the
    payload says so."""
    db_path, idx_path = _paths(ctx, "empty")
    try:
        db = await _open_db(db_path)
        await db.close()

        server = _fresh_tool(db_path, idx_path)
        res = await server.search_past_conversations("anything at all")
        assert res["ok"] is True, res
        assert res["hits"] == []
        assert res["index"]["sessions"] == 0, res
        hint = res.get("hint", "").lower()
        assert "not" in hint and "evidence" in hint, (
            f"empty index must disclaim absence, got: {res.get('hint')!r}"
        )

        # A populated index that simply misses must ALSO disclaim.
        db = await _open_db(db_path)
        await _seed(db, "chat-x", [_run([("user", "Something about otters.")])])
        await db.close()
        miss = await server.search_past_conversations("xylophone")
        assert miss["hits"] == []
        assert "words" in miss.get("hint", "").lower(), miss.get("hint")
        assert miss["index"]["sessions"] == 1, miss

        # An empty query is rejected without touching the index.
        bad = await server.search_past_conversations("   ")
        assert bad["ok"] is False and bad["hits"] == [], bad
        # Pure punctuation has no terms to match — no crash, no hits.
        punct = await server.search_past_conversations("!!! ???")
        assert punct["hits"] == [], punct
    finally:
        _cleanup(db_path, idx_path)


@test("transcript_index", "the cache refuses to answer out of another agent's DB")
async def t_source_db_rebinds(ctx: TestContext) -> None:
    """The index is keyed to the DB it was built from. Pointed at a different
    one it wipes rather than blending — mixing two agents' conversations is
    the worst failure this component could have."""
    from src.memory.transcript_index import TranscriptIndex

    db_a, idx_path = _paths(ctx, "srca")
    db_b = ctx.db_path.with_name(f"ti-srcb-{uuid.uuid4().hex[:8]}.db")
    try:
        db = await _open_db(db_a)
        await _seed(db, "chat-a", [_run([("user", "Alpha secret: nightingale.")])])
        await db.close()
        db = await _open_db(db_b)
        await _seed(db, "chat-b", [_run([("user", "Beta secret: cormorant.")])])
        await db.close()

        idx = TranscriptIndex(db_a, idx_path)
        idx.sync()
        assert idx.search("nightingale"), "precondition failed"
        idx.close()

        # Same cache file, different source DB.
        idx2 = TranscriptIndex(db_b, idx_path)
        idx2.sync()
        assert idx2.search("nightingale") == [], (
            "cache leaked another agent's conversation across a source change"
        )
        assert idx2.search("cormorant"), "did not rebuild for the new source"
        idx2.close()
    finally:
        _cleanup(db_a, db_b, idx_path)


@test("transcript_index", "default index path derives from the db path, not platform defaults")
async def t_index_path_keyed_to_db(ctx: TestContext) -> None:
    """A subprocess MCP that re-resolves platform defaults can land on another
    agent's data — the bug that forced ``OPENAGENT_DB_PATH`` injection. The
    cache location is derived from the injected db path so it cannot."""
    from src.memory.transcript_index import default_index_path

    root = ctx.db_path.parent
    a = default_index_path(root / "agent-one" / "openagent.db")
    b = default_index_path(root / "agent-two" / "openagent.db")
    c = default_index_path(root / "agent-one" / "other.db")
    assert a != b, "two agents share one cache file"
    assert a != c, "two DBs in one directory share one cache file"
    # Beside the db it describes, so it travels with the agent's data dir.
    assert a.parent == (root / "agent-one").resolve(), a
    assert a.name.startswith("transcript_index_"), a
    # Stable across calls — not a fresh temp name each boot.
    assert a == default_index_path(root / "agent-one" / "openagent.db")
    # Keyed on the CANONICAL path, so two routes to one db share one cache
    # rather than silently building it twice.
    assert a == default_index_path(root / "agent-one" / "." / "openagent.db")
