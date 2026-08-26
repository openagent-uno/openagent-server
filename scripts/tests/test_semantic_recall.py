"""Semantic recall — the index (Layer A) and the auto-recall hook (Layer B).

The gap this closes: keyword recall matches WORDS, so "launch deadline" cannot
find "the ship date we agreed", and the agent answers "no record" while the note
exists — a confident miss. These tests prove the semantic layer finds it, AND
that it is safe: off by default, inert without an embedding model, thresholded,
verify-framed, cache-safe, and a rebuildable cache that honours purges.

No network: the embedder is a DETERMINISTIC concept-mapping fake (synonyms share
a strong vector dimension; unrelated words hash to near-orthogonal residual
dims). That proves the index/search/injection PLUMBING and the ranking LOGIC
without a live endpoint — it does not claim real-model quality. A reachable
endpoint is used only by the out-of-band measurement script, not here.

Pure-unit: temp SQLite DBs, real ``MemoryDB`` (so ``purge_session`` is the
shipped code), real ``TranscriptIndex`` (so the keyword-miss half is honest),
and the real ``_with_recall`` hook. No pool, no gateway, no provider.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time as _time
import uuid
from pathlib import Path
from typing import Any, Optional

from ._framework import TestContext, test


# ── deterministic fake embedder ───────────────────────────────────────

# Synonyms share a reserved STRONG dimension (0..7), so paraphrases with zero
# shared words are cosine-near; every other word hashes into a residual dim
# (8..63), so unrelated texts are near-orthogonal and fall below any sane floor.
_CONCEPTS: dict[str, int] = {}
for _dim, _group in enumerate([
    {"launch", "deadline", "ship", "date", "target", "due", "migration"},
    {"customer", "complain", "complained", "complaint", "refund", "unhappy", "angry"},
    {"sourdough", "bread", "bake", "baking", "starter", "proof"},
    {"tungsten", "shipment", "arrive", "arrives", "delivery"},
]):
    for _w in _group:
        _CONCEPTS[_w] = _dim
_DIM = 64


class ConceptEmbedder:
    """Fake embedder: concept-aware, deterministic, no network."""

    model_id = "fake:concept-v1"

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * _DIM
            for w in re.findall(r"[a-z]+", (t or "").lower()):
                if w in _CONCEPTS:
                    v[_CONCEPTS[w]] += 3.0
                else:
                    h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                    v[8 + (h % (_DIM - 8))] += 1.0
            out.append(v)
        return out


class _OtherModelEmbedder(ConceptEmbedder):
    """Same vectors, different ``model_id`` — used to prove a model change wipes
    the cache (vectors from two models are incomparable)."""
    model_id = "fake:other-v2"


# ── fixtures ──────────────────────────────────────────────────────────


async def _open_db(db_path: Path):
    from src.memory.db import MemoryDB
    db = MemoryDB(str(db_path))
    await db.connect()
    return db


async def _seed_session(db, sid: str, pairs: list[tuple[str, str]], *,
                        title: str = "Chat", origin: str = "chat") -> None:
    conn = await db._ensure_connected()
    now = int(_time.time())
    runs = [{
        "run_id": f"run-{uuid.uuid4().hex[:8]}",
        "messages": [{"role": r, "content": t} for r, t in pairs],
        "created_at": now,
    }]
    last = next((t for r, t in reversed(pairs) if r == "assistant"), None)
    if last:
        runs[0]["content"] = last
    meta = json.dumps({"title": title, "origin": origin})
    await conn.execute(
        "INSERT INTO sessions (session_id, session_type, user_id, metadata, "
        "runs, created_at, updated_at) VALUES (?, 'agent', 'u', ?, ?, ?, ?)",
        (sid, meta, json.dumps(runs), now, now),
    )
    await conn.commit()


def _write_note(vault: Path, name: str, title: str, body: str) -> Path:
    p = vault / f"{name}.md"
    p.write_text(f"---\ntitle: {title}\n---\n{body}\n")
    return p


def _paths(ctx: TestContext, tag: str) -> tuple[Path, Path, Path]:
    stem = f"sr-{tag}-{uuid.uuid4().hex[:8]}"
    db = ctx.db_path.with_name(f"{stem}.db")
    idx = ctx.db_path.with_name(f"{stem}-sem.db")
    vault = ctx.db_path.with_name(f"{stem}-vault")
    vault.mkdir(exist_ok=True)
    return db, idx, vault


def _cleanup(*paths: Optional[Path]) -> None:
    import shutil
    for p in paths:
        if p is None:
            continue
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            continue
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(p) + suffix).unlink()
            except (FileNotFoundError, OSError):
                pass
    for k in ("OPENAGENT_AUTO_RECALL_ENABLED", "OPENAGENT_AUTO_RECALL_WARM_BUDGET",
              "OPENAGENT_AUTO_RECALL_MIN_SCORE", "OPENAGENT_AUTO_RECALL_TOP_K",
              "OPENAGENT_EMBEDDING_MODEL", "OPENAGENT_EMBEDDING_BASE_URL"):
        os.environ.pop(k, None)


# ── Layer A: the index ────────────────────────────────────────────────


@test("semantic_recall", "inert without an embedding model — sync no-op, search empty")
async def t_inert_without_model(ctx: TestContext) -> None:
    """The self-hosted default (§17): no ``OPENAGENT_EMBEDDING_MODEL`` means no
    semantic layer. ``resolve_embedder`` returns None, the index is inert, and
    retrieval falls back to FTS byte-identically."""
    from src.memory.semantic_index import SemanticIndex, resolve_embedder

    os.environ.pop("OPENAGENT_EMBEDDING_MODEL", None)
    assert resolve_embedder() is None, "an embedder resolved with no model configured"

    db, idx_path, vault = _paths(ctx, "inert")
    try:
        _write_note(vault, "n1", "Anything", "some body text about anything")
        idx = SemanticIndex(db, vault_root=vault, index_path=idx_path, embedder=None)
        assert idx.active is False
        stats = idx.sync()
        assert stats["vault"].embedded == 0 and stats["sessions"].embedded == 0
        assert idx.search("anything at all") == [], "inert index returned hits"
        assert idx.stats()["notes"] == 0
        idx.close()
    finally:
        _cleanup(db, idx_path, vault)


@test("semantic_recall", "SEMANTIC beats KEYWORD on a paraphrase (the whole justification)")
async def t_semantic_beats_keyword(ctx: TestContext) -> None:
    """The headline. A session says "the ship date we agreed"; the query is
    "launch deadline" — zero shared words. Real FTS (``TranscriptIndex``) MISSES
    it; the semantic index FINDS it. Keyword and semantic are complements, and
    this is the case only semantic can serve."""
    from src.memory.semantic_index import SemanticIndex
    from src.memory.transcript_index import TranscriptIndex

    db, idx_path, vault = _paths(ctx, "win")
    ti_path = ctx.db_path.with_name(f"sr-win-{uuid.uuid4().hex[:8]}-fts.db")
    try:
        d = await _open_db(db)
        await _seed_session(d, "chat-a", [
            ("user", "When is the ship date we agreed for the react migration?"),
            ("assistant", "We settled on targeting it for the fifteenth."),
        ], title="Migration planning")
        await d.close()

        # Keyword FTS: a real miss — the WORDS "launch"/"deadline" are absent.
        ti = TranscriptIndex(db, ti_path)
        ti.sync()
        assert ti.search("launch deadline") == [], (
            "precondition broken: FTS was supposed to MISS the paraphrase"
        )
        ti.close()

        # Semantic: a hit, because "ship date" and "launch deadline" share the
        # meaning even though they share no words.
        idx = SemanticIndex(db, vault_root=vault, index_path=idx_path,
                            embedder=ConceptEmbedder())
        idx.sync()
        hits = idx.search("launch deadline", scope="sessions", min_score=0.5)
        assert hits, f"SEMANTIC MISS on a clear paraphrase — the whole point: {hits}"
        assert hits[0]["session_id"] == "chat-a", hits
        assert hits[0]["score"] >= 0.5, hits
        idx.close()
    finally:
        _cleanup(db, idx_path, vault, ti_path)


@test("semantic_recall", "the index is a rebuildable cache; sync is incremental")
async def t_rebuildable_and_incremental(ctx: TestContext) -> None:
    """§5's contract, restated for embeddings: delete the file, the next sync
    rebuilds it; an unchanged corpus re-embeds NOTHING (each embed is a network
    call, so the gate matters more here than for FTS)."""
    from src.memory.semantic_index import SemanticIndex

    db, idx_path, vault = _paths(ctx, "rebuild")
    try:
        _write_note(vault, "bread", "Sourdough", "bake bread with a long proof")
        _write_note(vault, "ship", "Delivery", "the tungsten shipment arrives thursday")
        d = await _open_db(db)
        await _seed_session(d, "chat-x", [("user", "customer complained about a refund")])
        await d.close()

        idx = SemanticIndex(db, vault_root=vault, index_path=idx_path,
                            embedder=ConceptEmbedder())
        first = idx.sync()
        assert first["vault"].embedded == 2 and first["sessions"].embedded == 1, first

        # Incremental: nothing changed → nothing re-embedded.
        second = idx.sync()
        assert second["vault"].embedded == 0 and second["sessions"].embedded == 0, second
        assert second["vault"].unchanged == 2 and second["sessions"].unchanged == 1

        # Touch one note → only it re-embeds.
        _time.sleep(0.01)
        _write_note(vault, "bread", "Sourdough", "bake bread with a SHORT proof")
        third = idx.sync()
        assert third["vault"].embedded == 1 and third["vault"].updated == 1, third

        assert idx_path.exists()
        idx.close()

        # Nuke the cache exactly as an operator would — it rebuilds from source.
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(idx_path) + suffix).unlink()
            except FileNotFoundError:
                pass
        assert not idx_path.exists()
        idx2 = SemanticIndex(db, vault_root=vault, index_path=idx_path,
                            embedder=ConceptEmbedder())
        rebuilt = idx2.sync()
        assert rebuilt["vault"].embedded == 2 and rebuilt["sessions"].embedded == 1, rebuilt
        assert idx2.search("customer complaint", scope="sessions", min_score=0.5), (
            "index did not rebuild after deletion"
        )
        idx2.close()
    finally:
        _cleanup(db, idx_path, vault)


@test("semantic_recall", "a purged session and a deleted note stop being findable")
async def t_purge_propagation(ctx: TestContext) -> None:
    """Mirrors ``test_session_delete``'s invariant: deletion at the source must
    propagate to the cache on the next sync — a PULL, so ``db.py`` needs to know
    nothing about this index. Uses the REAL ``purge_session``."""
    from src.memory.semantic_index import SemanticIndex

    db, idx_path, vault = _paths(ctx, "purge")
    try:
        secret_note = _write_note(vault, "secret", "Passphrase", "the tungsten shipment secret")
        d = await _open_db(db)
        await _seed_session(d, "chat-secret", [
            ("user", "the customer complained and demanded a refund")])
        await d.close()

        idx = SemanticIndex(db, vault_root=vault, index_path=idx_path,
                            embedder=ConceptEmbedder())
        idx.sync()
        assert idx.search("customer complaint", scope="sessions", min_score=0.5)
        assert idx.search("tungsten delivery", scope="vault", min_score=0.5)

        # Purge the session (real code) and delete the note file.
        d = await _open_db(db)
        await d.purge_session("chat-secret")
        await d.close()
        secret_note.unlink()

        after = idx.sync()
        assert after["sessions"].deleted == 1 and after["vault"].deleted == 1, after
        assert idx.search("customer complaint", scope="sessions", min_score=0.5) == [], (
            "PURGED SESSION STILL FINDABLE via semantic search"
        )
        assert idx.search("tungsten delivery", scope="vault", min_score=0.5) == [], (
            "DELETED NOTE STILL FINDABLE via semantic search"
        )
        idx.close()
    finally:
        _cleanup(db, idx_path, vault)


@test("semantic_recall", "changing the embedding model wipes the cache")
async def t_model_change_wipes(ctx: TestContext) -> None:
    """Vectors from two models live in different spaces; keeping both would rank
    by noise. A model change is a rebuild, cheap because the source text stayed."""
    from src.memory.semantic_index import SemanticIndex

    db, idx_path, vault = _paths(ctx, "model")
    try:
        _write_note(vault, "n", "Bread", "bake sourdough bread")
        idx = SemanticIndex(db, vault_root=vault, index_path=idx_path,
                            embedder=ConceptEmbedder())
        idx.sync()
        assert idx.stats()["notes"] == 1
        idx.close()

        # Reopen with a DIFFERENT model id → old vectors are wiped on open.
        idx2 = SemanticIndex(db, vault_root=vault, index_path=idx_path,
                            embedder=_OtherModelEmbedder())
        assert idx2.stats()["notes"] == 0, "stale vectors survived a model change"
        idx2.sync()
        assert idx2.stats()["notes"] == 1
        idx2.close()
    finally:
        _cleanup(db, idx_path, vault)


@test("semantic_recall", "default index path is keyed to the db path, not platform defaults")
async def t_index_path_keyed_to_db(ctx: TestContext) -> None:
    from src.memory.semantic_index import default_semantic_index_path

    root = ctx.db_path.parent
    a = default_semantic_index_path(root / "agent-one" / "openagent.db")
    b = default_semantic_index_path(root / "agent-two" / "openagent.db")
    c = default_semantic_index_path(root / "agent-one" / "other.db")
    assert a != b and a != c
    assert a.name.startswith("semantic_index_")
    assert a == default_semantic_index_path(root / "agent-one" / "." / "openagent.db")


# ── Layer B: the auto-recall hook ─────────────────────────────────────


def _fake_agent(db_path: Path, vault: Path):
    from types import SimpleNamespace
    return SimpleNamespace(
        _db=SimpleNamespace(db_path=str(db_path)),
        _providers_config=[],
        _resolve_vault_path=lambda: str(vault),
    )


def _prime_recall_cache(db_path: Path, vault: Path) -> Any:
    """Pre-build a fake-embedder index and install it in the hook's cache, so
    ``_with_recall`` uses it without needing a live endpoint."""
    import src.core.agent as agent_mod
    from src.memory.semantic_index import SemanticIndex

    idx = SemanticIndex(db_path, vault_root=vault, embedder=ConceptEmbedder())
    idx.sync()
    agent_mod._RECALL_INDEX_CACHE[str(db_path)] = idx
    return idx


@test("semantic_recall", "auto-recall is OFF by default — injects nothing")
async def t_auto_recall_off_by_default(ctx: TestContext) -> None:
    import src.core.agent as agent_mod

    db, idx_path, vault = _paths(ctx, "off")
    try:
        d = await _open_db(db)
        await _seed_session(d, "chat", [("user", "customer complained about refund")])
        await d.close()
        idx = _prime_recall_cache(db, vault)  # warm, but flag is OFF

        os.environ.pop("OPENAGENT_AUTO_RECALL_ENABLED", None)
        out = await agent_mod._with_recall(
            _fake_agent(db, vault), "sess", "customer complaint", "USER MESSAGE")
        assert out == "USER MESSAGE", f"recall fired while disabled: {out!r}"
        idx.close()
    finally:
        agent_mod._RECALL_INDEX_CACHE.pop(str(db), None)
        _cleanup(db, idx_path, vault)


@test("semantic_recall", "auto-recall is inert with no embedding model even when enabled")
async def t_auto_recall_inert_without_model(ctx: TestContext) -> None:
    """Enabled + no model = still nothing. ``_get_recall_index`` resolves no
    embedder and returns None, so the turn text is byte-identical."""
    import src.core.agent as agent_mod

    db, idx_path, vault = _paths(ctx, "noembed")
    try:
        agent_mod._RECALL_INDEX_CACHE.pop(str(db), None)  # force live resolution
        os.environ["OPENAGENT_AUTO_RECALL_ENABLED"] = "1"
        os.environ.pop("OPENAGENT_EMBEDDING_MODEL", None)
        out = await agent_mod._with_recall(
            _fake_agent(db, vault), "sess", "customer complaint", "USER MESSAGE")
        assert out == "USER MESSAGE", f"recall fired with no embedder: {out!r}"
    finally:
        agent_mod._RECALL_INDEX_CACHE.pop(str(db), None)
        _cleanup(db, idx_path, vault)


@test("semantic_recall", "a strong match injects a bounded, verify-framed block")
async def t_auto_recall_strong_match(ctx: TestContext) -> None:
    import src.core.agent as agent_mod

    db, idx_path, vault = _paths(ctx, "strong")
    try:
        _write_note(vault, "acme", "Acme refund dispute",
                    "the customer complained and demanded a refund")
        d = await _open_db(db)
        await d.close()
        idx = _prime_recall_cache(db, vault)

        os.environ["OPENAGENT_AUTO_RECALL_ENABLED"] = "1"
        os.environ["OPENAGENT_AUTO_RECALL_WARM_BUDGET"] = "0"  # already warm
        os.environ["OPENAGENT_AUTO_RECALL_MIN_SCORE"] = "0.5"
        out = await agent_mod._with_recall(
            _fake_agent(db, vault), "sess",
            "has this customer complained before?", "USER MESSAGE")

        assert out != "USER MESSAGE", "strong match injected nothing"
        assert out.endswith("USER MESSAGE"), "user message must stay at the end"
        assert "<system-reminder>" in out and "</system-reminder>" in out
        # Framed as unverified, NOT asserted as fact — the safety property.
        assert "UNVERIFIED" in out and "acme.md" in out
        assert "similarity" in out.lower()
        # Bounded.
        assert len(out) < 4000, f"injected block unbounded: {len(out)}"
        idx.close()
    finally:
        agent_mod._RECALL_INDEX_CACHE.pop(str(db), None)
        _cleanup(db, idx_path, vault)


@test("semantic_recall", "a weak match injects nothing (thresholded — no noise)")
async def t_auto_recall_weak_match(ctx: TestContext) -> None:
    """The vault is full of orphans and contradictions; a weak match must inject
    NOTHING or auto-recall becomes a hallucination engine. The floor is what
    keeps stale notes out."""
    import src.core.agent as agent_mod

    db, idx_path, vault = _paths(ctx, "weak")
    try:
        _write_note(vault, "acme", "Acme refund dispute",
                    "the customer complained and demanded a refund")
        d = await _open_db(db)
        await d.close()
        idx = _prime_recall_cache(db, vault)

        os.environ["OPENAGENT_AUTO_RECALL_ENABLED"] = "1"
        os.environ["OPENAGENT_AUTO_RECALL_WARM_BUDGET"] = "0"
        os.environ["OPENAGENT_AUTO_RECALL_MIN_SCORE"] = "0.75"
        # A query about something unrelated — no shared concept, near-orthogonal.
        out = await agent_mod._with_recall(
            _fake_agent(db, vault), "sess",
            "photosynthesis rates in alpine ferns", "USER MESSAGE")
        assert out == "USER MESSAGE", (
            f"a weak match leaked into the turn — threshold not enforced: {out!r}"
        )
        idx.close()
    finally:
        agent_mod._RECALL_INDEX_CACHE.pop(str(db), None)
        _cleanup(db, idx_path, vault)


@test("semantic_recall", "the injected block is on the user path, NOT the cached system prefix")
async def t_auto_recall_cache_safe(ctx: TestContext) -> None:
    """Per-turn content in the ~10.8k-token cached system prefix busts the cache
    every turn (the regression the ``<session-id>`` split guards). The recall
    block lands on the USER message; the system prompt never sees it, and its
    trailing tag still splits cleanly."""
    import src.core.agent as agent_mod
    from src.models.providers.anthropic.claude import _split_session_id_tag

    db, idx_path, vault = _paths(ctx, "cachesafe")
    try:
        _write_note(vault, "acme", "Acme refund dispute",
                    "the customer complained and demanded a refund")
        d = await _open_db(db)
        await d.close()
        idx = _prime_recall_cache(db, vault)

        os.environ["OPENAGENT_AUTO_RECALL_ENABLED"] = "1"
        os.environ["OPENAGENT_AUTO_RECALL_WARM_BUDGET"] = "0"
        os.environ["OPENAGENT_AUTO_RECALL_MIN_SCORE"] = "0.5"
        user_out = await agent_mod._with_recall(
            _fake_agent(db, vault), "sess",
            "has this customer complained before?", "USER MESSAGE")
        assert "<system-reminder>" in user_out  # it DID inject, on the user path

        # The REAL combined system prompt (fake model) must NOT contain it, and
        # the <session-id> tag must still be the trailing token.
        from types import SimpleNamespace
        from src.core.agent import Agent
        agent = Agent.__new__(Agent)
        agent.system_prompt = ""
        agent._mcp = SimpleNamespace()
        agent._resolve_vault_path = lambda: str(vault)
        agent._resolve_db_path = lambda: str(db)
        orig = agent_mod.build_mcp_catalog_summary
        agent_mod.build_mcp_catalog_summary = lambda _pool: "(catalog)"
        try:
            system = agent._combined_system_prompt(session_id="tg:42")
        finally:
            agent_mod.build_mcp_catalog_summary = orig

        # The unique recall-block header must be absent from the system prefix.
        # (A bare "<system-reminder>"/"recall" check would be wrong — the
        # framework prompt legitimately DOCUMENTS both in its prose.)
        assert "Possibly-relevant memory" not in system, (
            "recall block leaked into the CACHED system prefix — busts the cache"
        )
        assert "[similarity" not in system, "a recall hit line leaked into the prefix"
        _body, tag = _split_session_id_tag(system)
        assert tag == "<session-id>tg:42</session-id>", (
            "the session-id split broke — cache tail no longer isolates"
        )
        idx.close()
    finally:
        agent_mod._RECALL_INDEX_CACHE.pop(str(db), None)
        _cleanup(db, idx_path, vault)


# ── the plumbing: env forwarding to the memory-search subprocess ──────


@test("semantic_recall", "resolve_default_entry forwards embedding + DB/vault env to memory-search")
async def t_memory_search_env_forwarded(ctx: TestContext) -> None:
    """The MCP SDK spawns a subprocess with a minimal env, so an operator's
    ``OPENAGENT_EMBEDDING_MODEL`` does NOT inherit — ``resolve_default_entry``
    must inject it (alongside DB_PATH for sessions and VAULT_PATH for notes),
    or ``semantic_recall`` stays inert in the subprocess even when configured.
    This locks that wiring so it can't silently regress to keyword-only."""
    from src.mcp.builtins import resolve_default_entry

    saved = {k: os.environ.get(k) for k in (
        "OPENAGENT_EMBEDDING_MODEL", "OPENAGENT_EMBEDDING_BASE_URL",
        "OPENAGENT_EMBEDDING_API_KEY")}
    try:
        os.environ["OPENAGENT_EMBEDDING_MODEL"] = "local:nomic-embed-text"
        os.environ["OPENAGENT_EMBEDDING_BASE_URL"] = "http://localhost:11434/v1"
        os.environ.pop("OPENAGENT_EMBEDDING_API_KEY", None)  # unset must NOT forward

        resolved = resolve_default_entry({"builtin": "memory-search"}, db_path="/tmp/x.db")
        assert resolved is not None, "memory-search resolved to None (Node/deps missing?)"
        env = resolved.get("env") or {}

        assert env.get("OPENAGENT_EMBEDDING_MODEL") == "local:nomic-embed-text", (
            "embedding model not forwarded — semantic_recall would be inert in the subprocess")
        assert env.get("OPENAGENT_EMBEDDING_BASE_URL") == "http://localhost:11434/v1", (
            "embedding base_url not forwarded — the ONLY channel that reaches the subprocess")
        assert "OPENAGENT_EMBEDDING_API_KEY" not in env, (
            "an unset embedding var must not be forwarded as empty")
        assert os.path.basename(env.get("OPENAGENT_DB_PATH", "")) == "x.db", (
            "DB_PATH not forwarded — semantic index over sessions can't find the DB")
        assert env.get("OPENAGENT_VAULT_PATH"), (
            "VAULT_PATH not forwarded — semantic index over notes can't find the vault")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@test("semantic_recall", "unconfigured embedding env is NOT forwarded (subprocess stays inert)")
async def t_memory_search_env_absent_when_unset(ctx: TestContext) -> None:
    """The inverse guard: with no embedding model configured, nothing embedding-
    related is forwarded, so the subprocess resolves ``None`` and degrades to
    keyword-only — the proven self-hosted default (§17), not a half-wired state."""
    from src.mcp.builtins import resolve_default_entry

    saved = {k: os.environ.get(k) for k in (
        "OPENAGENT_EMBEDDING_MODEL", "OPENAGENT_EMBEDDING_BASE_URL")}
    try:
        os.environ.pop("OPENAGENT_EMBEDDING_MODEL", None)
        os.environ.pop("OPENAGENT_EMBEDDING_BASE_URL", None)

        resolved = resolve_default_entry({"builtin": "memory-search"}, db_path="/tmp/x.db")
        assert resolved is not None
        env = resolved.get("env") or {}
        assert "OPENAGENT_EMBEDDING_MODEL" not in env
        assert "OPENAGENT_EMBEDDING_BASE_URL" not in env
        # DB/VAULT are unconditional (FTS needs them); only embedding is gated.
        assert env.get("OPENAGENT_DB_PATH"), "DB_PATH must still forward for FTS"
        assert env.get("OPENAGENT_VAULT_PATH"), "VAULT_PATH must still forward for FTS"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@test("semantic_recall", "the numpy-FREE fallback finds the same matches (bundle-proof)")
async def t_numpy_free_search(ctx: TestContext) -> None:
    """numpy shipped ABSENT from the frozen bundle twice, silently breaking
    recall with 'No module named numpy'. So the cosine is numpy-OPTIONAL: this
    forces the pure-Python path and asserts it still finds a strong match and
    respects the threshold — recall can never again be disabled by a missing C
    library."""
    import src.memory.semantic_index as si
    from src.memory.semantic_index import SemanticIndex

    db, idx_path, vault = _paths(ctx, "nonumpy")
    saved = si._HAS_NUMPY
    try:
        _write_note(vault, "deadline", "Launch deadline",
                    "the ship date we agreed for the migration")
        d = await _open_db(db); await d.close()
        si._HAS_NUMPY = False   # force the pure-Python dot-product path
        idx = SemanticIndex(db, vault_root=vault, index_path=idx_path,
                            embedder=ConceptEmbedder())
        idx.sync()
        assert idx.stats()["notes"] == 1, "pure-Python path failed to store a vector"
        # paraphrase with zero shared words must still match via the concept embedder
        hits = idx.search("when is the launch target", limit=3, min_score=0.3)
        assert hits and hits[0]["path"].endswith("deadline.md"), (
            f"numpy-free search missed the match: {hits}")
        # threshold still bites without numpy
        assert idx.search("sourdough bread baking", limit=3, min_score=0.5) == [], (
            "numpy-free path ignored the min_score floor")
        idx.close()
    finally:
        si._HAS_NUMPY = saved
        _cleanup(db, idx_path, vault)


@test("semantic_recall", "the background builder fills the index off the turn path")
async def t_background_builder(ctx: TestContext) -> None:
    """The on-turn hook is time-boxed and can't build a 2000-note index — it did
    so nowhere in prod, leaving recall empty. The builder does it in the
    background, un-time-boxed. Prove it embeds every note + is a no-op when inert."""
    import src.memory.semantic_index as si
    from src.memory import semantic_index_builder as bld

    db, idx_path, vault = _paths(ctx, "builder")
    try:
        for i in range(5):
            _write_note(vault, f"n{i}", f"Note {i}", f"content about topic {i} launch deadline")
        d = await _open_db(db); await d.close()

        # inert: no embedder → start() returns a task that exits immediately, no index
        import os as _os
        _os.environ.pop("OPENAGENT_EMBEDDING_MODEL", None)
        # ── build with a real (fake) embedder, forced via monkeypatch ──
        import src.memory.semantic_index as si_mod
        orig = si_mod.resolve_embedder
        si_mod.resolve_embedder = lambda *a, **k: ConceptEmbedder()
        try:
            import asyncio
            # short resync so the loop keeps running but we only need the first build
            _os.environ["OPENAGENT_SEMANTIC_RESYNC_SECONDS"] = "30"
            task = bld.start(str(db), str(vault), None)
            assert task is not None
            # wait for the first build to embed all 5 notes (well under one pass)
            for _ in range(40):
                await asyncio.sleep(0.05)
                idx = si_mod.SemanticIndex(str(db), vault_root=str(vault), embedder=ConceptEmbedder(),
                                           index_path=idx_path if False else None)
                n = idx.stats()["notes"]; idx.close()
                if n >= 5:
                    break
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            assert n >= 5, f"builder did not embed all notes: {n}/5"
        finally:
            si_mod.resolve_embedder = orig
            _os.environ.pop("OPENAGENT_SEMANTIC_RESYNC_SECONDS", None)
    finally:
        _cleanup(db, idx_path, vault)


# ── Hybrid recall (FTS ∪ semantic) ────────────────────────────────────
# The eval finding: semantic cosine bands overlap (nomic ~0.59–0.83) so no
# min_score cleanly separates relevant from noise — a refund-policy note scored
# 0.592, BELOW the floor, while a generic thread scored 0.604 above it. FTS
# keyword search catches the exact term the floor drops; hybrid fuses both.

def _prime_fts_cache(vault: Path, index_path: Path) -> Any:
    """Build a real FTS ``VaultIndex`` over ``vault`` and install it in the
    hybrid hook's cache, so ``_recall_block`` uses it without touching
    ``data_dir()`` (mirrors ``_prime_recall_cache`` for the semantic side)."""
    import src.core.agent as agent_mod
    from src.memory.vault.index import VaultIndex

    fts = VaultIndex(vault, index_path)
    fts.sync()
    agent_mod._FTS_INDEX_CACHE[str(vault)] = fts
    return fts


def _clear_recall_env() -> None:
    # Prefix-based so it also clears the scoping knobs (SCOPE / INCLUDE_PATHS /
    # EXCLUDE_PATHS / RESERVE_PREFIX) and their per-origin ``_<ORIGIN>`` suffixes.
    for _k in [k for k in os.environ if k.startswith("OPENAGENT_AUTO_RECALL_")]:
        os.environ.pop(_k, None)


def _write_note_at(vault: Path, relpath: str, title: str, body: str) -> Path:
    """Write a note at a vault-RELATIVE path (creating parent dirs) so tests can
    exercise path-prefix corpus scoping."""
    p = vault / relpath
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"---\ntitle: {title}\n---\n{body}\n")
    return p


@test("semantic_recall", "recall scoping helpers: origin parse, per-origin env selection, path filter")
async def t_recall_scoping_helpers(ctx: TestContext) -> None:
    from src.core.agent import _origin_of, _recall_scoping, _path_allowed

    assert _origin_of("event:aa:bb") == "event"
    assert _origin_of("scheduler:x") == "scheduler"
    assert _origin_of("bare") == "" and _origin_of(None) == ""

    assert _path_allowed("esound/procedures/refund.md", ["esound/"], []) is True
    assert _path_allowed("devops/build.md", ["esound/"], []) is False
    assert _path_allowed("esound/x.md", [], ["esound/triage/"]) is True
    assert _path_allowed("esound/triage/t.md", [], ["esound/triage/"]) is False

    _clear_recall_env()
    try:
        # unconfigured = identity (byte-identical to pre-scoping behaviour)
        # `reserve` e' una LISTA dal 26-ago-2026: due sottoalberi autorevoli
        # non possono contendersi un posto solo, o a ogni turno uno dei due
        # sparisce in silenzio. Il default resta l'identita'.
        assert _recall_scoping("event") == ("all", [], [], [])
        # default applies to every origin
        os.environ["OPENAGENT_AUTO_RECALL_EXCLUDE_PATHS"] = "devops/, arc/"
        assert _recall_scoping("chat")[2] == ["devops/", "arc/"]
        # per-origin override wins for that origin only; others keep the default
        os.environ["OPENAGENT_AUTO_RECALL_SCOPE_EVENT"] = "vault"
        os.environ["OPENAGENT_AUTO_RECALL_EXCLUDE_PATHS_EVENT"] = "devops/"
        assert _recall_scoping("event")[0] == "vault"
        assert _recall_scoping("event")[2] == ["devops/"]
        assert _recall_scoping("chat")[0] == "all"          # default, not the EVENT override
        assert _recall_scoping("chat")[2] == ["devops/", "arc/"]
        # an unknown scope degrades to 'all'
        os.environ["OPENAGENT_AUTO_RECALL_SCOPE"] = "bogus"
        assert _recall_scoping("chat")[0] == "all"
    finally:
        _clear_recall_env()


@test("semantic_recall", "SemanticIndex.search filters notes by include/exclude path prefixes")
async def t_search_path_prefix_filter(ctx: TestContext) -> None:
    from src.memory.semantic_index import SemanticIndex

    db, idx_path, vault = _paths(ctx, "prefix")
    try:
        _write_note_at(vault, "esound/procedures/refund.md", "Refund",
                       "the customer demanded a refund")
        _write_note_at(vault, "devops/build.md", "Build",
                       "the customer demanded a refund pipeline build")
        idx = SemanticIndex(db, vault_root=vault, index_path=idx_path,
                            embedder=ConceptEmbedder())
        idx.sync()
        q = "customer refund"
        allp = {h["path"] for h in idx.search(q, scope="vault", limit=5, min_score=0.0)}
        assert any("devops/build.md" in p for p in allp), f"devops note not matched at all: {allp}"
        exc = idx.search(q, scope="vault", limit=5, min_score=0.0,
                         exclude_prefixes=["devops/"])
        assert exc and all("devops/" not in h["path"] for h in exc), f"exclude leaked: {exc}"
        inc = idx.search(q, scope="vault", limit=5, min_score=0.0,
                         include_prefixes=["esound/procedures/"])
        assert inc and all(h["path"].startswith("esound/procedures/") for h in inc), \
            f"include admitted a non-matching prefix: {inc}"
        idx.close()
    finally:
        _cleanup(db, idx_path, vault)


@test("semantic_recall", "per-origin scoping: an event turn drops a dev-ops note a chat turn keeps")
async def t_per_origin_scoping_e2e(ctx: TestContext) -> None:
    import src.core.agent as agent_mod

    db, idx_path, vault = _paths(ctx, "origin")
    try:
        _write_note_at(vault, "devops/build.md", "Build ops",
                       "the customer complained about the refund build pipeline")
        d = await _open_db(db)
        await d.close()
        idx = _prime_recall_cache(db, vault)
        _clear_recall_env()
        try:
            os.environ["OPENAGENT_AUTO_RECALL_ENABLED"] = "1"
            os.environ["OPENAGENT_AUTO_RECALL_WARM_BUDGET"] = "0"
            os.environ["OPENAGENT_AUTO_RECALL_MIN_SCORE"] = "0.5"
            os.environ["OPENAGENT_AUTO_RECALL_HYBRID"] = "0"   # isolate the corpus filter
            os.environ["OPENAGENT_AUTO_RECALL_EXCLUDE_PATHS_EVENT"] = "devops/"
            q = "customer complained about a refund"
            chat = await agent_mod._with_recall(_fake_agent(db, vault), "chat:1", q, "MSG")
            assert "devops/build.md" in chat, f"chat turn should KEEP the note: {chat!r}"
            ev = await agent_mod._with_recall(_fake_agent(db, vault), "event:abc:def", q, "MSG")
            assert "devops/build.md" not in ev, f"event turn must DROP the devops note: {ev!r}"
        finally:
            _clear_recall_env()
        idx.close()
    finally:
        agent_mod._RECALL_INDEX_CACHE.pop(str(db), None)
        _cleanup(db, idx_path, vault)


@test("semantic_recall", "reserve_prefix surfaces the authoritative playbook alongside precedent")
async def t_reserve_prefix(ctx: TestContext) -> None:
    import src.core.agent as agent_mod

    db, idx_path, vault = _paths(ctx, "reserve")
    try:
        # A prior-thread precedent that matches the query strongly, and a playbook
        # under a reserve prefix that DOESN'T match the query (would be dropped).
        _write_note_at(vault, "esound/triage/thread-1.md", "Prior thread",
                       "the customer complained and demanded a refund twice")
        _write_note_at(vault, "esound/procedures/customer-response/refund-policy.md",
                       "Dashboard config", "quarterly metrics dashboard configuration")
        d = await _open_db(db)
        await d.close()
        idx = _prime_recall_cache(db, vault)
        _clear_recall_env()
        try:
            os.environ["OPENAGENT_AUTO_RECALL_ENABLED"] = "1"
            os.environ["OPENAGENT_AUTO_RECALL_WARM_BUDGET"] = "0"
            os.environ["OPENAGENT_AUTO_RECALL_MIN_SCORE"] = "0.5"
            os.environ["OPENAGENT_AUTO_RECALL_HYBRID"] = "0"
            os.environ["OPENAGENT_AUTO_RECALL_TOP_K"] = "2"
            q = "customer complained and demanded a refund"
            base = await agent_mod._with_recall(_fake_agent(db, vault), "event:1", q, "MSG")
            assert "thread-1.md" in base, f"precedent should surface: {base!r}"
            assert "refund-policy.md" not in base, \
                f"the off-topic playbook should NOT surface on its own: {base!r}"
            os.environ["OPENAGENT_AUTO_RECALL_RESERVE_PREFIX"] = \
                "esound/procedures/customer-response/"
            withres = await agent_mod._with_recall(_fake_agent(db, vault), "event:1", q, "MSG")
            assert "refund-policy.md" in withres, \
                f"reserve failed to surface the playbook: {withres!r}"
            assert "thread-1.md" in withres, \
                f"reserve must keep the precedent ALONGSIDE the playbook: {withres!r}"
        finally:
            _clear_recall_env()
        idx.close()
    finally:
        agent_mod._RECALL_INDEX_CACHE.pop(str(db), None)
        _cleanup(db, idx_path, vault)


@test("semantic_recall", "hybrid: an FTS keyword hit BELOW the semantic floor is still recalled (the eval finding)")
async def t_hybrid_rescues_below_floor(ctx: TestContext) -> None:
    """The whole justification: pin ``min_score`` so high the semantic side
    returns NOTHING, and show the note is injected anyway because FTS matched the
    exact term — then that with hybrid OFF it is lost (the pre-hybrid regression
    the eval measured on the refund rule)."""
    import src.core.agent as agent_mod

    db, idx_path, vault = _paths(ctx, "hybrid-floor")
    fts_path = ctx.db_path.with_name(f"{vault.name}-fts.db")
    try:
        _write_note(vault, "refund-policy", "Refund policy",
                    "refund double charges within 14 days via Stripe")
        d = await _open_db(db); await d.close()
        sem = _prime_recall_cache(db, vault)     # semantic index active
        fts = _prime_fts_cache(vault, fts_path)  # FTS index active

        os.environ["OPENAGENT_AUTO_RECALL_ENABLED"] = "1"
        os.environ["OPENAGENT_AUTO_RECALL_WARM_BUDGET"] = "0"
        os.environ["OPENAGENT_AUTO_RECALL_MIN_SCORE"] = "0.999"  # semantic → nothing
        os.environ["OPENAGENT_AUTO_RECALL_HYBRID"] = "1"
        out = await agent_mod._with_recall(
            _fake_agent(db, vault), "sess",
            "customer wants a refund", "USER MESSAGE")
        assert "refund-policy.md" in out, \
            f"FTS hit below the semantic floor was NOT recalled: {out!r}"
        assert "keyword match" in out, f"FTS-only hit not tagged: {out!r}"
        assert out.endswith("USER MESSAGE")

        # Hybrid OFF → the below-floor note is lost, exactly the regression.
        os.environ["OPENAGENT_AUTO_RECALL_HYBRID"] = "0"
        out2 = await agent_mod._with_recall(
            _fake_agent(db, vault), "sess",
            "customer wants a refund", "USER MESSAGE")
        assert out2 == "USER MESSAGE", \
            f"semantic-only leaked/rescued a below-floor hit: {out2!r}"
        sem.close(); fts.close()
    finally:
        agent_mod._RECALL_INDEX_CACHE.pop(str(db), None)
        agent_mod._FTS_INDEX_CACHE.pop(str(vault), None)
        _clear_recall_env()
        _cleanup(db, idx_path, vault, fts_path)


@test("semantic_recall", "hybrid: a note found by BOTH sides is deduped and tagged as both")
async def t_hybrid_dedup(ctx: TestContext) -> None:
    import src.core.agent as agent_mod

    db, idx_path, vault = _paths(ctx, "hybrid-dedup")
    fts_path = ctx.db_path.with_name(f"{vault.name}-fts.db")
    try:
        _write_note(vault, "acme", "Acme refund dispute",
                    "the customer complained and demanded a refund")
        d = await _open_db(db); await d.close()
        sem = _prime_recall_cache(db, vault)
        fts = _prime_fts_cache(vault, fts_path)

        os.environ["OPENAGENT_AUTO_RECALL_ENABLED"] = "1"
        os.environ["OPENAGENT_AUTO_RECALL_WARM_BUDGET"] = "0"
        os.environ["OPENAGENT_AUTO_RECALL_MIN_SCORE"] = "0.4"  # semantic returns it too
        os.environ["OPENAGENT_AUTO_RECALL_HYBRID"] = "1"
        out = await agent_mod._with_recall(
            _fake_agent(db, vault), "sess",
            "customer complained refund", "USER MESSAGE")
        assert out.count("`acme.md`") == 1, \
            f"note not deduped across FTS+semantic: {out!r}"
        assert "+ keyword" in out, \
            f"a both-sides hit should read 'similarity X + keyword': {out!r}"
        sem.close(); fts.close()
    finally:
        agent_mod._RECALL_INDEX_CACHE.pop(str(db), None)
        agent_mod._FTS_INDEX_CACHE.pop(str(vault), None)
        _clear_recall_env()
        _cleanup(db, idx_path, vault, fts_path)


@test("semantic_recall", "hybrid degrades to FTS-only when the embedder is down")
async def t_hybrid_degrades_no_embedder(ctx: TestContext) -> None:
    """Embedder unreachable → ``_get_recall_index`` is None → recall must still
    work off FTS alone (§17). The pre-hybrid code bailed here and injected
    nothing."""
    import src.core.agent as agent_mod

    db, idx_path, vault = _paths(ctx, "hybrid-noembed")
    fts_path = ctx.db_path.with_name(f"{vault.name}-fts.db")
    try:
        _write_note(vault, "refund-policy", "Refund policy",
                    "refund double charges within 14 days")
        agent_mod._RECALL_INDEX_CACHE.pop(str(db), None)  # force live resolution
        os.environ.pop("OPENAGENT_EMBEDDING_MODEL", None)  # embedder inert
        fts = _prime_fts_cache(vault, fts_path)

        os.environ["OPENAGENT_AUTO_RECALL_ENABLED"] = "1"
        os.environ["OPENAGENT_AUTO_RECALL_WARM_BUDGET"] = "0"
        os.environ["OPENAGENT_AUTO_RECALL_HYBRID"] = "1"
        out = await agent_mod._with_recall(
            _fake_agent(db, vault), "sess", "refund policy", "USER MESSAGE")
        assert "refund-policy.md" in out, \
            f"FTS-only recall failed with the embedder down: {out!r}"
        assert "keyword match" in out
        fts.close()
    finally:
        agent_mod._RECALL_INDEX_CACHE.pop(str(db), None)
        agent_mod._FTS_INDEX_CACHE.pop(str(vault), None)
        _clear_recall_env()
        _cleanup(db, idx_path, vault, fts_path)


@test("semantic_recall", "hybrid degrades to semantic-only when the FTS index is unavailable")
async def t_hybrid_degrades_no_fts(ctx: TestContext) -> None:
    import src.core.agent as agent_mod

    db, idx_path, vault = _paths(ctx, "hybrid-nofts")
    orig = agent_mod._get_vault_fts_index
    try:
        _write_note(vault, "acme", "Acme refund dispute",
                    "the customer complained and demanded a refund")
        d = await _open_db(db); await d.close()
        sem = _prime_recall_cache(db, vault)
        agent_mod._get_vault_fts_index = lambda _a: None  # FTS unavailable

        os.environ["OPENAGENT_AUTO_RECALL_ENABLED"] = "1"
        os.environ["OPENAGENT_AUTO_RECALL_WARM_BUDGET"] = "0"
        os.environ["OPENAGENT_AUTO_RECALL_MIN_SCORE"] = "0.4"
        os.environ["OPENAGENT_AUTO_RECALL_HYBRID"] = "1"
        out = await agent_mod._with_recall(
            _fake_agent(db, vault), "sess",
            "has this customer complained before?", "USER MESSAGE")
        assert "acme.md" in out, \
            f"semantic-only recall broke when FTS was unavailable: {out!r}"
        sem.close()
    finally:
        agent_mod._get_vault_fts_index = orig
        agent_mod._RECALL_INDEX_CACHE.pop(str(db), None)
        _clear_recall_env()
        _cleanup(db, idx_path, vault)


@test("semantic_recall", "RRF merge: dedup, FTS-only inclusion, both-sides boost, limit")
async def t_rrf_merge_unit(ctx: TestContext) -> None:
    from src.core.agent import _rrf_merge

    sem = [{"kind": "note", "path": "a.md", "score": 0.8, "title": "A"},
           {"kind": "note", "path": "b.md", "score": 0.7, "title": "B"}]
    fts = [{"path": "c.md", "title": "C"}, {"path": "a.md", "title": "A"}]
    merged = _rrf_merge(sem, fts, limit=5)
    paths = [h["path"] for h in merged]

    assert paths.count("a.md") == 1, f"a.md not deduped: {paths}"
    assert "c.md" in paths, f"FTS-only note dropped: {paths}"
    assert paths[0] == "a.md", f"both-sides hit should rank first: {paths}"

    a = next(h for h in merged if h["path"] == "a.md")
    assert a.get("fts_matched") is True and a["score"] == 0.8, \
        "both-sides hit must keep its cosine and be marked fts_matched"
    c = next(h for h in merged if h["path"] == "c.md")
    assert c["score"] is None and c.get("fts_matched") is True, \
        "FTS-only hit must have no cosine and be marked fts_matched"
    assert len(_rrf_merge(sem, fts, limit=2)) == 2, "limit not respected"


# ── Layer C: semantic SKILL recall (SKILL.md indexed by MEANING) ──────
# Gap 2: the file-backed skills subsystem gains semantic recall. sync_skills
# embeds each SKILL.md into a THIRD SemanticIndex source; search(scope="skills")
# finds a skill by a PARAPHRASE the substring scan misses; skill_search routes
# through it when an embedder is active and degrades to the substring scan when
# not; and _recall_block surfaces the top skill as a "load with skill_view" line.
# SKILL.md stays the source of truth — skill_vectors is a rebuildable cache.


def _write_skill(skills_root: Path, folder: str, name: str, description: str,
                 body: str, *, category: str = "support",
                 created_by: str = "agent", status: str | None = None) -> Path:
    d = skills_root / folder
    d.mkdir(parents=True, exist_ok=True)
    fm = (f"---\nname: {name}\ndescription: {description}\n"
          f"category: {category}\n")
    if created_by:
        fm += f"created_by: {created_by}\n"
    if status:
        fm += f"status: {status}\n"
    fm += "---\n\n" + body + "\n"
    (d / "SKILL.md").write_text(fm)
    return d


def _skills_dir(ctx: TestContext, tag: str) -> Path:
    p = ctx.db_path.with_name(f"sr-{tag}-{uuid.uuid4().hex[:8]}-skills")
    p.mkdir(exist_ok=True)
    return p


@test("semantic_recall", "sync_skills embeds SKILL.md; scope=skills matches a PARAPHRASE")
async def t_skill_index_semantic(ctx: TestContext) -> None:
    """The headline for Gap 2: a skill about a 'customer refund complaint' is
    found by the query 'angry unhappy client' — zero shared words, same meaning.
    Also proves the leg is opt-in (scope='all' never surfaces a skill), archived
    skills are not indexed, and the sync is incremental + rebuildable."""
    from src.memory.semantic_index import SemanticIndex

    db, idx_path, vault = _paths(ctx, "skillidx")
    skills = _skills_dir(ctx, "skillidx")
    try:
        _write_skill(skills, "refunds", "refund-playbook",
                     "how to handle a customer complaint about a refund",
                     "Offer the refund and apologise to the customer.")
        _write_skill(skills, "bread", "sourdough-guide",
                     "bake a good loaf of bread", "Proof the starter overnight.")
        # An archived skill must NOT be indexed (parity with the frozen index).
        _write_skill(skills, "old", "stale-refund-note",
                     "an old refund complaint skill", "Outdated.",
                     status="archived")

        idx = SemanticIndex(db, vault_root=vault, skills_root=str(skills),
                            index_path=idx_path, embedder=ConceptEmbedder())
        stats = idx.sync()
        assert "skills" in stats, "sync() dropped the skills leg despite a skills_root"
        assert stats["skills"].embedded == 2, (
            f"expected 2 live skills embedded (archived excluded): {stats['skills']}")
        assert idx.stats()["skills"] == 2

        # PARAPHRASE: shares NO words with the refund skill, only the concept.
        hits = idx.search("angry unhappy client", scope="skills", limit=3, min_score=0.4)
        assert hits, f"SEMANTIC MISS on a clear skill paraphrase: {hits}"
        assert hits[0]["name"] == "refund-playbook", hits
        assert hits[0]["kind"] == "skill" and hits[0]["score"] >= 0.4, hits

        # The archived skill never surfaces even at min_score 0.
        names = {h["name"] for h in idx.search("refund complaint", scope="skills",
                                               limit=10, min_score=0.0)}
        assert "stale-refund-note" not in names, "an archived skill leaked into recall"

        # scope="all" is byte-identical to before skills existed — NO skill hits.
        allhits = idx.search("angry unhappy client", scope="all", limit=10, min_score=0.0)
        assert all(h.get("kind") != "skill" for h in allhits), (
            "scope='all' surfaced a skill — it must stay vault+sessions only")

        # Incremental: nothing changed → nothing re-embedded.
        again = idx.sync()
        assert again["skills"].embedded == 0 and again["skills"].unchanged == 2, again

        # Touch one skill → only it re-embeds.
        _time.sleep(0.01)
        _write_skill(skills, "refunds", "refund-playbook",
                     "how to handle a customer complaint about a refund",
                     "Offer the refund FAST and apologise to the customer.")
        third = idx.sync()
        assert third["skills"].embedded == 1 and third["skills"].updated == 1, third
        idx.close()

        # Rebuildable: nuke the cache, it rebuilds from the SKILL.md files.
        for suffix in ("", "-wal", "-shm"):
            try:
                Path(str(idx_path) + suffix).unlink()
            except FileNotFoundError:
                pass
        idx2 = SemanticIndex(db, vault_root=vault, skills_root=str(skills),
                             index_path=idx_path, embedder=ConceptEmbedder())
        assert idx2.sync()["skills"].embedded == 2, "skills leg did not rebuild"
        assert idx2.search("angry unhappy client", scope="skills", min_score=0.4)
        idx2.close()
    finally:
        _cleanup(db, idx_path, vault, skills)


@test("semantic_recall", "sync_skills / scope=skills are INERT without an embedder")
async def t_skill_index_inert(ctx: TestContext) -> None:
    """The §17 fallback for the skills leg: no model → sync_skills embeds nothing
    and search(scope='skills') returns [] — no second opaque store, byte-identical
    to before the leg existed."""
    from src.memory.semantic_index import SemanticIndex

    db, idx_path, vault = _paths(ctx, "skillinert")
    skills = _skills_dir(ctx, "skillinert")
    try:
        _write_skill(skills, "refunds", "refund-playbook",
                     "customer refund complaint", "body")
        idx = SemanticIndex(db, vault_root=vault, skills_root=str(skills),
                            index_path=idx_path, embedder=None)
        assert idx.active is False
        st = idx.sync()
        assert st["skills"].embedded == 0, st
        assert idx.stats()["skills"] == 0
        assert idx.search("anything", scope="skills") == []
        idx.close()
    finally:
        _cleanup(db, idx_path, vault, skills)


@test("semantic_recall", "skill_search routes semantic with an embedder, substring without")
async def t_skill_search_routing(ctx: TestContext) -> None:
    """Gap 2 routing: with an embedder, skill_search finds a paraphrase hit the
    substring scan MISSES; with no embedder, it is the byte-identical substring
    scan (no 'semantic' matched_in, no score key)."""
    import src.mcp.servers.skills.handlers as handlers
    import src.memory.semantic_index as si_mod

    skills = _skills_dir(ctx, "search")
    db = ctx.db_path.with_name(f"sr-search-{uuid.uuid4().hex[:8]}.db")
    prev_skills = os.environ.get("OPENAGENT_SKILLS_PATH")
    prev_db = os.environ.get("OPENAGENT_DB_PATH")
    prev_model = os.environ.get("OPENAGENT_EMBEDDING_MODEL")
    orig_resolve = si_mod.resolve_embedder
    try:
        _write_skill(skills, "refunds", "refund-playbook",
                     "how to handle a customer complaint about a refund",
                     "Offer the refund and apologise.")
        os.environ["OPENAGENT_SKILLS_PATH"] = str(skills)
        os.environ["OPENAGENT_DB_PATH"] = str(db)

        # ── (a) NO embedder → substring scan, byte-identical to the original ──
        os.environ.pop("OPENAGENT_EMBEDDING_MODEL", None)
        # The paraphrase shares no full substring with the skill → substring MISS.
        miss = await handlers.skill_search("angry unhappy client")
        assert miss["count"] == 0 and miss["results"] == [], (
            f"substring scan should MISS the paraphrase: {miss}")
        # A literal term IS found by the substring scan, tagged by field (not
        # 'semantic'), and with NO score key — proof it is the original path.
        kw = await handlers.skill_search("refund")
        assert kw["count"] == 1, kw
        r0 = kw["results"][0]
        assert r0["name"] == "refund-playbook", kw
        assert r0["matched_in"] != ["semantic"] and "matched_in" in r0, r0
        assert "score" not in r0, "substring result must not carry a semantic score"

        # ── (b) WITH an embedder → semantic path finds the paraphrase ──
        os.environ["OPENAGENT_EMBEDDING_MODEL"] = "fake:concept-v1"
        si_mod.resolve_embedder = lambda *a, **k: ConceptEmbedder()
        hit = await handlers.skill_search("angry unhappy client")
        assert hit["count"] == 1, f"semantic path missed the paraphrase: {hit}"
        h0 = hit["results"][0]
        assert h0["name"] == "refund-playbook", hit
        assert h0["matched_in"] == ["semantic"], h0
        assert "score" in h0 and h0["score"] >= 0.4, h0
    finally:
        si_mod.resolve_embedder = orig_resolve
        for k, v in (("OPENAGENT_SKILLS_PATH", prev_skills),
                     ("OPENAGENT_DB_PATH", prev_db),
                     ("OPENAGENT_EMBEDDING_MODEL", prev_model)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _cleanup(db, skills)


def _prime_recall_cache_with_skills(db_path: Path, vault: Path, skills: Path) -> Any:
    """Pre-build a fake-embedder index that ALSO carries a skills_root, sync it
    (so skill_vectors are populated), and install it in the hook's cache — the
    skills-leg analogue of ``_prime_recall_cache``."""
    import src.core.agent as agent_mod
    from src.memory.semantic_index import SemanticIndex

    idx = SemanticIndex(db_path, vault_root=vault, skills_root=str(skills),
                        embedder=ConceptEmbedder())
    idx.sync()
    agent_mod._RECALL_INDEX_CACHE[str(db_path)] = idx
    return idx


@test("semantic_recall", "_recall_block surfaces the top skill as a 'skill_view' line")
async def t_recall_block_skills_leg(ctx: TestContext) -> None:
    """Gap 2's per-turn payoff: a written skill becomes discoverable by MEANING
    on the turn path. With a fake embedder + a relevant skill, recall injects a
    verify-framed 'load it with skill_view <name>' line; with no embedder, the
    turn text is byte-identical."""
    import src.core.agent as agent_mod

    db, idx_path, vault = _paths(ctx, "skillleg")
    skills = _skills_dir(ctx, "skillleg")
    _clear_recall_env()
    try:
        _write_skill(skills, "refunds", "refund-playbook",
                     "how to handle a customer complaint about a refund",
                     "Offer the refund and apologise.")
        d = await _open_db(db)
        await d.close()
        idx = _prime_recall_cache_with_skills(db, vault, skills)

        os.environ["OPENAGENT_AUTO_RECALL_ENABLED"] = "1"
        os.environ["OPENAGENT_AUTO_RECALL_WARM_BUDGET"] = "0"  # already synced
        os.environ["OPENAGENT_AUTO_RECALL_MIN_SCORE"] = "0.5"
        os.environ["OPENAGENT_AUTO_RECALL_HYBRID"] = "0"  # isolate the skills leg

        out = await agent_mod._with_recall(
            _fake_agent(db, vault), "sess",
            "the client is angry and unhappy", "USER MESSAGE")

        assert out != "USER MESSAGE", "the skills leg surfaced nothing"
        assert out.endswith("USER MESSAGE"), "user message must stay at the end"
        assert "<system-reminder>" in out and "</system-reminder>" in out
        assert "skill_view refund-playbook" in out, (
            f"recall did not point at skill_view: {out!r}")
        assert "load it" in out.lower(), "skill line not framed as an action"
        idx.close()

        # No embedder → the skills leg is inert; the turn text is unchanged.
        agent_mod._RECALL_INDEX_CACHE.pop(str(db), None)
        os.environ.pop("OPENAGENT_EMBEDDING_MODEL", None)
        out2 = await agent_mod._with_recall(
            _fake_agent(db, vault), "sess",
            "the client is angry and unhappy", "USER MESSAGE")
        assert out2 == "USER MESSAGE", f"skills leg fired with no embedder: {out2!r}"
    finally:
        agent_mod._RECALL_INDEX_CACHE.pop(str(db), None)
        _clear_recall_env()
        _cleanup(db, idx_path, vault, skills)


# ─── the recall QUERY: marker-scoped span ─────────────────────────────────────
# Measured 2026-08-25 on the live replio-thread lane: the orchestration prompt
# around the customer's sentence had grown to 12,159 chars, and embedding all of
# it put the note that answers the question at #244-#262 instead of #1. At
# top_k 6 the right note was never injected. These pin the fix and its fallbacks.

@test("semantic_recall", "query marker: the SPAN is embedded, not the boilerplate around it")
async def t_query_marker_span_is_embedded(ctx: TestContext) -> None:
    import src.core.agent as agent_mod

    db, idx_path, vault = _paths(ctx, "qmarker")
    try:
        # Two notes: one answers the boilerplate, one answers the customer.
        _write_note(vault, "escalation-rules", "Escalation rules",
                    "escalation ticket triage workflow queue assignment")
        _write_note(vault, "refund-policy", "Refund policy",
                    "refund double charges within 14 days via Stripe")
        d = await _open_db(db); await d.close()
        sem = _prime_recall_cache(db, vault)

        os.environ["OPENAGENT_AUTO_RECALL_ENABLED"] = "1"
        os.environ["OPENAGENT_AUTO_RECALL_WARM_BUDGET"] = "0"
        os.environ["OPENAGENT_AUTO_RECALL_MIN_SCORE"] = "0.1"
        os.environ["OPENAGENT_AUTO_RECALL_TOP_K"] = "1"   # only the best survives
        # Semantic leg only: the marker governs what gets EMBEDDED, and with
        # hybrid on the FTS side rescues the answer by exact keyword and hides
        # the very dilution under test.
        os.environ["OPENAGENT_AUTO_RECALL_HYBRID"] = "0"

        prompt = (
            "escalation ticket triage workflow queue assignment " * 40
            + "\n<customer-message>\ncustomer wants a refund\n</customer-message>\n"
            + "escalation ticket triage workflow queue assignment " * 40
        )

        # Without the marker the boilerplate wins — the dilution, reproduced.
        out = await agent_mod._with_recall(_fake_agent(db, vault), "event:1", prompt, "MSG")
        assert "escalation-rules.md" in out, f"boilerplate did NOT dominate: {out!r}"
        assert "refund-policy.md" not in out, f"unexpectedly found the answer: {out!r}"

        # With the marker the customer's sentence is the query.
        os.environ["OPENAGENT_AUTO_RECALL_QUERY_MARKER"] = "customer-message"
        out2 = await agent_mod._with_recall(_fake_agent(db, vault), "event:1", prompt, "MSG")
        assert "refund-policy.md" in out2, f"marked span was NOT used as the query: {out2!r}"
        assert "escalation-rules.md" not in out2, f"boilerplate still won: {out2!r}"
        # What the model reads is untouched.
        assert out2.endswith("MSG")
        sem.close()
    finally:
        agent_mod._RECALL_INDEX_CACHE.pop(str(db), None)
        _clear_recall_env()
        _cleanup(db, idx_path, vault)


@test("semantic_recall", "query marker: unset / absent / empty / bogus all fall back to the whole message")
async def t_query_marker_fallbacks(ctx: TestContext) -> None:
    """Every degradation returns the FULL message, so a deployment that has not
    configured the marker — or a turn whose prompt lost it — behaves exactly as
    it did before the marker existed."""
    import src.core.agent as agent_mod

    msg = "before <customer-message>the span</customer-message> after"
    try:
        # unset
        os.environ.pop("OPENAGENT_AUTO_RECALL_QUERY_MARKER", None)
        assert agent_mod._recall_query(msg) == msg

        os.environ["OPENAGENT_AUTO_RECALL_QUERY_MARKER"] = "customer-message"
        assert agent_mod._recall_query(msg) == "the span"

        # tag configured but absent from the message
        assert agent_mod._recall_query("no tags here") == "no tags here"

        # present but wrapping only whitespace
        blank = "a <customer-message>   \n  </customer-message> b"
        assert agent_mod._recall_query(blank) == blank

        # unclosed
        unclosed = "a <customer-message>the span b"
        assert agent_mod._recall_query(unclosed) == unclosed

        # a tag name that is not a plain identifier is ignored, never compiled
        os.environ["OPENAGENT_AUTO_RECALL_QUERY_MARKER"] = "cust(o.*)mer"
        assert agent_mod._recall_query(msg) == msg

        # empty message
        os.environ["OPENAGENT_AUTO_RECALL_QUERY_MARKER"] = "customer-message"
        assert agent_mod._recall_query("") == ""
    finally:
        _clear_recall_env()


@test("semantic_recall", "query marker: multi-line span, and only the FIRST occurrence is taken")
async def t_query_marker_multiline(ctx: TestContext) -> None:
    import src.core.agent as agent_mod
    try:
        os.environ["OPENAGENT_AUTO_RECALL_QUERY_MARKER"] = "customer-message"
        multi = "x <customer-message>line one\nline two</customer-message> y"
        assert agent_mod._recall_query(multi) == "line one\nline two"
        two = ("<customer-message>first</customer-message> mid "
               "<customer-message>second</customer-message>")
        assert agent_mod._recall_query(two) == "first"
    finally:
        _clear_recall_env()
