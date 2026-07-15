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
