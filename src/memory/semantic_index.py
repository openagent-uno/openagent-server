"""Rebuildable embedding cache over the vault notes AND the session transcripts.

This is the semantic-recall LAYER A. Keyword search (the vault FTS index and
``transcript_index``) matches WORDS; this matches MEANING — "launch deadline"
finds "the ship date we agreed", which FTS cannot. For a support agent asked
"has this customer complained before?" a thousand different ways, keyword recall
misses and the agent answers "no record" while the note exists — a confident
miss, i.e. a hallucination. This index closes that gap.

WHY THIS IS A CACHE, NOT A STORE (vision §5)
--------------------------------------------
§5 is explicit: "Memory is not a hidden vector store and not an opaque database
— it is human-inspectable Markdown." The Markdown vault and the ``sessions``
table stay the sole sources of truth; this SQLite file is a DERIVED cache of
their embeddings, exactly as ``src/memory/vault/index.py`` (FTS over notes) and
``src/memory/transcript_index.py`` (FTS over runs) are derived caches. Delete it
and the next ``sync`` rebuilds it. Nothing here is a second record: every vector
is one embedding call away from the source text, and the source text is one
``read_note`` / session-open away from the user.

The predecessor (``learning/semantic_search.py``, deleted v0.15.12) violated
this on three counts — it was a hidden vector store §5 rules out, it hardcoded
OpenAI (§17), and its writer had zero callers so it never held a row. See
``transcript_index.py`` for the full autopsy. This module keeps the ONE thing
that was worth rebuilding — semantic recall — while fixing all three: it lives
beside the FTS caches as a peer cache, degrades to nothing without a provider,
and is actually written (by the auto-recall hook and the memory-search MCP).

WHY IT DEGRADES TO NOTHING (vision §17)
---------------------------------------
§17: "Removing any single provider — any LLM API, any MCP, any channel — must
leave the agent operational with what remains." Embeddings need a model, and
the deleted code made ONE vendor (OpenAI) structurally required to search your
own memory — an agent on purely local models could never do it. So the embedder
here is provider-agnostic: any OpenAI-compatible ``/v1/embeddings`` endpoint,
named by ``OPENAGENT_EMBEDDING_MODEL=<provider>:<model>`` and resolved through
the SAME providers config the chat models use (the ``_SUMMARY_MODEL_ENV``
pattern in ``core/compaction.py``: the operator NAMES the model, we never infer
one). The recommended default is a LOCAL model — Ollama's ``nomic-embed-text``
at ``http://localhost:11434/v1`` — which is $0, private, and needs no cloud.
When no model is configured the whole layer is INERT: ``sync`` is a no-op,
``search`` returns ``[]``, and retrieval falls back to the existing FTS, byte-
identical to before this module existed. That is not a failure mode; it is the
self-hosted default.

BRUTE-FORCE COSINE IS HONEST AT THIS SCALE
------------------------------------------
Search stacks the stored unit vectors into one matrix and dots it with the
(unit) query vector — the top-k is a single ``numpy`` matmul over a few
thousand rows, sub-millisecond. The real eSound vault is ~2,100 notes; the
deleted code's own docstring flagged brute force as a stopgap only past ~50k.
An ANN index (faiss/hnsw) would be a second dependency and a second thing to
keep in sync for a corpus that does not need it. When a deployment's corpus
crosses ~50k, revisit — until then, brute force is the honest, dependency-free
answer, and this comment is the ranking note the vault FTS index has for bm25.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import array
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, Sequence

# numpy makes the cosine matmul ~100x faster, but it is FRAGILE to bundle into
# the PyInstaller onefile (it shipped absent twice, silently disabling recall
# with "No module named 'numpy'"). So it is OPTIONAL: with it, search is a fast
# matmul; without it, a pure-Python dot product (stdlib ``array`` + a loop) does
# the same over the few-thousand unit vectors a vault holds — slower per query
# but correct, and it can NEVER be missing from the bundle.
try:
    import numpy as np  # type: ignore
    _HAS_NUMPY = True
except Exception:  # noqa: BLE001 — a broken/absent numpy must degrade, not crash
    np = None  # type: ignore
    _HAS_NUMPY = False

# Bump when the stored-row SHAPE changes (a new column, a different blob
# encoding). A pure cache, so a bump just re-embeds from the sources that were
# always the truth — same contract as the vault index's ``_SCHEMA_VERSION``.
_SCHEMA_VERSION = "1"

# Per-sync embedding budget. A sync re-embeds at most this many changed items
# and reports the rest as ``pending`` — the same "bounded cold build" the
# transcript index uses, and it matters MORE here: each item is a network round
# trip to the embedding endpoint, not a local parse. A first sync on the 2,100-
# note eSound vault must not fire 2,100 sequential HTTP calls on one call.
_MAX_ITEMS_PER_SYNC = 128

# One embedding request carries at most this many texts. Every OpenAI-compatible
# ``/v1/embeddings`` endpoint accepts a list ``input``; batching cuts the round
# trips by ~this factor. Ollama honours it too.
_EMBED_BATCH = 32

# Text handed to the embedder per item, capped. A note body or a long session
# runs to tens of kilobytes; embedding all of it is slow, costs tokens, and
# dilutes the vector with boilerplate. The head carries the topic — the title,
# the summary, the opening exchange — which is what recall keys on. Mirrors
# ``transcript_index._MAX_MESSAGE_CHARS`` in spirit (index the recognisable
# head, the rest is one open away).
_MAX_EMBED_CHARS = 6_000

# Directory names never descended into when walking the vault — copied from
# ``vault/index.py._PRUNE_DIRS`` so this cache and the FTS cache agree on what
# "the vault" is (indexing ``_showcase`` would embed a derived artifact).
_PRUNE_DIRS = {".git", ".obsidian", "node_modules", ".openagent", "__pycache__",
               ".trash", ".DS_Store", "_showcase"}

# Convenience default endpoints per provider name, used ONLY when neither an
# explicit ``OPENAGENT_EMBEDDING_BASE_URL`` nor a providers-config row supplies
# one. This is NOT a vendor hardcode in the §17 sense — nothing here is
# required; it is a default URL the operator opts into by naming that provider.
# ``local`` deliberately has NO default: a self-hosted server's address is
# operator-specific (Ollama 11434, vLLM 8000, LM Studio 1234), so we make them
# state it rather than guess and silently hit the wrong port.
_DEFAULT_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
}


@dataclass
class SyncStats:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    pending: int = 0
    embedded: int = 0
    elapsed_ms: int = 0
    errored: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ── embedder resolution (the provider seam) ───────────────────────────


class EmbeddingError(RuntimeError):
    """The embedding endpoint failed. Callers degrade (skip recall / return
    ``[]``); a semantic miss must never fail a turn."""


class Embedder(Protocol):
    """Anything that turns texts into vectors. The Protocol is what keeps
    tests honest: a deterministic fake embedder can prove the index/search
    plumbing with no network, exactly the seam §17 asks for."""

    model_id: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...


class HttpEmbedder:
    """Calls any OpenAI-compatible ``POST {base_url}/embeddings``.

    Deliberately spoken over raw ``httpx`` rather than an SDK: the ``openai``
    package would re-pin the one vendor §17 forbids, and the wire format
    (``{"model", "input"}`` -> ``{"data":[{"embedding":[…]}]}``) is identical
    across OpenAI, Ollama, vLLM, LM Studio and llama.cpp. One code path serves
    every one of them, hosted or local.
    """

    def __init__(self, base_url: str, api_key: str, model_id: str,
                 *, provider: str = "", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.provider = provider
        self.timeout = timeout

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        import httpx

        url = f"{self.base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        # Local servers ignore the key but the OpenAI SDK convention is a
        # Bearer header; harmless to send one a keyless server drops.
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = httpx.post(
                url,
                headers=headers,
                json={"model": self.model_id, "input": list(texts)},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001 — normalise every failure
            raise EmbeddingError(f"{self.base_url}: {exc}") from exc

        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise EmbeddingError(
                f"{self.base_url}: expected {len(texts)} embeddings, got "
                f"{len(data) if isinstance(data, list) else type(data).__name__}"
            )
        out: list[list[float]] = []
        for row in data:
            vec = row.get("embedding") if isinstance(row, dict) else None
            if not isinstance(vec, list) or not vec:
                raise EmbeddingError(f"{self.base_url}: malformed embedding row")
            out.append([float(x) for x in vec])
        return out


def _find_provider_entry(providers_config: Any, name: str) -> dict[str, Any] | None:
    """The api-based provider row for ``name`` in a v0.12 flat list (or the
    legacy dict). Returns ``None`` when absent — resolution then falls back to
    env / defaults. Kept local (no ``iter_configured_models`` import) because
    we only need the provider's ``base_url``/``api_key``, not its model catalog,
    and the MCP subprocess must resolve this without the models machinery."""
    if isinstance(providers_config, list):
        entries: Iterable[dict] = (e for e in providers_config if isinstance(e, dict))
    elif isinstance(providers_config, dict):
        entries = (
            {**v, "name": k} for k, v in providers_config.items()
            if isinstance(v, dict)
        )
    else:
        return None
    for entry in entries:
        if str(entry.get("name") or "").strip() == name:
            return entry
    return None


def resolve_embedder(providers_config: Any = None) -> Optional[Embedder]:
    """Build the configured embedder, or ``None`` when the layer is inert.

    Resolution, in the ``_SUMMARY_MODEL_ENV`` shape (explicit config, never an
    inference over ``tier_hint``/``is_classifier`` — vision §3):

    1. ``OPENAGENT_EMBEDDING_MODEL`` unset -> ``None``. This is the self-hosted
       default: no embedding model, no semantic layer, retrieval is the
       existing FTS unchanged. Prove-able and proven (see the tests).
    2. ``<provider>:<model>`` -> split. A bare value is treated as the model on
       a provider of the same name.
    3. ``base_url`` / ``api_key``: an explicit ``OPENAGENT_EMBEDDING_BASE_URL`` /
       ``OPENAGENT_EMBEDDING_API_KEY`` wins (this is the ONLY channel that
       reaches the memory-search subprocess, whose env is not the parent's);
       else the matching providers-config row; else ``_DEFAULT_BASE_URLS``.
    4. No resolvable ``base_url`` -> ``None`` (inert) + a log line. A configured-
       but-unreachable embedder must degrade like an unconfigured one, not
       raise on the turn path.

    Recommended, and the §17-perfect default:
        OPENAGENT_EMBEDDING_MODEL=local:nomic-embed-text
        OPENAGENT_EMBEDDING_BASE_URL=http://localhost:11434/v1
    (Ollama; $0, private, offline). For a hosted default:
        OPENAGENT_EMBEDDING_MODEL=openai:text-embedding-3-small
    with the key on the ``openai`` provider or ``OPENAGENT_EMBEDDING_API_KEY``.
    """
    spec = (os.environ.get("OPENAGENT_EMBEDDING_MODEL") or "").strip()
    if not spec:
        return None

    provider, sep, model_id = spec.partition(":")
    if not sep:  # bare "model" — provider name doubles as the model's home
        provider, model_id = provider, provider

    base_url = (os.environ.get("OPENAGENT_EMBEDDING_BASE_URL") or "").strip() or None
    api_key = (os.environ.get("OPENAGENT_EMBEDDING_API_KEY") or "").strip() or None

    if (base_url is None or api_key is None) and providers_config:
        entry = _find_provider_entry(providers_config, provider)
        if entry:
            base_url = base_url or (str(entry.get("base_url") or "").strip() or None)
            api_key = api_key or (str(entry.get("api_key") or "").strip() or None)

    if base_url is None:
        base_url = _DEFAULT_BASE_URLS.get(provider)

    if base_url is None:
        from src.core.logging import elog
        elog(
            "semantic.embedder_unresolved",
            level="warning",
            configured=spec,
            reason="no_base_url",
            hint="set OPENAGENT_EMBEDDING_BASE_URL (e.g. http://localhost:11434/v1)",
        )
        return None

    # A keyless local server still needs *something* so the Bearer header is
    # well-formed; a real key configured above wins.
    if not api_key:
        api_key = "local"

    timeout = 30.0
    raw_to = (os.environ.get("OPENAGENT_EMBEDDING_TIMEOUT") or "").strip()
    if raw_to:
        try:
            timeout = max(1.0, float(raw_to))
        except ValueError:
            pass

    return HttpEmbedder(base_url, api_key, model_id, provider=provider, timeout=timeout)


# ── index location (keyed to the source DB, like transcript_index) ────


def default_semantic_index_path(db_path: str | Path) -> Path:
    """Where the semantic cache for ``db_path`` lives.

    Keyed on the ABSOLUTE SOURCE DB PATH for the exact reason
    ``transcript_index.default_index_path`` is: a subprocess that re-resolves
    platform defaults can land on another agent's data (the bug that forced
    ``OPENAGENT_DB_PATH`` injection). Deriving from the injected db path means
    the cache can never describe a database other than the one it was built
    from — and two agents, or two DBs in one directory, never share a cache.
    """
    src = Path(db_path).expanduser().resolve()
    h = hashlib.sha1(str(src).encode()).hexdigest()[:10]
    return src.parent / f"semantic_index_{h}.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- One row per embedded NOTE. ``(mtime, byte_size)`` is the invalidation pair,
-- identical to the vault FTS index's gate, so a touch-free vault re-syncs on a
-- stat scan with zero embedding calls.
CREATE TABLE IF NOT EXISTS vault_vectors (
    path        TEXT PRIMARY KEY,
    mtime       REAL,
    byte_size   INTEGER,
    title       TEXT,
    updated     TEXT,
    dim         INTEGER NOT NULL,
    vec         BLOB NOT NULL,
    embedded_at REAL NOT NULL
);

-- One row per embedded SESSION. ``(updated_at, runs_len)`` is the invalidation
-- pair, identical to the transcript index's gate. One vector per session (a
-- capped digest of its user/assistant text), not per message: session count is
-- bounded and this keeps the cold build's embedding-call count bounded with it.
CREATE TABLE IF NOT EXISTS session_vectors (
    session_id  TEXT PRIMARY KEY,
    updated_at  INTEGER,
    runs_len    INTEGER,
    title       TEXT,
    origin      TEXT,
    dim         INTEGER NOT NULL,
    vec         BLOB NOT NULL,
    embedded_at REAL NOT NULL
);
"""


def _unit(vec: Sequence[float]) -> array.array:
    """Return ``vec`` as a float32 ``array`` normalised to unit length (numpy-free)."""
    a = array.array("f", (float(x) for x in vec))
    norm = math.sqrt(sum(x * x for x in a))
    if norm > 0:
        a = array.array("f", (x / norm for x in a))
    return a


def _to_blob(vec: Sequence[float]) -> tuple[bytes, int]:
    """Normalise to unit length and pack as float32. Storing UNIT vectors turns
    cosine similarity into a plain dot product at search time — no per-query
    renormalisation over the whole matrix."""
    if _HAS_NUMPY:
        arr = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr.tobytes(), int(arr.shape[0])
    a = _unit(vec)
    return a.tobytes(), len(a)


def _prep_text(text: str) -> str:
    return (text or "").strip()[:_MAX_EMBED_CHARS]


class SemanticIndex:
    """A rebuildable embedding cache over one agent's vault + sessions.

    Guarded by a lock so ``asyncio.to_thread`` calls from one loop serialize
    safely, matching ``VaultIndex`` / ``TranscriptIndex``. INERT when
    constructed without an embedder: ``sync_*`` no-op and ``search`` returns
    ``[]`` — the whole point of the "degrades to nothing" contract.
    """

    def __init__(self, db_path: str | Path, *,
                 vault_root: str | Path | None = None,
                 index_path: str | Path | None = None,
                 embedder: Optional[Embedder] = None):
        self.db_path = str(Path(db_path).expanduser())
        self.vault_root = Path(vault_root).expanduser() if vault_root else None
        self.index_path = (Path(index_path) if index_path
                           else default_semantic_index_path(self.db_path))
        self.embedder = embedder
        self._lock = threading.RLock()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.index_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._ensure_meta()

    # ── lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    @property
    def active(self) -> bool:
        """True when an embedder is wired — i.e. the layer is not inert."""
        return self.embedder is not None

    def _ensure_meta(self) -> None:
        """Wipe the cache when it was built from a different DB, a different
        embedding MODEL, or an older schema.

        The model check is the one this index adds over the FTS caches: vectors
        from ``text-embedding-3-small`` and ``nomic-embed-text`` live in
        different spaces and different dimensions, so mixing them would rank by
        noise (or crash the matmul on a dim mismatch). Changing the model is a
        rebuild, exactly like changing the schema — and cheaply so, because the
        source text was never thrown away.
        """
        with self._lock:
            cur = self._conn.execute("SELECT key, value FROM meta")
            meta = {r["key"]: r["value"] for r in cur.fetchall()}
            src = str(Path(self.db_path).resolve())
            model = self.embedder.model_id if self.embedder else ""
            if (meta.get("source_db") != src
                    or meta.get("schema") != _SCHEMA_VERSION
                    or meta.get("embed_model") != model):
                self._conn.executescript(
                    "DELETE FROM vault_vectors; DELETE FROM session_vectors;"
                )
                for k, v in (("source_db", src), ("schema", _SCHEMA_VERSION),
                             ("embed_model", model)):
                    self._conn.execute(
                        "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)", (k, v))
                self._conn.commit()

    def _open_source(self) -> Optional[sqlite3.Connection]:
        """Open the agent DB read-write (a ``mode=ro`` connection cannot recover
        a hot WAL beside a live writer — same reasoning as transcript_index)."""
        if not Path(self.db_path).exists():
            return None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 10000")
            return conn
        except sqlite3.Error:
            return None

    def _embed_batch(self, texts: list[str]) -> list[tuple[bytes, int]]:
        """Embed ``texts`` (already truncated) into ``(blob, dim)`` pairs.
        Raises ``EmbeddingError`` on any endpoint failure; callers catch it and
        mark the sync errored rather than letting it reach the turn path."""
        assert self.embedder is not None
        out: list[tuple[bytes, int]] = []
        for i in range(0, len(texts), _EMBED_BATCH):
            chunk = texts[i:i + _EMBED_BATCH]
            for vec in self.embedder.embed(chunk):
                out.append(_to_blob(vec))
        return out

    # ── sync: vault notes ─────────────────────────────────────────────

    def _walk_md(self) -> list[Path]:
        import os as _os
        out: list[Path] = []
        if not self.vault_root or not self.vault_root.exists():
            return out
        for dirpath, dirnames, filenames in _os.walk(self.vault_root):
            dirnames[:] = [d for d in dirnames if d not in _PRUNE_DIRS]
            for fn in filenames:
                if fn.lower().endswith(".md"):
                    out.append(Path(dirpath) / fn)
        return out

    def sync_vault(self, *, force: bool = False,
                   max_items: int = _MAX_ITEMS_PER_SYNC) -> SyncStats:
        """Reconcile the note vectors with the Markdown on disk.

        Only notes whose ``(mtime, byte_size)`` changed are re-embedded; a
        vanished file drops its vector (a purged note must stop being findable,
        mirroring ``test_session_delete``'s invariant one table over). Bounded
        per call and newest-changed-first so a cold build costs a fixed number
        of embedding round trips and reports the remainder as ``pending``.
        """
        t0 = time.monotonic()
        stats = SyncStats()
        if not self.active or not self.vault_root:
            stats.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return stats

        from src.memory.vault.parser import parse_note_text, split_frontmatter

        with self._lock:
            existing = {
                r["path"]: (r["mtime"], r["byte_size"])
                for r in self._conn.execute(
                    "SELECT path, mtime, byte_size FROM vault_vectors")
            }
            seen: set[str] = set()
            stale: list[tuple[str, Path, float, int]] = []
            for abs_path in self._walk_md():
                try:
                    st = abs_path.stat()
                except OSError:
                    continue
                rel = abs_path.relative_to(self.vault_root).as_posix()
                seen.add(rel)
                prev = existing.get(rel)
                if (not force and prev is not None
                        and abs(prev[0] - st.st_mtime) < 1e-6
                        and prev[1] == st.st_size):
                    stats.unchanged += 1
                    continue
                stale.append((rel, abs_path, st.st_mtime, st.st_size))

            for rel in existing.keys() - seen:
                self._conn.execute("DELETE FROM vault_vectors WHERE path = ?", (rel,))
                stats.deleted += 1

            stale.sort(key=lambda t: t[2], reverse=True)  # newest first
            if len(stale) > max_items:
                stats.pending = len(stale) - max_items
                stale = stale[:max_items]

            texts: list[str] = []
            rows: list[tuple[str, float, int, str, str]] = []
            for rel, abs_path, mtime, size in stale:
                try:
                    content = abs_path.read_text(errors="replace")
                except OSError:
                    continue
                note = parse_note_text(rel, content, mtime=mtime)
                _, body = split_frontmatter(content)
                digest = "\n".join(x for x in (note.title, note.summary, body) if x)
                texts.append(_prep_text(digest))
                rows.append((rel, mtime, size, note.title or "", note.updated or ""))

            if texts:
                try:
                    blobs = self._embed_batch(texts)
                except EmbeddingError as exc:
                    self._log_embed_error("vault", exc)
                    stats.errored = True
                    self._conn.commit()
                    stats.elapsed_ms = int((time.monotonic() - t0) * 1000)
                    return stats
                for (rel, mtime, size, title, updated), (blob, dim) in zip(rows, blobs):
                    self._conn.execute(
                        "INSERT INTO vault_vectors "
                        "(path, mtime, byte_size, title, updated, dim, vec, embedded_at) "
                        "VALUES (?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(path) DO UPDATE SET "
                        "  mtime=excluded.mtime, byte_size=excluded.byte_size, "
                        "  title=excluded.title, updated=excluded.updated, "
                        "  dim=excluded.dim, vec=excluded.vec, "
                        "  embedded_at=excluded.embedded_at",
                        (rel, mtime, size, title, updated, dim, blob, time.time()),
                    )
                    stats.embedded += 1
                    stats.added += 1 if rel not in existing else 0
                    stats.updated += 1 if rel in existing else 0
            self._conn.commit()
        stats.elapsed_ms = int((time.monotonic() - t0) * 1000)
        return stats

    # ── sync: sessions ────────────────────────────────────────────────

    def sync_sessions(self, *, force: bool = False,
                      max_items: int = _MAX_ITEMS_PER_SYNC) -> SyncStats:
        """Reconcile the session vectors with ``sessions.runs``.

        Same three cases the transcript index handles, same ``(updated_at,
        runs_len)`` gate: new -> embedded, rewritten (compaction) -> re-embedded
        from scratch, purged -> dropped. The purge case is a PULL (the id stops
        appearing in the gating query), so ``db.py`` needs to know nothing about
        this cache — the invariant that keeps a deleted conversation unfindable.
        """
        t0 = time.monotonic()
        stats = SyncStats()
        if not self.active:
            stats.elapsed_ms = int((time.monotonic() - t0) * 1000)
            return stats

        from src.memory.transcript_index import _iter_run_messages, _loads_maybe_double

        with self._lock:
            src = self._open_source()
            if src is None:
                stats.elapsed_ms = int((time.monotonic() - t0) * 1000)
                return stats
            try:
                try:
                    live = {
                        r["session_id"]: (int(r["updated_at"] or 0), int(r["runs_len"]))
                        for r in src.execute(
                            "SELECT session_id, updated_at, "
                            "COALESCE(length(runs),0) AS runs_len FROM sessions")
                    }
                except sqlite3.Error:
                    stats.elapsed_ms = int((time.monotonic() - t0) * 1000)
                    return stats

                existing = {
                    r["session_id"]: (int(r["updated_at"] or 0), int(r["runs_len"] or 0))
                    for r in self._conn.execute(
                        "SELECT session_id, updated_at, runs_len FROM session_vectors")
                }

                for sid in existing.keys() - live.keys():
                    self._conn.execute(
                        "DELETE FROM session_vectors WHERE session_id = ?", (sid,))
                    stats.deleted += 1

                stale: list[str] = []
                for sid, pair in live.items():
                    if not force and existing.get(sid) == pair:
                        stats.unchanged += 1
                    else:
                        stale.append(sid)
                stale.sort(key=lambda s: live[s][0], reverse=True)  # newest first
                if len(stale) > max_items:
                    stats.pending = len(stale) - max_items
                    stale = stale[:max_items]

                texts: list[str] = []
                metas: list[tuple[str, int, int, str, str]] = []
                for sid in stale:
                    row = src.execute(
                        "SELECT runs, metadata FROM sessions WHERE session_id = ?",
                        (sid,)).fetchone()
                    if row is None:
                        continue
                    runs = _loads_maybe_double(row["runs"])
                    meta = _loads_maybe_double(row["metadata"])
                    meta = meta if isinstance(meta, dict) else {}
                    title = str(meta.get("title") or "")
                    seen_txt: set[str] = set()
                    parts: list[str] = [title] if title else []
                    for _role, text, _ts in _iter_run_messages(runs):
                        text = text.strip()
                        if len(text) < 8 or text in seen_txt:
                            continue
                        seen_txt.add(text)
                        parts.append(text)
                    digest = _prep_text("\n".join(parts))
                    if not digest:
                        continue
                    texts.append(digest)
                    metas.append((sid, live[sid][0], live[sid][1], title[:200],
                                  str(meta.get("origin") or "")[:40]))

                if texts:
                    try:
                        blobs = self._embed_batch(texts)
                    except EmbeddingError as exc:
                        self._log_embed_error("sessions", exc)
                        stats.errored = True
                        self._conn.commit()
                        stats.elapsed_ms = int((time.monotonic() - t0) * 1000)
                        return stats
                    for (sid, upd, rlen, title, origin), (blob, dim) in zip(metas, blobs):
                        self._conn.execute(
                            "INSERT INTO session_vectors "
                            "(session_id, updated_at, runs_len, title, origin, dim, "
                            " vec, embedded_at) VALUES (?,?,?,?,?,?,?,?) "
                            "ON CONFLICT(session_id) DO UPDATE SET "
                            "  updated_at=excluded.updated_at, runs_len=excluded.runs_len, "
                            "  title=excluded.title, origin=excluded.origin, "
                            "  dim=excluded.dim, vec=excluded.vec, "
                            "  embedded_at=excluded.embedded_at",
                            (sid, upd, rlen, title, origin, dim, blob, time.time()),
                        )
                        stats.embedded += 1
                        stats.added += 1 if sid not in existing else 0
                        stats.updated += 1 if sid in existing else 0
                self._conn.commit()
            finally:
                try:
                    src.close()
                except Exception:
                    pass
        stats.elapsed_ms = int((time.monotonic() - t0) * 1000)
        return stats

    def sync(self, *, force: bool = False,
             max_items: int = _MAX_ITEMS_PER_SYNC) -> dict[str, SyncStats]:
        """Sync both sources. Returns per-source stats; a no-op when inert."""
        return {
            "vault": self.sync_vault(force=force, max_items=max_items),
            "sessions": self.sync_sessions(force=force, max_items=max_items),
        }

    def _log_embed_error(self, source: str, exc: Exception) -> None:
        try:
            from src.core.logging import elog
            elog("semantic.embed_error", level="warning", source=source,
                 model=self.embedder.model_id if self.embedder else "",
                 error=str(exc)[:200])
        except Exception:  # noqa: BLE001
            pass

    # ── search ────────────────────────────────────────────────────────

    def _sims_for(self, table: str, q: Sequence[float]) -> list[tuple[sqlite3.Row, float]]:
        """``(row, cosine)`` for every stored unit vector in ``table``. Both are
        unit vectors so cosine == dot product. numpy path: one matmul over the
        stacked matrix; numpy-free path: a stdlib dot-product loop (fine over the
        few-thousand vectors a vault holds). Rows whose ``dim`` mismatches the
        query — e.g. a leftover from a different embed model — are skipped."""
        rows = list(self._conn.execute(f"SELECT * FROM {table}"))
        if not rows:
            return []
        dim = len(q)
        if _HAS_NUMPY:
            keep = [r for r in rows if r["dim"] == dim]
            if not keep:
                return []
            mat = np.frombuffer(b"".join(r["vec"] for r in keep), dtype=np.float32)
            mat = mat.reshape(len(keep), dim)
            sims = (mat @ np.asarray(q, dtype=np.float32)).tolist()
            return list(zip(keep, sims))
        out: list[tuple[sqlite3.Row, float]] = []
        for r in rows:
            if r["dim"] != dim:
                continue
            v = array.array("f")
            v.frombytes(r["vec"])
            out.append((r, math.fsum(a * b for a, b in zip(v, q))))
        return out

    def search(self, query: str, *, scope: str = "all", limit: int = 5,
               min_score: float = 0.0,
               include_prefixes: Optional[Sequence[str]] = None,
               exclude_prefixes: Optional[Sequence[str]] = None
               ) -> list[dict[str, Any]]:
        """Cosine-nearest vault notes / sessions to ``query``, best first.

        ``scope`` is ``"all"`` | ``"vault"`` | ``"sessions"``. Returns
        ``[]`` when inert, on an empty query, or when nothing clears
        ``min_score`` — a weak match is NO match, which is what keeps auto-recall
        from injecting noise. Each hit carries a ``score`` (cosine, 0..1) so the
        caller can threshold and frame it as "verify", never assert it as fact.

        ``include_prefixes`` / ``exclude_prefixes`` scope the NOTE side by
        vault-relative path prefix (keep only / drop matches) — the corpus knob
        that lets support recall skip dev-ops notes. Both default to no filter
        (unchanged behaviour). Sessions carry no path and are governed by
        ``scope`` alone.
        """
        q = (query or "").strip()
        if not self.active or not q:
            return []
        try:
            qv = self.embedder.embed([_prep_text(q)])[0]  # type: ignore[union-attr]
        except EmbeddingError as exc:
            self._log_embed_error("query", exc)
            return []
        qunit = list(_unit(qv))  # unit float list — numpy-free, works for both paths

        want = {"vault", "sessions"} if scope == "all" else {scope}
        hits: list[dict[str, Any]] = []
        with self._lock:
            if "vault" in want:
                inc = tuple(p for p in (include_prefixes or ()) if p)
                exc = tuple(p for p in (exclude_prefixes or ()) if p)
                for r, s in self._sims_for("vault_vectors", qunit):
                    if float(s) < min_score:
                        continue
                    path = str(r["path"] or "").lstrip("/")
                    if inc and not path.startswith(inc):
                        continue
                    if exc and path.startswith(exc):
                        continue
                    hits.append({
                        "kind": "note", "score": round(float(s), 4),
                        "path": r["path"], "title": r["title"] or "",
                        "updated": r["updated"] or "",
                    })
            if "sessions" in want:
                for r, s in self._sims_for("session_vectors", qunit):
                    if float(s) >= min_score:
                        hits.append({
                            "kind": "session", "score": round(float(s), 4),
                            "session_id": r["session_id"],
                            "title": r["title"] or "", "origin": r["origin"] or "",
                        })
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:max(1, int(limit))]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            v = self._conn.execute("SELECT COUNT(*) FROM vault_vectors").fetchone()[0]
            s = self._conn.execute("SELECT COUNT(*) FROM session_vectors").fetchone()[0]
        return {
            "notes": int(v), "sessions": int(s),
            "active": self.active,
            "model": self.embedder.model_id if self.embedder else None,
        }
