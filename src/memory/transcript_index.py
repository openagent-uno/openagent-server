"""Incremental, FTS5-backed index over the session transcript.

Answers "remember when we discussed X?" — a question about *history*, which
is a different question from the one the vault answers. The vault (§5) is
*curated* knowledge: atomic notes a human wrote or approved. The transcript
(§16) is raw history, "stored durably with full fidelity". Both are real;
neither substitutes for the other, and until this module the second one had
no reader at all.

``sessions.runs`` is already the source of truth and already holds every
message of every session. This SQLite file is a *rebuildable cache* over it,
in the exact shape of :mod:`src.memory.vault.index` — delete it and the next
``sync`` reconstructs it. Nothing here is a second store of record: every row
is derived, and the derivation is one function away from the raw text.

WHAT THIS REPLACED AND WHY (v0.15.12)
-------------------------------------
The previous implementation embedded each turn with OpenAI's
``text-embedding-3-small`` into a ``conversation_embeddings`` table and
scanned it with a brute-force Python cosine loop. It was deleted rather than
fixed, for three reasons:

1. It never ran. Its writer (``learning/semantic_search.store_turn``) had
   ZERO callers on every deployment that ever existed, so the table it read
   could not contain a row. The tool was a permanent no-op that returned
   "no matches" — indistinguishable, to the model reading it, from "you never
   discussed that". It was double-gated on top (``OPENAGENT_SEMANTIC_SEARCH``
   defaulted off), which is how it stayed invisible.
2. It was a hidden vector store, which §5 rules out as the shape of memory
   ("not a hidden vector store and not an opaque database"). The sibling pair
   (``skills``, ``user_profiles``) was deleted for exactly this — see
   ``src/learning/__init__.py`` for the argument, which applies here verbatim.
3. It hardcoded one vendor. §17: "Removing any single provider … must leave
   the agent operational with what remains." An agent running only a local
   model could never search its own history. FTS5 ships inside SQLite, so the
   rewrite has no provider at all — there is nothing left to remove.

Full-text is a real downgrade from embeddings on synonyms: "the thing we
decided about shipping" will not find "launch deadline". That trade is
deliberate and is the honest one — a working keyword index beats a semantic
index that has never held a row, and the tool text says plainly what it does
and does not match rather than letting an empty result read as "never".

Design for scale (mirrors the vault index, which does this at 100k+ notes):
- **Incremental.** ``sync`` reads one cheap gating query —
  ``(session_id, updated_at, length(runs))`` — and only re-parses sessions
  whose pair changed. ``length(runs)`` is computed inside SQLite, so the
  gate never transfers a transcript blob into Python. A quiet agent
  re-syncs in the time it takes to scan a few hundred small values.
- **Bounded.** A single ``sync`` re-parses at most ``_MAX_SESSIONS_PER_SYNC``
  sessions, newest first, and reports the remainder as ``pending``. The first
  query on a large existing agent therefore costs a bounded amount of work
  instead of slurping every transcript at once — the ``read_tail``-on-the-
  event-loop hazard, which we are not repeating here.
- **Rebuildable.** No migration, no schema in ``memory/db.py``, no second
  source of truth. ``rm`` the file and the next query rebuilds it.

Runs synchronously; the MCP calls it via ``asyncio.to_thread``. Note the
server it backs is a *subprocess* MCP, so even the blocking parse happens off
the gateway process entirely — a live voice stream cannot be starved by it.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

_SCHEMA_VERSION = "1"

# Only these roles are indexed. This is the single most important line in
# the module: the runtime stores the FULL message list it sent to the model
# in ``run.messages``, and that list STARTS with the framework system prompt
# (``core/_runner/agent/_messages.py`` builds ``Message(role="system", …)``
# and inserts it at position 0). That prompt is ~10.8k tokens and byte-
# identical across every session on the agent — index it and every session
# matches every query, since they all literally contain the same text. Tool
# messages are skipped for a different reason: they are large, machine-shaped
# payloads, and the agent already has a purpose-built reader for that
# activity in the ``logs`` MCP (§14).
_INDEXED_ROLES = frozenset({"user", "assistant"})

# A single message longer than this is a pasted file or a dumped log, not
# something a person will recognise in a snippet. Index the head so it stays
# findable and stop — the whole message is always one session-open away.
_MAX_MESSAGE_CHARS = 20_000

# Messages shorter than this carry no recall value ("ok", "thanks", an emoji)
# and only dilute bm25. Same reasoning the deleted implementation used, which
# was the one thing it got right.
_MIN_MESSAGE_CHARS = 8

# Per-sync re-parse budget. See "Bounded" above.
_MAX_SESSIONS_PER_SYNC = 400


@dataclass
class SyncStats:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    pending: int = 0
    messages: int = 0
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def default_index_path(db_path: str | Path) -> Path:
    """Where the cache for ``db_path``'s transcript lives.

    Keyed by the ABSOLUTE SOURCE DB PATH, not by ``data_dir()``. A subprocess
    MCP that re-resolves platform defaults can silently land on a *different*
    agent's data — the bug that forced ``OPENAGENT_DB_PATH`` injection in the
    first place (see ``mcp/servers/logs/__init__.py``). The db path is the
    thing actually injected, so deriving from it means the index can never
    describe a database other than the one it was built from. Sits beside the
    db (normally the agent data dir); the hash keys it so two DBs in one
    directory get two caches.
    """
    src = Path(db_path).expanduser().resolve()
    h = hashlib.sha1(str(src).encode()).hexdigest()[:10]
    return src.parent / f"transcript_index_{h}.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- One row per indexed session. Mirrors the vault index's ``notes`` table:
-- it exists to hold the invalidation pair and the display metadata, so the
-- gate never needs to touch the FTS content.
CREATE TABLE IF NOT EXISTS indexed_sessions (
    session_id  TEXT PRIMARY KEY,
    updated_at  INTEGER NOT NULL,
    runs_len    INTEGER NOT NULL,
    title       TEXT,
    origin      TEXT,
    msg_count   INTEGER NOT NULL DEFAULT 0,
    indexed_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_isess_updated ON indexed_sessions(updated_at);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    session_id UNINDEXED,
    role       UNINDEXED,
    ts         UNINDEXED,
    seq        UNINDEXED,
    text,
    tokenize='unicode61'
);
"""


def _loads_maybe_double(raw: Any) -> Any:
    """``json.loads`` that unwraps the runtime's double-encoding.

    ``serialize_session_json_fields`` stores ``runs`` / ``metadata`` as a
    JSON-encoded STRING of a JSON value when handed a stringified one, so a
    single ``loads`` returns a ``str``. ``MemoryDB.list_session_runs`` and
    ``compaction._load_runs`` both unwrap exactly once more; a reader that
    doesn't sees an empty transcript on precisely the rows the runtime wrote.
    """
    if raw is None or raw == "":
        return None
    try:
        v = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except (TypeError, ValueError):
            return None
    return v


def _text_of(content: Any) -> str:
    """Flatten a message/run ``content`` into plain text.

    Same shape-tolerance as ``compaction._extract_run_text``: content is a
    str on most rows, a list of ``{"text": …}`` parts on multimodal ones, and
    occasionally something older. Anything unrecognised flattens to "".
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
        return "\n".join(chunks)
    return ""


def _iter_run_messages(runs: Any) -> Iterator[tuple[str, str, Optional[int]]]:
    """Yield ``(role, text, ts)`` for every indexable message across ``runs``.

    Emits the run's ``messages`` (user/assistant only) and then its
    ``content`` (the assistant's reply, which is usually ALSO present as an
    assistant message — the caller dedups, so whichever shape a given row
    uses, the text is indexed exactly once).
    """
    if not isinstance(runs, list):
        return
    for run in runs:
        if not isinstance(run, dict):
            continue
        ts = run.get("created_at")
        ts = int(ts) if isinstance(ts, (int, float)) else None
        for msg in run.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role not in _INDEXED_ROLES:
                continue
            text = _text_of(msg.get("content"))
            if text:
                yield role, text, ts
        content = _text_of(run.get("content"))
        if content:
            yield "assistant", content, ts


class TranscriptIndex:
    """A rebuildable FTS5 cache over one agent DB's ``sessions.runs``.

    Guarded by a lock so ``asyncio.to_thread`` calls from one loop serialize
    safely, matching ``VaultIndex``.
    """

    def __init__(self, db_path: str | Path, index_path: str | Path | None = None):
        self.db_path = str(Path(db_path).expanduser())
        self.index_path = Path(index_path) if index_path else default_index_path(self.db_path)
        self._lock = threading.RLock()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.index_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._ensure_source_meta()

    # ── lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def _ensure_source_meta(self) -> None:
        """Wipe the cache if it was built from a different DB or an older
        schema. A stale cache is worse than a cold one — it would answer with
        another agent's conversations."""
        with self._lock:
            cur = self._conn.execute("SELECT key, value FROM meta")
            meta = {r["key"]: r["value"] for r in cur.fetchall()}
            src = str(Path(self.db_path).resolve())
            if meta.get("source_db") != src or meta.get("schema") != _SCHEMA_VERSION:
                self._conn.executescript(
                    "DELETE FROM indexed_sessions; DELETE FROM messages_fts;"
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('source_db', ?)",
                    (src,),
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('schema', ?)",
                    (_SCHEMA_VERSION,),
                )
                self._conn.commit()

    def _open_source(self) -> Optional[sqlite3.Connection]:
        """Open the agent DB for reading.

        Opened read-WRITE despite this class never issuing anything but
        SELECTs: a ``mode=ro`` connection cannot recover a hot WAL, so it
        fails against exactly the live database this always runs beside. The
        other in-tree MCP subprocesses connect the same way (``_common.py``).
        """
        if not Path(self.db_path).exists():
            return None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 10000")
            return conn
        except sqlite3.Error:
            return None

    # ── sync ──────────────────────────────────────────────────────────

    def sync(self, *, force: bool = False,
             max_sessions: int = _MAX_SESSIONS_PER_SYNC) -> SyncStats:
        """Reconcile the cache with ``sessions``. Cheap when nothing changed.

        Three cases, all of which the invalidation pair handles uniformly:

        - **New session** → indexed.
        - **Session rewritten in place** → re-indexed from scratch. This is
          not hypothetical: ``core/compaction.py`` folds old runs into a
          recap and ``UPDATE sessions SET runs = ?, updated_at = ?``. An
          index that assumed append-only would keep serving the folded-away
          text forever — quoting the user paragraphs the system had already
          decided to forget. Because the delete-then-insert is keyed on the
          session, the compacted-away text is gone from the cache the moment
          the next sync sees the new ``(updated_at, runs_len)``.
        - **Session purged** → dropped. ``purge_session`` deletes the
          ``sessions`` row, so the id simply stops appearing in the gating
          query and the reconcile removes it here. This is a PULL, which is
          what makes it correct: nothing in ``db.py`` has to remember to
          notify this index, and since the MCP syncs *before* every query,
          a purged conversation is unfindable from the first query after the
          purge. See the class docstring in the server for the invariant.

        ``(updated_at, runs_len)`` is the ``(mtime, size)`` of a row, and is
        used for the same reason the vault uses that pair: either alone is
        forgeable (``updated_at`` is second-granular, and an edit can be
        size-neutral), together they are not, in practice. Compaction moves
        both — it rewrites the column AND stamps ``updated_at`` — so the case
        that actually matters is caught twice over.
        """
        t0 = time.monotonic()
        stats = SyncStats()
        with self._lock:
            src = self._open_source()
            if src is None:
                stats.elapsed_ms = int((time.monotonic() - t0) * 1000)
                return stats
            try:
                try:
                    cur = src.execute(
                        "SELECT session_id, updated_at, "
                        "       COALESCE(length(runs), 0) AS runs_len "
                        "FROM sessions"
                    )
                    live = {
                        r["session_id"]: (int(r["updated_at"] or 0), int(r["runs_len"]))
                        for r in cur.fetchall()
                    }
                except sqlite3.Error:
                    # No ``sessions`` table yet (brand-new agent DB).
                    stats.elapsed_ms = int((time.monotonic() - t0) * 1000)
                    return stats

                cur = self._conn.execute(
                    "SELECT session_id, updated_at, runs_len FROM indexed_sessions"
                )
                existing = {
                    r["session_id"]: (int(r["updated_at"]), int(r["runs_len"]))
                    for r in cur.fetchall()
                }

                # Purge propagation: anything we hold that the DB no longer
                # has is deleted history and must stop being findable.
                for sid in existing.keys() - live.keys():
                    self._delete_session(sid)
                    stats.deleted += 1

                stale: list[str] = []
                for sid, pair in live.items():
                    if not force and existing.get(sid) == pair:
                        stats.unchanged += 1
                    else:
                        stale.append(sid)

                # Newest first: recency is what "remember when we discussed X"
                # is usually reaching for, so a warming index is useful long
                # before it is complete.
                stale.sort(key=lambda s: live[s][0], reverse=True)
                if len(stale) > max_sessions:
                    stats.pending = len(stale) - max_sessions
                    stale = stale[:max_sessions]

                for sid in stale:
                    n = self._index_session(src, sid, live[sid])
                    if n is None:
                        continue
                    stats.messages += n
                    if sid in existing:
                        stats.updated += 1
                    else:
                        stats.added += 1
                self._conn.commit()
            finally:
                try:
                    src.close()
                except Exception:
                    pass
        stats.elapsed_ms = int((time.monotonic() - t0) * 1000)
        return stats

    def _delete_session(self, sid: str) -> None:
        self._conn.execute("DELETE FROM indexed_sessions WHERE session_id = ?", (sid,))
        self._conn.execute("DELETE FROM messages_fts WHERE session_id = ?", (sid,))

    def _index_session(self, src: sqlite3.Connection, sid: str,
                       pair: tuple[int, int]) -> Optional[int]:
        """(Re)index one session. Returns the message count, or None if the
        row vanished mid-sync (a purge racing us — the next sync reconciles)."""
        try:
            row = src.execute(
                "SELECT runs, metadata FROM sessions WHERE session_id = ?", (sid,)
            ).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None

        runs = _loads_maybe_double(row["runs"])
        meta = _loads_maybe_double(row["metadata"])
        meta = meta if isinstance(meta, dict) else {}

        # Delete-then-insert rather than a diff: a rewritten session must not
        # retain a single row of its previous text (see ``sync``).
        self._delete_session(sid)

        seen: set[tuple[str, str]] = set()
        rows: list[tuple] = []
        for role, text, ts in _iter_run_messages(runs):
            text = text.strip()
            if len(text) < _MIN_MESSAGE_CHARS:
                continue
            text = text[:_MAX_MESSAGE_CHARS]
            key = (role, text)
            if key in seen:
                # A run's ``content`` is normally repeated as its assistant
                # message; dedup so one reply is one hit, not two.
                continue
            seen.add(key)
            rows.append((sid, role, ts if ts is not None else pair[0], len(rows), text))

        if rows:
            self._conn.executemany(
                "INSERT INTO messages_fts(session_id, role, ts, seq, text) "
                "VALUES (?,?,?,?,?)",
                rows,
            )
        self._conn.execute(
            "INSERT INTO indexed_sessions "
            "(session_id, updated_at, runs_len, title, origin, msg_count, indexed_at) "
            "VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "  updated_at=excluded.updated_at, runs_len=excluded.runs_len, "
            "  title=excluded.title, origin=excluded.origin, "
            "  msg_count=excluded.msg_count, indexed_at=excluded.indexed_at",
            (
                sid, pair[0], pair[1],
                str(meta.get("title") or "")[:200],
                str(meta.get("origin") or "")[:40],
                len(rows), time.time(),
            ),
        )
        return len(rows)

    # ── queries ───────────────────────────────────────────────────────

    def search(self, query: str, *, limit: int = 5, offset: int = 0,
               session_id: Optional[str] = None,
               snippet_tokens: int = 16) -> list[dict[str, Any]]:
        """FTS5 search over indexed user/assistant messages, best match first.

        Returns ``{session_id, title, origin, role, ts, snippet}`` per hit.
        Only a snippet comes back, never the message — the full text is one
        ``GET /api/sessions/{id}/runs`` away, and a tool that returned whole
        transcripts into a context that already spends ~10.8k tokens on the
        framework prompt would be its own denial of service.
        """
        q = (query or "").strip()
        if not q:
            return []
        exprs = _fts_queries(q)
        if not exprs:
            return []
        with self._lock:
            sql = (
                "SELECT session_id, role, ts, seq, "
                f"       snippet(messages_fts, 4, '[', ']', '…', {int(snippet_tokens)}) AS snip "
                "FROM messages_fts WHERE messages_fts MATCH ? "
            )
            if session_id:
                sql += "AND session_id = ? "
            sql += "ORDER BY rank LIMIT ? OFFSET ?"

            rows: list[sqlite3.Row] = []
            for expr in exprs:
                params: list[Any] = [expr]
                if session_id:
                    params.append(session_id)
                params.extend([max(1, int(limit)), max(0, int(offset))])
                try:
                    rows = self._conn.execute(sql, params).fetchall()
                    break
                except sqlite3.OperationalError:
                    # A malformed MATCH — the caller's own FTS syntax didn't
                    # parse. Try the next (safer) rendering rather than
                    # failing the tool call.
                    continue
            else:
                return []

            titles = {
                r["session_id"]: (r["title"], r["origin"])
                for r in self._conn.execute(
                    "SELECT session_id, title, origin FROM indexed_sessions"
                )
            }
            out: list[dict[str, Any]] = []
            for r in rows:
                title, origin = titles.get(r["session_id"], ("", ""))
                out.append({
                    "session_id": r["session_id"],
                    "title": title or "",
                    "origin": origin or "",
                    "role": r["role"],
                    "ts": r["ts"],
                    "snippet": r["snip"],
                })
            return out

    def stats(self) -> dict[str, int]:
        with self._lock:
            s = self._conn.execute(
                "SELECT COUNT(*) FROM indexed_sessions").fetchone()[0]
            m = self._conn.execute(
                "SELECT COALESCE(SUM(msg_count), 0) FROM indexed_sessions").fetchone()[0]
            return {"sessions": int(s), "messages": int(m)}


def _fts_queries(q: str) -> list[str]:
    """Render a freeform query as FTS5 MATCH expressions, best first.

    The caller tries each in turn and keeps the first SQLite accepts, so a
    caller-supplied expression can never fail the whole tool call.

    A quoted span means the caller wants FTS5's own syntax (a phrase, NEAR,
    an explicit AND) and gets it, with the OR rendering behind it as a
    fallback. Otherwise each alphanumeric term becomes a prefix match joined
    by OR, exactly as ``vault/index.py`` does. OR rather than AND because the
    model asks in sentences ("the deadline we set for the migration"): AND
    returns nothing unless every word appears, whereas OR + bm25 ranks by
    inverse document frequency, so the rare words that carry the query
    ("deadline", "migration") dominate and "the"/"we" cost almost nothing.

    Returns ``[]`` when the query has no tokenizable terms at all (pure
    punctuation) — there is nothing to match and no expression to guess.
    """
    out: list[str] = []
    if '"' in q:
        out.append(q)
    terms = re.findall(r"\w+", q.lower())
    if terms:
        out.append(" OR ".join(f"{t}*" for t in terms))
    return out
