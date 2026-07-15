"""Incremental, FTS5-backed index over a vault of Markdown notes.

This is the piece that makes the gate scale to hundreds of thousands of
notes. The Markdown files remain the source of truth; this SQLite database
is a *rebuildable cache* of their parsed metadata and link graph. Delete it
and the next ``sync`` reconstructs it.

Design for scale:
- **Incremental parse.** ``sync`` walks the tree, stats each file, and only
  re-reads + re-parses files whose ``(mtime, size)`` changed. A touch-free
  vault re-syncs in the time it takes to ``stat`` the files.
- **Cheap global graph.** Link resolution, inbound-link counts, broken links
  and connected components are recomputed from the indexed rows (never by
  re-reading files) — a few hundred-thousand dict lookups, sub-second even
  at 100k notes. This avoids fragile incremental-graph bookkeeping.
- **Bodies are not stored** in the ``notes`` table (every body-derived gate
  signal is captured at parse time). Full text lives only in the standalone
  FTS5 table, so search scales without doubling storage of parsed metadata.

Runs synchronously; the async ``VaultService`` calls it via
``asyncio.to_thread`` so the event loop never blocks on a big sync.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from src.memory.vault.model import Note
from src.memory.vault.parser import link_key, parse_note_text

# Bumped to 2 when ``stem`` joined notes_fts: the FTS table's COLUMN COUNT
# changed, so an index built by an older build cannot be reused — see
# ``_ensure_vault_meta``, which drops it rather than clearing it.
_SCHEMA_VERSION = "2"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS notes (
    path              TEXT PRIMARY KEY,
    folder            TEXT,
    stem              TEXT,
    stem_lower        TEXT,
    path_key          TEXT,
    title             TEXT,
    summary           TEXT,
    status            TEXT,
    created           TEXT,
    updated           TEXT,
    tags_json         TEXT,
    outlinks_json     TEXT,
    related_json      TEXT,
    spaced_json       TEXT,
    missing_fm_json   TEXT,
    line_count        INTEGER,
    byte_size         INTEGER,
    mtime             REAL,
    content_hash      TEXT,
    has_frontmatter   INTEGER,
    is_index          INTEGER,
    is_journal        INTEGER,
    related_multiline INTEGER,
    em_dash           INTEGER,
    inlink_count      INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_notes_stem_lower ON notes(stem_lower);
CREATE INDEX IF NOT EXISTS idx_notes_path_key   ON notes(path_key);
CREATE INDEX IF NOT EXISTS idx_notes_folder     ON notes(folder);
CREATE INDEX IF NOT EXISTS idx_notes_hash       ON notes(content_hash);

CREATE TABLE IF NOT EXISTS links (
    src        TEXT,
    target_raw TEXT,
    target_key TEXT,
    resolved   TEXT          -- NULL when the link is broken
);
CREATE INDEX IF NOT EXISTS idx_links_src      ON links(src);
CREATE INDEX IF NOT EXISTS idx_links_resolved ON links(resolved);

-- ``stem`` is indexed alongside the prose because a note's FILENAME is the
-- most deliberate label a human puts on it, and search used to throw it
-- away. On the real vault the query "premium not active playbook" could not
-- reach premium-not-active-playbook.md at ANY rank: its title is "Rule 1b —
-- Premium Not Active / Not Verifiable", so "playbook" lived only in the
-- filename. Column ORDER is load-bearing — ``search`` addresses body by
-- index for snippet() and weights all five for bm25().
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    path UNINDEXED, stem, title, summary, body, tokenize='unicode61'
);
"""

# bm25 column weights, in notes_fts column order: path (UNINDEXED, never
# matches), stem, title, summary, body. Flat weighting is why the vault's
# canonical notes were unreachable: 1,210 of the owner's 2,116 notes are
# auto-written support-triage receipts that mention "premium" or "google
# play" in passing, and short repetitive notes are exactly what flat bm25
# rewards — so the receipts buried the one note that was ABOUT the topic.
# Title outranks the stem because the stem is usually a slug OF the title
# (counting both at full weight would double-count the same signal), and
# body stays at 1 as the long-tail floor. Measured: title-shaped recall
# went from MRR 0.746 to 0.967, and the numbers sit on a broad plateau
# (stem 4-10, title 6-10, summary 3-5 all score within 0.05), so these are
# a sane point in a flat region rather than a fit to the query set.
_BM25_WEIGHTS = (0.0, 5.0, 10.0, 5.0, 1.0)

# Tokens of context returned around a hit. Was 12, which often could not
# span the matched terms plus enough words to tell two notes apart — so the
# agent spent a ``read_note`` round-trip per candidate just to find out
# which one it wanted. 32 is still small next to a note.
_SNIPPET_TOKENS = 32

# Directory names never descended into during a walk. ``_showcase`` holds a
# derived artifact (regenerated from the index), so indexing it would be
# circular; the rest are caches / VCS / OS noise.
_PRUNE_DIRS = {".git", ".obsidian", "node_modules", ".openagent", "__pycache__",
               ".trash", ".DS_Store", "_showcase"}


@dataclass
class SyncStats:
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    total: int = 0
    links: int = 0
    broken: int = 0
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _stem_text(stem: str) -> str:
    """A filename as searchable words: ``premium-not-active-playbook`` ->
    ``premium not active playbook``. The unicode61 tokenizer already splits
    on hyphens, but doing it here keeps the indexed text readable and covers
    the underscore/dot separators the vault also uses."""
    return re.sub(r"[-_./]+", " ", stem or "").strip()


def _row_to_note(row: sqlite3.Row) -> Note:
    """Reconstruct a ``Note`` (sans body) from an indexed row."""
    return Note(
        path=row["path"],
        folder=row["folder"] or "",
        stem=row["stem"] or "",
        title=row["title"] or "",
        summary=row["summary"] or "",
        tags=json.loads(row["tags_json"] or "[]"),
        status=row["status"] or "",
        created=row["created"] or "",
        updated=row["updated"] or "",
        related=json.loads(row["related_json"] or "[]"),
        outlinks=json.loads(row["outlinks_json"] or "[]"),
        line_count=row["line_count"] or 0,
        byte_size=row["byte_size"] or 0,
        content_hash=row["content_hash"] or "",
        mtime=row["mtime"] or 0.0,
        has_frontmatter=bool(row["has_frontmatter"]),
        is_index=bool(row["is_index"]),
        is_journal=bool(row["is_journal"]),
        related_multiline=bool(row["related_multiline"]),
        spaced_wikilinks=json.loads(row["spaced_json"] or "[]"),
        missing_frontmatter_fields=json.loads(row["missing_fm_json"] or "[]"),
        body_has_em_dash=bool(row["em_dash"]),
    )


class VaultIndex:
    """A SQLite-backed index of one vault. Not safe to share a single
    instance across event loops, but its methods are guarded by a lock so
    ``asyncio.to_thread`` calls from one loop are serialized safely."""

    def __init__(self, vault_root: Path, index_path: Path,
                 journal_root: str = "workspace/journal"):
        self.vault_root = Path(vault_root)
        self.index_path = Path(index_path)
        self.journal_root = journal_root
        self._lock = threading.RLock()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.index_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._ensure_vault_meta()
        self._configure_rank()

    # ── lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def _ensure_vault_meta(self) -> None:
        """If the index belongs to a different vault root or an older schema,
        wipe it — a stale cache is worse than a cold one."""
        with self._lock:
            cur = self._conn.execute("SELECT key, value FROM meta")
            meta = {r["key"]: r["value"] for r in cur.fetchall()}
            root = str(self.vault_root.resolve())
            if meta.get("vault_root") != root or meta.get("schema") != _SCHEMA_VERSION:
                # DROP, not DELETE: a schema bump can change the FTS table's
                # column COUNT (v2 added ``stem``), and ``CREATE VIRTUAL
                # TABLE IF NOT EXISTS`` will happily leave an old 4-column
                # table in place — every insert would then fail with "table
                # notes_fts has 4 columns but 5 values were supplied" on a
                # vault that had merely been indexed by an older build.
                self._conn.executescript(
                    "DELETE FROM notes; DELETE FROM links; DROP TABLE IF EXISTS notes_fts;"
                )
                self._conn.executescript(_SCHEMA)
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('vault_root', ?)",
                    (root,),
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES('schema', ?)",
                    (_SCHEMA_VERSION,),
                )
                self._conn.commit()

    def _configure_rank(self) -> None:
        """Make ``ORDER BY rank`` mean the WEIGHTED bm25 for this table.

        Not a style choice — it is 4x faster than the identical
        ``ORDER BY bm25(notes_fts, ...)`` expression (p50 2.3ms vs 9.6ms on
        the real 2,116-note vault, byte-identical result lists). Only the
        ``rank`` keyword lets FTS5 use its internal top-k path; an explicit
        bm25() call in the ORDER BY is an opaque expression, so every
        matching row is scored and sorted — and an OR-of-prefixes query
        matches a LOT of rows.

        Re-applied on every open rather than once at creation: the config is
        persisted INSIDE the FTS table, so a weights change in this file
        would otherwise be silently ignored on an already-built index. If it
        ever fails to apply, ``ORDER BY rank`` quietly falls back to
        UNWEIGHTED bm25 — the exact ranking bug this all fixes — so the
        failure is loud here instead.
        """
        weights = ", ".join(str(w) for w in _BM25_WEIGHTS)
        with self._lock:
            self._conn.execute(
                "INSERT INTO notes_fts(notes_fts, rank) VALUES('rank', ?)",
                (f"bm25({weights})",),
            )
            self._conn.commit()

    # ── walk + sync ───────────────────────────────────────────────────

    def _walk_md(self) -> Iterator[Path]:
        for dirpath, dirnames, filenames in os.walk(self.vault_root):
            dirnames[:] = [d for d in dirnames if d not in _PRUNE_DIRS]
            for fn in filenames:
                if fn.lower().endswith(".md"):
                    yield Path(dirpath) / fn

    def sync(self, force: bool = False) -> SyncStats:
        """Reconcile the index with the files on disk. Only changed files are
        re-parsed (``force`` re-parses everything). Recomputes the link graph
        afterwards."""
        t0 = time.monotonic()
        stats = SyncStats()
        with self._lock:
            if not self.vault_root.exists():
                # Nothing on disk — clear the index.
                self._conn.executescript(
                    "DELETE FROM notes; DELETE FROM links; DELETE FROM notes_fts;"
                )
                self._conn.commit()
                stats.elapsed_ms = int((time.monotonic() - t0) * 1000)
                return stats

            cur = self._conn.execute("SELECT path, mtime, byte_size FROM notes")
            existing = {r["path"]: (r["mtime"], r["byte_size"]) for r in cur.fetchall()}
            seen: set[str] = set()

            for abs_path in self._walk_md():
                try:
                    st = abs_path.stat()
                except OSError:
                    continue
                rel = abs_path.relative_to(self.vault_root).as_posix()
                # Index everything walkable (so links into sources/workspace
                # resolve and the graph is complete); the *gate* decides what
                # to skip via its own config. Non-note dirs are already pruned.
                seen.add(rel)
                prev = existing.get(rel)
                # Quick-check on (mtime, raw on-disk size). We store the RAW
                # ``st_size`` (not the decoded byte length) so a CRLF or
                # non-UTF-8 note isn't seen as "changed" on every sync.
                if (not force and prev is not None
                        and abs(prev[0] - st.st_mtime) < 1e-6
                        and prev[1] == st.st_size):
                    stats.unchanged += 1
                    continue
                try:
                    content = abs_path.read_text(errors="replace")
                except OSError:
                    continue
                note = parse_note_text(rel, content, mtime=st.st_mtime,
                                       journal_root=self.journal_root)
                note.byte_size = st.st_size  # store raw disk size for invalidation
                self._upsert(note, content)
                if prev is None:
                    stats.added += 1
                else:
                    stats.updated += 1

            # Delete notes whose files vanished.
            for rel in existing.keys() - seen:
                self._delete(rel)
                stats.deleted += 1

            stats.total = len(seen)
            # The global link/graph rebuild is O(N); skip it entirely when no
            # file changed (the common case for a gate/stats call on a quiet
            # vault) and report the existing counts instead.
            if force or stats.added or stats.updated or stats.deleted:
                links, broken = self._rebuild_links()
                stats.links = links
                stats.broken = broken
            else:
                stats.links = self._conn.execute(
                    "SELECT COUNT(*) FROM links").fetchone()[0]
                stats.broken = self._conn.execute(
                    "SELECT COUNT(*) FROM links WHERE resolved IS NULL").fetchone()[0]
            self._conn.commit()
        stats.elapsed_ms = int((time.monotonic() - t0) * 1000)
        return stats

    def _upsert(self, note: Note, body: str) -> None:
        # ``body`` is the full file text; split the frontmatter off so the
        # FTS table indexes only the prose (not the YAML).
        from src.memory.vault.parser import split_frontmatter
        _, fts_body = split_frontmatter(body)
        self._conn.execute(
            """
            INSERT INTO notes (
                path, folder, stem, stem_lower, path_key, title, summary,
                status, created, updated, tags_json, outlinks_json,
                related_json, spaced_json, missing_fm_json, line_count,
                byte_size, mtime, content_hash, has_frontmatter, is_index,
                is_journal, related_multiline, em_dash, inlink_count
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)
            ON CONFLICT(path) DO UPDATE SET
                folder=excluded.folder, stem=excluded.stem,
                stem_lower=excluded.stem_lower, path_key=excluded.path_key,
                title=excluded.title, summary=excluded.summary,
                status=excluded.status, created=excluded.created,
                updated=excluded.updated, tags_json=excluded.tags_json,
                outlinks_json=excluded.outlinks_json,
                related_json=excluded.related_json, spaced_json=excluded.spaced_json,
                missing_fm_json=excluded.missing_fm_json,
                line_count=excluded.line_count, byte_size=excluded.byte_size,
                mtime=excluded.mtime, content_hash=excluded.content_hash,
                has_frontmatter=excluded.has_frontmatter, is_index=excluded.is_index,
                is_journal=excluded.is_journal,
                related_multiline=excluded.related_multiline, em_dash=excluded.em_dash
            """,
            (
                note.path, note.folder, note.stem, note.stem.lower(),
                link_key(note.path), note.title, note.summary, note.status,
                note.created, note.updated, json.dumps(note.tags),
                json.dumps(note.outlinks), json.dumps(note.related),
                json.dumps(note.spaced_wikilinks),
                json.dumps(note.missing_frontmatter_fields), note.line_count,
                note.byte_size, note.mtime, note.content_hash,
                int(note.has_frontmatter), int(note.is_index),
                int(note.is_journal), int(note.related_multiline),
                int(note.body_has_em_dash),
            ),
        )
        self._conn.execute("DELETE FROM notes_fts WHERE path = ?", (note.path,))
        self._conn.execute(
            "INSERT INTO notes_fts(path, stem, title, summary, body) VALUES (?,?,?,?,?)",
            (note.path, _stem_text(note.stem), note.title, note.summary, fts_body),
        )

    def _delete(self, rel: str) -> None:
        self._conn.execute("DELETE FROM notes WHERE path = ?", (rel,))
        self._conn.execute("DELETE FROM notes_fts WHERE path = ?", (rel,))

    def update_note(self, rel_path: str, content: str) -> None:
        """Single-note upsert from in-memory content (validate-on-write path).
        Does NOT recompute the global graph — call ``sync`` for that."""
        rel = rel_path.replace("\\", "/").lstrip("/")
        note = parse_note_text(rel, content, journal_root=self.journal_root)
        with self._lock:
            self._upsert(note, content)
            self._conn.commit()

    def remove_note(self, rel_path: str) -> None:
        rel = rel_path.replace("\\", "/").lstrip("/")
        with self._lock:
            self._delete(rel)
            self._conn.commit()

    # ── link graph ────────────────────────────────────────────────────

    def _rebuild_links(self) -> tuple[int, int]:
        """Recompute the full link table + inbound counts from the indexed
        notes. Two-tier resolution (exact path, then bare stem) mirrors the
        gateway graph view. Returns ``(total_links, broken_links)``."""
        # ORDER BY path makes bare-stem resolution deterministic: when two
        # notes share a stem, the alphabetically-first path always wins
        # (the gate separately flags the collision).
        cur = self._conn.execute(
            "SELECT path, path_key, stem_lower, outlinks_json FROM notes "
            "ORDER BY path"
        )
        rows = cur.fetchall()
        path_map: dict[str, str] = {}
        stem_map: dict[str, str] = {}
        outlinks: dict[str, list[str]] = {}
        for r in rows:
            path_map[r["path_key"]] = r["path"]
            stem_map.setdefault(r["stem_lower"], r["path"])
            outlinks[r["path"]] = json.loads(r["outlinks_json"] or "[]")

        self._conn.execute("DELETE FROM links")
        self._conn.execute("UPDATE notes SET inlink_count = 0")

        inbound: dict[str, int] = {}
        link_rows: list[tuple] = []
        total = 0
        broken = 0
        for src, targets in outlinks.items():
            seen: set[str] = set()
            for raw in targets:
                key = link_key(raw)
                if not key or key in seen:
                    continue
                seen.add(key)
                resolved = path_map.get(key) or stem_map.get(key.rsplit("/", 1)[-1])
                if resolved == src:
                    continue
                total += 1
                if resolved:
                    inbound[resolved] = inbound.get(resolved, 0) + 1
                else:
                    broken += 1
                link_rows.append((src, raw, key, resolved))

        self._conn.executemany(
            "INSERT INTO links(src, target_raw, target_key, resolved) VALUES (?,?,?,?)",
            link_rows,
        )
        self._conn.executemany(
            "UPDATE notes SET inlink_count = ? WHERE path = ?",
            [(cnt, path) for path, cnt in inbound.items()],
        )
        return total, broken

    # ── queries (used by the gate + REST + tools) ─────────────────────

    def all_notes(self) -> list[Note]:
        with self._lock:
            cur = self._conn.execute("SELECT * FROM notes")
            return [_row_to_note(r) for r in cur.fetchall()]

    def iter_rows(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute("SELECT * FROM notes").fetchall()

    def note_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]

    def inlink_counts(self) -> dict[str, int]:
        """``path -> inbound resolved link count`` for every note."""
        with self._lock:
            cur = self._conn.execute("SELECT path, inlink_count FROM notes")
            return {r["path"]: r["inlink_count"] or 0 for r in cur.fetchall()}

    def get_note(self, rel_path: str) -> Optional[Note]:
        rel = rel_path.replace("\\", "/").lstrip("/")
        with self._lock:
            r = self._conn.execute("SELECT * FROM notes WHERE path = ?", (rel,)).fetchone()
            return _row_to_note(r) if r else None

    def resolve_link(self, target: str) -> Optional[str]:
        """Resolve a wikilink target to a real note path using the same
        two-tier rule as the graph (exact path key, then bare stem), or
        ``None`` if it points nowhere."""
        key = link_key(target)
        if not key:
            return None
        with self._lock:
            r = self._conn.execute(
                "SELECT path FROM notes WHERE path_key = ? ORDER BY path LIMIT 1",
                (key,)
            ).fetchone()
            if r:
                return r["path"]
            bare = key.rsplit("/", 1)[-1]
            r = self._conn.execute(
                "SELECT path FROM notes WHERE stem_lower = ? ORDER BY path LIMIT 1",
                (bare,)
            ).fetchone()
            return r["path"] if r else None

    def broken_links(self) -> list[tuple[str, str]]:
        """``(src_path, target_raw)`` for every unresolved wikilink."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT src, target_raw FROM links WHERE resolved IS NULL"
            )
            return [(r["src"], r["target_raw"]) for r in cur.fetchall()]

    def backlinks(self, rel_path: str) -> list[str]:
        rel = rel_path.replace("\\", "/").lstrip("/")
        with self._lock:
            cur = self._conn.execute(
                "SELECT DISTINCT src FROM links WHERE resolved = ?", (rel,)
            )
            return [r["src"] for r in cur.fetchall()]

    def resolved_degree(self) -> dict[str, int]:
        """``path -> number of distinct resolved outgoing targets``."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT src, COUNT(DISTINCT resolved) AS d FROM links "
                "WHERE resolved IS NOT NULL GROUP BY src"
            )
            return {r["src"]: r["d"] for r in cur.fetchall()}

    def stem_collisions(self) -> dict[str, list[str]]:
        """``stem_lower -> [paths]`` for stems used by more than one note."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT stem_lower, GROUP_CONCAT(path, '\n') AS paths "
                "FROM notes GROUP BY stem_lower HAVING COUNT(*) > 1"
            )
            return {r["stem_lower"]: r["paths"].split("\n") for r in cur.fetchall()}

    def hash_collisions(self) -> dict[str, list[str]]:
        """``content_hash -> [paths]`` for byte-identical notes."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT content_hash, GROUP_CONCAT(path, '\n') AS paths "
                "FROM notes WHERE content_hash != '' "
                "GROUP BY content_hash HAVING COUNT(*) > 1"
            )
            return {r["content_hash"]: r["paths"].split("\n") for r in cur.fetchall()}

    def components(self) -> list[list[str]]:
        """Connected components of the *undirected* resolved-link graph,
        largest first. Isolated notes (no resolved links either way) are
        their own singleton components."""
        with self._lock:
            paths = [r["path"] for r in self._conn.execute("SELECT path FROM notes")]
            edges = self._conn.execute(
                "SELECT src, resolved FROM links WHERE resolved IS NOT NULL"
            ).fetchall()
        parent: dict[str, str] = {p: p for p in paths}

        def find(x: str) -> str:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        for e in edges:
            a, b = e["src"], e["resolved"]
            if a in parent and b in parent:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

        groups: dict[str, list[str]] = {}
        for p in paths:
            groups.setdefault(find(p), []).append(p)
        return sorted(groups.values(), key=len, reverse=True)

    def search_files(self, query: str, limit: int = 50) -> list[dict]:
        """Search note file names/paths only. Queries the ``path`` and
        ``stem_lower`` columns of the ``notes`` table with a case-insensitive
        substring match. Returns ``{path, title, tags}`` for each match."""
        q = (query or "").strip().lower()
        if not q:
            return []
        with self._lock:
            cur = self._conn.execute(
                "SELECT path, title, tags_json FROM notes "
                "WHERE lower(path) LIKE ? OR lower(stem) LIKE ? "
                "ORDER BY path LIMIT ?",
                (f"%{q}%", f"%{q}%", limit),
            )
            return [
                {
                    "path": r["path"],
                    "title": r["title"],
                    "tags": json.loads(r["tags_json"] or "[]"),
                }
                for r in cur.fetchall()
            ]

    def search(self, query: str, limit: int = 25) -> list[dict]:
        """Full-text search over stem/title/summary/body, best first.

        Supports ``"exact phrase"``, ``+required`` and ``-excluded`` terms;
        bare words are OR'd and ranked. See :func:`_fts_query` for why OR is
        the default and precision lives in the ranking.
        """
        q = (query or "").strip()
        if not q:
            return []
        with self._lock:
            try:
                # ``rank`` here is the WEIGHTED bm25 — see _configure_rank.
                cur = self._conn.execute(
                    "SELECT n.path, n.title, n.tags_json, "
                    f"       snippet(notes_fts, 4, '[', ']', '…', {_SNIPPET_TOKENS}) AS snip "
                    "FROM notes_fts f JOIN notes n ON n.path = f.path "
                    "WHERE notes_fts MATCH ? ORDER BY rank LIMIT ?",
                    (_fts_query(q), limit),
                )
                rows = cur.fetchall()
            except (sqlite3.OperationalError, ValueError):
                # ValueError: the query had no FTS-tokenizable terms (pure
                # punctuation). OperationalError: a malformed MATCH. Either
                # way, fall back to a substring scan.
                cur = self._conn.execute(
                    "SELECT path, title, tags_json, '' AS snip FROM notes "
                    "WHERE lower(title) LIKE ? OR lower(stem) LIKE ? LIMIT ?",
                    (f"%{q.lower()}%", f"%{q.lower()}%", limit),
                )
                rows = cur.fetchall()
            return [
                {
                    "path": r["path"],
                    "title": r["title"],
                    "tags": json.loads(r["tags_json"] or "[]"),
                    "snippet": r["snip"],
                }
                for r in rows
            ]

    def journal_anchor_status(self) -> dict[str, bool]:
        """For each journal note, ``True`` if it has at least one resolved
        link to a non-journal (static entity) note — the Company-Brain rule
        that a diary entry must always anchor to a real entity."""
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT n.path AS jpath,
                       MAX(CASE WHEN tn.is_journal = 0 THEN 1 ELSE 0 END) AS anchored
                FROM notes n
                LEFT JOIN links l ON l.src = n.path AND l.resolved IS NOT NULL
                LEFT JOIN notes tn ON tn.path = l.resolved
                WHERE n.is_journal = 1
                GROUP BY n.path
                """
            )
            return {r["jpath"]: bool(r["anchored"]) for r in cur.fetchall()}

    def title_collisions(self) -> dict[str, list[str]]:
        """``normalized_title -> [paths]`` for distinct notes sharing a title
        (a cheap, O(N) near-duplicate signal; deep semantic dedup is the
        AI/dream layer's job)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT lower(trim(title)) AS t, GROUP_CONCAT(path, '\n') AS paths "
                "FROM notes WHERE trim(title) != '' "
                "GROUP BY lower(trim(title)) HAVING COUNT(*) > 1"
            )
            return {r["t"]: r["paths"].split("\n") for r in cur.fetchall()}

    def stats(self) -> dict:
        with self._lock:
            n = self._conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
            links = self._conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]
            broken = self._conn.execute(
                "SELECT COUNT(*) FROM links WHERE resolved IS NULL"
            ).fetchone()[0]
            orphans = self._conn.execute(
                "SELECT COUNT(*) FROM notes WHERE inlink_count = 0 AND is_index = 0"
            ).fetchone()[0]
            by_folder = {
                r["folder"]: r["c"]
                for r in self._conn.execute(
                    "SELECT folder, COUNT(*) AS c FROM notes GROUP BY folder"
                )
            }
        comps = self.components()
        return {
            "notes": n,
            "links": links,
            "broken_links": broken,
            "orphans": orphans,
            "components": len(comps),
            "largest_component": len(comps[0]) if comps else 0,
            "notes_by_folder": by_folder,
        }


# Dropped from a BARE (non-required) term list before matching. English
# only, and deliberately short: the vault is half Italian, and a word we
# drop is a word the agent cannot search for. These are the fillers an
# agent's own phrasing adds ("how do I get to the k8s credentials"), which
# match almost every note and so drown the real terms in the bm25 score.
# Measured on the real vault: dropping them lifted natural-language MRR
# from 0.531 to 0.643 and left title/body-fact recall untouched.
_STOPWORDS = frozenset("""
a an the and or of to in on at is are was were for from by with this that it
be been do does did how what why when where which who i me my we our you your
can could should would get got there here about into over under not no if then
""".split())

# FTS5 operators the agent may type as bare words. ``AND`` is honoured
# (``premium AND google`` plainly means both) rather than searched
# literally, which is what the old builder did: it lower-cased and
# prefix-matched every token, so the AND itself became the term ``and*``.
_CONNECTIVE_AND = "AND"
_CONNECTIVE_OR = "OR"

_QUERY_TOKEN = re.compile(r'([+-]?)(?:"([^"]*)"|(\S+))')
_TERM_CHARS = re.compile(r"[\w]+")


def _quote(term: str) -> str:
    """Wrap a term as an FTS5 string literal, doubling any embedded quote.

    This is the whole safety story, and why the builder no longer has to
    throw the user's syntax away to stay safe: a quoted FTS5 string is
    matched as literal text, so an apostrophe, a stray ``*``, a bare
    ``NEAR(`` or a reserved word inside it cannot become syntax. The old
    builder stripped punctuation with ``re.findall`` for the same reason
    and paid for it by also stripping every operator the agent meant.
    """
    return '"' + term.replace('"', '""') + '"'


def _fts_query(q: str) -> str:
    """Build a safe FTS5 MATCH expression from a freeform query.

    The grammar the agent gets, and why each piece is here:

    * bare words       -> OR'd prefix matches (the DEFAULT).
    * ``"a phrase"``   -> required, matched as an exact phrase.
    * ``+term``        -> required (AND).
    * ``-term``        -> excluded (NOT).
    * ``AND`` / ``OR`` -> make every bare term required / explicitly optional.

    OR is the default because it MEASURED best, which is not the intuition:
    requiring every term looks more precise and scores better on queries
    shaped like a note's title (MRR 1.000 vs 0.967 on the real vault), but
    it collapses on the queries an agent actually writes — a full sentence
    like "how do I notarize the macos desktop app" scored 0.031 against OR's
    0.643, hit@1 of 0/8. The reason is not that AND finds too little: it is
    that AND finds the WRONG thing and so never looks wrong. Some long
    receipt happens to contain every word of the sentence, the result set is
    non-empty, and no "widen when empty" fallback can rescue it. Precision
    therefore comes from RANKING (bm25 weights + the indexed stem), not from
    filtering, and the agent that genuinely needs "both of these" asks for it
    with ``+`` or quotes.
    """
    required: list[str] = []
    optional: list[str] = []
    excluded: list[str] = []
    all_required = False
    saw_operator = False

    for sign, phrase, word in _QUERY_TOKEN.findall(q or ""):
        if phrase is not None and phrase != "":
            # A quoted phrase: keep the whole span, terms and all.
            terms = _TERM_CHARS.findall(phrase.lower())
            if not terms:
                continue
            saw_operator = True
            expr = _quote(" ".join(terms))
            (excluded if sign == "-" else required).append(expr)
            continue
        if not word:
            continue
        if not sign and word in (_CONNECTIVE_AND, _CONNECTIVE_OR):
            saw_operator = True
            all_required = all_required or word == _CONNECTIVE_AND
            continue
        terms = _TERM_CHARS.findall(word.lower())
        if not terms:
            continue
        # A token like ``com.lyramusic`` is several FTS terms; keep it
        # together as a phrase so it stays one unit.
        expr = _quote(" ".join(terms)) + ("*" if len(terms) == 1 else "")
        if sign == "-":
            saw_operator = True
            excluded.append(expr)
        elif sign == "+":
            saw_operator = True
            required.append(expr)
        else:
            optional.append(expr)

    if all_required:
        required, optional = required + optional, []

    # Filler words only ever drown the real terms; but never let them be the
    # reason a query matches nothing (a search for "the who" is legitimate).
    if optional and not required:
        kept = [e for e in optional
                if e.strip('"*').split()[0] not in _STOPWORDS]
        if kept:
            optional = kept

    parts = list(required)
    if optional:
        parts.append("(" + " OR ".join(optional) + ")"
                     if len(optional) > 1 else optional[0])
    if not parts:
        # Nothing searchable (pure punctuation, or only an exclusion) —
        # the caller falls back to a LIKE scan over title/stem.
        raise ValueError("no terms")

    expr = " AND ".join(parts)
    if excluded:
        expr = f"({expr}) NOT ({' OR '.join(excluded)})"
    return expr
