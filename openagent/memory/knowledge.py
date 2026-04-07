"""Knowledge base: Obsidian-compatible .md files indexed in SQLite FTS5.

Files are the source of truth. SQL is the search index.
Human-readable, git-versionable, editable in Obsidian or any text editor.

Each memory is a small .md file with YAML frontmatter:

    ---
    topic: deploy
    tags: [k8s, wardrobe, ovh]
    links: [mixout-server-architecture, ovh-vps-setup]
    created: 2026-04-07T12:00:00
    updated: 2026-04-07T12:00:00
    ---
    # Deploy Wardrobe Service
    To deploy the wardrobe service to OVH k3s...

The SQL index stores: file path, topic, tags, links, content hash, timestamps.
FTS5 provides fast full-text search across all memory content.
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

# Frontmatter parsing (simple YAML between --- delimiters)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

KNOWLEDGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_index (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL DEFAULT '',
    topic TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '',
    links TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_knowledge_topic ON knowledge_index(topic);
CREATE INDEX IF NOT EXISTS idx_knowledge_hash ON knowledge_index(content_hash);

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    file_path,
    title,
    topic,
    tags,
    body,
    content='knowledge_index',
    content_rowid='rowid'
);
"""


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content. Returns (metadata, body)."""
    match = _FRONTMATTER_RE.match(content)
    if not match:
        return {}, content

    raw_yaml = match.group(1)
    body = content[match.end():]

    # Simple YAML parser (avoids pyyaml dependency for frontmatter)
    metadata: dict[str, Any] = {}
    for line in raw_yaml.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            # Parse lists: [item1, item2]
            if value.startswith("[") and value.endswith("]"):
                items = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
                metadata[key] = items
            elif value.lower() in ("true", "false"):
                metadata[key] = value.lower() == "true"
            else:
                metadata[key] = value.strip("'\"")

    return metadata, body


def _build_frontmatter(metadata: dict[str, Any]) -> str:
    """Build YAML frontmatter string from metadata dict."""
    lines = ["---"]
    for key, value in metadata.items():
        if isinstance(value, list):
            lines.append(f"{key}: [{', '.join(str(v) for v in value)}]")
        elif isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def _slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:80] or "untitled"


def _content_hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


def _extract_title(body: str) -> str:
    """Extract title from first # heading or first line."""
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line and not line.startswith("---"):
            return line[:80]
    return "Untitled"


def _extract_wikilinks(content: str) -> list[str]:
    """Extract [[wikilink]] targets from content."""
    return _WIKILINK_RE.findall(content)


class KnowledgeBase:
    """Obsidian-compatible knowledge base with SQLite FTS5 index.

    Files are stored as .md in a directory. SQL provides fast search.

    Usage:
        kb = KnowledgeBase(db, Path("./memories"))
        await kb.initialize()
        await kb.add("Deploy Wardrobe", "rsync + docker build...", topic="deploy", tags=["k8s"])
        results = await kb.search("how to deploy wardrobe")
    """

    def __init__(self, db: aiosqlite.Connection, base_dir: str | Path = "./memories"):
        self._db = db
        self.base_dir = Path(base_dir)

    async def initialize(self) -> None:
        """Create tables and sync index with filesystem."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        await self._db.executescript(KNOWLEDGE_SCHEMA)
        await self._db.commit()
        await self.reindex()

    async def add(
        self,
        title: str,
        content: str,
        topic: str = "",
        tags: list[str] | None = None,
        links: list[str] | None = None,
    ) -> str:
        """Create a new memory file and index it. Returns the file path."""
        slug = _slugify(title)
        filename = f"{slug}.md"
        file_path = self.base_dir / filename

        # Avoid collisions
        counter = 1
        while file_path.exists():
            filename = f"{slug}-{counter}.md"
            file_path = self.base_dir / filename
            counter += 1

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        metadata = {
            "topic": topic,
            "tags": tags or [],
            "links": links or [],
            "created": now,
            "updated": now,
        }

        full_content = _build_frontmatter(metadata) + f"# {title}\n\n{content}\n"
        file_path.write_text(full_content, encoding="utf-8")

        await self._index_file(file_path)
        return str(file_path)

    async def update(self, file_path: str, content: str | None = None, **metadata_updates) -> None:
        """Update a memory file's content and/or metadata."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Memory file not found: {file_path}")

        existing = path.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(existing)

        if metadata_updates:
            meta.update(metadata_updates)
        meta["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

        new_body = content if content is not None else body
        full_content = _build_frontmatter(meta) + new_body
        path.write_text(full_content, encoding="utf-8")

        await self._index_file(path)

    async def delete(self, file_path: str) -> None:
        """Delete a memory file and remove from index."""
        path = Path(file_path)
        rel = str(path.relative_to(self.base_dir)) if path.is_absolute() else str(path)

        if path.exists():
            path.unlink()

        await self._db.execute("DELETE FROM knowledge_index WHERE file_path = ?", (rel,))
        await self._db.commit()

    async def search(self, query: str, topic: str | None = None, limit: int = 20) -> list[dict]:
        """Search memories using FTS5 full-text search.

        Returns list of {file_path, title, topic, tags, snippet, rank}.
        """
        if topic:
            cursor = await self._db.execute(
                """
                SELECT ki.file_path, ki.title, ki.topic, ki.tags, ki.links,
                       snippet(knowledge_fts, 4, '<b>', '</b>', '...', 32) as snippet,
                       rank
                FROM knowledge_fts
                JOIN knowledge_index ki ON ki.rowid = knowledge_fts.rowid
                WHERE knowledge_fts MATCH ?
                AND ki.topic = ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, topic, limit),
            )
        else:
            cursor = await self._db.execute(
                """
                SELECT ki.file_path, ki.title, ki.topic, ki.tags, ki.links,
                       snippet(knowledge_fts, 4, '<b>', '</b>', '...', 32) as snippet,
                       rank
                FROM knowledge_fts
                JOIN knowledge_index ki ON ki.rowid = knowledge_fts.rowid
                WHERE knowledge_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            )

        rows = await cursor.fetchall()
        results = []
        for row in rows:
            results.append({
                "file_path": row[0],
                "title": row[1],
                "topic": row[2],
                "tags": row[3],
                "links": row[4],
                "snippet": row[5],
                "rank": row[6],
            })
        return results

    async def get_by_topic(self, topic: str, limit: int = 50) -> list[dict]:
        """Get all memories for a given topic."""
        cursor = await self._db.execute(
            "SELECT * FROM knowledge_index WHERE topic = ? ORDER BY updated_at DESC LIMIT ?",
            (topic, limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_all_topics(self) -> list[str]:
        """List all distinct topics."""
        cursor = await self._db.execute(
            "SELECT DISTINCT topic FROM knowledge_index WHERE topic != '' ORDER BY topic"
        )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]

    async def get_linked(self, title_slug: str) -> list[dict]:
        """Find memories that link to a given title (backlinks)."""
        cursor = await self._db.execute(
            "SELECT * FROM knowledge_index WHERE links LIKE ? ORDER BY updated_at DESC",
            (f"%{title_slug}%",),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def read_file(self, file_path: str) -> str:
        """Read the full content of a memory file."""
        path = self.base_dir / file_path if not Path(file_path).is_absolute() else Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Memory file not found: {file_path}")
        return path.read_text(encoding="utf-8")

    async def build_context(self, query: str, max_tokens: int = 4000) -> str:
        """Build a context string from relevant memories for prompt injection.

        Searches, reads top files, concatenates until max_tokens (approx).
        """
        results = await self.search(query, limit=10)
        if not results:
            return ""

        parts = ["## Relevant Knowledge:"]
        total_chars = 0
        max_chars = max_tokens * 4  # rough char-to-token ratio

        for r in results:
            try:
                content = await self.read_file(r["file_path"])
                _, body = _parse_frontmatter(content)
                body = body.strip()
                if total_chars + len(body) > max_chars:
                    break
                parts.append(f"### {r['title']}\n{body}")
                total_chars += len(body)
            except FileNotFoundError:
                continue

        return "\n\n".join(parts) if len(parts) > 1 else ""

    # ── Internal indexing ──

    async def _index_file(self, path: Path) -> None:
        """Index a single file into SQLite + FTS5."""
        content = path.read_text(encoding="utf-8")
        rel_path = str(path.relative_to(self.base_dir))
        meta, body = _parse_frontmatter(content)
        c_hash = _content_hash(content)

        # Check if already indexed with same hash
        cursor = await self._db.execute(
            "SELECT content_hash FROM knowledge_index WHERE file_path = ?", (rel_path,)
        )
        existing = await cursor.fetchone()
        if existing and existing[0] == c_hash:
            return  # unchanged

        title = _extract_title(body)
        topic = meta.get("topic", "")
        tags = ",".join(meta.get("tags", [])) if isinstance(meta.get("tags"), list) else str(meta.get("tags", ""))
        links_from_meta = meta.get("links", [])
        links_from_body = _extract_wikilinks(content)
        all_links = list(set(links_from_meta if isinstance(links_from_meta, list) else []) | set(links_from_body))
        links_str = ",".join(all_links)
        now = time.time()

        if existing:
            # Update existing
            await self._db.execute(
                "UPDATE knowledge_index SET title=?, topic=?, tags=?, links=?, content_hash=?, updated_at=? WHERE file_path=?",
                (title, topic, tags, links_str, c_hash, now, rel_path),
            )
        else:
            # Insert new
            import uuid
            await self._db.execute(
                "INSERT INTO knowledge_index (id, file_path, title, topic, tags, links, content_hash, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), rel_path, title, topic, tags, links_str, c_hash, now, now),
            )

        # Rebuild FTS for this entry
        rowid_cursor = await self._db.execute(
            "SELECT rowid FROM knowledge_index WHERE file_path = ?", (rel_path,)
        )
        row = await rowid_cursor.fetchone()
        if row:
            rowid = row[0]
            await self._db.execute(
                "INSERT OR REPLACE INTO knowledge_fts(rowid, file_path, title, topic, tags, body) VALUES (?, ?, ?, ?, ?, ?)",
                (rowid, rel_path, title, topic, tags, body),
            )

        await self._db.commit()

    async def reindex(self) -> None:
        """Full reindex: scan all .md files and update index. Skip unchanged files."""
        if not self.base_dir.exists():
            return

        indexed_paths = set()
        for md_file in self.base_dir.rglob("*.md"):
            await self._index_file(md_file)
            indexed_paths.add(str(md_file.relative_to(self.base_dir)))

        # Remove index entries for deleted files
        cursor = await self._db.execute("SELECT file_path FROM knowledge_index")
        rows = await cursor.fetchall()
        for row in rows:
            if row[0] not in indexed_paths:
                await self._db.execute("DELETE FROM knowledge_index WHERE file_path = ?", (row[0],))

        await self._db.commit()

    async def stats(self) -> dict:
        """Return stats about the knowledge base."""
        cursor = await self._db.execute("SELECT COUNT(*) FROM knowledge_index")
        count = (await cursor.fetchone())[0]
        topics = await self.get_all_topics()
        return {
            "total_memories": count,
            "topics": topics,
            "base_dir": str(self.base_dir),
        }
