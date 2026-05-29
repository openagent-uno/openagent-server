"""Vault REST API — CRUD for markdown notes + graph data.

GET  /api/vault/notes           → list all notes with metadata
GET  /api/vault/notes/{path}    → read note content + frontmatter + links
PUT  /api/vault/notes/{path}    → write/update note
DELETE /api/vault/notes/{path}  → delete note
GET  /api/vault/graph           → {nodes, edges} from wikilinks
GET  /api/vault/search?q=...    → full-text search
"""

from __future__ import annotations

import re
from pathlib import Path

_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


def _sanitize(obj):
    """Convert datetime/date to ISO strings for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            import yaml
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except Exception:
                meta = {}
            return _sanitize(meta), parts[2].strip()
    return {}, content


def _tags_list(meta: dict) -> list[str]:
    """Normalize a frontmatter ``tags`` value to a list.

    YAML parses ``tags: foo`` as a bare scalar string (and ``tags:``
    with no value as ``None``). Clients type this field as a string
    array and call array ops on it (``tags.slice(...).join(...)``) — a
    stray scalar leaking through crashes them. Always hand back a list.
    """
    tags = meta.get("tags")
    if isinstance(tags, str):
        return [tags]
    if isinstance(tags, list):
        return tags
    return []


def _scan_wikilinks(content: str) -> list[str]:
    return _WIKILINK_RE.findall(content)


def _link_key(s: str) -> str:
    """Normalize a wikilink target — or a note's vault-relative path — to
    a comparison key: lowercased, no surrounding slashes, no ``.md``
    suffix, and with any ``#heading`` / ``^block`` anchor dropped.

    Vault notes link each other by folder-relative path
    (``[[infra/scheduled-jobs]]``) far more than by bare name; keying the
    graph lookup on the bare filename stem alone dropped ~70% of edges.
    """
    k = s.lower().strip().lstrip("/")
    k = k.split("#", 1)[0].split("^", 1)[0].strip()
    if k.startswith("./"):
        k = k[2:]
    if k.endswith(".md"):
        k = k[:-3]
    return k


def _resolve_vault(request) -> Path:
    gw = request.app["gateway"]
    if gw.vault_path:
        return Path(gw.vault_path).expanduser().resolve()
    from src.core.paths import default_vault_path
    return default_vault_path()


async def handle_list(request):
    from aiohttp import web
    vault = _resolve_vault(request)
    if not vault.exists():
        return web.json_response({"notes": []})

    notes = []
    for md in sorted(vault.rglob("*.md")):
        rel = str(md.relative_to(vault))
        content = md.read_text(errors="replace")
        meta, _ = _parse_frontmatter(content)
        stat = md.stat()
        notes.append({
            "path": rel,
            "title": meta.get("title", md.stem),
            "tags": _tags_list(meta),
            "type": meta.get("type", ""),
            "modified": stat.st_mtime,
            "size": stat.st_size,
        })
    return web.json_response({"notes": notes})


async def handle_read(request):
    from aiohttp import web
    vault = _resolve_vault(request)
    note_path = request.match_info["path"]
    full = vault / note_path
    if not full.exists() or not full.is_file():
        return web.json_response({"error": "Not found"}, status=404)

    content = full.read_text(errors="replace")
    meta, body = _parse_frontmatter(content)
    return web.json_response({
        "path": note_path,
        "content": content,
        "frontmatter": meta,
        "body": body,
        "links": _scan_wikilinks(content),
        "modified": full.stat().st_mtime,
    })


async def handle_write(request):
    from aiohttp import web
    vault = _resolve_vault(request)
    note_path = request.match_info["path"]
    full = vault / note_path
    existed = full.exists()
    full.parent.mkdir(parents=True, exist_ok=True)
    data = await request.json()
    full.write_text(data.get("content", ""))
    gw = request.app.get("gateway")
    if gw is not None:
        await gw.broadcast_resource(
            "vault", "updated" if existed else "created", note_path,
        )
    return web.json_response({"ok": True, "path": note_path})


async def handle_delete(request):
    from aiohttp import web
    vault = _resolve_vault(request)
    note_path = request.match_info["path"]
    full = vault / note_path
    if full.exists():
        full.unlink()
        gw = request.app.get("gateway")
        if gw is not None:
            await gw.broadcast_resource("vault", "deleted", note_path)
        return web.json_response({"ok": True})
    return web.json_response({"error": "Not found"}, status=404)


async def handle_search(request):
    from aiohttp import web
    vault = _resolve_vault(request)
    query = (request.query.get("q") or "").lower().strip()
    if not query:
        return web.json_response({"results": []})

    results = []
    for md in vault.rglob("*.md"):
        content = md.read_text(errors="replace")
        if query in content.lower() or query in md.stem.lower():
            meta, _ = _parse_frontmatter(content)
            results.append({
                "path": str(md.relative_to(vault)),
                "title": meta.get("title", md.stem),
                "tags": _tags_list(meta),
            })
    return web.json_response({"results": results})


async def handle_graph(request):
    from aiohttp import web
    vault = _resolve_vault(request)
    if not vault.exists():
        return web.json_response({"nodes": [], "edges": []})

    nodes, edges = [], []
    # Resolve wikilinks by folder path AND by bare note name: notes link
    # each other both ways — ``[[infra/scheduled-jobs]]`` (path) and
    # ``[[scheduled-jobs]]`` (bare). ``path_map`` is the exact match;
    # ``stem_map`` is the last-segment fallback for bare links.
    path_map: dict[str, str] = {}
    stem_map: dict[str, str] = {}
    note_data: dict[str, dict] = {}

    for md in vault.rglob("*.md"):
        rel = str(md.relative_to(vault))
        content = md.read_text(errors="replace")
        meta, _ = _parse_frontmatter(content)
        path_map[_link_key(rel)] = rel
        stem_map.setdefault(md.stem.lower(), rel)
        note_data[rel] = {"meta": meta, "links": _scan_wikilinks(content)}

    for rel, data in note_data.items():
        meta = data["meta"]
        nodes.append({
            "id": rel,
            "label": meta.get("title", Path(rel).stem),
            "tags": _tags_list(meta),
            "type": meta.get("type", ""),
        })
        # Dedup per source: a note that mentions the same target several
        # times must not stack duplicate edges (they skew the force
        # layout's degree-based sizing/colour).
        seen: set[str] = set()
        for link in data["links"]:
            key = _link_key(link)
            target = path_map.get(key) or stem_map.get(key.rsplit("/", 1)[-1])
            if target and target != rel and target not in seen:
                seen.add(target)
                edges.append({"source": rel, "target": target})

    return web.json_response({"nodes": nodes, "edges": edges})
