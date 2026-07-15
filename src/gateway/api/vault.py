"""Vault REST API — CRUD for markdown notes + graph data.

GET  /api/vault/notes           → list all notes with metadata
GET  /api/vault/notes/{path}    → read note content + frontmatter + links
PUT  /api/vault/notes/{path}    → write/update note
DELETE /api/vault/notes/{path}  → delete note
GET  /api/vault/graph           → {nodes, edges} from wikilinks
GET  /api/vault/search?q=...    → full-text search
"""

from __future__ import annotations

import asyncio
from pathlib import Path


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
    """Read a note's frontmatter the same way the rest of the vault does.

    This used to be a FOURTH independent frontmatter parser: its own fence
    scan, a bare ``yaml.safe_load``, and ``except Exception: meta = {}``. That
    last line is the whole problem — a note whose YAML is malformed rendered in
    the app and over REST with **every field blank** (no title, no tags, no
    related) and the reader had no way to tell "this note is empty" from "we
    could not read this note". Exactly the silent degrade gray-matter does on
    the Node side, reproduced in Python.

    It also disagreed with the gate on real notes: stock ``safe_load`` raises on
    ``created: 2024-13-04`` (PyYAML's timestamp resolver rejects month 13 in an
    otherwise perfect document), so the whole note went blank over one bad date
    — while ``_FrontmatterLoader`` drops that resolver and reads it, leaving the
    ``date_format`` rule to report the date, which is the rule that names it.

    ``parser.load_frontmatter_yaml`` is documented as "THE single answer to 'is
    this frontmatter valid YAML?'" and is already what the gate, the doctor's
    repair guard and ``_enforce_write`` ask. This was the one caller that
    didn't. Now there is one parser and one answer.
    """
    from src.memory.vault.parser import (
        FrontmatterSyntaxError,
        load_frontmatter_yaml,
        split_frontmatter,
    )

    raw_fm, body = split_frontmatter(content)
    if raw_fm is None:
        return {}, content
    try:
        meta = load_frontmatter_yaml(raw_fm)
    except FrontmatterSyntaxError:
        # Genuinely unparseable even by the tolerant loader. Still degrade to
        # an empty mapping — the clients type these fields and call array ops
        # on them — but do not pretend it parsed: the body is returned intact
        # so the note is readable, and the gate's ``frontmatter_yaml`` rule is
        # what reports it as broken.
        meta = {}
    return _sanitize(meta), body.strip()


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
    """Raw wikilink targets in ``content``, in order.

    Delegates to the vault parser. This module used to carry its OWN
    ``_WIKILINK_RE`` and ``_link_key`` — a second implementation of
    ``[[target|alias]]`` semantics kept in agreement with
    ``memory/vault/parser.py`` by convention alone. It drifted (see
    ``handle_graph``); the local regex was missing the parser's ``\\n`` guard
    and matched across line breaks. There is now one wikilink parser.
    """
    from src.memory.vault.parser import extract_wikilinks
    return [target for target, _spaced in extract_wikilinks(content)]


def _resolve_vault(request) -> Path:
    gw = request.app["gateway"]
    if gw.vault_path:
        return Path(gw.vault_path).expanduser().resolve()
    from src.core.paths import default_vault_path
    return default_vault_path()


def _service(request):
    """The vault quality service for the gateway's vault (cached per root).

    Resolves through ``_resolve_vault`` — the SAME path the REST CRUD
    handlers use — so the index/gate and the file operations never target
    different folders."""
    from src.memory.vault.service import get_service
    return get_service(_resolve_vault(request))


def _safe_full(vault: Path, note_path: str) -> Path | None:
    """Join ``note_path`` under ``vault`` and confirm it stays inside it.
    Returns ``None`` for a path that escapes the vault (``../`` traversal,
    absolute path, symlink-out), so handlers can reject it."""
    try:
        base = vault.resolve()
        full = (vault / note_path).resolve()
        full.relative_to(base)
    except (ValueError, OSError):
        return None
    return full


def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _validate_on_write() -> bool:
    import os
    return _truthy(os.environ.get("OPENAGENT_VAULT_VALIDATE_ON_WRITE"), default=True)


async def handle_list(request):
    from aiohttp import web
    vault = _resolve_vault(request)
    if not vault.exists():
        return web.json_response({"notes": []})

    notes = []
    for md in sorted(vault.rglob("*.md")):
        if not md.is_file():
            continue  # a directory literally named "*.md"
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
    full = _safe_full(vault, note_path)
    if full is None:
        return web.json_response({"error": "Invalid path"}, status=400)
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
    full = _safe_full(vault, note_path)
    if full is None:
        return web.json_response({"error": "Invalid path"}, status=400)
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    if not isinstance(data, dict):
        return web.json_response({"error": "Body must be a JSON object"}, status=400)
    content = data.get("content", "")
    # Atomic write+index+validate+commit under the service's mutation lock so
    # the change lands in history with its own precise provenance (never swept
    # by the autocommit loop). With the quality gate on (default), mechanical
    # issues are auto-fixed and a structurally-broken note is rejected with
    # 422 + ``errors`` (nothing written); otherwise warnings ride along.
    try:
        from src.memory.vault.vault_origin import origin_from_request
        result = await _service(request).write_note(
            note_path, content, origin_from_request(request),
            validate=_validate_on_write())
    except Exception:  # noqa: BLE001 — never lose the note on a downstream error
        existed = full.exists()
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        gw = request.app.get("gateway")
        if gw is not None:
            await gw.broadcast_resource(
                "vault", "updated" if existed else "created", note_path)
        return web.json_response(
            {"ok": True, "path": note_path, "warnings": [], "commit": None})

    if not result.get("ok", True):
        # Rejected by the quality gate — nothing was written.
        return web.json_response(
            {"ok": False, "path": note_path, "blocked": True,
             "errors": result.get("errors", []),
             "warnings": result.get("warnings", [])}, status=422)

    gw = request.app.get("gateway")
    if gw is not None:
        await gw.broadcast_resource(
            "vault", "updated" if result["existed"] else "created", note_path)
    return web.json_response(
        {"ok": True, "path": note_path, "warnings": result.get("warnings", []),
         "applied": result.get("applied", []), "commit": result.get("commit")})


async def handle_delete(request):
    from aiohttp import web
    vault = _resolve_vault(request)
    note_path = request.match_info["path"]
    full = _safe_full(vault, note_path)
    if full is None:
        return web.json_response({"error": "Invalid path"}, status=400)
    if full.exists():
        commit = None
        try:
            from src.memory.vault.vault_origin import origin_from_request
            result = await _service(request).delete_note(
                note_path, origin_from_request(request))
            commit = result["commit"]
        except Exception:  # noqa: BLE001 — ensure the file is gone regardless
            if full.exists():
                full.unlink()
        gw = request.app.get("gateway")
        if gw is not None:
            await gw.broadcast_resource("vault", "deleted", note_path)
        return web.json_response({"ok": True, "commit": commit})
    return web.json_response({"error": "Not found"}, status=404)


async def handle_search_files(request):
    """GET /api/vault/search/files?q=&limit=50

    Search only note file names/paths (not content). Uses simple case-insensitive
    substring matching against the note path/stem in the FTS5 index.
    """
    from aiohttp import web
    query = (request.query.get("q") or "").strip()
    if not query:
        return web.json_response({"results": []})
    try:
        limit = int(request.query.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    try:
        results = await _service(request).search_files(query, limit=limit)
        return web.json_response({"results": results})
    except Exception:
        return web.json_response({"results": []})


async def handle_search_in_file(request):
    """GET /api/vault/search/in-file?path=&q=&regex=false

    Search for ``q`` within a specific note's content. When ``regex=true``,
    treat ``q`` as a Python regex pattern. Returns line/column positions and
    the matching line text. Reads the file from disk.
    """
    from aiohttp import web
    import re as _re

    vault = _resolve_vault(request)
    note_path = (request.query.get("path") or "").strip()
    query = (request.query.get("q") or "")
    is_regex = _truthy(request.query.get("regex"))

    if not note_path:
        return web.json_response({"error": "path is required"}, status=400)
    if not query:
        return web.json_response({"error": "q is required"}, status=400)

    full = _safe_full(vault, note_path)
    if full is None:
        return web.json_response({"error": "Invalid path"}, status=400)
    if not full.exists() or not full.is_file():
        return web.json_response({"error": "Not found"}, status=404)

    try:
        content = full.read_text(errors="replace")
    except OSError as exc:
        return web.json_response({"error": str(exc)}, status=500)

    matches = []
    lines = content.split("\n")

    if is_regex:
        try:
            pattern = _re.compile(query)
        except _re.error as exc:
            return web.json_response({"error": f"Invalid regex: {exc}"}, status=400)
        for lineno, line in enumerate(lines, start=1):
            for m in pattern.finditer(line):
                matches.append({
                    "line": lineno,
                    "col": m.start() + 1,
                    "text": line,
                })
    else:
        ql = query.lower()
        for lineno, line in enumerate(lines, start=1):
            idx = line.lower().find(ql)
            if idx != -1:
                matches.append({
                    "line": lineno,
                    "col": idx + 1,
                    "text": line,
                })

    return web.json_response({
        "path": note_path,
        "query": query,
        "regex": is_regex,
        "matches": matches,
        "count": len(matches),
    })


async def handle_search(request):
    from aiohttp import web
    vault = _resolve_vault(request)
    query = (request.query.get("q") or "").strip()
    if not query:
        return web.json_response({"results": []})

    # Prefer the FTS5 index — O(log n) and scales to 100k+ notes. Fall back
    # to a linear scan only if the index is unavailable for some reason.
    try:
        limit = int(request.query.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    try:
        results = await _service(request).search(query, limit=limit)
        return web.json_response({"results": results})
    except Exception:  # noqa: BLE001 — degrade to the scan below
        pass

    ql = query.lower()
    results = []
    for md in vault.rglob("*.md"):
        if not md.is_file():
            continue
        content = md.read_text(errors="replace")
        if ql in content.lower() or ql in md.stem.lower():
            meta, _ = _parse_frontmatter(content)
            results.append({
                "path": str(md.relative_to(vault)),
                "title": meta.get("title", md.stem),
                "tags": _tags_list(meta),
            })
    return web.json_response({"results": results})


def _graph_from_index(idx) -> tuple[list[dict], list[dict]]:
    """Build ``(nodes, edges)`` from the index. Sync — the caller threads it.

    Every link decision here is DELEGATED to the index (``note.outlinks`` from
    the one parser, ``idx.resolve_link`` for the one two-tier path-then-stem
    resolution). Nothing about wikilink syntax is re-decided in the gateway.
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    for note in idx.all_notes():
        nodes.append({
            "id": note.path,
            "label": note.title or note.stem,
            "tags": note.tags,
        })
        # Dedup per source: a note that mentions the same target several
        # times must not stack duplicate edges (they skew the force
        # layout's degree-based sizing/colour).
        seen: set[str] = set()
        for raw in note.outlinks:
            target = idx.resolve_link(raw)
            if target and target != note.path and target not in seen:
                seen.add(target)
                edges.append({"source": note.path, "target": target})
    return nodes, edges


async def handle_graph(request):
    """GET /api/vault/graph — {nodes, edges}, served FROM THE VAULT INDEX.

    This endpoint used to do its own ``rglob`` + frontmatter parse + wikilink
    scan + two-tier link resolution, hand-kept "in agreement" with
    ``memory/vault/parser.py`` + ``index.py``. They were not in agreement.
    Measured against the owner's real 2,116-note vault, the two disagreed on:

    * **1 node** — ``_showcase/showcase.md``. The index prunes ``_showcase``,
      ``.git``, ``.obsidian``, ``node_modules`` … (``index._PRUNE_DIRS``); the
      raw ``rglob`` here walked straight into them, so the graph drew a
      derived artifact as if it were memory.
    * **11 edges pointing at the WRONG note.** Both sides fall back to a
      bare-stem match, but the index breaks ties deterministically
      (``ORDER BY path``) while ``rglob`` order is whatever the filesystem
      says. With ``anti-fabrication.md`` existing twice (and ``_index.md``
      sixteen times, ``dev-coverage.md`` three times), ``[[anti-fabrication]]``
      resolved to ``esound/…`` in the picture the user was shown and to
      ``_inherited-from-lyra/…`` in the graph the gate validated.
    * **wikilink syntax** — the local regex had no ``\\n`` guard, so it matched
      ``[[foo`` / newline / ``bar]]`` across a line break; the parser's does not.

    Serving from the index deletes all three classes at once: one walk, one
    parser, one resolver. The index is also the cheaper source (a few SQL
    queries beat re-reading 2,116 files on every graph refresh).
    """
    from aiohttp import web
    vault = _resolve_vault(request)
    if not vault.exists():
        return web.json_response({"nodes": [], "edges": []})

    svc = _service(request)
    idx = await svc._ensure_index()
    await asyncio.to_thread(idx.sync, False)
    nodes, edges = await asyncio.to_thread(_graph_from_index, idx)
    return web.json_response({"nodes": nodes, "edges": edges})


# ── Quality subsystem endpoints ───────────────────────────────────────

async def handle_gate(request):
    """GET /api/vault/gate?strict=&limit= — run the quality gate and return
    the structured report (violations grouped by rule + health stats)."""
    from aiohttp import web
    import dataclasses
    svc = _service(request)
    cfg = svc.config
    if _truthy(request.query.get("strict")):
        cfg = dataclasses.replace(cfg, strict=True)
    try:
        limit = int(request.query.get("limit", 500))
    except (TypeError, ValueError):
        limit = 500
    rep = await svc.gate(config=cfg)
    body = rep.to_dict()
    if len(body["violations"]) > limit:
        body["violations"] = body["violations"][:limit]
        body["violations_truncated"] = True
    return web.json_response(body)


async def handle_doctor(request):
    """POST /api/vault/doctor?apply= — mechanically fix what code can fix and
    list the rest as suggestions. ``apply=false`` is a dry run."""
    from aiohttp import web
    apply = _truthy(request.query.get("apply"))
    svc = _service(request)
    result = await svc.doctor(apply=apply)
    if apply and result["fix"]["files_changed"]:
        gw = request.app.get("gateway")
        if gw is not None:
            await gw.broadcast_resource("vault", "changed")
    return web.json_response(result)


async def handle_index_sync(request):
    """POST /api/vault/index/sync?force= — reconcile the index with disk."""
    from aiohttp import web
    force = _truthy(request.query.get("force"))
    svc = _service(request)
    return web.json_response(await svc.sync(force=force))


async def handle_derived(request):
    """POST /api/vault/derived — regenerate llms.txt + _showcase/showcase.md."""
    from aiohttp import web
    svc = _service(request)
    res = await svc.regenerate_derived()
    gw = request.app.get("gateway")
    if gw is not None:
        await gw.broadcast_resource("vault", "changed")
    return web.json_response(res)


async def handle_stats(request):
    """GET /api/vault/stats — vault health summary."""
    from aiohttp import web
    return web.json_response(await _service(request).stats())


async def handle_history(request):
    """GET /api/vault/history?path=&limit= — the vault's git history with
    provenance. ``path`` (optional) scopes it to one note/folder."""
    from aiohttp import web
    path = request.query.get("path") or None
    try:
        limit = int(request.query.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    commits = await _service(request).git_log(limit, path)
    return web.json_response({"commits": commits, "path": path})


async def handle_commit(request):
    """GET /api/vault/commit?hash= — the changes a single commit introduced
    (metadata, the files it touched, and the unified diff)."""
    from aiohttp import web
    ref = (request.query.get("hash") or "").strip()
    if not ref:
        return web.json_response({"error": "hash is required"}, status=400)
    detail = await _service(request).git_show(ref)
    if detail is None:
        return web.json_response({"error": "Unknown commit"}, status=404)
    return web.json_response(detail)


async def handle_restore(request):
    """POST /api/vault/restore {hash} — non-destructively roll the vault back
    to the state at a commit (adds a new commit; history is preserved)."""
    from aiohttp import web
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    ref = (body.get("hash") or "").strip()
    if not ref:
        return web.json_response({"error": "hash is required"}, status=400)
    from src.memory.vault.vault_origin import origin_from_request
    result = await _service(request).restore_to(ref, origin_from_request(request))
    if "error" in result:
        return web.json_response(result, status=409)
    gw = request.app.get("gateway")
    if gw is not None:
        await gw.broadcast_resource("vault", "changed")
    return web.json_response(result)


async def handle_reset(request):
    """POST /api/vault/reset {hash, confirm:true} — DESTRUCTIVELY make a
    commit the latest, deleting every commit after it. Requires an explicit
    ``confirm`` flag (the clients gate this behind a confirmation prompt)."""
    from aiohttp import web
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    ref = (body.get("hash") or "").strip()
    if not ref:
        return web.json_response({"error": "hash is required"}, status=400)
    if body.get("confirm") is not True:
        return web.json_response(
            {"error": "This permanently deletes later commits — set "
                      "\"confirm\": true to proceed."}, status=400)
    from src.memory.vault.vault_origin import origin_from_request
    result = await _service(request).reset_to(ref, origin_from_request(request))
    if "error" in result:
        return web.json_response(result, status=409)
    gw = request.app.get("gateway")
    if gw is not None:
        await gw.broadcast_resource("vault", "changed")
    return web.json_response(result)


async def handle_init(request):
    """POST /api/vault/init — scaffold the folder system + canon + journal."""
    from aiohttp import web
    res = await _service(request).init_taxonomy()
    gw = request.app.get("gateway")
    if gw is not None and res.get("created"):
        await gw.broadcast_resource("vault", "changed")
    return web.json_response(res)


async def handle_move(request):
    """POST /api/vault/move {from, to} — move/rename a note or folder and
    rewrite every inbound wikilink so nothing breaks."""
    from aiohttp import web
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    src = (body.get("from") or "").strip()
    dst = (body.get("to") or "").strip()
    if not src or not dst:
        return web.json_response({"error": "from and to are required"}, status=400)
    vault = _resolve_vault(request)
    if _safe_full(vault, src) is None or _safe_full(vault, dst) is None:
        return web.json_response({"error": "Invalid path"}, status=400)
    result = await _service(request).move(src, dst)
    if "error" in result:
        return web.json_response(result, status=409)
    gw = request.app.get("gateway")
    if gw is not None:
        await gw.broadcast_resource("vault", "changed")
    return web.json_response(result)
