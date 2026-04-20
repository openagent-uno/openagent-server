"""Marketplace — search and install MCP servers from the official registry.

Thin proxy + schema translator on top of
``https://registry.modelcontextprotocol.io/v0.1`` (the canonical, Apache-2.0,
no-auth community registry — Smithery and PulseMCP both mirror it). The
desktop UI calls these endpoints; the gateway hides the upstream so we can
later add PulseMCP as a fallback without touching the client.

Three endpoints:

  GET  /api/marketplace/search?q=&cursor=
       → list of {server, _meta} cards (forwarded from /v0.1/servers).

  GET  /api/marketplace/servers?name=&version=latest
       → {server: <server.json>, requirements: <synthesized>}
       The "requirements" object enumerates which env vars / headers /
       placeholders the user must fill in before installing — letting the
       UI render a generic form without parsing server.json itself.

  POST /api/marketplace/install
       Body: {name, version, choice:{kind, index}, install_name?, env, headers, placeholders}
       → 201 {ok, mcp}.
       Re-fetches the cached server.json, walks the chosen package/remote,
       resolves the runtime (npx/uvx/docker/...) into a concrete argv,
       substitutes {placeholders} in arg/header/url values, and writes a
       row to the ``mcps`` table with kind="custom" and
       source="marketplace:registry.modelcontextprotocol.io@{version}".
       The pool's hot-reload picks the new server up on the next message.

Caching: small in-memory LRU on the running gateway. Search responses live
300 s (queries change as users type); per-server-version detail lives 3600 s
because server.json is immutable per version. Cache evaporates on restart.

Rate-limit posture: the registry publishes no numerical limit. Its only
documented policy is for catalog aggregators ("scrape on a regular but
infrequent basis, e.g., once per hour"). One desktop user typing into a
search box, with the cache, is far inside that envelope. We still send a
descriptive User-Agent and back off on 5xx/429 so we're a polite client if
the registry ever introduces throttling.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import OrderedDict
from typing import Any
from urllib.parse import quote

from openagent.core.logging import elog
from openagent.gateway.api._common import gateway_db as _db


REGISTRY_BASE = "https://registry.modelcontextprotocol.io/v0.1"
USER_AGENT = "OpenAgent-Marketplace/1.0 (+https://github.com/openagent)"
HTTP_TIMEOUT = 10  # seconds for upstream calls
SEARCH_TTL = 300
SERVER_TTL = 3600
CACHE_MAX = 200
RETRY_BACKOFF = (1.0, 2.0, 4.0)  # 5xx / 429 retries
PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")

# The set of runtimes we know how to assemble into argv. Anything else → 422
# at install time so we don't silently emit broken commands.
_KNOWN_RUNTIMES = {"npx", "uvx", "docker", "dnx"}

# Many registry entries omit ``runtimeHint`` and only set ``registryType``
# (the registry the artefact lives in). Map those to the conventional
# runner that pulls + executes from that registry. Mirrors how a human
# developer would translate "npm package X" → ``npx X``.
_REGISTRY_TYPE_TO_RUNTIME = {
    "npm": "npx",
    "pypi": "uvx",
    "oci": "docker",
    "docker": "docker",
    "nuget": "dnx",
}


def _resolve_runtime(pkg: dict) -> str | None:
    """Pick a runtime for a package: explicit hint wins, else infer from registryType."""
    hint = pkg.get("runtimeHint")
    if hint:
        return hint
    rtype = pkg.get("registryType")
    if rtype:
        return _REGISTRY_TYPE_TO_RUNTIME.get(rtype, rtype)
    return None

# ────────────────────────────────────────────────────────────────────────────
# Cache
# ────────────────────────────────────────────────────────────────────────────


def _cache(request) -> "OrderedDict[tuple, tuple[float, Any]]":
    """LRU-ish dict bound to the running Gateway. Created lazily."""
    app = request.app
    cache = app.get("marketplace_cache")
    if cache is None:
        cache = OrderedDict()
        app["marketplace_cache"] = cache
    return cache


def _cache_get(cache, key, ttl: float):
    entry = cache.get(key)
    if entry is None:
        return None
    inserted, value = entry
    if time.time() - inserted > ttl:
        cache.pop(key, None)
        return None
    cache.move_to_end(key)
    return value


def _cache_put(cache, key, value):
    cache[key] = (time.time(), value)
    cache.move_to_end(key)
    while len(cache) > CACHE_MAX:
        cache.popitem(last=False)


# ────────────────────────────────────────────────────────────────────────────
# Upstream HTTP
# ────────────────────────────────────────────────────────────────────────────


async def _fetch_json(url: str) -> tuple[int, Any, str]:
    """GET ``url`` with retries on 5xx/429. Returns (status, json|None, text).

    Raises only on connection-level errors (those become 502 in the caller).
    """
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=HTTP_TIMEOUT)
    last_status = 0
    last_text = ""
    headers = {"User-Agent": USER_AGENT, "accept": "application/json"}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        for attempt, delay in enumerate((0.0, *RETRY_BACKOFF)):
            if delay:
                await asyncio.sleep(delay)
            async with session.get(url) as resp:
                last_status = resp.status
                if resp.status < 500 and resp.status != 429:
                    if resp.status >= 400:
                        last_text = (await resp.text())[:500]
                        return resp.status, None, last_text
                    try:
                        body = await resp.json(content_type=None)
                    except Exception as e:
                        return resp.status, None, f"upstream returned non-JSON: {e}"
                    return resp.status, body, ""
                # 5xx / 429 → record and retry
                last_text = (await resp.text())[:500]
                elog(
                    "marketplace.upstream_retry",
                    attempt=attempt,
                    status=last_status,
                    url=url,
                )
    return last_status, None, last_text


# ────────────────────────────────────────────────────────────────────────────
# server.json schema helpers
# ────────────────────────────────────────────────────────────────────────────


def _placeholders_in(value: str | None) -> list[str]:
    if not value:
        return []
    return list(dict.fromkeys(PLACEHOLDER_RE.findall(value)))


def _env_field(item: dict) -> dict:
    """Normalise an ``environmentVariables[]`` or ``headers[]`` item."""
    return {
        "name": item.get("name"),
        "isSecret": bool(item.get("isSecret", False)),
        "isRequired": bool(item.get("isRequired", False)),
        "description": item.get("description"),
        "default": item.get("default"),
        # The literal template; placeholders get substituted at install time.
        "value_template": item.get("value"),
    }


def _collect_arg_placeholders(args: list[dict]) -> list[dict]:
    """Walk runtimeArguments/packageArguments and collect placeholder tokens."""
    out: list[dict] = []
    seen: set[str] = set()
    for arg in args or []:
        for token in _placeholders_in(arg.get("value")):
            if token in seen:
                continue
            seen.add(token)
            out.append({
                "token": token,
                "description": arg.get("description"),
            })
    return out


def _synthesise_requirements(server: dict) -> dict:
    """Walk server.json packages[] and remotes[] into a UI-friendly shape."""
    packages_out: list[dict] = []
    for i, pkg in enumerate(server.get("packages") or []):
        runtime = _resolve_runtime(pkg) or "unknown"
        env_required = [_env_field(e) for e in (pkg.get("environmentVariables") or [])]
        placeholders = _collect_arg_placeholders(
            (pkg.get("runtimeArguments") or []) + (pkg.get("packageArguments") or [])
        )
        packages_out.append({
            "index": i,
            "runtime": runtime,
            "registryType": pkg.get("registryType"),
            "identifier": pkg.get("identifier") or pkg.get("name"),
            "version": pkg.get("version"),
            "transport": (pkg.get("transport") or {}).get("type"),
            "env_required": env_required,
            "placeholders": placeholders,
            "supported": runtime in _KNOWN_RUNTIMES,
        })

    remotes_out: list[dict] = []
    for i, remote in enumerate(server.get("remotes") or []):
        header_required = [_env_field(h) for h in (remote.get("headers") or [])]
        # Placeholders can appear in the URL too; surface them.
        placeholders = []
        for token in _placeholders_in(remote.get("url")):
            placeholders.append({"token": token, "description": "URL placeholder"})
        remotes_out.append({
            "index": i,
            "url": remote.get("url"),
            "transport": (remote.get("transport") or {}).get("type") or "streamable-http",
            "header_required": header_required,
            "placeholders": placeholders,
        })

    return {"packages": packages_out, "remotes": remotes_out}


def _substitute(template: str | None, values: dict) -> str | None:
    if template is None:
        return None
    return PLACEHOLDER_RE.sub(lambda m: str(values.get(m.group(1), m.group(0))), template)


def _resolve_arg_values(args: list[dict], placeholders: dict) -> list[str]:
    """Turn server.json runtime/package arguments into a flat argv slice.

    Each arg is either positional ({type: "positional", value, ...}) or
    named ({type: "named", name, value, ...}). For named with no value, we
    skip — that's a flag-only switch like ``--verbose`` controlled elsewhere.
    """
    out: list[str] = []
    for arg in args or []:
        atype = arg.get("type") or "positional"
        value = _substitute(arg.get("value"), placeholders)
        if atype == "named":
            name = arg.get("name")
            if name:
                out.append(name)
            if value is not None and value != "":
                out.append(value)
        else:  # positional
            if value is not None:
                out.append(value)
    return out


def _build_install_payload(
    server: dict,
    choice: dict,
    env_in: dict,
    headers_in: dict,
    placeholders: dict,
) -> tuple[dict, str | None]:
    """Map server.json + user inputs → kwargs for ``db.upsert_mcp``.

    Returns (kwargs, error). On unsupported runtime / missing required value,
    returns ({}, "<error message>") so the caller can 400/422.
    """
    kind = choice.get("kind")
    idx = choice.get("index", 0)

    # ── Package (stdio) ──
    if kind == "package":
        packages = server.get("packages") or []
        if not 0 <= idx < len(packages):
            return {}, f"package index {idx} out of range"
        pkg = packages[idx]
        runtime = _resolve_runtime(pkg)
        identifier = pkg.get("identifier") or pkg.get("name")
        version = pkg.get("version")
        if runtime not in _KNOWN_RUNTIMES:
            return {}, f"unsupported runtime {runtime!r} (no install recipe)"
        if not identifier:
            return {}, "package is missing 'identifier'"

        runtime_args = _resolve_arg_values(pkg.get("runtimeArguments") or [], placeholders)
        package_args = _resolve_arg_values(pkg.get("packageArguments") or [], placeholders)

        if runtime == "npx":
            command = ["npx"]
            spec = f"{identifier}@{version}" if version else identifier
            args = ["-y", *runtime_args, spec, *package_args]
        elif runtime == "uvx":
            command = ["uvx"]
            spec = f"{identifier}=={version}" if version else identifier
            args = [*runtime_args, spec, *package_args]
        elif runtime == "dnx":
            command = ["dnx"]
            args = [*runtime_args, identifier, *package_args]
        else:  # docker / oci
            command = ["docker"]
            ref = f"{identifier}:{version}" if version else identifier
            args = ["run", "-i", "--rm", *runtime_args, ref, *package_args]

        # Resolve env: defaults + caller overrides; substitute placeholders.
        env: dict[str, str] = {}
        missing: list[str] = []
        for entry in pkg.get("environmentVariables") or []:
            name = entry.get("name")
            if not name:
                continue
            supplied = env_in.get(name)
            if supplied is None or supplied == "":
                tmpl = _substitute(entry.get("value"), placeholders)
                if tmpl not in (None, ""):
                    env[name] = tmpl
                elif entry.get("default") is not None:
                    env[name] = str(entry["default"])
                elif entry.get("isRequired"):
                    missing.append(name)
            else:
                env[name] = str(supplied)
        if missing:
            return {}, f"missing required env vars: {', '.join(missing)}"

        return {
            "command": command,
            "args": args,
            "url": None,
            "env": env,
            "headers": {},
        }, None

    # ── Remote (HTTP/SSE) ──
    if kind == "remote":
        remotes = server.get("remotes") or []
        if not 0 <= idx < len(remotes):
            return {}, f"remote index {idx} out of range"
        remote = remotes[idx]
        url = _substitute(remote.get("url"), placeholders)
        if not url:
            return {}, "remote is missing 'url'"

        headers: dict[str, str] = {}
        missing: list[str] = []
        for entry in remote.get("headers") or []:
            name = entry.get("name")
            if not name:
                continue
            supplied = headers_in.get(name)
            if supplied is None or supplied == "":
                tmpl = _substitute(entry.get("value"), placeholders)
                if tmpl not in (None, ""):
                    headers[name] = tmpl
                elif entry.get("default") is not None:
                    headers[name] = str(entry["default"])
                elif entry.get("isRequired"):
                    missing.append(name)
            else:
                headers[name] = str(supplied)
        if missing:
            return {}, f"missing required headers: {', '.join(missing)}"

        return {
            "command": None,
            "args": [],
            "url": url,
            "env": {},
            "headers": headers,
        }, None

    return {}, f"unknown choice.kind {kind!r} (expected 'package' or 'remote')"


def _default_install_name(registry_name: str) -> str:
    """``io.github.foo/postgres`` → ``postgres``; fallback: full registry name."""
    if not registry_name:
        return "mcp-marketplace"
    tail = registry_name.rsplit("/", 1)[-1]
    # Normalise to a slug — names go straight into the mcps PRIMARY KEY.
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", tail).strip("-").lower()
    return slug or "mcp-marketplace"


async def _suggest_unique_name(db, base: str) -> str:
    for n in range(2, 100):
        candidate = f"{base}-{n}"
        if await db.get_mcp(candidate) is None:
            return candidate
    return f"{base}-{int(time.time())}"


# ────────────────────────────────────────────────────────────────────────────
# Handlers
# ────────────────────────────────────────────────────────────────────────────


async def handle_search(request):
    """GET /api/marketplace/search?q=&cursor=&limit="""
    from aiohttp import web

    q = (request.query.get("q") or "").strip()
    cursor = request.query.get("cursor") or ""
    limit_raw = request.query.get("limit") or "30"
    try:
        limit = max(1, min(100, int(limit_raw)))
    except ValueError:
        limit = 30

    cache = _cache(request)
    key = ("search", q, cursor, limit)
    cached = _cache_get(cache, key, SEARCH_TTL)
    if cached is not None:
        return web.json_response(cached)

    params = [f"limit={limit}", "version=latest"]
    if q:
        params.append(f"search={quote(q)}")
    if cursor:
        params.append(f"cursor={quote(cursor)}")
    url = f"{REGISTRY_BASE}/servers?{'&'.join(params)}"

    try:
        status, body, text = await _fetch_json(url)
    except Exception as exc:
        elog("marketplace.search_failed", error=str(exc))
        return web.json_response(
            {"error": f"registry unreachable: {exc}"}, status=502,
        )
    if body is None:
        return web.json_response(
            {"error": f"registry returned {status}", "detail": text},
            status=502 if status >= 500 else status,
        )

    raw_servers = body.get("servers") or []
    cards: list[dict] = []
    for entry in raw_servers:
        server = entry.get("server") or {}
        cards.append({
            "name": server.get("name"),
            "version": server.get("version"),
            "title": server.get("title"),
            "description": server.get("description"),
            "status": server.get("status"),
            # Pass through useful _meta extras (publisher, etc.) opaquely.
            "_meta": entry.get("_meta") or {},
        })
    metadata = body.get("metadata") or {}
    out = {
        "servers": cards,
        "nextCursor": metadata.get("nextCursor"),
        "count": metadata.get("count", len(cards)),
    }
    _cache_put(cache, key, out)
    return web.json_response(out)


async def handle_server_detail(request):
    """GET /api/marketplace/servers?name=&version="""
    from aiohttp import web

    name = (request.query.get("name") or "").strip()
    version = (request.query.get("version") or "latest").strip()
    if not name:
        return web.json_response({"error": "name is required"}, status=400)

    cache = _cache(request)
    key = ("server", name, version)
    cached = _cache_get(cache, key, SERVER_TTL)
    if cached is None:
        url = f"{REGISTRY_BASE}/servers/{quote(name, safe='')}/versions/{quote(version, safe='')}"
        try:
            status, body, text = await _fetch_json(url)
        except Exception as exc:
            elog("marketplace.detail_failed", name=name, error=str(exc))
            return web.json_response(
                {"error": f"registry unreachable: {exc}"}, status=502,
            )
        if body is None:
            return web.json_response(
                {"error": f"registry returned {status}", "detail": text},
                status=502 if status >= 500 else status,
            )
        # Detail endpoint returns a single server entry shaped like a list item:
        # {server: <server.json>, _meta: {...}}. Keep both halves.
        server = body.get("server") or body
        meta = body.get("_meta") or {}
        cached = {"server": server, "_meta": meta}
        _cache_put(cache, key, cached)

    server = cached["server"]
    return web.json_response({
        "server": server,
        "_meta": cached.get("_meta", {}),
        "requirements": _synthesise_requirements(server),
    })


async def handle_install(request):
    """POST /api/marketplace/install — write a row to the mcps table."""
    from aiohttp import web

    db = _db(request)
    if db is None:
        return web.json_response({"error": "memory DB not available"}, status=500)

    body = await request.json() if request.can_read_body else {}
    name = str(body.get("name") or "").strip()
    version = str(body.get("version") or "latest").strip()
    choice = body.get("choice") or {}
    env_in = dict(body.get("env") or {})
    headers_in = dict(body.get("headers") or {})
    placeholders = dict(body.get("placeholders") or {})
    install_name = str(body.get("install_name") or "").strip() or _default_install_name(name)

    if not name:
        return web.json_response({"error": "name is required"}, status=400)
    if not isinstance(choice, dict) or "kind" not in choice or "index" not in choice:
        return web.json_response(
            {"error": "choice must be {kind: 'package'|'remote', index: N}"},
            status=400,
        )

    # Re-fetch (cache hit) so install never trusts a stale UI snapshot.
    cache = _cache(request)
    cache_key = ("server", name, version)
    cached = _cache_get(cache, cache_key, SERVER_TTL)
    if cached is None:
        url = f"{REGISTRY_BASE}/servers/{quote(name, safe='')}/versions/{quote(version, safe='')}"
        try:
            status, upstream, text = await _fetch_json(url)
        except Exception as exc:
            return web.json_response(
                {"error": f"registry unreachable: {exc}"}, status=502,
            )
        if upstream is None:
            return web.json_response(
                {"error": f"registry returned {status}", "detail": text},
                status=502 if status >= 500 else status,
            )
        server = upstream.get("server") or upstream
        cached = {"server": server, "_meta": upstream.get("_meta") or {}}
        _cache_put(cache, cache_key, cached)
    server = cached["server"]
    resolved_version = server.get("version") or version

    # Map server.json → mcps row kwargs.
    kwargs, err = _build_install_payload(server, choice, env_in, headers_in, placeholders)
    if err:
        # Distinguish "missing user input" (400) from "schema we can't handle" (422).
        if err.startswith("missing required") or "out of range" in err or err.startswith("unknown choice"):
            return web.json_response({"error": err}, status=400)
        return web.json_response({"error": err}, status=422)

    # Duplicate-name check.
    if await db.get_mcp(install_name) is not None:
        suggested = await _suggest_unique_name(db, install_name)
        return web.json_response(
            {
                "error": f"an MCP named {install_name!r} already exists",
                "suggested_name": suggested,
            },
            status=409,
        )

    await db.upsert_mcp(
        install_name,
        kind="custom",
        command=kwargs["command"],
        args=kwargs["args"],
        url=kwargs["url"],
        env=kwargs["env"],
        headers=kwargs["headers"],
        oauth=False,
        enabled=True,
        source=f"marketplace:registry.modelcontextprotocol.io@{resolved_version}",
    )
    elog(
        "marketplace.install",
        registry_name=name,
        install_name=install_name,
        version=resolved_version,
        kind=choice.get("kind"),
    )
    return web.json_response(
        {"ok": True, "mcp": await db.get_mcp(install_name)},
        status=201,
    )
