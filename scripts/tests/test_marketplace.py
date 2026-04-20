"""Marketplace — schema mapping + REST endpoints for /api/marketplace/*.

Pure-unit coverage of the schema translator (server.json → mcps row): no
network. The live registry calls are exercised by the smoke section in the
plan file (curl). What we test here are the bits that would silently
break on a registry schema change:

  - placeholder collection in args / urls
  - env / header field normalisation (defaults, secrets, requireds)
  - argv assembly per runtime (npx / uvx / docker / dnx)
  - missing-required-env returns an error
  - placeholder substitution in arg values, URLs, headers
  - unsupported runtime is rejected at install (no silent broken commands)
  - default install_name slug derivation
  - in-memory LRU cache TTL + eviction

Plus one REST shape test that runs against the gateway fixture if present
(skipped otherwise — same pattern as test_mcps_rest.py).
"""
from __future__ import annotations

from collections import OrderedDict

from ._framework import TestContext, TestSkip, test


# ────────────────────────────────────────────────────────────────────────
# Schema synthesis
# ────────────────────────────────────────────────────────────────────────


@test("marketplace", "_synthesise_requirements walks packages + remotes")
async def t_synth_requirements(ctx: TestContext) -> None:
    from openagent.gateway.api.marketplace import _synthesise_requirements

    server = {
        "name": "io.example/test",
        "version": "1.0.0",
        "packages": [
            {
                "registryType": "npm",
                "runtimeHint": "npx",
                "identifier": "@example/mcp",
                "version": "1.0.0",
                "transport": {"type": "stdio"},
                "environmentVariables": [
                    {"name": "API_KEY", "isRequired": True, "isSecret": True,
                     "description": "Your example API key"},
                    {"name": "DEBUG", "default": "false"},
                ],
                "packageArguments": [
                    {"type": "positional", "value": "{target_dir}",
                     "description": "Directory to operate on"},
                ],
            }
        ],
        "remotes": [
            {
                "url": "https://api.example.com/mcp/{tenant}",
                "transport": {"type": "streamable-http"},
                "headers": [
                    {"name": "Authorization", "value": "Bearer {api_key}",
                     "isRequired": True, "isSecret": True},
                ],
            }
        ],
    }
    req = _synthesise_requirements(server)
    assert len(req["packages"]) == 1
    pkg = req["packages"][0]
    assert pkg["runtime"] == "npx"
    assert pkg["supported"] is True
    assert {f["name"] for f in pkg["env_required"]} == {"API_KEY", "DEBUG"}
    api_key_field = next(f for f in pkg["env_required"] if f["name"] == "API_KEY")
    assert api_key_field["isSecret"] is True
    assert api_key_field["isRequired"] is True
    debug_field = next(f for f in pkg["env_required"] if f["name"] == "DEBUG")
    assert debug_field["default"] == "false"
    assert any(p["token"] == "target_dir" for p in pkg["placeholders"])

    assert len(req["remotes"]) == 1
    rem = req["remotes"][0]
    assert rem["url"] == "https://api.example.com/mcp/{tenant}"
    assert rem["transport"] == "streamable-http"
    assert any(p["token"] == "tenant" for p in rem["placeholders"])
    assert rem["header_required"][0]["name"] == "Authorization"


@test("marketplace", "_synthesise_requirements flags unknown runtime as unsupported")
async def t_synth_unknown_runtime(ctx: TestContext) -> None:
    from openagent.gateway.api.marketplace import _synthesise_requirements

    server = {
        "packages": [
            {"runtimeHint": "weird-runtime", "identifier": "x", "version": "1"},
        ],
    }
    req = _synthesise_requirements(server)
    assert req["packages"][0]["supported"] is False


@test("marketplace", "registryType inferred to runtime when runtimeHint missing")
async def t_runtime_inferred(ctx: TestContext) -> None:
    """Real-world: most registry entries set ``registryType: 'npm'`` and
    omit ``runtimeHint`` entirely. We must default to ``npx`` for those,
    not silently mark them as unsupported."""
    from openagent.gateway.api.marketplace import (
        _resolve_runtime, _synthesise_requirements, _build_install_payload,
    )

    assert _resolve_runtime({"registryType": "npm"}) == "npx"
    assert _resolve_runtime({"registryType": "pypi"}) == "uvx"
    assert _resolve_runtime({"registryType": "oci"}) == "docker"
    assert _resolve_runtime({"registryType": "nuget"}) == "dnx"
    # Explicit hint always wins.
    assert _resolve_runtime({"registryType": "npm", "runtimeHint": "uvx"}) == "uvx"

    # End-to-end: an npm package with no runtimeHint installs as npx.
    server = {
        "packages": [{
            "registryType": "npm",
            "identifier": "@letta-ai/memory-mcp",
            "version": "2.0.2",
        }],
    }
    req = _synthesise_requirements(server)
    assert req["packages"][0]["supported"] is True
    assert req["packages"][0]["runtime"] == "npx"
    kwargs, err = _build_install_payload(
        server, {"kind": "package", "index": 0}, {}, {}, {},
    )
    assert err is None
    assert kwargs["command"] == ["npx"]
    assert kwargs["args"] == ["-y", "@letta-ai/memory-mcp@2.0.2"]


# ────────────────────────────────────────────────────────────────────────
# Install payload assembly
# ────────────────────────────────────────────────────────────────────────


@test("marketplace", "npx install builds [npx, -y, identifier@version, ...args]")
async def t_install_npx(ctx: TestContext) -> None:
    from openagent.gateway.api.marketplace import _build_install_payload

    server = {
        "packages": [{
            "runtimeHint": "npx",
            "identifier": "@modelcontextprotocol/server-memory",
            "version": "0.1.0",
            "packageArguments": [
                {"type": "positional", "value": "--readonly"},
            ],
            "environmentVariables": [
                {"name": "MEMORY_PATH", "default": "/tmp/mem"},
            ],
        }],
    }
    kwargs, err = _build_install_payload(
        server, {"kind": "package", "index": 0}, {}, {}, {},
    )
    assert err is None, err
    assert kwargs["command"] == ["npx"]
    assert kwargs["args"] == [
        "-y", "@modelcontextprotocol/server-memory@0.1.0", "--readonly",
    ]
    assert kwargs["env"] == {"MEMORY_PATH": "/tmp/mem"}
    assert kwargs["url"] is None


@test("marketplace", "uvx install builds [uvx, identifier==version, ...args]")
async def t_install_uvx(ctx: TestContext) -> None:
    from openagent.gateway.api.marketplace import _build_install_payload

    server = {
        "packages": [{
            "runtimeHint": "uvx",
            "identifier": "mcp-server-fetch",
            "version": "0.2.0",
        }],
    }
    kwargs, err = _build_install_payload(
        server, {"kind": "package", "index": 0}, {}, {}, {},
    )
    assert err is None
    assert kwargs["command"] == ["uvx"]
    assert kwargs["args"] == ["mcp-server-fetch==0.2.0"]


@test("marketplace", "docker install wraps in docker run -i --rm <image>:<tag>")
async def t_install_docker(ctx: TestContext) -> None:
    from openagent.gateway.api.marketplace import _build_install_payload

    server = {
        "packages": [{
            "runtimeHint": "docker",
            "identifier": "ghcr.io/example/mcp",
            "version": "v1.2.3",
        }],
    }
    kwargs, err = _build_install_payload(
        server, {"kind": "package", "index": 0}, {}, {}, {},
    )
    assert err is None
    assert kwargs["command"] == ["docker"]
    assert kwargs["args"] == ["run", "-i", "--rm", "ghcr.io/example/mcp:v1.2.3"]


@test("marketplace", "missing required env var returns 'missing required env vars'")
async def t_install_missing_env(ctx: TestContext) -> None:
    from openagent.gateway.api.marketplace import _build_install_payload

    server = {
        "packages": [{
            "runtimeHint": "npx",
            "identifier": "x",
            "version": "1.0.0",
            "environmentVariables": [
                {"name": "BRAVE_API_KEY", "isRequired": True, "isSecret": True},
            ],
        }],
    }
    _, err = _build_install_payload(
        server, {"kind": "package", "index": 0}, {}, {}, {},
    )
    assert err is not None
    assert "BRAVE_API_KEY" in err
    assert err.startswith("missing required")

    # Now supply it — should succeed.
    kwargs, err = _build_install_payload(
        server, {"kind": "package", "index": 0},
        {"BRAVE_API_KEY": "sk-test"}, {}, {},
    )
    assert err is None
    assert kwargs["env"] == {"BRAVE_API_KEY": "sk-test"}


@test("marketplace", "unsupported runtime returns schema error (422 territory)")
async def t_install_unsupported_runtime(ctx: TestContext) -> None:
    from openagent.gateway.api.marketplace import _build_install_payload

    server = {"packages": [{"runtimeHint": "scratch", "identifier": "x"}]}
    _, err = _build_install_payload(
        server, {"kind": "package", "index": 0}, {}, {}, {},
    )
    assert err is not None
    assert "unsupported runtime" in err


@test("marketplace", "remote install substitutes URL + header placeholders")
async def t_install_remote_placeholders(ctx: TestContext) -> None:
    from openagent.gateway.api.marketplace import _build_install_payload

    server = {
        "remotes": [{
            "url": "https://server.smithery.ai/{slug}/mcp",
            "transport": {"type": "streamable-http"},
            "headers": [
                {"name": "Authorization", "value": "Bearer {api_key}",
                 "isRequired": True, "isSecret": True},
            ],
        }],
    }
    kwargs, err = _build_install_payload(
        server, {"kind": "remote", "index": 0}, {}, {},
        {"slug": "memory-mcp", "api_key": "sk-abc"},
    )
    assert err is None
    assert kwargs["url"] == "https://server.smithery.ai/memory-mcp/mcp"
    assert kwargs["headers"] == {"Authorization": "Bearer sk-abc"}
    assert kwargs["command"] is None


@test("marketplace", "named runtime args emit name then value")
async def t_install_named_args(ctx: TestContext) -> None:
    from openagent.gateway.api.marketplace import _resolve_arg_values

    args = [
        {"type": "named", "name": "--port", "value": "{port}"},
        {"type": "positional", "value": "{root}"},
        {"type": "named", "name": "--verbose"},  # flag-only — value missing
    ]
    out = _resolve_arg_values(args, {"port": "8080", "root": "/srv"})
    assert out == ["--port", "8080", "/srv", "--verbose"]


# ────────────────────────────────────────────────────────────────────────
# install_name derivation
# ────────────────────────────────────────────────────────────────────────


@test("marketplace", "_default_install_name slugifies tail of registry name")
async def t_default_name(ctx: TestContext) -> None:
    from openagent.gateway.api.marketplace import _default_install_name

    assert _default_install_name("io.github.foo/postgres") == "postgres"
    assert _default_install_name("io.modelcontextprotocol/servers-memory") == "servers-memory"
    assert _default_install_name("Weird Name With Spaces") == "weird-name-with-spaces"
    assert _default_install_name("") == "mcp-marketplace"
    assert _default_install_name("///") == "mcp-marketplace"


# ────────────────────────────────────────────────────────────────────────
# LRU cache
# ────────────────────────────────────────────────────────────────────────


@test("marketplace", "cache returns hit within TTL, evicts past TTL, caps at MAX")
async def t_cache_lru(ctx: TestContext) -> None:
    import time
    from openagent.gateway.api import marketplace as mp

    cache: OrderedDict = OrderedDict()
    mp._cache_put(cache, ("a",), {"x": 1})
    assert mp._cache_get(cache, ("a",), ttl=10) == {"x": 1}

    # Past-TTL miss.
    cache.clear()
    cache[("b",)] = (time.time() - 1000, {"y": 2})
    assert mp._cache_get(cache, ("b",), ttl=10) is None
    assert ("b",) not in cache  # evicted on read

    # Cap eviction.
    cache.clear()
    for i in range(mp.CACHE_MAX + 5):
        mp._cache_put(cache, (i,), i)
    assert len(cache) == mp.CACHE_MAX
    # Oldest 5 dropped.
    assert (0,) not in cache
    assert (mp.CACHE_MAX + 4,) in cache


# ────────────────────────────────────────────────────────────────────────
# REST shape (only when gateway fixture exists)
# ────────────────────────────────────────────────────────────────────────


@test("marketplace", "POST /api/marketplace/install rejects missing 'name'")
async def t_install_rest_400(ctx: TestContext) -> None:
    import aiohttp

    port = ctx.extras.get("gateway_port")
    if not port:
        raise TestSkip("requires gateway fixture")
    agent = ctx.extras.get("agent")
    if not agent or not getattr(agent, "memory_db", None):
        raise TestSkip("gateway fixture has no MemoryDB wired")

    async with aiohttp.ClientSession() as sess:
        async with sess.post(
            f"http://127.0.0.1:{port}/api/marketplace/install",
            json={"choice": {"kind": "package", "index": 0}},
        ) as resp:
            assert resp.status == 400, await resp.text()
            body = await resp.json()
            assert "name" in body["error"].lower()
