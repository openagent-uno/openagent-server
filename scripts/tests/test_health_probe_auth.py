"""Health-probe auth exemption — ``GET /api/health`` bypasses auth, nothing else.

A k8s liveness/readiness probe issues a PLAIN httpGet with no device cert and no
bearer token; before this change every ``/api/*`` route (health included) 401'd,
so an earlier attempt smuggled a token into an exec probe, mangled its shell
quoting, and crash-looped a pod. The fix exempts EXACTLY ``/api/health`` in
``make_auth_middleware`` — an exact path match, never a prefix — so the probe
works while every other route (config, vault, mcps, and the sensitive
``/api/health/ingest``) still requires a credential.

This suite drives the REAL middleware (like ``test_llm_scoped_token``) and
asserts on whether the downstream handler ran, and separately proves the health
PAYLOAD leaks nothing sensitive even when ``runtime_info`` carries secrets.
"""
from __future__ import annotations

import json

from ._framework import TestContext, test


async def _dial(method: str, path: str):
    """Drive the real auth middleware with a plain (no cert, no token, no
    agent-ALPN) request. Returns ``(status, handler_ran)``. A request the
    middleware does NOT exempt falls through to the cert path → 401."""
    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from src.network.auth.middleware import NetworkAuthState, make_auth_middleware

    state = NetworkAuthState(
        coordinator_pubkey=Ed25519PrivateKey.generate().public_key(),
        network_id="net-test",
    )
    middleware = make_auth_middleware(state)

    ran = {"called": False}

    async def handler(request):
        ran["called"] = True
        return web.Response(status=200, text="handler-reached")

    req = make_mocked_request(method, path)
    resp = await middleware(req, handler)
    return resp.status, ran["called"]


@test("health_probe", "unauthenticated GET /api/health is exempt (handler runs, 200)")
async def t_health_exempt(ctx: TestContext) -> None:
    status, ran = await _dial("GET", "/api/health")
    assert ran and status == 200, (
        f"an unauthenticated liveness probe on /api/health was blocked "
        f"(status {status}, handler_ran={ran})"
    )


@test("health_probe", "unauthenticated /api/config and other /api/* still 401")
async def t_others_still_authed(ctx: TestContext) -> None:
    for method, path in (
        ("GET", "/api/config"),
        ("PUT", "/api/config"),
        ("GET", "/api/vault/notes/secrets.md"),
        ("POST", "/api/mcps"),
        ("GET", "/api/agent-info"),
    ):
        status, ran = await _dial(method, path)
        assert not ran, f"{method} {path} reached the handler unauthenticated — auth leaked"
        assert status == 401, f"{method} {path}: expected 401 unauthenticated, got {status}"


@test("health_probe", "the exemption is EXACT — no prefix/adjacent path is opened")
async def t_exact_match_only(ctx: TestContext) -> None:
    """Critically ``/api/health/ingest`` (accepts HealthKit metrics) must stay
    authed — a prefix match would have opened it. Adjacent names stay closed too."""
    for method, path in (
        ("POST", "/api/health/ingest"),
        ("GET", "/api/health/ingest"),
        ("GET", "/api/health/"),
        ("GET", "/api/healthz"),
        ("GET", "/api/healthcheck"),
        ("GET", "/api/health/extra"),
    ):
        status, ran = await _dial(method, path)
        assert not ran and status == 401, (
            f"exemption leaked to {method} {path} (status {status}, ran={ran}) — "
            f"must be an EXACT match on /api/health only"
        )


@test("health_probe", "the health payload leaks no secret/config/session data")
async def t_health_payload_minimal(ctx: TestContext) -> None:
    """Call the real ``handle_health`` with a gateway whose ``runtime_info``
    carries sensitive fields (node_id, api_key, network). The response must
    surface ONLY liveness/basic status — proving the now-unauthenticated
    endpoint reveals nothing sensitive regardless of what runtime_info holds."""
    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request

    from src.gateway.api import health

    SECRET_NODE = "SECRET_NODE_ID_deadbeef"
    SECRET_KEY = "sk-super-secret-should-never-appear"

    class _FakeGateway:
        clients: dict = {}

        def runtime_info(self):
            # Deliberately includes fields handle_health must NOT surface.
            return {
                "agent": "supportbot",
                "version": "9.9.9",
                "node_id": SECRET_NODE,
                "network": "prod-net",
                "role": "coordinator",
                "api_key": SECRET_KEY,
            }

    app = web.Application()
    app["gateway"] = _FakeGateway()
    req = make_mocked_request("GET", "/api/health", app=app)
    resp = await health.handle_health(req)

    raw = resp.body.decode("utf-8") if isinstance(resp.body, (bytes, bytearray)) else str(resp.body)
    body = json.loads(raw)

    # Liveness essentials are present…
    assert body.get("status") == "ok", body
    assert body.get("agent") == "supportbot" and body.get("version") == "9.9.9", body
    assert body.get("connected_clients") == 0, body

    # …and nothing sensitive leaked, by key OR by value.
    for leaked_key in ("node_id", "api_key", "network", "role"):
        assert leaked_key not in body, f"health payload leaked sensitive key {leaked_key!r}: {body}"
    assert SECRET_NODE not in raw and SECRET_KEY not in raw, (
        f"a secret value from runtime_info appeared in the health body: {raw}"
    )
    # The payload is a tight allow-list, not a filtered dump.
    assert set(body) == {"status", "agent", "version", "connected_clients"}, body
