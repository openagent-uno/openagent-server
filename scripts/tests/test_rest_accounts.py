"""``GET /api/accounts`` — the serving-account view, over fake proxies.

Covers the four provider shapes that exist on a real deployment: a proxy
that reports quota (Codex), one that reports accounts but NO quota (Claude),
one that is down, and an ``https://`` vendor API that must never be probed.

Two of these assertions guard things that would be silent if they broke:

* an account's ``quota`` must be ``null`` — not ``{}`` — when the upstream
  reports none, because an empty object renders as "quota known, all zeroes"
  and 0% used is exactly the number an operator would plan around;
* neither the proxy's own ``api_key`` nor any userinfo embedded in a
  provider ``base_url`` may leave the gateway, since this payload is
  rendered in a UI.
"""
from __future__ import annotations

import json

from aiohttp import web

from ._framework import TestContext, test

CODEX_HEALTH = {
    "status": "ok",
    "metrics": {"active": 2, "queued": 0, "account_switches": 3,
                # Not in the allowlist — must not be relayed.
                "internal_secret": "must-not-pass"},
    "accounts": [
        {"id": "acct-a", "name": "acct-a", "priority": 10, "plan": "pro",
         "managed": True, "limited": False, "limited_until_ms": 0,
         "expires_at_ms": 1788280291000, "has_refresh_token": True,
         # Not in the allowlist — must not be relayed.
         "api_key": "sk-must-not-pass",
         "quota": {"plan": "pro", "primary_used_percent": 40,
                   "primary_window_minutes": 10080,
                   "primary_reset_after_s": 556956, "credits_balance": 0}},
        {"id": "acct-b", "name": "acct-b", "priority": 20, "plan": "pro",
         "managed": True, "limited": True, "limited_until_ms": 1788280000000,
         "quota": {"plan": "pro", "primary_used_percent": 100,
                   "primary_window_minutes": 10080, "primary_reset_after_s": 7200}},
    ],
}

CLAUDE_HEALTH = {
    "status": "ok",
    "metrics": {"active": 1},
    # Accounts, but the upstream reports no quota at all.
    "accounts": [
        {"id": "acct-c", "name": "acct-c", "priority": 10, "managed": True,
         "limited": False, "limited_until_ms": 0, "expires_at_ms": 1817452632180},
    ],
}


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    async def list_providers(self, **_):
        return self._rows


class _FakeRequest:
    """Enough of ``aiohttp.web.Request`` for the handler under test."""

    def __init__(self, db):
        class _Holder:
            pass

        agent = _Holder()
        agent.memory_db = db
        gateway = _Holder()
        gateway.agent = agent
        self.app = {"gateway": gateway}


async def _serve(payload, port):
    app = web.Application()

    async def health(_):
        return web.json_response(payload)

    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


async def _call(rows):
    from src.gateway.api import accounts as api

    resp = await api.handle_list(_FakeRequest(_FakeDB(rows)))
    body = json.loads(resp.body.decode())
    return resp.status, {p["provider"]: p for p in body["providers"]}


@test("rest_accounts", "quota passes through; a quota-less upstream reports null, not zero")
async def t_quota(ctx: TestContext) -> None:
    a = await _serve(CODEX_HEALTH, 18981)
    b = await _serve(CLAUDE_HEALTH, 18982)
    try:
        status, provs = await _call([
            {"name": "withquota", "framework": "api-based", "enabled": 1,
             "base_url": "http://127.0.0.1:18981/v1"},
            {"name": "noquota", "framework": "api-based", "enabled": 1,
             "base_url": "http://127.0.0.1:18982/v1"},
        ])
        assert status == 200

        q = provs["withquota"]
        assert q["reachable"] and len(q["accounts"]) == 2
        assert q["accounts"][0]["quota"]["primary_used_percent"] == 40
        assert q["accounts"][1]["limited"] is True

        n = provs["noquota"]
        assert n["reachable"] and len(n["accounts"]) == 1
        # The whole point: absent quota is null, never an empty object.
        assert n["accounts"][0]["quota"] is None
    finally:
        await a.cleanup()
        await b.cleanup()


@test("rest_accounts", "no credential leaves the gateway (proxy key, base_url userinfo)")
async def t_no_credential_leak(ctx: TestContext) -> None:
    a = await _serve(CODEX_HEALTH, 18983)
    try:
        _status, provs = await _call([
            {"name": "withquota", "framework": "api-based", "enabled": 1,
             "base_url": "http://127.0.0.1:18983/v1"},
            {"name": "embedded", "framework": "api-based", "enabled": 1,
             "base_url": "https://user:tok-secret@api.example.com/v1"},
        ])
        blob = json.dumps(provs)
        assert "sk-must-not-pass" not in blob
        assert "must-not-pass" not in blob      # covers the metric too
        assert "tok-secret" not in blob
        assert provs["embedded"]["base_url"] == "https://***@api.example.com/v1"
    finally:
        await a.cleanup()


@test("rest_accounts", "a dead proxy is data, not a 500; an external API is never probed")
async def t_fail_soft(ctx: TestContext) -> None:
    status, provs = await _call([
        {"name": "down", "framework": "api-based", "enabled": 1,
         "base_url": "http://127.0.0.1:18999/v1"},
        {"name": "vendor", "framework": "api-based", "enabled": 1,
         "base_url": "https://api.deepseek.com"},
        {"name": "off", "framework": "api-based", "enabled": 0,
         "base_url": "http://127.0.0.1:18999/v1"},
    ])
    assert status == 200
    assert provs["down"]["reachable"] is False and provs["down"]["accounts"] == []
    assert provs["down"]["error"]
    assert "not a local proxy" in provs["vendor"]["error"]
    # A disabled provider is not probed at all.
    assert "off" not in provs
