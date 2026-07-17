"""Least-privilege scoped LLM token — ``OPENAGENT_LLM_TOKEN``.

The gateway's token bypass (``make_auth_middleware``) historically accepted a
single shared secret, ``OPENAGENT_HTTP_TOKEN``, as a FULL-ACCESS pass for every
``/api/*`` route. Replio was handed that token so its reply-guard could call
``/api/llm/chat/completions`` — but the same token also opens vault, config,
scheduled-tasks, terminal-backed chat, etc. Least-privilege: a caller that only
needs the LLM gateway should not be able to reach anything else.

The fix adds an OPTIONAL second token, ``OPENAGENT_LLM_TOKEN``, that is accepted
ONLY on paths under ``/api/llm/`` and rejected on every other route; the full
token keeps working everywhere. This suite drives the REAL middleware (not a
re-implementation of its logic) and asserts on what it DID — whether the
downstream handler ran — for both header forms (``X-OpenAgent-Token`` and
``Authorization: Bearer``), because a test that only proved "no exception"
would pass just as happily against a middleware that granted the scoped token
everywhere.

Rejection here means the request fell through the token bypass to the
device-cert path, which — with no cert wire and no agent-ALPN contextvar set —
returns ``401 missing device cert``. So ``status == 401 and not handler_ran`` is
the shape of "this token was NOT accepted on this path".
"""
from __future__ import annotations

import os
from contextlib import contextmanager

from ._framework import TestContext, test

# Two DISTINCT secrets so "the llm token was accepted" can never be confused
# with "the full token was accepted" — they must never be equal.
FULL_TOKEN = "full-access-http-token-AAAA1111"
LLM_TOKEN = "scoped-llm-only-token-BBBB2222"
GARBAGE = "totally-wrong-token-XXXX9999"

# A path under the LLM gateway, and a representative NON-llm route the scoped
# token must never reach. ``/api/config`` is the exact "everything else" the
# least-privilege split is protecting (vault/config/scheduled-tasks/chat/...).
LLM_PATH = "/api/llm/chat/completions"
OTHER_PATHS = ("/api/config", "/api/vault/notes/secrets.md", "/api/mcps")


@contextmanager
def _token_env(*, http_token: str | None, llm_token: str | None):
    """Set/clear BOTH token env vars and always restore them.

    The middleware captures both tokens at construction time, so they must be
    set BEFORE ``make_auth_middleware`` is called. Restore is mandatory:
    ``_TEST_MODULES`` load order is significant and a leaked token would arm the
    bypass for every later test module in the same process.
    """
    prev = {
        "OPENAGENT_HTTP_TOKEN": os.environ.get("OPENAGENT_HTTP_TOKEN"),
        "OPENAGENT_LLM_TOKEN": os.environ.get("OPENAGENT_LLM_TOKEN"),
    }
    want = {"OPENAGENT_HTTP_TOKEN": http_token, "OPENAGENT_LLM_TOKEN": llm_token}
    try:
        for name, val in want.items():
            if val is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = val
        yield
    finally:
        for name, val in prev.items():
            if val is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = val


async def _dial(method: str, path: str, headers: dict, *,
                http_token: str | None, llm_token: str | None):
    """Drive the real auth middleware with a plain (non-agent, non-cert)
    request carrying ``headers``, with the two token env vars pinned to the
    given values BEFORE the middleware is built. Returns ``(status, ran)``.

    No agent-ALPN contextvar and no device-cert wire are set, so a request the
    token bypass does NOT accept falls through to the cert path and gets
    ``401 missing device cert`` — that 401 is our "rejected" signal.
    """
    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from src.network.auth.middleware import NetworkAuthState, make_auth_middleware

    with _token_env(http_token=http_token, llm_token=llm_token):
        state = NetworkAuthState(
            coordinator_pubkey=Ed25519PrivateKey.generate().public_key(),
            network_id="net-test",
        )
        middleware = make_auth_middleware(state)

        ran = {"called": False}

        async def handler(request):
            ran["called"] = True
            return web.Response(status=200, text="handler-reached")

        req = make_mocked_request(method, path, headers=headers)
        resp = await middleware(req, handler)
        return resp.status, ran["called"]


def _x(token: str) -> dict:
    """Present a token via the original ``X-OpenAgent-Token`` header."""
    return {"X-OpenAgent-Token": token}


def _bearer(token: str) -> dict:
    """Present a token via the standard ``Authorization: Bearer`` header."""
    return {"Authorization": f"Bearer {token}"}


# ══ OPENAGENT_LLM_TOKEN SET ══════════════════════════════════════════
# In this block both tokens are configured; the scoped token must reach the
# LLM gateway and nothing else, and the full token must reach everything.


@test("llm_scoped_token", "llm token authenticates /api/llm/* via BOTH header forms")
async def t_llm_token_reaches_llm_path(ctx: TestContext) -> None:
    for label, headers in (("X-OpenAgent-Token", _x(LLM_TOKEN)),
                           ("Authorization: Bearer", _bearer(LLM_TOKEN))):
        status, ran = await _dial(
            "POST", LLM_PATH, headers, http_token=FULL_TOKEN, llm_token=LLM_TOKEN,
        )
        assert ran and status == 200, (
            f"scoped llm token via {label} was refused on {LLM_PATH} (status {status})"
        )


@test("llm_scoped_token", "llm token is REJECTED on non-llm paths (both header forms)")
async def t_llm_token_rejected_off_path(ctx: TestContext) -> None:
    """The whole point of the least-privilege split: the scoped token must not
    reach config/vault/mcps/etc. Rejected ⇒ falls through to the cert path → 401."""
    for path in OTHER_PATHS:
        for label, headers in (("X-OpenAgent-Token", _x(LLM_TOKEN)),
                               ("Authorization: Bearer", _bearer(LLM_TOKEN))):
            status, ran = await _dial(
                "GET", path, headers, http_token=FULL_TOKEN, llm_token=LLM_TOKEN,
            )
            assert not ran, f"scoped llm token via {label} REACHED {path} — scope leaked"
            assert status == 401, (
                f"scoped llm token via {label} on {path}: expected 401, got {status}"
            )


@test("llm_scoped_token", "llm token does not leak to a prefix-adjacent /api/llm... path")
async def t_llm_token_prefix_precision(ctx: TestContext) -> None:
    """Scope is ``startswith('/api/llm/')`` — the trailing slash matters. A route
    that merely starts with the literal ``/api/llm`` but is not under it must stay
    out, exactly like the federation-scope prefix test in test_peer_policy.py."""
    for path in ("/api/llmadmin/secret", "/api/llm-internal", "/api/llm"):
        status, ran = await _dial(
            "GET", path, _bearer(LLM_TOKEN), http_token=FULL_TOKEN, llm_token=LLM_TOKEN,
        )
        assert not ran and status == 401, (
            f"scoped llm token leaked to prefix-adjacent {path} (status {status})"
        )


@test("llm_scoped_token", "full http token authenticates BOTH the llm path and non-llm paths")
async def t_full_token_reaches_everything(ctx: TestContext) -> None:
    for label, hdr in (("X-OpenAgent-Token", _x), ("Authorization: Bearer", _bearer)):
        # LLM path.
        status, ran = await _dial(
            "POST", LLM_PATH, hdr(FULL_TOKEN), http_token=FULL_TOKEN, llm_token=LLM_TOKEN,
        )
        assert ran and status == 200, (
            f"full token via {label} was refused on {LLM_PATH} (status {status})"
        )
        # Every non-llm path.
        for path in OTHER_PATHS:
            status, ran = await _dial(
                "GET", path, hdr(FULL_TOKEN), http_token=FULL_TOKEN, llm_token=LLM_TOKEN,
            )
            assert ran and status == 200, (
                f"full token via {label} was refused on {path} (status {status})"
            )


@test("llm_scoped_token", "a wrong/garbage token is rejected on the llm path AND non-llm paths")
async def t_garbage_rejected_everywhere(ctx: TestContext) -> None:
    for path in (LLM_PATH, *OTHER_PATHS):
        for label, headers in (("X-OpenAgent-Token", _x(GARBAGE)),
                               ("Authorization: Bearer", _bearer(GARBAGE))):
            status, ran = await _dial(
                "POST", path, headers, http_token=FULL_TOKEN, llm_token=LLM_TOKEN,
            )
            assert not ran, f"garbage token via {label} was ACCEPTED on {path}"
            assert status == 401, (
                f"garbage token via {label} on {path}: expected 401, got {status}"
            )


# ══ OPENAGENT_LLM_TOKEN UNSET ════════════════════════════════════════
# Backward compatibility: with no scoped token configured, behaviour must be
# identical to before this token existed — only the full token works, on all
# paths, and no path grants anything to a non-full token.


@test("llm_scoped_token", "llm token UNSET: only the full token works, on all paths")
async def t_unset_full_token_only(ctx: TestContext) -> None:
    for label, hdr in (("X-OpenAgent-Token", _x), ("Authorization: Bearer", _bearer)):
        for path in (LLM_PATH, *OTHER_PATHS):
            status, ran = await _dial(
                path=path, method="POST", headers=hdr(FULL_TOKEN),
                http_token=FULL_TOKEN, llm_token=None,
            )
            assert ran and status == 200, (
                f"with llm token unset, full token via {label} was refused on "
                f"{path} (status {status})"
            )


@test("llm_scoped_token", "llm token UNSET: the would-be scoped token grants nothing, even on /api/llm/*")
async def t_unset_scoped_value_inert(ctx: TestContext) -> None:
    """The load-bearing backward-compat assertion: when ``OPENAGENT_LLM_TOKEN`` is
    unset, presenting what WOULD be the scoped token must NOT open ``/api/llm/*``
    (nor anything else). Otherwise the scoped grant would be armed by a value the
    operator never configured."""
    for path in (LLM_PATH, *OTHER_PATHS):
        for label, headers in (("X-OpenAgent-Token", _x(LLM_TOKEN)),
                               ("Authorization: Bearer", _bearer(LLM_TOKEN))):
            status, ran = await _dial(
                "POST", path, headers, http_token=FULL_TOKEN, llm_token=None,
            )
            assert not ran, (
                f"with llm token unset, the would-be scoped token via {label} "
                f"still reached {path} — the grant armed itself"
            )
            assert status == 401, (
                f"with llm token unset, scoped value via {label} on {path}: "
                f"expected 401, got {status}"
            )
