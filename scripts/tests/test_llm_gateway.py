"""Generic LLM gateway — ``POST /api/llm/chat/completions`` + the
``Authorization: Bearer`` auth bypass that makes it OpenAI-client usable.

Two suites, both hermetic (no network, no live gateway):

  llm_gateway  — drives ``llm.handle_chat_completions`` directly against a
                 fake providers DB and a patched ``httpx.AsyncClient``, so
                 we can pin the whole contract: the ``provider:model``
                 split, error statuses, the exact outbound URL / body /
                 headers, verbatim success relay, and clean error relays
                 (upstream non-2xx + timeout) with the handler never raising.

  llm_auth     — drives the REAL ``make_auth_middleware`` to prove the new
                 ``Authorization: Bearer <OPENAGENT_HTTP_TOKEN>`` bypass is
                 accepted, a wrong bearer is rejected, and the original
                 ``X-OpenAgent-Token`` header still works unchanged.

The provider name / model id / base_url / api_key here are all arbitrary
test data — the point of the endpoint is that NONE of them are hardcoded
in the handler.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from ._framework import TestContext, test


# ── Fakes ────────────────────────────────────────────────────────────


class _FakeDB:
    """Minimal stand-in for MemoryDB — only ``list_providers`` is used."""

    def __init__(self, providers: list[dict]) -> None:
        self._providers = providers

    async def list_providers(self, **_kw) -> list[dict]:
        return list(self._providers)


class _FakeResp:
    """httpx-response shape the handler reads: ``status_code`` / ``json()`` / ``text``."""

    def __init__(self, status_code: int, json_data=None, text: str = "") -> None:
        self.status_code = status_code
        self._json = json_data
        self.text = text if text else (json.dumps(json_data) if json_data is not None else "")

    def json(self):
        if self._json is None:
            raise ValueError("no json body")
        return self._json


class _FakeClient:
    """Async context-manager drop-in for ``httpx.AsyncClient`` that records
    the single ``post`` call and returns a canned response (or raises)."""

    def __init__(self, capture: dict, resp: _FakeResp | None, exc: BaseException | None, **kwargs) -> None:
        self._cap = capture
        self._resp = resp
        self._exc = exc
        capture["client_kwargs"] = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def post(self, url, json=None, headers=None):
        self._cap["url"] = url
        self._cap["json"] = json
        self._cap["headers"] = headers
        if self._exc is not None:
            raise self._exc
        return self._resp


@contextmanager
def _patch_httpx(capture: dict, *, resp: _FakeResp | None = None, exc: BaseException | None = None):
    """Patch the httpx.AsyncClient the llm module calls, capturing the request."""
    from src.gateway.api import llm

    def _factory(**kwargs):
        return _FakeClient(capture, resp, exc, **kwargs)

    with patch.object(llm.httpx, "AsyncClient", _factory):
        yield


def _make_request(db, body: dict | None):
    """Build a minimal aiohttp request the handler can read.

    The handler only touches ``request.app['gateway'].agent.memory_db``
    (via ``gateway_db``) and ``await request.json()`` — mirrors the shape
    used by test_gateway_network_api.py.
    """
    from aiohttp.streams import StreamReader
    from aiohttp.test_utils import make_mocked_request
    from unittest.mock import Mock

    raw = json.dumps(body if body is not None else {}).encode()
    headers = {"Content-Type": "application/json", "Content-Length": str(len(raw))}
    fake_app = {"gateway": SimpleNamespace(agent=SimpleNamespace(memory_db=db))}

    protocol = Mock(_reading_paused=False)
    reader = StreamReader(protocol, limit=2**16, loop=None)
    reader.feed_data(raw)
    reader.feed_eof()
    return make_mocked_request(
        "POST", "/api/llm/chat/completions",
        headers=headers, payload=reader, app=fake_app,
    )


def _providers(**overrides) -> list[dict]:
    """One enabled provider row 'local' with a sensible default base_url."""
    row = {
        "name": "local",
        "framework": "api-based",
        "base_url": "http://127.0.0.1:8787/v1",
        "api_key": "provider-secret-key",
        "enabled": True,
    }
    row.update(overrides)
    return [row]


def _parse(resp):
    return json.loads(resp.body)


# ── request validation ───────────────────────────────────────────────


@test("llm_gateway", "model 'provider:model' splits on the first colon → bare model out")
async def t_model_split(ctx: TestContext) -> None:
    from src.gateway.api.llm import handle_chat_completions

    db = _FakeDB(_providers())
    cap: dict = {}
    # A model id that itself contains a colon proves we split ONLY on the first.
    req = _make_request(db, {"model": "local:vendor:model-x", "messages": [{"role": "user", "content": "hi"}]})
    with _patch_httpx(cap, resp=_FakeResp(200, {"ok": True})):
        resp = await handle_chat_completions(req)
    assert resp.status == 200, _parse(resp)
    assert cap["json"]["model"] == "vendor:model-x", cap["json"]["model"]


@test("llm_gateway", "model with no colon → 400 explaining provider:model")
async def t_missing_colon(ctx: TestContext) -> None:
    from src.gateway.api.llm import handle_chat_completions

    db = _FakeDB(_providers())
    for bad in ("claude-haiku", "", "local", ":only-model", "only-provider:"):
        req = _make_request(db, {"model": bad, "messages": []})
        resp = await handle_chat_completions(req)
        assert resp.status == 400, f"model={bad!r} → {resp.status}: {_parse(resp)}"
        assert "provider" in _parse(resp)["error"]["message"].lower()


@test("llm_gateway", "unknown provider → 404 naming the provider")
async def t_unknown_provider(ctx: TestContext) -> None:
    from src.gateway.api.llm import handle_chat_completions

    db = _FakeDB(_providers())  # only 'local' exists
    req = _make_request(db, {"model": "deepseek:deepseek-chat", "messages": []})
    resp = await handle_chat_completions(req)
    assert resp.status == 404, _parse(resp)
    assert "deepseek" in _parse(resp)["error"]["message"]


@test("llm_gateway", "disabled provider → 400 naming the provider")
async def t_disabled_provider(ctx: TestContext) -> None:
    from src.gateway.api.llm import handle_chat_completions

    db = _FakeDB(_providers(enabled=False))
    req = _make_request(db, {"model": "local:m", "messages": []})
    resp = await handle_chat_completions(req)
    assert resp.status == 400, _parse(resp)
    assert "local" in _parse(resp)["error"]["message"]
    assert "disabled" in _parse(resp)["error"]["message"].lower()


@test("llm_gateway", "provider with no base_url → 502 (no hardcoded fallback URL)")
async def t_no_base_url(ctx: TestContext) -> None:
    from src.gateway.api.llm import handle_chat_completions

    for empty in (None, "", "   "):
        db = _FakeDB(_providers(base_url=empty))
        req = _make_request(db, {"model": "local:m", "messages": []})
        resp = await handle_chat_completions(req)
        assert resp.status == 502, f"base_url={empty!r} → {resp.status}"
        assert "base_url" in _parse(resp)["error"]["message"]


# ── outbound request shape ───────────────────────────────────────────


@test("llm_gateway", "base_url join is correct across trailing-slash / /v1 variants")
async def t_base_url_join(ctx: TestContext) -> None:
    from src.gateway.api.llm import handle_chat_completions

    cases = {
        "http://h/v1": "http://h/v1/chat/completions",
        "http://h/v1/": "http://h/v1/chat/completions",
        "http://h": "http://h/chat/completions",
        "http://h/": "http://h/chat/completions",
        "https://api.example.com/openai": "https://api.example.com/openai/chat/completions",
    }
    for base, expected in cases.items():
        db = _FakeDB(_providers(base_url=base))
        cap: dict = {}
        req = _make_request(db, {"model": "local:m", "messages": []})
        with _patch_httpx(cap, resp=_FakeResp(200, {"ok": True})):
            resp = await handle_chat_completions(req)
        assert resp.status == 200, _parse(resp)
        assert cap["url"] == expected, f"base={base!r} → {cap['url']!r} (want {expected!r})"


@test("llm_gateway", "outbound body: bare model + stream=false + pass-through fields")
async def t_outbound_body(ctx: TestContext) -> None:
    from src.gateway.api.llm import handle_chat_completions

    db = _FakeDB(_providers())
    cap: dict = {}
    caller_body = {
        "model": "local:claude-haiku-4-5",
        "messages": [{"role": "system", "content": "grade this"}, {"role": "user", "content": "x"}],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 256,
        "top_p": 0.9,
        "stream": True,               # must be overridden to False
        "some_future_field": "keep",  # arbitrary standard field passes through
    }
    req = _make_request(db, caller_body)
    with _patch_httpx(cap, resp=_FakeResp(200, {"ok": True})):
        resp = await handle_chat_completions(req)
    assert resp.status == 200, _parse(resp)
    out = cap["json"]
    assert out["model"] == "claude-haiku-4-5", out["model"]
    assert out["stream"] is False, out["stream"]
    assert out["messages"] == caller_body["messages"]
    assert out["response_format"] == {"type": "json_object"}
    assert out["temperature"] == 0.2 and out["max_tokens"] == 256 and out["top_p"] == 0.9
    assert out["some_future_field"] == "keep"


@test("llm_gateway", "Authorization header carries the PROVIDER key, not the gateway token")
async def t_provider_key_forwarded(ctx: TestContext) -> None:
    from src.gateway.api.llm import handle_chat_completions

    db = _FakeDB(_providers(api_key="sk-provider-xyz"))
    cap: dict = {}
    req = _make_request(db, {"model": "local:m", "messages": []})
    with _patch_httpx(cap, resp=_FakeResp(200, {"ok": True})):
        resp = await handle_chat_completions(req)
    assert resp.status == 200
    assert cap["headers"]["Authorization"] == "Bearer sk-provider-xyz"
    assert cap["headers"]["Content-Type"] == "application/json"


# ── response relay ───────────────────────────────────────────────────


@test("llm_gateway", "upstream JSON + status relayed verbatim")
async def t_verbatim_relay(ctx: TestContext) -> None:
    from src.gateway.api.llm import handle_chat_completions

    db = _FakeDB(_providers())
    upstream = {
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    cap: dict = {}
    req = _make_request(db, {"model": "local:m", "messages": []})
    with _patch_httpx(cap, resp=_FakeResp(200, upstream)):
        resp = await handle_chat_completions(req)
    assert resp.status == 200
    assert _parse(resp) == upstream, "choices/usage must pass straight through"


@test("llm_gateway", "upstream non-2xx → clean JSON error carrying status + body")
async def t_upstream_error(ctx: TestContext) -> None:
    from src.gateway.api.llm import handle_chat_completions

    db = _FakeDB(_providers())
    cap: dict = {}
    req = _make_request(db, {"model": "local:m", "messages": []})
    with _patch_httpx(cap, resp=_FakeResp(429, {"error": "rate limited"}, text='{"error":"rate limited"}')):
        resp = await handle_chat_completions(req)
    # The upstream status is preserved so a 429 stays a 429 for the client.
    assert resp.status == 429, _parse(resp)
    err = _parse(resp)["error"]
    assert err["upstream_status"] == 429
    assert "rate limited" in err["upstream_body"]


@test("llm_gateway", "timeout → clean 502 JSON error, handler never raises")
async def t_timeout(ctx: TestContext) -> None:
    import httpx

    from src.gateway.api.llm import handle_chat_completions

    db = _FakeDB(_providers())
    cap: dict = {}
    req = _make_request(db, {"model": "local:m", "messages": []})
    with _patch_httpx(cap, exc=httpx.TimeoutException("timed out")):
        resp = await handle_chat_completions(req)  # must not raise
    assert resp.status == 502, _parse(resp)
    assert _parse(resp)["error"]["type"] == "upstream_timeout"


@test("llm_gateway", "connection failure → clean 502 JSON error, handler never raises")
async def t_connect_error(ctx: TestContext) -> None:
    import httpx

    from src.gateway.api.llm import handle_chat_completions

    db = _FakeDB(_providers())
    cap: dict = {}
    req = _make_request(db, {"model": "local:m", "messages": []})
    with _patch_httpx(cap, exc=httpx.ConnectError("refused")):
        resp = await handle_chat_completions(req)  # must not raise
    assert resp.status == 502, _parse(resp)
    assert _parse(resp)["error"]["type"] == "upstream_unreachable"


@test("llm_gateway", "OPENAGENT_LLM_GATEWAY_TIMEOUT overrides the default")
async def t_timeout_env(ctx: TestContext) -> None:
    from src.gateway.api.llm import handle_chat_completions

    db = _FakeDB(_providers())
    cap: dict = {}
    req = _make_request(db, {"model": "local:m", "messages": []})
    prev = os.environ.get("OPENAGENT_LLM_GATEWAY_TIMEOUT")
    os.environ["OPENAGENT_LLM_GATEWAY_TIMEOUT"] = "7.5"
    try:
        with _patch_httpx(cap, resp=_FakeResp(200, {"ok": True})):
            resp = await handle_chat_completions(req)
    finally:
        if prev is None:
            os.environ.pop("OPENAGENT_LLM_GATEWAY_TIMEOUT", None)
        else:
            os.environ["OPENAGENT_LLM_GATEWAY_TIMEOUT"] = prev
    assert resp.status == 200
    assert cap["client_kwargs"].get("timeout") == 7.5, cap["client_kwargs"]


# ══ Auth middleware: Authorization: Bearer bypass ════════════════════


@contextmanager
def _http_token_env(token: str | None):
    """Set/clear OPENAGENT_HTTP_TOKEN and always restore it.

    The middleware captures the token at construction time, so it must be
    set BEFORE ``make_auth_middleware`` is called. Restore is mandatory:
    a leaked token would arm the bypass for later test modules.
    """
    prev = os.environ.get("OPENAGENT_HTTP_TOKEN")
    try:
        if token is None:
            os.environ.pop("OPENAGENT_HTTP_TOKEN", None)
        else:
            os.environ["OPENAGENT_HTTP_TOKEN"] = token
        yield
    finally:
        if prev is None:
            os.environ.pop("OPENAGENT_HTTP_TOKEN", None)
        else:
            os.environ["OPENAGENT_HTTP_TOKEN"] = prev


async def _dial_with_headers(headers: dict, *, token: str):
    """Drive the real auth middleware with a plain (non-agent, non-cert)
    request carrying ``headers``. Returns ``(status, handler_ran)``."""
    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from src.network.auth.middleware import NetworkAuthState, make_auth_middleware

    with _http_token_env(token):
        state = NetworkAuthState(
            coordinator_pubkey=Ed25519PrivateKey.generate().public_key(),
            network_id="net-test",
        )
        middleware = make_auth_middleware(state)

        ran = {"called": False}

        async def handler(request):
            ran["called"] = True
            return web.Response(status=200, text="handler-reached")

        req = make_mocked_request("POST", "/api/llm/chat/completions", headers=headers)
        resp = await middleware(req, handler)
        return resp.status, ran["called"]


@test("llm_auth", "Authorization: Bearer <OPENAGENT_HTTP_TOKEN> is accepted")
async def t_bearer_accepted(ctx: TestContext) -> None:
    token = "s3cret-shared-token"
    status, ran = await _dial_with_headers({"Authorization": f"Bearer {token}"}, token=token)
    assert ran and status == 200, f"valid bearer was refused (status {status})"
    # Prefix is case-insensitive per the OpenAI/HTTP spec.
    status, ran = await _dial_with_headers({"Authorization": f"bearer {token}"}, token=token)
    assert ran and status == 200, "lowercase 'bearer ' prefix was refused"


@test("llm_auth", "a wrong bearer token is rejected (falls through to cert path → 401)")
async def t_bearer_rejected(ctx: TestContext) -> None:
    status, ran = await _dial_with_headers(
        {"Authorization": "Bearer not-the-token"}, token="s3cret-shared-token",
    )
    assert not ran, "handler ran on a wrong bearer token"
    assert status == 401, f"expected 401 for wrong bearer, got {status}"


@test("llm_auth", "X-OpenAgent-Token still works exactly as before")
async def t_x_token_still_works(ctx: TestContext) -> None:
    token = "s3cret-shared-token"
    status, ran = await _dial_with_headers({"X-OpenAgent-Token": token}, token=token)
    assert ran and status == 200, "X-OpenAgent-Token bypass regressed"
    # And a wrong X-OpenAgent-Token is still rejected.
    status, ran = await _dial_with_headers({"X-OpenAgent-Token": "nope"}, token=token)
    assert not ran and status == 401, "wrong X-OpenAgent-Token was accepted"
