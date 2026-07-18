"""Credential pool: a native provider rotates across N accounts on 429/529
*before* the turn spills to the configured secondary (DeepSeek).

LLM-free. A ``_FakeModel`` with mutable ``api_key`` / ``base_url`` /
``client`` / ``async_client`` and a synthetic ``aresponse`` /
``aresponse_stream`` stands in for the runtime Model, so these pin the
rotation mechanism (pool selection + the fallback.py seam) without a live
provider. The header note in :mod:`src.models.credential_pool` documents the
inert-by-default contract these guard.
"""
from __future__ import annotations

import time

from src.core.runtime_errors import ModelProviderError, ModelRateLimitError
from src.models.providers.response import ModelResponse

from ._framework import TestContext, test


# ── Fakes ────────────────────────────────────────────────────────────
class _Resp:
    """Minimal stand-in for a ModelResponse carrying the key that served it."""

    def __init__(self, key: str) -> None:
        self.key = key


class _FakeModel:
    """Runtime-Model shape the seam mutates: api_key/base_url/client swap."""

    def __init__(self, *, api_key: str, base_url: str = "http://proxy/v1",
                 provider: str = "local", id: str = "local:claude") -> None:
        self.id = id
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider
        self.client = object()
        self.async_client = object()
        # aresponse behaviour keyed by the CURRENT api_key
        self.fail_keys: set[str] = set()          # -> ModelRateLimitError(429)
        self.error_map: dict[str, int] = {}        # key -> non-429 status code
        self.calls: list[str] = []                 # api_key per aresponse call
        # aresponse_stream behaviour
        self.stream_fail_keys: set[str] = set()    # raise BEFORE first event
        self.stream_midfail_keys: set[str] = set()  # yield one event THEN raise
        self.stream_calls: list[str] = []

    async def aresponse(self, **kwargs):
        self.calls.append(self.api_key)
        if self.api_key in self.error_map:
            raise ModelProviderError("boom", status_code=self.error_map[self.api_key])
        if self.api_key in self.fail_keys:
            raise ModelRateLimitError("rate limited", status_code=429)
        return _Resp(self.api_key)

    async def aresponse_stream(self, **kwargs):
        self.stream_calls.append(self.api_key)
        if self.api_key in self.stream_fail_keys:
            raise ModelRateLimitError("rate limited", status_code=429)
        if self.api_key in self.stream_midfail_keys:
            yield ModelResponse(event="content", content="partial-" + self.api_key)
            raise ModelRateLimitError("rate limited mid-stream", status_code=429)
        yield ModelResponse(event="content", content="ok-" + self.api_key)


class _FakeFallback:
    """A fallback (DeepSeek) model that records whether it was consulted."""

    def __init__(self, id: str = "deepseek:deepseek-v4-pro") -> None:
        self.id = id
        self.called = False
        self.stream_called = False

    async def aresponse(self, **kwargs):
        self.called = True
        return _Resp("fallback")

    async def aresponse_stream(self, **kwargs):
        self.stream_called = True
        yield ModelResponse(event="content", content="fallback")


def _attach_pool(model: _FakeModel, pool) -> None:
    """Simulate native_provider Seam A: seed the model from the pool."""
    sel = pool.select()
    model.api_key = sel.api_key
    model.base_url = sel.base_url
    model.client = model.async_client = None
    model._openagent_cred_pool = pool


def _acct(key: str, base_url: str = "http://proxy/v1"):
    from src.models.credential_pool import PooledAccount

    return PooledAccount(api_key=key, base_url=base_url)


# ── 1. Rotate before fallback ────────────────────────────────────────
@test("credential_pool", "rotates across accounts before falling back to deepseek")
async def t_rotate_before_fallback(_ctx: TestContext) -> None:
    from src.models.credential_pool import CredentialPool
    from src.models.providers.fallback import FallbackConfig, acall_model_with_fallback

    pool = CredentialPool([_acct("ka"), _acct("kb"), _acct("kc")], strategy="fill_first")
    model = _FakeModel(api_key="ka")
    model.fail_keys = {"ka", "kb"}
    _attach_pool(model, pool)  # selects ka (fill_first)
    fb = _FakeFallback()
    fc = FallbackConfig(on_rate_limit=[fb])

    result = await acall_model_with_fallback(model, fc, messages=[])

    assert result.key == "kc", result.key
    assert model.calls == ["ka", "kb", "kc"], model.calls
    assert fb.called is False, "DeepSeek must NOT be consulted while an account is free"
    statuses = {a.api_key: a.last_status for a in pool._accounts}
    assert statuses == {"ka": "EXHAUSTED", "kb": "EXHAUSTED", "kc": "OK"}, statuses


# ── 2. Pool exhausted → existing fallback fires ──────────────────────
@test("credential_pool", "pool exhausted falls through to the existing fallback")
async def t_pool_exhausted_falls_back(_ctx: TestContext) -> None:
    from src.models.credential_pool import CredentialPool
    from src.models.providers.fallback import FallbackConfig, acall_model_with_fallback

    pool = CredentialPool([_acct("ka"), _acct("kb"), _acct("kc")], strategy="fill_first")
    model = _FakeModel(api_key="ka")
    model.fail_keys = {"ka", "kb", "kc"}
    _attach_pool(model, pool)
    fb = _FakeFallback()
    fc = FallbackConfig(on_rate_limit=[fb])

    result = await acall_model_with_fallback(model, fc, messages=[])

    assert result.key == "fallback", result.key
    assert fb.called is True
    assert model.calls == ["ka", "kb", "kc"], model.calls
    assert all(a.last_status == "EXHAUSTED" for a in pool._accounts)


# ── 3. Inert by default (no pool attribute) ──────────────────────────
@test("credential_pool", "inert without a pool: one aresponse then existing fallback")
async def t_inert_no_pool(_ctx: TestContext) -> None:
    from src.models.providers.fallback import FallbackConfig, acall_model_with_fallback

    model = _FakeModel(api_key="ka")
    model.fail_keys = {"ka"}
    # No _attach_pool: model carries NO _openagent_cred_pool attribute.
    fb = _FakeFallback()
    fc = FallbackConfig(on_rate_limit=[fb])

    result = await acall_model_with_fallback(model, fc, messages=[])

    assert result.key == "fallback", result.key
    assert fb.called is True
    assert model.calls == ["ka"], "inert path must call the primary exactly once"
    assert not hasattr(model, "_openagent_cred_pool")


# ── 4. Strategy + cooldown + non-429 + auth units ────────────────────
@test("credential_pool", "fill_first sticks to entry[0] until it is exhausted")
async def t_fill_first_sticks(_ctx: TestContext) -> None:
    from src.models.credential_pool import CredentialPool

    pool = CredentialPool([_acct("ka"), _acct("kb"), _acct("kc")], strategy="fill_first")
    assert [pool.select().api_key for _ in range(3)] == ["ka", "ka", "ka"]
    pool.mark_exhausted_and_rotate(status_code=429, api_key_hint="ka")
    assert pool.select().api_key == "kb"


@test("credential_pool", "round_robin cycles across accounts")
async def t_round_robin_cycles(_ctx: TestContext) -> None:
    from src.models.credential_pool import CredentialPool

    pool = CredentialPool([_acct("ka"), _acct("kb"), _acct("kc")], strategy="round_robin")
    keys = [pool.select().api_key for _ in range(6)]
    assert keys == ["ka", "kb", "kc", "ka", "kb", "kc"], keys


@test("credential_pool", "least_used spreads load by request_count")
async def t_least_used_spreads(_ctx: TestContext) -> None:
    from src.models.credential_pool import CredentialPool

    pool = CredentialPool([_acct("ka"), _acct("kb"), _acct("kc")], strategy="least_used")
    keys = [pool.select().api_key for _ in range(4)]
    assert keys == ["ka", "kb", "kc", "ka"], keys


@test("credential_pool", "elapsed cooldown re-enters select()")
async def t_cooldown_reenters(_ctx: TestContext) -> None:
    from src.models.credential_pool import CredentialPool

    pool = CredentialPool([_acct("ka"), _acct("kb")], strategy="fill_first")
    pool.mark_exhausted_and_rotate(status_code=429, api_key_hint="ka")
    assert pool.select().api_key == "kb"  # ka still cooling
    # Force the cooldown to have elapsed.
    pool._accounts[0].last_error_reset_at = time.monotonic() - 1
    assert pool.has_available() is True
    assert pool.select().api_key == "ka"
    assert pool._accounts[0].last_status == "OK"


@test("credential_pool", "a non-429 error does NOT rotate")
async def t_non_ratelimit_no_rotate(_ctx: TestContext) -> None:
    from src.models.credential_pool import CredentialPool
    from src.models.providers.fallback import FallbackConfig, acall_model_with_fallback

    pool = CredentialPool([_acct("ka"), _acct("kb")], strategy="fill_first")
    model = _FakeModel(api_key="ka")
    model.error_map = {"ka": 400}  # non-retryable client error
    _attach_pool(model, pool)
    fb = _FakeFallback()
    fc = FallbackConfig(on_rate_limit=[fb])

    raised = None
    try:
        await acall_model_with_fallback(model, fc, messages=[])
    except ModelProviderError as e:
        raised = e
    assert raised is not None and raised.status_code == 400
    assert model.calls == ["ka"], "must not rotate on a non-429 error"
    assert all(a.last_status == "OK" for a in pool._accounts)
    assert fb.called is False


@test("credential_pool", "auth (401) marks DEAD and is never re-selected")
async def t_auth_dead_terminal(_ctx: TestContext) -> None:
    from src.models.credential_pool import CredentialPool

    pool = CredentialPool([_acct("ka"), _acct("kb")], strategy="fill_first")
    nxt = pool.mark_exhausted_and_rotate(status_code=401, api_key_hint="ka")
    assert nxt.api_key == "kb"
    assert pool._accounts[0].last_status == "DEAD"
    # DEAD is terminal: even a stale reset timestamp must not revive it.
    pool._accounts[0].last_error_reset_at = time.monotonic() - 10_000
    assert pool.select().api_key == "kb"
    pool.mark_exhausted_and_rotate(status_code=429, api_key_hint="kb")
    assert pool.select() is None  # ka DEAD, kb cooling


# ── 5. Config parse / inert gate / singleton ─────────────────────────
@test("credential_pool", "get_or_build_pool inert when accounts absent or <= 1")
async def t_config_parse_and_singleton(_ctx: TestContext) -> None:
    from src.models.credential_pool import _reset_registry_for_test, get_or_build_pool

    _reset_registry_for_test()
    assert get_or_build_pool("p_absent", {"base_url": "http://proxy/v1"}) is None
    assert get_or_build_pool("p_empty_meta", {"metadata": {}}) is None
    assert get_or_build_pool("p_empty_accts", {"metadata": {"accounts": []}}) is None
    assert get_or_build_pool("p_one", {"metadata": {"accounts": [{"api_key": "k1"}]}}) is None

    cfg = {
        "base_url": "http://proxy/v1",
        "metadata": {"accounts": [{"api_key": "k1"}, {"api_key": "k2", "base_url": "http://other/v1"}]},
    }
    pool = get_or_build_pool("p_two", cfg)
    assert pool is not None
    assert [a.api_key for a in pool._accounts] == ["k1", "k2"]
    # An account without base_url inherits the provider's; its own wins.
    assert pool._accounts[0].base_url == "http://proxy/v1"
    assert pool._accounts[1].base_url == "http://other/v1"
    # Singleton: same provider name returns the identical instance.
    assert get_or_build_pool("p_two", cfg) is pool


@test("credential_pool", "get_pool_strategy reads the top-level strategy map")
async def t_get_pool_strategy(_ctx: TestContext) -> None:
    from src.models.credential_pool import get_pool_strategy

    cfg = {"credential_pool_strategies": {"local": "round_robin", "openai": "least_used"}}
    assert get_pool_strategy("local", cfg) == "round_robin"
    assert get_pool_strategy("openai", cfg) == "least_used"
    assert get_pool_strategy("missing", cfg) == "fill_first"  # default
    assert get_pool_strategy("x", {"pool_strategy": "least_used"}) == "least_used"
    assert get_pool_strategy("x", {"pool_strategy": "nonsense"}) == "fill_first"  # validated
    assert get_pool_strategy("x", {}) == "fill_first"


# ── 6. Streaming: mid-stream 429 must NOT rotate ─────────────────────
@test("credential_pool", "streaming: mid-stream 429 does not rotate (would duplicate output)")
async def t_stream_midfail_no_rotate(_ctx: TestContext) -> None:
    from src.models.credential_pool import CredentialPool
    from src.models.providers.fallback import FallbackConfig, acall_model_stream_with_fallback

    pool = CredentialPool([_acct("ka"), _acct("kb")], strategy="fill_first")
    model = _FakeModel(api_key="ka")
    model.stream_midfail_keys = {"ka"}  # yields one event, then 429
    _attach_pool(model, pool)
    fb = _FakeFallback()
    fc = FallbackConfig(on_rate_limit=[fb])

    events = [e async for e in acall_model_stream_with_fallback(model, fc, messages=[])]
    contents = [getattr(e, "content", None) for e in events]

    assert "partial-ka" in contents, contents
    assert model.stream_calls == ["ka"], "must not re-stream on a mid-stream failure"
    assert pool._accounts[0].last_status == "OK", "account must not be marked mid-stream"
    assert fb.stream_called is True, "existing fallback path must fire instead"


@test("credential_pool", "streaming: pre-first-event 429 rotates to the next account")
async def t_stream_prefail_rotates(_ctx: TestContext) -> None:
    from src.models.credential_pool import CredentialPool
    from src.models.providers.fallback import FallbackConfig, acall_model_stream_with_fallback

    pool = CredentialPool([_acct("ka"), _acct("kb")], strategy="fill_first")
    model = _FakeModel(api_key="ka")
    model.stream_fail_keys = {"ka"}  # raises before the first event
    _attach_pool(model, pool)
    fb = _FakeFallback()
    fc = FallbackConfig(on_rate_limit=[fb])

    events = [e async for e in acall_model_stream_with_fallback(model, fc, messages=[])]
    contents = [getattr(e, "content", None) for e in events]

    assert contents == ["ok-kb"], contents
    assert model.stream_calls == ["ka", "kb"], model.stream_calls
    assert pool._accounts[0].last_status == "EXHAUSTED"
    assert pool._accounts[1].last_status == "OK"
    assert fb.stream_called is False, "rotation succeeded → DeepSeek untouched"
