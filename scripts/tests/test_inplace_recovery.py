"""In-place deterministic recovery: at the model-error boundary, a repairable
request is fixed in place and retried on the SAME model *before* the turn
spills to credential rotation / another provider.

Each recovery branch repairs a 400 that OpenAgent otherwise raises straight up
(non-retryable client error → ``get_fallback_models`` returns None → raise):

  • image_too_large / payload_too_large → strip image parts, retry text-only
  • thinking_signature                  → strip signed thinking blocks, retry
  • tool_schema_pattern                 → strip pattern/format from tool schemas

LLM-free. A ``_FakeModel`` raises the matching ``ModelProviderError`` once, then
succeeds on the repaired request, so these pin the seam in ``fallback.py``
(async + stream) without a live provider. They also lock the two safety
contracts: an UNMATCHED error takes the exact existing path (byte-identical),
and a bug inside recovery degrades to that same path (never worse than today).
"""
from __future__ import annotations

import src.models.providers.fallback as fb_mod
from src.core.runtime_errors import ModelProviderError, ModelRateLimitError
from src.models.providers.fallback import (
    FallbackConfig,
    acall_model_stream_with_fallback,
    acall_model_with_fallback,
)
from src.models.providers.message import Message
from src.models.providers.response import ModelResponse

from ._framework import TestContext, test


# ── Fakes ────────────────────────────────────────────────────────────
class _Resp:
    """Minimal stand-in carrying a tag identifying which model served it."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


class _FakeModel:
    """Raises ``error`` on the FIRST call, then succeeds — so the repaired
    retry (the SECOND call) is the one that lands, and its kwargs are captured
    for assertions. ``fail_forever`` keeps raising to exercise degrade paths."""

    def __init__(self, *, error: Exception, fail_forever: bool = False, id: str = "local:claude") -> None:
        self.id = id
        self._error = error
        self._fail_forever = fail_forever
        self.calls: list[dict] = []          # kwargs per aresponse call
        self.stream_calls: list[dict] = []    # kwargs per aresponse_stream call

    async def aresponse(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail_forever or len(self.calls) == 1:
            raise self._error
        return _Resp("recovered")

    async def aresponse_stream(self, **kwargs):
        self.stream_calls.append(kwargs)
        if self._fail_forever or len(self.stream_calls) == 1:
            raise self._error
        yield ModelResponse(event="content", content="recovered")


class _FakeFallback:
    """A configured fallback model that records whether it was consulted."""

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


class _NoFallbackGuard:
    """Swap ``get_fallback_models`` for a tripwire that raises if consulted —
    proves a matched recovery short-circuits BEFORE the existing fallback path."""

    def __enter__(self):
        self._orig = fb_mod.get_fallback_models

        def _boom(*_a, **_k):
            raise AssertionError("get_fallback_models must NOT be consulted after in-place recovery")

        fb_mod.get_fallback_models = _boom  # type: ignore[assignment]
        return self

    def __exit__(self, *exc):
        fb_mod.get_fallback_models = self._orig  # type: ignore[assignment]
        return False


def _schema_has_key(node, key) -> bool:
    if isinstance(node, dict):
        return key in node or any(_schema_has_key(v, key) for v in node.values())
    if isinstance(node, list):
        return any(_schema_has_key(i, key) for i in node)
    return False


# ── 1. image_too_large: strip images, retry same model text-only ─────
@test("inplace_recovery", "image-too-large strips image parts and retries the SAME model")
async def t_image_recovery(_ctx: TestContext) -> None:
    err = ModelProviderError("messages.0: image exceeds 5 MB maximum", status_code=400)
    model = _FakeModel(error=err)
    msg = Message.model_construct(
        role="user",
        content=[{"type": "text", "text": "look"}, {"type": "image", "source": {"data": "BIG"}}],
        images=["fake-image"],
    )
    fb = _FakeFallback()
    fc = FallbackConfig(on_error=[fb])

    with _NoFallbackGuard():
        result = await acall_model_with_fallback(model, fc, messages=[msg])

    assert result.tag == "recovered", result.tag
    assert len(model.calls) == 2, model.calls  # original + one repaired retry
    retried = model.calls[1]["messages"][0]
    assert retried.images is None, retried.images
    assert all(p.get("type") != "image" for p in retried.content), retried.content
    assert fb.called is False
    # Original message object untouched (recovery works on a copy).
    assert msg.images == ["fake-image"] and len(msg.content) == 2


# ── 2. thinking_signature: strip signed thinking blocks, retry ───────
@test("inplace_recovery", "thinking-signature strips reasoning + signature and retries")
async def t_thinking_recovery(_ctx: TestContext) -> None:
    err = ModelProviderError(
        "input.messages.1: thinking block signature is invalid", status_code=400
    )
    model = _FakeModel(error=err)
    msg = Message(
        role="assistant",
        content="answer",
        reasoning_content="secret chain of thought",
        provider_data={"signature": "sig-abc", "container": {"id": "c1"}},
    )

    with _NoFallbackGuard():
        result = await acall_model_with_fallback(model, None, messages=[msg])

    assert result.tag == "recovered"
    retried = model.calls[1]["messages"][0]
    assert retried.reasoning_content is None
    assert "signature" not in (retried.provider_data or {})
    assert retried.provider_data == {"container": {"id": "c1"}}  # other keys preserved
    # Original untouched.
    assert msg.reasoning_content == "secret chain of thought"
    assert msg.provider_data.get("signature") == "sig-abc"


# ── 3. tool_schema_pattern: strip pattern/format from tool schemas ───
@test("inplace_recovery", "tool-schema grammar error strips pattern/format and retries")
async def t_tool_schema_recovery(_ctx: TestContext) -> None:
    err = ModelProviderError("error parsing grammar: json-schema-to-grammar failed", status_code=400)
    model = _FakeModel(error=err)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"d": {"type": "string", "pattern": r"\d{4}", "format": "date"}},
                },
            },
        }
    ]

    with _NoFallbackGuard():
        result = await acall_model_with_fallback(model, None, messages=[], tools=tools)

    assert result.tag == "recovered"
    retried_tools = model.calls[1]["tools"]
    assert not _schema_has_key(retried_tools, "pattern"), retried_tools
    assert not _schema_has_key(retried_tools, "format"), retried_tools
    # Original tools untouched (recovery deep-copies).
    assert _schema_has_key(tools, "pattern") and _schema_has_key(tools, "format")


# ── 4. Byte-identical: an UNMATCHED 400 takes the exact existing path ─
@test("inplace_recovery", "unmatched 400 falls through byte-identically (raises, no extra call)")
async def t_unmatched_400_byte_identical(_ctx: TestContext) -> None:
    err = ModelProviderError("bad request: unknown field 'foo'", status_code=400)
    model = _FakeModel(error=err, fail_forever=True)
    fb = _FakeFallback()
    fc = FallbackConfig(on_error=[fb])

    raised = None
    try:
        await acall_model_with_fallback(model, fc, messages=[Message(role="user", content="hi")])
    except ModelProviderError as e:
        raised = e

    assert raised is err, raised  # same object, same raise as today
    assert len(model.calls) == 1, "no in-place retry for an unmatched error"
    assert fb.called is False  # on_error never masks a non-retryable 400


# ── 5. Byte-identical: rate-limit path is untouched (no recovery match) ─
@test("inplace_recovery", "rate-limit (429) is untouched: existing on_rate_limit fallback fires")
async def t_ratelimit_untouched(_ctx: TestContext) -> None:
    err = ModelRateLimitError("rate limited", status_code=429)
    model = _FakeModel(error=err, fail_forever=True)
    fb = _FakeFallback()
    fc = FallbackConfig(on_rate_limit=[fb])

    result = await acall_model_with_fallback(model, fc, messages=[])

    assert result.tag == "fallback"
    assert len(model.calls) == 1, "429 must not trigger an in-place retry"
    assert fb.called is True


# ── 6. Degrade: repaired retry still fails → existing path (raise) ───
@test("inplace_recovery", "matched but repaired retry still fails → existing fallback path")
async def t_repaired_retry_fails_degrades(_ctx: TestContext) -> None:
    err = ModelProviderError("image exceeds 5 MB maximum", status_code=400)
    model = _FakeModel(error=err, fail_forever=True)  # repaired retry fails too
    msg = Message.model_construct(role="user", content=[{"type": "image", "source": {"data": "X"}}])

    raised = None
    try:
        await acall_model_with_fallback(model, None, messages=[msg])
    except ModelProviderError as e:
        raised = e

    assert raised is err, raised
    assert len(model.calls) == 2, "one repaired retry, then existing path (raise)"


# ── 7. Degrade: a BUG inside recovery is swallowed → existing path ───
@test("inplace_recovery", "a bug in recovery planning degrades to the existing path")
async def t_recovery_bug_degrades(_ctx: TestContext) -> None:
    err = ModelProviderError("image exceeds 5 MB maximum", status_code=400)
    model = _FakeModel(error=err)
    orig = fb_mod._plan_inplace_recovery

    def _boom(*_a, **_k):
        raise RuntimeError("recovery is broken")

    fb_mod._plan_inplace_recovery = _boom  # type: ignore[assignment]
    try:
        raised = None
        try:
            await acall_model_with_fallback(model, None, messages=[Message(role="user", content="hi")])
        except ModelProviderError as e:
            raised = e
    finally:
        fb_mod._plan_inplace_recovery = orig  # type: ignore[assignment]

    assert raised is err, raised  # outer defensive wrapper swallowed the bug
    assert len(model.calls) == 1, "the bug must not add a retry"


# ── 8. Kill-switch: disabling the feature restores today's behaviour ─
@test("inplace_recovery", "kill-switch OFF disables recovery (byte-identical raise)")
async def t_kill_switch_off(_ctx: TestContext) -> None:
    import os

    err = ModelProviderError("image exceeds 5 MB maximum", status_code=400)
    model = _FakeModel(error=err, fail_forever=True)
    prev = os.environ.get("OPENAGENT_INPLACE_RECOVERY_ENABLED")
    os.environ["OPENAGENT_INPLACE_RECOVERY_ENABLED"] = "0"
    try:
        raised = None
        try:
            await acall_model_with_fallback(
                model, None, messages=[Message.model_construct(role="user", content=[{"type": "image"}])]
            )
        except ModelProviderError as e:
            raised = e
    finally:
        if prev is None:
            os.environ.pop("OPENAGENT_INPLACE_RECOVERY_ENABLED", None)
        else:
            os.environ["OPENAGENT_INPLACE_RECOVERY_ENABLED"] = prev

    assert raised is err
    assert len(model.calls) == 1, "disabled → no repaired retry"


# ── 9. Streaming: pre-first-event match repairs and streams cleanly ──
@test("inplace_recovery", "streaming: repairs a pre-first-event failure on the same model")
async def t_stream_recovery(_ctx: TestContext) -> None:
    err = ModelProviderError("image exceeds 5 MB maximum", status_code=400)
    model = _FakeModel(error=err)
    msg = Message.model_construct(
        role="user", content=[{"type": "text", "text": "hi"}, {"type": "image", "source": {"data": "BIG"}}]
    )
    fb = _FakeFallback()
    fc = FallbackConfig(on_error=[fb])

    with _NoFallbackGuard():
        events = [e async for e in acall_model_stream_with_fallback(model, fc, messages=[msg])]

    contents = [getattr(e, "content", None) for e in events]
    assert contents == ["recovered"], contents
    assert len(model.stream_calls) == 2, model.stream_calls
    retried = model.stream_calls[1]["messages"][0]
    assert all(p.get("type") != "image" for p in retried.content), retried.content
    assert fb.stream_called is False


# ── 10. Streaming byte-identical: unmatched 429 → existing fallback ──
@test("inplace_recovery", "streaming: unmatched 429 takes the existing fallback path")
async def t_stream_unmatched_byte_identical(_ctx: TestContext) -> None:
    err = ModelRateLimitError("rate limited", status_code=429)
    model = _FakeModel(error=err, fail_forever=True)
    fb = _FakeFallback()
    fc = FallbackConfig(on_rate_limit=[fb])

    events = [e async for e in acall_model_stream_with_fallback(model, fc, messages=[])]
    contents = [getattr(e, "content", None) for e in events]

    assert "fallback" in contents, contents
    assert len(model.stream_calls) == 1, "429 stream must not trigger an in-place retry"
    assert fb.stream_called is True


# ── 11. Streaming degrade: repaired stream also fails → existing path ─
@test("inplace_recovery", "streaming: repaired stream fails → buffer discarded, existing path")
async def t_stream_repaired_fails_degrades(_ctx: TestContext) -> None:
    err = ModelProviderError("image exceeds 5 MB maximum", status_code=400)
    model = _FakeModel(error=err, fail_forever=True)  # repaired stream fails too
    msg = Message.model_construct(role="user", content=[{"type": "image", "source": {"data": "X"}}])

    raised = None
    events: list = []
    try:
        async for e in acall_model_stream_with_fallback(model, None, messages=[msg]):
            events.append(e)
    except ModelProviderError as e:
        raised = e

    assert raised is err, raised
    assert events == [], "no committed output from the discarded repaired attempt"
    assert len(model.stream_calls) == 2, "one buffered repaired attempt, then existing path"
