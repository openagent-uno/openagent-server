"""HTTP 402 / Insufficient Balance gets a clear, actionable message.

When a workflow's model hits a billing failure (HTTP 402 — Insufficient
Balance from DeepSeek's chat API, or the same status from any other
provider that surfaces it), retrying won't help and the run becomes a
silent hammer on the SQLite writer lock. Operators should see a single
clear line that names the provider and points at billing, not the raw
``Error code: 402 - {...'message': 'Insufficient Balance'...}``
boilerplate.

Pins:
  - 'insufficient balance' (case-insensitive) substring is rewritten
  - explicit 'Error code: 402' substring is rewritten too (catches
    providers that don't use the same English string)
  - the rewritten message names the provider that failed
  - unrelated errors pass through unchanged
"""
from __future__ import annotations

from ._framework import TestContext, test


def _make_provider(model: str = "deepseek:deepseek-chat"):
    """Construct an NativeProvider without triggering provider init.

    NativeProvider.__init__ pulls in heavy Agno modules; tests just need
    the rewriter's bound methods, so we bypass __init__ and set the
    one field the rewriter reads (``self.model``)."""
    from src.models.native_provider import NativeProvider

    p = NativeProvider.__new__(NativeProvider)
    p.model = model
    p._providers_config = {}
    return p


@test("provider_402_rewrite", "deepseek 'Insufficient Balance' rewrites to billing hint")
async def t_deepseek_insufficient_balance(ctx: TestContext) -> None:
    p = _make_provider("deepseek:deepseek-chat")
    raw = (
        "Error code: 402 - {'error': {'message': 'Insufficient Balance', "
        "'type': 'unknown_error', 'param': None, 'code': 'invalid_request_error'}}"
    )
    rewritten = p._rewrite_provider_error_detail(raw)
    assert rewritten != raw, "402 message should have been rewritten"
    lowered = rewritten.lower()
    assert "deepseek" in lowered, f"provider name missing: {rewritten}"
    assert "402" in rewritten, f"http code missing for diagnostic clarity: {rewritten}"
    assert "billing" in lowered or "refill" in lowered, \
        f"actionable hint missing: {rewritten}"


@test("provider_402_rewrite", "openai 'Error code: 402' (no English text) still rewrites")
async def t_openai_402_no_english(ctx: TestContext) -> None:
    p = _make_provider("openai:gpt-4o-mini")
    raw = "Error code: 402 - {'error': {'message': 'paiement requis'}}"
    rewritten = p._rewrite_provider_error_detail(raw)
    assert rewritten != raw, "402 should rewrite even when message isn't English"
    assert "openai" in rewritten.lower()


@test("provider_402_rewrite", "unrelated 500 errors pass through untouched")
async def t_500_passthrough(ctx: TestContext) -> None:
    p = _make_provider("openai:gpt-4o-mini")
    raw = "Error code: 500 - upstream timeout"
    out = p._rewrite_provider_error_detail(raw)
    assert out == raw, f"500 should pass through unchanged, got: {out!r}"


@test("provider_402_rewrite", "the pre-existing deepseek image-url rewrite still wins")
async def t_image_url_rewrite_still_works(ctx: TestContext) -> None:
    """Regression guard: the new 402 branch must not shadow the
    existing image-url rewrite, since DeepSeek can return that one
    on a 200 response."""
    p = _make_provider("deepseek:deepseek-chat")
    raw = "unknown variant image_url in DeepSeek payload"
    out = p._rewrite_provider_error_detail(raw)
    assert out != raw, "image-url rewrite should still fire"
    assert "computer-control" in out.lower() or "text message content" in out.lower(), \
        f"expected image-url rewrite, got: {out!r}"
