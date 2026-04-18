"""Catalog + default pricing tests.

Covers ``openagent.models.catalog`` — runtime-id parsing, the
default-pricing JSON fallback, and user overrides.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("catalog", "split_runtime_id + model_id_from_runtime")
async def t_catalog_split(ctx: TestContext) -> None:
    from openagent.models.catalog import split_runtime_id, model_id_from_runtime
    assert split_runtime_id("openai:gpt-4o-mini") == ("openai", "gpt-4o-mini")
    assert split_runtime_id("claude-cli/claude-sonnet-4-6") == ("claude-cli", "claude-sonnet-4-6")
    assert split_runtime_id("just-a-name") == ("just-a-name", "just-a-name")
    assert model_id_from_runtime("openai:gpt-4o-mini") == "gpt-4o-mini"


@test("catalog", "default pricing fallback works for bare model entries")
async def t_pricing_fallback(ctx: TestContext) -> None:
    from openagent.models.catalog import get_model_pricing, compute_cost
    user_cfg = {"openai": {"models": ["gpt-4o-mini", "gpt-4.1"]}}
    p = get_model_pricing("gpt-4o-mini", user_cfg)
    assert p["input_cost_per_million"] == 0.15, f"unexpected: {p}"
    assert p["output_cost_per_million"] == 0.60
    cost = compute_cost("openai:gpt-4.1", 1000, 500, user_cfg)
    expected = (2.00 * 1000 + 8.00 * 500) / 1_000_000
    assert abs(cost - expected) < 1e-12


@test("catalog", "user pricing overrides defaults")
async def t_pricing_override(ctx: TestContext) -> None:
    from openagent.models.catalog import get_model_pricing
    cfg = {"openai": {"models": [
        {"id": "gpt-4o-mini", "input_cost_per_million": 99.0, "output_cost_per_million": 88.0}
    ]}}
    p = get_model_pricing("gpt-4o-mini", cfg)
    assert p["input_cost_per_million"] == 99.0


@test("catalog", "claude-cli models have zero pricing (subscription billing)")
async def t_claude_cli_zero_pricing(ctx: TestContext) -> None:
    """claude-cli dispatches via Claude Pro/Max; there is no per-token billing."""
    from openagent.models.catalog import get_model_pricing, compute_cost

    for ref in [
        "claude-cli:anthropic:claude-sonnet-4-6",
        "claude-cli/claude-sonnet-4-6",
        "claude-cli",
    ]:
        p = get_model_pricing(ref)
        assert p["input_cost_per_million"] == 0.0, f"{ref} leaked pricing: {p}"
        assert p["output_cost_per_million"] == 0.0, f"{ref} leaked pricing: {p}"

    # Even with a config entry that sets anthropic pricing, claude-cli
    # must not inherit — it's a different framework with different billing.
    cfg = {"anthropic": {"models": [
        {"id": "claude-sonnet-4-6", "input_cost_per_million": 3.0, "output_cost_per_million": 15.0}
    ]}}
    assert compute_cost("claude-cli:anthropic:claude-sonnet-4-6", 10_000, 5_000, cfg) == 0.0


@test("catalog", "OpenRouter cache primes pricing lookup")
async def t_openrouter_cache_pricing(ctx: TestContext) -> None:
    """After discovery fetches OpenRouter's catalog, cost lookups consult it."""
    import time
    from openagent.models import discovery
    from openagent.models.catalog import get_model_pricing

    prev = discovery._OPENROUTER_CACHE
    try:
        # Seed the cache with a single OpenRouter-shaped row. 0.000003 $/token
        # on the wire becomes 3.0 $/M after the *1e6 in catalog's lookup.
        discovery._OPENROUTER_CACHE = (time.time(), [
            {"id": "openai/gpt-synthetic", "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
        ])
        p = get_model_pricing("openai:gpt-synthetic")
        assert p["input_cost_per_million"] == 3.0, p
        assert p["output_cost_per_million"] == 15.0, p
    finally:
        discovery._OPENROUTER_CACHE = prev
