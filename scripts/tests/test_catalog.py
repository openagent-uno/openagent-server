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


@test("catalog", "pricing returns zero + 'missing' when no source has the model")
async def t_pricing_missing(ctx: TestContext) -> None:
    """With no bundled table and a cold OpenRouter cache, an unknown
    model resolves to zero cost logged as 'missing' — not a crash."""
    import time
    from openagent.models import discovery
    from openagent.models.catalog import get_model_pricing, compute_cost

    prev = discovery._OPENROUTER_CACHE
    try:
        discovery._OPENROUTER_CACHE = (time.time(), [])  # empty catalog
        p = get_model_pricing("openai:gpt-unknown-model")
        assert p == {"input_cost_per_million": 0.0, "output_cost_per_million": 0.0}
        assert compute_cost("openai:gpt-unknown-model", 10_000, 5_000) == 0.0
    finally:
        discovery._OPENROUTER_CACHE = prev


@test("catalog", "live OpenRouter pricing wins over any stale config metadata")
async def t_pricing_live(ctx: TestContext) -> None:
    """User config metadata is no longer consulted for pricing. The
    OpenRouter cache is the only source for non-claude-cli models."""
    import time
    from openagent.models import discovery
    from openagent.models.catalog import get_model_pricing

    prev = discovery._OPENROUTER_CACHE
    try:
        discovery._OPENROUTER_CACHE = (time.time(), [
            {"id": "openai/gpt-4o-mini",
             "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
        ])
        # Even if the user config still carries the old static cost
        # columns, the live cache value is what compute_cost uses.
        cfg = {"openai": {"models": [
            {"id": "gpt-4o-mini", "input_cost_per_million": 99.0, "output_cost_per_million": 88.0}
        ]}}
        p = get_model_pricing("openai:gpt-4o-mini", cfg)
        assert p["input_cost_per_million"] == 1.0, p
        assert p["output_cost_per_million"] == 2.0, p
    finally:
        discovery._OPENROUTER_CACHE = prev


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

    # claude-cli short-circuits even when the corresponding agno model
    # has live pricing in OpenRouter — different framework, different
    # billing surface.
    assert compute_cost("claude-cli:anthropic:claude-sonnet-4-6", 10_000, 5_000) == 0.0


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
