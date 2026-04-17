"""Dynamic LLM catalog discovery — parsers, OpenRouter mapping, fallback.

Live HTTP calls are intentionally NOT exercised here; they'd be flaky
without network. We test the parser shapes and the bundled fallback
derived from default_pricing.json (no separate known_models file
anymore — that JSON was de-duplicated against the catalog).
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("models_discovery", "bundled fallback lists openai + anthropic")
async def t_bundled_fallback(ctx: TestContext) -> None:
    from openagent.models.discovery import _bundled_fallback

    openai_models = _bundled_fallback("openai")
    ids = {m["id"] for m in openai_models}
    assert "gpt-4o-mini" in ids, ids

    anthropic_models = _bundled_fallback("anthropic")
    assert any(m["id"].startswith("claude-sonnet-") for m in anthropic_models)


@test("models_discovery", "unknown provider returns []")
async def t_unknown_provider(ctx: TestContext) -> None:
    from openagent.models.discovery import _bundled_fallback

    assert _bundled_fallback("not-a-real-provider") == []


@test("models_discovery", "openai-style parser handles /v1/models envelope")
async def t_openai_parser_shape(ctx: TestContext) -> None:
    from openagent.models.discovery import _parse_openai_style

    payload = {"data": [
        {"id": "gpt-4o-mini", "name": "GPT-4o mini"},
        {"id": "gpt-4.1"},  # no name
        {"no-id": True},    # skipped
    ]}
    parsed = _parse_openai_style(payload)
    ids = [m["id"] for m in parsed]
    assert ids == ["gpt-4o-mini", "gpt-4.1"]


@test("models_discovery", "google parser strips models/ prefix")
async def t_google_parser(ctx: TestContext) -> None:
    from openagent.models.discovery import _parse_google

    payload = {"models": [
        {"name": "models/gemini-2.5-pro", "displayName": "Gemini 2.5 Pro"},
        {"name": "models/gemini-2.5-flash"},
    ]}
    parsed = _parse_google(payload)
    ids = [m["id"] for m in parsed]
    assert ids == ["gemini-2.5-pro", "gemini-2.5-flash"]


@test("models_discovery", "OpenRouter filter strips vendor prefix + converts pricing")
async def t_openrouter_filter(ctx: TestContext) -> None:
    from openagent.models.discovery import _openrouter_filter_for

    catalog = [
        {
            "id": "openai/gpt-4o-mini",
            "name": "OpenAI: GPT-4o mini",
            "pricing": {"prompt": "0.00000015", "completion": "0.00000060"},
        },
        {
            "id": "anthropic/claude-sonnet-4.5",
            "name": "Anthropic: Claude Sonnet 4.5",
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        },
        {"id": "malformed-no-slash", "name": "bad"},
        {"id": "other/something"},  # vendor not in our map — skipped
    ]
    oa = _openrouter_filter_for("openai", catalog)
    ids = [m["id"] for m in oa]
    assert ids == ["gpt-4o-mini"]
    # pricing per token → per million (0.00000015 * 1e6 ≈ 0.15)
    assert abs(oa[0]["input_cost_per_million"] - 0.15) < 1e-6
    assert abs(oa[0]["output_cost_per_million"] - 0.60) < 1e-6

    anth = _openrouter_filter_for("anthropic", catalog)
    assert [m["id"] for m in anth] == ["claude-sonnet-4.5"]
