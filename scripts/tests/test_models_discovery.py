"""Dynamic LLM catalog discovery — parsers + OpenRouter mapping.

Live HTTP calls are intentionally NOT exercised here; they'd be flaky
without network. We test the parser shapes and the OpenRouter prefix
filter. Since v0.10.5 there is no bundled offline fallback — if a
provider isn't reachable and OpenRouter doesn't know it, the picker
returns an empty list rather than stale hardcoded data.
"""
from __future__ import annotations

from ._framework import TestContext, test


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


@test("models_discovery", "openai-style parser tolerates alternate envelopes")
async def t_openai_parser_envelopes(ctx: TestContext) -> None:
    """z.ai and some self-hosts use ``models[]`` or bare arrays instead
    of OpenAI's canonical ``data[]``. Parser must handle all three."""
    from openagent.models.discovery import _parse_openai_style

    # z.ai / self-hosted shape
    ids = [m["id"] for m in _parse_openai_style({"models": [{"id": "glm-5.1"}]})]
    assert ids == ["glm-5.1"]

    # bare list shape
    ids = [m["id"] for m in _parse_openai_style([{"id": "solo"}, {"id": "another"}])]
    assert ids == ["solo", "another"]

    # malformed input → empty
    assert _parse_openai_style(None) == []
    assert _parse_openai_style("not a dict") == []


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
