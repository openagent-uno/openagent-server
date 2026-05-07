"""Regression guards for provider-specific Agno toolkit filtering."""
from __future__ import annotations

import json

from ._framework import TestContext, test


class _FakeToolkit:
    def __init__(self, family: str, tool_count: int = 1) -> None:
        self.tool_name_prefix = family
        self.functions = {
            f"{family}_{idx}": object() for idx in range(tool_count)
        }


@test("agno_tool_filter", "deepseek filters incompatible computer_control toolkit families")
async def t_deepseek_filters_computer_control(_ctx: TestContext) -> None:
    from openagent.models.agno_provider import AgnoProvider

    provider = AgnoProvider(model="deepseek:deepseek-v4-flash")
    provider.set_mcp_toolkits([
        _FakeToolkit("computer_control", tool_count=3),
        _FakeToolkit("files", tool_count=2),
        _FakeToolkit("browser", tool_count=1),
    ])

    compatible, filtered = provider._compatible_mcp_toolkits()
    assert filtered == ["computer_control"], filtered
    assert [tk.tool_name_prefix for tk in compatible] == ["files", "browser"]

    families = provider._tool_families()
    assert sorted(families.keys()) == ["browser", "files"], sorted(families.keys())

    inventory = json.loads(provider._build_list_mcps_tool()())
    assert [row["server"] for row in inventory] == ["files", "browser"], inventory


@test("agno_tool_filter", "deepseek image_url provider error is rewritten into an actionable message")
async def t_deepseek_rewrites_image_url_error(_ctx: TestContext) -> None:
    from openagent.models.agno_provider import AgnoProvider

    provider = AgnoProvider(model="deepseek:deepseek-v4-flash")
    rewritten = provider._rewrite_provider_error_detail(
        "Failed to deserialize the JSON body into the target type: "
        "messages[101]: unknown variant image_url, expected text",
    )
    lowered = rewritten.lower()
    assert "deepseek" in lowered, rewritten
    assert "text" in lowered, rewritten
    assert "computer-control" in lowered or "gui" in lowered, rewritten
