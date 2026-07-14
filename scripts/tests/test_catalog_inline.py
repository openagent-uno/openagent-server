"""Catalog tool-key inlining — the anti-hallucination surface.

The MCP catalog summary injected into every turn inlines the EXACT registered
tool keys for a small allowlist of built-in high-traffic servers so the model
copies a key verbatim instead of guessing (and mis-prefixing) it. A deployment
can opt its OWN high-traffic MCP (an org/domain MCP) into full inlining via the
``OPENAGENT_INLINE_TOOL_KEYS_SERVERS`` env var, without OpenAgent hardcoding the
tenant's server name. These tests pin both behaviours.
"""
from __future__ import annotations

import os

from ._framework import TestContext, test


def _render(summary, descriptions, tool_names):
    from src.core.prompts import _render_catalog_summary_lines
    return _render_catalog_summary_lines(summary, descriptions, tool_names)


@test("catalog_inline", "builtin allowlist server inlines its keys (capped)")
async def t_builtin_inlined(ctx: TestContext) -> None:
    from src.core.prompts import _INLINE_TOOL_KEYS_CAP

    keys = [f"vault_tool_{i:02d}" for i in range(_INLINE_TOOL_KEYS_CAP + 5)]
    out = _render({"vault": len(keys)}, {}, {"vault": keys})
    assert "tools: vault_tool_00" in out, out
    # Beyond the cap the model is told to fall back to list_tools.
    assert f"(+5 more — use list_tools)" in out, out


@test("catalog_inline", "non-allowlisted MCP is NOT inlined by default")
async def t_thirdparty_not_inlined(ctx: TestContext) -> None:
    prev = os.environ.pop("OPENAGENT_INLINE_TOOL_KEYS_SERVERS", None)
    try:
        keys = ["replio_threads_respond", "replio_docs_search"]
        out = _render({"replio": len(keys)}, {"replio": "org support MCP"}, {"replio": keys})
        assert "``replio``" in out, out               # server line present
        assert "tools:" not in out, out               # but no key list
        assert "replio_docs_search" not in out, out
    finally:
        if prev is not None:
            os.environ["OPENAGENT_INLINE_TOOL_KEYS_SERVERS"] = prev


@test("catalog_inline", "operator-opted MCP inlines ALL keys (no cap, no guessing)")
async def t_operator_optin_full(ctx: TestContext) -> None:
    from src.core.prompts import _INLINE_TOOL_KEYS_CAP

    # 37 keys — more than the builtin cap; the whole point is none is dropped.
    keys = sorted(f"replio_tool_{i:02d}" for i in range(37))
    prev = os.environ.get("OPENAGENT_INLINE_TOOL_KEYS_SERVERS")
    os.environ["OPENAGENT_INLINE_TOOL_KEYS_SERVERS"] = "replio, billingbear"
    try:
        assert len(keys) > _INLINE_TOOL_KEYS_CAP  # guards the "no cap" claim
        out = _render({"replio": len(keys)}, {"replio": "org support MCP"}, {"replio": keys})
        assert "tools:" in out, out
        for k in keys:                                   # every key surfaced
            assert k in out, f"missing {k}\n{out}"
        assert "more — use list_tools" not in out, out   # no truncation suffix
    finally:
        if prev is None:
            os.environ.pop("OPENAGENT_INLINE_TOOL_KEYS_SERVERS", None)
        else:
            os.environ["OPENAGENT_INLINE_TOOL_KEYS_SERVERS"] = prev


@test("catalog_inline", "env parsing tolerates whitespace / empty / trailing commas")
async def t_env_parsing(ctx: TestContext) -> None:
    from src.core.prompts import _operator_inline_servers

    prev = os.environ.get("OPENAGENT_INLINE_TOOL_KEYS_SERVERS")
    try:
        os.environ["OPENAGENT_INLINE_TOOL_KEYS_SERVERS"] = "  replio , ,billingbear,, clickup "
        assert _operator_inline_servers() == frozenset({"replio", "billingbear", "clickup"})
        os.environ["OPENAGENT_INLINE_TOOL_KEYS_SERVERS"] = ""
        assert _operator_inline_servers() == frozenset()
        os.environ.pop("OPENAGENT_INLINE_TOOL_KEYS_SERVERS", None)
        assert _operator_inline_servers() == frozenset()
    finally:
        if prev is None:
            os.environ.pop("OPENAGENT_INLINE_TOOL_KEYS_SERVERS", None)
        else:
            os.environ["OPENAGENT_INLINE_TOOL_KEYS_SERVERS"] = prev
