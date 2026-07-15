"""A team member backed by a large MCP (BillingBear registers ~280 tools)
must not re-ship every tool name into the leader's system message on every
turn. ``get_members_system_message_content`` caps the inlined list to a sample
plus a count; the member keeps all its tools at runtime, and the leader routes
by role, so delegation is unaffected. Regression for the ~14k-char/turn bloat.
"""
from __future__ import annotations

import os
from typing import Any

from ._framework import TestContext, test

from src.core._runner.team import _messages as M


# ── _member_tools_cap: the tunable knob ─────────────────────────────

@test("member_tools_cap", "default cap is 40 when env unset")
async def t_default(_ctx: TestContext) -> None:
    os.environ.pop("OPENAGENT_MEMBER_TOOLS_CAP", None)
    assert M._member_tools_cap() == 40


@test("member_tools_cap", "env overrides the cap; invalid falls back to 40; negatives clamp to 0")
async def t_env(_ctx: TestContext) -> None:
    try:
        os.environ["OPENAGENT_MEMBER_TOOLS_CAP"] = "12"
        assert M._member_tools_cap() == 12
        os.environ["OPENAGENT_MEMBER_TOOLS_CAP"] = "0"
        assert M._member_tools_cap() == 0
        os.environ["OPENAGENT_MEMBER_TOOLS_CAP"] = "not-a-number"
        assert M._member_tools_cap() == 40
        os.environ["OPENAGENT_MEMBER_TOOLS_CAP"] = "-5"
        assert M._member_tools_cap() == 0
    finally:
        os.environ.pop("OPENAGENT_MEMBER_TOOLS_CAP", None)


# ── the listing itself caps a large member ──────────────────────────

class _FakeMember:
    def __init__(self, name: str, tools: list[str]) -> None:
        self.name = name
        self.id = name
        self.role = "billing specialist"
        self.description = None
        self.members = None
        self.tools = tools  # plain strings; _get_tool_names accepts them


def _render(monkeypatch_target: Any, member: Any, cap: str | None) -> str:
    """Call get_members_system_message_content with a single fake member,
    bypassing get_resolved_members (which needs a real run context)."""
    import src.core._runner.utils.callables as callables
    orig = callables.get_resolved_members
    callables.get_resolved_members = lambda team, run_context: [member]  # type: ignore
    if cap is None:
        os.environ.pop("OPENAGENT_MEMBER_TOOLS_CAP", None)
    else:
        os.environ["OPENAGENT_MEMBER_TOOLS_CAP"] = cap

    class _T:
        add_member_tools_to_context = True

    try:
        return M.get_members_system_message_content(_T(), indent=0, run_context=None)
    finally:
        callables.get_resolved_members = orig  # type: ignore
        os.environ.pop("OPENAGENT_MEMBER_TOOLS_CAP", None)


@test("member_tools_cap", "a 300-tool member is capped to a sample + (+N more), not 300 names")
async def t_caps_large_member(_ctx: TestContext) -> None:
    tools = [f"billingbear_tool_{i:03d}" for i in range(300)]
    out = _render(M, _FakeMember("billing", tools), cap="40")
    assert "billingbear_tool_000" in out           # the sample is present
    assert "billingbear_tool_299" not in out        # the tail is elided
    assert "(+260 more)" in out                     # 300 - 40
    assert out.count("billingbear_tool_") == 40      # exactly the cap listed
    assert len(out) < 3000                           # nowhere near the ~14k bloat


@test("member_tools_cap", "a small member is listed in full (unchanged behaviour)")
async def t_small_member_unchanged(_ctx: TestContext) -> None:
    tools = ["replio_respond", "replio_brief", "vault_write"]
    out = _render(M, _FakeMember("support", tools), cap="40")
    assert "Tools: replio_respond, replio_brief, vault_write" in out
    assert "more)" not in out


@test("member_tools_cap", "cap=0 omits the tool listing entirely")
async def t_cap_zero_omits(_ctx: TestContext) -> None:
    out = _render(M, _FakeMember("billing", ["billingbear_a", "billingbear_b"]), cap="0")
    assert "billingbear_a" not in out
    assert "Tools" not in out
