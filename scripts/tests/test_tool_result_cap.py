"""Regression guards for the oversized single-tool-result cap.

A single unbounded tool result (e.g. an MCP call returning a whole email
thread with quoted history, measured ~1.7 MB) can exceed the model context
window on its own; in-session compaction folds whole turns, not one giant
message, so the run would die with a non-retryable context-length error.
`_cap_tool_result` truncates such a result (head + tail + marker) at both
tool-execution choke points (`run_tool`/`arun_tool`, used by agents and the
team lead alike).
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("tool_result_cap", "short string results pass through unchanged")
async def t_passthrough_small(_ctx: TestContext) -> None:
    from src.core._runner.agent._tools import _cap_tool_result

    assert _cap_tool_result("hello world") == "hello world"


@test("tool_result_cap", "non-string results are returned as-is")
async def t_non_string_passthrough(_ctx: TestContext) -> None:
    from src.core._runner.agent._tools import _cap_tool_result

    assert _cap_tool_result({"a": 1}) == {"a": 1}
    assert _cap_tool_result(None) is None


@test("tool_result_cap", "oversized string is truncated with a marker and bounded in size")
async def t_truncates_oversized(_ctx: TestContext) -> None:
    from src.core._runner.agent._tools import _MAX_TOOL_RESULT_CHARS, _cap_tool_result

    big = "A" * (_MAX_TOOL_RESULT_CHARS * 3)
    out = _cap_tool_result(big)
    assert "truncated by OpenAgent" in out
    # bounded: head+tail (== limit) plus the short marker
    assert len(out) <= _MAX_TOOL_RESULT_CHARS + 500
    # head and tail of the original content are preserved
    assert out.startswith("A") and out.endswith("A")
