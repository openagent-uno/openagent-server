"""Regression guards for the oversized single-tool-result cap.

A tool result is not paid once. It becomes a ``role="tool"`` message, and every
following step of the turn re-sends it — as does every following turn, in a
session bound to a long-lived thread. So one unbounded result is charged again
and again.

The cap existed before 2026-07-14 and did nothing, because it was applied to
``ToolExecution.result`` — the record the UI renders — and NOT to the message
handed back to the model, which ``Model.create_function_call_result`` builds
separately from the raw output. A 642 KB vault note therefore entered the
context uncapped on every single run. The load-bearing test in this file is the
last one: it asserts the cap reaches the MODEL's message, not just the display
record.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("tool_result_cap", "short string results pass through unchanged")
async def t_passthrough_small(_ctx: TestContext) -> None:
    from src.core.tool_output import cap_tool_output

    assert cap_tool_output("hello world") == "hello world"


@test("tool_result_cap", "unknown result shapes are returned as-is")
async def t_non_string_passthrough(_ctx: TestContext) -> None:
    from src.core.tool_output import cap_tool_output

    assert cap_tool_output({"a": 1}) == {"a": 1}
    assert cap_tool_output(None) is None


@test("tool_result_cap", "oversized string is truncated with a marker and bounded in size")
async def t_truncates_oversized(_ctx: TestContext) -> None:
    from src.core.tool_output import cap_tool_output, max_tool_result_chars

    limit = max_tool_result_chars()
    big = "A" * (limit * 3)
    out = cap_tool_output(big)
    assert "truncated by OpenAgent" in out
    # bounded: head+tail (== limit) plus the short marker
    assert len(out) <= limit + 500
    # head and tail of the original content are preserved
    assert out.startswith("A") and out.endswith("A")


@test("tool_result_cap", "content-block lists are capped per text block, structure intact")
async def t_caps_content_blocks(_ctx: TestContext) -> None:
    from src.core.tool_output import cap_tool_output, max_tool_result_chars

    limit = max_tool_result_chars()
    blocks = [
        {"type": "text", "text": "B" * (limit * 2)},
        {"type": "image", "data": "…"},  # never mangled
        "C" * (limit * 2),
    ]
    out = cap_tool_output(blocks)
    assert isinstance(out, list) and len(out) == 3
    assert "truncated by OpenAgent" in out[0]["text"]
    assert out[0]["type"] == "text"
    assert out[1] == {"type": "image", "data": "…"}  # untouched
    assert "truncated by OpenAgent" in out[2]


@test("tool_result_cap", "the cap reaches the MODEL's tool message, not just the display record")
async def t_cap_applied_to_model_message(_ctx: TestContext) -> None:
    # THE regression. Before the fix, this message carried the full 642 KB.
    from src.core.tool_output import max_tool_result_chars
    from src.models.providers.base import Model

    limit = max_tool_result_chars()

    class _StubModel(Model):
        """Concrete only so the abstract base can be instantiated — the method
        under test is a pure message builder and touches none of these."""

        def invoke(self, *a, **k): ...
        def ainvoke(self, *a, **k): ...
        def invoke_stream(self, *a, **k): ...
        def ainvoke_stream(self, *a, **k): ...
        def _parse_provider_response(self, *a, **k): ...
        def _parse_provider_response_delta(self, *a, **k): ...

    class _Fn:
        name = "vault_read_note"
        stop_after_tool_call = False

    class _FunctionCall:
        call_id = "call_1"
        function = _Fn()
        arguments = {"note": "esound/config/support-coverage"}
        error = None

    huge = "N" * (limit * 5)
    msg = _StubModel(id="stub").create_function_call_result(
        function_call=_FunctionCall(),
        success=True,
        output=huge,
    )
    assert isinstance(msg.content, str)
    assert len(msg.content) <= limit + 500, (
        f"tool message reaching the model is {len(msg.content)} chars — the cap "
        "is not applied where it bites"
    )
    assert "truncated by OpenAgent" in msg.content
