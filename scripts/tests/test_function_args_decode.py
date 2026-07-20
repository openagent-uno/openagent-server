"""Tool-call argument decoding — pure-unit.

Models occasionally emit valid JSON arguments followed by extra data (a second
object, a stray token, trailing prose). ``json.loads`` raises "Extra data" and
the decoder fell through to failure, forcing a full retry round-trip (~13/10min
observed on esound-openagent). ``_decode_function_arguments`` now recovers the
leading object via ``raw_decode`` before giving up. Pure-unit (string in, dict
out), so it can run anywhere in the order.
"""
from __future__ import annotations

from src.core._runner.utils.functions import _decode_function_arguments

from ._framework import TestContext, test


@test("function-args", "clean JSON args decode unchanged")
async def test_clean_json_unchanged(ctx: TestContext) -> None:
    assert _decode_function_arguments('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


@test("function-args", "trailing second object is tolerated (Extra data)")
async def test_extra_data_second_object(ctx: TestContext) -> None:
    # The exact shape that produced "Extra data: line 2 column 1".
    assert _decode_function_arguments('{"a": 1}\n{"b": 2}') == {"a": 1}


@test("function-args", "trailing prose after args is tolerated")
async def test_trailing_prose(ctx: TestContext) -> None:
    assert _decode_function_arguments('{"q": "hi"} and then some words') == {"q": "hi"}


@test("function-args", "control chars in strings are still repaired")
async def test_control_chars_still_repaired(ctx: TestContext) -> None:
    assert _decode_function_arguments('{"text": "line1\nline2"}') == {"text": "line1\nline2"}


@test("function-args", "leading whitespace then trailing extra data")
async def test_leading_ws_then_extra_data(ctx: TestContext) -> None:
    assert _decode_function_arguments('  {"a": 1}  trailing') == {"a": 1}
