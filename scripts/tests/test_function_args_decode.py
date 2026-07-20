"""Regression: tolerate trailing content after valid tool-call JSON args.

Models occasionally emit valid JSON arguments followed by extra data (a second
object, a stray token, trailing prose). `json.loads` raises "Extra data" and the
old decoder fell through to failure -> the runner forced a full retry round-trip.
`_decode_function_arguments` now recovers the leading object via raw_decode.
"""
from src.core._runner.utils.functions import _decode_function_arguments


def test_clean_json_unchanged():
    assert _decode_function_arguments('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_extra_data_second_object():
    # This is the exact shape that produced "Extra data: line 2 column 1".
    assert _decode_function_arguments('{"a": 1}\n{"b": 2}') == {"a": 1}


def test_trailing_prose():
    assert _decode_function_arguments('{"q": "hi"} and then some words') == {"q": "hi"}


def test_control_chars_in_string_still_repaired():
    # The pre-existing repair path must keep working.
    out = _decode_function_arguments('{"text": "line1\nline2"}')
    assert out == {"text": "line1\nline2"}


def test_leading_whitespace_then_extra_data():
    assert _decode_function_arguments('  {"a": 1}  trailing') == {"a": 1}
