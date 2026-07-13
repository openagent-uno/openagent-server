from __future__ import annotations

from src.core._runner.utils.functions import get_function_call
from src.mcp._runtime.function import Function

from scripts.tests._framework import test


@test("function_arguments", "decodes_json_with_literal_newlines_in_string_values")
async def t_decodes_json_with_literal_newlines_in_string_values(ctx) -> None:
    fn = Function(name="tool_search_call_tool")
    arguments = (
        '{"server": "vault", "tool": "vault_write_note", "args": {'
        '"path": "ops/infra-weekly-2026-07-13.md", '
        '"content": "---\ntitle: Infrastructure Weekly Snapshot\\nstatus: active\nbody"}}'
    )

    call = get_function_call("tool_search_call_tool", arguments, functions={fn.name: fn})

    assert call is not None
    assert not call.error
    assert call.arguments is not None
    assert call.arguments["args"]["path"] == "ops/infra-weekly-2026-07-13.md"
    assert "Infrastructure Weekly Snapshot" in call.arguments["args"]["content"]
    assert "\nbody" in call.arguments["args"]["content"]


@test("function_arguments", "decodes_python_literal_with_literal_newlines_in_string_values")
async def t_decodes_python_literal_with_literal_newlines_in_string_values(ctx) -> None:
    fn = Function(name="shell_exec")
    arguments = "{'command': 'python3 << EOF\nprint(1)\nEOF'}"

    call = get_function_call("shell_exec", arguments, functions={fn.name: fn})

    assert call is not None
    assert not call.error
    assert call.arguments == {"command": "python3 << EOF\nprint(1)\nEOF"}
