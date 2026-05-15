"""Workflow templating filters — happy paths + the safety net.

These tests cover the small set of custom Jinja2 filters wired up in
``openagent.workflow.templating._build_env``. Each filter exists to
satisfy a concrete workflow author use case, and silently regressing
any of them would break live workflows that rely on the contract.

The most load-bearing case is ``fromjson`` / ``from_json``: it's the
only way to feed a JSON-emitting ``ai-prompt`` block into a downstream
``loop.items_expr``, since the AI node returns its result as a raw
string in ``output.text`` and ``items_expr`` requires a native list.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("workflow_templating", "fromjson decodes a JSON string emitted by an ai-prompt upstream")
async def t_fromjson_decodes_json_string(ctx: TestContext) -> None:
    from src.workflow.templating import resolve_templates

    # Mirrors a realistic loop.items_expr where the upstream ai-prompt
    # block ``n24`` emitted a JSON object whose ``batches`` key is the
    # list we want to iterate.
    template_ctx = {
        "nodes": {
            "n24": {
                "output": {
                    "text": '{"total": 3, "batches": [["a", "b"], ["c"], ["d"]]}'
                }
            }
        }
    }
    out = resolve_templates(
        "{{ (nodes.n24.output.text | fromjson).batches }}", template_ctx
    )
    assert out == [["a", "b"], ["c"], ["d"]], f"unexpected: {out!r}"


@test("workflow_templating", "from_json alias matches fromjson")
async def t_from_json_alias(ctx: TestContext) -> None:
    from src.workflow.templating import resolve_templates

    ctx_data = {"nodes": {"n": {"output": {"text": '[1, 2, 3]'}}}}
    via_fromjson = resolve_templates("{{ nodes.n.output.text | fromjson }}", ctx_data)
    via_from_json = resolve_templates("{{ nodes.n.output.text | from_json }}", ctx_data)
    assert via_fromjson == [1, 2, 3]
    assert via_from_json == via_fromjson


@test("workflow_templating", "fromjson passes non-string values through unchanged")
async def t_fromjson_passthrough(ctx: TestContext) -> None:
    from src.workflow.templating import resolve_templates

    # Already-decoded list — applying the filter is a no-op, so callers
    # can chain it defensively without worrying about the upstream shape.
    ctx_data = {"nodes": {"n": {"output": {"items": [1, 2, 3]}}}}
    out = resolve_templates("{{ nodes.n.output.items | fromjson }}", ctx_data)
    assert out == [1, 2, 3]


@test("workflow_templating", "fromjson then index works inside a single placeholder")
async def t_fromjson_then_index(ctx: TestContext) -> None:
    from src.workflow.templating import resolve_templates

    ctx_data = {"nodes": {"n": {"output": {"text": '{"k": [10, 20]}'}}}}
    out = resolve_templates("{{ (nodes.n.output.text | fromjson)['k'][1] }}", ctx_data)
    assert out == 20


@test("workflow_templating", "json filter still round-trips with fromjson")
async def t_json_roundtrip(ctx: TestContext) -> None:
    from src.workflow.templating import resolve_templates

    # The pre-existing ``json`` filter serializes; ``fromjson`` reverses
    # it. Keeping the round-trip green guards against accidental
    # asymmetry if either filter is touched in the future.
    ctx_data = {"vars": {"payload": {"a": 1, "b": [2, 3]}}}
    encoded = resolve_templates("{{ vars.payload | json }}", ctx_data)
    assert isinstance(encoded, str)
    decoded_ctx = {"nodes": {"n": {"output": {"text": encoded}}}}
    decoded = resolve_templates("{{ nodes.n.output.text | fromjson }}", decoded_ctx)
    assert decoded == {"a": 1, "b": [2, 3]}
