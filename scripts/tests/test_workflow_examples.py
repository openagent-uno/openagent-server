"""The canonical workflow examples in ``openagent/workflow/examples.py``
exist so the AI can anchor its ``create_workflow`` calls on a known-good
shape. They're only useful if every one of them survives structural
validation — otherwise we're shipping a broken reference manual.

This module is the contract enforcer: every example must
- be discoverable by name from ``WORKFLOW_EXAMPLES``,
- carry a non-empty description and at least one pattern tag,
- pass ``validate_graph`` cleanly (no inventory required — that's a
  pool-runtime concern).
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("workflow_examples", "every example passes structural validate_graph")
async def t_examples_validate(ctx: TestContext) -> None:
    from openagent.workflow.examples import WORKFLOW_EXAMPLES
    from openagent.workflow.validate import ValidationError, validate_graph

    assert WORKFLOW_EXAMPLES, "WORKFLOW_EXAMPLES is empty"

    failures: list[str] = []
    for name, ex in WORKFLOW_EXAMPLES.items():
        try:
            validate_graph(ex.graph)
        except ValidationError as exc:
            failures.append(f"{name}: {exc}")
    assert not failures, "examples failed validate_graph:\n  " + "\n  ".join(failures)


@test("workflow_examples", "every example has description + patterns")
async def t_examples_metadata(ctx: TestContext) -> None:
    from openagent.workflow.examples import WORKFLOW_EXAMPLES

    bad: list[str] = []
    for name, ex in WORKFLOW_EXAMPLES.items():
        if not ex.description.strip():
            bad.append(f"{name}: empty description")
        if not ex.patterns:
            bad.append(f"{name}: no patterns listed")
        if not ex.graph.get("nodes"):
            bad.append(f"{name}: graph has no nodes")
    assert not bad, "metadata problems:\n  " + "\n  ".join(bad)


@test("workflow_examples", "list_workflow_examples returns the right shape")
async def t_list_shape(ctx: TestContext) -> None:
    from openagent.workflow.examples import (
        WORKFLOW_EXAMPLES, list_workflow_examples,
    )

    listing = list_workflow_examples()
    assert len(listing) == len(WORKFLOW_EXAMPLES), listing
    keys_seen = set()
    for entry in listing:
        for k in ("name", "description", "patterns"):
            assert k in entry, f"missing {k} in {entry}"
        assert "graph" not in entry, "list view must omit graph (token cost)"
        keys_seen.add(entry["name"])
    assert keys_seen == set(WORKFLOW_EXAMPLES), keys_seen


@test("workflow_examples", "get_workflow_example returns a complete graph")
async def t_get_returns_graph(ctx: TestContext) -> None:
    from openagent.workflow.examples import (
        WORKFLOW_EXAMPLES, get_workflow_example,
    )

    sample = next(iter(WORKFLOW_EXAMPLES))
    full = get_workflow_example(sample)
    assert "graph" in full and full["graph"].get("nodes"), full
    assert full["name"] == sample
    # Unknown name must error with a useful suggestion.
    raised = False
    try:
        get_workflow_example("definitely-not-a-real-example")
    except KeyError as exc:
        raised = True
        assert "Known examples" in str(exc), str(exc)
    assert raised, "get_workflow_example should raise on unknown name"


@test("workflow_examples", "examples cover the LLM-pitfall block types")
async def t_examples_cover_pitfalls(ctx: TestContext) -> None:
    """The whole point of the examples is to teach the LLM the block
    types it most often gets wrong. If we ever lose coverage of one of
    these, the reference manual stops earning its keep."""
    from openagent.workflow.examples import WORKFLOW_EXAMPLES

    block_types_seen: set[str] = set()
    for ex in WORKFLOW_EXAMPLES.values():
        for node in ex.graph.get("nodes", []):
            block_types_seen.add(node.get("type"))

    must_cover = {"if", "loop", "parallel", "merge", "ai-prompt", "mcp-tool"}
    missing = must_cover - block_types_seen
    assert not missing, (
        f"examples don't cover these pitfall block types: {sorted(missing)}"
    )
