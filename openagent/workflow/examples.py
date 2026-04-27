"""Canonical workflow examples — the AI's "show me how" reference.

Each example pairs a short docstring (what the workflow does + which
patterns it demonstrates) with a complete, validation-passing
``graph_json``. The ``workflow-manager`` MCP exposes these via the
``get_workflow_examples`` / ``get_workflow_example`` tools so the AI
can anchor its own ``create_workflow`` calls on a known-good shape
instead of reinventing the schema each time.

The examples deliberately reference real builtin MCP tools
(``messaging_telegram_send_message``, ``shell_exec``, etc.) so they
stay copy-pasteable. They pass structural ``validate_graph`` regardless
of which MCPs are loaded — pool-level callability is checked at
run-time by the executor with a clear error.

Add new examples by appending to ``WORKFLOW_EXAMPLES``. The contract
the test in ``scripts/tests/test_workflow_examples.py`` enforces:

- Every example must round-trip through ``validate_graph`` cleanly.
- Every example must have a non-empty docstring + patterns list so the
  AI can pick one by intent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkflowExample:
    name: str
    description: str
    patterns: tuple[str, ...]
    graph: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "patterns": list(self.patterns),
            "graph": self.graph,
        }


# ── Example 1 ───────────────────────────────────────────────────────
# The Hello-World shape: cron trigger + one MCP tool call. Mirrors
# the "Greet Alessandro Every Minute" workflow on mixout-agent.

_SCHEDULED_TELEGRAM = WorkflowExample(
    name="scheduled-telegram-ping",
    description=(
        "Fire on a cron and send one Telegram message. The simplest "
        "two-block shape every other example builds on."
    ),
    patterns=("trigger-schedule", "mcp-tool", "linear DAG"),
    graph={
        "version": 1,
        "nodes": [
            {
                "id": "n1",
                "type": "trigger-schedule",
                "label": "Every minute",
                "position": {"x": 100, "y": 100},
                "config": {"cron_expression": "* * * * *"},
            },
            {
                "id": "n2",
                "type": "mcp-tool",
                "label": "Send Telegram message",
                "position": {"x": 400, "y": 100},
                "config": {
                    "mcp_name": "messaging",
                    "tool_name": "messaging_telegram_send_message",
                    "args": {
                        "chat_id": "<your_telegram_chat_id>",
                        "text": "Hello from OpenAgent at {{now}}!",
                    },
                },
            },
        ],
        "edges": [
            {
                "id": "e1",
                "source": "n1", "target": "n2",
                "sourceHandle": "out", "targetHandle": "in",
            },
        ],
        "variables": {},
    },
)


# ── Example 2 ───────────────────────────────────────────────────────
# Branching on a tool result. Demonstrates the if-block's true/false
# sourceHandles — the single most common LLM-authoring pitfall.

_BRANCH_ON_RESULT = WorkflowExample(
    name="branch-on-result",
    description=(
        "Run a shell command, branch on whether stdout contains 'OK'. "
        "Demonstrates the `if` block's true/false sourceHandles — every "
        "outgoing edge from an `if` MUST set sourceHandle to either "
        "'true' or 'false'; the default 'out' will never fire."
    ),
    patterns=("trigger-manual", "mcp-tool", "if (true/false handles)", "branching"),
    graph={
        "version": 1,
        "nodes": [
            {
                "id": "n1",
                "type": "trigger-manual",
                "label": "Run",
                "position": {"x": 100, "y": 100},
                "config": {},
            },
            {
                "id": "n2",
                "type": "mcp-tool",
                "label": "Health check",
                "position": {"x": 320, "y": 100},
                "config": {
                    "mcp_name": "shell",
                    "tool_name": "shell_exec",
                    "args": {"command": "curl -fsS https://example.com/health"},
                },
            },
            {
                "id": "n3",
                "type": "if",
                "label": "Healthy?",
                "position": {"x": 580, "y": 100},
                "config": {
                    "expression": "{{ 'OK' in nodes.n2.output.result.stdout }}",
                },
            },
            {
                "id": "n4",
                "type": "mcp-tool",
                "label": "Notify success",
                "position": {"x": 820, "y": 40},
                "config": {
                    "mcp_name": "messaging",
                    "tool_name": "messaging_telegram_send_message",
                    "args": {
                        "chat_id": "<your_telegram_chat_id>",
                        "text": "Health check OK at {{now}}",
                    },
                },
            },
            {
                "id": "n5",
                "type": "mcp-tool",
                "label": "Notify failure",
                "position": {"x": 820, "y": 180},
                "config": {
                    "mcp_name": "messaging",
                    "tool_name": "messaging_telegram_send_message",
                    "args": {
                        "chat_id": "<your_telegram_chat_id>",
                        "text": "Health check FAILED at {{now}} — {{nodes.n2.output.result.stdout}}",
                    },
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "n1", "target": "n2",
             "sourceHandle": "out", "targetHandle": "in"},
            {"id": "e2", "source": "n2", "target": "n3",
             "sourceHandle": "out", "targetHandle": "in"},
            {"id": "e3", "source": "n3", "target": "n4",
             "sourceHandle": "true", "targetHandle": "in"},
            {"id": "e4", "source": "n3", "target": "n5",
             "sourceHandle": "false", "targetHandle": "in"},
        ],
        "variables": {},
    },
)


# ── Example 3 ───────────────────────────────────────────────────────
# AI-then-tool: ai-prompt produces text, mcp-tool acts on it. The
# canonical "draft + send" pattern.

_AI_THEN_TOOL = WorkflowExample(
    name="ai-then-tool",
    description=(
        "Ask the AI to draft a daily standup summary, then send it to "
        "Telegram. Demonstrates referencing an ai-prompt's output via "
        "{{nodes.<id>.output.text}} from a downstream mcp-tool's args."
    ),
    patterns=("trigger-schedule", "ai-prompt", "mcp-tool", "templating into args"),
    graph={
        "version": 1,
        "nodes": [
            {
                "id": "n1",
                "type": "trigger-schedule",
                "label": "Daily 9am",
                "position": {"x": 100, "y": 100},
                "config": {"cron_expression": "0 9 * * *"},
            },
            {
                "id": "n2",
                "type": "ai-prompt",
                "label": "Draft standup",
                "position": {"x": 360, "y": 100},
                "config": {
                    "prompt": (
                        "Write a 3-bullet daily standup based on yesterday's "
                        "git log in this repo. Be terse — no preamble."
                    ),
                    "session_policy": "ephemeral",
                },
            },
            {
                "id": "n3",
                "type": "mcp-tool",
                "label": "Send to Telegram",
                "position": {"x": 660, "y": 100},
                "config": {
                    "mcp_name": "messaging",
                    "tool_name": "messaging_telegram_send_message",
                    "args": {
                        "chat_id": "<your_telegram_chat_id>",
                        "text": "Daily standup:\n{{nodes.n2.output.text}}",
                    },
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "n1", "target": "n2",
             "sourceHandle": "out", "targetHandle": "in"},
            {"id": "e2", "source": "n2", "target": "n3",
             "sourceHandle": "out", "targetHandle": "in"},
        ],
        "variables": {},
    },
)


# ── Example 4 ───────────────────────────────────────────────────────
# Parallel fan-out + merge. Each parallel branch must use a distinct
# branch_<i> sourceHandle.

_PARALLEL_FETCH_MERGE = WorkflowExample(
    name="parallel-fetch-merge",
    description=(
        "Fan out two HTTP requests concurrently, merge the responses. "
        "Every outgoing edge from `parallel` MUST set sourceHandle to a "
        "distinct 'branch_<i>' (branch_0, branch_1, ...). The downstream "
        "merge collects from both branches before its own out edge fires."
    ),
    patterns=(
        "trigger-manual", "parallel (branch_<i> handles)",
        "http-request", "merge (collect upstream)",
    ),
    graph={
        "version": 1,
        "nodes": [
            {
                "id": "n1",
                "type": "trigger-manual",
                "label": "Run",
                "position": {"x": 100, "y": 100},
                "config": {},
            },
            {
                "id": "n2",
                "type": "parallel",
                "label": "Fan out",
                "position": {"x": 320, "y": 100},
                "config": {"branches": 2},
            },
            {
                "id": "n3",
                "type": "http-request",
                "label": "GitHub status",
                "position": {"x": 580, "y": 40},
                "config": {
                    "method": "GET",
                    "url": "https://www.githubstatus.com/api/v2/status.json",
                    "timeout_s": 10,
                },
            },
            {
                "id": "n4",
                "type": "http-request",
                "label": "Cloudflare status",
                "position": {"x": 580, "y": 200},
                "config": {
                    "method": "GET",
                    "url": "https://www.cloudflarestatus.com/api/v2/status.json",
                    "timeout_s": 10,
                },
            },
            {
                "id": "n5",
                "type": "merge",
                "label": "Combine",
                "position": {"x": 860, "y": 100},
                "config": {"strategy": "all", "collect_as": "statuses"},
            },
        ],
        "edges": [
            {"id": "e1", "source": "n1", "target": "n2",
             "sourceHandle": "out", "targetHandle": "in"},
            {"id": "e2", "source": "n2", "target": "n3",
             "sourceHandle": "branch_0", "targetHandle": "in"},
            {"id": "e3", "source": "n2", "target": "n4",
             "sourceHandle": "branch_1", "targetHandle": "in"},
            {"id": "e4", "source": "n3", "target": "n5",
             "sourceHandle": "out", "targetHandle": "in"},
            {"id": "e5", "source": "n4", "target": "n5",
             "sourceHandle": "out", "targetHandle": "in"},
        ],
        "variables": {},
    },
)


# ── Example 5 ───────────────────────────────────────────────────────
# Loop over a list. The loop body's tail re-enters the loop via the
# 'body' targetHandle — the one cycle the validator allows.

_LOOP_OVER_LIST = WorkflowExample(
    name="loop-over-list",
    description=(
        "Read a list from a tool result, iterate over it, send a "
        "Telegram message for each item, then post a summary when the "
        "loop is done. The loop's body subgraph is a regular DAG that "
        "starts from sourceHandle='body' — the loop handler runs it "
        "once per item internally; do NOT add a back-edge from the "
        "body's tail to the loop. Reference the current item inside "
        "the body via {{vars.item}} (the default iteration_var). The "
        "'done' handle fires once after the last iteration."
    ),
    patterns=(
        "trigger-manual", "mcp-tool",
        "loop (body subgraph + done handle)", "templating from vars",
    ),
    graph={
        "version": 1,
        "nodes": [
            {
                "id": "n1",
                "type": "trigger-manual",
                "label": "Run",
                "position": {"x": 100, "y": 100},
                "config": {},
            },
            {
                "id": "n2",
                "type": "mcp-tool",
                "label": "List inbox",
                "position": {"x": 320, "y": 100},
                "config": {
                    "mcp_name": "shell",
                    "tool_name": "shell_exec",
                    "args": {"command": "ls /tmp/incoming"},
                },
            },
            {
                "id": "n3",
                "type": "loop",
                "label": "For each file",
                "position": {"x": 580, "y": 100},
                "config": {
                    "items_expr": "{{ nodes.n2.output.result.stdout.splitlines() }}",
                    "iteration_var": "item",
                    "max_iterations": 50,
                },
            },
            {
                "id": "n4",
                "type": "mcp-tool",
                "label": "Notify per file",
                "position": {"x": 840, "y": 40},
                "config": {
                    "mcp_name": "messaging",
                    "tool_name": "messaging_telegram_send_message",
                    "args": {
                        "chat_id": "<your_telegram_chat_id>",
                        "text": "Processing file: {{vars.item}}",
                    },
                },
            },
            {
                "id": "n5",
                "type": "mcp-tool",
                "label": "Summary",
                "position": {"x": 840, "y": 200},
                "config": {
                    "mcp_name": "messaging",
                    "tool_name": "messaging_telegram_send_message",
                    "args": {
                        "chat_id": "<your_telegram_chat_id>",
                        "text": "All files processed at {{now}}.",
                    },
                },
            },
        ],
        "edges": [
            {"id": "e1", "source": "n1", "target": "n2",
             "sourceHandle": "out", "targetHandle": "in"},
            {"id": "e2", "source": "n2", "target": "n3",
             "sourceHandle": "out", "targetHandle": "in"},
            # body subgraph: just a forward chain from the body handle.
            {"id": "e3", "source": "n3", "target": "n4",
             "sourceHandle": "body", "targetHandle": "in"},
            # done fires once after the last iteration.
            {"id": "e4", "source": "n3", "target": "n5",
             "sourceHandle": "done", "targetHandle": "in"},
        ],
        "variables": {},
    },
)


# ── Public registry ─────────────────────────────────────────────────

WORKFLOW_EXAMPLES: dict[str, WorkflowExample] = {
    ex.name: ex
    for ex in (
        _SCHEDULED_TELEGRAM,
        _BRANCH_ON_RESULT,
        _AI_THEN_TOOL,
        _PARALLEL_FETCH_MERGE,
        _LOOP_OVER_LIST,
    )
}


def list_workflow_examples() -> list[dict[str, Any]]:
    """Lightweight index — name + description + patterns, no graph.
    Cheap for the AI to scan when picking which example to pull in full."""
    return [
        {
            "name": ex.name,
            "description": ex.description,
            "patterns": list(ex.patterns),
        }
        for ex in WORKFLOW_EXAMPLES.values()
    ]


def get_workflow_example(name: str) -> dict[str, Any]:
    """Full example by name — raises KeyError with the known set on miss."""
    try:
        return WORKFLOW_EXAMPLES[name].as_dict()
    except KeyError:
        raise KeyError(
            f"Unknown workflow example {name!r}. "
            f"Known examples: {sorted(WORKFLOW_EXAMPLES)}"
        ) from None
