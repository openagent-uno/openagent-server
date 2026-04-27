"""Block type catalog for the workflow engine.

Single source of truth for every block the editor, the validator, the
executor, the ``GET /api/workflow-block-types`` endpoint, and the
``describe_block_type`` MCP tool all agree on. Adding a new block type
means editing this file and writing its executor handler — nothing
else.

Each ``BlockSpec`` carries:

- ``type``: the string stored in ``node.type`` inside ``graph_json``.
- ``category``: palette grouping (triggers, ai, tools, flow, utility).
- ``description``: shown in the palette and returned to the AI.
- ``config_schema``: lightweight JSON-schema-ish dict describing the
  fields the node's ``config`` may carry. Used by the validator and
  by the UI to render the properties panel. This is deliberately NOT
  full JSON-Schema — we want a compact shape the AI can skim.
- ``source_handles`` / ``target_handles``: the wire endpoints. Empty
  ``target_handles`` marks a trigger; empty ``source_handles`` marks
  a terminal.
- ``output_shape``: a short human-readable hint about what the block
  emits (used by ``describe_block_type`` to help the AI write
  templated references like ``{{n3.output.text}}``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BlockSpec:
    type: str
    category: str
    description: str
    config_schema: dict[str, Any]
    source_handles: tuple[str, ...] = ("out",)
    target_handles: tuple[str, ...] = ("in",)
    output_shape: str = "any"

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "category": self.category,
            "description": self.description,
            "config_schema": self.config_schema,
            "source_handles": list(self.source_handles),
            "target_handles": list(self.target_handles),
            "output_shape": self.output_shape,
        }


def _field(
    kind: str,
    *,
    required: bool = False,
    default: Any | None = None,
    enum: list[Any] | None = None,
    description: str = "",
    items: dict | None = None,
) -> dict[str, Any]:
    """Shorthand for a config-schema field entry."""
    spec: dict[str, Any] = {"type": kind, "required": required, "description": description}
    if default is not None:
        spec["default"] = default
    if enum is not None:
        spec["enum"] = enum
    if items is not None:
        spec["items"] = items
    return spec


# ── Block catalog ────────────────────────────────────────────────────

BLOCK_CATALOG: dict[str, BlockSpec] = {
    # ── triggers ──
    "trigger-manual": BlockSpec(
        type="trigger-manual",
        category="triggers",
        description=(
            "Entry point fired by the UI's Run button or a manual HTTP POST "
            "to /api/workflows/{id}/run. Renders a form from inputs_schema "
            "so the user can pass values that downstream blocks reach via "
            "{{inputs.<field>}}."
        ),
        config_schema={
            "inputs_schema": _field(
                "object",
                description=(
                    "Optional JSON-Schema-shaped description of the inputs. "
                    "Keys become form fields in the Run dialog."
                ),
            ),
        },
        target_handles=(),
        output_shape="the inputs object",
    ),
    "trigger-schedule": BlockSpec(
        type="trigger-schedule",
        category="triggers",
        description=(
            "Fires on a cron schedule. The cron_expression field also drives "
            "the row-level next_run_at, so the existing Scheduler loop picks "
            "up workflows alongside scheduled_tasks without a second poller."
        ),
        config_schema={
            "cron_expression": _field(
                "string", required=True,
                description="Standard 5-field cron or '@once:<epoch>' for one-shot.",
            ),
        },
        target_handles=(),
        output_shape='{"triggered_at": ISO8601}',
    ),
    "trigger-ai": BlockSpec(
        type="trigger-ai",
        category="triggers",
        description=(
            "Entry point invoked when the AI calls run_workflow(id_or_name, "
            "inputs). The description field is shown to the AI when it "
            "lists workflows, so write it like a tool docstring."
        ),
        config_schema={
            "description": _field(
                "string", required=True,
                description="Human-readable docstring the AI reads to decide when to invoke.",
            ),
        },
        target_handles=(),
        output_shape="the inputs object the AI passed",
    ),

    # ── MCP tool call ──
    "mcp-tool": BlockSpec(
        type="mcp-tool",
        category="tools",
        description=(
            "Call a tool from any connected MCP (builtin or user-configured). "
            "Arg values support {{...}} templating against ctx.nodes, "
            "ctx.inputs, and ctx.vars (e.g. \"{{nodes.n2.output.text}}\"). "
            "The tool_name MUST be the prefixed form the pool actually "
            "exposes (e.g. 'messaging_telegram_send_message', NOT bare "
            "'telegram_send_message') — call list_available_tools first or "
            "consult /api/mcp-tools to see the canonical names. The bare "
            "form is auto-repaired but emitting the prefixed form keeps "
            "trace_json clean and avoids relying on the repair."
        ),
        config_schema={
            "mcp_name": _field(
                "string", required=True,
                description=(
                    "MCP server name as it appears in /api/mcp-tools "
                    "(e.g. 'shell', 'messaging', 'scheduler')."
                ),
            ),
            "tool_name": _field(
                "string", required=True,
                description=(
                    "Prefixed tool name within that MCP, e.g. 'shell_exec', "
                    "'messaging_telegram_send_message', 'scheduler_create_one_shot_task'."
                ),
            ),
            "args": _field(
                "object", default={},
                description="Keyword args passed to the tool. String values are template-resolved.",
            ),
            "on_error": _field(
                "string", default="halt",
                enum=["halt", "continue", "branch"],
                description="What happens if the tool raises.",
            ),
        },
        output_shape='{"result": <whatever the tool returned>}',
    ),

    # ── AI prompt ──
    "ai-prompt": BlockSpec(
        type="ai-prompt",
        category="ai",
        description=(
            "Run a prompt through the OpenAgent AI (same agent.run path as "
            "scheduled tasks). Optionally pin a specific model via "
            "model_override to bypass the SmartRouter."
        ),
        config_schema={
            "prompt": _field(
                "string", required=True,
                description="User message. Supports {{...}} templating.",
            ),
            "system": _field(
                "string",
                description="Optional system prompt override.",
            ),
            "model_override": _field(
                "string",
                description=(
                    "runtime_id such as 'openai:gpt-4o-mini' or "
                    "'claude-cli:anthropic:claude-opus-4-7'. When set, "
                    "bypasses the SmartRouter and dispatches directly."
                ),
            ),
            "session_policy": _field(
                "string", default="ephemeral",
                enum=["ephemeral", "shared"],
                description=(
                    "'ephemeral' — fresh conversation per block. "
                    "'shared' — all ai-prompt blocks in a single run share "
                    "one session so the AI has rolling memory."
                ),
            ),
            "on_error": _field(
                "string", default="halt",
                enum=["halt", "continue", "branch"],
                description="What happens if the model call raises.",
            ),
        },
        output_shape='{"text": str, "usage"?: {input_tokens, output_tokens, cost}}',
    ),

    # ── flow control ──
    "if": BlockSpec(
        type="if",
        category="flow",
        description=(
            "Route control via a jinja expression evaluated against ctx. "
            "Edges with sourceHandle='true' run when the expression is "
            "truthy; 'false' otherwise. WIRING: every outgoing edge from "
            "an `if` MUST set sourceHandle to either 'true' or 'false' "
            "(default 'out' will never fire). The two branches cascade-"
            "skip independently — downstream merges can collect both."
        ),
        config_schema={
            "expression": _field(
                "string", required=True,
                description=(
                    "Jinja expression, e.g. \"{{n3.output.status == 'ok'}}\". "
                    "Evaluated via sandbox."
                ),
            ),
        },
        source_handles=("true", "false"),
        output_shape="no output; routes via sourceHandle",
    ),
    "loop": BlockSpec(
        type="loop",
        category="flow",
        description=(
            "Iterate over a list. items_expr resolves to a sequence; the "
            "body subgraph (sourceHandle='body') runs once per item with "
            "ctx.vars[iteration_var] set. 'done' fires once after the "
            "last iteration. WIRING: connect the loop's 'body' handle "
            "to the FIRST node of a forward DAG; the loop handler runs "
            "that subgraph once per item internally — do NOT add a "
            "back-edge from the body's tail to the loop (the validator "
            "rejects it). Wire the 'done' handle to whatever should run "
            "after the loop completes. Reference the current item via "
            "\"{{vars.<iteration_var>}}\" (default \"{{vars.item}}\")."
        ),
        config_schema={
            "items_expr": _field(
                "string", required=True,
                description="Jinja expression yielding a list, e.g. \"{{n2.output.items}}\".",
            ),
            "max_iterations": _field(
                "integer", default=100,
                description="Safety cap to prevent runaway loops.",
            ),
            "iteration_var": _field(
                "string", default="item",
                description="Name written to ctx.vars inside the body.",
            ),
        },
        source_handles=("body", "done"),
        output_shape='{"results": [<body output per iteration>]}',
    ),
    "wait": BlockSpec(
        type="wait",
        category="flow",
        description=(
            "Pause execution for a duration or until a specific time. "
            "Uses asyncio.sleep so the runtime stays responsive."
        ),
        config_schema={
            "mode": _field(
                "string", required=True, enum=["duration", "until"],
                description="'duration' sleeps seconds; 'until' sleeps to an ISO timestamp.",
            ),
            "seconds": _field("number", description="For mode='duration'."),
            "until_iso": _field("string", description="For mode='until', e.g. 2026-05-01T12:00:00Z."),
        },
        output_shape='{"waited_ms": int}',
    ),
    "parallel": BlockSpec(
        type="parallel",
        category="flow",
        description=(
            "Fan out to N branches that run concurrently. WIRING: each "
            "outgoing edge MUST set sourceHandle to a distinct "
            "'branch_<i>' (branch_0, branch_1, ...) — the default 'out' "
            "won't fire. Pair with a merge block downstream that gathers "
            "every branch's tail; the merge waits for all upstream edges "
            "before emitting."
        ),
        config_schema={
            "branches": _field(
                "integer", default=2,
                description="Layout hint; actual branches are derived from the edges.",
            ),
        },
        source_handles=("branch_0", "branch_1", "branch_2", "branch_3"),
        output_shape="no output; fans out via sourceHandle",
    ),
    "merge": BlockSpec(
        type="merge",
        category="flow",
        description=(
            "Wait for all upstream branches then emit a combined output. "
            "strategy='all' returns a list in upstream order; 'first' uses "
            "whichever finishes first; 'last' uses whichever finishes last."
        ),
        config_schema={
            "strategy": _field(
                "string", default="all",
                enum=["all", "first", "last"],
                description="How to combine upstream outputs.",
            ),
            "collect_as": _field(
                "string",
                description="Optional key name under which results are grouped.",
            ),
        },
        output_shape='{"collected": [...]} or the single chosen output',
    ),

    # ── utility ──
    "set-variable": BlockSpec(
        type="set-variable",
        category="utility",
        description=(
            "Write a value into ctx.vars so later blocks can reference it "
            "via {{vars.<key>}}. Useful for accumulators in loops and for "
            "naming derived values. The expression is evaluated against "
            "the same ctx as templating, so \"{{n2.output.text}}\" or "
            "\"{{inputs.user_id}}\" work directly."
        ),
        config_schema={
            "key": _field("string", required=True, description="Variable name."),
            "value_expr": _field(
                "string", required=True,
                description="Jinja expression yielding the value.",
            ),
        },
        output_shape='{"key": str, "value": <any>}',
    ),
    "http-request": BlockSpec(
        type="http-request",
        category="tools",
        description=(
            "Generic HTTP client. Method + URL + optional headers/body. "
            "Response is exposed as {status, headers, body, json?}."
        ),
        config_schema={
            "method": _field(
                "string", default="GET",
                enum=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"],
                description="HTTP verb.",
            ),
            "url": _field("string", required=True, description="Fully-qualified URL."),
            "headers": _field("object", default={}, description="Request headers."),
            "body": _field("string", description="Request body (string or JSON-stringified)."),
            "timeout_s": _field("number", default=30, description="Request timeout in seconds."),
            "on_error": _field(
                "string", default="halt",
                enum=["halt", "continue", "branch"],
                description="What happens if the request fails.",
            ),
        },
        output_shape='{"status": int, "headers": {...}, "body": str, "json"?: any}',
    ),
}


def get_block_spec(type_name: str) -> BlockSpec:
    """Look up a block spec or raise ``KeyError`` with a list of known types."""
    try:
        return BLOCK_CATALOG[type_name]
    except KeyError:
        raise KeyError(
            f"Unknown block type {type_name!r}. "
            f"Known types: {sorted(BLOCK_CATALOG)}"
        ) from None


def iter_block_specs() -> list[dict[str, Any]]:
    """Catalog-as-list for the REST API + ``describe_block_type`` MCP tool."""
    return [spec.as_dict() for spec in BLOCK_CATALOG.values()]
