"""Regression: ``mcp-tool`` blocks must dispatch to subprocess MCPs
correctly. Without the fix, ``_h_mcp_tool`` would do ``fn(**args)``
on the runtime ``Function`` Pydantic descriptor (which has no ``__call__``)
and the workflow would die with::

    TypeError: 'Function' object is not callable

The actual callable lives on ``Function.entrypoint``. The same
``getattr(fn, "entrypoint", None) or fn`` pattern is already used by
``tool_search/adapters.py:129`` — the executor was the lone hold-out.

This module also covers the validator side: when a callability snapshot
is supplied, ``validate_graph`` rejects mcp-tool blocks whose tool maps
to a non-callable, so the failure surfaces at workflow-save / run-start
instead of mid-DAG.
"""
from __future__ import annotations

import base64
from typing import Any

from ._framework import TestContext, test


# ── Shared stubs ────────────────────────────────────────────────────


class _RecordingAgent:
    """Minimal Agent stub. Workflow executor only pokes ``_mcp`` and
    ``forget_session`` (the latter from _finalize_run; we no-op it).
    """

    def __init__(self, pool: Any | None = None) -> None:
        self._mcp = pool
        self.model = None

    async def forget_session(self, session_id: str) -> None:
        return None

    async def release_session(self, session_id: str) -> None:
        return None


class _StubDB:
    """Minimal MemoryDB shape used by WorkflowExecutor."""

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self._counter = 0

    async def add_workflow_run(
        self,
        *,
        workflow_id: str,
        trigger: str,
        inputs: dict,
        run_id: str | None = None,
    ) -> str:
        self._counter += 1
        rid = run_id or f"run-{self._counter}"
        self.runs[rid] = {
            "id": rid,
            "workflow_id": workflow_id,
            "status": "running",
            "trigger": trigger,
            "inputs": inputs,
            "trace": [],
        }
        return rid

    async def update_workflow_run(self, run_id: str, **kwargs: Any) -> None:
        self.runs[run_id].update(kwargs)

    async def update_workflow(self, workflow_id: str, **kwargs: Any) -> None:
        return None

    async def get_workflow_run(self, run_id: str) -> dict | None:
        return self.runs.get(run_id)


def _agno_function_factory():
    """Return the runtime ``Function`` class. Skip the test when the runtime isn't
    importable rather than crashing the suite."""
    from src.mcp._runtime.function import Function
    return Function


def _make_pool(toolkits: dict[str, Any]) -> Any:
    """Build a shape-compatible pool stub: ``_toolkit_by_name`` (dict)
    + ``toolkit_by_name`` + ``list_mcp_tools`` so the validator's
    ``mcp_inventory_from_pool`` and the executor's ``_h_mcp_tool``
    both see the same toolkits."""
    class _PoolStub:
        def __init__(self, by_name):
            self._toolkit_by_name = by_name

        def toolkit_by_name(self, name):
            return self._toolkit_by_name.get(name)

        def list_mcp_tools(self):
            out = []
            for name, tk in self._toolkit_by_name.items():
                merged = {
                    **(getattr(tk, "functions", {}) or {}),
                    **(getattr(tk, "async_functions", {}) or {}),
                }
                out.append({
                    "mcp_name": name,
                    "tools": [
                        {
                            "name": tn,
                            "description": getattr(fn, "description", "") or "",
                            "parameters_schema": getattr(fn, "parameters", None) or {},
                        }
                        for tn, fn in merged.items()
                    ],
                })
            return out

    return _PoolStub(toolkits)


class _ToolkitStub:
    """Mimics the runtime ``MCPTools`` shape for our purposes: ``functions``
    dict (subprocess MCPs) + ``async_functions`` dict (in-process
    toolkits). The executor's ``_h_mcp_tool`` merges both, exactly
    like the live pool does."""

    def __init__(
        self,
        *,
        functions: dict[str, Any] | None = None,
        async_functions: dict[str, Any] | None = None,
    ) -> None:
        self.functions = functions or {}
        self.async_functions = async_functions or {}


# ── Direct executor tests (per-handler) ─────────────────────────────


@test("workflow_mcp_dispatch", "mcp-tool calls runtime Function.entrypoint")
async def t_function_entrypoint_called(ctx: TestContext) -> None:
    """Regression for ``TypeError: 'Function' object is not callable``.
    Without the executor fix, calling ``fn(**args)`` on a Pydantic
    ``Function`` blows up here. With the fix, the call routes to
    ``fn.entrypoint``."""
    from src.workflow.executor import (
        WorkflowExecutor, _RunCtx, _h_mcp_tool,
    )

    Function = _agno_function_factory()

    seen: dict[str, Any] = {}

    async def entrypoint(**kwargs: Any) -> str:
        seen["kwargs"] = kwargs
        return "ok"

    fn = Function(name="messaging_telegram_send_message", entrypoint=entrypoint)
    toolkit = _ToolkitStub(functions={fn.name: fn})
    pool = _make_pool({"messaging": toolkit})

    executor = WorkflowExecutor(agent=_RecordingAgent(pool=pool), db=_StubDB())  # type: ignore[arg-type]
    run_ctx = _RunCtx(run_id="r", workflow_id="w", inputs={}, vars={})

    result = await _h_mcp_tool(
        executor,
        {"id": "n1", "type": "mcp-tool"},
        {
            "mcp_name": "messaging",
            "tool_name": "messaging_telegram_send_message",
            "args": {"chat_id": "123", "text": "hi"},
        },
        run_ctx,
    )

    assert result == {"result": "ok"}, result
    assert seen["kwargs"] == {"chat_id": "123", "text": "hi"}, seen


@test("workflow_mcp_dispatch", "mcp-tool falls back to raw async callable")
async def t_async_function_raw_callable(ctx: TestContext) -> None:
    """In-process toolkits (e.g. ``tool-search``) register raw
    callables in ``async_functions`` — no ``Function`` wrapper. The
    ``or fn`` fallback in the executor must still invoke them."""
    from src.workflow.executor import (
        WorkflowExecutor, _RunCtx, _h_mcp_tool,
    )

    seen: dict[str, Any] = {}

    async def call_tool(**kwargs: Any) -> str:
        seen["kwargs"] = kwargs
        return "raw-ok"

    toolkit = _ToolkitStub(async_functions={"tool_search_call_tool": call_tool})
    pool = _make_pool({"tool-search": toolkit})

    executor = WorkflowExecutor(agent=_RecordingAgent(pool=pool), db=_StubDB())  # type: ignore[arg-type]
    run_ctx = _RunCtx(run_id="r", workflow_id="w", inputs={}, vars={})

    result = await _h_mcp_tool(
        executor,
        {"id": "n1", "type": "mcp-tool"},
        {
            "mcp_name": "tool-search",
            "tool_name": "tool_search_call_tool",
            "args": {"server": "x", "tool": "y", "args": {}},
        },
        run_ctx,
    )

    assert result == {"result": "raw-ok"}, result
    assert seen["kwargs"] == {"server": "x", "tool": "y", "args": {}}, seen


@test("workflow_mcp_dispatch", "mcp-tool handles sync callable without await")
async def t_sync_callable_no_await(ctx: TestContext) -> None:
    """A sync entrypoint returns a non-awaitable; the
    ``inspect.isawaitable`` branch must skip cleanly."""
    from src.workflow.executor import (
        WorkflowExecutor, _RunCtx, _h_mcp_tool,
    )

    Function = _agno_function_factory()

    def entrypoint(**kwargs: Any) -> dict:
        return {"echo": kwargs}

    fn = Function(name="util_echo", entrypoint=entrypoint)
    toolkit = _ToolkitStub(functions={fn.name: fn})
    pool = _make_pool({"util": toolkit})

    executor = WorkflowExecutor(agent=_RecordingAgent(pool=pool), db=_StubDB())  # type: ignore[arg-type]
    run_ctx = _RunCtx(run_id="r", workflow_id="w", inputs={}, vars={})

    result = await _h_mcp_tool(
        executor,
        {"id": "n1", "type": "mcp-tool"},
        {
            "mcp_name": "util",
            "tool_name": "util_echo",
            "args": {"a": 1},
        },
        run_ctx,
    )

    assert result == {"result": {"echo": {"a": 1}}}, result


@test(
    "workflow_mcp_dispatch",
    "mcp-tool preserves the complete rich runtime result envelope",
)
async def t_rich_runtime_result_is_lossless(ctx: TestContext) -> None:
    """A subprocess MCP is exposed to the executor as a runtime
    ``ToolResult``.  Its display ``content`` is intentionally only a string;
    the original CallToolResult envelope and media live on separate fields.
    Serialising that model with ``default=str`` therefore discarded protocol
    content, structured data, error/meta fields, artifacts and child lineage.
    """
    from src.mcp._runtime.function import ToolResult
    from src.stream.media import Audio, File, Image, Video
    from src.workflow.executor import (
        WorkflowExecutor, _RunCtx, _h_mcp_tool,
    )

    wire_content = [
        {"type": "text", "text": "hello"},
        {
            "type": "image",
            "data": base64.b64encode(b"wire-image").decode("ascii"),
            "mimeType": "image/png",
        },
        {
            "type": "audio",
            "data": base64.b64encode(b"wire-audio").decode("ascii"),
            "mimeType": "audio/mpeg",
        },
        {
            "type": "resource_link",
            "name": "report",
            "uri": "https://example.test/report.pdf",
            "mimeType": "application/pdf",
            "size": 321,
        },
    ]
    runtime_result = ToolResult(
        content="display-only summary",
        mcp_result={
            "content": wire_content,
            "structuredContent": {
                "rows": [{"id": 1, "active": True}],
                "nullable": None,
            },
            "isError": False,
            "_meta": {"vendor": "rich-test", "requestId": "req-1"},
        },
        images=[Image(id="image-1", content=b"runtime-image", mime_type="image/png")],
        audios=[Audio(id="audio-1", content=b"runtime-audio", mime_type="audio/mpeg")],
        videos=[Video(id="video-1", content=b"runtime-video", mime_type="video/mp4")],
        files=[File(
            id="file-1", content=b"runtime-file", mime_type="application/pdf",
            filename="report.pdf",
        )],
        child_session_id="session:child-1",
    )
    # Some runtime result types also carry the team member run identifier.  It
    # is not yet a declared ToolResult field, but the shared envelope codec
    # deliberately preserves it when present for forward compatibility.
    object.__setattr__(runtime_result, "child_run_id", "run:child-1")

    async def rich_tool() -> ToolResult:
        return runtime_result

    toolkit = _ToolkitStub(async_functions={"rich_tool": rich_tool})
    pool = _make_pool({"rich": toolkit})
    executor = WorkflowExecutor(agent=_RecordingAgent(pool=pool), db=_StubDB())  # type: ignore[arg-type]
    run_ctx = _RunCtx(run_id="r", workflow_id="w", inputs={}, vars={})

    output = await _h_mcp_tool(
        executor,
        {"id": "n-rich", "type": "mcp-tool"},
        {"mcp_name": "rich", "tool_name": "rich_tool", "args": {}},
        run_ctx,
    )
    envelope = output["result"]

    assert envelope["content"] == wire_content, envelope
    assert envelope["structuredContent"] == {
        "rows": [{"id": 1, "active": True}], "nullable": None,
    }, envelope
    assert envelope["isError"] is False, envelope
    assert envelope["_meta"] == {
        "vendor": "rich-test", "requestId": "req-1",
    }, envelope
    assert envelope["images"][0]["content"] == base64.b64encode(
        b"runtime-image",
    ).decode("ascii"), envelope
    assert envelope["audios"][0]["content"] == base64.b64encode(
        b"runtime-audio",
    ).decode("ascii"), envelope
    assert envelope["videos"][0]["content"] == base64.b64encode(
        b"runtime-video",
    ).decode("ascii"), envelope
    assert envelope["files"][0]["content"] == base64.b64encode(
        b"runtime-file",
    ).decode("ascii"), envelope
    assert envelope["child_session_id"] == "session:child-1", envelope
    assert envelope["child_run_id"] == "run:child-1", envelope


# ── Validator tests ─────────────────────────────────────────────────


@test("workflow_mcp_dispatch", "validate_graph rejects non-callable Function")
async def t_validate_rejects_uncallable_function(ctx: TestContext) -> None:
    """When a callability snapshot says a tool is non-callable,
    ``validate_graph`` must reject the graph before it ever runs."""
    from src.workflow.validate import ValidationError, validate_graph

    graph = {
        "version": 1,
        "nodes": [
            {
                "id": "n1",
                "type": "mcp-tool",
                "config": {
                    "mcp_name": "messaging",
                    "tool_name": "messaging_telegram_send_message",
                    "args": {"chat_id": "1", "text": "x"},
                },
            },
        ],
        "edges": [],
        "variables": {},
    }
    inventory = {
        "messaging": {"messaging_telegram_send_message": {}},
    }
    callability = {
        "messaging": {"messaging_telegram_send_message": False},
    }

    raised = False
    try:
        validate_graph(
            graph,
            mcp_inventory=inventory,
            mcp_callability=callability,
        )
    except ValidationError as exc:
        raised = True
        assert "non-callable" in str(exc), str(exc)
        assert exc.node_id == "n1", exc.node_id
        assert exc.field == "tool_name", exc.field
    assert raised, "validate_graph should have raised on non-callable tool"


@test("workflow_mcp_dispatch", "mcp_callability_from_pool snapshots the live pool")
async def t_callability_snapshot_shape(ctx: TestContext) -> None:
    """The helper must mark Function-with-entrypoint and raw callables
    as ``True``, and Function-without-entrypoint as ``False``."""
    from src.workflow.validate import mcp_callability_from_pool

    Function = _agno_function_factory()

    async def entrypoint(**_: Any) -> str:
        return ""

    good = Function(name="good_tool", entrypoint=entrypoint)
    bad = Function(name="bad_tool")  # entrypoint defaults to None
    toolkit_sub = _ToolkitStub(functions={good.name: good, bad.name: bad})

    async def raw(**_: Any) -> str:
        return ""

    toolkit_inproc = _ToolkitStub(async_functions={"raw_tool": raw})

    pool = _make_pool({"sub": toolkit_sub, "inproc": toolkit_inproc})
    snapshot = mcp_callability_from_pool(pool)

    assert snapshot is not None, snapshot
    assert snapshot["sub"]["good_tool"] is True, snapshot["sub"]
    assert snapshot["sub"]["bad_tool"] is False, snapshot["sub"]
    assert snapshot["inproc"]["raw_tool"] is True, snapshot["inproc"]


# ── End-to-end: executor re-validates at run-start ──────────────────


@test(
    "workflow_mcp_dispatch",
    "executor re-validates at run-start with live pool",
)
async def t_executor_revalidates_at_run_start(ctx: TestContext) -> None:
    """Workflows authored via the workflow-manager subprocess validate
    *without* the live pool. The executor must re-validate at run-start
    so a stale tool reference (or the Function-not-callable bug class)
    fails as a finalized ``failed`` run with a clear error in
    ``trace_json``, not a mid-DAG ``TypeError``."""
    from src.workflow.executor import WorkflowExecutor

    Function = _agno_function_factory()

    bad = Function(name="messaging_telegram_send_message")  # no entrypoint
    toolkit = _ToolkitStub(functions={bad.name: bad})
    pool = _make_pool({"messaging": toolkit})

    executor = WorkflowExecutor(agent=_RecordingAgent(pool=pool), db=_StubDB())  # type: ignore[arg-type]

    workflow = {
        "id": "wf-1",
        "name": "broken",
        "graph": {
            "version": 1,
            "nodes": [
                {
                    "id": "n2",
                    "type": "mcp-tool",
                    "config": {
                        "mcp_name": "messaging",
                        "tool_name": "messaging_telegram_send_message",
                        "args": {"chat_id": "1", "text": "x"},
                    },
                },
            ],
            "edges": [],
            "variables": {},
        },
    }

    final = await executor.run(workflow, trigger="manual")

    assert final["status"] == "failed", final
    error = final.get("error") or ""
    assert "non-callable" in error, error
    assert "messaging" in error, error
    # Crucially NOT the opaque mid-DAG TypeError we used to get.
    assert "'Function' object is not callable" not in error, error


@test(
    "workflow_mcp_dispatch",
    "validate_graph auto-repairs double-prefixed tool names (shell_shell_exec → shell_exec)",
)
async def t_validate_repairs_double_prefix(ctx: TestContext) -> None:
    """When an LLM emits mcp_name + '_' + already-prefixed tool_name
    (e.g. tool_name='shell_shell_exec' for MCP 'shell'), the validator
    must strip the redundant prefix and repair in place rather than
    raising a ValidationError. Regression for the mixout Git Sync
    workflow that kept crashing with ValidationError: shell_shell_exec."""
    from src.workflow.validate import validate_graph

    config = {
        "mcp_name": "shell",
        "tool_name": "shell_shell_exec",  # double-prefixed — the bug
        "args": {},
    }
    graph = {
        "version": 1,
        "nodes": [{"id": "n1", "type": "mcp-tool", "config": config}],
        "edges": [],
        "variables": {},
    }
    inventory = {
        "shell": {
            "shell_exec": {},
            "shell_input": {},
            "shell_kill": {},
        },
    }

    # Must not raise — double-prefix should be silently repaired.
    validate_graph(graph, mcp_inventory=inventory)

    # Verify the config was repaired in-place.
    assert config["tool_name"] == "shell_exec", config["tool_name"]


@test(
    "workflow_mcp_dispatch",
    "mcp-tool accepts 'arguments' as synonym for 'args'",
)
async def t_executor_accepts_arguments_alias(ctx: TestContext) -> None:
    """LLM-authored workflows commonly emit ``arguments`` (matching the
    MCP JSON-RPC ``CallToolRequest.params.arguments`` field) instead of
    the block schema's canonical ``args``. Without the fallback in
    ``_h_mcp_tool``, ``cfg.get('args')`` returned ``{}``, the wrapped
    function was called with no kwargs, and the workflow died with
    ``TypeError: shell_exec() missing 1 required positional argument:
    'command'``. Regression for the Mixout Daily Release Check workflow
    (run 9b0eb495-…)."""
    from src.workflow.executor import (
        WorkflowExecutor, _RunCtx, _h_mcp_tool,
    )

    Function = _agno_function_factory()

    seen: dict[str, Any] = {}

    async def entrypoint(*, command: str, timeout: int = 60) -> str:
        seen["command"] = command
        seen["timeout"] = timeout
        return "ran"

    fn = Function(name="shell_exec", entrypoint=entrypoint)
    toolkit = _ToolkitStub(functions={fn.name: fn})
    pool = _make_pool({"shell": toolkit})

    executor = WorkflowExecutor(agent=_RecordingAgent(pool=pool), db=_StubDB())  # type: ignore[arg-type]
    run_ctx = _RunCtx(run_id="r", workflow_id="w", inputs={}, vars={})

    result = await _h_mcp_tool(
        executor,
        {"id": "n12", "type": "mcp-tool"},
        {
            "mcp_name": "shell",
            "tool_name": "shell_exec",
            # NOTE: 'arguments', not 'args' — the bug-trigger field.
            "arguments": {"command": "echo hi", "timeout": 600},
        },
        run_ctx,
    )

    assert result == {"result": "ran"}, result
    assert seen == {"command": "echo hi", "timeout": 600}, seen


@test(
    "workflow_mcp_dispatch",
    "validate_graph repairs 'arguments' → 'args' in place",
)
async def t_validate_repairs_arguments_alias(ctx: TestContext) -> None:
    """The validator must rename ``arguments`` → ``args`` so trace_json
    stays consistent and the missing-required-args check uses the right
    keys. Mirrors the existing tool_name auto-repair pattern."""
    from src.workflow.validate import validate_graph

    config = {
        "mcp_name": "shell",
        "tool_name": "shell_exec",
        # Uses 'arguments' — should be renamed to 'args' in place.
        "arguments": {"command": "echo hi"},
    }
    graph = {
        "version": 1,
        "nodes": [{"id": "n1", "type": "mcp-tool", "config": config}],
        "edges": [],
        "variables": {},
    }
    inventory = {
        "shell": {"shell_exec": {"required": ["command"]}},
    }

    # Must not raise: the repair must run before the missing-required
    # check, so ``args.command`` is seen as present.
    validate_graph(graph, mcp_inventory=inventory)

    assert "arguments" not in config, config
    assert config["args"] == {"command": "echo hi"}, config["args"]


@test(
    "workflow_mcp_dispatch",
    "validate_graph keeps 'args' when both 'args' and 'arguments' set",
)
async def t_validate_args_wins_over_arguments(ctx: TestContext) -> None:
    """If a workflow has both keys (corrupt or hand-edited), the
    canonical ``args`` wins and ``arguments`` is left untouched.
    Repairing in this case would silently change behaviour."""
    from src.workflow.validate import validate_graph

    config = {
        "mcp_name": "shell",
        "tool_name": "shell_exec",
        "args": {"command": "echo args"},
        "arguments": {"command": "echo arguments"},
    }
    graph = {
        "version": 1,
        "nodes": [{"id": "n1", "type": "mcp-tool", "config": config}],
        "edges": [],
        "variables": {},
    }
    inventory = {"shell": {"shell_exec": {"required": ["command"]}}}

    validate_graph(graph, mcp_inventory=inventory)

    assert config["args"] == {"command": "echo args"}, config["args"]
    assert config["arguments"] == {"command": "echo arguments"}, config["arguments"]


@test(
    "workflow_mcp_dispatch",
    "validate_graph flags required args set to None or empty string",
)
async def t_validate_rejects_empty_required_args(ctx: TestContext) -> None:
    """Required args present as ``None`` or ``""`` are as broken as
    missing keys — they crash mid-DAG with an opaque MCP error.

    LLM-authored workflows routinely emit ``{"text": null}`` or
    ``{"text": ""}`` when the model couldn't synthesise a value but
    still wanted to honour the schema's key set. Passing those through
    means the user only learns about the bug after the run starts, in
    a ``trace_json`` failure that names the MCP error rather than the
    config field.

    Templated values (``"{{ ctx.inputs.x }}"``) are non-empty strings
    and must continue to pass — the validator doesn't resolve them.
    """
    from src.workflow.validate import ValidationError, validate_graph

    inventory = {"messaging": {"send": {"required": ["chat_id", "text"]}}}

    def _graph(args: dict) -> dict:
        return {
            "version": 1,
            "nodes": [
                {
                    "id": "n1",
                    "type": "mcp-tool",
                    "config": {
                        "mcp_name": "messaging",
                        "tool_name": "send",
                        "args": args,
                    },
                },
            ],
            "edges": [],
            "variables": {},
        }

    # text: None → flagged
    raised = False
    try:
        validate_graph(_graph({"chat_id": "1", "text": None}), mcp_inventory=inventory)
    except ValidationError as exc:
        raised = True
        assert "['text']" in str(exc), str(exc)
        assert exc.node_id == "n1", exc.node_id
        assert exc.field == "args", exc.field
    assert raised, "validate_graph should reject text=None"

    # text: "" → flagged
    raised = False
    try:
        validate_graph(_graph({"chat_id": "1", "text": ""}), mcp_inventory=inventory)
    except ValidationError as exc:
        raised = True
        assert "['text']" in str(exc), str(exc)
    assert raised, "validate_graph should reject text=''"

    # Both missing and empty → both reported
    raised = False
    try:
        validate_graph(_graph({"text": ""}), mcp_inventory=inventory)
    except ValidationError as exc:
        raised = True
        msg = str(exc)
        assert "chat_id" in msg and "text" in msg, msg
    assert raised, "validate_graph should report both missing chat_id and empty text"

    # Templated value → still passes (non-empty string, validator
    # doesn't resolve templates).
    validate_graph(
        _graph({"chat_id": "1", "text": "{{ nodes.n0.output.body }}"}),
        mcp_inventory=inventory,
    )

    # Literal non-empty values → still pass.
    validate_graph(
        _graph({"chat_id": "1", "text": "hello"}),
        mcp_inventory=inventory,
    )

    # Numeric / boolean falsy values for non-required args don't trip
    # the check (regression guard against ``in (None, "")`` mistakenly
    # treating ``0`` or ``False`` as empty when they're legitimate
    # values for typed args).
    inventory_with_num = {"foo": {"bar": {"required": ["count", "flag"]}}}
    graph_num = {
        "version": 1,
        "nodes": [
            {
                "id": "n1",
                "type": "mcp-tool",
                "config": {
                    "mcp_name": "foo",
                    "tool_name": "bar",
                    "args": {"count": 0, "flag": False},
                },
            },
        ],
        "edges": [],
        "variables": {},
    }
    validate_graph(graph_num, mcp_inventory=inventory_with_num)


@test(
    "workflow_mcp_dispatch",
    "validate_graph repairs a bare tool name for a HYPHENATED MCP (aaa-support)",
)
async def t_validate_repairs_hyphenated_prefix(ctx: TestContext) -> None:
    """The runtime prefixes remote tools with the SAFE name (non-alnum →
    "_"), so ``aaa-support``'s ``threads_list`` registers as
    ``aaa_support_threads_list``. The bare→prefixed auto-repair must use
    the safe prefix, not the raw mcp_name — otherwise it builds
    ``aaa-support_threads_list`` (hyphen), never matches, and the block
    fails validation against a tool the pool actually exposes. Regression
    for lyra's customer-support-coverage workflow."""
    from src.workflow.validate import validate_graph

    config = {"mcp_name": "aaa-support", "tool_name": "threads_list", "args": {}}
    graph = {
        "version": 1,
        "nodes": [{"id": "n1", "type": "mcp-tool", "config": config}],
        "edges": [],
        "variables": {},
    }
    inventory = {
        "aaa-support": {
            "aaa_support_threads_list": {},
            "aaa_support_thread_brief": {},
        },
    }
    # Must not raise — the bare name auto-repairs to the safe-prefixed one.
    validate_graph(graph, mcp_inventory=inventory)
    assert config["tool_name"] == "aaa_support_threads_list", config["tool_name"]


@test(
    "workflow_mcp_dispatch",
    "validate_graph explains a present-but-empty MCP with its connect error",
)
async def t_validate_empty_mcp_surfaces_connect_error(ctx: TestContext) -> None:
    """When an MCP is loaded but exposes 0 tools (a swallowed 401 /
    handshake failure), validation must say so and surface the captured
    cause — NOT the misleading "has no tool X. Available: []" that sent
    lyra (and a human triager) hunting for a nonexistent naming bug."""
    from src.workflow.validate import ValidationError, validate_graph

    config = {"mcp_name": "aaa-support", "tool_name": "threads_list", "args": {}}
    graph = {
        "version": 1,
        "nodes": [{"id": "n2", "type": "mcp-tool", "config": config}],
        "edges": [],
        "variables": {},
    }
    inventory = {"aaa-support": {}}  # present but empty
    errors = {"aaa-support": "HTTPStatusError: 401 Unauthorized"}

    raised = False
    try:
        validate_graph(graph, mcp_inventory=inventory, mcp_errors=errors)
    except ValidationError as exc:
        raised = True
        msg = str(exc)
        assert "exposes no tools" in msg, msg
        assert "401" in msg, msg
        # Points at the MCP/config, not a bogus tool-name typo.
        assert exc.field == "mcp_name", exc.field
        assert "Available: []" not in msg, msg
    assert raised, "validate_graph should explain a present-but-empty MCP"


@test(
    "workflow_mcp_dispatch",
    "mcp_errors_from_pool snapshots pool._last_connect_error (drops blanks)",
)
async def t_mcp_errors_from_pool(ctx: TestContext) -> None:
    from src.workflow.validate import mcp_errors_from_pool

    class _Pool:
        _last_connect_error = {
            "aaa-support": "HTTPStatusError: 401 Unauthorized",
            "healthy": "",  # connected fine → blank, must be dropped
        }

    out = mcp_errors_from_pool(_Pool())
    assert out == {"aaa-support": "HTTPStatusError: 401 Unauthorized"}, out
    assert mcp_errors_from_pool(None) is None
    assert mcp_errors_from_pool(object()) is None  # no attr → None
