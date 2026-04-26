"""Provider adapters for the in-process ``tool-search`` MCP.

This MCP gives the model a uniform, provider-agnostic way to discover
and invoke any tool from any other connected MCP. It exists because
deployments with many MCPs accumulate hundreds of tools; LLM providers
cap how many tools they accept per request (OpenAI: 128, Claude Code
"standard" mode: ~200), so above-budget tools get silently dropped
from the upfront tool list. Without a recovery channel the model
plain doesn't see those tools — the symptom this MCP is here to fix.

The companion logic in ``openagent.models.runtime.wire_model_runtime``
trims above-budget MCPs from the upfront list and relies on the four
tools below (``list_servers`` / ``list_tools`` / ``describe_tool`` /
``call_tool``) to recover them on demand.

Both factories accept a ``pool`` kwarg so the adapter can navigate
the live ``MCPPool``. The pool is injected by ``MCPPool.connect_all``
when it detects via ``inspect.signature`` that the factory accepts
it; existing in-process adapters that don't take ``pool`` (e.g.
``shell``) keep working unchanged.
"""
from __future__ import annotations

import inspect
import json
from typing import Any


# ── Shared helpers (provider-agnostic implementation) ───────────────


def _coerce_to_jsonable(value: Any, _depth: int = 0) -> Any:
    """Best-effort JSON-friendly coercion. Mirrors the behaviour of
    ``openagent.workflow.executor._coerce_to_jsonable`` so results from
    ``call_tool`` look the same as results from an ``mcp-tool`` block."""
    if _depth > 10:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _coerce_to_jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_to_jsonable(v, _depth + 1) for v in value]
    if hasattr(value, "content"):
        return _coerce_to_jsonable(value.content, _depth + 1)
    return str(value)


def _functions_dict(toolkit: Any) -> dict[str, Any]:
    """Merged sync + async functions for an Agno toolkit / MCPTools.

    Subprocess MCPs populate ``functions``; in-process Toolkits with
    async tools populate ``async_functions``. We treat both as
    callable handles for ``call_tool``.
    """
    out = dict(getattr(toolkit, "functions", {}) or {})
    out.update(getattr(toolkit, "async_functions", {}) or {})
    return out


def _list_servers_impl(pool: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, toolkit in pool._toolkit_by_name.items():
        if name == "tool-search":
            continue  # never list ourselves — would be infinite-mirror noise
        out.append({"name": name, "tool_count": len(_functions_dict(toolkit))})
    out.sort(key=lambda x: x["name"])
    return out


def _list_tools_impl(pool: Any, server: str) -> list[dict[str, Any]]:
    toolkit = pool.toolkit_by_name(server)
    if toolkit is None:
        raise ValueError(
            f"MCP {server!r} is not loaded. Known MCPs: "
            f"{sorted(pool._toolkit_by_name)}"
        )
    out: list[dict[str, Any]] = []
    for tool_name, fn in _functions_dict(toolkit).items():
        # Compact 1-line description so list_tools fits a reasonable
        # token budget even on MCPs with 40+ tools (see firebase: 44).
        desc = (getattr(fn, "description", "") or "").strip()
        first_line = desc.split("\n", 1)[0][:200] if desc else ""
        out.append({"name": tool_name, "description": first_line})
    out.sort(key=lambda x: x["name"])
    return out


def _describe_tool_impl(pool: Any, server: str, tool: str) -> dict[str, Any]:
    toolkit = pool.toolkit_by_name(server)
    if toolkit is None:
        raise ValueError(f"MCP {server!r} is not loaded.")
    fn = _functions_dict(toolkit).get(tool)
    if fn is None:
        avail = sorted(_functions_dict(toolkit))
        raise ValueError(
            f"MCP {server!r} has no tool {tool!r}. Available: {avail}"
        )
    return {
        "name": tool,
        "description": getattr(fn, "description", "") or "",
        "input_schema": getattr(fn, "parameters", None) or {},
    }


async def _call_tool_impl(
    pool: Any, server: str, tool: str, args: dict | None,
) -> Any:
    toolkit = pool.toolkit_by_name(server)
    if toolkit is None:
        raise ValueError(
            f"MCP {server!r} is not loaded. Known MCPs: "
            f"{sorted(pool._toolkit_by_name)}"
        )
    fn = _functions_dict(toolkit).get(tool)
    if fn is None:
        avail = sorted(_functions_dict(toolkit))
        raise ValueError(
            f"MCP {server!r} has no tool {tool!r}. Available: {avail}"
        )
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ValueError(f"args must be a dict, got {type(args).__name__}")
    # Agno's ``Function`` exposes ``entrypoint``; raw callables don't.
    # Prefer ``entrypoint`` when present (matches the test fixtures in
    # ``scripts/tests/test_mcp.py``) and fall back to direct call for
    # plain functions.
    callable_to_call = getattr(fn, "entrypoint", None) or fn
    result = callable_to_call(**args)
    if inspect.isawaitable(result):
        result = await result
    return _coerce_to_jsonable(result)


def _json_dump(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)


# ── Claude Agent SDK adapter ────────────────────────────────────────


def build_sdk_server(*, pool: Any | None = None) -> Any:
    """Return a ``McpSdkServerConfig`` exposing the four tool-search tools.

    ``pool`` is required: the adapter has nothing useful to do without
    a live ``MCPPool`` to navigate.
    """
    from claude_agent_sdk import create_sdk_mcp_server, tool as sdk_tool

    if pool is None:
        raise RuntimeError("tool-search SDK adapter requires a pool kwarg")

    @sdk_tool(
        "list_servers",
        "List every connected MCP with its tool count. Start here to "
        "discover tools beyond the upfront tool list (the OpenAgent "
        "runtime trims MCPs above the provider's tool budget).",
        {"type": "object", "properties": {}},
    )
    async def _list_servers(args: dict) -> dict:
        return {"content": [{"type": "text", "text": _json_dump(
            _list_servers_impl(pool)
        )}]}

    @sdk_tool(
        "list_tools",
        "List the tools of a single MCP (name + 1-line description).",
        {
            "type": "object",
            "properties": {"server": {"type": "string"}},
            "required": ["server"],
        },
    )
    async def _list_tools(args: dict) -> dict:
        return {"content": [{"type": "text", "text": _json_dump(
            _list_tools_impl(pool, args["server"])
        )}]}

    @sdk_tool(
        "describe_tool",
        "Return the full description and JSON schema of a specific tool.",
        {
            "type": "object",
            "properties": {
                "server": {"type": "string"},
                "tool": {"type": "string"},
            },
            "required": ["server", "tool"],
        },
    )
    async def _describe_tool(args: dict) -> dict:
        return {"content": [{"type": "text", "text": _json_dump(
            _describe_tool_impl(pool, args["server"], args["tool"])
        )}]}

    @sdk_tool(
        "call_tool",
        "Invoke any tool on any connected MCP and return its result. Use "
        "this when the tool you need was trimmed from the upfront list.",
        {
            "type": "object",
            "properties": {
                "server": {"type": "string"},
                "tool": {"type": "string"},
                "args": {"type": "object", "additionalProperties": True},
            },
            "required": ["server", "tool"],
        },
    )
    async def _call_tool(args: dict) -> dict:
        result = await _call_tool_impl(
            pool, args["server"], args["tool"], args.get("args"),
        )
        return {"content": [{"type": "text", "text": _json_dump(result)}]}

    return create_sdk_mcp_server(
        "tool-search",
        tools=[_list_servers, _list_tools, _describe_tool, _call_tool],
    )


# ── Agno adapter ────────────────────────────────────────────────────


def build_agno_toolkit(*, pool: Any | None = None) -> Any:
    """Return an Agno ``Toolkit`` with the same four tools.

    The Agno function names mirror the convention used by subprocess
    MCPs: ``<sanitised-server-name>_<tool>``. The pool's
    ``_safe_prefix`` would normally do this for subprocess specs;
    in-process Toolkits skip that step (no ``tool_name_prefix``
    constructor arg), so we apply the prefix manually.
    """
    from agno.tools import Toolkit

    if pool is None:
        raise RuntimeError("tool-search Agno adapter requires a pool kwarg")

    async def tool_search_list_servers() -> list[dict[str, Any]]:
        """List every connected MCP with its tool count.

        Start here to discover tools beyond the upfront tool list — the
        OpenAgent runtime trims MCPs above the provider's tool budget.
        """
        return _list_servers_impl(pool)

    async def tool_search_list_tools(server: str) -> list[dict[str, Any]]:
        """List the tools of a single MCP (name + 1-line description)."""
        return _list_tools_impl(pool, server)

    async def tool_search_describe_tool(server: str, tool: str) -> dict[str, Any]:
        """Return the full description and JSON schema of a specific tool."""
        return _describe_tool_impl(pool, server, tool)

    async def tool_search_call_tool(
        server: str, tool: str, args: dict | None = None,
    ) -> Any:
        """Invoke any tool on any connected MCP and return its result.

        Use this when the tool you need was trimmed from the upfront list.
        """
        return await _call_tool_impl(pool, server, tool, args)

    return Toolkit(
        name="tool-search",
        tools=[
            tool_search_list_servers,
            tool_search_list_tools,
            tool_search_describe_tool,
            tool_search_call_tool,
        ],
    )
