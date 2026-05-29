"""Provider adapters for the in-process delegation MCP.

Same pattern as the shell / tool-search / attachments servers: handlers
are framework-agnostic; here we wrap them once per provider with that
provider's native tool decorator. Claude Agent SDK gets an
``McpSdkServerConfig``; the native runtime gets a ``Toolkit``.
"""

from __future__ import annotations

import json
from typing import Any

from src.mcp.servers.delegation import handlers


def _json_dump(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)


# ── Claude Agent SDK ────────────────────────────────────────────────


def build_sdk_server() -> Any:
    """Return an SDK MCP server exposing ``delegate_task`` + ``list_delegatable_models``."""
    from claude_agent_sdk import create_sdk_mcp_server, tool as sdk_tool

    @sdk_tool(
        "delegate_task",
        (
            "Hand a self-contained task to another registered model and "
            "return its final answer. Use this when another model is "
            "better suited for the work (cheaper, faster, vision-capable, "
            "or has a more relevant scope). Call list_delegatable_models "
            "first to see what's available. The sub-agent runs with the "
            "same session memory and the same MCP tools you have."
        ),
        {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": (
                        "The runtime id of the target model, e.g. "
                        "'openai:gpt-4o', 'anthropic:claude-3-5-sonnet-20240620'."
                    ),
                },
                "task": {
                    "type": "string",
                    "description": (
                        "The complete prompt to send the delegated model. "
                        "Be specific — the sub-agent does not see your "
                        "running thoughts, only this string."
                    ),
                },
            },
            "required": ["model_id", "task"],
        },
    )
    async def _sdk_delegate_task(args: dict) -> dict:
        result = await handlers.delegate_task(
            model_id=args["model_id"],
            task=args["task"],
        )
        return {
            "content": [
                {"type": "text", "text": _json_dump(result)},
            ],
        }

    @sdk_tool(
        "list_delegatable_models",
        (
            "List every model registered in this OpenAgent that you can "
            "delegate to. Returns the runtime id, display name, and "
            "framework of each."
        ),
        {"type": "object", "properties": {}},
    )
    async def _sdk_list_models(_args: dict) -> dict:
        result = await handlers.list_delegatable_models()
        return {
            "content": [
                {"type": "text", "text": _json_dump(result)},
            ],
        }

    return create_sdk_mcp_server(
        name="delegation",
        version="1.0.0",
        tools=[_sdk_delegate_task, _sdk_list_models],
    )


# ── Native runtime (Toolkit) ────────────────────────────────────────


def build_runtime_toolkit() -> Any:
    """Return a Toolkit exposing the two delegation tools to api-based agents."""
    from src.mcp._runtime.toolkit import Toolkit

    async def delegate_task(model_id: str, task: str) -> str:
        result = await handlers.delegate_task(model_id=model_id, task=task)
        return _json_dump(result)

    async def list_delegatable_models() -> str:
        result = await handlers.list_delegatable_models()
        return _json_dump(result)

    tk = Toolkit(name="delegation")
    tk.register(delegate_task)
    tk.register(list_delegatable_models)
    return tk
