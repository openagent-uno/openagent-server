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
