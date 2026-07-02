"""In-process adapter for agent-federation (mirrors ``delegation.adapters``).

``build_runtime_toolkit`` registers the SAME ``ask_agent`` / ``list_agents``
tool implementations the standalone ``openagent-mcp`` server uses, bound to a
``PeerNetworksBackend`` — so an OpenAgent agent's federation surface is the
standalone client's surface. Concrete, typed wrappers (NOT ``lambda **kw``) so
the runtime builds a correct parameter schema.
"""

from __future__ import annotations

import json
from typing import Any

from openagent_mcp import tools as _tools

from .backend import PeerNetworksBackend


def _dump(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def build_runtime_toolkit() -> Any:
    from src.mcp._runtime.toolkit import Toolkit

    backend = PeerNetworksBackend()

    async def ask_agent(
        target: str,
        message: str,
        session_id: str | None = None,
        timeout_secs: int | None = None,
    ) -> str:
        """Send a message to a federated PEER OpenAgent agent and get its reply.

        Use this to retrieve context/information from, or delegate a task to,
        another agent this one is federated with (call list_agents for names).
        ``target`` is a peer name from list_agents. Omit ``session_id`` to start
        a NEW conversation — the returned ``session_id`` is the handle to
        continue it (pass it back to resume). The peer handles the message as a
        first-class conversation with its own tools and capabilities.
        """
        return _dump(await _tools.ask_agent(
            backend, target=target, message=message,
            session_id=session_id, timeout_secs=timeout_secs,
        ))

    async def list_agents() -> str:
        """List the federated peer OpenAgent agents reachable via ask_agent."""
        return _dump(await _tools.list_agents(backend))

    tk = Toolkit(name="agent-federation")
    tk.register(ask_agent)
    tk.register(list_agents)
    return tk
