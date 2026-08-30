"""In-process tools for this agent's user-defined name and persona prompt."""

from __future__ import annotations

from typing import Any

from src.core.agent_identity import AgentIdentityService
from src.core.on_behalf_context import current_on_behalf_identity


def build_runtime_toolkit(*, pool: Any) -> Any:
    """Build a principal-bound toolkit around the one live Agent runtime."""

    from src.mcp._runtime import Toolkit

    def service() -> AgentIdentityService:
        agent = getattr(pool, "agent_runtime", None)
        if agent is None:
            raise RuntimeError("agent-manager: live Agent runtime is not attached")
        return AgentIdentityService(
            agent=agent,
            db=getattr(agent, "_db", None) or getattr(pool, "_db", None),
            config_path=(getattr(agent, "config", {}) or {}).get("_config_path"),
            gateway=getattr(pool, "gateway_runtime", None),
        )

    async def agent_get_identity() -> dict[str, Any]:
        """Read this agent's display name and user-defined persona prompt.

        The returned ``system_prompt`` is the editable agent persona layered
        above OpenAgent. The immutable framework system prompt is never
        returned and cannot be changed by this MCP.
        """

        return await service().get(current_on_behalf_identity())

    async def agent_update_identity(
        name: str | None = None,
        system_prompt: str | None = None,
        expected_revision: str | None = None,
    ) -> dict[str, Any]:
        """Update this agent's display name and/or user-defined persona.

        Use only when the primary owner explicitly asks to rename the agent or
        revise how it should behave. Omitted fields remain unchanged;
        ``system_prompt=''`` clears the persona. The immutable OpenAgent
        framework prompt is not writable. Changes persist atomically and apply
        to the next agent turn without a restart. Pass ``expected_revision``
        from ``agent_get_identity`` when editing a previously-read profile.
        """

        result = await service().update(
            current_on_behalf_identity(),
            name=name,
            system_prompt=system_prompt,
            expected_revision=expected_revision,
        )
        # Do not duplicate a potentially sizeable private persona in tool
        # output/history. The read tool is the explicit surface for fetching
        # it; mutation returns only the public identity and concurrency data.
        return {
            key: value
            for key, value in result.items()
            if key != "system_prompt"
        }

    toolkit = Toolkit(
        name="agent-manager",
        tools=[agent_get_identity, agent_update_identity],
    )
    toolkit.async_functions["agent_get_identity"].classification = "read_only"
    return toolkit


__all__ = ["build_runtime_toolkit"]
