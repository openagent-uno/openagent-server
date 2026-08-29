"""Turn-scoped MCP catalog and dispatch backends.

The model-facing tool-search MCP must not know whether a tool is implemented
by the server's :class:`~src.mcp.pool.MCPPool` or by the authenticated client
that originated the current turn.  This module is that boundary.  Both
backends implement the same catalog and dispatch contracts, while routing is
still explicit through canonical ``server:`` / ``client:`` identifiers.

Client providers deliberately resolve :func:`current_execution_origin` for
every operation.  They therefore cannot turn a formerly interactive session
into a durable device binding, and a provider retained by a scheduler,
webhook, bridge, or detached workflow immediately becomes unavailable once
the trusted turn context is gone.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.core.execution_origin import (
    TurnExecutionOrigin,
    current_execution_origin,
)


SERVER_EXECUTION_HOST: dict[str, Any] = {
    "kind": "server",
    "device_label": "Server OpenAgent",
}


@runtime_checkable
class ToolCatalogProvider(Protocol):
    """A location-specific, per-turn view of MCP schemas.

    Server names passed to backend methods are bare names.  Canonical location
    prefixes are added by :meth:`list_servers` and selected by tool-search
    before calling the other methods.
    """

    location: str

    def list_servers(self) -> list[dict[str, Any]]:
        """List MCPs at this execution location using canonical IDs."""

    def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        """List the visible tools on one MCP at this location."""

    def describe_tool(
        self, server_name: str, tool_name: str,
    ) -> dict[str, Any]:
        """Return the schema for an exact tool on an exact MCP."""


@runtime_checkable
class ToolDispatcher(Protocol):
    """Dispatch one exact MCP call without location or device fallback."""

    location: str

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        args: dict[str, Any] | None,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute a tool and preserve the complete MCP result envelope."""


class ServerMCPProvider:
    """Catalog and dispatcher backend for the server's live ``MCPPool``."""

    location = "server"

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    def list_servers(self) -> list[dict[str, Any]]:
        # Lazy imports keep the legacy helpers in their stable module (PTC and
        # existing integrations import them directly) without introducing an
        # import cycle while tool-search itself imports this provider.
        from src.mcp.servers.tool_search.adapters import _list_servers_impl

        return [
            {
                **item,
                "name": f"server:{item['name']}",
                "execution_host": dict(SERVER_EXECUTION_HOST),
            }
            for item in _list_servers_impl(self.pool)
        ]

    def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        from src.mcp.servers.tool_search.adapters import _list_tools_impl

        return [
            {**item, "execution_host": dict(SERVER_EXECUTION_HOST)}
            for item in _list_tools_impl(self.pool, server_name)
        ]

    def describe_tool(
        self, server_name: str, tool_name: str,
    ) -> dict[str, Any]:
        from src.mcp.servers.tool_search.adapters import _describe_tool_impl

        return {
            **_describe_tool_impl(self.pool, server_name, tool_name),
            "execution_host": dict(SERVER_EXECUTION_HOST),
        }

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        args: dict[str, Any] | None,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        del session_id  # Server tools do not bind to an interactive client.
        from src.mcp.servers.tool_search.adapters import (
            _call_tool_impl,
            _stamp_execution_host,
        )

        result = await _call_tool_impl(self.pool, server_name, tool_name, args)
        return _stamp_execution_host(result, dict(SERVER_EXECUTION_HOST))


class InteractiveClientMCPProvider:
    """Catalog and dispatcher backend for the current authenticated client.

    The registry is Gateway-local.  The exact device, client instance and
    generation are *only* read from the trusted ``TurnExecutionOrigin`` and
    must refer back to this same registry.  There is intentionally no API for
    selecting another live host.
    """

    location = "client"

    def __init__(self, registry: Any) -> None:
        self.registry = registry

    def _origin(self) -> TurnExecutionOrigin:
        origin = current_execution_origin()
        if origin is None:
            raise PermissionError(
                "Client MCPs are unavailable: this is a server-owned turn or "
                "the originating client did not advertise local capabilities."
            )
        if origin.registry is not self.registry:
            # A provider from another Gateway can never be rebound merely
            # because the device/instance strings happen to match.
            raise PermissionError(
                "Client MCPs are unavailable for this Gateway execution origin."
            )
        return origin

    def list_servers(self) -> list[dict[str, Any]]:
        origin = self._origin()
        return self.registry.list_servers(origin)

    def list_tools(self, server_name: str) -> list[dict[str, Any]]:
        origin = self._origin()
        return self.registry.list_tools(origin, server_name)

    def describe_tool(
        self, server_name: str, tool_name: str,
    ) -> dict[str, Any]:
        origin = self._origin()
        return self.registry.describe_tool(origin, server_name, tool_name)

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        args: dict[str, Any] | None,
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        origin = self._origin()
        result = await self.registry.call_tool(
            origin,
            server_name,
            tool_name,
            args,
            session_id=session_id,
        )
        # Current registries stamp this themselves.  Keep the normalisation at
        # this abstraction boundary for compatibility with older/injected
        # registries while retaining every MCP envelope field.
        if isinstance(result, dict) and isinstance(
            result.get("execution_host"), dict,
        ):
            return result
        from src.mcp.servers.tool_search.adapters import _stamp_execution_host

        return _stamp_execution_host(result, origin.execution_host)


__all__ = [
    "InteractiveClientMCPProvider",
    "SERVER_EXECUTION_HOST",
    "ServerMCPProvider",
    "ToolCatalogProvider",
    "ToolDispatcher",
]
