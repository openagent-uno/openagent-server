"""Bundled MCP servers shipped with OpenAgent.

Each subdirectory is a standalone MCP server project (Node or Python). They
are spawned as subprocesses via :mod:`openagent.mcp.builtins`. This package
exists primarily so the Python ones (``scheduler``) are importable; the Node
projects are referenced by directory only.
"""
