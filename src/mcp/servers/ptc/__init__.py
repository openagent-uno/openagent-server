"""Programmatic Tool Calling (PTC) — the ``run_python`` tool.

The model writes a Python script that reaches the agent's OWN tools through a
local RPC bridge (``call_tool(server, tool, args)``), the script runs on the
sandbox exec backend, and only its stdout returns to the model. This collapses
a multi-tool pipeline (fan-out → filter → aggregate) into ONE model turn: the
per-call round-trips happen in code, not as separate model steps.

OFF BY DEFAULT (gated on ``ptc.enabled``). With the flag unset this package is
inert — the ``ptc`` builtin spec is never seeded (see ``config_gated_mcp_entries``),
``build_ptc_note`` renders "", and the system prompt / tool list / every code
path stay byte-identical to a build without this feature.

The bridge can only reach tools the agent already has: the RPC handler dispatches
through ``tool_search.adapters._call_tool_impl`` (the same entrypoint the
``tool_search_call_tool`` tool uses), so there is no privilege escalation and
dry-run stamping propagates for free.
"""
