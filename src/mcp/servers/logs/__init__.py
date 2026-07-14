"""In-process ``logs`` MCP — the agent's handle on its own event log.

Vision §14 says the unified log is queryable in three directions, and that
the third one is the agent itself: it "reads it as a tool to diagnose its own
behavior — answering questions like 'what went wrong yesterday?', 'why did
this scheduled task fail?', or 'which MCP call is slowing me down?'".

Every other domain already has an introspection MCP (``scheduler``,
``workflow-manager``, ``events-manager``, ``mcp-manager``, ``model-manager``);
the log did not. The only path the agent had was the one ``DREAM_MODE_PROMPT``
spells out in prose — shell out, ``find ~ -name events.jsonl``, ``tail -n
2000``, and reason over raw JSON lines. That burns a huge number of tokens on
unfiltered noise, hardcodes OS-specific paths into prompt text, and cannot
aggregate.

This MCP replaces that with three structured tools (``logs_query``,
``logs_summary``, ``logs_context``). It is **in-process on purpose**: the log
location comes from :func:`src.core.paths.log_dir`, which resolves against the
*live* agent directory set by ``set_agent_dir``. A subprocess MCP would
re-resolve it from platform defaults and silently read a *different* agent's
log (the same class of bug that forced ``OPENAGENT_DB_PATH`` injection for the
scheduler / model-manager subprocess MCPs — see ``resolve_default_entry``).

Read-only by design: see the module docstring in :mod:`.handlers`.
"""
