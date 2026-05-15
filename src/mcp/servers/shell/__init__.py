"""In-process shell MCP for OpenAgent.

Exposes shell_exec / shell_output / shell_input / shell_kill / shell_list /
shell_which tools. Replaces the Node-based subprocess MCP (pre-0.7) with
a true in-process implementation that shares a ShellHub singleton with
the agent run loop, allowing terminal background-shell events to wake
the current session via _run_inner's auto-continuation loop.
"""
