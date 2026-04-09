"""Scheduler MCP: expose OpenAgent's scheduled-task DB to the agent itself.

Runs as a subprocess MCP (stdio transport). Reads the DB path from the
OPENAGENT_DB_PATH env var (injected by the Agent at runtime), and writes
to the exact same `scheduled_tasks` table used by openagent.scheduler.
"""
