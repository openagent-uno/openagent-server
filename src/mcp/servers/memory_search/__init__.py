"""Memory-search MCP — full-text recall across past conversations.

Wraps :mod:`src.memory.transcript_index` as a subprocess MCP so the agent can
find what was said in any stored session (§16) — the counterpart to the vault,
which holds what it deliberately learned (§5).

Subprocess rather than in-process on purpose: keeping the index sync out of
the gateway process means a cold rebuild can never stall a live voice stream.
It reads the DB that ``OPENAGENT_DB_PATH`` points at, which is injected at
spawn, and derives the cache location from that same path so it can never
answer out of a different agent's history.
"""
