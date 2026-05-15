"""Shared helpers for the REST endpoint modules.

Keeping these in one place so every handler uses the same accessor and
the private-attribute path (``agent.memory_db``) stays a single-line
dependency that moves in lockstep if the Agent surface changes.
"""

from __future__ import annotations


def gateway_db(request):
    """Return the ``MemoryDB`` instance held by the running Agent, or ``None``."""
    return request.app["gateway"].agent.memory_db
