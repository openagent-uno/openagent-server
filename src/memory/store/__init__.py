"""Persistence-layer entry point.

OpenAgent ships SQLite as the sole supported backend (vision §17 —
self-hosted, no vendor lock-in). The lazy ``__getattr__`` that used to
fan out to Postgres / Mongo / Dynamo here has been removed; the
corresponding modules don't exist in this tree.
"""
from src.memory.store.base import BaseDb, SessionType

__all__ = [
    "BaseDb",
    "SessionType",
]
