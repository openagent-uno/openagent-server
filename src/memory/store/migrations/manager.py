"""Schema-version stub for the SQLite session store.

The upstream codebase tracked schema versions across multiple database
backends and supported v1→v2 in-place migration. OpenAgent ships SQLite-only
and starts at the current schema, so all we need is a fixed version
returned to ``SqliteDb`` when it stamps the ``versions`` table during table
creation. No actual migration logic runs.
"""

from __future__ import annotations

from packaging import version as packaging_version
from packaging.version import Version


_CURRENT_SCHEMA_VERSION = packaging_version.parse("2.5.6")


class MigrationManager:
    """Inert version tracker. Records the current schema version per table."""

    def __init__(self, db):
        self.db = db

    @property
    def latest_schema_version(self) -> Version:
        return _CURRENT_SCHEMA_VERSION

    async def up(self, *_args, **_kwargs):  # pragma: no cover - no-op
        return None
