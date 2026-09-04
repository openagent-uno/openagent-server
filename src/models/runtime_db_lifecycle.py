"""Release SQLite handles owned by cached runtime Agent/Team objects.

The model runtimes are cached for reuse, but each runtime may own a
``SqliteDb`` (and therefore a SQLAlchemy engine).  Dropping a cache entry
without disposing that database leaves pooled DB-API connections alive.  A
connection with a partially-consumed result keeps its WAL read snapshot and
can prevent checkpoints long after the turn that opened it has finished.

Keep this helper independent from the runner classes: both the single-model
and team adapters use it, and importing either runner here would introduce a
cycle.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def close_runtime_databases(runtime: Any) -> int:
    """Close every runtime-owned ``SqliteDb`` reachable from *runtime*.

    Team members can share the Team's database, so identities are deduplicated.
    The narrow ``db_engine`` + ``Session`` check deliberately avoids calling
    arbitrary model/provider ``close`` methods from a synchronous cache path.
    ``SqliteDb.close`` is idempotent and its disposed Engine remains reusable if
    a caller still holds the runtime briefly while an invalidation completes.
    """

    pending = [runtime]
    seen_objects: set[int] = set()
    seen_databases: set[int] = set()
    closed = 0

    while pending:
        current = pending.pop()
        if current is None or id(current) in seen_objects:
            continue
        seen_objects.add(id(current))

        db = getattr(current, "db", None)
        if (
            db is not None
            and id(db) not in seen_databases
            and hasattr(db, "db_engine")
            and hasattr(db, "Session")
        ):
            seen_databases.add(id(db))
            close = getattr(db, "close", None)
            if callable(close):
                try:
                    close()
                    closed += 1
                except Exception:
                    # Cache invalidation must never take the serving process
                    # down. A later WAL guard/restart remains the final backstop.
                    pass

        members = getattr(current, "members", None)
        if isinstance(members, Iterable) and not isinstance(
            members, (str, bytes, bytearray, dict)
        ):
            pending.extend(members)

    return closed
