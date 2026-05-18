"""Background maintenance for the learning tables.

Periodically prunes / archives stale entries so the bot's "memory" stays
relevant instead of growing forever:

  • ``skills``: rows whose ``last_used_at`` is older than
    ``skill_stale_days`` AND ``usage_count`` is zero get deleted —
    these are saves the matcher never confirmed as useful, so they were
    likely noise from the detector. Rows older than
    ``skill_archive_days`` (since ``last_used_at`` or ``updated_at``)
    are deleted regardless: the matcher would never surface them
    anyway, and keeping them inflates the matcher's candidate set.

  • ``user_profiles``: rows whose ``last_flushed_at`` is older than
    ``profile_archive_days`` get deleted. The user has likely abandoned
    that session_id; the next inbound message will recreate a fresh
    profile on first flush.

The curator runs as a single long-lived ``asyncio.Task`` started by
``core/server.AgentServer``. It sleeps ``interval_hours`` between
passes, so the steady-state CPU cost is one ``DELETE WHERE`` query per
hour — negligible next to the rest of the server.

Off by default. Enable via ``memory.curator.enabled: true`` in
``openagent.yaml``.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from src.core.logging import elog

_DAY_SECONDS = 86400.0

_DEFAULTS = {
    "interval_hours": 6,
    "skill_stale_days": 14,
    "skill_archive_days": 90,
    "profile_archive_days": 90,
    "session_retention_days": 90,
    # Backup knobs — separate cadence (daily, by default) from the
    # prune interval. ``backup_keep`` is "how many snapshots to retain
    # before deleting the oldest"; the resulting on-disk floor is
    # roughly ``backup_keep`` × the db size.
    "backup_interval_hours": 24,
    "backup_keep": 30,
}


def _is_enabled() -> bool:
    return (
        os.environ.get("OPENAGENT_CURATOR_ENABLED", "0").strip().lower()
        in ("1", "true", "yes", "on")
    )


def _int_env(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return v if v > 0 else default


async def run_once(db: Any) -> dict[str, int]:
    """Single curator pass. Returns the deletion counts so callers can
    log a summary; safe to call manually for testing.
    """
    conn = getattr(db, "_conn", None)
    if conn is None:
        return {"skills_unused_pruned": 0, "skills_archived": 0, "profiles_archived": 0}
    now = time.time()
    skill_stale_cutoff = now - _DAY_SECONDS * _int_env(
        "OPENAGENT_CURATOR_SKILL_STALE_DAYS", _DEFAULTS["skill_stale_days"]
    )
    skill_archive_cutoff = now - _DAY_SECONDS * _int_env(
        "OPENAGENT_CURATOR_SKILL_ARCHIVE_DAYS", _DEFAULTS["skill_archive_days"]
    )
    profile_archive_cutoff = now - _DAY_SECONDS * _int_env(
        "OPENAGENT_CURATOR_PROFILE_ARCHIVE_DAYS", _DEFAULTS["profile_archive_days"]
    )

    session_retention_cutoff = now - _DAY_SECONDS * _int_env(
        "OPENAGENT_CURATOR_SESSION_RETENTION_DAYS", _DEFAULTS["session_retention_days"]
    )

    counts = {
        "skills_unused_pruned": 0,
        "skills_archived": 0,
        "profiles_archived": 0,
        "sessions_pruned": 0,
    }

    try:
        # Prune unused skills past the stale cutoff. ``last_used_at``
        # being NULL falls under ``COALESCE(last_used_at, updated_at)``
        # so newly-created saves with no usage yet are also age-checked.
        cur = await conn.execute(
            "DELETE FROM skills "
            "WHERE usage_count <= 0 "
            "  AND COALESCE(last_used_at, updated_at) < ?",
            (skill_stale_cutoff,),
        )
        counts["skills_unused_pruned"] = cur.rowcount or 0
        await cur.close()

        # Archive (delete) any skill older than the archive cutoff
        # regardless of usage — the matcher would not surface it again
        # anyway after that horizon.
        cur = await conn.execute(
            "DELETE FROM skills "
            "WHERE COALESCE(last_used_at, updated_at) < ?",
            (skill_archive_cutoff,),
        )
        counts["skills_archived"] = cur.rowcount or 0
        await cur.close()

        # Prune profile rows that haven't been flushed in a long time.
        # ``last_flushed_at`` can be NULL when the row was bumped but
        # never reached the flush threshold — fall back to updated_at
        # so those rows still age out cleanly.
        cur = await conn.execute(
            "DELETE FROM user_profiles "
            "WHERE COALESCE(last_flushed_at, updated_at) < ?",
            (profile_archive_cutoff,),
        )
        counts["profiles_archived"] = cur.rowcount or 0
        await cur.close()

        # Prune dormant Agno sessions. ``updated_at`` is bumped on every
        # turn, so a row that hasn't moved in retention_days is a
        # session the user has clearly abandoned. ``agno_sessions``
        # stores ts as INTEGER epoch seconds (unlike the REAL we use
        # in our newer tables) — coerce the cutoff so the comparison
        # works against both shapes that might appear in a long-running
        # deployment.
        cur = await conn.execute(
            "DELETE FROM agno_sessions WHERE updated_at < ?",
            (int(session_retention_cutoff),),
        )
        counts["sessions_pruned"] = cur.rowcount or 0
        await cur.close()

        await conn.commit()
    except Exception as e:
        elog("curator.run_error", error=str(e)[:200])
        return counts

    elog(
        "curator.run",
        skills_unused_pruned=counts["skills_unused_pruned"],
        skills_archived=counts["skills_archived"],
        profiles_archived=counts["profiles_archived"],
        sessions_pruned=counts["sessions_pruned"],
    )
    return counts


async def _maybe_backup(db: Any) -> None:
    """Snapshot the SQLite DB to ``<agent_dir>/backups/`` when the last
    backup is older than ``backup_interval_hours``. Uses
    ``VACUUM INTO`` so the snapshot is atomic + WAL-consistent even
    while writes are in flight. Trim older backups beyond ``backup_keep``.

    No-op when the curator feature is off (the whole loop won't be
    running) or when the DB exposes no path we can copy.
    """
    interval_h = _int_env(
        "OPENAGENT_CURATOR_BACKUP_INTERVAL_HOURS",
        _DEFAULTS["backup_interval_hours"],
    )
    keep = _int_env("OPENAGENT_CURATOR_BACKUP_KEEP", _DEFAULTS["backup_keep"])
    db_path_str = getattr(db, "db_path", None) or os.environ.get("OPENAGENT_DB_PATH")
    if not db_path_str:
        return
    db_path = Path(db_path_str)
    backups_dir = db_path.parent / "backups"
    backups_dir.mkdir(exist_ok=True)

    # Find latest existing backup, decide whether we're due.
    existing = sorted(backups_dir.glob("openagent-*.db"))
    now = time.time()
    if existing:
        latest = existing[-1]
        try:
            age_s = now - latest.stat().st_mtime
        except OSError:
            age_s = float("inf")
        if age_s < interval_h * 3600:
            return  # last snapshot still fresh

    snapshot_path = backups_dir / time.strftime("openagent-%Y%m%d-%H%M.db")
    conn = getattr(db, "_conn", None)
    if conn is None:
        return
    try:
        # SQLite VACUUM INTO is the right tool: atomic, WAL-aware, no
        # checkpoint needed. ``?`` parametrisation doesn't work for
        # filenames in PRAGMA / VACUUM, so build the literal but
        # escape single quotes defensively even though our path is
        # under the agent_dir we own.
        target_sql = str(snapshot_path).replace("'", "''")
        await conn.execute(f"VACUUM INTO '{target_sql}'")
    except Exception as e:
        elog("curator.backup_error", error=str(e)[:200], target=str(snapshot_path))
        return

    # Retention: drop everything past the ``keep`` newest.
    existing = sorted(backups_dir.glob("openagent-*.db"))
    pruned = 0
    if len(existing) > keep:
        for p in existing[: len(existing) - keep]:
            try:
                p.unlink()
                pruned += 1
            except OSError:
                pass

    elog(
        "curator.backup",
        snapshot=str(snapshot_path),
        size_bytes=snapshot_path.stat().st_size if snapshot_path.exists() else 0,
        retained=min(len(existing), keep),
        pruned=pruned,
    )


async def _curator_loop(db: Any) -> None:
    """Long-lived task: run_once → sleep → repeat. Stops on cancellation
    (server shutdown). Exceptions inside ``run_once`` are caught so a
    one-off SQL failure doesn't kill the loop permanently.
    """
    while True:
        try:
            await run_once(db)
        except Exception as e:
            elog("curator.loop_error", error=str(e)[:200])
        try:
            await _maybe_backup(db)
        except Exception as e:
            elog("curator.backup_loop_error", error=str(e)[:200])
        interval_h = _int_env("OPENAGENT_CURATOR_INTERVAL_HOURS", _DEFAULTS["interval_hours"])
        try:
            await asyncio.sleep(max(60, interval_h * 3600))
        except asyncio.CancelledError:
            elog("curator.loop_cancelled")
            raise


def start(db: Any) -> asyncio.Task | None:
    """Schedule the curator on the current event loop. Returns the task
    so the caller can cancel it on shutdown. ``None`` when the feature
    is off — caller should treat that as success."""
    if not _is_enabled():
        return None
    elog("curator.start", interval_hours=_int_env("OPENAGENT_CURATOR_INTERVAL_HOURS", _DEFAULTS["interval_hours"]))
    return asyncio.create_task(_curator_loop(db), name="learning-curator")
