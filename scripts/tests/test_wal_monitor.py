"""WAL self-heal requires a stationary reader, idle work, and a cooldown."""

from __future__ import annotations

import sqlite3
import time

from ._framework import TestContext, test


@test("wal_monitor", "a moving checkpoint frontier is not a pinned WAL")
async def t_moving_frontier_is_healthy(ctx: TestContext) -> None:
    from src.core import wal_monitor

    samples = iter((
        wal_monitor.WalSample(0, 100, 10),
        wal_monitor.WalSample(0, 120, 20),
        wal_monitor.WalSample(0, 140, 30),
        wal_monitor.WalSample(0, 160, 40),
    ))
    original_checkpoint = wal_monitor._checkpoint
    original_path = wal_monitor.DB_PATH
    original_threshold = wal_monitor.PIN_THRESHOLD_BYTES
    original_samples = wal_monitor.CONFIRM_SAMPLES
    db_path = ctx.db_path.with_name("wal-monitor-moving.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (value INTEGER)")
    conn.commit()
    wal_path = db_path.with_name(db_path.name + "-wal")
    try:
        wal_monitor.DB_PATH = db_path
        wal_monitor.PIN_THRESHOLD_BYTES = 1
        wal_monitor.CONFIRM_SAMPLES = 4
        wal_monitor._checkpoint = lambda _conn: next(samples)
        wal_bytes, got, pinned = wal_monitor.sample_wal(sleep=lambda _seconds: None)
        assert wal_bytes >= 1
        assert len(got) == 4
        assert pinned is False
    finally:
        wal_monitor._checkpoint = original_checkpoint
        wal_monitor.DB_PATH = original_path
        wal_monitor.PIN_THRESHOLD_BYTES = original_threshold
        wal_monitor.CONFIRM_SAMPLES = original_samples
        conn.close()
        for path in (db_path, wal_path, db_path.with_name(db_path.name + "-shm")):
            try:
                path.unlink()
            except OSError:
                pass


@test("wal_monitor", "a stationary checkpoint frontier is pinned above the floor")
async def t_stationary_frontier_is_pinned(ctx: TestContext) -> None:
    from src.core import wal_monitor

    original_checkpoint = wal_monitor._checkpoint
    original_path = wal_monitor.DB_PATH
    original_threshold = wal_monitor.PIN_THRESHOLD_BYTES
    original_samples = wal_monitor.CONFIRM_SAMPLES
    db_path = ctx.db_path.with_name("wal-monitor-stationary.db")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (value INTEGER)")
    conn.commit()
    try:
        wal_monitor.DB_PATH = db_path
        wal_monitor.PIN_THRESHOLD_BYTES = 1
        wal_monitor.CONFIRM_SAMPLES = 4
        wal_monitor._checkpoint = lambda _conn: wal_monitor.WalSample(0, 200, 17)
        _wal_bytes, got, pinned = wal_monitor.sample_wal(sleep=lambda _seconds: None)
        assert len(got) == 4
        assert pinned is True
    finally:
        wal_monitor._checkpoint = original_checkpoint
        wal_monitor.DB_PATH = original_path
        wal_monitor.PIN_THRESHOLD_BYTES = original_threshold
        wal_monitor.CONFIRM_SAMPLES = original_samples
        conn.close()
        for suffix in ("", "-wal", "-shm"):
            try:
                db_path.with_name(db_path.name + suffix).unlink()
            except OSError:
                pass


@test("wal_monitor", "restart is deferred for active work and persisted cooldown")
async def t_restart_guards(ctx: TestContext) -> None:
    from src.core import wal_monitor

    original_active = wal_monitor.recent_active_work
    original_load = wal_monitor._load_state
    original_save = wal_monitor._save_state
    try:
        wal_monitor.recent_active_work = lambda _now: ["task=1"]
        wal_monitor._load_state = lambda: {}
        wal_monitor._save_state = lambda _state: None
        assert wal_monitor.maybe_restart(now=time.time(), apply=False) is False

        now = time.time()
        wal_monitor.recent_active_work = lambda _now: []
        wal_monitor._load_state = lambda: {"wal_restart_ts": now}
        assert wal_monitor.maybe_restart(now=now + 1, apply=False) is False
    finally:
        wal_monitor.recent_active_work = original_active
        wal_monitor._load_state = original_load
        wal_monitor._save_state = original_save
