#!/usr/bin/env python3
"""Session retention for OpenAgent's SQLite store — keeps openagent.db bounded.

This is the in-repo, version-controlled home of the DB-retention daemon. It
used to live ONLY on the pods' persistent PVC (``/data/agent/session_retention.py``)
and be wired in by a ``[program:session-retention]`` block that only existed in
the PVC copy of ``supervisord.conf`` — so a fresh PVC, pod migration, or config
reset silently disabled ALL pruning and the DB drifted back to the documented
~2 GB bloat + WAL-lock outage. Keeping the daemon here (and referencing it from
the image's ``deploy/supervisord.conf``) makes it durable.

NOTE: this is NOT ``src/learning/curator.py`` — that module is a dead/divergent
predecessor (90-day env-gated prune + ``VACUUM INTO`` backups, off by default,
no exempt prefixes, no run trimming, no scheduler short-retention). The behavior
below is the one that actually runs in production; the curator should be removed
in a later cleanup.

The module is deliberately stdlib-only (PyYAML is used only when present) so the
same file runs both:
  * imported as ``src.core.session_retention`` inside the frozen binary, and
  * as a standalone script under the pod's bare ``/usr/bin/python3`` — which is
    exactly how the supervisord ``session-retention`` program invokes it.

Without this the `sessions` table grows without limit: the scheduler writes a
new session per task run and the per-thread webhook binding keeps every
delivery's transcript, so the DB climbed to ~2GB and every model run reloaded
megabytes of history (the token/disk "history bloat").

Two knobs, both from openagent.yaml `retention:` (safe defaults if absent):
  * days      — delete sessions whose last activity is older than this
                (conversational sessions tg:/peer:/agent-brid are EXEMPT so
                the bot keeps its chat memory). Bounds infinite growth.
  * max_runs  — trim any session's `runs` transcript to the last N entries, so
                one chatty thread can't balloon *within* the window.

The binding stays per-thread (each thread keeps its own session) — this only
prunes OLD stuff, exactly the intended design.

Every write is BATCHED: a single 1.8GB UPDATE OOM-killed the process; 200-row
batches with per-batch commits stay flat in memory. Safe to run against the
live DB (WAL, one writer at a time).

Usage:
  session_retention.py                 # one-shot prune, then exit
  session_retention.py --daemon        # prune now, then every interval_hours
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time

DEFAULTS = {"days": 3, "max_runs": 5, "interval_hours": 6, "trim_threshold_bytes": 200_000}
# sessions whose id starts with one of these are conversational (Telegram, peer
# network, agent bridges) — keep them regardless of age.
EXEMPT_PREFIXES = ("tg:", "peer:", "agent-brid")


def _parse_retention_block(text: str) -> dict:
    """Minimal parser for the ``retention:`` block — no PyYAML needed, so the
    script runs under the pod's bare system python too. Reads the indented
    ``key: value`` lines until the block dedents."""
    out: dict = {}
    in_block = False
    for line in text.splitlines():
        if line.strip() == "retention:":
            in_block = True
            continue
        if in_block:
            if line.strip() and not line[0].isspace():  # dedent → block ended
                break
            body = line.split("#", 1)[0].strip()
            if ":" in body:
                key, val = body.split(":", 1)
                out[key.strip()] = val.strip()
    return out


def load_settings(yaml_path: str) -> dict:
    settings = dict(DEFAULTS)
    try:
        with open(yaml_path) as fh:
            text = fh.read()
    except FileNotFoundError:
        return settings
    try:
        import yaml  # PyYAML if present (agent venv)
        raw = (yaml.safe_load(text) or {}).get("retention") or {}
    except Exception:
        raw = _parse_retention_block(text)  # bare-python fallback
    for key in DEFAULTS:
        if key in raw:
            try:
                settings[key] = int(raw[key])
            except (TypeError, ValueError):
                pass
    return settings


def prune(db_path: str, days: int, max_runs: int, trim_threshold: int) -> tuple[int, int]:
    conn = sqlite3.connect(db_path, timeout=60)
    cur = conn.cursor()
    cur.execute("PRAGMA busy_timeout=60000")
    cutoff = f"strftime('%s','now','-{int(days)} days')"
    exempt = " AND ".join(f"session_id NOT LIKE '{p}%'" for p in EXEMPT_PREFIXES)

    # 1) delete sessions past the retention window — batched to bound memory.
    deleted = 0
    while True:
        cur.execute(
            "DELETE FROM sessions WHERE session_id IN ("
            f"  SELECT session_id FROM sessions WHERE updated_at < {cutoff} AND {exempt} LIMIT 200)"
        )
        batch = cur.rowcount
        conn.commit()
        deleted += batch
        if batch == 0:
            break

    # 2) trim over-long transcripts to the last `max_runs` runs — one at a time.
    cur.execute("SELECT session_id FROM sessions WHERE length(runs) > ?", (trim_threshold,))
    ids = [row[0] for row in cur.fetchall()]
    trimmed = 0
    for sid in ids:
        cur.execute("SELECT runs FROM sessions WHERE session_id=?", (sid,))
        raw = cur.fetchone()[0]
        try:
            runs = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(runs, list) and len(runs) > max_runs:
            cur.execute(
                "UPDATE sessions SET runs=? WHERE session_id=?",
                (json.dumps(runs[-max_runs:]), sid),
            )
            conn.commit()
            trimmed += 1

    # 2b) scheduler short-retention (added 2026-07-14 after the WAL-lock outage):
    # a completed cron run needs no transcript. Prune scheduler/agent run-logs
    # hard (1 day), and drop ANY oversized run-log immediately regardless of age
    # — a single 3-4MB `runs` blob monopolised the one WAL writer and starved
    # add_event_delivery (busy_timeout breach) → the support pipeline stalled.
    sched_cut = "strftime('%s','now','-1 days')"
    while True:
        cur.execute(
            "DELETE FROM sessions WHERE session_id IN ("
            "  SELECT session_id FROM sessions"
            "  WHERE (session_id LIKE 'scheduler:%' OR session_type='agent')"
            f"    AND (updated_at < {sched_cut} OR length(coalesce(runs,'')) > 500000)"
            f"    AND {exempt} LIMIT 200)"
        )
        b = cur.rowcount
        conn.commit()
        deleted += b
        if b == 0:
            break

    cur.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    # Reclaim free pages: delete/trim leave free pages that never shrink the file
    # without VACUUM (it sat at 2.7GB with 0.18GB of content). VACUUM takes an
    # exclusive lock, so only when free pages dominate; on bounded content it
    # finishes in seconds (< the agent's busy_timeout) so concurrent writes wait
    # rather than fail.
    try:
        free = cur.execute("PRAGMA freelist_count").fetchone()[0]
        total = cur.execute("PRAGMA page_count").fetchone()[0]
        if total and free / total > 0.25:
            conn.isolation_level = None
            cur.execute("VACUUM")
            print(f"[retention] VACUUM (freed {free}/{total} pages)", flush=True)
    except sqlite3.OperationalError as e:
        print(f"[retention] VACUUM skipped: {e}", flush=True)
    conn.close()
    return deleted, trimmed


def run_once(db_path: str, yaml_path: str) -> dict:
    settings = load_settings(yaml_path)
    deleted, trimmed = prune(
        db_path, settings["days"], settings["max_runs"], settings["trim_threshold_bytes"]
    )
    print(
        f"[retention] deleted={deleted} trimmed={trimmed} "
        f"days={settings['days']} max_runs={settings['max_runs']}",
        flush=True,
    )
    return settings


def main() -> None:
    ap = argparse.ArgumentParser(description="Prune OpenAgent session history.")
    ap.add_argument("--db", default="/data/agent/openagent.db")
    ap.add_argument("--config", default="/data/agent/openagent.yaml")
    ap.add_argument("--daemon", action="store_true", help="loop every interval_hours")
    args = ap.parse_args()

    if not args.daemon:
        run_once(args.db, args.config)
        return

    while True:
        settings = run_once(args.db, args.config)
        time.sleep(max(1, settings["interval_hours"]) * 3600)


if __name__ == "__main__":
    main()
