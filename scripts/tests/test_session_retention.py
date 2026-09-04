"""DB-retention daemon — import + prune behavior + image wiring.

The retention daemon (``src/core/session_retention.py``) keeps
``openagent.db`` bounded. It used to live ONLY on the pods' PVC and be
wired in by a ``[program:session-retention]`` block that only existed in
the PVC copy of supervisord.conf, so a fresh PVC silently disabled ALL
pruning. These tests pin that it is now durable:

  1. the module imports and its entry points are callable,
  2. ``run_once`` prunes a throwaway (mock) DB per the yaml `retention:`
     knobs — deletes the old window, keeps conversational (tg:/peer:)
     sessions, trims over-long transcripts, and applies the scheduler
     short-retention without nuking fresh scheduler rows, and
  3. the image's supervisord config (``deploy/supervisord.conf``)
     references the module so a FRESH pod runs it.

Pure-unit: a temp on-disk SQLite stands in for the DB (no real
/data/agent), no pool/gateway.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path

from ._framework import TestContext, test

# scripts/tests/test_session_retention.py → parents[2] == repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Minimal subset of the real ``sessions`` schema (src/memory/db.py) that the
# retention SQL touches: id, type, runs, and the two NOT NULL timestamps.
_SCHEMA = """
CREATE TABLE sessions (
    session_id   TEXT PRIMARY KEY,
    session_type TEXT,
    runs         TEXT,
    created_at   INTEGER NOT NULL,
    updated_at   INTEGER NOT NULL
);
"""


def _seed_db(db_path: str) -> None:
    now = int(time.time())
    old = now - 10 * 86_400  # 10 days ago → past the 3-day window
    runs5 = json.dumps([{"run_id": f"r{i}"} for i in range(1, 6)])
    rows = [
        # generic, past the window → deleted by step 1
        ("sess:old", "user", "[]", old, old),
        # conversational → EXEMPT, kept despite being old
        ("tg:keep", "user", "[]", old, old),
        ("peer:keep", "user", "[]", old, old),
        # recent but over-long transcript → kept, trimmed to last max_runs
        ("sess:recent", "user", runs5, now, now),
        # old scheduler run → deleted by the scheduler short-retention (2b)
        ("scheduler:old", "agent", "[]", old, old),
        # fresh scheduler run → survives (short-retention is age/size bounded)
        ("scheduler:recent", "agent", "[]", now, now),
    ]
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    conn.executemany(
        "INSERT INTO sessions "
        "(session_id, session_type, runs, created_at, updated_at) VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def _ids(db_path: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    ids = {r[0] for r in conn.execute("SELECT session_id FROM sessions").fetchall()}
    conn.close()
    return ids


@test("retention", "module imports and exposes callable entry points")
async def t_retention_importable(ctx: TestContext) -> None:
    import src.core.session_retention as sr

    for name in ("prune", "load_settings", "run_once", "main"):
        assert callable(getattr(sr, name, None)), f"missing/uncallable: {name}"
    assert sr.EXEMPT_PREFIXES == ("tg:", "peer:", "agent-brid"), sr.EXEMPT_PREFIXES
    assert set(sr.DEFAULTS) == {
        "days", "max_runs", "interval_hours", "trim_threshold_bytes"
    }, sr.DEFAULTS


@test("retention", "run_once prunes/keeps/trims a mock DB per the yaml knobs")
async def t_retention_run_once(ctx: TestContext) -> None:
    import src.core.session_retention as sr

    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "openagent.db")
        yaml_path = str(Path(tmp) / "openagent.yaml")
        _seed_db(db_path)
        # Custom knobs: tiny trim threshold so the 5-run transcript trips it.
        Path(yaml_path).write_text(
            "retention:\n"
            "  days: 3\n"
            "  max_runs: 2\n"
            "  interval_hours: 6\n"
            "  trim_threshold_bytes: 10\n"
        )

        settings = sr.run_once(db_path, yaml_path)

        # Knobs were parsed from the yaml, not the defaults.
        assert settings == {
            "days": 3, "max_runs": 2, "interval_hours": 6, "trim_threshold_bytes": 10
        }, settings

        remaining = _ids(db_path)
        # Old generic + old scheduler pruned; exempt + fresh rows kept.
        assert "sess:old" not in remaining, remaining
        assert "scheduler:old" not in remaining, remaining
        assert {"tg:keep", "peer:keep", "sess:recent", "scheduler:recent"} <= remaining, remaining

        # The over-long transcript was trimmed to the last max_runs (2) entries.
        conn = sqlite3.connect(db_path)
        raw = conn.execute(
            "SELECT runs FROM sessions WHERE session_id='sess:recent'"
        ).fetchone()[0]
        conn.close()
        run_ids = [r["run_id"] for r in json.loads(raw)]
        assert run_ids == ["r4", "r5"], run_ids


@test("retention", "image supervisord config wires the session-retention daemon")
async def t_retention_supervisord_wired(ctx: TestContext) -> None:
    conf = REPO_ROOT / "deploy" / "supervisord.conf"
    assert conf.exists(), f"missing image supervisord config: {conf}"
    text = conf.read_text()

    # The retention program must exist, invoke the repo module, and daemonize.
    assert "[program:session-retention]" in text, "no session-retention program"
    assert "session_retention.py" in text, "config does not reference the module"
    assert "--daemon" in text, "retention program is not run as a daemon"
    # Sanity: this is the real machine config (mirrors the image baseline),
    # not a stub — the agent program must still be present.
    assert "[program:openagent]" in text, "config is not the machine supervisord.conf"
    openagent_block = text.split("[program:openagent]", 1)[1].split("[program:", 1)[0]
    assert "stopasgroup=true" in openagent_block, "openagent stop can orphan its frozen child"
    assert "killasgroup=true" in openagent_block, "openagent kill can orphan its frozen child"

    runtime = (REPO_ROOT / "Dockerfile.openagent-runtime").read_text()
    assert "COPY deploy/supervisord.conf /etc/supervisord.conf" in runtime
    assert "COPY src/core/session_retention.py /opt/openagent/session_retention.py" in runtime
