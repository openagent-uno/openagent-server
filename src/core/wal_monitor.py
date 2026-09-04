#!/usr/bin/env python3
"""Deterministic SQLite WAL guard for self-hosted OpenAgent containers.

The monitor never deletes a WAL. It confirms that a checkpoint's copied mark
is genuinely stationary, verifies that no recent durable automation is active,
persists a restart cooldown, and then terminates PID 1 so the container runtime
can release every SQLite snapshot and restart the complete process group.

It is stdlib-only because the runtime image executes it with system Python.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time
from dataclasses import dataclass
from typing import Callable
import urllib.request


DATA_DIR = Path(os.environ.get("OPENAGENT_DATA_DIR", "/data/agent"))
DB_PATH = DATA_DIR / "openagent.db"
STATE_PATH = DATA_DIR / "home" / ".resource-monitor-state.json"
LOG_PATH = DATA_DIR / "logs" / "wal-monitor.log"
ENV_PATH = DATA_DIR / ".env"


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


CHECK_INTERVAL_SECONDS = _env_int("OPENAGENT_WAL_MONITOR_INTERVAL_SECONDS", 900)
PIN_THRESHOLD_BYTES = _env_int("OPENAGENT_WAL_PIN_THRESHOLD_MB", 256) * 1024 * 1024
TRUNCATE_THRESHOLD_BYTES = _env_int("OPENAGENT_WAL_TRUNCATE_MB", 64) * 1024 * 1024
CONFIRM_SAMPLES = _env_int("OPENAGENT_WAL_PIN_CONFIRM_SAMPLES", 4, minimum=2)
CONFIRM_SLEEP_SECONDS = _env_int("OPENAGENT_WAL_PIN_CONFIRM_SLEEP_SECONDS", 3)
RESTART_COOLDOWN_SECONDS = _env_int("OPENAGENT_WAL_RESTART_COOLDOWN_SECONDS", 7200)
RECENT_WORK_SECONDS = _env_int("OPENAGENT_WAL_RECENT_WORK_SECONDS", 3600)


@dataclass(frozen=True)
class WalSample:
    busy: int
    frames: int
    copied: int

    @property
    def partial(self) -> bool:
        return self.frames > 0 and self.copied < self.frames


def logline(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] {message}"
    print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def _load_state() -> dict:
    try:
        value = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp = STATE_PATH.with_suffix(".tmp")
        temp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        temp.replace(STATE_PATH)
    except OSError as exc:
        raise RuntimeError("cannot persist WAL restart cooldown") from exc


def _agent_label() -> str:
    configured = os.environ.get("OPENAGENT_RESOURCE_LABEL", "").strip()
    if configured:
        return configured
    try:
        for raw in (DATA_DIR / "openagent.yaml").read_text(encoding="utf-8").splitlines():
            if raw.startswith("name:"):
                value = raw.split(":", 1)[1].strip().strip("'\"")
                if value:
                    return value
    except OSError:
        pass
    return "OpenAgent"


def _telegram_token() -> str:
    direct = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if direct:
        return direct
    try:
        for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if raw.startswith("TELEGRAM_BOT_TOKEN="):
                return raw.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def send_alert(message: str) -> None:
    recipient = os.environ.get("OPENAGENT_RESOURCE_ALERT_CHAT_ID", "").strip()
    token = _telegram_token()
    if not recipient or not token:
        logline("WAL alert not sent: configure OPENAGENT_RESOURCE_ALERT_CHAT_ID and TELEGRAM_BOT_TOKEN")
        return
    payload = json.dumps({"chat_id": recipient, "text": message}).encode()
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            logline(f"WAL alert sent (HTTP {response.status})")
    except Exception as exc:  # noqa: BLE001 - monitoring must keep running
        logline(f"WAL alert failed: {type(exc).__name__}")


def _checkpoint(conn: sqlite3.Connection) -> WalSample:
    row = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
    return WalSample(*(int(value) for value in row))


def sample_wal(
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[int, list[WalSample], bool]:
    """Return ``(wal_bytes, samples, pinned)`` without modifying live work.

    ``copied < frames`` is normal when a checkpoint races a writer. A real pin
    requires the copied mark to remain identical across every sample. If it
    advances, the reader frontier is moving and the WAL remains recyclable.
    """

    wal_path = Path(f"{DB_PATH}-wal")
    try:
        wal_bytes = wal_path.stat().st_size
    except OSError:
        return 0, [], False

    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    try:
        conn.execute("PRAGMA busy_timeout=10000")
        samples = [_checkpoint(conn)]
        while len(samples) < CONFIRM_SAMPLES and samples[-1].partial:
            sleep(CONFIRM_SLEEP_SECONDS)
            samples.append(_checkpoint(conn))
        pinned = (
            wal_bytes >= PIN_THRESHOLD_BYTES
            and len(samples) == CONFIRM_SAMPLES
            and all(sample.partial for sample in samples)
            and len({sample.copied for sample in samples}) == 1
        )
        if not pinned and wal_bytes >= TRUNCATE_THRESHOLD_BYTES and not samples[-1].partial:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        return wal_bytes, samples, pinned
    finally:
        conn.close()


def recent_active_work(now: float | None = None) -> list[str]:
    """List recent durable work; database uncertainty fails closed."""

    cutoff = (now if now is not None else time.time()) - RECENT_WORK_SECONDS
    checks = (
        (
            "task",
            "SELECT count(*) FROM task_runs WHERE status IN ('running','cancelling') AND started_at>=?",
        ),
        (
            "workflow",
            "SELECT count(*) FROM workflow_runs WHERE status IN ('running','cancelling') AND started_at>=?",
        ),
        (
            "delivery",
            "SELECT count(*) FROM event_deliveries WHERE status IN ('received','running') AND started_at>=?",
        ),
    )
    active: list[str] = []
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    except Exception as exc:  # noqa: BLE001
        return [f"db-check-error={type(exc).__name__}"]
    try:
        for label, sql in checks:
            try:
                count = int(conn.execute(sql, (cutoff,)).fetchone()[0])
            except sqlite3.OperationalError as exc:
                # A missing legacy table is safe; any other inability to prove
                # idleness defers the restart.
                if "no such table" in str(exc).lower():
                    continue
                return [f"db-check-error={type(exc).__name__}"]
            if count:
                active.append(f"{label}={count}")
        return active
    finally:
        conn.close()


def maybe_restart(*, now: float | None = None, apply: bool = True) -> bool:
    current = now if now is not None else time.time()
    state = _load_state()
    try:
        last_restart = float(state.get("wal_restart_ts", 0) or 0)
    except (TypeError, ValueError):
        last_restart = 0
    if current - last_restart < RESTART_COOLDOWN_SECONDS:
        logline("WAL self-heal deferred: restart cooldown is active")
        return False
    active = recent_active_work(current)
    if active:
        logline("WAL self-heal deferred: recent durable work: " + ", ".join(active))
        return False
    if not apply:
        logline("WAL self-heal dry-run: container restart not requested")
        return False

    state["wal_restart_ts"] = current
    state["wal_restart_reason"] = "confirmed stationary WAL reader mark"
    _save_state(state)
    logline("WAL self-heal: idle agent confirmed; terminating PID 1 for a clean container restart")
    os.kill(1, signal.SIGTERM)
    return True


def run_once(*, apply: bool = True) -> bool:
    if not DB_PATH.exists():
        logline(f"WAL monitor idle: database not found at {DB_PATH}")
        return False
    try:
        wal_bytes, samples, pinned = sample_wal()
    except Exception as exc:  # noqa: BLE001
        logline(f"WAL monitor check failed: {type(exc).__name__}")
        return False
    if not samples:
        return False
    final = samples[-1]
    logline(
        f"WAL {wal_bytes / (1024 * 1024):.1f} MiB; "
        f"checkpoint {final.copied}/{final.frames}; pinned={pinned}"
    )
    if not pinned:
        return False
    label = _agent_label()
    send_alert(
        f"🔴 {label} WAL PINNED\n"
        f"openagent.db-wal={wal_bytes / (1024 * 1024):.1f} MiB\n"
        f"checkpoint={final.copied}/{final.frames} (stationary across {len(samples)} samples)\n"
        "Action: automatic clean container restart when durable work is idle"
    )
    return maybe_restart(apply=apply)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    daemon = "--daemon" in args
    apply = "--dry-run" not in args
    if not daemon:
        run_once(apply=apply)
        return 0
    logline(f"WAL monitor daemon started (interval={CHECK_INTERVAL_SECONDS}s, apply={apply})")
    while True:
        try:
            run_once(apply=apply)
        except Exception as exc:  # noqa: BLE001
            logline(f"WAL monitor loop failed: {type(exc).__name__}")
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
