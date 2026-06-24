"""Update state journal + boot guard + automatic rollback.

Why this exists
---------------
The self-updater swaps the on-disk binary and exits 75; the service
manager (launchd ``KeepAlive`` / systemd ``Restart=always``) then
relaunches the NEW binary. If that new binary is broken — a bad build,
an incompatible config migration, a crash-on-boot regression — the
supervisor restarts it forever (launchd) or eventually gives up and
leaves the unit dead (systemd ``StartLimitBurst``). Either way the
agent is bricked, and the operator typically **cannot reach the box**
to recover it. For OpenAgent that is a total, unrecoverable loss.

This module makes a bad release self-heal with zero remote access:

  ``updater.apply_update`` keeps the previous binary as ``<target>.old``
  (bare binary) or ``<bundle>.app.old`` (macOS .app) and
  ``record_pending`` writes a tiny JSON journal NEXT TO THE BINARY. On
  every ``serve`` start :func:`boot_guard` increments the pending
  update's boot-attempt counter BEFORE the heavy startup work. When the
  server actually reaches the serving state it calls
  :func:`mark_healthy`, which marks the update confirmed and resets the
  counter. A binary that crash-loops never reaches ``mark_healthy``;
  after :data:`MAX_BOOT_ATTEMPTS` boots the guard restores the ``.old``
  binary over the live one and exits so the supervisor brings the
  known-good version back up. The rolled-back version is remembered so
  the nightly auto-update never re-installs the same brick.

Design constraints
------------------
* **Stdlib only.** The guard runs at the very start of ``serve``,
  before the rest of the package is known to be healthy, and must never
  itself become the thing that breaks boot.
* **Anchored to the binary, not the agent dir.** Multiple tenants can
  run the same on-disk binary (e.g. several services sharing
  ``~/.local/bin/openagent``); the journal and the ``.old`` rollback
  target both live next to the binary so they are shared correctly.
* **Atomic + locked.** Every write is ``tmp + os.replace`` and
  serialised by the same cross-process flock the swap uses, so
  concurrent boots can't corrupt the journal.
* **Best-effort, fail-open.** Any internal error logs and lets boot
  proceed. The guard can only ever *help*; a bug in it must never be a
  new way to brick the box.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# How many consecutive boots a pending update may take WITHOUT reaching
# the healthy/serving milestone before the guard rolls back to the
# previous binary. 3 gives a genuinely-broken release a couple of
# retries (transient DB lock, slow network on first boot) while still
# self-healing within a minute or so under a 5 s restart throttle.
MAX_BOOT_ATTEMPTS = 3

# Journal schema version — bump if the on-disk shape changes so an old
# guard reading a newer journal can bail out safely.
_SCHEMA = 1

STATE_PENDING = "pending"
STATE_CONFIRMED = "confirmed"
STATE_ROLLED_BACK = "rolled_back"
STATE_ROLLBACK_FAILED = "rollback_failed"

# Keep the last N rolled-back versions so a daily auto-update never
# re-downloads and re-applies a version we already proved is broken.
_MAX_REMEMBERED_BAD = 10


# ── path anchoring ────────────────────────────────────────────────────


def _executable_target() -> Path | None:
    """Return the on-disk executable to guard, or None when not frozen.

    Self-update / rollback only applies to the PyInstaller build. A pip
    / source checkout has no ``.old`` binary to fall back to, so the
    guard is a no-op there.
    """
    try:
        from src._frozen import is_frozen, executable_path
        if not is_frozen():
            return None
        return executable_path()
    except Exception:  # noqa: BLE001 — never let path resolution break boot
        return None


def _find_app_bundle(path: Path) -> Path | None:
    """Walk up from *path* to the enclosing ``.app`` bundle, if any.

    Mirrors :func:`updater._find_app_bundle` but kept local so the guard
    has no import dependency on the heavier updater module at boot.
    """
    p = path.parent
    while p != p.parent:
        if p.suffix == ".app" and p.is_dir():
            return p
        p = p.parent
    return None


def _anchor(target: Path) -> Path:
    """Return the swap anchor for *target*: the ``.app`` bundle when the
    binary lives inside one, else the bare binary path. Sidecar files
    (``.old``, ``.update-state.json``, ``.swap-lock``) hang off this."""
    bundle = _find_app_bundle(target)
    return bundle if bundle is not None else target


def _journal_path(target: Path) -> Path:
    return _anchor(target).with_name(_anchor(target).name + ".update-state.json")


def _rollback_path(target: Path) -> Path:
    """The ``.old`` last-known-good kept beside the binary/bundle.

    For a bundle ``foo.app`` this is ``foo.app.old``; for a bare binary
    ``openagent`` it is ``openagent.old`` — matching exactly what
    :func:`updater.apply_update` writes.
    """
    anchor = _anchor(target)
    if anchor.suffix == ".app":
        return anchor.with_name(anchor.stem + ".app.old")
    return anchor.with_name(anchor.name + ".old")


# ── locked + atomic journal IO ────────────────────────────────────────


class _Lock:
    """Best-effort exclusive flock on ``<anchor>.swap-lock``.

    Shares the lock name with :func:`updater._swap_lock` so the boot
    guard and an in-flight ``apply_update`` on a sibling service can
    never interleave a half-written journal with a half-done swap.
    No-op on platforms without ``fcntl`` (Windows), which use a
    side-by-side ``.pending.exe`` and don't need it.
    """

    def __init__(self, target: Path):
        self._path = _anchor(target).with_name(_anchor(target).name + ".swap-lock")
        self._fd = None

    def __enter__(self):
        try:
            import fcntl
        except ImportError:
            return self
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fd = open(self._path, "w")
            fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX)
        except Exception:  # noqa: BLE001 — lock is advisory; proceed unlocked
            self._fd = None
        return self

    def __exit__(self, *_exc):
        if self._fd is not None:
            try:
                import fcntl
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
            except Exception:  # noqa: BLE001
                pass
            try:
                self._fd.close()
            except Exception:  # noqa: BLE001
                pass
        return False


def _read_journal(target: Path) -> dict | None:
    path = _journal_path(target)
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            return None
        return data
    except Exception as e:  # noqa: BLE001 — corrupt/partial journal is non-fatal
        logger.warning("update journal unreadable (%s): %s", path, e)
        return None


def _write_journal(target: Path, data: dict) -> None:
    path = _journal_path(target)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic
    except Exception as e:  # noqa: BLE001
        logger.warning("could not write update journal (%s): %s", path, e)
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:  # noqa: BLE001
            pass


# ── public API ────────────────────────────────────────────────────────


def record_pending(new_version: str, prev_version: str | None = None) -> None:
    """Record that *new_version* was just swapped in and is unconfirmed.

    Called by :func:`updater.perform_self_update_sync` immediately after
    a successful swap, BEFORE the process exits 75. The ``.old`` /
    ``.app.old`` rollback target must already exist (apply_update just
    created it).

    No-op when not running frozen (pip/source installs don't swap a
    binary and have nothing to roll back to).
    """
    target = _executable_target()
    if target is None:
        return
    rb = _rollback_path(target)
    try:
        with _Lock(target):
            existing = _read_journal(target) or {}
            bad = list(existing.get("rolled_back_versions", []))
            _write_journal(target, {
                "schema": _SCHEMA,
                "state": STATE_PENDING,
                "new_version": str(new_version),
                "prev_version": str(prev_version) if prev_version else existing.get("confirmed_version"),
                "applied_at": time.time(),
                "boot_attempts": 0,
                "rollback_path": str(rb),
                "rollback_available": rb.exists(),
                "rolled_back_versions": bad,
            })
        logger.info("update journal: pending %s (rollback=%s)", new_version, rb)
    except Exception as e:  # noqa: BLE001
        logger.warning("record_pending failed: %s", e)


def rolled_back_versions() -> set[str]:
    """Versions previously rolled back on this binary — never re-install.

    Lets :func:`updater.check_for_update` skip a release we already
    proved is broken so the nightly poller doesn't re-brick-and-rollback
    on a daily loop.
    """
    target = _executable_target()
    if target is None:
        return set()
    data = _read_journal(target) or {}
    try:
        return {str(v) for v in data.get("rolled_back_versions", [])}
    except Exception:  # noqa: BLE001
        return set()


def boot_guard() -> str | None:
    """Run at the very start of ``serve``. Returns:

    * ``None`` — proceed booting (no pending update, not frozen, or the
      pending update still has boot attempts left).
    * ``"rolled_back"`` — the guard restored the previous binary; the
      caller MUST exit with the restart code so the supervisor relaunches
      the known-good version.

    Counting happens BEFORE the heavy startup work so even a crash that
    happens during ``start()`` still increments the attempt counter and
    eventually triggers rollback.
    """
    target = _executable_target()
    if target is None:
        return None
    try:
        with _Lock(target):
            data = _read_journal(target)
            if not data or data.get("state") != STATE_PENDING:
                return None  # nothing pending → normal boot

            attempts = int(data.get("boot_attempts", 0)) + 1
            data["boot_attempts"] = attempts
            new_version = data.get("new_version", "?")

            if attempts <= MAX_BOOT_ATTEMPTS:
                _write_journal(target, data)
                logger.warning(
                    "update %s unconfirmed: boot attempt %d/%d",
                    new_version, attempts, MAX_BOOT_ATTEMPTS,
                )
                _elog("update.boot_attempt", level="warning",
                      version=new_version, attempt=attempts, max=MAX_BOOT_ATTEMPTS)
                return None

            # Exhausted: the new binary never reached a healthy serving
            # state. Restore the previous binary and let the supervisor
            # bring it back up.
            rb = data.get("rollback_path")
            ok = _perform_rollback(target, Path(rb) if rb else None)
            bad = list(data.get("rolled_back_versions", []))
            if new_version not in bad:
                bad.append(new_version)
            bad = bad[-_MAX_REMEMBERED_BAD:]
            data["rolled_back_versions"] = bad
            data["state"] = STATE_ROLLED_BACK if ok else STATE_ROLLBACK_FAILED
            data["rolled_back_at"] = time.time()
            _write_journal(target, data)

            if ok:
                logger.error(
                    "update %s failed %d boots — rolled back to previous binary",
                    new_version, MAX_BOOT_ATTEMPTS,
                )
                _elog("update.rolled_back", level="error",
                      version=new_version, attempts=MAX_BOOT_ATTEMPTS)
                return "rolled_back"

            # Could not roll back (no .old). Don't keep counting forever —
            # clear pending so the current binary at least keeps trying to
            # run rather than being stuck. Better a crash-looping new
            # binary the supervisor keeps reviving than a guard that
            # blocks boot.
            logger.error(
                "update %s failed but NO rollback target available — "
                "clearing pending and continuing on current binary",
                new_version,
            )
            _elog("update.rollback_unavailable", level="error", version=new_version)
            return None
    except Exception as e:  # noqa: BLE001 — guard must never block boot
        logger.warning("boot_guard error (continuing boot): %s", e)
        return None


def mark_healthy() -> None:
    """Confirm the pending update once the server has reached serving.

    Resets the boot-attempt counter and flips the journal to
    ``confirmed`` so subsequent normal restarts (config reloads, manual
    restarts, reboots) are never mistaken for a failing update. A
    crash-looping binary never gets here, which is exactly what lets the
    guard roll it back.

    Idempotent and best-effort. No-op when not frozen or nothing pending.
    """
    target = _executable_target()
    if target is None:
        return
    try:
        with _Lock(target):
            data = _read_journal(target)
            if not data or data.get("state") != STATE_PENDING:
                return
            new_version = data.get("new_version", "?")
            data["state"] = STATE_CONFIRMED
            data["confirmed_version"] = new_version
            data["confirmed_at"] = time.time()
            data["boot_attempts"] = 0
            _write_journal(target, data)
        logger.info("update %s confirmed healthy", new_version)
        _elog("update.confirmed", version=new_version)
    except Exception as e:  # noqa: BLE001
        logger.warning("mark_healthy failed: %s", e)


# ── rollback mechanics ────────────────────────────────────────────────


def _perform_rollback(target: Path, rollback_path: Path | None) -> bool:
    """Restore *rollback_path* (the ``.old``) over the live binary/bundle.

    Uses same-directory ``os.rename`` so the swap is atomic and needs no
    extra free disk (the broken current is renamed aside, the known-good
    ``.old`` is renamed into place). Returns True on success.

    Caller holds the swap lock.
    """
    if rollback_path is None or not rollback_path.exists():
        return False
    anchor = _anchor(target)
    failed = anchor.with_name(anchor.name + ".failed")
    try:
        # Clear any stale .failed from a previous rollback.
        if failed.exists():
            if failed.is_dir():
                shutil.rmtree(failed, ignore_errors=True)
            else:
                try:
                    failed.unlink()
                except Exception:  # noqa: BLE001
                    pass
        # Move the broken current aside, then the known-good into place.
        # Both renames are within the same parent dir → atomic.
        if anchor.exists():
            os.rename(anchor, failed)
        os.rename(rollback_path, anchor)
        # Reclaim the broken binary's disk; non-fatal if it lingers.
        try:
            if failed.is_dir():
                shutil.rmtree(failed, ignore_errors=True)
            elif failed.exists():
                failed.unlink()
        except Exception:  # noqa: BLE001
            pass
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("rollback rename failed: %s", e)
        # Best-effort: if we moved current aside but failed to put .old
        # in place, try to put current back so the supervisor still has
        # *something* to launch.
        try:
            if not anchor.exists() and failed.exists():
                os.rename(failed, anchor)
        except Exception:  # noqa: BLE001
            pass
        return False


def _elog(event: str, level: str = "info", **data) -> None:
    """Best-effort structured event; never raises (logging may be
    unwired this early in boot)."""
    try:
        from src.core.logging import elog
        elog(event, level=level, **data)
    except Exception:  # noqa: BLE001
        pass
