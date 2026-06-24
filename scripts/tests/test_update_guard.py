"""Update boot-guard + rollback — the self-healing core.

These tests exercise the journal lifecycle that makes a bad release
recover with no remote access: record a pending update, count failed
boots, roll back to the previous binary after the threshold, and confirm
a healthy update so normal restarts are never mistaken for failures.

The guard is a no-op unless running frozen, so every test patches
``src._frozen.is_frozen`` → True and ``executable_path`` → a sandbox
binary it fully controls.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from ._framework import TestContext, test


def _bare_binary(tmp: Path) -> Path:
    """A bare on-disk binary with a ``.old`` rollback target, mirroring
    what apply_update leaves behind on Linux."""
    cur = tmp / "openagent"
    cur.write_bytes(b"NEW-BROKEN")
    cur.chmod(0o755)
    old = tmp / "openagent.old"
    old.write_bytes(b"OLD-GOOD")
    old.chmod(0o755)
    return cur


def _patches(target: Path):
    return (
        patch("src._frozen.is_frozen", return_value=True),
        patch("src._frozen.executable_path", return_value=target),
    )


@test("update_guard", "record_pending writes a pending journal next to the binary")
async def t_record_pending(ctx: TestContext) -> None:
    import json
    import tempfile
    import src.update_guard as g

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cur = _bare_binary(tmp)
        with _patches(cur)[0], _patches(cur)[1]:
            g.record_pending("1.2.3", prev_version="1.2.2")
        journal = tmp / "openagent.update-state.json"
        assert journal.exists()
        data = json.loads(journal.read_text())
        assert data["state"] == "pending"
        assert data["new_version"] == "1.2.3"
        assert data["boot_attempts"] == 0
        assert data["rollback_path"] == str(tmp / "openagent.old")


@test("update_guard", "mark_healthy confirms a pending update and resets the counter")
async def t_mark_healthy(ctx: TestContext) -> None:
    import json
    import tempfile
    import src.update_guard as g

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cur = _bare_binary(tmp)
        with _patches(cur)[0], _patches(cur)[1]:
            g.record_pending("1.2.3")
            # Simulate one shaky boot, then a healthy serving milestone.
            assert g.boot_guard() is None
            g.mark_healthy()
        data = json.loads((tmp / "openagent.update-state.json").read_text())
        assert data["state"] == "confirmed"
        assert data["confirmed_version"] == "1.2.3"
        assert data["boot_attempts"] == 0
        # The binary is untouched — a confirmed update is never rolled back.
        assert cur.read_bytes() == b"NEW-BROKEN"


@test("update_guard", "boot_guard rolls back to .old after MAX_BOOT_ATTEMPTS unhealthy boots")
async def t_boot_guard_rollback(ctx: TestContext) -> None:
    import json
    import tempfile
    import src.update_guard as g

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cur = _bare_binary(tmp)
        with _patches(cur)[0], _patches(cur)[1]:
            g.record_pending("9.9.9")
            results = []
            # Each boot that never reaches mark_healthy increments the
            # counter; the (MAX+1)-th returns "rolled_back".
            for _ in range(g.MAX_BOOT_ATTEMPTS + 1):
                results.append(g.boot_guard())

        assert results[:g.MAX_BOOT_ATTEMPTS] == [None] * g.MAX_BOOT_ATTEMPTS, results
        assert results[-1] == "rolled_back", results
        # The previous (good) binary is now the live one; .old consumed.
        assert cur.read_bytes() == b"OLD-GOOD", "rollback must restore the known-good binary"
        assert not (tmp / "openagent.old").exists(), ".old is consumed by the rename"
        assert not (tmp / "openagent.failed").exists(), "broken binary cleaned up"
        data = json.loads((tmp / "openagent.update-state.json").read_text())
        assert data["state"] == "rolled_back"
        assert "9.9.9" in data["rolled_back_versions"]


@test("update_guard", "after rollback the guard is a no-op (no re-counting the restored binary)")
async def t_boot_guard_noop_after_rollback(ctx: TestContext) -> None:
    import tempfile
    import src.update_guard as g

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cur = _bare_binary(tmp)
        with _patches(cur)[0], _patches(cur)[1]:
            g.record_pending("9.9.9")
            for _ in range(g.MAX_BOOT_ATTEMPTS + 1):
                g.boot_guard()
            # State is now rolled_back; subsequent boots of the restored
            # binary must NOT increment or roll anything.
            assert g.boot_guard() is None
            assert g.boot_guard() is None
        assert cur.read_bytes() == b"OLD-GOOD"


@test("update_guard", "rolled_back_versions exposes the recorded bad versions")
async def t_rolled_back_versions(ctx: TestContext) -> None:
    import tempfile
    import src.update_guard as g

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cur = _bare_binary(tmp)
        with _patches(cur)[0], _patches(cur)[1]:
            g.record_pending("9.9.9")
            for _ in range(g.MAX_BOOT_ATTEMPTS + 1):
                g.boot_guard()
            assert g.rolled_back_versions() == {"9.9.9"}


@test("update_guard", "boot_guard with no rollback target clears pending instead of looping")
async def t_boot_guard_no_rollback_target(ctx: TestContext) -> None:
    """If the .old is gone (can't roll back), the guard must not block
    boot forever — it clears pending and lets the supervisor keep the
    current binary alive rather than wedging the box."""
    import json
    import tempfile
    import src.update_guard as g

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cur = _bare_binary(tmp)
        (tmp / "openagent.old").unlink()  # no rollback target
        with _patches(cur)[0], _patches(cur)[1]:
            g.record_pending("9.9.9")
            last = None
            for _ in range(g.MAX_BOOT_ATTEMPTS + 1):
                last = g.boot_guard()
        assert last is None, "no rollback target → don't return rolled_back"
        assert cur.read_bytes() == b"NEW-BROKEN", "current binary left in place"
        data = json.loads((tmp / "openagent.update-state.json").read_text())
        assert data["state"] == "rollback_failed"


@test("update_guard", "boot_guard rolls back an .app bundle to .app.old")
async def t_boot_guard_bundle_rollback(ctx: TestContext) -> None:
    """macOS production layout: the binary lives inside ~/Applications/
    openagent.app and the rollback target is the sibling openagent.app.old
    bundle. Rollback must swap whole bundles, not the inner binary."""
    import tempfile
    import src.update_guard as g

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        apps = tmp / "Applications"
        # current (broken) bundle
        cur_bundle = apps / "openagent.app"
        (cur_bundle / "Contents" / "MacOS").mkdir(parents=True)
        inner = cur_bundle / "Contents" / "MacOS" / "openagent"
        inner.write_bytes(b"NEW-BROKEN")
        inner.chmod(0o755)
        # previous good bundle kept as .app.old
        old_bundle = apps / "openagent.app.old"
        (old_bundle / "Contents" / "MacOS").mkdir(parents=True)
        (old_bundle / "Contents" / "MacOS" / "openagent").write_bytes(b"OLD-GOOD")

        with _patches(inner)[0], _patches(inner)[1]:
            g.record_pending("9.9.9")
            results = [g.boot_guard() for _ in range(g.MAX_BOOT_ATTEMPTS + 1)]

        assert results[-1] == "rolled_back", results
        assert (cur_bundle / "Contents" / "MacOS" / "openagent").read_bytes() == b"OLD-GOOD"
        assert not old_bundle.exists(), ".app.old consumed by the rollback rename"


@test("update_guard", "guard is a no-op when not running frozen")
async def t_guard_noop_unfrozen(ctx: TestContext) -> None:
    import src.update_guard as g

    with patch("src._frozen.is_frozen", return_value=False):
        assert g.boot_guard() is None
        g.record_pending("1.2.3")  # must not raise
        g.mark_healthy()           # must not raise
        assert g.rolled_back_versions() == set()
