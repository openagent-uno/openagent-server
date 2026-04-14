"""Frozen-environment detection for PyInstaller bundles.

Provides helpers to detect whether OpenAgent is running from a frozen
executable (PyInstaller) and resolve paths accordingly.

Since v0.5.2 the server ships as a PyInstaller **onefile** build: the
bundle extracts to ``sys._MEIPASS`` on first launch and the executable
lives at ``sys.executable`` outside of that temp tree.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def is_frozen() -> bool:
    """Return True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def bundle_dir() -> Path:
    """Return the directory containing bundled data files.

    In onefile mode this is ``sys._MEIPASS`` (the freshly-extracted temp
    tree). In onedir mode — not what we ship any more, but kept for dev
    setups — it's the directory next to the executable.
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(sys.executable).resolve().parent


def executable_path() -> Path:
    """Return the path to the running executable on disk.

    Important distinction in onefile mode: ``sys.executable`` points at
    the on-disk binary (e.g. ``~/bin/openagent``), NOT into the
    ``sys._MEIPASS`` extract tree. That's what the self-updater needs to
    swap, so resolve here without touching ``_MEIPASS``.
    """
    return Path(sys.executable).resolve()


def swap_pending_if_any() -> bool:
    """On Windows, promote a staged ``*.pending.exe`` file into place.

    The self-updater can't overwrite a running executable on Windows, so
    ``apply_update`` drops the new binary next to the current one with a
    ``.pending.exe`` suffix. The OS service manager restarts us on exit 75;
    the very next launch calls this from the CLI entry point BEFORE
    importing the rest of the package, moves the pending file into place,
    and re-execs so the user ends up running the new code.

    Returns True if a swap was performed (caller should re-exec or exit).
    """
    if not is_frozen():
        return False
    if sys.platform != "win32":
        return False
    exe = executable_path()
    pending = exe.with_name(exe.stem + ".pending.exe")
    if not pending.exists():
        return False
    old = exe.with_suffix(exe.suffix + ".old")
    try:
        if old.exists():
            old.unlink()
        exe.rename(old)
        shutil.move(str(pending), str(exe))
        # Re-exec so the user ends up running the new binary immediately.
        os.execv(str(exe), [str(exe)] + sys.argv[1:])
    except Exception:
        # Swap failed — keep running the current binary so the user isn't
        # stuck in a broken state. The next update attempt will retry.
        return False
    return True
