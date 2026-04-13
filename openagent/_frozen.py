"""Frozen-environment detection for PyInstaller bundles.

Provides helpers to detect whether OpenAgent is running from a frozen
executable (PyInstaller) and resolve paths accordingly.
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """Return True when running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def bundle_dir() -> Path:
    """Return the directory containing bundled data files.

    In PyInstaller ``--onedir`` mode this is the directory containing the
    executable. In ``--onefile`` mode it would be the temporary
    ``sys._MEIPASS`` directory (we use ``--onedir``).
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(sys.executable).resolve().parent


def executable_path() -> Path:
    """Return the path to the running executable."""
    return Path(sys.executable).resolve()
