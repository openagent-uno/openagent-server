"""Cross-platform path resolution for OpenAgent config, data, and logs.

Follows platform conventions (XDG on Linux, Application Support on macOS,
%APPDATA% on Windows). Every function returns a :class:`Path` and ensures
the directory exists.

Precedence for config loading (handled by :func:`config.load_config`):

1. Explicit ``--config`` / ``-c`` CLI flag — highest priority.
2. ``openagent.yaml`` in the current working directory.
3. ``<config_dir>/openagent.yaml`` — XDG/system default.

For data (DB, vault), the default is ``<data_dir>/`` unless overridden in
the YAML config via ``memory.db_path`` / ``memory.vault_path``.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

APP_NAME = "openagent"


def _system() -> str:
    return platform.system()  # "Darwin", "Linux", "Windows"


def config_dir() -> Path:
    """Return the platform-standard config directory, creating it if needed.

    - macOS:   ~/Library/Application Support/OpenAgent/
    - Linux:   ~/.config/openagent/
    - Windows: %APPDATA%\\OpenAgent\\
    """
    system = _system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "OpenAgent"
    elif system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "OpenAgent"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        base = Path(xdg) / APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def data_dir() -> Path:
    """Return the platform-standard data directory.

    - macOS:   ~/Library/Application Support/OpenAgent/
    - Linux:   ~/.local/share/openagent/
    - Windows: %APPDATA%\\OpenAgent\\
    """
    system = _system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "OpenAgent"
    elif system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "OpenAgent"
    else:
        xdg = os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
        base = Path(xdg) / APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def log_dir() -> Path:
    """Return the log directory (inside data_dir)."""
    d = data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def default_config_path() -> Path:
    """Return the default config file path inside the config directory."""
    return config_dir() / "openagent.yaml"


def default_db_path() -> Path:
    """Return the default SQLite database path."""
    return data_dir() / "openagent.db"


def default_vault_path() -> Path:
    """Return the default memory vault directory."""
    d = data_dir() / "memories"
    d.mkdir(parents=True, exist_ok=True)
    return d
