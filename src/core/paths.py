"""Cross-platform path resolution for OpenAgent config, data, and logs.

Follows platform conventions (XDG on Linux, Application Support on macOS,
%APPDATA% on Windows). Every function returns a :class:`Path` and ensures
the directory exists.

When an **agent directory** is set (via :func:`set_agent_dir`), all paths
are resolved relative to that directory instead of platform defaults. This
enables running multiple independent agents in parallel, each with its own
config, database, memories, and logs.

Precedence for config loading (handled by :func:`config.load_config`):

1. Explicit ``--config`` / ``-c`` CLI flag — highest priority.
2. ``<agent_dir>/openagent.yaml`` — if agent dir is set.
3. ``openagent.yaml`` in the current working directory.
4. ``<config_dir>/openagent.yaml`` — XDG/system default.

For data (DB, vault), the default is ``<data_dir>/`` unless overridden in
the YAML config via ``memory.db_path`` / ``memory.vault_path``.
"""

from __future__ import annotations

import os
import platform
import textwrap
from pathlib import Path

APP_NAME = "openagent"

# ── Agent directory singleton ──
# When set, all path functions return paths relative to this directory
# instead of platform-standard locations.

_agent_dir: Path | None = None


def set_agent_dir(path: Path | None) -> None:
    """Set the active agent directory. Pass ``None`` to reset to defaults."""
    global _agent_dir
    _agent_dir = path.resolve() if path is not None else None


def get_agent_dir() -> Path | None:
    """Return the active agent directory, or ``None`` if using defaults."""
    return _agent_dir


_DEFAULT_YAML = textwrap.dedent("""\
    # OpenAgent agent configuration
    # See https://github.com/geroale/OpenAgent for full reference.
    #
    # Providers, models, MCPs, and scheduled tasks are managed exclusively
    # through the SQLite database (configure them via the desktop app or
    # the /api/* REST endpoints). This file only holds server-level knobs.

    name: agent

    network:
      # Path to this agent's Iroh secret key (relative to the agent dir).
      # Generated automatically on first run. Keep at 0600 — leaking it
      # impersonates this agent on the entire network.
      identity_path: ./identity.key

      coordinator:
        # Path to the coordinator's signing key. Only used when this
        # agent has been promoted to network coordinator via
        # ``openagent network init``.
        key_path: ./coordinator.key

      # Optional: override Iroh's public DERP relay. Empty = use Iroh's
      # public network. Self-host one for full data locality.
      derp_url: ""

    channels:
      # Bridges (telegram, discord, whatsapp) connect as network clients
      # rather than via host:port. See `openagent network invite --role user`.
""")


def ensure_agent_dir(path: Path) -> Path:
    """Create an agent directory with default structure if it doesn't exist.

    Creates:
      <path>/openagent.yaml   (minimal config)
      <path>/memories/         (memory vault)
      <path>/logs/             (log files)

    Returns the resolved absolute path.
    """
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)

    config_file = path / "openagent.yaml"
    if not config_file.exists():
        config_file.write_text(_DEFAULT_YAML)

    (path / "memories").mkdir(exist_ok=True)
    (path / "logs").mkdir(exist_ok=True)

    return path


# ── Platform path helpers ──

def _system() -> str:
    return platform.system()  # "Darwin", "Linux", "Windows"


def _platform_dir(kind: str) -> Path:
    """Resolve the base config/data directory for the current platform."""
    if _agent_dir is not None:
        _agent_dir.mkdir(parents=True, exist_ok=True)
        return _agent_dir

    system = _system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Application Support" / "OpenAgent"
    elif system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")) / "OpenAgent"
    else:
        env_name = "XDG_CONFIG_HOME" if kind == "config" else "XDG_DATA_HOME"
        default = Path.home() / ".config" if kind == "config" else Path.home() / ".local" / "share"
        xdg = os.environ.get(env_name, str(default))
        base = Path(xdg) / APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def config_dir() -> Path:
    """Return the config directory, creating it if needed."""
    return _platform_dir("config")


def data_dir() -> Path:
    """Return the data directory, creating it if needed."""
    return _platform_dir("data")


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
