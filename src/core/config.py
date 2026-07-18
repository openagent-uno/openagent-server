"""Configuration loader for OpenAgent. Supports YAML config with env var substitution."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_FILE = "openagent.yaml"


def _substitute_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} patterns with environment variable values."""
    def replacer(match: re.Match) -> str:
        var_name = match.group(1)
        env_val = os.environ.get(var_name)
        if env_val is None:
            raise ValueError(f"Environment variable {var_name} is not set")
        return env_val
    return re.sub(r"\$\{([^}]+)\}", replacer, value)


def _resolve_env_vars(data: Any) -> Any:
    """Recursively resolve env vars in config data."""
    if isinstance(data, str):
        return _substitute_env_vars(data)
    if isinstance(data, dict):
        return {k: _resolve_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    return data


def load_config(path: str | Path | None = None) -> dict:
    """Load config from YAML file.

    Search order:
    1. Explicit *path* argument (from ``--config`` CLI flag).
    2. ``openagent.yaml`` in the current working directory.
    3. Platform-standard config directory (XDG on Linux, Application
       Support on macOS, %APPDATA% on Windows).

    Returns an empty dict if no config file is found anywhere.
    """
    if path:
        config_path = Path(path)
    else:
        cwd_path = Path(DEFAULT_CONFIG_FILE)
        if cwd_path.exists():
            config_path = cwd_path
        else:
            from src.core.paths import default_config_path
            config_path = default_config_path()

    if not config_path.exists():
        return {}
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return _resolve_env_vars(raw)


@dataclass(frozen=True)
class ShellSettings:
    """Runtime knobs for the in-process shell MCP.

    wake_wait_window_seconds:
        How long ``agent._run_inner`` sits after the model's final turn
        waiting for a background shell to complete (so short builds get
        auto-continuation). 0 disables; the default is 60.

    autoloop_cap:
        Maximum number of auto-continuation iterations per
        ``agent.run()`` call, protecting against a runaway shell →
        reminder → model → shell chain. Default 25.
    """
    wake_wait_window_seconds: float = 60.0
    autoloop_cap: int = 25


def shell_settings(config: dict) -> ShellSettings:
    """Parse ShellSettings out of the top-level ``openagent.yaml`` dict."""
    raw = (config or {}).get("shell") or {}
    return ShellSettings(
        wake_wait_window_seconds=float(raw.get("wake_wait_window_seconds", 60.0)),
        autoloop_cap=int(raw.get("autoloop_cap", 25)),
    )


@dataclass(frozen=True)
class SkillsSettings:
    """Runtime knobs for the native Skills subsystem (Hermes/Claude-Code
    SKILL.md progressive disclosure).

    OFF BY DEFAULT. With ``enabled=False`` nothing is injected into the
    system prompt and the ``skills`` builtin MCP is never registered — the
    production prompt and every code path stay byte-identical to a build
    without this feature.

    enabled:
        Master switch. Reads ``skills.enabled`` (default ``False``).

    path:
        Optional override for the skills directory. Reads ``skills.path``.
        ``None`` falls back to ``paths.default_skills_path()``
        (``<data_dir>/skills``), which itself honours the
        ``OPENAGENT_SKILLS_PATH`` env var — parity with
        ``memory.vault_path`` / ``OPENAGENT_VAULT_PATH``.

    curator_enabled:
        Master switch for the self-improving skill-curator scheduled task
        ("dream-mode for skills"). Reads ``skills.curator_enabled`` (default
        ``False``). SECOND gate on top of ``enabled``: the curator only
        seeds/enables when BOTH are true, so with the default the
        ``scheduled_tasks`` table gains no row and the system is
        byte-identical to a build without the feature. Mirrors how
        ``dream_mode.enabled`` gates the nightly vault pass.

    curator_schedule:
        Optional cron for the curator run. Reads ``skills.curator_schedule``.
        ``None`` falls back to a weekly default (Sunday 04:00) — parity with
        ``dream_mode.cron`` / ``auto_update.check_interval``.
    """
    enabled: bool = False
    path: str | None = None
    curator_enabled: bool = False
    curator_schedule: str | None = None


def skills_settings(config: dict) -> SkillsSettings:
    """Parse SkillsSettings out of the top-level ``openagent.yaml`` dict.

    Defensive: a missing/empty ``skills:`` stanza yields the OFF default,
    so a deployment that never heard of skills behaves exactly as before.
    """
    raw = (config or {}).get("skills") or {}
    if not isinstance(raw, dict):
        raw = {}
    path = raw.get("path")
    schedule = raw.get("curator_schedule")
    return SkillsSettings(
        enabled=bool(raw.get("enabled", False)),
        path=str(path) if path else None,
        curator_enabled=bool(raw.get("curator_enabled", False)),
        curator_schedule=str(schedule) if schedule else None,
    )
