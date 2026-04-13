"""Configuration loader for OpenAgent. Supports YAML config with env var substitution."""

from __future__ import annotations

import os
import re
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
            from openagent.core.paths import default_config_path
            config_path = default_config_path()

    if not config_path.exists():
        return {}
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return _resolve_env_vars(raw)


def build_model_from_config(config: dict):
    """Instantiate a model from config dict."""
    import logging
    from openagent.models.agno_provider import AgnoProvider
    from openagent.models.claude_cli import ClaudeCLI

    _logger = logging.getLogger(__name__)
    model_cfg = config.get("model", {})
    provider = model_cfg.get("provider", "agno")
    providers_config = config.get("providers", {})

    # Backward compat: map legacy provider names to the Agno-backed API runtime.
    if provider == "claude-api":
        _logger.info("provider 'claude-api' mapped to 'agno' with model 'anthropic:%s'",
                      model_cfg.get("model_id", "claude-sonnet-4-6"))
        return AgnoProvider(
            model=f"anthropic:{model_cfg.get('model_id', 'claude-sonnet-4-6')}",
            api_key=model_cfg.get("api_key"),
            providers_config=providers_config,
        )
    elif provider in ("zhipu", "zai"):
        model_id = model_cfg.get("model_id", "glm-5")
        base_url = model_cfg.get("base_url", "https://api.z.ai/api/paas/v4")
        _logger.info("provider '%s' mapped to 'agno' with model 'zai:%s'", provider, model_id)
        return AgnoProvider(
            model=f"zai:{model_id}",
            api_key=model_cfg.get("api_key"),
            base_url=base_url,
            providers_config=providers_config,
        )
    elif provider == "claude-cli":
        return ClaudeCLI(
            model=model_cfg.get("model_id"),
            permission_mode=model_cfg.get("permission_mode", "bypass"),
        )
    elif provider in ("litellm", "agno"):
        return AgnoProvider(
            model=model_cfg.get("model_id", "anthropic:claude-sonnet-4-20250514"),
            api_key=model_cfg.get("api_key"),
            base_url=model_cfg.get("base_url"),
            providers_config=providers_config,
        )
    elif provider == "smart":
        from openagent.models.smart_router import SmartRouter
        return SmartRouter(
            routing=model_cfg.get("routing") or None,
            api_key=model_cfg.get("api_key"),
            monthly_budget=float(model_cfg.get("monthly_budget", 0)),
            classifier_model=model_cfg.get("classifier_model"),
            providers_config=providers_config,
            claude_permission_mode=model_cfg.get("permission_mode", "bypass"),
        )
    else:
        raise ValueError(f"Unknown model provider: {provider}")
