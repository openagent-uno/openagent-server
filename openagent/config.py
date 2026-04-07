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
    """Load config from YAML file. Returns empty dict if file doesn't exist."""
    config_path = Path(path) if path else Path(DEFAULT_CONFIG_FILE)
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return _resolve_env_vars(raw)


def build_model_from_config(config: dict):
    """Instantiate a model from config dict."""
    from openagent.models.claude_api import ClaudeAPI
    from openagent.models.claude_cli import ClaudeCLI
    from openagent.models.zhipu import ZhipuGLM

    model_cfg = config.get("model", {})
    provider = model_cfg.get("provider", "claude-api")

    if provider == "claude-api":
        return ClaudeAPI(
            model=model_cfg.get("model_id", "claude-sonnet-4-6"),
            api_key=model_cfg.get("api_key"),
        )
    elif provider == "claude-cli":
        return ClaudeCLI(
            model=model_cfg.get("model_id"),
            permission_mode=model_cfg.get("permission_mode", "bypass"),
        )
    elif provider == "zhipu":
        return ZhipuGLM(
            model=model_cfg.get("model_id", "glm-4"),
            api_key=model_cfg.get("api_key"),
            base_url=model_cfg.get("base_url", "https://open.bigmodel.cn/api/paas/v4"),
        )
    else:
        raise ValueError(f"Unknown model provider: {provider}")
