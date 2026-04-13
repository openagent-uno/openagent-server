"""Backward-compatible shim for the old LiteLLM-backed provider name.

The runtime on this branch uses Agno for API-backed models. The old class name
is kept temporarily so existing imports continue to work while the rest of the
codebase migrates.
"""

from __future__ import annotations

from openagent.models.agno_provider import AgnoProvider
from openagent.models.catalog import get_default_model_for_provider


class LiteLLMProvider(AgnoProvider):
    """Deprecated compatibility alias for API-backed models."""


def get_cheapest_model(provider: str, providers_config: dict | None = None) -> str | None:
    """Backward-compatible helper name used by older CLI/API code paths."""
    return get_default_model_for_provider(provider, providers_config or {})
