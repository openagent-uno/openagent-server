"""Shared runtime helpers for building and testing chat models."""

from __future__ import annotations

from typing import Any

from src.models.base import BaseModel, ModelResponse
from src.models.catalog import (
    _iter_provider_entries,
    get_default_model_for_provider,
    normalize_runtime_model_id,
)


_UNSET = object()


# ── Defer-all MCP wiring (v0.14+) ───────────────────────────────────
#
# Every LLM gets ONLY the ``tool-search`` MCP up front. Its four meta-
# tools (``list_servers`` / ``list_tools`` / ``describe_tool`` /
# ``call_tool``) are the model's uniform interface to every other
# connected MCP — there is no upfront/deferred split that varies by
# tool count. Replaces the previous budget-trim heuristic that handed
# the model some MCPs upfront and put the rest behind tool-search; in
# practice that split confused the model about which capabilities
# needed discovery.
#
# Pre-knowledge of what's behind tool-search comes via the catalog
# summary injected into the system prompt (see
# :func:`src.core.prompts.build_mcp_catalog_summary`), so the model
# rarely needs a discovery turn at all.


def wire_model_runtime(
    model: BaseModel,
    *,
    db: Any = None,
    mcp_pool: Any = None,
    fallback_config: Any = _UNSET,
) -> BaseModel:
    """Attach runtime dependencies (DB, MCP pool, fallback config) to a model.

    Wires only the ``tool-search`` MCP into the model's upfront tool
    list. Every other MCP connected in ``mcp_pool`` stays reachable
    via ``tool-search.call_tool``. NativeProvider gets the in-process
    ``MCPTools`` instance.
    """
    if db is not None:
        set_db = getattr(model, "set_db", None)
        if callable(set_db):
            set_db(db)
    if mcp_pool is not None:
        # NativeProvider / TeamRouterProvider: in-process MCPTools instance(s).
        set_mcp_toolkits = getattr(model, "set_mcp_toolkits", None)
        if callable(set_mcp_toolkits):
            set_mcp_toolkits(mcp_pool.runtime_toolkits_tool_search_only())
        # SmartRouter holds the pool itself so it can re-wire newly
        # constructed per-session providers as they're lazily built.
        set_mcp_pool = getattr(model, "set_mcp_pool", None)
        if callable(set_mcp_pool):
            set_mcp_pool(mcp_pool)
    if fallback_config is not _UNSET:
        set_fallback_config = getattr(model, "set_fallback_config", None)
        if callable(set_fallback_config):
            set_fallback_config(fallback_config)
    return model


def create_model_from_spec(
    spec: str,
    *,
    providers_config: Any = None,
    db: Any = None,
    mcp_pool: Any = None,
) -> BaseModel:
    """Create a model instance from a compact OpenAgent runtime spec."""
    if providers_config is None:
        providers_config = []

    if spec == "smart":
        from src.models.dispatcher import ModelDispatcher

        model: BaseModel = ModelDispatcher(providers_config=providers_config)
    else:
        from src.models.native_provider import NativeProvider

        model = NativeProvider(
            model=spec,
            providers_config=providers_config,
            db_path=getattr(db, "db_path", None),
        )

    return wire_model_runtime(model, db=db, mcp_pool=mcp_pool)


def create_model_from_config(config: dict) -> BaseModel:
    """Instantiate the active model from the resolved OpenAgent config.

    Always returns a SmartRouter — SmartRouter is the single top-level
    runtime and dispatches each session to the native API provider
    internally (see ``openagent.models.smart_router``). The ``providers`` / ``models``
    SQLite tables are the sole source of truth for the catalog;
    SmartRouter starts empty and gets its routing populated by
    ``Agent.initialize`` (and every hot-reload tick) via
    ``rebuild_routing``. The yaml is never consulted for provider or
    model state.
    """
    del config  # catalog comes from the DB, not yaml
    return create_model_from_spec("smart", providers_config=[])


async def run_provider_smoke_test(
    provider_name: str,
    providers_config: Any,
    *,
    model_id: str | None = None,
    framework: str | None = None,
    session_id: str = "provider-test",
    prompt: str = "Say 'ok' and nothing else.",
) -> tuple[str, ModelResponse]:
    """Run a minimal prompt through the configured runtime for one provider.

    When the same vendor is registered under more than one framework,
    pass ``framework=`` to disambiguate — otherwise the first matching
    entry wins.
    """
    # Resolve the provider row by (name, framework) pair. Fall back to
    # the first entry that matches by name when framework is unspecified.
    cfg: dict[str, Any] | None = None
    for entry in _iter_provider_entries(providers_config):
        if str(entry.get("name") or "").strip() != provider_name:
            continue
        if framework and entry.get("framework") != framework:
            continue
        cfg = dict(entry)
        break
    if cfg is None:
        raise ValueError(f"Provider '{provider_name}' not configured")

    # Resolve a model: the caller's explicit model_id, else a default
    # scoped to the provider row's framework.
    runtime_model = model_id or get_default_model_for_provider(
        provider_name,
        providers_config,
        framework=cfg.get("framework"),
    )
    if not runtime_model:
        raise ValueError(f"No models configured for provider '{provider_name}'")
    runtime_model = normalize_runtime_model_id(runtime_model, providers_config)

    provider = create_model_from_spec(
        runtime_model,
        providers_config=providers_config,
    )
    resp = await provider.generate(
        messages=[{"role": "user", "content": prompt}],
        session_id=session_id,
    )
    return runtime_model, resp
