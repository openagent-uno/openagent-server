"""One-shot import of yaml-configured MCPs and models into the DB.

Before this module, the MCP server list and per-provider model list lived
in ``openagent.yaml`` and were read on every boot. Moving them to the DB
lets the agent itself edit them at runtime (via mcp-manager / model-manager
MCPs) and hot-reload without a restart. To preserve existing user configs
without a manual migration, on first boot we copy the yaml entries into
the DB and set a ``config_state`` flag so we never run the import twice
against the same database.

Idempotency rests on two layers:

  1. The ``mcps_imported`` / ``models_imported`` flag in ``config_state``
     short-circuits the whole function after the first successful run.
  2. Every row-write uses ``upsert_*``'s ``ON CONFLICT DO UPDATE`` — so
     even if the flag is missing (someone deleted the row manually) the
     re-import just refreshes the existing rows instead of duplicating.

Subsequent yaml edits to ``mcp:`` / ``providers.X.models:`` are NOT
reflected in the DB — we log a one-line warning the first time we detect
the flag is set but the yaml still has entries. Users who want to change
MCPs/models after first boot go through the manager MCPs or the REST
endpoints.
"""

from __future__ import annotations

import logging
from typing import Any

from openagent.memory.db import MemoryDB

logger = logging.getLogger(__name__)


async def import_yaml_mcps_once(
    db: MemoryDB,
    mcp_config: list[dict] | None,
    include_defaults: bool,
    disable: list[str] | None,
) -> bool:
    """Seed the ``mcps`` table from yaml on first boot.

    Returns True iff rows were written. No-op (returns False) on
    subsequent boots because the ``mcps_imported`` state flag is set.

    Layout of rows written:
      - Every entry in ``DEFAULT_MCPS`` (if ``include_defaults``) becomes
        a ``kind='default'`` row. Disabled entries stay in the table with
        ``enabled=0`` so the UI can show them for re-enabling.
      - User-supplied ``mcp:`` entries become ``kind='builtin'`` when they
        use the ``{builtin: ...}`` shape, else ``kind='custom'``.

    The ``command``/``args``/``env`` fields are copied verbatim — spec
    resolution (relative→absolute paths, token injection) still happens
    in ``MCPPool._normalise_spec`` on load.
    """
    if await db.get_state("mcps_imported") == "1":
        if mcp_config or include_defaults:
            logger.debug("mcps already imported; ignoring yaml mcp: entries")
        return False

    # Deferred import: avoids a module-level cycle with openagent.mcp.
    from openagent.mcp.builtins import DEFAULT_MCPS

    disabled = set(disable or [])
    user_names = {
        str(entry.get("name") or entry.get("builtin") or "").strip()
        for entry in (mcp_config or [])
        if entry.get("name") or entry.get("builtin")
    }

    written = 0
    if include_defaults:
        for entry in DEFAULT_MCPS:
            name = str(entry.get("name") or entry.get("builtin") or "").strip()
            if not name:
                continue
            if name in user_names:
                # User override is written by the next loop; skip to avoid duplicates.
                continue
            enabled = name not in disabled
            if "builtin" in entry:
                await db.upsert_mcp(
                    name,
                    kind="default",
                    builtin_name=str(entry["builtin"]),
                    enabled=enabled,
                    source="yaml-default",
                )
            else:
                await db.upsert_mcp(
                    name,
                    kind="default",
                    command=list(entry.get("command") or []) or None,
                    args=list(entry.get("args") or []),
                    url=entry.get("url"),
                    env=dict(entry.get("env") or {}),
                    enabled=enabled,
                    source="yaml-default",
                )
            written += 1

    for entry in mcp_config or []:
        name = str(entry.get("name") or entry.get("builtin") or "").strip()
        if not name:
            continue
        enabled = name not in disabled
        if "builtin" in entry:
            await db.upsert_mcp(
                name,
                kind="builtin",
                builtin_name=str(entry["builtin"]),
                env=dict(entry.get("env") or {}),
                enabled=enabled,
                source="yaml",
            )
        else:
            await db.upsert_mcp(
                name,
                kind="custom",
                command=list(entry.get("command") or []) or None,
                args=list(entry.get("args") or []),
                url=entry.get("url"),
                env=dict(entry.get("env") or {}),
                headers=dict(entry.get("headers") or {}),
                oauth=bool(entry.get("oauth")),
                enabled=enabled,
                source="yaml",
            )
        written += 1

    await db.set_state("mcps_imported", "1")
    logger.info("bootstrap: imported %d MCP rows from yaml", written)
    return True


async def import_yaml_models_once(
    db: MemoryDB,
    providers_config: dict | None,
    model_cfg: dict | None = None,
) -> bool:
    """Seed the ``models`` table from the yaml ``providers:`` + ``model:`` sections.

    For each provider entry, every item in ``models:`` becomes a row. The
    ``runtime_id`` is computed via ``catalog.build_runtime_model_id`` so it
    matches the shape used everywhere else in the code (``openai:gpt-4o-mini``,
    ``claude-cli/claude-sonnet-4-6``, etc.).

    We ALSO register any model referenced by the ``model:`` section itself —
    ``model_id``, ``classifier_model``, and every ``routing.*`` entry — so
    deployments that don't bother listing models per provider (common for
    SmartRouter configs that only declare tier routing) still end up with
    enabled rows. Without this, the "no models enabled" rejection gate
    would fire on boot for those users.

    Disabled models (in ``providers.X.disabled_models``) are written with
    ``enabled=0`` so the UI can flip them back without losing the entry.
    """
    if await db.get_state("models_imported") == "1":
        return False

    from openagent.models.catalog import (
        build_runtime_model_id,
        normalize_runtime_model_id,
        split_runtime_id,
        is_claude_cli_model,
    )

    providers_config = providers_config or {}
    written = 0
    for provider_name, cfg in providers_config.items():
        cfg = cfg or {}
        disabled_ids = {
            str(item).strip() for item in (cfg.get("disabled_models") or [])
        }
        for entry in cfg.get("models") or []:
            model_id = _entry_model_id(entry)
            if not model_id:
                continue
            runtime_id = build_runtime_model_id(str(provider_name), model_id)
            if not runtime_id:
                continue
            meta = _entry_metadata(entry)
            await db.upsert_model(
                runtime_id,
                provider=str(provider_name),
                model_id=model_id,
                display_name=meta.get("display_name") or meta.get("name"),
                input_cost=_coerce_cost(meta.get("input_cost_per_million")),
                output_cost=_coerce_cost(meta.get("output_cost_per_million")),
                tier_hint=meta.get("tier_hint"),
                enabled=model_id not in disabled_ids,
                metadata=meta or None,
            )
            written += 1

    # Implicit models referenced by model.routing / model_id / classifier_model
    # that aren't listed under providers.X.models. Without this the rejection
    # gate would trip on SmartRouter-only configs.
    from openagent.models.catalog import _load_default_pricing
    pricing_keys = set(_load_default_pricing().keys())

    for ref in _extract_model_refs(model_cfg or {}):
        runtime_id = normalize_runtime_model_id(ref, providers_config)
        if not runtime_id:
            continue
        if ":" not in runtime_id and "/" not in runtime_id:
            # Bare id — scan the pricing table for a matching provider prefix.
            guess = None
            for key in pricing_keys:
                if ":" in key and key.split(":", 1)[1] == runtime_id:
                    guess = key.split(":", 1)[0]
                    break
            if guess is None:
                logger.warning(
                    "bootstrap: cannot resolve bare model ref %r to a provider — "
                    "skipping (add it to providers.X.models or prefix with "
                    "provider:)",
                    runtime_id,
                )
                continue
            runtime_id = f"{guess}:{runtime_id}"
        if is_claude_cli_model(runtime_id):
            provider_name = "claude-cli"
        else:
            provider_name, _ = split_runtime_id(runtime_id)
        if not provider_name:
            continue
        _, bare_model_id = split_runtime_id(runtime_id)
        if await db.get_model(runtime_id):
            continue  # already written above
        await db.upsert_model(
            runtime_id,
            provider=provider_name,
            model_id=bare_model_id,
            enabled=True,
            metadata={"source": "model_cfg"},
        )
        written += 1

    await db.set_state("models_imported", "1")
    logger.info("bootstrap: imported %d model rows from yaml", written)
    return True


def _extract_model_refs(model_cfg: dict) -> list[str]:
    """Collect every model id mentioned in the ``model:`` yaml section."""
    refs: list[str] = []
    direct = str(model_cfg.get("model_id") or "").strip()
    if direct:
        refs.append(direct)
    classifier = str(model_cfg.get("classifier_model") or "").strip()
    if classifier:
        refs.append(classifier)
    for tier_ref in (model_cfg.get("routing") or {}).values():
        if isinstance(tier_ref, str) and tier_ref.strip():
            refs.append(tier_ref.strip())
    return refs


def _entry_model_id(entry: Any) -> str:
    """Same coercion rules as openagent.models.catalog._entry_model_id."""
    if isinstance(entry, dict):
        for key in ("id", "model_id", "model"):
            value = entry.get(key)
            if value:
                return str(value).strip()
        return ""
    return str(entry or "").strip()


def _entry_metadata(entry: Any) -> dict:
    return dict(entry) if isinstance(entry, dict) else {}


def _coerce_cost(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
