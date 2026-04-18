"""Model-manager MCP server.

Exposes the ``models`` table over MCP so the agent can inspect and edit
its own LLM catalog at runtime. Writes land directly in SQLite; the
gateway polls ``MAX(updated_at)`` per message and rebuilds the
SmartRouter routing dict — so additions take effect on the next turn
without a process restart.

Transport: stdio. Storage: the shared OpenAgent SQLite DB via
``OPENAGENT_DB_PATH`` (set by MCPPool at launch).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import aiosqlite
from mcp.server.fastmcp import FastMCP
from openagent.memory.db import MemoryDB
from openagent.mcp.servers._common import SharedConnection, ensure_row_exists, run_stdio

logger = logging.getLogger(__name__)

_shared = SharedConnection("model-manager")


async def _get_conn() -> aiosqlite.Connection:
    return await _shared.get()


_row_to_dict = MemoryDB._row_to_model


mcp = FastMCP("model-manager")


@mcp.tool()
async def list_models(
    provider: str | None = None,
    enabled_only: bool = False,
) -> list[dict[str, Any]]:
    """List every LLM model currently registered in the DB.

    Pass ``provider`` to filter (``openai``, ``anthropic``, ``google``,
    ``claude-cli``, etc.). Each row has ``runtime_id`` (canonical id
    used by SmartRouter), ``provider``, ``model_id`` (bare id),
    ``display_name``, ``input_cost_per_million``, ``output_cost_per_million``,
    ``tier_hint`` (optional simple/medium/hard for routing), ``enabled``.
    """
    conn = await _get_conn()
    clauses: list[str] = []
    params: list[Any] = []
    if provider:
        clauses.append("provider = ?")
        params.append(provider)
    if enabled_only:
        clauses.append("enabled = 1")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    cursor = await conn.execute(
        f"SELECT * FROM models {where} ORDER BY provider ASC, model_id ASC",
        params,
    )
    rows = await cursor.fetchall()
    return [_row_to_dict(r) for r in rows]


@mcp.tool()
async def get_model(runtime_id: str) -> dict[str, Any]:
    """Fetch one model row by its canonical runtime id."""
    conn = await _get_conn()
    cursor = await conn.execute(
        "SELECT * FROM models WHERE runtime_id = ?", (runtime_id,)
    )
    row = await cursor.fetchone()
    if not row:
        raise ValueError(f"No model with runtime_id={runtime_id!r}")
    return _row_to_dict(row)


@mcp.tool()
async def list_supported_providers() -> list[str]:
    """Every vendor OpenAgent knows how to drive (anthropic, openai, …).

    Listing something here does NOT mean the install can use it — the
    user still needs to register an API key via ``add_provider``.
    ``claude-cli`` is NOT a provider — it's a framework (runtime) that
    dispatches Anthropic models through the local ``claude`` binary
    instead of the API. See ``list_supported_frameworks``.
    """
    from openagent.models.catalog import SUPPORTED_PROVIDERS

    return sorted(SUPPORTED_PROVIDERS)


@mcp.tool()
async def list_supported_frameworks() -> list[str]:
    """Every runtime OpenAgent can dispatch through.

    - ``agno``: direct provider API call via the Agno SDK. Works for
      every supported provider.
    - ``claude-cli``: the local ``claude`` binary (Claude Pro/Max
      subscription). Only dispatches Anthropic models.
    """
    from openagent.models.catalog import SUPPORTED_FRAMEWORKS

    return list(SUPPORTED_FRAMEWORKS)


@mcp.tool()
async def list_providers() -> list[dict[str, Any]]:
    """What providers are currently configured in the yaml.

    Returns one entry per configured provider with ``name``, ``has_api_key``,
    ``base_url``, and ``configured_model_count`` (rows in the ``models``
    table). Keys are not surfaced in cleartext — only the presence flag.
    """
    providers_cfg = _load_providers_from_yaml()
    conn = await _get_conn()

    out: list[dict[str, Any]] = []
    for name, cfg in providers_cfg.items():
        cfg = cfg or {}
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM models WHERE provider = ?", (name,),
        )
        row = await cursor.fetchone()
        count = int(row[0]) if row else 0
        out.append({
            "name": name,
            "has_api_key": bool(cfg.get("api_key")),
            "base_url": cfg.get("base_url") or None,
            "configured_model_count": count,
        })
    return out


@mcp.tool()
async def add_provider(
    name: str,
    api_key: str,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Register a new LLM provider + credentials.

    Writes ``providers.<name>`` to ``openagent.yaml`` with
    ``api_key`` (and optional ``base_url``). Keys are the yaml's
    source-of-truth — not stored in the DB. Once the provider is
    configured, use ``list_available_models(provider=<name>)`` to see
    what the provider exposes with that key, then ``add_model`` to
    register specific runtime ids.
    """
    if not name or not name.strip():
        raise ValueError("name is required")
    if not api_key or not api_key.strip():
        raise ValueError("api_key is required")
    raw = _read_yaml()
    providers = dict(raw.get("providers") or {})
    entry: dict[str, Any] = dict(providers.get(name, {}) or {})
    entry["api_key"] = api_key.strip()
    if base_url is not None:
        if base_url.strip():
            entry["base_url"] = base_url.strip()
        else:
            entry.pop("base_url", None)
    providers[name] = entry
    raw["providers"] = providers
    _write_yaml(raw)
    # Return a sanitized entry (no cleartext key).
    return {
        "name": name,
        "has_api_key": True,
        "base_url": entry.get("base_url") or None,
    }


@mcp.tool()
async def update_provider(
    name: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Patch ``providers.<name>`` in the yaml.

    Only the fields you pass are changed. Pass ``base_url=''`` to clear
    an existing base_url.
    """
    raw = _read_yaml()
    providers = dict(raw.get("providers") or {})
    if name not in providers:
        raise ValueError(f"Provider {name!r} is not configured")
    entry = dict(providers[name] or {})
    if api_key is not None:
        if not api_key.strip():
            raise ValueError("api_key cannot be empty")
        entry["api_key"] = api_key.strip()
    if base_url is not None:
        if base_url.strip():
            entry["base_url"] = base_url.strip()
        else:
            entry.pop("base_url", None)
    providers[name] = entry
    raw["providers"] = providers
    _write_yaml(raw)
    return {
        "name": name,
        "has_api_key": bool(entry.get("api_key")),
        "base_url": entry.get("base_url") or None,
    }


@mcp.tool()
async def remove_provider(name: str) -> dict[str, Any]:
    """Remove a provider from the yaml AND purge its models from the DB.

    Provider keys live in yaml, models live in SQLite — without the
    cascade, removing a provider orphans every model row that referenced
    it. Those would keep showing in the catalog and the router would
    try to dispatch them, failing with a confusing "missing API key" at
    send time. Cleanup happens in a single tool call.
    """
    raw = _read_yaml()
    providers = dict(raw.get("providers") or {})
    if name not in providers:
        raise ValueError(f"Provider {name!r} is not configured")
    providers.pop(name)
    raw["providers"] = providers
    _write_yaml(raw)

    models_purged = 0
    try:
        conn = await _get_conn()
        cursor = await conn.execute("DELETE FROM models WHERE provider = ?", (name,))
        await conn.commit()
        models_purged = cursor.rowcount or 0
    except Exception:
        # yaml write succeeded; DB cascade failure is survivable — the
        # next provider read will flag the orphans as unreachable.
        pass
    return {"removed": True, "name": name, "models_purged": models_purged}


@mcp.tool()
async def add_model(
    provider: str,
    model_id: str,
    framework: str = "agno",
    display_name: str | None = None,
    input_cost_per_million: float | None = None,
    output_cost_per_million: float | None = None,
    tier_hint: str | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Register a new LLM model.

    - ``provider`` is the vendor: ``anthropic``, ``openai``, ``google``,
      ``zai``, ``groq``, ``local``, …
    - ``framework`` is the runtime that dispatches the model: ``agno``
      (direct API call via the Agno SDK — default, works for every
      provider) or ``claude-cli`` (the local ``claude`` binary, which
      only wraps Anthropic models and uses the user's Pro/Max
      subscription instead of API keys).
    - ``model_id`` is the bare vendor id (``gpt-4o-mini``,
      ``claude-sonnet-4-6``, ``glm-5``, …).

    The canonical ``runtime_id`` comes out as ``<provider>:<model>`` for
    agno rows and ``claude-cli:<provider>:<model>`` for claude-cli rows.
    Use ``tier_hint=simple|medium|hard`` to force placement in the
    SmartRouter routing table; otherwise auto-routing sorts by cost.
    """
    from openagent.models.catalog import (
        FRAMEWORK_AGNO,
        FRAMEWORK_CLAUDE_CLI,
        SUPPORTED_FRAMEWORKS,
        build_runtime_model_id,
    )

    if not provider or not provider.strip():
        raise ValueError("provider is required")
    if not model_id or not model_id.strip():
        raise ValueError("model_id is required")
    framework = (framework or FRAMEWORK_AGNO).strip()
    if framework not in SUPPORTED_FRAMEWORKS:
        raise ValueError(
            f"invalid framework {framework!r}; expected one of {SUPPORTED_FRAMEWORKS}"
        )
    # Legacy shorthand: caller passed provider="claude-cli". Rewrite to
    # the modern vocabulary (framework=claude-cli, provider=anthropic).
    if provider.strip() == FRAMEWORK_CLAUDE_CLI:
        provider = "anthropic"
        framework = FRAMEWORK_CLAUDE_CLI
    runtime_id = build_runtime_model_id(provider.strip(), model_id.strip(), framework)
    if not runtime_id:
        raise ValueError(
            f"could not build runtime_id from provider={provider!r} model_id={model_id!r}"
        )
    conn = await _get_conn()
    now = time.time()
    await conn.execute(
        "INSERT INTO models (runtime_id, provider, framework, model_id, display_name, "
        "input_cost_per_million, output_cost_per_million, tier_hint, enabled, "
        "metadata_json, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?) "
        "ON CONFLICT(runtime_id) DO UPDATE SET "
        "provider = excluded.provider, framework = excluded.framework, "
        "model_id = excluded.model_id, display_name = excluded.display_name, "
        "input_cost_per_million = excluded.input_cost_per_million, "
        "output_cost_per_million = excluded.output_cost_per_million, "
        "tier_hint = excluded.tier_hint, enabled = excluded.enabled, "
        "updated_at = excluded.updated_at",
        (
            runtime_id,
            provider.strip(),
            framework,
            model_id.strip(),
            display_name,
            input_cost_per_million,
            output_cost_per_million,
            tier_hint,
            1 if enabled else 0,
            now,
            now,
        ),
    )
    await conn.commit()
    return await get_model(runtime_id)


@mcp.tool()
async def update_model(
    runtime_id: str,
    display_name: str | None = None,
    input_cost_per_million: float | None = None,
    output_cost_per_million: float | None = None,
    tier_hint: str | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Partially update a model row (only fields you pass are changed)."""
    conn = await _get_conn()
    cursor = await conn.execute(
        "SELECT 1 FROM models WHERE runtime_id = ?", (runtime_id,)
    )
    if not await cursor.fetchone():
        raise ValueError(f"No model with runtime_id={runtime_id!r}")

    updates: dict[str, Any] = {}
    if display_name is not None:
        updates["display_name"] = display_name
    if input_cost_per_million is not None:
        updates["input_cost_per_million"] = float(input_cost_per_million)
    if output_cost_per_million is not None:
        updates["output_cost_per_million"] = float(output_cost_per_million)
    if tier_hint is not None:
        updates["tier_hint"] = tier_hint or None
    if enabled is not None:
        updates["enabled"] = 1 if enabled else 0
    if not updates:
        raise ValueError("No fields to update")
    updates["updated_at"] = time.time()
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [runtime_id]
    await conn.execute(
        f"UPDATE models SET {set_clause} WHERE runtime_id = ?", values
    )
    await conn.commit()
    return await get_model(runtime_id)


@mcp.tool()
async def enable_model(runtime_id: str) -> dict[str, Any]:
    """Enable one model (takes effect on next message)."""
    conn = await _get_conn()
    await ensure_row_exists(conn, "models", "runtime_id", runtime_id)
    await conn.execute(
        "UPDATE models SET enabled = 1, updated_at = ? WHERE runtime_id = ?",
        (time.time(), runtime_id),
    )
    await conn.commit()
    return await get_model(runtime_id)


@mcp.tool()
async def disable_model(runtime_id: str) -> dict[str, Any]:
    """Disable one model (row preserved for re-enable)."""
    conn = await _get_conn()
    await ensure_row_exists(conn, "models", "runtime_id", runtime_id)
    await conn.execute(
        "UPDATE models SET enabled = 0, updated_at = ? WHERE runtime_id = ?",
        (time.time(), runtime_id),
    )
    await conn.commit()
    return await get_model(runtime_id)


@mcp.tool()
async def remove_model(runtime_id: str) -> dict[str, Any]:
    """Remove a model permanently.

    Refuses if this would leave zero enabled models — the agent would
    start rejecting every incoming message. Use ``disable_model``
    instead if you want to keep the row around.
    """
    conn = await _get_conn()
    await ensure_row_exists(conn, "models", "runtime_id", runtime_id)
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM models WHERE enabled = 1 AND runtime_id <> ?",
        (runtime_id,),
    )
    row = await cursor.fetchone()
    # COUNT(*) always returns one row with an integer value.
    remaining = int(row[0])
    if remaining == 0:
        raise ValueError(
            "Refusing to remove the last enabled model — the agent would "
            "reject every incoming message. Add another model first."
        )
    await conn.execute("DELETE FROM models WHERE runtime_id = ?", (runtime_id,))
    await conn.commit()
    return {"removed": True, "runtime_id": runtime_id}


@mcp.tool()
async def pin_session(session_id: str, runtime_id: str) -> dict[str, Any]:
    """Pin ``session_id`` to a specific model ``runtime_id`` forever.

    Subsequent turns on this session skip the SmartRouter classifier
    and dispatch straight to ``runtime_id``. Use this when the user
    asks "force/always use model X for me" — e.g.
    ``pin_session(session_id, "claude-cli:anthropic:claude-opus-4-6")``.

    The agent can find its current ``session_id`` in the
    ``<session-id>...</session-id>`` tag of the framework system
    prompt.

    Raises if the pinned model belongs to a different framework than
    the session's existing binding (pinning a claude-cli session to an
    agno model would split conversation history across two stores).
    Use ``unpin_session`` to release.
    """
    session_id = (session_id or "").strip()
    runtime_id = (runtime_id or "").strip()
    if not session_id:
        raise ValueError("session_id is required")
    if not runtime_id:
        raise ValueError("runtime_id is required")
    conn = await _get_conn()
    # Enabled-model precheck: pin to a missing or disabled model would
    # start failing every turn with "no model available".
    cursor = await conn.execute(
        "SELECT enabled FROM models WHERE runtime_id = ?", (runtime_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        raise ValueError(f"Model {runtime_id!r} is not registered. Use add_model first.")
    if not row[0]:
        raise ValueError(f"Model {runtime_id!r} is disabled. Enable it before pinning.")
    await _pin_on_shared_conn(conn, session_id, runtime_id)
    return {"session_id": session_id, "runtime_id": runtime_id, "pinned": True}


@mcp.tool()
async def unpin_session(session_id: str) -> dict[str, Any]:
    """Clear the per-session model pin on ``session_id``.

    The session returns to normal SmartRouter routing (classifier →
    tier → model) on the next turn, while keeping its framework
    binding (agno or claude-cli) intact.
    """
    session_id = (session_id or "").strip()
    if not session_id:
        raise ValueError("session_id is required")
    conn = await _get_conn()
    await conn.execute(
        "UPDATE session_bindings SET runtime_id = NULL, bound_at = ? WHERE session_id = ?",
        (time.time(), session_id),
    )
    await conn.commit()
    return {"session_id": session_id, "pinned": False}


async def _pin_on_shared_conn(conn, session_id: str, runtime_id: str) -> None:
    """Mirror of ``MemoryDB.pin_session_model`` against the subprocess's
    shared aiosqlite connection, so the tool doesn't pay the cost of
    opening a second connection + re-running schema migrations every
    pin. Kept in sync with the MemoryDB method's framework-lock logic.
    """
    target_framework = "claude-cli" if runtime_id.startswith("claude-cli") else "agno"
    # Look up the existing framework binding — sdk_sessions (source of
    # truth for claude-cli) then session_bindings (agno).
    cursor = await conn.execute(
        "SELECT provider FROM sdk_sessions WHERE session_id = ?", (session_id,),
    )
    row = await cursor.fetchone()
    existing = row[0] if row and row[0] else None
    if not existing:
        cursor = await conn.execute(
            "SELECT provider FROM session_bindings WHERE session_id = ?", (session_id,),
        )
        row = await cursor.fetchone()
        existing = row[0] if row and row[0] else None
    if existing and existing != target_framework:
        raise ValueError(
            f"session {session_id!r} is bound to framework={existing!r} and "
            f"cannot be pinned to a {target_framework!r} model — conversation "
            "history lives in the current framework's store. Use /clear first."
        )
    await conn.execute(
        "INSERT INTO session_bindings (session_id, provider, bound_at, runtime_id) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(session_id) DO UPDATE SET "
        "provider = excluded.provider, bound_at = excluded.bound_at, "
        "runtime_id = excluded.runtime_id",
        (session_id, target_framework, time.time(), runtime_id),
    )
    await conn.commit()


@mcp.tool()
async def list_available_models(provider: str) -> list[dict[str, Any]]:
    """List models available from a provider (based on its API key).

    Uses ``openagent.models.discovery`` — queries the provider's
    ``/v1/models`` endpoint when the user has a key configured, falls
    back to a bundled catalog otherwise. Returns ``{id, display_name}``
    entries. Read-only: use ``add_model`` to actually register one.
    """
    from openagent.models.discovery import list_provider_models_cached

    return await list_provider_models_cached(provider)


@mcp.tool()
async def test_model(runtime_id: str) -> dict[str, Any]:
    """Send a 1-token probe through a model to confirm the key works.

    Reuses ``openagent.models.runtime.run_provider_smoke_test``. Does NOT
    write to the DB; use this before ``enable_model`` to confirm a key
    is valid.
    """
    from openagent.models.runtime import run_provider_smoke_test
    from openagent.models.catalog import split_runtime_id

    provider, _ = split_runtime_id(runtime_id)
    # Providers config is owned by yaml; load it fresh on each call so
    # key rotations are picked up without a restart.
    providers_config = _load_providers_from_yaml()
    _, resp = await run_provider_smoke_test(
        provider,
        providers_config,
        model_id=runtime_id,
        session_id="model-manager-probe",
    )
    return {"ok": True, "runtime_id": runtime_id, "response": resp.content}


def _load_providers_from_yaml() -> dict:
    """Read the providers section of the live openagent.yaml (env-resolved).

    Returns ``{}`` when the config file is missing or unreadable — those
    are expected in test harnesses and first-boot scenarios. Anything
    else (unexpected exception) is logged so real config errors surface.
    """
    from openagent.core.config import load_config
    try:
        return load_config(os.environ.get("OPENAGENT_CONFIG_PATH")).get("providers", {}) or {}
    except (FileNotFoundError, PermissionError, OSError):
        return {}
    except Exception as e:  # noqa: BLE001 — upstream YAML errors, etc.
        logger.warning("load_config failed in model-manager: %s", e)
        return {}


def _yaml_path() -> str:
    """Resolve the yaml path we're allowed to edit.

    Precedence matches ``openagent.core.config.load_config``: env var
    ``OPENAGENT_CONFIG_PATH`` wins, then CWD's ``openagent.yaml``, then
    the platform-standard path. Writing to a missing path just creates
    the file under its parent directory.
    """
    explicit = os.environ.get("OPENAGENT_CONFIG_PATH")
    if explicit:
        return os.path.expanduser(explicit)
    cwd = os.path.abspath("openagent.yaml")
    if os.path.exists(cwd):
        return cwd
    from openagent.core.paths import default_config_path
    return str(default_config_path())


def _read_yaml() -> dict[str, Any]:
    """Read the yaml WITHOUT env-var resolution.

    We want to write back later; resolving ``${VAR}`` now would
    substitute the value and destroy the reference. The live
    ``load_config`` path is still the right one for READERS, but this
    helper is for read-modify-write cycles on the on-disk file.
    """
    import yaml

    path = _yaml_path()
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _write_yaml(data: dict[str, Any]) -> None:
    import yaml

    path = _yaml_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def main() -> None:
    run_stdio(mcp, loglevel_env="OPENAGENT_MODEL_MANAGER_LOGLEVEL")


if __name__ == "__main__":
    main()
