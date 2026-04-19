"""Model-manager MCP server (v0.12 schema).

Exposes the ``providers`` + ``models`` tables over MCP so the agent
can inspect and edit its own LLM catalog at runtime. Writes land
directly in SQLite; the gateway polls ``MAX(updated_at)`` per message
and rebuilds SmartRouter's providers_config — so additions take effect
on the next turn without a process restart.

Vocabulary:
  - **provider_id**: the surrogate integer PK on ``providers`` rows.
    Returned by ``add_provider`` / ``list_providers``. All model
    writes use this to join.
  - **model id**: the surrogate integer PK on ``models`` rows. Returned
    by ``add_model`` / ``list_models``. Used by
    ``enable_model`` / ``disable_model`` / ``remove_model``.
  - **runtime_id**: composite string like
    ``claude-cli:anthropic:claude-opus-4-7`` — derived at read time,
    used for session pins and classifier output.

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
from openagent.memory.db import MemoryDB, VALID_FRAMEWORKS
from openagent.mcp.servers._common import SharedConnection, run_stdio

logger = logging.getLogger(__name__)

_shared = SharedConnection("model-manager")


async def _get_conn() -> aiosqlite.Connection:
    return await _shared.get()


mcp = FastMCP("model-manager")


# ── Providers ─────────────────────────────────────────────────────────────


@mcp.tool()
async def list_providers() -> list[dict[str, Any]]:
    """What providers are currently configured.

    Returns one entry per row with ``id``, ``name``, ``framework``,
    ``has_api_key``, ``base_url``, ``enabled``, and
    ``configured_model_count``. API keys are never surfaced in
    cleartext — only the presence flag.
    """
    conn = await _get_conn()
    rows = await (await conn.execute(
        "SELECT id, name, framework, api_key, base_url, enabled "
        "FROM providers ORDER BY name ASC, framework ASC"
    )).fetchall()

    out: list[dict[str, Any]] = []
    for pid, name, framework, api_key, base_url, enabled in rows:
        cnt_row = await (await conn.execute(
            "SELECT COUNT(*) FROM models WHERE provider_id = ?", (pid,),
        )).fetchone()
        out.append({
            "id": int(pid),
            "name": name,
            "framework": framework,
            "has_api_key": bool(api_key),
            "base_url": base_url or None,
            "enabled": bool(enabled),
            "configured_model_count": int(cnt_row[0]) if cnt_row else 0,
        })
    return out


@mcp.tool()
async def add_provider(
    name: str,
    framework: str = "agno",
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Register (or update) an LLM provider row.

    Under v0.12 the same vendor can be registered twice — once with
    ``framework='agno'`` (direct API via Agno SDK, needs ``api_key``)
    and once with ``framework='claude-cli'`` (the local ``claude``
    binary / Pro/Max subscription, ``api_key`` MUST be None).

    Provider keys live in the SQLite ``providers`` table. Writes are
    hot-reloaded on the next message via ``Agent.refresh_registries``.
    Once the provider is configured, use ``list_available_models`` to
    see what the vendor exposes under that key, then ``add_model`` to
    register specific model ids.
    """
    if not name or not name.strip():
        raise ValueError("name is required")
    if framework not in VALID_FRAMEWORKS:
        raise ValueError(
            f"invalid framework {framework!r}; expected one of {list(VALID_FRAMEWORKS)}"
        )
    if framework == "claude-cli" and api_key:
        raise ValueError(
            "claude-cli providers must not carry an api_key — "
            "the local `claude` binary uses your Pro/Max subscription."
        )
    if framework == "agno" and not (api_key or "").strip():
        raise ValueError("agno providers require an api_key")

    now = time.time()
    conn = await _get_conn()
    await conn.execute(
        """
        INSERT INTO providers (name, framework, api_key, base_url, enabled, metadata_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, 1, '{}', ?, ?)
        ON CONFLICT(name, framework) DO UPDATE SET
            api_key = excluded.api_key,
            base_url = COALESCE(excluded.base_url, providers.base_url),
            enabled = 1,
            updated_at = excluded.updated_at
        """,
        (
            name.strip(),
            framework,
            (api_key or "").strip() or None,
            (base_url or "").strip() or None,
            now,
            now,
        ),
    )
    await conn.commit()
    row = await (await conn.execute(
        "SELECT id FROM providers WHERE name = ? AND framework = ?",
        (name.strip(), framework),
    )).fetchone()
    pid = int(row[0])
    return {
        "id": pid,
        "name": name.strip(),
        "framework": framework,
        "has_api_key": bool((api_key or "").strip()),
        "base_url": (base_url or "").strip() or None,
    }


@mcp.tool()
async def update_provider(
    provider_id: int,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Patch a provider row.

    Only the fields you pass are changed. Pass ``base_url=''`` to clear
    an existing base_url. ``framework`` is immutable — delete+recreate
    to change it.
    """
    conn = await _get_conn()
    cursor = await conn.execute(
        "SELECT name, framework, api_key, base_url FROM providers WHERE id = ?",
        (int(provider_id),),
    )
    row = await cursor.fetchone()
    if row is None:
        raise ValueError(f"Provider id={provider_id} not found")
    name, framework, current_key, current_base = row
    new_key = current_key
    new_base = current_base
    if api_key is not None:
        if framework == "claude-cli" and api_key.strip():
            raise ValueError("claude-cli providers must not carry an api_key")
        new_key = api_key.strip() or None
    if base_url is not None:
        new_base = base_url.strip() or None
    await conn.execute(
        "UPDATE providers SET api_key = ?, base_url = ?, updated_at = ? WHERE id = ?",
        (new_key, new_base, time.time(), int(provider_id)),
    )
    await conn.commit()
    return {
        "id": int(provider_id),
        "name": name,
        "framework": framework,
        "has_api_key": bool(new_key),
        "base_url": new_base,
    }


@mcp.tool()
async def remove_provider(provider_id: int) -> dict[str, Any]:
    """Remove a provider AND cascade-purge its models from the DB.

    FK ``ON DELETE CASCADE`` on ``models.provider_id`` handles the
    model cleanup — no separate call needed. Returns the number of
    models that were wiped so the caller can surface it.
    """
    conn = await _get_conn()
    cursor = await conn.execute(
        "SELECT name, framework FROM providers WHERE id = ?", (int(provider_id),),
    )
    row = await cursor.fetchone()
    if row is None:
        raise ValueError(f"Provider id={provider_id} not found")
    name, framework = row
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM models WHERE provider_id = ?", (int(provider_id),),
    )
    cnt_row = await cursor.fetchone()
    purged = int(cnt_row[0]) if cnt_row else 0
    await conn.execute("DELETE FROM providers WHERE id = ?", (int(provider_id),))
    await conn.commit()
    return {
        "removed": True,
        "id": int(provider_id),
        "name": name,
        "framework": framework,
        "models_purged": purged,
    }


# ── Models ────────────────────────────────────────────────────────────────


def _model_row(d: dict[str, Any]) -> dict[str, Any]:
    """Shape an enriched model row for the MCP response."""
    from openagent.models.catalog import build_runtime_model_id

    runtime_id = d.get("runtime_id") or build_runtime_model_id(
        d["provider_name"], d["model"], d["framework"],
    )
    return {
        "id": int(d["id"]),
        "provider_id": int(d["provider_id"]),
        "provider_name": d["provider_name"],
        "framework": d["framework"],
        "model": d["model"],
        "display_name": d.get("display_name"),
        "tier_hint": d.get("tier_hint"),
        "enabled": bool(d.get("enabled", True)),
        "runtime_id": runtime_id,
    }


@mcp.tool()
async def list_models(
    provider_id: int | None = None,
    framework: str | None = None,
    enabled_only: bool = False,
) -> list[dict[str, Any]]:
    """List every LLM model currently registered in the DB.

    Pass ``provider_id`` to filter to one provider row, or ``framework``
    to filter by dispatch runtime. Each row carries ``id``,
    ``provider_id``, ``provider_name``, ``framework``, ``model`` (bare
    vendor id), ``display_name``, ``tier_hint`` (free-form classifier
    guidance), ``enabled``, and a derived ``runtime_id``.
    """
    db = MemoryDB(os.environ.get("OPENAGENT_DB_PATH", "openagent.db"))
    await db.connect()
    try:
        provider_name = None
        if provider_id is not None:
            prow = await db.get_provider(int(provider_id))
            if prow is None:
                raise ValueError(f"Provider id={provider_id} not found")
            # list_models_enriched filters on (name, framework) — derive
            # from the provider row to keep the join unambiguous.
            provider_name = prow["name"]
            framework = prow["framework"]
        rows = await db.list_models_enriched(
            enabled_only=enabled_only,
            framework=framework,
            provider_name=provider_name,
        )
        return [_model_row(r) for r in rows]
    finally:
        await db.close()


@mcp.tool()
async def get_model(model_id: int) -> dict[str, Any]:
    """Fetch one model row by its surrogate id."""
    db = MemoryDB(os.environ.get("OPENAGENT_DB_PATH", "openagent.db"))
    await db.connect()
    try:
        base = await db.get_model(int(model_id))
        if base is None:
            raise ValueError(f"Model id={model_id} not found")
        # Re-fetch via the enriched view so the caller gets provider +
        # framework + runtime_id in one dict.
        enriched = await db.list_models_enriched(enabled_only=False)
        for r in enriched:
            if r["id"] == int(model_id):
                return _model_row(r)
        raise RuntimeError(f"Model id={model_id} could not be enriched")
    finally:
        await db.close()


@mcp.tool()
async def list_supported_providers() -> list[str]:
    """Every vendor OpenAgent knows how to drive (anthropic, openai, …).

    Listing something here does NOT mean the install can use it — the
    user still needs to register it as a provider row via
    ``add_provider``. Under v0.12 the ``name`` and ``framework`` together
    identify a provider; the same vendor can appear twice under both
    frameworks.
    """
    from openagent.models.catalog import SUPPORTED_PROVIDERS

    return sorted(SUPPORTED_PROVIDERS)


@mcp.tool()
async def list_supported_frameworks() -> list[str]:
    """Every runtime OpenAgent can dispatch through.

    - ``agno``: direct provider API call via the Agno SDK. Works for
      every supported vendor.
    - ``claude-cli``: the local ``claude`` binary (Claude Pro/Max
      subscription). Only dispatches Anthropic models.
    """
    return list(VALID_FRAMEWORKS)


@mcp.tool()
async def add_model(
    provider_id: int,
    model: str,
    display_name: str | None = None,
    tier_hint: str | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    """Register a new LLM model row under a provider.

    - ``provider_id`` is the surrogate id of the provider row the model
      belongs to (see ``add_provider`` / ``list_providers``). The
      framework is inherited — no separate ``framework`` argument.
    - ``model`` is the bare vendor id (``gpt-4o-mini``,
      ``claude-sonnet-4-6``, ``glm-5``, …).
    - ``tier_hint`` (optional, free-form) is a soft hint to the
      classifier describing the model's strengths: ``"vision"``,
      ``"200k context"``, ``"best for code"``, ``"fast + cheap"``, etc.
      The classifier treats it as advice and overrides freely.

    Pricing is resolved live from OpenRouter on every billing event,
    so there is no cost field to set here. Returns the enriched row.
    """
    db = MemoryDB(os.environ.get("OPENAGENT_DB_PATH", "openagent.db"))
    await db.connect()
    try:
        mid = await db.upsert_model(
            provider_id=int(provider_id),
            model=model.strip(),
            display_name=display_name,
            tier_hint=tier_hint,
            enabled=enabled,
        )
        return await get_model(mid)
    finally:
        await db.close()


@mcp.tool()
async def update_model(
    model_id: int,
    display_name: str | None = None,
    tier_hint: str | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Partially update a model row (only fields you pass are changed).

    Pricing isn't editable — it's resolved live on every billing event.
    """
    db = MemoryDB(os.environ.get("OPENAGENT_DB_PATH", "openagent.db"))
    await db.connect()
    try:
        existing = await db.get_model(int(model_id))
        if existing is None:
            raise ValueError(f"Model id={model_id} not found")
        await db.upsert_model(
            provider_id=existing["provider_id"],
            model=existing["model"],
            display_name=display_name if display_name is not None else existing.get("display_name"),
            tier_hint=tier_hint if tier_hint is not None else existing.get("tier_hint"),
            enabled=enabled if enabled is not None else bool(existing.get("enabled", True)),
            metadata=existing.get("metadata") or None,
        )
        return await get_model(int(model_id))
    finally:
        await db.close()


@mcp.tool()
async def enable_model(model_id: int) -> dict[str, Any]:
    """Enable one model (takes effect on next message)."""
    db = MemoryDB(os.environ.get("OPENAGENT_DB_PATH", "openagent.db"))
    await db.connect()
    try:
        if await db.get_model(int(model_id)) is None:
            raise ValueError(f"Model id={model_id} not found")
        await db.set_model_enabled(int(model_id), True)
        return await get_model(int(model_id))
    finally:
        await db.close()


@mcp.tool()
async def disable_model(model_id: int) -> dict[str, Any]:
    """Disable one model (row preserved for re-enable)."""
    db = MemoryDB(os.environ.get("OPENAGENT_DB_PATH", "openagent.db"))
    await db.connect()
    try:
        if await db.get_model(int(model_id)) is None:
            raise ValueError(f"Model id={model_id} not found")
        await db.set_model_enabled(int(model_id), False)
        return await get_model(int(model_id))
    finally:
        await db.close()


@mcp.tool()
async def remove_model(model_id: int) -> dict[str, Any]:
    """Remove a model permanently.

    Refuses if this would leave zero enabled models — the agent would
    start rejecting every incoming message. Use ``disable_model``
    instead if you want to keep the row around.
    """
    db = MemoryDB(os.environ.get("OPENAGENT_DB_PATH", "openagent.db"))
    await db.connect()
    try:
        existing = await db.get_model(int(model_id))
        if existing is None:
            raise ValueError(f"Model id={model_id} not found")
        # Count remaining enabled (model AND its provider enabled).
        conn = await db._ensure_connected()
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM models m "
            "JOIN providers p ON p.id = m.provider_id "
            "WHERE m.enabled = 1 AND p.enabled = 1 AND m.id <> ?",
            (int(model_id),),
        )
        row = await cursor.fetchone()
        remaining = int(row[0]) if row else 0
        if remaining == 0 and existing.get("enabled"):
            raise ValueError(
                "Refusing to remove the last enabled model — the agent would "
                "reject every incoming message. Add another model first."
            )
        await db.delete_model(int(model_id))
        return {"removed": True, "id": int(model_id)}
    finally:
        await db.close()


# ── Session pins ──────────────────────────────────────────────────────────


@mcp.tool()
async def pin_session(session_id: str, runtime_id: str) -> dict[str, Any]:
    """Pin ``session_id`` to a specific model ``runtime_id`` forever.

    ``runtime_id`` stays a human-readable composite string
    (``openai:gpt-4o-mini`` or ``claude-cli:anthropic:claude-opus-4-7``)
    because the agent finds its current pin via the
    ``<session-id>...</session-id>`` tag in the framework system
    prompt, and those tags are rendered with the derived runtime_id.

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
    db = MemoryDB(os.environ.get("OPENAGENT_DB_PATH", "openagent.db"))
    await db.connect()
    try:
        # Enabled-model precheck: pin to a missing or disabled model
        # would start failing every turn with "no model available".
        row = await db.get_model_by_runtime_id(runtime_id)
        if row is None:
            raise ValueError(f"Model {runtime_id!r} is not registered. Use add_model first.")
        if not row.get("enabled") or not row.get("provider_enabled"):
            raise ValueError(f"Model {runtime_id!r} is disabled. Enable it before pinning.")
        await db.pin_session_model(session_id, runtime_id)
        return {"session_id": session_id, "runtime_id": runtime_id, "pinned": True}
    finally:
        await db.close()


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
    db = MemoryDB(os.environ.get("OPENAGENT_DB_PATH", "openagent.db"))
    await db.connect()
    try:
        await db.unpin_session_model(session_id)
        return {"session_id": session_id, "pinned": False}
    finally:
        await db.close()


# ── Discovery + smoke test ────────────────────────────────────────────────


@mcp.tool()
async def list_available_models(provider_id: int) -> list[dict[str, Any]]:
    """List models available from a provider (based on its API key).

    Uses ``openagent.models.discovery`` — queries the provider's
    ``/v1/models`` endpoint when the user has a key configured, falls
    back to a bundled catalog otherwise. Returns ``{id, display_name}``
    entries. Read-only: use ``add_model`` to actually register one.
    """
    from openagent.models.discovery import list_provider_models

    db = MemoryDB(os.environ.get("OPENAGENT_DB_PATH", "openagent.db"))
    await db.connect()
    try:
        provider_row = await db.get_provider(int(provider_id))
        if provider_row is None:
            raise ValueError(f"Provider id={provider_id} not found")
        return await list_provider_models(
            provider_row["name"],
            api_key=provider_row.get("api_key"),
            base_url=provider_row.get("base_url"),
        )
    finally:
        await db.close()


@mcp.tool()
async def test_model(runtime_id: str) -> dict[str, Any]:
    """Send a 1-token probe through a model to confirm the key works.

    Resolves the model via its runtime_id to a (provider_id, framework)
    pair, then reuses ``run_provider_smoke_test``. Does NOT write to
    the DB; use this before ``enable_model`` to confirm a new key.
    """
    from openagent.models.runtime import run_provider_smoke_test

    db = MemoryDB(os.environ.get("OPENAGENT_DB_PATH", "openagent.db"))
    await db.connect()
    try:
        model_row = await db.get_model_by_runtime_id(runtime_id)
        if model_row is None:
            raise ValueError(f"Model {runtime_id!r} is not registered")
        # Materialise the full providers_config so AgnoProvider sees
        # sibling entries (classifier model, etc.) too.
        provider_rows = await db.list_providers()
        model_rows = await db.list_models()
        by_id: dict[int, dict[str, Any]] = {}
        for p in provider_rows:
            by_id[p["id"]] = {**p, "models": []}
        for m in model_rows:
            if m["provider_id"] in by_id:
                by_id[m["provider_id"]]["models"].append({
                    "id": m["id"], "model": m["model"],
                    "display_name": m.get("display_name"),
                    "tier_hint": m.get("tier_hint"),
                    "enabled": m.get("enabled", True),
                })
        providers_config = list(by_id.values())
        _, resp = await run_provider_smoke_test(
            model_row["provider_name"],
            providers_config,
            model_id=runtime_id,
            framework=model_row["framework"],
            session_id="model-manager-probe",
        )
        return {"ok": True, "runtime_id": runtime_id, "response": resp.content}
    finally:
        await db.close()


def main() -> None:
    run_stdio(mcp, loglevel_env="OPENAGENT_MODEL_MANAGER_LOGLEVEL")


if __name__ == "__main__":
    main()
