"""/api/models — DB-backed model catalog (v0.12 schema).

Models join to their parent provider row via ``models.provider_id``.
Framework is inherited from the provider; ``runtime_id`` is derived at
read time (``build_runtime_model_id(provider_name, model, framework)``).

GET    /api/models                → enriched list with derived runtime_id
POST   /api/models                → {provider_id, model, display_name?, tier_hint?, enabled?}
GET    /api/models/{id}           → fetch one (enriched)
PUT    /api/models/{id}           → update one
DELETE /api/models/{id}           → delete one
POST   /api/models/{id}/enable|disable

GET    /api/models/catalog        → iter_configured_models view w/ pricing
GET    /api/models/available?provider_id=N → discovery-driven per-provider catalog
GET    /api/models/providers      → supported provider list
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiohttp import web

from openagent.gateway.api._common import gateway_db as _db


async def handle_available_providers(request: web.Request) -> web.Response:
    """GET /api/models/providers — provider catalog exposed by OpenAgent."""
    from aiohttp import web as _web
    from openagent.models.catalog import supported_providers

    db = _db(request)
    providers_cfg = (
        await db.materialise_providers_config(enabled_only=False) if db else []
    )
    return _web.json_response({"providers": supported_providers(providers_cfg)})


async def handle_catalog(request: web.Request) -> web.Response:
    """GET /api/models/catalog?provider=openai — configured models with live pricing."""
    from aiohttp import web as _web
    from openagent.models.catalog import get_model_pricing, iter_configured_models

    provider_filter = request.query.get("provider", "")
    db = _db(request)
    providers_cfg = (
        await db.materialise_providers_config(enabled_only=False) if db else []
    )
    results = []
    for entry in iter_configured_models(providers_cfg):
        if provider_filter and entry.provider != provider_filter:
            continue
        pricing = get_model_pricing(entry.runtime_id)
        results.append(
            {
                "provider": entry.provider,
                "framework": entry.framework,
                "model": entry.model_id,
                "runtime_id": entry.runtime_id,
                "history_mode": entry.history_mode,
                "tier_hint": entry.tier_hint,
                "input_cost_per_million": round(float(pricing["input_cost_per_million"] or 0.0), 4),
                "output_cost_per_million": round(float(pricing["output_cost_per_million"] or 0.0), 4),
            }
        )
    results.sort(key=lambda item: (item["provider"], item["framework"], item["input_cost_per_million"], item["model"]))
    return _web.json_response({"models": results})


# ──────────────────────────────────────────────────────────────────────
# DB-backed model CRUD. These endpoints hit the ``models`` table the
# model-manager MCP writes to; the gateway's hot-reload loop picks up
# changes on the next message.
# ──────────────────────────────────────────────────────────────────────


def _parse_model_id(request: "web.Request") -> int | None:
    raw = request.match_info.get("id")
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _shape_model(row: dict[str, Any]) -> dict[str, Any]:
    """Public response shape for an enriched model row."""
    from openagent.channels.tts import DEFAULT_VOICE_BY_VENDOR
    from openagent.models.catalog import get_model_pricing

    pricing = get_model_pricing(row["runtime_id"])
    kind = row.get("kind") or "llm"
    display_name = row.get("display_name")
    # Synthesize a display_name for audio rows that don't carry one — old
    # migrated rows, or anything added through APIs whose discovery layer
    # only returned a bare model id. The UI renders rows as
    # "<model> · <KIND> · <display_name>", so this keeps the column shape
    # consistent with LLM rows (which usually inherit a vendor-prefixed
    # display_name from OpenRouter, e.g. "OpenAI: GPT-5.4").
    if not display_name and kind in ("tts", "stt"):
        display_name = f"{row['provider_name']}: {row['model']}"
    # For TTS: surface the *effective* voice_id (configured or per-vendor
    # default) so the UI mirrors what the runtime actually plays. The
    # fallback chain matches ``tts.resolve_tts_provider``.
    metadata = dict(row.get("metadata") or {})
    if kind == "tts" and not metadata.get("voice_id"):
        default_voice = DEFAULT_VOICE_BY_VENDOR.get(row["provider_name"])
        if default_voice:
            metadata["voice_id"] = default_voice
            metadata["voice_id_source"] = "default"
    return {
        "id": row["id"],
        "provider_id": row["provider_id"],
        "provider_name": row["provider_name"],
        "framework": row["framework"],
        "kind": kind,
        "runtime_id": row["runtime_id"],
        "model": row["model"],
        "display_name": display_name,
        "tier_hint": row.get("tier_hint"),
        "enabled": bool(row.get("enabled", True)),
        "is_classifier": bool(row.get("is_classifier", False)),
        "provider_enabled": bool(row.get("provider_enabled", True)),
        "metadata": metadata,
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "input_cost_per_million": round(float(pricing["input_cost_per_million"] or 0.0), 4),
        "output_cost_per_million": round(float(pricing["output_cost_per_million"] or 0.0), 4),
    }


def _synthetic_piper_row() -> dict[str, Any]:
    """Surface the bundled Piper local TTS as a read-only catalog entry.

    Negative id (``-1``) so the universal client can detect it (
    ``id < 0`` → can't be edited or deleted via PUT/DELETE — those
    endpoints reject non-positive ids). The Voice tab uses the
    presence of any ``kind='tts'`` row to suppress its "no TTS"
    banner; this entry makes that check naturally pass when Piper is
    available.
    """
    from openagent.channels import tts_local

    voice = tts_local._resolve_voice_name(None)
    return {
        "id": -1,
        "synthetic": True,
        "provider_id": -1,
        "provider_name": "(local)",
        "framework": "local",
        "kind": "tts",
        "runtime_id": f"piper:{voice}",
        "model": "piper",
        "display_name": f"(local) Piper · {voice}",
        "tier_hint": "Bundled offline TTS — auto-downloads ~25 MB on first use",
        "enabled": True,
        "is_classifier": False,
        "provider_enabled": True,
        "metadata": {"local": True, "voice_id": voice, "voice_id_source": "default"},
        "created_at": None,
        "updated_at": None,
        "input_cost_per_million": 0.0,
        "output_cost_per_million": 0.0,
    }


async def handle_list_db(request: web.Request) -> web.Response:
    """GET /api/models — list every configured model row with live pricing.

    Query params:
      - ``provider_id`` (int) — filter to a single provider row
      - ``framework`` (``agno`` / ``claude-cli``) — filter by framework
      - ``enabled_only`` (bool) — skip disabled model rows

    When Piper is importable AND no kind='tts' row exists yet, a
    synthetic local-Piper TTS row is appended so the Voice tab knows
    there's a working backend even before the user adds a cloud row.
    """
    from aiohttp import web as _web
    from openagent.channels import tts_local

    db = _db(request)
    if db is None:
        return _web.json_response({"error": "memory DB not available"}, status=500)
    framework = request.query.get("framework") or None
    provider_id_raw = request.query.get("provider_id")
    provider_id: int | None = None
    if provider_id_raw:
        try:
            provider_id = int(provider_id_raw)
        except ValueError:
            return _web.json_response({"error": "invalid provider_id"}, status=400)
    enabled_only = request.query.get("enabled_only", "").lower() in ("1", "true", "yes")

    rows = await db.list_models_enriched(
        enabled_only=enabled_only,
        framework=framework,
        provider_id=provider_id,
    )
    shaped = [_shape_model(r) for r in rows]

    # Append the synthetic Piper row only when no real TTS row exists
    # AND no provider/framework filter is in play (those filters
    # implicitly target a specific real provider, where the synthetic
    # entry would be confusing). Skip when piper isn't installed —
    # ``resolve_tts_provider`` would fall through to text-only anyway,
    # so don't lie to the UI about audio availability.
    if (
        provider_id is None
        and framework is None
        and tts_local.is_available()
        and not any(m.get("kind") == "tts" for m in shaped)
    ):
        shaped.append(_synthetic_piper_row())

    return _web.json_response({"models": shaped})


async def handle_get_db(request: web.Request) -> web.Response:
    from aiohttp import web as _web

    db = _db(request)
    mid = _parse_model_id(request)
    if db is None or mid is None:
        return _web.json_response({"error": "invalid model id"}, status=400)
    row = await db.get_model_enriched(mid)
    if row is None:
        return _web.json_response({"error": f"model id={mid} not found"}, status=404)
    return _web.json_response({"model": _shape_model(row)})


async def handle_create_db(request: web.Request) -> web.Response:
    from aiohttp import web as _web

    db = _db(request)
    if db is None:
        return _web.json_response({"error": "memory DB not available"}, status=500)
    body = await request.json() if request.can_read_body else {}
    provider_id_raw = body.get("provider_id")
    model = str(body.get("model") or body.get("model_id") or "").strip()
    if provider_id_raw is None or not model:
        return _web.json_response(
            {"error": "provider_id and model are required"}, status=400,
        )
    try:
        provider_id = int(provider_id_raw)
    except (TypeError, ValueError):
        return _web.json_response({"error": "provider_id must be an integer"}, status=400)
    kind = (body.get("kind") or "llm").strip()
    if kind not in ("llm", "tts", "stt"):
        return _web.json_response({"error": "kind must be llm/tts/stt"}, status=400)
    try:
        mid = await db.upsert_model(
            provider_id=provider_id,
            model=model,
            display_name=body.get("display_name"),
            tier_hint=body.get("tier_hint") or body.get("notes"),
            enabled=bool(body.get("enabled", True)),
            is_classifier=bool(body.get("is_classifier", False)),
            metadata=body.get("metadata") or None,
            kind=kind,
        )
    except ValueError as e:
        return _web.json_response({"error": str(e)}, status=400)
    # is_classifier is persisted by upsert_model directly; multiple
    # rows are allowed to carry the flag, so no clear-others step.
    enriched = await db.get_model_enriched(mid)
    return _web.json_response(
        {"ok": True, "model": _shape_model(enriched) if enriched else {"id": mid}},
        status=201,
    )


async def handle_update_db(request: web.Request) -> web.Response:
    from aiohttp import web as _web

    db = _db(request)
    mid = _parse_model_id(request)
    if db is None or mid is None:
        return _web.json_response({"error": "invalid model id"}, status=400)
    existing = await db.get_model(mid)
    if existing is None:
        return _web.json_response({"error": f"model id={mid} not found"}, status=404)
    body = await request.json() if request.can_read_body else {}
    # Multi-classifier semantics: each row's flag is independent. Body
    # can omit the field (preserve existing value) or pass a bool to
    # toggle it on this row only — never touches other rows.
    desired_classifier = body.get("is_classifier")
    try:
        await db.upsert_model(
            provider_id=existing["provider_id"],
            model=body.get("model", existing["model"]),
            display_name=body.get("display_name", existing.get("display_name")),
            tier_hint=body.get("tier_hint", existing.get("tier_hint")),
            enabled=bool(body.get("enabled", existing.get("enabled", True))),
            is_classifier=(
                bool(desired_classifier)
                if desired_classifier is not None
                else bool(existing.get("is_classifier", False))
            ),
            metadata=body.get("metadata", existing.get("metadata") or None),
            kind=existing.get("kind", "llm"),
        )
    except ValueError as e:
        return _web.json_response({"error": str(e)}, status=400)
    enriched = await db.get_model_enriched(mid)
    return _web.json_response(
        {"ok": True, "model": _shape_model(enriched) if enriched else {"id": mid}},
    )


async def handle_delete_db(request: web.Request) -> web.Response:
    from aiohttp import web as _web

    db = _db(request)
    mid = _parse_model_id(request)
    if db is None or mid is None:
        return _web.json_response({"error": "invalid model id"}, status=400)
    existing = await db.get_model(mid)
    if existing is None:
        return _web.json_response({"error": f"model id={mid} not found"}, status=404)
    # Deleting the last enabled row is allowed — the rejection gate in
    # gateway/server.py will then surface a clear "No models are enabled"
    # error on the next message, which is what the user wants when they
    # intentionally empty the catalog.
    await db.delete_model(mid)
    return _web.json_response({"ok": True})


async def handle_enable_db(request: web.Request) -> web.Response:
    return await _toggle_model(request, True)


async def handle_disable_db(request: web.Request) -> web.Response:
    return await _toggle_model(request, False)


async def _toggle_model(request: web.Request, enabled: bool) -> web.Response:
    from aiohttp import web as _web

    db = _db(request)
    mid = _parse_model_id(request)
    if db is None or mid is None:
        return _web.json_response({"error": "invalid model id"}, status=400)
    if await db.get_model(mid) is None:
        return _web.json_response({"error": f"model id={mid} not found"}, status=404)
    await db.set_model_enabled(mid, enabled)
    enriched = await db.get_model_enriched(mid)
    return _web.json_response(
        {"ok": True, "model": _shape_model(enriched) if enriched else {"id": mid}},
    )


async def handle_available_models(request: web.Request) -> web.Response:
    """GET /api/models/available?provider_id=N

    Dynamic provider catalog: tries the provider's /v1/models endpoint
    with the configured API key, falls back to the bundled catalog.
    Claude-cli providers (no api_key) still return their bundled list
    through discovery's offline path.
    """
    import asyncio
    from aiohttp import web as _web
    from openagent.models.catalog import build_runtime_model_id
    from openagent.models.discovery import list_provider_models

    db = _db(request)
    if db is None:
        return _web.json_response({"error": "memory DB not available"}, status=500)

    provider_id_raw = request.query.get("provider_id")
    provider_name_q = (request.query.get("provider") or "").strip()
    if provider_id_raw:
        try:
            pid = int(provider_id_raw)
        except ValueError:
            return _web.json_response({"error": "invalid provider_id"}, status=400)
        provider_row = await db.get_provider(pid)
    elif provider_name_q:
        # Back-compat: ``?provider=openai`` without framework. If the user
        # has multiple rows under this name, pick the first (stable by id).
        candidates = [
            p for p in await db.list_providers(kind="llm")
            if p["name"] == provider_name_q
        ]
        if not candidates:
            return _web.json_response({"error": f"no provider named {provider_name_q!r}"}, status=404)
        provider_row = candidates[0]
    else:
        return _web.json_response(
            {"error": "provider_id (preferred) or provider query param required"},
            status=400,
        )
    if provider_row is None:
        return _web.json_response({"error": "provider not found"}, status=404)

    discovery_task = list_provider_models(
        provider_row["name"],
        api_key=provider_row.get("api_key"),
        base_url=provider_row.get("base_url"),
    )
    db_task = db.list_models(provider_id=provider_row["id"])
    models_list, db_rows = await asyncio.gather(discovery_task, db_task)
    configured = {r["model"] for r in db_rows}
    for m in models_list:
        m["runtime_id"] = build_runtime_model_id(
            provider_row["name"], m["id"], provider_row["framework"],
        )
        m["added"] = m["id"] in configured

    return _web.json_response({
        "provider_id": provider_row["id"],
        "provider": provider_row["name"],
        "framework": provider_row["framework"],
        "models": models_list,
    })
