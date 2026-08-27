"""Events REST API — CRUD for the webhook Events channel + deliveries.

GET    /api/events                      → { "events": [...] }
POST   /api/events                      → created event + one-time secret (201)
GET    /api/events/{id}                 → event | 404
PATCH  /api/events/{id}                 → updated event | 404
DELETE /api/events/{id}                 → { "ok": true, "id" } | 404
POST   /api/events/{id}/rotate-secret   → { ...event, secret } (one-time)
POST   /api/events/{id}/trigger         → dispatch now (peer/manual delivery)
GET    /api/events/{id}/deliveries      → { "deliveries": [...] }
GET    /api/event-deliveries/{id}       → delivery | 404
GET    /api/event-types                 → provider presets for the type picker

The secret is returned in CLEAR exactly once — on create and on rotate. Every
read returns only ``secret_hint``. All handlers go through the same device-cert
auth middleware as the rest of ``/api/*``; this is the surface the app, the CLI
and Iroh peers use. The unauthenticated external doorway is the separate
``/hooks/{slug}`` listener (see ``gateway/webhook_site.py``).
"""

from __future__ import annotations

import asyncio

from src.core.event_secret import (
    make_secret_material, slugify, random_slug_suffix,
)
from src.core.event_types import is_valid_type, public_types, DEFAULT_TYPE
from src.core.execution_policy import encode_execution_policy, normalize_execution_policy
from src.core.logging import elog

_VALID_ACTION_KINDS = ("workflow", "scheduled_task", "prompt")


def _db(request):
    return request.app["gateway"].agent.memory_db


def _db_path(request):
    db = _db(request)
    return getattr(db, "db_path", None) if db is not None else None


def _require_db(request):
    from aiohttp import web
    db = _db(request)
    if db is None:
        return None, web.json_response({"error": "No database configured"}, status=503)
    return db, None


def _public_url(request) -> str | None:
    """The configured webhook public base URL, for building an event's hook
    URL in responses. Read from ``channels.webhook.public_url``; None when the
    listener isn't configured (the app then shows setup instructions)."""
    from .config import _read_resolved
    try:
        cfg = _read_resolved(request)
    except Exception:  # noqa: BLE001
        return None
    wh = ((cfg or {}).get("channels") or {}).get("webhook") or {}
    base = (wh.get("public_url") or "").strip()
    return base.rstrip("/") or None


def _serialize_event(request, ev: dict) -> dict:
    out = dict(ev)
    base = _public_url(request)
    out["webhook_path"] = f"/hooks/{ev.get('slug')}"
    out["webhook_url"] = f"{base}/hooks/{ev.get('slug')}" if base else None
    return out


async def _unique_slug(db, name: str, *, desired: str | None = None) -> str:
    slug = slugify(desired or name)
    if not await db.slug_exists(slug):
        return slug
    for _ in range(5):
        candidate = f"{slug}-{random_slug_suffix()}"
        if not await db.slug_exists(candidate):
            return candidate
    return f"{slug}-{random_slug_suffix()}"


async def handle_list(request):
    from aiohttp import web
    db, err = _require_db(request)
    if err is not None:
        return err
    enabled_only = request.query.get("enabled_only", "").lower() in ("1", "true", "yes")
    rows = await db.list_events(enabled_only=enabled_only)
    return web.json_response({"events": [_serialize_event(request, r) for r in rows]})


async def handle_get(request):
    from aiohttp import web
    db, err = _require_db(request)
    if err is not None:
        return err
    ev = await db.get_event(request.match_info["id"])
    if ev is None:
        return web.json_response({"error": "Event not found"}, status=404)
    return web.json_response(_serialize_event(request, ev))


def _validate_action(body) -> tuple[str, str | None, str | None, str | None]:
    """Return (action_kind, action_ref, prompt_template, error)."""
    action_kind = (body.get("action_kind") or "").strip()
    if action_kind not in _VALID_ACTION_KINDS:
        return "", None, None, f"action_kind must be one of {_VALID_ACTION_KINDS}"
    action_ref = (body.get("action_ref") or "").strip() or None
    prompt_template = body.get("prompt_template") or None
    if action_kind in ("workflow", "scheduled_task") and not action_ref:
        return "", None, None, f"action_ref is required for action_kind={action_kind}"
    if action_kind == "prompt" and not (prompt_template and str(prompt_template).strip()):
        return "", None, None, "prompt_template is required for action_kind=prompt"
    return action_kind, action_ref, prompt_template, None


def _validate_session_binding(body) -> tuple[bool | None, str | None, str | None]:
    """Return (enabled?, path?, error) for optional event-session binding."""
    enabled = None
    if "session_binding_enabled" in body:
        enabled = bool(body.get("session_binding_enabled"))
    raw_path = body.get("session_binding_path")
    path = (str(raw_path).strip() if raw_path is not None else "") or None
    effective_enabled = bool(enabled)
    if enabled is None and raw_path is not None:
        effective_enabled = bool(body.get("session_binding_enabled", False))
    if effective_enabled and not path:
        return enabled, path, "session_binding_path is required when session binding is enabled"
    return enabled, path, None


async def handle_create(request):
    from aiohttp import web
    db, err = _require_db(request)
    if err is not None:
        return err
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    name = (body.get("name") or "").strip()
    if not name:
        return web.json_response({"error": "name is required"}, status=400)

    event_type = (body.get("type") or DEFAULT_TYPE).strip()
    if not is_valid_type(event_type):
        return web.json_response({"error": f"unknown type {event_type!r}"}, status=400)

    action_kind, action_ref, prompt_template, verr = _validate_action(body)
    if verr:
        return web.json_response({"error": verr}, status=400)
    binding_enabled, binding_path, berr = _validate_session_binding(body)
    if berr:
        return web.json_response({"error": berr}, status=400)
    try:
        execution_policy = normalize_execution_policy(body.get("execution_policy"))
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    # Referenced workflow / task must exist so a broken binding can't be saved.
    if action_kind == "workflow" and await db.get_workflow(action_ref) is None:
        return web.json_response({"error": f"workflow {action_ref!r} not found"}, status=400)
    if action_kind == "scheduled_task" and await db.get_task(action_ref) is None:
        return web.json_response({"error": f"scheduled task {action_ref!r} not found"}, status=400)

    slug = await _unique_slug(db, name, desired=body.get("slug"))
    clear, secret_enc, hint = make_secret_material(db_path=_db_path(request))

    event_id = await db.add_event(
        name=name,
        action_kind=action_kind,
        slug=slug,
        secret_enc=secret_enc,
        secret_hint=hint,
        event_type=event_type,
        description=(body.get("description") or None),
        input_schema=body.get("input_schema") or [],
        action_ref=action_ref,
        prompt_template=prompt_template,
        model=(body.get("model") or "").strip() or None,
        session_binding_enabled=bool(binding_enabled),
        session_binding_path=binding_path,
        execution_policy=execution_policy,
        rate_limit_per_min=int(body.get("rate_limit_per_min", 60)),
        max_payload_bytes=int(body.get("max_payload_bytes", 262144)),
        enabled=bool(body.get("enabled", True)),
    )
    ev = await db.get_event(event_id)
    elog("event.create", id=event_id, name=name, type=event_type, action=action_kind)
    await request.app["gateway"].broadcast_resource("event", "created", event_id)
    out = _serialize_event(request, ev)
    out["secret"] = clear  # one-time reveal
    return web.json_response(out, status=201)


async def handle_update(request):
    from aiohttp import web
    db, err = _require_db(request)
    if err is not None:
        return err
    event_id = request.match_info["id"]
    existing = await db.get_event(event_id)
    if existing is None:
        return web.json_response({"error": "Event not found"}, status=404)
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    updates: dict = {}
    for key in ("name", "description", "prompt_template", "model"):
        if key in body:
            updates[key] = body[key]
    if "type" in body:
        if not is_valid_type(body["type"]):
            return web.json_response({"error": f"unknown type {body['type']!r}"}, status=400)
        updates["type"] = body["type"]
    if "enabled" in body:
        updates["enabled"] = bool(body["enabled"])
    if "input_schema" in body:
        updates["input_schema"] = body["input_schema"] or []
    if "execution_policy" in body:
        try:
            updates["execution_policy_json"] = encode_execution_policy(
                body.get("execution_policy")
            )
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
    if any(k in body for k in ("session_binding_enabled", "session_binding_path")):
        merged_binding = {
            "session_binding_enabled": body.get(
                "session_binding_enabled",
                existing.get("session_binding_enabled"),
            ),
            "session_binding_path": body.get(
                "session_binding_path",
                existing.get("session_binding_path"),
            ),
        }
        binding_enabled, binding_path, berr = _validate_session_binding(merged_binding)
        if berr:
            return web.json_response({"error": berr}, status=400)
        updates["session_binding_enabled"] = bool(binding_enabled)
        updates["session_binding_path"] = binding_path
    for key in ("rate_limit_per_min", "max_payload_bytes"):
        if key in body:
            try:
                updates[key] = int(body[key])
            except (TypeError, ValueError):
                return web.json_response({"error": f"{key} must be an integer"}, status=400)
    # Action rebinding — validate as a unit if any action field is present.
    if any(k in body for k in ("action_kind", "action_ref", "prompt_template")):
        merged = {
            "action_kind": body.get("action_kind", existing.get("action_kind")),
            "action_ref": body.get("action_ref", existing.get("action_ref")),
            "prompt_template": body.get("prompt_template", existing.get("prompt_template")),
        }
        action_kind, action_ref, prompt_template, verr = _validate_action(merged)
        if verr:
            return web.json_response({"error": verr}, status=400)
        if action_kind == "workflow" and await db.get_workflow(action_ref) is None:
            return web.json_response({"error": f"workflow {action_ref!r} not found"}, status=400)
        if action_kind == "scheduled_task" and await db.get_task(action_ref) is None:
            return web.json_response({"error": f"scheduled task {action_ref!r} not found"}, status=400)
        updates["action_kind"] = action_kind
        updates["action_ref"] = action_ref
        updates["prompt_template"] = prompt_template

    if not updates:
        return web.json_response({"error": "No fields to update"}, status=400)

    await db.update_event(event_id, **updates)
    ev = await db.get_event(event_id)
    elog("event.update", id=event_id, fields=list(updates.keys()))
    await request.app["gateway"].broadcast_resource("event", "updated", event_id)
    return web.json_response(_serialize_event(request, ev))


async def handle_delete(request):
    from aiohttp import web
    db, err = _require_db(request)
    if err is not None:
        return err
    event_id = request.match_info["id"]
    if await db.get_event(event_id) is None:
        return web.json_response({"error": "Event not found"}, status=404)
    await db.delete_event(event_id)
    elog("event.delete", id=event_id)
    await request.app["gateway"].broadcast_resource("event", "deleted", event_id)
    return web.json_response({"ok": True, "id": event_id})


async def handle_rotate_secret(request):
    from aiohttp import web
    db, err = _require_db(request)
    if err is not None:
        return err
    event_id = request.match_info["id"]
    if await db.get_event(event_id) is None:
        return web.json_response({"error": "Event not found"}, status=404)
    clear, secret_enc, hint = make_secret_material(db_path=_db_path(request))
    await db.rotate_event_secret(event_id, secret_enc=secret_enc, secret_hint=hint)
    ev = await db.get_event(event_id)
    elog("event.rotate_secret", id=event_id)
    await request.app["gateway"].broadcast_resource("event", "updated", event_id)
    out = _serialize_event(request, ev)
    out["secret"] = clear
    return web.json_response(out)


async def handle_trigger(request):
    """POST /api/events/{id}/trigger — dispatch an event now from an
    authenticated caller (an Iroh peer/agent, or the app/CLI Test button).
    Body: ``{payload, wait, timeout_s, source}``. Bypasses webhook auth (the
    caller already presented a device cert / agent handshake)."""
    from aiohttp import web
    db, err = _require_db(request)
    if err is not None:
        return err
    event_id = request.match_info["id"]
    ev = await db.get_event(event_id)
    if ev is None:
        return web.json_response({"error": "Event not found"}, status=404)

    try:
        body = await request.json() if request.can_read_body else {}
    except Exception:
        body = {}
    payload = body.get("payload") or {}
    if not isinstance(payload, dict):
        return web.json_response({"error": "payload must be an object"}, status=400)
    wait = body.get("wait", True)
    timeout_s = int(body.get("timeout_s", 120))
    # An agent peer triggering over ALPN is 'peer'; an app/CLI test is 'manual'.
    is_agent = bool(request.get("device_cert") and getattr(request["device_cert"], "capabilities", None) and "agent" in request["device_cert"].capabilities)
    source = body.get("source") or ("peer" if is_agent else "manual")

    gw = request.app["gateway"]
    scheduler = getattr(gw, "_scheduler", None)
    if scheduler is None:
        return web.json_response({"error": "Scheduler is not running"}, status=503)

    delivery_id = await db.add_event_delivery(
        event_id=event_id, source=source, payload=payload, claimed=True,
    )
    elog("event.trigger", id=event_id, source=source, delivery=delivery_id)

    from src.core.event_dispatcher import dispatch_event
    coro = dispatch_event(
        agent=gw.agent, db=db, scheduler=scheduler,
        event=ev, payload=payload, delivery_id=delivery_id, source=source,
        broadcast=gw.broadcast_resource_sync,
    )
    task = scheduler._spawn_workflow(coro)
    if not wait:
        return web.json_response({"delivery_id": delivery_id, "status": "running"}, status=202)
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
    except asyncio.TimeoutError:
        return web.json_response(
            {"delivery_id": delivery_id, "status": "running",
             "note": f"still running after {timeout_s}s"}, status=202,
        )
    except Exception as e:  # noqa: BLE001 — dispatch records its own failure
        elog("event.trigger_failed", id=event_id, delivery=delivery_id, error=str(e))
    row = await db.get_event_delivery(delivery_id)
    return web.json_response(row or {"delivery_id": delivery_id})


async def handle_deliveries_list(request):
    from aiohttp import web
    db, err = _require_db(request)
    if err is not None:
        return err
    event_id = request.match_info["id"]
    try:
        limit = int(request.query.get("limit", 20))
    except ValueError:
        limit = 20
    rows = await db.list_event_deliveries(event_id, limit=limit)
    return web.json_response({"deliveries": rows})


async def handle_delivery_get(request):
    from aiohttp import web
    db, err = _require_db(request)
    if err is not None:
        return err
    row = await db.get_event_delivery(request.match_info["id"])
    if row is None:
        return web.json_response({"error": "Delivery not found"}, status=404)
    return web.json_response(row)


async def handle_types(request):
    from aiohttp import web
    return web.json_response({"types": public_types()})
