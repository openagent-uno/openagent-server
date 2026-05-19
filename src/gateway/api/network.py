"""``/api/network/*`` — viewer + invite-minting endpoints.

Three things the desktop app and the openagent-cli need from the
coordinator without dropping to a shell:

  - GET  /api/network/users         — who's registered on this network
  - GET  /api/network/agents        — what agents users can talk to
  - GET  /api/network/invitations   — active (unspent, unexpired) invites
  - POST /api/network/invitations   — mint a new invite (smart auto-pick)
  - DELETE /api/network/invitations/{code} — revoke a still-unused invite

All endpoints require a valid device cert (the gateway-wide auth
middleware handles that before the handler runs). The endpoints are
coordinator-only: member-mode agents don't hold the invite table and
return 404 — clients should ask the network's coordinator directly.

Authorization: any authenticated network member can read the lists
(they were already invited into this network — the user/agent
directory isn't a secret) and mint invites. Tighter scoping (per-
user/admin roles) is a future change; today every user can onboard
their friends or pair their own new devices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web

from src.gateway.api._common import gateway_db


async def _store_or_404(request: "web.Request"):
    """Open a ``CoordinatorStore`` against this agent's DB or return
    a 404 ``web.Response`` if this agent isn't a coordinator (members
    don't keep the invite/user tables locally).

    Returns either ``(store, network_row)`` or ``web.Response``.
    """
    from aiohttp import web
    from src.network.coordinator.store import CoordinatorStore

    db = gateway_db(request)
    if db is None:
        return web.json_response(
            {"error": "agent DB not available"}, status=503,
        )
    store = CoordinatorStore(db)
    row = await store.get_network_role()
    if row is None or row["role"] != "coordinator":
        return web.json_response(
            {
                "error": "not a coordinator-mode agent",
                "hint": "user / agent / invite management lives on the "
                        "network's coordinator; this agent is a member.",
            },
            status=404,
        )
    return store, row


async def handle_list_users(request: "web.Request") -> "web.Response":
    """GET /api/network/users — every registered network member."""
    from aiohttp import web

    result = await _store_or_404(request)
    if isinstance(result, web.Response):
        return result
    store, _row = result
    users = await store.list_users()
    return web.json_response({
        "users": [
            {
                "handle": u.handle,
                "status": u.status,
                "pake_algo": u.pake_algo,
                "created_at": u.created_at,
            }
            for u in users
        ],
    })


async def handle_list_agents(request: "web.Request") -> "web.Response":
    """GET /api/network/agents — *federated* agents on this network.

    Filters out the row whose ``node_id == coordinator_node_id`` —
    that's the agent the caller is already talking to, and is
    routing/storage state, not a manageable peer. The desktop
    Members tab and the ``openagent-cli agents`` view both render
    this list directly; without the filter they'd surface a
    "delete" affordance that 409s on every click.

    The underlying ``network_agents`` row is *kept* (the coordinator
    RPC ``list_agents`` still returns it — that's how new clients
    discover the gateway target on first contact). Only the
    authenticated-admin view hides it.
    """
    from aiohttp import web

    result = await _store_or_404(request)
    if isinstance(result, web.Response):
        return result
    store, row = result
    coord_node_id = (row or {}).get("coordinator_node_id") or ""
    agents = await store.list_agents()
    return web.json_response({
        "agents": [
            {
                "handle": a.handle,
                "node_id": a.node_id,
                "label": a.label,
                "owner_handle": a.owner_handle,
                "added_at": a.added_at,
                "last_seen": a.last_seen,
            }
            for a in agents
            if a.node_id != coord_node_id
        ],
    })


async def handle_list_invitations(request: "web.Request") -> "web.Response":
    """GET /api/network/invitations — unspent, unexpired invites only.

    Skips burned/expired entries — those are operator forensics
    questions, not user-facing UI questions. If a future audit
    surface needs them, expose ``?include_spent=1`` then.
    """
    from aiohttp import web

    result = await _store_or_404(request)
    if isinstance(result, web.Response):
        return result
    store, _row = result
    invites = await store.list_invitations(include_expired=False)
    out = []
    for inv in invites:
        if inv.uses_left <= 0:
            continue
        out.append({
            "code": inv.code,
            "role": inv.role,
            "bind_to": inv.bind_to_handle or "",
            "uses_left": inv.uses_left,
            "created_at": inv.created_at,
            "expires_at": inv.expires_at,
            "created_by": inv.created_by,
        })
    return web.json_response({"invitations": out})


async def handle_mint_invitation(request: "web.Request") -> "web.Response":
    """POST /api/network/invitations — auto-pick role from handle.

    Body (all fields optional):
        {
          "handle": "marco",   # if exists → device-bound; else → user-role
          "role":   "user",    # advanced: force role; bypasses auto-detect
          "ttl":    604800,    # seconds, defaults to 7 days
          "uses":   1,         # advanced: redemption count
        }

    Returns:
        {
          "ticket": "oa1...",            # the full string the recipient pastes
          "code": "xxxxxxxxxxxxxxxxxxxx",
          "role": "user|device|agent",
          "bind_to": "marco" | "",
          "intent": "onboard marco (new user)",  # operator-readable label
          "expires_at": <unix-seconds>,
        }
    """
    from aiohttp import web

    from src.core import paths as core_paths
    from src.network.cli_commands import resolve_invite_intent
    from src.network.coordinator_addr_cache import read_cache
    from src.network.identity import load_or_create_identity
    from src.network.iroh_node import _node_id_from_secret
    from src.network.ticket import InviteTicket

    result = await _store_or_404(request)
    if isinstance(result, web.Response):
        return result
    store, row = result

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    raw_handle = body.get("handle")
    handle = raw_handle if isinstance(raw_handle, str) else None
    raw_role = body.get("role")
    role = raw_role if isinstance(raw_role, str) else None
    if role is not None and role not in ("user", "device", "agent"):
        return web.json_response(
            {"error": f"invalid role: {role!r}; must be user|device|agent"},
            status=400,
        )
    ttl = body.get("ttl", 7 * 24 * 3600)
    if not isinstance(ttl, int) or ttl <= 0 or ttl > 90 * 24 * 3600:
        return web.json_response(
            {"error": "ttl must be a positive int ≤ 90 days"},
            status=400,
        )
    uses = body.get("uses", 1)
    if not isinstance(uses, int) or uses < 1 or uses > 1000:
        return web.json_response(
            {"error": "uses must be an int in [1, 1000]"},
            status=400,
        )

    resolved_role, bind_to, intent = await resolve_invite_intent(
        store, handle=handle, role=role,
    )

    minted_by = request.get("user_handle") or "gateway"
    invite = await store.create_invitation(
        role=resolved_role,
        created_by=f"gateway:{minted_by}",
        ttl_seconds=ttl,
        uses=uses,
        bind_to_handle=bind_to,
    )

    # Tickets need the coordinator's NodeId + optional address hints.
    # ``get_agent_dir`` is set early at serve-time, so this is a cheap
    # read (no FS scan).
    agent_dir = core_paths.get_agent_dir()
    if agent_dir is None:
        # Shouldn't happen for a running coordinator gateway, but if
        # something's mis-wired we still want a useful response.
        return web.json_response(
            {"error": "no agent_dir active — server mis-wired"},
            status=500,
        )
    identity = load_or_create_identity(agent_dir / "identity.key")
    relay_url, addresses = read_cache(agent_dir)
    ticket = InviteTicket(
        code=invite.code,
        coordinator_node_id=_node_id_from_secret(identity.secret_bytes),
        network_name=row["name"],
        network_id=row["network_id"],
        role=invite.role,
        bind_to=bind_to or "",
        relay_url=relay_url,
        addresses=addresses or None,
    )

    return web.json_response({
        "ticket": ticket.encode(),
        "code": invite.code,
        "role": invite.role,
        "bind_to": invite.bind_to_handle or "",
        "intent": intent,
        "expires_at": invite.expires_at,
        "uses_left": invite.uses_left,
    }, status=201)


async def handle_patch_user(request: "web.Request") -> "web.Response":
    """PATCH /api/network/users/{handle} — flip status (active/suspended).

    Soft-delete: leaves the account + devices in place so the operator
    can re-activate without re-onboarding. Body: ``{"status": "active"
    | "suspended"}``.
    """
    from aiohttp import web

    result = await _store_or_404(request)
    if isinstance(result, web.Response):
        return result
    store, _row = result

    handle = (request.match_info.get("handle") or "").strip().lower()
    if not handle:
        return web.json_response({"error": "missing handle"}, status=400)

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    status = body.get("status")
    if status not in ("active", "suspended"):
        return web.json_response(
            {"error": "status must be 'active' or 'suspended'"},
            status=400,
        )

    user = await store.get_user(handle)
    if user is None:
        return web.json_response({"error": f"user {handle!r} not found"}, status=404)
    updated = await store.set_user_status(handle, status)
    return web.json_response({"handle": handle, "status": status, "updated": updated})


async def handle_delete_user(request: "web.Request") -> "web.Response":
    """DELETE /api/network/users/{handle} — hard-delete a user + devices.

    Cascades into ``network_devices`` so existing certs for this user
    stop verifying at the auth middleware (cert sig still valid; row
    lookup fails). Existing live WS sessions stay open until they
    drop; new connections are rejected.

    Guard: an authenticated user can't delete themselves through this
    endpoint — kicking yourself off mid-request would race the
    response. Operators who really need to remove their own account
    can do it from a *different* user's session, or by hand-editing
    the DB while the agent is stopped.
    """
    from aiohttp import web

    result = await _store_or_404(request)
    if isinstance(result, web.Response):
        return result
    store, _row = result

    handle = (request.match_info.get("handle") or "").strip().lower()
    if not handle:
        return web.json_response({"error": "missing handle"}, status=400)

    caller = (request.get("user_handle") or "").strip().lower()
    if caller == handle:
        return web.json_response(
            {
                "error": "can't delete yourself through the API",
                "hint": "use a different account to remove this one, "
                        "or stop the agent and edit the DB by hand.",
            },
            status=409,
        )

    deleted = await store.delete_user(handle)
    if not deleted:
        return web.json_response({"error": f"user {handle!r} not found"}, status=404)
    return web.json_response({"deleted": True, "handle": handle})


async def handle_patch_agent(request: "web.Request") -> "web.Response":
    """PATCH /api/network/agents/{handle} — edit label and/or owner_handle.

    ``handle`` and ``node_id`` are intentionally not editable here —
    they're the routing key clients cache; changing them silently
    breaks every paired device's local agent picker. If you need to
    re-key an agent, delete + re-register (and accept that clients
    will need to re-resolve their target).
    """
    from aiohttp import web

    result = await _store_or_404(request)
    if isinstance(result, web.Response):
        return result
    store, _row = result

    handle = (request.match_info.get("handle") or "").strip().lower()
    if not handle:
        return web.json_response({"error": "missing handle"}, status=400)

    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    label = body.get("label")
    owner_handle = body.get("owner_handle")
    if label is not None and not isinstance(label, str):
        return web.json_response({"error": "label must be a string"}, status=400)
    if owner_handle is not None and not isinstance(owner_handle, str):
        return web.json_response(
            {"error": "owner_handle must be a string"},
            status=400,
        )
    if label is None and owner_handle is None:
        return web.json_response(
            {"error": "PATCH must include at least one of: label, owner_handle"},
            status=400,
        )

    updated = await store.update_agent(
        handle,
        label=label.strip() if isinstance(label, str) else None,
        owner_handle=(owner_handle.strip().lower()
                      if isinstance(owner_handle, str) and owner_handle.strip()
                      else None),
    )
    if not updated:
        return web.json_response({"error": f"agent {handle!r} not found"}, status=404)
    return web.json_response({"updated": True, "handle": handle})


async def handle_delete_agent(request: "web.Request") -> "web.Response":
    """DELETE /api/network/agents/{handle} — drop an agent from the registry.

    Guard: the agent whose ``node_id`` matches this coordinator's own
    NodeId is the network's only working gateway — deleting it
    strands every client (list_agents returns empty → no agent to
    pick → WS dials fail). Block that path. Foreign-network rows
    (the failure mode that surfaced as WS code 1006 on lyra) can be
    removed freely.
    """
    from aiohttp import web

    result = await _store_or_404(request)
    if isinstance(result, web.Response):
        return result
    store, net_row = result

    handle = (request.match_info.get("handle") or "").strip().lower()
    if not handle:
        return web.json_response({"error": "missing handle"}, status=400)

    # Probe the row before deleting so we can refuse the self-agent
    # case with a clear message instead of just rejecting the call.
    agents = await store.list_agents()
    target = next((a for a in agents if a.handle == handle), None)
    if target is None:
        return web.json_response({"error": f"agent {handle!r} not found"}, status=404)
    coord_node_id = (net_row or {}).get("coordinator_node_id") or ""
    if target.node_id == coord_node_id:
        return web.json_response(
            {
                "error": "refusing to delete this agent: it's the "
                         "coordinator's own gateway — removing it would "
                         "strand every paired device.",
                "hint": "use ``openagent network init`` on a fresh agent "
                        "if you want to migrate to a different one.",
            },
            status=409,
        )

    deleted = await store.remove_agent(handle)
    return web.json_response({"deleted": deleted, "handle": handle})


async def handle_revoke_invitation(request: "web.Request") -> "web.Response":
    """DELETE /api/network/invitations/{code} — mark an invite spent.

    Idempotent: returning 200 with ``revoked: False`` when the code
    didn't match anything (or was already spent) lets the desktop app
    re-render its list without surfacing scary 404s for stale rows.
    """
    from aiohttp import web

    result = await _store_or_404(request)
    if isinstance(result, web.Response):
        return result
    store, _row = result

    code = request.match_info.get("code", "").strip()
    if not code:
        return web.json_response({"error": "missing code"}, status=400)

    # No dedicated ``revoke_invitation`` method exists yet; an UPDATE
    # to zero ``uses_left`` is functionally equivalent (the consume
    # path already short-circuits when ``uses_left <= 0``).
    conn = store._conn  # noqa: SLF001 — see store.py: this is a property
    cur = await conn.execute(
        "UPDATE network_invitations SET uses_left=0 "
        "WHERE code=? AND uses_left>0",
        (code,),
    )
    await conn.commit()
    return web.json_response({"revoked": (cur.rowcount or 0) > 0})
