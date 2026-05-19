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
    """GET /api/network/agents — agents registered on this network.

    Order matches what default-picking clients (CLI / app) see: the
    coordinator's own agent first, then federated entries by
    registration time. See ``CoordinatorStore.list_agents``.
    """
    from aiohttp import web

    result = await _store_or_404(request)
    if isinstance(result, web.Response):
        return result
    store, _row = result
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
