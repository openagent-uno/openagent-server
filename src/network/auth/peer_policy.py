"""Federation peer policy — the enforcement half of ``network.peers``.

WHY THIS MODULE EXISTS
----------------------
``src/network/transport/agent_iroh_site.py`` accepts the
``openagent/agent/1`` ALPN and hands the stream to the SAME aiohttp
``AppRunner`` the gateway serves to first-party clients. Its own docstring
says so plainly: "Every route registered on the shared aiohttp app is
reachable". The auth middleware then synthesises a ``DeviceCert`` with
``capabilities=["agent"]`` for whoever dialled, and its own comment admitted
the consequence: "the agent-ALPN endpoint trusts any node that can dial it".

The Iroh QUIC handshake proves the peer owns its node_id. It does not prove
the peer is anyone we know. There is no inbound enrolment on this path — no
invite is consumed, no coordinator cert is checked — so "authenticated" here
means "has a keypair", which every host on the internet can arrange.

WHAT THAT ACTUALLY OPENED — MEASURE IT, DON'T ASSUME
----------------------------------------------------
It is tempting to read ``capabilities=["agent"]`` as a scope. It is not one.
Exactly two places in the tree look at that field, and NEITHER is an
authorisation check:

  - ``src/gateway/api/events.py:326`` — picks the delivery's ``source``
    label ("peer" vs "manual"). A provenance string for the audit trail.
  - ``src/gateway/api/chat.py:146``  — namespaces the session id
    (``peer:<node16>:…``) so peer threads don't collide on "default".

There is no ``require_capability`` helper anywhere in ``src/``. So the
capability is a *label*, and an unenrolled dialler reaches every route the
gateway registers, including:

  - ``POST /api/mcps``               — arbitrary argv, spawned by the pool's
                                       hot-reload → remote code execution,
                                       no marketplace round-trip needed.
  - ``POST /api/marketplace/install``— the same, via the registry.
  - ``GET/PUT /api/config``          — read and rewrite config.
  - ``PUT /api/vault/notes/{path}``  — read and rewrite memory.
  - ``POST /api/network/invitations``— mint an invite into the network.
  - ``POST /api/update``/``restart`` — control the process.

So the honest blast radius of the ALPN path is not "a peer can fire an
event". It is "a stranger who can dial this node owns it". ``/api/events/
{id}/trigger`` is one of roughly a hundred doors, and far from the worst.

TWO INDEPENDENT QUESTIONS, TWO TOGGLES
--------------------------------------
Conflating them produces a gate that is either useless or unshippable:

  1. WHO may dial            → ``allowlist``. Needs per-deployment data
                               (which node_ids are my peers?).
  2. WHAT a peer may reach   → ``scope``. Has a knowable correct answer,
                               derived from the code, independent of
                               deployment.

``scope`` is the one that shrinks the blast radius even for an enrolled peer
— vision §11 says federated agents "exchange messages, collaborate on tasks,
and share context with explicit permission", which is ``/api/chat`` and
``/api/events/{id}/trigger``. It never says a colleague may install software
on your host. An allowlist alone still hands every allowlisted peer root.

BOTH DEFAULT OFF, AND THAT IS LOAD-BEARING
------------------------------------------
``check_peer_request`` returns ``None`` before reading any list unless a
toggle is explicitly on. These agents run unattended in k8s; an allowlist
defaulting on would stop agents talking to each other with no warning, on
someone's nightly cron, at 3am. A protection that fires on real traffic
turns a silent risk into a live outage — so the off path is a literal no-op
you can read, not a property you infer from a test.

THE MIGRATION PATH ALREADY RAN
------------------------------
The reason this is enable-able on a live mesh without downtime: the
``agent.contact`` audit line in ``middleware.py`` has been recording every
dialling node_id since the ALPN shipped — it was added as "Phase-0 security:
record every first-contact agent node_id so the allowlist can be built /
audited". So the operator does not have to guess or schedule a window:

    grep '"event":"agent.contact"' ~/.openagent/events.jsonl \\
      | jq -r .node_id | sort -u

is the list of peers that actually dial them. Copy it into
``network.peers.allowlist.node_ids``, then flip ``enabled``. A peer that has
not dialled within the log's retention will not appear — which is why
``scope`` (no list to maintain) is the one to turn on first, and why
enforcement logs ``peer.denied`` rather than failing silently.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from src.core.logging import elog

_ALLOWLIST_ENABLED_ENV = "OPENAGENT_NETWORK_PEER_ALLOWLIST_ENABLED"
_ALLOWLIST_ENV = "OPENAGENT_NETWORK_PEER_ALLOWLIST"
_SCOPE_ENABLED_ENV = "OPENAGENT_NETWORK_PEER_SCOPE_ENABLED"
_SCOPE_EXTRA_PATHS_ENV = "OPENAGENT_NETWORK_PEER_SCOPE_EXTRA_PATHS"

# The routes federation actually uses over ``openagent/agent/1``. Derived
# from the code, not from taste — a wrong entry here is a broken mesh the
# moment someone enables the scope:
#
#   POST /api/chat                  ``src/network/peers.py:569`` — the only
#                                   request handle_peer_chat sends through
#                                   AgentDialer/LoopbackProxy.
#   POST /api/events/{id}/trigger   vision §8.5 ("a member or a federated
#                                   peer can fire an event over the peer-to-
#                                   peer network with their device identity")
#                                   and ``src/memory/db.py:708``, which
#                                   documents the ``peer`` delivery source as
#                                   arriving via "agent ALPN".
#   GET  /api/health                liveness; read-only, no agent state.
#   GET  /api/agent-info            identity/name discovery; read-only.
#
# Anything else a peer asks for is administration, not collaboration. If a
# future federation feature adds a route, add it here — extending the tuple
# is the deliberate act; ``extra_paths`` exists so an operator is never
# forced to choose between "scope off" and "wait for a release".
_FEDERATION_ROUTES: tuple[tuple[str, str], ...] = (
    ("POST", r"^/api/chat/?$"),
    ("POST", r"^/api/events/[^/]+/trigger/?$"),
    ("GET", r"^/api/health/?$"),
    ("GET", r"^/api/agent-info/?$"),
)


def _flag(env: str) -> bool:
    """Read a boolean toggle. Default OFF, and anything unrecognised is OFF.

    Same contract as ``src.core.safety.approvals_enabled`` and inverted from
    ``compaction._flag_enabled`` for the same reason: a typo in this var must
    never silently arm a gate that can reject live federation traffic.
    ``server.py`` only ever writes "1" or "0"; the wider set is for operators
    exporting the var by hand.
    """
    return os.environ.get(env, "").strip().lower() in {"1", "true", "yes", "on"}


def peer_allowlist_enabled() -> bool:
    """Whether ``network.peers.allowlist`` is on. Default OFF."""
    return _flag(_ALLOWLIST_ENABLED_ENV)


def peer_scope_enabled() -> bool:
    """Whether ``network.peers.scope`` is on. Default OFF."""
    return _flag(_SCOPE_ENABLED_ENV)


@lru_cache(maxsize=8)
def _parse_allowlist(raw: str) -> frozenset[str]:
    """Parse the comma-separated node_id allowlist.

    Keyed on the raw env string rather than cached module-level: ``server.py``
    exports these during agent build, after this module may already have been
    imported, and tests flip them between cases. A module-level cache would
    freeze policy at import time — the same trap ``safety._compile`` documents.

    Node ids are compared case-insensitively: iroh renders them as hex and the
    coordinator itself compares with ``.lower()`` (``coordinator/service.py``
    ``peer_node_id.lower() != node_id.lower()``), so being stricter here than
    the enrolment path would reject peers the coordinator just enrolled.
    """
    return frozenset(
        seg.strip().lower() for seg in raw.split(",") if seg.strip()
    )


@lru_cache(maxsize=8)
def _parse_extra_paths(raw: str) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """Parse ``scope.extra_paths`` — ``METHOD /path/re`` fragments.

    An unparseable fragment is dropped with an audit line rather than raising:
    a typo must not take the gateway's auth middleware down on the next
    inbound stream. Note the failure mode is inverted from a blocklist and
    worse — a dropped ALLOW fragment fails *closed*, silently narrowing the
    scope — hence the dedicated audit event.
    """
    out: list[tuple[str, re.Pattern[str]]] = []
    for frag in (seg.strip() for seg in raw.split(",")):
        if not frag:
            continue
        parts = frag.split(None, 1)
        if len(parts) != 2:
            elog(
                "peer.bad_scope_path",
                pattern=frag,
                error="expected '<METHOD> <path-regex>'",
            )
            continue
        method, path_re = parts[0].strip().upper(), parts[1].strip()
        try:
            out.append((method, re.compile(path_re)))
        except re.error as e:
            elog("peer.bad_scope_path", pattern=frag, error=str(e))
    return tuple(out)


def _scope_permits(method: str, path: str) -> bool:
    for m, pat in _FEDERATION_ROUTES:
        if m == method and re.match(pat, path):
            return True
    for m, pat in _parse_extra_paths((os.environ.get(_SCOPE_EXTRA_PATHS_ENV) or "").strip()):
        if m == method and pat.match(path):
            return True
    return False


def check_peer_request(*, node_id: str, method: str, path: str) -> str | None:
    """Return a denial reason for an agent-ALPN request, or ``None`` to allow.

    No-op unless a toggle is explicitly on. The disabled path returns before
    reading any list — that is what makes "config absent ⇒ byte-identical
    behaviour" something you read off the code rather than infer from a test.

    Returns a reason string instead of raising: the caller is aiohttp
    middleware that must turn this into a 403 response, and an exception there
    would surface as a 500 and read like a server bug to the peer.
    """
    allowlist_on = peer_allowlist_enabled()
    scope_on = peer_scope_enabled()
    if not allowlist_on and not scope_on:
        return None

    if allowlist_on:
        allowed = _parse_allowlist((os.environ.get(_ALLOWLIST_ENV) or "").strip())
        if node_id.strip().lower() not in allowed:
            elog(
                "peer.denied",
                level="warning",
                reason="not_in_allowlist",
                node_id=node_id,
                method=method,
                path=path,
            )
            return "peer not in network.peers.allowlist"

    if scope_on and not _scope_permits(method, path):
        # Logged at warning with the full path: this is the line an operator
        # greps to discover that a legitimate federation feature needs an
        # ``extra_paths`` entry — the same way ``agent.contact`` is how they
        # build the allowlist. A denial nobody can diagnose gets the whole
        # gate switched back off.
        elog(
            "peer.denied",
            level="warning",
            reason="outside_scope",
            node_id=node_id,
            method=method,
            path=path,
        )
        return f"{method} {path} is outside network.peers.scope"

    return None
