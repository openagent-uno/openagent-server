"""Self-contained invite tickets — one string carries everything a client needs.

A ticket is the entire join-payload (network name, network_id, coordinator
NodeId, invite code, role, optional bind-to-handle) packed into a single
copy-pasteable string. UX-wise this collapses the legacy

    openagent-cli connect alice@homelab \\
        --coordinator c843dfbb25e9... --invite gl6o-h3l7-...

into

    openagent-cli connect oa1abcdef...

The CLI/app prompts only for what a ticket can't carry (the user's chosen
handle for ``role=user``, and the password). For ``role=device`` invites
the handle is bound by ``bind_to``, so the user only enters a password.

Wire format::

    "oa1" || base32-no-pad-lowercase(CBOR{
        v: 1,
        code: <invite code>,
        node_id: <coordinator NodeId hex>,
        name: <network display name>,
        network_id: <network UUID>,
        role: "user" | "device" | "agent",
        bind_to: <handle or empty>,
        # Optional fields (added in v0.12.54). Embed the coordinator's
        # iroh relay URL + direct addresses so first-contact dials skip
        # iroh discovery — a hard requirement on macOS DMG builds where
        # mDNS is gated behind Local Network access permission and pkarr
        # DNS doesn't always resolve same-machine coordinators.
        relay_url?: <coordinator home relay URL or empty/missing>,
        addresses?: <list of "ip:port" direct UDP addrs or missing>,
    })

Base32 (no padding, lowercase) keeps the string URL-safe + double-clickable
without losing density. Length: ~120-180 chars for typical inputs — fine
for terminals, QR codes, and chat messages.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cbor2


TICKET_PREFIX = "oa1"
TICKET_VERSION = 1


class TicketError(ValueError):
    """Raised on any malformed-ticket failure (wrong prefix, bad CBOR, wrong version)."""


@dataclass(frozen=True)
class InviteTicket:
    """Decoded contents of an invite ticket. Use ``encode``/``decode``."""

    code: str
    coordinator_node_id: str
    network_name: str
    network_id: str
    role: str
    bind_to: str = ""
    # Optional address hints — see module docstring. ``None`` (rather
    # than empty containers) means "this ticket was minted before the
    # field existed"; the client falls back to iroh discovery.
    relay_url: str | None = None
    addresses: tuple[str, ...] | None = None

    def encode(self) -> str:
        # Only emit the optional fields when populated — keeps tickets
        # minted by older deployments byte-identical to before, and
        # avoids a wave of "ticket changed!" support tickets after the
        # rollout.
        payload: dict = {
            "v": TICKET_VERSION,
            "code": self.code,
            "node_id": self.coordinator_node_id,
            "name": self.network_name,
            "network_id": self.network_id,
            "role": self.role,
            "bind_to": self.bind_to,
        }
        if self.relay_url:
            payload["relay_url"] = self.relay_url
        if self.addresses:
            payload["addresses"] = list(self.addresses)
        return _encode_payload(payload)

    @classmethod
    def decode(cls, s: str) -> "InviteTicket":
        obj = _decode_payload(s)
        try:
            raw_addresses = obj.get("addresses")
            addresses: tuple[str, ...] | None = None
            if isinstance(raw_addresses, list):
                cleaned = tuple(a for a in raw_addresses if isinstance(a, str) and a)
                addresses = cleaned or None
            raw_relay = obj.get("relay_url")
            relay_url = raw_relay if isinstance(raw_relay, str) and raw_relay else None
            return cls(
                code=str(obj["code"]),
                coordinator_node_id=str(obj["node_id"]),
                network_name=str(obj["name"]),
                network_id=str(obj["network_id"]),
                role=str(obj.get("role", "user")),
                bind_to=str(obj.get("bind_to", "")),
                relay_url=relay_url,
                addresses=addresses,
            )
        except KeyError as e:
            raise TicketError(f"ticket missing field: {e}") from e


def looks_like_ticket(s: str) -> bool:
    """Cheap prefix-check used by argument routers."""
    return isinstance(s, str) and s.startswith(TICKET_PREFIX)


# ── internals ──────────────────────────────────────────────────────────


def _encode_payload(obj: dict) -> str:
    raw = cbor2.dumps(obj)
    body = base64.b32encode(raw).rstrip(b"=").decode("ascii").lower()
    return TICKET_PREFIX + body


def _decode_payload(s: str) -> dict:
    if not isinstance(s, str) or not s.startswith(TICKET_PREFIX):
        raise TicketError(f"not an OpenAgent ticket: {s[:8]!r}")
    body = s[len(TICKET_PREFIX):].upper()
    # Base32 wants padding to a multiple of 8 chars; we strip it for
    # cosmetics on encode and add it back here for decode.
    body += "=" * (-len(body) % 8)
    try:
        raw = base64.b32decode(body, casefold=True)
    except (ValueError, base64.binascii.Error) as e:  # type: ignore[attr-defined]
        raise TicketError(f"ticket isn't valid base32: {e}") from e
    try:
        obj = cbor2.loads(raw)
    except cbor2.CBORDecodeError as e:
        raise TicketError(f"ticket payload isn't valid CBOR: {e}") from e
    if not isinstance(obj, dict):
        raise TicketError("ticket payload is not a CBOR map")
    if obj.get("v") != TICKET_VERSION:
        raise TicketError(f"unsupported ticket version: {obj.get('v')}")
    return obj
