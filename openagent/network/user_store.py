"""User-side networks file: ``~/.openagent/user/networks.toml``.

Stores the list of networks this user (CLI/app install) has joined,
each with the pinned coordinator NodeId+pubkey and the cached cert
location. The CLI's ``connect alice@homelab`` looks up ``homelab``
here; if it's missing, the CLI prompts for the coordinator NodeId
plus an invite code so first-time onboarding is one command.

Schema is versioned so we can evolve the file format without
silently breaking existing installs.

Pure-Python implementation using ``tomllib`` (read) + a tiny
hand-rolled writer (TOML stdlib doesn't ship a writer until 3.13).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import tomllib  # 3.11+

from openagent.network.identity import user_identity_path


SCHEMA_VERSION = 1


@dataclass
class StoredNetwork:
    name: str
    network_id: str
    coordinator_node_id: str
    coordinator_pubkey_hex: str
    handle: str
    added_at: float
    cert_path: str  # relative to the user dir
    last_login_at: float | None = None

    @property
    def coordinator_pubkey_bytes(self) -> bytes:
        return bytes.fromhex(self.coordinator_pubkey_hex)


@dataclass
class UserStore:
    """Versioned dict that survives across CLI / app sessions."""

    networks: list[StoredNetwork] = field(default_factory=list)
    active_network: str | None = None
    active_agent: str | None = None  # agent handle the user last connected to
    schema_version: int = SCHEMA_VERSION


def _user_dir() -> Path:
    p = Path.home() / ".openagent" / "user"
    p.mkdir(parents=True, exist_ok=True)
    return p


def store_path() -> Path:
    return _user_dir() / "networks.toml"


def cert_path_for(network_id: str, handle: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{network_id}__{handle}")
    return _user_dir() / "certs" / f"{safe}.cert"


def load() -> UserStore:
    p = store_path()
    if not p.exists():
        return UserStore()
    raw = tomllib.loads(p.read_text(encoding="utf-8"))
    if raw.get("schema_version", 0) > SCHEMA_VERSION:
        # Forward-compat: a newer client wrote this file. Don't try to
        # parse — return an empty store and let the caller surface a
        # message.
        return UserStore()
    networks = [
        StoredNetwork(
            name=n["name"],
            network_id=n["network_id"],
            coordinator_node_id=n["coordinator_node_id"],
            coordinator_pubkey_hex=n["coordinator_pubkey_hex"],
            handle=n["handle"],
            added_at=float(n.get("added_at", time.time())),
            cert_path=n.get("cert_path") or "",
            last_login_at=n.get("last_login_at"),
        )
        for n in raw.get("networks", [])
    ]
    return UserStore(
        networks=networks,
        active_network=raw.get("active_network"),
        active_agent=raw.get("active_agent"),
        schema_version=raw.get("schema_version", SCHEMA_VERSION),
    )


def save(store: UserStore) -> None:
    """Write the user store to disk atomically.

    We hand-write TOML because tomli-w is an extra dep and the file is
    small + structured.
    """
    p = store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"schema_version = {store.schema_version}")
    if store.active_network:
        lines.append(f'active_network = "{_escape(store.active_network)}"')
    if store.active_agent:
        lines.append(f'active_agent = "{_escape(store.active_agent)}"')
    for n in store.networks:
        lines.append("")
        lines.append("[[networks]]")
        lines.append(f'name = "{_escape(n.name)}"')
        lines.append(f'network_id = "{_escape(n.network_id)}"')
        lines.append(f'coordinator_node_id = "{_escape(n.coordinator_node_id)}"')
        lines.append(f'coordinator_pubkey_hex = "{_escape(n.coordinator_pubkey_hex)}"')
        lines.append(f'handle = "{_escape(n.handle)}"')
        lines.append(f"added_at = {n.added_at}")
        lines.append(f'cert_path = "{_escape(n.cert_path)}"')
        if n.last_login_at is not None:
            lines.append(f"last_login_at = {n.last_login_at}")
    body = "\n".join(lines) + "\n"

    tmp = p.with_suffix(".toml.tmp")
    tmp.write_text(body, encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def add_or_update(
    store: UserStore,
    *,
    name: str,
    network_id: str,
    coordinator_node_id: str,
    coordinator_pubkey_hex: str,
    handle: str,
) -> StoredNetwork:
    """Idempotent insert/update by ``name``. Returns the stored row."""
    for i, existing in enumerate(store.networks):
        if existing.name == name:
            updated = StoredNetwork(
                name=name,
                network_id=network_id,
                coordinator_node_id=coordinator_node_id,
                coordinator_pubkey_hex=coordinator_pubkey_hex,
                handle=handle,
                added_at=existing.added_at,
                cert_path=str(cert_path_for(network_id, handle)),
                last_login_at=existing.last_login_at,
            )
            store.networks[i] = updated
            return updated
    new_row = StoredNetwork(
        name=name,
        network_id=network_id,
        coordinator_node_id=coordinator_node_id,
        coordinator_pubkey_hex=coordinator_pubkey_hex,
        handle=handle,
        added_at=time.time(),
        cert_path=str(cert_path_for(network_id, handle)),
    )
    store.networks.append(new_row)
    if store.active_network is None:
        store.active_network = name
    return new_row


def find(store: UserStore, name: str) -> StoredNetwork | None:
    """Look up a network by human name or by network_id.

    Older saved accounts may carry the network UUID instead of its
    name (a gateway bug echoed ``network_id`` as the ``network`` field
    in ``auth_ok``); accepting both keeps those accounts working
    without forcing a re-add.
    """
    for n in store.networks:
        if n.name == name or n.network_id == name:
            return n
    return None


def remove(store: UserStore, name: str) -> bool:
    for i, n in enumerate(store.networks):
        if n.name == name:
            del store.networks[i]
            cert = Path(n.cert_path)
            if cert.exists():
                try:
                    cert.unlink()
                except OSError:
                    pass
            if store.active_network == name:
                store.active_network = store.networks[0].name if store.networks else None
            return True
    return False


def write_cert(stored: StoredNetwork, cert_wire: bytes) -> None:
    """Persist a freshly-issued cert to ``stored.cert_path`` (0600)."""
    p = Path(stored.cert_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_bytes(cert_wire)
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)


def read_cert(stored: StoredNetwork) -> bytes | None:
    p = Path(stored.cert_path)
    if not p.exists():
        return None
    return p.read_bytes()


def ensure_user_identity_dir() -> Path:
    """Make sure ``user_identity_path()`` and ``certs/`` exist."""
    p = user_identity_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    (p.parent / "certs").mkdir(exist_ok=True)
    return p.parent
