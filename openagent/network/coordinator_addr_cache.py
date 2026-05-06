"""On-disk handoff: the running coordinator publishes its iroh
``NodeAddr`` (relay URL + direct UDP addresses) to a file inside
``agent_dir`` so the *separate* ``openagent network invite`` CLI
process can embed those addresses in the tickets it mints.

Why a file instead of an RPC? The invite command runs synchronously
against the SQLite DB and never opens an iroh node — adding an iroh
client just to fetch the addr would double the command's cold-start
time. The file is written once on each ``serve`` start (cheap, ~1 KB)
and read at invite-mint time. If the file is missing (coordinator
never started) or empty, the invite is still valid — the client
just falls back to iroh discovery, exactly as before this feature
existed.

Staleness: the file is rewritten on every ``serve`` start, so a
client running for hours has a fresh-ish snapshot. Direct addresses
can drift (NAT rebinding, network change) but the relay URL is
stable; even partially-stale addresses help the client skip the
worst case (full discovery from scratch).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_FILENAME = "coordinator_addr.json"


def cache_path(agent_dir: Path) -> Path:
    return agent_dir / CACHE_FILENAME


def write_cache(
    agent_dir: Path,
    *,
    node_id: str,
    relay_url: str | None,
    direct_addresses: tuple[str, ...] | list[str],
) -> None:
    """Atomically rewrite the addr cache. Quiet on failure — best-effort.

    Errors here would only delay the discovery-bypass optimisation by
    one round; logging at WARNING so an operator running with a
    locked-down ``agent_dir`` can debug, but never raising.
    """
    payload = {
        "node_id": node_id,
        "relay_url": relay_url or None,
        "addresses": list(direct_addresses or ()),
        "written_at": int(time.time()),
    }
    p = cache_path(agent_dir)
    tmp = p.with_suffix(".tmp")
    try:
        agent_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # POSIX mode 0600 — the file leaks no secrets but the addr list
        # is mildly fingerprintable, so keep it owner-readable only.
        os.chmod(tmp, 0o600)
        tmp.replace(p)
    except OSError as e:
        logger.warning("coordinator addr cache write failed at %s: %s", p, e)


def read_cache(agent_dir: Path) -> tuple[str | None, tuple[str, ...]]:
    """Return ``(relay_url, addresses)`` from the cache, or ``(None, ())``
    when the file is missing/empty/malformed/expired.

    Caller may safely treat ``(None, ())`` as "no hint, fall back to
    iroh discovery".
    """
    p = cache_path(agent_dir)
    if not p.exists():
        return None, ()
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        logger.debug("coordinator addr cache read failed: %s", e)
        return None, ()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, ()
    if not isinstance(data, dict):
        return None, ()
    relay = data.get("relay_url")
    if not isinstance(relay, str) or not relay:
        relay = None
    addrs_raw = data.get("addresses")
    if not isinstance(addrs_raw, list):
        return relay, ()
    addrs = tuple(a for a in addrs_raw if isinstance(a, str) and a)
    return relay, addrs
