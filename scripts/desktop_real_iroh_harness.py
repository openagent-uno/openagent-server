#!/usr/bin/env python3
"""Opt-in real-Iroh server harness for the Desktop Playwright E2E.

The harness owns the server half of one isolated acceptance run:

* a temporary coordinator database and identity;
* the production Gateway and coordinator services over a real Iroh node;
* a one-use ``oa1`` user invitation consumed by the real Electron client;
* the production Agent/runtime tool loop backed by a deterministic local
  OpenAI-compatible endpoint.

It prints one ``OPENAGENT_DESKTOP_IROH_READY {...}`` line, then remains alive
until stdin closes or receives ``stop``.  The Playwright process owns the
temporary root and can therefore inspect the client-local sentinel and model
transcript before removing it.  No public TCP Gateway or synthetic auth token
is involved: Electron enrolls its own device identity and every chat/tool
stream crosses its native Iroh loopback.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import threading
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any


READY_PREFIX = "OPENAGENT_DESKTOP_IROH_READY "
SERVER_ROOT = Path(__file__).resolve().parents[1]
# Executing ``python scripts/<name>.py`` places ``scripts/`` rather than the
# repository root on sys.path.  Add the root explicitly so both ``src`` and
# the reusable test fixture package resolve identically in local and CI runs.
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


async def _wait_for_stop() -> None:
    """Wait for the owning Playwright process, stdin EOF, or a signal."""

    loop = asyncio.get_running_loop()
    stopped = asyncio.Event()
    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        with suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, stopped.set)

    # A cancelled ``asyncio.to_thread(sys.stdin.readline)`` remains inside the
    # loop's default executor and can make ``asyncio.run`` hang forever while
    # stdin is still open. A tiny daemon reader gives stdin EOF/"stop" the same
    # wakeup semantics without making signal-driven shutdown wait on that OS
    # thread (notably on Windows, where add_reader is unavailable).
    loop = asyncio.get_running_loop()

    def read_stdin() -> None:
        try:
            sys.stdin.readline()
        finally:
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(stopped.set)

    threading.Thread(
        target=read_stdin,
        name="desktop-real-iroh-harness-stdin",
        daemon=True,
    ).start()
    await stopped.wait()


async def _persist_model_evidence(
    path: Path,
    calls: list[dict[str, Any]],
) -> None:
    """Expose deterministic-provider inputs without adding a test HTTP API."""

    previous = -1
    try:
        while True:
            if len(calls) != previous:
                _atomic_json(path, calls)
                previous = len(calls)
            await asyncio.sleep(0.025)
    finally:
        _atomic_json(path, calls)


async def _seed_deterministic_model(db: Any, model_base_url: str) -> None:
    """Register the harness provider in the canonical dispatch catalog."""

    provider_id = await db.upsert_provider(
        name="local",
        framework="api-based",
        api_key="local",
        base_url=model_base_url,
        enabled=True,
    )
    await db.upsert_model(
        provider_id=provider_id,
        model="deterministic-client-e2e",
        display_name="Deterministic Client E2E",
        enabled=True,
    )


async def run(root: Path) -> None:
    # Import after argument parsing so ``--help`` works even when this script
    # is inspected outside the server virtualenv.
    from scripts.tests.test_real_iroh_client_e2e import (
        _start_deterministic_model_endpoint,
        _wait_for_direct_addresses,
    )
    from src.core import child_session as child_session_hooks
    from src.core.agent import Agent
    from src.gateway.server import Gateway
    from src.memory.db import MemoryDB
    from src.mcp.pool import MCPPool
    from src.mcp.servers.agent_federation import handlers as federation_handlers
    from src.models.native_provider import NativeProvider
    from src.network.coordinator.store import CoordinatorStore
    from src.network.identity import load_or_create_identity
    from src.network.state import NetworkState
    from src.network.ticket import InviteTicket
    from src.stream import child_stream as child_stream_hooks
    from src.stream import resource_events as resource_event_hooks

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "desktop-machine" / "sentinel.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    evidence_path = root / "model-calls.json"
    _atomic_json(evidence_path, [])

    db = MemoryDB(str(root / "gateway.db"))
    await db.connect()
    store = CoordinatorStore(db)
    network_id = "desktop-real-iroh-" + uuid.uuid4().hex[:12]
    network_name = "desktop-real-iroh-e2e"
    identity_path = root / "coordinator.key"
    coordinator_identity = load_or_create_identity(identity_path)
    await store.set_network_role(
        role="coordinator",
        network_id=network_id,
        name=network_name,
        coordinator_node_id=coordinator_identity.public_hex,
        coordinator_pubkey=coordinator_identity.public_bytes,
    )
    await store.register_agent(
        handle="coordinator",
        node_id=coordinator_identity.public_hex,
        owner_handle="system",
        label="Desktop Real Iroh E2E",
    )

    model_runner, model_base_url, model_calls = (
        await _start_deterministic_model_endpoint({
            "desktop": target,
            "cli": root / "unused-cli-machine" / "sentinel.txt",
        })
    )
    # The canonical catalog is the dispatch authority even when the harness
    # supplies a concrete NativeProvider instance.  Keep it in sync with that
    # runtime provider so Gateway's pre-dispatch enabled-model gate accepts
    # the turn and Agent hot-reload can materialise the same configuration.
    await _seed_deterministic_model(db, model_base_url)
    model_evidence_task = asyncio.create_task(
        _persist_model_evidence(evidence_path, model_calls),
        name="desktop-real-iroh-model-evidence",
    )

    pool = MCPPool.from_config(
        mcp_config=[{"builtin": "tool-search"}],
        include_defaults=False,
        db_path=str(root / "runtime.db"),
    )
    model = NativeProvider(
        model="local:deterministic-client-e2e",
        api_key="local",
        base_url=model_base_url,
        providers_config=[{
            "name": "local",
            "framework": "api-based",
            "api_key": "local",
            "base_url": model_base_url,
        }],
        db_path=str(root / "runtime.db"),
    )
    agent = Agent(
        name="Desktop Real Iroh E2E",
        model=model,
        system_prompt=(
            "You are the deterministic Desktop acceptance agent. "
            "Client paths are never server paths."
        ),
        mcp_pool=pool,
        # The harness is a production-shaped vertical slice: Agent lifecycle
        # services and Gateway REST surfaces must share the coordinator's
        # canonical store.  A separate/absent DB leaves Custom Views without
        # an authoritative repository and makes Gateway startup fail.
        memory=db,
    )
    state = await NetworkState.from_db(db=db, identity_path=identity_path)
    gateway = Gateway(agent=agent, network_state=state)

    # The production Gateway binds a few process-level stream hooks. Restore
    # them because this executable is also useful from an in-process CI
    # launcher, not only as a one-shot child process.
    previous_process_hooks = (
        child_session_hooks._listener,
        child_stream_hooks._broadcast_sink,
        resource_event_hooks._sink,
        federation_handlers._iroh_node,
        federation_handlers._db,
    )

    try:
        gateway._prepare_iroh_site()
        await state.start()
        await gateway.start()

        coordinator_node_id = await state.node_id()
        relay_url, addresses = await _wait_for_direct_addresses(state.iroh_node)
        if not relay_url and not addresses:
            raise RuntimeError("coordinator published no Iroh address hints")

        handle = "desktop-e2e"
        password = "desktop-real-iroh-password"
        invitation = await store.create_invitation(
            role="user",
            created_by="desktop-real-iroh-playwright",
            ttl_seconds=900,
            uses=1,
            bind_to_handle=handle,
        )
        ticket = InviteTicket(
            code=invitation.code,
            coordinator_node_id=coordinator_node_id,
            network_name=network_name,
            network_id=network_id,
            role="user",
            bind_to="",
            relay_url=relay_url,
            addresses=tuple(addresses) or None,
        ).encode()

        ready = {
            "ticket": ticket,
            "password": password,
            "handle": handle,
            "network_id": network_id,
            "coordinator_node_id": coordinator_node_id,
            "target_path": str(target),
            "evidence_path": str(evidence_path),
        }
        print(READY_PREFIX + json.dumps(ready, sort_keys=True), flush=True)
        await _wait_for_stop()
    finally:
        model_evidence_task.cancel()
        await asyncio.gather(model_evidence_task, return_exceptions=True)
        with suppress(Exception):
            await gateway.stop()
        with suppress(Exception):
            await agent.shutdown()
        with suppress(Exception):
            await state.stop()
        with suppress(Exception):
            await model_runner.cleanup()
        with suppress(Exception):
            await db.close()

        (
            previous_child_listener,
            previous_child_broadcast,
            previous_resource_sink,
            previous_federation_node,
            previous_federation_db,
        ) = previous_process_hooks
        child_session_hooks.set_child_session_listener(previous_child_listener)
        child_stream_hooks.set_child_broadcast_sink(previous_child_broadcast)
        resource_event_hooks.set_resource_event_sink(previous_resource_sink)
        federation_handlers.set_agent_runtime(
            previous_federation_node, previous_federation_db,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Playwright-owned isolated state/evidence directory",
    )
    args = parser.parse_args()
    asyncio.run(run(args.root))


if __name__ == "__main__":
    main()
