"""Helpers for the ``workflow_run_requests`` hand-off table.

The ``workflow-manager`` MCP runs as a subprocess — it can't touch the
live ``Agent`` / ``MCPPool`` in the gateway process. When the AI (or
an HTTP caller going through the MCP) asks for ``run_workflow``, we
drop a row into this table and the main-process ``Scheduler`` loop
claims it.

Kept as a thin module so both the MCP subprocess and the main-process
worker have one implementation to import. All functions take a live
``MemoryDB`` — no connection pooling magic.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from openagent.memory.db import MemoryDB


async def enqueue_run_request(
    db: MemoryDB,
    *,
    workflow_id: str,
    trigger: str,
    inputs: dict[str, Any] | None = None,
) -> str:
    """Insert a request row; return its id. The caller polls
    ``wait_for_run_id`` (or ``wait_for_completion``) next."""
    return await db.enqueue_workflow_run_request(
        workflow_id=workflow_id,
        trigger=trigger,
        inputs=inputs,
    )


async def wait_for_run_id(
    db: MemoryDB,
    request_id: str,
    *,
    timeout_s: float = 30.0,
    poll_s: float = 0.25,
) -> str:
    """Block until the main-process worker claims the request and
    attaches a ``run_id``. Raises ``TimeoutError`` if it doesn't
    happen in time."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        req = await db.get_workflow_run_request(request_id)
        if req is not None and req.get("run_id"):
            return req["run_id"]
        await asyncio.sleep(poll_s)
    raise TimeoutError(
        f"workflow_run_request {request_id!r} was not picked up within "
        f"{timeout_s}s — is the main-process scheduler running?"
    )


async def wait_for_completion(
    db: MemoryDB,
    run_id: str,
    *,
    timeout_s: float = 300.0,
    poll_s: float = 0.5,
) -> dict[str, Any]:
    """Poll ``workflow_runs`` until status leaves ``running``. Returns
    the final run row (including trace)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        run = await db.get_workflow_run(run_id)
        if run is None:
            await asyncio.sleep(poll_s)
            continue
        if run.get("status") != "running":
            return run
        await asyncio.sleep(poll_s)
    raise TimeoutError(
        f"workflow_run {run_id!r} did not finish within {timeout_s}s."
    )
