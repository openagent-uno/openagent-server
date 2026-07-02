"""Global ``resource_event`` sink for in-process producers.

Most ``resource_event`` broadcasts originate from the gateway's own REST
handlers or the Scheduler (which holds ``gateway.broadcast_resource_sync``
directly, so a cron firing flips the app's "Recent" feed live). But a few
in-process producers mutate run state from deep inside an agent turn with no
gateway handle threaded down to them — notably the ``run_dream_mode``
delegation MCP tool, which opens/closes a ``task_runs`` row for a manual
dream-mode firing. Those runs must refresh the feed exactly as a scheduled
firing does.

The gateway registers its ``broadcast_resource_sync`` here on start (and
clears it on stop) — mirroring ``child_stream.set_child_broadcast_sink``.
``emit_resource_event`` is then a best-effort, fire-and-forget any in-process
code can call without importing the gateway. Stays a silent no-op in
headless / test runs where no sink is registered.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ``(resource, action, id) -> None`` — same shape as
# ``Gateway.broadcast_resource_sync``.
ResourceEventSink = Callable[[str, str, "str | None"], None]

_sink: Optional[ResourceEventSink] = None


def set_resource_event_sink(cb: Optional[ResourceEventSink]) -> None:
    """Register (or clear with None) the gateway's resource-broadcast hook."""
    global _sink
    _sink = cb


def emit_resource_event(resource: str, action: str, id: str | None = None) -> None:
    """Best-effort: broadcast a ``resource_event`` to connected clients.

    A no-op when no sink is registered (headless / tests). Never raises — a
    feed refresh must never break the run it's mirroring.
    """
    cb = _sink
    if cb is None:
        return
    try:
        cb(resource, action, id)
    except Exception as e:  # noqa: BLE001 — feed refresh is best-effort
        logger.debug("resource_events: emit %s/%s failed: %s", resource, action, e)
