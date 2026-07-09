"""Guard against anyio#695 — ``_deliver_cancellation`` infinite loop.

When MCP SDK tasks inside a cancel scope don't respond to ``CancelledError``
(e.g. ``post_writer`` waiting on a ``MemoryObjectSendStream`` whose peer has
dropped), anyio's ``_deliver_cancellation`` can enter an infinite
``call_soon`` loop, burning 100% CPU until the process is restarted.

This module applies a bounded-iteration wrapper at startup so the loop
terminates after ``_MAX_ITERATIONS`` instead of spinning forever.

Refs:
  - anyio#695 (upstream)
  - IBM ContextForge cpu-spin-loop-mitigation guide
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 200

_original_deliver = None


def _patch_deliver_cancellation() -> bool:
    """Apply the bounded-iteration wrapper. Idempotent — safe to call
    multiple times (e.g. from a reload or test fixture)."""
    global _original_deliver
    if _original_deliver is not None:
        return False  # already patched

    try:
        from anyio._backends._asyncio import CancelScope as _CancelScope
    except ImportError:
        logger.debug("anyio_cancel_guard: anyio._backends._asyncio not available")
        return False

    _original_deliver = _CancelScope._deliver_cancellation  # type: ignore[attr-defined]

    def _patched_deliver(self, origin):
        iteration = getattr(origin, "_oa_cancel_iter", 0) + 1
        origin._oa_cancel_iter = iteration  # type: ignore[attr-defined]
        if iteration > _MAX_ITERATIONS:
            logger.warning(
                "anyio cancel delivery exceeded %d iterations on scope %r — "
                "forcing termination to break spin loop",
                _MAX_ITERATIONS,
                origin,
            )
            # Clear the cancel handle on the origin scope to break the
            # ``call_soon`` chain. The stuck MCP tasks remain but the
            # event loop is released.
            if hasattr(origin, "_cancel_handle") and origin._cancel_handle is not None:
                origin._cancel_handle.cancel()
                origin._cancel_handle = None
            return False
        return _original_deliver(self, origin)

    _CancelScope._deliver_cancellation = _patched_deliver  # type: ignore[attr-defined]
    logger.info("anyio_cancel_guard: patched _deliver_cancellation (max %d iterations)", _MAX_ITERATIONS)
    return True


def _unpatch_deliver_cancellation() -> bool:
    """Restore the original method. For testing / cleanup only."""
    global _original_deliver
    if _original_deliver is None:
        return False
    try:
        from anyio._backends._asyncio import CancelScope as _CancelScope
    except ImportError:
        return False
    _CancelScope._deliver_cancellation = _original_deliver  # type: ignore[attr-defined]
    _original_deliver = None
    return True
