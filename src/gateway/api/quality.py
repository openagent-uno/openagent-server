"""Quality REST API — the correctness meter beside the spend meter.

GET /api/quality?window=<seconds>  → { window_seconds, quality, usage, recall }

Same device-cert auth middleware as the rest of ``/api/*``. Where
``/api/budgets/usage`` answers "how much did we spend?", this answers "were the
answers any good?" — by aggregating the ``quality.score`` (LLM-judge),
``router.cost_recorded`` (spend) and ``recall.metric`` (semantic-recall
hit-rate) events over a window, in one reverse scan of ``events.jsonl``.

Read-only and store-free: everything comes from the append-only event log the
rest of observability already writes, so there is no new table to keep in sync.
Returns zeroed sections (not an error) when the monitor is off or the window is
empty — the app can always draw the panel.
"""
from __future__ import annotations

_DEFAULT_WINDOW = 86400.0  # 24h
_MAX_WINDOW = 60.0 * 60.0 * 24.0 * 90.0  # 90 days — bound the reverse scan


async def handle_report(request):
    from aiohttp import web

    from src.core import quality_monitor

    raw = request.query.get("window", "")
    try:
        window = float(raw) if raw else _DEFAULT_WINDOW
    except ValueError:
        return web.json_response(
            {"error": f"invalid window: {raw!r}"}, status=400)
    window = max(1.0, min(window, _MAX_WINDOW))

    # The scan is plain (blocking) file I/O — offload it so the gateway's event
    # loop (live WebSocket streams, voice audio) is never stalled, exactly as
    # ``GET /api/logs`` hands ``read_tail`` to a thread.
    import asyncio

    report = await asyncio.to_thread(quality_monitor.aggregate, window)
    report["enabled"] = quality_monitor.enabled()
    return web.json_response(report)
