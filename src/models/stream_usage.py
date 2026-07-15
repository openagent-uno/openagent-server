"""Token accounting for the STREAMING path.

``ModelDispatcher.generate`` records every call into ``usage_log`` — the
canonical cost ledger behind Settings → Costs, ``/context`` and the budget.
``ModelDispatcher.stream`` recorded nothing at all. Every surface that matters
streams: chat, the channel bridges, scheduled tasks, and — the one that made
this expensive — inbound webhook events.

So on 2026-07-13/14 two agents burned ~412M input tokens through the Replio
webhook lane and the ledger showed **zero rows** for it. The fire was not
merely unnoticed; it was structurally invisible. It stopped when DeepSeek
returned HTTP 402 "Insufficient Balance", which is a hell of a way to find out.

The runtime emits a ``RunCompletedEvent`` carrying ``metrics`` at the end of a
streamed run. The pieces that see that event (``_arun_runtime_stream`` for the
Team path, ``NativeProvider.stream`` for the single-model path) live in modules
that cannot import the dispatcher without a cycle — so the sink lives here, and
everybody imports this.

A ContextVar holds a plain dict that the dispatcher creates per call. The inner
generators only ever MUTATE that dict, never rebind the ContextVar, so nothing
depends on context propagating back out of an async generator.
"""
from __future__ import annotations

import contextvars
from typing import Any, Optional

_SINK: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "openagent_stream_usage_sink", default=None
)


def open_sink() -> tuple[dict, contextvars.Token]:
    """Start collecting usage for one streamed call."""
    sink: dict[str, Any] = {
        "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "model": None,
    }
    return sink, _SINK.set(sink)


def close_sink(token: contextvars.Token) -> None:
    _SINK.reset(token)


def record(
    *,
    input_tokens: int,
    output_tokens: int,
    model: str | None = None,
    cache_read_tokens: int = 0,
) -> None:
    """Add one streamed run's tokens to the sink, if one is open.

    A no-op when nothing is collecting — a provider streamed outside the
    dispatcher (a test, a direct call) must not blow up on accounting.
    ``cache_read_tokens`` (a subset of ``input_tokens``) lets the streamed
    ledger price a re-sent, server-cached prefix at the cheap cache-read rate.
    """
    sink = _SINK.get()
    if sink is None:
        return
    try:
        sink["input_tokens"] += int(input_tokens or 0)
        sink["output_tokens"] += int(output_tokens or 0)
        sink["cache_read_tokens"] = sink.get("cache_read_tokens", 0) + int(cache_read_tokens or 0)
    except (TypeError, ValueError):
        return
    if model and not sink.get("model"):
        sink["model"] = model


def metrics_to_tokens(metrics: Any) -> tuple[int, int]:
    """Pull (input, output) out of a runtime ``RunMetrics``-shaped object.

    Mirrors ``NativeProvider._extract_metric``: the runtime has changed the
    shape of this object before, and a metrics object we fail to read must
    cost us a log line, never a turn.
    """
    if metrics is None:
        return 0, 0
    data: dict[str, Any]
    if isinstance(metrics, dict):
        data = metrics
    else:
        dump = getattr(metrics, "model_dump", None) or getattr(metrics, "dict", None)
        if callable(dump):
            try:
                data = dump()
            except Exception:  # noqa: BLE001
                data = {}
        else:
            data = {
                k: getattr(metrics, k, None)
                for k in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens")
            }

    def _pick(*names: str) -> int:
        for name in names:
            val = data.get(name)
            if val is None:
                continue
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
        return 0

    return (
        _pick("input_tokens", "prompt_tokens", "input"),
        _pick("output_tokens", "completion_tokens", "output"),
    )


def metrics_to_cache_read(metrics: Any) -> int:
    """Pull server-side prefix-cache read tokens out of a ``RunMetrics`` object.

    Separate from :func:`metrics_to_tokens` so that function's ``(input, output)``
    return stays stable. Returns 0 when absent (a non-caching model / older
    metrics shape), which makes cost accounting fall back to flat pricing.
    """
    if metrics is None:
        return 0
    if isinstance(metrics, dict):
        data: Any = metrics
    else:
        dump = getattr(metrics, "model_dump", None) or getattr(metrics, "dict", None)
        if callable(dump):
            try:
                data = dump()
            except Exception:  # noqa: BLE001
                data = {}
        else:
            data = {k: getattr(metrics, k, None)
                    for k in ("cache_read_tokens", "cached_tokens")}
    for name in ("cache_read_tokens", "cached_tokens", "cache_read"):
        val = data.get(name) if isinstance(data, dict) else None
        if val is None:
            continue
        try:
            return int(val)
        except (TypeError, ValueError):
            continue
    return 0
