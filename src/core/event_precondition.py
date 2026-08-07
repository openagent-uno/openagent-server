"""Cheap pre-delivery check: is there still work to do?

A delivery is enqueued when something happens and runs when a worker gets to
it. Those are not the same moment. Under load — exactly when it matters — the
gap is minutes, and by then a teammate may have answered, the record may have
been closed, or another delivery for the same object may have handled it.

The prompt usually knows this. A support event's instructions open with "read
the thread; if it was already answered, stop", and that check is correct — it
is just being performed by the most expensive component in the system. Measured
on the eSound support webhook (2026-08-07): **~22% of deliveries** — around 200
a day — spent a full brief plus a model turn to conclude there was nothing to
do. That is roughly 11M input tokens a day to answer a question one HTTP GET
can answer.

So the event may declare the check declaratively and let the dispatcher settle
it first::

    {
      "type": "http",
      "url": "http://replio-api/api/v1/orgs/ORG/threads/{payload.payload.thread_id}",
      "headers": {"Authorization": "Bearer ${REPLIO_API_KEY}"},
      "skip_when": {"path": "waiting_for_team", "equals": false},
      "skip_output": "already handled — nobody is waiting on us"
    }

``{payload.…}`` interpolates from the delivery payload; ``${VAR}`` reads the
process environment, so a credential lives in the pod's env and never in the
database or in an events export.

**This is an optimisation, and it fails open.** A timeout, a non-2xx, a missing
field, a malformed spec — anything other than an unambiguous "skip" — runs the
delivery. Skipping a real customer message to save a model call would be a far
worse bug than the cost it saves, so every uncertain path costs money rather
than risking silence.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from src.core.logging import elog

# Hard ceiling on the check itself. It exists to save a model turn; one that
# blocks the worker for longer than the turn would have taken is worse than
# useless, so a slow endpoint fails open quickly rather than holding the queue.
_DEFAULT_TIMEOUT_MS = 5_000
_MAX_TIMEOUT_MS = 30_000

_PAYLOAD_REF = re.compile(r"\{payload\.([A-Za-z0-9_.\[\]-]+)\}")
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _dotted(obj: Any, path: str) -> Any:
    """Walk a dotted path through nested dicts. ``None`` when it doesn't exist.

    Deliberately dict-only: a precondition reads a field out of a JSON body,
    and silently indexing into lists or strings would turn a typo'd path into a
    plausible-looking value instead of the miss that makes us fail open.
    """
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _render(template: str, payload: dict[str, Any]) -> tuple[str, bool]:
    """Substitute ``{payload.a.b}`` and ``${ENV_VAR}`` into *template*.

    Returns ``(rendered, resolved)``. ``resolved`` is False when any reference
    had nothing behind it — a typo'd payload path, an unset environment
    variable.

    It is reported rather than inferred from the output on purpose: an earlier
    version guessed by looking for an empty result or a leftover ``{payload.``
    marker, and both miss the ordinary case. ``http://host/{payload.typo}``
    renders to ``http://host/`` — non-empty, no marker, and a request to
    entirely the wrong URL whose answer we would have believed.
    """
    resolved = True

    def _payload_sub(m: re.Match[str]) -> str:
        nonlocal resolved
        val = _dotted(payload, m.group(1))
        if val is None or str(val) == "":
            resolved = False
            return ""
        return str(val)

    def _env_sub(m: re.Match[str]) -> str:
        nonlocal resolved
        val = os.environ.get(m.group(1), "")
        if not val:
            resolved = False
        return val

    rendered = _ENV_REF.sub(_env_sub, _PAYLOAD_REF.sub(_payload_sub, template))
    return rendered, resolved


def _parse_ts(value: Any) -> float | None:
    """ISO-8601 (with or without a trailing ``Z``) → epoch seconds, else None."""
    if not isinstance(value, str) or not value:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _matches(body: Any, skip_when: dict[str, Any]) -> bool:
    """Evaluate the (single, deliberately simple) skip condition.

    Two forms, one comparison each:

    * ``{"path": P, "equals": V}`` — the field at P equals V.
    * ``{"path": P, "after": Q}`` — the timestamp at P is strictly later than
      the one at Q. This is the "someone already replied" test: an outbound
      newer than the newest inbound.

    Why the second form exists, rather than reading a ready-made boolean: on
    the system this was built for, the obvious flag (``waiting_for_team``) is
    not that predicate. Measured over 14 days, 774 of 1325 threads with an
    unanswered inbound had it false — using it would have silently dropped 58%
    of real customer messages. The two timestamps are load-bearing state and
    say what actually happened; the flag is derived, and drifts.

    Anything richer than this belongs in the endpoint being called, not in a
    config string interpreted at dispatch time — a condition language here
    would become a second place where "should we answer this customer?" gets
    decided, and the wrong answer is silence.
    """
    path = skip_when.get("path")
    if not isinstance(path, str) or not path:
        return False
    actual = _dotted(body, path)

    if "equals" in skip_when:
        if actual is None:
            return False    # field absent → we don't know → run
        return actual == skip_when["equals"]

    if "after" in skip_when:
        other = skip_when["after"]
        if not isinstance(other, str) or not other:
            return False
        lhs, rhs = _parse_ts(actual), _parse_ts(_dotted(body, other))
        if lhs is None or rhs is None:
            return False    # either side missing/unparseable → run
        return lhs > rhs

    return False            # no comparator → not a condition → run


async def _fetch_json(url: str, headers: dict[str, str], timeout_s: float) -> Any:
    """GET *url* and parse JSON, off the event loop thread."""
    import urllib.request

    def _blocking() -> Any:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status < 200 or resp.status >= 300:
                raise RuntimeError(f"HTTP {resp.status}")
            return json.loads(resp.read().decode("utf-8", "replace"))

    return await asyncio.to_thread(_blocking)


async def should_skip(
    event: dict[str, Any], payload: dict[str, Any],
) -> tuple[bool, str]:
    """Return ``(skip, reason)`` for this delivery.

    ``(False, "")`` — no precondition, or the check didn't clearly say skip.
    ``(True, reason)`` — the condition matched; the delivery can be closed
    without a model run, and *reason* is what to record on it.
    """
    raw = event.get("precondition_json")
    if not raw:
        return False, ""

    event_id = event.get("id", "")
    try:
        spec = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(spec, dict) or spec.get("type") != "http":
            raise ValueError(f"unsupported precondition type {spec!r}"[:120])
        url_tpl = spec["url"]
        skip_when = spec["skip_when"]
        if not isinstance(skip_when, dict):
            raise ValueError("skip_when must be an object")
    except Exception as exc:  # noqa: BLE001 — a bad spec must not eat a delivery
        elog("event.precondition_invalid", level="warning",
             id=event_id, error=str(exc)[:200])
        return False, ""

    url, ok = _render(str(url_tpl), payload)
    if not ok:
        elog("event.precondition_unresolved", level="warning",
             id=event_id, url_template=str(url_tpl)[:200])
        return False, ""

    headers: dict[str, str] = {}
    for key, value in (spec.get("headers") or {}).items():
        rendered, ok = _render(str(value), payload)
        if not ok:
            # Usually a missing env var, i.e. a credential. Sending the request
            # without it would 401 and fail open anyway — but noisily, and
            # against someone else's service.
            elog("event.precondition_unresolved", level="warning",
                 id=event_id, header=str(key)[:60])
            return False, ""
        headers[str(key)] = rendered

    try:
        timeout_ms = int(spec.get("timeout_ms") or _DEFAULT_TIMEOUT_MS)
    except (TypeError, ValueError):
        timeout_ms = _DEFAULT_TIMEOUT_MS
    timeout_ms = max(1, min(timeout_ms, _MAX_TIMEOUT_MS))

    try:
        body = await _fetch_json(url, headers, timeout_ms / 1000.0)
    except Exception as exc:  # noqa: BLE001 — unreachable check → run the delivery
        elog("event.precondition_error", level="warning",
             id=event_id, error=f"{type(exc).__name__}: {exc}"[:200])
        return False, ""

    if not _matches(body, skip_when):
        return False, ""

    reason = str(spec.get("skip_output") or "precondition matched")
    elog("event.precondition_skip", id=event_id,
         path=str(skip_when.get("path"))[:80], reason=reason[:160])
    return True, reason
