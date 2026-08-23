"""Per-turn execution profiles for constrained local-model event runs.

The normal OpenAgent prompt and Team runtime are deliberately broad.  That is
useful for interactive, multi-model work, but wasteful for an unattended event
whose model was explicitly pinned to a self-hosted endpoint.  This module keeps
the narrower behaviour coroutine-local so concurrent cloud/chat turns remain
byte-identical.
"""
from __future__ import annotations

import contextvars
import ipaddress
import os
from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import urlparse


_LEAN_LOCAL_EVENT = contextvars.ContextVar("lean_local_event", default=False)
_STRICT_LOCAL_ONLY = contextvars.ContextVar("strict_local_only", default=False)
_STATELESS_COMPLETION = contextvars.ContextVar(
    "stateless_completion", default=False,
)
_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}


def lean_local_event_active() -> bool:
    return bool(_LEAN_LOCAL_EVENT.get())


_LEAN_LOCAL_TASK = contextvars.ContextVar("lean_local_task", default=False)


def lean_local_task_active() -> bool:
    """True inside a locally-pinned SCHEDULED TASK, as opposed to an event.

    Both lanes share the lean profile, but not the shape of their output: a
    support reply is a few sentences, a scheduled task emits a report and long
    tool calls. Measured: a quality-scorer tool call was cut off mid-JSON at
    the event budget and the whole run died on a parse error.
    """
    return bool(_LEAN_LOCAL_TASK.get())


def strict_local_only_active() -> bool:
    """True when a turn must never use a configured cloud fallback."""
    return bool(_STRICT_LOCAL_ONLY.get())


@contextmanager
def lean_local_task_scope(enabled: bool = True) -> Iterator[None]:
    token = _LEAN_LOCAL_TASK.set(bool(enabled))
    try:
        yield
    finally:
        _LEAN_LOCAL_TASK.reset(token)


@contextmanager
def lean_local_event_scope(enabled: bool = True) -> Iterator[None]:
    token = _LEAN_LOCAL_EVENT.set(bool(enabled))
    try:
        yield
    finally:
        _LEAN_LOCAL_EVENT.reset(token)


def stateless_completion_active() -> bool:
    """True for a one-shot, tool-less generation that must persist nothing.

    The controller's reply composer is a pure function: no tools, no history,
    one turn. Persisting a session row per call bought nothing and cost real
    contention - measured at 8 concurrent deliveries, SQLite answered
    "database is locked" on upsert_session and five composes out of
    twenty-four timed out into the deterministic fallback.
    """
    return bool(_STATELESS_COMPLETION.get())


@contextmanager
def stateless_completion_scope(enabled: bool = True) -> Iterator[None]:
    token = _STATELESS_COMPLETION.set(bool(enabled))
    try:
        yield
    finally:
        _STATELESS_COMPLETION.reset(token)


@contextmanager
def strict_local_only_scope(enabled: bool = True) -> Iterator[None]:
    """Coroutine-local hard boundary for controller-owned local inference."""
    token = _STRICT_LOCAL_ONLY.set(bool(enabled))
    try:
        yield
    finally:
        _STRICT_LOCAL_ONLY.reset(token)


def _enabled_by_default() -> bool:
    raw = os.environ.get("OPENAGENT_LEAN_LOCAL_EVENTS", "1").strip().lower()
    if raw in _FALSEY:
        return False
    return raw in _TRUTHY or not raw


def _scheduled_tasks_enabled_by_default() -> bool:
    raw = os.environ.get("OPENAGENT_LEAN_LOCAL_SCHEDULED_TASKS", "1").strip().lower()
    if raw in _FALSEY:
        return False
    return raw in _TRUTHY or not raw


# A model from one of these families is that vendor's model wherever it is
# reached from. Two of our own endpoints are proxies to Anthropic - one on
# 127.0.0.1, one on a *.svc.cluster.local name - so a URL test alone called
# cloud Claude "self-hosted" and handed it the lean local profile.
_CLOUD_MODEL_FAMILIES = (
    "claude", "gpt-", "gpt4", "gpt5", "o1-", "o3-", "o4-",
    "gemini", "grok", "deepseek", "mistral-large", "command-r",
)


def _is_cloud_model_id(runtime_id: str) -> bool:
    model = runtime_id.split(":", 1)[-1].strip().lower()
    return any(family in model for family in _CLOUD_MODEL_FAMILIES)


def _is_local_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    try:
        host = (urlparse(base_url).hostname or "").strip().lower()
        if host in {"localhost", "host.docker.internal"} or host.endswith(".local"):
            return True
        address = ipaddress.ip_address(host)
        # Tailscale hands out 100.64.0.0/10 (CGNAT). Python calls that
        # "shared", not "private", so the one endpoint that really is our own
        # GPU box was the only one being judged remote.
        if address in ipaddress.ip_network("100.64.0.0/10"):
            return True
        if address.version == 6 and address in ipaddress.ip_network("fd7a:115c:a1e0::/48"):
            return True
        return address.is_private
    except (TypeError, ValueError):
        return False


async def _is_pinned_self_hosted(runtime_id: str, db: Any) -> bool:
    """Resolve one explicit runtime pin to a private/self-hosted provider."""
    if db is None:
        return False
    if ":" not in runtime_id:
        return False
    explicit = {
        item.strip()
        for item in os.environ.get("OPENAGENT_LOCAL_INFERENCE_MODELS", "").split(",")
        if item.strip()
    }
    if explicit:
        return runtime_id in explicit
    # A cloud model is never self-hosted, whatever it is proxied through.
    if _is_cloud_model_id(runtime_id):
        return False
    provider_name = runtime_id.split(":", 1)[0]
    try:
        row = await db.get_provider_by_name(provider_name, "api-based")
    except Exception:  # noqa: BLE001 - profile detection must never fail a run
        return False
    return bool(row and _is_local_url(row.get("base_url")))


async def should_use_lean_local_event(event: dict[str, Any], db: Any) -> bool:
    """True only for an explicitly pinned event on a self-hosted endpoint."""
    if not _enabled_by_default():
        return False
    runtime_id = str((event or {}).get("model") or "").strip()
    return await _is_pinned_self_hosted(runtime_id, db)


async def should_use_lean_local_scheduled_task(
    task: dict[str, Any], db: Any,
) -> bool:
    """True for an explicitly pinned scheduled task on a self-hosted model.

    Unlike an interactive turn, a scheduled firing has no human available to
    notice a silent provider fallback or a runaway context.  The scheduler
    uses this result for both the compact prompt profile and the hard
    local-only boundary.
    """
    if not _scheduled_tasks_enabled_by_default():
        return False
    runtime_id = str((task or {}).get("model") or "").strip()
    return await _is_pinned_self_hosted(runtime_id, db)


def lean_local_tool_families(prompt: str, available: "Iterable[str]") -> list[str] | None:
    """Tool families a lean local scheduled task should be allowed to use.

    A pinned local task ran against the agent's whole MCP surface: measured on
    Qwen3-30B, a task that only had to send one approved sentence spent ten
    tool calls exploring BillingBear and never reached the send. A small model
    treats a large surface as an invitation.

    So: when the prompt names servers explicitly, restrict the run to those
    (plus ``vault``, which every policy read needs, and ``tool-search``, the
    navigation surface itself). When it names none, return ``None`` — no
    restriction, byte-identical to today, because guessing would be worse than
    the status quo.
    """
    low = str(prompt or "").lower()
    names = [str(name) for name in available or ()]
    named = [
        name for name in names
        if name.lower() in low
        or name.lower().replace("-", " ") in low
        or name.lower().replace("-", "_") in low
    ]
    if not named:
        return None
    always = [
        name for name in names
        if name.lower() in {"vault", "tool-search", "tool_search"}
    ]
    return sorted({*named, *always})
