"""Opt-in per-child tool scoping — the one seam the delegation layer and the
model runtime share without importing each other.

A delegated child today runs on its parent's own ``Agent`` and its exact
``self._mcp`` pool, so the child's toolset is byte-identical to the parent's and
can never exceed it (a ``model_id`` child can only get *fewer* tools, when a
provider drops an incompatible family — never more). The "child ⊆ parent"
invariant therefore already holds by construction.

What this module adds is the *optional* Hermes-style ability to run a child on a
NARROWER toolset — a named subset of the parent's grant — without touching the
durable-session / tiered-concurrency machinery or the shared model runtime's
defaults.

The mechanism is a single contextvar holding an allowlist of MCP tool FAMILIES
(server names, e.g. ``vault`` / ``shell`` / ``web``). It is:

  * SET by ``core.child_session.run_child_session`` around a child's run when —
    and only when — a caller passed ``allowed_tools`` (default ``None``);
  * READ by ``models.native_provider`` when it composes a run's toolkits, so the
    child's runtime is built with only the allowed families.

Default is ``None`` → NO restriction → byte-identical to today. Nothing reads or
writes this contextvar on the ordinary (unrestricted) delegation path, so the
production ``support-coverage`` fan-out is completely unaffected.

Family names are normalised through :func:`normalize_family` so a caller can
name a server the human way (``computer-control``) and it still matches the
runtime's ``tool_name_prefix`` (``computer_control``). The normalisation matches
``src.mcp.pool._safe_prefix`` exactly, and is idempotent, so wrapping an
already-normalised name is a no-op.
"""

from __future__ import annotations

import contextvars
from typing import FrozenSet, Iterable, Optional


def normalize_family(name: object) -> str:
    """Coerce a server/family name into the runtime's tool-name prefix form.

    Mirrors ``src.mcp.pool._safe_prefix``: every char that is not alphanumeric
    or ``_`` becomes ``_`` (so ``computer-control`` → ``computer_control``).
    Idempotent — normalising an already-normalised name returns it unchanged.
    """
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))


# The per-run tool allowlist. ``None`` (the default) means "no restriction" and
# is what every unrestricted delegation / chat / automation run leaves in place.
# A frozenset of normalised family names restricts the run to those families.
#
# A ContextVar propagates into ``asyncio.create_task`` children (the context is
# COPIED at task-creation time) and into the coroutines a provider fans tool
# calls out through — the same propagation ``child_session._depth_var`` relies
# on — so a value set around one child's run is seen by every generate/tool call
# that run makes while remaining invisible to its siblings and to its parent.
_allowlist_var: contextvars.ContextVar[Optional[FrozenSet[str]]] = contextvars.ContextVar(
    "openagent_child_tool_allowlist", default=None,
)


def current_tool_allowlist() -> Optional[FrozenSet[str]]:
    """The tool-family allowlist for the run executing on this context, or
    ``None`` when the run is unrestricted (the default)."""
    return _allowlist_var.get()


def set_tool_allowlist(allow: Optional[Iterable[str]]) -> contextvars.Token:
    """Install an allowlist for the current context; returns a reset token.

    ``None`` clears any restriction (explicitly unrestricted). An iterable is
    normalised and frozen. Pass the returned token to :func:`reset_tool_allowlist`
    in the same context to restore the prior value.
    """
    value: Optional[FrozenSet[str]] = (
        None if allow is None else frozenset(normalize_family(a) for a in allow)
    )
    return _allowlist_var.set(value)


def reset_tool_allowlist(token: contextvars.Token) -> None:
    """Restore the allowlist to what it was before the matching
    :func:`set_tool_allowlist`."""
    _allowlist_var.reset(token)
