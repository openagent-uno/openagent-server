"""MCP install policy — the enforcement half of ``mcps.install_policy``.

WHY THIS MODULE EXISTS
----------------------
Registering an MCP is not "installing an app". It is handing a third party's
argv the agent's whole environment. Three paths reach that outcome and every
one of them ends at the same place — a row in the ``mcps`` table with
``kind="custom"`` and a ``command``, which ``MCPPool``'s hot-reload spawns on
the next message (``src/mcp/pool.py`` ``_build_and_enter_toolkit``):

  1. ``POST /api/marketplace/install`` (``src/gateway/api/marketplace.py``)
     re-fetches ``server.json`` from ``registry.modelcontextprotocol.io``,
     resolves ``runtimeHint``/``registryType`` into argv via ``_KNOWN_RUNTIMES``
     = {npx, uvx, docker, dnx}, and writes the row. Accepting a card in the UI
     therefore runs ``npx -y <whatever that registry entry names>`` on the
     host. No signature check, no publisher allowlist, no sandbox: a registry
     entry is a remote-code-execution primitive, on a box whose own default
     config gives the agent a filesystem MCP and every secret it can read.
  2. ``POST /api/mcps`` (``src/gateway/api/mcps.py``) skips the registry and
     takes arbitrary ``command``/``args`` directly.
  3. ``add_custom_mcp`` / ``update_mcp`` on the ``mcp-manager`` MCP — which
     means the AGENT can do this to itself. That is the path that matters for
     an unattended deployment: prompt-injected content in a web page or a
     ticket is enough to reach it, with nobody watching.

Gating only the marketplace while (3) stays open would be theatre — the
marketplace is the *narrow* path (it at least has a named package behind it);
(2) and (3) accept raw argv. So the check is factored out here rather than
inlined into one handler, and every surface calls the same function with the
same descriptor. (Cf. ``src/core/safety.py``, factored out for exactly this
reason after a blocklist that guarded ``shell`` while the editor MCP roamed
free.)

SCOPE — READ THIS BEFORE TRUSTING IT
------------------------------------
This covers surfaces (1) and (3). It does NOT yet cover (2) ``POST /api/mcps``
or the pool's spawn itself, because those files are owned elsewhere this
session; the one-line call each needs is in the handover notes. That gap is
narrower than it looks — (2) requires a device cert, i.e. an operator
administering their own agent on purpose, which is the one case that should
work — but it is a gap, and the honest place to close it for good is the pool
spawn, which is the single point all three paths funnel through. Until then
this is a policy on *registration*, not a sandbox on *execution*: it decides
what may be written, and anything already in the table keeps running.

OFF BY DEFAULT, AND THAT IS LOAD-BEARING
----------------------------------------
``check_mcp_install_allowed`` returns before compiling anything unless
``mcps.install_policy.enabled: true``. Existing deployments install MCPs at
runtime by design — that is vision §6 ("Users can register custom MCPs at any
time, by command, URL, or marketplace pick... no restart is required") — so
arming this by default would break a documented capability on someone's
nightly cron.

WHY ENABLING IT IS ONE LINE FOR THE FLEET
-----------------------------------------
The default-on posture, with no allowlist at all, is "freeze the capability
set": no new MCPs at runtime. For an unattended production agent (eSound,
Lyra, Spicysparks) that is the correct posture and needs no per-package
knowledge — their capability set is provisioned once and should not change
because a model read a web page. ``allow_patterns`` is the escape hatch for
the agents that genuinely install, and exists for the same reason
``safety.approvals.allow_patterns`` does: without an exception mechanism the
operator picks between "protection off entirely" and "break my agent", and
picks off — which is how a safety feature ends up switched off fleet-wide.
"""
from __future__ import annotations

import os
import re
import shlex
from functools import lru_cache

from src.core.logging import elog

_POLICY_ENV = "OPENAGENT_MCP_INSTALL_POLICY"
_ALLOW_PATTERNS_ENV = "OPENAGENT_MCP_INSTALL_ALLOW_PATTERNS"


class BlockedInstallError(PermissionError):
    """Raised when an MCP registration is refused by policy.

    Subclasses ``PermissionError`` so a caller that only cares "was this
    refused" can catch a stdlib type — same contract as
    ``safety.BlockedCommandError``. The ``mcp-manager`` tool lets this
    propagate: the runtime turns a tool exception into a structured tool error
    the model reads, so the agent sees the refusal and picks another path
    rather than the run dying. The REST surface turns it into a 403.

    The message names the descriptor verbatim because that string is what an
    operator must paste into ``allow_patterns`` to make an exception — a
    refusal you cannot act on gets the whole policy disabled.
    """

    def __init__(self, descriptor: str, surface: str) -> None:
        self.descriptor = descriptor
        self.surface = surface
        super().__init__(
            f"Blocked by MCP install policy ({surface}): {descriptor!r}. "
            "Registering an MCP runs its command with this agent's full "
            "environment. To permit it, add a matching fragment to "
            "``mcps.install_policy.allow_patterns`` in openagent.yaml, or set "
            "``mcps.install_policy.enabled: false`` to turn the policy off."
        )


def install_policy_enabled() -> bool:
    """Whether ``mcps.install_policy`` is on. Default OFF.

    Anything not explicitly recognised as true is OFF: a typo in this var must
    never silently arm a gate that refuses installs on a running deployment.
    ``server.py`` only ever writes "1" or "0"; the wider set is for operators
    exporting the var by hand.
    """
    return os.environ.get(_POLICY_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def describe_install(
    *,
    command: list[str] | None = None,
    args: list[str] | None = None,
    url: str | None = None,
    registry_name: str | None = None,
) -> str:
    """Build the one-line descriptor that policy matches against.

    ONE string rather than a field-per-dimension allowlist, because the three
    install shapes (registry pick / raw argv / remote URL) would otherwise need
    three parallel config lists, and an operator would have to know which path
    their agent used to write the right one. The format is stable and
    documented in ``examples/openagent.full.yaml`` — it is config surface, so
    changing it silently breaks every deployed ``allow_patterns``:

        marketplace:io.github.foo/postgres npx -y @foo/postgres@1.0.0
        npx -y @modelcontextprotocol/server-filesystem /srv
        url:https://mcp.example.com/sse

    ``shlex.join`` so an argument containing a space cannot forge a token
    boundary — without it, an arg of ``"x npx"`` would let a package smuggle a
    fake ``npx`` into a descriptor an operator's pattern anchors on.
    """
    parts: list[str] = []
    if registry_name:
        parts.append(f"marketplace:{registry_name}")
    argv = [*(command or []), *(args or [])]
    if argv:
        parts.append(shlex.join(str(a) for a in argv))
    elif url:
        parts.append(f"url:{url}")
    return " ".join(parts)


@lru_cache(maxsize=8)
def _compile_allow(allow: str) -> tuple[re.Pattern[str], ...]:
    """Compile ``mcps.install_policy.allow_patterns``.

    Keyed on the raw env string, not cached module-level: ``server.py`` exports
    the var during agent build — after this module may already have been
    imported — and tests flip it between cases. A module-level cache would
    freeze policy at import time (the trap ``safety._compile`` documents).

    Comma-separated, case-insensitive, matching ``safety``'s parsing contract
    so an operator learns one format; a fragment needing a literal comma must
    escape it (``[,]``). An unparseable fragment is dropped with an audit line
    rather than raising — a typo must not take the agent down at first install
    — but note this list is an ALLOW list, so a dropped fragment fails
    **closed**: the exception silently stops applying and installs that used to
    work start getting refused. Hence the dedicated audit event.
    """
    pats: list[re.Pattern[str]] = []
    for frag in (seg.strip() for seg in allow.split(",")):
        if not frag:
            continue
        try:
            pats.append(re.compile(frag, re.IGNORECASE))
        except re.error as e:
            elog("mcp.bad_install_allow_pattern", pattern=frag, error=str(e))
    return tuple(pats)


def check_mcp_install_allowed(
    *,
    name: str,
    surface: str,
    command: list[str] | None = None,
    args: list[str] | None = None,
    url: str | None = None,
    registry_name: str | None = None,
) -> None:
    """Raise ``BlockedInstallError`` if this registration is refused.

    No-op unless ``mcps.install_policy.enabled: true``. The disabled path
    returns before building a descriptor or compiling a pattern — that is what
    makes "config absent ⇒ byte-identical behaviour" readable off the code
    rather than inferred from a test.

    Deny-by-default once on: with no ``allow_patterns`` the capability set is
    frozen, which is the posture an unattended agent wants and the reason this
    is one line to enable rather than a package-inventory project.
    """
    if not install_policy_enabled():
        return

    descriptor = describe_install(
        command=command, args=args, url=url, registry_name=registry_name,
    )
    for pat in _compile_allow((os.environ.get(_ALLOW_PATTERNS_ENV) or "").strip()):
        if pat.search(descriptor):
            elog(
                "mcp.install_allowed",
                name=name,
                surface=surface,
                pattern=pat.pattern,
                descriptor=descriptor[:200],
            )
            return

    # Logged at warning with the descriptor verbatim: this is the line an
    # operator greps to build ``allow_patterns`` after turning the policy on,
    # the same way ``agent.contact`` is how they build the peer allowlist.
    elog(
        "mcp.install_blocked",
        level="warning",
        name=name,
        surface=surface,
        descriptor=descriptor[:200],
    )
    raise BlockedInstallError(descriptor, surface)
