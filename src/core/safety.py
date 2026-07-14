"""Command blocklist — the enforcement half of ``safety.approvals``.

WHY THIS MODULE EXISTS
----------------------
``safety.approvals`` has been in ``examples/openagent.full.yaml`` for a long
time, advertising that it blocks "destructive shell / git push --force / etc.".
It did not. A blocklist used to live in ``src/models/claude_cli.py`` and ran
from the Claude SDK's ``can_use_tool`` callback; that file was deleted whole in
``e8f5d68`` ("refactor: consolidate model layer onto Agno multi-framework
runtime") and the blocklist went with it, silently. What survived was the
config plumbing in ``src/core/server.py``: it still parsed the stanza and
exported ``OPENAGENT_SAFETY_APPROVALS`` /
``OPENAGENT_SAFETY_BLOCK_EXTRA_PATTERNS`` into the environment, where nothing
read them. The toggle looked live from the outside — ``grep`` found the env
var being set — while every command ran unchecked.

That gap was not theoretical. After the May 2026 incident (an agent hard-coded
production iOS signing secrets into a shared build script and pushed them to
master) the operator's post-incident notes recorded
``OPENAGENT_SAFETY_APPROVALS=1`` as the mitigation. It did nothing. This module
is what makes that note true.

SCOPE — READ THIS BEFORE TRUSTING IT
------------------------------------
This is a guardrail against *accidents*, not a sandbox against an adversary.
It is a substring/regex matcher over the command string, so it is trivially
bypassable by anything that defers the text: base64, a variable, a heredoc, a
script file, or piping the command through stdin. It buys "the model typo'd
``rm -rf /``", not "the model is trying to get out". Do not treat a passing
check as an authorisation boundary.

It also covers exactly ONE surface: ``shell_exec`` (see
``src/mcp/servers/shell/handlers.py``). The editor MCP and any registered
filesystem MCP can still destroy things unchecked — they are separate servers
(the editor one is out-of-process TypeScript), and a blocklist that guards
``shell`` while ``filesystem_write_file`` roams free is security theatre if you
believe it covers more than it does. The check is factored out here, rather
than inlined into the shell handler, so those surfaces can adopt the same
policy without re-deriving it.

OFF BY DEFAULT, AND THAT IS LOAD-BEARING
----------------------------------------
``check_command_allowed`` returns before compiling or matching anything unless
``safety.approvals.enabled: true`` is set. Existing deployments run agents that
legitimately do what this list blocks — the Lyra agent force-pushes its own
branch by design and runs ``kubectl`` — so enabling this by default would turn
a silent risk into a live outage on someone's nightly cron.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from src.core.logging import elog

# Destructive shell / VCS / k8s patterns rejected when
# ``safety.approvals.enabled: true``. Carried over verbatim from the retired
# ``claude_cli.py`` list so that turning the toggle on restores exactly the
# protection the config has always claimed — no more (inventing new policy
# here would surprise operators mid-upgrade) and no less.
#
# Matched case-insensitively, substring-wise, against the command string.
# Tighter parsing (shlex-aware, AST for SQL) belongs in a future iteration;
# see the SCOPE note above for why that would not change the trust story much.
_DEFAULT_RISKY_PATTERNS: tuple[str, ...] = (
    r"\brm\s+-[rfRF]+\s+/",               # rm -rf / or rm -rf /...
    r"\bsudo\s+rm\b",
    r"\bdd\s+if=",                        # dd raw disk write
    r"\bmkfs(\.\w+)?\b",                  # filesystem reformat
    r">\s*/dev/sd[a-z]",                  # write to raw block device
    r"\bchmod\s+(-R\s+)?[0-7]{3,4}\s+/",  # chmod -R on root paths
    r"\bgit\s+push\s+(?:--force\b|-f\b)",
    r"\bgit\s+reset\s+--hard\s+(HEAD|origin)",
    r"\bkubectl\s+delete\b",
    r"\bdocker\s+(?:rm\s+-f|rmi\s+-f|system\s+prune\s+-a)",
    r"\bdrop\s+(?:database|table)\s+",    # SQL destructive
    r"\bshutdown\b|\breboot\b",
    r"\bhalt\b|\bpoweroff\b",
)

_APPROVALS_ENV = "OPENAGENT_SAFETY_APPROVALS"
_EXTRA_PATTERNS_ENV = "OPENAGENT_SAFETY_BLOCK_EXTRA_PATTERNS"
_ALLOW_PATTERNS_ENV = "OPENAGENT_SAFETY_ALLOW_PATTERNS"


class BlockedCommandError(PermissionError):
    """Raised when a command matches the blocklist.

    Subclasses ``PermissionError`` so a caller that only cares "was this
    refused" can catch a stdlib type. The shell handler lets this propagate:
    the runtime turns a tool exception into a structured tool error the model
    reads, which is the behaviour the retired ``can_use_tool`` callback had
    (it returned ``{"behavior": "deny", "message": ...}``) — the model sees the
    refusal and picks another path rather than the run dying.
    """

    def __init__(self, command: str, pattern: str, matched: str) -> None:
        self.command = command
        self.pattern = pattern
        self.matched = matched
        super().__init__(
            f"Blocked by safety policy: pattern {pattern!r} matched {matched!r}. "
            "To allow this operation, either rephrase the command, ask the user "
            "to run it directly, or set ``safety.approvals.enabled: false`` in "
            "openagent.yaml."
        )


def approvals_enabled() -> bool:
    """Whether ``safety.approvals`` is on. Default OFF.

    Inverted from ``compaction._flag_enabled`` on purpose: there, an unset var
    means "on" because the feature shipped on. Here an unset var — and any
    value we don't explicitly recognise as true — means OFF, because a typo in
    this var must never silently arm a blocklist on a running deployment.
    ``server.py`` only ever writes "1" or "0"; the wider set is for operators
    exporting the var by hand.
    """
    val = os.environ.get(_APPROVALS_ENV, "").strip().lower()
    return val in {"1", "true", "yes", "on"}


@lru_cache(maxsize=8)
def _compile_allow(allow: str) -> tuple[re.Pattern[str], ...]:
    """Compile ``safety.approvals.allow_patterns`` — the escape hatch.

    Without this the blocklist is all-or-nothing, and that makes it unusable
    for the deployments that need it most. ``git push --force`` is in the
    default set, but an autonomous agent that owns its own branch force-pushes
    it *by design* (rebase, then force-push ``agent/<ticket>-…``). Such an
    operator had exactly two options: leave the blocklist off and keep every
    other protection off with it, or turn it on and break their agent's normal
    workflow on the next run. Both are bad, so most would pick "off" — which is
    how a safety feature ends up switched off fleet-wide and everyone stops
    thinking about it.

    An allow pattern wins over a block pattern. That ordering is the point: the
    operator is stating an exception to a rule they otherwise keep. It is
    deliberately narrower than editing the default list — the defaults stay
    intact and the exception is written down, auditable, and per-deployment.

    Same parsing contract as :func:`_compile`: comma-separated, case-insensitive,
    an unparseable fragment is dropped with an audit line rather than raising.
    But note the failure mode is *inverted* and worse: a typo'd BLOCK pattern
    fails safe (that command stays unblocked, and the rest of the list still
    applies), whereas a typo'd ALLOW pattern fails **closed** — the exception
    silently stops applying and the agent starts getting blocked on work that
    used to run. Hence the dedicated audit event.
    """
    pats: list[re.Pattern[str]] = []
    for frag in (seg.strip() for seg in allow.split(",")):
        if not frag:
            continue
        try:
            pats.append(re.compile(frag, re.IGNORECASE))
        except re.error as e:
            elog("safety.bad_allow_pattern", pattern=frag, error=str(e))
    return tuple(pats)


@lru_cache(maxsize=8)
def _compile(extra: str) -> tuple[re.Pattern[str], ...]:
    """Compile defaults + ``extra`` (comma-separated regex fragments).

    Keyed on the raw env string rather than cached module-level, because
    ``server.py`` exports the var during agent build — after this module may
    already have been imported — and tests flip it between cases. A stale
    module-level cache would freeze policy at import time.

    Comma is the separator the retired code documented and the one
    ``server.py`` joins the YAML list with, so a fragment needing a literal
    comma must escape it (``[,]`` or ``\\x2c``). An unparseable fragment is
    dropped with an audit line rather than raising: a typo'd extra pattern
    must not take the agent down at first shell call, and the built-in list
    still applies.
    """
    pats: list[re.Pattern[str]] = [
        re.compile(p, re.IGNORECASE) for p in _DEFAULT_RISKY_PATTERNS
    ]
    for frag in (seg.strip() for seg in extra.split(",")):
        if not frag:
            continue
        try:
            pats.append(re.compile(frag, re.IGNORECASE))
        except re.error as e:
            elog("safety.bad_extra_pattern", pattern=frag, error=str(e))
    return tuple(pats)


def check_command_allowed(command: str, *, tool: str = "shell_exec") -> None:
    """Raise ``BlockedCommandError`` if *command* is blocked. Else return None.

    No-op unless ``safety.approvals.enabled: true``. The disabled path returns
    before touching the pattern list at all — that is what makes "config off ⇒
    byte-identical behaviour" something you can read off the code rather than
    infer from a test.
    """
    if not approvals_enabled():
        return

    # Allow wins over block: an operator naming an exception has said more
    # about their intent than a default pattern written years ago. Checked
    # FIRST so an exempt command never reaches the block loop — which also
    # means it is never audited as blocked, because it wasn't.
    for pat in _compile_allow((os.environ.get(_ALLOW_PATTERNS_ENV) or "").strip()):
        if pat.search(command):
            elog(
                "safety.command_allowed",
                tool=tool,
                pattern=pat.pattern,
                command=command[:120],
            )
            return

    for pat in _compile((os.environ.get(_EXTRA_PATTERNS_ENV) or "").strip()):
        m = pat.search(command)
        if m:
            elog(
                "safety.command_blocked",
                tool=tool,
                pattern=pat.pattern,
                matched=m.group(0)[:120],
            )
            raise BlockedCommandError(command, pat.pattern, m.group(0)[:120])
