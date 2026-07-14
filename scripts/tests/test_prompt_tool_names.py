"""Prompt-vs-reality — every tool the framework prompt names must exist.

``FRAMEWORK_SYSTEM_PROMPT`` is injected into EVERY run (chat turn,
sub-agent, scheduled task, workflow AI block), and ``DREAM_MODE_PROMPT``
seeds every nightly maintenance session. Both spend their tokens telling
the model which tools to call — so a name that drifted out of sync with
the real MCP registrations is a defect that fires on every run: the
prompt itself warns that a bad key "yields ``Function X not found`` and
burns a turn", and then hands the model exactly such a key.

That is not hypothetical. Caught by this test when it was written:

  * ``list_notes`` / ``vault_list_notes`` advertised as vault tools in
    four places — the real leaf is ``vault_list_directory``, and the
    prompt said so itself twenty lines earlier.
  * ``get_backlinks`` listed as a vault tool — no tool by that name
    exists anywhere; inbound links are ``vault_backlinks``, and they
    live on a DIFFERENT MCP (``vault-gate``).
  * ``shell_shell_exec`` (+ five siblings) documented as the shell
    keys — the registered keys are ``shell_exec`` & co., and the same
    prompt calls ``shell_shell_exec`` the mistake to avoid.

So the test derives ground truth from the REAL registrations rather than
a hand-kept list — that is the whole point, since a hand-kept list is
what drifted in the first place. Pure-unit: imports adapter modules and
parses the vendored vault server's source; no subprocess, no Node, no
network.
"""
from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

from ._framework import TestContext, test

# ── Ground truth ────────────────────────────────────────────────────


def _safe_prefix(name: str) -> str:
    """Mirror of ``MCPPool._safe_prefix`` — the runtime keys subprocess
    tools as ``<safe_prefix>_<tool>`` (``vault-gate`` → ``vault_gate_``).
    """
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)


class _StubPool:
    """Minimal duck-typed pool for factories that take a ``pool`` kwarg
    (``tool-search``). We only ever read the registered tool NAMES, so
    the factory never dispatches through it."""

    _toolkit_by_name: dict = {}

    def toolkit_by_name(self, _name):  # noqa: D102
        return None


# Tools the RUNTIME injects directly, not via any MCP — so they can't be
# discovered from ``BUILTIN_MCP_SPECS``. Keep this list tiny and cite the
# definition site so it can be re-verified.
_RUNTIME_TOOLS: frozenset[str] = frozenset({
    # Team-leader delegation tool, built per-run by the Team runner:
    # src/core/_runner/team/_default_tools.py (``def delegate_task_to_member``).
    # Only present when a ``<team_members>`` block is set, which is exactly
    # how the prompt documents it.
    "delegate_task_to_member",
})

# Node MCPs whose registered names we parse straight from source (no
# ``npm install`` / ``dist/`` build needed at test time). Only the vault
# is listed: it is the sole Node MCP whose individual tool names the
# framework prompt enumerates. If a prompt edit starts naming ``editor``
# or ``web-search`` tools, this test fails with "not registered" — add
# the server here rather than deleting the assertion.
_NODE_TOOL_SOURCES: dict[str, str] = {
    "vault": "src/createServer.ts",
}
_TS_TOOL_NAME = re.compile(r'^\s*name:\s*"([a-z][a-z0-9_]*)"', re.M)


def _repo_root() -> Path:
    # scripts/tests/test_prompt_tool_names.py → repo root is 2 up
    return Path(__file__).resolve().parent.parent.parent


def _registered_tool_keys() -> tuple[set[str], list[str]]:
    """Return ``(every registered tool key, notes)`` built from the real
    sources: in-process adapter toolkits, the Python MCP servers' FastMCP
    registries, and the vendored vault server's TypeScript source.
    """
    from src.mcp.builtins import BUILTIN_MCP_SPECS, BUILTIN_MCPS_DIR

    keys: set[str] = set(_RUNTIME_TOOLS)
    notes: list[str] = []

    for name, spec in BUILTIN_MCP_SPECS.items():
        # 1. In-process Toolkits keep the BARE python function name — the
        #    pool passes no ``tool_name_prefix`` for them, so the ``shell``
        #    server's key really is ``shell_exec``, never ``shell_shell_exec``.
        if spec.get("in_process"):
            try:
                mod = importlib.import_module(spec["adapter_module"])
                factory = getattr(
                    mod, spec.get("runtime_toolkit_factory", "build_runtime_toolkit"),
                )
                if "pool" in inspect.signature(factory).parameters:
                    toolkit = factory(pool=_StubPool())
                else:
                    toolkit = factory()
            except Exception as exc:  # noqa: BLE001
                # agent-federation imports the sibling ``openagent_mcp``
                # package, absent in a clean venv (see test_repo_hygiene's
                # _SMOKE_IMPORT_SKIP). A server we can't introspect is
                # recorded, not fatal — the prompt names none of its tools.
                notes.append(f"{name}: not introspectable ({type(exc).__name__})")
                continue
            fns = dict(getattr(toolkit, "functions", {}) or {})
            fns.update(getattr(toolkit, "async_functions", {}) or {})
            keys |= set(fns)
            continue

        # 2. Python subprocess MCPs go through the pool's prefixing.
        if spec.get("python"):
            try:
                mod = importlib.import_module(f"src.mcp.servers.{spec['dir']}.server")
                registry = mod.mcp._tool_manager.list_tools()
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{name}: not introspectable ({type(exc).__name__})")
                continue
            pfx = _safe_prefix(name) + "_"
            keys |= {pfx + t.name for t in registry}

    # 3. Vendored Node vault server — parse its ListTools handler.
    for name, rel in _NODE_TOOL_SOURCES.items():
        src = BUILTIN_MCPS_DIR / BUILTIN_MCP_SPECS[name]["dir"] / rel
        if not src.exists():
            notes.append(f"{name}: source missing at {src}")
            continue
        pfx = _safe_prefix(name) + "_"
        keys |= {pfx + t for t in _TS_TOOL_NAME.findall(src.read_text())}

    return keys, notes


# ── Extracting the names the prompt claims ──────────────────────────

# A backticked snake_case token, either ``x`` / `x` or ``x(``…, plus the
# ``tool="x"`` form used by every tool_search_call_tool example.
_BACKTICKED = re.compile(r"``([a-z][a-z0-9_]*)(?:\(|``)|`([a-z][a-z0-9_]*)(?:\(|`)")
_TOOL_KWARG = re.compile(r'tool="([a-z][a-z0-9_]*)"')

# Backticked snake_case tokens that are NOT tool names. The prompt marks
# up SQL identifiers, JSON fields and shell binaries the same way it marks
# up tools, so they have to be named explicitly. Grouped by what they are:
# when a prompt edit trips this test, classify the new token into a group
# below or fix the tool name — do not delete the check.
_NON_TOOL_TOKENS: frozenset[str] = frozenset({
    # -- SQLite tables + columns (the "Operational history" section) --
    "sessions", "runs", "summary", "session_data", "session_metrics",
    "metadata", "client_id", "title", "model", "framework",
    "parent_session_id", "origin", "kind", "child_run_id",
    "scheduled_tasks", "task_runs", "status", "output", "error",
    "session_id", "task_id", "task_run_requests", "workflow_tasks",
    "workflow_schedules", "workflow_runs", "trace_json",
    "workflow_run_requests", "providers", "models", "mcps",
    "pinned_sessions", "config_state", "network", "peer_networks",
    "device_certs", "network_users", "network_devices", "network_agents",
    "network_invitations", "usage_log", "conversation_embeddings",
    "user_profiles", "skills", "vault_save_reminders", "updated_at",
    "started_at", "workflow_id", "json_extract",
    # -- MCP SERVER names (not tools) --
    "vault", "scheduler", "delegation", "attachments", "workflow", "chat",
    # -- tool PARAMETERS / result fields --
    "model_id", "runtime_id", "top_k", "session_binding_path", "task",
    "query", "path", "tool", "marker", "shell_id", "open_suggestions",
    "child_session_id", "id",
    # ``error_like`` is a RESULT FIELD of the logs MCP, not a tool.
    # DREAM_MODE_PROMPT names it to warn that severity is inferred rather
    # than authoritative (log entries carry no level field), so the flag
    # must be confirmed against the event payload before acting on it.
    "error_like",
    # -- vault note frontmatter fields --
    "tags", "created", "updated",
    # -- vault folder taxonomy (the Company-Brain set) --
    "self", "areas", "projects", "sources", "concepts", "docs",
    "entities", "data", "code", "outputs", "workspace",
    # -- host binaries / Claude Code surfaces the prompt FORBIDS --
    "crontab", "launchd", "systemd", "at", "cat", "grep", "find",
    "read_file", "str_replace", "loop", "schedule", "claude",
    # -- invitation protocol roles --
    "agent", "user", "device",
})

# The prompt deliberately names a few keys BECAUSE they are wrong — the
# "Two slips waste a turn" block teaches ``shell_shell_exec`` and
# ``vault_list_notes`` as the mistakes to avoid. Those mentions are
# correct and must survive.
#
# We tell a lesson from a bug by CONTEXT, not by maintaining a list of
# blessed names: naming a tool that does not exist is only allowed when
# the prose is explicitly telling the model not to call it. A blanket
# per-name exemption would be worse than useless here — it would have
# hidden the four real ``vault_list_notes`` / ``list_notes`` defects,
# which sat in the same prompt as the counter-example that names them.
#
# Markers are matched case-insensitively on the token's own line and the
# one above it (the prompt hard-wraps, so the marker often lands on the
# previous line).
_NEGATION_MARKERS: tuple[str, ...] = (
    "never", "not the", "there is no", "no tool", "do not", "don't",
    "instead of", "rather than", "not a real",
)
_NEGATION_LOOKBEHIND = 1


def _claimed_names(text: str) -> dict[str, list[int]]:
    """``{name: [1-based line numbers]}`` for every tool-ish token that is
    NOT presented as a counter-example."""
    lines = text.split("\n")
    found: dict[str, list[int]] = {}
    for i, line in enumerate(lines):
        hits = [m.group(1) or m.group(2) for m in _BACKTICKED.finditer(line)]
        hits += _TOOL_KWARG.findall(line)
        for name in hits:
            if not name or name in _NON_TOOL_TOKENS:
                continue
            # A trailing underscore means the prompt is quoting a KEY
            # PREFIX (``shell_``), not a tool — no registered name ends
            # in one.
            if name.endswith("_"):
                continue
            if _is_counter_example(lines, i):
                continue
            found.setdefault(name, []).append(i + 1)
    return found


def _is_counter_example(lines: list[str], idx: int) -> bool:
    """True if the prose around ``lines[idx]`` marks its tool name as one
    the model must NOT call."""
    window = lines[max(0, idx - _NEGATION_LOOKBEHIND): idx + 1]
    haystack = " ".join(window).lower()
    return any(m in haystack for m in _NEGATION_MARKERS)


def _counter_example_names(text: str) -> dict[str, list[int]]:
    """``{name: [lines]}`` for tool-ish tokens the prose marks as wrong."""
    lines = text.split("\n")
    found: dict[str, list[int]] = {}
    for i, line in enumerate(lines):
        if not _is_counter_example(lines, i):
            continue
        hits = [m.group(1) or m.group(2) for m in _BACKTICKED.finditer(line)]
        hits += _TOOL_KWARG.findall(line)
        for name in hits:
            if name and name not in _NON_TOOL_TOKENS and not name.endswith("_"):
                found.setdefault(name, []).append(i + 1)
    return found


def _resolves(name: str, keys: set[str]) -> bool:
    """True if ``name`` is usable as-typed.

    Two accepted shapes, mirroring ``tool_search``'s ``_candidate_names``:
      * the exact registered key (``vault_write_note``, ``shell_exec``);
      * a bare leaf the resolver repairs by adding a server prefix
        (``patch_note`` → ``vault_patch_note``), which the prompt uses
        legitimately in prose.
    A DOUBLED prefix (``shell_shell_exec``) is deliberately NOT accepted:
    the resolver forgives it, but the prompt's own rule is "copy the
    registered key verbatim", so advertising one is still a defect.
    """
    if name in keys:
        return True
    return any(k.endswith("_" + name) for k in keys)


def _assert_prompt_tools_exist(text: str, label: str) -> None:
    keys, notes = _registered_tool_keys()
    assert len(keys) > 50, (
        f"ground truth looks broken — only {len(keys)} tool keys "
        f"discovered. Notes: {notes}"
    )

    bad: list[str] = []
    for name, lines in sorted(_claimed_names(text).items()):
        if not _resolves(name, keys):
            bad.append(f"{name!r} (line{'s' if len(lines) > 1 else ''} {lines})")

    assert not bad, (
        f"{label} names {len(bad)} tool(s) that resolve to NO registered "
        f"tool — the model will get 'Function X not found' and burn a turn:"
        f"\n  - " + "\n  - ".join(bad)
        + "\n\nFix the name, or — if it is not a tool — add the token to "
        "_NON_TOOL_TOKENS in this file with the group it belongs to."
        + (f"\n\n(servers not introspected: {notes})" if notes else "")
    )


# ── Tests ───────────────────────────────────────────────────────────


@test("prompt_tool_names", "every tool named in FRAMEWORK_SYSTEM_PROMPT is registered")
async def t_framework_prompt_tools_exist(_ctx: TestContext) -> None:
    from src.core.prompts import FRAMEWORK_SYSTEM_PROMPT

    _assert_prompt_tools_exist(FRAMEWORK_SYSTEM_PROMPT, "FRAMEWORK_SYSTEM_PROMPT")


@test("prompt_tool_names", "every tool named in DREAM_MODE_PROMPT is registered")
async def t_dream_prompt_tools_exist(_ctx: TestContext) -> None:
    from src.core.server import DREAM_MODE_PROMPT

    _assert_prompt_tools_exist(DREAM_MODE_PROMPT, "DREAM_MODE_PROMPT")


@test("prompt_tool_names", "the vault tool list matches the vault MCP exactly")
async def t_vault_tool_list_is_complete(_ctx: TestContext) -> None:
    """The prompt hardcodes the vault's tool surface. Pin it to the real
    registration so the list can't silently drift again — it is the exact
    list that carried ``list_notes`` and ``get_backlinks``.
    """
    from src.core.prompts import FRAMEWORK_SYSTEM_PROMPT

    keys, _ = _registered_tool_keys()
    vault_keys = {k for k in keys if k.startswith("vault_")}
    # vault-gate's tools are also ``vault_*``; keep only the vault MCP's.
    from src.mcp.builtins import BUILTIN_MCPS_DIR, BUILTIN_MCP_SPECS

    src = BUILTIN_MCPS_DIR / BUILTIN_MCP_SPECS["vault"]["dir"] / "src/createServer.ts"
    real = {"vault_" + t for t in _TS_TOOL_NAME.findall(src.read_text())}
    assert len(real) == 15, f"vault MCP tool count changed: {len(real)} — {sorted(real)}"
    assert real <= vault_keys

    missing = sorted(k for k in real if k not in FRAMEWORK_SYSTEM_PROMPT)
    assert not missing, (
        "FRAMEWORK_SYSTEM_PROMPT's vault tool list omits registered "
        f"tool(s): {missing}. The model can only copy a key it has seen."
    )


@test("prompt_tool_names", "the counter-example keys are still genuinely wrong")
async def t_counter_examples_still_wrong(_ctx: TestContext) -> None:
    """The other direction of the same contract.

    ``vault_list_notes`` / ``shell_shell_exec`` are taught as the two
    "slips that waste a turn". If one ever became a real registered tool
    the lesson would quietly turn into a lie — the model would be steered
    away from a key that works. Assert every name the prose marks as
    wrong is in fact unregistered.
    """
    from src.core.prompts import FRAMEWORK_SYSTEM_PROMPT
    from src.core.server import DREAM_MODE_PROMPT

    keys, _ = _registered_tool_keys()

    # Named explicitly rather than harvested by context: the negation
    # window that guards the forward direction is deliberately generous
    # (it exempts a non-existent name whenever nearby prose says "never" /
    # "do not"), and lines like "Do NOT emit ``vault_write_note`` as a
    # top-level call" legitimately pair a negation with a REAL key. For
    # this direction we want the two specific lessons, so we spell them out.
    marked = _counter_example_names(FRAMEWORK_SYSTEM_PROMPT)
    marked.update(_counter_example_names(DREAM_MODE_PROMPT))

    for name in ("vault_list_notes", "shell_shell_exec"):
        assert name in marked, (
            f"{name!r} is no longer presented as a counter-example. It is "
            "among the most plausible keys for the model to hallucinate — "
            "keep the 'Two slips waste a turn' block, or update this test."
        )
        assert not _resolves(name, keys), (
            f"the prompt teaches {name!r} as a mistake, but it now resolves "
            "to a real registered tool. Steering the model away from a "
            "working key wastes turns just like the reverse — update the "
            "'Two slips waste a turn' block."
        )
