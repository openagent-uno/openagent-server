"""Tool labels vs reality — the friendly verbs must match real registrations.

``src/channels/tool_labels.py`` maps a tool's bare op to the human verb the
bridges narrate ("📖 Recalling", "🗂️ Browsing memory"). It is a hand-kept
table, and it had drifted in BOTH directions at once:

  * ``list_notes`` and ``get_backlinks`` carried labels — neither tool has
    ever existed on either vault server.
  * ``list_directory``, ``move_file`` and ``get_notes_info`` are real and
    had NO label, so they leaked a raw tool name into the user's chat.
    ``list_directory`` is the vault browsing leaf DREAM_MODE_PROMPT opens
    with ("use `list_directory` and `search_notes` to survey the vault"),
    so every nightly run narrated a raw name.

This is the same defect class ``test_prompt_tool_names`` catches in the
prompts, on a surface that test does not reach. Both exist because a
hand-kept list of tool names drifts away from the code that registers
them — so, as there, ground truth here is DERIVED from the real servers,
never restated.

Scope note: only the memory-vault verbs are checked. ``_MEMORY_VERBS`` is
the only op-keyed table in the module; the rest of the labelling is
keyed by MCP server name, which cannot drift the same way.
"""
from __future__ import annotations

import re
from pathlib import Path

from ._framework import TestContext, test

# The vault-gate MCP is an in-process toolkit, so the pool passes no
# tool_name_prefix and its keys are the bare python function names
# (``vault_backlinks``). The vault MCP is a Node subprocess, so its keys
# ARE prefixed (``vault_list_directory``). Both reduce through
# ``_memory_op`` to the same bare-op space this table is keyed on.
_TS_TOOL_NAME = re.compile(r'^\s*name:\s*"([a-z][a-z0-9_]*)"', re.M)

# Ops that legitimately have no MCP registration behind them.
_NON_MCP_OPS: frozenset[str] = frozenset({
    # Emitted by the vendored runtime's own memory manager, not an MCP.
    # Guarded by is_memory_tool()'s explicit name check.
    "update_user_memory",
    # Alias kept for the obsidian-MCP raw name vs the vault-gate name
    # (see the module docstring's "a few ops carry aliases").
    "search",
    "stats",
})


def _real_vault_ops() -> set[str]:
    """Every bare op the two vault servers actually register."""
    from src.channels.tool_labels import _memory_op
    from src.mcp.builtins import BUILTIN_MCP_SPECS, BUILTIN_MCPS_DIR

    ops: set[str] = set()

    # vault — vendored Node server; parse its ListTools handler rather than
    # requiring a built dist/ at test time.
    src = BUILTIN_MCPS_DIR / BUILTIN_MCP_SPECS["vault"]["dir"] / "src/createServer.ts"
    assert src.exists(), f"vault server source missing at {src}"
    for t in _TS_TOOL_NAME.findall(src.read_text()):
        ops.add(_memory_op(f"vault_{t}"))

    # vault-gate — in-process toolkit; ask the factory what it registered.
    import importlib

    spec = BUILTIN_MCP_SPECS["vault-gate"]
    mod = importlib.import_module(spec["adapter_module"])
    factory = getattr(mod, spec.get("runtime_toolkit_factory", "build_runtime_toolkit"))
    toolkit = factory()
    fns = dict(getattr(toolkit, "functions", {}) or {})
    fns.update(getattr(toolkit, "async_functions", {}) or {})
    for key in fns:
        ops.add(_memory_op(key))

    return ops


@test("tool_labels", "every labelled memory op is a real registered tool")
async def t_no_phantom_labels(ctx: TestContext) -> None:
    from src.channels.tool_labels import _MEMORY_VERBS

    real = _real_vault_ops()
    phantom = sorted(
        op for op in _MEMORY_VERBS
        if op not in real and op not in _NON_MCP_OPS
    )
    assert not phantom, (
        f"tool_labels labels ops no server registers: {phantom}. Dead weight, "
        "and a sign the table drifted from the code. Real ops: "
        f"{sorted(real)}"
    )


@test("tool_labels", "every real vault tool has a friendly label")
async def t_no_unlabelled_tools(ctx: TestContext) -> None:
    """The one that actually bites a user.

    An unlabelled op falls through to the raw tool name in a Telegram /
    Discord / Slack / WhatsApp message. `list_directory` was in this state
    while `list_notes` — which does not exist — held the label meant for it.
    """
    from src.channels.tool_labels import _MEMORY_VERBS

    unlabelled = sorted(_real_vault_ops() - set(_MEMORY_VERBS))
    assert not unlabelled, (
        f"real vault tools with no friendly label: {unlabelled}. Each one "
        "narrates a raw tool name into the user's chat. Add a verb to "
        "_MEMORY_VERBS (and mirror it in the app's toolDisplay)."
    )


@test("tool_labels", "the vault browsing leaf resolves to its verb end-to-end")
async def t_list_directory_resolves(ctx: TestContext) -> None:
    """Pin the specific regression, through the real public entry point.

    _memory_op strips the server prefix, so both the prefixed key and the
    dispatcher-unwrapped bare name must land on the same verb.
    """
    from src.channels.base import ToolStatusEvent
    from src.channels.tool_labels import _memory_op, is_memory_tool, status_line

    assert _memory_op("vault_list_directory") == "list_directory"
    assert _memory_op("list_directory") == "list_directory"

    for tool in ("vault_list_directory", "list_directory"):
        evt = ToolStatusEvent(tool=tool, status="running", server="vault")
        assert is_memory_tool(evt), f"{tool} not recognised as a memory tool"
        line = status_line(evt)
        assert "Browsing memory" in line, (
            f"{tool} narrated as {line!r} — the raw name leaked instead of "
            "the friendly verb."
        )


@test("tool_labels", "the module names no tool that doesn't exist")
async def t_docstring_names_real_tools(ctx: TestContext) -> None:
    """The docstring cited ``shell_shell_exec`` as an example raw name.

    That key does not exist — in-process toolkits keep the bare function
    name, so the shell server registers ``shell_exec``. A doc example is
    how the next person learns the naming rule; teaching the wrong one is
    how this drift propagates.
    """
    src = Path("src/channels/tool_labels.py").read_text()
    assert "shell_shell_exec" not in src, (
        "tool_labels.py names `shell_shell_exec`, which no server registers "
        "(the real key is `shell_exec` — in-process toolkits are unprefixed)."
    )
