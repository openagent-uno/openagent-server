"""There is exactly ONE dream mode, and it is the scheduled task.

Two things were called "dream mode" and neither did the whole job:

  * ``dream_mode.enabled`` — a built-in scheduled task firing
    ``DREAM_MODE_PROMPT`` as a full child session (the AI/judgement half:
    curate the vault, triage the logs, notice recurring work).
  * ``memory.vault.maintenance.enabled`` — a separate 12-hourly asyncio loop
    in ``learning/vault_maintenance.py`` calling ``VaultService.maintenance()``
    (the mechanical half: sync → gate → doctor → derive), plus one AI
    suggestion step, writing its own differently-formatted log to
    ``workspace/_dream/``.

Both defaulted off, both could run at once, and neither knew the other
existed — the loop's own docstring said "Independent of the curator loop".

Vision §12 describes ONE thing: "The agent runs a scheduled 'dream' task that
maintains its memory vault ... nightly by default, at a time the user can
adjust. It does not compete with user-facing work." A scheduled task is that.
An interval loop counting 12h from boot fires at 2pm mid-conversation half the
time, and no ``time:`` or ``timezone:`` setting could ever move it.

The loop was deleted in v0.16.1. These tests pin the three ways that deletion
could rot:

  1. the loop comes back (or a new one is added beside it);
  2. the retired config keys turn from inert into fatal, so an existing
     ``openagent.yaml`` stops booting on upgrade;
  3. the retired keys stay half-alive — still parsed into env vars nothing
     reads, which is the exact defect ``t_no_write_only_safety_env`` in
     ``test_safety.py`` exists to punish. A config key that does nothing is
     worse than no key: it greps like a live feature.

``test_dream.py::t_dream_runs_mechanical_pass`` is the other half of this
contract and is now load-bearing in a way it was not before: with the loop
gone, that assertion is the only thing keeping the mechanical pass wired to
anything at all.
"""
from __future__ import annotations

import os
import re
import tokenize
from pathlib import Path

from ._framework import TestContext, test

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"

# Every env var the retired ``memory.vault.maintenance.*`` / ``memory.
# learning_model`` wiring used to export. None may be written OR read again:
# the loop that read them is gone, so a writer would be write-only (defect
# class above) and a reader would be a resurrected second dream mode.
_RETIRED_ENV = (
    "OPENAGENT_VAULT_MAINTENANCE_ENABLED",
    "OPENAGENT_VAULT_MAINTENANCE_AUTOFIX",
    "OPENAGENT_VAULT_MAINTENANCE_DERIVED",
    "OPENAGENT_VAULT_MAINTENANCE_INTERVAL_HOURS",
    "OPENAGENT_LEARNING_MODEL",
)

# A v0.16.0 config, verbatim from the shipped ``examples/openagent.full.yaml``
# plus the ``learning_model`` key the docs advertised. This is what a real
# deployment's file looks like the moment before it is upgraded.
_OLD_YAML = """\
name: TestAgent
model:
  provider: anthropic
  model: claude-opus-4-8
memory:
  learning_model: "groq:llama-3.3-70b-versatile"
  vault:
    enabled: true
    max_lines: 300
    git:
      enabled: false
    maintenance:
      enabled: true
      interval_hours: 12
      autofix: true
      regenerate_derived: true
dream_mode:
  enabled: true
  time: "3:00"
"""


def _code_only(path: Path) -> str:
    """Source with comments blanked out.

    Load-bearing, and copied from ``t_no_write_only_safety_env`` for the same
    reason it needed it: ``server.py`` now *discusses* every one of these
    retired vars in prose, at length. Counting a comment as a reader (or a
    writer) would let the real thing hide behind its own tombstone — which is
    how that test first shipped vacuous.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    try:
        with open(path, "rb") as f:
            toks = list(tokenize.tokenize(f.readline))
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return text
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            row, col = tok.start
            lines[row - 1] = lines[row - 1][:col]
    return "\n".join(lines)


@test("dream", "the second dream mode (the 12h loop) is gone, not just disabled")
async def t_no_vault_maintenance_module(ctx: TestContext) -> None:
    """The loop and the model helper that existed only to feed it.

    ``_model.py`` goes with the loop because the loop's AI-suggestion step was
    its ONLY caller, and that step is strictly weaker than what replaced it: it
    asked a cheap model for prose advice about the issues code could not fix
    ("this orphan should link to X") and wrote the advice into a log note that
    nothing read back. The scheduled task gets the same ``open_suggestions``
    from ``vault_dream()`` while holding ``write_note``/``patch_note``/
    ``delete_note`` — it merges the duplicate instead of noting that someone
    ought to.
    """
    import importlib

    for mod in ("src.learning.vault_maintenance", "src.learning._model"):
        try:
            importlib.import_module(mod)
        except ImportError:
            continue
        raise AssertionError(
            f"{mod} is back. Dream mode is the `dream-mode` SCHEDULED TASK "
            "(vision §12: 'nightly by default, at a time the user can "
            "adjust'). A 12-hourly asyncio loop cannot be aimed at a time and "
            "fires mid-conversation; if the mechanical pass needs to run, it "
            "runs as `vault_dream()` inside the task's Mission 1."
        )

    assert not (_SRC / "learning" / "vault_maintenance.py").exists()
    assert not (_SRC / "learning" / "_model.py").exists()


@test("dream", "nothing under src/learning runs a vault pass on a timer")
async def t_learning_holds_no_vault_loop(ctx: TestContext) -> None:
    """Deleting the file is not the same as deleting the pattern.

    The failure this guards is the loop coming back under a new name — the
    cheapest possible way to "fix" a vault problem is to add a background
    sweep, and it would be off by default, so nobody would notice it landing
    beside the task again. ``VaultService.maintenance()`` has exactly one
    production caller now: the ``vault_dream`` MCP tool, which the scheduled
    task invokes.

    Parsed with ``ast`` rather than grepped — like ``t_no_hardcoded_provider``
    next door, and for the identical reason. ``learning/__init__.py`` now
    *explains* that the deleted loop called ``svc.maintenance(...)``; a text
    scan reads that tombstone as the crime it describes. (It did: this test
    failed on its own docstring before this comment existed.)
    """
    import ast

    offenders = []
    for path in sorted((_SRC / "learning").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        rel = path.relative_to(_SRC).as_posix()
        for node in ast.walk(tree):
            # `svc.maintenance(...)` — the pass itself.
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "maintenance"
            ):
                offenders.append(f"{rel}: .maintenance() call")
            # `from src.memory.vault.service import get_service` — the handle
            # you would need to reach it.
            elif isinstance(node, ast.ImportFrom) and "vault" in (node.module or ""):
                offenders.append(f"{rel}: imports {node.module}")
    assert not offenders, (
        "src/learning is driving the vault maintenance pass again: "
        f"{offenders}. That pass belongs to dream mode's Mission 1 via the "
        "`vault_dream` MCP tool, where the user can aim it at an hour."
    )


@test("dream", "retired maintenance keys are neither written nor read (no write-only env)")
async def t_no_write_only_maintenance_env(ctx: TestContext) -> None:
    """The bar this whole change is held to.

    Five ``OPENAGENT_SAFETY_*`` vars sat in ``_build_agent`` for months, set
    from yaml and read by nobody, while the example config advertised them as
    protection. Retiring ``memory.vault.maintenance.*`` by deleting its READER
    and keeping its WRITER would have reproduced that exactly — and this time
    the var is named ...MAINTENANCE_ENABLED, so it would grep like the vault
    was being maintained.

    Deliberately asserts BOTH directions. No writer (it would be write-only);
    no reader (that would mean the loop is back).
    """
    written: dict[str, str] = {}
    read: dict[str, str] = {}
    for name in _RETIRED_ENV:
        w_re = re.compile(r"os\.environ\[\s*[\"']" + name + r"[\"']\s*\]\s*=[^=]")
        # Reads come in two shapes here: the stdlib lookups, and the
        # ``_int_env``/``_bool_env`` helpers the learning loops used — the
        # scanner in test_safety.py only knows the former, and would have
        # called the live OPENAGENT_CURATOR_SESSION_RETENTION_DAYS write-only
        # for exactly that reason. Match the name inside ANY call.
        r_re = re.compile(
            r"(?:os\.environ\.get\(|os\.getenv\(|_int_env\(|_bool_env\(|"
            r"[A-Za-z_]\w*_env\()\s*\n?\s*[\"']" + name + r"[\"']"
            r"|os\.environ\[\s*[\"']" + name + r"[\"']\s*\](?!\s*=[^=])"
            r"|^\s*[A-Za-z_]\w*\s*=\s*[\"']" + name + r"[\"']\s*$",
            re.MULTILINE,
        )
        for py in _SRC.rglob("*.py"):
            code = _code_only(py)
            if w_re.search(code):
                written.setdefault(name, py.relative_to(_SRC).as_posix())
            if r_re.search(code):
                read.setdefault(name, py.relative_to(_SRC).as_posix())

    assert not written, (
        "these retired vars are exported from yaml again, but the loop that "
        f"read them is deleted — that is a write-only env var: {written}"
    )
    assert not read, (
        "something reads a retired vault-maintenance var — the second dream "
        f"mode is back: {read}"
    )


@test("dream", "an old openagent.yaml with maintenance keys still boots (inert, not fatal)")
async def t_old_config_still_boots(ctx: TestContext) -> None:
    """Retired keys degrade, they do not explode.

    Precedent and the reason it exists: ``memory.user_profile`` /
    ``memory.skills`` (v0.15.11) and ``safety.guardrails`` /
    ``safety.compression``. An operator upgrading a server should never have to
    edit yaml to boot — a config carrying a block we retired is a config we
    made stale, not a user error. So an unknown/retired key must be ignored,
    and must not leak an env var on its way past.
    """
    import yaml

    from src.core.server import _build_agent

    cfg = yaml.safe_load(_OLD_YAML)
    # Snapshot the WHOLE environment, not just ``_RETIRED_ENV``.
    #
    # ``_build_agent`` exports the yaml's config as process-global env vars —
    # that is how the subprocess MCPs inherit it — so it leaves far more behind
    # than the keys this test is about. ``_OLD_YAML`` sets
    # ``memory.vault.git.enabled: false``, which lands as
    # ``OPENAGENT_VAULT_GIT_ENABLED=0`` and STAYS there. Restoring only the
    # retired keys leaked it into every later test in the run: five vault-git
    # tests then failed with ``commit: None`` because git was globally off, and
    # they passed in isolation, which made it look like flakiness.
    #
    # A hand-listed set of keys to restore is the same defect this whole test
    # file is about — it drifts the moment ``_build_agent`` learns a new key.
    # Snapshot everything instead.
    prev_environ = dict(os.environ)
    for k in _RETIRED_ENV:
        os.environ.pop(k, None)
    try:
        # The whole point: this must not raise on the retired block.
        agent = _build_agent(cfg)
        assert agent is not None

        leaked = {k: os.environ[k] for k in _RETIRED_ENV if k in os.environ}
        assert not leaked, (
            "a retired key was still parsed into the environment; it is "
            f"supposed to be ignored entirely: {leaked}"
        )
        # And the live key beside it must still work — "inert" applies to the
        # retired block only, not to the section containing it.
        assert os.environ.get("OPENAGENT_VAULT_MAX_LINES") == "300", (
            "memory.vault.max_lines stopped being read; removing the "
            "maintenance sub-block broke its parent section"
        )
    finally:
        os.environ.clear()
        os.environ.update(prev_environ)


@test("dream", "examples/openagent.full.yaml documents no retired maintenance keys")
async def t_example_yaml_drops_maintenance(ctx: TestContext) -> None:
    """The reference config must not advertise a pass that never runs.

    This is the same assertion ``test_safety.py::t_example_yaml_safety_is_true``
    makes about the safety block, for the same reason: the example config is
    what operators copy, and it said ``maintenance.enabled`` toggled the
    vault's upkeep long after the toggle would have done nothing.
    """
    import yaml

    p = _ROOT / "examples" / "openagent.full.yaml"
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    vault = ((cfg.get("memory") or {}).get("vault") or {})

    assert "maintenance" not in vault, (
        "the reference config advertises memory.vault.maintenance.*, which "
        "the server no longer reads. Dream mode is the `dream_mode` task."
    )
    assert "learning_model" not in (cfg.get("memory") or {}), (
        "the reference config advertises memory.learning_model, whose only "
        "consumer (the maintenance loop's AI step) is deleted."
    )
    # The git sub-block sits directly above where maintenance was; a
    # copy-paste slip that removed the wrong key would still pass everything
    # above this line.
    assert vault.get("git", {}).get("enabled") is True, (
        "memory.vault.git went missing along with the maintenance block"
    )


@test("dream", "AgentServer no longer starts or stops a vault-maintenance task")
async def t_no_maint_task_lifecycle(ctx: TestContext) -> None:
    """A cancelled handle for a task nobody starts is how dead loops linger.

    The shutdown path used to cancel ``_vault_maint_task`` in the same tuple as
    ``_vault_autocommit_task``. Leaving that name in the tuple would be
    harmless at runtime (``getattr`` returns ``None``) and precisely the kind
    of residue that makes the next reader think the loop still exists.
    """
    code = _code_only(_SRC / "core" / "server.py")
    assert "_vault_maint_task" not in code, (
        "the vault-maintenance task handle is still in server.py's lifecycle "
        "code; the loop it belonged to is deleted"
    )
    assert "vault_maintenance" not in code, (
        "server.py still imports/starts the deleted vault_maintenance module"
    )
    # The autocommit loop shares that shutdown block and MUST survive — it is
    # the safety net for edits made outside OpenAgent's tools (Obsidian).
    assert "_vault_autocommit_task" in code, (
        "the vault autocommit loop was deleted along with the maintenance "
        "loop; it is unrelated and still load-bearing"
    )


@test("dream", "the boot test restores the environment it dirtied")
async def t_boot_test_restores_env(ctx: TestContext) -> None:
    """The bug this file shipped, pinned so it cannot come back.

    ``_build_agent`` exports the whole config as process-global env vars — that
    is its job, it is how subprocess MCPs inherit the config. So any test that
    calls it MUST put the environment back, or it mutates every test that runs
    after it.

    ``t_old_config_still_boots`` restored a hand-listed set of keys and missed
    the rest. Its ``_OLD_YAML`` carries ``memory.vault.git.enabled: false``,
    which lands as ``OPENAGENT_VAULT_GIT_ENABLED=0`` and stayed set for the
    remainder of the run: five vault-git tests then failed with
    ``commit: None`` because git was globally off. They passed in isolation, so
    it read as flakiness and cost a bisect to find.

    So this asserts the real invariant — the sibling test cleans up after
    itself — by running it and diffing the environment around it. It does not
    assert ``_build_agent`` is pure; it is not, and it should not be.
    """
    # Clear the keys FIRST. Order matters and this is where my own first
    # version of this guard was vacuous: t_old_config_still_boots runs earlier
    # in the file, so by the time this ran the leak was already in the
    # environment — snapshotting it as the baseline made the diff empty and the
    # guard passed against the very defect it was written for. Start from a
    # known-clean slate for exactly the keys _OLD_YAML sets.
    leaky = ("OPENAGENT_VAULT_GIT_ENABLED", "OPENAGENT_VAULT_ENABLED",
             "OPENAGENT_VAULT_MAX_LINES") + _RETIRED_ENV
    outer = dict(os.environ)
    for k in leaky:
        os.environ.pop(k, None)
    before = dict(os.environ)
    try:
        await t_old_config_still_boots(ctx)
        after = dict(os.environ)
    finally:
        os.environ.clear()
        os.environ.update(outer)

    dirty = {k: (before.get(k), after.get(k))
             for k in set(before) | set(after)
             if before.get(k) != after.get(k)}
    assert not dirty, (
        "t_old_config_still_boots leaked env into the rest of the run. "
        "Snapshot and restore the WHOLE environment (dict(os.environ), then "
        "clear() + update()) — never a hand-listed key set, which drifts the "
        "moment _build_agent learns a new key. Leaked: "
        + ", ".join(f"{k}: {o!r} -> {n!r}" for k, (o, n) in sorted(dirty.items()))
    )
