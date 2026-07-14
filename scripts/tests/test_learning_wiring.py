"""Learning hooks — wiring + provider-agnosticism tests.

``test_vault_reminder.py`` covers the reminder MODULE (counter mechanic, text).
It passed for the entire time the feature was silently off in production,
because the module's default was always correct — it was the sole CALL SITE
(``bridges/base.py``) that re-implemented the enabled check with the opposite
default and won. So the tests here deliberately test the things that failure
mode lives in, and that a module-level unit test structurally cannot see:

  1. the reminder actually reaches a turn on the SHARED agent run path, so
     every origin gets it (§15) rather than bridge-mediated chats only;
  2. no call site anywhere re-implements the enabled check;
  3. nothing under ``src/learning`` pins a specific vendor SDK (§17).
"""
from __future__ import annotations

import os
from pathlib import Path

from ._framework import TestContext, test

_SRC = Path(__file__).resolve().parents[2] / "src"


async def _make_db() -> object:
    """Return a connected in-memory DB shim with the real schema."""
    import aiosqlite

    class _Shim:
        def __init__(self, conn):
            self._conn = conn

    conn = await aiosqlite.connect(":memory:")
    from src.memory.db import SCHEMA_SQL
    await conn.executescript(SCHEMA_SQL)
    return _Shim(conn)


@test("learning_wiring", "shared run path prepends the reminder to the turn input")
async def t_shared_path_injects(ctx: TestContext) -> None:
    # Default-on (no env set) — the state a fresh install boots in, and the
    # exact case the bridge call site used to override to off.
    for k in ("OPENAGENT_VAULT_REMINDER_ENABLED", "OPENAGENT_VAULT_REMINDER_EVERY_N_TURNS"):
        os.environ.pop(k, None)
    import importlib
    from src.learning import vault_reminder
    importlib.reload(vault_reminder)

    from src.core.agent import _with_vault_reminder

    db = await _make_db()
    out = await _with_vault_reminder(db, "sess-wired", "what is the deploy command?")
    assert out.endswith("what is the deploy command?"), f"user text not preserved: {out!r}"
    assert out != "what is the deploy command?", "expected a reminder to be prepended on turn 1"
    assert "memory checkpoint" in out.lower(), f"reminder text missing: {out!r}"


@test("learning_wiring", "reminder hook is a no-op when disabled / no session")
async def t_hook_noop(ctx: TestContext) -> None:
    os.environ["OPENAGENT_VAULT_REMINDER_ENABLED"] = "0"
    try:
        import importlib
        from src.learning import vault_reminder
        importlib.reload(vault_reminder)
        from src.core.agent import _with_vault_reminder

        db = await _make_db()
        assert await _with_vault_reminder(db, "s", "hi") == "hi", "disabled must pass text through"
        # No session id (one-off runs) and no db must both pass through cleanly.
        assert await _with_vault_reminder(db, None, "hi") == "hi"
        assert await _with_vault_reminder(None, "s", "hi") == "hi"
    finally:
        os.environ.pop("OPENAGENT_VAULT_REMINDER_ENABLED", None)


@test("learning_wiring", "reminder hook never fails a turn when the DB misbehaves")
async def t_hook_never_raises(ctx: TestContext) -> None:
    for k in ("OPENAGENT_VAULT_REMINDER_ENABLED",):
        os.environ.pop(k, None)
    import importlib
    from src.learning import vault_reminder
    importlib.reload(vault_reminder)
    from src.core.agent import _with_vault_reminder

    class _Exploding:
        @property
        def _conn(self):
            raise RuntimeError("db is on fire")

    # A memory nudge must never be able to take down a user turn.
    out = await _with_vault_reminder(_Exploding(), "sess-boom", "hello")
    assert out == "hello", f"expected clean passthrough on DB failure, got {out!r}"


@test("learning_wiring", "only vault_reminder.py owns the enabled flag's default")
async def t_no_reimplemented_flag_check(ctx: TestContext) -> None:
    """The v0.15.11 bug, pinned.

    ``OPENAGENT_VAULT_REMINDER_ENABLED`` defaulted to "1" in the module and
    "0" at its only call site, so the feature was off everywhere while the
    module, its tests, and the docs all said on-by-default. A re-implemented
    flag check at a call site is the defect — not the literal that was wrong.

    Allowed: the module that defines the default, and server.py's yaml→env
    mapping (which sets the var rather than defaulting it).
    """
    allowed = {"learning/vault_reminder.py", "core/server.py"}
    offenders = []
    for path in _SRC.rglob("*.py"):
        rel = path.relative_to(_SRC).as_posix()
        if rel in allowed:
            continue
        if "OPENAGENT_VAULT_REMINDER_ENABLED" in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(rel)
    assert not offenders, (
        "these files read OPENAGENT_VAULT_REMINDER_ENABLED directly; call "
        "maybe_render_reminder() unconditionally instead (it self-gates): "
        f"{offenders}"
    )


@test("learning_wiring", "learning model no-ops when unconfigured, never picks one itself")
async def t_learning_model_degrades(ctx: TestContext) -> None:
    os.environ.pop("OPENAGENT_LEARNING_MODEL", None)
    from src.learning._model import complete

    class _DB:
        db_path = ":memory:"

        async def materialise_providers_config(self, *, enabled_only=False):
            # A fully-configured catalog. Unconfigured learning model must
            # still no-op rather than helpfully grabbing the user's model:
            # this runs unattended on a timer, so an inferred pick is a
            # silent recurring bill. See learning/_model.py.
            return [{
                "id": 1, "name": "anthropic", "framework": "api-based",
                "api_key": "sk-test", "enabled": True,
                "models": [{"id": 9, "model": "claude-opus-4-8", "enabled": True}],
            }]

    assert await complete(db=_DB(), prompt="hi", log_event="test.learning") is None, (
        "unset OPENAGENT_LEARNING_MODEL must no-op, not auto-select a model"
    )


# Modules under src/learning that are KNOWN to still pin a vendor and are
# awaiting deletion rather than repair. Every entry here is a live §17
# violation; the list must only ever shrink.
#
# semantic_search: embeds via OpenAI's text-embedding-3-small and scans with a
#   brute-force Python cosine loop. Recommended for deletion along with the
#   memory-search MCP that is its only reader — the table it writes has never
#   had a writer wired, and §5 rules out a "hidden vector store" as the shape
#   of the agent's memory in the first place. Deleting it needs the
#   registration pulled from src/mcp/builtins.py + src/cli.py and the mention
#   dropped from src/core/prompts.py, which are owned elsewhere; until then it
#   stays put, opt-in and off by default, and the prompt already tells the
#   model to treat it as empty.
_KNOWN_VENDOR_PINNED = {"semantic_search.py"}

_BANNED_IMPORTS = {"groq", "openai"}
_BANNED_NAMES = {"AsyncGroq", "AsyncOpenAI", "OpenAI", "Groq"}
_BANNED_STRINGS = {"GROQ_API_KEY", "OPENAI_API_KEY"}


@test("learning_wiring", "src/learning pins no vendor SDK (vision §17)")
async def t_no_hardcoded_provider(ctx: TestContext) -> None:
    """§17: "Removing any single provider … must leave the agent operational
    with what remains."

    The learning loops used to import the ``groq`` SDK and resolve a
    GROQ_API_KEY directly, which made one vendor a structural requirement for
    the agent to maintain its own memory — an agent on purely local models
    could never dream (§12). Model access now goes through
    ``_model.complete`` → the catalog → NativeProvider.

    Parsed with ``ast`` rather than grepped, so the docstrings that explain
    *why* those vendors are gone don't read as the thing they warn about.
    """
    import ast

    offenders: list[str] = []
    for path in sorted((_SRC / "learning").rglob("*.py")):
        if path.name in _KNOWN_VENDOR_PINNED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        # Docstrings are the only bare string-expression statements; skip them
        # so prose about GROQ_API_KEY isn't mistaken for a lookup of it.
        docstrings = {
            id(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        }
        rel = path.relative_to(_SRC).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in _BANNED_IMPORTS:
                        offenders.append(f"{rel}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in _BANNED_IMPORTS:
                    offenders.append(f"{rel}: from {node.module} import …")
            elif isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
                offenders.append(f"{rel}: {node.id}")
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and node.value in _BANNED_STRINGS
            ):
                offenders.append(f"{rel}: {node.value!r}")
    assert not offenders, (
        "vendor SDK pinned in src/learning — route model access through "
        f"_model.complete() instead: {offenders}"
    )


@test("learning_wiring", "the known vendor-pinned allowlist stays accurate")
async def t_known_vendor_allowlist_is_real(ctx: TestContext) -> None:
    """An allowlist that outlives what it excuses is how a violation becomes
    permanent. Fail once ``semantic_search.py`` is gone (or repaired) so the
    entry — and this test — get removed with it.
    """
    for name in _KNOWN_VENDOR_PINNED:
        path = _SRC / "learning" / name
        assert path.exists(), (
            f"{name} is allowlisted as vendor-pinned but no longer exists — "
            "drop it from _KNOWN_VENDOR_PINNED."
        )
        assert "AsyncOpenAI" in path.read_text(encoding="utf-8", errors="ignore"), (
            f"{name} no longer pins a vendor — drop it from _KNOWN_VENDOR_PINNED."
        )
