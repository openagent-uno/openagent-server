"""Repo-hygiene guards — designed to catch the *kind* of bug we
just fixed (PR #1 / commit a61b943: ``executor.py`` was truncated
from 990 lines to 60 because an LLM-generated patch put a literal
docstring saying "CONTENT IS OMITTED FOR BREVITY - see actual fix
below" verbatim into the file and the commit slipped past zero
reviews).

Three tests, each catching a different layer the original bug crossed:

1. **No LLM placeholder strings in source** — grep ``src/`` for the
   stock "I'm summarising the unchanged part" phrases that AI
   assistants emit when they don't want to repeat a large file. Cheap,
   fast, catches the exact failure mode of a61b943.

2. **Every test_*.py is registered** — compare the on-disk file list
   under ``scripts/tests/`` against the explicit ``_TEST_MODULES``
   tuple in the runner. Without this assertion a ``test_*.py`` written
   but never wired up (caught today: ``test_agno_stream.py`` was in the
   tree since v0.13.x but the runner never imported it) reports
   "everything green" while silently skipping the coverage.

3. **Every src module imports cleanly** — walk ``src/**/*.py`` and
   ``importlib.import_module`` each. Catches missing-symbol /
   broken-import bugs even in modules that no unit test currently
   exercises (the most general case of a61b943: a truncated file
   whose missing class is referenced only at production runtime, like
   ``WorkflowExecutor`` being imported from ``src.core.scheduler``).

These are not behavioural tests — they're structural guards. They
should be cheap to run and very rare to flake. Keep their failure
messages explicit so a future reader knows exactly which file/line
tripped them.
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from ._framework import TestContext, test


# Strings that LLM coding assistants commonly emit as "compact diff"
# placeholders. None of these belong in a real source file — every
# entry here is empirically observed in production AI-generated
# patches that were committed verbatim. Match is substring + case
# insensitive on the lowercased line.
_PLACEHOLDER_NEEDLES: tuple[str, ...] = (
    "content is omitted for brevity",
    "content omitted for brevity",
    "rest of the file unchanged",
    "rest of file unchanged",
    "rest of file remains the same",
    "unchanged content omitted",
    "base64-encoded-meaningful-content-omitted",
    "// ... existing code ...",
    "# ... existing code ...",
    "<existing code>",
    "[file content omitted",
)

# A literal one-line file that ONLY contains a docstring saying
# "see fix below" is a degenerate signal we've also seen in the
# wild. Catch it separately so the error message can be specific.
_PLACEHOLDER_REGEX_PHRASE = "see actual fix below"


def _repo_root() -> Path:
    """Resolve the openagent-server repo root from this test file."""
    here = Path(__file__).resolve()
    # scripts/tests/test_repo_hygiene.py → repo root is 2 up
    return here.parent.parent.parent


@test("repo_hygiene", "no LLM placeholder strings in src/")
async def t_no_llm_placeholders(_ctx: TestContext) -> None:
    """Walk src/ and fail if any .py file contains an obvious
    AI-generated "content omitted" placeholder. Specifically guards
    against the regression of PR #1 (commit a61b943): a docstring
    saying "CONTENT IS OMITTED FOR BREVITY - see actual fix below"
    was committed as the entire file, deleting 990 lines of
    WorkflowExecutor + handlers. The placeholder was syntactically
    valid Python, so nothing automated caught it until production
    workflow scheduler failures surfaced 2 days later.
    """
    root = _repo_root() / "src"
    assert root.is_dir(), f"src/ not found at {root}"

    hits: list[str] = []
    for path in sorted(root.rglob("*.py")):
        # Skip __pycache__ / generated noise (rglob already does, but
        # be explicit so a future addition to the tree doesn't bite).
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            hits.append(f"{path}: cannot read ({e})")
            continue
        lower = text.lower()
        for needle in _PLACEHOLDER_NEEDLES:
            if needle in lower:
                # Show the offending line + needle so a real
                # legitimate occurrence (e.g. a test fixture string)
                # is easy to spot and either fix or whitelist.
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if needle in line.lower():
                        hits.append(
                            f"{path.relative_to(_repo_root())}:{lineno}: "
                            f"contains LLM placeholder {needle!r}: "
                            f"{line.strip()[:120]}"
                        )
                        break
        if _PLACEHOLDER_REGEX_PHRASE in lower:
            hits.append(
                f"{path.relative_to(_repo_root())}: file contains the "
                f"phrase {_PLACEHOLDER_REGEX_PHRASE!r} — almost certainly "
                "an AI-generated 'see fix below' placeholder that was "
                "committed without expansion"
            )

    assert not hits, (
        "LLM placeholder strings found in source — these are almost "
        "always evidence of an AI-generated patch that was committed "
        "without expanding the omitted content:\n  "
        + "\n  ".join(hits)
    )


@test("repo_hygiene", "every test_*.py is registered in _TEST_MODULES")
async def t_every_test_file_registered(_ctx: TestContext) -> None:
    """Every ``scripts/tests/test_*.py`` file must appear in
    ``scripts/test_openagent.py``'s ``_TEST_MODULES`` tuple. Without
    this guard, a test author can write a perfectly good ``test_*.py``
    and have it silently never run (caught today: ``test_agno_stream.py``
    was in tree but unregistered, so the ``stream zero-delta fallback``
    coverage was dead since it shipped).
    """
    from scripts.test_openagent import _TEST_MODULES

    tests_dir = _repo_root() / "scripts" / "tests"
    assert tests_dir.is_dir(), f"scripts/tests not found at {tests_dir}"

    on_disk: set[str] = set()
    for path in sorted(tests_dir.glob("test_*.py")):
        if path.name.startswith("_") or path.name == "__init__.py":
            continue
        on_disk.add(path.stem)

    registered: set[str] = set(_TEST_MODULES)
    unregistered: set[str] = on_disk - registered - _REGISTRATION_EXCEPTIONS
    missing_from_disk: set[str] = registered - on_disk

    msg_parts: list[str] = []
    if unregistered:
        msg_parts.append(
            "test files on disk but NOT in _TEST_MODULES (will silently "
            "never run): " + ", ".join(sorted(unregistered))
        )
    if missing_from_disk:
        msg_parts.append(
            "_TEST_MODULES entries with no file on disk (typo / "
            "deletion): " + ", ".join(sorted(missing_from_disk))
        )
    assert not msg_parts, "; ".join(msg_parts)


# Modules that legitimately cannot be imported at test time (no real
# bug — they're conditional / platform / entry-point modules with
# import-time side effects). Keep this list short and annotate WHY
# each entry is here so it's reviewed when the file changes.
_SMOKE_IMPORT_SKIP: frozenset[str] = frozenset({
    # CLI entry point — uses argparse at module-import time in some
    # PyInstaller invocations and is not meant to be imported as a
    # library. Coverage already comes from the live CLI tests.
    "src.cli",
    # Frozen-build runtime helper; some branches assume sys.frozen.
    "src._frozen",
})


# Test files under ``scripts/tests/`` that use a different test
# framework (pytest, etc.) and intentionally do NOT belong in
# ``_TEST_MODULES``. Keep this list short and explain why each entry
# bypasses the runner.
_REGISTRATION_EXCEPTIONS: frozenset[str] = frozenset({
    # Runs under ``python -m pytest`` — needs the native
    # ``openagent-computer-control`` binary on the host (macOS
    # Accessibility prompt on first run). Not hermetic, not in CI.
    "test_computer_control_native",
})


@test("repo_hygiene", "no stale 'from src.X' imports anywhere in src/ (incl. lazy)")
async def t_no_stale_src_imports(_ctx: TestContext) -> None:
    """AST-walk every ``src/`` file and verify every ``from src.X`` /
    ``import src.X`` statement points at a module that actually exists,
    INCLUDING lazy imports nested inside functions, classes, or
    try/except blocks.

    The plain smoke-import test (below) only catches stale module-level
    imports — a stale ``from src.models.smart_router import SmartRouter``
    hidden inside a handler function fires only when the HTTP endpoint
    is called. This test catches it before deployment.
    """
    import ast
    import pkgutil
    import src as _src_pkg

    valid_modules: set[str] = {"src"}
    for _finder, name, _ispkg in pkgutil.walk_packages(
        path=_src_pkg.__path__,  # type: ignore[attr-defined]
        prefix="src.",
    ):
        valid_modules.add(name)

    src_root = Path(_src_pkg.__file__).parent  # type: ignore[arg-type]
    failures: list[tuple[str, int, str]] = []

    for py_file in src_root.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            mods: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
                mods.append((node.module, node.lineno))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("src."):
                        mods.append((alias.name, node.lineno))
            for mod_name, lineno in mods:
                if mod_name not in valid_modules:
                    rel = py_file.relative_to(src_root.parent)
                    failures.append((str(rel), lineno, mod_name))

    assert not failures, (
        "Stale src.* imports — targets don't exist as modules. Lazy "
        "imports inside function bodies aren't caught by the smoke-"
        "import test because the function never runs at import time:\n  "
        + "\n  ".join(f"{file}:{line} → {mod}" for file, line, mod in failures)
    )


@test("repo_hygiene", "every src/ module imports cleanly (smoke)")
async def t_every_src_module_imports(_ctx: TestContext) -> None:
    """Walk ``src/`` and ``importlib.import_module`` each Python
    module. Catches truncated files / missing symbols / broken
    intra-package imports even in code paths no other test exercises.

    This is the *most general* version of the WorkflowExecutor
    regression guard: if scheduler.py does ``from src.workflow.executor
    import WorkflowExecutor`` and that name is gone, importing
    ``src.core.scheduler`` raises ImportError — and we catch it here
    without needing a behavioural test for scheduler itself.
    """
    import src as _src_pkg

    failures: list[tuple[str, str]] = []
    skipped: list[str] = []

    src_root = Path(_src_pkg.__file__).parent  # type: ignore[arg-type]
    # ``pkgutil.walk_packages`` follows __init__.py correctly and
    # gives dotted module names directly, no manual translation
    # from file paths to ``src.foo.bar`` strings.
    for finder, name, ispkg in pkgutil.walk_packages(
        path=_src_pkg.__path__,  # type: ignore[attr-defined]
        prefix="src.",
    ):
        if name in _SMOKE_IMPORT_SKIP:
            skipped.append(name)
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 — surface every flavour
            failures.append((name, f"{type(exc).__name__}: {exc}"))

    assert not failures, (
        "src/ modules failed to import — this catches truncated files, "
        "missing symbols, and broken intra-package imports even in code "
        "paths no behavioural test exercises:\n  "
        + "\n  ".join(f"{name}: {err}" for name, err in failures)
        + (f"\n(skipped: {sorted(skipped)})" if skipped else "")
    )
