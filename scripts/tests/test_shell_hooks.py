"""``src.core.hooks`` — the shell-command hook registry + quick commands.

This module had **no test at all** until now, which is exactly how its
central defect survived: every ``elog`` call in it passed
``event=event_name`` while ``elog(event, level, ...)`` already takes the
event name as its first positional, so each one raised
``TypeError: elog() got multiple values for argument 'event'``.

The failure was invisible by construction — the logging call *was* the
thing raising, so the log could not report it. Worse, ``_run_hook``'s
``except`` handler repeated the mistake, so the TypeError from the
success-path ``hooks.fired`` re-raised inside the handler and escaped as
an unretrieved task exception. Hooks ran; the operator saw nothing.

``test_task_hooks.py`` did not and could not catch this: it covers the
scheduler's ``run_task`` hook registry — a different subsystem that
happens to share the word "hook".

These tests pin the *call sites*, not the module's internals. That
distinction is the whole point: this repo has shipped module-level tests
that passed for a feature's entire lifetime while the feature was inert
in production (see ``test_vault_reminder``, which asserted "on by
default" while its only call site defaulted it off).
"""
from __future__ import annotations

import asyncio
import inspect

from ._framework import TestContext, test


@test("shell_hooks", "elog rejects a kwarg named 'event' (the bug's shape)")
async def t_elog_event_kwarg_collides(ctx: TestContext) -> None:
    """Pin the signature that makes ``hook_event=`` mandatory.

    If someone renames ``elog``'s first parameter, ``hook_event`` stops
    being necessary and this test should be revisited deliberately,
    rather than the collision quietly becoming possible again.
    """
    from src.core.logging import elog

    params = list(inspect.signature(elog).parameters)
    assert params[0] == "event", (
        f"elog's first positional is no longer 'event' but {params[0]!r} — "
        "the reason src/core/hooks.py must say hook_event= has changed."
    )

    try:
        elog("some.event", event="collides")
    except TypeError as e:
        assert "event" in str(e)
    else:
        raise AssertionError(
            "elog(positional, event=...) no longer raises — the collision "
            "this guards is gone; revisit src/core/hooks.py's comment."
        )


@test("shell_hooks", "no elog call in hooks.py passes a bare 'event' kwarg")
async def t_no_event_kwarg_in_hooks(ctx: TestContext) -> None:
    """Static guard over every call site at once.

    The runtime tests below only reach the paths they exercise; this one
    catches a fourth (or fifth) call site added later. Found exactly that
    way: ``fire()``'s ``hooks.skipped`` call had the same defect as the
    three in ``_run_hook``.
    """
    import ast
    import pathlib

    src = pathlib.Path("src/core/hooks.py").read_text()
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Name) and fn.id == "elog"):
            continue
        for kw in node.keywords:
            if kw.arg == "event":
                offenders.append(node.lineno)

    assert not offenders, (
        f"src/core/hooks.py:{offenders} call elog(..., event=...), which "
        "collides with elog's first positional and raises TypeError. Use "
        "hook_event= instead."
    )


@test("shell_hooks", "a fired hook logs hooks.fired with the hook's identity")
async def t_hook_success_path_logs(ctx: TestContext) -> None:
    from src.core import hooks as hooks_mod
    from src.core.logging import elog as _real_elog

    seen: list[tuple[str, dict]] = []

    def _spy(event: str, level: str = "info", exc_info: bool = False, **data):
        # Bind exactly like the real elog, so a collision still raises here.
        seen.append((event, data))

    hooks_mod.elog = _spy
    try:
        # set_hooks takes str values (not lists) and strips a leading
        # ``on_``, so this registers under ``turn_end``.
        hooks_mod.set_hooks({"on_turn_end": "true"})
        assert "turn_end" in hooks_mod._HOOKS, "hook did not register"

        hooks_mod.fire("turn_end", session_id="s1")
        for _ in range(300):
            await asyncio.sleep(0.01)
            if any(e == "hooks.fired" for e, _ in seen):
                break
    finally:
        hooks_mod.elog = _real_elog
        hooks_mod.set_hooks({})

    events = [e for e, _ in seen]
    assert "hooks.fired" in events, (
        f"hooks.fired never logged — got {events!r}. Before the hook_event "
        "fix this raised TypeError, was swallowed by the except handler, "
        "and the handler then raised the same TypeError again."
    )
    fired = next(d for e, d in seen if e == "hooks.fired")
    assert fired.get("hook_event") == "on_turn_end" or fired.get("hook_event") == "turn_end", (
        f"hooks.fired lost the hook's identity: {fired!r} — a log line that "
        "cannot say WHICH hook fired is not worth writing."
    )
    assert "event" not in fired, (
        "payload carries a bare 'event' key again — that is the collision."
    )
    assert fired.get("exit_code") == 0


@test("shell_hooks", "a hook that cannot spawn logs hooks.error, never escapes")
async def t_hook_error_path_logs(ctx: TestContext) -> None:
    """The except handler carried the same bug, so the error path was
    doubly broken: it could neither report a failure nor survive
    reporting it.

    Note ``_run_hook`` uses ``create_subprocess_shell``, so a missing
    binary is exit 127 on the SUCCESS path, not an exception. To drive
    the handler we make the spawn itself fail.
    """
    from src.core import hooks as hooks_mod
    from src.core.logging import elog as _real_elog

    seen: list[tuple[str, dict]] = []
    hooks_mod.elog = lambda event, level="info", exc_info=False, **d: seen.append((event, d))

    async def _boom(*a, **k):
        raise OSError("cannot spawn")

    real_spawn = asyncio.create_subprocess_shell
    asyncio.create_subprocess_shell = _boom
    try:
        hooks_mod.set_hooks({"on_turn_end": "true"})
        hooks_mod.fire("turn_end")
        for _ in range(300):
            await asyncio.sleep(0.01)
            if seen:
                break
    finally:
        asyncio.create_subprocess_shell = real_spawn
        hooks_mod.elog = _real_elog
        hooks_mod.set_hooks({})

    assert seen, "the error path logged nothing at all"
    event, data = seen[0]
    assert event == "hooks.error", f"expected hooks.error, got {event!r}"
    assert "hook_event" in data, f"hooks.error lost the hook's identity: {data!r}"
    assert "cannot spawn" in data.get("error", "")


@test("shell_hooks", "quick commands expand; normal text is left alone")
async def t_quick_commands(ctx: TestContext) -> None:
    from src.core import hooks as hooks_mod

    try:
        # Keys are stored slash-stripped and lowercased, so the yaml may
        # say ``recap`` or ``/recap`` and both match ``/RECAP``.
        hooks_mod.set_quick_commands({"/recap": "Summarise the last 20 turns."})
        assert hooks_mod.expand_quick_command("/recap") == "Summarise the last 20 turns."
        assert hooks_mod.expand_quick_command("/RECAP") == "Summarise the last 20 turns."

        # Trailing text is preserved and appended, not dropped.
        assert hooks_mod.expand_quick_command("/recap last week") == (
            "Summarise the last 20 turns.\nlast week"
        )

        # Non-triggers return None (the "not a quick command" signal) —
        # a registry that rewrote ordinary messages would be worse than none.
        assert hooks_mod.expand_quick_command("hello") is None
        assert hooks_mod.expand_quick_command("/unknown") is None
        assert hooks_mod.expand_quick_command("") is None
    finally:
        hooks_mod.set_quick_commands({})

    # Empty registry -> no-op, the documented default.
    assert hooks_mod.expand_quick_command("/recap") is None
