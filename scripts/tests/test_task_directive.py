"""Unit tests for the scheduled-task deterministic execution block."""
from __future__ import annotations

from typing import Any

from ._framework import TestContext, test


class _Toolkit:
    def __init__(self, functions: dict[str, Any]) -> None:
        from types import SimpleNamespace

        self.functions = {
            name: SimpleNamespace(entrypoint=fn, parameters={
                "type": "object", "properties": {},
            })
            for name, fn in functions.items()
        }
        self.async_functions: dict[str, Any] = {}


class _Pool:
    def __init__(self, toolkits: dict[str, _Toolkit]) -> None:
        self._toolkit_by_name = toolkits

    def toolkit_by_name(self, name: str) -> Any:
        return self._toolkit_by_name.get(name)


@test("task_directive", "a block is parsed, and a prompt without one stays untouched")
async def t_parse(_ctx: TestContext) -> None:
    from src.core import task_directive

    assert task_directive.parse("just a normal prompt") == []
    directives = task_directive.parse(
        "Send the approved reply.\n\n"
        "[[execute]]\n"
        "server: replio\n"
        "tool: threads_respond\n"
        'args: {"thread_id": "t-1", "body_text": "Hola"}\n'
        "[[/execute]]\n"
    )
    assert len(directives) == 1, directives
    assert directives[0].server == "replio"
    assert directives[0].tool == "threads_respond"
    assert directives[0].args == {"thread_id": "t-1", "body_text": "Hola"}
    assert "[[execute]]" not in task_directive.strip(
        "before [[execute]]\nserver: a\ntool: b\n[[/execute]] after"
    )


@test("task_directive", "a malformed block is refused, never silently skipped")
async def t_parse_errors(_ctx: TestContext) -> None:
    from src.core.task_directive import DirectiveError, parse

    for body in (
        "[[execute]]\ntool: threads_respond\n[[/execute]]",          # no server
        "[[execute]]\nserver: replio\n[[/execute]]",                 # no tool
        "[[execute]]\nserver: r\ntool: t\nargs: {oops\n[[/execute]]",  # bad JSON
        '[[execute]]\nserver: r\ntool: t\nargs: ["a"]\n[[/execute]]',  # not an object
    ):
        try:
            parse(body)
        except DirectiveError:
            continue
        raise AssertionError(f"malformed block was accepted: {body!r}")


@test("task_directive", "execution stops at the first failure and proves bytes by digest")
async def t_execute(_ctx: TestContext) -> None:
    import hashlib

    from src.core.task_directive import Directive, execute

    seen: list[dict[str, Any]] = []

    async def threads_respond(thread_id: str, body_text: str) -> dict[str, Any]:
        seen.append({"thread_id": thread_id, "body_text": body_text})
        return {"ok": True, "success": True, "simulated": True}

    async def threads_patch(thread_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        seen.append({"patch": patch})
        return {"ok": False, "status": 500}

    async def never_called(thread_id: str) -> dict[str, Any]:
        raise AssertionError("a directive after a failed one must not run")

    pool = _Pool({"replio": _Toolkit({
        "replio_threads_respond": threads_respond,
        "replio_threads_patch": threads_patch,
        "replio_threads_tags_add": never_called,
    })})

    text = "Hola, tu Premium ya esta activo."
    ok, receipts = await execute(pool, [
        Directive("replio", "threads_respond", {"thread_id": "t", "body_text": text}),
    ])
    assert ok is True, receipts
    assert seen[0]["body_text"] == text
    # The payload is never echoed back; the digest is what proves fidelity.
    assert text not in str(receipts), receipts
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    assert receipts[0]["arg_digests"]["body_text"] == digest

    # A failure fails the task and never falls through to the next block.
    ok, receipts = await execute(pool, [
        Directive("replio", "threads_patch", {"thread_id": "t", "patch": {"x": 1}}),
        Directive("replio", "threads_tags_add", {"thread_id": "t"}),
    ])
    assert ok is False, receipts
    assert len(receipts) == 1, receipts


@test("task_directive", "an ok:false or bare 4xx receipt counts as a failure")
async def t_failure_markers(_ctx: TestContext) -> None:
    from src.core.reply_guard import _trace_result_succeeded as ok

    # These four shapes all mean "the call did not do what it says". Only
    # success:false was recognised before, so an ok:false receipt was read as
    # a completed action.
    assert ok('{"ok": false, "status": 500}') is False
    assert ok('{"success": false}') is False
    assert ok('{"status": 404}') is False
    assert ok('{"isError": true}') is False
    # And the shapes that must stay successful.
    assert ok('{"ok": true}') is True
    assert ok('{"ok": true, "status": 200, "patch": {"status": "open"}}') is True
    assert ok('{"ok": true, "captured_chars": 405}') is True
