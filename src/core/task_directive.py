"""Deterministic execution block for scheduled tasks.

Some scheduled tasks carry no judgement at all. The three ``reply-*`` tasks in
the eSound agent say it outright: send THIS EXACT TEXT, do not rephrase, do not
add a signature. A human already approved the wording; routing it through a
model can only subtract.

Measured on Qwen3-30B against the real scheduler, two firings out of three
spent the whole tool budget reading policy notes nobody asked for and never
reached the send — while the task still reported ``success`` and the log still
showed the approved text. That failure is invisible: nothing distinguishes "the
reply went out" from "the model talked about the reply".

So an operator can make the intent machine-readable:

    [[execute]]
    server: replio
    tool: threads_respond
    args: {"thread_id": "abc", "body_text": "Hola, ..."}
    [[/execute]]

Blocks run in order, before any model call, and the model is skipped entirely
when they all succeed. A failing block fails the task: it never falls through
to "let the model try instead", because a half-executed approved action is
worse than one that visibly did not run.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


_BLOCK = re.compile(
    r"\[\[execute\]\](?P<body>.*?)\[\[/execute\]\]",
    re.IGNORECASE | re.DOTALL,
)
_FIELD = re.compile(r"^\s*(server|tool|args)\s*:\s*(.*)$", re.IGNORECASE)


@dataclass(frozen=True)
class Directive:
    server: str
    tool: str
    args: dict[str, Any]


class DirectiveError(ValueError):
    """A block exists but cannot be executed exactly as written."""


def parse(prompt: str) -> list[Directive]:
    """Extract every execution block, in order. ``[]`` when there are none."""
    out: list[Directive] = []
    for match in _BLOCK.finditer(prompt or ""):
        fields: dict[str, str] = {}
        key = ""
        for line in match.group("body").splitlines():
            found = _FIELD.match(line)
            if found:
                key = found.group(1).lower()
                fields[key] = found.group(2).strip()
            elif key == "args" and line.strip():
                # A JSON payload may legitimately span lines.
                fields["args"] = fields.get("args", "") + line.strip()
        server = fields.get("server", "").strip()
        tool = fields.get("tool", "").strip()
        raw_args = fields.get("args", "").strip() or "{}"
        if not server or not tool:
            raise DirectiveError("execute block needs both 'server' and 'tool'")
        try:
            args = json.loads(raw_args)
        except (TypeError, ValueError) as exc:
            raise DirectiveError(f"execute block has invalid args JSON: {exc}") from exc
        if not isinstance(args, dict):
            raise DirectiveError("execute block 'args' must be a JSON object")
        out.append(Directive(server=server, tool=tool, args=args))
    return out


def strip(prompt: str) -> str:
    """The prompt without its execution blocks."""
    return _BLOCK.sub("", prompt or "").strip()


async def execute(pool: Any, directives: list[Directive]) -> tuple[bool, list[dict]]:
    """Run every directive in order. Stops at the first failure.

    Returns ``(ok, receipts)``. ``ok`` is False when any call raised or came
    back as an error envelope, so the caller can fail the task loudly instead
    of reporting a success nobody performed.
    """
    from src.core import reply_guard
    from src.mcp.servers.tool_search.adapters import _call_tool_impl

    receipts: list[dict] = []
    for directive in directives:
        entry: dict[str, Any] = {
            "server": directive.server,
            "tool": directive.tool,
            # Never echo the payload back into a stored receipt: it can carry
            # a customer's name or address. A digest still proves, after the
            # fact, that exactly the approved bytes were sent - which a
            # substring check on a redacted log never could.
            "args_keys": sorted(directive.args),
            "arg_digests": {
                key: hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
                for key, value in directive.args.items()
                if isinstance(value, str)
            },
        }
        try:
            result = await _call_tool_impl(
                pool, directive.server, directive.tool, dict(directive.args),
            )
        except Exception as exc:  # noqa: BLE001 - a failed directive fails the task
            entry.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]})
            receipts.append(entry)
            return False, receipts
        rendered = result if isinstance(result, str) else json.dumps(result, default=str)
        ok = reply_guard._trace_result_succeeded(rendered)
        entry.update({"ok": ok, "result": rendered[:600]})
        receipts.append(entry)
        if not ok:
            return False, receipts
    return True, receipts
