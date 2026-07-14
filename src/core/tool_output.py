"""The hard ceiling on a single tool result, applied where it actually bites.

A tool result is not a transient thing. It becomes a `role="tool"` message in
the conversation, and every subsequent model call in the turn — and, in a bound
session, every subsequent turn — re-sends it. So one unbounded result is not
paid once: it is paid on every step that follows it, forever.

On 2026-07-13 an eSound vault note had grown to 642 KB (~160k tokens) of
appended run-logs, and the support prompt mandated reading it before acting.
It entered the context uncapped, was replayed on every agentic step (3.3 on
average, 13 at worst), and the webhook lane alone burned ~412M input tokens in
19 hours across two agents.

A cap existed. It just never reached the model: it was applied to
``ToolExecution.result`` — the record the UI renders — on a code path the
normal (non-HITL) tool loop does not take. The message handed back to the
provider was built separately, from the raw output, in
``Model.create_function_call_result``. That is the one place every provider
funnels through, so that is where the ceiling lives now.

Truncation keeps the head and a small tail with a loud marker between them: the
run survives, and the model can see it was cut and narrow its next query,
instead of the whole call dying on a non-retryable context-length error.
"""
from __future__ import annotations

import os
from typing import Any

# ~12k tokens for a SINGLE tool result. Generous for a real answer, and far
# below what one runaway note or a whole re-quoted email thread (measured at
# ~1.7 MB) would otherwise inject into every following step.
DEFAULT_MAX_TOOL_RESULT_CHARS = 50_000


def max_tool_result_chars() -> int:
    """The cap, read at call time so it can be tuned without a restart."""
    try:
        return int(
            os.environ.get(
                "OPENAGENT_MAX_TOOL_RESULT_CHARS", DEFAULT_MAX_TOOL_RESULT_CHARS
            )
        )
    except (TypeError, ValueError):
        return DEFAULT_MAX_TOOL_RESULT_CHARS


def _cap_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.85)
    tail = max(0, limit - head)
    dropped = len(text) - head - tail
    marker = (
        f"\n\n[... {dropped} characters truncated by OpenAgent: this single tool "
        f"result exceeded {limit} chars and would be re-sent to the model on every "
        f"following step. Narrow the query, request fewer items, or fetch specifics. ...]\n\n"
    )
    return text[:head] + marker + (text[-tail:] if tail else "")


def cap_tool_output(output: Any) -> Any:
    """Cap an oversized tool result before it becomes part of the context.

    Strings are truncated. A list (a provider's content-block form) has its
    string members truncated in place, and its dict members' ``text`` field —
    so an image block or other structured content is never mangled. Anything
    else passes through untouched: guessing at the shape of a value we do not
    understand is how you corrupt a tool call.
    """
    limit = max_tool_result_chars()
    if limit <= 0:
        return output

    if isinstance(output, str):
        return _cap_text(output, limit)

    if isinstance(output, list):
        capped: list[Any] = []
        for item in output:
            if isinstance(item, str):
                capped.append(_cap_text(item, limit))
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                capped.append({**item, "text": _cap_text(item["text"], limit)})
            else:
                capped.append(item)
        return capped

    return output
