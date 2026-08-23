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

import hashlib
import os
import time
from pathlib import Path
from typing import Any

# ~12k tokens for a SINGLE tool result. Generous for a real answer, and far
# below what one runaway note or a whole re-quoted email thread (measured at
# ~1.7 MB) would otherwise inject into every following step.
DEFAULT_MAX_TOOL_RESULT_CHARS = 50_000


def max_tool_result_chars() -> int:
    """The cap, read at call time so it can be tuned without a restart."""
    from src.core.execution_profile import lean_local_event_active

    if lean_local_event_active():
        try:
            lean_cap = int(os.environ.get("OPENAGENT_LEAN_EVENT_TOOL_RESULT_CHARS", "2500"))
        except (TypeError, ValueError):
            lean_cap = 2500
        return max(500, lean_cap)
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


# ── Lossless offload (opt-in) ─────────────────────────────────────────
#
# Truncation is lossy: the dropped bytes are gone, and a support agent that was
# told to read a 200 KB KB article or a long re-quoted email thread cannot
# recover them. Offload trades that for a spill-to-disk — the FULL result is
# written to a file and the in-context value becomes a compact preview plus the
# path, which the agent re-reads with its ``read_file`` / editor tool on demand.
#
# It is OPT-IN (default OFF) and strictly additive: with offload disabled,
# ``cap_tool_output`` truncates byte-identically to before. Policy is read from
# the environment at call time (parity with ``max_tool_result_chars``);
# ``src/core/server.py`` exports it from the ``tool_output:`` config stanza so
# this in-process reader sees it without any config plumbing.

# Chars of the original kept inline as a preview ahead of the read handle.
OFFLOAD_PREVIEW_CHARS = 1_500

_TRUTHY = {"1", "true", "yes", "on"}


def _offload_enabled() -> bool:
    return (
        os.environ.get("OPENAGENT_TOOL_OFFLOAD_ENABLED", "0").strip().lower()
        in _TRUTHY
    )


def _offload_threshold() -> int:
    """Chars above which a result is offloaded. Defaults to the truncation cap."""
    raw = os.environ.get("OPENAGENT_TOOL_OFFLOAD_THRESHOLD", "").strip()
    if raw:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    return max_tool_result_chars()


def _offload_keep() -> int:
    """Retention cap — how many offload files to keep. Default 200."""
    raw = os.environ.get("OPENAGENT_TOOL_OFFLOAD_KEEP", "").strip()
    if raw:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    return 200


def _offload_dir() -> Path:
    """The directory offloaded results are written to.

    Defaults to ``paths.data_dir()/tool_outputs`` — inside the data dir that
    the filesystem/editor MCP root already covers by default, so the handle the
    preview hands back is re-readable without widening any root.
    """
    raw = os.environ.get("OPENAGENT_TOOL_OFFLOAD_DIR", "").strip()
    if raw:
        d = Path(raw).expanduser()
    else:
        from src.core.paths import data_dir

        d = data_dir() / "tool_outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _prune_offload_dir(directory: Path, keep: int) -> None:
    """Keep only the ``keep`` newest ``*.txt`` files so the dir stays bounded.

    A tiny prune-on-write — no daemon. ``keep <= 0`` disables pruning.
    """
    if keep <= 0:
        return
    files = sorted(
        directory.glob("*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in files[keep:]:
        stale.unlink(missing_ok=True)


def _offload_text(text: str) -> str:
    """Write the FULL ``text`` to the offload dir; return preview + read handle.

    Lossless: the file on disk equals ``text`` byte-for-byte (surrogatepass on
    both the hash and the write, so nothing is mangled). The returned in-context
    value is a short preview followed by one line naming the path and how to
    read it back.
    """
    directory = _offload_dir()
    digest = hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()[:12]
    name = f"{time.strftime('%Y%m%dT%H%M%S')}-{digest}.txt"
    path = directory / name
    path.write_text(text, encoding="utf-8", errors="surrogatepass")
    _prune_offload_dir(directory, _offload_keep())

    preview = text[:OFFLOAD_PREVIEW_CHARS]
    dropped = len(text) - len(preview)
    marker = (
        f"\n\n[... {dropped} characters truncated by OpenAgent; FULL output "
        f"({len(text)} chars) saved to {path} — read it with the read_file/editor "
        f"tool if you need the rest ...]\n"
    )
    return preview + marker


def _cap_or_offload(text: str, limit: int) -> str:
    """Offload branch of the cap: spill over-threshold results, else truncate.

    Only reached when offload is ENABLED. A result at or below the offload
    threshold is handed to the normal cap (returned inline if it also fits under
    ``limit``, truncated otherwise), so small results stay untouched.
    """
    if len(text) <= _offload_threshold():
        return _cap_text(text, limit)
    return _offload_text(text)


def cap_tool_output(output: Any) -> Any:
    """Cap an oversized tool result before it becomes part of the context.

    Strings are truncated. A list (a provider's content-block form) has its
    string members truncated in place, and its dict members' ``text`` field —
    so an image block or other structured content is never mangled. Anything
    else passes through untouched: guessing at the shape of a value we do not
    understand is how you corrupt a tool call.

    When ``tool_output.offload_enabled`` is set (default OFF), an over-threshold
    result is spilled LOSSLESSLY to disk and replaced with a preview + path
    instead of being truncated. With offload disabled this is byte-identical to
    the historical truncation.
    """
    limit = max_tool_result_chars()
    if limit <= 0:
        return output

    # OPT-IN: with offload disabled (default), ``transform`` is the historical
    # truncation, so every byte this function returns is identical to before.
    transform = _cap_or_offload if _offload_enabled() else _cap_text

    if isinstance(output, str):
        return transform(output, limit)

    if isinstance(output, list):
        capped: list[Any] = []
        for item in output:
            if isinstance(item, str):
                capped.append(transform(item, limit))
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                capped.append({**item, "text": transform(item["text"], limit)})
            else:
                capped.append(item)
        return capped

    return output
