"""Pure-unit tests for ``openagent.channels.base``.

No server, no model — these just exercise the string-level utilities
that every bridge (telegram/discord/whatsapp/ws) depends on for
attachment parsing, long-message splitting, and blocked-extension
filtering.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("channels", "is_blocked_attachment rejects executables")
async def t_is_blocked_attachment(ctx: TestContext) -> None:
    from openagent.channels.base import is_blocked_attachment

    for bad in ("malware.exe", "script.bat", "payload.vbs", "trick.scr",
                "weird.ps1", "macro.jse"):
        assert is_blocked_attachment(bad), f"{bad!r} should be blocked"
    for ok in ("report.pdf", "photo.png", "notes.md", "voice.mp3"):
        assert not is_blocked_attachment(ok), f"{ok!r} should NOT be blocked"
    # None and empty string → False (no filename = nothing to block)
    assert is_blocked_attachment(None) is False
    assert is_blocked_attachment("") is False


@test("channels", "build_attachment_context + prepend_context_block shape")
async def t_build_attachment_context(ctx: TestContext) -> None:
    from openagent.channels.base import (
        build_attachment_context, prepend_context_block,
    )
    ctx_block = build_attachment_context(
        ["- image: chart.png — local path: /tmp/chart.png"],
    )
    assert "The user attached files:" in ctx_block
    assert "chart.png" in ctx_block
    assert "Read tool" in ctx_block  # ATTACHMENT_READ_HINT

    merged = prepend_context_block("tell me about this image", ctx_block)
    assert merged.startswith("The user attached files:")
    # Blank-line separator between block and user text
    assert "\n\n" in merged
    assert merged.endswith("tell me about this image")

    # Empty user text: context is the whole string
    solo = prepend_context_block("", ctx_block)
    assert solo == ctx_block


@test("channels", "split_preserving_code_blocks keeps ``` balanced")
async def t_split_code_blocks(ctx: TestContext) -> None:
    from openagent.channels.base import split_preserving_code_blocks

    # Short text → returned verbatim (single chunk)
    out = split_preserving_code_blocks("short", max_len=100)
    assert out == ["short"]

    # Empty → empty list
    assert split_preserving_code_blocks("", max_len=100) == []

    # Whitespace only → empty list (nothing useful to send)
    assert split_preserving_code_blocks("   ", max_len=10) == []

    # Long text with a code block that crosses a chunk boundary: every
    # output chunk must have a balanced number of triple-backticks.
    body = (
        "Here is some output:\n\n"
        "```python\n"
        + "\n".join(f"line {i}" for i in range(50))
        + "\n```\n\n"
        "And some prose after. " * 10
    )
    chunks = split_preserving_code_blocks(body, max_len=200)
    assert len(chunks) >= 2, f"expected split, got {len(chunks)} chunks"
    for i, ch in enumerate(chunks):
        count = ch.count("```")
        assert count % 2 == 0, f"chunk {i} has unbalanced ``` ({count}): {ch!r}"
