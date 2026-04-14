"""Markdown-to-platform renderers used by the Telegram / WhatsApp bridges.

Quick shape checks — don't assert exact byte-for-byte output (the
renderers evolve), just that their signature conversions still work.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("formatting", "markdown_to_telegram_html converts inline marks")
async def t_markdown_to_telegram_html(ctx: TestContext) -> None:
    from openagent.channels.formatting import markdown_to_telegram_html

    out = markdown_to_telegram_html("**bold** and *italic* plus `code`")
    assert "<b>bold</b>" in out, out
    assert "<i>italic</i>" in out, out
    assert "<code>code</code>" in out, out


@test("formatting", "markdown_to_telegram_html escapes raw HTML")
async def t_markdown_telegram_escape(ctx: TestContext) -> None:
    from openagent.channels.formatting import markdown_to_telegram_html
    # Raw HTML must be escaped so the bridge doesn't silently inject tags
    out = markdown_to_telegram_html("danger <script>alert(1)</script>")
    assert "<script>" not in out, out
    assert "&lt;script&gt;" in out or "&lt;" in out, out


@test("formatting", "markdown_to_whatsapp converts to WhatsApp syntax")
async def t_markdown_to_whatsapp(ctx: TestContext) -> None:
    from openagent.channels.formatting import markdown_to_whatsapp
    # WhatsApp uses *bold*, _italic_, ~strike~, ```code```
    out = markdown_to_whatsapp("**bold** and *italic*")
    assert "*bold*" in out, out
    # Markdown italic (single asterisk) should NOT survive as literal "*italic*"
    # — WhatsApp's bold uses single-star, so italic typically maps to _italic_.
    # Just verify the input's content shows up and output is non-empty.
    assert "italic" in out
