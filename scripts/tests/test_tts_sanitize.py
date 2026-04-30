"""``sanitize_for_tts`` — strip markdown / emojis / URLs before synthesis.

Local Piper (and most cloud TTS engines) literally pronounce every
character. Without sanitization, an LLM reply with markdown formatting
becomes "asterisk asterisk hello asterisk asterisk", emojis become
"smiling face with smiling eyes", code fences become "backtick backtick
backtick", and bare URLs become a tedious letter-by-letter spelling.

These tests pin the sanitizer's behaviour:

  * idempotent — sanitising already-clean text is a no-op
  * markdown markers stripped, inner text preserved
  * emojis dropped (multiple unicode ranges + ZWJ sequences)
  * code fences and inline backticks dropped (chunker handles fences in
    the streaming path; this is the bridges' synthesize_full backstop)
  * bare URLs dropped, link text preserved
  * lists, headers, blockquotes, table pipes flattened
  * pronouncable punctuation (``. , ! ? ; :``) preserved for prosody
"""
from __future__ import annotations

from ._framework import TestContext, test


def _sanitize(text: str) -> str:
    from openagent.channels.tts import sanitize_for_tts
    return sanitize_for_tts(text)


@test("tts_sanitize", "no-op on plain prose")
async def t_no_op_on_plain_prose(_ctx: TestContext) -> None:
    plain = "The quick brown fox jumps over the lazy dog."
    assert _sanitize(plain) == plain
    # Idempotent: running it twice is the same as once.
    assert _sanitize(_sanitize(plain)) == plain


@test("tts_sanitize", "strips bold / italic / strikethrough markers")
async def t_strips_inline_emphasis(_ctx: TestContext) -> None:
    assert _sanitize("Hello **world**!") == "Hello world!"
    assert _sanitize("This is __really__ important.") == "This is really important."
    assert _sanitize("An *italic* phrase.") == "An italic phrase."
    assert _sanitize("And _another italic_ phrase.") == "And another italic phrase."
    assert _sanitize("It was ~~deleted~~ then.") == "It was deleted then."


@test("tts_sanitize", "strips inline code backticks but keeps inner text")
async def t_strips_inline_code(_ctx: TestContext) -> None:
    out = _sanitize("Run `npm test` to verify.")
    assert "`" not in out
    assert "npm test" in out


@test("tts_sanitize", "strips fenced code blocks entirely")
async def t_strips_code_fences(_ctx: TestContext) -> None:
    src = "Look at this:\n```python\nprint('hi')\n```\nDone."
    out = _sanitize(src)
    assert "```" not in out
    assert "print" not in out, f"code body leaked: {out!r}"
    assert "Look at this:" in out and "Done." in out


@test("tts_sanitize", "strips emojis across multiple unicode ranges")
async def t_strips_emojis(_ctx: TestContext) -> None:
    # Smiley, heart, rocket, thumbs-up, sparkles
    out = _sanitize("Great work 😀❤️🚀👍✨ keep it up.")
    for emoji in ("😀", "❤", "🚀", "👍", "✨"):
        assert emoji not in out, f"emoji leaked: {out!r}"
    assert "Great work" in out
    assert "keep it up." in out


@test("tts_sanitize", "drops bare URLs but keeps surrounding text")
async def t_drops_bare_urls(_ctx: TestContext) -> None:
    out = _sanitize("See https://example.com/path?x=1 for details.")
    assert "https://" not in out
    assert "example.com" not in out
    assert "See" in out and "for details." in out


@test("tts_sanitize", "rewrites markdown links to plain anchor text")
async def t_link_text_preserved(_ctx: TestContext) -> None:
    out = _sanitize("Check the [docs](https://docs.example.com) for syntax.")
    assert "https://" not in out
    assert "docs.example.com" not in out
    assert "Check the docs for syntax." == out


@test("tts_sanitize", "image alt text preserved, URL dropped")
async def t_image_alt_preserved(_ctx: TestContext) -> None:
    out = _sanitize("Here ![a chart of sales](https://img.example.com/x.png) data.")
    assert "img.example.com" not in out
    # Alt text reads naturally as the inline content.
    assert "a chart of sales" in out


@test("tts_sanitize", "strips list bullets, headers, blockquotes")
async def t_strips_block_markers(_ctx: TestContext) -> None:
    src = (
        "# Heading\n"
        "## Subheading\n"
        "- first item\n"
        "- second item\n"
        "1. numbered one\n"
        "2. numbered two\n"
        "> quoted text\n"
    )
    out = _sanitize(src)
    assert "#" not in out
    assert ">" not in out
    # Bullet markers gone, item text preserved
    for marker in ("- first", "- second", "1. numbered", "2. numbered"):
        assert marker not in out, f"bullet marker leaked: {out!r}"
    assert "first item" in out
    assert "second item" in out
    assert "numbered one" in out
    assert "quoted text" in out
    assert "Heading" in out and "Subheading" in out


@test("tts_sanitize", "preserves prosody punctuation")
async def t_preserves_prosody_punct(_ctx: TestContext) -> None:
    """Periods, commas, semicolons, colons, exclamation marks and
    question marks carry intonation. Sanitizer must NOT strip them."""
    src = "Wait, really? Yes! Here's the deal: it works; trust me."
    out = _sanitize(src)
    for ch in (",", "?", "!", ":", ";", "."):
        assert ch in out, f"prosody char {ch!r} stripped: {out!r}"


@test("tts_sanitize", "empty / whitespace input → empty string")
async def t_empty_input(_ctx: TestContext) -> None:
    assert _sanitize("") == ""
    assert _sanitize("   ") == ""
    assert _sanitize("\n\n\t") == ""


@test("tts_sanitize", "html tags dropped, inner text preserved")
async def t_strips_html_tags(_ctx: TestContext) -> None:
    out = _sanitize("Hello <strong>world</strong>!")
    assert "<" not in out and ">" not in out
    assert "world" in out


@test("tts_sanitize", "table pipes collapsed to spaces")
async def t_table_pipes_collapsed(_ctx: TestContext) -> None:
    src = "| Header A | Header B |\n|---|---|\n| cell 1 | cell 2 |"
    out = _sanitize(src)
    assert "|" not in out
    assert "Header A" in out and "cell 2" in out
