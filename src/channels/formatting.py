"""Markdown → channel-native rich-text converters.

The LLM backends emit CommonMark-ish markdown (``**bold**``, ``*italic*``,
``` ```code``` ```, ``[link](url)``, ``# headers``). Each channel renders
this differently:

- **Discord**: native markdown — emits as-is, no conversion needed.
- **Telegram**: ignores ``**bold**`` unless ``parse_mode`` is set. We
  convert to Telegram HTML (``<b>``, ``<i>``, ``<code>``, ``<pre>``,
  ``<a>``, ``<s>``) because Telegram HTML is much more permissive than
  MarkdownV2 (MarkdownV2 requires escaping half the ASCII table).
- **WhatsApp**: uses its own syntax — ``*bold*`` (single asterisk),
  ``_italic_``, ``~strike~``, ``` ```code``` ```. We rewrite the doubled
  markers to the single-marker WhatsApp form.

Both converters preserve the original text if no markup is present, so
passing plain text through them is a no-op.
"""

from __future__ import annotations

import html
import re


# ── Telegram HTML ──────────────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(r"```([A-Za-z0-9_+\-]*)\n?(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_BOLD_STAR_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_BOLD_UND_RE = re.compile(r"__(.+?)__", re.DOTALL)
_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
# Single-asterisk italic after bold has been stripped (bold regex runs first).
_ITALIC_STAR_RE = re.compile(r"(?<!\w)\*([^\*\n]+?)\*(?!\w)")
_ITALIC_UND_RE = re.compile(r"(?<!\w)_([^_\n]+?)_(?!\w)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_BULLET_RE = re.compile(r"^(\s*)[-*]\s+", re.MULTILINE)

# Placeholder sentinel — must be something that won't appear in real text
# and won't get mangled by html.escape.
_PLACEHOLDER_PREFIX = "\x00OA_CODE_"
_PLACEHOLDER_RE = re.compile(r"\x00OA_CODE_(\d+)\x00")


def markdown_to_telegram_html(text: str) -> str:
    """Convert markdown to Telegram-HTML with ``parse_mode="HTML"``.

    Handles bold (``**`` / ``__``), italic (``*`` / ``_``), strikethrough
    (``~~``), inline code, fenced code blocks with language hints, links
    (``[t](u)``), ATX headers, and unordered lists. Everything else is
    HTML-escaped. The result is safe to pass directly to Telegram with
    ``parse_mode="HTML"``.
    """
    if not text:
        return text

    # 1. Extract code blocks and inline code — replace with placeholders so
    #    their inner content isn't touched by the HTML escaper or the
    #    bold/italic regexes.
    placeholders: list[str] = []

    def _protect(match: re.Match) -> str:
        placeholders.append(match.group(0))
        return f"{_PLACEHOLDER_PREFIX}{len(placeholders) - 1}\x00"

    protected = _CODE_BLOCK_RE.sub(_protect, text)
    protected = _INLINE_CODE_RE.sub(_protect, protected)

    # 2. HTML-escape the remaining text. Placeholders are ASCII-safe.
    protected = html.escape(protected, quote=False)

    # 3. Convert markdown tokens to HTML tags. Order matters: bold BEFORE
    #    italic, otherwise ``**x**`` would match ``*x*`` twice.
    protected = _BOLD_STAR_RE.sub(r"<b>\1</b>", protected)
    protected = _BOLD_UND_RE.sub(r"<b>\1</b>", protected)
    protected = _STRIKE_RE.sub(r"<s>\1</s>", protected)
    protected = _ITALIC_STAR_RE.sub(r"<i>\1</i>", protected)
    protected = _ITALIC_UND_RE.sub(r"<i>\1</i>", protected)

    def _link_sub(match: re.Match) -> str:
        label, url = match.group(1), match.group(2)
        return f'<a href="{html.escape(url, quote=True)}">{label}</a>'

    protected = _LINK_RE.sub(_link_sub, protected)
    # ATX headers → bold. Telegram has no <h1>.
    protected = _HEADER_RE.sub(r"<b>\2</b>", protected)
    # Bullet lists → Unicode bullet (Telegram HTML has no list tags).
    protected = _BULLET_RE.sub(r"\1• ", protected)

    # 4. Restore code placeholders. Their contents ARE escaped here.
    def _restore(match: re.Match) -> str:
        idx = int(match.group(1))
        orig = placeholders[idx]
        cb = _CODE_BLOCK_RE.fullmatch(orig)
        if cb:
            lang = cb.group(1)
            code = cb.group(2).rstrip("\n")
            code_esc = html.escape(code, quote=False)
            if lang:
                return (
                    f'<pre><code class="language-{html.escape(lang, quote=True)}">'
                    f'{code_esc}</code></pre>'
                )
            return f"<pre>{code_esc}</pre>"
        ic = _INLINE_CODE_RE.fullmatch(orig)
        if ic:
            return f"<code>{html.escape(ic.group(1), quote=False)}</code>"
        # Should never happen, but fall back to an escaped literal.
        return html.escape(orig, quote=False)

    return _PLACEHOLDER_RE.sub(_restore, protected)


# ── WhatsApp ────────────────────────────────────────────────────────────

_WA_BOLD_STAR_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_WA_BOLD_UND_RE = re.compile(r"__(.+?)__", re.DOTALL)
_WA_STRIKE_RE = re.compile(r"~~(.+?)~~", re.DOTALL)
_WA_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_WA_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def markdown_to_whatsapp(text: str) -> str:
    """Convert markdown to WhatsApp's native rich-text syntax.

    WhatsApp syntax:

    - ``*bold*``   — single asterisk (not double)
    - ``_italic_`` — single underscore
    - ``~strike~`` — single tilde
    - ``` `code` ``` and triple-backtick blocks — identical to markdown
    - links and headers are not rendered; we keep the link text + bare URL
      and promote headers to ``*bold*``

    Code blocks are left untouched so their inner text isn't rewritten.
    """
    if not text:
        return text

    placeholders: list[str] = []

    def _protect(match: re.Match) -> str:
        placeholders.append(match.group(0))
        return f"{_PLACEHOLDER_PREFIX}{len(placeholders) - 1}\x00"

    protected = _CODE_BLOCK_RE.sub(_protect, text)
    protected = _INLINE_CODE_RE.sub(_protect, protected)

    protected = _WA_BOLD_STAR_RE.sub(r"*\1*", protected)
    protected = _WA_BOLD_UND_RE.sub(r"*\1*", protected)
    protected = _WA_STRIKE_RE.sub(r"~\1~", protected)
    protected = _WA_LINK_RE.sub(r"\1 (\2)", protected)
    protected = _WA_HEADER_RE.sub(r"*\2*", protected)

    def _restore(match: re.Match) -> str:
        idx = int(match.group(1))
        return placeholders[idx]

    return _PLACEHOLDER_RE.sub(_restore, protected)
