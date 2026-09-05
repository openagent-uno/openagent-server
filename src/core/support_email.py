"""Keep the customer's request separate from a forwarded store receipt."""
from __future__ import annotations

import re
import unicodedata

_FORWARD = re.compile(
    r"(?im)^\s*(?:-{2,}\s*(?:forwarded message|messaggio inoltrato|mensaje reenviado|"
    r"message transf[eé]r[eé])\s*-{2,}|begin forwarded message:|inizio messaggio inoltrato:)\s*$"
)
_RECEIPT = re.compile(r"\b(?:ricevuta|receipt|order number|numero ordine|GPA\.\d|invoice)\b", re.I)


def without_support_echo(text: str, prior_support: list[str]) -> str:
    """Remove an unmarked trailing copy of an actual support message.

    Some mail clients quote with whitespace alone. Topic words in that copy
    must not reopen a resolved problem. Require a substantial, exact token
    match through the end: inline or bottom-posted customer answers survive.
    """
    tokens = list(re.finditer(r"\S+", text or ""))
    def fold(value: str) -> str:
        return unicodedata.normalize("NFC", value).replace("’", "'").casefold()
    words = [fold(token.group()) for token in tokens]
    boundary = len(text)
    for previous in prior_support:
        quoted = [fold(word) for word in previous.split()]
        if len(quoted) < 16 or len(quoted) > len(words):
            continue
        start = len(words) - len(quoted)
        if words[start:] != quoted:
            continue
        offset = tokens[start].start()
        # Only a separate quoted block, never a quote embedded in a sentence.
        if offset and text[:offset].rstrip(" \t").endswith("\n") is False:
            continue
        boundary = min(boundary, offset)
    return text[:boundary].strip()


def receipt_request(text: str) -> tuple[str, bool]:
    """Only an explicit forwarded boundary identifies quoted document text.

    Preserve the customer's prose before the boundary, including real refund
    requests. A receipt is evidence of what was forwarded, not payment proof.
    """
    marker = _FORWARD.search(text or "")
    if marker is None or not _RECEIPT.search(text[marker.end():]):
        return text, False
    return text[:marker.start()].strip(), True
