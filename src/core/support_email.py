"""Keep the customer's request separate from a forwarded store receipt."""
from __future__ import annotations

import re

_FORWARD = re.compile(
    r"(?im)^\s*(?:-{2,}\s*(?:forwarded message|messaggio inoltrato|mensaje reenviado|"
    r"message transf[eé]r[eé])\s*-{2,}|begin forwarded message:|inizio messaggio inoltrato:)\s*$"
)
_RECEIPT = re.compile(r"\b(?:ricevuta|receipt|order number|numero ordine|GPA\.\d|invoice)\b", re.I)


def receipt_request(text: str) -> tuple[str, bool]:
    """Only an explicit forwarded boundary identifies quoted document text.

    Preserve the customer's prose before the boundary, including real refund
    requests. A receipt is evidence of what was forwarded, not payment proof.
    """
    marker = _FORWARD.search(text or "")
    if marker is None or not _RECEIPT.search(text[marker.end():]):
        return text, False
    return text[:marker.start()].strip(), True
