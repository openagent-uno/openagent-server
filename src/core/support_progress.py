"""Reconstruct a case's pending work from its durable conversation.

Customer-supplied identifiers are lookup hints, never identity or payment proof.
Keeping this derived from the transcript avoids a second, stale case database.
"""
from __future__ import annotations

import re
from typing import Any

_REFUND = re.compile(r"\b(?:refund\w*|money back|reimburs\w*|rimbor\w*|reembols\w*|rembours\w*|rückerstatt\w*|erstattung|estorno)\b", re.I)
_CONDITIONAL = re.compile(r"\b(?:otherwise|or else|if|unless|se no|altrimenti|si no|sinon|caso contr[aá]rio)\b", re.I)
_ORDER = re.compile(r"\bGPA\.\d{4}-\d{4}-\d{4}-\d{5}(?:\.\.\d+)?\b", re.I)
_PAYMENT = re.compile(r"\b(?:payment|purchase|pagamento|pago|cobro|paiement|zahlung|compra)\b", re.I)
_PENDING = re.compile(r"\b(?:pending|processing|in progress|in elaborazione|pendente|pendiente|en cours|ausstehend)\b", re.I)
_BOT_QUESTION = re.compile(r"\b(?:are you (?:a |an )?(?:robot|bot|human|person)|sei (?:un |una )?(?:bot|robot|persona)|eres (?:un |una )?(?:bot|robot|persona)|voc[eê] [ée] (?:um |uma )?(?:bot|rob[oô]|pessoa))\b", re.I)


def explicit_refund(text: str) -> bool:
    """A payment word or an exact citation is not a refund request."""
    return any(_REFUND.search(s) and not _CONDITIONAL.search(s)
               and not re.search(r"\b(?:no|not|don't|do not|non|não|nao)\s+(?:want|need|voglio|quero|quiero)?\s*(?:a |un |um )?refund", s, re.I)
               for s in re.split(r"[.!?\n]", text or ""))


def payment_pending(text: str) -> bool:
    return bool(_PAYMENT.search(text or "") and _PENDING.search(text or ""))


def bot_question(text: str) -> bool:
    return bool(_BOT_QUESTION.search(text or ""))


def case_frame(exchange: list[dict[str, Any]], latest: str) -> dict[str, Any]:
    customer = [str(t.get("text") or "") for t in exchange if t.get("from") == "customer"]
    support = [str(t.get("text") or "") for t in exchange if t.get("from") != "customer"]
    text = "\n".join([*customer, latest])
    orders = list(dict.fromkeys(_ORDER.findall(text)))
    last_question = support[-1] if support else ""
    requested = "playlist_link" if re.search(r"playlist.{0,35}link|link.{0,35}playlist", last_question, re.I) else ""
    return {
        "order_received": bool(orders),
        "order_references": orders[-3:],
        "payment_pending_reported": payment_pending(text),
        "refund_requested": explicit_refund(text),
        "pending_field": requested,
        "latest_is_question": bool("?" in latest or re.search(r"\b(?:how|what|where|should|can i|como|come|dove)\b", latest, re.I)),
    }


def pending_link_question(frame: dict[str, Any], latest: str) -> bool:
    return bool(frame.get("pending_field") == "playlist_link" and
                re.search(r"\b(?:link|playlist)\b", latest, re.I) and
                re.search(r"\?|\b(?:should|can i|send|devo|inviare|enviar|mandar)\b", latest, re.I))


def public_frame(frame: dict[str, Any]) -> dict[str, Any]:
    """Safe operational summary; identifiers stay inside the case lookup."""
    return {k: v for k, v in frame.items() if k != "order_references"}


def courtesy_only(text: str) -> bool:
    """A pure courtesy cannot revive an older technical request."""
    return bool(re.fullmatch(
        r"\s*(?:(?:ok(?:ay)?|thanks?(?: a lot)?|thank you(?: very much)?|thx|"
        r"grazie|gracias|merci|obrigad[oa]|danke)[\s,.!🙏👍]*){1,4}\s*",
        text or "", re.I))


def authorization_only(text: str) -> bool:
    """Recognize a short go-ahead for routing, never as identity or authority."""
    return bool(re.fullmatch(
        r"\s*(?:(?:yes|s[ií]|sim|oui)[,!.\s]+)?(?:you can (?:do it|proceed)|"
        r"go ahead|puedes hacerlo|puoi (?:farlo|procedere)|procedi|"
        r"pode (?:fazer|prosseguir)|vous pouvez (?:le faire|continuer))[.!\s]*",
        text or "", re.I))
