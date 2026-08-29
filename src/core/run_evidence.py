"""Un'esecuzione che dichiara successo senza aver toccato niente.

Preso da Hermes (``agent/verification_evidence.py`` +
``agent/verification_stop.py``), tradotto nel guasto che qui e' successo
davvero. Da loro la domanda e' "hai modificato del codice e stai per dire
fatto senza aver verificato niente?". Da noi il codice non e' il mestiere: i
compiti autonomi sono. E il modo in cui falliscono e' che riportano
``success`` avendo prodotto solo prosa — nessuna chiamata a un tool, nessun
dato letto, nessun messaggio inviato. Un resoconto plausibile costruito su
niente, che nell'archivio delle esecuzioni e' indistinguibile da un lavoro
fatto.

Il modulo e' PASSIVO, esattamente come il loro. Non declassa un'esecuzione a
fallita e non blocca niente: un compito che legittimamente non ha bisogno di
tool esiste, e trasformarlo in un errore renderebbe rumorosa la cosa che deve
restare leggibile. Registra soltanto — e lo dice a voce alta abbastanza da
essere trovato quando qualcuno chiede "ma questo lavora davvero?".

L'asimmetria e' voluta: un falso allarme costa una riga di log da leggere, un
allarme mancato costa settimane di compito che sembra vivo.
"""

from __future__ import annotations

import json
from typing import Any

# Chiavi sotto cui i vari runtime registrano le chiamate ai tool dentro un run
# salvato. Se ne cambia una, il conteggio va a zero e il modulo direbbe che
# TUTTO e' senza prove: per questo il chiamante distingue "zero" da "non so".
_TOOL_KEYS = ("tool_calls", "tools", "tool_executions")


def count_tool_calls(run: dict[str, Any] | None) -> int | None:
    """Quante chiamate a tool contiene il run salvato, o None se non si sa.

    None NON e' zero. Se la forma del run e' cambiata e non si trova nessuna
    delle chiavi note, la risposta onesta e' "non lo so" — dire zero
    trasformerebbe un difetto di lettura in un'accusa a ogni compito.
    """
    if not isinstance(run, dict):
        return None

    total = 0
    found_any_key = False

    def walk(node: Any) -> None:
        nonlocal total, found_any_key
        if isinstance(node, dict):
            for key, value in node.items():
                if key in _TOOL_KEYS:
                    found_any_key = True
                    if isinstance(value, (list, tuple)):
                        total += len(value)
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    try:
        walk(run)
    except RecursionError:
        return None
    return total if found_any_key else None


def unevidenced_reason(
    *, status: str, run: dict[str, Any] | None, output: str | None,
) -> str | None:
    """Perche' questa esecuzione e' un'affermazione invece di un risultato.

    Restituisce None quando non c'e' niente da segnalare — che e' il caso
    normale e deve restarlo.
    """
    if (status or "").strip().lower() != "success":
        return None  # un fallimento dichiarato e' gia' onesto

    calls = count_tool_calls(run)
    if calls is None or calls > 0:
        return None

    preview = " ".join((output or "").split())[:200]
    return (
        "reported success with zero tool calls: nothing was read, written or "
        "sent, so the outcome is a claim rather than a result"
        + (f" — it said: {preview!r}" if preview else " — and it said nothing")
    )
