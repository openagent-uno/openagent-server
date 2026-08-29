"""La stessa chiamata, di nuovo, con gli stessi argomenti.

Abbiamo gia' un tetto: 60 chiamate per esecuzione, oltre il quale il runtime
smette di eseguire. Il tetto pero' non ha visto il caso vero. Il 25-ago su
SpicySparks ``docs-keeper`` e' arrivato a 46 chiamate — SOTTO il tetto — e ha
bruciato 857.876 token di input freschi, perche' ogni passo rimanda l'INTERO
contesto e un agent che continua a chiedere si ripaga tutta la sua storia a
ogni giro. Un tetto ferma la corsa dopo che il danno e' fatto; non riconosce
che i giri erano lo stesso giro.

Preso da Hermes (``agent/tool_guardrails.py``), ridotto al pezzo che qui
serve: la stessa firma — nome piu' argomenti — ripetuta dentro una singola
esecuzione. La terza volta non si esegue: si restituisce quello che era gia'
tornato, dicendo che e' gia' tornato. Il modello ha la risposta e non paga il
giro, che e' esattamente cio' che voleva.

Due prudenze deliberate.

**Si contano solo le letture.** Un nome che sembra mutare qualcosa —
scrivere, creare, mandare, cancellare — non viene mai intercettato. Ripetere
una scrittura identica puo' essere voluto (un tentativo dopo un errore
passeggero), e rifiutarla trasformerebbe una guardia contro lo spreco in una
guardia contro il lavoro. Lo spreco costoso e' comunque dalla parte delle
letture: sono quelle che un agent bloccato ripete all'infinito.

**Alla terza, non alla seconda.** Una ripetizione capita per motivi legittimi
— un tentativo dopo un fallimento passeggero e' il piu' comune. Tre chiamate
identiche in una sola esecuzione non sono piu' un tentativo, sono un giro.

Lo stato vive in una ContextVar per esecuzione, non sull'agent: gli agent sono
in cache e riusati fra turni, e un contatore appiccicato li' bloccherebbe
domani una lettura legittima perche' e' gia' stata fatta ieri.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
from typing import Any

# Alla terza chiamata identica si smette di eseguire.
REPEAT_LIMIT = 3

# Parole che indicano che il tool CAMBIA qualcosa. Chi le porta nel nome non
# viene mai intercettato: meglio pagare un giro sprecato che rifiutare una
# scrittura che qualcuno voleva davvero ripetere.
_MUTATING_HINTS = (
    "write", "create", "update", "delete", "remove", "send", "patch", "post",
    "put", "archive", "manage", "set_", "add_", "move", "rename", "upload",
    "install", "deploy", "restart", "run_", "exec", "shell", "kill", "stop",
)

_counts: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "openagent_tool_repeat_counts", default=None,
)
_results: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "openagent_tool_repeat_results", default=None,
)


def begin_run() -> None:
    """Azzera i contatori. Da chiamare all'inizio di ogni esecuzione."""
    _counts.set({})
    _results.set({})


def is_mutating(name: str) -> bool:
    lowered = (name or "").lower()
    return any(hint in lowered for hint in _MUTATING_HINTS)


def signature(name: str, arguments: Any) -> str:
    """Firma stabile di una chiamata: nome piu' argomenti canonicalizzati.

    Le chiavi vanno ordinate, altrimenti due chiamate identiche scritte in
    ordine diverso sembrerebbero diverse e il rilevatore non vedrebbe il giro
    proprio quando il giro c'e'.
    """
    try:
        canonical = json.dumps(arguments or {}, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        canonical = str(arguments)
    digest = hashlib.sha256(canonical.encode("utf-8", "replace")).hexdigest()[:16]
    return f"{name}:{digest}"


def observe(name: str, arguments: Any) -> str | None:
    """Registra la chiamata; restituisce cosa rispondere invece di eseguirla.

    None significa "esegui pure", che e' il caso normale.
    """
    if is_mutating(name):
        return None
    counts = _counts.get()
    if counts is None:
        return None  # nessuna esecuzione aperta: non e' compito nostro indovinare

    key = signature(name, arguments)
    counts[key] = counts.get(key, 0) + 1
    if counts[key] < REPEAT_LIMIT:
        return None

    previous = (_results.get() or {}).get(key)
    body = (
        f"[chiamata ripetuta] {name} e' gia' stato chiamato {counts[key] - 1} volte "
        "in questa esecuzione con questi stessi argomenti, e gli argomenti "
        "identici danno la stessa risposta. Non e' stato eseguito di nuovo."
    )
    if previous:
        body += f"\n\nCio' che aveva restituito:\n{previous}"
    else:
        body += (
            "\n\nUsa quello che hai gia' ricevuto, oppure cambia gli argomenti "
            "se ti serve qualcosa di diverso."
        )
    return body


def remember_result(name: str, arguments: Any, result: Any, *, limit: int = 1200) -> None:
    """Tiene un estratto della risposta, per poterla ridare invece di rifarla."""
    results = _results.get()
    if results is None:
        return
    key = signature(name, arguments)
    if key in results:
        return  # la prima risposta e' quella che conta
    try:
        text = result if isinstance(result, str) else json.dumps(result, default=str)
    except Exception:  # noqa: BLE001
        text = str(result)
    results[key] = text[:limit]


def repeat_guard(name: str, next_func: Any, arguments: dict[str, Any] | None = None):
    """Hook sincrono per la catena dei tool del runtime."""
    args = arguments or {}
    verdict = observe(name, args)
    if verdict is not None:
        return verdict
    result = next_func(**args)
    remember_result(name, args, result)
    return result


async def repeat_guard_async(
    name: str, next_func: Any, arguments: dict[str, Any] | None = None,
):
    """Variante asincrona: stessa politica, stessa firma dell'hook."""
    args = arguments or {}
    verdict = observe(name, args)
    if verdict is not None:
        return verdict
    result = await next_func(**args)
    remember_result(name, args, result)
    return result
