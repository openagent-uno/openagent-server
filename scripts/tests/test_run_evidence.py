"""L'esecuzione che dichiara successo senza aver toccato niente.

Preso da Hermes (``verification_evidence`` + ``verification_stop``), tradotto
nel guasto che qui e' successo davvero. Da loro la domanda e' "hai modificato
codice e stai per dire fatto senza aver verificato?". Da noi il codice non e'
il mestiere: i compiti autonomi sono, e falliscono riportando ``success``
avendo prodotto solo prosa. Un resoconto plausibile costruito su niente, che
nell'archivio delle esecuzioni e' indistinguibile da un lavoro fatto — ed e'
cosi' che un compito rotto resta vivo per settimane.

Passivo come il loro: non declassa, non blocca. Un compito che legittimamente
non usa tool esiste, e trasformarlo in un errore renderebbe rumorosa proprio
la cosa che deve restare leggibile.

La distinzione che questi test difendono e' fra **zero** e **non lo so**. Se
la forma del run cambia e non si trova nessuna delle chiavi note, il conteggio
non e' zero: e' ignoto. Dire zero trasformerebbe un difetto di lettura in
un'accusa contro ogni compito, tutti insieme, lo stesso giorno.
"""
from __future__ import annotations

from ._framework import TestContext, test


@test("run_evidence", "successo senza una sola chiamata a tool viene nominato")
async def t_zero_calls_is_flagged(ctx: TestContext) -> None:
    from src.core.run_evidence import unevidenced_reason

    run = {"status": "COMPLETED", "messages": [
        {"role": "assistant", "content": "Fatto: ho controllato tutto.",
         "tool_calls": []},
    ]}
    reason = unevidenced_reason(
        status="success", run=run, output="Fatto: ho controllato tutto.")
    assert reason, "successo senza prove passato in silenzio"
    assert "zero tool calls" in reason
    # Il testo dichiarato finisce nella segnalazione: senza, chi legge il log
    # deve andarsi a cercare cosa aveva affermato.
    assert "ho controllato tutto" in reason


@test("run_evidence", "un'esecuzione che ha lavorato non viene disturbata")
async def t_real_work_is_silent(ctx: TestContext) -> None:
    from src.core.run_evidence import count_tool_calls, unevidenced_reason

    run = {"messages": [
        {"role": "assistant", "tool_calls": [{"function": {"name": "logs_query"}}]},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "read_note"}},
            {"function": {"name": "send_message"}},
        ]},
    ]}
    assert count_tool_calls(run) == 3
    assert unevidenced_reason(status="success", run=run, output="ok") is None


@test("run_evidence", "zero e 'non lo so' non sono la stessa cosa")
async def t_unknown_is_not_zero(ctx: TestContext) -> None:
    from src.core.run_evidence import count_tool_calls, unevidenced_reason

    # Nessuna chiave nota: la forma del run e' cambiata, oppure non e' un run.
    # La risposta onesta e' None. Dire zero qui accuserebbe OGNI compito nello
    # stesso momento, e la valanga di allarmi identici e' esattamente il modo
    # in cui una segnalazione utile diventa rumore da silenziare.
    assert count_tool_calls({"forma": "sconosciuta"}) is None
    assert count_tool_calls(None) is None
    assert count_tool_calls("non un dizionario") is None
    assert unevidenced_reason(
        status="success", run={"forma": "sconosciuta"}, output="ok") is None

    # Una chiave nota ma vuota invece e' uno zero vero, e va segnalato.
    assert count_tool_calls({"tool_calls": []}) == 0
    assert unevidenced_reason(status="success", run={"tool_calls": []}, output="x")


@test("run_evidence", "un fallimento dichiarato e' gia' onesto: non si commenta")
async def t_declared_failure_is_left_alone(ctx: TestContext) -> None:
    from src.core.run_evidence import unevidenced_reason

    # Il difetto e' la BUGIA, non l'assenza di lavoro. Un'esecuzione che dice
    # di essere fallita sta gia' dicendo la verita', anche senza tool.
    for status in ("failed", "cancelled", "FAILED", " failed "):
        assert unevidenced_reason(
            status=status, run={"tool_calls": []}, output="non ci sono riuscito") is None


@test("run_evidence", "i tool contano ovunque siano annidati")
async def t_counts_at_any_depth(ctx: TestContext) -> None:
    from src.core.run_evidence import count_tool_calls

    # I run salvati annidano diversamente a seconda del runtime; un conteggio
    # che guarda un solo livello direbbe zero su una struttura sana, che e' il
    # falso allarme piu' facile da produrre.
    nested = {"a": {"b": [{"c": {"tool_calls": [1, 2]}}]},
              "d": [{"tool_executions": [1]}]}
    assert count_tool_calls(nested) == 3


@test("run_evidence", "il controllo non puo' rompere l'esecuzione che osserva")
async def t_never_raises(ctx: TestContext) -> None:
    from src.core.run_evidence import count_tool_calls

    # Una struttura ciclica: un testimone che esplode sul caso strano
    # trasformerebbe un'esecuzione sana in un fallimento riportato.
    cyclic: dict = {"tool_calls": []}
    cyclic["self"] = cyclic
    result = count_tool_calls(cyclic)
    assert result in (0, None)
