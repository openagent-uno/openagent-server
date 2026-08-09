"""_arun_runtime_stream — la rete sull'evento di completamento.

Un turno che non emette delta non deve costare una seconda chiamata al modello.

Contesto (produzione, 9-ago-2026): il 16% dei run di eSound e il 25% di Lyra
finivano in ``agent.run_stream.fallback_to_generate`` con motivo unico
``no_deltas_yielded``. Quel fallback rifa' l'INTERO turno con ``generate()``:
una seconda chiamata al modello, a prezzo pieno, per una risposta che il runtime
aveva gia' prodotto. Su una quota settimanale gia' corta del 30% e' la chiamata
duplicata piu' cara del sistema.

Le due direzioni che contano:
  * nessun delta ma il run e' COMPLETATO con del testo -> quel testo esce
    (e il fallback non serve piu');
  * i delta arrivano regolarmente -> la rete NON aggiunge niente, altrimenti
    la risposta comparirebbe due volte.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._framework import TestContext, test


class _FakeRuntime:
    """``runtime.arun(prompt, **kwargs)`` -> async iterator di eventi."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    def arun(self, prompt: str, **kwargs: Any):
        events = self._events

        async def _gen():
            for e in events:
                yield e

        return _gen()


async def _drain(events: list[Any], session_id: str = "s1") -> str:
    from src.models.dispatcher import _arun_runtime_stream

    out: list[str] = []
    async for chunk in _arun_runtime_stream(
        _FakeRuntime(events),
        prompt="ciao",
        session_id=session_id,
        user_id="u1",
        on_status=None,
        error_event="test.stream_error",
    ):
        out.append(chunk)
    return "".join(out)


@test("stream_completed_net",
      "zero deltas + completed run with text → the text is recovered (no re-run)")
async def test_completed_content_recovered(ctx: TestContext) -> None:
    # Si usano le classi VERE: un finto evento passerebbe il test senza
    # dimostrare che l'isinstance in produzione lo riconosce.
    from src.core._run_state.agent import RunCompletedEvent

    got = await _drain([RunCompletedEvent(session_id="s1", content="La risposta vera.")])
    assert "La risposta vera." in got, (
        f"il testo del run completato non e' stato recuperato: {got!r} — "
        "senza questa rete il chiamante rifa' l'intero turno con generate()"
    )


@test("stream_completed_net",
      "deltas arrived → the net stays out of the way (no duplicated answer)")
async def test_no_duplicate_when_deltas_arrived(ctx: TestContext) -> None:
    from src.core._run_state.agent import RunCompletedEvent, RunContentEvent

    got = await _drain([
        RunContentEvent(session_id="s1", content="La risposta "),
        RunContentEvent(session_id="s1", content="vera."),
        RunCompletedEvent(session_id="s1", content="La risposta vera."),
    ])
    assert got.count("La risposta vera.") == 1, (
        f"risposta duplicata: {got!r} — la rete non deve scattare quando "
        "i delta sono arrivati"
    )


@test("stream_completed_net",
      "empty completed run → nothing yielded (the fallback still owns that case)")
async def test_empty_completion_yields_nothing(ctx: TestContext) -> None:
    from src.core._run_state.agent import RunCompletedEvent

    got = await _drain([RunCompletedEvent(session_id="s1", content="")])
    assert got == "", (
        f"un run completato senza testo non deve produrre niente: {got!r} — "
        "quel caso resta del fallback di run_stream"
    )
