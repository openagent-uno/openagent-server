"""La rete sull'evento di completamento: un turno che non emette delta non
deve costare una seconda chiamata al modello.

Contesto (produzione, 9-ago-2026): il 16% dei run di eSound e il 25% di Lyra
finivano in ``agent.run_stream.fallback_to_generate`` con motivo unico
``no_deltas_yielded``. Quel fallback rifa' l'INTERO turno con ``generate()``:
una seconda chiamata al modello, a prezzo pieno, per una risposta che il runtime
aveva gia' prodotto. Su una quota settimanale gia' corta del 30% e' lo spreco
piu' caro del sistema.

I due test coprono le due direzioni che contano:
  - se non e' arrivato nessun delta ma il run e' COMPLETATO con del testo,
    quel testo deve uscire (niente fallback);
  - se i delta sono arrivati regolarmente, la rete NON deve aggiungere nulla —
    altrimenti la risposta comparirebbe doppia.
"""
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class _FakeCompleted:
    """Sta al posto di RunCompletedEvent: al dispatcher servono solo
    ``session_id``, ``content`` e ``metrics`` (che puo' essere None)."""
    session_id: Optional[str] = None
    content: Optional[Any] = None
    metrics: Optional[Any] = None
    run_id: Optional[str] = None


@dataclass
class _FakeContent:
    session_id: Optional[str] = None
    content: Optional[str] = None
    run_id: Optional[str] = None


@dataclass
class _FakeRuntime:
    events: list = field(default_factory=list)

    def arun(self, prompt, **kwargs):
        events = self.events

        async def _gen():
            for e in events:
                yield e
        return _gen()


async def _drain(runtime, session_id="s1"):
    import src.models.dispatcher as d
    out = []
    async for chunk in d._arun_runtime_stream(
        runtime, prompt="ciao", session_id=session_id, user_id="u1",
        on_status=None, error_event="test.error",
    ):
        out.append(chunk)
    return out


def test_completed_content_recovered_when_no_deltas(ctx):
    """Zero delta + run completato con testo => il testo esce comunque."""
    import asyncio
    import src.models.dispatcher as d

    # L'evento di completamento deve essere riconosciuto come tale: si riusa la
    # classe vera, cosi' il test non passa per via di un finto isinstance.
    from src.core._run_state.agent import RunCompletedEvent

    ev = RunCompletedEvent(session_id="s1", content="La risposta vera.")
    got = asyncio.run(_drain(_FakeRuntime([ev])))
    joined = "".join(got)
    assert "La risposta vera." in joined, (
        f"il testo del run completato non e' stato recuperato: {got!r} — "
        "senza questo il chiamante rifa' l'intero turno con generate()"
    )
    return "recuperato dal completamento, nessuna seconda chiamata"


def test_no_duplicate_when_deltas_arrived(ctx):
    """Delta regolari => la rete non deve aggiungere la risposta una seconda volta."""
    import asyncio
    from src.core._run_state.agent import RunCompletedEvent, RunContentEvent

    evs = [
        RunContentEvent(session_id="s1", content="La risposta "),
        RunContentEvent(session_id="s1", content="vera."),
        RunCompletedEvent(session_id="s1", content="La risposta vera."),
    ]
    got = asyncio.run(_drain(_FakeRuntime(evs)))
    joined = "".join(got)
    assert joined.count("La risposta vera.") == 1, (
        f"risposta duplicata: {joined!r} — la rete non deve scattare "
        "quando i delta sono arrivati"
    )
    return "nessun doppione quando lo streaming funziona"


TESTS = [
    ("stream", "completed_content_recovered", test_completed_content_recovered_when_no_deltas),
    ("stream", "no_duplicate_when_deltas_arrived", test_no_duplicate_when_deltas_arrived),
]
