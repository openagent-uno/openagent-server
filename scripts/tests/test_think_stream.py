"""Il ragionamento del modello che finiva in chat, in streaming.

``extract_thinking_content`` e' corretto su una stringa intera. In streaming
pero' il testo arriva a pezzi, e il tag si spezza dove capita. Misurato sul
nostro codice prima della correzione, con i delta spezzati come arrivano
davvero:

    '<think>' -> '<think>' | 'Controllo la sua ' -> 'Controllo la sua '
    'configurazione' -> 'configurazione' | '</think>' -> '' | 'Ecco.' -> 'Ecco.'

cioe' in chat: ``<think>Controllo la sua configurazioneEcco.`` — il pensiero
del modello riversato all'utente, tag di apertura incluso, mentre sul
messaggio intero la stessa funzione taglia correttamente.

Preso da Hermes (``agent/think_scrubber.py``). Il caso che rende la cosa non
banale e' il delta che finisce a meta' di un tag: ``"<thi"`` non e' testo da
mostrare e non e' ancora un tag. Si trattiene finche' non si sa.
"""
from __future__ import annotations

from ._framework import TestContext, test


def _run(deltas: list[str]) -> tuple[str, str]:
    from src.core._runner.utils.think_stream import ThinkStreamScrubber

    s = ThinkStreamScrubber()
    shown = "".join(s.feed(d) for d in deltas)
    shown += s.flush()
    return shown, s.thinking


@test("think_stream", "il caso misurato: il blocco spezzato non arriva all'utente")
async def t_the_measured_leak(ctx: TestContext) -> None:
    shown, thinking = _run(
        ["<think>", "Controllo la sua ", "configurazione", "</think>", "Ecco la risposta."])
    assert shown == "Ecco la risposta.", shown
    assert "<think>" not in shown
    # E il ragionamento non e' perso, e' solo altrove: chi vuole mostrarlo a
    # parte ce l'ha.
    assert thinking == "Controllo la sua configurazione"


@test("think_stream", "un tag spezzato a meta' delta non stampa mezzo tag")
async def t_tag_split_across_deltas(ctx: TestContext) -> None:
    # E' il caso che rende necessario lo stato: nessun delta contiene un tag
    # intero, quindi qualunque controllo per-delta li manca tutti.
    shown, thinking = _run(["<", "thi", "nk>", "ragiono", "</", "think", ">", "risposta"])
    assert shown == "risposta", shown
    assert thinking == "ragiono"

    # Anche il caso peggiore: un carattere per volta.
    shown, _ = _run(list("<think>nascosto</think>visibile"))
    assert shown == "visibile", shown


@test("think_stream", "il testo prima e dopo il blocco resta intatto")
async def t_surrounding_text_survives(ctx: TestContext) -> None:
    shown, _ = _run(["Certo. ", "<think>", "mm", "</think>", " Ecco:", " tre passi."])
    assert shown == "Certo.  Ecco: tre passi.", repr(shown)

    # Piu' blocchi nello stesso flusso.
    shown, thinking = _run(["a", "<think>x</think>", "b", "<think>y</think>", "c"])
    assert shown == "abc"
    assert thinking == "xy"


@test("think_stream", "senza blocchi non si trattiene e non si perde niente")
async def t_plain_text_is_untouched(ctx: TestContext) -> None:
    deltas = ["Il ", "certificato ", "e' scaduto ", "il 3 marzo."]
    shown, thinking = _run(deltas)
    assert shown == "".join(deltas)
    assert thinking == ""

    # Un '<' che non apre niente non deve sparire: e' testo.
    shown, _ = _run(["se x ", "< 3 ", "allora"])
    assert shown == "se x < 3 allora", repr(shown)

    # Nemmeno un frammento che ASSOMIGLIA a un tag ma non lo diventa.
    shown, _ = _run(["<thin", "king about it"])
    assert shown == "<thinking about it", repr(shown)


@test("think_stream", "un blocco mai chiuso non stampa il pensiero a meta'")
async def t_unterminated_block_stays_hidden(ctx: TestContext) -> None:
    # Il modello viene troncato mentre pensa. Cio' che e' trattenuto e'
    # ragionamento: mostrarlo perche' il flusso e' finito sarebbe il guasto
    # originale con un passo in piu'.
    shown, thinking = _run(["<think>", "stavo ragionando quando"])
    assert shown == "", repr(shown)
    assert "stavo ragionando" in thinking


@test("think_stream", "la coda trattenuta fuori da un blocco viene emessa alla fine")
async def t_flush_emits_real_text(ctx: TestContext) -> None:
    # Il flusso finisce con qualcosa che POTEVA diventare un tag e non lo e'
    # diventato. E' testo vero e non va perso: senza il flush la risposta
    # perderebbe la sua ultima parola.
    from src.core._runner.utils.think_stream import ThinkStreamScrubber

    s = ThinkStreamScrubber()
    assert s.feed("risposta finita <") == "risposta finita "
    assert s.flush() == "<"

    # Una chiusura ORFANA — senza apertura — si cancella invece di comparire.
    # Capita davvero: alcuni modelli ragionano implicitamente e chiudono
    # soltanto. Cio' che e' gia' partito non si ritira, ma stampare anche il
    # tag aggiungerebbe al danno l'evidenza del danno.
    s2 = ThinkStreamScrubber()
    assert s2.feed("x</thi") == "x"
    assert s2.feed("nk>") == "", repr(s2.feed(""))
    assert s2.feed(" poi il resto") == " poi il resto"


@test("think_stream", "lo stato non sopravvive al turno")
async def t_state_is_per_stream(ctx: TestContext) -> None:
    from src.core._runner.utils.think_stream import ThinkStreamScrubber

    # Uno scrubber lasciato dentro un blocco non deve poter zittire la
    # risposta successiva: se ne crea uno per flusso, ed e' la ragione per cui
    # e' una classe e non una funzione con una variabile globale.
    a = ThinkStreamScrubber()
    a.feed("<think>mai chiuso")
    assert a.inside is True

    b = ThinkStreamScrubber()
    assert b.inside is False
    assert b.feed("risposta pulita") == "risposta pulita"


@test("think_stream", "non e' un tag solo: la famiglia intera")
async def t_the_whole_tag_family(ctx: TestContext) -> None:
    from src.core._runner.utils.think_stream import TAG_NAMES

    # Coprire <think> e dire "fatto" lascia passare gli altri, e i modelli li
    # usano davvero: <thinking> e <reasoning> sono comuni quanto <think>.
    for name in ("think", "thinking", "reasoning", "thought"):
        assert name in TAG_NAMES
        shown, thinking = _run([f"<{name}>", "nascosto", f"</{name}>", "visibile"])
        assert shown == "visibile", (name, shown)
        assert thinking == "nascosto"

    # E anche quello scritto in maiuscolo, che e' il piu' facile da dimenticare.
    shown, _ = _run(["<REASONING_SCRATCHPAD>x</REASONING_SCRATCHPAD>", "ok"])
    assert shown == "ok", shown


@test("think_stream", "il parser dei delta esce pulito, e senza scrubber non cambia niente")
async def t_wired_into_the_delta_parser(ctx: TestContext) -> None:
    from src.core._runner.utils.think_stream import ThinkStreamScrubber
    from src.models.providers.openai.chat import OpenAIChat

    class _Delta:
        def __init__(self, content):
            self.content = content
            self.tool_calls = None

    class _Choice:
        def __init__(self, content):
            self.delta = _Delta(content)

    class _Chunk:
        def __init__(self, content):
            self.choices = [_Choice(content)]
            self.id = None
            self.system_fingerprint = None
            self.model_extra = None
            self.usage = None

    model = OpenAIChat.__new__(OpenAIChat)
    scrubber = ThinkStreamScrubber()
    shown = ""
    for piece in ["<think>", "rifletto", "</think>", "Ecco."]:
        r = model._parse_provider_response_delta(_Chunk(piece), scrubber)
        shown += r.content or ""
    shown += scrubber.flush()
    assert shown == "Ecco.", repr(shown)

    # Senza scrubber il comportamento e' identico a prima, byte per byte: un
    # chiamante che non passa nulla non deve accorgersi che questo esiste.
    plain = model._parse_provider_response_delta(_Chunk("<think>grezzo"))
    assert plain.content == "<think>grezzo"
