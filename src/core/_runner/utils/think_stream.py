"""Scrubber a stato per i blocchi di ragionamento nel testo in streaming.

``extract_thinking_content`` e' corretto su una stringa INTERA: trova
``</think>`` e taglia. In streaming pero' il testo arriva a pezzi, e i pezzi
non sono stringhe intere — il tag si spezza dove capita. Misurato sul nostro
codice:

    delta '<think>'              -> emesso '<think>'
    delta 'Controllo la sua '    -> emesso 'Controllo la sua '
    delta 'configurazione'       -> emesso 'configurazione'
    delta '</think>'             -> emesso ''
    delta 'Ecco la risposta.'    -> emesso 'Ecco la risposta.'

cioe' in chat finisce ``<think>Controllo la sua configurazioneEcco la
risposta.``: il ragionamento del modello riversato all'utente, tag di apertura
compreso. Sul messaggio intero la stessa funzione da' correttamente ``Ecco la
risposta.``. Il percorso non-streaming e' giusto, quello in streaming perde.

Riguarda i modelli che mettono il ragionamento IN LINEA nel contenuto invece
che in un campo separato (i Qwen locali, MiniMax, diversi modelli aperti). Chi
usa ``reasoning_content`` non passa di qui e non e' toccato.

Preso da Hermes (``agent/think_scrubber.py``), incluse due cose che andando a
intuito avrei sbagliato:

* **Non e' un tag solo.** I modelli usano ``think``, ``thinking``,
  ``reasoning``, ``thought``, ``REASONING_SCRATCHPAD``. Coprirne uno e dire
  "fatto" lascia passare gli altri.
* **Una chiusura orfana si cancella, non si stampa.** ``</think>`` senza
  apertura capita davvero — ci sono modelli che iniziano a ragionare
  implicitamente e chiudono soltanto. Cio' che e' gia' stato emesso non si puo'
  ritirare, ma almeno il tag non deve comparire.

Il nucleo e' che un delta puo' finire a meta' di un tag: ``"<thi"`` non e'
testo da mostrare, e non e' ancora un tag — e' un forse. Quindi si TRATTIENE
finche' non si sa, invece di indovinare. Trattenere qualche carattere per un
istante e' invisibile; mostrare mezzo tag no.
"""

from __future__ import annotations

TAG_NAMES = ("think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD")

OPEN_TAGS = tuple(f"<{n}>" for n in TAG_NAMES)
CLOSE_TAGS = tuple(f"</{n}>" for n in TAG_NAMES)

# Il piu' lungo prefisso che potrebbe ancora diventare un tag.
_MAX_PARTIAL = max(len(t) for t in OPEN_TAGS + CLOSE_TAGS) - 1


def _find_first(text: str, tags: tuple[str, ...]) -> tuple[int, int]:
    """(indice, lunghezza) del primo tag trovato, o (-1, 0)."""
    best, best_len = -1, 0
    for tag in tags:
        i = text.find(tag)
        if i != -1 and (best == -1 or i < best):
            best, best_len = i, len(tag)
    return best, best_len


def _held_tail(text: str) -> int:
    """Quanti caratteri finali potrebbero essere l'inizio di un tag.

    Zero quando la coda non e' prefisso di nessun tag — cioe' quando puo'
    uscire subito.
    """
    for n in range(min(_MAX_PARTIAL, len(text)), 0, -1):
        tail = text[-n:]
        for tag in OPEN_TAGS + CLOSE_TAGS:
            if tag.startswith(tail):
                return n
    return 0


def _strip_orphan_closes(text: str) -> str:
    """Toglie le chiusure senza apertura.

    Capitano: alcuni modelli ragionano implicitamente e emettono solo il tag
    di chiusura. In streaming il testo precedente e' gia' partito e non si
    ritira — ma stampare anche ``</think>`` aggiungerebbe al danno l'evidenza
    del danno.
    """
    for tag in CLOSE_TAGS:
        text = text.replace(tag, "")
    return text


class ThinkStreamScrubber:
    """Toglie i blocchi di ragionamento da un flusso di delta, con stato.

    Uno per turno, per flusso. Non e' riusabile fra turni: lo stato "sono
    dentro un blocco" e' precisamente cio' che non deve sopravvivere alla fine
    di una risposta, ed e' la ragione per cui e' una classe e non una funzione
    con una variabile globale.
    """

    def __init__(self) -> None:
        self._inside = False
        self._held = ""
        self._thinking = ""

    @property
    def inside(self) -> bool:
        return self._inside

    @property
    def thinking(self) -> str:
        """Il ragionamento visto finora, per chi lo mostra a parte."""
        return self._thinking

    def feed(self, delta: str) -> str:
        """Il testo da mostrare per questo delta. Puo' essere vuoto."""
        if not delta:
            return ""

        buf = self._held + delta
        self._held = ""
        out: list[str] = []

        while buf:
            if self._inside:
                idx, tag_len = _find_first(buf, CLOSE_TAGS)
                if idx == -1:
                    keep = _held_tail(buf)
                    if keep:
                        self._thinking += buf[:-keep]
                        self._held = buf[-keep:]
                    else:
                        self._thinking += buf
                    return "".join(out)
                self._thinking += buf[:idx]
                buf = buf[idx + tag_len:]
                self._inside = False
                continue

            idx, tag_len = _find_first(buf, OPEN_TAGS)
            if idx == -1:
                keep = _held_tail(buf)
                visible = buf[:-keep] if keep else buf
                if keep:
                    self._held = buf[-keep:]
                out.append(_strip_orphan_closes(visible))
                return "".join(out)
            out.append(_strip_orphan_closes(buf[:idx]))
            buf = buf[idx + tag_len:]
            self._inside = True

        return "".join(out)

    def flush(self) -> str:
        """Cio' che resta trattenuto a fine flusso.

        Se il flusso finisce DENTRO un blocco mai chiuso, il trattenuto e'
        ragionamento e resta nascosto: un modello troncato a meta' pensiero non
        deve stampare l'inizio del suo pensiero. Se finisce fuori, il
        trattenuto era testo vero non ancora giudicabile e va emesso —
        altrimenti la risposta perderebbe la sua ultima parola.
        """
        held, self._held = self._held, ""
        if self._inside:
            self._thinking += held
            return ""
        return _strip_orphan_closes(held)
