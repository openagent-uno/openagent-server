"""Il fork di revisione dopo il turno: cosa ha insegnato questo turno, se qualcosa.

Preso da Hermes. Quando un turno finisce bene, un figlio rigioca cio' che e'
appena successo e si fa una domanda sola — c'e' una skill da scrivere o da
aggiornare? — con i tool ristretti a skill e memoria. Il turno non lo tocca.

Non sostituisce il distiller e non e' lo stesso mestiere. Il distiller lavora
su cio' che e' SOPRAVVISSUTO in archivio, il giorno dopo, riassunto; il fork
guarda un turno mentre e' ancora caldo, quando i comandi ci sono ancora e i
vicoli ciechi pure. Il primo trova gli schemi fra i giorni, il secondo prende
la singola volta che ha funzionato prima che sbiadisca.

Le tre cose che questi test sorvegliano, perche' sono quelle che
trasformerebbero la funzione da utile a pericolosa:

1. **Propone, non scrive.** Il rifiuto sta nel codice, non nel prompt.
2. **Solo i turni finiti bene.** Imparare da un turno annullato o esploso
   significa insegnare alla libreria una procedura che non ha funzionato.
3. **Il costo segue la cache.** Sul modello del padre la trascrizione e' calda
   e si manda intera; su un altro modello no, quindi si manda un digest — e il
   digest DICE di essere tale invece di sembrare completo.
"""
from __future__ import annotations

from ._framework import TestContext, test


class _Settings:
    def __init__(self, **kw):
        self.enabled = kw.get("enabled", True)
        self.review_enabled = kw.get("review_enabled", True)
        self.review_mode = kw.get("review_mode", "propose")
        self.review_model = kw.get("review_model", None)


@test("skill_review", "in modalita' proposta NIENTE si puo' scrivere")
async def t_propose_refuses_everything(ctx: TestContext) -> None:
    from src.mcp.servers.skills.provenance import (
        PROPOSE, mutation_refusal, reset_write_origin, set_write_origin,
    )

    token = set_write_origin(PROPOSE)
    try:
        # Nemmeno cio' che l'agent ha scritto lui e che non e' pinnato — cioe'
        # l'unica cosa che in background sarebbe passata. Il valore di questa
        # modalita' e' che il giudizio si vede PRIMA di essere applicato.
        refusal = mutation_refusal("update", "self-remediation",
                                   created_by="agent", pinned=False)
        assert refusal and "proposal-only" in refusal
        assert "A person decides" in refusal

        for action in ("create", "update", "archive", "remove"):
            assert mutation_refusal(action, "qualunque",
                                    created_by="agent", pinned=False)
    finally:
        reset_write_origin(token)

    # E fuori dal blocco si torna liberi: la modalita' non e' appiccicosa.
    assert mutation_refusal("update", "x", created_by=None, pinned=False) is None


@test("skill_review", "in modalita' scrittura torna a valere il confine normale")
async def t_write_mode_keeps_the_ordinary_boundary(ctx: TestContext) -> None:
    from src.mcp.servers.skills.provenance import (
        BACKGROUND, mutation_refusal, reset_write_origin, set_write_origin,
    )

    token = set_write_origin(BACKGROUND)
    try:
        # Sua e non pinnata: passa.
        assert mutation_refusal("update", "self-remediation",
                                created_by="agent", pinned=False) is None
        # Pinnata: no, nemmeno se l'ha scritta lui.
        assert mutation_refusal("update", "esound-support-thread-triage",
                                created_by="agent", pinned=True)
        # Di seed: no.
        assert mutation_refusal("archive", "esound-project-ops",
                                created_by=None, pinned=False)
    finally:
        reset_write_origin(token)


@test("skill_review", "si rivedono solo i turni finiti bene")
async def t_only_completed_turns(ctx: TestContext) -> None:
    from src.core.skill_review import should_review

    on = _Settings()
    assert should_review(on, reason="completed") is True
    # Annullato: l'utente se n'e' andato. Errore: non e' mai arrivato in fondo.
    # Vuoto: non c'e' niente dentro. Nessuno dei tre ha insegnato qualcosa.
    for reason in ("cancelled", "error", "empty", ""):
        assert should_review(on, reason=reason) is False, reason

    # E i due cancelli restano cancelli.
    assert should_review(_Settings(review_enabled=False), reason="completed") is False
    assert should_review(_Settings(enabled=False), reason="completed") is False


@test("skill_review", "sul modello del padre la trascrizione va intera")
async def t_same_model_sends_everything(ctx: TestContext) -> None:
    from src.core.skill_review import review_payload

    long_transcript = "UTENTE: ciao\n\n" + ("AGENT: " + "x" * 200 + "\n\n") * 200
    assert len(long_transcript) > 20_000

    # review_model non impostato: gira sul modello del turno, dove la
    # trascrizione e' gia' calda nel prefix cache. Mandarla intera e' l'opzione
    # ECONOMICA, non quella costosa.
    payload, whole = review_payload(
        long_transcript, parent_model="local:claude-opus-5", review_model=None)
    assert whole is True
    assert payload == long_transcript

    # Stesso modello dichiarato esplicitamente: idem.
    payload, whole = review_payload(
        long_transcript, parent_model="local:claude-opus-5",
        review_model="local:claude-opus-5")
    assert whole is True and payload == long_transcript


@test("skill_review", "su un altro modello va un digest, e il digest lo dichiara")
async def t_other_model_gets_a_digest(ctx: TestContext) -> None:
    from src.core.skill_review import DIGEST_CHAR_BUDGET, review_payload

    head = "UTENTE: come si rigenera il certificato?\n\n"
    tail = "\n\nAGENT: fatto, il comando giusto era `openagent network renew`."
    long_transcript = head + ("TOOL: " + "z" * 300 + "\n\n") * 200 + tail
    assert len(long_transcript) > DIGEST_CHAR_BUDGET * 2

    payload, whole = review_payload(
        long_transcript, parent_model="local:claude-opus-5",
        review_model="codex:gpt-5.6-luna")
    assert whole is False
    assert len(payload) < len(long_transcript)

    # Le due estremita' sopravvivono: l'inizio dice cosa e' stato chiesto, la
    # fine dice cosa alla fine ha funzionato. E' il centro — i vicoli ciechi —
    # a essere sacrificabile per questa domanda.
    assert "come si rigenera il certificato" in payload
    assert "openagent network renew" in payload

    # E soprattutto: il taglio si dichiara. Un digest che sembra completo e'
    # peggio di uno che ammette cosa manca.
    assert "omessi" in payload

    # Una trascrizione corta non viene tagliata solo perche' il modello cambia.
    short = "UTENTE: ciao\n\nAGENT: ciao!"
    payload, whole = review_payload(
        short, parent_model="a", review_model="b")
    assert whole is True and payload == short


@test("skill_review", "il fork vede solo skill e memoria")
async def t_tool_whitelist_is_narrow(ctx: TestContext) -> None:
    from src.core.skill_review import REVIEW_TOOL_FAMILIES

    # Il revisore deve accorgersi di una procedura da tenere. Non ha motivo di
    # eseguire comandi o di scrivere a qualcuno, e la differenza fra un
    # revisore e un secondo agent libero nella stessa sessione e' esattamente
    # quanto e' stretta questa lista.
    assert set(REVIEW_TOOL_FAMILIES) == {"skills", "vault"}
    for forbidden in ("shell", "messaging", "web", "browser", "scheduler"):
        assert forbidden not in REVIEW_TOOL_FAMILIES


@test("skill_review", "la missione dice al revisore che 'niente' e' una risposta giusta")
async def t_mission_permits_the_empty_verdict(ctx: TestContext) -> None:
    from src.core.skill_review import MODE_PROPOSE, MODE_WRITE, _mission

    proposing = _mission(MODE_PROPOSE)
    # Un revisore che sente di dover produrre qualcosa produce comunque
    # qualcosa: e' cosi' che una libreria si riempie di quasi-doppioni.
    assert "Most turns do not" in proposing
    assert "Do not manufacture a skill" in proposing
    # E deve cercare prima di dichiarare che una cosa e' nuova.
    assert "Search the existing skills" in proposing
    # In proposta gli si dice che i tool lo rifiuteranno, cosi' non spreca
    # il turno a sbattere contro il confine.
    assert "CANNOT write" in proposing

    writing = _mission(MODE_WRITE)
    assert "CANNOT write" not in writing
    assert "pinned" in writing


@test("skill_review", "un fallimento della revisione non rompe il turno")
async def t_failure_never_breaks_the_turn(ctx: TestContext) -> None:
    import asyncio

    from src.core.skill_review import schedule_review

    # Il turno e' finito, l'utente sta leggendo la risposta. Un revisore che
    # esplode deve sparire in silenzio nel log, non portarsi dietro la chat.
    schedule_review(agent=None, db=None, parent_session_id="s",
                    transcript="UTENTE: x", settings=_Settings())
    await asyncio.sleep(0.05)  # lascia girare il task di sfondo

    # Nessuna eccezione e' uscita fin qui: e' questa l'asserzione.
    assert True


# ── l'aggancio a fine turno ─────────────────────────────────────────────────
#
# I pezzi provati sopra non servono a niente se il turno non li chiama, o se
# li chiama con la trascrizione sbagliata. Questi due provano la giuntura.

class _JournalDB:
    def __init__(self, events):
        self._events = events

    async def list_session_events(self, session_id, *, after_seq=0, limit=500):
        return [e for e in self._events if e["seq"] > after_seq][:limit]


def _ev(seq, typ, text=""):
    return {"seq": seq, "ts_ms": seq, "type": typ, "data": {"text": text}}


def _session_with(agent):
    """Uno StreamSession abbastanza vivo da eseguire i due metodi."""
    from src.stream.session import StreamSession

    ss = StreamSession.__new__(StreamSession)
    ss._agent = agent
    ss.session_id = "s-review"
    return ss


@test("skill_review", "si manda l'ULTIMO turno, non tutta la conversazione")
async def t_transcript_is_this_turn_only(ctx: TestContext) -> None:
    db = _JournalDB([
        _ev(1, "user/message", "prima domanda"),
        _ev(2, "assistant/message", "prima risposta"),
        _ev(3, "turn/end"),
        _ev(4, "user/message", "il certificato non si rinnova"),
        _ev(5, "tool/status", "shell_run: openagent network renew"),
        _ev(6, "assistant/message", "risolto, serviva --force"),
        _ev(7, "turn/end"),
    ])
    text = await _session_with(object())._turn_transcript(db)

    # Il turno appena chiuso c'e' tutto...
    assert "il certificato non si rinnova" in text
    assert "openagent network renew" in text
    assert "serviva --force" in text
    # ...e quello di prima no. Mandare l'intera sessione farebbe rileggere la
    # stessa storia a ogni revisione di una conversazione lunga, e la domanda
    # e' su QUESTO turno.
    assert "prima domanda" not in text
    assert "prima risposta" not in text

    # I ruoli sono etichettati: il revisore deve poter distinguere cosa ha
    # chiesto l'utente da cosa ha fatto l'agent.
    assert "UTENTE:" in text and "AGENT:" in text and "TOOL:" in text


@test("skill_review", "col cancello chiuso il turno non paga niente")
async def t_gate_off_costs_nothing(ctx: TestContext) -> None:
    calls = []

    class _Agent:
        config = {"skills": {"enabled": True}}  # review_enabled assente = off
        memory_db = _JournalDB([_ev(1, "user/message", "x")])

    import src.core.skill_review as review_mod
    original = review_mod.schedule_review
    review_mod.schedule_review = lambda **kw: calls.append(kw)
    try:
        await _session_with(_Agent())._maybe_review_turn("completed")
        assert calls == [], "ha lanciato la revisione col cancello chiuso"

        # E con il cancello aperto invece parte, con la trascrizione giusta.
        _Agent.config = {"skills": {"enabled": True, "review_enabled": True}}
        await _session_with(_Agent())._maybe_review_turn("completed")
        assert len(calls) == 1, calls
        assert calls[0]["parent_session_id"] == "s-review"
        assert "x" in calls[0]["transcript"]

        # Un turno annullato non paga nemmeno col cancello aperto.
        await _session_with(_Agent())._maybe_review_turn("cancelled")
        assert len(calls) == 1
    finally:
        review_mod.schedule_review = original
